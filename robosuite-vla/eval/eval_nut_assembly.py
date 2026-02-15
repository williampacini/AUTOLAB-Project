#!/usr/bin/env python3
"""Evaluate SmolVLA on robosuite NutAssembly with MP4 video recording.

Every episode is recorded and tagged success/fail. Supports both single-task
and multi-task checkpoints — auto-detects expected state dimension from the
checkpoint's preprocessor stats.

Uses LeRobot's official preprocessing pipeline (make_pre_post_processors)
which handles language tokenization and input/output normalization.

Usage:
    python eval/eval_nut_assembly.py \
        --checkpoint outputs/checkpoints/smolvla_nut_assembly \
        --episodes 20 \
        --max-steps 1000 \
        --speedup 3 \
        --output-dir outputs \
        --camera-size 256
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Headless rendering — MUST be before any robosuite/mujoco import
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

ROOT = Path(__file__).resolve().parent.parent


def find_pretrained_dir(checkpoint_path):
    """Resolve the actual pretrained_model directory from a checkpoint path.

    LeRobot saves checkpoints in nested structure:
        <output_dir>/checkpoints/last/pretrained_model/
    Users may pass any level of this hierarchy.
    """
    p = Path(checkpoint_path)

    # If it already contains config.json, use it directly
    if (p / "config.json").exists():
        return p

    # Try common subdirectory patterns
    candidates = [
        p / "checkpoints" / "last" / "pretrained_model",
        p / "last" / "pretrained_model",
        p / "pretrained_model",
    ]
    for c in candidates:
        if (c / "config.json").exists():
            return c

    # Search for config.json recursively
    configs = list(p.rglob("config.json"))
    if configs:
        return configs[0].parent

    # Fall back to original path — from_pretrained will give a clear error
    return p


def get_state_dim_from_checkpoint(pretrained_dir):
    """Read expected state dimension from preprocessor stats."""
    try:
        from safetensors.torch import load_file
    except ImportError:
        return None

    for sf in sorted(pretrained_dir.glob("*preprocessor*.safetensors")):
        try:
            tensors = load_file(str(sf))
            for key, tensor in tensors.items():
                if "observation.state" in key and "mean" in key:
                    return tensor.shape[-1]
        except Exception:
            continue

    return None


def load_policy_and_processors(checkpoint_path, device="cuda"):
    """Load SmolVLA policy with its preprocessor and postprocessor.

    The preprocessor handles:
      - Language tokenization (task string → observation.language.tokens)
      - Image normalization (using learned dataset stats)
      - State normalization (using learned dataset stats)

    The postprocessor handles:
      - Action unnormalization

    Returns (policy, preprocess, postprocess, state_dim).
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    pretrained_dir = find_pretrained_dir(checkpoint_path)
    print(f"Loading policy from: {pretrained_dir}")

    # Auto-detect state dim
    state_dim = get_state_dim_from_checkpoint(pretrained_dir)

    # Load model
    policy = SmolVLAPolicy.from_pretrained(pretrained_dir)
    policy.to(device)
    policy.eval()

    # Read n_action_steps from config
    config_path = pretrained_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        n_action_steps = cfg.get("n_action_steps", "unknown")
        print(f"  n_action_steps: {n_action_steps}")
        if n_action_steps == 1:
            print("  WARNING: n_action_steps=1 — inference will be 50x slower!")
            print("  Run: python train/train_smolvla.py --fix-config <checkpoint>")

    # Create preprocessor and postprocessor using LeRobot's factory
    # This handles tokenization, normalization, and unnormalization
    try:
        from lerobot.policies.factory import make_pre_post_processors
    except ImportError:
        from lerobot.common.policies.factory import make_pre_post_processors

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    print(f"  State dim (from preprocessor): {state_dim}")
    print(f"  Preprocessor + postprocessor loaded")
    return policy, preprocess, postprocess, state_dim


def create_env(camera_size=256):
    """Create robosuite NutAssembly environment."""
    import robosuite as suite

    env = suite.make(
        env_name="NutAssembly",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=camera_size,
        camera_widths=camera_size,
        reward_shaping=True,
        single_object_mode=0,  # Both nuts
        horizon=1000,
    )
    return env


def build_state(obs, expected_dim=None):
    """Build state vector from robosuite observation.

    Constructs: eef_pos(3) + eef_quat(4) [+ gripper_qpos(2)]
    and trims/pads to match the expected dimension from training.
    """
    parts = [
        obs["robot0_eef_pos"],       # (3,)
        obs["robot0_eef_quat"],      # (4,)
    ]

    # Include gripper if we need more than 7 dims
    if expected_dim is not None and expected_dim > 7:
        gripper = obs.get("robot0_gripper_qpos")
        if gripper is not None:
            parts.append(gripper)

    state = np.concatenate(parts).astype(np.float32)

    # Trim or pad to exact expected dim
    if expected_dim is not None:
        if len(state) > expected_dim:
            state = state[:expected_dim]
        elif len(state) < expected_dim:
            state = np.pad(state, (0, expected_dim - len(state)))

    return state


def get_action(policy, preprocess, postprocess, obs, task_language,
               device="cuda", img_size=256, expected_state_dim=None):
    """Get action from policy given robosuite observation.

    Uses LeRobot's preprocessor for tokenization + normalization,
    and postprocessor for action unnormalization.
    """
    from PIL import Image as PILImage

    # Camera 1: agentview (flip for MuJoCo rendering convention)
    agentview = np.flip(obs["agentview_image"], axis=0)
    agentview_pil = PILImage.fromarray(agentview).resize(
        (img_size, img_size), PILImage.LANCZOS
    )

    # Camera 2: wrist / eye-in-hand camera
    wrist = np.flip(obs["robot0_eye_in_hand_image"], axis=0)
    wrist_pil = PILImage.fromarray(wrist).resize(
        (img_size, img_size), PILImage.LANCZOS
    )

    # State vector (auto-matched to training dim)
    state = build_state(obs, expected_state_dim)

    # Build observation frame — images as [0,1] float tensors, state as float tensor
    # The preprocessor will handle tokenization + normalization
    obs_frame = {
        "observation.images.image": (
            torch.from_numpy(np.array(agentview_pil))
            .permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        ),
        "observation.images.image2": (
            torch.from_numpy(np.array(wrist_pil))
            .permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        ),
        "observation.state": (
            torch.from_numpy(state).unsqueeze(0).to(device)
        ),
        "task": task_language,
    }

    # Preprocess: tokenizes task string + normalizes images/state
    batch = preprocess(obs_frame)

    # Inference
    with torch.no_grad():
        action = policy.select_action(batch)

    # Postprocess: unnormalize action
    action = postprocess(action)

    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy().flatten()
    elif isinstance(action, dict):
        # Some postprocessors return a dict with "action" key
        action = action.get("action", action)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()

    return np.clip(action[:7], -1, 1)


def save_video(frames, path, fps):
    """Save frames as MP4 video."""
    import imageio

    writer = imageio.get_writer(
        str(path), fps=fps, codec="h264",
        output_params=["-pix_fmt", "yuv420p"],
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def evaluate(args):
    """Run evaluation on NutAssembly."""
    policy, preprocess, postprocess, state_dim = load_policy_and_processors(
        args.checkpoint, args.device,
    )
    env = create_env(args.camera_size)

    task_language = (
        args.task
        or "Assemble the round nut and square nut onto their respective pegs"
    )

    video_dir = Path(args.output_dir) / "videos" / "nut_assembly"
    results_dir = Path(args.output_dir) / "results"
    video_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    control_freq = getattr(env, "control_freq", 20)
    video_fps = int(control_freq * args.speedup)

    print(f"\n{'='*60}")
    print(f"  NutAssembly Evaluation")
    print(f"{'='*60}")
    print(f"  Episodes:    {args.episodes}")
    print(f"  Max steps:   {args.max_steps}")
    print(f"  Task:        {task_language}")
    print(f"  State dim:   {state_dim}")
    print(f"  Camera size: {args.camera_size}")
    print(f"  Video FPS:   {video_fps} ({args.speedup}x speedup)")
    print()

    results = {
        "task": "NutAssembly",
        "task_language": task_language,
        "checkpoint": str(args.checkpoint),
        "state_dim": state_dim,
        "n_episodes": args.episodes,
        "video_speedup": args.speedup,
        "episodes": [],
    }

    n_success = 0
    total_time = 0.0

    for ep in range(args.episodes):
        obs = env.reset()
        policy.reset()

        frames = []
        total_reward = 0.0
        success = False
        steps_taken = 0
        t0 = time.time()

        for step in range(args.max_steps):
            action = get_action(
                policy, preprocess, postprocess,
                obs, task_language, args.device,
                args.camera_size, state_dim,
            )
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps_taken = step + 1

            # Record frame for video
            cam_img = obs.get("agentview_image")
            if cam_img is not None:
                frames.append(np.flip(cam_img, axis=0).copy())

            # Check task success
            if env._check_success():
                success = True
                break

            if done:
                break

        elapsed = time.time() - t0
        total_time += elapsed

        if success:
            n_success += 1

        # Save video
        tag = "success" if success else "fail"
        video_path = video_dir / f"ep{ep:03d}_{tag}.mp4"
        if frames:
            save_video(frames, video_path, video_fps)

        results["episodes"].append({
            "scenario_idx": ep,
            "seed": ep,
            "success": success,
            "total_reward": float(total_reward),
            "steps": steps_taken,
            "video_path": str(video_path),
            "time_s": round(elapsed, 1),
        })

        status = "SUCCESS" if success else "FAIL"
        print(f"  Episode {ep:3d}: {status} | "
              f"reward={total_reward:7.2f} | "
              f"steps={steps_taken:4d} | "
              f"time={elapsed:.1f}s")

    env.close()

    results["n_success"] = n_success
    results["n_fail"] = args.episodes - n_success
    results["success_rate"] = n_success / args.episodes
    results["total_time_s"] = round(total_time, 1)

    # Save results
    results_path = results_dir / "nut_assembly_eval.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  NutAssembly: {n_success}/{args.episodes} "
          f"({results['success_rate']:.1%})")
    print(f"  Total time: {total_time:.0f}s "
          f"({total_time/args.episodes:.1f}s/episode)")
    print(f"  Results: {results_path}")
    print(f"  Videos:  {video_dir}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SmolVLA on robosuite NutAssembly"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to model checkpoint (any level of the directory hierarchy)",
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--speedup", type=float, default=3,
        help="Video playback speedup factor",
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Output directory for videos and results",
    )
    parser.add_argument(
        "--camera-size", type=int, default=256,
        help="Camera observation resolution",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for inference",
    )
    parser.add_argument(
        "--task", default=None,
        help="Override task language description",
    )
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
