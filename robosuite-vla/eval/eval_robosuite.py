#!/usr/bin/env python3
"""Evaluate a trained SmolVLA model on robosuite tasks.

Standard evaluation: 6 tasks x N episodes per task.
Reports per-task and overall success rates.
Saves MP4 video of every evaluation episode for verification.

Usage:
    # Evaluate all tasks, record every episode
    python eval/eval_robosuite.py --checkpoint outputs/checkpoints/smolvla_robosuite --all --record-all

    # Nut assembly only, record everything
    python eval/eval_robosuite.py --checkpoint outputs/checkpoints/smolvla_robosuite --nut-assembly --record-all

    # Single task
    python eval/eval_robosuite.py --checkpoint outputs/checkpoints/smolvla_robosuite --env Lift --episodes 50

    # Quick smoke test
    python eval/eval_robosuite.py --checkpoint outputs/checkpoints/smolvla_robosuite --env Lift --episodes 2
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Headless rendering — MUST be before any robosuite import
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

ROOT = Path(__file__).resolve().parent.parent

# Reuse env configs from collect_demos
ENV_CONFIGS = {
    "Lift": {
        "max_steps": 200,
        "task_description": "pick up the red cube and lift it",
    },
    "Stack": {
        "max_steps": 400,
        "task_description": "pick up the red cube and stack it on the green cube",
    },
    "PickPlaceSingle": {
        "max_steps": 400,
        "task_description": "pick up the object and place it in the bin",
    },
    "Door": {
        "max_steps": 300,
        "task_description": "open the door by turning the handle",
    },
    "NutAssemblySingle": {
        "max_steps": 400,
        "task_description": "pick up the nut and place it on the peg",
    },
    "NutAssembly": {
        "max_steps": 600,
        "task_description": "pick up each nut and place it on the correct peg",
    },
}

NUT_ASSEMBLY_ENVS = ["NutAssemblySingle", "NutAssembly"]


def make_env(env_name, camera_res=128):
    """Create a robosuite environment for evaluation."""
    import robosuite as suite
    from robosuite.controllers import load_controller_config

    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=False,  # Not needed for learned policy
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=camera_res,
        camera_widths=camera_res,
        reward_shaping=True,
        control_freq=20,
    )
    return env


def load_policy(checkpoint_path, device="cuda"):
    """Load a trained SmolVLA policy from checkpoint."""
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
        policy.to(device)
        policy.eval()
        print(f"Loaded policy from {checkpoint_path}")
        return policy
    except Exception as e:
        print(f"Failed to load policy: {e}")
        print("Using random policy for testing.")
        return None


def get_action(policy, obs, task_language, device="cuda"):
    """Get action from policy given observation."""
    if policy is None:
        return np.random.uniform(-0.3, 0.3, size=7)

    from PIL import Image as PILImage

    # Camera 1: agentview
    agentview = np.flip(
        obs.get("agentview_image", np.zeros((128, 128, 3), dtype=np.uint8)),
        axis=0,
    )
    agentview_pil = PILImage.fromarray(agentview).resize((256, 256))

    # Camera 2: wrist / eye-in-hand
    wrist = obs.get("robot0_eye_in_hand_image", None)
    if wrist is not None:
        wrist = np.flip(wrist, axis=0)
        wrist_pil = PILImage.fromarray(wrist).resize((256, 256))
    else:
        wrist_pil = PILImage.fromarray(
            np.zeros((128, 128, 3), dtype=np.uint8)
        ).resize((256, 256))

    # Build state: eef_pos(3) + eef_quat(4)
    eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
    eef_quat = obs.get("robot0_eef_quat", np.zeros(4))
    state = np.concatenate([eef_pos, eef_quat]).astype(np.float32)

    # SmolVLA observation dict
    obs_dict = {
        "observation.images.image": (
            torch.from_numpy(np.array(agentview_pil))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
            / 255.0
        ),
        "observation.images.image2": (
            torch.from_numpy(np.array(wrist_pil))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
            / 255.0
        ),
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "task": task_language,
    }

    with torch.no_grad():
        action = policy.select_action(obs_dict)

    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy().flatten()

    return np.clip(action[:7], -1, 1)


def run_episode(env, policy, env_name, episode_idx, max_steps=400,
                device="cuda", record=False):
    """Run a single evaluation episode.

    Args:
        record: If True, captures EVERY frame for video saving.

    Returns:
        success: bool
        total_reward: float
        frames: list of RGB frames (if record=True), every frame captured
    """
    # Use different seeds from training (offset by 10000)
    seed = 10000 + episode_idx
    np.random.seed(seed)
    obs = env.reset()

    task_language = ENV_CONFIGS[env_name]["task_description"]

    # Reset policy state at episode start
    if policy is not None:
        policy.reset()

    frames = []
    total_reward = 0.0

    for step in range(max_steps):
        # Capture frame BEFORE action (shows what policy sees)
        if record:
            cam_img = obs.get("agentview_image")
            if cam_img is not None:
                frames.append(np.flip(cam_img, axis=0).copy())

        action = get_action(policy, obs, task_language, device=device)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Check success
        try:
            task_success = env._check_success()
        except AttributeError:
            task_success = bool(reward > 0.5)

        if task_success:
            # Capture final success frame
            if record:
                cam_img = obs.get("agentview_image")
                if cam_img is not None:
                    frames.append(np.flip(cam_img, axis=0).copy())
            return True, total_reward, frames

        if done:
            break

    return False, total_reward, frames


def _save_video(frames, path, fps=20):
    """Save frames as MP4 video."""
    import imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def _save_gif(frames, path):
    """Save episode frames as GIF (smaller, for quick preview)."""
    from PIL import Image as PILImage

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Subsample to keep GIF size reasonable
    step = max(1, len(frames) // 60)
    sampled = frames[::step]

    pil_frames = [PILImage.fromarray(f).resize((256, 256)) for f in sampled]
    if pil_frames:
        pil_frames[0].save(
            path, save_all=True, append_images=pil_frames[1:],
            duration=50, loop=0, optimize=True,
        )


def evaluate_env(checkpoint_path, env_name, n_episodes=50, max_steps=None,
                 device="cuda", record_all=False, record_freq=10,
                 output_dir=None):
    """Evaluate on a single robosuite environment.

    Args:
        record_all: If True, save MP4 video for EVERY episode.
        record_freq: If record_all is False, save GIF every N episodes.

    Returns:
        dict with task results
    """
    max_steps = max_steps or ENV_CONFIGS[env_name]["max_steps"]
    policy = load_policy(checkpoint_path, device=device)
    env = make_env(env_name)

    video_dir = Path(output_dir or "outputs") / "videos" / "robosuite" / env_name
    video_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    rewards = []
    video_paths = []

    pbar = tqdm(range(n_episodes), desc=f"  {env_name}")
    for ep in pbar:
        # Record this episode?
        should_record = record_all or (record_freq > 0 and ep % record_freq == 0)

        success, reward, frames = run_episode(
            env, policy, env_name, ep,
            max_steps=max_steps, device=device, record=should_record,
        )

        if success:
            successes += 1
        rewards.append(reward)
        pbar.set_postfix(sr=f"{successes}/{ep + 1}")

        # Save video/gif
        if should_record and frames and output_dir:
            tag = "success" if success else "fail"
            if record_all:
                # Full MP4 for every episode
                mp4_path = video_dir / f"ep{ep:03d}_{tag}.mp4"
                _save_video(frames, mp4_path, fps=20)
                video_paths.append(str(mp4_path))
            else:
                # Sampled GIF
                gif_path = video_dir / f"ep{ep:03d}_{tag}.gif"
                _save_gif(frames, gif_path)
                video_paths.append(str(gif_path))

    env.close()

    sr = successes / n_episodes
    print(f"  {env_name}: {sr:.1%} ({successes}/{n_episodes})")
    if video_paths:
        print(f"    Videos saved: {video_dir}/ ({len(video_paths)} files)")

    return {
        "env_name": env_name,
        "task_description": ENV_CONFIGS[env_name]["task_description"],
        "successes": successes,
        "n_episodes": n_episodes,
        "success_rate": sr,
        "mean_reward": float(np.mean(rewards)),
        "video_dir": str(video_dir),
        "n_videos": len(video_paths),
    }


def evaluate_all(checkpoint_path, env_names=None, n_episodes=50, device="cuda",
                 record_all=False, record_freq=10, output_dir=None):
    """Evaluate on multiple robosuite environments."""
    env_names = env_names or list(ENV_CONFIGS.keys())

    print(f"\n{'='*60}")
    print(f"  Evaluating on {len(env_names)} robosuite tasks x {n_episodes} episodes")
    if record_all:
        print(f"  Recording MP4 video for EVERY episode")
    print(f"{'='*60}")

    results = {
        "checkpoint": str(checkpoint_path),
        "n_episodes": n_episodes,
        "record_all": record_all,
        "tasks": [],
    }

    total_successes = 0
    total_trials = 0

    for env_name in env_names:
        task_result = evaluate_env(
            checkpoint_path, env_name, n_episodes=n_episodes,
            device=device, record_all=record_all, record_freq=record_freq,
            output_dir=output_dir,
        )
        results["tasks"].append(task_result)
        total_successes += task_result["successes"]
        total_trials += n_episodes

    results["overall_success_rate"] = total_successes / total_trials
    results["total_successes"] = total_successes
    results["total_trials"] = total_trials

    print(f"\n  Overall: {results['overall_success_rate']:.1%} "
          f"({total_successes}/{total_trials})")

    return results


def save_results(results, output_dir):
    """Save results as JSON and CSV."""
    out = Path(output_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out / "robosuite_eval.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {json_path}")

    # CSV
    csv_path = out / "robosuite_eval.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env_name", "task_description", "successes",
                         "episodes", "success_rate", "mean_reward"])
        for t in results["tasks"]:
            writer.writerow([
                t["env_name"], t["task_description"],
                t["successes"], t["n_episodes"],
                f"{t['success_rate']:.4f}",
                f"{t['mean_reward']:.4f}",
            ])
        writer.writerow([
            "OVERALL", "", results["total_successes"],
            results["total_trials"],
            f"{results['overall_success_rate']:.4f}", "",
        ])
    print(f"CSV saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA on robosuite tasks")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")

    # Task selection (mutually exclusive group)
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--env", type=str, default=None,
                            choices=list(ENV_CONFIGS.keys()),
                            help="Single environment to evaluate")
    task_group.add_argument("--all", action="store_true",
                            help="Evaluate all environments")
    task_group.add_argument("--nut-assembly", action="store_true",
                            help="Evaluate NutAssemblySingle + NutAssembly only")

    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per task (default: 50)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps per episode (overrides env default)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--record-all", action="store_true",
                        help="Record MP4 video of EVERY episode for verification")
    parser.add_argument("--record-freq", type=int, default=10,
                        help="Record GIF every N episodes when --record-all is off (0=off)")
    args = parser.parse_args()

    if not args.all and not args.env and not args.nut_assembly:
        parser.error("Specify --env ENV_NAME, --all, or --nut-assembly")

    # Determine which envs to evaluate
    if args.env:
        env_names = [args.env]
    elif args.nut_assembly:
        env_names = NUT_ASSEMBLY_ENVS
    else:
        env_names = list(ENV_CONFIGS.keys())

    if len(env_names) == 1:
        task_result = evaluate_env(
            checkpoint_path=args.checkpoint,
            env_name=env_names[0],
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            device=args.device,
            record_all=args.record_all,
            record_freq=args.record_freq,
            output_dir=args.output_dir,
        )
        results = {
            "checkpoint": args.checkpoint,
            "n_episodes": args.episodes,
            "record_all": args.record_all,
            "tasks": [task_result],
            "overall_success_rate": task_result["success_rate"],
            "total_successes": task_result["successes"],
            "total_trials": args.episodes,
        }
    else:
        results = evaluate_all(
            checkpoint_path=args.checkpoint,
            env_names=env_names,
            n_episodes=args.episodes,
            device=args.device,
            record_all=args.record_all,
            record_freq=args.record_freq,
            output_dir=args.output_dir,
        )

    save_results(results, args.output_dir)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*60}")
    for t in results["tasks"]:
        print(f"  {t['env_name']:20s}  {t['success_rate']:.1%}  "
              f"({t['successes']}/{t['n_episodes']})")
    print(f"  {'OVERALL':20s}  {results['overall_success_rate']:.1%}  "
          f"({results['total_successes']}/{results['total_trials']})")
    if args.record_all:
        print(f"\n  Videos: {args.output_dir}/videos/robosuite/<env>/")


if __name__ == "__main__":
    main()
