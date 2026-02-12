#!/usr/bin/env python3
"""Evaluate a trained SmolVLA model on robosuite tasks.

Standard evaluation: 6 tasks x N episodes per task.
Reports per-task and overall success rates.

Usage:
    # Evaluate all tasks
    python eval/eval_robosuite.py --checkpoint outputs/checkpoints/smolvla_robosuite --all

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

    Returns:
        success: bool
        total_reward: float
        frames: list of frames (if record=True)
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
        action = get_action(policy, obs, task_language, device=device)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if record and step % 3 == 0:
            cam_img = obs.get("agentview_image")
            if cam_img is not None:
                frames.append(np.flip(cam_img, axis=0).copy())

        # Check success
        try:
            task_success = env._check_success()
        except AttributeError:
            task_success = bool(reward > 0.5)

        if task_success:
            return True, total_reward, frames

        if done:
            break

    return False, total_reward, frames


def evaluate_env(checkpoint_path, env_name, n_episodes=50, max_steps=None,
                 device="cuda", record_freq=10, output_dir=None):
    """Evaluate on a single robosuite environment.

    Returns:
        dict with task results
    """
    max_steps = max_steps or ENV_CONFIGS[env_name]["max_steps"]
    policy = load_policy(checkpoint_path, device=device)
    env = make_env(env_name)

    successes = 0
    rewards = []

    pbar = tqdm(range(n_episodes), desc=f"  {env_name}")
    for ep in pbar:
        record = (ep % record_freq == 0) if record_freq > 0 else False
        success, reward, frames = run_episode(
            env, policy, env_name, ep,
            max_steps=max_steps, device=device, record=record,
        )

        if success:
            successes += 1
        rewards.append(reward)
        pbar.set_postfix(sr=f"{successes}/{ep + 1}")

        if record and frames and output_dir:
            _save_gif(frames, output_dir, env_name, ep, success)

    env.close()

    sr = successes / n_episodes
    print(f"  {env_name}: {sr:.1%} ({successes}/{n_episodes})")

    return {
        "env_name": env_name,
        "task_description": ENV_CONFIGS[env_name]["task_description"],
        "successes": successes,
        "n_episodes": n_episodes,
        "success_rate": sr,
        "mean_reward": float(np.mean(rewards)),
    }


def evaluate_all(checkpoint_path, n_episodes=50, device="cuda",
                 record_freq=10, output_dir=None):
    """Evaluate on all robosuite environments."""
    print(f"\n{'='*60}")
    print(f"  Evaluating on {len(ENV_CONFIGS)} robosuite tasks x {n_episodes} episodes")
    print(f"{'='*60}")

    results = {
        "checkpoint": str(checkpoint_path),
        "n_episodes": n_episodes,
        "tasks": [],
    }

    total_successes = 0
    total_trials = 0

    for env_name in ENV_CONFIGS:
        task_result = evaluate_env(
            checkpoint_path, env_name, n_episodes=n_episodes,
            device=device, record_freq=record_freq, output_dir=output_dir,
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


def _save_gif(frames, output_dir, env_name, ep, success):
    """Save episode frames as GIF."""
    from PIL import Image as PILImage

    out = Path(output_dir) / "videos" / "robosuite"
    out.mkdir(parents=True, exist_ok=True)

    tag = "success" if success else "fail"
    gif_path = out / f"{env_name}_ep{ep}_{tag}.gif"

    pil_frames = [PILImage.fromarray(f).resize((256, 256)) for f in frames]
    if pil_frames:
        pil_frames[0].save(
            gif_path, save_all=True, append_images=pil_frames[1:],
            duration=50, loop=0, optimize=True,
        )


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
        writer.writerow(["env_name", "task_description", "successes", "episodes", "success_rate"])
        for t in results["tasks"]:
            writer.writerow([
                t["env_name"], t["task_description"],
                t["successes"], t["n_episodes"],
                f"{t['success_rate']:.4f}",
            ])
        writer.writerow([
            "OVERALL", "", results["total_successes"],
            results["total_trials"],
            f"{results['overall_success_rate']:.4f}",
        ])
    print(f"CSV saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA on robosuite tasks")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--env", type=str, default=None,
                        choices=list(ENV_CONFIGS.keys()),
                        help="Single environment to evaluate")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all environments")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per task (default: 50)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps per episode (overrides env default)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--record-freq", type=int, default=10,
                        help="Record GIF every N episodes (0=off)")
    args = parser.parse_args()

    if not args.all and not args.env:
        parser.error("Specify --env ENV_NAME or --all")

    if args.all:
        results = evaluate_all(
            checkpoint_path=args.checkpoint,
            n_episodes=args.episodes,
            device=args.device,
            record_freq=args.record_freq,
            output_dir=args.output_dir,
        )
    else:
        task_result = evaluate_env(
            checkpoint_path=args.checkpoint,
            env_name=args.env,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            device=args.device,
            record_freq=args.record_freq,
            output_dir=args.output_dir,
        )
        results = {
            "checkpoint": args.checkpoint,
            "n_episodes": args.episodes,
            "tasks": [task_result],
            "overall_success_rate": task_result["success_rate"],
            "total_successes": task_result["successes"],
            "total_trials": args.episodes,
        }

    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
