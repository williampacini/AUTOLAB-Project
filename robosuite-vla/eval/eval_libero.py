#!/usr/bin/env python3
"""Evaluate a trained model on standard LIBERO benchmarks.

Standard evaluation: 10 tasks × 50 episodes = 500 trials per suite.
Reports per-task and overall success rates.

Usage:
    python eval/eval_libero.py --checkpoint outputs/checkpoints/smolvla_spatial --suite libero_spatial
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

# Monkey-patch torch.load for LIBERO compatibility (PyTorch 2.6+ defaults to
# weights_only=True but LIBERO init states contain numpy arrays)
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

ROOT = Path(__file__).resolve().parent.parent


def load_policy(checkpoint_path, device="cuda"):
    """Load a trained SmolVLA policy with preprocessor/postprocessor pipelines.

    Returns (policy, preprocess, postprocess) tuple.
    The preprocessor handles language tokenization and state normalization.
    The postprocessor handles action unnormalization.
    """
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors

        policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
        policy.to(device)
        policy.eval()

        # Load preprocessor/postprocessor from saved model configs.
        # These contain the dataset normalization stats (mean/std for state & action)
        # and the tokenizer config for language instructions.
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            checkpoint_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

        print(f"Loaded policy from {checkpoint_path}")
        return policy, preprocess, postprocess
    except Exception as e:
        print(f"Failed to load policy: {e}")
        print("Using random policy for testing.")
        return None, None, None


def create_env(suite_name, task_id):
    """Create a LIBERO environment for a specific task."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)

    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=128,
        camera_widths=128,
    )

    return env, task, task_suite


def _get_model_image_keys(policy):
    """Get expected image keys from model config.
    Base model uses camera1/camera2/camera3; fine-tuned uses image/image2.
    """
    if hasattr(policy.config, 'input_features'):
        keys = sorted([
            k for k in policy.config.input_features
            if k.startswith("observation.images")
        ])
        if keys:
            return keys
    return ["observation.images.image", "observation.images.image2"]


def _get_model_state_dim(policy):
    """Get expected state dimension from model config."""
    if hasattr(policy.config, 'input_features'):
        feat = policy.config.input_features.get("observation.state")
        if feat is not None:
            shape = getattr(feat, 'shape', None)
            if shape is None and isinstance(feat, dict):
                shape = feat.get('shape')
            if shape is not None:
                return shape[0] if isinstance(shape, (list, tuple)) else shape
    return 7


def get_action(policy, preprocess, postprocess, obs, task_language, device="cuda"):
    """Get action from policy given observation.

    Dynamically adapts to the model's expected image keys and state dimension
    so this works for both the base model (camera1/camera2/camera3, state=6)
    and fine-tuned models (image/image2, state=7).
    """
    if policy is None:
        return np.random.uniform(-0.3, 0.3, size=7)

    from PIL import Image as PILImage
    from lerobot.policies.utils import prepare_observation_for_inference

    image_keys = _get_model_image_keys(policy)
    expected_state_dim = _get_model_state_dim(policy)

    # Camera 1: agentview (main workspace camera)
    agentview = np.flip(
        obs.get("agentview_image", np.zeros((128, 128, 3), dtype=np.uint8)), axis=0
    ).copy()
    agentview = np.array(PILImage.fromarray(agentview).resize((256, 256)))

    # Camera 2: wrist / eye-in-hand camera
    wrist = obs.get("robot0_eye_in_hand_image", None)
    if wrist is not None:
        wrist = np.flip(wrist, axis=0).copy()
        wrist = np.array(PILImage.fromarray(wrist).resize((256, 256)))
    else:
        wrist = np.zeros((256, 256, 3), dtype=np.uint8)

    # Build state from actual robot proprioception, adjust to expected dim
    eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
    eef_quat = obs.get("robot0_eef_quat", np.zeros(4))
    state = np.concatenate([eef_pos, eef_quat]).astype(np.float32)
    if len(state) > expected_state_dim:
        state = state[:expected_state_dim]
    elif len(state) < expected_state_dim:
        state = np.pad(state, (0, expected_state_dim - len(state)))

    # Build observation dict using model's expected image keys
    cameras = [agentview, wrist]
    raw_obs = {}
    for i, key in enumerate(image_keys):
        if i < len(cameras):
            raw_obs[key] = cameras[i]
        else:
            raw_obs[key] = np.zeros((256, 256, 3), dtype=np.uint8)
    raw_obs["observation.state"] = state

    # Convert to tensors, normalize images to [0,1], add batch dim, move to device
    obs_frame = prepare_observation_for_inference(raw_obs, device, task=task_language)

    # Preprocess: tokenize language, normalize state (MEAN_STD)
    obs_preprocessed = preprocess(obs_frame)

    # Model inference
    with torch.no_grad():
        action = policy.select_action(obs_preprocessed)

    # Postprocess: unnormalize action (MEAN_STD)
    action = postprocess(action)

    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy().flatten()

    return np.clip(action[:7], -1, 1)


def run_episode(env, policy, preprocess, postprocess, task, task_suite,
                task_id, episode_idx, max_steps=600, device="cuda", record=False):
    """Run a single evaluation episode.

    Returns:
        success: bool
        total_reward: float
        frames: list of frames (if record=True)
    """
    env.seed(episode_idx)
    obs = env.reset()

    # Set initial state for reproducibility
    init_states = task_suite.get_task_init_states(task_id)
    if episode_idx < len(init_states):
        env.set_init_state(init_states[episode_idx])
        obs = env.reset()

    # Reset policy internal state at start of each episode
    if policy is not None:
        policy.reset()

    frames = []
    total_reward = 0.0

    for step in range(max_steps):
        action = get_action(policy, preprocess, postprocess, obs,
                            task.language, device=device)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if record and step % 3 == 0:  # Every 3rd frame to save memory
            cam_img = obs.get("agentview_image")
            if cam_img is not None:
                frames.append(np.flip(cam_img, axis=0).copy())

        if done:
            break

    # Check success
    success = bool(done and reward > 0)

    return success, total_reward, frames


def evaluate_suite(checkpoint_path, suite_name, n_episodes=50, max_steps=600,
                   device="cuda", seed=42, record_freq=10, output_dir=None):
    """Evaluate on all tasks in a LIBERO suite.

    Returns:
        results: dict with per-task and overall success rates
    """
    from libero.libero import benchmark

    np.random.seed(seed)
    policy, preprocess, postprocess = load_policy(checkpoint_path, device=device)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    n_tasks = task_suite.n_tasks

    print(f"\n{'='*60}")
    print(f"  Evaluating on {suite_name} ({n_tasks} tasks × {n_episodes} episodes)")
    print(f"{'='*60}")

    results = {
        "suite": suite_name,
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "n_episodes": n_episodes,
        "tasks": [],
    }

    total_successes = 0
    total_trials = 0

    for task_id in range(n_tasks):
        env, task, task_suite_obj = create_env(suite_name, task_id)
        task_successes = 0

        pbar = tqdm(range(n_episodes), desc=f"  Task {task_id}: {task.language[:40]}")
        for ep in pbar:
            record = (ep % record_freq == 0) if record_freq > 0 else False
            success, reward, frames = run_episode(
                env, policy, preprocess, postprocess, task,
                task_suite_obj, task_id, ep,
                max_steps=max_steps, device=device, record=record,
            )

            if success:
                task_successes += 1

            pbar.set_postfix(sr=f"{task_successes}/{ep+1}")

            # Save GIF for recorded episodes
            if record and frames and output_dir:
                _save_gif(frames, output_dir, suite_name, task_id, ep, success)

        sr = task_successes / n_episodes
        results["tasks"].append({
            "task_id": task_id,
            "language": task.language,
            "successes": task_successes,
            "success_rate": sr,
        })
        total_successes += task_successes
        total_trials += n_episodes

        print(f"  Task {task_id}: {sr:.1%} ({task_successes}/{n_episodes}) — {task.language}")
        env.close()

    results["overall_success_rate"] = total_successes / total_trials
    results["total_successes"] = total_successes
    results["total_trials"] = total_trials

    print(f"\n  Overall: {results['overall_success_rate']:.1%} ({total_successes}/{total_trials})")

    return results


def _save_gif(frames, output_dir, suite, task_id, ep, success):
    """Save episode frames as GIF."""
    from PIL import Image as PILImage

    output_dir = Path(output_dir) / "videos" / suite
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = "success" if success else "fail"
    gif_path = output_dir / f"task{task_id}_ep{ep}_{tag}.gif"

    pil_frames = [PILImage.fromarray(f).resize((256, 256)) for f in frames]
    if pil_frames:
        pil_frames[0].save(
            gif_path, save_all=True, append_images=pil_frames[1:],
            duration=50, loop=0, optimize=True,
        )


def save_results(results, output_dir):
    """Save results as JSON and CSV."""
    output_dir = Path(output_dir) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    suite = results["suite"]
    seed = results["seed"]

    # JSON
    json_path = output_dir / f"{suite}_seed{seed}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {json_path}")

    # CSV
    csv_path = output_dir / f"{suite}_seed{seed}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "language", "successes", "episodes", "success_rate"])
        for t in results["tasks"]:
            writer.writerow([t["task_id"], t["language"], t["successes"],
                           results["n_episodes"], f"{t['success_rate']:.4f}"])
        writer.writerow(["OVERALL", "", results["total_successes"],
                        results["total_trials"], f"{results['overall_success_rate']:.4f}"])
    print(f"CSV saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on standard LIBERO")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--suite", type=str, default="libero_spatial",
                       choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per task")
    parser.add_argument("--max-steps", type=int, default=600, help="Max steps per episode")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--record-freq", type=int, default=10, help="Record GIF every N episodes (0=off)")
    args = parser.parse_args()

    results = evaluate_suite(
        checkpoint_path=args.checkpoint,
        suite_name=args.suite,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
        record_freq=args.record_freq,
        output_dir=args.output_dir,
    )

    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
