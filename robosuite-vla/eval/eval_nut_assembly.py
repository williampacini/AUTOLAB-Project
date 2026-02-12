#!/usr/bin/env python3
"""Evaluate a trained SmolVLA model on RobotSuite NutAssembly.

Records an MP4 video for EVERY test episode so each scenario can be
visually inspected for success or failure. Videos are sped up by a
configurable factor (default 3x) and tagged with success/fail in the filename.

Usage:
    python eval/eval_nut_assembly.py --checkpoint outputs/checkpoints/smolvla_nut_assembly
    python eval/eval_nut_assembly.py --checkpoint outputs/checkpoints/smolvla_nut_assembly \
        --episodes 20 --speedup 4 --output-dir outputs
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


def create_env(camera_size=256):
    """Create a NutAssembly environment with two cameras."""
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
        single_object_mode=0,  # Both round and square nuts
        horizon=1000,
    )
    return env


def get_action(policy, obs, task_language, device="cuda"):
    """Get action from SmolVLA policy given observation.

    Follows the same pattern as eval_libero.py:get_action().
    """
    if policy is None:
        # Random policy for testing the evaluation loop
        return np.random.uniform(-0.3, 0.3, size=7)

    from PIL import Image as PILImage

    # Camera 1: agentview (flip for MuJoCo upside-down rendering)
    agentview = np.flip(obs["agentview_image"], axis=0)
    agentview_pil = PILImage.fromarray(agentview).resize((256, 256))

    # Camera 2: wrist / eye-in-hand camera
    wrist = obs.get("robot0_eye_in_hand_image", None)
    if wrist is not None:
        wrist = np.flip(wrist, axis=0)
        wrist_pil = PILImage.fromarray(wrist).resize((256, 256))
    else:
        wrist_pil = PILImage.fromarray(
            np.zeros((256, 256, 3), dtype=np.uint8)
        )

    # Build state: eef_pos (3) + eef_quat (4) = 7D
    eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
    eef_quat = obs.get("robot0_eef_quat", np.zeros(4))
    state = np.concatenate([eef_pos, eef_quat]).astype(np.float32)

    # SmolVLA observation dict
    obs_dict = {
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

    with torch.no_grad():
        action = policy.select_action(obs_dict)

    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy().flatten()

    return np.clip(action[:7], -1, 1)


def run_episode(env, policy, task_language, seed, max_steps=1000, device="cuda"):
    """Run a single evaluation episode.

    Records EVERY frame for video output.

    Returns:
        success: bool
        total_reward: float
        frames: list of (H, W, 3) uint8 arrays (flipped agentview)
        n_steps: int
    """
    np.random.seed(seed)
    env.seed(seed)
    obs = env.reset()

    # Reset policy internal state
    if policy is not None:
        policy.reset()

    frames = []
    total_reward = 0.0
    success = False

    for step in range(max_steps):
        action = get_action(policy, obs, task_language, device=device)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Record every frame for video
        agentview = np.flip(obs["agentview_image"], axis=0).copy()
        frames.append(agentview)

        # Check success
        if env._check_success():
            success = True
            break

        if done:
            break

    return success, total_reward, frames, step + 1


def save_video_mp4(frames, path, fps=20, speedup=1):
    """Save frames as MP4 video with optional speed-up.

    Args:
        frames: list of (H, W, 3) uint8 arrays
        path: output file path
        fps: native capture FPS
        speedup: playback speed multiplier (e.g., 3 = 3x faster)
    """
    import imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    effective_fps = int(fps * speedup)
    writer = imageio.get_writer(str(path), fps=effective_fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def add_text_overlay(frame, text, position=(10, 20)):
    """Add text overlay to a frame (optional, requires PIL)."""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    pil_img = PILImage.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    # Draw text with background for readability
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()

    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill="black")
    draw.text(position, text, fill="white", font=font)

    return np.array(pil_img)


def evaluate(checkpoint_path, n_episodes=20, seeds=None, max_steps=1000,
             device="cuda", output_dir="outputs", speedup=3, camera_size=256):
    """Full evaluation with video recording for every episode.

    Every episode is recorded as an MP4 video, tagged with success/fail.
    A summary JSON/CSV is output showing which scenarios passed/failed.

    Args:
        checkpoint_path: path to SmolVLA checkpoint
        n_episodes: number of evaluation episodes
        seeds: list of seeds (one per episode). Defaults to range(n_episodes).
        max_steps: max simulation steps per episode
        device: "cuda" or "cpu"
        output_dir: base output directory
        speedup: video playback speed multiplier
        camera_size: camera resolution

    Returns:
        summary dict with results
    """
    output_dir = Path(output_dir)
    video_dir = output_dir / "videos" / "nut_assembly"
    results_dir = output_dir / "results"
    video_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    task_language = "Assemble the round nut and square nut onto their respective pegs"

    # Load policy
    policy = load_policy(checkpoint_path, device=device)

    # Create environment
    env = create_env(camera_size=camera_size)

    # Generate seeds
    if seeds is None:
        seeds = list(range(n_episodes))

    print(f"\n{'='*60}")
    print(f"  NutAssembly Evaluation")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Episodes:   {n_episodes}")
    print(f"  Max steps:  {max_steps}")
    print(f"  Video dir:  {video_dir}")
    print(f"  Speedup:    {speedup}x")
    print()

    results = []
    start_time = time.time()

    for idx, seed in enumerate(tqdm(seeds[:n_episodes], desc="  Evaluating")):
        success, reward, frames, steps = run_episode(
            env, policy, task_language, seed, max_steps, device
        )

        # Add text overlay to first and last frames showing status
        if frames:
            status_text = f"Scenario {idx} | Seed {seed}"
            frames[0] = add_text_overlay(frames[0], status_text)
            result_text = f"{'SUCCESS' if success else 'FAIL'} | Steps: {steps} | Reward: {reward:.2f}"
            frames[-1] = add_text_overlay(frames[-1], result_text)

        # Save video for this episode
        tag = "success" if success else "fail"
        video_path = video_dir / f"scenario_{idx:03d}_{tag}.mp4"
        save_video_mp4(frames, video_path, fps=20, speedup=speedup)

        results.append({
            "scenario_idx": idx,
            "seed": seed,
            "success": success,
            "total_reward": float(reward),
            "steps": steps,
            "video_path": str(video_path),
        })

        # Print inline result
        icon = "PASS" if success else "FAIL"
        tqdm.write(f"  [{icon}] Scenario {idx:3d} (seed={seed}) — "
                    f"reward={reward:.2f}, steps={steps}, video={video_path.name}")

    elapsed = time.time() - start_time
    env.close()

    # Compute summary statistics
    n_success = sum(r["success"] for r in results)
    n_total = len(results)
    success_rate = n_success / n_total if n_total > 0 else 0.0

    summary = {
        "task": "NutAssembly",
        "task_language": task_language,
        "checkpoint": str(checkpoint_path),
        "n_episodes": n_total,
        "n_success": n_success,
        "n_fail": n_total - n_success,
        "success_rate": success_rate,
        "video_speedup": speedup,
        "max_steps": max_steps,
        "elapsed_seconds": elapsed,
        "episodes": results,
    }

    # Save JSON results
    json_path = results_dir / "nut_assembly_eval.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults JSON: {json_path}")

    # Save CSV results
    csv_path = results_dir / "nut_assembly_eval.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_idx", "seed", "success", "reward", "steps", "video_path"])
        for r in results:
            writer.writerow([
                r["scenario_idx"], r["seed"], r["success"],
                f"{r['total_reward']:.4f}", r["steps"], r["video_path"],
            ])
        writer.writerow([])
        writer.writerow(["OVERALL", "", f"{success_rate:.4f}",
                         "", "", f"{n_success}/{n_total}"])
    print(f"Results CSV:  {csv_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  RESULTS: {n_success}/{n_total} succeeded ({success_rate:.1%})")
    print(f"{'='*60}")
    print(f"  Successes: {n_success}")
    print(f"  Failures:  {n_total - n_success}")
    print(f"  Time:      {elapsed:.1f}s ({elapsed/n_total:.1f}s per episode)")
    print(f"\n  Videos saved to: {video_dir}/")
    print(f"  Each video is {speedup}x speed, named scenario_NNN_success.mp4 or scenario_NNN_fail.mp4")

    # List failed scenarios for quick review
    failed = [r for r in results if not r["success"]]
    if failed:
        print(f"\n  Failed scenarios ({len(failed)}):")
        for r in failed:
            print(f"    - Scenario {r['scenario_idx']:3d} (seed={r['seed']}): {r['video_path']}")

    # List succeeded scenarios
    succeeded = [r for r in results if r["success"]]
    if succeeded:
        print(f"\n  Succeeded scenarios ({len(succeeded)}):")
        for r in succeeded:
            print(f"    - Scenario {r['scenario_idx']:3d} (seed={r['seed']}): {r['video_path']}")

    return summary


def generate_summary_html(summary, output_dir):
    """Generate an HTML page with embedded video links for easy review.

    This is especially useful in Colab where you can display it inline.
    """
    output_dir = Path(output_dir)
    html_path = output_dir / "results" / "nut_assembly_eval.html"

    episodes = summary["episodes"]
    success_rate = summary["success_rate"]

    rows = ""
    for r in episodes:
        color = "#4CAF50" if r["success"] else "#f44336"
        status = "SUCCESS" if r["success"] else "FAIL"
        video_name = Path(r["video_path"]).name
        rows += f"""
        <tr style="background-color: {'#e8f5e9' if r['success'] else '#ffebee'}">
            <td>{r['scenario_idx']}</td>
            <td>{r['seed']}</td>
            <td style="color: {color}; font-weight: bold">{status}</td>
            <td>{r['total_reward']:.2f}</td>
            <td>{r['steps']}</td>
            <td><a href="../videos/nut_assembly/{video_name}">{video_name}</a></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>NutAssembly Evaluation Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .summary .rate {{ font-size: 2em; font-weight: bold; color: {'#4CAF50' if success_rate > 0.5 else '#f44336'}; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #333; color: white; }}
    </style>
</head>
<body>
    <h1>NutAssembly Evaluation Results</h1>

    <div class="summary">
        <p class="rate">{summary['n_success']}/{summary['n_episodes']} ({success_rate:.1%})</p>
        <p>Checkpoint: <code>{summary['checkpoint']}</code></p>
        <p>Video speedup: {summary['video_speedup']}x | Max steps: {summary['max_steps']}</p>
    </div>

    <table>
        <tr>
            <th>Scenario</th>
            <th>Seed</th>
            <th>Status</th>
            <th>Reward</th>
            <th>Steps</th>
            <th>Video</th>
        </tr>
        {rows}
    </table>
</body>
</html>"""

    with open(html_path, "w") as f:
        f.write(html)
    print(f"HTML report: {html_path}")
    return html_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA on NutAssembly with video output")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SmolVLA checkpoint")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max steps per episode")
    parser.add_argument("--speedup", type=int, default=3,
                        help="Video playback speed multiplier (e.g., 3 = 3x faster)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Base output directory")
    parser.add_argument("--camera-size", type=int, default=256,
                        help="Camera resolution")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Specific seeds to use (one per episode)")
    args = parser.parse_args()

    summary = evaluate(
        checkpoint_path=args.checkpoint,
        n_episodes=args.episodes,
        seeds=args.seeds,
        max_steps=args.max_steps,
        device=args.device,
        output_dir=args.output_dir,
        speedup=args.speedup,
        camera_size=args.camera_size,
    )

    # Generate HTML report for easy review
    generate_summary_html(summary, args.output_dir)


if __name__ == "__main__":
    main()
