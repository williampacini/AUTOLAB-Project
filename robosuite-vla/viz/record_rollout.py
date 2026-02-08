#!/usr/bin/env python3
"""Record GIF of a model rollout on a LIBERO task.

Usage:
    python viz/record_rollout.py --checkpoint outputs/checkpoints/smolvla_spatial \
        --suite libero_spatial --task 0 --output rollout.gif
"""

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

ROOT = Path(__file__).resolve().parent.parent


def record(checkpoint_path, suite_name, task_id, output_path,
           max_steps=600, device="cuda", seed=42, fps=20):
    """Record a single episode as GIF."""
    from eval.eval_libero import load_policy, create_env, get_action
    from PIL import Image as PILImage

    policy = load_policy(checkpoint_path, device=device)
    env, task, task_suite = create_env(suite_name, task_id)

    print(f"Recording: {task.language}")

    np.random.seed(seed)
    env.seed(seed)
    obs = env.reset()

    init_states = task_suite.get_task_init_states(task_id)
    if init_states is not None and len(init_states) > 0:
        env.set_init_state(init_states[0])
        obs = env.reset()

    frames = []
    total_reward = 0.0

    for step in range(max_steps):
        action = get_action(policy, obs, task.language, device=device)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        cam_img = obs.get("agentview_image")
        if cam_img is not None:
            frame = np.flip(cam_img, axis=0).copy()
            frames.append(frame)

        if done:
            print(f"  Task completed at step {step}!")
            break

    env.close()

    # Save GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = max(1, 1000 // fps)
    pil_frames = [PILImage.fromarray(f).resize((256, 256)) for f in frames]
    if pil_frames:
        pil_frames[0].save(
            str(output_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )
        print(f"GIF saved: {output_path} ({len(frames)} frames)")
    else:
        print("No frames captured!")

    print(f"Total reward: {total_reward:.4f}")
    print(f"Success: {done and reward > 0}")


def main():
    parser = argparse.ArgumentParser(description="Record rollout GIF")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = args.output or f"outputs/videos/{args.suite}_task{args.task}.gif"
    record(args.checkpoint, args.suite, args.task, output,
           max_steps=args.max_steps, device=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
