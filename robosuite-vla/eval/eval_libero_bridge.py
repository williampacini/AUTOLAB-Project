#!/usr/bin/env python3
"""Evaluate SmolVLA on custom BDDL tasks that bridge LIBERO and robosuite.

Runs the model through LIBERO's OffScreenRenderEnv using custom BDDL task
files that approximate robosuite tasks (Lift, NutAssembly) within LIBERO's
environment framework.

Two task sets are provided:

  1. **Proxy tasks** (bddl_tasks/lift/, bddl_tasks/nut_assembly/):
     Use LIBERO's native objects (bowls, mugs, baskets) as stand-ins for
     robosuite objects. Work out-of-the-box with any LIBERO installation.

  2. **Bridge tasks** (bddl_tasks/robosuite_bridge/):
     Import robosuite's actual MuJoCo objects (Cube, RoundNut, SquareNut)
     into LIBERO's environment via the registration module. Requires both
     robosuite and LIBERO to be installed.

Usage (CLI):
    # Run proxy tasks only (no robosuite objects needed in LIBERO):
    python eval/eval_libero_bridge.py \\
        --checkpoint outputs/checkpoints/smolvla_spatial \\
        --task-set proxy --episodes 20

    # Run bridge tasks (robosuite objects registered in LIBERO):
    python eval/eval_libero_bridge.py \\
        --checkpoint outputs/checkpoints/smolvla_spatial \\
        --task-set bridge --episodes 20

    # Run all tasks:
    python eval/eval_libero_bridge.py \\
        --checkpoint outputs/checkpoints/smolvla_spatial \\
        --task-set all --episodes 20

Usage (Colab):
    See "# === COLAB CELL ===" markers below.
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

# Monkey-patch torch.load for LIBERO compatibility
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

ROOT = Path(__file__).resolve().parent.parent
BDDL_DIR = ROOT / "bddl_tasks"

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

# Proxy tasks: use LIBERO's existing objects as stand-ins
PROXY_TASKS = {
    "lift_bowl": {
        "bddl_file": str(BDDL_DIR / "lift" / "lift_bowl_from_table.bddl"),
        "language": "pick up the black bowl from the table",
        "horizon": 500,
        "robosuite_equivalent": "Lift",
    },
    "lift_mug": {
        "bddl_file": str(BDDL_DIR / "lift" / "lift_mug_from_table.bddl"),
        "language": "pick up the red mug from the table",
        "horizon": 500,
        "robosuite_equivalent": "Lift",
    },
    "lift_bowl_distractors": {
        "bddl_file": str(BDDL_DIR / "lift" / "lift_bowl_with_distractors.bddl"),
        "language": "pick up the black bowl from the center of the table",
        "horizon": 500,
        "robosuite_equivalent": "Lift",
    },
    "place_bowl_on_plate": {
        "bddl_file": str(BDDL_DIR / "nut_assembly" / "place_bowl_on_plate.bddl"),
        "language": "pick up the black bowl and place it on the plate",
        "horizon": 600,
        "robosuite_equivalent": "NutAssembly (pick-and-place)",
    },
    "place_mug_on_plate": {
        "bddl_file": str(BDDL_DIR / "nut_assembly" / "place_mug_on_plate.bddl"),
        "language": "pick up the red mug and place it on the plate",
        "horizon": 600,
        "robosuite_equivalent": "NutAssembly (pick-and-place)",
    },
    "place_butter_in_basket": {
        "bddl_file": str(BDDL_DIR / "nut_assembly" / "place_butter_in_basket.bddl"),
        "language": "pick up the butter and place it in the basket",
        "horizon": 600,
        "robosuite_equivalent": "NutAssembly (place-in-container)",
    },
    "place_two_in_basket": {
        "bddl_file": str(BDDL_DIR / "nut_assembly" / "place_two_objects_in_basket.bddl"),
        "language": "pick up the butter and the cream cheese and place them both in the basket",
        "horizon": 1000,
        "robosuite_equivalent": "NutAssembly (two-object)",
    },
}

# Bridge tasks: use robosuite's actual objects registered in LIBERO
BRIDGE_TASKS = {
    "lift_cube_native": {
        "bddl_file": str(BDDL_DIR / "robosuite_bridge" / "lift_cube.bddl"),
        "language": "pick up the red cube from the table",
        "horizon": 500,
        "robosuite_equivalent": "Lift (native cube)",
        "requires_bridge": True,
    },
    "round_nut_on_plate": {
        "bddl_file": str(BDDL_DIR / "robosuite_bridge" / "place_round_nut_on_plate.bddl"),
        "language": "pick up the round nut and place it on the plate",
        "horizon": 1000,
        "robosuite_equivalent": "NutAssemblyRound (native nut)",
        "requires_bridge": True,
    },
    "square_nut_on_plate": {
        "bddl_file": str(BDDL_DIR / "robosuite_bridge" / "place_square_nut_on_plate.bddl"),
        "language": "pick up the square nut and place it on the plate",
        "horizon": 1000,
        "robosuite_equivalent": "NutAssemblySquare (native nut)",
        "requires_bridge": True,
    },
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_policy_and_tokenizer(checkpoint_path, device="cuda"):
    """Load SmolVLA policy and tokenizer."""
    tokenizer = None
    policy = None

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    except Exception as e:
        print(f"WARNING: Could not load tokenizer: {e}")

    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
        policy.to(device)
        policy.eval()
        print(f"Loaded SmolVLA from {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load policy: {e}")
        print("Using RANDOM policy for pipeline testing.")

    return policy, tokenizer


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def create_bddl_env(bddl_file, requires_bridge=False):
    """Create a LIBERO OffScreenRenderEnv from a BDDL file.

    Args:
        bddl_file: Path to .bddl task file.
        requires_bridge: If True, register robosuite objects first.

    Returns:
        env: LIBERO OffScreenRenderEnv instance.
    """
    if requires_bridge:
        try:
            sys.path.insert(0, str(ROOT))
            import bddl_tasks.robosuite_bridge  # noqa: F401 — triggers registration
        except ImportError as e:
            print(f"WARNING: Bridge registration failed: {e}")
            print("  Ensure both LIBERO and robosuite are installed.")
            return None

    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=128,
        camera_widths=128,
    )
    return env


# ---------------------------------------------------------------------------
# Action inference
# ---------------------------------------------------------------------------

def get_action(policy, obs, language, tokenizer, device="cuda"):
    """Get action from policy given LIBERO observation dict."""
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

    # State: eef_pos + eef_quat
    eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
    eef_quat = obs.get("robot0_eef_quat", np.zeros(4))
    state = np.concatenate([eef_pos, eef_quat]).astype(np.float32)

    # Tokenize language
    tokens = tokenizer(
        language,
        return_tensors="pt",
        padding="max_length",
        max_length=256,
        truncation=True,
    )

    obs_dict = {
        "observation.images.image": (
            torch.from_numpy(np.array(agentview_pil))
            .permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        ),
        "observation.images.image2": (
            torch.from_numpy(np.array(wrist_pil))
            .permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        ),
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "observation.language.tokens": tokens["input_ids"].to(device),
        "observation.language.attention_mask": tokens["attention_mask"].bool().to(device),
    }

    with torch.no_grad():
        action = policy.select_action(obs_dict)

    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy().flatten()

    return np.clip(action[:7], -1, 1)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, policy, language, tokenizer, max_steps=600,
                device="cuda", record=False):
    """Run a single evaluation episode.

    Returns:
        success: bool — True if task goal is achieved
        total_reward: float
        steps: int
        frames: list (if record=True)
    """
    obs = env.reset()

    if policy is not None:
        policy.reset()

    frames = []
    total_reward = 0.0

    for step in range(max_steps):
        action = get_action(policy, obs, language, tokenizer, device=device)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if record and step % 3 == 0:
            cam_img = obs.get("agentview_image")
            if cam_img is not None:
                frames.append(np.flip(cam_img, axis=0).copy())

        if done:
            break

    success = bool(done and reward > 0)
    return success, total_reward, step + 1, frames


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_tasks(
    checkpoint_path,
    task_set="proxy",
    n_episodes=20,
    max_steps=None,
    device="cuda",
    seed=42,
    record_freq=10,
    output_dir="outputs",
):
    """Evaluate on custom BDDL tasks.

    Args:
        task_set: "proxy" (LIBERO objects), "bridge" (robosuite objects), or "all"
    """
    np.random.seed(seed)

    # Select task set
    tasks = {}
    if task_set in ("proxy", "all"):
        tasks.update(PROXY_TASKS)
    if task_set in ("bridge", "all"):
        tasks.update(BRIDGE_TASKS)

    if not tasks:
        print(f"ERROR: Unknown task_set '{task_set}'. Use 'proxy', 'bridge', or 'all'.")
        return None

    print(f"\n{'='*70}")
    print(f"  LIBERO-BRIDGE EVALUATION")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Task set:   {task_set} ({len(tasks)} tasks x {n_episodes} episodes)")
    print(f"{'='*70}\n")

    # Load model
    policy, tokenizer = load_policy_and_tokenizer(checkpoint_path, device)

    results = {
        "checkpoint": str(checkpoint_path),
        "task_set": task_set,
        "seed": seed,
        "n_episodes": n_episodes,
        "tasks": {},
    }

    total_successes = 0
    total_trials = 0

    for task_name, task_info in tasks.items():
        bddl_file = task_info["bddl_file"]
        language = task_info["language"]
        horizon = max_steps or task_info["horizon"]
        requires_bridge = task_info.get("requires_bridge", False)

        print(f"\n  --- {task_name}: \"{language}\" (horizon={horizon}) ---\n")

        # Check BDDL file exists
        if not os.path.exists(bddl_file):
            print(f"    SKIP: BDDL file not found: {bddl_file}")
            continue

        # Create environment
        try:
            env = create_bddl_env(bddl_file, requires_bridge=requires_bridge)
            if env is None:
                print(f"    SKIP: Could not create environment")
                continue
        except Exception as e:
            print(f"    SKIP: Environment creation failed: {e}")
            continue

        task_successes = 0
        task_rewards = []
        task_steps = []

        for ep in range(n_episodes):
            t0 = time.time()
            env.seed(seed + ep)

            record = (ep % record_freq == 0) if record_freq > 0 else False
            success, reward, steps, frames = run_episode(
                env, policy, language, tokenizer,
                max_steps=horizon, device=device, record=record,
            )

            elapsed = time.time() - t0
            tag = "OK" if success else "--"
            print(f"    ep {ep:3d}:  {tag}  R={reward:7.2f}  steps={steps:4d}  ({elapsed:.1f}s)")

            if success:
                task_successes += 1
            task_rewards.append(reward)
            task_steps.append(steps)

            # Save GIF
            if record and frames and output_dir:
                _save_gif(frames, output_dir, task_name, ep, success)

        sr = task_successes / n_episodes
        results["tasks"][task_name] = {
            "language": language,
            "bddl_file": bddl_file,
            "robosuite_equivalent": task_info["robosuite_equivalent"],
            "horizon": horizon,
            "successes": task_successes,
            "success_rate": sr,
            "mean_reward": float(np.mean(task_rewards)),
            "mean_steps": float(np.mean(task_steps)),
        }

        total_successes += task_successes
        total_trials += n_episodes

        print(f"\n    Result: {sr:.1%} ({task_successes}/{n_episodes})  "
              f"avg_R={np.mean(task_rewards):.2f}  avg_steps={np.mean(task_steps):.0f}")

        env.close()

    # Overall
    overall_sr = total_successes / max(total_trials, 1)
    results["overall_success_rate"] = overall_sr
    results["total_successes"] = total_successes
    results["total_trials"] = total_trials

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY: LIBERO-Bridge Evaluation")
    print(f"{'='*70}")
    print(f"  {'Task':<30s} {'Equiv':<25s} {'Success':>10s}")
    print(f"  {'-'*65}")
    for tname, tres in results["tasks"].items():
        sr_str = f"{tres['success_rate']:.1%} ({tres['successes']}/{n_episodes})"
        print(f"  {tname:<30s} {tres['robosuite_equivalent']:<25s} {sr_str:>10s}")
    print(f"  {'-'*65}")
    print(f"  {'OVERALL':<55s} {overall_sr:.1%} ({total_successes}/{total_trials})")
    print(f"{'='*70}")

    # Save results
    if output_dir:
        _save_results(results, output_dir)

    return results


def _save_gif(frames, output_dir, task_name, ep, success):
    """Save episode frames as GIF."""
    from PIL import Image as PILImage

    out_dir = Path(output_dir) / "videos" / "libero_bridge"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = "success" if success else "fail"
    gif_path = out_dir / f"{task_name}_ep{ep}_{tag}.gif"

    pil_frames = [PILImage.fromarray(f).resize((256, 256)) for f in frames]
    if pil_frames:
        pil_frames[0].save(
            gif_path, save_all=True, append_images=pil_frames[1:],
            duration=50, loop=0, optimize=True,
        )


def _save_results(results, output_dir):
    """Save results as JSON."""
    out_dir = Path(output_dir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"libero_bridge_{results['task_set']}_seed{results['seed']}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SmolVLA on LIBERO-Bridge tasks (custom BDDL files)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Task sets:
  proxy  — LIBERO objects as stand-ins (bowls, mugs, baskets)
  bridge — Robosuite native objects (cube, nuts) registered in LIBERO
  all    — Both proxy and bridge tasks

Examples:
  python eval/eval_libero_bridge.py --checkpoint outputs/checkpoints/smolvla --task-set proxy
  python eval/eval_libero_bridge.py --checkpoint outputs/checkpoints/smolvla --task-set all --episodes 50
        """,
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task-set", type=str, default="proxy",
                        choices=["proxy", "bridge", "all"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override per-task horizon")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-freq", type=int, default=10,
                        help="Record GIF every N episodes (0=off)")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    evaluate_tasks(
        checkpoint_path=args.checkpoint,
        task_set=args.task_set,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
        record_freq=args.record_freq,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
