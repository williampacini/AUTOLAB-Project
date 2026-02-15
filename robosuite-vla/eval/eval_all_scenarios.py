#!/usr/bin/env python3
"""Comprehensive evaluation: all robosuite tasks + base SmolVLA comparison.

Evaluates THREE configurations side-by-side:
  1. Fine-tuned SmolVLA (WITH postprocessing / action unnormalization)
  2. Fine-tuned SmolVLA (NO postprocessing / raw actions)
  3. Base SmolVLA pretrained weights (lerobot/smolvla_base, NO postprocessing)

Across ALL robosuite single-arm scenarios:
  - Lift, Door, NutAssemblySquare, NutAssemblyRound, NutAssembly,
    PickPlaceSingle, Stack, Wipe

Includes action magnitude diagnostics to identify normalization issues.
Produces a JSON results file + comparison table.

Usage (CLI):
    python eval/eval_all_scenarios.py \
        --finetuned /path/to/your/checkpoint \
        --episodes 20 \
        --max-steps 600

Usage (Colab):
    Paste cells from this file — see "# === COLAB CELL ===" markers.
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

# ---------------------------------------------------------------------------
# Robosuite task definitions
# ---------------------------------------------------------------------------
ROBOSUITE_TASKS = {
    "Lift": {
        "env_name": "Lift",
        "language": "Pick up the cube",
        "horizon": 500,
        "kwargs": {},
    },
    "Door": {
        "env_name": "Door",
        "language": "Open the door",
        "horizon": 500,
        "kwargs": {},
    },
    "NutAssemblySquare": {
        "env_name": "NutAssembly",
        "language": "Assemble the square nut onto the peg",
        "horizon": 1000,
        "kwargs": {"single_object_mode": 2},
    },
    "NutAssemblyRound": {
        "env_name": "NutAssembly",
        "language": "Assemble the round nut onto the peg",
        "horizon": 1000,
        "kwargs": {"single_object_mode": 1},
    },
    "NutAssembly": {
        "env_name": "NutAssembly",
        "language": "Assemble the round nut and square nut onto their respective pegs",
        "horizon": 1000,
        "kwargs": {"single_object_mode": 0},
    },
    "PickPlaceSingle": {
        "env_name": "PickPlace",
        "language": "Pick up the object and place it in the bin",
        "horizon": 1000,
        "kwargs": {"single_object_mode": 2},
    },
    "Stack": {
        "env_name": "Stack",
        "language": "Stack one cube on top of another",
        "horizon": 1000,
        "kwargs": {},
    },
    "Wipe": {
        "env_name": "Wipe",
        "language": "Wipe the dirty surface clean",
        "horizon": 500,
        "kwargs": {},
    },
}


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------
def create_robosuite_env(task_key, camera_size=256):
    """Create a robosuite environment for the given task key."""
    import robosuite as suite

    task = ROBOSUITE_TASKS[task_key]
    env = suite.make(
        env_name=task["env_name"],
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=camera_size,
        camera_widths=camera_size,
        reward_shaping=True,
        horizon=task["horizon"],
        **task["kwargs"],
    )
    return env


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def find_pretrained_dir(checkpoint_path):
    """Resolve the actual pretrained_model directory."""
    p = Path(checkpoint_path)
    if (p / "config.json").exists():
        return p
    candidates = [
        p / "checkpoints" / "last" / "pretrained_model",
        p / "last" / "pretrained_model",
        p / "pretrained_model",
    ]
    for c in candidates:
        if (c / "config.json").exists():
            return c
    configs = list(p.rglob("config.json"))
    if configs:
        return configs[0].parent
    return p


def get_state_dim_from_checkpoint(pretrained_dir):
    """Read expected state dimension from preprocessor stats."""
    try:
        from safetensors.torch import load_file
    except ImportError:
        return None
    for sf in sorted(Path(pretrained_dir).glob("*preprocessor*.safetensors")):
        try:
            tensors = load_file(str(sf))
            for key, tensor in tensors.items():
                if "observation.state" in key and "mean" in key:
                    return tensor.shape[-1]
        except Exception:
            continue
    return None


def load_finetuned_model(checkpoint_path, device="cuda"):
    """Load the fine-tuned SmolVLA with preprocessor + postprocessor.

    Returns: (policy, postprocess_fn, state_dim, tokenizer, model_name)
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from transformers import AutoTokenizer

    pretrained_dir = find_pretrained_dir(checkpoint_path)
    print(f"[fine-tuned] Loading from: {pretrained_dir}")

    state_dim = get_state_dim_from_checkpoint(pretrained_dir)
    policy = SmolVLAPolicy.from_pretrained(pretrained_dir)
    policy.to(device)
    policy.eval()

    # Load postprocessor for action unnormalization
    postprocess_fn = None
    try:
        try:
            from lerobot.policies.factory import make_pre_post_processors
        except ImportError:
            from lerobot.common.policies.factory import make_pre_post_processors

        _pre, postprocess_fn = make_pre_post_processors(
            policy.config,
            str(pretrained_dir),
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        print("[fine-tuned] Postprocessor loaded (action unnormalization ON)")
    except Exception as e:
        print(f"[fine-tuned] WARNING: Could not load postprocessor: {e}")
        print("[fine-tuned] Will use raw actions (no unnormalization)")

    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    )
    model_name = Path(checkpoint_path).name
    print(f"[fine-tuned] state_dim={state_dim}, model={model_name}")

    return policy, postprocess_fn, state_dim, tokenizer, f"finetuned:{model_name}"


def load_base_model(device="cuda"):
    """Load the base SmolVLA pretrained weights — NO postprocessing.

    This is the out-of-the-box model from HuggingFace, evaluated without
    any dataset-specific normalization / unnormalization.

    Returns: (policy, postprocess_fn, state_dim, tokenizer, model_name)
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from transformers import AutoTokenizer

    print("[base] Loading lerobot/smolvla_base from HuggingFace ...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.to(device)
    policy.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    )

    # Base model: no postprocessing, default state_dim = 7
    print("[base] Loaded. No postprocessor (raw actions).")
    return policy, None, 7, tokenizer, "base:smolvla_base"


# ---------------------------------------------------------------------------
# Observation + Action helpers
# ---------------------------------------------------------------------------
def build_state(obs, expected_dim=7):
    """Build state vector: eef_pos(3) + eef_quat(4) [+ gripper_qpos(2)]."""
    parts = [obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
    if expected_dim is not None and expected_dim > 7:
        g = obs.get("robot0_gripper_qpos")
        if g is not None:
            parts.append(g)
    s = np.concatenate(parts).astype(np.float32)
    if expected_dim is not None:
        if len(s) > expected_dim:
            s = s[:expected_dim]
        elif len(s) < expected_dim:
            s = np.pad(s, (0, expected_dim - len(s)))
    return s


def get_action(policy, obs, task_language, tokenizer, device="cuda",
               img_size=256, state_dim=7, postprocess_fn=None):
    """Get action from policy given a robosuite observation.

    If postprocess_fn is provided, actions are unnormalized (fine-tuned path).
    Otherwise, raw actions are used (base model path).

    Returns (clipped_action, raw_action_before_clip) for diagnostics.
    """
    from PIL import Image as PILImage

    # Cameras
    agentview = np.flip(obs["agentview_image"], axis=0)
    agentview_pil = PILImage.fromarray(agentview).resize(
        (img_size, img_size), PILImage.LANCZOS
    )
    wrist = np.flip(obs["robot0_eye_in_hand_image"], axis=0)
    wrist_pil = PILImage.fromarray(wrist).resize(
        (img_size, img_size), PILImage.LANCZOS
    )

    # State
    state = build_state(obs, state_dim)

    # Tokenize
    tokens = tokenizer(
        task_language,
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

    # Postprocess if available (fine-tuned model)
    if postprocess_fn is not None:
        action = postprocess_fn(action)

    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy().flatten()
    elif isinstance(action, dict):
        a = action.get("action", action)
        if isinstance(a, torch.Tensor):
            action = a.cpu().numpy().flatten()
        else:
            action = np.array(a).flatten()

    raw = action[:7].copy()
    return np.clip(raw, -1, 1), raw


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------
def run_episode(env, policy, task_language, tokenizer, device, img_size,
                state_dim, postprocess_fn, max_steps):
    """Run one episode. Returns (success, total_reward, steps, action_stats)."""
    obs = env.reset()
    policy.reset()

    total_reward = 0.0
    success = False
    raw_actions = []

    for step in range(max_steps):
        action, raw = get_action(
            policy, obs, task_language, tokenizer,
            device, img_size, state_dim, postprocess_fn,
        )
        raw_actions.append(raw)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if env._check_success():
            success = True
            break
        if done:
            break

    # Action magnitude diagnostics
    raw_arr = np.array(raw_actions)
    action_stats = {
        "mean_abs": float(np.mean(np.abs(raw_arr))),
        "max_abs": float(np.max(np.abs(raw_arr))),
        "mean_per_dim": np.mean(np.abs(raw_arr), axis=0).tolist(),
    }

    return success, total_reward, step + 1, action_stats


# ---------------------------------------------------------------------------
# Evaluate one model on all tasks
# ---------------------------------------------------------------------------
def evaluate_model_all_tasks(
    policy, postprocess_fn, state_dim, tokenizer, model_name,
    task_keys, n_episodes, max_steps, device, camera_size,
):
    """Evaluate a single model across all specified robosuite tasks.

    Returns a dict of results keyed by task name.
    """
    model_results = {}

    for task_key in task_keys:
        task_info = ROBOSUITE_TASKS[task_key]
        task_lang = task_info["language"]
        horizon = min(task_info["horizon"], max_steps)

        print(f"\n  --- {task_key}: \"{task_lang}\" (horizon={horizon}) ---")

        try:
            env = create_robosuite_env(task_key, camera_size)
        except Exception as e:
            print(f"  SKIP: Could not create env for {task_key}: {e}")
            model_results[task_key] = {
                "task": task_key,
                "language": task_lang,
                "error": str(e),
                "success_rate": 0.0,
                "n_success": 0,
                "n_episodes": n_episodes,
                "episodes": [],
            }
            continue

        task_successes = 0
        episodes = []
        all_action_stats = []

        for ep in range(n_episodes):
            t0 = time.time()
            success, reward, steps, act_stats = run_episode(
                env, policy, task_lang, tokenizer,
                device, camera_size, state_dim, postprocess_fn, horizon,
            )
            elapsed = time.time() - t0

            if success:
                task_successes += 1

            all_action_stats.append(act_stats)
            episodes.append({
                "episode": ep,
                "success": success,
                "reward": float(reward),
                "steps": steps,
                "time_s": round(elapsed, 1),
                "action_mean_abs": act_stats["mean_abs"],
                "action_max_abs": act_stats["max_abs"],
            })

            tag = "OK" if success else "  "
            print(f"    ep {ep:3d}: {tag} R={reward:7.2f} steps={steps:4d} "
                  f"|act|={act_stats['mean_abs']:.4f} max={act_stats['max_abs']:.4f} ({elapsed:.1f}s)")

        env.close()

        # Aggregate action diagnostics
        mean_act_mag = float(np.mean([s["mean_abs"] for s in all_action_stats]))
        max_act_mag = float(np.max([s["max_abs"] for s in all_action_stats]))

        sr = task_successes / n_episodes
        model_results[task_key] = {
            "task": task_key,
            "language": task_lang,
            "success_rate": sr,
            "n_success": task_successes,
            "n_episodes": n_episodes,
            "episodes": episodes,
            "avg_reward": float(np.mean([e["reward"] for e in episodes])),
            "avg_steps": float(np.mean([e["steps"] for e in episodes])),
            "action_diagnostics": {
                "mean_abs": mean_act_mag,
                "max_abs": max_act_mag,
            },
        }
        print(f"  => {task_key}: {task_successes}/{n_episodes} ({sr:.1%})  "
              f"action magnitude: avg={mean_act_mag:.4f} max={max_act_mag:.4f}")

    return model_results


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def print_comparison_table(all_results, task_keys):
    """Print a side-by-side comparison table of all models."""

    model_names = list(all_results.keys())
    col_w = 26
    header = f"{'Task':<22}"
    for m in model_names:
        header += f" | {m:^{col_w}}"
    sep = "=" * len(header)

    print(f"\n{sep}")
    print("  COMPARISON: Success Rate | Avg Reward | Action Magnitude")
    print(sep)
    print(header)
    print("-" * len(header))

    for task_key in task_keys:
        row = f"{task_key:<22}"
        for m in model_names:
            r = all_results[m].get(task_key)
            if r is None or "error" in r:
                row += f" | {'SKIP':^{col_w}}"
            else:
                sr = r["success_rate"]
                ns = r["n_success"]
                ne = r["n_episodes"]
                avg_r = r["avg_reward"]
                act_m = r.get("action_diagnostics", {}).get("mean_abs", 0)
                cell = f"{sr:.0%}({ns}/{ne}) R={avg_r:.1f} |a|={act_m:.3f}"
                row += f" | {cell:^{col_w}}"
        print(row)

    # Overall
    print("-" * len(header))
    row = f"{'OVERALL':<22}"
    for m in model_names:
        total_s = sum(r["n_success"] for r in all_results[m].values() if "error" not in r)
        total_e = sum(r["n_episodes"] for r in all_results[m].values() if "error" not in r)
        avg_act = np.mean([
            r.get("action_diagnostics", {}).get("mean_abs", 0)
            for r in all_results[m].values() if "error" not in r
        ]) if total_e > 0 else 0
        if total_e > 0:
            overall_sr = total_s / total_e
            cell = f"{overall_sr:.0%}({total_s}/{total_e}) |a|={avg_act:.3f}"
        else:
            cell = "N/A"
        row += f" | {cell:^{col_w}}"
    print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def dump_postprocessor_stats(checkpoint_path):
    """Print the action normalization stats from the postprocessor.

    These stats reveal whether unnormalization is collapsing action magnitudes.
    """
    pretrained_dir = find_pretrained_dir(checkpoint_path)
    try:
        from safetensors.torch import load_file
    except ImportError:
        print("  [diag] safetensors not installed, cannot dump stats")
        return

    print("\n  [DIAGNOSTICS] Postprocessor normalization statistics:")
    for sf in sorted(pretrained_dir.glob("*postprocessor*.safetensors")):
        tensors = load_file(str(sf))
        for key in sorted(tensors.keys()):
            t = tensors[key]
            print(f"    {key}: shape={list(t.shape)} values={t.cpu().numpy().tolist()}")

    print("  [DIAGNOSTICS] Preprocessor state statistics:")
    for sf in sorted(pretrained_dir.glob("*preprocessor*.safetensors")):
        tensors = load_file(str(sf))
        for key in sorted(tensors.keys()):
            if "state" in key or "action" in key.lower():
                t = tensors[key]
                print(f"    {key}: shape={list(t.shape)} values={t.cpu().numpy().tolist()}")
    print()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def run_full_evaluation(
    finetuned_ckpt=None,
    task_keys=None,
    n_episodes=10,
    max_steps=600,
    device="cuda",
    camera_size=256,
    output_dir="outputs",
    skip_base=False,
):
    """Run the full comparison evaluation.

    Args:
        finetuned_ckpt: Path to fine-tuned checkpoint (None to skip)
        task_keys: List of task keys from ROBOSUITE_TASKS (None = all)
        n_episodes: Episodes per task per model
        max_steps: Max steps per episode
        device: torch device
        camera_size: Camera resolution
        output_dir: Where to save results JSON
        skip_base: If True, skip the base model evaluation
    """
    if task_keys is None:
        task_keys = list(ROBOSUITE_TASKS.keys())

    all_results = {}

    # ── 1. Fine-tuned model WITH postprocessing ──
    if finetuned_ckpt is not None:
        print("\n" + "=" * 70)
        print("  MODEL 1: Fine-tuned SmolVLA (WITH postprocessing)")
        print("=" * 70)
        policy, postproc, sdim, tok, name = load_finetuned_model(
            finetuned_ckpt, device
        )

        # Print postprocessor normalization stats for diagnosis
        dump_postprocessor_stats(finetuned_ckpt)

        results = evaluate_model_all_tasks(
            policy, postproc, sdim, tok, name,
            task_keys, n_episodes, max_steps, device, camera_size,
        )
        all_results["ft+postproc"] = results

        # ── 2. Same fine-tuned model WITHOUT postprocessing ──
        print("\n" + "=" * 70)
        print("  MODEL 2: Fine-tuned SmolVLA (NO postprocessing / raw actions)")
        print("=" * 70)
        results_raw = evaluate_model_all_tasks(
            policy, None, sdim, tok, name,  # postprocess_fn=None
            task_keys, n_episodes, max_steps, device, camera_size,
        )
        all_results["ft-raw"] = results_raw

        del policy
        torch.cuda.empty_cache()

    # ── 3. Base SmolVLA (no postprocessing) ──
    if not skip_base:
        print("\n" + "=" * 70)
        print("  MODEL 3: Base SmolVLA (NO postprocessing / raw pretrained)")
        print("=" * 70)
        policy, postproc, sdim, tok, name = load_base_model(device)
        results = evaluate_model_all_tasks(
            policy, postproc, sdim, tok, name,
            task_keys, n_episodes, max_steps, device, camera_size,
        )
        all_results["base-raw"] = results

        del policy
        torch.cuda.empty_cache()

    # ── 3. Print comparison ──
    print_comparison_table(all_results, task_keys)

    # ── 4. Save results ──
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "all_scenarios_comparison.json"

    # Make JSON-serializable
    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_episodes": n_episodes,
        "max_steps": max_steps,
        "task_keys": task_keys,
        "models": {},
    }
    for model_label, model_res in all_results.items():
        save_data["models"][model_label] = model_res

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved: {results_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned vs base SmolVLA on all robosuite tasks"
    )
    parser.add_argument(
        "--finetuned", type=str, default=None,
        help="Path to fine-tuned checkpoint (skip if not provided)",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=None,
        choices=list(ROBOSUITE_TASKS.keys()),
        help="Specific tasks to evaluate (default: all)",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--skip-base", action="store_true",
        help="Skip the base model evaluation",
    )
    args = parser.parse_args()

    run_full_evaluation(
        finetuned_ckpt=args.finetuned,
        task_keys=args.tasks,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        device=args.device,
        camera_size=args.camera_size,
        output_dir=args.output_dir,
        skip_base=args.skip_base,
    )


if __name__ == "__main__":
    main()
