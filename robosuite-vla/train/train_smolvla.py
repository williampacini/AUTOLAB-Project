#!/usr/bin/env python3
"""Fine-tune SmolVLA on LIBERO or robosuite demonstration data via LeRobot.

Usage:
    # LIBERO (HuggingFace dataset)
    python train/train_smolvla.py --config config/libero_spatial.yaml
    python train/train_smolvla.py --suite libero_spatial --steps 20000

    # Robosuite (local dataset after convert_to_lerobot.py)
    python train/train_smolvla.py --config config/robosuite_multitask.yaml
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path):
    """Load and merge config with defaults."""
    default_path = ROOT / "config" / "default.yaml"

    with open(default_path) as f:
        config = yaml.safe_load(f)

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            override = yaml.safe_load(f)
        _deep_update(config, override)

    return config


def _deep_update(base, override):
    """Recursively update base dict with override."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def check_lerobot_installed():
    """Verify LeRobot is installed with SmolVLA support."""
    try:
        import lerobot
        print(f"LeRobot version: {lerobot.__version__}")
        return True
    except ImportError:
        print("LeRobot not installed. Install with:")
        print("  git clone https://github.com/huggingface/lerobot.git")
        print('  cd lerobot && pip install -e ".[smolvla]"')
        return False


def check_gpu():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            return True
        else:
            print("WARNING: No GPU detected. Training will be very slow.")
            return False
    except ImportError:
        print("PyTorch not installed")
        return False


def train_with_lerobot_cli(config):
    """Launch training using LeRobot's CLI."""
    training = config["training"]
    data = config.get("data", {})
    logging_cfg = config.get("logging", {})

    # Build the LeRobot training command (lerobot-train CLI, LeRobot v0.4+)
    # Use --policy.type=smolvla to auto-infer camera features from dataset.
    # Do NOT use --policy.path=lerobot/smolvla_base (expects 3 cameras, fails with LIBERO's 2).
    cmd = [
        "lerobot-train",
        "--policy.type=smolvla",
        "--policy.load_vlm_weights=true",
        f"--batch_size={training['batch_size']}",
        f"--steps={training['total_steps']}",
        f"--output_dir={training.get('output_dir', 'outputs/checkpoints/smolvla')}",
        f"--save_freq={training['save_freq']}",
        f"--eval_freq={training.get('eval_freq', 2000)}",
        f"--log_freq={logging_cfg.get('log_freq', 100)}",
        f"--seed={training['seed']}",
    ]

    # Dataset source: HuggingFace Hub repo OR local directory
    hf_repo = data.get("hf_repo")
    local_dir = data.get("local_dir")

    if hf_repo:
        cmd.append(f"--dataset.repo_id={hf_repo}")
    elif local_dir:
        # Local dataset (e.g., robosuite demos converted to LeRobot format)
        local_path = Path(local_dir)
        if not local_path.is_absolute():
            local_path = ROOT / local_path
        cmd.append(f"--dataset.root={local_path}")
        # When using local root, repo_id is used as the dataset name
        cmd.append("--dataset.repo_id=robosuite_multitask")
    else:
        # Fallback
        cmd.append("--dataset.repo_id=HuggingFaceVLA/libero")

    # GPU device
    cmd.append("--policy.device=cuda")

    # Mixed precision
    if training.get("mixed_precision"):
        cmd.append(f"--policy.dtype={training['mixed_precision']}")

    # Wandb logging
    if logging_cfg.get("wandb_project"):
        cmd.append("--wandb.enable=true")
        cmd.append(f"--wandb.project={logging_cfg['wandb_project']}")
        if logging_cfg.get("wandb_entity"):
            cmd.append(f"--wandb.entity={logging_cfg['wandb_entity']}")

    print("\n" + "=" * 60)
    print("  SmolVLA Training")
    print("=" * 60)
    print(f"  Model: {training['model']}")
    print(f"  Dataset: {data.get('hf_repo', 'N/A')}")
    print(f"  Steps: {training['total_steps']}")
    print(f"  Batch size: {training['batch_size']}")
    print(f"  Output: {training.get('output_dir', 'outputs/checkpoints/smolvla')}")
    print()

    print("Command:")
    print("  " + " \\\n    ".join(cmd))
    print()

    result = subprocess.run(cmd)
    return result.returncode


def train_manual(config):
    """Manual training loop (fallback if LeRobot CLI doesn't work)."""
    print("Manual training loop — for custom training setups")
    print("This requires LeRobot's Python API. Example:")
    print()
    print("  from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy")
    print("  from lerobot.datasets.lerobot_dataset import LeRobotDataset")
    print()
    print("  dataset = LeRobotDataset(repo_id)")
    print("  policy = SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')")
    print("  # ... training loop ...")
    print()
    print("See: https://github.com/huggingface/lerobot/tree/main/examples")


def fix_n_action_steps(checkpoint_dir):
    """Post-training fix: set n_action_steps to 50 in config.json."""
    import json

    config_path = Path(checkpoint_dir) / "config.json"
    if not config_path.exists():
        # Try finding it in subdirectories
        for p in Path(checkpoint_dir).rglob("config.json"):
            config_path = p
            break

    if not config_path.exists():
        print(f"WARNING: config.json not found in {checkpoint_dir}")
        print("You must manually set n_action_steps=50 before inference.")
        return

    with open(config_path) as f:
        cfg = json.load(f)

    if cfg.get("n_action_steps") != 50:
        cfg["n_action_steps"] = 50
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Fixed n_action_steps=50 in {config_path}")
    else:
        print(f"n_action_steps already set to 50 in {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA on LIBERO or robosuite")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--suite", type=str, default=None, help="LIBERO suite name (loads config/<suite>.yaml)")
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")
    parser.add_argument("--fix-config", type=str, default=None, help="Fix n_action_steps in a checkpoint dir")
    args = parser.parse_args()

    # Post-training config fix
    if args.fix_config:
        fix_n_action_steps(args.fix_config)
        return

    # Load config
    config_path = args.config
    if args.suite and not config_path:
        config_path = ROOT / "config" / f"{args.suite}.yaml"

    config = load_config(config_path)

    # Apply overrides
    if args.steps:
        config["training"]["total_steps"] = args.steps
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    # Pre-flight checks
    if not check_lerobot_installed():
        return
    check_gpu()

    if args.dry_run:
        print("\n[DRY RUN] Would run training with above config")
        return

    # Train
    returncode = train_with_lerobot_cli(config)

    if returncode == 0:
        # Fix n_action_steps after training
        output_dir = config["training"].get("output_dir", "outputs/checkpoints/smolvla")
        fix_n_action_steps(output_dir)
        print("\nTraining complete!")
    else:
        print(f"\nTraining failed with code {returncode}")
        sys.exit(returncode)


if __name__ == "__main__":
    main()
