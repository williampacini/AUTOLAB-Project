#!/usr/bin/env python3
"""End-to-end robosuite SmolVLA pipeline.

Runs the complete workflow:
  1. Collect scripted expert demos from robosuite environments
  2. Convert demos to LeRobot v3.0 format
  3. Post-train SmolVLA on the demo dataset
  4. Evaluate on nut assembly tasks with video recording of EVERY episode

Usage:
    # Full pipeline (all envs, train, eval nut assembly with videos)
    python run_pipeline.py

    # Skip collection + conversion if already done
    python run_pipeline.py --skip-collect --skip-convert

    # Train only (data already converted)
    python run_pipeline.py --skip-collect --skip-convert --skip-eval

    # Eval only (model already trained)
    python run_pipeline.py --skip-collect --skip-convert --skip-train

    # Quick test (2 demos, 100 training steps, 2 eval episodes)
    python run_pipeline.py --quick-test

    # Custom demo counts
    python run_pipeline.py --n-demos 25 --nut-assembly-demos 50 --steps 30000
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Default paths
DEMO_DIR = ROOT / "data" / "robosuite_demos"
LEROBOT_DIR = ROOT / "outputs" / "lerobot" / "robosuite_multitask"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "smolvla_robosuite"
OUTPUT_DIR = ROOT / "outputs"


def run_cmd(cmd, description, check=True):
    """Run a shell command with nice formatting."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  $ {' '.join(cmd)}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0

    if check and result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.0f}s")
        sys.exit(result.returncode)
    else:
        print(f"\n  Done in {elapsed:.0f}s")

    return result.returncode


def step_collect(n_demos=50, nut_assembly_demos=75, camera_res=128):
    """Step 1: Collect scripted expert demonstrations."""
    # Collect all envs except NutAssembly with standard count
    for env_name in ["Lift", "Stack", "PickPlaceSingle", "Door", "NutAssemblySingle"]:
        run_cmd(
            [sys.executable, "data/collect_demos.py",
             "--env", env_name,
             "--n-demos", str(n_demos),
             "--output-dir", str(DEMO_DIR),
             "--camera-res", str(camera_res)],
            f"Collecting {n_demos} demos for {env_name}",
        )

    # NutAssembly gets more demos
    run_cmd(
        [sys.executable, "data/collect_demos.py",
         "--env", "NutAssembly",
         "--n-demos", str(nut_assembly_demos),
         "--output-dir", str(DEMO_DIR),
         "--camera-res", str(camera_res)],
        f"Collecting {nut_assembly_demos} demos for NutAssembly",
    )


def step_convert(image_size=256):
    """Step 2: Convert HDF5 demos to LeRobot format."""
    run_cmd(
        [sys.executable, "data/convert_to_lerobot.py",
         "--source", "robosuite",
         "--hdf5-dir", str(DEMO_DIR),
         "--output-dir", str(LEROBOT_DIR),
         "--image-size", str(image_size)],
        "Converting robosuite demos to LeRobot format",
    )


def step_train(steps=50000, batch_size=4):
    """Step 3: Post-train SmolVLA on robosuite demos."""
    run_cmd(
        [sys.executable, "train/train_smolvla.py",
         "--config", "config/robosuite_multitask.yaml",
         "--steps", str(steps),
         "--batch-size", str(batch_size)],
        f"Training SmolVLA ({steps} steps, batch_size={batch_size})",
    )

    # Post-training fix: set n_action_steps = 50
    run_cmd(
        [sys.executable, "train/train_smolvla.py",
         "--fix-config", str(CHECKPOINT_DIR)],
        "Fixing n_action_steps in checkpoint config",
    )


def step_eval(episodes=50, device="cuda", eval_all=False):
    """Step 4: Evaluate on nut assembly with video of every episode."""

    # Always evaluate nut assembly tasks with full video recording
    nut_cmd = [
        sys.executable, "eval/eval_robosuite.py",
        "--checkpoint", str(CHECKPOINT_DIR),
        "--nut-assembly",
        "--episodes", str(episodes),
        "--record-all",
        "--device", device,
        "--output-dir", str(OUTPUT_DIR),
    ]
    run_cmd(nut_cmd, f"Evaluating nut assembly ({episodes} episodes, recording ALL videos)")

    # Optionally evaluate all other tasks too
    if eval_all:
        all_cmd = [
            sys.executable, "eval/eval_robosuite.py",
            "--checkpoint", str(CHECKPOINT_DIR),
            "--all",
            "--episodes", str(episodes),
            "--record-all",
            "--device", device,
            "--output-dir", str(OUTPUT_DIR),
        ]
        run_cmd(all_cmd, f"Evaluating ALL tasks ({episodes} episodes, recording ALL videos)")


def print_summary():
    """Print final summary with paths to results and videos."""
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Demos:       {DEMO_DIR}/")
    print(f"  LeRobot:     {LEROBOT_DIR}/")
    print(f"  Checkpoint:  {CHECKPOINT_DIR}/")
    print(f"  Videos:      {OUTPUT_DIR}/videos/robosuite/")
    print(f"  Results:     {OUTPUT_DIR}/results/")

    # Print eval results if they exist
    results_file = OUTPUT_DIR / "results" / "robosuite_eval.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        print(f"\n  Evaluation Results:")
        for t in results.get("tasks", []):
            print(f"    {t['env_name']:20s}  {t['success_rate']:.1%}  "
                  f"({t['successes']}/{t['n_episodes']})")
        print(f"    {'OVERALL':20s}  {results.get('overall_success_rate', 0):.1%}")

    # Count video files
    video_dir = OUTPUT_DIR / "videos" / "robosuite"
    if video_dir.exists():
        mp4s = list(video_dir.rglob("*.mp4"))
        gifs = list(video_dir.rglob("*.gif"))
        print(f"\n  Videos: {len(mp4s)} MP4s, {len(gifs)} GIFs")
        for env_dir in sorted(video_dir.iterdir()):
            if env_dir.is_dir():
                env_mp4s = list(env_dir.glob("*.mp4"))
                env_successes = [p for p in env_mp4s if "success" in p.name]
                print(f"    {env_dir.name}/: {len(env_mp4s)} videos "
                      f"({len(env_successes)} successes)")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end robosuite SmolVLA pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                     # Full pipeline
  python run_pipeline.py --quick-test        # Quick smoke test
  python run_pipeline.py --skip-collect --skip-convert --skip-train  # Eval only
        """,
    )

    # Skip stages
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip demo collection (use existing HDF5 files)")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip LeRobot conversion (use existing dataset)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (use existing checkpoint)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation")

    # Demo collection
    parser.add_argument("--n-demos", type=int, default=50,
                        help="Demos per environment (default: 50)")
    parser.add_argument("--nut-assembly-demos", type=int, default=75,
                        help="Demos for NutAssembly (default: 75)")

    # Training
    parser.add_argument("--steps", type=int, default=50000,
                        help="Training steps (default: 50000)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size (default: 4)")

    # Evaluation
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Eval episodes per task (default: 50)")
    parser.add_argument("--eval-all", action="store_true",
                        help="Evaluate ALL tasks (default: nut assembly only)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for training/eval")

    # Quick test mode
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick smoke test: 2 demos, 100 steps, 2 eval episodes")

    args = parser.parse_args()

    # Quick test overrides
    if args.quick_test:
        args.n_demos = 2
        args.nut_assembly_demos = 2
        args.steps = 100
        args.eval_episodes = 2
        args.eval_all = False
        print("QUICK TEST MODE: 2 demos, 100 training steps, 2 eval episodes")

    t_pipeline_start = time.time()

    # Step 1: Collect demos
    if not args.skip_collect:
        step_collect(
            n_demos=args.n_demos,
            nut_assembly_demos=args.nut_assembly_demos,
        )
    else:
        print("\n  Skipping demo collection (--skip-collect)")

    # Step 2: Convert to LeRobot format
    if not args.skip_convert:
        step_convert()
    else:
        print("\n  Skipping conversion (--skip-convert)")

    # Step 3: Train SmolVLA
    if not args.skip_train:
        step_train(steps=args.steps, batch_size=args.batch_size)
    else:
        print("\n  Skipping training (--skip-train)")

    # Step 4: Evaluate on nut assembly with video recording
    if not args.skip_eval:
        step_eval(
            episodes=args.eval_episodes,
            device=args.device,
            eval_all=args.eval_all,
        )
    else:
        print("\n  Skipping evaluation (--skip-eval)")

    total_time = time.time() - t_pipeline_start
    print(f"\n  Total pipeline time: {total_time / 60:.1f} minutes")

    print_summary()


if __name__ == "__main__":
    main()
