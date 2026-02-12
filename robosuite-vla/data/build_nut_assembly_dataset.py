#!/usr/bin/env python3
"""Orchestrate the full NutAssembly dataset build pipeline.

Combines:
  - 3/4 from HuggingFace: ~75 NutAssemblySquare + ~75 NutAssemblyRound demos (robomimic)
  - 1/4 from scratch: ~50 demos via scripted policy on full NutAssembly

Outputs a local LeRobot-format dataset at outputs/lerobot/nut_assembly/

Usage:
    python data/build_nut_assembly_dataset.py
    python data/build_nut_assembly_dataset.py --skip-collect    # Skip scripted demo collection
    python data/build_nut_assembly_dataset.py --skip-download   # Skip HF download
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TASK_LANGUAGE = "Assemble the round nut and square nut onto their respective pegs"
DEFAULT_OUTPUT = ROOT / "outputs" / "lerobot" / "nut_assembly"
SCRIPTED_HDF5 = ROOT / "data" / "nut_assembly_scripted" / "demos.hdf5"
ROBOMIMIC_DIR = ROOT / "data" / "robomimic_nut_assembly"


def step_download_hf(max_demos_per_variant=75):
    """Step 1: Download NutAssembly HDF5 from HuggingFace."""
    print("\n" + "=" * 60)
    print("  Step 1: Download HuggingFace Data")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(ROOT / "data" / "download_nut_assembly.py"),
        "--output-dir", str(ROBOMIMIC_DIR),
        "--max-demos", str(max_demos_per_variant),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("WARNING: HuggingFace download had issues. Check output above.")
        print("Continuing with whatever data is available...")
    return result.returncode == 0


def step_collect_scripted(num_demos=50, noise_std=0.01):
    """Step 2: Collect demonstrations via scripted policy."""
    print("\n" + "=" * 60)
    print("  Step 2: Collect Scripted Demos")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(ROOT / "data" / "collect_nut_assembly.py"),
        "--num-demos", str(num_demos),
        "--output", str(SCRIPTED_HDF5),
        "--noise-std", str(noise_std),
        "--visualize",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("WARNING: Scripted demo collection had issues.")
    return result.returncode == 0


def step_convert_and_combine(output_dir, image_size=256, max_demos_per_file=None):
    """Step 3: Convert all HDF5 sources to LeRobot format."""
    print("\n" + "=" * 60)
    print("  Step 3: Convert & Combine into LeRobot Format")
    print("=" * 60)

    # Collect all HDF5 files
    hdf5_files = []

    # Add HuggingFace downloads
    if ROBOMIMIC_DIR.exists():
        hf_files = sorted(ROBOMIMIC_DIR.rglob("*.hdf5"))
        hdf5_files.extend(hf_files)
        print(f"  HuggingFace HDF5 files: {len(hf_files)}")
    else:
        print("  No HuggingFace data directory found.")

    # Add scripted demos
    if SCRIPTED_HDF5.exists():
        hdf5_files.append(SCRIPTED_HDF5)
        print(f"  Scripted HDF5: {SCRIPTED_HDF5}")
    else:
        print("  No scripted demo file found.")

    if not hdf5_files:
        print("ERROR: No HDF5 files found to convert!")
        return False

    print(f"  Total HDF5 files: {len(hdf5_files)}")

    # Build the conversion command
    cmd = [
        sys.executable,
        str(ROOT / "data" / "convert_to_lerobot.py"),
        "--hdf5",
    ] + [str(f) for f in hdf5_files] + [
        "--task-name", "nut_assembly",
        "--task-language", TASK_LANGUAGE,
        "--output-dir", str(output_dir),
        "--image-size", str(image_size),
        "--include-wrist",
    ]

    if max_demos_per_file:
        cmd.extend(["--max-demos", str(max_demos_per_file)])

    result = subprocess.run(cmd)
    return result.returncode == 0


def step_verify(output_dir):
    """Step 4: Verify the built dataset."""
    print("\n" + "=" * 60)
    print("  Step 4: Verify Dataset")
    print("=" * 60)

    output_dir = Path(output_dir)

    # Check directory structure
    expected_dirs = ["data", "videos", "meta"]
    for d in expected_dirs:
        p = output_dir / d
        exists = p.exists()
        print(f"  {d}/: {'OK' if exists else 'MISSING'}")

    # Check parquet
    parquet_files = list((output_dir / "data").rglob("*.parquet")) if (output_dir / "data").exists() else []
    print(f"  Parquet files: {len(parquet_files)}")

    # Check videos
    video_files = list((output_dir / "videos").rglob("*.mp4")) if (output_dir / "videos").exists() else []
    print(f"  Video files: {len(video_files)}")

    # Check metadata
    meta_path = output_dir / "meta" / "info.json"
    if meta_path.exists():
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Episodes: {meta.get('n_episodes', '?')}")
        print(f"  Total frames: {meta.get('total_frames', '?')}")
        print(f"  Image size: {meta.get('image_size', '?')}")
        print(f"  Cameras: {meta.get('cameras', '?')}")
    else:
        print("  meta/info.json: MISSING")

    print()
    return len(parquet_files) > 0


def main():
    parser = argparse.ArgumentParser(description="Build NutAssembly LeRobot dataset")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output directory for LeRobot dataset")
    parser.add_argument("--scripted-demos", type=int, default=50,
                        help="Number of scripted demos to collect")
    parser.add_argument("--hf-demos", type=int, default=75,
                        help="Max demos per HF variant (Square/Round)")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Output image resolution")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip HuggingFace download step")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip scripted demo collection step")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip conversion step (verify only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("  NutAssembly Dataset Build Pipeline")
    print("=" * 60)
    print(f"  Output:         {output_dir}")
    print(f"  HF demos:       {args.hf_demos}/variant (Square+Round)")
    print(f"  Scripted demos: {args.scripted_demos}")
    print(f"  Total target:   ~{args.hf_demos * 2 + args.scripted_demos} demos")
    print(f"  Task:           {TASK_LANGUAGE}")

    # Step 1: Download
    if not args.skip_download:
        step_download_hf(max_demos_per_variant=args.hf_demos)
    else:
        print("\n  [SKIPPED] HuggingFace download")

    # Step 2: Collect
    if not args.skip_collect:
        step_collect_scripted(num_demos=args.scripted_demos)
    else:
        print("\n  [SKIPPED] Scripted demo collection")

    # Step 3: Convert
    if not args.skip_convert:
        success = step_convert_and_combine(output_dir, image_size=args.image_size)
        if not success:
            print("\nERROR: Dataset conversion failed!")
            sys.exit(1)
    else:
        print("\n  [SKIPPED] Conversion")

    # Step 4: Verify
    step_verify(output_dir)

    print("=" * 60)
    print("  Dataset build complete!")
    print(f"  Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
