#!/usr/bin/env python3
"""Inspect and visualize LIBERO demonstration data.

Supports both HDF5 format (original LIBERO) and LeRobot format (parquet + mp4).
Generates GIF previews and prints statistics.
"""

import argparse
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "libero_datasets"


def inspect_hdf5(hdf5_path, demo_idx=0, save_gif=True, output_dir=None):
    """Inspect a single HDF5 demo file."""
    import h5py

    print(f"\nInspecting: {hdf5_path}")
    print("-" * 60)

    with h5py.File(hdf5_path, "r") as f:
        # List all demos
        if "data" in f:
            demo_keys = sorted(f["data"].keys())
            print(f"Total demos: {len(demo_keys)}")

            if demo_idx >= len(demo_keys):
                print(f"Demo index {demo_idx} out of range (max {len(demo_keys) - 1})")
                return

            demo = f["data"][demo_keys[demo_idx]]
        else:
            demo = f
            demo_keys = ["root"]

        # Print structure
        print(f"\nDemo: {demo_keys[demo_idx]}")

        if "attrs" in dir(demo):
            for attr_name in demo.attrs:
                print(f"  attr/{attr_name}: {demo.attrs[attr_name]}")

        def print_tree(group, prefix="  "):
            for key in sorted(group.keys()):
                item = group[key]
                if hasattr(item, "shape"):
                    print(f"{prefix}{key}: shape={item.shape}, dtype={item.dtype}")
                else:
                    print(f"{prefix}{key}/")
                    print_tree(item, prefix + "  ")

        print_tree(demo)

        # Extract episode statistics
        if "actions" in demo:
            actions = demo["actions"][:]
            print(f"\nAction stats:")
            print(f"  Length: {len(actions)} steps ({len(actions)/20:.1f}s at 20Hz)")
            print(f"  Mean: {actions.mean(axis=0).round(4)}")
            print(f"  Std:  {actions.std(axis=0).round(4)}")
            print(f"  Min:  {actions.min(axis=0).round(4)}")
            print(f"  Max:  {actions.max(axis=0).round(4)}")

            # Gripper analysis
            gripper = actions[:, -1]
            n_close = (gripper > 0).sum()
            n_open = (gripper <= 0).sum()
            print(f"  Gripper: {n_close} close, {n_open} open actions")

        # Save GIF preview
        if save_gif and "obs" in demo:
            cam_key = None
            for k in ["agentview_image", "agentview_rgb"]:
                if k in demo["obs"]:
                    cam_key = k
                    break

            if cam_key:
                images = demo["obs"][cam_key][:]
                _save_gif(images, hdf5_path, demo_idx, output_dir)
            else:
                print(f"  No camera images found in obs/ (keys: {list(demo['obs'].keys())})")


def inspect_lerobot(suite_dir, task_idx=0, demo_idx=0, save_gif=True, output_dir=None):
    """Inspect LeRobot format data (parquet + mp4)."""
    import pandas as pd

    print(f"\nInspecting LeRobot data: {suite_dir}")
    print("-" * 60)

    parquet_files = sorted(suite_dir.rglob("*.parquet"))
    mp4_files = sorted(suite_dir.rglob("*.mp4"))

    print(f"  Parquet files: {len(parquet_files)}")
    print(f"  MP4 files: {len(mp4_files)}")

    if parquet_files:
        df = pd.read_parquet(parquet_files[0])
        print(f"\n  Parquet schema ({parquet_files[0].name}):")
        for col in df.columns:
            print(f"    {col}: dtype={df[col].dtype}, shape example={df[col].iloc[0] if len(df) > 0 else 'empty'}")
        print(f"  Total rows: {len(df)}")

    # Check metadata
    meta_file = suite_dir / "meta" / "info.json"
    if meta_file.exists():
        import json
        with open(meta_file) as mf:
            meta = json.load(mf)
        print(f"\n  Metadata:")
        for k, v in meta.items():
            print(f"    {k}: {v}")


def _save_gif(images, source_path, demo_idx, output_dir):
    """Save camera images as a GIF."""
    from PIL import Image as PILImage

    output_dir = output_dir or (ROOT / "outputs" / "videos")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = Path(source_path).stem
    gif_path = output_dir / f"{name}_demo{demo_idx}.gif"

    # Flip images (MuJoCo renders upside-down) and resize
    frames = []
    step = max(1, len(images) // 100)  # Cap at ~100 frames for reasonable GIF size
    for i in range(0, len(images), step):
        img = np.flip(images[i], axis=0)
        pil_img = PILImage.fromarray(img).resize((256, 256))
        frames.append(pil_img)

    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0,
            optimize=True,
        )
        print(f"  GIF saved: {gif_path} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Inspect LIBERO demo data")
    parser.add_argument("--suite", type=str, default="libero_spatial", help="Suite name")
    parser.add_argument("--task", type=int, default=0, help="Task index")
    parser.add_argument("--demo", type=int, default=0, help="Demo index within task")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for GIFs")
    args = parser.parse_args()

    suite_dir = DATA_DIR / args.suite

    if not suite_dir.exists():
        print(f"Suite directory not found: {suite_dir}")
        print(f"Run: python data/download_libero.py --suite {args.suite}")
        return

    # Try HDF5 files first
    hdf5_files = sorted(suite_dir.rglob("*.hdf5"))
    if hdf5_files:
        if args.task < len(hdf5_files):
            inspect_hdf5(
                hdf5_files[args.task],
                demo_idx=args.demo,
                save_gif=not args.no_gif,
                output_dir=args.output_dir,
            )
        else:
            print(f"Task {args.task} out of range. Found {len(hdf5_files)} HDF5 files:")
            for i, f in enumerate(hdf5_files):
                print(f"  [{i}] {f.name}")
        return

    # Try LeRobot format
    parquet_files = sorted(suite_dir.rglob("*.parquet"))
    if parquet_files:
        inspect_lerobot(
            suite_dir,
            task_idx=args.task,
            demo_idx=args.demo,
            save_gif=not args.no_gif,
            output_dir=args.output_dir,
        )
        return

    print(f"No data files found in {suite_dir}")
    print("Contents:")
    for p in sorted(suite_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
