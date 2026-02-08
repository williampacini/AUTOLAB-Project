#!/usr/bin/env python3
"""Convert LIBERO HDF5 demonstrations to LeRobot dataset format.

LeRobot datasets use:
  - Parquet files for state/action data
  - MP4 videos for camera observations
  - HuggingFace dataset metadata

This script converts LIBERO's HDF5 format and optionally uploads to HuggingFace Hub.
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "libero_datasets"


def get_task_language(hdf5_path):
    """Extract natural language task description from LIBERO HDF5."""
    with h5py.File(hdf5_path, "r") as f:
        # LIBERO stores task description in file attributes
        if "task_description" in f.attrs:
            return str(f.attrs["task_description"])
        if "problem_info" in f.attrs:
            info = f.attrs["problem_info"]
            if isinstance(info, bytes):
                info = info.decode()
            return info

    # Fall back to filename-based description
    name = Path(hdf5_path).stem
    return name.replace("_", " ")


def convert_hdf5_to_lerobot(hdf5_path, output_dir, image_size=256, task_language=None):
    """Convert a single LIBERO HDF5 file to LeRobot-compatible format.

    Returns list of episode dicts with keys:
      - episode_index, task, observations, actions, etc.
    """
    from PIL import Image as PILImage

    task_language = task_language or get_task_language(hdf5_path)
    episodes = []

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            print(f"  [WARN] No 'data' group in {hdf5_path}")
            return []

        demo_keys = sorted(f["data"].keys())
        print(f"  Task: {task_language}")
        print(f"  Demos: {len(demo_keys)}")

        for demo_key in tqdm(demo_keys, desc=f"  Converting {Path(hdf5_path).stem}"):
            demo = f["data"][demo_key]
            actions = demo["actions"][:]
            n_steps = len(actions)

            # Extract camera images
            images = None
            for cam_key in ["agentview_image", "agentview_rgb"]:
                if cam_key in demo["obs"]:
                    images = demo["obs"][cam_key][:]
                    break

            # Extract state observations
            state_keys = {}
            for key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos"]:
                if key in demo["obs"]:
                    state_keys[key] = demo["obs"][key][:]

            # Build episode data
            episode_data = {
                "task": task_language,
                "n_steps": n_steps,
                "actions": actions,
                "states": state_keys,
            }

            # Process and save images
            if images is not None:
                processed_frames = []
                for frame in images:
                    # Flip (MuJoCo renders upside-down)
                    frame = np.flip(frame, axis=0).copy()
                    # Resize if needed
                    if frame.shape[0] != image_size or frame.shape[1] != image_size:
                        pil_img = PILImage.fromarray(frame).resize(
                            (image_size, image_size), PILImage.LANCZOS
                        )
                        frame = np.array(pil_img)
                    processed_frames.append(frame)
                episode_data["images"] = np.stack(processed_frames)

            episodes.append(episode_data)

    return episodes


def save_lerobot_format(episodes, output_dir, task_name):
    """Save episodes in LeRobot-compatible format."""
    import pandas as pd

    output_dir = Path(output_dir)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    all_rows = []
    episode_lengths = []

    for ep_idx, ep in enumerate(tqdm(episodes, desc="  Saving")):
        n_steps = ep["n_steps"]
        episode_lengths.append(n_steps)

        for t in range(n_steps):
            row = {
                "episode_index": ep_idx,
                "frame_index": t,
                "timestamp": t / 20.0,  # 20Hz
                "task": ep["task"],
            }

            # Actions
            for i, name in enumerate(["dx", "dy", "dz", "dax", "day", "daz", "gripper"]):
                row[f"action_{name}"] = float(ep["actions"][t, i])

            # States
            for key, values in ep.get("states", {}).items():
                if t < len(values):
                    for i, v in enumerate(values[t]):
                        row[f"{key}_{i}"] = float(v)

            all_rows.append(row)

        # Save video if images available
        if "images" in ep:
            _save_episode_video(
                ep["images"],
                output_dir / "videos" / f"episode_{ep_idx:04d}.mp4",
            )

    # Save as parquet
    df = pd.DataFrame(all_rows)
    parquet_path = output_dir / "data" / f"{task_name}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"  Saved {len(df)} rows to {parquet_path}")

    # Save metadata
    meta = {
        "task_name": task_name,
        "n_episodes": len(episodes),
        "episode_lengths": episode_lengths,
        "total_frames": sum(episode_lengths),
        "fps": 20,
        "action_dim": 7,
        "action_names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper"],
        "image_size": episodes[0].get("images", np.zeros((1, 128, 128, 3))).shape[1]
        if episodes else 128,
    }
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(meta, f, indent=2)

    return df


def _save_episode_video(images, path, fps=20):
    """Save images as MP4 video."""
    import imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps)
    for frame in images:
        writer.append_data(frame)
    writer.close()


def upload_to_hub(output_dir, repo_id):
    """Upload converted dataset to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install huggingface_hub: pip install huggingface-hub")
        return

    api = HfApi()
    print(f"\nUploading to HuggingFace Hub: {repo_id}")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Convert LIBERO HDF5 to LeRobot format")
    parser.add_argument("--suite", type=str, default="libero_spatial", help="LIBERO suite name")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--image-size", type=int, default=256, help="Output image size")
    parser.add_argument("--repo-id", type=str, default=None, help="HuggingFace repo ID for upload")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace Hub")
    args = parser.parse_args()

    suite_dir = DATA_DIR / args.suite
    if not suite_dir.exists():
        print(f"Suite not found: {suite_dir}")
        print(f"Run: python data/download_libero.py --suite {args.suite}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / "lerobot" / args.suite
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Converting {args.suite} to LeRobot format")
    print("=" * 60)

    # Find HDF5 files
    hdf5_files = sorted(suite_dir.rglob("*.hdf5"))
    if not hdf5_files:
        print(f"No HDF5 files found in {suite_dir}")
        print("Data may already be in LeRobot format — check for .parquet files")
        return

    all_episodes = []
    for hdf5_path in hdf5_files:
        episodes = convert_hdf5_to_lerobot(hdf5_path, output_dir, image_size=args.image_size)
        all_episodes.extend(episodes)

    if all_episodes:
        save_lerobot_format(all_episodes, output_dir, args.suite)

    if args.upload and args.repo_id:
        upload_to_hub(output_dir, args.repo_id)

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
