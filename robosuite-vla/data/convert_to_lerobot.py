#!/usr/bin/env python3
"""Convert LIBERO or robomimic HDF5 demonstrations to LeRobot dataset format.

LeRobot datasets use:
  - Parquet files for state/action data
  - MP4 videos for camera observations
  - HuggingFace dataset metadata

Supports:
  - LIBERO HDF5 format (agentview_image, robot0_eye_in_hand_image)
  - robomimic HDF5 format (NutAssembly, Lift, etc.)
  - Dual cameras (agentview + wrist)

This script converts HDF5 to local LeRobot format and optionally uploads to HuggingFace Hub.
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


def _process_camera_frames(raw_frames, image_size, flip=True):
    """Process camera frames: flip and resize.

    Args:
        raw_frames: (T, H, W, 3) uint8 array
        image_size: target square resolution
        flip: whether to flip vertically (MuJoCo renders upside-down)

    Returns:
        (T, image_size, image_size, 3) uint8 array
    """
    from PIL import Image as PILImage

    processed = []
    for frame in raw_frames:
        if flip:
            frame = np.flip(frame, axis=0).copy()
        if frame.shape[0] != image_size or frame.shape[1] != image_size:
            pil_img = PILImage.fromarray(frame).resize(
                (image_size, image_size), PILImage.LANCZOS
            )
            frame = np.array(pil_img)
        processed.append(frame)
    return np.stack(processed)


def convert_hdf5_to_lerobot(hdf5_path, output_dir, image_size=256, task_language=None,
                             max_demos=None, include_wrist=True, flip_images=True):
    """Convert a single HDF5 file (LIBERO or robomimic) to LeRobot-compatible format.

    Supports both LIBERO and robomimic HDF5 layouts.

    Args:
        hdf5_path: path to HDF5 file
        output_dir: output directory
        image_size: target image resolution (square)
        task_language: override task description (uses HDF5 attrs if None)
        max_demos: limit number of demos to convert (None = all)
        include_wrist: include wrist camera (robot0_eye_in_hand_image) as image2
        flip_images: flip images vertically (MuJoCo upside-down fix)

    Returns list of episode dicts with keys:
      - task, n_steps, actions, states, images, images2 (optional)
    """
    task_language = task_language or get_task_language(hdf5_path)
    episodes = []

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            print(f"  [WARN] No 'data' group in {hdf5_path}")
            return []

        demo_keys = sorted(f["data"].keys())
        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]

        print(f"  Task: {task_language}")
        print(f"  Demos: {len(demo_keys)}")

        for demo_key in tqdm(demo_keys, desc=f"  Converting {Path(hdf5_path).stem}"):
            demo = f["data"][demo_key]
            actions = demo["actions"][:]
            n_steps = len(actions)

            # Extract main camera images (agentview)
            images = None
            for cam_key in ["agentview_image", "agentview_rgb"]:
                if cam_key in demo["obs"]:
                    images = demo["obs"][cam_key][:]
                    break

            # Extract wrist camera images
            images2 = None
            if include_wrist:
                for cam_key in ["robot0_eye_in_hand_image", "robot0_eye_in_hand_rgb"]:
                    if cam_key in demo["obs"]:
                        images2 = demo["obs"][cam_key][:]
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

            # Process agentview images
            if images is not None:
                episode_data["images"] = _process_camera_frames(
                    images, image_size, flip=flip_images
                )

            # Process wrist camera images
            if images2 is not None:
                episode_data["images2"] = _process_camera_frames(
                    images2, image_size, flip=flip_images
                )

            episodes.append(episode_data)

    return episodes


def save_lerobot_format(episodes, output_dir, task_name):
    """Save episodes in LeRobot-compatible format.

    Handles dual cameras: agentview → observation.images.image,
    wrist → observation.images.image2.
    """
    import pandas as pd

    output_dir = Path(output_dir)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "observation.images.image").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "observation.images.image2").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    has_wrist = any("images2" in ep for ep in episodes)

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

            # Actions (7D)
            for i, name in enumerate(["dx", "dy", "dz", "dax", "day", "daz", "gripper"]):
                row[f"action_{name}"] = float(ep["actions"][t, i])

            # State: eef_pos (3) + eef_quat (4) = 7D observation.state
            eef_pos = ep.get("states", {}).get("robot0_eef_pos")
            eef_quat = ep.get("states", {}).get("robot0_eef_quat")
            if eef_pos is not None and eef_quat is not None and t < len(eef_pos):
                state = np.concatenate([eef_pos[t], eef_quat[t]])
                for i, v in enumerate(state):
                    row[f"observation.state_{i}"] = float(v)

            # Extra states (gripper, joints)
            for key, values in ep.get("states", {}).items():
                if key in ("robot0_eef_pos", "robot0_eef_quat"):
                    continue  # Already in observation.state
                if t < len(values):
                    for i, v in enumerate(values[t]):
                        row[f"{key}_{i}"] = float(v)

            all_rows.append(row)

        # Save agentview video
        if "images" in ep:
            _save_episode_video(
                ep["images"],
                output_dir / "videos" / "observation.images.image" / f"episode_{ep_idx:04d}.mp4",
            )

        # Save wrist camera video
        if "images2" in ep:
            _save_episode_video(
                ep["images2"],
                output_dir / "videos" / "observation.images.image2" / f"episode_{ep_idx:04d}.mp4",
            )

    # Save as parquet
    df = pd.DataFrame(all_rows)
    parquet_path = output_dir / "data" / f"{task_name}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"  Saved {len(df)} rows to {parquet_path}")

    # Determine image size
    img_size = 128
    if episodes and "images" in episodes[0]:
        img_size = episodes[0]["images"].shape[1]

    # Save metadata
    meta = {
        "task_name": task_name,
        "n_episodes": len(episodes),
        "episode_lengths": episode_lengths,
        "total_frames": sum(episode_lengths),
        "fps": 20,
        "action_dim": 7,
        "action_names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper"],
        "image_size": img_size,
        "cameras": ["observation.images.image"] + (["observation.images.image2"] if has_wrist else []),
        "state_dim": 7,  # eef_pos (3) + eef_quat (4)
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


def convert_multiple_hdf5(hdf5_paths, output_dir, task_name, image_size=256,
                          task_language=None, max_demos_per_file=None,
                          include_wrist=True, flip_images=True):
    """Convert multiple HDF5 files into a single LeRobot dataset.

    Useful for combining NutAssemblySquare + NutAssemblyRound + scripted demos.

    Args:
        hdf5_paths: list of HDF5 file paths
        output_dir: output directory for the combined dataset
        task_name: name for the output dataset
        image_size: target image resolution
        task_language: override task description for all episodes
        max_demos_per_file: limit demos per HDF5 file
        include_wrist: include wrist camera
        flip_images: flip images vertically

    Returns:
        DataFrame of the combined dataset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_episodes = []
    for hdf5_path in hdf5_paths:
        print(f"\nProcessing: {hdf5_path}")
        episodes = convert_hdf5_to_lerobot(
            hdf5_path, output_dir,
            image_size=image_size,
            task_language=task_language,
            max_demos=max_demos_per_file,
            include_wrist=include_wrist,
            flip_images=flip_images,
        )
        all_episodes.extend(episodes)

    if all_episodes:
        print(f"\nTotal episodes: {len(all_episodes)}")
        return save_lerobot_format(all_episodes, output_dir, task_name)
    else:
        print("No episodes to save!")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 demos to LeRobot format")
    parser.add_argument("--suite", type=str, default=None, help="LIBERO suite name")
    parser.add_argument("--hdf5", type=str, nargs="+", default=None,
                        help="Direct HDF5 file paths (for robomimic / NutAssembly)")
    parser.add_argument("--task-name", type=str, default=None,
                        help="Dataset name (defaults to suite name)")
    parser.add_argument("--task-language", type=str, default=None,
                        help="Override task description for all episodes")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--image-size", type=int, default=256, help="Output image size")
    parser.add_argument("--max-demos", type=int, default=None,
                        help="Max demos per HDF5 file")
    parser.add_argument("--include-wrist", action="store_true", default=True,
                        help="Include wrist camera as image2")
    parser.add_argument("--no-flip", action="store_true",
                        help="Don't flip images (if already corrected)")
    parser.add_argument("--repo-id", type=str, default=None, help="HuggingFace repo ID for upload")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace Hub")
    args = parser.parse_args()

    # Determine input sources
    if args.hdf5:
        # Direct HDF5 file paths (robomimic / NutAssembly mode)
        hdf5_files = [Path(p) for p in args.hdf5]
        for p in hdf5_files:
            if not p.exists():
                print(f"File not found: {p}")
                return
        task_name = args.task_name or "nut_assembly"
    elif args.suite:
        # LIBERO suite mode (original behavior)
        suite_dir = DATA_DIR / args.suite
        if not suite_dir.exists():
            print(f"Suite not found: {suite_dir}")
            print(f"Run: python data/download_libero.py --suite {args.suite}")
            return
        hdf5_files = sorted(suite_dir.rglob("*.hdf5"))
        task_name = args.task_name or args.suite
    else:
        print("Specify --suite for LIBERO or --hdf5 for direct HDF5 files")
        return

    if not hdf5_files:
        print("No HDF5 files found!")
        return

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / "lerobot" / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Converting to LeRobot format: {task_name}")
    print(f"  Input files: {len(hdf5_files)}")
    print(f"  Output dir:  {output_dir}")
    print("=" * 60)

    convert_multiple_hdf5(
        hdf5_files, output_dir, task_name,
        image_size=args.image_size,
        task_language=args.task_language,
        max_demos_per_file=args.max_demos,
        include_wrist=args.include_wrist,
        flip_images=not args.no_flip,
    )

    if args.upload and args.repo_id:
        upload_to_hub(output_dir, args.repo_id)

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
