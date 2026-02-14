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


def _build_state_vector(states, t):
    """Concatenate state observations into a single vector for timestep t.

    Order: eef_pos (3) + eef_quat (4) + gripper_qpos (2) + joint_pos (7) = 16
    Falls back gracefully if some keys are missing.
    """
    parts = []
    for key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos"]:
        if key in states and t < len(states[key]):
            parts.append(states[key][t].astype(np.float32))
    if parts:
        return np.concatenate(parts)
    return np.zeros(0, dtype=np.float32)


def save_lerobot_format(episodes, output_dir, task_name):
    """Save episodes in LeRobot v0.4+ compatible format.

    Produces:
      - data/chunk-000/file-000.parquet  (array columns: action, observation.state, etc.)
      - videos/observation.images.image/chunk-000/file-{ep_idx:03d}.mp4
      - meta/info.json  (with 'features' dict required by LeRobot v0.4+)
      - meta/episodes.parquet  (episode metadata)
      - meta/tasks.parquet  (task descriptions)
    """
    import pandas as pd

    output_dir = Path(output_dir)
    chunk_dir = output_dir / "data" / "chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    # Determine state dimension from first episode
    sample_states = episodes[0].get("states", {})
    sample_state = _build_state_vector(sample_states, 0)
    state_dim = len(sample_state)

    # Determine image size
    has_images = "images" in episodes[0]
    if has_images:
        img_shape = list(episodes[0]["images"].shape[1:])  # [H, W, C]
    else:
        img_shape = [256, 256, 3]

    # Build state motor names
    state_motor_names = []
    for key, dim_names in [
        ("robot0_eef_pos", ["x", "y", "z"]),
        ("robot0_eef_quat", ["qx", "qy", "qz", "qw"]),
        ("robot0_gripper_qpos", ["finger0", "finger1"]),
        ("robot0_joint_pos", ["j0", "j1", "j2", "j3", "j4", "j5", "j6"]),
    ]:
        if key in sample_states:
            state_motor_names.extend(dim_names)

    # Collect unique tasks
    tasks = list(dict.fromkeys(ep["task"] for ep in episodes))
    task_to_idx = {t: i for i, t in enumerate(tasks)}

    all_rows = []
    episode_lengths = []
    global_idx = 0

    for ep_idx, ep in enumerate(tqdm(episodes, desc="  Saving")):
        n_steps = ep["n_steps"]
        episode_lengths.append(n_steps)
        task_idx = task_to_idx[ep["task"]]

        for t in range(n_steps):
            row = {
                "index": global_idx,
                "episode_index": ep_idx,
                "frame_index": t,
                "timestamp": t / 20.0,
                "task_index": task_idx,
                "action": ep["actions"][t].astype(np.float32).tolist(),
            }

            # State as single concatenated vector
            if ep.get("states"):
                row["observation.state"] = _build_state_vector(ep["states"], t).tolist()

            # Image path reference (LeRobot loads videos separately)
            if has_images:
                row["observation.images.image"] = f"videos/observation.images.image/chunk-000/episode_{ep_idx:06d}.mp4"

            all_rows.append(row)
            global_idx += 1

        # Save video
        if "images" in ep:
            video_dir = output_dir / "videos" / "observation.images.image" / "chunk-000"
            video_dir.mkdir(parents=True, exist_ok=True)
            _save_episode_video(
                ep["images"],
                video_dir / f"episode_{ep_idx:06d}.mp4",
            )

    # Save as parquet
    df = pd.DataFrame(all_rows)
    parquet_path = chunk_dir / "file-000.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"  Saved {len(df)} rows to {parquet_path}")

    # Build features dict (required by LeRobot v0.4+)
    features = {
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": {"motors": ["dx", "dy", "dz", "dax", "day", "daz", "gripper"]},
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }

    if state_dim > 0:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [state_dim],
            "names": {"motors": state_motor_names} if state_motor_names else None,
        }

    if has_images:
        features["observation.images.image"] = {
            "dtype": "video",
            "shape": img_shape,
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": 20.0,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    total_frames = sum(episode_lengths)

    # Build splits
    splits = {"train": f"0:{len(episodes)}"}

    # Save info.json (LeRobot v0.4+ format)
    meta = {
        "codebase_version": "v3.0",
        "robot_type": "panda",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(episodes) if has_images else 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 20,
        "splits": splits,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/episode_{episode_index:06d}.mp4"
        if has_images else None,
        "features": features,
    }
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save episodes.parquet (LeRobot v3.0 expects parquet, not jsonl)
    episodes_records = []
    for ep_idx, length in enumerate(episode_lengths):
        episodes_records.append({
            "episode_index": ep_idx,
            "tasks": json.dumps([episodes[ep_idx]["task"]]),
            "length": length,
        })
    pd.DataFrame(episodes_records).to_parquet(
        output_dir / "meta" / "episodes.parquet", index=False
    )

    # Save tasks.parquet
    tasks_records = [{"task_index": i, "task": t} for i, t in enumerate(tasks)]
    pd.DataFrame(tasks_records).to_parquet(
        output_dir / "meta" / "tasks.parquet", index=False
    )

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
