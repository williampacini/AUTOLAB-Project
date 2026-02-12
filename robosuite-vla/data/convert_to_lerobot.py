#!/usr/bin/env python3
"""Convert LIBERO or robosuite HDF5 demonstrations to LeRobot dataset format.

Uses the LeRobotDataset.create() API (v3.0) which handles all format details:
  - Chunked parquet files with correct schema
  - MP4 video encoding for camera observations
  - Metadata (info.json, stats.json, tasks.jsonl, episodes/)
  - Normalization statistics

Supports two sources:
  --source libero     (default) Reads from data/libero_datasets/<suite>/
  --source robosuite  Reads from data/robosuite_demos/ (or --hdf5-dir)

This script converts HDF5 demos and optionally uploads to HuggingFace Hub.
"""

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "libero_datasets"
ROBOSUITE_DEMO_DIR = ROOT / "data" / "robosuite_demos"

# State vector: eef_pos(3) + eef_quat(4) = 7D
# Must match eval/eval_robosuite.py get_action() state construction
STATE_DIM = 7
ACTION_DIM = 7
IMAGE_SIZE = 256
FPS = 20  # robosuite control_freq


def get_task_language(hdf5_path):
    """Extract natural language task description from HDF5."""
    with h5py.File(hdf5_path, "r") as f:
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


def _flip_and_resize(frame, image_size):
    """Flip MuJoCo-rendered image (upside-down) and resize."""
    from PIL import Image as PILImage

    frame = np.flip(frame, axis=0).copy()
    if frame.shape[0] != image_size or frame.shape[1] != image_size:
        pil_img = PILImage.fromarray(frame).resize(
            (image_size, image_size), PILImage.LANCZOS
        )
        frame = np.array(pil_img)
    return frame


def convert_robosuite(hdf5_dir, output_dir, image_size=IMAGE_SIZE, repo_id=None):
    """Convert robosuite HDF5 demos to LeRobot v3.0 format.

    Uses LeRobotDataset.create() API which handles:
      - Chunked parquet with correct columns (index, frame_index,
        episode_index, task_index, next.done, timestamp, etc.)
      - MP4 video encoding per camera key
      - meta/info.json, meta/stats.json, meta/tasks.jsonl, meta/episodes/
      - Normalization statistics

    Args:
        hdf5_dir: Directory containing per-environment HDF5 files.
        output_dir: Where to save the LeRobot dataset.
        image_size: Target image resolution (square).
        repo_id: HuggingFace repo ID for upload (optional).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    hdf5_dir = Path(hdf5_dir)
    output_dir = Path(output_dir)
    dataset_name = repo_id or "robosuite_multitask"

    # Clean previous output to avoid stale data
    if output_dir.exists():
        print(f"  Removing existing output: {output_dir}")
        shutil.rmtree(output_dir)

    # Discover HDF5 files
    hdf5_files = sorted(hdf5_dir.glob("*.hdf5"))
    if not hdf5_files:
        print(f"No HDF5 files found in {hdf5_dir}")
        return None

    # Count total episodes for progress display
    total_episodes = 0
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as f:
            if "data" in f:
                total_episodes += len(f["data"].keys())
    print(f"  Found {len(hdf5_files)} HDF5 files, {total_episodes} total episodes")

    # Create the LeRobot dataset with proper feature schema
    # NOTE: Do NOT include index, frame_index, episode_index, task_index,
    # next.done, timestamp — the API adds these automatically.
    dataset = LeRobotDataset.create(
        repo_id=dataset_name,
        fps=FPS,
        robot_type="panda",
        root=output_dir,
        features={
            "observation.images.image": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.image2": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": ["actions"],
            },
        },
        use_videos=True,
    )

    # Process each HDF5 file (one per robosuite environment)
    ep_count = 0
    for hdf5_path in hdf5_files:
        task_language = get_task_language(hdf5_path)
        env_name = hdf5_path.stem

        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                print(f"  [WARN] No 'data' group in {hdf5_path}, skipping")
                continue

            demo_keys = sorted(f["data"].keys())
            print(f"\n  {env_name}: {len(demo_keys)} demos — \"{task_language}\"")

            for demo_key in tqdm(demo_keys, desc=f"    {env_name}"):
                demo = f["data"][demo_key]
                actions = demo["actions"][:]
                n_steps = len(actions)

                # Extract camera images
                agentview = None
                for cam_key in ["agentview_image", "agentview_rgb"]:
                    if cam_key in demo["obs"]:
                        agentview = demo["obs"][cam_key]
                        break

                wrist = None
                for cam_key in ["robot0_eye_in_hand_image", "robot0_eye_in_hand_rgb"]:
                    if cam_key in demo["obs"]:
                        wrist = demo["obs"][cam_key]
                        break

                # Extract state components
                eef_pos = demo["obs"].get("robot0_eef_pos")
                eef_quat = demo["obs"].get("robot0_eef_quat")

                for t in range(n_steps):
                    # Build agentview image
                    if agentview is not None:
                        img = _flip_and_resize(agentview[t], image_size)
                    else:
                        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

                    # Build wrist image
                    if wrist is not None:
                        wrist_img = _flip_and_resize(wrist[t], image_size)
                    else:
                        wrist_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

                    # Build state: eef_pos(3) + eef_quat(4) = 7D
                    pos = eef_pos[t] if eef_pos is not None else np.zeros(3)
                    quat = eef_quat[t] if eef_quat is not None else np.zeros(4)
                    state = np.concatenate([pos, quat]).astype(np.float32)

                    # Action: 7D [dx, dy, dz, dax, day, daz, gripper]
                    action = actions[t].astype(np.float32)

                    dataset.add_frame({
                        "observation.images.image": img,
                        "observation.images.image2": wrist_img,
                        "observation.state": state,
                        "action": action,
                        "task": task_language,
                    })

                dataset.save_episode()
                ep_count += 1

    # Finalize: computes stats, flushes metadata, validates integrity
    print(f"\n  Finalizing dataset ({ep_count} episodes)...")
    dataset.finalize()

    print(f"\n  Dataset saved to: {output_dir}")
    print(f"  Episodes: {ep_count}")
    return output_dir


def convert_libero(suite_dir, output_dir, image_size=IMAGE_SIZE, repo_id=None):
    """Convert LIBERO HDF5 demos to LeRobot v3.0 format.

    Same approach as robosuite — uses LeRobotDataset.create() API.
    State vector: eef_pos(3) + eef_quat(4) = 7D (matches robosuite).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    suite_dir = Path(suite_dir)
    output_dir = Path(output_dir)
    dataset_name = repo_id or suite_dir.name

    if output_dir.exists():
        print(f"  Removing existing output: {output_dir}")
        shutil.rmtree(output_dir)

    hdf5_files = sorted(suite_dir.rglob("*.hdf5"))
    if not hdf5_files:
        print(f"No HDF5 files found in {suite_dir}")
        print("Data may already be in LeRobot format — check for .parquet files")
        return None

    total_episodes = 0
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as f:
            if "data" in f:
                total_episodes += len(f["data"].keys())
    print(f"  Found {len(hdf5_files)} HDF5 files, {total_episodes} total episodes")

    dataset = LeRobotDataset.create(
        repo_id=dataset_name,
        fps=FPS,
        robot_type="panda",
        root=output_dir,
        features={
            "observation.images.image": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.image2": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": ["actions"],
            },
        },
        use_videos=True,
    )

    ep_count = 0
    for hdf5_path in hdf5_files:
        task_language = get_task_language(hdf5_path)

        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                continue

            demo_keys = sorted(f["data"].keys())
            print(f"\n  {hdf5_path.stem}: {len(demo_keys)} demos — \"{task_language}\"")

            for demo_key in tqdm(demo_keys, desc=f"    {hdf5_path.stem}"):
                demo = f["data"][demo_key]
                actions = demo["actions"][:]
                n_steps = len(actions)

                agentview = None
                for cam_key in ["agentview_image", "agentview_rgb"]:
                    if cam_key in demo["obs"]:
                        agentview = demo["obs"][cam_key]
                        break

                wrist = None
                for cam_key in ["robot0_eye_in_hand_image", "robot0_eye_in_hand_rgb"]:
                    if cam_key in demo["obs"]:
                        wrist = demo["obs"][cam_key]
                        break

                eef_pos = demo["obs"].get("robot0_eef_pos")
                eef_quat = demo["obs"].get("robot0_eef_quat")

                for t in range(n_steps):
                    if agentview is not None:
                        img = _flip_and_resize(agentview[t], image_size)
                    else:
                        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

                    if wrist is not None:
                        wrist_img = _flip_and_resize(wrist[t], image_size)
                    else:
                        wrist_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

                    pos = eef_pos[t] if eef_pos is not None else np.zeros(3)
                    quat = eef_quat[t] if eef_quat is not None else np.zeros(4)
                    state = np.concatenate([pos, quat]).astype(np.float32)
                    action = actions[t].astype(np.float32)

                    dataset.add_frame({
                        "observation.images.image": img,
                        "observation.images.image2": wrist_img,
                        "observation.state": state,
                        "action": action,
                        "task": task_language,
                    })

                dataset.save_episode()
                ep_count += 1

    print(f"\n  Finalizing dataset ({ep_count} episodes)...")
    dataset.finalize()

    print(f"\n  Dataset saved to: {output_dir}")
    print(f"  Episodes: {ep_count}")
    return output_dir


def upload_to_hub(output_dir, repo_id):
    """Upload converted dataset to HuggingFace Hub."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("LeRobot not installed")
        return

    dataset = LeRobotDataset(repo_id=repo_id, root=output_dir)
    dataset.push_to_hub(tags=["robosuite", "panda"])
    print(f"Uploaded: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LIBERO/robosuite HDF5 to LeRobot v3.0 format"
    )
    parser.add_argument(
        "--source", type=str, default="libero",
        choices=["libero", "robosuite"],
        help="Source format: 'libero' or 'robosuite'",
    )
    parser.add_argument(
        "--suite", type=str, default="libero_spatial",
        help="LIBERO suite name (used when --source=libero)",
    )
    parser.add_argument(
        "--hdf5-dir", type=str, default=None,
        help="Directory containing robosuite HDF5 files (used when --source=robosuite)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--image-size", type=int, default=IMAGE_SIZE,
        help=f"Output image size (default: {IMAGE_SIZE})",
    )
    parser.add_argument(
        "--repo-id", type=str, default=None,
        help="HuggingFace repo ID (used as dataset name and for upload)",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload to HuggingFace Hub after conversion",
    )
    args = parser.parse_args()

    if args.source == "robosuite":
        hdf5_dir = Path(args.hdf5_dir) if args.hdf5_dir else ROBOSUITE_DEMO_DIR
        if not hdf5_dir.exists():
            print(f"Robosuite demo dir not found: {hdf5_dir}")
            print("Run: python data/collect_demos.py --all")
            return

        dataset_name = args.repo_id or "robosuite_multitask"
        output_dir = (Path(args.output_dir) if args.output_dir
                      else ROOT / "outputs" / "lerobot" / dataset_name)

        print("=" * 60)
        print(f"  Converting robosuite demos to LeRobot v3.0 format")
        print(f"  Source: {hdf5_dir}")
        print(f"  Output: {output_dir}")
        print("=" * 60)

        convert_robosuite(
            hdf5_dir, output_dir,
            image_size=args.image_size,
            repo_id=dataset_name,
        )

    else:
        suite_dir = DATA_DIR / args.suite
        if not suite_dir.exists():
            print(f"Suite not found: {suite_dir}")
            print(f"Run: python data/download_libero.py --suite {args.suite}")
            return

        dataset_name = args.repo_id or args.suite
        output_dir = (Path(args.output_dir) if args.output_dir
                      else ROOT / "outputs" / "lerobot" / dataset_name)

        print("=" * 60)
        print(f"  Converting {args.suite} to LeRobot v3.0 format")
        print(f"  Output: {output_dir}")
        print("=" * 60)

        convert_libero(
            suite_dir, output_dir,
            image_size=args.image_size,
            repo_id=dataset_name,
        )

    if args.upload and args.repo_id:
        upload_to_hub(output_dir, args.repo_id)

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
