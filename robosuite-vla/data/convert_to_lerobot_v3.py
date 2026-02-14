#!/usr/bin/env python3
"""Convert robosuite HDF5 demonstrations to LeRobot v3.0 dataset format.

Usage:
    python data/convert_to_lerobot_v3.py \
        --hdf5 collected_demos/lift_demos.hdf5 \
        --repo-id lift \
        --output-dir outputs/lerobot/lift \
        --task "Pick up the cube" \
        --image-size 256 \
        --vcodec h264

    # Multiple HDF5 files:
    python data/convert_to_lerobot_v3.py \
        --hdf5 demos1.hdf5 demos2.hdf5 \
        --repo-id nut_assembly \
        --output-dir outputs/lerobot/nut_assembly \
        --task "Assemble the nuts onto their pegs"

LeRobot v3.0 datasets use:
  - Chunked Parquet files for state/action data
  - MP4 videos for camera observations
  - Metadata in meta/ directory
"""

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# Task descriptions for common robosuite environments
DEFAULT_TASKS = {
    "lift": "Pick up the cube",
    "stack": "Stack one cube on top of another",
    "pickplacesingle": "Pick up the object and place it in the bin",
    "pickplace": "Pick up objects and place them in the correct bins",
    "nutassemblysquare": "Assemble the square nut onto the peg",
    "nutassemblyround": "Assemble the round nut onto the peg",
    "nutassembly": "Assemble the round nut and square nut onto their respective pegs",
    "door": "Open the door",
    "wipe": "Wipe the dirty surface clean",
}

# Cameras to look for in HDF5 obs, in priority order
CAMERA_KEYS = [
    "agentview_image",
    "robot0_eye_in_hand_image",
    "agentview_rgb",
    "frontview_image",
    "sideview_image",
]

# State keys to concatenate into observation.state
STATE_KEYS = [
    "robot0_eef_pos",       # (3,) end-effector position
    "robot0_eef_quat",      # (4,) end-effector quaternion
    "robot0_gripper_qpos",  # (2,) gripper joint positions
]


def detect_cameras(hdf5_file):
    """Detect available camera keys in the HDF5 file."""
    with h5py.File(hdf5_file, "r") as f:
        demo_keys = sorted(f["data"].keys())
        obs = f["data"][demo_keys[0]]["obs"]
        found = [k for k in CAMERA_KEYS if k in obs]
    return found


def detect_state_dim(hdf5_file):
    """Detect the state vector dimension from available observation keys."""
    dim = 0
    keys_used = []
    with h5py.File(hdf5_file, "r") as f:
        demo_keys = sorted(f["data"].keys())
        obs = f["data"][demo_keys[0]]["obs"]
        for key in STATE_KEYS:
            if key in obs:
                d = obs[key].shape[-1]
                dim += d
                keys_used.append((key, d))
    return dim, keys_used


def infer_task(hdf5_path):
    """Infer task description from the HDF5 filename."""
    stem = Path(hdf5_path).stem.lower()
    # Strip common suffixes like _demos, _demo, _data
    for suffix in ["_demos", "_demo", "_data"]:
        stem = stem.replace(suffix, "")
    return DEFAULT_TASKS.get(stem, stem.replace("_", " "))


def build_state_vector(obs_group, step_idx, state_keys_used):
    """Concatenate state observation arrays into a single vector."""
    parts = []
    for key, _ in state_keys_used:
        parts.append(obs_group[key][step_idx].astype(np.float32))
    return np.concatenate(parts)


def process_image(image, target_size):
    """Flip (MuJoCo renders upside-down) and resize an image."""
    frame = np.flip(image, axis=0).copy()
    if frame.shape[0] != target_size or frame.shape[1] != target_size:
        pil_img = Image.fromarray(frame).resize(
            (target_size, target_size), Image.LANCZOS
        )
        frame = np.array(pil_img)
    return frame


def convert(hdf5_paths, repo_id, output_dir, task, image_size, vcodec, fps):
    """Convert HDF5 demo files to LeRobot v3.0 dataset."""
    # Try current import path first, fall back to older path
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output_dir = Path(output_dir)
    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    # Detect cameras and state from first HDF5 file
    cameras = detect_cameras(hdf5_paths[0])
    state_dim, state_keys_used = detect_state_dim(hdf5_paths[0])

    if not cameras:
        print("ERROR: No camera observations found in HDF5 file.")
        sys.exit(1)

    print(f"Cameras found: {cameras}")
    print(f"State keys: {[(k, d) for k, d in state_keys_used]} (dim={state_dim})")

    # Map HDF5 camera keys to LeRobot feature names
    camera_feature_map = {}
    for cam_key in cameras:
        # agentview_image -> observation.images.agentview
        feature_name = cam_key.replace("_image", "").replace("_rgb", "")
        feature_name = f"observation.images.{feature_name}"
        camera_feature_map[cam_key] = feature_name

    # Build features dict
    features = {}
    for cam_key, feature_name in camera_feature_map.items():
        features[feature_name] = {
            "dtype": "video",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        }

    if state_dim > 0:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": None,
        }

    features["action"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "axes": ["dx", "dy", "dz", "dax", "day", "daz", "gripper"],
        },
    }

    print(f"\nFeatures: {list(features.keys())}")

    # Create the LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=str(output_dir),
        robot_type="panda",
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        vcodec=vcodec,
    )

    total_episodes = 0
    total_frames = 0

    for hdf5_path in hdf5_paths:
        hdf5_path = Path(hdf5_path)
        print(f"\nProcessing: {hdf5_path}")

        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                print(f"  WARNING: No 'data' group in {hdf5_path}, skipping.")
                continue

            demo_keys = sorted(f["data"].keys())
            print(f"  Demos: {len(demo_keys)}")

            for demo_key in tqdm(demo_keys, desc=f"  {hdf5_path.stem}"):
                demo = f["data"][demo_key]
                actions = demo["actions"][:]
                n_steps = len(actions)

                # Determine task description
                episode_task = task
                if not episode_task:
                    episode_task = infer_task(hdf5_path)

                for t in range(n_steps):
                    frame = {
                        "action": actions[t].astype(np.float32),
                        "task": episode_task,
                    }

                    # Add camera images
                    for cam_key, feature_name in camera_feature_map.items():
                        if cam_key in demo["obs"]:
                            img = process_image(demo["obs"][cam_key][t], image_size)
                            frame[feature_name] = img

                    # Add state vector
                    if state_dim > 0:
                        state = build_state_vector(demo["obs"], t, state_keys_used)
                        frame["observation.state"] = state

                    dataset.add_frame(frame)

                dataset.save_episode()
                total_episodes += 1
                total_frames += n_steps

    # Finalize the dataset (writes parquet footers)
    dataset.finalize()

    print(f"\n{'='*60}")
    print(f"  Conversion complete!")
    print(f"  Output: {output_dir}")
    print(f"  Episodes: {total_episodes}")
    print(f"  Frames: {total_frames}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert robosuite HDF5 demos to LeRobot v3.0 format"
    )
    parser.add_argument(
        "--hdf5", nargs="+", required=True,
        help="Path(s) to HDF5 demo files",
    )
    parser.add_argument(
        "--repo-id", required=True,
        help="Dataset identifier (e.g. 'lift', 'nut_assembly')",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for the LeRobot dataset",
    )
    parser.add_argument(
        "--task", default=None,
        help="Task description string. If not provided, inferred from filename.",
    )
    parser.add_argument(
        "--image-size", type=int, default=256,
        help="Output image size (height and width). Default: 256",
    )
    parser.add_argument(
        "--vcodec", default="h264",
        choices=["h264", "hevc", "libsvtav1"],
        help="Video codec for MP4 encoding. Default: h264",
    )
    parser.add_argument(
        "--fps", type=int, default=20,
        help="Frame rate of the demonstrations. Default: 20",
    )
    args = parser.parse_args()

    # Validate HDF5 paths
    for path in args.hdf5:
        if not Path(path).exists():
            print(f"ERROR: HDF5 file not found: {path}")
            sys.exit(1)

    convert(
        hdf5_paths=args.hdf5,
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        task=args.task,
        image_size=args.image_size,
        vcodec=args.vcodec,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
