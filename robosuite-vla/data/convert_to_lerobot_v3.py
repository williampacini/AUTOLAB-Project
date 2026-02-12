#!/usr/bin/env python3
"""Convert robomimic/robosuite HDF5 demonstrations to LeRobot v3.0 dataset format.

Uses the LeRobotDataset API directly to create properly formatted datasets
that are compatible with lerobot-train.

Usage:
    python data/convert_to_lerobot_v3.py \
        --hdf5 data/robomimic_nut_assembly/mimictest_extracted/square/mh/image.hdf5 \
               data/robomimic_nut_assembly/mimictest_extracted/square/ph/image.hdf5 \
        --repo-id nut_assembly \
        --output-dir outputs/lerobot/nut_assembly \
        --task "Assemble the round nut and square nut onto their respective pegs"
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def get_features(image_size=256, state_dim=7, action_dim=7, use_videos=True):
    """Define the dataset feature schema."""
    img_dtype = "video" if use_videos else "image"
    return {
        "observation.images.image": {
            "dtype": img_dtype,
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.image2": {
            "dtype": img_dtype,
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
        },
    }


def process_image(raw_frame, image_size, flip=True):
    """Flip (MuJoCo renders upside-down) and resize a single frame."""
    from PIL import Image as PILImage

    if flip:
        raw_frame = np.flip(raw_frame, axis=0).copy()
    if raw_frame.shape[0] != image_size or raw_frame.shape[1] != image_size:
        pil = PILImage.fromarray(raw_frame).resize((image_size, image_size), PILImage.LANCZOS)
        raw_frame = np.array(pil)
    return raw_frame


def convert(hdf5_paths, repo_id, output_dir, task_language, image_size=256,
            max_demos_per_file=None, flip_images=True, vcodec="libx264"):
    """Convert HDF5 files to a LeRobot v3.0 dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_dir = Path(output_dir)

    features = get_features(image_size=image_size, use_videos=True)

    print("=" * 60)
    print("  Converting to LeRobot v3.0 format")
    print("=" * 60)
    print(f"  Repo ID:    {repo_id}")
    print(f"  Output:     {output_dir}")
    print(f"  HDF5 files: {len(hdf5_paths)}")
    print(f"  Task:       {task_language}")
    print(f"  Image size: {image_size}")
    print(f"  Codec:      {vcodec}")
    print()

    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(output_dir),
        fps=20,
        features=features,
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
        print(f"\nProcessing: {hdf5_path.name}")

        with h5py.File(str(hdf5_path), "r") as f:
            if "data" not in f:
                print(f"  [WARN] No 'data' group in {hdf5_path}")
                continue

            demo_keys = sorted(f["data"].keys())
            if max_demos_per_file:
                demo_keys = demo_keys[:max_demos_per_file]

            print(f"  Demos: {len(demo_keys)}")

            for demo_key in tqdm(demo_keys, desc=f"  {hdf5_path.stem}"):
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

                if agentview is None:
                    continue

                # Extract state: eef_pos (3) + eef_quat (4) = 7D
                eef_pos = demo["obs"].get("robot0_eef_pos")
                eef_quat = demo["obs"].get("robot0_eef_quat")

                for t in range(n_steps):
                    frame = {}

                    # Images
                    frame["observation.images.image"] = process_image(
                        agentview[t], image_size, flip=flip_images
                    )
                    if wrist is not None:
                        frame["observation.images.image2"] = process_image(
                            wrist[t], image_size, flip=flip_images
                        )
                    else:
                        # Black placeholder if no wrist camera
                        frame["observation.images.image2"] = np.zeros(
                            (image_size, image_size, 3), dtype=np.uint8
                        )

                    # State
                    if eef_pos is not None and eef_quat is not None:
                        state = np.concatenate([eef_pos[t], eef_quat[t]]).astype(np.float32)
                    else:
                        state = np.zeros(7, dtype=np.float32)
                    frame["observation.state"] = state

                    # Action
                    frame["action"] = actions[t].astype(np.float32)

                    # Task
                    frame["task"] = task_language

                    dataset.add_frame(frame)

                dataset.save_episode()
                total_episodes += 1
                total_frames += n_steps

    print(f"\nFinalizing dataset...")
    dataset.finalize()

    print(f"\n{'=' * 60}")
    print(f"  Done!")
    print(f"  Episodes: {total_episodes}")
    print(f"  Frames:   {total_frames}")
    print(f"  Output:   {output_dir}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 demos to LeRobot v3.0 format")
    parser.add_argument("--hdf5", type=str, nargs="+", required=True,
                        help="HDF5 file paths to convert")
    parser.add_argument("--repo-id", type=str, default="nut_assembly",
                        help="Dataset repo ID")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for the dataset")
    parser.add_argument("--task", type=str,
                        default="Assemble the round nut and square nut onto their respective pegs",
                        help="Task description")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Output image resolution (square)")
    parser.add_argument("--max-demos", type=int, default=None,
                        help="Max demos per HDF5 file")
    parser.add_argument("--no-flip", action="store_true",
                        help="Don't flip images (if already corrected)")
    parser.add_argument("--vcodec", type=str, default="h264",
                        help="Video codec (h264, hevc, libsvtav1)")
    args = parser.parse_args()

    hdf5_paths = [Path(p) for p in args.hdf5]
    for p in hdf5_paths:
        if not p.exists():
            print(f"File not found: {p}")
            return

    convert(
        hdf5_paths=hdf5_paths,
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        task_language=args.task,
        image_size=args.image_size,
        max_demos_per_file=args.max_demos,
        flip_images=not args.no_flip,
        vcodec=args.vcodec,
    )


if __name__ == "__main__":
    main()
