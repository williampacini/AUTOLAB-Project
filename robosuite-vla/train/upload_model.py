#!/usr/bin/env python3
"""Upload a trained SmolVLA checkpoint to HuggingFace Hub."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repo ID (user/model-name)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.exists():
        print(f"Checkpoint not found: {checkpoint_dir}")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install huggingface-hub: pip install huggingface-hub")
        return

    api = HfApi()

    print(f"Uploading {checkpoint_dir} to {args.repo_id}")
    api.create_repo(args.repo_id, exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=args.repo_id,
    )
    print(f"Uploaded: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
