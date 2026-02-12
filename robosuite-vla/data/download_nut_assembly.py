#!/usr/bin/env python3
"""Download NutAssembly demonstration data from HuggingFace (robomimic).

Downloads NutAssemblySquare and NutAssemblyRound proficient-human demos
from the amandlek/robomimic dataset repository.

Usage:
    python data/download_nut_assembly.py
    python data/download_nut_assembly.py --output-dir data/robomimic_nut_assembly --max-demos 75
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data" / "robomimic_nut_assembly"

# robomimic HuggingFace repo and known file paths for NutAssembly variants
HF_REPO = "amandlek/robomimic"

# These are the expected paths inside the robomimic HF dataset repo.
# The exact paths may vary — the script will search for alternatives if not found.
NUT_ASSEMBLY_FILES = {
    "square": {
        "variants": [
            "datasets/square/ph/image.hdf5",
            "square/ph/image.hdf5",
            "NutAssemblySquare/ph/image.hdf5",
        ],
        "task_name": "NutAssemblySquare",
    },
    "round": {
        "variants": [
            "datasets/round/ph/image.hdf5",
            "round/ph/image.hdf5",
            "NutAssemblyRound/ph/image.hdf5",
        ],
        "task_name": "NutAssemblyRound",
    },
}


def list_repo_files():
    """List files in the robomimic HF repo to find NutAssembly datasets."""
    from huggingface_hub import list_repo_tree

    print(f"Listing files in {HF_REPO}...")
    all_files = []
    try:
        for item in list_repo_tree(HF_REPO, repo_type="dataset", recursive=True):
            if hasattr(item, "rfilename"):
                all_files.append(item.rfilename)
    except Exception as e:
        print(f"  Warning: Could not list repo tree: {e}")
        # Fallback to list_repo_files
        from huggingface_hub import list_repo_files as hf_list
        all_files = list(hf_list(HF_REPO, repo_type="dataset"))

    # Filter for nut-related files
    nut_files = [f for f in all_files if "nut" in f.lower() or "square" in f.lower() or "round" in f.lower()]
    hdf5_files = [f for f in all_files if f.endswith(".hdf5")]

    print(f"  Total files: {len(all_files)}")
    print(f"  HDF5 files:  {len(hdf5_files)}")
    if nut_files:
        print(f"  Nut-related: {nut_files[:20]}")
    if hdf5_files:
        print(f"  HDF5 files:  {hdf5_files[:20]}")

    return all_files


def download_variant(variant_key, output_dir, all_repo_files=None):
    """Download a single NutAssembly variant from HuggingFace.

    Args:
        variant_key: "square" or "round"
        output_dir: directory to save into
        all_repo_files: optional pre-fetched list of repo files

    Returns:
        Path to downloaded file, or None if not found
    """
    from huggingface_hub import hf_hub_download

    variant_info = NUT_ASSEMBLY_FILES[variant_key]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try each known path variant
    for file_path in variant_info["variants"]:
        try:
            print(f"  Trying: {file_path}")
            local_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=file_path,
                repo_type="dataset",
                local_dir=str(output_dir),
            )
            print(f"  Downloaded: {local_path}")
            return Path(local_path)
        except Exception:
            continue

    # If none of the known paths work, search the repo
    if all_repo_files:
        keyword = variant_key.lower()
        candidates = [
            f for f in all_repo_files
            if keyword in f.lower() and f.endswith(".hdf5") and "image" in f.lower()
        ]
        if not candidates:
            # Try without 'image' filter
            candidates = [f for f in all_repo_files if keyword in f.lower() and f.endswith(".hdf5")]

        for file_path in candidates[:3]:
            try:
                print(f"  Trying discovered path: {file_path}")
                local_path = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=file_path,
                    repo_type="dataset",
                    local_dir=str(output_dir),
                )
                print(f"  Downloaded: {local_path}")
                return Path(local_path)
            except Exception:
                continue

    print(f"  WARNING: Could not download {variant_info['task_name']} from HuggingFace.")
    print(f"  You can manually download from: https://huggingface.co/datasets/{HF_REPO}")
    print(f"  Or use robomimic's download script:")
    print(f"    python robomimic/scripts/download_datasets.py --tasks {variant_key}")
    return None


def verify_hdf5(path):
    """Verify an HDF5 file contains valid demo data."""
    import h5py

    path = Path(path)
    if not path.exists():
        print(f"  File not found: {path}")
        return False

    try:
        with h5py.File(str(path), "r") as f:
            if "data" not in f:
                print(f"  No 'data' group in {path}")
                return False

            demos = list(f["data"].keys())
            if not demos:
                print(f"  No demos found in {path}")
                return False

            # Check first demo structure
            first_demo = f["data"][demos[0]]
            has_actions = "actions" in first_demo
            has_obs = "obs" in first_demo

            obs_keys = list(first_demo["obs"].keys()) if has_obs else []
            n_steps = len(first_demo["actions"]) if has_actions else 0

            print(f"  OK: {len(demos)} demos, {n_steps} steps in first demo")
            print(f"      obs keys: {obs_keys[:5]}")
            return True
    except Exception as e:
        print(f"  Error reading {path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download NutAssembly data from HuggingFace")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output directory for downloaded files")
    parser.add_argument("--max-demos", type=int, default=75,
                        help="Max demos to use per variant")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list available files, don't download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("  NutAssembly Data Download (HuggingFace / robomimic)")
    print("=" * 60)
    print(f"  Source:      {HF_REPO}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Max demos:   {args.max_demos} per variant")
    print()

    # List repo files to discover paths
    all_files = list_repo_files()

    if args.list_only:
        return

    # Download both variants
    downloaded = {}
    for variant in ["square", "round"]:
        print(f"\nDownloading {NUT_ASSEMBLY_FILES[variant]['task_name']}...")
        path = download_variant(variant, output_dir, all_repo_files=all_files)
        if path:
            print(f"\nVerifying {variant}...")
            if verify_hdf5(path):
                downloaded[variant] = path
            else:
                print(f"  Verification failed for {variant}")

    # Summary
    print(f"\n{'='*60}")
    print("  Download Summary")
    print(f"{'='*60}")
    for variant, path in downloaded.items():
        print(f"  {variant}: {path}")
    if not downloaded:
        print("  No files downloaded successfully.")
        print("  The robomimic HF repo may require authentication or have a different structure.")
        print("  Alternative: use robomimic's own download script:")
        print("    pip install robomimic")
        print("    python -m robomimic.scripts.download_datasets --tasks square round")


if __name__ == "__main__":
    main()
