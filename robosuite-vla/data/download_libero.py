#!/usr/bin/env python3
"""Download LIBERO demonstration datasets and set up LIBERO/LIBERO-PRO repos."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]

# HuggingFaceVLA/libero is the public v3.0 format dataset compatible with LeRobot v0.4+
# The yifengzhu-hf per-suite datasets are gated and use v2.0 format (incompatible).
HF_DATASETS = {
    "libero_spatial": "HuggingFaceVLA/libero",
    "libero_object": "HuggingFaceVLA/libero",
    "libero_goal": "HuggingFaceVLA/libero",
    "libero_10": "HuggingFaceVLA/libero",
    "libero_90": "HuggingFaceVLA/libero",
}


def run(cmd, cwd=None):
    """Run a shell command and print output."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"  [WARN] Command exited with code {result.returncode}")
    return result.returncode


def clone_libero():
    """Clone the LIBERO repository."""
    libero_dir = DATA_DIR / "LIBERO"
    if libero_dir.exists():
        print(f"LIBERO already cloned at {libero_dir}")
        return libero_dir

    print("Cloning LIBERO repository...")
    run(f"git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git {libero_dir}")

    # Install LIBERO in editable mode
    print("Installing LIBERO...")
    run(f"{sys.executable} -m pip install -e .", cwd=libero_dir)
    return libero_dir


def clone_libero_pro():
    """Clone the LIBERO-PRO repository."""
    pro_dir = DATA_DIR / "LIBERO-PRO"
    if pro_dir.exists():
        print(f"LIBERO-PRO already cloned at {pro_dir}")
        return pro_dir

    print("Cloning LIBERO-PRO repository...")
    run(f"git clone --depth 1 https://github.com/Zxy-MLlab/LIBERO-PRO.git {pro_dir}")
    return pro_dir


def download_demos_hf(suite_name):
    """Download demonstration data from HuggingFace Hub."""
    if suite_name not in HF_DATASETS:
        print(f"Unknown suite: {suite_name}. Available: {list(HF_DATASETS.keys())}")
        return

    repo_id = HF_DATASETS[suite_name]
    out_dir = DATA_DIR / "libero_datasets" / suite_name

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"Dataset already exists at {out_dir}")
        return

    print(f"Downloading {suite_name} from HuggingFace: {repo_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(out_dir),
        )
        print(f"Downloaded {suite_name} to {out_dir}")
    except ImportError:
        print("huggingface_hub not installed. Trying CLI...")
        run(f"huggingface-cli download --repo-type dataset {repo_id} --local-dir {out_dir}")


def download_demos_libero_script(suite_name, libero_dir):
    """Download using LIBERO's built-in download script."""
    script = libero_dir / "benchmark_scripts" / "download_libero_datasets.py"
    if not script.exists():
        print(f"LIBERO download script not found at {script}")
        print("Falling back to HuggingFace download...")
        download_demos_hf(suite_name)
        return

    print(f"Downloading {suite_name} using LIBERO script...")
    run(
        f"{sys.executable} {script} --datasets {suite_name}",
        cwd=libero_dir,
    )


def verify_download(suite_name):
    """Verify that demo data was downloaded correctly."""
    data_dir = DATA_DIR / "libero_datasets" / suite_name
    if not data_dir.exists():
        print(f"[FAIL] {suite_name}: directory not found at {data_dir}")
        return False

    # Look for HDF5 files or parquet files (LeRobot format)
    hdf5_files = list(data_dir.rglob("*.hdf5"))
    parquet_files = list(data_dir.rglob("*.parquet"))
    mp4_files = list(data_dir.rglob("*.mp4"))

    total = len(hdf5_files) + len(parquet_files) + len(mp4_files)
    if total == 0:
        print(f"[FAIL] {suite_name}: no data files found in {data_dir}")
        return False

    print(f"[OK]   {suite_name}: {len(hdf5_files)} HDF5, {len(parquet_files)} parquet, {len(mp4_files)} mp4")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download LIBERO data and repositories")
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_spatial",
        choices=SUITES + ["all"],
        help="Which LIBERO suite to download",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="huggingface",
        choices=["huggingface", "libero_script"],
        help="Download method",
    )
    parser.add_argument("--skip-repos", action="store_true", help="Skip cloning LIBERO/LIBERO-PRO repos")
    args = parser.parse_args()

    print("=" * 60)
    print("  LIBERO Data Download")
    print("=" * 60)

    # Step 1: Clone repos
    if not args.skip_repos:
        libero_dir = clone_libero()
        clone_libero_pro()
    else:
        libero_dir = DATA_DIR / "LIBERO"

    # Step 2: Download demo data
    suites_to_download = SUITES if args.suite == "all" else [args.suite]

    for suite in suites_to_download:
        print(f"\n--- Downloading {suite} ---")
        if args.method == "huggingface":
            download_demos_hf(suite)
        else:
            download_demos_libero_script(suite, libero_dir)

    # Step 3: Verify
    print("\n--- Verification ---")
    for suite in suites_to_download:
        verify_download(suite)

    print("\nDone.")


if __name__ == "__main__":
    main()
