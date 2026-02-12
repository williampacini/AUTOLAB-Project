#!/usr/bin/env python3
"""Download NutAssembly demonstration data from HuggingFace (robomimic).

Downloads NutAssemblySquare and NutAssemblyRound proficient-human demos
from multiple sources. Prefers pre-rendered image HDF5 when available,
falls back to raw demo files (which need rendering via render_demos.py).

Usage:
    python data/download_nut_assembly.py
    python data/download_nut_assembly.py --output-dir data/robomimic_nut_assembly --max-demos 75
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data" / "robomimic_nut_assembly"

# Primary source: official robomimic HuggingFace repo
HF_REPO = "amandlek/robomimic"

# Alternative source: EDiRobotics/mimictest_data (has pre-rendered images)
MIMICTEST_REPO = "EDiRobotics/mimictest_data"
MIMICTEST_FILES = {
    "square": "robomimic_image/square.zip",
}

# File paths to try in priority order per variant.
# Priority: image.hdf5 > ph/demo > mh/demo (prefer proficient-human over multi-human)
NUT_ASSEMBLY_FILES = {
    "square": {
        "image_variants": [
            "datasets/square/ph/image.hdf5",
            "square/ph/image.hdf5",
            "NutAssemblySquare/ph/image.hdf5",
        ],
        "raw_variants": [
            "v1.5/square/ph/demo_v15.hdf5",
            "v1.5/square/mh/demo_v15.hdf5",
        ],
        "task_name": "NutAssemblySquare",
    },
    "round": {
        "image_variants": [
            "datasets/round/ph/image.hdf5",
            "round/ph/image.hdf5",
            "NutAssemblyRound/ph/image.hdf5",
        ],
        "raw_variants": [
            "v1.5/round/ph/demo_v15.hdf5",
            "v1.5/round/mh/demo_v15.hdf5",
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
        from huggingface_hub import list_repo_files as hf_list
        all_files = list(hf_list(HF_REPO, repo_type="dataset"))

    nut_files = [f for f in all_files if "nut" in f.lower() or "square" in f.lower() or "round" in f.lower()]
    hdf5_files = [f for f in all_files if f.endswith(".hdf5")]

    print(f"  Total files: {len(all_files)}")
    print(f"  HDF5 files:  {len(hdf5_files)}")
    if nut_files:
        print(f"  Nut-related: {nut_files[:20]}")
    if hdf5_files:
        print(f"  HDF5 files:  {hdf5_files[:20]}")

    return all_files


def download_mimictest_variant(variant_key, output_dir):
    """Try downloading pre-rendered image data from EDiRobotics/mimictest_data.

    Returns:
        Path to extracted HDF5 file, or None
    """
    if variant_key not in MIMICTEST_FILES:
        return None

    try:
        from huggingface_hub import hf_hub_download
        import zipfile
    except ImportError:
        return None

    zip_filename = MIMICTEST_FILES[variant_key]
    output_dir = Path(output_dir)

    try:
        print(f"  Trying EDiRobotics/mimictest_data: {zip_filename}")
        local_zip = hf_hub_download(
            repo_id=MIMICTEST_REPO,
            filename=zip_filename,
            repo_type="dataset",
            local_dir=str(output_dir),
        )
        extract_dir = output_dir / "mimictest_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(local_zip, "r") as zf:
            zf.extractall(str(extract_dir))
        extracted = sorted(extract_dir.rglob("*.hdf5"))
        if extracted:
            print(f"  Downloaded and extracted: {extracted[0]}")
            return extracted[0]
    except Exception as e:
        print(f"  EDiRobotics download failed: {e}")

    return None


def download_variant(variant_key, output_dir, all_repo_files=None):
    """Download a single NutAssembly variant with fallback strategy.

    Priority:
    1. EDiRobotics/mimictest_data (pre-rendered images)
    2. amandlek/robomimic image.hdf5 (pre-rendered)
    3. amandlek/robomimic ph/demo_v15.hdf5 (proficient-human raw, needs rendering)
    4. amandlek/robomimic mh/demo_v15.hdf5 (multi-human raw, fallback)
    5. Discovery search in repo

    Returns:
        Path to downloaded file, or None if not found
    """
    from huggingface_hub import hf_hub_download

    variant_info = NUT_ASSEMBLY_FILES[variant_key]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Pre-rendered from EDiRobotics
    result = download_mimictest_variant(variant_key, output_dir)
    if result:
        return result

    # Strategy 2: Try image variants from amandlek/robomimic
    for file_path in variant_info.get("image_variants", []):
        try:
            print(f"  Trying (image): {file_path}")
            local_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=file_path,
                repo_type="dataset",
                local_dir=str(output_dir),
            )
            print(f"  Downloaded (image): {local_path}")
            return Path(local_path)
        except Exception:
            continue

    # Strategy 3: Try raw demo variants (PH first, then MH)
    for file_path in variant_info.get("raw_variants", []):
        try:
            print(f"  Trying (raw): {file_path}")
            local_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=file_path,
                repo_type="dataset",
                local_dir=str(output_dir),
            )
            print(f"  Downloaded (raw, needs rendering): {local_path}")
            return Path(local_path)
        except Exception:
            continue

    # Strategy 4: Discovery search in repo
    if all_repo_files:
        keyword = variant_key.lower()
        # Prefer image files, then any HDF5
        candidates = [
            f for f in all_repo_files
            if keyword in f.lower() and f.endswith(".hdf5") and "image" in f.lower()
        ]
        if not candidates:
            # Prefer ph over mh
            candidates = [
                f for f in all_repo_files
                if keyword in f.lower() and f.endswith(".hdf5") and "ph" in f.lower()
            ]
        if not candidates:
            candidates = [
                f for f in all_repo_files
                if keyword in f.lower() and f.endswith(".hdf5")
            ]

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
    return None


def verify_hdf5(path):
    """Verify an HDF5 file and classify its data type."""
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

            demos = [k for k in f["data"].keys() if k.startswith("demo")]
            if not demos:
                print(f"  No demos found in {path}")
                return False

            first_demo = f["data"][demos[0]]
            has_actions = "actions" in first_demo
            has_obs = "obs" in first_demo
            has_states = "states" in first_demo

            obs_keys = list(first_demo["obs"].keys()) if has_obs else []
            n_steps = len(first_demo["actions"]) if has_actions else 0

            has_images = any("image" in k for k in obs_keys)
            has_proprio = any(k in obs_keys for k in
                              ["robot0_eef_pos", "robot0_eef_quat"])

            # Classify
            if has_images:
                data_type = "IMAGE (has camera observations)"
            elif has_proprio:
                data_type = "LOW_DIM (proprioception only, no images)"
            elif has_states:
                data_type = "RAW (states only, needs rendering via render_demos.py)"
            else:
                data_type = "UNKNOWN"

            print(f"  OK: {len(demos)} demos, {n_steps} steps in first demo")
            print(f"      obs keys: {obs_keys[:8]}")
            print(f"      has states: {has_states}")
            print(f"      type: {data_type}")
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
    print(f"  Source:      {HF_REPO} + {MIMICTEST_REPO}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Max demos:   {args.max_demos} per variant")
    print()

    all_files = list_repo_files()

    if args.list_only:
        return

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

    print(f"\n{'='*60}")
    print("  Download Summary")
    print(f"{'='*60}")
    for variant, path in downloaded.items():
        print(f"  {variant}: {path}")
    if not downloaded:
        print("  No files downloaded successfully.")
        print("  Alternative: use robomimic's own download script:")
        print("    pip install robomimic")
        print("    python -m robomimic.scripts.download_datasets --tasks square round")
    else:
        # Check if any files need rendering
        needs_rendering = []
        for variant, path in downloaded.items():
            import h5py
            with h5py.File(str(path), "r") as f:
                demos = [k for k in f["data"].keys() if k.startswith("demo")]
                if demos:
                    first = f["data"][demos[0]]
                    obs_keys = list(first["obs"].keys()) if "obs" in first else []
                    if not any("image" in k for k in obs_keys):
                        needs_rendering.append(path)
        if needs_rendering:
            print(f"\n  NOTE: {len(needs_rendering)} file(s) need image rendering.")
            print("  Run build_nut_assembly_dataset.py (it auto-renders), or manually:")
            for p in needs_rendering:
                print(f"    python data/render_demos.py --input {p}")


if __name__ == "__main__":
    main()
