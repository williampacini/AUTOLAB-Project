#!/usr/bin/env python3
"""Orchestrate the full NutAssembly dataset build pipeline.

Combines:
  - 3/4 from HuggingFace: ~75 NutAssemblySquare + ~75 NutAssemblyRound demos (robomimic)
  - 1/4 from scratch: ~50 demos via scripted policy on full NutAssembly

Auto-detects raw demo HDF5 files (states only, no images) and renders
image observations via render_demos.py before conversion.

Outputs a local LeRobot-format dataset at outputs/lerobot/nut_assembly/

Usage:
    python data/build_nut_assembly_dataset.py
    python data/build_nut_assembly_dataset.py --skip-collect    # Skip scripted demo collection
    python data/build_nut_assembly_dataset.py --skip-download   # Skip HF download
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TASK_LANGUAGE = "Assemble the round nut and square nut onto their respective pegs"
DEFAULT_OUTPUT = ROOT / "outputs" / "lerobot" / "nut_assembly"
SCRIPTED_HDF5 = ROOT / "data" / "nut_assembly_scripted" / "demos.hdf5"
ROBOMIMIC_DIR = ROOT / "data" / "robomimic_nut_assembly"


def _hdf5_has_images(hdf5_path):
    """Check if an HDF5 file has image observations."""
    import h5py

    try:
        with h5py.File(str(hdf5_path), "r") as f:
            if "data" not in f:
                return False
            demos = [k for k in f["data"].keys() if k.startswith("demo")]
            if not demos:
                return False
            first = f["data"][demos[0]]
            if "obs" not in first:
                return False
            obs_keys = list(first["obs"].keys())
            return any("image" in k for k in obs_keys)
    except Exception:
        return False


def _hdf5_has_states(hdf5_path):
    """Check if an HDF5 file has MuJoCo states (for rendering)."""
    import h5py

    try:
        with h5py.File(str(hdf5_path), "r") as f:
            if "data" not in f:
                return False
            demos = [k for k in f["data"].keys() if k.startswith("demo")]
            if not demos:
                return False
            return "states" in f["data"][demos[0]]
    except Exception:
        return False


def _infer_env_name(hdf5_path):
    """Infer robosuite environment name from HDF5 attrs or path heuristics."""
    import h5py

    try:
        with h5py.File(str(hdf5_path), "r") as f:
            for group in [f.get("data"), f]:
                if group is not None and "env_args" in group.attrs:
                    env_args_str = group.attrs["env_args"]
                    if isinstance(env_args_str, bytes):
                        env_args_str = env_args_str.decode()
                    env_args = json.loads(env_args_str)
                    if "env_name" in env_args:
                        return env_args["env_name"]
    except Exception:
        pass

    path_str = str(hdf5_path).lower()
    if "square" in path_str:
        return "NutAssemblySquare"
    if "round" in path_str:
        return "NutAssemblyRound"
    if "nut" in path_str:
        return "NutAssembly"
    return None


def step_download_hf(max_demos_per_variant=75):
    """Step 1: Download NutAssembly HDF5 from HuggingFace."""
    print("\n" + "=" * 60)
    print("  Step 1: Download HuggingFace Data")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(ROOT / "data" / "download_nut_assembly.py"),
        "--output-dir", str(ROBOMIMIC_DIR),
        "--max-demos", str(max_demos_per_variant),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("WARNING: HuggingFace download had issues. Check output above.")
        print("Continuing with whatever data is available...")
    return result.returncode == 0


def step_collect_scripted(num_demos=50, noise_std=0.01):
    """Step 2: Collect demonstrations via scripted policy."""
    print("\n" + "=" * 60)
    print("  Step 2: Collect Scripted Demos")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(ROOT / "data" / "collect_nut_assembly.py"),
        "--num-demos", str(num_demos),
        "--output", str(SCRIPTED_HDF5),
        "--noise-std", str(noise_std),
        "--visualize",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("WARNING: Scripted demo collection had issues.")
    return result.returncode == 0


def step_render_if_needed(camera_size=256):
    """Step 2.5: Render image observations for raw demo HDF5 files.

    Scans all HDF5 files in the robomimic data directory. For files that
    have MuJoCo states but no image observations, runs render_demos.py
    to generate camera images by replaying states through robosuite.
    """
    print("\n" + "=" * 60)
    print("  Step 2.5: Check & Render Image Observations")
    print("=" * 60)

    if not ROBOMIMIC_DIR.exists():
        print("  No robomimic data directory. Skipping.")
        return

    hdf5_files = sorted(ROBOMIMIC_DIR.rglob("*.hdf5"))
    files_needing_rendering = []

    for hdf5_path in hdf5_files:
        # Skip rendered output files
        if "_rendered" in hdf5_path.stem:
            continue
        # Skip files that already have a rendered counterpart
        rendered_path = hdf5_path.parent / f"{hdf5_path.stem}_rendered.hdf5"
        if rendered_path.exists():
            continue
        # Check if file has images
        if _hdf5_has_images(hdf5_path):
            continue
        # Check if file has states (renderable)
        if _hdf5_has_states(hdf5_path):
            files_needing_rendering.append(hdf5_path)
        else:
            print(f"  WARNING: {hdf5_path.name} has no images and no states — unusable")

    if not files_needing_rendering:
        print("  All HDF5 files have image observations (or rendered counterparts). OK.")
        return

    print(f"  {len(files_needing_rendering)} file(s) need image rendering:")
    for p in files_needing_rendering:
        print(f"    - {p.name}")

    render_script = ROOT / "data" / "render_demos.py"
    if not render_script.exists():
        print(f"  ERROR: render_demos.py not found at {render_script}")
        return

    for hdf5_path in files_needing_rendering:
        output_path = hdf5_path.parent / f"{hdf5_path.stem}_rendered.hdf5"
        env_name = _infer_env_name(hdf5_path)

        cmd = [
            sys.executable,
            str(render_script),
            "--input", str(hdf5_path),
            "--output", str(output_path),
            "--camera-size", str(camera_size),
        ]
        if env_name:
            cmd.extend(["--env-name", env_name])

        print(f"\n  Rendering: {hdf5_path.name} -> {output_path.name}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  WARNING: Rendering failed for {hdf5_path.name}")


def step_convert_and_combine(output_dir, image_size=256, max_demos_per_file=None):
    """Step 3: Convert all HDF5 sources to LeRobot format.

    Prefers rendered files over raw ones. Skips files without images.
    """
    print("\n" + "=" * 60)
    print("  Step 3: Convert & Combine into LeRobot Format")
    print("=" * 60)

    hdf5_files = []

    # Add HuggingFace downloads (prefer rendered versions)
    if ROBOMIMIC_DIR.exists():
        all_hf_files = sorted(ROBOMIMIC_DIR.rglob("*.hdf5"))
        selected = []
        seen_stems = set()

        # First pass: collect rendered files
        for f in all_hf_files:
            if "_rendered" in f.stem:
                selected.append(f)
                # Mark the original stem as covered
                original_stem = f.stem.replace("_rendered", "")
                seen_stems.add(original_stem)

        # Second pass: add non-rendered files that don't have a rendered counterpart
        for f in all_hf_files:
            if "_rendered" not in f.stem and f.stem not in seen_stems:
                # Only include if it has images
                if _hdf5_has_images(f):
                    selected.append(f)

        hdf5_files.extend(selected)
        print(f"  HuggingFace HDF5 files: {len(selected)}")
        for f in selected:
            has_img = _hdf5_has_images(f)
            print(f"    {f.name} {'(images OK)' if has_img else '(NO images — will skip)'}")
    else:
        print("  No HuggingFace data directory found.")

    # Add scripted demos
    if SCRIPTED_HDF5.exists():
        hdf5_files.append(SCRIPTED_HDF5)
        print(f"  Scripted HDF5: {SCRIPTED_HDF5}")
    else:
        print("  No scripted demo file found.")

    if not hdf5_files:
        print("ERROR: No HDF5 files found to convert!")
        return False

    # Filter to only files with images
    files_with_images = [f for f in hdf5_files if _hdf5_has_images(f)]
    if not files_with_images:
        print("ERROR: No HDF5 files have image observations!")
        print("  Run rendering step first: python data/render_demos.py --input <raw_file>")
        return False

    if len(files_with_images) < len(hdf5_files):
        skipped = len(hdf5_files) - len(files_with_images)
        print(f"  Skipping {skipped} file(s) without images")

    print(f"  Converting {len(files_with_images)} HDF5 file(s)")

    cmd = [
        sys.executable,
        str(ROOT / "data" / "convert_to_lerobot.py"),
        "--hdf5",
    ] + [str(f) for f in files_with_images] + [
        "--task-name", "nut_assembly",
        "--task-language", TASK_LANGUAGE,
        "--output-dir", str(output_dir),
        "--image-size", str(image_size),
        "--include-wrist",
    ]

    if max_demos_per_file:
        cmd.extend(["--max-demos", str(max_demos_per_file)])

    result = subprocess.run(cmd)
    return result.returncode == 0


def step_verify(output_dir):
    """Step 4: Verify the built dataset."""
    print("\n" + "=" * 60)
    print("  Step 4: Verify Dataset")
    print("=" * 60)

    output_dir = Path(output_dir)

    expected_dirs = ["data", "videos", "meta"]
    for d in expected_dirs:
        p = output_dir / d
        exists = p.exists()
        print(f"  {d}/: {'OK' if exists else 'MISSING'}")

    parquet_files = list((output_dir / "data").rglob("*.parquet")) if (output_dir / "data").exists() else []
    print(f"  Parquet files: {len(parquet_files)}")

    video_files = list((output_dir / "videos").rglob("*.mp4")) if (output_dir / "videos").exists() else []
    print(f"  Video files: {len(video_files)}")

    meta_path = output_dir / "meta" / "info.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Episodes: {meta.get('n_episodes', '?')}")
        print(f"  Total frames: {meta.get('total_frames', '?')}")
        print(f"  Image size: {meta.get('image_size', '?')}")
        print(f"  Cameras: {meta.get('cameras', '?')}")
    else:
        print("  meta/info.json: MISSING")

    print()
    return len(parquet_files) > 0


def main():
    parser = argparse.ArgumentParser(description="Build NutAssembly LeRobot dataset")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output directory for LeRobot dataset")
    parser.add_argument("--scripted-demos", type=int, default=50,
                        help="Number of scripted demos to collect")
    parser.add_argument("--hf-demos", type=int, default=75,
                        help="Max demos per HF variant (Square/Round)")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Output image resolution")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip HuggingFace download step")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip scripted demo collection step")
    parser.add_argument("--skip-render", action="store_true",
                        help="Skip image rendering step")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip conversion step (verify only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("  NutAssembly Dataset Build Pipeline")
    print("=" * 60)
    print(f"  Output:         {output_dir}")
    print(f"  HF demos:       {args.hf_demos}/variant (Square+Round)")
    print(f"  Scripted demos: {args.scripted_demos}")
    print(f"  Total target:   ~{args.hf_demos * 2 + args.scripted_demos} demos")
    print(f"  Task:           {TASK_LANGUAGE}")

    # Step 1: Download
    if not args.skip_download:
        step_download_hf(max_demos_per_variant=args.hf_demos)
    else:
        print("\n  [SKIPPED] HuggingFace download")

    # Step 2: Collect scripted
    if not args.skip_collect:
        step_collect_scripted(num_demos=args.scripted_demos)
    else:
        print("\n  [SKIPPED] Scripted demo collection")

    # Step 2.5: Render images if needed
    if not args.skip_render:
        step_render_if_needed(camera_size=args.image_size)
    else:
        print("\n  [SKIPPED] Image rendering")

    # Step 3: Convert
    if not args.skip_convert:
        success = step_convert_and_combine(output_dir, image_size=args.image_size)
        if not success:
            print("\nERROR: Dataset conversion failed!")
            sys.exit(1)
    else:
        print("\n  [SKIPPED] Conversion")

    # Step 4: Verify
    step_verify(output_dir)

    print("=" * 60)
    print("  Dataset build complete!")
    print(f"  Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
