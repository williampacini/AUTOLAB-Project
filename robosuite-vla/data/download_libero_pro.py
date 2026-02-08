#!/usr/bin/env python3
"""Download and set up LIBERO-PRO perturbation configs."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PRO_DIR = DATA_DIR / "LIBERO-PRO"


def clone_libero_pro():
    """Clone LIBERO-PRO repo if not present."""
    if PRO_DIR.exists():
        print(f"LIBERO-PRO already at {PRO_DIR}")
        return

    print("Cloning LIBERO-PRO...")
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/Zxy-MLlab/LIBERO-PRO.git", str(PRO_DIR)],
        check=True,
    )
    print(f"Cloned to {PRO_DIR}")


def list_perturbations():
    """List available perturbation configs."""
    print("\nLIBERO-PRO Perturbation Types:")
    print("-" * 40)

    for ptype in ["object", "position", "instruction", "environment", "combined"]:
        config_dir = PRO_DIR / "configs" / ptype
        if config_dir.exists():
            configs = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
            print(f"  {ptype}: {len(configs)} configs")
        else:
            # Check alternative structures
            bddl_dir = PRO_DIR / "bddl_files"
            init_dir = PRO_DIR / "init_files"
            if bddl_dir.exists() or init_dir.exists():
                print(f"  {ptype}: available (BDDL/init file based)")
            else:
                print(f"  {ptype}: checking structure...")

    # Show actual structure
    print("\nRepository structure:")
    for p in sorted(PRO_DIR.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_dir():
            children = [c.name for c in sorted(p.iterdir())[:5]]
            print(f"  {p.name}/  [{', '.join(children)}{'...' if len(list(p.iterdir())) > 5 else ''}]")
        else:
            print(f"  {p.name}")


def main():
    parser = argparse.ArgumentParser(description="Set up LIBERO-PRO")
    parser.add_argument("--list", action="store_true", help="List available perturbation configs")
    args = parser.parse_args()

    clone_libero_pro()

    if args.list:
        list_perturbations()
    else:
        list_perturbations()

    print("\nLIBERO-PRO ready for evaluation.")


if __name__ == "__main__":
    main()
