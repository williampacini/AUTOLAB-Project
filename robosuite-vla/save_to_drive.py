#!/usr/bin/env python3
"""Save AUTOLAB project outputs to Google Drive.

Works in two modes:
  1. Google Colab  — uses google.colab.drive.mount (zero setup)
  2. Standalone    — uses Google Drive API via service account or OAuth

Usage (Colab — simplest):
    from save_to_drive import save_all
    save_all()                       # saves everything
    save_all(what=["results"])       # saves only results

Usage (CLI):
    python save_to_drive.py --all
    python save_to_drive.py --results --videos
    python save_to_drive.py --checkpoint outputs/checkpoints/smolvla_nut_assembly

Usage (Standalone with service account):
    export GDRIVE_SERVICE_ACCOUNT=/path/to/service-account.json
    export GDRIVE_FOLDER_ID=1abc...xyz
    python save_to_drive.py --all
"""

import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Default paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"
DEFAULT_DRIVE_BASE = "AUTOLAB/robosuite-vla"


def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _sizeof_fmt(num):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _dir_size(path):
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


# ===================================================================
# Mode 1: Google Colab (mount-based, zero config)
# ===================================================================

def _is_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _mount_drive():
    """Mount Google Drive in Colab. Returns mount point."""
    from google.colab import drive
    mount_point = "/content/drive"
    if not os.path.ismount(mount_point):
        drive.mount(mount_point)
    return Path(mount_point) / "MyDrive"


def _copy_tree(src, dst, label=""):
    """Copy directory tree with progress reporting."""
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        print(f"  SKIP {label or src.name}: {src} not found")
        return 0

    size = _dir_size(src)
    print(f"  Copying {label or src.name} ({_sizeof_fmt(size)}) ...")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
    print(f"    -> {dst}")
    return size


def _copy_file(src, dst, label=""):
    """Copy single file."""
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        print(f"  SKIP {label or src.name}: not found")
        return 0
    size = src.stat().st_size
    print(f"  Copying {label or src.name} ({_sizeof_fmt(size)}) ...")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    print(f"    -> {dst}")
    return size


# ===================================================================
# Mode 2: Google Drive API (standalone / headless)
# ===================================================================

def _upload_via_api(local_path, remote_folder_id, credentials_path=None):
    """Upload a file or directory via Google Drive API.

    Requires:
        pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

    Set GDRIVE_SERVICE_ACCOUNT env var to path to service account JSON,
    or GDRIVE_FOLDER_ID to the target folder ID.
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        print("ERROR: Install google-api-python-client:")
        print("  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        return False

    creds_path = credentials_path or os.environ.get("GDRIVE_SERVICE_ACCOUNT")
    if not creds_path:
        print("ERROR: Set GDRIVE_SERVICE_ACCOUNT env var or pass credentials_path")
        return False

    creds = service_account.Credentials.from_service_account_file(
        creds_path, scopes=["https://www.googleapis.com/auth/drive.file"]
    )
    service = build("drive", "v3", credentials=creds)

    local_path = Path(local_path)

    def _upload_file(fpath, parent_id):
        media = MediaFileUpload(str(fpath), resumable=True)
        metadata = {"name": fpath.name, "parents": [parent_id]}
        request = service.files().create(body=metadata, media_body=media, fields="id,name")
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                print(f"    Uploading {fpath.name}: {pct}%", end="\r")
        print(f"    Uploaded {fpath.name} (id={response['id']})")
        return response["id"]

    def _create_folder(name, parent_id):
        metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        folder = service.files().create(body=metadata, fields="id").execute()
        return folder["id"]

    def _upload_dir(dirpath, parent_id):
        folder_id = _create_folder(dirpath.name, parent_id)
        for item in sorted(dirpath.iterdir()):
            if item.is_file():
                _upload_file(item, folder_id)
            elif item.is_dir():
                _upload_dir(item, folder_id)
        return folder_id

    if local_path.is_file():
        _upload_file(local_path, remote_folder_id)
    elif local_path.is_dir():
        _upload_dir(local_path, remote_folder_id)
    else:
        print(f"ERROR: {local_path} does not exist")
        return False

    return True


# ===================================================================
# High-level save functions
# ===================================================================

def save_all(
    output_dir=None,
    drive_base=None,
    checkpoint_path=None,
    what=None,
    use_api=False,
    api_folder_id=None,
    api_credentials=None,
    run_name=None,
):
    """Save project outputs to Google Drive.

    Args:
        output_dir: Path to outputs/ directory (default: auto-detect)
        drive_base: Subfolder inside Google Drive (default: AUTOLAB/robosuite-vla)
        checkpoint_path: Explicit checkpoint path (default: outputs/checkpoints/*)
        what: List of what to save. Options: "results", "videos", "checkpoints",
              "configs", "notebooks". Default (None) = save everything.
        use_api: If True, use Google Drive API instead of mount (for non-Colab).
        api_folder_id: Google Drive folder ID for API uploads.
        api_credentials: Path to service account JSON for API uploads.
        run_name: Optional name for this run (used in folder name).

    Returns:
        Path to the save directory on Google Drive.
    """
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUTS
    drive_base = drive_base or DEFAULT_DRIVE_BASE
    what = what or ["results", "videos", "checkpoints", "configs"]

    ts = _timestamp()
    run_label = run_name or f"run_{ts}"

    print("=" * 60)
    print(f"  SAVING TO GOOGLE DRIVE")
    print(f"  Run: {run_label}")
    print(f"  Items: {', '.join(what)}")
    print("=" * 60)

    # Determine save method
    if use_api or not _is_colab():
        folder_id = api_folder_id or os.environ.get("GDRIVE_FOLDER_ID")
        creds = api_credentials or os.environ.get("GDRIVE_SERVICE_ACCOUNT")

        if folder_id and creds:
            print(f"\nUsing Google Drive API (folder_id={folder_id[:12]}...)")
            return _save_via_api(output_dir, what, checkpoint_path, folder_id, creds, run_label)
        elif _is_colab():
            pass  # Fall through to mount-based
        else:
            print("\nNo Colab detected and no API credentials set.")
            print("Options:")
            print("  1. Run in Google Colab (automatic Drive mount)")
            print("  2. Set env vars: GDRIVE_SERVICE_ACCOUNT + GDRIVE_FOLDER_ID")
            print("  3. Use --local-backup to save a tar.gz instead")
            return None

    # Colab mount-based save
    print("\nMounting Google Drive...")
    drive_root = _mount_drive()
    save_dir = drive_root / drive_base / run_label

    total_bytes = 0

    if "results" in what:
        results_dir = output_dir / "results"
        if results_dir.exists():
            total_bytes += _copy_tree(results_dir, save_dir / "results", "Results (JSON/CSV/plots)")

    if "videos" in what:
        videos_dir = output_dir / "videos"
        if videos_dir.exists():
            total_bytes += _copy_tree(videos_dir, save_dir / "videos", "Videos (MP4/GIF)")

    if "checkpoints" in what:
        if checkpoint_path:
            cp = Path(checkpoint_path)
            total_bytes += _copy_tree(cp, save_dir / "checkpoints" / cp.name, f"Checkpoint: {cp.name}")
        else:
            cp_dir = output_dir / "checkpoints"
            if cp_dir.exists():
                for cp in sorted(cp_dir.iterdir()):
                    if cp.is_dir():
                        total_bytes += _copy_tree(cp, save_dir / "checkpoints" / cp.name, f"Checkpoint: {cp.name}")

    if "configs" in what:
        config_dir = PROJECT_ROOT / "config"
        if config_dir.exists():
            total_bytes += _copy_tree(config_dir, save_dir / "config", "Config files")
        # Also save CLAUDE.md for context
        claude_md = PROJECT_ROOT / "CLAUDE.md"
        total_bytes += _copy_file(claude_md, save_dir / "CLAUDE.md", "CLAUDE.md")

    if "notebooks" in what:
        nb_dir = PROJECT_ROOT / "notebooks"
        if nb_dir.exists():
            total_bytes += _copy_tree(nb_dir, save_dir / "notebooks", "Notebooks")

    print(f"\nTotal saved: {_sizeof_fmt(total_bytes)}")
    print(f"Location: {save_dir}")
    print("=" * 60)
    return save_dir


def _save_via_api(output_dir, what, checkpoint_path, folder_id, creds, run_label):
    """Save using Google Drive API (non-Colab environments)."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    credentials = service_account.Credentials.from_service_account_file(
        creds, scopes=["https://www.googleapis.com/auth/drive.file"]
    )
    service = build("drive", "v3", credentials=credentials)

    # Create run folder
    run_folder = service.files().create(
        body={
            "name": run_label,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [folder_id],
        },
        fields="id",
    ).execute()
    run_id = run_folder["id"]
    print(f"  Created folder: {run_label} (id={run_id})")

    if "results" in what:
        _upload_via_api(output_dir / "results", run_id, creds)
    if "videos" in what:
        _upload_via_api(output_dir / "videos", run_id, creds)
    if "checkpoints" in what:
        if checkpoint_path:
            _upload_via_api(checkpoint_path, run_id, creds)
        else:
            cp_dir = output_dir / "checkpoints"
            if cp_dir.exists():
                _upload_via_api(cp_dir, run_id, creds)
    if "configs" in what:
        _upload_via_api(PROJECT_ROOT / "config", run_id, creds)

    return run_id


def save_local_backup(output_dir=None, backup_dir=None, run_name=None):
    """Create a local tar.gz backup (fallback when no Drive access).

    Returns path to the created archive.
    """
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUTS
    backup_dir = Path(backup_dir) if backup_dir else PROJECT_ROOT / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    ts = _timestamp()
    label = run_name or f"run_{ts}"
    archive_path = backup_dir / f"{label}.tar.gz"

    print(f"Creating local backup: {archive_path}")

    import tarfile
    with tarfile.open(str(archive_path), "w:gz") as tar:
        if (output_dir / "results").exists():
            tar.add(str(output_dir / "results"), arcname=f"{label}/results")
            print("  Added results/")
        if (output_dir / "videos").exists():
            tar.add(str(output_dir / "videos"), arcname=f"{label}/videos")
            print("  Added videos/")
        if (output_dir / "checkpoints").exists():
            tar.add(str(output_dir / "checkpoints"), arcname=f"{label}/checkpoints")
            print("  Added checkpoints/")
        config_dir = PROJECT_ROOT / "config"
        if config_dir.exists():
            tar.add(str(config_dir), arcname=f"{label}/config")
            print("  Added config/")

    size = archive_path.stat().st_size
    print(f"\nBackup created: {archive_path} ({_sizeof_fmt(size)})")
    return archive_path


# ===================================================================
# Convenience functions for notebooks
# ===================================================================

def save_results(run_name=None):
    """Quick save: results only (JSON, CSV, plots)."""
    return save_all(what=["results"], run_name=run_name)


def save_checkpoint(checkpoint_path, run_name=None):
    """Quick save: single checkpoint."""
    return save_all(what=["checkpoints"], checkpoint_path=checkpoint_path, run_name=run_name)


def save_videos(run_name=None):
    """Quick save: evaluation videos only."""
    return save_all(what=["videos"], run_name=run_name)


def save_results_and_videos(run_name=None):
    """Quick save: results + videos (skip large checkpoints)."""
    return save_all(what=["results", "videos"], run_name=run_name)


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Save AUTOLAB outputs to Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Colab — save everything:
  python save_to_drive.py --all

  # Save only results and videos:
  python save_to_drive.py --results --videos

  # Save a specific checkpoint:
  python save_to_drive.py --checkpoint outputs/checkpoints/smolvla_nut_assembly

  # Create local tar.gz backup (no Drive needed):
  python save_to_drive.py --local-backup

  # Standalone with service account:
  GDRIVE_SERVICE_ACCOUNT=/path/to/creds.json GDRIVE_FOLDER_ID=1abc... \\
      python save_to_drive.py --all
        """,
    )
    parser.add_argument("--all", action="store_true", help="Save everything")
    parser.add_argument("--results", action="store_true", help="Save results (JSON/CSV/plots)")
    parser.add_argument("--videos", action="store_true", help="Save evaluation videos")
    parser.add_argument("--checkpoints", action="store_true", help="Save model checkpoints")
    parser.add_argument("--configs", action="store_true", help="Save config files")
    parser.add_argument("--notebooks", action="store_true", help="Save notebooks")
    parser.add_argument("--checkpoint", type=str, help="Path to specific checkpoint to save")
    parser.add_argument("--output-dir", type=str, help="Path to outputs/ directory")
    parser.add_argument("--drive-base", type=str, help="Subfolder in Google Drive")
    parser.add_argument("--run-name", type=str, help="Name for this run")
    parser.add_argument("--local-backup", action="store_true", help="Create local tar.gz instead")
    parser.add_argument("--use-api", action="store_true", help="Force Google Drive API mode")

    args = parser.parse_args()

    if args.local_backup:
        save_local_backup(output_dir=args.output_dir, run_name=args.run_name)
        return

    # Build what list
    if args.all:
        what = ["results", "videos", "checkpoints", "configs", "notebooks"]
    else:
        what = []
        if args.results:
            what.append("results")
        if args.videos:
            what.append("videos")
        if args.checkpoints or args.checkpoint:
            what.append("checkpoints")
        if args.configs:
            what.append("configs")
        if args.notebooks:
            what.append("notebooks")

    if not what:
        parser.print_help()
        print("\nError: Specify --all or at least one of --results, --videos, --checkpoints, etc.")
        sys.exit(1)

    save_all(
        output_dir=args.output_dir,
        drive_base=args.drive_base,
        checkpoint_path=args.checkpoint,
        what=what,
        use_api=args.use_api,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
