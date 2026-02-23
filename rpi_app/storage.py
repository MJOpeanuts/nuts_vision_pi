#!/usr/bin/env python3
"""
storage.py – External-disk discovery and job-folder management.

Looks for a disk mounted under /media/<USER>/* that contains a
"nuts_vision" sub-folder, or creates one on the first available mount.
Falls back to ~/nuts_vision_fallback if no external disk is found.
"""

import os
from pathlib import Path


NUTS_FOLDER = "nuts_vision"
JOBS_SUBDIR = "jobs"


def find_external_root() -> Path:
    """
    Return the path to the nuts_vision data root on an external disk.

    Search order:
      1. /media/<USER>/*/nuts_vision  (existing folder)
      2. /media/<USER>/*              (first writable mount → create nuts_vision)
      3. ~/nuts_vision_fallback        (microSD fallback)
    """
    user = os.environ.get("USER", os.environ.get("LOGNAME", "pi"))
    media_base = Path(f"/media/{user}")

    candidates: list[Path] = []
    if media_base.is_dir():
        candidates = sorted(media_base.iterdir())

    # 1. Prefer a mount that already has the nuts_vision folder
    for mount in candidates:
        target = mount / NUTS_FOLDER
        if target.is_dir():
            return target

    # 2. Use first writable mount and create the folder
    for mount in candidates:
        if os.access(str(mount), os.W_OK):
            target = mount / NUTS_FOLDER
            target.mkdir(parents=True, exist_ok=True)
            return target

    # 3. MicroSD fallback
    fallback = Path.home() / "nuts_vision_fallback"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def get_jobs_dir() -> Path:
    """Return (and create if needed) the jobs directory."""
    jobs = find_external_root() / JOBS_SUBDIR
    jobs.mkdir(parents=True, exist_ok=True)
    return jobs


def get_db_path() -> Path:
    """Return the SQLite database file path on the external disk."""
    return find_external_root() / "nuts_vision.db"


def create_job_dirs(job_name: str) -> tuple[Path, Path]:
    """
    Create a new job directory structure.

    Returns:
        (job_dir, crops_dir)
    """
    job_dir = get_jobs_dir() / job_name
    crops_dir = job_dir / "crops"
    job_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(exist_ok=True)
    return job_dir, crops_dir
