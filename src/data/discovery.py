"""
PAMAP2 dataset discovery utilities.

Locates the PAMAP2 dataset within the project directory tree,
regardless of exact placement.  All paths are returned as pathlib.Path
objects relative to the discovered project root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROTOCOL_DIR_NAME = "Protocol"
OPTIONAL_DIR_NAME = "Optional"
DATASET_ROOT_NAME = "PAMAP2_Dataset"
DAT_EXTENSION = ".dat"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class DatasetPaths:
    """Container for all resolved dataset paths."""

    project_root: Path
    dataset_root: Path
    protocol_dir: Path
    optional_dir: Optional[Path]
    protocol_files: List[Path] = field(default_factory=list)
    optional_files: List[Path] = field(default_factory=list)
    documentation_files: List[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery functions
# ---------------------------------------------------------------------------
def find_project_root(start: Optional[Path] = None) -> Path:
    """Walk up from *start* (default: this file's directory) until we find
    a directory that looks like the MotionLatent project root (contains
    a ``.git`` directory **or** one of the known dataset markers)."""
    if start is None:
        start = Path(__file__).resolve().parent

    current = start
    for _ in range(10):  # safety limit
        if (current / ".git").exists():
            return current
        # Also accept if dataset folder lives here
        if any(current.rglob(DATASET_ROOT_NAME)):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    raise FileNotFoundError(
        f"Could not find MotionLatent project root starting from {start}"
    )


def find_dataset_root(project_root: Path) -> Path:
    """Locate the innermost ``PAMAP2_Dataset`` directory that actually
    contains ``Protocol/`` and/or data files."""
    candidates: list[Path] = []
    for p in project_root.rglob(DATASET_ROOT_NAME):
        if p.is_dir():
            # Prefer the deepest one that contains Protocol/
            if (p / PROTOCOL_DIR_NAME).is_dir():
                candidates.append(p)
    if not candidates:
        raise FileNotFoundError(
            f"No {DATASET_ROOT_NAME}/{PROTOCOL_DIR_NAME} found under {project_root}"
        )
    # Return the deepest match (handles nested PAMAP2_Dataset/PAMAP2_Dataset)
    return max(candidates, key=lambda c: len(c.parts))


def list_dat_files(directory: Path) -> List[Path]:
    """Return sorted list of ``.dat`` files in *directory*."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix == DAT_EXTENSION)


def list_documentation_files(dataset_root: Path) -> List[Path]:
    """Return non-dat files (PDFs, READMEs, etc.) in *dataset_root*."""
    return sorted(
        p
        for p in dataset_root.iterdir()
        if p.is_file() and p.suffix != DAT_EXTENSION
    )


def discover_dataset(project_root: Optional[Path] = None) -> DatasetPaths:
    """High-level discovery: find everything and return a ``DatasetPaths``."""
    if project_root is None:
        project_root = find_project_root()
    else:
        project_root = Path(project_root).resolve()

    dataset_root = find_dataset_root(project_root)
    protocol_dir = dataset_root / PROTOCOL_DIR_NAME
    optional_dir = dataset_root / OPTIONAL_DIR_NAME

    return DatasetPaths(
        project_root=project_root,
        dataset_root=dataset_root,
        protocol_dir=protocol_dir,
        optional_dir=optional_dir if optional_dir.is_dir() else None,
        protocol_files=list_dat_files(protocol_dir),
        optional_files=list_dat_files(optional_dir) if optional_dir.is_dir() else [],
        documentation_files=list_documentation_files(dataset_root),
    )


def file_size_mb(path: Path) -> float:
    """Return file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)


def extract_subject_id(filename: str) -> int:
    """Extract numeric subject ID from a filename like ``subject101.dat``."""
    name = Path(filename).stem  # e.g. "subject101"
    digits = "".join(c for c in name if c.isdigit())
    if not digits:
        raise ValueError(f"Cannot extract subject ID from '{filename}'")
    return int(digits)
