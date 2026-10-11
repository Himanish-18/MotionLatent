"""
LOSO (Leave-One-Subject-Out) Dataset Loader & Splitter -- Phase 3.

Provides leakage-safe loading, normalization, and window filtering for
evaluating unsupervised representations across all 9 subjects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.preprocessing import (
    NUM_CHANNELS,
    WINDOW_SAMPLES,
    FittedScaler,
    load_subject_windows,
    load_window_metadata,
)

ALL_SUBJECTS = [101, 102, 103, 104, 105, 106, 107, 108, 109]
FLATTENED_DIM = WINDOW_SAMPLES * NUM_CHANNELS  # 256 * 27 = 6912


@dataclass
class FoldData:
    """Encapsulates all train and test arrays for a single LOSO fold."""
    test_subject: int
    training_subjects: List[int]
    X_train: np.ndarray          # (N_train, 6912), float32, normalized
    y_train: np.ndarray          # (N_train,), int
    train_subject_ids: np.ndarray# (N_train,), int
    X_test: np.ndarray           # (N_test, 6912), float32, normalized
    y_test: np.ndarray           # (N_test,), int
    test_subject_ids: np.ndarray # (N_test,), int
    scaler: FittedScaler

    @property
    def unique_train_classes(self) -> np.ndarray:
        return np.unique(self.y_train)


def load_fold_scaler(processed_dir: Path, test_subject: int) -> FittedScaler:
    """Load the pre-computed training scaler for the specified test fold."""
    scaler_path = Path(processed_dir) / "scalers" / f"scaler_test{test_subject}.json"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    with open(scaler_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return FittedScaler(
        mean=np.array(data["mean"], dtype=np.float32),
        std=np.array(data["std"], dtype=np.float32),
        training_subjects=data["training_subjects"],
        test_subject=data["test_subject"],
        n_training_samples=data["n_training_samples"],
    )


def load_fold_data(
    test_subject: int,
    processed_dir: Optional[Path] = None,
    pure_only: bool = True,
    exclude_transient: bool = True,
) -> FoldData:
    """Load and prepare train and test arrays for a LOSO fold.

    Parameters
    ----------
    test_subject : int
        The subject held out for testing.
    processed_dir : Path, optional
        Directory containing processed artifacts (data/processed).
        Defaults to data/processed relative to repo root.
    pure_only : bool, default=True
        Whether to filter to windows meeting the 80% purity threshold.
    exclude_transient : bool, default=True
        Whether to exclude Activity 0 (transient) windows from evaluation.

    Returns
    -------
    FoldData containing normalized flattened arrays and labels.
    """
    if processed_dir is None:
        processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    processed_dir = Path(processed_dir)
    scaler = load_fold_scaler(processed_dir, test_subject)
    training_subjects = sorted(s for s in ALL_SUBJECTS if s != test_subject)

    all_meta = load_window_metadata(processed_dir)
    meta_by_subject: Dict[int, List[Dict[str, Any]]] = {}
    for m in all_meta:
        meta_by_subject.setdefault(m["subject_id"], []).append(m)

    # Helper to process a single subject
    def _get_subject_arrays(sid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_wins = load_subject_windows(processed_dir, sid)  # (N, 256, 27)
        s_meta = meta_by_subject.get(sid, [])
        assert len(raw_wins) == len(s_meta), f"Window/metadata count mismatch for S{sid}"

        # Build filter mask
        mask = np.ones(len(s_meta), dtype=bool)
        if pure_only:
            mask &= np.array([not m["is_mixed_label"] for m in s_meta], dtype=bool)
        if exclude_transient:
            mask &= np.array([not m["is_transient"] for m in s_meta], dtype=bool)

        if not np.any(mask):
            return (
                np.empty((0, FLATTENED_DIM), dtype=np.float32),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=int),
            )

        selected_wins = raw_wins[mask]
        # Normalize using the training scaler
        norm_wins = scaler.transform(selected_wins).astype(np.float32)
        # Flatten to (N, 6912)
        flat_wins = norm_wins.reshape(len(selected_wins), FLATTENED_DIM)

        labels = np.array([m["dominant_activity"] for m in s_meta], dtype=int)[mask]
        sids = np.full(len(labels), sid, dtype=int)

        return flat_wins, labels, sids

    # 1. Process training subjects
    train_X_list, train_y_list, train_sid_list = [], [], []
    for sid in training_subjects:
        X_s, y_s, sid_s = _get_subject_arrays(sid)
        if len(X_s) > 0:
            train_X_list.append(X_s)
            train_y_list.append(y_s)
            train_sid_list.append(sid_s)

    X_train = np.concatenate(train_X_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)
    train_subject_ids = np.concatenate(train_sid_list, axis=0)

    # 2. Process test subject
    X_test, y_test, test_subject_ids = _get_subject_arrays(test_subject)

    return FoldData(
        test_subject=test_subject,
        training_subjects=training_subjects,
        X_train=X_train,
        y_train=y_train,
        train_subject_ids=train_subject_ids,
        X_test=X_test,
        y_test=y_test,
        test_subject_ids=test_subject_ids,
        scaler=scaler,
    )
