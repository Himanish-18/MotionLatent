"""
PAMAP2 preprocessing pipeline -- Phase 2.

Implements leakage-safe preprocessing from raw PAMAP2 data to
fixed-length windowed tensors ready for Phase 3 experiments.

Pipeline stages:
  1. Load raw data (reuse Phase 1 loader)
  2. Select primary motion channels (27 channels)
  3. Handle missing values (conservative interpolation)
  4. Generate fixed-length windows (256 samples, stride 128)
  5. Compute window labels and purity
  6. Subject-aware normalization (LOSO-compatible)
  7. Save processed dataset
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.schema import (
    ACTIVITY_MAP,
    COLUMN_NAMES,
    NOMINAL_SAMPLING_FREQ_HZ,
    get_primary_motion_columns,
)
from src.data.discovery import DatasetPaths, extract_subject_id
from src.data.audit import load_dat_file


# ===================================================================
# Constants
# ===================================================================
WINDOW_SAMPLES = 256
WINDOW_STRIDE = 128  # 50% overlap
NUM_CHANNELS = 27
PURITY_THRESHOLD = 0.80
MAX_INTERP_GAP = 10  # max consecutive NaN samples for interpolation

# Metadata columns retained alongside sensor data
METADATA_COLS = ["subject_id", "timestamp", "activity_id"]

PRIMARY_MOTION_COLS: List[str] = get_primary_motion_columns()


# ===================================================================
# Channel selection
# ===================================================================
def select_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Select metadata + primary motion channels from a raw DataFrame.

    Returns a copy with only the 27 motion channels and metadata.
    """
    required = METADATA_COLS + PRIMARY_MOTION_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[required].copy()


# ===================================================================
# Missing-value handling
# ===================================================================
@dataclass
class MissingValueReport:
    """Report on missing-value handling for one subject."""
    subject_id: int
    total_rows: int
    missing_before: int          # total NaN cells in motion cols
    missing_after_interp: int    # remaining after interpolation
    rows_dropped: int            # rows dropped due to unresolved NaN
    rows_retained: int
    long_gaps_detected: int      # gaps > MAX_INTERP_GAP


def _find_gap_lengths(mask: np.ndarray) -> np.ndarray:
    """Return array of consecutive-True run lengths in a boolean mask."""
    if not mask.any():
        return np.array([], dtype=int)
    # Find transitions
    padded = np.concatenate([[False], mask, [False]])
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return ends - starts


def handle_missing_values(
    df: pd.DataFrame,
    max_gap: int = MAX_INTERP_GAP,
) -> Tuple[pd.DataFrame, MissingValueReport]:
    """Handle missing values for a single-subject DataFrame.

    Strategy:
    1. For each motion channel, find NaN gaps.
    2. Short gaps (<= max_gap): linear interpolation.
    3. Long gaps (> max_gap): mark rows as unresolvable.
    4. Drop rows that still contain NaN in any motion channel.

    Parameters
    ----------
    df : pd.DataFrame
        Single-subject data with metadata + motion columns.
    max_gap : int
        Maximum gap length for interpolation.

    Returns
    -------
    (cleaned_df, report)
    """
    assert df["subject_id"].nunique() == 1, "Expected single-subject DataFrame"
    sid = int(df["subject_id"].iloc[0])

    motion_cols = [c for c in PRIMARY_MOTION_COLS if c in df.columns]
    total_rows = len(df)

    # Count missing before
    missing_before = int(df[motion_cols].isnull().sum().sum())

    # Detect long gaps -- mark them so interpolation won't fill them
    long_gap_count = 0
    unresolvable_rows = np.zeros(len(df), dtype=bool)

    for col in motion_cols:
        mask = df[col].isnull().values
        gap_lengths = _find_gap_lengths(mask)
        long_gaps = gap_lengths[gap_lengths > max_gap]
        long_gap_count += len(long_gaps)

        # Mark rows within long gaps as unresolvable
        if len(long_gaps) > 0:
            padded = np.concatenate([[False], mask, [False]])
            diffs = np.diff(padded.astype(int))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            for s, e in zip(starts, ends):
                if (e - s) > max_gap:
                    unresolvable_rows[s:e] = True

    # Interpolate short gaps only.
    # Strategy: replace long-gap NaN positions with a sentinel value
    # BEFORE interpolation so they cannot be filled, then restore as NaN.
    result = df.copy()
    for col in motion_cols:
        mask = result[col].isnull().values.copy()
        if not mask.any():
            continue

        # Identify long gap positions
        padded = np.concatenate([[False], mask, [False]])
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        long_gap_positions = np.zeros(len(result), dtype=bool)
        for s, e in zip(starts, ends):
            if (e - s) > max_gap:
                long_gap_positions[s:e] = True

        # Temporarily fill long-gap positions so interpolation ignores them
        # Use np.inf as sentinel (will be restored to NaN after)
        result.loc[long_gap_positions, col] = np.inf

        # Interpolate only the short gaps
        result[col] = result[col].interpolate(
            method="linear", limit=max_gap, limit_direction="both"
        )

        # Restore long-gap positions to NaN
        result.loc[long_gap_positions, col] = np.nan

    # After interpolation, check remaining NaN
    missing_after = int(result[motion_cols].isnull().sum().sum())

    # Drop rows still containing NaN in motion columns
    valid_mask = result[motion_cols].notna().all(axis=1)
    rows_dropped = int((~valid_mask).sum())
    result = result[valid_mask].reset_index(drop=True)

    report = MissingValueReport(
        subject_id=sid,
        total_rows=total_rows,
        missing_before=missing_before,
        missing_after_interp=missing_after,
        rows_dropped=rows_dropped,
        rows_retained=len(result),
        long_gaps_detected=long_gap_count,
    )

    return result, report


# ===================================================================
# Window generation
# ===================================================================
@dataclass
class WindowInfo:
    """Metadata for a single window."""
    window_idx: int
    subject_id: int
    start_timestamp: float
    end_timestamp: float
    dominant_activity: int
    activity_purity: float
    is_mixed_label: bool
    is_transient: bool          # dominant activity == 0
    activity_proportions: Dict[int, float]


@dataclass
class WindowingReport:
    """Summary of window generation for one subject."""
    subject_id: int
    input_rows: int
    total_windows: int
    pure_windows: int
    mixed_windows: int
    transient_windows: int
    rejected_windows: int       # windows with NaN/Inf
    windows_per_activity: Dict[int, int]


def generate_windows_for_subject(
    df: pd.DataFrame,
    window_size: int = WINDOW_SAMPLES,
    stride: int = WINDOW_STRIDE,
    purity_threshold: float = PURITY_THRESHOLD,
) -> Tuple[np.ndarray, List[WindowInfo], WindowingReport]:
    """Generate fixed-length overlapping windows for a single subject.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed single-subject data with metadata + motion cols.
    window_size : int
        Window length in samples (default: 256).
    stride : int
        Step size between windows (default: 128).
    purity_threshold : float
        Minimum fraction of dominant activity for pure label (default: 0.80).

    Returns
    -------
    (windows_array, window_infos, report)
        windows_array: shape (N, window_size, num_channels)
        window_infos: metadata per window
        report: summary statistics
    """
    assert df["subject_id"].nunique() == 1
    sid = int(df["subject_id"].iloc[0])

    motion_cols = [c for c in PRIMARY_MOTION_COLS if c in df.columns]
    assert len(motion_cols) == NUM_CHANNELS, (
        f"Expected {NUM_CHANNELS} motion channels, got {len(motion_cols)}"
    )

    sensor_data = df[motion_cols].values  # (N, 27)
    timestamps = df["timestamp"].values
    activities = df["activity_id"].values.astype(int)

    n_samples = len(df)
    n_windows = max(0, (n_samples - window_size) // stride + 1)

    windows_list = []
    infos = []
    rejected = 0
    activity_counts: Dict[int, int] = {}

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window_data = sensor_data[start:end]  # (256, 27)

        # Validate: no NaN/Inf in this window
        if np.any(~np.isfinite(window_data)):
            rejected += 1
            continue

        # Activity analysis
        window_activities = activities[start:end]
        unique, counts = np.unique(window_activities, return_counts=True)
        proportions = {int(a): float(c / window_size) for a, c in zip(unique, counts)}
        dominant_idx = np.argmax(counts)
        dominant_act = int(unique[dominant_idx])
        purity = float(counts[dominant_idx] / window_size)

        is_mixed = purity < purity_threshold
        is_transient = dominant_act == 0

        info = WindowInfo(
            window_idx=len(windows_list),
            subject_id=sid,
            start_timestamp=float(timestamps[start]),
            end_timestamp=float(timestamps[end - 1]),
            dominant_activity=dominant_act,
            activity_purity=round(purity, 4),
            is_mixed_label=is_mixed,
            is_transient=is_transient,
            activity_proportions=proportions,
        )

        windows_list.append(window_data)
        infos.append(info)

        # Count for report
        if not is_mixed:
            activity_counts[dominant_act] = activity_counts.get(dominant_act, 0) + 1

    if windows_list:
        windows_array = np.stack(windows_list, axis=0)  # (N, 256, 27)
    else:
        windows_array = np.empty((0, window_size, NUM_CHANNELS), dtype=np.float64)

    pure_count = sum(1 for w in infos if not w.is_mixed_label and not w.is_transient)
    mixed_count = sum(1 for w in infos if w.is_mixed_label)
    transient_count = sum(1 for w in infos if w.is_transient and not w.is_mixed_label)

    report = WindowingReport(
        subject_id=sid,
        input_rows=n_samples,
        total_windows=len(infos),
        pure_windows=pure_count,
        mixed_windows=mixed_count,
        transient_windows=transient_count,
        rejected_windows=rejected,
        windows_per_activity=activity_counts,
    )

    return windows_array, infos, report


# ===================================================================
# Subject-aware normalization (LOSO compatible)
# ===================================================================
@dataclass
class FittedScaler:
    """Per-channel StandardScaler statistics fitted on training data."""
    mean: np.ndarray   # shape (NUM_CHANNELS,)
    std: np.ndarray    # shape (NUM_CHANNELS,)
    training_subjects: List[int]
    test_subject: int
    n_training_samples: int

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Normalize windows. X shape: (N, T, C) or (T, C)."""
        return (X - self.mean) / np.where(self.std == 0, 1.0, self.std)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "training_subjects": self.training_subjects,
            "test_subject": self.test_subject,
            "n_training_samples": self.n_training_samples,
        }


def fit_loso_scaler(
    all_windows: Dict[int, np.ndarray],
    test_subject: int,
) -> FittedScaler:
    """Fit a StandardScaler on all subjects EXCEPT test_subject.

    Parameters
    ----------
    all_windows : dict
        Mapping subject_id -> windows array (N_i, 256, 27).
    test_subject : int
        Subject to hold out.

    Returns
    -------
    FittedScaler with per-channel mean and std from training subjects.
    """
    train_subjects = sorted(s for s in all_windows if s != test_subject)
    train_arrays = [all_windows[s] for s in train_subjects if all_windows[s].shape[0] > 0]

    if not train_arrays:
        # Edge case: no training data
        return FittedScaler(
            mean=np.zeros(NUM_CHANNELS),
            std=np.ones(NUM_CHANNELS),
            training_subjects=train_subjects,
            test_subject=test_subject,
            n_training_samples=0,
        )

    # Concatenate and reshape to (total_samples, channels)
    train_data = np.concatenate(train_arrays, axis=0)  # (N, 256, 27)
    reshaped = train_data.reshape(-1, NUM_CHANNELS)  # (N*256, 27)

    mean = reshaped.mean(axis=0)  # (27,)
    std = reshaped.std(axis=0)    # (27,)

    return FittedScaler(
        mean=mean,
        std=std,
        training_subjects=train_subjects,
        test_subject=test_subject,
        n_training_samples=int(train_data.shape[0]),
    )


def generate_all_loso_scalers(
    all_windows: Dict[int, np.ndarray],
) -> Dict[int, FittedScaler]:
    """Generate LOSO scalers for each subject as test."""
    scalers = {}
    for test_sid in sorted(all_windows.keys()):
        scalers[test_sid] = fit_loso_scaler(all_windows, test_sid)
    return scalers


# ===================================================================
# Full pipeline
# ===================================================================
@dataclass
class PipelineReport:
    """Complete Phase 2 pipeline report."""
    raw_observations: int
    observations_after_channel_select: int
    observations_after_missing_handling: int
    observations_rejected: int
    missing_value_reports: List[Dict[str, Any]]
    total_windows: int
    pure_windows: int
    mixed_windows: int
    transient_windows: int
    rejected_windows: int
    windows_per_subject: Dict[int, int]
    windows_per_activity: Dict[int, int]
    subject_activity_coverage: Dict[int, List[int]]
    final_tensor_shape: List[int]
    normalization: str
    loso_subjects: List[int]
    data_quality_warnings: List[str]


def run_preprocessing_pipeline(
    paths: DatasetPaths,
    output_dir: Optional[Path] = None,
) -> Tuple[Dict[int, np.ndarray], List[WindowInfo], PipelineReport]:
    """Run the complete Phase 2 preprocessing pipeline.

    Parameters
    ----------
    paths : DatasetPaths
        Discovered dataset paths.
    output_dir : Path, optional
        If provided, save processed data and report here.

    Returns
    -------
    (all_windows, all_infos, report)
    """
    print("=" * 60)
    print("MotionLatent Phase 2 -- Preprocessing Pipeline")
    print("=" * 60)

    all_windows: Dict[int, np.ndarray] = {}
    all_infos: List[WindowInfo] = []
    mv_reports: List[Dict[str, Any]] = []
    windowing_reports: List[Dict[str, Any]] = []
    warnings_list: List[str] = []

    raw_obs_total = 0
    selected_obs_total = 0
    retained_obs_total = 0
    rejected_obs_total = 0

    total_pure = 0
    total_mixed = 0
    total_transient = 0
    total_rejected_windows = 0
    windows_per_subject: Dict[int, int] = {}
    windows_per_activity: Dict[int, int] = {}
    subject_activity_coverage: Dict[int, List[int]] = {}

    # Process each subject independently
    for f in paths.protocol_files:
        sid = extract_subject_id(f.name)
        print(f"\n[Subject {sid}] Loading {f.name}...")

        # 1. Load raw
        df_raw = load_dat_file(f, subject_id=sid)
        raw_obs = len(df_raw)
        raw_obs_total += raw_obs

        # 2. Select channels
        print(f"  Channel selection: {NUM_CHANNELS} motion channels + metadata")
        df_selected = select_channels(df_raw)
        selected_obs = len(df_selected)
        selected_obs_total += selected_obs

        # 3. Handle missing values
        print(f"  Handling missing values (max_gap={MAX_INTERP_GAP})...")
        df_clean, mv_report = handle_missing_values(df_selected)
        retained_obs_total += mv_report.rows_retained
        rejected_obs_total += mv_report.rows_dropped
        mv_reports.append(asdict(mv_report))

        if mv_report.rows_dropped > 0:
            print(f"  -> {mv_report.rows_dropped} rows dropped "
                  f"({mv_report.long_gaps_detected} long gaps)")
        if mv_report.rows_retained < WINDOW_SAMPLES:
            msg = (f"Subject {sid}: only {mv_report.rows_retained} rows after "
                   f"missing-value handling (< {WINDOW_SAMPLES} required for 1 window)")
            warnings_list.append(msg)
            print(f"  [WARN] {msg}")
            all_windows[sid] = np.empty((0, WINDOW_SAMPLES, NUM_CHANNELS))
            windows_per_subject[sid] = 0
            subject_activity_coverage[sid] = []
            continue

        # 4. Generate windows
        print(f"  Generating windows (size={WINDOW_SAMPLES}, stride={WINDOW_STRIDE})...")
        windows, infos, w_report = generate_windows_for_subject(df_clean)

        all_windows[sid] = windows
        all_infos.extend(infos)
        windowing_reports.append(asdict(w_report))

        windows_per_subject[sid] = w_report.total_windows
        total_pure += w_report.pure_windows
        total_mixed += w_report.mixed_windows
        total_transient += w_report.transient_windows
        total_rejected_windows += w_report.rejected_windows

        # Aggregate activity counts
        for act, cnt in w_report.windows_per_activity.items():
            windows_per_activity[act] = windows_per_activity.get(act, 0) + cnt

        # Track coverage
        subject_activities = sorted(set(
            info.dominant_activity for info in infos
            if not info.is_mixed_label
        ))
        subject_activity_coverage[sid] = subject_activities

        print(f"  -> {w_report.total_windows} windows "
              f"(pure={w_report.pure_windows}, mixed={w_report.mixed_windows}, "
              f"transient={w_report.transient_windows}, rejected={w_report.rejected_windows})")

    # --- Validate all windows ---
    print("\n[Validation] Checking all windows...")
    total_windows = sum(w.shape[0] for w in all_windows.values())
    for sid, wins in all_windows.items():
        if wins.shape[0] == 0:
            continue
        assert wins.shape[1] == WINDOW_SAMPLES, (
            f"Subject {sid}: window length {wins.shape[1]} != {WINDOW_SAMPLES}")
        assert wins.shape[2] == NUM_CHANNELS, (
            f"Subject {sid}: channels {wins.shape[2]} != {NUM_CHANNELS}")
        if np.any(~np.isfinite(wins)):
            msg = f"Subject {sid}: NaN/Inf detected in accepted windows!"
            warnings_list.append(msg)
            print(f"  [WARN] {msg}")
        else:
            print(f"  Subject {sid}: {wins.shape[0]} windows OK "
                  f"(shape {wins.shape})")

    # --- Build report ---
    # Compute combined tensor shape for reporting
    combined_shape = [total_windows, WINDOW_SAMPLES, NUM_CHANNELS]

    report = PipelineReport(
        raw_observations=raw_obs_total,
        observations_after_channel_select=selected_obs_total,
        observations_after_missing_handling=retained_obs_total,
        observations_rejected=rejected_obs_total,
        missing_value_reports=mv_reports,
        total_windows=total_windows,
        pure_windows=total_pure,
        mixed_windows=total_mixed,
        transient_windows=total_transient,
        rejected_windows=total_rejected_windows,
        windows_per_subject=windows_per_subject,
        windows_per_activity=windows_per_activity,
        subject_activity_coverage=subject_activity_coverage,
        final_tensor_shape=combined_shape,
        normalization="per_channel_StandardScaler_LOSO",
        loso_subjects=sorted(all_windows.keys()),
        data_quality_warnings=warnings_list,
    )

    # --- Save ---
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_dataset(all_windows, all_infos, report, output_dir)

    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Raw observations:     {raw_obs_total:,}")
    print(f"  After channel select: {selected_obs_total:,}")
    print(f"  After missing-value:  {retained_obs_total:,}")
    print(f"  Observations dropped: {rejected_obs_total:,}")
    print(f"  Total windows:        {total_windows:,}")
    print(f"  Pure windows:         {total_pure:,}")
    print(f"  Mixed windows:        {total_mixed:,}")
    print(f"  Transient windows:    {total_transient:,}")
    print(f"  Rejected windows:     {total_rejected_windows:,}")
    print(f"  Final tensor shape:   {combined_shape}")
    print(f"  LOSO subjects:        {sorted(all_windows.keys())}")
    if warnings_list:
        print(f"\n  WARNINGS:")
        for w in warnings_list:
            print(f"    - {w}")

    return all_windows, all_infos, report


# ===================================================================
# Dataset I/O
# ===================================================================
def _save_dataset(
    all_windows: Dict[int, np.ndarray],
    all_infos: List[WindowInfo],
    report: PipelineReport,
    output_dir: Path,
) -> None:
    """Save processed windows and metadata to disk.

    Format: one .npz per subject + metadata JSON + report JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-subject windows
    for sid, windows in all_windows.items():
        npz_path = output_dir / f"windows_subject{sid}.npz"
        np.savez_compressed(npz_path, windows=windows)
        print(f"  Saved {npz_path.name} ({windows.shape})")

    # Save window metadata as JSON
    meta_records = []
    for info in all_infos:
        meta_records.append({
            "window_idx": info.window_idx,
            "subject_id": info.subject_id,
            "start_timestamp": info.start_timestamp,
            "end_timestamp": info.end_timestamp,
            "dominant_activity": info.dominant_activity,
            "activity_name": ACTIVITY_MAP.get(info.dominant_activity, "unknown"),
            "activity_purity": info.activity_purity,
            "is_mixed_label": info.is_mixed_label,
            "is_transient": info.is_transient,
            "activity_proportions": info.activity_proportions,
        })
    meta_path = output_dir / "window_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_records, f, indent=2)
    print(f"  Saved {meta_path.name} ({len(meta_records)} records)")

    # Save report
    report_path = output_dir / "phase2_pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print(f"  Saved {report_path.name}")

    # Save channel order for reproducibility
    channels_path = output_dir / "channel_order.json"
    with open(channels_path, "w", encoding="utf-8") as f:
        json.dump(PRIMARY_MOTION_COLS, f, indent=2)
    print(f"  Saved {channels_path.name}")


def load_subject_windows(output_dir: Path, subject_id: int) -> np.ndarray:
    """Load saved windows for a specific subject."""
    npz_path = Path(output_dir) / f"windows_subject{subject_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No saved windows for subject {subject_id}")
    data = np.load(npz_path)
    return data["windows"]


def load_window_metadata(output_dir: Path) -> List[Dict[str, Any]]:
    """Load window metadata from JSON."""
    meta_path = Path(output_dir) / "window_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
