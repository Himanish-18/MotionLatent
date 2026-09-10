"""
PAMAP2 dataset audit utilities.

Provides functions for comprehensive programmatic inspection of the
PAMAP2 Physical Activity Monitoring dataset.  All functions accept
raw DataFrames and return structured results — they do NOT modify
the underlying data.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.discovery import DatasetPaths, extract_subject_id, file_size_mb
from src.data.schema import (
    ACTIVITY_MAP,
    COLUMN_NAMES,
    TOTAL_COLUMNS,
    NOMINAL_SAMPLING_FREQ_HZ,
    classify_columns,
    get_primary_motion_columns,
    get_excluded_columns,
)


# ===================================================================
# Data loading
# ===================================================================
def load_dat_file(filepath: Path, subject_id: Optional[int] = None) -> pd.DataFrame:
    """Load a single PAMAP2 ``.dat`` file into a DataFrame.

    Parameters
    ----------
    filepath : Path
        Path to the .dat file.
    subject_id : int, optional
        If provided, added as a ``subject_id`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with named columns per the PAMAP2 schema.
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES if len(COLUMN_NAMES) == TOTAL_COLUMNS else None,
        na_values=["NaN", "nan"],
        engine="c",
    )
    # Validate column count
    if df.shape[1] != TOTAL_COLUMNS:
        warnings.warn(
            f"{filepath.name}: expected {TOTAL_COLUMNS} columns, "
            f"got {df.shape[1]}"
        )
    else:
        df.columns = COLUMN_NAMES

    if subject_id is not None:
        df.insert(0, "subject_id", subject_id)

    return df


def load_all_protocol(paths: DatasetPaths) -> pd.DataFrame:
    """Load and concatenate all Protocol .dat files."""
    frames = []
    for f in paths.protocol_files:
        sid = extract_subject_id(f.name)
        frames.append(load_dat_file(f, subject_id=sid))
    return pd.concat(frames, ignore_index=True)


def load_all_optional(paths: DatasetPaths) -> pd.DataFrame:
    """Load and concatenate all Optional .dat files."""
    frames = []
    for f in paths.optional_files:
        sid = extract_subject_id(f.name)
        frames.append(load_dat_file(f, subject_id=sid))
    return pd.concat(frames, ignore_index=True)


# ===================================================================
# File-level audit
# ===================================================================
@dataclass
class FileAudit:
    filename: str
    subject_id: int
    size_mb: float
    num_rows: int
    num_columns: int
    directory: str  # "Protocol" or "Optional"


def audit_files_from_frames(
    paths: DatasetPaths,
    protocol_frames: Dict[int, pd.DataFrame],
    optional_frames: Optional[Dict[int, pd.DataFrame]] = None,
) -> List[FileAudit]:
    """Produce per-file audit records from pre-loaded DataFrames."""
    results: List[FileAudit] = []
    for f in paths.protocol_files:
        sid = extract_subject_id(f.name)
        df = protocol_frames.get(sid)
        nrows = len(df) if df is not None else 0
        ncols = df.shape[1] if df is not None else 0
        results.append(FileAudit(
            filename=f.name,
            subject_id=sid,
            size_mb=round(file_size_mb(f), 2),
            num_rows=nrows,
            num_columns=ncols,
            directory="Protocol",
        ))
    if optional_frames:
        for f in paths.optional_files:
            sid = extract_subject_id(f.name)
            df = optional_frames.get(sid)
            nrows = len(df) if df is not None else 0
            ncols = df.shape[1] if df is not None else 0
            results.append(FileAudit(
                filename=f.name,
                subject_id=sid,
                size_mb=round(file_size_mb(f), 2),
                num_rows=nrows,
                num_columns=ncols,
                directory="Optional",
            ))
    return results


# ===================================================================
# Missing-value analysis
# ===================================================================
def missing_value_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing-value counts and percentages."""
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(4)
    return pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_pct": pct.values,
        "total_rows": total,
    })


def missing_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value percentage per subject per column."""
    groups = df.groupby("subject_id")
    records = []
    for sid, grp in groups:
        total = len(grp)
        for col in grp.columns:
            if col == "subject_id":
                continue
            miss = int(grp[col].isnull().sum())
            records.append({
                "subject_id": sid,
                "column": col,
                "missing_count": miss,
                "missing_pct": round(miss / total * 100, 4),
            })
    return pd.DataFrame(records)


# ===================================================================
# Activity analysis
# ===================================================================
def activity_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return activity frequency table with names."""
    counts = df["activity_id"].value_counts().sort_index()
    total = counts.sum()
    records = []
    for aid, cnt in counts.items():
        records.append({
            "activity_id": int(aid),
            "activity_name": ACTIVITY_MAP.get(int(aid), f"unknown_{aid}"),
            "count": int(cnt),
            "percentage": round(cnt / total * 100, 4),
        })
    return pd.DataFrame(records)


def activity_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Return observations per subject per activity."""
    ct = df.groupby(["subject_id", "activity_id"]).size().reset_index(name="count")
    ct["activity_name"] = ct["activity_id"].map(ACTIVITY_MAP).fillna("unknown")
    return ct


def missing_activities_per_subject(df: pd.DataFrame) -> Dict[int, List[int]]:
    """Return dict of subject_id → list of activity IDs not present."""
    all_activities = set(df["activity_id"].unique())
    result: Dict[int, List[int]] = {}
    for sid, grp in df.groupby("subject_id"):
        present = set(grp["activity_id"].unique())
        missing = sorted(all_activities - present)
        if missing:
            result[int(sid)] = missing
    return result


# ===================================================================
# Subject / session analysis
# ===================================================================
def subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-subject summary: observations, activities, duration."""
    records = []
    for sid, grp in df.groupby("subject_id"):
        ts = grp["timestamp"]
        duration_s = (ts.max() - ts.min()) if len(ts) > 1 else 0.0
        records.append({
            "subject_id": int(sid),
            "observations": len(grp),
            "num_activities": grp["activity_id"].nunique(),
            "activities": sorted(grp["activity_id"].unique().tolist()),
            "duration_seconds": round(float(duration_s), 2),
            "duration_minutes": round(float(duration_s) / 60, 2),
        })
    return pd.DataFrame(records)


# ===================================================================
# Timestamp / sampling analysis
# ===================================================================
@dataclass
class SamplingStats:
    subject_id: int
    num_samples: int
    duration_s: float
    mean_dt: float
    median_dt: float
    std_dt: float
    min_dt: float
    max_dt: float
    empirical_freq_hz: float
    num_gaps: int          # intervals > 2× median
    num_duplicates: int    # dt == 0
    is_monotonic: bool


def sampling_analysis(df: pd.DataFrame) -> List[SamplingStats]:
    """Analyze timestamp characteristics per subject."""
    results: List[SamplingStats] = []
    for sid, grp in df.groupby("subject_id"):
        ts = grp["timestamp"].values
        if len(ts) < 2:
            continue
        dt = np.diff(ts)
        median_dt = float(np.median(dt))
        gap_threshold = 2.0 * median_dt if median_dt > 0 else 0.02
        results.append(SamplingStats(
            subject_id=int(sid),
            num_samples=len(ts),
            duration_s=round(float(ts[-1] - ts[0]), 4),
            mean_dt=round(float(np.mean(dt)), 6),
            median_dt=round(float(median_dt), 6),
            std_dt=round(float(np.std(dt)), 6),
            min_dt=round(float(np.min(dt)), 6),
            max_dt=round(float(np.max(dt)), 6),
            empirical_freq_hz=round(1.0 / float(np.median(dt)), 2) if median_dt > 0 else 0.0,
            num_gaps=int(np.sum(dt > gap_threshold)),
            num_duplicates=int(np.sum(dt == 0.0)),
            is_monotonic=bool(np.all(dt >= 0)),
        ))
    return results


# ===================================================================
# Data quality analysis
# ===================================================================
@dataclass
class ChannelQuality:
    column: str
    nan_count: int
    nan_pct: float
    inf_count: int
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    variance: float
    is_constant: bool
    is_near_zero_var: bool  # std < 1e-6


def channel_quality_analysis(df: pd.DataFrame) -> List[ChannelQuality]:
    """Analyze quality metrics for each numeric column."""
    results = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    total = len(df)
    for col in numeric_cols:
        if col == "subject_id":
            continue
        vals = df[col]
        nan_count = int(vals.isnull().sum())
        clean = vals.dropna()
        if len(clean) == 0:
            results.append(ChannelQuality(
                column=col, nan_count=nan_count, nan_pct=round(nan_count / total * 100, 4),
                inf_count=0, min_val=float("nan"), max_val=float("nan"),
                mean_val=float("nan"), std_val=float("nan"), variance=float("nan"),
                is_constant=True, is_near_zero_var=True,
            ))
            continue
        inf_count = int(np.isinf(clean.values).sum())
        finite = clean[np.isfinite(clean.values)]
        std_val = float(finite.std()) if len(finite) > 1 else 0.0
        results.append(ChannelQuality(
            column=col,
            nan_count=nan_count,
            nan_pct=round(nan_count / total * 100, 4),
            inf_count=inf_count,
            min_val=round(float(finite.min()), 6) if len(finite) > 0 else float("nan"),
            max_val=round(float(finite.max()), 6) if len(finite) > 0 else float("nan"),
            mean_val=round(float(finite.mean()), 6) if len(finite) > 0 else float("nan"),
            std_val=round(std_val, 6),
            variance=round(std_val ** 2, 6),
            is_constant=bool(finite.nunique() <= 1),
            is_near_zero_var=bool(std_val < 1e-6),
        ))
    return results


def detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect duplicate rows and duplicate timestamps."""
    dup_rows = int(df.duplicated().sum())
    dup_ts = 0
    if "timestamp" in df.columns and "subject_id" in df.columns:
        dup_ts = int(
            df.duplicated(subset=["subject_id", "timestamp"]).sum()
        )
    return {
        "duplicate_rows": dup_rows,
        "duplicate_timestamps_per_subject": dup_ts,
    }


# ===================================================================
# Windowing analysis
# ===================================================================
def estimate_window_samples(
    window_durations: List[float],
    overlaps: List[float],
    total_observations: int,
    sampling_freq: float,
) -> List[Dict[str, Any]]:
    """Estimate number of windows for various configurations.

    Parameters
    ----------
    window_durations : list of float
        Window durations in seconds.
    overlaps : list of float
        Overlap fractions (0.0 to 1.0).
    total_observations : int
        Total number of raw observations.
    sampling_freq : float
        Sampling frequency in Hz.

    Returns
    -------
    List of dicts with estimation results.
    """
    results = []
    for dur in window_durations:
        window_samples = int(dur * sampling_freq)
        for ovl in overlaps:
            step = int(window_samples * (1.0 - ovl))
            if step < 1:
                step = 1
            n_windows = max(0, (total_observations - window_samples) // step + 1)
            feature_dim = window_samples * len(get_primary_motion_columns())
            results.append({
                "window_seconds": dur,
                "overlap_fraction": ovl,
                "window_samples": window_samples,
                "step_samples": step,
                "estimated_windows": n_windows,
                "feature_dimension": feature_dim,
                "feature_dimension_flat": feature_dim,
            })
    return results


# ===================================================================
# Normalization analysis
# ===================================================================
def sensor_scale_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compare scales across sensor modalities."""
    groups = classify_columns()
    records = []
    for group_name, cols in [
        ("accelerometer_16g", groups.accelerometer_16g),
        ("gyroscope", groups.gyroscope),
        ("magnetometer", groups.magnetometer),
        ("temperature", groups.temperature),
    ]:
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        vals = df[present].values.flatten()
        clean = vals[np.isfinite(vals)]
        if len(clean) == 0:
            continue
        records.append({
            "modality": group_name,
            "num_channels": len(present),
            "min": round(float(np.min(clean)), 4),
            "max": round(float(np.max(clean)), 4),
            "mean": round(float(np.mean(clean)), 4),
            "std": round(float(np.std(clean)), 4),
            "range": round(float(np.max(clean) - np.min(clean)), 4),
        })
    return pd.DataFrame(records)


# ===================================================================
# Full audit orchestrator
# ===================================================================
def run_full_audit(
    paths: DatasetPaths,
    output_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run the complete Phase 1 audit and return structured results.

    Optionally writes machine-readable JSON to *output_dir*.
    """
    print("=" * 60)
    print("PAMAP2 Phase 1 Data Audit")
    print("=" * 60)

    # --- Load data once ---
    print("\n[1/9] Loading protocol data...")
    protocol_frames: Dict[int, pd.DataFrame] = {}
    for f in paths.protocol_files:
        sid = extract_subject_id(f.name)
        print(f"       Loading {f.name}...")
        protocol_frames[sid] = load_dat_file(f, subject_id=sid)
    df_protocol = pd.concat(protocol_frames.values(), ignore_index=True)

    optional_frames: Optional[Dict[int, pd.DataFrame]] = None
    df_optional = None
    if paths.optional_files:
        print("       Loading optional data...")
        optional_frames = {}
        for f in paths.optional_files:
            sid = extract_subject_id(f.name)
            optional_frames[sid] = load_dat_file(f, subject_id=sid)
        df_optional = pd.concat(optional_frames.values(), ignore_index=True)

    # --- File audit (uses pre-loaded frames) ---
    print("[2/9] Auditing files...")
    file_audits = audit_files_from_frames(paths, protocol_frames, optional_frames)

    # --- Missing values ---
    print("[3/9] Analyzing missing values...")
    mv_analysis = missing_value_analysis(df_protocol)
    mv_by_subject = missing_by_subject(df_protocol)

    # --- Activity analysis ---
    print("[4/9] Analyzing activities...")
    act_dist = activity_distribution(df_protocol)
    act_by_subj = activity_by_subject(df_protocol)
    missing_acts = missing_activities_per_subject(df_protocol)

    # --- Subject summary ---
    print("[5/9] Analyzing subjects...")
    subj_summ = subject_summary(df_protocol)

    # --- Sampling analysis ---
    print("[6/9] Analyzing sampling/timestamps...")
    samp_stats = sampling_analysis(df_protocol)

    # --- Channel quality ---
    print("[7/9] Analyzing channel quality...")
    chan_quality = channel_quality_analysis(df_protocol)

    # --- Duplicates ---
    print("[8/9] Detecting duplicates...")
    dup_info = detect_duplicates(df_protocol)

    # --- Sensor scales ---
    print("[9/9] Analyzing sensor scales...")
    scale_analysis = sensor_scale_analysis(df_protocol)

    # --- Windowing estimation ---
    window_est = estimate_window_samples(
        window_durations=[1.0, 2.0, 3.0, 5.0],
        overlaps=[0.0, 0.25, 0.50],
        total_observations=len(df_protocol),
        sampling_freq=NOMINAL_SAMPLING_FREQ_HZ,
    )

    # --- Assemble results ---
    results: Dict[str, Any] = {
        "dataset": {
            "protocol_files": len(paths.protocol_files),
            "optional_files": len(paths.optional_files),
            "total_protocol_observations": len(df_protocol),
            "total_optional_observations": len(df_optional) if df_optional is not None else 0,
            "num_columns": df_protocol.shape[1],
            "column_names": list(df_protocol.columns),
        },
        "file_audits": [asdict(fa) for fa in file_audits],
        "missing_values": {
            "summary": mv_analysis.to_dict(orient="records"),
            "by_subject_high_missing": mv_by_subject[
                mv_by_subject["missing_pct"] > 0
            ].to_dict(orient="records"),
        },
        "activities": {
            "distribution": act_dist.to_dict(orient="records"),
            "by_subject": act_by_subj.to_dict(orient="records"),
            "missing_per_subject": {
                str(k): v for k, v in missing_acts.items()
            },
        },
        "subjects": subj_summ.to_dict(orient="records"),
        "sampling": [asdict(s) for s in samp_stats],
        "channel_quality": [asdict(c) for c in chan_quality],
        "duplicates": dup_info,
        "sensor_scales": scale_analysis.to_dict(orient="records"),
        "windowing_estimates": window_est,
    }

    # --- Write JSON ---
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "audit_output.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nAudit JSON written to: {json_path}")

    print("\n[OK] Audit complete.")
    return results, df_protocol
