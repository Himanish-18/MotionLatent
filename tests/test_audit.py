"""
Tests for MotionLatent data audit utilities.

Uses small synthetic fixtures to avoid requiring the full PAMAP2 dataset.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.schema import (
    ACTIVITY_MAP,
    COLUMN_NAMES,
    TOTAL_COLUMNS,
    build_column_names,
    classify_columns,
    get_primary_motion_columns,
    get_excluded_columns,
)
from src.data.discovery import extract_subject_id
from src.data.audit import (
    missing_value_analysis,
    activity_distribution,
    activity_by_subject,
    missing_activities_per_subject,
    subject_summary,
    channel_quality_analysis,
    detect_duplicates,
    estimate_window_samples,
    load_dat_file,
)


# ===================================================================
# Fixtures
# ===================================================================
@pytest.fixture
def synthetic_dat_file(tmp_path: Path) -> Path:
    """Create a minimal synthetic .dat file with 54 columns."""
    rng = np.random.RandomState(42)
    n_rows = 200
    data = rng.randn(n_rows, TOTAL_COLUMNS)
    # Column 0: timestamp (monotonically increasing)
    data[:, 0] = np.linspace(0, 2.0, n_rows)
    # Column 1: activity_id (integers)
    data[:, 1] = rng.choice([1, 2, 3, 4], size=n_rows)
    # Column 2: heart_rate
    data[:, 2] = rng.uniform(60, 150, n_rows)
    # Add some NaNs in orientation columns (indices 14-17, 31-34, 48-51)
    for col_idx in [14, 15, 16, 17, 31, 32, 33, 34, 48, 49, 50, 51]:
        data[:, col_idx] = np.nan

    filepath = tmp_path / "subject101.dat"
    np.savetxt(filepath, data, fmt="%.6f")
    return filepath


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Create a synthetic DataFrame mimicking PAMAP2 format."""
    rng = np.random.RandomState(42)
    n = 500
    data = {}
    data["subject_id"] = np.repeat([101, 102], n // 2)
    data["timestamp"] = np.concatenate([
        np.linspace(0, 5.0, n // 2),
        np.linspace(0, 5.0, n // 2),
    ])
    data["activity_id"] = rng.choice([0, 1, 2, 3, 4, 5], size=n)
    data["heart_rate"] = rng.uniform(60, 160, n)

    # Generate IMU columns
    cols = build_column_names()
    for col in cols:
        if col in data:
            continue
        if "orientation" in col:
            data[col] = np.full(n, np.nan)
        else:
            data[col] = rng.randn(n)

    return pd.DataFrame(data)


# ===================================================================
# Schema tests
# ===================================================================
class TestSchema:
    def test_column_count(self):
        assert len(COLUMN_NAMES) == TOTAL_COLUMNS == 54

    def test_build_column_names_is_deterministic(self):
        a = build_column_names()
        b = build_column_names()
        assert a == b

    def test_activity_map_has_transient(self):
        assert 0 in ACTIVITY_MAP
        assert ACTIVITY_MAP[0] == "transient"

    def test_classify_columns_coverage(self):
        groups = classify_columns()
        all_cols = (
            groups.metadata
            + groups.heart_rate
            + groups.temperature
            + groups.accelerometer_16g
            + groups.accelerometer_6g
            + groups.gyroscope
            + groups.magnetometer
            + groups.orientation
        )
        # All 54 columns should be covered
        assert set(all_cols) == set(COLUMN_NAMES)

    def test_primary_motion_columns_count(self):
        # 3 locations × 3 modalities × 3 axes = 27
        cols = get_primary_motion_columns()
        assert len(cols) == 27

    def test_excluded_columns_have_rationale(self):
        excluded = get_excluded_columns()
        for key, (cols, rationale) in excluded.items():
            assert len(cols) > 0, f"No columns for {key}"
            assert len(rationale) > 0, f"No rationale for {key}"


# ===================================================================
# Discovery tests
# ===================================================================
class TestDiscovery:
    def test_extract_subject_id(self):
        assert extract_subject_id("subject101.dat") == 101
        assert extract_subject_id("subject109.dat") == 109

    def test_extract_subject_id_invalid(self):
        with pytest.raises(ValueError):
            extract_subject_id("nodigits.dat")


# ===================================================================
# Audit tests
# ===================================================================
class TestAudit:
    def test_missing_value_analysis(self, synthetic_df: pd.DataFrame):
        result = missing_value_analysis(synthetic_df)
        assert "column" in result.columns
        assert "missing_count" in result.columns
        assert "missing_pct" in result.columns
        # Orientation columns should have 100% missing
        orient_rows = result[result["column"].str.contains("orientation")]
        assert all(orient_rows["missing_pct"] == 100.0)

    def test_activity_distribution(self, synthetic_df: pd.DataFrame):
        result = activity_distribution(synthetic_df)
        assert "activity_id" in result.columns
        assert "activity_name" in result.columns
        assert "count" in result.columns
        assert result["count"].sum() == len(synthetic_df)

    def test_activity_by_subject(self, synthetic_df: pd.DataFrame):
        result = activity_by_subject(synthetic_df)
        assert "subject_id" in result.columns
        assert "activity_id" in result.columns

    def test_missing_activities_per_subject(self, synthetic_df: pd.DataFrame):
        result = missing_activities_per_subject(synthetic_df)
        assert isinstance(result, dict)

    def test_subject_summary(self, synthetic_df: pd.DataFrame):
        result = subject_summary(synthetic_df)
        assert len(result) == 2  # two subjects
        assert "observations" in result.columns
        assert "duration_seconds" in result.columns

    def test_channel_quality(self, synthetic_df: pd.DataFrame):
        result = channel_quality_analysis(synthetic_df)
        assert len(result) > 0
        # Check orientation columns are flagged as constant/all-nan
        orient_quality = [c for c in result if "orientation" in c.column]
        for cq in orient_quality:
            assert cq.nan_pct == 100.0

    def test_detect_duplicates(self, synthetic_df: pd.DataFrame):
        result = detect_duplicates(synthetic_df)
        assert "duplicate_rows" in result
        assert "duplicate_timestamps_per_subject" in result

    def test_estimate_windows(self):
        results = estimate_window_samples(
            window_durations=[1.0, 2.0],
            overlaps=[0.0, 0.5],
            total_observations=10000,
            sampling_freq=100.0,
        )
        assert len(results) == 4  # 2 durations × 2 overlaps
        for r in results:
            assert "estimated_windows" in r
            assert r["estimated_windows"] > 0

    def test_load_dat_file(self, synthetic_dat_file: Path):
        df = load_dat_file(synthetic_dat_file, subject_id=101)
        assert "subject_id" in df.columns
        assert df.shape[1] == TOTAL_COLUMNS + 1  # +1 for subject_id
        assert len(df) == 200


# ===================================================================
# Invalid column detection
# ===================================================================
class TestInvalidColumns:
    def test_orientation_columns_all_nan(self, synthetic_df: pd.DataFrame):
        """Verify that orientation columns are detected as invalid."""
        groups = classify_columns()
        for col in groups.orientation:
            if col in synthetic_df.columns:
                assert synthetic_df[col].isnull().all(), (
                    f"Orientation column {col} should be all NaN in test data"
                )
