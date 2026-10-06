"""
Tests for MotionLatent Phase 2 preprocessing pipeline.

Tests cover channel selection, windowing, activity purity, transient handling,
missing-value interpolation, gap rejection, NaN/Inf checks, LOSO subject
isolation, and scaler fitting correctness.

Uses synthetic fixtures -- does not require the full PAMAP2 dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.data.schema import (
    ACTIVITY_MAP,
    COLUMN_NAMES,
    TOTAL_COLUMNS,
    build_column_names,
    get_primary_motion_columns,
)
from src.data.preprocessing import (
    WINDOW_SAMPLES,
    WINDOW_STRIDE,
    NUM_CHANNELS,
    PURITY_THRESHOLD,
    MAX_INTERP_GAP,
    PRIMARY_MOTION_COLS,
    select_channels,
    handle_missing_values,
    generate_windows_for_subject,
    fit_loso_scaler,
    generate_all_loso_scalers,
    FittedScaler,
)


# ===================================================================
# Fixtures
# ===================================================================
def _make_subject_df(
    subject_id: int,
    n_rows: int = 1000,
    activities: list | None = None,
    rng_seed: int = 42,
    inject_nan_at: list | None = None,
    inject_long_gap: bool = False,
) -> pd.DataFrame:
    """Create a synthetic single-subject DataFrame with metadata + all columns."""
    rng = np.random.RandomState(rng_seed)
    if activities is None:
        activities = [1, 2, 3, 4]

    data = {}
    data["subject_id"] = np.full(n_rows, subject_id)
    data["timestamp"] = np.arange(n_rows) * 0.01  # 100 Hz
    data["activity_id"] = rng.choice(activities, size=n_rows)
    data["heart_rate"] = rng.uniform(60, 160, n_rows)

    cols = build_column_names()
    for col in cols:
        if col in data:
            continue
        if "orientation" in col:
            data[col] = np.full(n_rows, np.nan)
        else:
            data[col] = rng.randn(n_rows)

    df = pd.DataFrame(data)

    # Inject short NaN gaps in motion columns if requested
    if inject_nan_at:
        for idx in inject_nan_at:
            for col in PRIMARY_MOTION_COLS:
                df.loc[idx, col] = np.nan

    # Inject a long gap (> MAX_INTERP_GAP) if requested
    if inject_long_gap:
        gap_start = n_rows // 2
        gap_end = gap_start + MAX_INTERP_GAP + 5  # longer than allowed
        for col in PRIMARY_MOTION_COLS:
            df.loc[gap_start:gap_end, col] = np.nan

    return df


@pytest.fixture
def subject101_df() -> pd.DataFrame:
    return _make_subject_df(101, n_rows=1000, activities=[1, 2, 3, 4])


@pytest.fixture
def subject102_df() -> pd.DataFrame:
    return _make_subject_df(102, n_rows=800, activities=[1, 2, 3], rng_seed=99)


@pytest.fixture
def subject_with_short_gaps() -> pd.DataFrame:
    """Subject with a few short NaN gaps (<=MAX_INTERP_GAP)."""
    return _make_subject_df(
        103, n_rows=600,
        inject_nan_at=[50, 51, 52, 200, 201],  # short gaps of 3 and 2
    )


@pytest.fixture
def subject_with_long_gap() -> pd.DataFrame:
    """Subject with a long NaN gap (>MAX_INTERP_GAP)."""
    return _make_subject_df(104, n_rows=600, inject_long_gap=True)


@pytest.fixture
def pure_activity_df() -> pd.DataFrame:
    """Subject with uniform activity labels (all activity 4)."""
    return _make_subject_df(105, n_rows=600, activities=[4])


@pytest.fixture
def transient_df() -> pd.DataFrame:
    """Subject where all windows will be transient (activity 0)."""
    return _make_subject_df(106, n_rows=600, activities=[0])


@pytest.fixture
def mixed_activity_df() -> pd.DataFrame:
    """Subject with rapidly alternating activities to produce mixed windows."""
    rng = np.random.RandomState(77)
    n = 600
    df = _make_subject_df(107, n_rows=n, rng_seed=77)
    # Alternate activity every 10 samples -> no window will have 80% purity
    acts = np.tile(np.repeat([1, 2, 3, 4, 5], 10), n // 50 + 1)[:n]
    df["activity_id"] = acts
    return df


# ===================================================================
# Channel selection tests
# ===================================================================
class TestChannelSelection:
    def test_select_channels_returns_exact_27_motion(self, subject101_df):
        result = select_channels(subject101_df)
        motion_cols = [c for c in result.columns if c not in ["subject_id", "timestamp", "activity_id"]]
        assert len(motion_cols) == 27

    def test_select_channels_preserves_metadata(self, subject101_df):
        result = select_channels(subject101_df)
        assert "subject_id" in result.columns
        assert "timestamp" in result.columns
        assert "activity_id" in result.columns

    def test_select_channels_excludes_heart_rate(self, subject101_df):
        result = select_channels(subject101_df)
        assert "heart_rate" not in result.columns

    def test_select_channels_excludes_orientation(self, subject101_df):
        result = select_channels(subject101_df)
        orient_cols = [c for c in result.columns if "orientation" in c]
        assert len(orient_cols) == 0

    def test_select_channels_excludes_temperature(self, subject101_df):
        result = select_channels(subject101_df)
        temp_cols = [c for c in result.columns if "temperature" in c]
        assert len(temp_cols) == 0

    def test_select_channels_excludes_acc_6g(self, subject101_df):
        result = select_channels(subject101_df)
        acc6g_cols = [c for c in result.columns if "acc_6g" in c]
        assert len(acc6g_cols) == 0

    def test_channel_order_matches_schema(self, subject101_df):
        result = select_channels(subject101_df)
        motion_cols = [c for c in result.columns if c not in ["subject_id", "timestamp", "activity_id"]]
        assert motion_cols == PRIMARY_MOTION_COLS


# ===================================================================
# Window generation tests
# ===================================================================
class TestWindowGeneration:
    def test_window_length_is_256(self, subject101_df):
        df = select_channels(subject101_df)
        df_clean, _ = handle_missing_values(df)
        windows, _, _ = generate_windows_for_subject(df_clean)
        if windows.shape[0] > 0:
            assert windows.shape[1] == 256

    def test_window_channels_is_27(self, subject101_df):
        df = select_channels(subject101_df)
        df_clean, _ = handle_missing_values(df)
        windows, _, _ = generate_windows_for_subject(df_clean)
        if windows.shape[0] > 0:
            assert windows.shape[2] == 27

    def test_stride_is_128(self, subject101_df):
        df = select_channels(subject101_df)
        df_clean, _ = handle_missing_values(df)
        n = len(df_clean)
        expected_windows = max(0, (n - WINDOW_SAMPLES) // WINDOW_STRIDE + 1)
        windows, _, _ = generate_windows_for_subject(df_clean)
        assert windows.shape[0] == expected_windows

    def test_no_nan_in_accepted_windows(self, subject101_df):
        df = select_channels(subject101_df)
        df_clean, _ = handle_missing_values(df)
        windows, _, _ = generate_windows_for_subject(df_clean)
        assert np.all(np.isfinite(windows))

    def test_no_inf_in_accepted_windows(self, subject101_df):
        df = select_channels(subject101_df)
        df_clean, _ = handle_missing_values(df)
        windows, _, _ = generate_windows_for_subject(df_clean)
        assert not np.any(np.isinf(windows))

    def test_window_metadata_count_matches(self, subject101_df):
        df = select_channels(subject101_df)
        df_clean, _ = handle_missing_values(df)
        windows, infos, report = generate_windows_for_subject(df_clean)
        assert len(infos) == windows.shape[0]
        assert report.total_windows == windows.shape[0]

    def test_empty_subject_returns_empty(self):
        """Subject with fewer than 256 samples should produce 0 windows."""
        df = _make_subject_df(110, n_rows=100)
        df = select_channels(df)
        df_clean, _ = handle_missing_values(df)
        windows, infos, report = generate_windows_for_subject(df_clean)
        assert windows.shape[0] == 0
        assert len(infos) == 0


# ===================================================================
# Activity purity tests
# ===================================================================
class TestActivityPurity:
    def test_pure_activity_windows(self, pure_activity_df):
        df = select_channels(pure_activity_df)
        df_clean, _ = handle_missing_values(df)
        windows, infos, report = generate_windows_for_subject(df_clean)

        for info in infos:
            assert info.activity_purity == 1.0
            assert not info.is_mixed_label
            assert info.dominant_activity == 4

        assert report.pure_windows == len(infos)
        assert report.mixed_windows == 0

    def test_transient_windows(self, transient_df):
        df = select_channels(transient_df)
        df_clean, _ = handle_missing_values(df)
        windows, infos, report = generate_windows_for_subject(df_clean)

        for info in infos:
            assert info.is_transient
            assert info.dominant_activity == 0

        assert report.transient_windows == len(infos)

    def test_mixed_windows_detected(self, mixed_activity_df):
        df = select_channels(mixed_activity_df)
        df_clean, _ = handle_missing_values(df)
        windows, infos, report = generate_windows_for_subject(df_clean)

        # With activities alternating every 10 samples in a 256-sample window,
        # no single activity can reach 80% purity
        mixed_count = sum(1 for i in infos if i.is_mixed_label)
        assert mixed_count > 0
        assert report.mixed_windows > 0

    def test_purity_threshold(self, subject101_df):
        df = select_channels(subject101_df)
        df_clean, _ = handle_missing_values(df)
        _, infos, _ = generate_windows_for_subject(df_clean)

        for info in infos:
            if not info.is_mixed_label:
                assert info.activity_purity >= PURITY_THRESHOLD


# ===================================================================
# Missing-value tests
# ===================================================================
class TestMissingValues:
    def test_short_gap_interpolated(self, subject_with_short_gaps):
        df = select_channels(subject_with_short_gaps)
        df_clean, report = handle_missing_values(df)

        assert report.missing_before > 0
        assert report.rows_dropped == 0  # short gaps should be interpolated
        assert report.rows_retained == report.total_rows

    def test_long_gap_causes_row_drop(self, subject_with_long_gap):
        df = select_channels(subject_with_long_gap)
        df_clean, report = handle_missing_values(df)

        assert report.long_gaps_detected > 0
        assert report.rows_dropped > 0

    def test_no_nan_after_handling(self, subject_with_short_gaps):
        df = select_channels(subject_with_short_gaps)
        df_clean, _ = handle_missing_values(df)
        motion_cols = [c for c in PRIMARY_MOTION_COLS if c in df_clean.columns]
        assert df_clean[motion_cols].isnull().sum().sum() == 0

    def test_clean_data_unchanged(self, subject101_df):
        df = select_channels(subject101_df)
        df_clean, report = handle_missing_values(df)
        assert report.missing_before == 0
        assert report.rows_dropped == 0
        assert report.rows_retained == len(df)


# ===================================================================
# LOSO and scaler tests
# ===================================================================
class TestLOSOScaler:
    def _make_windows_dict(self) -> Dict[int, np.ndarray]:
        """Create synthetic windows for 3 subjects."""
        rng = np.random.RandomState(42)
        return {
            101: rng.randn(50, WINDOW_SAMPLES, NUM_CHANNELS) * 10 + 5,
            102: rng.randn(40, WINDOW_SAMPLES, NUM_CHANNELS) * 3 + 1,
            103: rng.randn(30, WINDOW_SAMPLES, NUM_CHANNELS) * 7 - 2,
        }

    def test_scaler_excludes_test_subject(self):
        windows = self._make_windows_dict()
        scaler = fit_loso_scaler(windows, test_subject=101)
        assert 101 not in scaler.training_subjects
        assert scaler.test_subject == 101

    def test_scaler_fitted_on_training_only(self):
        windows = self._make_windows_dict()
        scaler = fit_loso_scaler(windows, test_subject=103)
        # Scaler should be fitted on subjects 101 and 102 only
        assert set(scaler.training_subjects) == {101, 102}

        # Compute expected stats from training data
        train_data = np.concatenate([windows[101], windows[102]], axis=0)
        expected_mean = train_data.reshape(-1, NUM_CHANNELS).mean(axis=0)
        expected_std = train_data.reshape(-1, NUM_CHANNELS).std(axis=0)

        np.testing.assert_allclose(scaler.mean, expected_mean, rtol=1e-10)
        np.testing.assert_allclose(scaler.std, expected_std, rtol=1e-10)

    def test_scaler_transform_produces_correct_shape(self):
        windows = self._make_windows_dict()
        scaler = fit_loso_scaler(windows, test_subject=101)
        transformed = scaler.transform(windows[101])
        assert transformed.shape == windows[101].shape

    def test_scaler_transform_normalizes_training_data(self):
        windows = self._make_windows_dict()
        scaler = fit_loso_scaler(windows, test_subject=103)

        # Transform training data -- should have ~zero mean, ~unit std
        train_data = np.concatenate([windows[101], windows[102]], axis=0)
        transformed = scaler.transform(train_data)
        reshaped = transformed.reshape(-1, NUM_CHANNELS)

        np.testing.assert_allclose(reshaped.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(reshaped.std(axis=0), 1, atol=1e-10)

    def test_generate_all_loso_scalers(self):
        windows = self._make_windows_dict()
        scalers = generate_all_loso_scalers(windows)
        assert len(scalers) == 3  # one per subject
        for test_sid, scaler in scalers.items():
            assert scaler.test_subject == test_sid
            assert test_sid not in scaler.training_subjects

    def test_no_subject_leakage(self):
        """Verify test subject data never influences scaler statistics."""
        windows = self._make_windows_dict()
        # Give subject 101 extreme values
        windows[101] = np.full_like(windows[101], 1000.0)

        scaler_101_test = fit_loso_scaler(windows, test_subject=101)
        scaler_102_test = fit_loso_scaler(windows, test_subject=102)

        # When 101 is the test subject, its extreme values should NOT
        # appear in the scaler mean
        assert np.all(scaler_101_test.mean < 100), (
            "Test subject 101's extreme values leaked into scaler!"
        )

        # When 101 is a training subject (testing 102), its values SHOULD
        # influence the mean
        # (mean will be high because 101's 1000s are included)
        assert np.any(scaler_102_test.mean > 100), (
            "Training subject 101's values should influence the scaler"
        )

    def test_scaler_serialization(self):
        windows = self._make_windows_dict()
        scaler = fit_loso_scaler(windows, test_subject=101)
        d = scaler.to_dict()
        assert "mean" in d
        assert "std" in d
        assert "training_subjects" in d
        assert "test_subject" in d
        assert len(d["mean"]) == NUM_CHANNELS


# ===================================================================
# Metadata consistency tests
# ===================================================================
class TestMetadataConsistency:
    def test_subject_id_preserved_in_window_info(self, subject101_df):
        df = select_channels(subject101_df)
        df_clean, _ = handle_missing_values(df)
        _, infos, _ = generate_windows_for_subject(df_clean)
        for info in infos:
            assert info.subject_id == 101

    def test_timestamps_ordered_in_windows(self, subject101_df):
        df = select_channels(subject101_df)
        df_clean, _ = handle_missing_values(df)
        _, infos, _ = generate_windows_for_subject(df_clean)
        timestamps = [info.start_timestamp for info in infos]
        assert timestamps == sorted(timestamps)

    def test_activity_proportions_sum_to_one(self, subject101_df):
        df = select_channels(subject101_df)
        df_clean, _ = handle_missing_values(df)
        _, infos, _ = generate_windows_for_subject(df_clean)
        for info in infos:
            total = sum(info.activity_proportions.values())
            assert abs(total - 1.0) < 1e-6


# ===================================================================
# Reproducibility
# ===================================================================
class TestReproducibility:
    def test_deterministic_output(self, subject101_df):
        df = select_channels(subject101_df)
        df1, _ = handle_missing_values(df)
        df2, _ = handle_missing_values(df)
        pd.testing.assert_frame_equal(df1, df2)

        w1, _, _ = generate_windows_for_subject(df1)
        w2, _, _ = generate_windows_for_subject(df2)
        np.testing.assert_array_equal(w1, w2)
