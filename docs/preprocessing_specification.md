# MotionLatent — Preprocessing Specification

**Phase 2 Implementation Guide**

This document specifies the preprocessing pipeline that Phase 2 will implement.
It is based on the findings from the Phase 1 data audit.

---

## Pipeline Overview

```
Raw PAMAP2 .dat files
    │
    ▼
[1] Dataset Discovery
    │
    ▼
[2] Schema Validation
    │
    ▼
[3] Invalid-Field Removal
    │
    ▼
[4] Missing-Value Handling
    │
    ▼
[5] Timestamp Validation & Resampling
    │
    ▼
[6] Signal Filtering (if justified)
    │
    ▼
[7] Normalization
    │
    ▼
[8] Temporal Windowing
    │
    ▼
[9] Window-Level Dataset
    │
    ▼
[10] Metadata Preservation
```

---

## Stage 1: Dataset Discovery

- **Input:** Project root directory
- **Transformation:** Recursively locate `PAMAP2_Dataset/Protocol/` and `Optional/` directories
- **Output:** `DatasetPaths` object with resolved file paths
- **Rationale:** Avoids hardcoded paths; works across different machines
- **Parameters:** None
- **Risks:** Dataset may be missing; fail clearly with error message

## Stage 2: Schema Validation

- **Input:** Raw `.dat` files
- **Transformation:** Verify 54-column structure; assign column names per PAMAP2 specification
- **Output:** DataFrame with named columns + `subject_id`
- **Rationale:** PAMAP2 files have no headers; column order must match documentation exactly
- **Parameters:**
  - Expected columns: 54
  - Column names: as defined in `schema.py`
- **Risks:** Corrupted files may have different column counts; warn and skip

## Stage 3: Invalid-Field Removal

- **Input:** Full 54-column DataFrame
- **Transformation:** Remove columns that are known invalid or excluded from primary representation
- **Output:** DataFrame with only valid columns retained
- **Rationale:**
  - **Orientation (12 cols):** PAMAP2 readme states "not validated," all values are NaN — MUST remove
  - **Heart rate (1 col):** Physiological, not motion — excluded from primary representation
  - **Temperature (3 cols):** Environmental, not motion — excluded
  - **Accelerometer 6g (9 cols):** Redundant with 16g, narrower range — excluded (retain for later comparison)
- **Parameters:**
  - Remove: `{loc}_orientation_{0-3}` for all locations
  - Remove: `heart_rate`
  - Remove: `{loc}_temperature` for all locations
  - Remove: `{loc}_acc_6g_{x,y,z}` for all locations
- **Risks:** Future experiments may want to re-include acc_6g; keep raw data untouched

## Stage 4: Missing-Value Handling

- **Input:** DataFrame with valid columns only
- **Transformation:**
  1. For gaps ≤ 10 consecutive NaN samples: linear interpolation
  2. For remaining NaNs after interpolation: forward fill, then backward fill
  3. Drop any rows still containing NaN (edge cases at recording boundaries)
- **Output:** DataFrame with no NaN values in motion channels
- **Rationale:**
  - Short gaps are common in IMU data due to sensor dropouts
  - Linear interpolation preserves signal continuity for short gaps
  - Longer gaps indicate sensor failure — forward/backward fill is a fallback
- **Parameters:**
  - `max_gap_samples`: 10 (100ms at 100Hz)
  - `interpolation_method`: linear
  - `fallback`: forward_fill → backward_fill → drop
- **Risks:**
  - Interpolating long gaps introduces artificial data
  - Must track how many rows are dropped per subject

## Stage 5: Timestamp Validation & Resampling

- **Input:** Interpolated DataFrame
- **Transformation:**
  1. Verify timestamp monotonicity per subject
  2. Compute empirical sampling frequency
  3. If deviation from 100Hz is > 1%, resample to exactly 100Hz
  4. Handle timestamp duplicates (average values)
- **Output:** DataFrame with uniform 100Hz sampling
- **Rationale:**
  - Phase 1 audit revealed minor jitter around the nominal 100Hz
  - Uniform sampling is required for fixed-length windowing
- **Parameters:**
  - `target_freq_hz`: 100
  - `tolerance_pct`: 1.0
- **Risks:**
  - Resampling introduces interpolation artifacts
  - Must preserve activity label alignment during resampling

## Stage 6: Signal Filtering

- **Input:** Resampled DataFrame
- **Transformation:** Apply low-pass Butterworth filter (PROVISIONAL)
- **Output:** Filtered DataFrame
- **Rationale:**
  - High-frequency noise may be present in accelerometer/gyroscope data
  - Phase 1 audit did not find compelling evidence to mandate filtering
  - This stage should be **skippable** (configurable flag)
- **Parameters:**
  - `apply`: false (default; enable only if Phase 2 experiments warrant it)
  - `type`: butterworth_lowpass
  - `cutoff_hz`: 20
  - `order`: 4
- **Risks:**
  - Aggressive filtering removes meaningful high-frequency content
  - Different activities may require different cutoff frequencies

## Stage 7: Normalization

- **Input:** Filtered (or unfiltered) DataFrame
- **Transformation:** Standardize each sensor channel to zero mean, unit variance
- **Output:** Normalized DataFrame
- **Rationale:**
  - Accelerometer (±16g), gyroscope (rad/s), and magnetometer (μT) operate at vastly different scales
  - PCA/ICA/Autoencoders are sensitive to scale differences
  - StandardScaler is preferred over MinMaxScaler for robustness
- **Parameters:**
  - `method`: StandardScaler
  - `scope`: per_channel (fit independently per column)
  - `fit_on`: **training set only** (CRITICAL for leakage prevention)
- **Risks:**
  - Fitting on test data leaks information
  - RobustScaler may be preferable if extreme outliers persist; evaluate in Phase 2

## Stage 8: Temporal Windowing

- **Input:** Normalized DataFrame (per subject)
- **Transformation:** Segment into fixed-length, overlapping windows
- **Output:** Array of shape `(N_windows, T, C)` + labels
- **Rationale:**
  - Fixed-length windows are required for batch processing in ML pipelines
  - 2.56s (256 samples) at 50% overlap is recommended:
    - Captures sufficient temporal context for most activities
    - Power-of-2 length is FFT-friendly
    - 50% overlap balances sample count vs redundancy
- **Parameters:**
  - `window_samples`: 256 (= 2.56s at 100Hz)
  - `overlap_fraction`: 0.50
  - `step_samples`: 128
- **Risks:**
  - Windows spanning activity transitions contain mixed labels
  - Majority-vote labeling for transition windows (document threshold)

### Window-Level Labeling Strategy

For each window:
1. Compute mode (most frequent) activity ID within the window
2. If the mode represents < 80% of samples, flag as "mixed" transition window
3. Mixed windows should be retained but flagged for potential exclusion from evaluation

## Stage 9: Window-Level Dataset

- **Input:** Windowed arrays per subject
- **Output:** Final dataset with structure:
  ```
  X: (N_total_windows, 256, 27)   # T=256, C=27 motion channels
  y: (N_total_windows,)            # activity labels
  subjects: (N_total_windows,)     # subject IDs for grouping
  metadata: {window_start_time, subject_id, mixed_label_flag}
  ```
- **Rationale:** This is the "MotionLatent Data Contract" — the input format for all Phase 3+ experiments

## Stage 10: Metadata Preservation

- **Input:** Window-level dataset
- **Transformation:** Attach and persist metadata for each window
- **Output:** HDF5 or NumPy archive with:
  - `X_windows`: sensor data
  - `y_labels`: activity labels
  - `subject_ids`: for leakage-safe splitting
  - `window_metadata`: start timestamps, mixed-label flags
- **Rationale:**
  - Subject IDs must be preserved for GroupKFold / LOSO evaluation
  - Window timestamps enable temporal analysis of errors
  - Mixed-label flags enable principled handling of transitions
- **Parameters:** Output format: HDF5 (recommended) or `.npz`
- **Risks:** Large file sizes; compress with gzip

---

## MotionLatent Data Contract

A single sample in the MotionLatent pipeline is defined as:

```
X_i  ∈  R^(T × C)

where:
  T = 256   (temporal samples per window = 2.56s at 100Hz)
  C = 27    (motion channels: 3 locations × 3 modalities × 3 axes)

Channels (ordered):
  hand_acc_16g_x, hand_acc_16g_y, hand_acc_16g_z,
  chest_acc_16g_x, chest_acc_16g_y, chest_acc_16g_z,
  ankle_acc_16g_x, ankle_acc_16g_y, ankle_acc_16g_z,
  hand_gyro_x, hand_gyro_y, hand_gyro_z,
  chest_gyro_x, chest_gyro_y, chest_gyro_z,
  ankle_gyro_x, ankle_gyro_y, ankle_gyro_z,
  hand_mag_x, hand_mag_y, hand_mag_z,
  chest_mag_x, chest_mag_y, chest_mag_z,
  ankle_mag_x, ankle_mag_y, ankle_mag_z

After dimensionality reduction:
  Z_i  ∈  R^d     where d << T × C = 6912
```

### Metadata per window:
- `subject_id`: int (101–109)
- `activity_id`: int (from PAMAP2 activity map)
- `window_start_timestamp`: float (seconds)
- `is_mixed_label`: bool (activity transition flag)

---

## Evaluation Strategy

### Leakage-Safe Splitting

**Primary:** Leave-One-Subject-Out (LOSO) cross-validation
- 9 folds, one per subject
- Guarantees no temporal or subject-level leakage
- Most conservative evaluation

**Alternative:** GroupKFold with `subject_id` as group
- Same guarantee, fewer folds if needed

**NEVER:** Random window splitting (overlapping windows from the same subject would leak across train/test)

### Activity-0 (Transient) Policy

- Activity 0 is retained in the dataset
- Excluded from primary activity-structure evaluation metrics
- Rationale: transitions have no consistent motion pattern; including them would artificially degrade evaluation metrics
