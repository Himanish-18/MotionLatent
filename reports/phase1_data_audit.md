# MotionLatent Phase 1 -- PAMAP2 Data Audit Report

**Date:** 2026-09-10
**Author:** MotionLatent ML Engineering Team
**Status:** Complete

---

## 1. Executive Summary

This report documents a complete programmatic audit of the PAMAP2 Physical Activity Monitoring dataset in preparation for the MotionLatent representation-learning project. The audit establishes the formal data and preprocessing specification for subsequent phases.

**Key findings:**
- **2,872,533 observations** across 9 subjects, 13 activities, at a verified 100 Hz sampling rate
- **27 primary motion channels** selected (accelerometer 16g + gyroscope + magnetometer x 3 locations)
- **Heart rate is 90.87% missing** -- confirmed exclusion from primary representation
- **Orientation columns contain data** but are documented as "not validated" -- excluded from pipeline
- **Subject 109 is severely limited** (1.4 min, 2 activities) -- requires special handling
- **Activity 0 (transient) = 32.4%** of data -- must be excluded from activity-structure evaluation
- **Zero duplicate rows, zero duplicate timestamps, perfect monotonic timestamps** across all subjects
- **Exact 100 Hz sampling** confirmed empirically for all subjects with zero gaps
- **Missing IMU values are low**: hand 0.46%, chest 0.12%, ankle 0.41% -- linear interpolation feasible

---

## 2. Project Context

MotionLatent investigates how much motion-sensor data can be compressed into a compact representation while preserving meaningful activity structure, neighbourhood structure, and acceptable reconstruction quality.

The project compares: Raw baseline, PCA, ICA, and Nonlinear Autoencoder representations.

This Phase 1 report covers dataset audit only -- no dimensionality reduction or classification is performed.

---

## 3. Dataset Overview

| Property | Value |
|----------|-------|
| Dataset | PAMAP2 Physical Activity Monitoring |
| Source | UCI Machine Learning Repository |
| Total columns per file | 54 (no header) |
| Protocol files | 9 (.dat) |
| Optional files | 5 (.dat) |
| Documentation files | 5 (PDFs) |
| Total protocol observations | 2,872,533 |
| Total optional observations | 977,972 |
| Sampling frequency | 100 Hz (verified) |
| Total recording duration | ~7.97 hours (protocol) |

---

## 4. Dataset File Structure

### Protocol Files

| File | Subject | Size (MB) | Rows | Duration (min) |
|------|---------|-----------|------|----------------|
| subject101.dat | 101 | 135.13 | 376,417 | 62.7 |
| subject102.dat | 102 | 197.72 | 447,000 | 74.5 |
| subject103.dat | 103 | 112.43 | 252,833 | 42.1 |
| subject104.dat | 104 | 146.14 | 329,576 | 54.9 |
| subject105.dat | 105 | 165.75 | 374,783 | 62.5 |
| subject106.dat | 106 | 160.49 | 361,817 | 60.3 |
| subject107.dat | 107 | 139.07 | 313,599 | 52.3 |
| subject108.dat | 108 | 181.24 | 408,031 | 68.0 |
| subject109.dat | 109 | 3.71 | 8,477 | 1.4 |

### Optional Files

| File | Subject | Size (MB) |
|------|---------|-----------|
| subject101.dat | 101 | 115.65 |
| subject105.dat | 105 | 69.15 |
| subject106.dat | 106 | 58.09 |
| subject108.dat | 108 | 81.46 |
| subject109.dat | 109 | 86.40 |

---

## 5. Subject Analysis

| Subject | Observations | Activities | Duration (min) | Notes |
|---------|-------------|------------|----------------|-------|
| S101 | 376,417 | 13 | 62.7 | Complete |
| S102 | 447,000 | 13 | 74.5 | Complete, longest recording |
| S103 | 252,833 | 9 | 42.1 | Missing: running, cycling, Nordic walking, rope jumping |
| S104 | 329,576 | 12 | 54.9 | Missing: rope jumping |
| S105 | 374,783 | 13 | 62.5 | Complete |
| S106 | 361,817 | 13 | 60.3 | Complete |
| S107 | 313,599 | 12 | 52.3 | Missing: rope jumping |
| S108 | 408,031 | 13 | 68.0 | Complete |
| S109 | 8,477 | 2 | 1.4 | **SEVERELY LIMITED** -- only transient + rope jumping |

### Subject Imbalance Assessment

- **Subject 109** is an extreme outlier: 0.3% of total data, only 2 activities (transient + rope_jumping), 1.4 minutes of recording
- **Subject 103** is missing 4 activities
- **Subjects 104, 107** are each missing rope_jumping only
- **5 of 9 subjects** have complete activity coverage
- Recording durations range from 1.4 to 74.5 minutes (53x variation)

**Recommendation:** Subject 109 should be flagged for special handling in LOSO evaluation. It cannot meaningfully represent most activities. Consider using it only as a supplementary test case.

---

## 6. Activity Analysis

### Activity Distribution (Protocol Data)

| ID | Activity | Count | Percentage |
|----|----------|-------|------------|
| 0 | transient | 929,661 | 32.4% |
| 1 | lying | 192,523 | 6.7% |
| 2 | sitting | 185,188 | 6.4% |
| 3 | standing | 189,931 | 6.6% |
| 4 | walking | 238,761 | 8.3% |
| 5 | running | 98,199 | 3.4% |
| 6 | cycling | 164,600 | 5.7% |
| 7 | Nordic_walking | 188,107 | 6.5% |
| 12 | ascending_stairs | 117,216 | 4.1% |
| 13 | descending_stairs | 104,944 | 3.7% |
| 16 | vacuum_cleaning | 175,353 | 6.1% |
| 17 | ironing | 238,690 | 8.3% |
| 24 | rope_jumping | 49,360 | 1.7% |

### Activity Imbalance

- **Activity 0 (transient)** dominates at 32.4% -- this is transition/unlabeled data
- **Rope jumping** is the rarest labeled activity at 1.7% (49,360 observations)
- Excluding transient, the labeled activities range from 1.7% to 8.3%
- The imbalance is moderate but should be considered in evaluation

### Missing Activities Per Subject

| Subject | Missing Activities |
|---------|--------------------|
| S103 | running (5), cycling (6), Nordic walking (7), rope jumping (24) |
| S104 | rope jumping (24) |
| S107 | rope jumping (24) |
| S109 | lying (1), sitting (2), standing (3), walking (4), running (5), cycling (6), Nordic walking (7), ascending stairs (12), descending stairs (13), vacuum cleaning (16), ironing (17) |

---

## 7. Sensor Schema

### Complete PAMAP2 Column Layout (54 columns)

| Index | Column | Category |
|-------|--------|----------|
| 0 | timestamp | Metadata |
| 1 | activity_id | Metadata |
| 2 | heart_rate | Physiological |
| 3-6 | hand: temperature, acc_16g (x,y,z) | IMU - Hand |
| 7-9 | hand: acc_6g (x,y,z) | IMU - Hand |
| 10-12 | hand: gyro (x,y,z) | IMU - Hand |
| 13-15 | hand: mag (x,y,z) | IMU - Hand |
| 16-19 | hand: orientation (0,1,2,3) | IMU - Hand |
| 20-23 | chest: temperature, acc_16g (x,y,z) | IMU - Chest |
| 24-26 | chest: acc_6g (x,y,z) | IMU - Chest |
| 27-29 | chest: gyro (x,y,z) | IMU - Chest |
| 30-32 | chest: mag (x,y,z) | IMU - Chest |
| 33-36 | chest: orientation (0,1,2,3) | IMU - Chest |
| 37-40 | ankle: temperature, acc_16g (x,y,z) | IMU - Ankle |
| 41-43 | ankle: acc_6g (x,y,z) | IMU - Ankle |
| 44-46 | ankle: gyro (x,y,z) | IMU - Ankle |
| 47-49 | ankle: mag (x,y,z) | IMU - Ankle |
| 50-53 | ankle: orientation (0,1,2,3) | IMU - Ankle |

### Primary Motion Representation (27 channels)

Selected for MotionLatent:

| Modality | Channels | Locations | Total |
|----------|----------|-----------|-------|
| Accelerometer (16g) | x, y, z | hand, chest, ankle | 9 |
| Gyroscope | x, y, z | hand, chest, ankle | 9 |
| Magnetometer | x, y, z | hand, chest, ankle | 9 |
| **Total** | | | **27** |

### Excluded Columns and Rationale

| Column(s) | Reason | Revisit? |
|-----------|--------|----------|
| heart_rate | Physiological, not motion. 90.87% missing. | No |
| {loc}_temperature (3) | Environmental/skin, not motion. | No |
| {loc}_acc_6g_{xyz} (9) | Redundant with 16g, narrower range. | Phase 3 comparison |
| {loc}_orientation_{0-3} (12) | PAMAP2 documentation: "not validated". Data present but reliability unverified. | Only if independently validated |

### Orientation Column Finding

The orientation quaternion columns **do contain non-NaN values** (contrary to some references claiming they are all NaN). The values are in [-1, 1] range consistent with unit quaternions. However, the PAMAP2 readme states the orientation output was "not validated." Given this, the columns are excluded from the primary pipeline as their accuracy cannot be guaranteed. They may be revisited in a later experiment if independently validated.

---

## 8. Missing Data Analysis

### Per-Sensor-Location Missing Rates

| Sensor Location | Missing Count | Missing % |
|-----------------|--------------|-----------|
| Heart rate | 2,610,265 | 90.87% |
| Hand IMU | 13,141 | 0.46% |
| Chest IMU | 3,563 | 0.12% |
| Ankle IMU | 11,749 | 0.41% |

**Notes:**
- All channels at a given location share the same missing pattern (sensor-level dropout)
- Heart rate missingness is extreme because the HR sensor operates at ~9 Hz and was interpolated to 100 Hz, with many failures
- IMU missing rates are low (<0.5%) and amenable to linear interpolation

### Recommendation

- **Heart rate:** Excluded entirely (90.87% missing + not motion)
- **IMU channels:** Linear interpolation for gaps <= 10 consecutive samples (100ms), forward-fill then drop for longer gaps
- Expected data loss from dropping: negligible (<0.5%)

---

## 9. Timestamp/Sampling Analysis

### Empirical Verification

| Property | Value (all subjects) |
|----------|---------------------|
| Nominal frequency | 100 Hz |
| Empirical frequency | **100.0 Hz** (verified) |
| Mean dt | 0.01 s |
| Median dt | 0.01 s |
| Std dt | 0.0 s |
| Min dt | 0.01 s |
| Max dt | 0.01 s |
| Gaps (>2x median) | **0** across all subjects |
| Duplicate timestamps | **0** across all subjects |
| Monotonic | **Yes** for all subjects |

### Assessment

The PAMAP2 timestamps are perfectly regular at exactly 100 Hz with zero jitter, zero gaps, and zero duplicates. **No resampling is required.** This is exceptionally clean temporal data.

---

## 10. Sensor Quality Analysis

### Sensor Value Ranges

| Modality | Channels | Min | Max | Mean | Std | Range |
|----------|----------|-----|-----|------|-----|-------|
| Accelerometer 16g | 9 | -158.93 | 160.52 | 1.79 | 6.75 | 319.44 |
| Gyroscope | 9 | -28.14 | 26.42 | -0.00 | 1.05 | 54.55 |
| Magnetometer | 9 | -497.63 | 183.91 | -5.76 | 27.86 | 681.54 |
| Temperature | 3 | 24.75 | 38.56 | 34.16 | 2.14 | 13.81 |

### Scale Differences (Critical for Normalization)

The three primary motion modalities operate at vastly different scales:
- Accelerometer range: ~320
- Gyroscope range: ~55
- Magnetometer range: ~682

**This confirms that per-channel normalization is mandatory before PCA/ICA/Autoencoder.**

### Data Quality Findings

- **No infinities** detected in any channel
- **No constant channels** detected
- **No near-zero variance channels** (all have meaningful variation)
- **Zero duplicate rows** across the entire dataset
- **Zero duplicate timestamps** per subject

---

## 11. Invalid/Excluded Features

| Feature | Status | Rationale |
|---------|--------|-----------|
| Orientation (12 cols) | **EXCLUDED** | PAMAP2 documentation: "not validated". Contains data but accuracy unverified. |
| Heart rate (1 col) | **EXCLUDED** | Physiological signal, 90.87% missing, not motion-relevant |
| Temperature (3 cols) | **EXCLUDED** | Environmental/skin temperature, not motion |
| Accelerometer 6g (9 cols) | **EXCLUDED (provisional)** | Redundant with 16g. May compare in Phase 3. |

---

## 12. Activity-0 / Transient Activity Analysis

### Quantification

- **929,661 observations** labeled as activity 0 (transient)
- **32.4%** of all protocol data
- Present in **all 9 subjects**
- Represents transitions between structured activities

### Where It Occurs

Activity 0 appears between all activity segments, representing the time when subjects transition from one activity to another. It does not represent a single consistent motion pattern.

### Decision

- **Retained in dataset** for transparency and reproducibility
- **Excluded from primary activity-structure evaluation** metrics
- **Rationale:** Transient periods contain mixed, inconsistent motion patterns. Including them in activity-structure metrics (e.g., cluster purity, neighbourhood preservation) would artificially degrade results and not reflect the system's ability to distinguish actual activities.

---

## 13. Windowing Analysis

### Candidate Configurations

| Window (s) | Overlap | Samples | Step | Est. Windows | Feature Dim |
|------------|---------|---------|------|--------------|-------------|
| 1.0 | 0% | 100 | 100 | 28,725 | 2,700 |
| 1.0 | 25% | 100 | 75 | 38,300 | 2,700 |
| 1.0 | 50% | 100 | 50 | 57,449 | 2,700 |
| 2.0 | 0% | 200 | 200 | 14,362 | 5,400 |
| 2.0 | 25% | 200 | 150 | 19,149 | 5,400 |
| 2.0 | 50% | 200 | 100 | 28,724 | 5,400 |
| 3.0 | 0% | 300 | 300 | 9,575 | 8,100 |
| 3.0 | 50% | 300 | 150 | 19,149 | 8,100 |
| 5.0 | 0% | 500 | 500 | 5,745 | 13,500 |
| 5.0 | 50% | 500 | 250 | 11,489 | 13,500 |

### Recommendation: 2.56s window, 50% overlap (256 samples)

**Rationale:**
1. **Temporal context:** 2.56s captures 2-3 full gait cycles for walking/running, sufficient for most activities
2. **Power-of-2:** 256 samples is FFT-friendly for potential frequency-domain features
3. **Sample count:** ~28,000 windows at 50% overlap provides adequate data for ML
4. **Feature dimension:** 256 x 27 = 6,912 -- large enough for meaningful compression, not so large as to be computationally prohibitive
5. **Activity transitions:** 50% overlap ensures transition windows can be identified and flagged
6. **Edge deployment:** 256 samples at 100Hz = 2.56s latency, acceptable for real-time applications
7. **Literature alignment:** 2-3s windows are standard in HAR literature

---

## 14. Normalization Analysis

### Scale Comparison

| Modality | Std Dev | Range |
|----------|---------|-------|
| Accelerometer 16g | 6.75 | 319.4 |
| Gyroscope | 1.05 | 54.6 |
| Magnetometer | 27.86 | 681.5 |

The scale differences are dramatic (6.4x between gyro std and acc std; 26.5x between gyro std and mag std).

### Recommendation: Per-Channel StandardScaler

- **Method:** StandardScaler (zero mean, unit variance per channel)
- **Scope:** Per-channel (each of the 27 channels independently)
- **Fit on:** Training set only (to prevent data leakage)

**Rationale:**
- StandardScaler is preferred over MinMaxScaler because:
  - More robust to the occasional extreme values in accelerometer data
  - Preserves the distributional shape better than range-based scaling
  - Standard in PCA/ICA preprocessing (both methods assume centered data)
- RobustScaler is a viable alternative if extreme outliers prove problematic
- Per-channel (not global) normalization is required because each axis at each location has its own characteristic distribution

---

## 15. Data Leakage Risks

### Critical Risk: Overlapping Windows

If windows are generated with overlap (e.g., 50%), adjacent windows from the same subject share temporal samples. Random splitting of windows across train/test sets would cause:
- **Temporal leakage:** Near-identical windows from the same recording appearing in both train and test
- **Subject leakage:** The model learning subject-specific patterns rather than generalizable activity patterns

### Recommended Strategy: Leave-One-Subject-Out (LOSO)

- **Primary:** LOSO cross-validation (9 folds, one per subject)
- **Alternative:** GroupKFold with `subject_id` as group
- All windows from one subject are held out entirely
- No temporal overlap between train and test sets
- Most conservative evaluation strategy

**Why not random splitting:** With 50% overlap windows, approximately 50% of each window's data also appears in adjacent windows. Random splitting would place highly correlated windows in both train and test, inflating performance estimates by 10-30% (based on HAR literature).

### Subject 109 Consideration

With only 2 activities and 1.4 minutes of data, Subject 109 as a LOSO test fold will produce near-zero performance for most activities. This fold should be reported separately and not averaged into overall metrics without caveats.

---

## 16. Recommended Preprocessing Pipeline

```
Raw PAMAP2 .dat files
    |
    v
[1] Dataset Discovery (pathlib, relative paths)
    |
    v
[2] Schema Validation (verify 54 columns)
    |
    v
[3] Column Selection (keep 27 motion channels + metadata)
    |
    v
[4] Invalid-Field Removal (drop orientation, heart rate, temperature, acc_6g)
    |
    v
[5] Missing-Value Handling (linear interpolation <= 10 samples, ffill, drop)
    |
    v
[6] Normalization (per-channel StandardScaler, fit on train only)
    |
    v
[7] Temporal Windowing (256 samples, 50% overlap)
    |
    v
[8] Window-Level Dataset (X: N x 256 x 27, y: activity labels, subject IDs)
```

---

## 17. MotionLatent Data Contract

### Input Sample

```
X_i  in  R^(T x C)

where:
  T = 256   (temporal samples per window = 2.56s at 100Hz)
  C = 27    (motion channels: 3 locations x 3 modalities x 3 axes)
```

### Compact Representation

```
Z_i  in  R^d     where d << T x C = 6,912
```

### Metadata per Window

| Field | Type | Description |
|-------|------|-------------|
| subject_id | int | 101-109 |
| activity_id | int | PAMAP2 activity map |
| window_start_timestamp | float | Seconds from recording start |
| is_mixed_label | bool | True if activity transition within window |

### Channel Order

```
hand_acc_16g_x, hand_acc_16g_y, hand_acc_16g_z,
chest_acc_16g_x, chest_acc_16g_y, chest_acc_16g_z,
ankle_acc_16g_x, ankle_acc_16g_y, ankle_acc_16g_z,
hand_gyro_x, hand_gyro_y, hand_gyro_z,
chest_gyro_x, chest_gyro_y, chest_gyro_z,
ankle_gyro_x, ankle_gyro_y, ankle_gyro_z,
hand_mag_x, hand_mag_y, hand_mag_z,
chest_mag_x, chest_mag_y, chest_mag_z,
ankle_mag_x, ankle_mag_y, ankle_mag_z
```

---

## 18. Decisions Made

| # | Decision | Rationale | Source |
|---|----------|-----------|--------|
| D1 | Use accelerometer 16g (not 6g) | Wider dynamic range for vigorous activities | PAMAP2 readme: 16g and 6g available |
| D2 | Exclude heart rate | 90.87% missing, physiological not motion | Measured: 2,610,265 / 2,872,533 NaN |
| D3 | Exclude orientation | PAMAP2 documentation: "not validated" | PAMAP2 readme |
| D4 | Exclude temperature | Environmental, not motion-relevant | Domain knowledge |
| D5 | Exclude activity 0 from evaluation | Inconsistent transition patterns | 32.4% of data, no consistent motion |
| D6 | Use LOSO cross-validation | Prevent subject and temporal leakage | Standard HAR methodology |
| D7 | 2.56s window, 50% overlap | Balance context, sample count, FFT compatibility | Windowing analysis |
| D8 | Per-channel StandardScaler | Scale differences: 6-27x between modalities | Measured sensor ranges |
| D9 | Linear interpolation for missing values | IMU missing rate < 0.5% | Measured missing rates |
| D10 | 100 Hz, no resampling needed | Empirically verified exact 100 Hz | Sampling analysis |

---

## 19. Open Questions

| # | Question | Impact | Status |
|---|----------|--------|--------|
| Q1 | Should Subject 109 be excluded from LOSO? | Affects evaluation fairness | Recommend: report separately |
| Q2 | Is a low-pass filter needed? | Affects signal quality | PROVISIONAL: no filter by default |
| Q3 | Should acc_6g be added for comparison? | Affects channel count | Deferred to Phase 3 |
| Q4 | Can orientation be independently validated? | 12 additional channels | Deferred: requires external validation |
| Q5 | Optimal window for different activity types? | May vary by activity | Deferred to Phase 2 experiments |
| Q6 | RobustScaler vs StandardScaler? | Outlier sensitivity | Evaluate in Phase 2 |

---

## 20. Phase 2 Readiness Checklist

- [x] PAMAP2 dataset located and audited
- [x] Dataset schema documented
- [x] All relevant sensors identified
- [x] Invalid/problematic fields identified
- [x] Activity distribution documented
- [x] Subject/session structure documented
- [x] Missing values quantified
- [x] Timestamp/sampling behaviour analyzed
- [x] Data-quality issues documented
- [x] Activity-0/transient handling documented
- [x] Windowing strategy evaluated
- [x] Normalization strategy recommended
- [x] Subject-aware leakage-safe evaluation strategy defined
- [x] Future MotionLatent data contract defined
- [x] Phase 2 preprocessing specification documented
- [x] Reproducible audit code exists
- [x] Tests pass (18/18)
- [x] Raw dataset excluded from Git
- [x] Documentation complete

**Phase 1 is COMPLETE. Ready for Phase 2.**
