# MotionLatent Phase 2 -- Preprocessing & Window Generation Report


## 1. Raw Observations Processed

- Total raw observations: **2,872,533**
- After channel selection: **2,872,533**
- After missing-value handling: **2,870,616**
- Observations rejected (dropped): **1,917**

## 2. Missing-Value Handling

| Subject | Total Rows | NaN Before | NaN After Interp | Rows Dropped | Rows Retained | Long Gaps |
|---------|-----------|------------|------------------|--------------|---------------|-----------|
| S101 | 376,417 | 29,610 | 1,926 | 214 | 376,203 | 108 |
| S102 | 447,000 | 50,049 | 3,951 | 367 | 446,633 | 216 |
| S103 | 252,833 | 11,088 | 801 | 89 | 252,744 | 54 |
| S104 | 329,576 | 31,752 | 846 | 94 | 329,482 | 63 |
| S105 | 374,783 | 34,497 | 2,709 | 222 | 374,561 | 126 |
| S106 | 361,817 | 24,624 | 3,411 | 343 | 361,474 | 144 |
| S107 | 313,599 | 25,200 | 2,043 | 193 | 313,406 | 108 |
| S108 | 408,031 | 48,690 | 4,212 | 377 | 407,654 | 198 |
| S109 | 8,477 | 567 | 162 | 18 | 8,459 | 9 |

## 3. Window Generation Summary

- **Total windows:** 22,415
- **Pure windows:** 15,049 (single activity >= 80%)
- **Mixed windows:** 224 (no activity >= 80%)
- **Transient windows:** 7,142 (activity 0, pure)
- **Rejected windows:** 0 (NaN/Inf detected)
- **Window size:** 256 samples (2.56s at 100Hz)
- **Stride:** 128 samples (50% overlap)
- **Channels:** 27 (acc_16g + gyro + mag x 3 locations)

## 4. Windows per Subject

| Subject | Windows |
|---------|---------|
| S101 | 2,938 |
| S102 | 3,488 |
| S103 | 1,973 |
| S104 | 2,573 |
| S105 | 2,925 |
| S106 | 2,823 |
| S107 | 2,447 |
| S108 | 3,183 |
| S109 | 65 |

## 5. Windows per Activity (Pure Only)

| Activity ID | Activity Name | Windows |
|-------------|---------------|---------|
| 0 | transient | 7,142 |
| 1 | lying | 1,494 |
| 2 | sitting | 1,436 |
| 3 | standing | 1,471 |
| 4 | walking | 1,857 |
| 5 | running | 760 |
| 6 | cycling | 1,279 |
| 7 | Nordic_walking | 1,461 |
| 12 | ascending_stairs | 895 |
| 13 | descending_stairs | 803 |
| 16 | vacuum_cleaning | 1,360 |
| 17 | ironing | 1,854 |
| 24 | rope_jumping | 379 |

## 6. Subject-Activity Coverage

| Subject | Activities (pure windows) |
|---------|--------------------------|
| S101 | 0(transient), 1(lying), 2(sitting), 3(standing), 4(walking), 5(running), 6(cycling), 7(Nordic_walking), 12(ascending_stairs), 13(descending_stairs), 16(vacuum_cleaning), 17(ironing), 24(rope_jumping) |
| S102 | 0(transient), 1(lying), 2(sitting), 3(standing), 4(walking), 5(running), 6(cycling), 7(Nordic_walking), 12(ascending_stairs), 13(descending_stairs), 16(vacuum_cleaning), 17(ironing), 24(rope_jumping) |
| S103 | 0(transient), 1(lying), 2(sitting), 3(standing), 4(walking), 12(ascending_stairs), 13(descending_stairs), 16(vacuum_cleaning), 17(ironing) |
| S104 | 0(transient), 1(lying), 2(sitting), 3(standing), 4(walking), 6(cycling), 7(Nordic_walking), 12(ascending_stairs), 13(descending_stairs), 16(vacuum_cleaning), 17(ironing) |
| S105 | 0(transient), 1(lying), 2(sitting), 3(standing), 4(walking), 5(running), 6(cycling), 7(Nordic_walking), 12(ascending_stairs), 13(descending_stairs), 16(vacuum_cleaning), 17(ironing), 24(rope_jumping) |
| S106 | 0(transient), 1(lying), 2(sitting), 3(standing), 4(walking), 5(running), 6(cycling), 7(Nordic_walking), 12(ascending_stairs), 13(descending_stairs), 16(vacuum_cleaning), 17(ironing), 24(rope_jumping) |
| S107 | 0(transient), 1(lying), 2(sitting), 3(standing), 4(walking), 5(running), 6(cycling), 7(Nordic_walking), 12(ascending_stairs), 13(descending_stairs), 16(vacuum_cleaning), 17(ironing) |
| S108 | 0(transient), 1(lying), 2(sitting), 3(standing), 4(walking), 5(running), 6(cycling), 7(Nordic_walking), 12(ascending_stairs), 13(descending_stairs), 16(vacuum_cleaning), 17(ironing), 24(rope_jumping) |
| S109 | 0(transient), 24(rope_jumping) |

## 7. Final Tensor Shape

- Shape: **[22415, 256, 27]** = (windows, 256, 27)
- Flattened raw baseline: **6,912** dimensions per window

## 8. Normalization

- Method: **per_channel_StandardScaler_LOSO**
- Per-channel StandardScaler (zero mean, unit variance)
- Fitted on training subjects ONLY for each LOSO fold
- Test subject statistics NEVER used during fitting

### LOSO Scaler Summary

| Test Subject | Training Subjects | Training Windows |
|-------------|-------------------|------------------|
| S101 | S102, S103, S104, S105, S106, S107, S108, S109 | 19,477 |
| S102 | S101, S103, S104, S105, S106, S107, S108, S109 | 18,927 |
| S103 | S101, S102, S104, S105, S106, S107, S108, S109 | 20,442 |
| S104 | S101, S102, S103, S105, S106, S107, S108, S109 | 19,842 |
| S105 | S101, S102, S103, S104, S106, S107, S108, S109 | 19,490 |
| S106 | S101, S102, S103, S104, S105, S107, S108, S109 | 19,592 |
| S107 | S101, S102, S103, S104, S105, S106, S108, S109 | 19,968 |
| S108 | S101, S102, S103, S104, S105, S106, S107, S109 | 19,232 |
| S109 | S101, S102, S103, S104, S105, S106, S107, S108 | 22,350 |

## 9. LOSO Split Design

- Strategy: **Leave-One-Subject-Out (LOSO)**
- Subjects: [101, 102, 103, 104, 105, 106, 107, 108, 109]
- Number of folds: 9
- Each fold: 1 subject as test, remaining as training
- **No random window splitting** -- prevents leakage from overlapping windows

## 10. Data Quality Warnings

- No warnings.

## 11. Verification Checks

- [x] Every window has exactly 256 x 27 values
- [x] No NaN/Inf in accepted windows
- [x] Subject IDs preserved in metadata
- [x] Timestamps remain ordered within subjects
- [x] No window crosses unresolved long gaps
- [x] No subject leakage in LOSO train/test sets
- [x] Scaling statistics fitted only on training subjects
