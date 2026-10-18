# MotionLatent — Phase 3.3: Raw Baseline Pooled-Metric Reconstruction Report

**Author:** MotionLatent Core ML Research & Independent Verification  
**Date:** October 2026  
**Status:** Validated & Complete  
**Evaluation Verdict:** **PASS**  
**Repository:** [https://github.com/Himanish-18/MotionLatent](https://github.com/Himanish-18/MotionLatent)  
**Reconstructed Prediction Artifact:** [`reports/raw_baseline_predictions.json`](file:///c:/Users/himan/OneDrive/Documents/MotionLatent/reports/raw_baseline_predictions.json)  

---

## 1. Executive Summary & Verdict

### Final Verdict: **PASS**

Phase 3.3 resolves the final empirical evaluation gap identified in Phase 3.2 by rigorously reconstructing the held-out test predictions for the uncompressed, 6,912-dimensional raw-input $k$-NN baseline across all nine Leave-One-Subject-Out (LOSO) folds (subjects 101–109). 

In Phase 3.2, all representation models (PCA, FastICA, Autoencoder) were audited, reconciled, and frozen. However, while fold-level aggregate metrics for the raw baseline were preserved in `reports/phase3_loso_results.json`, per-window prediction vectors ($\hat{y}$) had not been serialized to disk, leaving True Pooled classification metrics for the raw baseline recorded as `N/A (unserialized)`.

In Phase 3.3:
1. **Zero Data Leakage & Exact Protocol Fidelity**: Using the exact Phase 2.1 preprocessed data, authoritative window metadata, and fold-specific `StandardScaler` instances fitted strictly on training subjects, we reconstructed held-out predictions for all 14,973 pure non-transient evaluation windows under the original $k=5$ Euclidean $k$-NN classifier.
2. **Deterministic Metric Fidelity**: Fold-level balanced accuracy, accuracy, and macro-F1 reproduced the historical Phase 3 results with **zero discrepancy** across all nine folds.
3. **True Pooled Metric Resolution**: With all 14,973 held-out predictions serialized, we calculated true pooled metrics across the global evaluation population:
   - **True Pooled Balanced Accuracy (9-sub)**: **0.5322** (53.22%)
   - **True Pooled Balanced Accuracy (8-sub, excl. S109)**: **0.5322** (53.22%)
   - **True Pooled Accuracy (9-sub)**: **0.6022** (60.22%)
   - **True Pooled Accuracy (8-sub, excl. S109)**: **0.6042** (60.42%)
   - **True Pooled Macro-F1 (9-sub)**: **0.5157** (51.57%)
   - **True Pooled Macro-F1 (8-sub, excl. S109)**: **0.5181** (51.81%)
4. **Closing the Evaluation Record**: The benchmark comparison between the compact latent representations (PCA-32 at 69.4% pooled balanced accuracy, Autoencoder-16 at 70.9% weighted score) and the raw high-dimensional baseline (53.2% pooled balanced accuracy) is now mathematically complete and permanently reproducible.

---

## 2. Baseline Configuration & Artifact Traceability

The raw baseline evaluates the downstream activity clustering directly in the standardized input signal space without learned latent projection.

### 2.1 Input Data & Channel Contract
- **Dataset**: PAMAP2 Physical Activity Monitoring Benchmark.
- **Sensor Modalities**: 27 primary channels across 3 IMU placements (Hand, Chest, Ankle).
  * 16g Triaxial Accelerometer ($X, Y, Z$)
  * Triaxial Gyroscope ($X, Y, Z$)
  * Triaxial Magnetometer ($X, Y, Z$)
- **Exclusions**: 6g accelerometer, temperature, orientation, and heart rate are strictly excluded as established in Phase 1 and validated in Phase 2.1.
- **Window Shape**: $256 \text{ samples} \times 27 \text{ channels} = 6,912 \text{ flattened features}$ (2.56 seconds at 100 Hz, stride 128 = 50% overlap).
- **Channel Ordering**: Validated invariant from `src/data/preprocessing.py` and `src/data/loso.py`:
  `[hand_16g_acc_xyz, hand_gyro_xyz, hand_mag_xyz, chest_16g_acc_xyz, chest_gyro_xyz, chest_mag_xyz, ankle_16g_acc_xyz, ankle_gyro_xyz, ankle_mag_xyz]`.

### 2.2 Normalization & Fold Isolation
- **Scaler**: `sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)`.
- **Leakage Prevention**: For each fold $k \in \{101, \dots, 109\}$, the scaler is fitted strictly on the $N_{\text{train}}^{(k)}$ training subject windows ($\mu_{\text{train}}, \sigma_{\text{train}}$).
- Zero held-out test windows, labels, or test-subject statistics are exposed to the scaler or classifier during fitting.

### 2.3 Downstream Classifier Specification
- **Algorithm**: $k$-Nearest Neighbors (`sklearn.neighbors.KNeighborsClassifier`).
- **Hyperparameters**:
  * $k = 5$ neighbors
  * Metric: Euclidean ($L_2$ norm)
  * Neighbor weighting: Uniform (`weights='uniform'`)
  * Algorithm: Auto (`algorithm='auto'`)
- **Training Population**: Scaled training-fold windows $X_{\text{train}} \in \mathbb{R}^{N_{\text{train}} \times 6912}$ with training labels $y_{\text{train}}$.
- **Inference**: Predict $\hat{y}_{\text{test}} = \operatorname{kNN}(X_{\text{test}} \in \mathbb{R}^{N_{\text{test}} \times 6912})$.

---

## 3. Evaluation Population Invariants & Coverage

The primary evaluation population is strictly restricted to **pure non-transient windows**:
- Purity criterion: `activity_purity >= 0.80` (`is_mixed_label == False`).
- Transient criterion: `dominant_activity != 0` (`is_transient == False`).

### 3.1 Per-Subject Window Allocation
Across all nine LOSO folds, the 14,973 eligible windows are partitioned with exactly-once held-out coverage:

| Test Subject | Held-Out Test Windows ($N_{\text{test}}$) | Training Windows ($N_{\text{train}}$) | Total Evaluation Windows |
|---|---|---|---|
| **S101** | 1,927 | 13,046 | 14,973 |
| **S102** | 2,029 | 12,944 | 14,973 |
| **S103** | 1,342 | 13,631 | 14,973 |
| **S104** | 1,787 | 13,186 | 14,973 |
| **S105** | 2,098 | 12,875 | 14,973 |
| **S106** | 1,927 | 13,046 | 14,973 |
| **S107** | 1,794 | 13,179 | 14,973 |
| **S108** | 2,020 | 12,953 | 14,973 |
| **S109** | 49 | 14,924 | 14,973 |
| **Total** | **14,973** | — | **14,973** |

### 3.2 Coverage & Uniqueness Invariants
- Total held-out prediction records generated: **14,973**.
- Uniqueness: Every `(subject_id, window_idx)` pair in the eligible population appears as a test instance in exactly one fold.
- Zero duplicate evaluations, zero dropped evaluation windows.

---

## 4. Reconstructed Per-Subject Results & Fidelity Check

We evaluated the reconstructed predictions against the saved Phase 3 fold results stored in `reports/phase3_loso_results.json`.

| Subject | Test Windows | Accuracy (Recon) | Accuracy (Saved) | Bal. Acc (Recon) | Bal. Acc (Saved) | Macro-F1 (Recon) | Macro-F1 (Saved) | Delta / Match |
|---|---|---|---|---|---|---|---|---|
| **S101** | 1,927 | 0.5044 | 0.5044 | 0.4694 | 0.4694 | 0.4170 | 0.4170 | **Exact Match** |
| **S102** | 2,029 | 0.6481 | 0.6481 | 0.6017 | 0.6017 | 0.5797 | 0.5797 | **Exact Match** |
| **S103** | 1,342 | 0.7377 | 0.7377 | 0.6652 | 0.6652 | 0.4344 | 0.4344 | **Exact Match** |
| **S104** | 1,787 | 0.7140 | 0.7140 | 0.6834 | 0.6834 | 0.5421 | 0.5421 | **Exact Match** |
| **S105** | 2,098 | 0.6020 | 0.6020 | 0.5367 | 0.5367 | 0.5069 | 0.5069 | **Exact Match** |
| **S106** | 1,927 | 0.6969 | 0.6969 | 0.6765 | 0.6765 | 0.5940 | 0.5940 | **Exact Match** |
| **S107** | 1,794 | 0.7007 | 0.7007 | 0.6506 | 0.6506 | 0.5658 | 0.5658 | **Exact Match** |
| **S108** | 2,020 | 0.2970 | 0.2970 | 0.2691 | 0.2691 | 0.2254 | 0.2254 | **Exact Match** |
| **S109** | 49 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **Exact Match** |

*Finding*: All per-subject metrics match the historical records down to four decimal places. The reconstruction is 100% faithful to the original Phase 3 experimental setup.

---

## 5. Metric Aggregations: Definitions & Empirical Results

Phase 3.2 established clear mathematical boundaries between three distinct aggregation paradigms. Here we report all three for the reconstructed raw baseline:

### 5.1 Definitions

1. **LOSO Macro-Mean Across Folds**:
   $$\bar{M} = \frac{1}{K} \sum_{k=1}^K M_k, \quad \sigma_M = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (M_k - \bar{M})^2}$$
   Treats each subject as an independent, equally weighted domain.

2. **Window-Count-Weighted Fold-Score Mean**:
   $$\bar{M}_w = \frac{\sum_{k=1}^K N_k M_k}{\sum_{k=1}^K N_k}$$
   Weights each fold's metric by that subject's window count.

3. **True Pooled Metric**:
   $$M_{\text{pooled}} = f\left(\bigoplus_{k=1}^K y^{(k)}, \bigoplus_{k=1}^K \hat{y}^{(k)}\right)$$
   Concatenates all 14,973 test predictions and ground-truth labels across all folds, evaluating the metric once across the entire combined evaluation population.

### 5.2 Comparative Metric Table

| Metric Aggregation Method | Balanced Accuracy | Accuracy | Macro-F1 |
|---|---|---|---|
| **9-Subject LOSO Macro-Mean** | $0.5058 \pm 0.2187$ | $0.5445 \pm 0.2326$ | $0.4295 \pm 0.1865$ |
| **8-Subject Macro-Mean (excl. S109)** | $0.5691 \pm 0.1335$ | $0.6126 \pm 0.1418$ | $0.4832 \pm 0.1264$ |
| **9-Subject Weighted Fold-Score Mean** | **0.5597** | **0.6021** | **0.4815** |
| **8-Subject Weighted Fold-Score Mean (excl. S109)** | **0.5615** | **0.6041** | **0.4831** |
| **9-Subject True Pooled Metric** | **0.5322** | **0.6022** | **0.5157** |
| **8-Subject True Pooled Metric (excl. S109)** | **0.5322** | **0.6042** | **0.5181** |

### 5.3 Mathematical Explanation: True Pooled (0.5322) vs. Weighted Fold-Score Mean (0.5597)

A crucial question in metric reconciliation is why True Pooled Balanced Accuracy (0.5322) is lower than the Weighted Fold-Score Mean (0.5597):
1. **Per-Class Recall in Pooled Space**: Balanced accuracy gives equal $1/12 \approx 8.33\%$ weight to each of the 12 active activity classes.
2. **Behavior on Activity 24 (Rope Jumping)**:
   - In raw 6,912-dimensional Euclidean space, rope jumping exhibits high high-frequency acceleration variance that overlaps heavily with running and cycling.
   - Out of 373 total pure rope jumping windows in the dataset (49 in S109, 324 in other subjects), raw $k$-NN achieves only **1 correct prediction** (recall = $1 / 373 \approx 0.0027$).
3. **Weighting Discrepancy**:
   - In **Weighted Fold-Score Mean**, S109 (where rope jumping recall is 0.0) contributes only $49 / 14,973 \approx 0.33\%$ of the weight. The remaining 324 rope jumping windows in other subjects are averaged inside multi-class fold balanced accuracies where 11 other well-performing classes mask rope jumping's near-zero recall.
   - In **True Pooled Balanced Accuracy**, rope jumping is treated as 1 of 12 equal classes ($8.33\%$ weight) with recall $0.0027$, directly depressing global balanced accuracy to $0.5322$.
4. **Accuracy Equivalence**: In contrast to balanced accuracy, standard accuracy satisfies the property that window-weighted fold accuracy ($0.6021$) identically equals true pooled accuracy ($0.6022$) within rounding precision, because standard accuracy weights every individual window equally.

---

## 6. Updated Representation Benchmark Comparison

With the raw baseline true pooled metrics established, we present the complete, unified benchmark table comparing raw inputs against compact representations:

| Representation Paradigm | Latent Dim ($d$) | Compression Ratio | 9-Sub Macro Bal. Acc | 8-Sub Macro Bal. Acc | 9-Sub Weighted Fold Bal. Acc | True Pooled Bal. Acc | True Pooled Accuracy | True Pooled Macro-F1 | Status |
|---|---|---|---|---|---|---|---|---|---|
| **Raw Baseline** | **6,912** | **$1.0\times$** | **0.5058** | **0.5691** | **0.5597** | **0.5322** | **0.6022** | **0.5157** | **Reconstructed (Phase 3.3)** |
| **PCA** | 16 | $432.0\times$ | 0.6272 | 0.7030 | 0.6908 | 0.6936 | 0.7224 | 0.6860 | Validated (Phase 3.2) |
| **PCA** | **32** | **$216.0\times$** | **0.7096** | **0.7065** | **0.6979** | **0.6936** | **0.7270** | **0.6980** | **Strict MVD Winner** |
| **FastICA** | 16 | $432.0\times$ | 0.6933 | 0.7104 | 0.6997 | *N/A* | *N/A* | *N/A* | Validated (Phase 3.2) |
| **FastICA** | 32 | $216.0\times$ | 0.6933 | 0.6800 | 0.6698 | *N/A* | *N/A* | *N/A* | Validated (Phase 3.2) |
| **Autoencoder** | **16** | **$432.0\times$** | **0.7017** | **0.7179** | **0.7087** | *N/A* | *N/A* | *N/A* | **Edge Challenger** |

### Scientific Conclusions Reaffirmed:
1. **Super-Baseline Advantage**:
   - Compressing the raw signal by $216\times$ (PCA-32) improves True Pooled Balanced Accuracy from **53.22%** to **69.36%** (+16.14 percentage points) and Macro-F1 from **51.57%** to **69.80%** (+18.23 percentage points).
2. **Resolution of Dimensionality Curse**:
   - In raw $\mathbb{R}^{6912}$, noise across 27 channels disperses activity clusters. Low-rank projection to $\mathbb{R}^{32}$ acts as an optimal spatial-temporal filter, preserving cross-subject kinematic invariants while filtering high-frequency noise.

---

## 7. Automated Test Suite Verification

A dedicated regression test suite was implemented in [`tests/test_phase3_3_raw_baseline.py`](file:///c:/Users/himan/OneDrive/Documents/MotionLatent/tests/test_phase3_3_raw_baseline.py) and executed via pytest:

1. `TestRawBaselinePopulationCoverage::test_total_prediction_record_count`: PASSED
   - Verifies exactly 14,973 prediction records in `reports/raw_baseline_predictions.json`.
2. `TestRawBaselinePopulationCoverage::test_exactly_once_held_out_coverage`: PASSED
   - Verifies that every eligible `(subject_id, window_idx)` appears as a held-out test instance exactly once.
3. `TestRawBaselinePopulationCoverage::test_per_subject_window_counts`: PASSED
   - Verifies exact window distribution across subjects S101–S109 matching Phase 2.1 metadata.
4. `TestFoldIsolationAndLeakage::test_scalers_exclude_test_subject`: PASSED
   - Verifies that fold scalers contain strictly 8 training subjects and zero held-out test leakage.
5. `TestMetricReconstructionFidelity::test_reconstructed_fold_metrics_match_saved_phase3`: PASSED
   - Verifies that reconstructed per-subject metrics match `reports/phase3_loso_results.json` exactly.
6. `TestMetricReconstructionFidelity::test_macro_means_and_weighted_means`: PASSED
   - Verifies 9-subject and 8-subject macro-means and window-weighted fold-score means.
7. `TestTruePooledMetricCalculations::test_true_pooled_metrics_from_concatenated_predictions`: PASSED
   - Verifies true pooled balanced accuracy (0.5322), accuracy (0.6022), and macro-F1 (0.5157).
8. `TestTruePooledMetricCalculations::test_pooled_vs_weighted_mean_distinction`: PASSED
   - Verifies mathematical inequality between window-weighted fold balanced accuracy and true pooled balanced accuracy.
9. `TestTruePooledMetricCalculations::test_s109_single_class_fold_properties`: PASSED
   - Verifies single-class fold handling for Subject 109 without runtime warnings or NaN values.

---

## 8. Summary of Changes & Artifact Manifest

### 8.1 Files Created
- [`reconstruct_raw_baseline.py`](file:///c:/Users/himan/OneDrive/Documents/MotionLatent/reconstruct_raw_baseline.py): Standalone, reproducible reconstruction script executing the 9-fold LOSO baseline evaluation and serializing predictions.
- [`reports/raw_baseline_predictions.json`](file:///c:/Users/himan/OneDrive/Documents/MotionLatent/reports/raw_baseline_predictions.json): Authoritative machine-readable predictions artifact containing metadata, per-subject metric summaries, true pooled metrics, and 14,973 window prediction records (`subject_id`, `window_idx`, `start_timestamp`, `end_timestamp`, `activity_purity`, `y_true`, `y_pred`).
- [`tests/test_phase3_3_raw_baseline.py`](file:///c:/Users/himan/OneDrive/Documents/MotionLatent/tests/test_phase3_3_raw_baseline.py): Complete automated regression test suite for Phase 3.3.
- [`reports/phase3_3_raw_baseline_validation_report.md`](file:///c:/Users/himan/OneDrive/Documents/MotionLatent/reports/phase3_3_raw_baseline_validation_report.md): This validation and audit report.

### 8.2 Files Updated
- [`reports/phase3_summary.json`](file:///c:/Users/himan/OneDrive/Documents/MotionLatent/reports/phase3_summary.json): Annotated the `Raw Baseline` entry with true pooled metrics (`true_pooled_balanced_accuracy: 0.5322`, `true_pooled_accuracy: 0.6022`, `true_pooled_macro_f1: 0.5157`).
- [`reports/phase3_representation_report.md`](file:///c:/Users/himan/OneDrive/Documents/MotionLatent/reports/phase3_representation_report.md): Updated methodological notes and comparative discussion to cite verified raw baseline pooled metrics.

---

## 9. Final Statement

With the completion of Phase 3.3, all evaluation gaps identified in Phase 3.2 are resolved. The raw 6,912-dimensional baseline predictions across all nine LOSO folds are reconstructed, verified against historical records, and stored in a standardized machine-readable format. True pooled metrics have been computed and mathematically reconciled with fold-level averages. 

**The Phase 3 evaluation record is now completely frozen, mathematically reconciled, and verified.**
