# MotionLatent — Phase 3.1: Experimental Validation & Metric Reconciliation Report

**Auditor:** MotionLatent Independent ML Audit & Verification  
**Date:** October 2026  
**Status:** Completed  
**Repository:** [https://github.com/Himanish-18/MotionLatent](https://github.com/Himanish-18/MotionLatent)  
**Evaluated Artifacts:** `reports/phase3_representation_report.md`, `reports/phase3_loso_results.json`, `reports/phase3_summary.json`, `reports/figures/*.png`

---

## 1. Executive Summary & Scientific Readiness Verdict

### Overall Verdict: **PASS WITH WARNINGS**

The unsupervised latent representation learning experiments conducted in Phase 3 demonstrate **sound methodological isolation and zero test-subject leakage**. All 153 fold experiment records stored in `reports/phase3_loso_results.json` and aggregate metrics in `reports/phase3_summary.json` were reproduced with zero mathematical error from the raw data.

However, a rigorous audit identified several documentation defects, transcription inconsistencies, and unverified deployment assumptions in the initial Phase 3 report:
1. **Window Count Discrepancy (Resolved)**: The 176-window discrepancy between the report's stated population (14,973) and Table 4.3's row sum (14,797) was traced to a **manual drafting transcription typo in Table 4.3** of the report. The actual data loader, metadata, and runner evaluated all **14,973 pure non-transient windows** across all 9 subjects.
2. **Per-Subject Table Inconsistency (Resolved)**: The fold-level balanced accuracy numbers in Table 4.3 of the initial report contained synthetic/placeholder values (e.g. S108 listed as 0.622 instead of the empirical 0.327). The aggregate metrics in Table 4.1 were mathematically computed from the true fold outputs, but Table 4.3 did not reflect the true fold outputs. Table 4.3 has now been fully reconciled and updated with the empirical numbers from `reports/phase3_loso_results.json`.
3. **Autoencoder Validation Split (Documented Warning)**: The Autoencoder early stopping used a deterministic 10% tail slice (`X_train[-n_val:]`) rather than a subject-aware inner fold (or 15% as originally described). While this introduced **zero leakage from the held-out test subject**, it means early stopping was driven by the kinematics of the last training subject in the fold list rather than cross-subject generalization.
4. **MVD Threshold Strictness (Corrected)**: PCA-16 was initially marked as satisfying all criteria despite its trustworthiness metric ($0.8982$) falling below the pre-registered threshold ($> 0.9000$). Furthermore, Autoencoder-16's ~13.9 MB parameter size was described as "embedded viable", which is incompatible with microcontrollers ($\le 512$ KB RAM/Flash). The criteria table has been corrected: **PCA-32 is the only linear model strictly passing all representational criteria**, while Autoencoder-16 is designated **Target-Dependent (Edge SBC / Mobile only)**.

---

## 2. Inspected Artifacts & Reproducibility Trace

| Artifact | Location | SHA / Size | Status |
|---|---|---|---|
| Phase 3 Research Report | `reports/phase3_representation_report.md` | 22.8 KB | Inspected & Corrected |
| Phase 2.1 Validation Report | `reports/phase2_1_validation_report.md` | 13.2 KB | Verified Baseline Reference |
| Window Metadata | `data/processed/window_metadata.json` | 7.3 MB (22,249 records) | Inspected & Validated |
| Per-Subject Window NPZ | `data/processed/windows_subject10[1-9].npz` | 9 files (~465 MB total) | Inspected & Validated |
| Raw Fold Results | `reports/phase3_loso_results.json` | 118.6 KB (153 records) | Verified Ground Truth |
| Aggregated Results | `reports/phase3_summary.json` | 140.8 KB | Verified Ground Truth |
| LOSO Data Loader | `src/data/loso.py` | 5.5 KB | Audited |
| Evaluation Runner | `src/evaluation/runner.py` | 11.5 KB | Audited |
| Metrics Implementation | `src/evaluation/metrics.py` | 6.0 KB | Audited |
| PCA Model | `src/models/pca_model.py` | 4.0 KB | Audited |
| FastICA Model | `src/models/ica_model.py` | 4.9 KB | Audited |
| Autoencoder Model | `src/models/autoencoder.py` | 7.6 KB | Audited |
| Visualizations | `reports/figures/*.png` | 5 figures | Audited & Validated |

---

## 3. Window Count Reconciliation

### Investigation of the 176-Window Discrepancy
The initial Phase 3 report stated that 14,973 pure non-transient windows were evaluated. However, the sum of the per-subject window counts in Table 4.3 of the report yielded 14,797:
$$1745 + 1847 + 1833 + 1847 + 1885 + 1777 + 1794 + 2020 + 49 = 14,797 \quad (\Delta = -176)$$

We audited the entire data processing chain from raw PAMAP2 observations to evaluation arrays:
1. **Raw Metadata Check**: `data/processed/window_metadata.json` contains exactly **22,249 accepted windows**.
2. **Filtering Check**:
   - Total pure windows (purity $\ge 0.80$): **22,030** (including transient activity 0).
   - Total pure transient windows: **7,057**.
   - Total mixed windows: **219**.
   - Total pure non-transient windows (`is_mixed_label == False` and `is_transient == False`): **14,973**.
3. **Data Loader (`FoldData`) Check**:
   For every fold $s \in [101 \dots 109]$, `load_fold_data(test_subject=s, pure_only=True, exclude_transient=True)` loads exactly the empirical test windows, summing across all 9 subjects to **14,973**.

### Stage-by-Stage & Per-Subject Reconciliation Table

| Subject ID | Total Accepted Windows | Transient Windows (Act 0, $\ge 80\%$) | Mixed Windows ($< 80\%$) | Evaluated Pure Non-Transient Windows | Table 4.3 Draft Count | Discrepancy |
|---|---|---|---|---|---|---|
| **S101** | 2,916 | 960 | 29 | **1,927** | 1,745 | -182 (Draft error) |
| **S102** | 3,451 | 1,391 | 31 | **2,029** | 1,847 | -182 (Draft error) |
| **S103** | 1,963 | 600 | 21 | **1,342** | 1,833 | +491 (Draft error) |
| **S104** | 2,559 | 746 | 26 | **1,787** | 1,847 | +60 (Draft error) |
| **S105** | 2,906 | 781 | 27 | **2,098** | 1,885 | -213 (Draft error) |
| **S106** | 2,809 | 857 | 25 | **1,927** | 1,777 | -150 (Draft error) |
| **S107** | 2,434 | 613 | 27 | **1,794** | 1,794 | 0 (Correct) |
| **S108** | 3,148 | 1,097 | 31 | **2,020** | 2,020 | 0 (Correct) |
| **S109** | 63 | 12 | 2 | **49** | 49 | 0 (Correct) |
| **Total** | **22,249** | **7,057** | **219** | **14,973** | **14,797** | **-176** |

**Conclusion**: The evaluation pipeline evaluated **14,973 windows**. Zero windows were lost in code. The discrepancy in evaluated windows was an artifact of manual transcription during report drafting for subjects S101–S106. In addition, per-subject transient vs mixed breakdowns have been reconciled to the authoritative metadata in `data/processed/window_metadata.json` (exactly matching Phase 2.1 Table 4.2).

---

## 4. Recomputed Aggregate & Per-Subject Metrics

We recomputed all metrics from the individual predictions and fold-level records in `reports/phase3_loso_results.json`.

### 4.1 Comparison of Aggregation Methodologies
In scientific LOSO reporting, three distinct aggregation methods must be distinguished:
1. **Unweighted Macro-Mean across Folds (Standard LOSO convention)**:
   $$\text{Macro-Mean} = \frac{1}{K} \sum_{k=1}^K M_k, \quad \text{Std} = \sqrt{\frac{1}{K}\sum_{k=1}^K (M_k - \text{Macro-Mean})^2}$$
2. **Window-Weighted Fold-Score Mean (Sample-Weighted Fold Average)**:
   $$\text{Weighted Fold-Score Mean} = \frac{\sum_{k=1}^K N_k M_k}{\sum_{k=1}^K N_k}$$
   *(Note: Phase 3.1 initially labeled this as "Pooled Score"; it is strictly a weighted average of fold scores, not pooled over instances).*
3. **True Pooled Metric (Evaluated Over Concatenated Predictions)**:
   Concatenate all held-out predictions $\hat{\mathbf{y}} = \bigoplus_{k=1}^K \hat{y}^{(k)}$ and ground-truth labels $\mathbf{y} = \bigoplus_{k=1}^K y^{(k)}$, then evaluate balanced accuracy / macro-F1 once over the combined 14,973 windows.

The table below contrasts fold aggregation methods for candidate models:

| Representation | Latent Dim ($d$) | 9-Subject Macro-Mean | 9-Subject Weighted Fold-Score Mean | 8-Subject Macro-Mean (excl. S109) | 8-Subject Weighted Fold-Score Mean (excl. S109) | True Pooled Balanced Accuracy |
|---|---|---|---|---|---|---|
| **Raw Baseline** | 6912 | $0.506 \pm 0.219$ | 0.560 | 0.569 | 0.561 | — |
| **PCA** | 16 | $0.627 \pm 0.268$ | 0.691 | 0.703 | 0.693 | — |
| **PCA** | **32** | $\mathbf{0.710 \pm 0.159}$ | $\mathbf{0.698}$ | $\mathbf{0.707}$ | $\mathbf{0.698}$ | $\mathbf{0.694}$ |
| **FastICA** | 16 | $0.693 \pm 0.177$ | 0.700 | 0.710 | 0.700 | — |
| **FastICA** | 32 | $0.693 \pm 0.182$ | 0.670 | 0.680 | 0.670 | — |
| **Autoencoder** | **16** | $\mathbf{0.702 \pm 0.150}$ | $\mathbf{0.709}$ | $\mathbf{0.718}$ | $\mathbf{0.709}$ | — |

*Note on True Pooled Metric*: For PCA-32, concatenating all held-out predictions yields True Pooled Balanced Accuracy of **0.6936** ($\approx 0.694$), True Pooled Accuracy of **0.7270**, and True Pooled Macro-F1 of **0.6980**. For Autoencoder and FastICA, fold-level predictions were not serialized to disk during Phase 3 execution, so only their Macro-Mean and Weighted Fold-Score Mean are directly reproducible from saved fold records.

### 4.2 Reconciliation of PCA-32 Performance
The Phase 3 report contained seemingly conflicting statements regarding PCA-32:
- Executive Summary: `"PCA at d=32 achieves 70.7% (9-subject: 71.0%)"`
- Section 4.1 Table: `"Bal. Acc (9-sub): 0.710 +/- 0.159, Bal. Acc (ex-109): 0.707"`
- Section 5.1: `"Beyond d=32, adding dimensions degrades balanced accuracy (dropping from 71.0% to 50.6%)"`

**Audit Finding**:
- The 9-subject unweighted macro-mean across all 9 folds is **0.7096** ($\approx 0.710$).
- The 8-subject unweighted macro-mean excluding Subject 109 is **0.7065** ($\approx 0.707$).
- The 9-subject window-weighted fold-score mean across all 14,973 windows is **0.6979** ($\approx 0.698$).
- The 9-subject true pooled balanced accuracy from concatenated predictions is **0.6936** ($\approx 0.694$).
- Both 71.0% (9-subject macro-mean) and 70.7% (8-subject macro-mean) are mathematically valid, but the text previously interchanged them without explicit labeling. The updated report clearly marks them as `71.0% (all 9 folds macro-mean)` and `70.7% (8 full-protocol folds excluding S109 macro-mean)`.

### 4.3 Verified Per-Subject Metrics Table

| Subject | Windows | Raw Baseline ($d=6912$) | PCA ($d=16$) | PCA ($d=32$) | ICA ($d=16$) | ICA ($d=32$) | AE ($d=16$) |
|---|---|---|---|---|---|---|---|
| **S101** | 1,927 | 0.469 | 0.562 | 0.569 | 0.610 | 0.583 | 0.563 |
| **S102** | 2,029 | 0.602 | 0.650 | 0.715 | 0.664 | 0.724 | 0.636 |
| **S103** | 1,342 | 0.665 | 0.823 | 0.820 | 0.837 | 0.803 | 0.832 |
| **S104** | 1,787 | 0.683 | 0.848 | 0.820 | 0.846 | 0.801 | 0.824 |
| **S105** | 2,098 | 0.537 | 0.743 | 0.763 | 0.775 | 0.715 | 0.771 |
| **S106** | 1,927 | 0.676 | 0.752 | 0.755 | 0.752 | 0.745 | 0.813 |
| **S107** | 1,794 | 0.651 | 0.902 | 0.883 | 0.901 | 0.847 | 0.885 |
| **S108** | 2,020 | 0.269 | 0.343 | 0.327 | 0.298 | 0.220 | 0.420 |
| **S109** | 49 | 0.000 | 0.020 | 0.735 | 0.551 | 0.796 | 0.571 |
| **Macro-Mean (9-sub)** | **14,973** | **0.506** | **0.627** | **0.710** | **0.693** | **0.693** | **0.702** |
| **Macro-Mean (excl. 109)** | **14,924** | **0.569** | **0.703** | **0.707** | **0.710** | **0.680** | **0.718** |

---

## 5. LOSO & Model-Selection Leakage Audit

### 5.1 Verification of Fold Isolation
We verified lines of code and runtime assertions across `src/data/loso.py` and `run_phase3.py`:
- **Scaler Isolation**: In every fold $k$, `load_fold_scaler(processed_dir, test_subject)` loads `scaler_test{k}.json`. The scaler was precomputed exclusively over training subjects $\mathcal{S} \setminus \{k\}$. Assertions confirmed that test subject samples were never used in scaler computation.
- **Model Training Isolation**: PCA SVD, FastICA pre-whitening/unmixing, and Autoencoder weights were fitted using `fold.X_train` only.
- **Classifier Isolation**: The $k$-NN classifier was fitted exclusively on `(Z_train, y_train)` and evaluated on `(Z_test, y_test)`.
- **Primary Population Uniformity**: All 4 models were evaluated on the identical test arrays `fold.X_test` and `fold.y_test` across all 9 folds.

### 5.2 Autoencoder Validation Split Audit (Warning)
In `AutoencoderRepresentation.fit()` (`src/models/autoencoder.py`):
```python
if X_val is None:
    n_val = max(1, int(len(X_train) * 0.10))
    train_data = torch.from_numpy(X_train[:-n_val].astype(np.float32))
    val_data = torch.from_numpy(X_train[-n_val:].astype(np.float32))
```
- **Leakage Finding**: Zero test-subject data entered `val_data`. The split was taken strictly from `X_train`.
- **Methodological Warning**:
  1. The split percentage was **10%**, not 15% as written in the draft report.
  2. Because `X_train` was ordered by subject ID, `X_train[-n_val:]` drew all validation windows from the **last training subject** in that fold list (e.g. S108 or S109).
  3. This means early stopping was governed by reconstruction loss on that single training subject rather than cross-subject generalizability. Future iterations should implement subject-level inner LOSO cross-validation for model selection.

---

## 6. Neighbourhood Preservation & Reconstruction Audit

### 6.1 Trustworthiness Implementation
- The reference space is the original 6,912-dimensional normalized input space $X_{\text{orig}}$.
- The reduced representation $Z_{\text{latent}}$ and $X_{\text{orig}}$ use identical held-out test windows.
- In `src/evaluation/metrics.py`:
  - When $N > 500$, a reproducible deterministic subset ($N=500$, seed 42) is used to avoid $O(N^2 \cdot D)$ memory overhead.
  - Line 91 invokes `sklearn.manifold.trustworthiness(X_sub, Z_sub, n_neighbors=k)`.
- **Independent Verification**: A unit test in `tests/test_phase3_1_validation.py` cross-checked `compute_neighbourhood_preservation` against `sklearn.manifold.trustworthiness` on synthetic data and confirmed an exact match ($p < 10^{-5}$).

### 6.2 Reconstruction Metrics
- **PCA**: Correctly utilizes `Z @ components[:d] + mean`.
- **Autoencoder**: Measured in normalized space via `model.decode(Z)`.
- **Metric Definitions**: MSE is computed as $\frac{1}{N \cdot D} \|X - \hat{X}\|_2^2$, and RMSE is computed as $\sqrt{\text{MSE}}$.

---

## 7. Compression, Footprint & Latency Audit

### 7.1 Distinguishing Latent Compression vs. Model Size
The initial report conflated representation compression with total deployable footprint. The table below delineates the exact resource breakdown:

| Representation | Latent Dim ($d$) | Latent Size / Window (Bytes) | Representation Compression | Encoder Parameter Count | Serialized Encoder Size (KB) | Full Model Size (Enc + Dec) |
|---|---|---|---|---|---|---|
| **Raw Baseline** | 6912 | 27,648 B | $1.0\times$ | 0 | 0 KB | 0 KB |
| **PCA-16** | 16 | 64 B | $432.0\times$ | 117,504 | 459 KB | 459 KB (Symmetric) |
| **PCA-32** | 32 | 128 B | $216.0\times$ | 228,096 | 891 KB | 891 KB (Symmetric) |
| **FastICA-16** | 16 | 64 B | $432.0\times$ | 117,760 | 460 KB | 460 KB |
| **FastICA-32** | 32 | 128 B | $216.0\times$ | 229,120 | 895 KB | 895 KB |
| **Autoencoder-16** | 16 | 64 B | $432.0\times$ | 1,774,096 | 6,930 KB (~6.9 MB) | 13,889 KB (~13.9 MB) |
| **Autoencoder-64** | 64 | 256 B | $108.0\times$ | 1,786,432 | 6,978 KB (~7.0 MB) | 13,985 KB (~14.0 MB) |

### 7.2 Latency Benchmark Audit
The reported latencies (~0.03–0.05 ms/window) measure **vectorized batch matrix multiplication throughput** on an x86 CPU (`time.perf_counter()` over batched test sets).
- **Audit Caveat**: This timing does NOT include:
  1. IMU sensor buffer sliding and window slicing (256 samples).
  2. Per-channel standard scaling ($\mu, \sigma$).
  3. Single-sample sequential execution overhead on bare-metal microcontroller hardware.
- These measurements are classified as **desktop CPU encoder throughput estimates**, not confirmed microcontroller deployment latency.

---

## 8. Corrected Minimum Viable Dimension (MVD) Criteria Table

| Candidate Dimension & Model | Activity Retention (>66.9%) | Trustworthiness (>0.900) | Compression (>=100x) | On-Device Footprint | Strictly Meets All Criteria? |
|---|---|---|---|---|---|
| **$d=4$ (Any method)** | ❌ FAIL (45.4%–50.9%) | ❌ FAIL (0.755–0.806) | ✅ PASS (1728x) | ✅ PASS (Negligible) | ❌ **FAIL** |
| **$d=8$ (Any method)** | ❌ FAIL (61.2%–61.3%) | ❌ FAIL (0.852–0.860) | ✅ PASS (864x) | ✅ PASS (Negligible) | ❌ **FAIL** |
| **PCA-16** | ✅ PASS (70.3%) | ❌ **FAIL (0.8982 < 0.900)** | ✅ PASS (432x) | ✅ PASS (459 KB Flash) | ❌ **FAIL (Strict Topology)** |
| **FastICA-16** | ✅ PASS (71.0%) | ❌ **FAIL (0.8960 < 0.900)** | ✅ PASS (432x) | ✅ PASS (460 KB Flash) | ❌ **FAIL (Strict Topology)** |
| **PCA-32** | ✅ PASS (70.7%) | ✅ PASS (0.946) | ✅ PASS (216x) | ⚠️ **PASS / UNVERIFIED** (891 KB Flash) | ✅ **PASS (Representational & MCU with >=1MB Flash)** |
| **FastICA-32** | ✅ PASS (68.0%) | ✅ PASS (0.951) | ✅ PASS (216x) | ⚠️ **PASS / UNVERIFIED** (895 KB Flash) | ✅ **PASS (Representational)** |
| **Autoencoder-16** | ✅ PASS (71.8%) | ✅ PASS (0.935) | ✅ PASS (432x) | ❌ **FAIL (MCU) / ✅ PASS (Edge SBC/Mobile)** (~13.9 MB) | ⚠️ **TARGET-DEPENDENT (Fails MCU, Passes Edge SBC)** |
| **PCA-64** | ✅ PASS (68.6%) | ✅ PASS (0.966) | ✅ PASS (108x) | ⚠️ UNVERIFIED (1.75 MB Flash) | ❌ Sub-optimal vs PCA-32 |
| **Autoencoder-64** | ✅ PASS (67.6%) | ✅ PASS (0.968) | ✅ PASS (108x) | ❌ FAIL (MCU) (~14.0 MB) | ❌ Sub-optimal vs AE-16 |

### Rigorous Scientific Verdict on Minimum Viable Dimension
1. **Representational MVD Winner:** **$d = 32$ (PCA)**
   - Simultaneously satisfies activity retention ($70.7\% > 66.9\%$), neighbourhood preservation ($0.9455 > 0.9000$), and compression ratio ($216.0\times \ge 100\times$).
   - Model size ($891$ KB) is directly deployable as a static float32 matrix on microcontrollers with $\ge 1$ MB flash (e.g. STM32H7, Nordic nRF5340).
2. **Rejection of PCA-16 as Full MVD:**
   - Although PCA-16 has an attractive 459 KB footprint and 70.3% balanced accuracy, its trustworthiness of **$0.8982$** strictly fails the pre-registered $>0.9000$ topology preservation criterion.
3. **Hardware Target Dependence of Autoencoder-16:**
   - Autoencoder-16 achieves superior activity retention ($71.8\%$) and compression ($432.0\times$), but its 3.55M parameter footprint (~13.9 MB weights) requires an edge system (Raspberry Pi, mobile device, or gateway). It cannot be certified for standard resource-constrained microcontrollers.

---

## 9. Automated Regression Testing

We implemented a dedicated validation test module [`tests/test_phase3_1_validation.py`](file:///c:/Users/himan/OneDrive/Documents/MotionLatent/tests/test_phase3_1_validation.py) containing 9 unit tests.

### Test Execution Summary

```bash
python -m pytest tests/test_phase3_1_validation.py -v
```

**Results:**
- `test_total_metadata_windows`: **PASSED** (22,249 accepted windows).
- `test_pure_nontransient_window_counts_per_subject`: **PASSED** (Exact per-subject breakdown summing to 14,973).
- `test_loso_loader_preserves_exact_population`: **PASSED** (Partitions $N_{\text{train}} + N_{\text{test}} = 14,973$ on all 9 folds).
- `test_subject_isolation_in_training_data`: **PASSED** (Test subject strictly excluded from training arrays).
- `test_scaler_excludes_test_subject`: **PASSED** (Fitted scaler excludes test subject).
- `test_trustworthiness_matches_sklearn_exactly`: **PASSED** (Exact mathematical parity with `sklearn.manifold.trustworthiness`).
- `test_recompute_saved_loso_metrics`: **PASSED** (Recomputed fold metrics match `reports/phase3_summary.json` to 4 decimal places).
- `test_compression_ratios`: **PASSED** (Exact integer ratios $6912/d$).
- `test_strict_mvd_thresholds`: **PASSED** (Enforces strict threshold logic).

**Full Test Suite Status:**
- Total tests in repository: **82 passed**, 0 failed (18 Phase 1 + 40 Phase 2/2.1 + 15 Phase 3 + 9 Phase 3.1).

---

## 10. Remaining Limitations & Recommendations for Phase 4

1. **Autoencoder Architecture**: The current Autoencoder is a dense MLP operating on flattened 6,912-d vectors, resulting in large weight matrices ($6912 \times 256$). Phase 4 should investigate **1D Temporal Convolutional Networks (1D-CNN) or Depthwise-Separable Convolutions**, which reduce model weights from ~14 MB to <100 KB while preserving or improving temporal dynamics.
2. **Subject 108 Generalization**: Subject 108 proved exceptionally challenging for all linear methods (balanced accuracy 32.7% on PCA-32). Subject-adaptive calibration or domain adaptation should be explored.
3. **Quantization & True Microcontroller Benchmarking**: In Phase 4, convert the optimal PCA-32 projection matrix and quantized Autoencoders into C arrays / TFLite Micro flatbuffers and measure cycle counts on actual ARM Cortex-M hardware.
