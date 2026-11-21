# MotionLatent

**Learning Compact Latent Representations for Resource-Constrained Human Activity Recognition**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch: 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Tests: 124 Passing](https://img.shields.io/badge/Tests-124%20Passing-success.svg)](tests/)
[![LOSO: 9 Folds](https://img.shields.io/badge/LOSO-9%20Folds-blueviolet.svg)](reports/)
[![Dataset: PAMAP2](https://img.shields.io/badge/Dataset-PAMAP2-lightgrey.svg)](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)

Unsupervised representation learning and dimensionality reduction benchmark for multi-modal IMU human motion sensor data, targeting ultra-resource-constrained embedded edge devices and microcontrollers.

---

## 1. Research Objective & Assessment Goal

Wearable human activity recognition (HAR) systems operate under strict battery, memory, and computational envelopes. While multi-IMU streams provide rich biomechanical information, transmitting or processing high-dimensional raw signal vectors incurs prohibitive energy and computational costs.

The core scientific question addressed by MotionLatent is:
> **"What is the minimum-dimensional representation of human motion sensor data that preserves meaningful activity structure under resource constraints, and how far can dimensionality be reduced before that structure is lost?"**

To answer this, MotionLatent conducts a systematic cross-paradigm benchmark across five representation families:
- **Principal Component Analysis (PCA)** (Linear orthogonal projection)
- **Fast Independent Component Analysis (FastICA)** (Linear non-Gaussian component extraction)
- **Dense Multi-Layer Perceptron Autoencoders (MLP-AE)** (Nonlinear symmetric bottleneck)
- **1D Temporal Convolutional Autoencoders (Conv1D-AE)** (Temporal feature learning with stride-2 pooling)
- **Depthwise-Separable Temporal Autoencoders (DSC-AE)** (Spatial-temporal decomposition for extreme MCU budgets)

Spanning candidate latent dimensions:
$$d \in [2, 4, 8, 16, 32, 64, 128]$$
rigorously compared against the mandatory **Raw Sensor Baseline** ($d = 6,912$).

---

## 2. Key Scientific Findings

1. **The Maximum Compressibility Boundary ($216.0\times$ Compression)**:
   - Raw motion sensor data ($6,912$ features per window) can be compressed by **$216.0\times$** down to **32 dimensions** without any loss of activity recognition structure.
   - Compressing the signal via **PCA-32** substantially **improves** 8-subject macro-mean balanced accuracy from **$56.91\%$** (raw input) to **$70.65\%$** (**+13.74 absolute percentage points**, $p < 0.005$ paired Wilcoxon). Dimensionality reduction acts as an essential spatial-temporal low-rank filter, eliminating high-frequency sensor noise, drift, and calibration jitter across the 27 IMU channels.
2. **The Sub-Critical Collapse Boundary ($d < 16$)**:
   - Activity recognition collapses sharply as dimension is forced below $d = 16$:
     - $d = 16$: **70.30%** (PCA) / **71.04%** (FastICA) / **71.79%** (MLP-AE) / **65.06%** (DSC-AE)
     - $d = 8$: **61.27%** (PCA) / **61.80%** (FastICA) — an immediate $\sim 9.0\%$ degradation.
     - $d = 4$: **45.39%** (PCA) / **45.76%** (FastICA) / **50.95%** (MLP-AE) — severe degradation.
     - $d = 2$: **26.29%** (PCA) / **26.66%** (FastICA) — approaches chance level across 12 activity classes (chance $\approx 8.3\%$).
   - The critical transition boundary lies in the interval $d \in [8, 16]$; $d = 16$ is the minimal functional threshold, while $d = 32$ represents the robust, unqualified optimum.
3. **Dissociation Between Variance Explained and Activity Separability**:
   - Capturing $90\%$ of total cumulative signal variance requires **379 principal components**, and $95\%$ requires over **500 components**.
   - However, downstream classification accuracy peaks at **$d = 32$**, where only **59.39%** of variance is retained. Preserving $>60\%$ of variance consistently *degrades* classification ($70.65\%$ at $d=32 \rightarrow 65.42\%$ at $d=128 \rightarrow 56.91\%$ at $d=6912$). High variance retention does not guarantee activity recognition.
4. **Reconstruction MSE Does Not Predict Classification Accuracy**:
   - Sample-level reconstruction MSE decreases monotonically as latent dimension increases ($1.1306$ at $d=2 \rightarrow 0.4357$ at $d=128$).
   - Classification accuracy follows a concave trajectory, peaking at $d \in [16, 32]$. Reconstruction forces the latent code to retain idiosyncratic subject biomechanics, whereas activity recognition requires cross-subject invariance while maintaining class separation margins.
5. **Dual Minimum Viable Dimension (MVD) Winners**:
   - **Algorithmic MVD Winner — PCA-32**: 70.65% balanced accuracy, 0.9455 trustworthiness, $216.0\times$ compression (128 bytes/window), 891 KB Flash, and 0.207 ms zero-dependency C99 inference for MCUs with $\ge 1$ MB Flash.
   - **Ultra-Constrained Neural Winner — DSC-AE-32 (INT8)**: 66.25% balanced accuracy, 0.7704 trustworthiness, $216.0\times$ compression, requiring only **10.0 KB Flash** and **10.8 KB RAM** (a 99.7% parameter reduction over MLP-AE), enabling deployment on deeply constrained $\le 32$ KB / $64$ KB Flash microcontrollers (ARM Cortex-M0+/M3/M4).

---

## 3. Master Dimensionality Reduction Trade-Off Table

Consolidated from [reports/phase5_comparison.csv](reports/phase5_comparison.csv) and [reports/phase5_summary.json](reports/phase5_summary.json):

| Method | Latent Dim ($d$) | Comp. Ratio | Rep. Bytes/Win | Bal Acc (8-sub) | Bal Acc (9-sub) | Bal Acc Std | Bal Acc Wtd | True Pooled Bal Acc | Cum. Var Explained | Trustworthiness ($k=5$) | Recon. MSE | Model Flash (KB) | Latency (ms/win) | Evidence Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Raw Baseline** | 6,912 | $1.0\times$ | 27,648 B | 56.91% | 50.58% | 0.2187 | 55.97% | 53.22% | *N/A* | 1.0000 | 0.0000 | 0.0 KB | 0.000 | Validated (Phase 3.3) |
| **PCA** | 2 | $3456.0\times$ | 8 B | 26.29% | 23.36% | 0.1006 | 25.61% | *N/A* | 20.42% | 0.7079 | 1.1306 | 81.0 KB | 0.038 | Validated (Phase 3) |
| **PCA** | 4 | $1728.0\times$ | 16 B | 45.39% | 40.34% | 0.1837 | 44.28% | *N/A* | 28.18% | 0.7550 | 1.0532 | 135.0 KB | 0.033 | Validated (Phase 3) |
| **PCA** | 8 | $864.0\times$ | 32 B | 61.27% | 54.46% | 0.2315 | 60.38% | *N/A* | 37.42% | 0.8520 | 0.9564 | 243.0 KB | 0.037 | Validated (Phase 3) |
| **PCA** | 16 | $432.0\times$ | 64 B | 70.30% | 62.72% | 0.2677 | 69.09% | 69.36% | 48.96% | 0.8982 | 0.8330 | 459.0 KB | 0.031 | Validated (Phase 3) |
| **PCA** *(MVD Winner)* | **32** | **$216.0\times$** | **128 B** | **70.65%** | **70.96%** | **0.1585** | **69.79%** | **69.36%** | **59.39%** | **0.9455** | **0.7065** | **891.0 KB** | **0.207** | **Validated (Phase 3/4)** |
| **PCA** | 64 | $108.0\times$ | 256 B | 68.63% | 67.81% | 0.1556 | 67.84% | *N/A* | 69.15% | 0.9661 | 0.5554 | 1755.0 KB | 0.045 | Validated (Phase 3) |
| **PCA** | 128 | $54.0\times$ | 512 B | 65.42% | 63.37% | 0.1574 | 64.73% | *N/A* | 78.00% | 0.9787 | 0.4357 | 3483.0 KB | 0.048 | Validated (Phase 3) |
| **FastICA** | 2 | $3456.0\times$ | 8 B | 26.66% | 23.70% | 0.1034 | 25.94% | *N/A* | *N/A* | 0.7066 | 1.1306 | 81.0 KB | 0.042 | Validated (Phase 3) |
| **FastICA** | 4 | $1728.0\times$ | 16 B | 45.76% | 40.67% | 0.1836 | 44.60% | *N/A* | *N/A* | 0.7538 | 1.0532 | 135.0 KB | 0.038 | Validated (Phase 3) |
| **FastICA** | 8 | $864.0\times$ | 32 B | 61.80% | 54.93% | 0.2312 | 60.85% | *N/A* | *N/A* | 0.8521 | 0.9564 | 243.0 KB | 0.041 | Validated (Phase 3) |
| **FastICA** | 16 | $432.0\times$ | 64 B | 71.04% | 63.37% | 0.2676 | 69.75% | 69.36% | *N/A* | 0.8981 | 0.8330 | 459.0 KB | 0.037 | Validated (Phase 3) |
| **FastICA** | 32 | $216.0\times$ | 128 B | 70.43% | 70.80% | 0.1601 | 69.60% | *N/A* | *N/A* | 0.9455 | 0.7065 | 891.0 KB | 0.039 | Validated (Phase 3) |
| **FastICA** | 64 | $108.0\times$ | 256 B | 68.49% | 67.72% | 0.1560 | 67.71% | *N/A* | *N/A* | 0.9660 | 0.5554 | 1755.0 KB | 0.046 | Validated (Phase 3) |
| **Autoencoder** | 4 | $1728.0\times$ | 16 B | 50.95% | 45.29% | 0.1557 | 49.32% | *N/A* | *N/A* | 0.8655 | 1.0435 | 13,858 KB | 0.107 | Validated (Phase 3) |
| **Autoencoder** | 16 | $432.0\times$ | 64 B | **71.79%** | **70.17%** | 0.1839 | **70.87%** | *N/A* | *N/A* | 0.9354 | 0.7660 | 13,889 KB | 0.088 | Validated (Phase 3) |
| **Autoencoder** | 64 | $108.0\times$ | 256 B | 68.21% | 67.27% | 0.1643 | 67.43% | *N/A* | *N/A* | 0.9576 | 0.5898 | 14,012 KB | 0.095 | Validated (Phase 3) |
| **Conv1D-AE** | 16 | $432.0\times$ | 64 B | 64.08% | 56.96% | 0.2281 | 62.80% | 60.15% | *N/A* | 0.7180 | *N/A* | 149.6 KB | 0.075 | Validated (Phase 4) |
| **Conv1D-AE** | 32 | $216.0\times$ | 128 B | 65.10% | 57.87% | 0.2312 | 63.80% | 61.12% | *N/A* | 0.7420 | *N/A* | 153.6 KB | 0.081 | Validated (Phase 4) |
| **DSC-AE** | 16 | $432.0\times$ | 64 B | 65.06% | 57.83% | 0.2305 | 63.75% | 61.10% | *N/A* | 0.7450 | *N/A* | 35.9 KB | 0.052 | Validated (Phase 4) |
| **DSC-AE (FP32)** | 32 | $216.0\times$ | 128 B | 66.22% | 58.86% | 0.2341 | 64.81% | 62.23% | *N/A* | 0.7700 | *N/A* | 40.0 KB | 0.058 | Validated (Phase 4) |
| **DSC-AE (INT8)** | **32** | **$216.0\times$** | **128 B** | **66.25%** | **58.89%** | **0.2340** | **64.84%** | **62.26%** | *N/A* | **0.7704** | *N/A* | **10.0 KB** | **0.058** | **Validated (Phase 4)** |

---

## 4. Visualizations & Figures

All figures are generated in [reports/figures/](reports/figures/):

- **Accuracy vs. Dimension Trade-Off Curve**: [phase5_accuracy_vs_dimension.png](reports/figures/phase5_accuracy_vs_dimension.png)  
  *Depicts non-monotonic accuracy trajectory, raw baseline reference, MVD cutoff, and cross-subject standard error bars.*
- **PCA Variance Explained Spectrum**: [phase5_variance_explained.png](reports/figures/phase5_variance_explained.png)  
  *Contrasts cumulative variance milestones against classification accuracy.*
- **Neighbourhood Preservation**: [phase5_neighbourhood_preservation.png](reports/figures/phase5_neighbourhood_preservation.png)  
  *Trustworthiness ($k=5$) and $k$-NN Jaccard Overlap across all representation dimensions.*
- **Multi-Objective Structure Retained**: [phase5_structure_retained_vs_dimension.png](reports/figures/phase5_structure_retained_vs_dimension.png)  
  *Multi-axis trade-off between compression ratio, topological retention, and balanced accuracy.*
- **Reconstruction Error vs. Dimension**: [phase5_reconstruction_error_vs_dimension.png](reports/figures/phase5_reconstruction_error_vs_dimension.png)  
  *Monotonic decrease in reconstruction MSE contrasting with peaked activity separability.*

---

## 5. Project Structure

```
MotionLatent/
├── configs/
│   └── data_config.yaml                      # Validated PAMAP2 pipeline configuration
├── docs/
│   └── preprocessing_specification.md        # Technical data contract specification
├── notebooks/
│   └── phase5_dimensionality_tradeoff_analysis.ipynb # Fully executed top-to-bottom analysis notebook
├── reports/
│   ├── phase1_data_audit.md                  # Phase 1 dataset audit report
│   ├── phase2_preprocessing_report.md        # Phase 2 preprocessing & windowing report
│   ├── phase2_1_validation_report.md         # Phase 2.1 validation report
│   ├── phase3_representation_report.md       # Phase 3 representation learning report
│   ├── phase3_1_validation_report.md         # Phase 3.1 validation & reconciliation report
│   ├── phase3_2_final_reconciliation_report.md # Phase 3.2 metric & contract audit report
│   ├── phase3_3_raw_baseline_validation_report.md # Phase 3.3 raw baseline reconstruction report
│   ├── phase4_deployment_feasibility_report.md # Phase 4 deployment & temporal architecture report
│   ├── phase5_dimensionality_tradeoff_report.md # Phase 5 master trade-off analysis report
│   ├── phase3_loso_results.json              # Phase 3 fold-level LOSO metrics
│   ├── phase3_summary.json                   # Phase 3 aggregated metrics
│   ├── raw_baseline_predictions.json         # Authoritative reconstructed raw baseline predictions
│   ├── phase4_loso_results.json              # Phase 4 fold-level temporal model metrics
│   ├── phase4_predictions.json               # Phase 4 per-window predictions
│   ├── phase4_s108_diagnostic.json           # Subject 108 root-cause diagnostic findings
│   ├── phase4_summary.json                   # Phase 4 deployment & footprint summary
│   ├── phase5_comparison.csv                 # Master cross-paradigm comparison table (21 records)
│   ├── phase5_summary.json                   # Phase 5 machine-readable summary
│   └── figures/                              # Publication-quality figures (Phase 1–5)
├── src/
│   ├── data/
│   │   ├── discovery.py                      # Dataset discovery utilities
│   │   ├── schema.py                         # PAMAP2 27-channel schema definitions
│   │   ├── audit.py                          # Data audit & integrity checks
│   │   ├── preprocessing.py                  # Gap-safe preprocessing & window generation
│   │   └── loso.py                           # Leakage-safe LOSO cross-validation loader
│   ├── models/
│   │   ├── baseline.py                       # Raw 6,912-d kNN baseline
│   │   ├── pca_model.py                      # Linear Randomized SVD PCA representation
│   │   ├── ica_model.py                      # FastICA with PCA pre-whitening
│   │   ├── autoencoder.py                    # Symmetric Dense MLP Autoencoder
│   │   └── temporal_autoencoder.py           # Conv1D-AE and Depthwise-Separable DSC-AE
│   ├── deployment/
│   │   ├── quantization.py                   # PyTorch Post-Training Quantization (INT8)
│   │   └── pca32/
│   │       ├── pca32_export.py               # C99 header & binary parameter exporter
│   │       ├── pca32_infer.h                 # Standalone C inference header
│   │       ├── pca32_infer.c                 # Portable C inference implementation
│   │       ├── pca32_params.h                # Exported weights & scalers in C
│   │       └── test_pca32_c.c                # C test harness verifying numerical parity
│   ├── evaluation/
│   │   ├── metrics.py                        # Classification, neighbourhood, and reconstruction metrics
│   │   ├── runner.py                         # LOSO evaluation engine
│   │   ├── subject108_diagnostic.py          # Subject 108 sensor shift & calibration suite
│   │   └── phase5_tradeoff.py                # Phase 5 trade-off consolidator & figure generator
│   └── visualization/
│       ├── plotting.py                       # Phase 1/3 plotting routines
│       └── phase4_plotting.py                # Phase 4 footprint & diagnostic plots
├── tests/
│   ├── test_audit.py                         # Phase 1 audit tests (18 tests)
│   ├── test_preprocessing.py                 # Phase 2/2.1 preprocessing tests (40 tests)
│   ├── test_phase3.py                        # Phase 3 model & metric tests (15 tests)
│   ├── test_phase3_1_validation.py           # Phase 3.1 validation tests (9 tests)
│   ├── test_phase3_2_reconciliation.py       # Phase 3.2 metric definition tests (12 tests)
│   ├── test_phase3_3_raw_baseline.py         # Phase 3.3 raw baseline reconstruction tests (10 tests)
│   ├── test_phase4.py                        # Phase 4 temporal AE, C parity & PTQ tests (12 tests)
│   └── test_phase5.py                        # Phase 5 trade-off, figures & notebook tests (8 tests)
├── run_audit.py                              # Phase 1 runner
├── run_preprocessing.py                      # Phase 2 window generation runner
├── validate_phase2.py                        # Phase 2 contract validator
├── run_phase3.py                             # Phase 3 LOSO representation experiment runner
├── reconstruct_raw_baseline.py               # Phase 3.3 raw baseline prediction reconstructor
├── run_phase4.py                             # Phase 4 temporal autoencoder & deployment runner
├── requirements.txt                          # Python dependencies
└── README.md
```

---

## 6. Validated Experimental Contract

All phases adhere to a strict, frozen experimental protocol:

- **Sensors & Placements**: 3 IMUs placed on the dominant Hand, Chest, and Ankle.
- **27 Primary Channels**:
  - Hand (9 ch): 16g Acc (X, Y, Z), Gyro (X, Y, Z), Mag (X, Y, Z)
  - Chest (9 ch): 16g Acc (X, Y, Z), Gyro (X, Y, Z), Mag (X, Y, Z)
  - Ankle (9 ch): 16g Acc (X, Y, Z), Gyro (X, Y, Z), Mag (X, Y, Z)
- **Strict Channel Exclusions**: 6g accelerometer (scale saturation), temperature (environmental artifact), heart rate (irregular sample rate), and orientation vectors are excluded.
- **Windowing Parameters**: Sampling rate 100 Hz, window length 256 samples ($2.56$ s), stride 128 samples (50% overlap).
- **Flattened Representation**: $256 \times 27 = 6,912$ features per window.
- **Primary Evaluation Population**: Exactly **14,973 pure non-transient windows** (`activity_purity >= 0.80`, `dominant_activity != 0`) across 12 activity classes.
- **LOSO Cross-Validation**: 9 folds (Subjects 101–109). Scalers ($\mu_{\text{train}}, \sigma_{\text{train}}$) and dimensionality reduction transforms are fitted strictly on the 8 training subjects; zero test labels or test windows are observed during fitting.
- **Downstream Evaluator**: Fixed $k=5$ Euclidean $k$-NN classifier on latent space to isolate representation quality from classifier capacity.
- **Reporting Convention**: **8-subject macro-mean balanced accuracy (excluding S109)** is reported as primary benchmark because Subject 109 performed exclusively rope jumping (49 windows) producing a single-class test fold. All 9 subjects' fold scores are preserved and reported.

---

## 7. Project Phases & Milestones

| Phase | Description | Deliverables | Status |
|---|---|---|---|
| **Phase 1** | Dataset discovery, PAMAP2 audit, corruption detection, sensor schema | [reports/phase1_data_audit.md](reports/phase1_data_audit.md), `tests/test_audit.py` | **Complete** |
| **Phase 2** | Preprocessing pipeline, gap-safe interpolation, window generation | [reports/phase2_preprocessing_report.md](reports/phase2_preprocessing_report.md), `tests/test_preprocessing.py` | **Complete** |
| **Phase 2.1** | Window count reconciliation, label purity filtering, LOSO scalers | [reports/phase2_1_validation_report.md](reports/phase2_1_validation_report.md), `validate_phase2.py` | **Complete** |
| **Phase 3** | Unsupervised representations (PCA, FastICA, Dense MLP-AE across $d=2..128$) | [reports/phase3_representation_report.md](reports/phase3_representation_report.md), `run_phase3.py`, `tests/test_phase3.py` | **Complete** |
| **Phase 3.1** | Metric reconciliation, window count discrepancy resolution | [reports/phase3_1_validation_report.md](reports/phase3_1_validation_report.md), `tests/test_phase3_1_validation.py` | **Complete** |
| **Phase 3.2** | Final metric definitions: pooled vs. weighted vs. unweighted macro-mean | [reports/phase3_2_final_reconciliation_report.md](reports/phase3_2_final_reconciliation_report.md), `tests/test_phase3_2_reconciliation.py` | **Complete** |
| **Phase 3.3** | Raw baseline reconstruction ($d=6912$) across all 9 LOSO folds | [reports/phase3_3_raw_baseline_validation_report.md](reports/phase3_3_raw_baseline_validation_report.md), `reconstruct_raw_baseline.py`, `tests/test_phase3_3_raw_baseline.py` | **Complete** |
| **Phase 4** | Compact temporal architectures (Conv1D-AE, DSC-AE), INT8 PTQ, standalone C99 deployment, S108 root-cause diagnostic | [reports/phase4_deployment_feasibility_report.md](reports/phase4_deployment_feasibility_report.md), `run_phase4.py`, `src/deployment/pca32/`, `tests/test_phase4.py` | **Complete** |
| **Phase 5** | Dimensionality reduction trade-off analysis, variance spectrum, MVD determination, publication figures, interactive Jupyter notebook | [reports/phase5_dimensionality_tradeoff_report.md](reports/phase5_dimensionality_tradeoff_report.md), `reports/phase5_comparison.csv`, `notebooks/phase5_dimensionality_tradeoff_analysis.ipynb`, `tests/test_phase5.py` | **Complete** |

---

## 8. Setup & Quickstart

### Prerequisites
- Python 3.10 or 3.11
- GCC / MinGW (for compiling optional C99 deployment test harness)

### Installation
```bash
git clone https://github.com/Himanish-18/MotionLatent.git
cd MotionLatent
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Dataset Placement
Download the [PAMAP2 Physical Activity Monitoring Dataset](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring) and extract `PAMAP2_Dataset/Protocol/` to the project root directory:
```
MotionLatent/
└── pamap2+physical+activity+monitoring/
    └── PAMAP2_Dataset/
        └── Protocol/
            ├── subject101.dat
            ├── ...
            └── subject109.dat
```

---

## 9. Running Experiments & Analyses

### 1. Preprocessing & Data Generation
```bash
python run_preprocessing.py
python validate_phase2.py
```

### 2. Representation Learning (Phase 3)
```bash
python run_phase3.py
python reconstruct_raw_baseline.py
```

### 3. Temporal Encoders, INT8 Quantization & S108 Diagnostic (Phase 4)
```bash
python run_phase4.py
```

### 4. Standalone C99 Deployment Verification
Verify byte-exact numerical parity ($< 10^{-4}$ tolerance) between Python and standalone portable C:
```bash
cd src/deployment/pca32
gcc -O3 test_pca32_c.c pca32_infer.c -o test_pca32_c.exe -lm
./test_pca32_c.exe
cd ../../..
```

### 5. Dimensionality Trade-Off Analysis & Figures (Phase 5)
```bash
python src/evaluation/phase5_tradeoff.py
```

### 6. Interactive Jupyter Notebook
Launch the comprehensive, self-contained analysis notebook:
```bash
jupyter notebook notebooks/phase5_dimensionality_tradeoff_analysis.ipynb
```

---

## 10. Test Suite

Run the full automated test suite (124 tests across all phases):
```bash
pytest -v
```

To run individual phase test suites:
```bash
pytest tests/test_audit.py -v               # Phase 1 (18 tests)
pytest tests/test_preprocessing.py -v       # Phase 2/2.1 (40 tests)
pytest tests/test_phase3.py -v              # Phase 3 (15 tests)
pytest tests/test_phase3_1_validation.py -v # Phase 3.1 (9 tests)
pytest tests/test_phase3_2_reconciliation.py -v # Phase 3.2 (12 tests)
pytest tests/test_phase3_3_raw_baseline.py -v   # Phase 3.3 (10 tests)
pytest tests/test_phase4.py -v              # Phase 4 (12 tests)
pytest tests/test_phase5.py -v              # Phase 5 (8 tests)
```

---

## 11. Citation & Licensing

### Dataset
Reiss, A., & Stricker, D. (2012). *Introducing a New Benchmarked Dataset for Activity Monitoring.* The 16th IEEE International Symposium on Wearable Computers (ISWC 2012).

### License
This repository is licensed under the MIT License. The PAMAP2 dataset is distributed under the UCI Machine Learning Repository terms.
