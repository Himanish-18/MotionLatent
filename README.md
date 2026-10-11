# MotionLatent

Unsupervised representation-learning for motion-sensor data, based on the PAMAP2 Physical Activity Monitoring dataset.

## Research Objective

Determine how much motion-sensor data can be compressed into a compact representation while preserving meaningful activity structure, neighbourhood structure, and acceptable reconstruction quality for resource-constrained/on-board activity recognition.

## Project Structure

```
MotionLatent/
├── configs/
│   └── data_config.yaml         # Pipeline configuration
├── docs/
│   └── preprocessing_specification.md
├── reports/
│   ├── phase1_data_audit.md             # Phase 1 audit report
│   ├── phase2_preprocessing_report.md   # Phase 2 preprocessing report
│   ├── phase2_1_validation_report.md    # Phase 2.1 validation report
│   ├── phase3_representation_report.md  # Phase 3 representation learning report
│   ├── phase3_loso_results.json         # Fold-level LOSO metric records
│   ├── phase3_summary.json              # Aggregated LOSO metrics
│   └── figures/                         # Generated plots
├── src/
│   ├── data/
│   │   ├── discovery.py                 # Dataset discovery utilities
│   │   ├── schema.py                    # PAMAP2 schema definitions
│   │   ├── audit.py                     # Audit functions
│   │   ├── preprocessing.py             # Preprocessing & windowing
│   │   └── loso.py                      # Leakage-safe LOSO loader
│   ├── models/
│   │   ├── baseline.py                  # Raw 6,912-d baseline
│   │   ├── pca_model.py                 # Linear PCA representation
│   │   ├── ica_model.py                 # FastICA representation
│   │   └── autoencoder.py               # Nonlinear MLP Autoencoder
│   ├── evaluation/
│   │   ├── metrics.py                   # Reconstruction, neighbourhood, kNN metrics
│   │   └── runner.py                    # LOSO experiment runner
│   └── visualization/
│       └── plotting.py                  # Trade-off curve visualizations
├── tests/
│   ├── test_audit.py                    # Phase 1 tests (18 tests)
│   ├── test_preprocessing.py            # Phase 2/2.1 tests (40 tests)
│   └── test_phase3.py                   # Phase 3 tests (15 tests)
├── run_audit.py                         # Phase 1 audit runner
├── run_preprocessing.py                 # Phase 2 windowing runner
├── run_phase3.py                        # Phase 3 LOSO experiment runner
└── requirements.txt                     # Dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

The PAMAP2 dataset must be downloaded separately and placed under the project directory.
It is excluded from version control. See the [Phase 1 report](reports/phase1_data_audit.md) for details.

**Source:** [UCI ML Repository — PAMAP2](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)

## Running Experiments

### Phase 1: Data Audit
```bash
python run_audit.py
```

### Phase 2: Preprocessing & Window Generation
```bash
python run_preprocessing.py
```

### Phase 3: Representation Learning Experiments
```bash
python run_phase3.py
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Data audit & preprocessing specification | Complete |
| 2 | Preprocessing pipeline implementation | Complete |
| 2.1 | Validation pass & contract verification | Complete |
| 3 | Unsupervised latent representation experiments (PCA, ICA, Autoencoder) | Complete |
| 4 | Downstream classification & deployment optimization | Next |

## License

This project is for research/educational purposes. The PAMAP2 dataset has its own licensing terms from UCI.
