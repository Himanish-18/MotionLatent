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
│   ├── phase1_data_audit.md     # Phase 1 audit report
│   ├── audit_output.json        # Machine-readable audit data
│   └── figures/                 # Generated plots
├── src/
│   └── data/
│       ├── discovery.py         # Dataset discovery utilities
│       ├── schema.py            # PAMAP2 schema definitions
│       ├── audit.py             # Audit functions
│       └── visualize.py         # Visualization utilities
├── tests/
│   └── test_audit.py            # Unit tests
├── run_audit.py                 # Main audit runner
└── requirements.txt             # Dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

The PAMAP2 dataset must be downloaded separately and placed under the project directory.
It is excluded from version control. See the [Phase 1 report](reports/phase1_data_audit.md) for details.

**Source:** [UCI ML Repository — PAMAP2](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)

## Running the Audit

```bash
python run_audit.py
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1     | Data audit & preprocessing specification | Complete |
| 2     | Preprocessing pipeline implementation | Planned |
| 3     | PCA / ICA / Autoencoder experiments | Planned |
| 4     | Evaluation & comparison | Planned |

## License

This project is for research/educational purposes. The PAMAP2 dataset has its own licensing terms from UCI.
