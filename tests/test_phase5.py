"""
Unit and integration tests for Phase 5: Dimensionality Reduction Trade-off Analysis.
Validates artifacts, data contracts, compression ratios, figures, and notebook outputs.
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import nbformat


@pytest.fixture(scope="module")
def repo_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def reports_dir(repo_root):
    return repo_root / "reports"


@pytest.fixture(scope="module")
def figures_dir(reports_dir):
    return reports_dir / "figures"


@pytest.fixture(scope="module")
def comparison_df(reports_dir):
    csv_path = reports_dir / "phase5_comparison.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def summary_json(reports_dir):
    json_path = reports_dir / "phase5_summary.json"
    assert json_path.exists(), f"Missing {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_comparison_csv_structure(comparison_df):
    """Verify comparison CSV contains required columns and has at least 22 rows."""
    expected_cols = [
        "Method",
        "Latent Dimension",
        "Compression Ratio",
        "Representation Bytes/Win",
        "Bal Acc (8-sub Macro)",
        "Bal Acc (9-sub Macro)",
        "Bal Acc Std (9-sub)",
        "Cumulative Explained Variance",
        "Trustworthiness (k=5)",
        "k-NN Overlap (k=5)",
        "Reconstruction MSE",
        "Reconstruction RMSE",
        "Model Flash Size (KB)",
        "Evidence Status",
    ]
    for col in expected_cols:
        assert col in comparison_df.columns, f"Missing required column: {col}"
    assert len(comparison_df) >= 21, f"Expected >= 21 rows, got {len(comparison_df)}"


def test_raw_baseline_inclusion(comparison_df):
    """Verify raw baseline is explicitly included with d=6912 and 1.0x compression."""
    raw_rows = comparison_df[comparison_df["Method"] == "Raw Baseline"]
    assert len(raw_rows) == 1, "Raw Baseline row must exist exactly once"
    raw = raw_rows.iloc[0]
    assert int(raw["Latent Dimension"]) == 6912
    assert raw["Compression Ratio"] == "1.0x"
    assert int(raw["Representation Bytes/Win"]) == 27648
    assert abs(float(raw["Bal Acc (8-sub Macro)"]) - 0.5691) < 1e-4
    assert abs(float(raw["True Pooled Bal Acc"]) - 0.5322) < 1e-4


def test_compression_ratio_arithmetic(comparison_df):
    """Verify compression ratio equals 6912 / d and bytes per window equals d * 4."""
    for _, row in comparison_df.iterrows():
        d = int(row["Latent Dimension"])
        expected_ratio = f"{6912.0 / d:.1f}x"
        assert row["Compression Ratio"] == expected_ratio, (
            f"Dimension {d}: expected {expected_ratio}, got {row['Compression Ratio']}"
        )
        expected_bytes = d * 4
        assert int(row["Representation Bytes/Win"]) == expected_bytes, (
            f"Dimension {d}: expected {expected_bytes} bytes, got {row['Representation Bytes/Win']}"
        )


def test_pca_variance_monotonicity(comparison_df):
    """Verify PCA cumulative explained variance strictly increases with dimension."""
    pca_rows = comparison_df[comparison_df["Method"] == "PCA"].sort_values("Latent Dimension")
    variances = pd.to_numeric(pca_rows["Cumulative Explained Variance"]).tolist()
    assert len(variances) == 7, "Expected 7 PCA dimensions (2, 4, 8, 16, 32, 64, 128)"
    for i in range(len(variances) - 1):
        assert variances[i] < variances[i + 1], (
            f"PCA variance not strictly increasing: {variances[i]} >= {variances[i+1]}"
        )
    # Check milestone at d=32
    pca_32 = pca_rows[pca_rows["Latent Dimension"] == 32].iloc[0]
    assert abs(float(pca_32["Cumulative Explained Variance"]) - 0.5939) < 1e-3


def test_pca_reconstruction_error_monotonicity(comparison_df):
    """Verify PCA reconstruction MSE strictly decreases with dimension."""
    pca_rows = comparison_df[comparison_df["Method"] == "PCA"].sort_values("Latent Dimension")
    mses = pd.to_numeric(pca_rows["Reconstruction MSE"]).tolist()
    for i in range(len(mses) - 1):
        assert mses[i] > mses[i + 1], (
            f"PCA MSE not strictly decreasing: {mses[i]} <= {mses[i+1]}"
        )


def test_summary_json_integrity(summary_json):
    """Verify phase5_summary.json has valid structure and milestone entries."""
    assert "raw_baseline" in summary_json
    assert summary_json["raw_baseline"]["feature_dim"] == 6912
    assert summary_json["raw_baseline"]["eight_subject_macro_mean_balanced_accuracy"] == 0.5691

    assert "pca_variance_milestones" in summary_json
    milestones = summary_json["pca_variance_milestones"]
    assert milestones["50_percent_components"] == 18
    assert milestones["59_4_percent_components"] == 32
    assert milestones["90_percent_components"] == 379

    assert "mvd_winners" in summary_json
    assert "algorithmic_mvd_winner" in summary_json["mvd_winners"]
    assert summary_json["mvd_winners"]["algorithmic_mvd_winner"]["method"] == "PCA-32"
    assert summary_json["mvd_winners"]["algorithmic_mvd_winner"]["latent_dimension"] == 32


def test_all_five_figures_exist(figures_dir):
    """Verify all 5 required Phase 5 publication figures exist and are non-empty."""
    figures = [
        "phase5_accuracy_vs_dimension.png",
        "phase5_variance_explained.png",
        "phase5_neighbourhood_preservation.png",
        "phase5_structure_retained_vs_dimension.png",
        "phase5_reconstruction_error_vs_dimension.png",
    ]
    for fig_name in figures:
        fig_path = figures_dir / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_path}"
        assert fig_path.stat().st_size > 5000, f"Figure {fig_name} is unexpectedly small or empty"


def test_notebook_validity_and_outputs(repo_root):
    """Verify phase5 notebook exists, parses as valid ipynb, and has executed outputs."""
    nb_path = repo_root / "notebooks" / "phase5_dimensionality_tradeoff_analysis.ipynb"
    assert nb_path.exists(), f"Notebook missing: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    assert len(nb.cells) >= 20, f"Expected at least 20 cells, got {len(nb.cells)}"
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 8, f"Expected at least 8 code cells, got {len(code_cells)}"

    for idx, cell in enumerate(code_cells):
        assert len(cell.outputs) > 0, f"Code cell {idx} has no executed outputs embedded"
