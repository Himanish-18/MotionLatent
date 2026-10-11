"""
Trade-Off Curves & Visualization -- Phase 3.

Generates high-resolution figures for accuracy, reconstruction, neighbourhood preservation,
and compression trade-offs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np


def generate_all_tradeoff_plots(
    summary_data: Dict[str, Any],
    output_dir: Path,
) -> List[Path]:
    """Generate and save all publication-grade trade-off plots.

    Parameters
    ----------
    summary_data : dict
        Aggregated results dictionary containing 'summary' and 'raw_results'.
    output_dir : Path
        Directory to save generated image files (reports/figures).

    Returns
    -------
    list of Paths to saved figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    rows = summary_data["summary"]
    raw_results = summary_data.get("raw_results", [])

    # Extract Baseline values
    baseline_row = next((r for r in rows if r["method"] == "Raw Baseline"), None)
    base_bal_acc = baseline_row["balanced_accuracy_mean"] if baseline_row else None
    base_f1 = baseline_row["macro_f1_mean"] if baseline_row else None
    base_bal_acc_ex = baseline_row["balanced_accuracy_ex109_mean"] if baseline_row else None

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        "PCA": "#1f77b4",
        "FastICA": "#ff7f0e",
        "Autoencoder": "#2ca02c",
        "Raw Baseline": "#d62728",
    }
    markers = {
        "PCA": "o",
        "FastICA": "s",
        "Autoencoder": "^",
        "Raw Baseline": "D",
    }

    # -------------------------------------------------------------
    # 1. Balanced Accuracy & Macro-F1 vs Latent Dimension
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    for method in ["PCA", "FastICA", "Autoencoder"]:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        method_rows = sorted(method_rows, key=lambda x: x["latent_dim"])
        dims = [r["latent_dim"] for r in method_rows]
        bal_accs = [r["balanced_accuracy_mean"] for r in method_rows]
        bal_stds = [r["balanced_accuracy_std"] for r in method_rows]
        f1s = [r["macro_f1_mean"] for r in method_rows]
        f1_stds = [r["macro_f1_std"] for r in method_rows]

        ax1.errorbar(
            dims, bal_accs, yerr=bal_stds, label=method,
            color=colors[method], marker=markers[method], capsize=4, linewidth=2, markersize=7
        )
        ax2.errorbar(
            dims, f1s, yerr=f1_stds, label=method,
            color=colors[method], marker=markers[method], capsize=4, linewidth=2, markersize=7
        )

    if base_bal_acc is not None:
        ax1.axhline(base_bal_acc, color=colors["Raw Baseline"], linestyle="--", linewidth=1.8, label=f"Raw Baseline (d=6912: {base_bal_acc:.3f})")
    if base_f1 is not None:
        ax2.axhline(base_f1, color=colors["Raw Baseline"], linestyle="--", linewidth=1.8, label=f"Raw Baseline (d=6912: {base_f1:.3f})")

    for ax, title, ylabel in zip([ax1, ax2], ["Balanced Accuracy vs Dimension", "Macro-F1 vs Dimension"], ["Balanced Accuracy", "Macro-F1"]):
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Latent Dimension (d)", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    p1 = output_dir / "phase3_accuracy_vs_dim.png"
    fig.savefig(p1)
    plt.close(fig)
    saved_files.append(p1)

    # -------------------------------------------------------------
    # 2. Reconstruction MSE & RMSE vs Latent Dimension
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    for method in ["PCA", "FastICA", "Autoencoder"]:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        method_rows = sorted(method_rows, key=lambda x: x["latent_dim"])
        dims = [r["latent_dim"] for r in method_rows]
        mses = [r["mse_mean"] for r in method_rows]
        mse_stds = [r["mse_std"] for r in method_rows]
        rmses = [r["rmse_mean"] for r in method_rows]

        ax1.errorbar(
            dims, mses, yerr=mse_stds, label=method,
            color=colors[method], marker=markers[method], capsize=4, linewidth=2, markersize=7
        )
        ax2.plot(
            dims, rmses, label=method,
            color=colors[method], marker=markers[method], linewidth=2, markersize=7
        )

    for ax, title, ylabel in zip([ax1, ax2], ["Reconstruction MSE vs Dimension", "Reconstruction RMSE vs Dimension"], ["Test Reconstruction MSE", "Test Reconstruction RMSE"]):
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Latent Dimension (d)", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    p2 = output_dir / "phase3_reconstruction_vs_dim.png"
    fig.savefig(p2)
    plt.close(fig)
    saved_files.append(p2)

    # -------------------------------------------------------------
    # 3. PCA Cumulative Explained Variance vs Dimension
    # -------------------------------------------------------------
    pca_rows = sorted([r for r in rows if r["method"] == "PCA"], key=lambda x: x["latent_dim"])
    if pca_rows:
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
        dims = [r["latent_dim"] for r in pca_rows]
        cum_vars = [r["cumulative_explained_variance"] for r in pca_rows]

        ax.plot(dims, cum_vars, color=colors["PCA"], marker="o", linewidth=2.5, markersize=8, label="Cumulative Variance Explained")
        for d, v in zip(dims, cum_vars):
            ax.annotate(f"{v*100:.1f}%", (d, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")

        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Principal Components (d)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Cumulative Explained Variance Ratio", fontsize=11, fontweight="bold")
        ax.set_title("PCA Cumulative Explained Variance vs Dimension", fontsize=13, fontweight="bold")
        ax.axhline(0.80, color="gray", linestyle="--", alpha=0.7, label="80% Threshold")
        ax.axhline(0.90, color="red", linestyle=":", alpha=0.7, label="90% Threshold")
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

        fig.tight_layout()
        p3 = output_dir / "phase3_pca_variance.png"
        fig.savefig(p3)
        plt.close(fig)
        saved_files.append(p3)

    # -------------------------------------------------------------
    # 4. Neighbourhood Preservation vs Dimension
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    for method in ["PCA", "FastICA", "Autoencoder"]:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        method_rows = sorted(method_rows, key=lambda x: x["latent_dim"])
        dims = [r["latent_dim"] for r in method_rows]
        trusts = [r["trustworthiness_mean"] for r in method_rows]
        overlaps = [r["knn_overlap_mean"] for r in method_rows]

        ax1.plot(dims, trusts, label=method, color=colors[method], marker=markers[method], linewidth=2, markersize=7)
        ax2.plot(dims, overlaps, label=method, color=colors[method], marker=markers[method], linewidth=2, markersize=7)

    for ax, title, ylabel in zip([ax1, ax2], ["Trustworthiness vs Dimension", "k-NN Jaccard Overlap vs Dimension"], ["Trustworthiness (k=5)", "k-NN Overlap (k=5)"]):
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Latent Dimension (d)", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    p4 = output_dir / "phase3_neighbourhood_vs_dim.png"
    fig.savefig(p4)
    plt.close(fig)
    saved_files.append(p4)

    # -------------------------------------------------------------
    # 5. Multi-panel Trade-Off Curve (Performance vs Compression Ratio)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    for method in ["PCA", "FastICA", "Autoencoder"]:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        method_rows = sorted(method_rows, key=lambda x: x["compression_ratio"], reverse=True)
        crs = [r["compression_ratio"] for r in method_rows]
        bal_accs = [r["balanced_accuracy_mean"] for r in method_rows]
        mses = [r["mse_mean"] for r in method_rows]

        ax1.plot(crs, bal_accs, label=method, color=colors[method], marker=markers[method], linewidth=2, markersize=7)
        ax2.plot(crs, mses, label=method, color=colors[method], marker=markers[method], linewidth=2, markersize=7)

    if base_bal_acc is not None:
        ax1.scatter([1.0], [base_bal_acc], color=colors["Raw Baseline"], marker=markers["Raw Baseline"], s=100, zorder=5, label=f"Raw Baseline (1.0x)")

    ax1.set_xscale("log", base=10)
    ax1.set_xlabel("Compression Ratio (6912 / d)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Balanced Accuracy", fontsize=11, fontweight="bold")
    ax1.set_title("Balanced Accuracy vs Compression Ratio", fontsize=13, fontweight="bold")
    ax1.legend(frameon=True, fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.set_xscale("log", base=10)
    ax2.set_xlabel("Compression Ratio (6912 / d)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Reconstruction MSE", fontsize=11, fontweight="bold")
    ax2.set_title("Reconstruction MSE vs Compression Ratio", fontsize=13, fontweight="bold")
    ax2.legend(frameon=True, fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    p5 = output_dir / "phase3_tradeoff_curves.png"
    fig.savefig(p5)
    plt.close(fig)
    saved_files.append(p5)

    return saved_files
