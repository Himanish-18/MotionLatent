"""
Phase 4 Publication Trade-Off Plotting -- Phase 4.

Generates high-resolution publication figures:
1. phase4_accuracy_vs_footprint.png: Balanced Accuracy vs Model Size (Flash KB, log-scale).
2. phase4_fp32_vs_int8.png: FP32 vs INT8 comparison (Accuracy, Footprint, Latency).
3. phase4_s108_diagnostic.png: Subject 108 error distribution and sensor distribution shifts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_vs_footprint(
    summary_data: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Plot downstream Balanced Accuracy vs. Model Footprint (KB) on a log scale."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Style definitions
    colors = {
        "Raw Baseline": "#7f7f7f",
        "PCA": "#1f77b4",
        "Autoencoder": "#ff7f0e",
        "Conv1D-AE": "#2ca02c",
        "DSC-AE": "#d62728",
        "DSC-AE (INT8)": "#9467bd",
    }
    markers = {
        "Raw Baseline": "X",
        "PCA": "o",
        "Autoencoder": "s",
        "Conv1D-AE": "^",
        "DSC-AE": "D",
        "DSC-AE (INT8)": "v",
    }

    for item in summary_data:
        m = item["method"]
        d = item.get("latent_dim", "")
        acc = item["balanced_accuracy_mean"]
        size_kb = max(item["model_size_kb"], 0.1)  # avoid log(0) for baseline

        color = colors.get(m, "#333333")
        marker = markers.get(m, "o")
        label = f"{m} (d={d})" if d else m

        ax.scatter(size_kb, acc * 100.0, color=color, marker=marker, s=120, edgecolors="k", zorder=4)
        ax.annotate(
            label,
            (size_kb, acc * 100.0),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

    # Reference threshold lines
    ax.axhline(66.9, color="#e41a1c", linestyle="--", alpha=0.8, label="Strict MVD Threshold (66.9%)")
    ax.axvline(1024.0, color="#4daf4a", linestyle=":", alpha=0.8, label="1 MB MCU Flash Limit")
    ax.axvline(256.0, color="#377eb8", linestyle=":", alpha=0.8, label="256 KB Constrained Flash")

    ax.set_xscale("log")
    ax.set_xlabel("Model Encoder Size in Flash (KB, log scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("9-Subject Macro Balanced Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("MotionLatent Phase 4: Accuracy vs. Deployment Memory Footprint", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_fp32_vs_int8(
    quant_results: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Compare FP32 and INT8 quantization impact across metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

    # 1. Balanced Accuracy comparison
    labels = ["FP32", "INT8 (PTQ)"]
    baccs = [
        quant_results["downstream_fp32"]["balanced_accuracy"] * 100.0,
        quant_results["downstream_quantized"]["balanced_accuracy"] * 100.0,
    ]
    axes[0].bar(labels, baccs, color=["#1f77b4", "#ff7f0e"], width=0.5, edgecolor="k")
    axes[0].set_ylabel("Balanced Accuracy (%)", fontweight="bold")
    axes[0].set_title("Downstream Accuracy", fontweight="bold")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(baccs):
        axes[0].text(i, v + 1.5, f"{v:.2f}%", ha="center", fontweight="bold")

    # 2. Flash Size comparison
    sizes = [quant_results["fp32_size_kb"], quant_results["quantized_size_kb"]]
    axes[1].bar(labels, sizes, color=["#2ca02c", "#d62728"], width=0.5, edgecolor="k")
    axes[1].set_ylabel("Encoder Size (KB)", fontweight="bold")
    axes[1].set_title("Flash Memory Footprint", fontweight="bold")
    for i, v in enumerate(sizes):
        axes[1].text(i, v + (max(sizes)*0.03), f"{v:.2f} KB", ha="center", fontweight="bold")

    # 3. Latency comparison
    lats = [quant_results["fp32_latency_ms"], quant_results["quantized_latency_ms"]]
    axes[2].bar(labels, lats, color=["#9467bd", "#8c564b"], width=0.5, edgecolor="k")
    axes[2].set_ylabel("Latency (ms / window)", fontweight="bold")
    axes[2].set_title("Inference Latency", fontweight="bold")
    for i, v in enumerate(lats):
        axes[2].text(i, v + (max(lats)*0.03), f"{v:.4f} ms", ha="center", fontweight="bold")

    plt.suptitle(
        f"Numerical Quantization Impact: {quant_results.get('model_name', 'Temporal AE')}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_s108_diagnostic(
    diag_results: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Plot Subject 108 per-class recalls and top shifted sensor channels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    # Subplot 1: Per-class recall on S108
    per_class = diag_results["confusion_analysis"]["per_class_performance"]
    c_names = list(per_class.keys())
    recalls = [per_class[k]["recall"] * 100.0 for k in c_names]
    y_pos = np.arange(len(c_names))

    ax1.barh(y_pos, recalls, color="#1f77b4", edgecolor="k", height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(c_names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Recall on Subject 108 (%)", fontweight="bold")
    ax1.set_title("Subject 108 Activity Recall (PCA-32)", fontweight="bold")
    ax1.axvline(50.0, color="gray", linestyle="--", alpha=0.5)

    for i, v in enumerate(recalls):
        ax1.text(v + 1.0, i, f"{v:.1f}%", va="center", fontsize=8)

    # Subplot 2: Top sensor channel distribution shifts (Z-score difference)
    top_channels = diag_results["sensor_distribution_shift"]["top_shifted_channels"][:10]
    ch_names = [c["channel_name"] for c in top_channels]
    z_shifts = [c["z_shift"] for c in top_channels]
    y_pos2 = np.arange(len(ch_names))

    ax2.barh(y_pos2, z_shifts, color="#d62728", edgecolor="k", height=0.6)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(ch_names, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean Absolute Shift (Standard Deviations)", fontweight="bold")
    ax2.set_title("Top 10 Sensor Distribution Shifts for S108", fontweight="bold")

    for i, v in enumerate(z_shifts):
        ax2.text(v + 0.05, i, f"{v:.2f} sigma", va="center", fontsize=8)

    plt.suptitle("MotionLatent Phase 4D: Subject 108 Diagnostic Decomposition", fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path
