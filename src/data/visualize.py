"""
PAMAP2 exploratory visualization utilities.

Generates publication-quality plots for the Phase 1 data audit.
Every plot answers a specific data-quality or preprocessing question.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.schema import ACTIVITY_MAP, classify_columns, SENSOR_LOCATIONS


# ===================================================================
# Style configuration
# ===================================================================
def _apply_style() -> None:
    """Apply consistent publication-quality style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.edgecolor": "#333333",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "font.family": "sans-serif",
        "grid.alpha": 0.3,
    })
    sns.set_palette("deep")


# ===================================================================
# 1. Activity distribution
# ===================================================================
def plot_activity_distribution(
    df: pd.DataFrame, save_path: Optional[Path] = None
) -> None:
    """Bar chart of activity counts with names."""
    _apply_style()
    counts = df["activity_id"].value_counts().sort_index()
    labels = [ACTIVITY_MAP.get(int(a), f"ID {a}") for a in counts.index]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(labels, counts.values, color=sns.color_palette("viridis", len(labels)))
    ax.set_xlabel("Number of Observations")
    ax.set_title("PAMAP2 Activity Distribution (Protocol Data)")
    ax.invert_yaxis()

    for bar, val in zip(bars, counts.values):
        ax.text(val + 500, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# 2. Subject distribution
# ===================================================================
def plot_subject_distribution(
    df: pd.DataFrame, save_path: Optional[Path] = None
) -> None:
    """Bar chart of observations per subject."""
    _apply_style()
    counts = df["subject_id"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        [f"S{s}" for s in counts.index],
        counts.values,
        color=sns.color_palette("mako", len(counts)),
    )
    ax.set_xlabel("Subject")
    ax.set_ylabel("Number of Observations")
    ax.set_title("PAMAP2 Observations per Subject")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 500,
                f"{val:,}", ha="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# 3. Missing-value heatmap
# ===================================================================
def plot_missing_heatmap(
    df: pd.DataFrame, save_path: Optional[Path] = None
) -> None:
    """Heatmap of missing-value percentage per subject per sensor group."""
    _apply_style()
    groups = classify_columns()
    sensor_groups = {
        "heart_rate": groups.heart_rate,
        "temperature": groups.temperature,
        "acc_16g": groups.accelerometer_16g,
        "acc_6g": groups.accelerometer_6g,
        "gyroscope": groups.gyroscope,
        "magnetometer": groups.magnetometer,
        "orientation": groups.orientation,
    }

    subjects = sorted(df["subject_id"].unique())
    matrix = []
    group_names = []
    for gname, cols in sensor_groups.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        group_names.append(gname)
        row = []
        for sid in subjects:
            sub_df = df[df["subject_id"] == sid]
            miss_pct = sub_df[present].isnull().mean().mean() * 100
            row.append(round(miss_pct, 2))
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(12, 6))
    data = np.array(matrix)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([f"S{s}" for s in subjects])
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names)
    ax.set_title("Missing Value % by Subject × Sensor Group")

    for i in range(len(group_names)):
        for j in range(len(subjects)):
            ax.text(j, i, f"{data[i, j]:.1f}%",
                    ha="center", va="center", fontsize=8,
                    color="white" if data[i, j] > 50 else "black")

    fig.colorbar(im, ax=ax, label="Missing %")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# 4. Sampling interval distribution
# ===================================================================
def plot_sampling_intervals(
    df: pd.DataFrame, save_path: Optional[Path] = None
) -> None:
    """Histogram of inter-sample time intervals."""
    _apply_style()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    subjects = sorted(df["subject_id"].unique())

    for idx, sid in enumerate(subjects):
        if idx >= 9:
            break
        ax = axes[idx // 3][idx % 3]
        ts = df[df["subject_id"] == sid]["timestamp"].values
        dt = np.diff(ts) * 1000  # to ms
        dt_clipped = dt[dt < 50]  # clip for visibility
        ax.hist(dt_clipped, bins=100, color="#3498db", alpha=0.7, edgecolor="none")
        ax.axvline(10, color="red", linestyle="--", linewidth=1, label="10ms (100Hz)")
        ax.set_title(f"Subject {sid}", fontsize=11)
        ax.set_xlabel("Δt (ms)")
        ax.set_ylabel("Count")
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Sampling Interval Distribution per Subject", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# 5–7. Representative sensor signals
# ===================================================================
def plot_representative_signal(
    df: pd.DataFrame,
    subject_id: int,
    signal_cols: List[str],
    title: str,
    ylabel: str,
    save_path: Optional[Path] = None,
    max_seconds: float = 30.0,
) -> None:
    """Plot a short segment of sensor data for one subject."""
    _apply_style()
    sub = df[df["subject_id"] == subject_id].copy()
    if sub.empty:
        return
    t0 = sub["timestamp"].iloc[0]
    sub = sub[sub["timestamp"] <= t0 + max_seconds]
    t = sub["timestamp"].values - t0

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = sns.color_palette("Set2", len(signal_cols))
    for col, c in zip(signal_cols, colors):
        if col in sub.columns:
            ax.plot(t, sub[col].values, label=col.split("_", 1)[-1],
                    linewidth=0.8, alpha=0.85, color=c)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} — Subject {subject_id} (first {max_seconds}s)")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# 8. Activity timeline
# ===================================================================
def plot_activity_timeline(
    df: pd.DataFrame, subject_id: int, save_path: Optional[Path] = None
) -> None:
    """Timeline of activity labels for one subject."""
    _apply_style()
    sub = df[df["subject_id"] == subject_id].copy()
    if sub.empty:
        return
    t = (sub["timestamp"].values - sub["timestamp"].values[0]) / 60  # minutes

    activities = sorted(sub["activity_id"].unique())
    act_to_idx = {a: i for i, a in enumerate(activities)}
    y = [act_to_idx[a] for a in sub["activity_id"].values]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.scatter(t, y, c=sub["activity_id"].values, cmap="tab20",
               s=1, alpha=0.5, edgecolors="none")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Activity")
    ax.set_yticks(range(len(activities)))
    ax.set_yticklabels([ACTIVITY_MAP.get(int(a), f"ID {a}") for a in activities],
                       fontsize=9)
    ax.set_title(f"Activity Timeline — Subject {subject_id}")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# 9. Sensor magnitude distributions
# ===================================================================
def plot_sensor_magnitude_distributions(
    df: pd.DataFrame, save_path: Optional[Path] = None
) -> None:
    """Box plots of sensor magnitudes across modalities."""
    _apply_style()
    groups = classify_columns()
    modalities = {
        "Acc 16g": groups.accelerometer_16g,
        "Gyroscope": groups.gyroscope,
        "Magnetometer": groups.magnetometer,
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (name, cols) in zip(axes, modalities.items()):
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        data = df[present].sample(min(50000, len(df)), random_state=42)
        data_melted = data.melt(var_name="channel", value_name="value")
        data_melted = data_melted.dropna()
        # Shorten labels
        data_melted["channel"] = data_melted["channel"].apply(
            lambda x: x.replace("_acc_16g", "").replace("_gyro", "").replace("_mag", "")
        )
        sns.boxplot(data=data_melted, x="channel", y="value", ax=ax,
                    fliersize=1, palette="Set3")
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Sensor Value Distributions by Modality", fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Generate all plots
# ===================================================================
def generate_all_plots(
    df: pd.DataFrame,
    output_dir: Path,
    representative_subject: int = 101,
) -> List[Path]:
    """Generate all Phase 1 exploratory plots and return saved paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    groups = classify_columns()

    plots = [
        ("01_activity_distribution.png",
         lambda p: plot_activity_distribution(df, p)),
        ("02_subject_distribution.png",
         lambda p: plot_subject_distribution(df, p)),
        ("03_missing_value_heatmap.png",
         lambda p: plot_missing_heatmap(df, p)),
        ("04_sampling_intervals.png",
         lambda p: plot_sampling_intervals(df, p)),
        ("05_accelerometer_signal.png",
         lambda p: plot_representative_signal(
             df, representative_subject,
             [f"hand_acc_16g_{a}" for a in "xyz"],
             "Hand Accelerometer (16g)", "Acceleration (m/s²)", p)),
        ("06_gyroscope_signal.png",
         lambda p: plot_representative_signal(
             df, representative_subject,
             [f"chest_gyro_{a}" for a in "xyz"],
             "Chest Gyroscope", "Angular velocity (rad/s)", p)),
        ("07_magnetometer_signal.png",
         lambda p: plot_representative_signal(
             df, representative_subject,
             [f"ankle_mag_{a}" for a in "xyz"],
             "Ankle Magnetometer", "Magnetic field (μT)", p)),
        ("08_activity_timeline.png",
         lambda p: plot_activity_timeline(df, representative_subject, p)),
        ("09_sensor_magnitude_distributions.png",
         lambda p: plot_sensor_magnitude_distributions(df, p)),
    ]

    for fname, plot_fn in plots:
        path = output_dir / fname
        print(f"  Generating {fname}...")
        try:
            plot_fn(path)
            saved.append(path)
        except Exception as e:
            print(f"  [WARN] Failed to generate {fname}: {e}")

    return saved
