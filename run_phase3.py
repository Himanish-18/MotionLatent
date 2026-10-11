"""
MotionLatent Phase 3 Experiment Runner.

Executes Leave-One-Subject-Out (LOSO) cross-validation across subjects 101-109
for:
1. Raw Baseline (d=6912)
2. PCA (d in [2, 4, 8, 16, 32, 64, 128])
3. FastICA (d in [2, 4, 8, 16, 32, 64])
4. Autoencoder (d in [4, 16, 64])

Outputs:
- reports/phase3_loso_results.json
- reports/phase3_summary.json
- reports/figures/*.png
"""

import sys
import json
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loso import ALL_SUBJECTS, load_fold_data
from src.evaluation.runner import (
    PCA_DIMS,
    ICA_DIMS,
    AE_DIMS,
    run_single_fold_experiments,
    aggregate_loso_results,
)
from src.visualization.plotting import generate_all_tradeoff_plots


def main():
    print("=" * 80)
    print("MotionLatent -- Phase 3 Unsupervised Representation Learning Experiments")
    print("=" * 80)
    print(f"Subjects: {ALL_SUBJECTS}")
    print(f"PCA dimensions: {PCA_DIMS}")
    print(f"ICA dimensions: {ICA_DIMS}")
    print(f"Autoencoder dimensions: {AE_DIMS}")
    print("Protocol: Strict LOSO, train-only normalization & fitting, pure non-transient evaluation.")
    print("=" * 80)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_fold_results = []
    total_start = time.time()

    for i, sid in enumerate(ALL_SUBJECTS, 1):
        print(f"\n>>> [{i}/{len(ALL_SUBJECTS)}] Processing LOSO Fold: Test Subject {sid} <<<")
        fold_start = time.time()
        
        fold = load_fold_data(test_subject=sid, pure_only=True, exclude_transient=True)
        print(f"  Loaded Fold S{sid}: Train={fold.X_train.shape[0]} windows, Test={fold.X_test.shape[0]} windows, {len(fold.unique_train_classes)} train classes")

        fold_results = run_single_fold_experiments(
            fold=fold,
            pca_dims=PCA_DIMS,
            ica_dims=ICA_DIMS,
            ae_dims=AE_DIMS,
            verbose=True,
        )
        all_fold_results.extend(fold_results)
        print(f"  Completed Fold S{sid} in {time.time() - fold_start:.2f}s ({len(fold_results)} experiment records)")

        # Save partial progress
        partial_path = reports_dir / "phase3_partial_results.json"
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(all_fold_results, f, indent=2)

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"All 9 LOSO folds completed in {total_time:.2f}s ({total_time / 60:.2f} min)")
    print("=" * 80)

    # Aggregate results
    print("\nAggregating results across subjects...")
    summary_data = aggregate_loso_results(all_fold_results)

    # Save machine-readable outputs
    loso_results_path = reports_dir / "phase3_loso_results.json"
    summary_path = reports_dir / "phase3_summary.json"

    with open(loso_results_path, "w", encoding="utf-8") as f:
        json.dump(all_fold_results, f, indent=2)
    print(f"Saved full fold results: {loso_results_path} ({len(all_fold_results)} records)")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved aggregated summary: {summary_path}")

    # Generate figures
    print("\nGenerating publication trade-off plots...")
    saved_plots = generate_all_tradeoff_plots(summary_data, figures_dir)
    for p in saved_plots:
        print(f"  Saved plot: {p}")

    # Print summary table
    print("\n" + "=" * 105)
    print(f"{'Method':<14} | {'d':<5} | {'Comp.Ratio':<10} | {'Bal.Acc (9-sub)':<16} | {'Macro-F1 (9-sub)':<16} | {'Bal.Acc (ex109)':<16} | {'MSE':<10}")
    print("-" * 105)
    for row in summary_data["summary"]:
        m = row["method"]
        d = row["latent_dim"]
        cr = f"{row['compression_ratio']:.1f}x"
        ba9 = f"{row['balanced_accuracy_mean']:.3f} +/- {row['balanced_accuracy_std']:.3f}"
        f19 = f"{row['macro_f1_mean']:.3f} +/- {row['macro_f1_std']:.3f}"
        ba_ex = f"{row['balanced_accuracy_ex109_mean']:.3f}"
        mse = f"{row['mse_mean']:.4f}"
        print(f"{m:<14} | {d:<5} | {cr:<10} | {ba9:<16} | {f19:<16} | {ba_ex:<16} | {mse:<10}")
    print("=" * 105)


if __name__ == "__main__":
    main()
