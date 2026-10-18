"""
MotionLatent Phase 3.3 -- Raw Baseline Prediction Reconstruction & True Pooled Metrics.

Reconstructs the raw 6,912-dimensional kNN baseline's held-out predictions across all
nine LOSO folds (subjects 101-109) using the exact Phase 3.2 data contract, training scalers,
and kNN configuration (k=5, Euclidean distance).

Computes:
- True pooled balanced accuracy, accuracy, and macro-F1 over concatenated held-out predictions.
- Per-subject balanced accuracy, accuracy, and macro-F1.
- 9-subject macro-mean (+/- std) and 8-subject macro-mean (excl. S109).
- Window-weighted fold-score means.

Saves:
- reports/raw_baseline_predictions.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.data.loso import ALL_SUBJECTS, load_fold_data
from src.data.preprocessing import load_window_metadata
from src.evaluation.metrics import (
    evaluate_downstream_knn,
    compute_pooled_classification_metrics,
    compute_weighted_fold_score,
    compute_macro_mean,
)
from src.evaluation.runner import SUBJECT_WINDOW_COUNTS


def reconstruct_raw_baseline(
    processed_dir: Path | None = None,
    output_path: Path | None = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parent
    if processed_dir is None:
        processed_dir = project_root / "data" / "processed"
    if output_path is None:
        output_path = project_root / "reports" / "raw_baseline_predictions.json"

    all_meta = load_window_metadata(processed_dir)

    all_window_records: List[Dict[str, Any]] = []
    per_subject_results: Dict[str, Dict[str, Any]] = {}
    fold_baccs: List[float] = []
    fold_accs: List[float] = []
    fold_f1s: List[float] = []
    window_counts: List[int] = []

    concat_y_true: List[int] = []
    concat_y_pred: List[int] = []

    total_start = time.time()
    if verbose:
        print("=" * 80)
        print("MotionLatent Phase 3.3: Raw Baseline (6,912-d) Prediction Reconstruction")
        print("=" * 80)

    for idx, sid in enumerate(ALL_SUBJECTS, 1):
        fold_start = time.time()
        if verbose:
            print(f"[{idx}/9] Loading and evaluating held-out Subject {sid}...")

        # Load exact fold data with train-fitted scaler and pure non-transient filter
        fold = load_fold_data(test_subject=sid, processed_dir=processed_dir, pure_only=True, exclude_transient=True)

        # Get corresponding test window metadata in exact order
        s_meta = [
            m for m in all_meta
            if m["subject_id"] == sid and not m["is_mixed_label"] and not m["is_transient"]
        ]
        assert len(s_meta) == len(fold.y_test), (
            f"Subject {sid}: metadata count ({len(s_meta)}) != fold test count ({len(fold.y_test)})"
        )
        assert np.array_equal([m["dominant_activity"] for m in s_meta], fold.y_test), (
            f"Subject {sid}: label ordering divergence between metadata and fold test array"
        )

        # Run raw baseline downstream kNN (k=5, Euclidean, n_jobs=-1)
        res = evaluate_downstream_knn(
            Z_train=fold.X_train,
            y_train=fold.y_train,
            Z_test=fold.X_test,
            y_test=fold.y_test,
            n_neighbors=5,
            return_predictions=True,
        )

        y_true = res["y_true"]
        y_pred = res["y_pred"]
        n_windows = len(y_test_array := fold.y_test)

        bacc = res["balanced_accuracy"]
        acc = res["accuracy"]
        macro_f1 = res["macro_f1"]

        fold_baccs.append(bacc)
        fold_accs.append(acc)
        fold_f1s.append(macro_f1)
        window_counts.append(n_windows)

        concat_y_true.extend(y_true.tolist())
        concat_y_pred.extend(y_pred.tolist())

        per_subject_results[str(sid)] = {
            "subject_id": sid,
            "n_windows": n_windows,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "macro_f1": macro_f1,
            "n_classes_test": res["n_classes_test"],
            "elapsed_seconds": round(time.time() - fold_start, 2),
        }

        if verbose:
            print(f"     S{sid}: N={n_windows}, Bal.Acc={bacc:.4f}, Acc={acc:.4f}, F1={macro_f1:.4f} ({time.time() - fold_start:.2f}s)")

        # Record per-window predictions
        for m, yt, yp in zip(s_meta, y_true, y_pred):
            all_window_records.append({
                "subject_id": sid,
                "window_idx": m["window_idx"],
                "start_timestamp": m["start_timestamp"],
                "end_timestamp": m["end_timestamp"],
                "activity_purity": m["activity_purity"],
                "y_true": int(yt),
                "y_pred": int(yp),
            })

    total_time = time.time() - total_start
    arr_true = np.array(concat_y_true, dtype=int)
    arr_pred = np.array(concat_y_pred, dtype=int)

    assert len(arr_true) == 14973, f"Expected 14,973 total evaluation windows, got {len(arr_true)}"

    # 1. True pooled classification metrics (evaluated once over full 14,973 predictions)
    pooled_metrics = compute_pooled_classification_metrics(arr_true, arr_pred)

    # True pooled metrics excluding S109 (14,924 windows)
    mask_ex109 = np.array([r["subject_id"] != 109 for r in all_window_records], dtype=bool)
    pooled_metrics_ex109 = compute_pooled_classification_metrics(arr_true[mask_ex109], arr_pred[mask_ex109])

    # 2. LOSO macro-means
    macro_bacc_9, std_bacc_9 = compute_macro_mean(fold_baccs)
    macro_acc_9, std_acc_9 = compute_macro_mean(fold_accs)
    macro_f1_9, std_f1_9 = compute_macro_mean(fold_f1s)

    macro_bacc_8, std_bacc_8 = compute_macro_mean(fold_baccs[:-1])
    macro_acc_8, std_acc_8 = compute_macro_mean(fold_accs[:-1])
    macro_f1_8, std_f1_8 = compute_macro_mean(fold_f1s[:-1])

    # 3. Window-weighted fold-score means
    weighted_bacc_9 = compute_weighted_fold_score(fold_baccs, window_counts)
    weighted_acc_9 = compute_weighted_fold_score(fold_accs, window_counts)
    weighted_f1_9 = compute_weighted_fold_score(fold_f1s, window_counts)

    weighted_bacc_8 = compute_weighted_fold_score(fold_baccs[:-1], window_counts[:-1])
    weighted_acc_8 = compute_weighted_fold_score(fold_accs[:-1], window_counts[:-1])
    weighted_f1_8 = compute_weighted_fold_score(fold_f1s[:-1], window_counts[:-1])

    summary_payload = {
        "metadata": {
            "method": "Raw Baseline",
            "feature_dim": 6912,
            "classifier": "KNeighborsClassifier(n_neighbors=5, metric='euclidean')",
            "normalization": "LOSO StandardScaler (fitted on training subjects only)",
            "evaluation_population": "Pure Non-Transient (14,973 windows, dominant act != 0, purity >= 80%)",
            "total_windows": len(arr_true),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_elapsed_seconds": round(total_time, 2),
        },
        "true_pooled_metrics_9_subjects": {
            "pooled_balanced_accuracy": pooled_metrics["pooled_balanced_accuracy"],
            "pooled_accuracy": pooled_metrics["pooled_accuracy"],
            "pooled_macro_f1": pooled_metrics["pooled_macro_f1"],
            "n_windows": len(arr_true),
            "n_classes": int(len(np.unique(arr_true))),
        },
        "true_pooled_metrics_8_subjects_excl_109": {
            "pooled_balanced_accuracy": pooled_metrics_ex109["pooled_balanced_accuracy"],
            "pooled_accuracy": pooled_metrics_ex109["pooled_accuracy"],
            "pooled_macro_f1": pooled_metrics_ex109["pooled_macro_f1"],
            "n_windows": int(mask_ex109.sum()),
            "n_classes": int(len(np.unique(arr_true[mask_ex109]))),
        },
        "loso_macro_means": {
            "balanced_accuracy_mean_9sub": macro_bacc_9,
            "balanced_accuracy_std_9sub": std_bacc_9,
            "accuracy_mean_9sub": macro_acc_9,
            "accuracy_std_9sub": std_acc_9,
            "macro_f1_mean_9sub": macro_f1_9,
            "macro_f1_std_9sub": std_f1_9,
            "balanced_accuracy_mean_8sub_ex109": macro_bacc_8,
            "balanced_accuracy_std_8sub_ex109": std_bacc_8,
            "accuracy_mean_8sub_ex109": macro_acc_8,
            "accuracy_std_8sub_ex109": std_acc_8,
            "macro_f1_mean_8sub_ex109": macro_f1_8,
            "macro_f1_std_8sub_ex109": std_f1_8,
        },
        "weighted_fold_score_means": {
            "balanced_accuracy_weighted_9sub": weighted_bacc_9,
            "accuracy_weighted_9sub": weighted_acc_9,
            "macro_f1_weighted_9sub": weighted_f1_9,
            "balanced_accuracy_weighted_8sub_ex109": weighted_bacc_8,
            "accuracy_weighted_8sub_ex109": weighted_acc_8,
            "macro_f1_weighted_8sub_ex109": weighted_f1_8,
        },
        "per_subject_results": per_subject_results,
        "predictions": all_window_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    if verbose:
        print("\n" + "=" * 80)
        print("RECONSTRUCTION COMPLETE")
        print("=" * 80)
        print(f"Total evaluated windows: {len(arr_true)}")
        print(f"True Pooled Balanced Accuracy (9-sub):  {pooled_metrics['pooled_balanced_accuracy']:.4f}")
        print(f"True Pooled Balanced Accuracy (ex109): {pooled_metrics_ex109['pooled_balanced_accuracy']:.4f}")
        print(f"True Pooled Accuracy (9-sub):           {pooled_metrics['pooled_accuracy']:.4f}")
        print(f"True Pooled Macro-F1 (9-sub):           {pooled_metrics['pooled_macro_f1']:.4f}")
        print(f"LOSO Macro-Mean Bal.Acc (9-sub):        {macro_bacc_9:.4f} +/- {std_bacc_9:.4f}")
        print(f"LOSO Macro-Mean Bal.Acc (ex109):        {macro_bacc_8:.4f} +/- {std_bacc_8:.4f}")
        print(f"Weighted Fold-Score Bal.Acc (9-sub):    {weighted_bacc_9:.4f}")
        print(f"Weighted Fold-Score Bal.Acc (ex109):    {weighted_bacc_8:.4f}")
        print(f"Saved machine-readable predictions: {output_path}")

    return summary_payload


if __name__ == "__main__":
    reconstruct_raw_baseline()
