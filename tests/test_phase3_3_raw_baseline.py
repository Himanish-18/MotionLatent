"""
Phase 3.3 Raw Baseline Validation & True Pooled Metric Tests.

Validates:
1. Exactly 14,973 eligible evaluation records in reconstructed raw baseline predictions.
2. Exactly-once held-out coverage for every eligible window across all 9 LOSO folds.
3. No subject leakage in training sets or scaler fitting.
4. Deterministic reproduction of fold-level metrics matching saved Phase 3 results.
5. Correct calculation and differentiation between macro-mean, weighted fold-score mean, and true pooled metrics.
6. Correct handling of Subject 109 single-class test fold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.data.loso import ALL_SUBJECTS, load_fold_scaler
from src.data.preprocessing import load_window_metadata
from src.evaluation.metrics import (
    compute_macro_mean,
    compute_pooled_classification_metrics,
    compute_weighted_fold_score,
)
from src.evaluation.runner import SUBJECT_WINDOW_COUNTS


TOTAL_PURE_NONTRANSIENT = 14973


@pytest.fixture(scope="module")
def baseline_payload() -> Dict:
    path = Path("reports/raw_baseline_predictions.json")
    assert path.exists(), f"Raw baseline predictions artifact missing: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def saved_loso_results() -> List[Dict]:
    path = Path("reports/phase3_loso_results.json")
    assert path.exists(), f"Saved LOSO results missing: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def window_metadata() -> List[Dict]:
    processed_dir = Path("data/processed")
    return load_window_metadata(processed_dir)


class TestRawBaselinePopulationCoverage:
    """Verifies that all 14,973 pure non-transient windows are evaluated exactly once."""

    def test_total_prediction_record_count(self, baseline_payload):
        preds = baseline_payload["predictions"]
        assert len(preds) == TOTAL_PURE_NONTRANSIENT, (
            f"Expected {TOTAL_PURE_NONTRANSIENT} evaluation records, got {len(preds)}"
        )

    def test_exactly_once_held_out_coverage(self, baseline_payload, window_metadata):
        """Cross-check each prediction record against metadata to verify 1-to-1 bijection."""
        eligible_meta = [
            m for m in window_metadata
            if not m["is_mixed_label"] and not m["is_transient"]
        ]
        assert len(eligible_meta) == TOTAL_PURE_NONTRANSIENT

        meta_keys = {(m["subject_id"], m["window_idx"], round(m["start_timestamp"], 2)) for m in eligible_meta}
        pred_keys = {(p["subject_id"], p["window_idx"], round(p["start_timestamp"], 2)) for p in baseline_payload["predictions"]}

        assert len(pred_keys) == TOTAL_PURE_NONTRANSIENT, "Duplicate predictions detected across folds"
        assert meta_keys == pred_keys, "Set of evaluated windows diverges from eligible metadata windows"

    def test_per_subject_window_counts(self, baseline_payload):
        preds = baseline_payload["predictions"]
        counts: Dict[int, int] = {}
        for p in preds:
            sid = p["subject_id"]
            counts[sid] = counts.get(sid, 0) + 1

        assert counts == SUBJECT_WINDOW_COUNTS, (
            f"Per-subject counts mismatch: got {counts}, expected {SUBJECT_WINDOW_COUNTS}"
        )


class TestFoldIsolationAndLeakage:
    """Verifies that training sets and scalers strictly isolate the test subject."""

    def test_scalers_exclude_test_subject(self):
        processed_dir = Path("data/processed")
        for sid in ALL_SUBJECTS:
            scaler = load_fold_scaler(processed_dir, sid)
            assert scaler.test_subject == sid
            assert sid not in scaler.training_subjects
            assert len(scaler.training_subjects) == 8


class TestMetricReconstructionFidelity:
    """Verifies that reconstructed fold metrics reproduce saved Phase 3 results deterministically."""

    def test_reconstructed_fold_metrics_match_saved_phase3(self, baseline_payload, saved_loso_results):
        raw_saved = [r for r in saved_loso_results if r["method"] == "Raw Baseline"]
        assert len(raw_saved) == 9

        per_sub = baseline_payload["per_subject_results"]

        for saved_record in raw_saved:
            sid = saved_record["subject_id"]
            saved_knn = saved_record["downstream_knn"]
            recon_knn = per_sub[str(sid)]

            assert pytest.approx(recon_knn["accuracy"], abs=1e-4) == saved_knn["accuracy"], (
                f"S{sid} accuracy divergence: recon={recon_knn['accuracy']}, saved={saved_knn['accuracy']}"
            )
            assert pytest.approx(recon_knn["balanced_accuracy"], abs=1e-4) == saved_knn["balanced_accuracy"], (
                f"S{sid} balanced accuracy divergence: recon={recon_knn['balanced_accuracy']}, saved={saved_knn['balanced_accuracy']}"
            )
            assert pytest.approx(recon_knn["macro_f1"], abs=1e-4) == saved_knn["macro_f1"], (
                f"S{sid} macro-F1 divergence: recon={recon_knn['macro_f1']}, saved={saved_knn['macro_f1']}"
            )

    def test_macro_means_and_weighted_means(self, baseline_payload):
        per_sub = baseline_payload["per_subject_results"]
        baccs_9 = [per_sub[str(s)]["balanced_accuracy"] for s in ALL_SUBJECTS]
        counts_9 = [SUBJECT_WINDOW_COUNTS[s] for s in ALL_SUBJECTS]

        # 9-subject macro-mean
        macro_mean, macro_std = compute_macro_mean(baccs_9)
        assert pytest.approx(baseline_payload["loso_macro_means"]["balanced_accuracy_mean_9sub"], abs=1e-4) == round(macro_mean, 4)
        assert pytest.approx(baseline_payload["loso_macro_means"]["balanced_accuracy_std_9sub"], abs=1e-4) == round(macro_std, 4)
        assert round(macro_mean, 4) == 0.5058

        # 8-subject macro-mean (excl S109)
        macro_mean_8, _ = compute_macro_mean(baccs_9[:-1])
        assert pytest.approx(baseline_payload["loso_macro_means"]["balanced_accuracy_mean_8sub_ex109"], abs=1e-4) == round(macro_mean_8, 4)
        assert round(macro_mean_8, 4) == 0.5691

        # Weighted fold-score means
        w_mean_9 = compute_weighted_fold_score(baccs_9, counts_9)
        w_mean_8 = compute_weighted_fold_score(baccs_9[:-1], counts_9[:-1])
        assert pytest.approx(baseline_payload["weighted_fold_score_means"]["balanced_accuracy_weighted_9sub"], abs=1e-4) == round(w_mean_9, 4)
        assert round(w_mean_9, 4) == 0.5597
        assert round(w_mean_8, 4) == 0.5615


class TestTruePooledMetricCalculations:
    """Verifies true pooled metrics computed over concatenated predictions."""

    def test_true_pooled_metrics_from_concatenated_predictions(self, baseline_payload):
        preds = baseline_payload["predictions"]
        y_true = np.array([p["y_true"] for p in preds], dtype=int)
        y_pred = np.array([p["y_pred"] for p in preds], dtype=int)

        assert len(y_true) == TOTAL_PURE_NONTRANSIENT
        assert len(np.unique(y_true)) == 12, "Evaluation population must span all 12 active activities"

        # Direct scikit-learn computation
        direct_acc = float(accuracy_score(y_true, y_pred))
        direct_bacc = float(balanced_accuracy_score(y_true, y_pred))
        direct_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        pooled_9 = baseline_payload["true_pooled_metrics_9_subjects"]
        assert pytest.approx(pooled_9["pooled_accuracy"], abs=1e-4) == round(direct_acc, 4)
        assert pytest.approx(pooled_9["pooled_balanced_accuracy"], abs=1e-4) == round(direct_bacc, 4)
        assert pytest.approx(pooled_9["pooled_macro_f1"], abs=1e-4) == round(direct_f1, 4)

        assert round(direct_bacc, 4) == 0.5322
        assert round(direct_acc, 4) == 0.6022
        assert round(direct_f1, 4) == 0.5157

    def test_pooled_vs_weighted_mean_distinction(self, baseline_payload):
        """Verifies that true pooled balanced accuracy is distinct from weighted fold-score mean."""
        pooled_bacc = baseline_payload["true_pooled_metrics_9_subjects"]["pooled_balanced_accuracy"]
        weighted_bacc = baseline_payload["weighted_fold_score_means"]["balanced_accuracy_weighted_9sub"]

        # True pooled balanced accuracy is 0.5322; weighted fold-score mean is 0.5597
        assert pooled_bacc != weighted_bacc
        assert abs(pooled_bacc - weighted_bacc) > 0.025, (
            f"Expected significant difference: pooled={pooled_bacc}, weighted={weighted_bacc}"
        )

    def test_s109_single_class_fold_properties(self, baseline_payload):
        s109_preds = [p for p in baseline_payload["predictions"] if p["subject_id"] == 109]
        assert len(s109_preds) == 49
        assert all(p["y_true"] == 24 for p in s109_preds)

        s109_res = baseline_payload["per_subject_results"]["109"]
        assert s109_res["n_classes_test"] == 1
        assert s109_res["accuracy"] == 0.0
        assert s109_res["balanced_accuracy"] == 0.0
        assert s109_res["macro_f1"] == 0.0
