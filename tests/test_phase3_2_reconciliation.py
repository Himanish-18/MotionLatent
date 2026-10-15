"""
Phase 3.2 Metric & Data-Contract Reconciliation Regression Tests.

Validates:
1. True pooled metrics vs. window-weighted fold-score means vs. macro-means.
2. Single-class test fold behavior (robustness on Subject 109).
3. Transient vs. mixed category semantics and exact metadata count invariants.
4. Per-subject and global count reconciliation matching the authoritative data contract.
5. Exact LOSO evaluation-population coverage (14,973 pure non-transient windows, zero duplicates, zero missing).
6. Metric recomputation from saved fold outputs in reports/phase3_loso_results.json.
7. Strict MVD criteria enforcement and embedded deployment constraint verification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.data.loso import ALL_SUBJECTS, load_fold_data
from src.data.preprocessing import load_window_metadata, PURITY_THRESHOLD
from src.evaluation.metrics import (
    compute_macro_mean,
    compute_pooled_classification_metrics,
    compute_weighted_fold_score,
)
from src.evaluation.runner import SUBJECT_WINDOW_COUNTS


AUTHORITATIVE_COUNTS = {
    101: {"pure_nontransient": 1927, "pure_transient": 960, "mixed": 29, "accepted": 2916},
    102: {"pure_nontransient": 2029, "pure_transient": 1391, "mixed": 31, "accepted": 3451},
    103: {"pure_nontransient": 1342, "pure_transient": 600, "mixed": 21, "accepted": 1963},
    104: {"pure_nontransient": 1787, "pure_transient": 746, "mixed": 26, "accepted": 2559},
    105: {"pure_nontransient": 2098, "pure_transient": 781, "mixed": 27, "accepted": 2906},
    106: {"pure_nontransient": 1927, "pure_transient": 857, "mixed": 25, "accepted": 2809},
    107: {"pure_nontransient": 1794, "pure_transient": 613, "mixed": 27, "accepted": 2434},
    108: {"pure_nontransient": 2020, "pure_transient": 1097, "mixed": 31, "accepted": 3148},
    109: {"pure_nontransient": 49, "pure_transient": 12, "mixed": 2, "accepted": 63},
}

TOTAL_ACCEPTED = 22249
TOTAL_PURE_NONTRANSIENT = 14973
TOTAL_PURE_TRANSIENT = 7057
TOTAL_MIXED = 219
TOTAL_REJECTED = 166
TOTAL_CANDIDATE = 22415


class TestMetricDefinitionsAndAggregation:
    """Verifies mathematical distinction between true pooled and weighted fold metrics."""

    def test_weighted_fold_mean_differs_from_true_pooled_on_unequal_distributions(self):
        """Mathematical demonstration that sum(N_k * BA_k) / sum(N_k) != true pooled balanced accuracy.
        
        Fold 1: 100 samples, 2 classes with unequal sizes (80 of class A, 20 of class B)
        Fold 2: 200 samples, 2 classes with unequal sizes (20 of class A, 180 of class B)
        """
        # Controlled synthetic predictions
        y_true_1 = np.array([0] * 80 + [1] * 20)
        y_pred_1 = np.array([0] * 70 + [1] * 10 + [0] * 5 + [1] * 15)  # rec0=70/80=0.875, rec1=15/20=0.75
        ba_1 = balanced_accuracy_score(y_true_1, y_pred_1)  # (0.875 + 0.75) / 2 = 0.8125

        y_true_2 = np.array([0] * 20 + [1] * 180)
        y_pred_2 = np.array([0] * 10 + [1] * 10 + [0] * 36 + [1] * 144)  # rec0=10/20=0.5, rec1=144/180=0.8
        ba_2 = balanced_accuracy_score(y_true_2, y_pred_2)  # (0.5 + 0.8) / 2 = 0.6500

        # 1. Macro-mean across folds
        macro_mean = (ba_1 + ba_2) / 2.0  # 0.73125
        macro_func, _ = compute_macro_mean([ba_1, ba_2])
        assert pytest.approx(macro_func, abs=1e-4) == round(macro_mean, 4)

        # 2. Window-weighted fold-score mean
        weighted_mean = (len(y_true_1) * ba_1 + len(y_true_2) * ba_2) / (len(y_true_1) + len(y_true_2))
        weighted_func = compute_weighted_fold_score([ba_1, ba_2], [len(y_true_1), len(y_true_2)])
        assert pytest.approx(weighted_func, abs=1e-4) == round(weighted_mean, 4)

        # 3. True pooled balanced accuracy
        y_true_pooled = np.concatenate([y_true_1, y_true_2])
        y_pred_pooled = np.concatenate([y_pred_1, y_pred_2])
        true_pooled_ba = balanced_accuracy_score(y_true_pooled, y_pred_pooled)
        pooled_dict = compute_pooled_classification_metrics(y_true_pooled, y_pred_pooled)
        assert pytest.approx(pooled_dict["pooled_balanced_accuracy"], abs=1e-4) == round(true_pooled_ba, 4)

        # Formally assert that weighted mean does NOT equal true pooled balanced accuracy
        assert abs(weighted_mean - true_pooled_ba) > 0.01, (
            f"Expected distinct values: weighted_mean={weighted_mean}, true_pooled={true_pooled_ba}"
        )

    def test_single_class_fold_robustness(self):
        """Verify handling of single-class folds (such as Subject 109 with rope jumping)."""
        # Test fold with only class 24 (49 samples, e.g., 36 correct, 13 misclassified as running)
        y_true_s109 = np.full(49, 24)
        y_pred_s109 = np.array([24] * 36 + [5] * 13)

        # Scikit-learn balanced_accuracy on single class returns its recall
        ba_single = balanced_accuracy_score(y_true_s109, y_pred_s109)
        assert pytest.approx(ba_single, abs=1e-4) == round(36 / 49, 4)

        # Concatenating with multi-class training folds creates full 12-class pooled evaluation
        y_true_other = np.array([1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17] * 10)
        y_pred_other = y_true_other.copy()

        y_true_all = np.concatenate([y_true_other, y_true_s109])
        y_pred_all = np.concatenate([y_pred_other, y_pred_s109])

        pooled_res = compute_pooled_classification_metrics(y_true_all, y_pred_all)
        assert 0.0 < pooled_res["pooled_balanced_accuracy"] <= 1.0
        assert 0.0 < pooled_res["pooled_macro_f1"] <= 1.0


class TestAuthoritativeDataContract:
    """Verifies category semantics, count invariants, and coverage from metadata."""

    @pytest.fixture(scope="class")
    def metadata(self) -> List[Dict]:
        processed_dir = Path("data/processed")
        return load_window_metadata(processed_dir)

    def test_global_count_invariants(self, metadata):
        assert len(metadata) == TOTAL_ACCEPTED

        pure_nontransient = [m for m in metadata if not m["is_mixed_label"] and not m["is_transient"]]
        pure_transient = [m for m in metadata if not m["is_mixed_label"] and m["is_transient"]]
        mixed = [m for m in metadata if m["is_mixed_label"]]

        assert len(pure_nontransient) == TOTAL_PURE_NONTRANSIENT
        assert len(pure_transient) == TOTAL_PURE_TRANSIENT
        assert len(mixed) == TOTAL_MIXED
        assert len(pure_nontransient) + len(pure_transient) + len(mixed) == TOTAL_ACCEPTED

    def test_per_subject_count_reconciliation(self, metadata):
        for sid, expected in AUTHORITATIVE_COUNTS.items():
            s_meta = [m for m in metadata if m["subject_id"] == sid]
            assert len(s_meta) == expected["accepted"], f"S{sid} accepted mismatch"

            s_pnt = [m for m in s_meta if not m["is_mixed_label"] and not m["is_transient"]]
            s_pt = [m for m in s_meta if not m["is_mixed_label"] and m["is_transient"]]
            s_mix = [m for m in s_meta if m["is_mixed_label"]]

            assert len(s_pnt) == expected["pure_nontransient"], f"S{sid} pure non-transient mismatch"
            assert len(s_pt) == expected["pure_transient"], f"S{sid} pure transient mismatch"
            assert len(s_mix) == expected["mixed"], f"S{sid} mixed mismatch"
            assert len(s_pnt) + len(s_pt) + len(s_mix) == expected["accepted"]

    def test_category_semantics_and_threshold_agreement(self, metadata):
        for m in metadata:
            purity = m["activity_purity"]
            is_mixed = m["is_mixed_label"]
            is_transient = m["is_transient"]
            dom = m["dominant_activity"]

            # 1. Purity threshold agreement
            if is_mixed:
                assert purity < PURITY_THRESHOLD, f"Window {m['window_idx']} mixed but purity >= {PURITY_THRESHOLD}"
            else:
                assert purity >= PURITY_THRESHOLD, f"Window {m['window_idx']} pure but purity < {PURITY_THRESHOLD}"

            # 2. Transient label agreement
            if is_transient:
                assert dom == 0, f"Window {m['window_idx']} transient but dominant != 0"
            else:
                assert dom != 0, f"Window {m['window_idx']} non-transient but dominant == 0"

            # 3. Proportions sum to 1.0
            prop_sum = sum(m["activity_proportions"].values())
            assert abs(prop_sum - 1.0) < 1e-4

    def test_loso_evaluation_coverage_and_uniqueness(self, metadata):
        """Verify that pure non-transient windows partition exactly across the 9 test folds."""
        eligible_windows = [m for m in metadata if not m["is_mixed_label"] and not m["is_transient"]]
        assert len(eligible_windows) == TOTAL_PURE_NONTRANSIENT

        # Check subject mapping uniqueness
        subject_fold_assignments = {}
        for m in eligible_windows:
            w_idx = (m["subject_id"], m["start_timestamp"])
            assert w_idx not in subject_fold_assignments, f"Duplicate window record detected: {w_idx}"
            subject_fold_assignments[w_idx] = m["subject_id"]

        assert len(subject_fold_assignments) == TOTAL_PURE_NONTRANSIENT

        # Check per-subject distribution in SUBJECT_WINDOW_COUNTS
        for sid in ALL_SUBJECTS:
            count = sum(1 for s in subject_fold_assignments.values() if s == sid)
            assert count == SUBJECT_WINDOW_COUNTS[sid]


class TestSavedResultRecomputation:
    """Verifies that all reported macro and weighted metrics reproduce from saved records."""

    def test_recompute_all_summary_rows_from_raw_results(self):
        results_path = Path("reports/phase3_loso_results.json")
        summary_path = Path("reports/phase3_summary.json")

        assert results_path.exists(), "phase3_loso_results.json missing"
        assert summary_path.exists(), "phase3_summary.json missing"

        with open(results_path, "r", encoding="utf-8") as f:
            raw_results = json.load(f)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        groups: Dict[tuple, List[Dict]] = {}
        for r in raw_results:
            key = (r["method"], r["latent_dim"])
            groups.setdefault(key, []).append(r)

        for summary_row in summary_data["summary"]:
            key = (summary_row["method"], summary_row["latent_dim"])
            records = sorted(groups[key], key=lambda x: x["subject_id"])
            assert len(records) == 9

            baccs = [r["downstream_knn"]["balanced_accuracy"] for r in records]
            baccs_ex = [r["downstream_knn"]["balanced_accuracy"] for r in records if r["subject_id"] != 109]
            counts = [SUBJECT_WINDOW_COUNTS[r["subject_id"]] for r in records]
            counts_ex = [SUBJECT_WINDOW_COUNTS[r["subject_id"]] for r in records if r["subject_id"] != 109]

            # Recompute macro-mean
            expected_macro_9 = float(np.mean(baccs))
            expected_macro_8 = float(np.mean(baccs_ex))
            assert pytest.approx(summary_row["balanced_accuracy_mean"], abs=1e-4) == round(expected_macro_9, 4)
            assert pytest.approx(summary_row["balanced_accuracy_ex109_mean"], abs=1e-4) == round(expected_macro_8, 4)

            # Recompute weighted fold-score mean
            expected_weighted_9 = sum(n * s for n, s in zip(counts, baccs)) / sum(counts)
            expected_weighted_8 = sum(n * s for n, s in zip(counts_ex, baccs_ex)) / sum(counts_ex)
            assert pytest.approx(summary_row["balanced_accuracy_weighted_mean"], abs=1e-4) == round(expected_weighted_9, 4)
            assert pytest.approx(summary_row["balanced_accuracy_weighted_ex109_mean"], abs=1e-4) == round(expected_weighted_8, 4)


class TestStrictMVDVerification:
    """Verifies strict MVD criteria without silent threshold shifts."""

    @pytest.mark.parametrize(
        "model,bal_acc_ex,trust,comp,size_kb,expected_verdict",
        [
            ("PCA-16", 0.7030, 0.8982, 432.0, 459.0, "FAIL_TOPOLOGY"),
            ("FastICA-16", 0.7104, 0.8960, 432.0, 460.0, "FAIL_TOPOLOGY"),
            ("PCA-32", 0.7065, 0.9455, 216.0, 891.0, "PASS_MCU_1MB"),
            ("Autoencoder-16", 0.7179, 0.9354, 432.0, 13889.06, "PASS_EDGE_SBC_ONLY"),
            ("PCA-4", 0.4539, 0.7550, 1728.0, 135.0, "FAIL_ACCURACY"),
        ],
    )
    def test_strict_mvd_decision_logic(self, model, bal_acc_ex, trust, comp, size_kb, expected_verdict):
        # Stated criteria:
        # 1. Retention: bal_acc_ex > 0.669 (56.9% baseline + 10%)
        # 2. Topology: trust > 0.9000
        # 3. Compression: comp >= 100.0x
        # 4. Storage: MCU <= 1024 KB, Edge SBC > 1024 KB

        if bal_acc_ex <= 0.669:
            verdict = "FAIL_ACCURACY"
        elif trust <= 0.9000:
            verdict = "FAIL_TOPOLOGY"
        elif comp < 100.0:
            verdict = "FAIL_COMPRESSION"
        elif size_kb <= 1024.0:
            verdict = "PASS_MCU_1MB"
        else:
            verdict = "PASS_EDGE_SBC_ONLY"

        assert verdict == expected_verdict
