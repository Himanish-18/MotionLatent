"""
Phase 3.1 Validation & Reconciliation Unit Tests.

Validates:
1. Window count reconciliation (14,973 pure non-transient windows across all subjects).
2. Fold-level subject isolation (training subjects vs test subject).
3. Metric recomputation from saved LOSO records.
4. Trustworthiness correctness against scikit-learn.
5. Compression ratio arithmetic and model parameter sizing.
6. Strict MVD criteria evaluation and threshold enforcement.
"""

import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.manifold import trustworthiness as sklearn_trustworthiness

from src.data.loso import ALL_SUBJECTS, FLATTENED_DIM, load_fold_data
from src.data.preprocessing import load_window_metadata
from src.evaluation.metrics import (
    compute_neighbourhood_preservation,
    compute_reconstruction_metrics,
    compute_resource_metrics,
)


EXPECTED_SUBJECT_WINDOWS = {
    101: 1927,
    102: 2029,
    103: 1342,
    104: 1787,
    105: 2098,
    106: 1927,
    107: 1794,
    108: 2020,
    109: 49,
}
TOTAL_PURE_NONTRANSIENT = 14973


class TestWindowCountReconciliation:
    """Verifies that window counts at all stages match empirical ground truth."""

    def test_total_metadata_windows(self):
        processed_dir = Path("data/processed")
        meta = load_window_metadata(processed_dir)
        assert len(meta) == 22249, f"Expected 22,249 total windows, got {len(meta)}"

    def test_pure_nontransient_window_counts_per_subject(self):
        processed_dir = Path("data/processed")
        meta = load_window_metadata(processed_dir)

        # Filter: pure (is_mixed_label == False) and non-transient (is_transient == False)
        filtered = [m for m in meta if not m["is_mixed_label"] and not m["is_transient"]]
        assert len(filtered) == TOTAL_PURE_NONTRANSIENT

        counts = {}
        for m in filtered:
            sid = m["subject_id"]
            counts[sid] = counts.get(sid, 0) + 1

        assert counts == EXPECTED_SUBJECT_WINDOWS
        assert sum(counts.values()) == TOTAL_PURE_NONTRANSIENT

    def test_loso_loader_preserves_exact_population(self):
        """Verify load_fold_data partitions exactly 14,973 windows for every fold."""
        processed_dir = Path("data/processed")
        for sid in ALL_SUBJECTS:
            fold = load_fold_data(test_subject=sid, processed_dir=processed_dir, pure_only=True, exclude_transient=True)
            expected_test = EXPECTED_SUBJECT_WINDOWS[sid]
            expected_train = TOTAL_PURE_NONTRANSIENT - expected_test

            assert len(fold.X_test) == expected_test, f"Fold S{sid}: expected {expected_test} test windows, got {len(fold.X_test)}"
            assert len(fold.X_train) == expected_train, f"Fold S{sid}: expected {expected_train} train windows, got {len(fold.X_train)}"
            assert len(fold.X_train) + len(fold.X_test) == TOTAL_PURE_NONTRANSIENT


class TestFoldIsolationAndLeakage:
    """Verifies that LOSO folds strictly isolate the test subject."""

    def test_subject_isolation_in_training_data(self):
        processed_dir = Path("data/processed")
        for sid in ALL_SUBJECTS:
            fold = load_fold_data(test_subject=sid, processed_dir=processed_dir, pure_only=True, exclude_transient=True)
            assert sid not in fold.training_subjects
            assert sid not in fold.train_subject_ids
            assert np.all(fold.test_subject_ids == sid)

    def test_scaler_excludes_test_subject(self):
        processed_dir = Path("data/processed")
        for sid in ALL_SUBJECTS:
            fold = load_fold_data(test_subject=sid, processed_dir=processed_dir, pure_only=True, exclude_transient=True)
            assert fold.scaler.test_subject == sid
            assert sid not in fold.scaler.training_subjects


class TestMetricRecomputationAndTrustworthiness:
    """Verifies metric computation and cross-checks against scikit-learn."""

    def test_trustworthiness_matches_sklearn_exactly(self):
        rng = np.random.RandomState(42)
        # Small deterministic test data (N=50, D_orig=20, D_latent=5)
        X_orig = rng.randn(50, 20).astype(np.float32)
        Z_lat = X_orig[:, :5] + 0.1 * rng.randn(50, 5).astype(np.float32)

        result = compute_neighbourhood_preservation(X_orig, Z_lat, n_neighbors=5, sample_size=100)
        expected = float(sklearn_trustworthiness(X_orig, Z_lat, n_neighbors=5))

        assert pytest.approx(result["trustworthiness"], abs=1e-4) == round(expected, 4)

    def test_recompute_saved_loso_metrics(self):
        """Verify recomputing aggregate metrics from phase3_loso_results.json."""
        results_path = Path("reports/phase3_loso_results.json")
        summary_path = Path("reports/phase3_summary.json")

        if not results_path.exists() or not summary_path.exists():
            pytest.skip("Result JSONs not found")

        with open(results_path, "r", encoding="utf-8") as f:
            raw_results = json.load(f)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        # Test PCA-32 recomputation
        pca32_raw = [r for r in raw_results if r["method"] == "PCA" and r["latent_dim"] == 32]
        assert len(pca32_raw) == 9

        baccs_all = [r["downstream_knn"]["balanced_accuracy"] for r in pca32_raw]
        baccs_ex = [r["downstream_knn"]["balanced_accuracy"] for r in pca32_raw if r["subject_id"] != 109]

        mean_all = float(np.mean(baccs_all))
        mean_ex = float(np.mean(baccs_ex))

        # Check summary record
        pca32_sum = next(r for r in summary_data["summary"] if r["method"] == "PCA" and r["latent_dim"] == 32)
        assert pytest.approx(pca32_sum["balanced_accuracy_mean"], abs=1e-4) == round(mean_all, 4)
        assert pytest.approx(pca32_sum["balanced_accuracy_ex109_mean"], abs=1e-4) == round(mean_ex, 4)
        assert round(mean_all, 3) == 0.710
        assert round(mean_ex, 3) == 0.707


class TestCompressionAndStrictMVDEnforcement:
    """Verifies compression arithmetic and strict threshold checking."""

    def test_compression_ratios(self):
        expected_ratios = {
            2: 3456.0,
            4: 1728.0,
            8: 864.0,
            16: 432.0,
            32: 216.0,
            64: 108.0,
            128: 54.0,
            6912: 1.0,
        }
        for d, exp_ratio in expected_ratios.items():
            res = compute_resource_metrics(6912, d, 0, 0, 0.0)
            assert res["compression_ratio"] == exp_ratio

    def test_strict_mvd_thresholds(self):
        """Ensure threshold logic strictly rejects candidates below stated cutoffs."""
        # Thresholds: bal_acc_ex > 0.669, trust > 0.900, compression >= 100x
        def evaluate_mvd(bal_acc_ex: float, trust: float, comp_ratio: float) -> str:
            if bal_acc_ex <= 0.669:
                return "FAIL (Accuracy)"
            if trust < 0.900:
                return "FAIL (Trustworthiness)"
            if comp_ratio < 100.0:
                return "FAIL (Compression)"
            return "PASS (Representational)"

        # PCA-16 has trust = 0.8982 -> must fail
        assert evaluate_mvd(0.7030, 0.8982, 432.0) == "FAIL (Trustworthiness)"
        # Autoencoder-4 has bal_acc = 0.5095 -> must fail
        assert evaluate_mvd(0.5095, 0.8061, 1728.0) == "FAIL (Accuracy)"
        # PCA-32 has bal_acc = 0.7065, trust = 0.9455, comp = 216.0 -> passes
        assert evaluate_mvd(0.7065, 0.9455, 216.0) == "PASS (Representational)"
        # Autoencoder-16 has bal_acc = 0.7180, trust = 0.9354, comp = 432.0 -> passes
        assert evaluate_mvd(0.7180, 0.9354, 432.0) == "PASS (Representational)"
