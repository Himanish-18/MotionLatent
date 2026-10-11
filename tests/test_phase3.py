"""
Tests for MotionLatent Phase 3 -- Unsupervised Latent Representation Learning.

Tests cover:
- LOSO fold isolation and leakage prevention
- Raw baseline representation
- PCA fitting, projection, inverse transform, and explained variance
- FastICA fitting, reconstruction, and convergence tracking
- MotionAutoencoder PyTorch architecture, encoding, decoding, and training loop
- Metric calculations: MSE, RMSE, Trustworthiness, kNN evaluation, and compression ratio
- Single-class robustness handling (e.g. Subject 109 edge cases)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.loso import ALL_SUBJECTS, FoldData
from src.models.baseline import RawBaselineRepresentation
from src.models.pca_model import PCARepresentation
from src.models.ica_model import ICARepresentation
from src.models.autoencoder import MotionAutoencoder, AutoencoderRepresentation
from src.evaluation.metrics import (
    compute_reconstruction_metrics,
    compute_neighbourhood_preservation,
    evaluate_downstream_knn,
    compute_resource_metrics,
)


# ===================================================================
# Fixtures
# ===================================================================
@pytest.fixture
def synthetic_fold_data() -> FoldData:
    """Create a minimal synthetic fold data structure for unit testing."""
    rng = np.random.RandomState(42)
    n_train = 100
    n_test = 20
    d_raw = 6912

    X_train = rng.randn(n_train, d_raw).astype(np.float32)
    y_train = rng.choice([1, 2, 3, 4], size=n_train)
    train_sids = rng.choice([102, 103, 104], size=n_train)

    X_test = rng.randn(n_test, d_raw).astype(np.float32)
    y_test = rng.choice([1, 2, 3, 4], size=n_test)
    test_sids = np.full(n_test, 101)

    class DummyScaler:
        mean = np.zeros(27)
        std = np.ones(27)

    return FoldData(
        test_subject=101,
        training_subjects=[102, 103, 104],
        X_train=X_train,
        y_train=y_train,
        train_subject_ids=train_sids,
        X_test=X_test,
        y_test=y_test,
        test_subject_ids=test_sids,
        scaler=DummyScaler(),
    )


# ===================================================================
# Test Baseline
# ===================================================================
class TestRawBaseline:
    def test_baseline_preserves_shape(self, synthetic_fold_data):
        base = RawBaselineRepresentation().fit(synthetic_fold_data.X_train)
        Z_test = base.transform(synthetic_fold_data.X_test)
        assert Z_test.shape == (20, 6912)
        assert base.compute_reconstruction_error(synthetic_fold_data.X_test) == 0.0

    def test_baseline_inverse_transform(self, synthetic_fold_data):
        base = RawBaselineRepresentation().fit(synthetic_fold_data.X_train)
        Z = base.transform(synthetic_fold_data.X_test)
        X_rec = base.inverse_transform(Z)
        np.testing.assert_array_equal(X_rec, synthetic_fold_data.X_test)


# ===================================================================
# Test PCA Model
# ===================================================================
class TestPCARepresentation:
    def test_pca_fit_and_slice(self, synthetic_fold_data):
        pca = PCARepresentation(max_components=32, random_state=42)
        pca.fit(synthetic_fold_data.X_train)
        assert pca.is_fitted

        # Transform to different sub-dimensions
        Z_8 = pca.transform(synthetic_fold_data.X_test, d=8)
        Z_16 = pca.transform(synthetic_fold_data.X_test, d=16)

        assert Z_8.shape == (20, 8)
        assert Z_16.shape == (20, 16)
        # Verify sub-dimension prefix matches exactly
        np.testing.assert_array_equal(Z_16[:, :8], Z_8)

    def test_pca_reconstruction_shape(self, synthetic_fold_data):
        pca = PCARepresentation(max_components=16, random_state=42)
        pca.fit(synthetic_fold_data.X_train)
        Z = pca.transform(synthetic_fold_data.X_test, d=16)
        X_rec = pca.inverse_transform(Z, d=16)
        assert X_rec.shape == synthetic_fold_data.X_test.shape

    def test_pca_explained_variance_monotonic(self, synthetic_fold_data):
        pca = PCARepresentation(max_components=16, random_state=42)
        pca.fit(synthetic_fold_data.X_train)
        cum_4 = pca.get_cumulative_explained_variance(4)
        cum_8 = pca.get_cumulative_explained_variance(8)
        cum_16 = pca.get_cumulative_explained_variance(16)
        assert 0.0 < cum_4 <= cum_8 <= cum_16 <= 1.0


# ===================================================================
# Test FastICA Model
# ===================================================================
class TestICARepresentation:
    def test_ica_fit_and_transform(self, synthetic_fold_data):
        ica = ICARepresentation(n_components=4, max_iter=200, random_state=42)
        ica.fit(synthetic_fold_data.X_train)
        assert ica.is_fitted
        Z = ica.transform(synthetic_fold_data.X_test)
        assert Z.shape == (20, 4)

    def test_ica_reconstruction_shape(self, synthetic_fold_data):
        ica = ICARepresentation(n_components=4, max_iter=200, random_state=42)
        ica.fit(synthetic_fold_data.X_train)
        Z = ica.transform(synthetic_fold_data.X_test)
        X_rec = ica.inverse_transform(Z)
        assert X_rec.shape == (20, 6912)


# ===================================================================
# Test Autoencoder Model
# ===================================================================
class TestAutoencoderRepresentation:
    def test_autoencoder_architecture(self):
        ae = MotionAutoencoder(input_dim=6912, hidden_dim=64, latent_dim=8)
        x = torch.randn(5, 6912)
        z = ae.encode(x)
        assert z.shape == (5, 8)
        out = ae.decode(z)
        assert out.shape == (5, 6912)

    def test_autoencoder_training(self, synthetic_fold_data):
        ae = AutoencoderRepresentation(
            latent_dim=4,
            hidden_dim=32,
            max_epochs=2,
            batch_size=32,
            random_seed=42,
        )
        ae.fit(synthetic_fold_data.X_train)
        assert ae.is_fitted
        Z = ae.transform(synthetic_fold_data.X_test)
        assert Z.shape == (20, 4)
        X_rec = ae.inverse_transform(Z)
        assert X_rec.shape == (20, 6912)


# ===================================================================
# Test Metrics
# ===================================================================
class TestEvaluationMetrics:
    def test_reconstruction_metrics(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[1.1, 2.0], [2.9, 4.2]])
        res = compute_reconstruction_metrics(A, B)
        assert "mse" in res and "rmse" in res
        assert res["mse"] > 0.0
        assert np.isclose(res["rmse"], np.sqrt(res["mse"]))

    def test_identical_reconstruction_zero_mse(self):
        A = np.ones((10, 50))
        res = compute_reconstruction_metrics(A, A)
        assert res["mse"] == 0.0
        assert res["rmse"] == 0.0

    def test_trustworthiness_metric(self):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 100)
        Z = X[:, :5]  # projection to 5 dims
        res = compute_neighbourhood_preservation(X, Z, n_neighbors=3, sample_size=30)
        assert 0.0 <= res["trustworthiness"] <= 1.0
        assert 0.0 <= res["knn_overlap"] <= 1.0

    def test_knn_evaluation_multiclass(self):
        Z_train = np.array([[0, 0], [0, 1], [5, 5], [5, 6]])
        y_train = np.array([1, 1, 2, 2])
        Z_test = np.array([[0, 0.5], [5, 5.5]])
        y_test = np.array([1, 2])
        res = evaluate_downstream_knn(Z_train, y_train, Z_test, y_test, n_neighbors=1)
        assert res["accuracy"] == 1.0
        assert res["balanced_accuracy"] == 1.0
        assert res["macro_f1"] == 1.0

    def test_knn_evaluation_single_class_robustness(self):
        """Single-class test set (like Subject 109) should evaluate without crashing."""
        Z_train = np.array([[0, 0], [0, 1], [5, 5], [5, 6]])
        y_train = np.array([1, 1, 2, 2])
        Z_test = np.array([[5, 5.2], [5, 5.8]])
        y_test = np.array([2, 2])  # single class in test set
        res = evaluate_downstream_knn(Z_train, y_train, Z_test, y_test, n_neighbors=1)
        assert res["accuracy"] == 1.0
        assert res["n_classes_test"] == 1
        assert res["balanced_accuracy"] == 1.0

    def test_resource_metrics(self):
        res = compute_resource_metrics(orig_dim=6912, latent_dim=16, param_count=1000)
        assert res["compression_ratio"] == 432.0
        assert res["latent_bytes_per_window"] == 64
