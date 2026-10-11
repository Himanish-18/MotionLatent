"""
PCA (Principal Component Analysis) Representation -- Phase 3.

Primary linear dimensionality reduction method. Fits orthogonal components
exclusively on training subjects and supports sub-dimension slicing for the
sweep d in [2, 4, 8, 16, 32, 64, 128].
"""

from __future__ import annotations

import time
from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA


class PCARepresentation:
    """PCA representation model with exact sub-dimension slicing."""

    def __init__(self, max_components: int = 128, random_state: int = 42) -> None:
        self.name = "PCA"
        self.max_components = max_components
        self.random_state = random_state
        self.pca = PCA(
            n_components=max_components,
            svd_solver="randomized",
            random_state=random_state,
        )
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None  # (max_components, 6912)
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PCARepresentation":
        """Fit PCA on training data only."""
        self.pca.fit(X)
        self.mean_ = self.pca.mean_.astype(np.float32)
        self.components_ = self.pca.components_.astype(np.float32)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_.astype(np.float64)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray, d: Optional[int] = None) -> np.ndarray:
        """Project X into latent space of dimension d (<= max_components)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform.")
        if d is None:
            d = self.max_components
        if d > self.max_components:
            raise ValueError(f"Requested dimension {d} exceeds max_components {self.max_components}")
        
        # Exact projection: (X - mean) @ components[:d].T
        X_centered = X.astype(np.float32) - self.mean_
        return np.dot(X_centered, self.components_[:d].T).astype(np.float32)

    def inverse_transform(self, Z: np.ndarray, d: Optional[int] = None) -> np.ndarray:
        """Reconstruct original 6,912-d space from latent codes Z."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before inverse_transform.")
        if d is None:
            d = Z.shape[1]
        
        # Exact reconstruction: Z @ components[:d] + mean
        return (np.dot(Z[:, :d], self.components_[:d]) + self.mean_).astype(np.float32)

    def get_explained_variance_ratio(self, d: Optional[int] = None) -> np.ndarray:
        """Returns explained variance ratio for each component up to d."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        if d is None:
            d = self.max_components
        return self.explained_variance_ratio_[:d]

    def get_cumulative_explained_variance(self, d: Optional[int] = None) -> float:
        """Returns cumulative explained variance ratio up to dimension d."""
        return float(np.sum(self.get_explained_variance_ratio(d)))

    def get_param_count(self, d: int) -> int:
        """Number of parameters for d components (projection matrix + mean)."""
        n_features = self.components_.shape[1] if self.components_ is not None else 6912
        return int(n_features * d + n_features)

    def get_model_size_bytes(self, d: int) -> int:
        """Model size in bytes (float32)."""
        return self.get_param_count(d) * 4

    def benchmark_latency(self, X: np.ndarray, d: int, n_repeats: int = 5) -> float:
        """Measure average encoding latency in milliseconds per window."""
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            _ = self.transform(X, d=d)
        dt = (time.perf_counter() - t0) / (n_repeats * len(X))
        return float(dt * 1000.0)  # ms per window
