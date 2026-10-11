"""
FastICA Representation -- Phase 3.

Secondary linear method. Learns non-Gaussian independent components
exclusively on training subjects using randomized PCA pre-whitening for
computational efficiency on 6,912-dimensional inputs. Handles convergence
warnings explicitly and supports linear inverse reconstruction.
"""

from __future__ import annotations

import time
import warnings
from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.exceptions import ConvergenceWarning


class ICARepresentation:
    """FastICA representation model for a specified component count."""

    def __init__(
        self,
        n_components: int = 16,
        max_iter: int = 200,
        tol: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.name = f"FastICA (d={n_components})"
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Randomized PCA whitening followed by FastICA rotation
        self.pca_whiten = PCA(
            n_components=n_components,
            whiten=True,
            svd_solver="randomized",
            random_state=random_state,
        )
        self.ica = FastICA(
            whiten=False,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.is_fitted = False
        self.converged = False
        self.n_iter_ = 0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ICARepresentation":
        """Fit FastICA on training data only, tracking convergence."""
        X_f32 = X.astype(np.float32)
        X_white = self.pca_whiten.fit_transform(X_f32)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            warnings.filterwarnings("ignore", message=".*Ignoring n_components with whiten=False.*")
            self.ica.fit(X_white)
            
            conv_warn = any(issubclass(w.category, ConvergenceWarning) for w in caught_warnings)
            self.converged = not conv_warn
            self.n_iter_ = getattr(self.ica, "n_iter_", self.max_iter)

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data into independent component representation."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform.")
        X_white = self.pca_whiten.transform(X.astype(np.float32))
        return self.ica.transform(X_white).astype(np.float32)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct original 6,912-d observation from independent components."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before inverse_transform.")
        X_white_rec = self.ica.inverse_transform(Z)
        return self.pca_whiten.inverse_transform(X_white_rec).astype(np.float32)

    def get_param_count(self) -> int:
        """Parameter count: PCA components (d*6912) + mean (6912) + ICA unmixing (d*d)."""
        n_features = self.pca_whiten.components_.shape[1] if hasattr(self.pca_whiten, "components_") else 6912
        pca_params = n_features * self.n_components + n_features
        ica_params = self.n_components * self.n_components
        return int(pca_params + ica_params)

    def get_model_size_bytes(self) -> int:
        """Model size in bytes (float32)."""
        return self.get_param_count() * 4

    def benchmark_latency(self, X: np.ndarray, n_repeats: int = 5) -> float:
        """Measure average encoding latency in milliseconds per window."""
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            _ = self.transform(X)
        dt = (time.perf_counter() - t0) / (n_repeats * len(X))
        return float(dt * 1000.0)  # ms per window
