"""
Raw Input Baseline Representation -- Phase 3.

Maintains the full 6,912-dimensional flattened window representation without
compression (compression ratio = 1.0x) as the mandatory evaluation benchmark.
"""

from __future__ import annotations

import time
from typing import Optional
import numpy as np


class RawBaselineRepresentation:
    """Identity baseline representation keeping all 6,912 features."""

    def __init__(self) -> None:
        self.name = "Raw Baseline"
        self.latent_dim = 6912
        self.n_features_in_ = 6912
        self.param_count = 0
        self.model_size_bytes = 0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RawBaselineRepresentation":
        """Fits baseline (no-op identity)."""
        self.n_features_in_ = X.shape[1]
        self.latent_dim = self.n_features_in_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Returns input uncompressed (N, 6912)."""
        return np.ascontiguousarray(X, dtype=np.float32)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Returns representation unchanged (N, 6912)."""
        return np.ascontiguousarray(Z, dtype=np.float32)

    def compute_reconstruction_error(self, X: np.ndarray) -> float:
        """Identity reconstruction error is exactly 0.0."""
        return 0.0

    def benchmark_latency(self, X: np.ndarray, n_repeats: int = 5) -> float:
        """Measure average latency in milliseconds per window."""
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            _ = self.transform(X)
        dt = (time.perf_counter() - t0) / (n_repeats * len(X))
        return float(dt * 1000.0)  # ms per window
