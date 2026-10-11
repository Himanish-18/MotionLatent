"""
Structure-Preservation & Performance Metrics -- Phase 3.

Implements rigorous, reproducible metrics for:
- Reconstruction quality (MSE, RMSE)
- Neighbourhood preservation (Trustworthiness, kNN overlap)
- Downstream activity recognition (kNN: Accuracy, Balanced Accuracy, Macro-F1)
- Computational and resource costs (Compression ratio, parameter count, model size, latency)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.manifold import trustworthiness


def compute_reconstruction_metrics(
    X_true: np.ndarray,
    X_recon: np.ndarray,
) -> Dict[str, float]:
    """Compute MSE and RMSE in the normalized input space.

    Parameters
    ----------
    X_true : np.ndarray, shape (N, D)
        Original normalized observations.
    X_recon : np.ndarray, shape (N, D)
        Reconstructed observations from latent codes.

    Returns
    -------
    dict with 'mse' and 'rmse'
    """
    diff = (X_true.astype(np.float64) - X_recon.astype(np.float64)).ravel()
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    return {
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
    }


def compute_neighbourhood_preservation(
    X_orig: np.ndarray,
    Z_latent: np.ndarray,
    n_neighbors: int = 5,
    sample_size: int = 500,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute trustworthiness and kNN overlap between original and latent space.

    Parameters
    ----------
    X_orig : np.ndarray, shape (N, 6912)
        Original normalized observations.
    Z_latent : np.ndarray, shape (N, d)
        Compressed latent representations.
    n_neighbors : int, default=5
        Number of nearest neighbors to consider.
    sample_size : int, default=500
        Subset size for efficient, reproducible computation.
    random_state : int, default=42
        Seed for subsampling.

    Returns
    -------
    dict with 'trustworthiness' and 'knn_overlap'
    """
    n_samples = len(X_orig)
    if n_samples <= n_neighbors:
        return {"trustworthiness": 1.0, "knn_overlap": 1.0}

    # Subsample if sample_size < n_samples for computational feasibility
    if n_samples > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_samples, size=sample_size, replace=False)
        X_sub = X_orig[idx]
        Z_sub = Z_latent[idx]
    else:
        X_sub = X_orig
        Z_sub = Z_latent

    # 1. Scikit-learn Trustworthiness
    # Trustworthiness is defined for n_neighbors < n_samples / 2
    k = min(n_neighbors, len(X_sub) // 2 - 1)
    if k < 1:
        k = 1
    t_val = float(trustworthiness(X_sub, Z_sub, n_neighbors=k))

    # 2. k-NN Jaccard Overlap
    # What fraction of top-k neighbors in original space remain top-k in latent space?
    nn_orig = NearestNeighbors(n_neighbors=k + 1).fit(X_sub)
    orig_nbrs = nn_orig.kneighbors(X_sub, return_distance=False)[:, 1:]  # exclude self

    nn_lat = NearestNeighbors(n_neighbors=k + 1).fit(Z_sub)
    lat_nbrs = nn_lat.kneighbors(Z_sub, return_distance=False)[:, 1:]

    overlaps = []
    for i in range(len(X_sub)):
        set_orig = set(orig_nbrs[i])
        set_lat = set(lat_nbrs[i])
        intersection = len(set_orig.intersection(set_lat))
        union = len(set_orig.union(set_lat))
        overlaps.append(intersection / union if union > 0 else 1.0)
    knn_overlap = float(np.mean(overlaps))

    return {
        "trustworthiness": round(t_val, 4),
        "knn_overlap": round(knn_overlap, 4),
    }


def evaluate_downstream_knn(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    n_neighbors: int = 5,
) -> Dict[str, Any]:
    """Evaluate activity structure preservation using fixed kNN classifier.

    Parameters
    ----------
    Z_train : np.ndarray, shape (N_train, d)
        Training latent representations.
    y_train : np.ndarray, shape (N_train,)
        Training activity labels.
    Z_test : np.ndarray, shape (N_test, d)
        Test latent representations.
    y_test : np.ndarray, shape (N_test,)
        Test activity labels.
    n_neighbors : int, default=5
        k parameter for kNN.

    Returns
    -------
    dict with 'accuracy', 'balanced_accuracy', 'macro_f1', 'n_classes_test'
    """
    if len(Z_test) == 0 or len(Z_train) == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "n_classes_test": 0,
        }

    k = min(n_neighbors, len(Z_train))
    clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    clf.fit(Z_train, y_train)
    y_pred = clf.predict(Z_test)

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "n_classes_test": int(len(np.unique(y_test))),
    }


def compute_resource_metrics(
    orig_dim: int = 6912,
    latent_dim: int = 16,
    param_count: int = 0,
    model_size_bytes: int = 0,
    latency_ms_per_window: float = 0.0,
) -> Dict[str, Any]:
    """Compute compression ratio and compute metrics."""
    compression_ratio = float(orig_dim / latent_dim) if latent_dim > 0 else 0.0
    latent_storage_bytes = int(latent_dim * 4)  # float32

    return {
        "orig_dim": orig_dim,
        "latent_dim": latent_dim,
        "compression_ratio": round(compression_ratio, 2),
        "latent_bytes_per_window": latent_storage_bytes,
        "param_count": param_count,
        "model_size_kb": round(model_size_bytes / 1024.0, 2),
        "encoding_latency_ms": round(latency_ms_per_window, 4),
    }
