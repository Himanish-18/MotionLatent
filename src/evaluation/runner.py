"""
LOSO Experiment Runner -- Phase 3.

Executes Leave-One-Subject-Out evaluation across Baseline, PCA, FastICA, and
Autoencoder models with common evaluation protocols and metric aggregation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.data.loso import ALL_SUBJECTS, FoldData, load_fold_data
from src.models.baseline import RawBaselineRepresentation
from src.models.pca_model import PCARepresentation
from src.models.ica_model import ICARepresentation
from src.models.autoencoder import AutoencoderRepresentation
from src.evaluation.metrics import (
    compute_reconstruction_metrics,
    compute_neighbourhood_preservation,
    evaluate_downstream_knn,
    compute_resource_metrics,
    compute_weighted_fold_score,
)

PCA_DIMS = [2, 4, 8, 16, 32, 64, 128]
ICA_DIMS = [2, 4, 8, 16, 32, 64]
AE_DIMS = [4, 16, 64]

SUBJECT_WINDOW_COUNTS = {
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


def run_single_fold_experiments(
    fold: FoldData,
    pca_dims: List[int] = PCA_DIMS,
    ica_dims: List[int] = ICA_DIMS,
    ae_dims: List[int] = AE_DIMS,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run all representation models on a single LOSO fold.

    Parameters
    ----------
    fold : FoldData
        Loaded and normalized train/test data for the fold.
    pca_dims : list of int
        Candidate dimensions for PCA.
    ica_dims : list of int
        Candidate dimensions for FastICA.
    ae_dims : list of int
        Candidate dimensions for Autoencoder.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    list of dict records for each (method, dimension) on this fold.
    """
    sid = fold.test_subject
    results = []

    # -------------------------------------------------------------
    # 1. Raw Baseline (d=6912)
    # -------------------------------------------------------------
    if verbose:
        print(f"  [S{sid}] Running Raw Baseline (d=6912)...")
    baseline = RawBaselineRepresentation().fit(fold.X_train)
    Z_train_base = baseline.transform(fold.X_train)
    Z_test_base = baseline.transform(fold.X_test)
    knn_base = evaluate_downstream_knn(Z_train_base, fold.y_train, Z_test_base, fold.y_test)
    recon_base = {"mse": 0.0, "rmse": 0.0}
    trust_base = {"trustworthiness": 1.0, "knn_overlap": 1.0}
    res_base = compute_resource_metrics(
        orig_dim=6912,
        latent_dim=6912,
        param_count=0,
        model_size_bytes=0,
        latency_ms_per_window=baseline.benchmark_latency(fold.X_test),
    )

    results.append({
        "subject_id": sid,
        "method": "Raw Baseline",
        "latent_dim": 6912,
        "reconstruction": recon_base,
        "neighbourhood": trust_base,
        "downstream_knn": knn_base,
        "resources": res_base,
        "explained_variance_ratio": None,
        "cumulative_explained_variance": None,
        "converged": True,
    })

    # -------------------------------------------------------------
    # 2. PCA Sweep (d in pca_dims)
    # -------------------------------------------------------------
    if verbose:
        print(f"  [S{sid}] Fitting PCA (max_d={max(pca_dims)})...")
    t0 = time.time()
    pca_model = PCARepresentation(max_components=max(pca_dims), random_state=42)
    pca_model.fit(fold.X_train)
    if verbose:
        print(f"  [S{sid}] PCA fit complete in {time.time()-t0:.2f}s")

    for d in pca_dims:
        Z_train_pca = pca_model.transform(fold.X_train, d=d)
        Z_test_pca = pca_model.transform(fold.X_test, d=d)
        X_recon_pca = pca_model.inverse_transform(Z_test_pca, d=d)

        recon_pca = compute_reconstruction_metrics(fold.X_test, X_recon_pca)
        trust_pca = compute_neighbourhood_preservation(fold.X_test, Z_test_pca)
        knn_pca = evaluate_downstream_knn(Z_train_pca, fold.y_train, Z_test_pca, fold.y_test)
        res_pca = compute_resource_metrics(
            orig_dim=6912,
            latent_dim=d,
            param_count=pca_model.get_param_count(d),
            model_size_bytes=pca_model.get_model_size_bytes(d),
            latency_ms_per_window=pca_model.benchmark_latency(fold.X_test, d=d),
        )

        results.append({
            "subject_id": sid,
            "method": "PCA",
            "latent_dim": d,
            "reconstruction": recon_pca,
            "neighbourhood": trust_pca,
            "downstream_knn": knn_pca,
            "resources": res_pca,
            "explained_variance_ratio": float(pca_model.get_explained_variance_ratio(d)[-1]),
            "cumulative_explained_variance": pca_model.get_cumulative_explained_variance(d),
            "converged": True,
        })

    # -------------------------------------------------------------
    # 3. FastICA Sweep (d in ica_dims)
    # -------------------------------------------------------------
    for d in ica_dims:
        if verbose:
            print(f"  [S{sid}] Fitting FastICA (d={d})...")
        t0 = time.time()
        ica_model = ICARepresentation(n_components=d, max_iter=500, tol=1e-3, random_state=42)
        ica_model.fit(fold.X_train)
        if verbose:
            print(f"  [S{sid}] FastICA (d={d}) complete in {time.time()-t0:.2f}s (converged={ica_model.converged})")

        Z_train_ica = ica_model.transform(fold.X_train)
        Z_test_ica = ica_model.transform(fold.X_test)
        X_recon_ica = ica_model.inverse_transform(Z_test_ica)

        recon_ica = compute_reconstruction_metrics(fold.X_test, X_recon_ica)
        trust_ica = compute_neighbourhood_preservation(fold.X_test, Z_test_ica)
        knn_ica = evaluate_downstream_knn(Z_train_ica, fold.y_train, Z_test_ica, fold.y_test)
        res_ica = compute_resource_metrics(
            orig_dim=6912,
            latent_dim=d,
            param_count=ica_model.get_param_count(),
            model_size_bytes=ica_model.get_model_size_bytes(),
            latency_ms_per_window=ica_model.benchmark_latency(fold.X_test),
        )

        results.append({
            "subject_id": sid,
            "method": "FastICA",
            "latent_dim": d,
            "reconstruction": recon_ica,
            "neighbourhood": trust_ica,
            "downstream_knn": knn_ica,
            "resources": res_ica,
            "explained_variance_ratio": None,
            "cumulative_explained_variance": None,
            "converged": ica_model.converged,
        })

    # -------------------------------------------------------------
    # 4. Autoencoder Sweep (d in ae_dims)
    # -------------------------------------------------------------
    for d in ae_dims:
        if verbose:
            print(f"  [S{sid}] Training Autoencoder (d={d})...")
        t0 = time.time()
        ae_model = AutoencoderRepresentation(
            latent_dim=d,
            hidden_dim=256,
            max_epochs=12,
            batch_size=128,
            patience=3,
            random_seed=42,
        )
        ae_model.fit(fold.X_train)
        if verbose:
            print(f"  [S{sid}] Autoencoder (d={d}) trained in {time.time()-t0:.2f}s "
                  f"({len(ae_model.train_losses)} epochs, final val loss={ae_model.val_losses[-1]:.4f})")

        Z_train_ae = ae_model.transform(fold.X_train)
        Z_test_ae = ae_model.transform(fold.X_test)
        X_recon_ae = ae_model.inverse_transform(Z_test_ae)

        recon_ae = compute_reconstruction_metrics(fold.X_test, X_recon_ae)
        trust_ae = compute_neighbourhood_preservation(fold.X_test, Z_test_ae)
        knn_ae = evaluate_downstream_knn(Z_train_ae, fold.y_train, Z_test_ae, fold.y_test)
        res_ae = compute_resource_metrics(
            orig_dim=6912,
            latent_dim=d,
            param_count=ae_model.get_param_count(),
            model_size_bytes=ae_model.get_model_size_bytes(),
            latency_ms_per_window=ae_model.benchmark_latency(fold.X_test),
        )

        results.append({
            "subject_id": sid,
            "method": "Autoencoder",
            "latent_dim": d,
            "reconstruction": recon_ae,
            "neighbourhood": trust_ae,
            "downstream_knn": knn_ae,
            "resources": res_ae,
            "explained_variance_ratio": None,
            "cumulative_explained_variance": None,
            "converged": True,
        })

    return results


def aggregate_loso_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate fold-level results into means, standard deviations, and robustness views."""
    # Group results by (method, latent_dim)
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in all_results:
        key = (r["method"], r["latent_dim"])
        groups.setdefault(key, []).append(r)

    summary_rows = []
    for (method, d), records in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        # Full 9 subjects
        accs_all = [r["downstream_knn"]["accuracy"] for r in records]
        bal_accs_all = [r["downstream_knn"]["balanced_accuracy"] for r in records]
        f1s_all = [r["downstream_knn"]["macro_f1"] for r in records]
        mses_all = [r["reconstruction"]["mse"] for r in records]
        rmses_all = [r["reconstruction"]["rmse"] for r in records]
        trusts_all = [r["neighbourhood"]["trustworthiness"] for r in records]
        overlaps_all = [r["neighbourhood"]["knn_overlap"] for r in records]

        # 8 subjects excluding S109 (robustness view)
        records_ex109 = [r for r in records if r["subject_id"] != 109]
        accs_ex = [r["downstream_knn"]["accuracy"] for r in records_ex109]
        bal_accs_ex = [r["downstream_knn"]["balanced_accuracy"] for r in records_ex109]
        f1s_ex = [r["downstream_knn"]["macro_f1"] for r in records_ex109]

        # Subject window counts for weighted fold-score averaging
        counts_all = [SUBJECT_WINDOW_COUNTS.get(r["subject_id"], 1) for r in records]
        counts_ex = [SUBJECT_WINDOW_COUNTS.get(r["subject_id"], 1) for r in records_ex109]

        cum_var = records[0]["cumulative_explained_variance"]
        res = records[0]["resources"]

        summary_rows.append({
            "method": method,
            "latent_dim": d,
            "compression_ratio": res["compression_ratio"],
            # 9-subject LOSO macro-means
            "accuracy_mean": round(float(np.mean(accs_all)), 4),
            "accuracy_std": round(float(np.std(accs_all)), 4),
            "balanced_accuracy_mean": round(float(np.mean(bal_accs_all)), 4),
            "balanced_accuracy_std": round(float(np.std(bal_accs_all)), 4),
            "macro_f1_mean": round(float(np.mean(f1s_all)), 4),
            "macro_f1_std": round(float(np.std(f1s_all)), 4),
            # 9-subject weighted fold-score means: sum(N_k * S_k) / sum(N_k)
            "balanced_accuracy_weighted_mean": compute_weighted_fold_score(bal_accs_all, counts_all),
            "accuracy_weighted_mean": compute_weighted_fold_score(accs_all, counts_all),
            "macro_f1_weighted_mean": compute_weighted_fold_score(f1s_all, counts_all),
            # 8-subject metrics (excl. S109)
            "accuracy_ex109_mean": round(float(np.mean(accs_ex)), 4),
            "balanced_accuracy_ex109_mean": round(float(np.mean(bal_accs_ex)), 4),
            "macro_f1_ex109_mean": round(float(np.mean(f1s_ex)), 4),
            "balanced_accuracy_weighted_ex109_mean": compute_weighted_fold_score(bal_accs_ex, counts_ex),
            # Reconstruction & Neighbourhood
            "mse_mean": round(float(np.mean(mses_all)), 6),
            "mse_std": round(float(np.std(mses_all)), 6),
            "rmse_mean": round(float(np.mean(rmses_all)), 6),
            "trustworthiness_mean": round(float(np.mean(trusts_all)), 4),
            "knn_overlap_mean": round(float(np.mean(overlaps_all)), 4),
            # PCA variance
            "cumulative_explained_variance": round(cum_var, 4) if cum_var is not None else None,
            # Resources
            "param_count": res["param_count"],
            "model_size_kb": res["model_size_kb"],
            "encoding_latency_ms": round(float(np.mean([r["resources"]["encoding_latency_ms"] for r in records])), 4),
        })

    return {
        "n_folds": len(ALL_SUBJECTS),
        "subjects": ALL_SUBJECTS,
        "summary": summary_rows,
        "raw_results": all_results,
    }
