"""
Subject 108 Generalization Diagnostic Engine -- Phase 4D.

Investigates why Subject 108 exhibits poor cross-subject generalization across
all representations without compromising primary LOSO isolation.

Performs:
1. Activity class distribution & per-class error analysis.
2. Confusion matrix & representation-level error clustering.
3. Sensor channel distribution shift (mean, std, L2 shift across Hand/Chest/Ankle IMUs).
4. Exploratory unsupervised target-domain calibration using unlabeled S108 windows
   (zero ground-truth labels exposed).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from src.data.loso import load_fold_data, FoldData
from src.models.pca_model import PCARepresentation


ACTIVITY_NAMES = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "Nordic walking",
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
    24: "rope jumping",
}


def analyze_subject108_class_distribution(fold: FoldData) -> Dict[str, Any]:
    """Examine Subject 108's activity class distribution compared to training population."""
    assert fold.test_subject == 108, "Fold test subject must be 108"

    s108_classes, s108_counts = np.unique(fold.y_test, return_counts=True)
    train_classes, train_counts = np.unique(fold.y_train, return_counts=True)

    distribution_comparison = {}
    for c in sorted(s108_classes):
        c_name = ACTIVITY_NAMES.get(int(c), f"Class_{c}")
        train_cnt = int(train_counts[train_classes == c][0]) if c in train_classes else 0
        distribution_comparison[c_name] = {
            "activity_id": int(c),
            "s108_count": int(s108_counts[s108_classes == c][0]),
            "s108_percentage": round(float(s108_counts[s108_classes == c][0]) / len(fold.y_test) * 100.0, 2),
            "train_count": train_cnt,
            "train_percentage": round(float(train_cnt) / len(fold.y_train) * 100.0, 2),
        }

    return {
        "n_test_windows": len(fold.y_test),
        "n_train_windows": len(fold.y_train),
        "classes_present_in_s108": [int(c) for c in s108_classes],
        "class_breakdown": distribution_comparison,
    }


def analyze_subject108_confusion(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "PCA-32",
) -> Dict[str, Any]:
    """Analyze confusion matrix and per-activity recall on Subject 108."""
    clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    clf.fit(Z_train, y_train)
    y_pred = clf.predict(Z_test)

    classes = sorted(np.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    per_class_recalls = {}
    for i, c in enumerate(classes):
        c_name = ACTIVITY_NAMES.get(int(c), f"Class_{c}")
        total = int(np.sum(cm[i]))
        correct = int(cm[i, i])
        rec = float(correct / total) if total > 0 else 0.0

        # Most frequent misclassification
        row = cm[i].copy()
        row[i] = 0
        top_err_idx = int(np.argmax(row))
        top_err_class = classes[top_err_idx]
        top_err_name = ACTIVITY_NAMES.get(int(top_err_class), f"Class_{top_err_class}")

        per_class_recalls[c_name] = {
            "activity_id": int(c),
            "samples": total,
            "correct": correct,
            "recall": round(rec, 4),
            "primary_confusion": top_err_name if row[top_err_idx] > 0 else "None",
            "confusion_count": int(row[top_err_idx]),
        }

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    return {
        "model_name": model_name,
        "overall_accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_performance": per_class_recalls,
        "confusion_matrix": cm.tolist(),
        "classes": [int(c) for c in classes],
    }


def analyze_sensor_distribution_shift(fold: FoldData) -> Dict[str, Any]:
    """Compare channel-wise statistics between Subject 108 and training population.

    Computes channel means and stds across 27 channels (reshaped from 6912).
    """
    # Reshape from (N, 6912) to (N, 256, 27) and average over time
    X_train_chan = fold.X_train.reshape(-1, 256, 27).mean(axis=1)  # (N_train, 27)
    X_test_chan = fold.X_test.reshape(-1, 256, 27).mean(axis=1)    # (N_test, 27)

    train_mean = np.mean(X_train_chan, axis=0)  # (27,)
    test_mean = np.mean(X_test_chan, axis=0)    # (27,)

    train_std = np.std(X_train_chan, axis=0)
    test_std = np.std(X_test_chan, axis=0)

    channel_shifts = []
    imu_names = ["Hand", "Chest", "Ankle"]
    mod_names = ["16g_Acc_X", "16g_Acc_Y", "16g_Acc_Z",
                 "Gyro_X", "Gyro_Y", "Gyro_Z",
                 "Mag_X", "Mag_Y", "Mag_Z"]

    for ch_idx in range(27):
        imu = imu_names[ch_idx // 9]
        mod = mod_names[ch_idx % 9]
        name = f"{imu}_{mod}"
        mean_diff = float(abs(test_mean[ch_idx] - train_mean[ch_idx]))
        z_shift = float(mean_diff / (train_std[ch_idx] + 1e-6))

        channel_shifts.append({
            "channel_idx": ch_idx,
            "channel_name": name,
            "train_mean": round(float(train_mean[ch_idx]), 4),
            "test_mean": round(float(test_mean[ch_idx]), 4),
            "mean_shift": round(mean_diff, 4),
            "z_shift": round(z_shift, 4),
        })

    # Sort channels by magnitude of distribution shift
    top_shifted_channels = sorted(channel_shifts, key=lambda x: x["z_shift"], reverse=True)

    return {
        "mean_global_z_shift": round(float(np.mean([c["z_shift"] for c in channel_shifts])), 4),
        "top_shifted_channels": top_shifted_channels[:10],
        "all_channel_shifts": channel_shifts,
    }


def evaluate_unsupervised_target_calibration(
    fold: FoldData,
    model: Any,
) -> Dict[str, Any]:
    """Test unsupervised test-time target domain calibration on Subject 108.

    Strict Rules Enforced:
    1. ZERO ground-truth labels for Subject 108 are used during adaptation.
    2. Primary unadapted benchmark is preserved unchanged.
    3. Method: Unsupervised test-time feature centering (shifting S108 unlabeled sensor mean).
    """
    assert fold.test_subject == 108

    # 1. Unadapted baseline (standard primary LOSO)
    Z_train = model.transform(fold.X_train)
    Z_test_unadapted = model.transform(fold.X_test)

    clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    clf.fit(Z_train, fold.y_train)
    pred_unadapted = clf.predict(Z_test_unadapted)

    bacc_unadapted = float(balanced_accuracy_score(fold.y_test, pred_unadapted))
    acc_unadapted = float(accuracy_score(fold.y_test, pred_unadapted))
    f1_unadapted = float(f1_score(fold.y_test, pred_unadapted, average="macro", zero_division=0))

    # 2. Unsupervised Test-Time Channel Centering
    # Reshape X_test to (N, 256, 27)
    X_test_3d = fold.X_test.reshape(-1, 256, 27).copy()
    s108_unlabeled_channel_mean = np.mean(X_test_3d, axis=(0, 1), keepdims=True)  # (1, 1, 27)
    
    # Center S108 around zero in sensor space (without labels)
    X_test_centered_3d = X_test_3d - s108_unlabeled_channel_mean
    X_test_centered_flat = X_test_centered_3d.reshape(-1, 6912)

    Z_test_centered = model.transform(X_test_centered_flat)
    pred_centered = clf.predict(Z_test_centered)

    bacc_centered = float(balanced_accuracy_score(fold.y_test, pred_centered))
    acc_centered = float(accuracy_score(fold.y_test, pred_centered))
    f1_centered = float(f1_score(fold.y_test, pred_centered, average="macro", zero_division=0))

    # 3. Unsupervised Latent Covariance Whitening (CORAL / Domain Alignment)
    # Align latent covariance of S108 to training latent covariance without labels
    cov_train = np.cov(Z_train, rowvar=False) + 1e-4 * np.eye(Z_train.shape[1])
    cov_test = np.cov(Z_test_unadapted, rowvar=False) + 1e-4 * np.eye(Z_test_unadapted.shape[1])

    U_tr, S_tr, _ = np.linalg.svd(cov_train)
    U_te, S_te, _ = np.linalg.svd(cov_test)

    # Whitening operator: cov_train^(1/2) * cov_test^(-1/2)
    whiten_test = U_te @ np.diag(1.0 / np.sqrt(S_te)) @ U_te.T
    color_train = U_tr @ np.diag(np.sqrt(S_tr)) @ U_tr.T
    A_align = whiten_test @ color_train

    mean_z_test = np.mean(Z_test_unadapted, axis=0, keepdims=True)
    mean_z_train = np.mean(Z_train, axis=0, keepdims=True)
    Z_test_aligned = (Z_test_unadapted - mean_z_test) @ A_align + mean_z_train

    pred_aligned = clf.predict(Z_test_aligned)
    bacc_aligned = float(balanced_accuracy_score(fold.y_test, pred_aligned))
    acc_aligned = float(accuracy_score(fold.y_test, pred_aligned))
    f1_aligned = float(f1_score(fold.y_test, pred_aligned, average="macro", zero_division=0))

    return {
        "unadapted": {
            "balanced_accuracy": round(bacc_unadapted, 4),
            "accuracy": round(acc_unadapted, 4),
            "macro_f1": round(f1_unadapted, 4),
        },
        "unsupervised_channel_centering": {
            "balanced_accuracy": round(bacc_centered, 4),
            "accuracy": round(acc_centered, 4),
            "macro_f1": round(f1_centered, 4),
            "delta_balanced_accuracy": round(bacc_centered - bacc_unadapted, 4),
        },
        "unsupervised_latent_covariance_alignment": {
            "balanced_accuracy": round(bacc_aligned, 4),
            "accuracy": round(acc_aligned, 4),
            "macro_f1": round(f1_aligned, 4),
            "delta_balanced_accuracy": round(bacc_aligned - bacc_unadapted, 4),
        },
        "deployment_requirements": (
            "Unsupervised calibration requires buffered target-domain unlabeled windows "
            "prior to classification. In real-time streaming, unadapted inference remains the "
            "primary zero-buffering deployment paradigm."
        ),
    }


def run_complete_subject108_diagnostic(fold_data: Optional[FoldData] = None) -> Dict[str, Any]:
    """Execute complete diagnostic suite for Subject 108."""
    if fold_data is None:
        fold_data = load_fold_data(test_subject=108, pure_only=True, exclude_transient=True)

    # 1. Class distribution analysis
    class_dist = analyze_subject108_class_distribution(fold_data)

    # 2. Sensor shift analysis
    sensor_shift = analyze_sensor_distribution_shift(fold_data)

    # 3. Model confusion analysis with PCA-32
    pca32 = PCARepresentation(max_components=32, random_state=42).fit(fold_data.X_train)
    Z_train_pca = pca32.transform(fold_data.X_train, d=32)
    Z_test_pca = pca32.transform(fold_data.X_test, d=32)

    confusion_pca32 = analyze_subject108_confusion(
        Z_train_pca, fold_data.y_train, Z_test_pca, fold_data.y_test, model_name="PCA-32"
    )

    # 4. Unsupervised target-domain calibration experiment
    calibration_results = evaluate_unsupervised_target_calibration(fold_data, pca32)

    return {
        "class_distribution": class_dist,
        "sensor_distribution_shift": sensor_shift,
        "confusion_analysis": confusion_pca32,
        "unsupervised_calibration": calibration_results,
    }


if __name__ == "__main__":
    diag = run_complete_subject108_diagnostic()
    print("Subject 108 Diagnostic Complete:")
    print("  Global Mean Z-Shift:", diag["sensor_distribution_shift"]["mean_global_z_shift"])
    print("  Top shifted channel:", diag["sensor_distribution_shift"]["top_shifted_channels"][0])
    print("  Unadapted Bal. Acc (PCA-32):", diag["unsupervised_calibration"]["unadapted"]["balanced_accuracy"])
    print("  Latent Aligned Bal. Acc:    ", diag["unsupervised_calibration"]["unsupervised_latent_covariance_alignment"]["balanced_accuracy"])
