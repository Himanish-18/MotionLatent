"""
Phase 2.1 Validation Script -- MotionLatent

Validates every Phase 2 artifact against the code and contract.
Outputs structured findings for the validation report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.schema import get_primary_motion_columns, ACTIVITY_MAP
from src.data.preprocessing import (
    WINDOW_SAMPLES, WINDOW_STRIDE, NUM_CHANNELS, PURITY_THRESHOLD,
    load_subject_windows, load_window_metadata,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SCALERS_DIR = PROCESSED_DIR / "scalers"
SUBJECTS = [101, 102, 103, 104, 105, 106, 107, 108, 109]


def main():
    findings = []
    errors = []

    print("=" * 70)
    print("MotionLatent Phase 2.1 -- Validation Script")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Validate window shapes and NaN/Inf
    # ---------------------------------------------------------------
    print("\n[1] Validating window shapes and data integrity...")
    total_windows = 0
    per_subject_counts = {}
    for sid in SUBJECTS:
        wins = load_subject_windows(PROCESSED_DIR, sid)
        n = wins.shape[0]
        per_subject_counts[sid] = n
        total_windows += n

        if n == 0:
            findings.append(f"S{sid}: 0 windows (empty)")
            continue

        # Shape check
        assert wins.shape[1] == WINDOW_SAMPLES, f"S{sid}: bad window length {wins.shape[1]}"
        assert wins.shape[2] == NUM_CHANNELS, f"S{sid}: bad channel count {wins.shape[2]}"

        # NaN/Inf check
        nan_count = int(np.isnan(wins).sum())
        inf_count = int(np.isinf(wins).sum())
        if nan_count > 0:
            errors.append(f"S{sid}: {nan_count} NaN values in accepted windows!")
        if inf_count > 0:
            errors.append(f"S{sid}: {inf_count} Inf values in accepted windows!")

        print(f"  S{sid}: {n:>5} windows, shape {wins.shape}, "
              f"NaN={nan_count}, Inf={inf_count}")

    findings.append(f"Total windows across all subjects: {total_windows}")
    findings.append(f"Per-subject: {per_subject_counts}")

    # ---------------------------------------------------------------
    # 2. Validate metadata
    # ---------------------------------------------------------------
    print("\n[2] Validating window metadata...")
    meta = load_window_metadata(PROCESSED_DIR)
    findings.append(f"Metadata records: {len(meta)}")

    if len(meta) != total_windows:
        errors.append(
            f"Metadata count ({len(meta)}) != total windows ({total_windows})"
        )

    # Categorize windows
    pure_non_transient = 0
    pure_transient = 0
    mixed_total = 0
    transient_mixed = 0
    per_subject_meta = {s: 0 for s in SUBJECTS}
    per_activity_pure = {}

    for m in meta:
        sid = m["subject_id"]
        per_subject_meta[sid] = per_subject_meta.get(sid, 0) + 1

        is_mixed = m["is_mixed_label"]
        is_transient = m["is_transient"]
        dominant = m["dominant_activity"]
        purity = m["activity_purity"]

        if is_mixed:
            mixed_total += 1
            if is_transient:
                transient_mixed += 1
        else:
            if is_transient:
                pure_transient += 1
            else:
                pure_non_transient += 1
                per_activity_pure[dominant] = per_activity_pure.get(dominant, 0) + 1

        # Verify purity labeling
        if not is_mixed and purity < PURITY_THRESHOLD:
            errors.append(
                f"Window {m['window_idx']} S{sid}: marked pure but "
                f"purity={purity} < {PURITY_THRESHOLD}"
            )
        if is_mixed and purity >= PURITY_THRESHOLD:
            errors.append(
                f"Window {m['window_idx']} S{sid}: marked mixed but "
                f"purity={purity} >= {PURITY_THRESHOLD}"
            )

        # Verify proportions sum to 1
        prop_sum = sum(m["activity_proportions"].values())
        if abs(prop_sum - 1.0) > 1e-4:
            errors.append(
                f"Window {m['window_idx']} S{sid}: proportions sum to {prop_sum}"
            )

    print(f"  Pure (non-transient): {pure_non_transient}")
    print(f"  Pure (transient):     {pure_transient}")
    print(f"  Mixed:                {mixed_total}")
    print(f"    (of which transient-dominant mixed: {transient_mixed})")
    print(f"  Total:                {pure_non_transient + pure_transient + mixed_total}")

    # Check category arithmetic dynamically against pipeline report
    with open(PROCESSED_DIR / "phase2_pipeline_report.json", "r") as f:
        pipe_report = json.load(f)

    report_claims_pure = pipe_report["pure_windows"]
    report_claims_mixed = pipe_report["mixed_windows"]
    report_claims_transient = pipe_report["transient_windows"]
    report_claims_total = pipe_report["total_windows"]

    findings.append(f"Recomputed pure (non-transient): {pure_non_transient}")
    findings.append(f"Recomputed pure (transient): {pure_transient}")
    findings.append(f"Recomputed mixed: {mixed_total}")

    expected_total = report_claims_pure + report_claims_transient + report_claims_mixed
    findings.append(f"Report sum: {report_claims_pure}+{report_claims_transient}+"
                    f"{report_claims_mixed} = {expected_total} "
                    f"(claimed total={report_claims_total})")

    actual_total = pure_non_transient + pure_transient + mixed_total
    if actual_total != total_windows:
        errors.append(
            f"Category sum ({actual_total}) != total windows ({total_windows})"
        )
    if pure_non_transient != report_claims_pure:
        errors.append(
            f"Pure non-transient ({pure_non_transient}) != report ({report_claims_pure})"
        )
    if pure_transient != report_claims_transient:
        errors.append(
            f"Pure transient ({pure_transient}) != report ({report_claims_transient})"
        )
    if mixed_total != report_claims_mixed:
        errors.append(
            f"Mixed total ({mixed_total}) != report ({report_claims_mixed})"
        )

    # Check per-subject metadata counts match npz counts
    for sid in SUBJECTS:
        if per_subject_meta[sid] != per_subject_counts[sid]:
            errors.append(
                f"S{sid}: metadata count ({per_subject_meta[sid]}) "
                f"!= npz count ({per_subject_counts[sid]})"
            )

    # ---------------------------------------------------------------
    # 3. Validate timestamp ordering per subject
    # ---------------------------------------------------------------
    print("\n[3] Validating timestamp ordering...")
    for sid in SUBJECTS:
        subject_meta = [m for m in meta if m["subject_id"] == sid]
        timestamps = [m["start_timestamp"] for m in subject_meta]
        if timestamps != sorted(timestamps):
            errors.append(f"S{sid}: timestamps NOT in order!")
        else:
            print(f"  S{sid}: timestamps ordered ({len(timestamps)} windows)")

    # ---------------------------------------------------------------
    # 4. Validate LOSO scalers
    # ---------------------------------------------------------------
    print("\n[4] Validating LOSO scalers...")
    for test_sid in SUBJECTS:
        scaler_path = SCALERS_DIR / f"scaler_test{test_sid}.json"
        if not scaler_path.exists():
            errors.append(f"Missing scaler file for test subject {test_sid}")
            continue

        with open(scaler_path, "r") as f:
            sc = json.load(f)

        train_subs = sc["training_subjects"]
        test_sub = sc["test_subject"]

        # Verify test subject excluded
        if test_sub in train_subs:
            errors.append(f"Scaler test{test_sid}: test subject IN training!")
        if test_sub != test_sid:
            errors.append(f"Scaler test{test_sid}: test_subject={test_sub} != {test_sid}")

        # Verify all other subjects included
        expected_train = sorted(s for s in SUBJECTS if s != test_sid)
        if sorted(train_subs) != expected_train:
            errors.append(
                f"Scaler test{test_sid}: training={train_subs}, "
                f"expected={expected_train}"
            )

        # Verify mean/std shape
        if len(sc["mean"]) != NUM_CHANNELS:
            errors.append(f"Scaler test{test_sid}: mean length={len(sc['mean'])}")
        if len(sc["std"]) != NUM_CHANNELS:
            errors.append(f"Scaler test{test_sid}: std length={len(sc['std'])}")

        # Verify no zero std (which would indicate a constant channel)
        zero_std = sum(1 for s in sc["std"] if s == 0)
        if zero_std > 0:
            errors.append(f"Scaler test{test_sid}: {zero_std} channels with std=0")

        print(f"  Scaler test{test_sid}: train={train_subs}, "
              f"test={test_sub}, n_train_windows={sc['n_training_samples']}")

    # ---------------------------------------------------------------
    # 5. Validate channel order
    # ---------------------------------------------------------------
    print("\n[5] Validating channel order...")
    with open(PROCESSED_DIR / "channel_order.json", "r") as f:
        saved_channels = json.load(f)
    expected_channels = get_primary_motion_columns()
    if saved_channels != expected_channels:
        errors.append("Saved channel order doesn't match schema!")
    else:
        print(f"  Channel order matches schema ({len(saved_channels)} channels)")

    # ---------------------------------------------------------------
    # 6. Validate pipeline report JSON
    # ---------------------------------------------------------------
    print("\n[6] Validating pipeline report JSON...")
    with open(PROCESSED_DIR / "phase2_pipeline_report.json", "r") as f:
        report = json.load(f)

    # Cross-check key counts
    if report["total_windows"] != total_windows:
        errors.append(
            f"Report total_windows ({report['total_windows']}) "
            f"!= actual ({total_windows})"
        )
    if report["pure_windows"] != pure_non_transient:
        errors.append(
            f"Report pure_windows ({report['pure_windows']}) "
            f"!= actual pure non-transient ({pure_non_transient})"
        )
    if report["mixed_windows"] != mixed_total:
        errors.append(
            f"Report mixed_windows ({report['mixed_windows']}) "
            f"!= actual ({mixed_total})"
        )
    if report["transient_windows"] != pure_transient:
        errors.append(
            f"Report transient_windows ({report['transient_windows']}) "
            f"!= actual pure transient ({pure_transient})"
        )

    findings.append(f"Pipeline JSON report cross-check against metadata: "
                    f"{'MATCH' if not any('Report' in e for e in errors) else 'MISMATCH'}")

    # ---------------------------------------------------------------
    # 7. Subject 109 validation
    # ---------------------------------------------------------------
    print("\n[7] Subject 109 analysis...")
    s109_meta = [m for m in meta if m["subject_id"] == 109]
    s109_activities = set(m["dominant_activity"] for m in s109_meta)
    s109_pure = [m for m in s109_meta if not m["is_mixed_label"]]
    s109_pure_acts = set(m["dominant_activity"] for m in s109_pure)
    print(f"  Windows: {len(s109_meta)}")
    print(f"  Activities (all): {sorted(s109_activities)}")
    print(f"  Activities (pure): {sorted(s109_pure_acts)}")
    print(f"  Pure: {sum(1 for m in s109_meta if not m['is_mixed_label'])}")
    print(f"  Mixed: {sum(1 for m in s109_meta if m['is_mixed_label'])}")
    findings.append(f"S109: {len(s109_meta)} windows, activities={sorted(s109_activities)}")

    # ---------------------------------------------------------------
    # 8. Verify window stride against actual data
    # ---------------------------------------------------------------
    print("\n[8] Verifying window stride consistency...")
    expected_dt = WINDOW_STRIDE * 0.01  # 1.28 seconds
    under_stride_issues = 0
    gap_stride_steps = 0
    for sid in SUBJECTS:
        subject_meta = [m for m in meta if m["subject_id"] == sid]
        if len(subject_meta) < 2:
            continue
        for i in range(1, len(subject_meta)):
            actual_dt = subject_meta[i]["start_timestamp"] - subject_meta[i-1]["start_timestamp"]
            if actual_dt < expected_dt - 0.01:
                under_stride_issues += 1
                errors.append(
                    f"S{sid} window {i}: under-stride dt={actual_dt:.4f} < expected {expected_dt}"
                )
            elif actual_dt > expected_dt + 0.02:
                gap_stride_steps += 1
    if under_stride_issues == 0:
        print(f"  All continuous window strides match nominal {expected_dt}s")
        print(f"  ({gap_stride_steps} inter-window steps span gap boundaries where invalid windows were rejected)")
    else:
        print(f"  {under_stride_issues} under-stride violations found")
    findings.append(f"Under-stride violations: {under_stride_issues}")
    findings.append(f"Inter-window gap boundaries: {gap_stride_steps}")

    # ---------------------------------------------------------------
    # 9. Verify NO accepted window crosses a gap (duration <= 2.57s)
    # ---------------------------------------------------------------
    print("\n[9] Verifying window continuity (no gap crossing)...")
    gap_crossing_count = 0
    for m in meta:
        dur = m["end_timestamp"] - m["start_timestamp"]
        if dur > 2.57:
            gap_crossing_count += 1
            if gap_crossing_count <= 3:
                errors.append(
                    f"Window {m['window_idx']} S{m['subject_id']}: duration={dur:.4f}s > 2.57s (crosses gap!)"
                )
    if gap_crossing_count == 0:
        print(f"  All accepted windows are continuous (duration exactly ~2.55s, no gaps)")
    else:
        print(f"  {gap_crossing_count} windows cross gaps!")
    findings.append(f"Gap-crossing windows in accepted data: {gap_crossing_count}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nFindings ({len(findings)}):")
    for f_item in findings:
        print(f"  [INFO] {f_item}")

    print(f"\nErrors ({len(errors)}):")
    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")
    else:
        print("  None!")

    verdict = "PASS" if not errors else "FAIL"
    print(f"\nVERDICT: {verdict}")
    return verdict, findings, errors


if __name__ == "__main__":
    verdict, findings, errors = main()
    sys.exit(0 if verdict == "PASS" else 1)
