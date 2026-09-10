"""
MotionLatent Phase 1 — Main Audit Runner

Execute with:
    python run_audit.py

This script runs the complete PAMAP2 data audit pipeline,
generates visualizations, and produces the machine-readable
audit output JSON.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.discovery import discover_dataset
from src.data.audit import run_full_audit
from src.data.visualize import generate_all_plots


def main() -> None:
    """Run the complete Phase 1 audit."""
    print("MotionLatent Phase 1 -- PAMAP2 Data Audit")
    print("=" * 60)

    # --- Discover dataset ---
    print("\n[DISCOVERY] Discovering dataset...")
    paths = discover_dataset(PROJECT_ROOT)
    print(f"   Project root:  {paths.project_root}")
    print(f"   Dataset root:  {paths.dataset_root}")
    print(f"   Protocol files: {len(paths.protocol_files)}")
    print(f"   Optional files: {len(paths.optional_files)}")
    print(f"   Documentation:  {len(paths.documentation_files)}")

    # --- Run audit ---
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    results, df = run_full_audit(paths, output_dir=reports_dir)

    # --- Generate visualizations ---
    print("\n[PLOTS] Generating visualizations...")
    figures_dir = reports_dir / "figures"
    saved_plots = generate_all_plots(df, figures_dir)
    print(f"   Saved {len(saved_plots)} plots to {figures_dir}")

    # --- Summary ---
    ds = results["dataset"]
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"  Protocol files:        {ds['protocol_files']}")
    print(f"  Optional files:        {ds['optional_files']}")
    print(f"  Protocol observations: {ds['total_protocol_observations']:,}")
    print(f"  Optional observations: {ds['total_optional_observations']:,}")
    print(f"  Columns:               {ds['num_columns']}")
    print(f"  Duplicates:            {results['duplicates']}")
    print(f"\n  Activities found:")
    for act in results["activities"]["distribution"]:
        print(f"    [{act['activity_id']:>2}] {act['activity_name']:<20s} "
              f"{act['count']:>8,} ({act['percentage']:.1f}%)")
    print(f"\n  Subjects:")
    for subj in results["subjects"]:
        print(f"    S{subj['subject_id']} — {subj['observations']:>8,} obs, "
              f"{subj['num_activities']} activities, "
              f"{subj['duration_minutes']:.1f} min")
    print(f"\n  Reports written to: {reports_dir}")
    print("\n[DONE] Phase 1 audit complete.")


if __name__ == "__main__":
    main()
