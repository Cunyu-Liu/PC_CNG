"""P4-G6: Real HTE External Validation — orchestration.

Runs the full pipeline:
  1. Data normalization (raw HiTEA -> parquet with audit fields)
  2. Screen-aware split manifest
  3. Data audit JSON
  4. 5 methods × 5 tasks evaluation with cluster-aware bootstrap
  5. Aggregation + go/no_go.json

Usage::

    python3 -m pc_cng.run_p4_hte_validation --device cpu

    # Or step-by-step:
    python3 -m pc_cng.run_p4_hte_validation data  # data pipeline only
    python3 -m pc_cng.run_p4_hte_validation eval  # evaluation only
    python3 -m pc_cng.run_p4_hte_validation aggregate  # aggregation only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from pc_cng.p4_g6_hte_data import run_data_pipeline  # noqa: E402
from pc_cng.p4_g6_hte_eval import run_evaluation  # noqa: E402
from pc_cng.aggregate_p4_g6 import aggregate  # noqa: E402

# Default paths (relative to repo root)
DEFAULT_RAW = _REPO_ROOT / "external/HiTEA/data/8_SEPT_APPROVED_full_dataset.csv"
DEFAULT_NORM = _REPO_ROOT / "data/processed/hitea_full_normalized.csv"
DEFAULT_MANIFEST = _REPO_ROOT / "data/p4/manifests/hte_feasibility_v2.json"
DEFAULT_RISK_ARTIFACTS = _REPO_ROOT / "results/p4_risk_aware/risk_artifacts.json"
DEFAULT_OUTPUT = _REPO_ROOT


def cmd_data(args):
    """Run data pipeline: normalize -> split -> audit."""
    summary = run_data_pipeline(
        raw_path=Path(args.raw),
        norm_csv_path=Path(args.norm_csv),
        output_dir=Path(args.output),
    )
    print("\n=== Data Pipeline Complete ===")
    print(f"Records: {summary['normalize']['n_records']}")
    print(f"Screens: {summary['normalize']['n_screens']}")
    print(f"Families: {summary['normalize']['n_families']}")
    print(f"Split: {summary['split']['reaction_counts']}")
    print(f"Audit verified: {summary['audit']['hte_authenticity_verified']}")


def cmd_eval(args):
    """Run evaluation: 5 methods × 5 tasks."""
    parquet = Path(args.output) / "data/processed/p4_hte_normalized.parquet"
    manifest = Path(args.manifest)
    risk_artifacts = Path(args.risk_artifacts) if args.risk_artifacts else None
    output = Path(args.output) / "results/p4_hte_external_validation"

    results = run_evaluation(
        parquet_path=parquet,
        manifest_path=manifest,
        risk_artifacts_path=risk_artifacts,
        output_dir=output,
        seed=args.seed,
    )
    print("\n=== Evaluation Complete ===")
    for method, res in results.items():
        pe = res["point_estimates"]
        print(f"  {method}: t1_auprc_5={pe.get('t1_low_yield_auprc_5', 0):.4f}, "
              f"t4_ndcg={pe.get('t4_plate_ndcg', 0):.4f}, "
              f"ece={pe.get('ece', 0):.4f}")


def cmd_aggregate(args):
    """Run aggregation: produce go_no_go.json."""
    summary = Path(args.output) / "results/p4_hte_external_validation/summary.csv"
    audit = Path(args.output) / "results/p4_hte_external_validation/data_audit.json"
    output = Path(args.output) / "results/p4_hte_external_validation/go_no_go.json"

    go_no_go = aggregate(summary, audit, output)
    print("\n=== Aggregation Complete ===")
    print(f"Verdict: {go_no_go['status']}")
    print(f"Reason: {go_no_go['reason']}")
    print(f"next_phase_allowed: {go_no_go['next_phase_allowed']}")


def cmd_all(args):
    """Run full pipeline: data -> eval -> aggregate."""
    t0 = time.time()
    cmd_data(args)
    cmd_eval(args)
    cmd_aggregate(args)
    print(f"\n=== P4-G6 Complete ({time.time() - t0:.1f}s) ===")


def main():
    parser = argparse.ArgumentParser(description="P4-G6: Real HTE External Validation")
    parser.add_argument("command", nargs="?", default="all",
                        choices=["all", "data", "eval", "aggregate"])
    parser.add_argument("--raw", default=str(DEFAULT_RAW),
                        help="Raw HiTEA CSV path")
    parser.add_argument("--norm-csv", default=str(DEFAULT_NORM),
                        help="Normalized HTEa CSV path")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                        help="PC-CNG manifest (G3/G5 frozen)")
    parser.add_argument("--risk-artifacts", default=str(DEFAULT_RISK_ARTIFACTS),
                        help="G5 risk artifacts JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output root directory")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.command == "data":
        cmd_data(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "aggregate":
        cmd_aggregate(args)
    else:
        cmd_all(args)


if __name__ == "__main__":
    main()
