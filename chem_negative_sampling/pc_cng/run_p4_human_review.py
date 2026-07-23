"""P4-G7: Human Expert Calibration — orchestration.

Usage::

    python3 -m pc_cng.run_p4_human_review pilot     # Generate pilot (80 samples)
    python3 -m pc_cng.run_p4_human_review main      # Generate main review (250 samples)
    python3 -m pc_cng.run_p4_human_review analyze   # Analyze expert responses
    python3 -m pc_cng.run_p4_human_review all       # pilot + go_no_go (DEFERRED)
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

from pc_cng.p4_g7_sampling import (  # noqa: E402
    run_pilot, run_main_review, SEED, PILOT_PER_STRATUM, MAIN_PER_STRATUM,
    STRATA, SCORING_DIMENSIONS, REASON_CODES,
)
from pc_cng.p4_g7_agreement import analyze_responses  # noqa: E402

DEFAULT_MANIFEST = _REPO_ROOT / "data/p4/manifests/hte_feasibility_v2.json"
DEFAULT_RISK = _REPO_ROOT / "results/p4_risk_aware/risk_artifacts.json"
DEFAULT_G6_PRED = _REPO_ROOT / "results/p4_hte_external_validation/raw_predictions/risk_aware_pc_cng.csv"
DEFAULT_OUTPUT = _REPO_ROOT / "results/p4_human_review"


def cmd_pilot(args):
    """Generate pilot review materials."""
    t0 = time.time()
    result = run_pilot(
        manifest_path=Path(args.manifest),
        risk_artifacts_path=Path(args.risk),
        g6_predictions_path=Path(args.g6_pred),
        output_dir=Path(args.output),
        seed=args.seed,
    )
    print(f"\n=== Pilot Generation Complete ({time.time() - t0:.1f}s) ===")
    print(f"  Samples: {result['n_samples']}")
    print(f"  Strata covered: {result['all_strata_covered']}")
    print(f"  Stratum counts: {result['stratum_counts']}")
    print(f"  Reviewers: {result['n_reviewers']}")
    print(f"  Output: {result['output_dir']}")


def cmd_main(args):
    """Generate main review materials."""
    t0 = time.time()
    result = run_main_review(
        manifest_path=Path(args.manifest),
        risk_artifacts_path=Path(args.risk),
        g6_predictions_path=Path(args.g6_pred),
        output_dir=Path(args.output),
        n_reviewers=args.n_reviewers,
        seed=args.seed,
    )
    print(f"\n=== Main Review Generation Complete ({time.time() - t0:.1f}s) ===")
    print(f"  Samples: {result['n_samples']}")
    print(f"  Strata covered: {result['all_strata_covered']}")
    print(f"  Stratum counts: {result['stratum_counts']}")
    print(f"  Reviewers: {result['n_reviewers']}")
    print(f"  Output: {result['output_dir']}")


def cmd_analyze(args):
    """Analyze expert responses (after receiving forms back)."""
    report = analyze_responses(
        responses_path=Path(args.responses),
        manifest_path=Path(args.manifest_review),
        output_path=Path(args.output) / "agreement_report.json",
    )
    print("\n=== Analysis Complete ===")
    v = report.get("verdict", {})
    print(f"  Verdict: {v.get('verdict', 'N/A')}")
    print(f"  Reason: {v.get('reason', 'N/A')}")
    print(f"  Weighted kappa: {v.get('max_weighted_kappa', 0)}")
    print(f"  Krippendorff alpha: {v.get('krippendorff_alpha', 0)}")
    print(f"  Controls discriminated: {v.get('controls_discriminated', False)}")
    print(f"  PC-CNG superior: {v.get('pc_cng_superior', False)}")


def cmd_all(args):
    """Run pilot + generate DEFERRED go_no_go.json."""
    cmd_pilot(args)
    _write_deferred_go_no_go(Path(args.output))


def _write_deferred_go_no_go(output_dir: Path):
    """Write go_no_go.json with DEFERRED status (waiting for expert results)."""
    go_no_go = {
        "phase": "P4-G7",
        "status": "DEFERRED",
        "reason": "Expert review materials prepared. Waiting for real expert responses. "
                  "Cannot use 'expert validated' until forms are returned and analyzed.",
        "primary_method": "human_expert_review",
        "n_samples_prepared": sum(PILOT_PER_STRATUM.values()),
        "n_reviewers": 2,
        "strata": STRATA,
        "scoring_dimensions": SCORING_DIMENSIONS,
        "reason_codes": REASON_CODES,
        "evidence_paths": [
            "results/p4_human_review/sampling_manifest.json",
            "results/p4_human_review/samples.csv",
            "results/p4_human_review/blinded_forms/",
            "docs/p4_07_human_review_protocol.md",
        ],
        "limitations": [
            "No expert responses received yet",
            "Cannot compute kappa/alpha without expert data",
            "PC-CNG vs baseline comparison requires expert scores",
        ],
        "next_phase_allowed": False,
        "note": "Run 'python3 -m pc_cng.run_p4_human_review analyze' after receiving expert forms",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "go_no_go.json", "w") as f:
        json.dump(go_no_go, f, indent=2)
    print(f"\n=== go_no_go.json written (DEFERRED) ===")
    print(f"  next_phase_allowed: false")
    print(f"  Awaiting expert responses at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="P4-G7: Human Expert Calibration")
    parser.add_argument("command", nargs="?", default="all",
                        choices=["all", "pilot", "main", "analyze"])
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                        help="Candidate manifest (G3/G5 frozen)")
    parser.add_argument("--risk", default=str(DEFAULT_RISK),
                        help="G5 risk artifacts JSON")
    parser.add_argument("--g6-pred", default=str(DEFAULT_G6_PRED),
                        help="G6 raw predictions CSV (risk_aware_pc_cng)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-reviewers", type=int, default=3,
                        help="Number of reviewers (main review)")
    parser.add_argument("--responses", default=None,
                        help="Expert responses CSV (for analyze command)")
    parser.add_argument("--manifest-review", default=None,
                        help="Sampling manifest JSON (for analyze command)")
    args = parser.parse_args()

    if args.command == "pilot":
        cmd_pilot(args)
    elif args.command == "main":
        cmd_main(args)
    elif args.command == "analyze":
        if not args.responses:
            print("Error: --responses required for analyze command")
            sys.exit(1)
        if not args.manifest_review:
            args.manifest_review = str(Path(args.output) / "sampling_manifest.json")
        cmd_analyze(args)
    else:
        cmd_all(args)


if __name__ == "__main__":
    main()
