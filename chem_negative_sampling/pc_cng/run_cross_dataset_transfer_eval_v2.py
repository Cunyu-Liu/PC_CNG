"""P2-05: Cross-dataset transfer v2 expanded (L5 fix).

Fixes the P1-02 NO-GO (only 1/4 migration pairs had significant CIs).
P2-05 expands to 10 migration pairs drawn from
{RegioSQM20, HiTEA, USPTO, ORD, NiCOlit}, focused on the small -> large
direction per the manuscript claim.  Each pair runs with ``pccng-limit=1000``
(up from 200 in v1 default) and ``epochs=15`` across 10 seeds.

The 10 default pairs::

    regiosqm20 -> {hitea, uspto, ord, nicolit}    (4 pairs)
    hitea      -> {uspto, ord, nicolit}           (3 pairs)
    uspto      -> {ord, nicolit}                  (2 pairs)
    ord        -> {nicolit}                       (1 pair)
    ---------------------------------------------------
    total:                                         10 pairs

Go/No-Go rule: >= 3/10 pairs with CI95_low > 0 (entire bootstrap CI positive)
=> fixes L5 NO-GO.

This module reuses :mod:`pc_cng.run_cross_dataset_transfer_eval` (v1) for the
per-pair training/evaluation machinery.  We extend v1's ``DATASET_REGISTRY``
to include ``ord`` and ``nicolit`` (only available on the v2 path; v1 stays
backwards-compatible).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

# Ensure ``PYTHONPATH=.`` works from the chem_negative_sampling root.
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from pc_cng import run_cross_dataset_transfer_eval as _v1  # noqa: E402
from pc_cng.run_cross_dataset_transfer_eval import (  # noqa: E402
    parse_seeds,
    run_transfer_pair,
)


# --------------------------------------------------------------------------- #
# Dataset registry (extends v1 with ord and nicolit).
# --------------------------------------------------------------------------- #
V2_DATASET_REGISTRY: Dict[str, str] = {
    "regiosqm20": "data/processed/regiosqm20_normalized.csv",
    "hitea": "data/processed/hitea_full_normalized.csv",
    "uspto": "data/processed/uspto_openmolecules_normalized.csv",
    "ord": "data/processed/ord_normalized.csv",
    "nicolit": "data/processed/ni_coupling_supplement.csv",
}

# Patch v1's module-level DATASET_REGISTRY so v1's ``resolve_dataset_path``
# can resolve ``ord`` and ``nicolit`` when ``run_transfer_pair`` is called.
for _name, _path in V2_DATASET_REGISTRY.items():
    _v1.DATASET_REGISTRY.setdefault(_name, _path)


# --------------------------------------------------------------------------- #
# Default 10 migration pairs and 10 seeds.
# --------------------------------------------------------------------------- #
DEFAULT_PAIRS: List[Tuple[str, str]] = [
    ("regiosqm20", "hitea"),
    ("regiosqm20", "uspto"),
    ("regiosqm20", "ord"),
    ("regiosqm20", "nicolit"),
    ("hitea", "uspto"),
    ("hitea", "ord"),
    ("hitea", "nicolit"),
    ("uspto", "ord"),
    ("uspto", "nicolit"),
    ("ord", "nicolit"),
]

DEFAULT_SEEDS: str = (
    "20260710,20260711,20260712,20260713,20260714,"
    "20260715,20260716,20260717,20260718,20260719"
)

# Go/No-Go threshold: number of pairs (out of 10) that must have CI95_low > 0
# for the L5 NO-GO to be considered fixed.
GO_THRESHOLD: int = 3


# --------------------------------------------------------------------------- #
# CLI parsing helpers.
# --------------------------------------------------------------------------- #
def parse_pairs(raw: str) -> List[Tuple[str, str]]:
    """Parse a comma-separated list of ``src->tgt`` strings into pairs.

    Raises ``ValueError`` if any item doesn't contain ``->``.
    """
    out: List[Tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "->" not in item:
            raise ValueError(f"Invalid pair {item!r}; expected 'src->tgt'")
        src, tgt = item.split("->", 1)
        out.append((src.strip(), tgt.strip()))
    return out


# --------------------------------------------------------------------------- #
# Aggregate / decision helpers.
# --------------------------------------------------------------------------- #
def is_ci_all_positive(paired_payload: Dict[str, object]) -> bool:
    """Return True if the pooled delta CI95_low > 0 (entire CI is positive)."""
    sig = paired_payload.get("paired_significance_pooled") or {}
    try:
        ci_low = float(sig.get("delta_ci95_low", 0.0))
    except (TypeError, ValueError):
        return False
    return ci_low > 0.0


def build_aggregate_summary(per_pair_results: List[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate per-pair payloads into a single summary dict."""
    pairs_summary: List[Dict[str, object]] = []
    ci_positive_count = 0
    for res in per_pair_results:
        src = res.get("source")
        tgt = res.get("target")
        sig = res.get("paired_significance_pooled") or {}
        ci_low = float(sig.get("delta_ci95_low", 0.0) or 0.0)
        ci_high = float(sig.get("delta_ci95_high", 0.0) or 0.0)
        ci_pos = (ci_low > 0.0) and (ci_high > 0.0)
        if ci_pos:
            ci_positive_count += 1
        pairs_summary.append({
            "source": src,
            "target": tgt,
            "pair": f"{src}_to_{tgt}",
            "delta_mean": sig.get("delta_mean"),
            "delta_ci95_low": ci_low,
            "delta_ci95_high": ci_high,
            "paired_permutation_p": sig.get("paired_permutation_p"),
            "sign_test_p": sig.get("sign_test_p"),
            "n_pooled": sig.get("n"),
            "ci_all_positive": ci_pos,
        })
    return {
        "n_pairs_total": len(per_pair_results),
        "n_pairs_ci_all_positive": ci_positive_count,
        "pairs": pairs_summary,
    }


def build_go_no_go(aggregate: Dict[str, object], threshold: int = GO_THRESHOLD) -> Dict[str, object]:
    """Build the GO/NO-GO decision payload.

    Rule: GO if ``count_ci_all_positive >= threshold``.
    """
    count = int(aggregate.get("n_pairs_ci_all_positive", 0) or 0)
    total = int(aggregate.get("n_pairs_total", 0) or 0)
    decision = "GO" if count >= threshold else "NO-GO"
    return {
        "decision": decision,
        "count_ci_all_positive": count,
        "n_pairs_total": total,
        "threshold_for_go": threshold,
        "rule": f"GO if count_ci_all_positive >= {threshold}",
        "fixes_L5_NOGO": decision == "GO",
    }


def write_per_pair_summary(pair_dir: str, payload: Dict[str, object]) -> None:
    """Write ``summary.json`` for a single pair, derived from the v1 payload.

    The v1 ``paired_significance.json`` already contains the full payload; the
    v2 ``summary.json`` is a focused subset used by downstream dashboards.
    """
    summary = {
        "source": payload.get("source"),
        "target": payload.get("target"),
        "seeds": payload.get("seeds"),
        "config": payload.get("config"),
        "per_seed": payload.get("per_seed"),
        "paired_significance_pooled": payload.get("paired_significance_pooled"),
        "seed_level_significance": payload.get("seed_level_significance"),
    }
    os.makedirs(pair_dir, exist_ok=True)
    with open(os.path.join(pair_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "P2-05 cross-dataset transfer v2 (10 pairs, pccng-limit=1000, "
            "epochs=15, 10 seeds). Fixes P1-02 L5 NO-GO."
        )
    )
    parser.add_argument("--output-dir", required=True, help="Top-level output directory")
    parser.add_argument("--pccng-limit", type=int, default=1000,
                        help="Max source positives used for PC-CNG generation (default: 1000)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Training epochs per reranker (default: 15)")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS,
                        help="Comma-separated seeds (default: 10 seeds 20260710..20260719)")
    parser.add_argument("--pairs", default=None,
                        help="Comma-separated 'src->tgt' pairs (overrides default 10 pairs)")
    parser.add_argument("--pair-only", default=None,
                        help="Run only one pair, e.g. 'regiosqm20->hitea' (for testing)")
    parser.add_argument("--research-root", default=None,
                        help="Override research root (default: cwd)")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default=None,
                        help="cuda:0 / cpu / etc.  Defaults to cuda if available.")
    parser.add_argument("--target-limit", type=int, default=None,
                        help="Cap target rows for tractability on large targets.")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()

    research_root = args.research_root or os.getcwd()
    seeds = parse_seeds(args.seeds)

    # Determine which pairs to run.
    if args.pair_only:
        pairs = parse_pairs(args.pair_only)
        if len(pairs) != 1:
            parser.error(f"--pair-only expects exactly one pair; got {len(pairs)}")
    elif args.pairs:
        pairs = parse_pairs(args.pairs)
        if not pairs:
            parser.error("--pairs was empty after parsing")
    else:
        pairs = list(DEFAULT_PAIRS)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[v2] running {len(pairs)} pair(s) with pccng_limit={args.pccng_limit} "
          f"epochs={args.epochs} seeds={seeds}")
    print(f"[v2] pairs: {pairs}")

    per_pair_results: List[Dict[str, object]] = []
    for source, target in pairs:
        print(f"[v2] === pair {source}->{target} ===")
        res = run_transfer_pair(
            source=source,
            target=target,
            seeds=seeds,
            output_dir=args.output_dir,
            research_root=research_root,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            n_bits=args.n_bits,
            batch_size=args.batch_size,
            device_name=args.device,
            pccng_limit=args.pccng_limit,
            target_limit=args.target_limit,
            bootstrap_iterations=args.bootstrap_iterations,
            smoke=False,
        )

        # Write per-pair summary.json (in addition to v1's paired_significance.json).
        pair_dir = os.path.join(args.output_dir, f"{source}_to_{target}")
        write_per_pair_summary(pair_dir, res)
        per_pair_results.append(res)

    # Write aggregate_summary.json
    aggregate = build_aggregate_summary(per_pair_results)
    agg_path = os.path.join(args.output_dir, "aggregate_summary.json")
    with open(agg_path, "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, ensure_ascii=False)
    print(f"[v2] wrote {agg_path}")

    # Write go_no_go_decision.json
    decision = build_go_no_go(aggregate, threshold=GO_THRESHOLD)
    dec_path = os.path.join(args.output_dir, "go_no_go_decision.json")
    with open(dec_path, "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, ensure_ascii=False)
    print(f"[v2] wrote {dec_path}")

    print(f"[v2] {aggregate['n_pairs_ci_all_positive']}/{aggregate['n_pairs_total']} "
          f"pairs have CI all positive (threshold={GO_THRESHOLD}).")
    print(f"[v2] decision: {decision['decision']} -> fixes_L5_NOGO={decision['fixes_L5_NOGO']}")


if __name__ == "__main__":
    main()
