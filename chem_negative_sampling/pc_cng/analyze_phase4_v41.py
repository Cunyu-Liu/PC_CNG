"""Phase 4 v4.1 final aggregation: horizontal table + H1/H2/H3 verdicts.

Recomputes every metric from the SAVED per-scenario CSV records (no
retraining).  All arms share the same fixed semi_hard pool records, so every
comparison is exactly paired at the record level.

Hypotheses (pre-registered in run_phase4_fixed_testset.py; union/null arms
added as follow-ups):

  H1 (learned SOTA):   learned_structured vs {rule, random, shuffled}
  H1u (union SOTA):    learned_union     vs {rule, random, learned,
                                             shuffled, diff_semihard}
  H2 (hard control):   shuffled_parent srcMacro (transferred compatibility
                       prior; expected >> 0.5 per diag_shuffled_transfer) and
                       null_randomized_label srcMacro (must be ~0.5 - the
                       literal null control).
  H3 (inverted-U):     diff_semihard     vs {diff_easy, diff_hard}

Holm correction is applied ACROSS ALL scenarios jointly per hypothesis
family (not per scenario), matching the multi-scenario SOTA claim.

Run:
    python3 -m pc_cng.analyze_phase4_v41 \
        --base-results results/phase4_fixed_testset_v41
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from pc_cng.paired_cluster_inference import (  # noqa: E402
    holm_correction,
    paired_cluster_bootstrap,
)
from pc_cng.run_phase4_fixed_testset import (  # noqa: E402
    METHOD_LEARNED,
    METHOD_RULE,
    METHOD_SHUFFLED_PARENT,
    PRIMARY_POOL,
    source_macro_auprc_metric,
)
from pc_cng.run_phase3_external_validation import METHOD_RANDOM  # noqa: E402
from pc_cng.run_phase4_union_arm import (  # noqa: E402
    METHOD_UNION,
    load_baseline_scores,
)
from pc_cng.run_phase4_null_control import METHOD_NULL  # noqa: E402

DIFF_ARMS = ("diff_easy", "diff_semihard", "diff_hard")
UNION_V2 = METHOD_UNION + "_v2"
ALL_ARMS = (METHOD_RULE, METHOD_RANDOM, METHOD_LEARNED, METHOD_SHUFFLED_PARENT,
            *DIFF_ARMS, METHOD_UNION, UNION_V2, METHOD_NULL)

H1_PAIRS = [(METHOD_LEARNED, b) for b in
            (METHOD_RULE, METHOD_RANDOM, METHOD_SHUFFLED_PARENT)]
H1U_PAIRS = [(METHOD_UNION, b) for b in
             (METHOD_RULE, METHOD_RANDOM, METHOD_LEARNED,
              METHOD_SHUFFLED_PARENT, "diff_semihard")]
H1U_V2_PAIRS = [(UNION_V2, b) for b in
                (METHOD_RULE, METHOD_RANDOM, METHOD_LEARNED,
                 METHOD_SHUFFLED_PARENT, "diff_semihard", METHOD_UNION)]
H3_PAIRS = [("diff_semihard", "diff_easy"), ("diff_semihard", "diff_hard")]


def _load_arm(records_dir: Path, scenario: str, arm: str
              ) -> Optional[List[Dict]]:
    recs = load_baseline_scores(records_dir, scenario, arm)
    return recs if recs and len(recs) >= 10 else None


def _record_signature(record: Dict[str, Any]) -> tuple:
    """Identity fields that must match for a paired comparison.

    A paired bootstrap requires the same held-out records, not merely the
    same row count and cluster labels.  Scores are intentionally excluded so
    challenger/baseline predictions can differ.
    """
    return (
        str(record.get("reaction_smiles", "")),
        int(record.get("label", 0)),
        bool(record.get("is_positive", False)),
        str(record.get("source", "?")),
        str(record.get("experimental_group", "default")),
    )


def _alignment_error(challenger: List[Dict], baseline: List[Dict]) -> Optional[str]:
    if len(challenger) != len(baseline):
        return f"record_count_mismatch:{len(challenger)}!={len(baseline)}"
    for idx, (ch, bl) in enumerate(zip(challenger, baseline)):
        ch_sig, bl_sig = _record_signature(ch), _record_signature(bl)
        if ch_sig != bl_sig:
            return f"record_identity_mismatch_at:{idx}:{ch_sig!r}!={bl_sig!r}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-results", type=Path, required=True)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    records_dir = args.base_results / "per_scenario_records"
    scenarios = sorted({f.name.split("__")[0]
                        for f in records_dir.glob(f"*__{PRIMARY_POOL}.csv")})
    print(f"[agg] scenarios discovered: {scenarios}")

    # ----- per-arm srcMacro matrix
    matrix: Dict[str, Dict[str, float]] = {}
    rec_cache: Dict[str, Dict[str, List[Dict]]] = {}
    for sc in scenarios:
        matrix[sc] = {}
        rec_cache[sc] = {}
        for arm in ALL_ARMS:
            recs = _load_arm(records_dir, sc, arm)
            if recs is None:
                continue
            rec_cache[sc][arm] = recs
            matrix[sc][arm] = float(source_macro_auprc_metric(recs))

    # ----- horizontal table
    hdr_arms = [a for a in ALL_ARMS if any(a in matrix[sc] for sc in scenarios)]
    print(f"\n[agg] === source-macro AUPRC[{PRIMARY_POOL}] horizontal table ===")
    print(f"  {'scenario':<16}" + "".join(f" {a[:13]:>13}" for a in hdr_arms))
    for sc in scenarios:
        row = f"  {sc:<16}"
        for a in hdr_arms:
            v = matrix[sc].get(a)
            row += f" {v:>13.4f}" if isinstance(v, float) else f" {'-':>13}"
        print(row)

    # ----- paired CIs per hypothesis family
    alignment_audit: List[Dict[str, Any]] = []

    def _run_pairs(pairs, family):
        tests: List[Dict[str, Any]] = []
        for sc in scenarios:
            for chal, base in pairs:
                rc, rb = rec_cache.get(sc, {}).get(chal), \
                    rec_cache.get(sc, {}).get(base)
                if rc is None or rb is None:
                    continue
                alignment_error = _alignment_error(rc, rb)
                alignment_audit.append({
                    "family": family,
                    "scenario": sc,
                    "pair": f"{chal}_vs_{base}",
                    "ok": alignment_error is None,
                    "error": alignment_error,
                })
                if alignment_error is not None:
                    tests.append({
                        "family": family,
                        "scenario": sc,
                        "pair": f"{chal}_vs_{base}",
                        "error": f"paired_alignment_failed:{alignment_error}",
                    })
                    continue
                try:
                    ci = paired_cluster_bootstrap(
                        rc, rb, metric_fn=source_macro_auprc_metric,
                        cluster_key="experimental_group",
                        n_bootstrap=args.n_bootstrap, seed=args.seed)
                except Exception as exc:
                    ci = {"error": str(exc)}
                tests.append({"family": family, "scenario": sc,
                              "pair": f"{chal}_vs_{base}", **ci})
        return tests

    all_tests = (_run_pairs(H1_PAIRS, "H1") + _run_pairs(H1U_PAIRS, "H1u")
                 + _run_pairs(H1U_V2_PAIRS, "H1u_v2")
                 + _run_pairs(H3_PAIRS, "H3"))

    # ----- Holm per family across scenarios
    holm_by_family: Dict[str, Any] = {}
    for fam in ("H1", "H1u", "H1u_v2", "H3"):
        fam_tests = [t for t in all_tests
                     if t["family"] == fam and "error" not in t]
        holm = holm_correction([t["p_value"] for t in fam_tests],
                               alpha=args.alpha) if fam_tests else []
        for t, h in zip(fam_tests, holm):
            t["adjusted_p"] = h["adjusted_p"]
            t["significant"] = h["rejected"]
        holm_by_family[fam] = {
            "n_significant": sum(1 for t in fam_tests if t["significant"]
                                 and t["delta_mean"] > 0),
            "n_total": len(fam_tests),
        }

    # ----- verdicts
    def _scenario_sota(chal: str, bases: tuple, fam: str) -> Dict[str, Any]:
        per = {}
        for sc in scenarios:
            wins = 0
            n = 0
            for b in bases:
                t = next((t for t in all_tests
                          if t["family"] == fam and t["scenario"] == sc
                          and t["pair"] == f"{chal}_vs_{b}"), None)
                if t is None or "error" in t:
                    continue
                n += 1
                if t.get("significant") and t["delta_mean"] > 0:
                    wins += 1
            per[sc] = {"wins": wins, "n": n,
                       "status": "SOTA" if n > 0 and wins == n else
                                 ("partial" if wins > 0 else "fail")}
        return per

    h1_per = _scenario_sota(METHOD_LEARNED,
                            (METHOD_RULE, METHOD_RANDOM,
                             METHOD_SHUFFLED_PARENT), "H1")
    h1u_per = _scenario_sota(METHOD_UNION,
                             (METHOD_RULE, METHOD_RANDOM, METHOD_LEARNED,
                              METHOD_SHUFFLED_PARENT, "diff_semihard"), "H1u")
    h1u_v2_per = _scenario_sota(UNION_V2,
                                (METHOD_RULE, METHOD_RANDOM, METHOD_LEARNED,
                                 METHOD_SHUFFLED_PARENT, "diff_semihard",
                                 METHOD_UNION), "H1u_v2")
    h3_per = _scenario_sota("diff_semihard", ("diff_easy", "diff_hard"), "H3")

    n_h1 = sum(1 for v in h1_per.values() if v["status"] == "SOTA")
    n_h1u = sum(1 for v in h1u_per.values() if v["status"] == "SOTA")
    n_h1u_v2 = sum(1 for v in h1u_v2_per.values() if v["status"] == "SOTA")
    n_h3 = sum(1 for v in h3_per.values() if v["status"] == "SOTA")

    # H2: control-arm levels.  For AUPRC the correct null level is the
    # slice base rate (n_pos / (n_pos + n_neg)), NOT 0.5; the
    # randomized-label arm must sit at that rate.  |null - base_rate|
    # >> 0 would indicate evaluation leakage.
    def _base_rate(sc: str) -> Optional[float]:
        recs = rec_cache.get(sc, {}).get(METHOD_NULL) or \
            rec_cache.get(sc, {}).get(METHOD_SHUFFLED_PARENT)
        if not recs:
            return None
        pos = [r for r in recs if r.get("is_positive")]
        neg_by_src: Dict[str, int] = {}
        for r in recs:
            if not r.get("is_positive"):
                src = str(r.get("source", "?"))
                neg_by_src[src] = neg_by_src.get(src, 0) + 1
        rates = [len(pos) / (len(pos) + n) for n in neg_by_src.values()
                 if n >= 8]
        return float(np.mean(rates)) if rates else None

    h2 = {
        "shuffled_parent_srcmacro": {sc: matrix[sc].get(METHOD_SHUFFLED_PARENT)
                                     for sc in scenarios},
        "null_randomized_label_srcmacro": {sc: matrix[sc].get(METHOD_NULL)
                                           for sc in scenarios},
        "null_base_rate": {sc: _base_rate(sc) for sc in scenarios},
    }
    dev = {sc: abs(h2["null_randomized_label_srcmacro"][sc]
                   - h2["null_base_rate"][sc])
           for sc in scenarios
           if isinstance(h2["null_randomized_label_srcmacro"].get(sc), float)
           and isinstance(h2["null_base_rate"].get(sc), float)}
    h2["null_deviation_from_base_rate"] = dev
    h2["null_ok"] = bool(dev) and max(dev.values()) < 0.10

    verdict = {
        "H1_learned_sota": {"per_scenario": h1_per, "n_sota": n_h1,
                            "n_scenarios": len(h1_per)},
        "H1u_union_sota": {"per_scenario": h1u_per, "n_sota": n_h1u,
                           "n_scenarios": len(h1u_per)},
        "H1u_v2_union_sota": {"per_scenario": h1u_v2_per, "n_sota": n_h1u_v2,
                               "n_scenarios": len(h1u_v2_per)},
        "H2_controls": h2,
        "H3_inverted_u": {"per_scenario": h3_per, "n_sota": n_h3,
                          "n_scenarios": len(h3_per)},
    }

    print(f"\n[agg] === VERDICT ===")
    print(f"  H1  learned_structured SOTA: {n_h1}/{len(h1_per)} scenarios")
    print(f"  H1u union SOTA:              {n_h1u}/{len(h1u_per)} scenarios")
    print(f"  H1u_v2 difficulty-matched union SOTA: {n_h1u_v2}/{len(h1u_v2_per)} scenarios")
    print(f"  H3  inverted-U (semihard>easy&hard): {n_h3}/{len(h3_per)}")
    if dev:
        print(f"  H2  null-control max |AUPRC - base_rate|: "
              f"{max(dev.values()):.4f} "
              f"({'OK no leakage' if h2['null_ok'] else 'LEAKAGE?'})")
    shuf_pts = [v for v in h2["shuffled_parent_srcmacro"].values()
                if isinstance(v, float)]
    if shuf_pts:
        print(f"  H2  shuffled_parent median srcMacro: "
              f"{float(np.median(shuf_pts)):.4f} (compatibility transfer)")

    out = {"matrix": matrix, "tests": all_tests,
           "alignment_audit": alignment_audit,
           "holm_by_family": holm_by_family, "verdict": verdict}
    with open(args.base_results / "phase4_v41_aggregation.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"[agg] saved -> {args.base_results / 'phase4_v41_aggregation.json'}")


if __name__ == "__main__":
    main()
