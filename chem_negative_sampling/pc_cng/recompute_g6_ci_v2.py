#!/usr/bin/env python3
"""Recompute G6 v2 paired cluster bootstrap CIs with corrected metric functions.

Bug fixed:
- metric_fn selection previously checked ``"auprc" in metric_name`` BEFORE
  ``"macro_auprc" in metric_name``.  Since "macro_auprc" contains "auprc",
  every macro_auprc endpoint (incl. the primary endpoint
  ``T5_condition_feasibility_macro_auprc``) was incorrectly evaluated with the
  binary ``auprc_metric`` instead of a macro-averaged AUPRC.
- Additionally, ``macro_auprc_metric`` groups by ``yield_bin`` by default, while
  the G6 primary-endpoint point estimate (in ``compute_all_metrics``) groups by
  ``reaction_family``.  The new ``family_macro_auprc_metric`` aligns the CI
  metric function with the point-estimate definition.

This script:
1. Loads ``scored_records_v2.json`` and ``go_no_go_v2.json``.
2. Recomputes ALL paired cluster bootstrap CIs using the CORRECT metric
   functions (``family_macro_auprc_metric`` for macro_auprc, ``auprc_metric``
   for binary auprc, etc.).  Same cluster_key="experimental_group",
   n_bootstrap=2000, seed=20260723 as the original.
3. For B4_vs_B3: keeps the original binary ``auprc_metric`` CI under the
   existing key (backward compatibility) AND adds a macro_auprc version under
   ``B4_vs_B3_<primary>_macro``.
4. Recomputes the verdict based on the corrected primary-endpoint CI.
5. Saves a backup ``go_no_go_v2.json.bak_pre_metric_fix`` before overwriting.
6. Prints an old-vs-new comparison table for the primary endpoint and all T5
   metrics.

Parallelism: each paired_cluster_bootstrap call is independent and resets its
RNG with the same seed (20260723), so CIs are computed in parallel via a
multiprocessing pool (fork inherits the scored-records memory).  Results are
bit-identical to a sequential computation.

No models are retrained; only CIs are recomputed from saved scored records.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path("/home/cunyuliu/pc_cng_research")
sys.path.insert(0, str(REPO_ROOT / "chem_negative_sampling"))

import numpy as np  # noqa: E402

from pc_cng.paired_cluster_inference import (  # noqa: E402
    auprc_metric,
    ece_metric,
    family_macro_auprc_metric,
    mae_metric,
    ndcg_metric,
    paired_cluster_bootstrap,
    spearman_metric,
)

OUTPUT_DIR = REPO_ROOT / "results" / "p4_hte_external_validation_v2"
SCORED_FILE = OUTPUT_DIR / "scored_records_v2.json"
RESULT_FILE = OUTPUT_DIR / "go_no_go_v2.json"
BACKUP_FILE = OUTPUT_DIR / "go_no_go_v2.json.bak_pre_metric_fix"

N_BOOTSTRAP = 2000
SEED = 20260723
CLUSTER_KEY = "experimental_group"
BASELINE_ARM = "B0"
PRIMARY_METRIC = "T5_condition_feasibility_macro_auprc"
N_WORKERS = 12

# Module-level globals inherited by forked workers.
_ARM_SCORED: dict | None = None
_N_BOOTSTRAP: int = N_BOOTSTRAP


def choose_metric_fn(metric_name: str):
    """Choose metric function (check macro_auprc BEFORE auprc)."""
    if "macro_auprc" in metric_name:
        return lambda recs: family_macro_auprc_metric(recs)
    if "auprc" in metric_name:
        return lambda recs: auprc_metric(recs)
    if "mae" in metric_name:
        return lambda recs: mae_metric(recs)
    if "spearman" in metric_name:
        return lambda recs: spearman_metric(recs)
    if "ndcg" in metric_name:
        return lambda recs: ndcg_metric(recs)
    if "ece" in metric_name:
        return lambda recs: ece_metric(recs)
    if "brier" in metric_name:
        return lambda recs: float(np.mean([
            (r.get("score", 0.5) - r.get("label", 0)) ** 2 for r in recs
        ]))
    return None


def task_id_for_metric(metric_name: str):
    if metric_name.startswith("T1"):
        return "T1"
    if metric_name.startswith("T2"):
        return "T2"
    if metric_name.startswith("T3"):
        return "T3"
    if metric_name.startswith("T4"):
        return "T4"
    if metric_name.startswith("T5"):
        return "T5"
    if metric_name in ("ece", "brier"):
        return "T5"
    return None


def _worker_compute(task) -> tuple[str, dict]:
    """Compute a single paired cluster CI. Runs in a forked worker.

    task = (key, kind, arm_id_or_none, task_id_or_none, metric_name_or_none)
    kind in {"main", "b4b3_binary", "b4b3_macro"}.
    """
    key, kind, arm_id, task_id, metric_name = task
    if kind == "main":
        ch_recs = _ARM_SCORED[arm_id][task_id]
        bl_recs = _ARM_SCORED[BASELINE_ARM][task_id]
        metric_fn = choose_metric_fn(metric_name)
    elif kind == "b4b3_binary":
        ch_recs = _ARM_SCORED["B4"]["T5"]
        bl_recs = _ARM_SCORED["B3"]["T5"]
        metric_fn = lambda recs: auprc_metric(recs)
    elif kind == "b4b3_macro":
        ch_recs = _ARM_SCORED["B4"]["T5"]
        bl_recs = _ARM_SCORED["B3"]["T5"]
        metric_fn = lambda recs: family_macro_auprc_metric(recs)
    else:
        raise ValueError(f"unknown kind: {kind}")

    ci = paired_cluster_bootstrap(
        challenger_records=ch_recs,
        baseline_records=bl_recs,
        metric_fn=metric_fn,
        cluster_key=CLUSTER_KEY,
        n_bootstrap=_N_BOOTSTRAP,
        seed=SEED,
    )
    return key, ci


def fmt_ci(ci: dict) -> str:
    if not ci:
        return "<missing>"
    return (
        f"delta={ci.get('delta_mean'):.6f} "
        f"CI=[{ci.get('delta_ci_low'):.6f}, {ci.get('delta_ci_high'):.6f}] "
        f"all_pos={ci.get('ci_all_positive')} "
        f"ch={ci.get('challenger_point'):.6f} bl={ci.get('baseline_point'):.6f} "
        f"n_cl={ci.get('n_clusters')}"
    )


def main() -> None:
    global _ARM_SCORED, _N_BOOTSTRAP

    if not SCORED_FILE.exists():
        print(f"ERROR: {SCORED_FILE} not found"); sys.exit(1)
    if not RESULT_FILE.exists():
        print(f"ERROR: {RESULT_FILE} not found"); sys.exit(1)

    print(f"Loading {SCORED_FILE.name} ...", flush=True)
    with open(SCORED_FILE) as f:
        scored = json.load(f)
    with open(RESULT_FILE) as f:
        result = json.load(f)

    old_paired_cis: dict = dict(result.get("paired_cis", {}))
    old_verdict = result.get("verdict")
    old_reason = result.get("reason")
    _N_BOOTSTRAP = result.get("n_bootstrap", N_BOOTSTRAP)

    _ARM_SCORED = scored["arm_scored_by_task"]
    arms = [a for a in _ARM_SCORED.keys() if a != BASELINE_ARM]
    print(f"Arms (vs {BASELINE_ARM}): {arms}", flush=True)
    print(f"n_bootstrap={_N_BOOTSTRAP}, seed={SEED}, cluster_key={CLUSTER_KEY!r}", flush=True)
    print(f"n_workers={N_WORKERS}", flush=True)

    # Derive the metric set from the original paired_cis keys (vs B0 / vs B3).
    metric_names = set()
    for k in old_paired_cis:
        if "_vs_" not in k:
            continue
        _arm, rest = k.split("_vs_", 1)
        bl = rest.split("_", 1)[0]
        metric = rest[len(bl) + 1:]
        metric_names.add(metric)
    metric_names = sorted(metric_names)
    print(f"Metrics ({len(metric_names)}): {metric_names}", flush=True)

    # Build task list.
    tasks = []
    for metric_name in metric_names:
        if metric_name == "collision_sensitivity":
            continue
        task_id = task_id_for_metric(metric_name)
        if task_id is None or choose_metric_fn(metric_name) is None:
            continue
        for arm_id in arms:
            ch_recs = _ARM_SCORED.get(arm_id, {}).get(task_id, [])
            bl_recs = _ARM_SCORED.get(BASELINE_ARM, {}).get(task_id, [])
            if not ch_recs or not bl_recs:
                continue
            key = f"{arm_id}_vs_{BASELINE_ARM}_{metric_name}"
            tasks.append((key, "main", arm_id, task_id, metric_name))

    b4_vs_b3_key = f"B4_vs_B3_{PRIMARY_METRIC}"
    if _ARM_SCORED.get("B4", {}).get("T5") and _ARM_SCORED.get("B3", {}).get("T5"):
        tasks.append((b4_vs_b3_key, "b4b3_binary", None, None, None))
        tasks.append((f"B4_vs_B3_{PRIMARY_METRIC}_macro", "b4b3_macro", None, None, None))

    print(f"\nComputing {len(tasks)} paired cluster CIs in parallel ...", flush=True)

    paired_cis: dict[str, dict] = {}
    # Use fork (Linux default) so workers inherit _ARM_SCORED via COW.
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(N_WORKERS, len(tasks))) as pool:
        done = 0
        for key, ci in pool.imap_unordered(_worker_compute, tasks):
            paired_cis[key] = ci
            done += 1
            print(f"  [{done}/{len(tasks)}] {key}  all_pos={ci.get('ci_all_positive')}", flush=True)

    # ------------------------------------------------------------------
    # Verdict (recomputed from corrected primary-endpoint CI).
    #   STRONG_GO: B4 primary CI all positive AND B4 beats B3
    #   WEAK_GO:   B4 primary CI all positive
    #   NO_GO:     otherwise
    # B4_vs_B3 "beats" check uses the backward-compat binary CI (as original).
    # ------------------------------------------------------------------
    primary_ci_key = f"B4_vs_{BASELINE_ARM}_{PRIMARY_METRIC}"
    primary_ci = paired_cis.get(primary_ci_key, {})
    b4_ci_positive = primary_ci.get("ci_all_positive", False)
    b4_beats_b3 = paired_cis.get(b4_vs_b3_key, {}).get("ci_all_positive", False)

    if b4_ci_positive and b4_beats_b3:
        verdict = "STRONG_GO"
        reason = (
            f"Primary endpoint {PRIMARY_METRIC}: B4 CI all positive AND B4 beats B3 "
            f"(binary auprc B4_vs_B3 CI all positive, backward-compat)"
        )
    elif b4_ci_positive:
        verdict = "WEAK_GO"
        reason = (
            f"Primary endpoint {PRIMARY_METRIC}: B4 CI all positive but B4 does not beat B3 "
            f"(binary auprc B4_vs_B3)"
        )
    else:
        verdict = "NO_GO"
        reason = f"Primary endpoint {PRIMARY_METRIC}: B4 CI not all positive"

    # ------------------------------------------------------------------
    # Update result, backup original, write.
    # ------------------------------------------------------------------
    if not BACKUP_FILE.exists():
        shutil.copy2(RESULT_FILE, BACKUP_FILE)
        print(f"\n[backup] {RESULT_FILE.name} -> {BACKUP_FILE.name}", flush=True)
    else:
        print(f"\n[backup] {BACKUP_FILE.name} already exists (keeping existing backup)", flush=True)

    result["paired_cis"] = {k: {kk: vv for kk, vv in v.items()} for k, v in paired_cis.items()}
    result["verdict"] = verdict
    result["reason"] = reason
    result["metric_fix_applied"] = (
        "Recomputed paired cluster CIs with corrected metric_fn selection "
        "(macro_auprc checked before auprc; macro_auprc uses "
        "family_macro_auprc_metric grouping by reaction_family to match the "
        "primary-endpoint point estimate). B4_vs_B3 primary kept as binary "
        "auprc (backward compat); macro version under B4_vs_B3_<primary>_macro."
    )

    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[wrote] {RESULT_FILE}", flush=True)

    # ------------------------------------------------------------------
    # Comparison table.
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("OLD vs NEW VERDICT")
    print("=" * 100)
    print(f"  old verdict: {old_verdict}  |  new verdict: {verdict}")
    print(f"  old reason : {old_reason}")
    print(f"  new reason : {reason}")

    print("\n" + "=" * 100)
    print(f"PRIMARY ENDPOINT: {primary_ci_key}")
    print("=" * 100)
    print(f"  OLD: {fmt_ci(old_paired_cis.get(primary_ci_key, {}))}")
    print(f"  NEW: {fmt_ci(primary_ci)}")

    # Full comparison for all T5 metrics (incl. ece/brier) + B4_vs_B3.
    t5_table_keys = sorted(
        k for k in paired_cis.keys()
        if "T5_condition_feasibility" in k
        or k in (b4_vs_b3_key, f"B4_vs_B3_{PRIMARY_METRIC}_macro")
        or k.endswith("_ece") or k.endswith("_brier")
    )

    print("\n" + "=" * 100)
    print("OLD vs NEW CIs — all T5 metrics (incl. ece/brier)")
    print("=" * 100)
    header = f"{'key':<58} {'OLD delta/CI':<40} {'NEW delta/CI':<40} {'changed?'}"
    print(header)
    print("-" * len(header))
    for k in t5_table_keys:
        old = old_paired_cis.get(k, {})
        new = paired_cis.get(k, {})
        old_s = f"{(old.get('delta_mean') if old else float('nan')):.4f} [{(old.get('delta_ci_low') if old else float('nan')):.4f},{(old.get('delta_ci_high') if old else float('nan')):.4f}]"
        new_s = f"{(new.get('delta_mean') if new else float('nan')):.4f} [{(new.get('delta_ci_low') if new else float('nan')):.4f},{(new.get('delta_ci_high') if new else float('nan')):.4f}]"
        changed = "CHANGED" if (not old or not new or
                                abs(old.get('delta_mean', 0) - new.get('delta_mean', 0)) > 1e-9 or
                                abs(old.get('delta_ci_low', 0) - new.get('delta_ci_low', 0)) > 1e-9 or
                                abs(old.get('delta_ci_high', 0) - new.get('delta_ci_high', 0)) > 1e-9) else "same"
        print(f"{k:<58} {old_s:<40} {new_s:<40} {changed}")

    macro_key = f"B4_vs_B3_{PRIMARY_METRIC}_macro"
    print("\n" + "-" * 100)
    print(f"B4_vs_B3 macro_auprc (newly computed): {macro_key}")
    print(f"  {fmt_ci(paired_cis.get(macro_key, {}))}")
    print(f"B4_vs_B3 binary (backward-compat):    {b4_vs_b3_key}")
    print(f"  {fmt_ci(paired_cis.get(b4_vs_b3_key, {}))}")

    print("\nDone.")


if __name__ == "__main__":
    main()
