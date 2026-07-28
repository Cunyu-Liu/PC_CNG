"""Phase 4 follow-up: RANDOMIZED-LABEL null control arm on the frozen pools.

Motivation
----------
The pre-registered H2 hypothesis ("shuffled_parent-trained classifier should
achieve ~0.5 AUPRC on the fixed semi_hard pool") was falsified by the v4.1
evidence: on balanced, RDKit-valid, similarity-matched boundary pools the
shuffled_parent-TRAINED classifier still reaches 0.83-0.95 AUPRC.  The
diag_shuffled_transfer.py analysis showed this is NOT a residual shortcut:
the transfer survives on the formula-preserved slice, i.e. training on
real-reactant -> shuffled-real-product pairs teaches a genuine broad
reactant/product compatibility prior, which is exactly the knowledge needed
to rank real reactions above boundary-edited ones.

The scientifically correct "~0.5 hard control" is therefore NOT the
shuffled_parent arm but a RANDOMIZED-LABEL arm: identical training pairs
(positives + shuffled-real negatives, same seed/offset as the
shuffled_parent arm) but with the binary labels randomly permuted.  Any
AUPRC significantly above 0.5 for this arm would indicate evaluation
leakage in the fixed-pool pipeline; AUPRC ~ 0.5 validates the pipeline and
provides the literal null control.

Run:
    python3 -m pc_cng.run_phase4_null_control \
        --base-results results/phase4_fixed_testset_v41 --gpu 0 --use-gnn
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

os.environ.setdefault("RDKitRDLogger", "0")

from pc_cng.paired_cluster_inference import auprc_metric  # noqa: E402
from pc_cng.run_phase4_fixed_testset import (  # noqa: E402
    METHOD_SHUFFLED_PARENT,
    PRIMARY_POOL,
    build_main_arm_train,
    source_macro_auprc_metric,
    train_classifier,
    score_records,
)
from pc_cng.run_phase4_union_arm import (  # noqa: E402
    _scenario_rows,
    load_saved_pool,
)
from pc_cng.run_phase3_external_validation import (  # noqa: E402
    DEFAULT_NI_CSV,
    DEFAULT_OOD_DIR,
    DEFAULT_PARQUET,
)

METHOD_NULL = "null_randomized_label"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-results", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--ni-csv", type=Path, default=DEFAULT_NI_CSV)
    ap.add_argument("--ood-dir", type=Path, default=DEFAULT_OOD_DIR)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--use-gnn", action="store_true")
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--max-train", type=int, default=500)
    ap.add_argument("--max-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    base_records = args.base_results / "per_scenario_records"
    if not base_records.exists():
        raise SystemExit(f"no per_scenario_records under {args.base_results}")

    import torch
    print(f"[null] torch.cuda.is_available() = {torch.cuda.is_available()}")
    if args.gpu is not None and torch.cuda.is_available():
        print(f"[null] device = {torch.cuda.get_device_name(args.gpu)}")

    scenarios = sorted({f.name.split("__")[0] for f in base_records.glob(
        f"*__{METHOD_SHUFFLED_PARENT}__{PRIMARY_POOL}.csv")})
    if args.splits:
        scenarios = [s for s in scenarios if s in set(args.splits)]
    print(f"[null] scenarios: {scenarios}")

    all_results: Dict[str, Any] = {}
    for scenario in scenarios:
        print(f"\n[null] === Scenario: {scenario} ===")
        try:
            rows = _scenario_rows(args, scenario)
        except Exception as exc:
            print(f"  [skip] cannot reload split rows: {exc}")
            continue
        train_rows = rows["train"]

        # Identical (pos, shuffled-neg) pairs as the main run's
        # shuffled_parent arm (same builder, same seed), then permute labels.
        X_tr, y_tr, rec_tr = build_main_arm_train(
            METHOD_SHUFFLED_PARENT, train_rows, {}, args.seed)
        if X_tr is None or len(X_tr) < 10:
            print("  [skip] insufficient train data")
            continue
        rng = np.random.default_rng(args.seed + 777)
        y_null = rng.permutation(y_tr)  # destroy the label signal only
        t0 = time.time()
        clf = train_classifier(X_tr, y_null, rec_tr, args.seed,
                               use_gnn=args.use_gnn)
        train_sec = time.time() - t0

        pool_recs = load_saved_pool(base_records, scenario)
        if not pool_recs:
            print("  [skip] no saved pool records")
            continue
        scored = score_records(clf, pool_recs, use_gnn=args.use_gnn)

        out_f = base_records / f"{scenario}__{METHOD_NULL}__{PRIMARY_POOL}.csv"
        with open(out_f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(scored[0].keys()))
            w.writeheader()
            w.writerows(scored)

        sma = float(source_macro_auprc_metric(scored))
        pooled = float(auprc_metric(scored))
        all_results[scenario] = {"source_macro": sma, "pooled": pooled,
                                 "train_sec": train_sec, "n_train": len(X_tr)}
        print(f"  [null] trained on PERMUTED labels ({len(X_tr)} samples, "
              f"{train_sec:.1f}s) srcMacroAUPRC[{PRIMARY_POOL}]={sma:.4f} "
              f"pooled={pooled:.4f}")

    pts = [v["source_macro"] for v in all_results.values()]
    med = float(np.median(pts)) if pts else None
    print(f"\n[null] === NULL CONTROL SUMMARY ===")
    print(f"  median srcMacroAUPRC = {med}")
    print(f"  expectation: ~ slice base rate n_pos/(n_pos+n_neg) "
          f"(0.5 only for balanced slices); a systematic excess would "
          f"indicate evaluation leakage")
    with open(args.base_results / "null_control_results.json", "w") as fh:
        json.dump({"arm": METHOD_NULL, "median_src_macro": med,
                   "results": all_results}, fh, indent=2)
    print(f"[null] saved -> {args.base_results / 'null_control_results.json'}")


if __name__ == "__main__":
    main()
