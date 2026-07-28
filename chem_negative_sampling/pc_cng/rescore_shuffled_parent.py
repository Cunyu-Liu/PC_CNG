#!/usr/bin/env python3
"""Re-score shuffled_parent arm with FIXED negative generation (both reactants
and products shuffled).  Reuses existing test-pool records from a completed
Phase 4 run, retrains the shuffled_parent classifier, and overwrites the
shuffled_parent CSV files with the corrected scores.

This fixes the issue where the original shuffled_parent only shuffled products,
preserving reactant-product compatibility and yielding AUPRC 0.80-0.99.
"""
import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

# Setup path
_THIS_DIR = Path(__file__).resolve().parent
# pc_cng_work is at <repo_root>/chem_negative_sampling/p4_g0_staging/pc_cng_work/
# pc_cng module is at <repo_root>/chem_negative_sampling/pc_cng/
# So CNS_ROOT = parents[2] from this file
_CNS_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

os.environ.setdefault("RDKitRDLogger", "0")

from pc_cng.run_phase3_external_validation import (  # noqa: E402
    DEFAULT_NI_CSV,
    DEFAULT_OOD_DIR,
    DEFAULT_PARQUET,
    METHOD_RANDOM,
    NegativeGenerator,
    load_hitea_split,
    load_ni_coupling,
    load_uspto_patent,
)
from pc_cng.run_phase4_fixed_testset import (  # noqa: E402
    METHOD_LEARNED,
    METHOD_RULE,
    METHOD_SHUFFLED_PARENT,
    PRIMARY_POOL,
    SHUFFLED_OFFSET,
    _product_of,
    _row_meta,
    score_records,
    source_macro_auprc_metric,
    train_classifier,
)
from pc_cng.paired_cluster_inference import auprc_metric  # noqa: E402


def stable_scenario_seed(base_seed: int, scenario: str) -> int:
    """Derive a reproducible per-scenario seed across Python processes.

    ``hash(scenario)`` is intentionally randomized by Python, so using it in
    a rescore command made two otherwise identical runs produce different
    shuffled pairs.  A digest keeps the handover command reproducible.
    """
    digest = hashlib.sha256(scenario.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big") % 1000
    return int(base_seed) + offset


def _scenario_data(args, scenario: str):
    """Load scenario data directly."""
    if scenario == "ni_coupling":
        data = load_ni_coupling(args.ni_csv, args.max_train, args.max_test)
        for part in ("train", "test"):
            data[part]["reaction_family"] = "NI_COUPLING"
            data[part]["experimental_group"] = data[part]["split_key"]
            data[part]["yield_bin"] = 0
        return data
    if scenario == "uspto_patent":
        uspto_csv = Path(
            "/home/cunyuliu/pc_cng_research/data/processed/"
            "uspto_openmolecules_normalized.csv")
        data = load_uspto_patent(uspto_csv, args.ood_dir, args.max_train,
                                 args.max_test)
        for part in ("train", "test"):
            data[part]["reaction_family"] = "USPTO_PATENT"
            data[part]["experimental_group"] = data[part]["split_key"]
            data[part]["yield_bin"] = 0
        return data
    return load_hitea_split(args.parquet, args.ood_dir, scenario,
                            args.max_train, args.max_test)


def _build_fixed_shuffled_parent_train(train_rows, seed: int):
    """Build training data for shuffled_parent with BOTH reactants and
    products fully shuffled to destroy chemistry-specific signals."""
    import random
    from rdkit import Chem

    rng = random.Random(seed)
    parsed: List[Tuple[str, str, str, str, Dict]] = []
    for _, row in train_rows.iterrows():
        rxn = row.get("reaction_smiles", "")
        if not isinstance(rxn, str) or ">" not in rxn:
            continue
        parts = rxn.split(">")
        if len(parts) < 3 or not parts[0] or not parts[2]:
            continue
        reactants = parts[0].strip()
        agents = parts[1].strip()
        products = parts[2].strip()
        meta = _row_meta(row)
        parsed.append((reactants, agents, products, rxn, meta))

    n = len(parsed)
    if n < 2:
        return None, None, None

    X_list = []
    y_list = []
    records: List[Dict[str, Any]] = []

    for i in range(n):
        r, a, p, pos_rxn, meta = parsed[i]
        # Fully shuffled: reactants from one row, products from another
        j = (i + SHUFFLED_OFFSET) % n
        r_shuf = parsed[j][0]
        a_shuf = parsed[j][1]
        # Products from a DIFFERENT row (not the same as reactants source)
        k = (j + SHUFFLED_OFFSET) % n
        p_shuf = parsed[k][2]

        # Ensure reactants differ from original
        if r_shuf == r and a_shuf == a:
            j2 = (j + 1) % n
            r_shuf = parsed[j2][0]
            a_shuf = parsed[j2][1]

        # Verify validity
        try:
            if r_shuf == r and p_shuf == p:
                continue
            neg_rxn = f"{r_shuf}>{a_shuf}>{p_shuf}"
            mol = Chem.MolFromSmiles(p_shuf)
            if mol is None:
                continue
        except Exception:
            continue

        # Use the same reaction representation as score_records() and the
        # main Phase 4 arm builder.  A previous version trained on an
        # absolute positive-vs-negative difference vector but scored raw
        # reaction fingerprints, silently making the rescore train/test
        # feature semantics inconsistent.
        from pc_cng.phase3_enhanced import reaction_fp_enhanced
        pos_feats = reaction_fp_enhanced(pos_rxn)
        neg_feats = reaction_fp_enhanced(neg_rxn)
        if pos_feats is None or neg_feats is None:
            continue

        # The rescore must mirror build_main_arm_train: every shuffled
        # negative is paired with its real positive.  Training on negatives
        # alone creates a one-class classifier and is not a valid control.
        X_list.extend([pos_feats, neg_feats])
        y_list.extend([1, 0])

        records.append({
            "reaction_smiles": pos_rxn,
            "negative_smiles": pos_rxn,
            "label": 1,
            "score": 0.0,
            "method": METHOD_SHUFFLED_PARENT,
            "is_positive": True,
            "split_key": meta.get("split_key", ""),
            "reaction_family": meta.get("reaction_family", ""),
            "experimental_group": meta.get("experimental_group", ""),
        })
        records.append({
            "reaction_smiles": neg_rxn,
            "negative_smiles": p_shuf,
            "label": 0,
            "score": 0.0,
            "method": METHOD_SHUFFLED_PARENT,
            "is_positive": False,
            "positive_rxn": pos_rxn,
            "negative_product": p_shuf,
            "negative_source": METHOD_SHUFFLED_PARENT,
            "split_key": meta.get("split_key", ""),
            "reaction_family": meta.get("reaction_family", ""),
            "experimental_group": meta.get("experimental_group", ""),
        })

    if not X_list:
        return None, None, None
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, records


def _featurize_reaction(pos_rxn: str, neg_rxn: str) -> np.ndarray:
    """Simple difference-feature between positive and negative reactions."""
    from pc_cng.phase3_enhanced import reaction_fp_enhanced
    pos_fp = reaction_fp_enhanced(pos_rxn)
    neg_fp = reaction_fp_enhanced(neg_rxn)
    if pos_fp is None or neg_fp is None:
        return None
    return np.abs(pos_fp - neg_fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-results", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--ni-csv", type=Path, default=DEFAULT_NI_CSV)
    ap.add_argument("--ood-dir", type=Path, default=DEFAULT_OOD_DIR)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--use-gnn", action="store_true")
    ap.add_argument("--max-train", type=int, default=500)
    ap.add_argument("--max-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    print(f"[rescore] parquet: {args.parquet}")
    print(f"[rescore] ni_csv:  {args.ni_csv}")
    print(f"[rescore] ood_dir: {args.ood_dir}")

    if not args.parquet.exists():
        raise SystemExit(f"parquet not found: {args.parquet}")
    if not args.ood_dir.exists():
        raise SystemExit(f"ood_dir not found: {args.ood_dir}")

    records_dir = args.base_results / "per_scenario_records"
    if not records_dir.exists():
        raise SystemExit(f"no per_scenario_records under {args.base_results}")

    import torch
    device = None
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"[rescore] device = {torch.cuda.get_device_name(args.gpu)}")

    # Discover scenarios from existing CSVs
    scenarios = sorted({f.name.split("__")[0] for f in records_dir.glob(
        f"*__learned_structured__{PRIMARY_POOL}.csv")})
    print(f"[rescore] scenarios: {scenarios}")

    for scenario in scenarios:
        print(f"\n[rescore] === {scenario} ===")
        try:
            rows = _scenario_data(args, scenario)
        except Exception as exc:
            import traceback
            print(f"  [skip] cannot reload: {exc}")
            traceback.print_exc()
            continue
        train_rows = rows["train"]
        test_rows = rows["test"]

        # Build FIXED shuffled_parent training data (both reactants + products shuffled)
        result = _build_fixed_shuffled_parent_train(
            train_rows, seed=stable_scenario_seed(args.seed, scenario))
        if result is None or result[0] is None:
            print(f"  [skip] no training data for {scenario}")
            continue
        X_train, y_train, rec_train = result
        print(f"  [train] {len(y_train)} shuffled_parent negatives (fully shuffled R+P)")

        # Train classifier
        try:
            clf = train_classifier(
                X_train, y_train, rec_train, seed=args.seed,
                use_gnn=args.use_gnn,
            )
        except Exception as exc:
            import traceback
            print(f"  [skip] training failed: {exc}")
            traceback.print_exc()
            continue

        # Load existing test pool records
        pool_f = records_dir / f"{scenario}__learned_structured__{PRIMARY_POOL}.csv"
        if not pool_f.exists():
            print(f"  [skip] no pool records at {pool_f}")
            continue
        with open(pool_f) as fh:
            pool_recs = list(csv.DictReader(fh))
        if not pool_recs:
            print(f"  [skip] empty pool records")
            continue
        # Ensure labels are ints (CSV stores them as strings)
        for r in pool_recs:
            try:
                r["label"] = int(r.get("label", 0))
            except (ValueError, TypeError):
                r["label"] = 0
            # Ensure is_positive is bool
            if isinstance(r.get("is_positive"), str):
                r["is_positive"] = r["is_positive"].lower() in ("true", "1", "yes")

        # Score with new classifier
        scored = score_records(clf, pool_recs, use_gnn=args.use_gnn)

        # Save new shuffled_parent scores
        out_f = records_dir / f"{scenario}__{METHOD_SHUFFLED_PARENT}__{PRIMARY_POOL}.csv"
        with open(out_f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(scored[0].keys()))
            w.writeheader()
            w.writerows(scored)

        # Compute metrics
        sma = float(source_macro_auprc_metric(scored))
        pooled = float(auprc_metric(scored))
        print(f"  [done] srcMacroAUPRC={sma:.4f} pooledAUPRC={pooled:.4f}")
        print(f"  [saved] -> {out_f}")

    print("\n[rescore] done! All shuffled_parent scores updated.")


if __name__ == "__main__":
    main()
