#!/usr/bin/env python3
"""Phase 3 RegioSQM20 external validation.

RegioSQM20 is a regioselectivity dataset with REAL negative labels
(1,872 minor/unobserved regioisomers vs 552 major/observed regioisomers).
This is the spec-required "RegioSQM 或严格受控 regioselectivity 数据".

Validation protocol:
  1. Train Morgan-MLP on HiTEA train set + PC-CNG negatives (same as OOD eval)
  2. Test on RegioSQM20 test split using REAL labels (no generated negatives)
  3. Compute AUPRC, macro-AUPRC, paired cluster bootstrap CI

This is a true cross-dataset transfer test: the classifier trained on HiTEA
(cross-coupling HTE) is evaluated on RegioSQM20 (aromatic regioselectivity).
The test negatives are REAL unobserved regioisomers, not synthetic.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reuse infrastructure from the main external validation script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pc_cng.run_phase3_external_validation import (
    DEFAULT_PARQUET,
    DEFAULT_OOD_DIR,
    DEFAULT_OUTPUT,
    MORGAN_RADIUS,
    MORGAN_BITS,
    MLP_EPOCHS,
    MLP_BATCH,
    MLP_LR,
    METHOD_LEARNED,
    METHOD_RULE,
    METHOD_RANDOM,
    BASELINE_METHODS,
    MorganMLP,
    NegativeGenerator,
    load_g8c_model,
    morgan_fingerprint,
    reaction_fp,
    reaction_fp_dim,
    build_dataset,
    evaluate_method,
)
from pc_cng.paired_cluster_inference import (
    auprc_metric,
    macro_auprc_metric,
    paired_cluster_bootstrap,
)

REGIOSQM_CSV = Path("/home/cunyuliu/pc_cng_research/data/processed/regiosqm20_normalized.csv")
DEFAULT_OUTPUT_RS = Path("/home/cunyuliu/pc_cng_research/results/phase3_regiosqm20_validation")


def load_regiosqm20_test(csv_path: Path, max_test: int = 200) -> List[Dict]:
    """Load RegioSQM20 test split with REAL labels.

    Returns list of records with:
      - smiles: product SMILES
      - is_positive: True if major regioisomer (positive), False if minor (real_negative)
      - score: None (will be filled by classifier)
      - experimental_group: split_key (for cluster bootstrap)
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    test_df = df[df["split"] == "test"].copy()
    if len(test_df) > max_test:
        test_df = test_df.sample(n=max_test, random_state=20260725)
    records = []
    for _, row in test_df.iterrows():
        product = row.get("products", "")
        rxn = row.get("reaction_smiles", "")
        if not product or not isinstance(product, str):
            continue
        is_pos = row.get("label_type", "") == "positive"
        records.append({
            "smiles": product,
            "reaction_smiles": rxn,
            "is_positive": is_pos,
            "score": 0.0,
            "experimental_group": str(row.get("split_key", "default")),
            "reaction_family": "regioselectivity",
            "yield_bin": "1" if is_pos else "0",
        })
    return records


def load_hitea_train(parquet_path: Path, ood_dir: Path,
                     split_name: str = "random",
                     max_train: int = 500) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """Load HiTEA train/val DataFrames from the random split.

    Returns DataFrames (NOT lists) because build_dataset expects rows with
    .iterrows() and a 'reaction_smiles' column.
    """
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    train_idx_path = ood_dir / f"{split_name}_train_idx.json"
    val_idx_path = ood_dir / f"{split_name}_val_idx.json"
    import json as _json
    with open(train_idx_path) as f:
        train_idx = _json.load(f)
    with open(val_idx_path) as f:
        val_idx = _json.load(f)
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    if max_train and len(train_df) > max_train:
        train_df = train_df.sample(n=max_train, random_state=20260725)
    if max_train and len(val_df) > max_train:
        val_df = val_df.sample(n=max_train, random_state=20260725)
    return train_df, val_df


def evaluate_on_regiosqm20(train_rows,  # pd.DataFrame (has .iterrows())
                           test_records: List[Dict],
                           generator: NegativeGenerator,
                           seed: int) -> Dict:
    """Train MLP on HiTEA+negatives, evaluate on RegioSQM20 REAL labels."""
    t0 = time.time()
    # Build training set (HiTEA positives + generated negatives)
    X_train, y_train, _ = build_dataset(train_rows, generator)
    if X_train is None or len(X_train) < 10:
        return {"error": "insufficient training data", "n_train": 0}

    # Build test set from RegioSQM20 REAL labels.
    # Use reaction_fp (same as training) for dimension consistency.
    fps = []
    labels = []
    valid_indices = []
    for i, rec in enumerate(test_records):
        rxn = rec.get("reaction_smiles", "")
        fp = reaction_fp(rxn) if rxn else None
        if fp is None:
            # Fallback: try product Morgan fingerprint, but pad/truncate to
            # match reaction_fp_dim.  This should rarely trigger since
            # RegioSQM20 has reaction_smiles.
            fp = morgan_fingerprint(rec["smiles"])
            if fp is not None:
                # Pad to reaction_fp_dim by concatenating with zeros
                target_dim = reaction_fp_dim()
                if len(fp) < target_dim:
                    fp = np.concatenate([fp, np.zeros(target_dim - len(fp), dtype=fp.dtype)])
                elif len(fp) > target_dim:
                    fp = fp[:target_dim]
        if fp is not None:
            fps.append(fp)
            labels.append(1.0 if rec["is_positive"] else 0.0)
            valid_indices.append(i)
    if len(fps) < 4:
        return {"error": "insufficient RegioSQM20 test data", "n_train": len(X_train)}

    X_test = np.vstack(fps)
    y_test = np.array(labels, dtype=np.float32)
    # Keep only valid test records for metric computation
    test_records = [test_records[i] for i in valid_indices]

    # Train MLP
    mlp = MorganMLP(input_dim=X_train.shape[1], seed=seed)
    mlp.train(X_train, y_train, epochs=MLP_EPOCHS, batch_size=MLP_BATCH,
              lr=MLP_LR, verbose=False)

    # Predict on RegioSQM20 test
    scores = mlp.predict_proba(X_test)
    for i, rec in enumerate(test_records):
        if i < len(scores):
            rec["score"] = float(scores[i])
        # auprc_metric uses "label" (int), not "is_positive" (bool)
        rec["label"] = 1 if rec["is_positive"] else 0

    # Compute metrics
    # Only use records that have valid fingerprints
    valid_records = [r for i, r in enumerate(test_records) if i < len(scores)]
    auprc = auprc_metric(valid_records)
    macro = macro_auprc_metric(valid_records, bin_key="yield_bin")

    elapsed = time.time() - t0
    n_pos = sum(1 for r in valid_records if r["is_positive"])
    n_neg = sum(1 for r in valid_records if not r["is_positive"])
    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_test_positive": n_pos,
        "n_test_negative": n_neg,
        "auprc": float(auprc),
        "macro_auprc": float(macro),
        "family_macro_auprc": float(auprc),  # single family
        "elapsed_sec": float(elapsed),
        "test_records": valid_records,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase 3 RegioSQM20 external validation (REAL negative labels).")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--ood-dir", type=Path, default=DEFAULT_OOD_DIR)
    parser.add_argument("--regiosqm-csv", type=Path, default=REGIOSQM_CSV)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/home/cunyuliu/pc_cng_research/results/p4_g8c_phase2_full/model_checkpoint.pt"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_RS)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=500)
    parser.add_argument("--max-test", type=int, default=200)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("[regiosqm20] === Phase 3 RegioSQM20 External Validation ===")
    print(f"  parquet: {args.parquet}")
    print(f"  regiosqm: {args.regiosqm_csv}")
    print(f"  checkpoint: {args.checkpoint}")

    # Load RegioSQM20 test set (REAL labels)
    test_records = load_regiosqm20_test(args.regiosqm_csv, args.max_test)
    n_pos = sum(1 for r in test_records if r["is_positive"])
    n_neg = sum(1 for r in test_records if not r["is_positive"])
    print(f"[regiosqm20] test set: {len(test_records)} records ({n_pos} pos, {n_neg} neg REAL labels)")

    # Load HiTEA train set
    train_rows, val_rows = load_hitea_train(args.parquet, args.ood_dir,
                                            split_name="random",
                                            max_train=args.max_train)
    print(f"[regiosqm20] HiTEA train: {len(train_rows)} rows")

    # Load G8-C model if available
    model = None
    device = None
    if args.checkpoint.exists():
        try:
            model, device = load_g8c_model(args.checkpoint)
            print(f"[regiosqm20] G8-C model loaded from {args.checkpoint}")
        except Exception as exc:
            print(f"[regiosqm20] WARNING: could not load G8-C model: {exc}")
    else:
        print(f"[regiosqm20] G8-C checkpoint not found; running baselines only")

    # Methods
    methods = list(BASELINE_METHODS)
    if model is not None:
        methods = [METHOD_LEARNED] + methods
    print(f"[regiosqm20] methods: {methods}")

    # Run each method
    results: Dict[str, Dict] = {}
    test_records_by_method: Dict[str, List[Dict]] = {}
    import copy
    for method in methods:
        print(f"\n  [{method}] training on HiTEA+negatives, testing on RegioSQM20 REAL labels ...")
        if method == METHOD_LEARNED and model is None:
            results[method] = {"error": "model_not_available", "skipped": True}
            continue
        try:
            if method == METHOD_LEARNED:
                gen = NegativeGenerator(method, model=model, device=device, seed=args.seed)
            else:
                gen = NegativeGenerator(method, seed=args.seed)
            # Deep copy test_records so each method gets its own copy
            test_copy = copy.deepcopy(test_records)
            res = evaluate_on_regiosqm20(train_rows, test_copy, gen, args.seed)
            results[method] = {k: v for k, v in res.items() if k != "test_records"}
            if "test_records" in res:
                test_records_by_method[method] = res["test_records"]
                print(f"    AUPRC={res['auprc']:.4f}  macro={res['macro_auprc']:.4f}  "
                      f"(pos={res['n_test_positive']}, neg={res['n_test_negative']})  "
                      f"({res['elapsed_sec']:.1f}s)")
            else:
                print(f"    ERROR: {res.get('error', 'unknown')}")
        except Exception as exc:
            results[method] = {"error": str(exc)}
            print(f"    ERROR: {exc}")

    # Paired CI for all available method pairs
    paired_ci: Dict = {}
    _pairs = [
        (METHOD_LEARNED, METHOD_RULE),
        (METHOD_LEARNED, METHOD_RANDOM),
        (METHOD_RULE, METHOD_RANDOM),
    ]
    for challenger, baseline in _pairs:
        if challenger not in test_records_by_method or baseline not in test_records_by_method:
            continue
        pair_key = f"{challenger}_vs_{baseline}"
        print(f"\n  [paired CI] {pair_key} ...")
        try:
            ch_recs = list(test_records_by_method[challenger])
            bl_recs = list(test_records_by_method[baseline])
            n = min(len(ch_recs), len(bl_recs))
            ch_recs = ch_recs[:n]
            bl_recs = bl_recs[:n]
            ci = paired_cluster_bootstrap(
                ch_recs, bl_recs,
                metric_fn=auprc_metric,
                cluster_key="experimental_group",
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            paired_ci[pair_key] = ci
            print(f"    delta={ci['delta_mean']:.4f}  "
                  f"CI=[{ci['delta_ci_low']:.4f}, {ci['delta_ci_high']:.4f}]  "
                  f"sig={ci['ci_all_positive']}  p={ci['p_value']:.4f}")
        except Exception as exc:
            paired_ci[pair_key] = {"error": str(exc)}
            print(f"    ERROR: {exc}")

    # Save results
    output = {
        "dataset": "regiosqm20",
        "description": "RegioSQM20 regioselectivity external validation with REAL negative labels",
        "n_test_real_positive": n_pos,
        "n_test_real_negative": n_neg,
        "methods": results,
        "paired_ci": paired_ci,
        "checkpoint_loaded": model is not None,
        "elapsed_sec": time.time() - t_start,
    }
    with open(args.output / "regiosqm20_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "regiosqm_csv": str(args.regiosqm_csv),
        "parquet": str(args.parquet),
        "checkpoint": str(args.checkpoint),
        "model_loaded": model is not None,
        "methods": methods,
        "max_train": args.max_train,
        "max_test": args.max_test,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
    }
    with open(args.output / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("[regiosqm20] === RegioSQM20 Validation Summary ===")
    print("=" * 70)
    print(f"  Test set: {n_pos} real positives, {n_neg} real negatives")
    for method, res in results.items():
        if "auprc" in res:
            print(f"  {method:20s}  AUPRC={res['auprc']:.4f}  macro={res['macro_auprc']:.4f}")
    for pair_key, ci in paired_ci.items():
        if isinstance(ci, dict) and "delta_mean" in ci:
            print(f"  {pair_key:20s}  delta={ci['delta_mean']:.4f}  "
                  f"CI=[{ci['delta_ci_low']:.4f}, {ci['delta_ci_high']:.4f}]  "
                  f"sig={ci['ci_all_positive']}")
    print(f"\n[regiosqm20] results saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
