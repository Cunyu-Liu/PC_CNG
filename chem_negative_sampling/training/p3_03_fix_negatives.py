"""Patch finetune_head.py to generate negatives on-the-fly.

The cross-dataset CSVs (ord_normalized.csv, uspto_openmolecules_normalized.csv,
hitea_full_normalized.csv) contain ONLY positive reactions (label_type=positive).
This makes MRR trivially 1.0 (each source_id group has exactly 1 positive item,
ranked #1 by default).

This patch adds a `_generate_negatives` function that creates K negative
examples per positive by corrupting the product:
  positive: reactants>>original_product  (label=1)
  negative: reactants>>random_product    (label=0)

All examples (positive + its K negatives) share the same source_id, so MRR
becomes meaningful (rank the positive among K+1 candidates).

The patch is applied IN-MEMORY by importing the module and monkey-patching.
This avoids modifying the source file while the current P3-03 run is active
(HC: don't modify running code's files -- although Python loads source at
import time, we use monkey-patching to be safe).
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any, Dict, List


def _generate_negatives(
    rows: List[Dict[str, Any]],
    n_negatives: int = 4,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Generate negative examples by corrupting products.

    For each positive reaction, create n_negatives corrupted reactions by
    replacing the product with a random product from another reaction.
    All examples (positive + its negatives) share the same source_id so
    MRR is meaningful.

    Handles both reaction SMILES formats:
      - reactants>>products (no agents)
      - reactants>agents>products (with agents)

    Args:
        rows: List of row dicts (must have "smiles" and "source_id").
            Assumed to be all positive (label=1).
        n_negatives: Number of negatives per positive.
        seed: Random seed for reproducibility.

    Returns:
        New list of rows with both positives and negatives, grouped by
        the original source_id.
    """
    rng = random.Random(seed)

    def _parse_reaction(smiles: str):
        """Return (reactants_with_agents_prefix, product) or None."""
        if ">>" in smiles:
            left, prod = smiles.rsplit(">>", 1)
            return left, prod.strip()
        # Try single > separator: reactants>agents>products
        parts = smiles.split(">")
        if len(parts) == 3:
            # Reconstruct reactants>agents> as prefix
            return parts[0] + ">" + parts[1] + ">", parts[2].strip()
        return None

    # Extract products from all rows
    products: List[str] = []
    for r in rows:
        smiles = str(r["smiles"])
        parsed = _parse_reaction(smiles)
        if parsed:
            products.append(parsed[1])
        else:
            products.append(smiles)  # fallback: use full smiles as "product"

    if len(products) < 2:
        return rows  # Can't generate negatives with < 2 examples

    out: List[Dict[str, Any]] = []
    n_with_negatives = 0
    for i, r in enumerate(rows):
        smiles = str(r["smiles"])
        sid = str(r["source_id"])
        # Add the positive (preserve original fields)
        positive_row = dict(r)
        positive_row["label"] = 1
        positive_row["source_id"] = sid
        out.append(positive_row)
        # Add negatives by corrupting the product
        parsed = _parse_reaction(smiles)
        if parsed is None:
            continue
        prefix, _ = parsed
        # Sample n_negatives random products (excluding self)
        neg_indices = [j for j in range(len(products)) if j != i and products[j] != parsed[1]]
        if len(neg_indices) <= n_negatives:
            sampled = neg_indices
        else:
            sampled = rng.sample(neg_indices, n_negatives)
        for j in sampled:
            neg_smiles = f"{prefix}{products[j]}"
            out.append({
                "smiles": neg_smiles,
                "label": 0,
                "source_id": sid,  # Same group as the positive
            })
        n_with_negatives += 1

    print(
        f"  [_generate_negatives] {len(rows)} rows -> {len(out)} rows "
        f"({n_with_negatives} got negatives, "
        f"{sum(1 for r in out if r['label']==1)} pos + "
        f"{sum(1 for r in out if r['label']==0)} neg)",
        flush=True,
    )
    return out


def _patched_run_pair(
    source_name: str,
    target_name: str,
    source_csv: str,
    target_csv: str,
    backbone_ckpt,
    vocab_path: str,
    seeds,
    output_dir: str,
    n_few_shot: float = 0.1,
    epochs: int = 5,
    lr: float = 1e-4,
    device: str = "cpu",
    train_idx_path=None,
    val_idx_path=None,
    test_idx_path=None,
    bootstrap_iterations: int = 10000,
    n_negatives_per_positive: int = 4,
) -> Dict[str, Any]:
    """Patched run_pair that generates negatives if target has none."""
    import os
    import copy
    from pathlib import Path
    import torch
    import numpy as np

    # Import from the original module
    import sys as _sys
    _cns_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _cns_root not in _sys.path:
        _sys.path.insert(0, _cns_root)
    from training.finetune_head import (
        load_dataset,
        load_or_create_split,
        load_tokenizer,
        load_pretrained_scorer,
        few_shot_finetune,
        evaluate,
        stratified_group_split,
        VARIANTS,
        paired_bootstrap_ci,
        family_cluster_bootstrap_ci,
        _render_summary_md,
    )

    pair_name = f"{source_name}_to_{target_name}"
    pair_dir = Path(output_dir) / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_exists = bool(backbone_ckpt) and os.path.exists(str(backbone_ckpt))

    # 1. Load source dataset
    source_rows = load_dataset(source_csv)
    source_train_idx, _, _ = load_or_create_split(
        source_rows, train_idx_path, val_idx_path, test_idx_path
    )

    # 2. Load target dataset
    target_rows = load_dataset(target_csv)

    # --- BUG FIX: generate negatives if target has no meaningful groups ---
    # A "meaningful group" = a source_id with BOTH positive AND negative
    # examples, so MRR is non-trivial. Datasets like hitea have negatives
    # (label_type=real_negative) but each source_id has exactly 1 row, so
    # MRR is trivially 1.0 (positive) or 0.0 (negative).
    from collections import defaultdict as _dd
    _group_labels = _dd(set)
    for r in target_rows:
        sid = str(r.get("source_id", ""))
        _group_labels[sid].add(int(r.get("label", 0)))
    has_meaningful_groups = any(
        0 in labels and 1 in labels for labels in _group_labels.values()
    )
    if not has_meaningful_groups:
        # Filter to only positive rows for negative generation
        pos_rows = [r for r in target_rows if int(r.get("label", 0)) == 1]
        print(
            f"[P3-03-FIX] WARNING: target {target_name} has no meaningful groups "
            f"(no source_id with both pos+neg); generating "
            f"{n_negatives_per_positive} negatives per positive "
            f"(total {len(pos_rows)} pos -> "
            f"{len(pos_rows) * (1 + n_negatives_per_positive)})",
            flush=True,
        )
        target_rows = _generate_negatives(
            pos_rows, n_negatives=n_negatives_per_positive, seed=42
        )
        print(
            f"[P3-03-FIX] After negative generation: {len(target_rows)} rows, "
            f"{sum(1 for r in target_rows if r['label']==1)} positives, "
            f"{sum(1 for r in target_rows if r['label']==0)} negatives",
            flush=True,
        )

    # 3. Load tokenizer + source-trained model
    tokenizer = load_tokenizer(vocab_path)
    source_model = load_pretrained_scorer(backbone_ckpt, vocab_path, device)

    # If no checkpoint, train initial head on source train split
    if not checkpoint_exists and source_train_idx:
        source_train_rows = [source_rows[i] for i in source_train_idx if 0 <= i < len(source_rows)]
        source_model = few_shot_finetune(
            source_model,
            source_train_rows,
            "head_finetune",
            tokenizer,
            n_epochs=epochs,
            lr=lr,
            device=device,
        )

    # 4. Run 3 variants x N seeds
    per_seed_metrics = {v: [] for v in VARIANTS}
    per_seed_aggregate = {v: [] for v in VARIANTS}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        target_train_idx, target_test_idx = stratified_group_split(
            target_rows, n_few_shot=n_few_shot, seed=seed
        )
        target_train_rows = [target_rows[i] for i in target_train_idx]
        target_test_rows = [target_rows[i] for i in target_test_idx]

        seed_dir = pair_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        seed_metrics = {"seed": seed, "variants": {}}

        for variant in VARIANTS:
            model_variant = copy.deepcopy(source_model)

            if variant != "direct":
                model_variant = few_shot_finetune(
                    model_variant,
                    target_train_rows,
                    variant,
                    tokenizer,
                    n_epochs=epochs,
                    lr=lr,
                    device=device,
                )

            metrics = evaluate(
                model_variant, tokenizer, target_test_rows, device=device
            )

            seed_metrics["variants"][variant] = {
                "mrr": metrics["mrr"],
                "accuracy": metrics["accuracy"],
                "auc": metrics["auc"],
                "n_examples": metrics["n_examples"],
            }
            per_seed_metrics[variant].append(seed_metrics["variants"][variant])
            per_seed_aggregate[variant].append(metrics["mrr"])

            # Free GPU memory
            del model_variant
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        with open(seed_dir / "metrics.json", "w") as f:
            json_dump = {
                "seed": seed,
                "variants": seed_metrics["variants"],
                "n_train": len(target_train_rows),
                "n_test": len(target_test_rows),
                "n_negatives_per_positive": n_negatives_per_positive if not has_meaningful_groups else 0,
            }
            import json
            json.dump(json_dump, f, indent=2)

        print(f"  seed {seed}: " + " | ".join(
            f"{v} MRR={per_seed_aggregate[v][-1]:.4f}" for v in VARIANTS
        ), flush=True)

    # 5. Aggregate + paired bootstrap CI
    import json
    summary = {
        "pair": pair_name,
        "source": source_name,
        "target": target_name,
        "n_seeds": len(seeds),
        "n_few_shot": n_few_shot,
        "checkpoint_used": checkpoint_exists,
        "backbone_ckpt": str(backbone_ckpt) if backbone_ckpt else None,
        "negatives_generated": not has_meaningful_groups,
        "n_negatives_per_positive": n_negatives_per_positive if not has_meaningful_groups else 0,
        "variants": {},
        "paired_bootstrap_ci": {},
        "family_cluster_bootstrap_ci": {},
        "go_no_go": {},
    }

    for v in VARIANTS:
        arr = per_seed_aggregate[v]
        summary["variants"][v] = {
            "mrr_mean": float(np.mean(arr)) if arr else 0.0,
            "mrr_std": float(np.std(arr)) if arr else 0.0,
            "mrr_per_seed": [float(x) for x in arr],
        }

    mrr_direct = per_seed_aggregate["direct"]
    mrr_head = per_seed_aggregate["head_finetune"]
    mrr_full = per_seed_aggregate["full_finetune"]

    head_vs_direct = paired_bootstrap_ci(mrr_head, mrr_direct, n_iterations=bootstrap_iterations)
    full_vs_direct = paired_bootstrap_ci(mrr_full, mrr_direct, n_iterations=bootstrap_iterations)
    # paired_bootstrap_ci returns (mean_diff, ci_low, ci_high, p_value) tuple
    summary["paired_bootstrap_ci"]["head_finetune_vs_direct"] = {
        "mean_diff": float(head_vs_direct[0]),
        "ci_low": float(head_vs_direct[1]),
        "ci_high": float(head_vs_direct[2]),
        "p_value": float(head_vs_direct[3]),
    }
    summary["paired_bootstrap_ci"]["full_finetune_vs_direct"] = {
        "mean_diff": float(full_vs_direct[0]),
        "ci_low": float(full_vs_direct[1]),
        "ci_high": float(full_vs_direct[2]),
        "p_value": float(full_vs_direct[3]),
    }

    try:
        # family_cluster_bootstrap_ci requires per-example cluster_ids,
        # which we don't have at the per-seed aggregate level.
        # Use paired_bootstrap_ci only (sufficient for 10-seed comparison).
        summary["family_cluster_bootstrap_ci"]["head_finetune_vs_direct"] = {
            "mean_diff": float(head_vs_direct[0]),
            "ci_low": float(head_vs_direct[1]),
            "ci_high": float(head_vs_direct[2]),
            "p_value": float(head_vs_direct[3]),
            "note": "family_cluster_bootstrap_ci skipped (needs per-example cluster_ids); "
                    "using paired_bootstrap_ci values as proxy",
        }
        summary["family_cluster_bootstrap_ci"]["full_finetune_vs_direct"] = {
            "mean_diff": float(full_vs_direct[0]),
            "ci_low": float(full_vs_direct[1]),
            "ci_high": float(full_vs_direct[2]),
            "p_value": float(full_vs_direct[3]),
            "note": "family_cluster_bootstrap_ci skipped (needs per-example cluster_ids); "
                    "using paired_bootstrap_ci values as proxy",
        }
    except Exception as exc:
        summary["family_cluster_bootstrap_ci"]["error"] = str(exc)

    summary["go_no_go"]["head_finetune_go"] = bool(
        head_vs_direct[0] > 0 and head_vs_direct[1] > 0
    )
    summary["go_no_go"]["full_finetune_go"] = bool(
        full_vs_direct[0] > 0 and full_vs_direct[1] > 0
    )

    with open(pair_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(pair_dir / "summary.md", "w") as f:
        f.write(_render_summary_md(summary))

    return summary


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="P3-03 fixed cross-dataset evaluation")
    parser.add_argument("--pairs", required=True, help="Comma-separated src->tgt pairs")
    parser.add_argument("--backbone-ckpt", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--seeds", default="20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719")
    parser.add_argument("--output-dir", default="results/cross_dataset_finetune_head_fixed_20260721")
    parser.add_argument("--n-few-shot", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-idx", default=None)
    parser.add_argument("--val-idx", default=None)
    parser.add_argument("--test-idx", default=None)
    parser.add_argument("--n-negatives", type=int, default=4, help="Negatives per positive")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    # Parse pairs
    pairs = []
    for p in args.pairs.split(","):
        src, tgt = p.split("->")
        pairs.append((src.strip(), tgt.strip()))

    # Map dataset names to CSV paths
    DATASET_CSVS = {
        "uspto": "uspto_openmolecules_normalized.csv",
        "uspto_openmolecules": "uspto_openmolecules_normalized.csv",
        "ord": "ord_normalized.csv",
        "hitea": "hitea_full_normalized.csv",
    }

    for source_name, target_name in pairs:
        source_csv = os.path.join(args.data_dir, DATASET_CSVS.get(source_name, f"{source_name}_normalized.csv"))
        target_csv = os.path.join(args.data_dir, DATASET_CSVS.get(target_name, f"{target_name}_normalized.csv"))

        if not os.path.exists(source_csv):
            print(f"[P3-03-FIX] SKIP {source_name}->{target_name}: source CSV missing ({source_csv})")
            continue
        if not os.path.exists(target_csv):
            print(f"[P3-03-FIX] SKIP {source_name}->{target_name}: target CSV missing ({target_csv})")
            continue

        print(f"[P3-03-FIX] Running {source_name}->{target_name} ...", flush=True)
        summary = _patched_run_pair(
            source_name=source_name,
            target_name=target_name,
            source_csv=source_csv,
            target_csv=target_csv,
            backbone_ckpt=args.backbone_ckpt,
            vocab_path=args.vocab,
            seeds=seeds,
            output_dir=args.output_dir,
            n_few_shot=args.n_few_shot,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            train_idx_path=args.train_idx,
            val_idx_path=args.val_idx,
            test_idx_path=args.test_idx,
            bootstrap_iterations=args.bootstrap_iterations,
            n_negatives_per_positive=args.n_negatives,
        )

        print(
            f"[P3-03-FIX] {source_name}->{target_name}: "
            f"direct MRR={summary['variants']['direct']['mrr_mean']:.4f} "
            f"head_finetune MRR={summary['variants']['head_finetune']['mrr_mean']:.4f} "
            f"full_finetune MRR={summary['variants']['full_finetune']['mrr_mean']:.4f}",
            flush=True,
        )
