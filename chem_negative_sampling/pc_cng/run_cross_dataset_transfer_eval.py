"""P1-02: Cross-dataset transfer evaluation for PC-CNG boundary negatives.

Tests whether PC-CNG boundary negatives learned on a source dataset transfer
to a held-out target dataset.  For each transfer pair, two rerankers are
trained on the source:

* ``baseline``: real labels only (BCE supervised loss).
* ``treatment``: real labels + PC-CNG boundary negatives (the negatives are
  added as extra ``label=0`` training rows, mirroring the PC-CNG v2 data
  augmentation recipe).

Both models are evaluated on the target dataset's test split (reranking
Top-1, grouped by reactants).  A 10-seed paired significance test compares
baseline vs. treatment using:

* paired bootstrap 95% CI on the per-group delta,
* two-sided paired sign-flip permutation p-value,
* two-sided sign-test p-value (overflow-safe via ``scipy.stats.binomtest``).

The script is intentionally self-contained: it generates PC-CNG negatives for
the source once (cached), trains both rerankers in-process, scores the target,
and writes ``paired_significance.json`` plus per-seed diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .evaluate_candidate_reranking import (
    build_real_candidate_rows,
    build_synthetic_candidate_rows,
    positive_lookup,
    ranking_metrics,
    read_real_rows as read_real_rows_for_rerank,
    CheckpointScorer,
)
from .paired_reranking_significance import (
    bootstrap_ci,
    mean,
    paired_permutation_p_value,
    percentile,
)
from .train_feasibility_mlp import (
    FeasibilityMLP,
    featurize_rows,
    make_reaction_featurizer,
    predict,
    read_real_rows,
    read_synthetic_rows,
    set_seed,
)


DATASET_REGISTRY: Dict[str, str] = {
    "regiosqm20": "data/processed/regiosqm20_normalized.csv",
    "hitea": "data/processed/hitea_full_normalized.csv",
    "uspto": "data/processed/uspto_openmolecules_normalized.csv",
}

# Datasets whose normalized CSV has only positives (no real_negative labels)
# need a separate synthetic-negative source for reranking evaluation.  USPTO
# uses the v3 test-expansion candidates (PC-CNG negatives generated from USPTO
# test-split positives, reviewed and kept as ``keep_synthetic_negative``).
# Each entry maps dataset name -> (positives_csv, synthetic_negatives_csv).
DATASET_SYNTHETIC_REGISTRY: Dict[str, Tuple[str, str]] = {
    "uspto": (
        "results/original_test_expansion_uspto_negatives_20260712/"
        "uspto_original_test_expansion_positive_parents.csv",
        "results/original_test_expansion_uspto_negatives_20260712/"
        "uspto_test_expansion_candidates_reviewed.csv",
    ),
}

TRANSFER_PAIRS: List[Tuple[str, str]] = [
    ("regiosqm20", "hitea"),
    ("hitea", "regiosqm20"),
    ("regiosqm20", "uspto"),
    ("hitea", "uspto"),
]


def sign_test_p_value_safe(values: Sequence[float]) -> float:
    """Overflow-safe two-sided sign-test p-value.

    The original :func:`paired_reranking_significance.sign_test_p_value` uses
    ``math.comb`` which overflows for large ``n`` (thousands of pooled groups).
    We fall back to ``scipy.stats.binomtest`` which is numerically stable.
    """
    positive = sum(1 for value in values if value > 0.0)
    negative = sum(1 for value in values if value < 0.0)
    n = positive + negative
    if n == 0:
        return 1.0
    k = min(positive, negative)
    try:
        from scipy.stats import binomtest

        return float(binomtest(k, n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        # Last-resort fallback for very small n (keeps the legacy path).
        if n > 30:
            # Normal approximation to avoid overflow.
            mu = n * 0.5
            sigma = math.sqrt(n * 0.25)
            z = (k - mu) / max(sigma, 1e-12)
            return float(min(1.0, 2.0 * 0.5 * (1.0 + math.erf(-abs(z) / math.sqrt(2.0)))))
        cdf = sum(math.comb(n, i) * (0.5**n) for i in range(k + 1))
        return float(min(1.0, 2.0 * cdf))


def resolve_dataset_path(name: str, research_root: str) -> str:
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {list(DATASET_REGISTRY)}")
    path = DATASET_REGISTRY[key]
    if os.path.isabs(path):
        return path
    return os.path.join(research_root, path)


def generate_pccng_negatives(
    source_csv: str,
    output_csv: str,
    limit: int | None,
    max_candidates_per_reaction: int = 4,
) -> Dict[str, object]:
    """Generate PC-CNG boundary negatives from the source's train-split positives.

    The output CSV is written in the BoundaryCandidate schema so that
    :func:`train_feasibility_mlp.read_synthetic_rows` can ingest it directly.
    """
    from .reaction_boundary_generator import (
        BoundaryCandidate,
        ReactionBoundaryGenerator,
    )

    real_rows, _ = read_real_rows(source_csv)
    positives = [row for row in real_rows if row.get("split") == "train" and int(row.get("label", 0)) == 1]
    if limit is not None and len(positives) > limit:
        positives = positives[:limit]

    generator = ReactionBoundaryGenerator(
        allow_unmapped_fallback=True,
        max_candidates_per_reaction=max_candidates_per_reaction,
    )
    fieldnames = list(BoundaryCandidate.__dataclass_fields__.keys())
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    processed = 0
    generated = 0
    failed = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for pos in positives:
            processed += 1
            try:
                candidates = generator.generate_for_reaction(
                    str(pos["reaction_smiles"]),
                    source_id=str(pos["source_id"]),
                )
            except Exception:
                failed += 1
                continue
            if not candidates:
                failed += 1
                continue
            for cand in candidates:
                writer.writerow(cand.to_dict())
                generated += 1

    return {
        "source_csv": source_csv,
        "output_csv": output_csv,
        "processed_positives": processed,
        "generated_negatives": generated,
        "failed_or_empty": failed,
    }


def train_reranker(
    real_csv: str,
    synthetic_csv: str | None,
    output_dir: str,
    seed: int,
    epochs: int,
    hidden_dim: int,
    n_bits: int,
    batch_size: int,
    device_name: str | None,
    checkpoint_metric: str = "val_roc_auc",
) -> str:
    """Train a FeasibilityMLP reranker on the source data.

    If ``synthetic_csv`` is provided, PC-CNG negatives are added as extra
    ``label=0`` training rows (mirroring the v2 data-augmentation recipe).
    Returns the model directory containing ``best_feasibility_mlp.pt``.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    real_rows, source_split = read_real_rows(real_csv)
    train_rows = [row for row in real_rows if row.get("split") == "train"]
    val_rows = [row for row in real_rows if row.get("split") == "val"]
    test_rows = [row for row in real_rows if row.get("split") == "test"]

    synthetic_rows: List[Dict[str, object]] = []
    if synthetic_csv and os.path.exists(synthetic_csv):
        synthetic_rows = read_synthetic_rows(synthetic_csv, source_split)

    all_train_rows = list(train_rows) + list(synthetic_rows)

    featurizer = make_reaction_featurizer(feature_mode="morgan", n_bits=n_bits, fp_mode="binary")
    x_train, y_train, _, train_kept = featurize_rows(all_train_rows, featurizer)
    x_val, y_val, _, val_kept = featurize_rows(val_rows, featurizer)
    x_test, y_test, _, test_kept = featurize_rows(test_rows, featurizer)
    if len(x_train) == 0:
        raise RuntimeError(f"No train rows survived featurization for {real_csv}")

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FeasibilityMLP(in_dim=x_train.shape[1], hidden_dim=hidden_dim, dropout=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    neg = max(float((y_train == 0).sum()), 1.0)
    pos = max(float((y_train == 1).sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    best_val = -1.0
    best_path = os.path.join(output_dir, "best_feasibility_mlp.pt")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        val_scores = predict(model, x_val, device, batch_size)
        from .train_feasibility_mlp import compute_metrics

        val_metrics = compute_metrics(y_val, val_scores) if len(y_val) else {}
        val_key = val_metrics.get("roc_auc", val_metrics.get("accuracy", 0.0))
        if val_key == val_key and val_key > best_val:
            best_val = float(val_key)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "n_bits": n_bits,
                    "fp_mode": "binary",
                    "include_descriptors": False,
                    "feature_mode": "morgan",
                    "hidden_dim": hidden_dim,
                    "input_dim": x_train.shape[1],
                    "epoch": epoch,
                    "best_val": best_val,
                },
                best_path,
            )

    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    # Write metrics.json with counts for downstream auditing.
    from .train_feasibility_mlp import compute_metrics, save_predictions

    val_scores = predict(model, x_val, device, batch_size) if len(y_val) else np.zeros((0,), dtype=np.float32)
    test_scores = predict(model, x_test, device, batch_size) if len(y_test) else np.zeros((0,), dtype=np.float32)
    if len(y_val):
        save_predictions(os.path.join(output_dir, "val_predictions.csv"), val_kept, y_val, val_scores)
    if len(y_test):
        save_predictions(os.path.join(output_dir, "test_predictions.csv"), test_kept, y_test, test_scores)

    metrics = {
        "config": {
            "real_csv": real_csv,
            "synthetic_csv": synthetic_csv,
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "n_bits": n_bits,
            "batch_size": batch_size,
            "seed": seed,
        },
        "device": str(device),
        "counts": {
            "real_train": len(train_rows),
            "synthetic_train": len(synthetic_rows),
            "val": len(val_kept),
            "test": len(test_kept),
        },
        "val": compute_metrics(y_val, val_scores) if len(y_val) else {},
        "test": compute_metrics(y_test, test_scores) if len(y_test) else {},
        "best_checkpoint": best_path,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    return output_dir


def per_group_top1(scored_rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    """Return ``group_id -> Top-1`` (1.0 if positive ranks first, else 0.0)."""
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in scored_rows:
        grouped[str(row["group_id"])].append(row)
    out: Dict[str, float] = {}
    for gid, rows in grouped.items():
        labels = [int(row["label"]) for row in rows]
        if not any(labels) or all(labels):
            continue
        ranked = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
        out[gid] = 1.0 if int(ranked[0]["label"]) == 1 else 0.0
    return out


def evaluate_on_target(
    model_dir: str,
    target_csv: str,
    batch_size: int,
    device_name: str | None,
    target_limit: int | None,
    target_name: str | None = None,
    research_root: str | None = None,
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, object]]]:
    """Evaluate a trained reranker on the target dataset.

    Returns ``(aggregate_metrics, per_group_top1_map, scored_rows)``.

    For datasets with no ``real_negative`` labels in their normalized CSV
    (e.g. USPTO), we fall back to the synthetic test-expansion candidates
    (PC-CNG negatives generated from the target's own test-split positives).
    The candidate_source field is set to ``"synthetic"`` so downstream code
    can distinguish the two regimes.
    """
    import torch

    # Determine whether this target needs the synthetic-negative fallback.
    synthetic_paths: List[str] = []
    if target_name and target_name.lower() in DATASET_SYNTHETIC_REGISTRY:
        pos_rel, synth_rel = DATASET_SYNTHETIC_REGISTRY[target_name.lower()]
        base = research_root or os.getcwd()
        pos_csv = pos_rel if os.path.isabs(pos_rel) else os.path.join(base, pos_rel)
        synth_csv = synth_rel if os.path.isabs(synth_rel) else os.path.join(base, synth_rel)
        if not os.path.exists(pos_csv) or not os.path.exists(synth_csv):
            raise FileNotFoundError(
                f"Synthetic-negative files for {target_name} not found: "
                f"{pos_csv}, {synth_csv}"
            )
        # Load positives from the test-expansion parents CSV only (already
        # label_type=positive, test split).  Skipping target_csv here avoids
        # loading the 200k-row USPTO normalized CSV just to extract positives.
        real_rows = read_real_rows_for_rerank([pos_csv])
        synthetic_paths = [synth_csv]
    else:
        real_rows = read_real_rows_for_rerank([target_csv])

    if target_limit is not None and not synthetic_paths:
        # Keep all splits but cap total rows for tractability on USPTO.
        if len(real_rows) > target_limit:
            real_rows = real_rows[:target_limit]

    if synthetic_paths:
        positives = positive_lookup(real_rows)
        candidates = build_synthetic_candidate_rows(
            synthetic_paths=synthetic_paths,
            positives=positives,
            review_statuses=["keep_synthetic_negative"],
        )
    else:
        candidates = build_real_candidate_rows(
            real_rows=real_rows,
            group_by="reactants",
            candidate_scope="same_split",
        )
    if not candidates:
        return {"groups": 0, "top1": 0.0, "mrr": 0.0, "ndcg": 0.0}, {}, []

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    scorer = CheckpointScorer(model_dir, device)
    scores, kept = scorer.score(candidates, batch_size=batch_size)
    for row, score in zip(kept, scores.tolist()):
        row["score"] = float(score)

    aggregate = ranking_metrics(kept)
    group_map = per_group_top1(kept)
    return aggregate, group_map, kept


def paired_significance(
    deltas: Sequence[float],
    bootstrap_iterations: int,
    seed: int,
) -> Dict[str, float]:
    """Compute paired bootstrap CI + permutation p + sign-test p on deltas."""
    if not deltas:
        return {
            "n": 0,
            "delta_mean": 0.0,
            "delta_ci95_low": 0.0,
            "delta_ci95_high": 0.0,
            "paired_permutation_p": 1.0,
            "sign_test_p": 1.0,
            "positive_deltas": 0,
            "negative_deltas": 0,
            "zero_deltas": 0,
        }
    ci_low, ci_high = bootstrap_ci(deltas, bootstrap_iterations, seed)
    perm_p = paired_permutation_p_value(deltas, bootstrap_iterations, seed + 100)
    sign_p = sign_test_p_value_safe(deltas)
    return {
        "n": len(deltas),
        "delta_mean": mean(deltas),
        "delta_ci95_low": ci_low,
        "delta_ci95_high": ci_high,
        "paired_permutation_p": perm_p,
        "sign_test_p": sign_p,
        "positive_deltas": sum(1 for d in deltas if d > 0.0),
        "negative_deltas": sum(1 for d in deltas if d < 0.0),
        "zero_deltas": sum(1 for d in deltas if d == 0.0),
    }


def run_transfer_pair(
    source: str,
    target: str,
    seeds: Sequence[int],
    output_dir: str,
    research_root: str,
    epochs: int,
    hidden_dim: int,
    n_bits: int,
    batch_size: int,
    device_name: str | None,
    pccng_limit: int,
    target_limit: int | None,
    bootstrap_iterations: int,
    smoke: bool,
) -> Dict[str, object]:
    """Run one transfer pair across all seeds and write paired_significance.json."""
    source_csv = resolve_dataset_path(source, research_root)
    target_csv = resolve_dataset_path(target, research_root)
    pair_dir = os.path.join(output_dir, f"{source}_to_{target}")
    os.makedirs(pair_dir, exist_ok=True)

    # 1. Generate PC-CNG negatives from source (cached, seed-independent).
    pccng_csv = os.path.join(pair_dir, f"pccng_negatives_{source}.csv")
    if not os.path.exists(pccng_csv):
        print(f"[{source}->{target}] generating PC-CNG negatives from {source_csv}")
        gen_stats = generate_pccng_negatives(source_csv, pccng_csv, limit=pccng_limit)
        with open(os.path.join(pair_dir, "pccng_generation_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(gen_stats, handle, indent=2, ensure_ascii=False)
        print(f"[{source}->{target}] generated {gen_stats['generated_negatives']} negatives")
    else:
        print(f"[{source}->{target}] reusing cached PC-CNG negatives at {pccng_csv}")

    # 2. For each seed: train baseline + treatment, evaluate on target.
    per_seed_records: List[Dict[str, object]] = []
    baseline_group_maps: Dict[int, Dict[str, float]] = {}
    treatment_group_maps: Dict[int, Dict[str, float]] = {}
    for seed in seeds:
        seed_dir = os.path.join(pair_dir, f"seed{seed}")
        baseline_dir = os.path.join(seed_dir, "baseline")
        treatment_dir = os.path.join(seed_dir, "treatment")
        os.makedirs(baseline_dir, exist_ok=True)
        os.makedirs(treatment_dir, exist_ok=True)

        t0 = time.time()
        print(f"[{source}->{target}] seed={seed} training baseline")
        train_reranker(
            real_csv=source_csv,
            synthetic_csv=None,
            output_dir=baseline_dir,
            seed=seed,
            epochs=epochs,
            hidden_dim=hidden_dim,
            n_bits=n_bits,
            batch_size=batch_size,
            device_name=device_name,
        )
        baseline_metrics, baseline_groups, _ = evaluate_on_target(
            baseline_dir,
            target_csv,
            batch_size,
            device_name,
            target_limit,
            target_name=target,
            research_root=research_root,
        )
        print(f"[{source}->{target}] seed={seed} baseline target top1={baseline_metrics.get('top1', 0.0):.4f} groups={baseline_metrics.get('groups', 0)}")

        print(f"[{source}->{target}] seed={seed} training treatment (with PC-CNG)")
        train_reranker(
            real_csv=source_csv,
            synthetic_csv=pccng_csv,
            output_dir=treatment_dir,
            seed=seed,
            epochs=epochs,
            hidden_dim=hidden_dim,
            n_bits=n_bits,
            batch_size=batch_size,
            device_name=device_name,
        )
        treatment_metrics, treatment_groups, _ = evaluate_on_target(
            treatment_dir,
            target_csv,
            batch_size,
            device_name,
            target_limit,
            target_name=target,
            research_root=research_root,
        )
        print(f"[{source}->{target}] seed={seed} treatment target top1={treatment_metrics.get('top1', 0.0):.4f} groups={treatment_metrics.get('groups', 0)}")

        baseline_group_maps[seed] = baseline_groups
        treatment_group_maps[seed] = treatment_groups
        elapsed = time.time() - t0
        per_seed_records.append(
            {
                "seed": seed,
                "elapsed_sec": elapsed,
                "baseline_target_top1": baseline_metrics.get("top1", 0.0),
                "treatment_target_top1": treatment_metrics.get("top1", 0.0),
                "baseline_target_groups": baseline_metrics.get("groups", 0),
                "treatment_target_groups": treatment_metrics.get("groups", 0),
            }
        )
        if smoke:
            break

    # 3. Pool per-group deltas across seeds for paired significance test.
    pooled_deltas: List[float] = []
    for seed in seeds:
        bmap = baseline_group_maps.get(seed, {})
        tmap = treatment_group_maps.get(seed, {})
        common = set(bmap) & set(tmap)
        for gid in common:
            pooled_deltas.append(tmap[gid] - bmap[gid])

    sig = paired_significance(pooled_deltas, bootstrap_iterations, seed=20260719)
    seed_level_deltas = [
        rec["treatment_target_top1"] - rec["baseline_target_top1"] for rec in per_seed_records
    ]
    seed_ci_low, seed_ci_high = (
        bootstrap_ci(seed_level_deltas, bootstrap_iterations, 20260719 + 1)
        if seed_level_deltas
        else (0.0, 0.0)
    )
    seed_significance = {
        "n_seeds": len(seed_level_deltas),
        "mean_delta": mean(seed_level_deltas),
        "ci95_low": seed_ci_low,
        "ci95_high": seed_ci_high,
        "sign_test_p": sign_test_p_value_safe(seed_level_deltas),
        "per_seed_deltas": seed_level_deltas,
    }

    # 4. Write outputs.
    with open(os.path.join(pair_dir, "per_seed_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(per_seed_records, handle, indent=2, ensure_ascii=False)

    payload = {
        "source": source,
        "target": target,
        "source_csv": source_csv,
        "target_csv": target_csv,
        "seeds": list(seeds),
        "pccng_negatives_csv": pccng_csv,
        "config": {
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "n_bits": n_bits,
            "batch_size": batch_size,
            "pccng_limit": pccng_limit,
            "target_limit": target_limit,
            "bootstrap_iterations": bootstrap_iterations,
            "smoke": smoke,
        },
        "per_seed": per_seed_records,
        "paired_significance_pooled": sig,
        "seed_level_significance": seed_significance,
        "decision_rule": {
            "go_to_main_paper": "CI95_low > 0 AND sign_test_p < 0.05",
            "supplementary_only": "CI crosses 0 OR sign_test_p >= 0.05",
        },
    }
    with open(os.path.join(pair_dir, "paired_significance.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(
        f"[{source}->{target}] pooled delta={sig['delta_mean']:.4f} "
        f"CI=[{sig['delta_ci95_low']:.4f},{sig['delta_ci95_high']:.4f}] "
        f"perm_p={sig['paired_permutation_p']:.4g} sign_p={sig['sign_test_p']:.4g}"
    )
    return payload


def parse_seeds(raw: str) -> List[int]:
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="P1-02 cross-dataset transfer evaluation")
    parser.add_argument("--source", default=None, help="Source dataset name (regiosqm20|hitea|uspto). Required unless --all-pairs.")
    parser.add_argument("--target", default=None, help="Target dataset name. Required unless --all-pairs.")
    parser.add_argument("--seeds", required=True, help="Comma-separated seed list")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--research-root", default=None, help="Override research root (default: cwd)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default=None, help="cuda:0 / cpu / etc.  Defaults to cuda if available.")
    parser.add_argument("--pccng-limit", type=int, default=1000, help="Max source positives used for PC-CNG generation.")
    parser.add_argument("--target-limit", type=int, default=None, help="Cap target rows (USPTO subset).")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--smoke", action="store_true", help="Run 1 seed only for pipeline validation.")
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Ignore --source/--target and run all 4 transfer pairs.",
    )
    args = parser.parse_args()

    research_root = args.research_root or os.getcwd()
    seeds = parse_seeds(args.seeds)
    if args.smoke:
        seeds = seeds[:1]

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all_pairs:
        if args.source is not None or args.target is not None:
            print("[warn] --all-pairs set; ignoring --source/--target")
        all_results: List[Dict[str, object]] = []
        for source, target in TRANSFER_PAIRS:
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
                smoke=args.smoke,
            )
            all_results.append(res)
        with open(os.path.join(args.output_dir, "all_pairs_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2, ensure_ascii=False)
    else:
        if args.source is None or args.target is None:
            parser.error("--source and --target are required when --all-pairs is not set")
        run_transfer_pair(
            source=args.source,
            target=args.target,
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
            smoke=args.smoke,
        )


if __name__ == "__main__":
    main()
