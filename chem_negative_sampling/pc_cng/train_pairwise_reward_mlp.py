"""Train a reward-style feasibility MLP with pairwise negative feedback.

This is closer to the Science Advances lesson than direct BCE mixing: synthetic
counterfactual negatives are used as preferences, i.e. the observed positive
reaction should score above its paired failed/counterfactual outcome. Real
positive/negative labels still provide a supervised anchor.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .ranking_metrics import ranking_metrics
from .train_feasibility_mlp import (
    FeasibilityMLP,
    compute_group_metrics,
    compute_metrics,
    featurize_rows,
    make_reaction_featurizer,
    predict,
    read_real_rows,
    read_synthetic_rows,
    save_predictions,
    set_seed,
)


def build_positive_lookup(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    lookup: Dict[str, Dict[str, object]] = {}
    for row in rows:
        if row.get("split") == "train" and int(row.get("label", 0)) == 1:
            lookup[str(row["source_id"])] = row
    return lookup


def paired_rows(
    synthetic_rows: List[Dict[str, object]],
    positive_by_source: Dict[str, Dict[str, object]],
    max_pairs: int | None,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    pos_rows: List[Dict[str, object]] = []
    neg_rows: List[Dict[str, object]] = []
    for row in synthetic_rows:
        source_id = str(row.get("source_id", ""))
        pos = positive_by_source.get(source_id)
        if pos is None:
            continue
        pos_rows.append(pos)
        neg_rows.append(row)
        if max_pairs is not None and len(pos_rows) >= max_pairs:
            break
    return pos_rows, neg_rows


def parse_key_float(items: List[str]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        values[key] = float(value)
    return values


def pair_family(row: Dict[str, object]) -> str:
    family = str(row.get("action_family", "") or "").strip()
    if family:
        return family
    failure_type = str(row.get("failure_type", "") or "").strip()
    if failure_type.endswith("_hard_negative"):
        return failure_type[: -len("_hard_negative")]
    return failure_type or "unknown"


def pair_reaction_class(row: Dict[str, object]) -> str:
    reaction_class = str(row.get("reaction_class", "") or "").strip()
    if reaction_class:
        return reaction_class
    return "unknown"


def make_ranking_rows(
    rows: List[Dict[str, object]],
    labels: np.ndarray,
    scores: np.ndarray,
    group_by: str,
) -> List[Dict[str, object]]:
    """Build real-row candidate groups for checkpoint-selection ranking metrics."""
    out: List[Dict[str, object]] = []
    for row, label, score in zip(rows, labels.tolist(), scores.tolist()):
        dataset = str(row.get("dataset", "") or "real")
        split = str(row.get("split", "") or "unknown")
        group_value = (
            row.get(group_by)
            or row.get("reactants")
            or row.get("split_key")
            or row.get("source_id")
            or "unknown"
        )
        out.append(
            {
                "group_id": f"real|{dataset}|{split}|{group_value}",
                "source_id": row.get("source_id", ""),
                "reaction_smiles": row.get("reaction_smiles", ""),
                "label": int(label),
                "score": float(score),
                "split": split,
                "dataset": dataset,
                "candidate_source": "real",
                "candidate_family": "observed_positive" if int(label) == 1 else "real_negative",
                "reaction_class": row.get("reaction_class", ""),
            }
        )
    return out


def checkpoint_metric_value(
    metric_name: str,
    binary_metrics: Dict[str, float],
    ranking: Dict[str, float | int],
) -> float:
    if metric_name.startswith("val_"):
        key = metric_name[len("val_") :]
    else:
        key = metric_name
    if key in {"top1", "mrr", "ndcg", "top3"}:
        return float(ranking.get(key, 0.0))
    return float(binary_metrics.get(key, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", action="append", required=True)
    parser.add_argument("--synthetic-csv", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--feature-mode", choices=["morgan", "graph_stats", "combined"], default="morgan")
    parser.add_argument("--n-bits", type=int, default=4096)
    parser.add_argument("--fp-mode", choices=["binary", "count", "binary_count"], default="binary")
    parser.add_argument("--include-descriptors", action="store_true")
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--family-margin", action="append", default=[], help="Optional action-family margin as family=value")
    parser.add_argument("--family-weight", action="append", default=[], help="Optional action-family pair weight as family=value")
    parser.add_argument("--class-margin", action="append", default=[], help="Optional reaction-class margin as class=value; overrides family/base margin")
    parser.add_argument("--class-weight", action="append", default=[], help="Optional reaction-class pair weight as class=value; multiplies family weight")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default="none", help="Learning rate scheduler")
    parser.add_argument("--lr-min", type=float, default=1e-5, help="Minimum learning rate for cosine scheduler")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Number of warmup epochs (linear warmup)")
    parser.add_argument(
        "--checkpoint-metric",
        choices=["val_roc_auc", "val_auprc", "val_f1", "val_top1", "val_top3", "val_mrr", "val_ndcg"],
        default="val_roc_auc",
        help="Validation metric used to select best checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-group-by",
        default="reactants",
        help="Real validation row field used for val_* ranking checkpoint metrics.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional path to a FeasibilityMLP checkpoint to warm-start from (curriculum training).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    family_margins = parse_key_float(args.family_margin)
    family_weights = parse_key_float(args.family_weight)
    class_margins = parse_key_float(args.class_margin)
    class_weights = parse_key_float(args.class_weight)

    real_rows: List[Dict[str, object]] = []
    source_split: Dict[str, str] = {}
    for path in args.real_csv:
        rows, split_map = read_real_rows(path)
        real_rows.extend(rows)
        source_split.update(split_map)

    synthetic_rows: List[Dict[str, object]] = []
    for path in args.synthetic_csv:
        synthetic_rows.extend(read_synthetic_rows(path, source_split, args.max_pairs))

    train_rows = [row for row in real_rows if row["split"] == "train"]
    val_rows = [row for row in real_rows if row["split"] == "val"]
    test_rows = [row for row in real_rows if row["split"] == "test"]
    pos_pair_rows, neg_pair_rows = paired_rows(synthetic_rows, build_positive_lookup(real_rows), args.max_pairs)
    if not pos_pair_rows:
        raise RuntimeError("No positive/synthetic pairs found for pairwise reward training")

    featurizer = make_reaction_featurizer(
        feature_mode=args.feature_mode,
        n_bits=args.n_bits,
        fp_mode=args.fp_mode,
        include_descriptors=args.include_descriptors,
    )
    x_train, y_train, _, train_kept = featurize_rows(train_rows, featurizer)
    x_val, y_val, _, val_kept = featurize_rows(val_rows, featurizer)
    x_test, y_test, _, test_kept = featurize_rows(test_rows, featurizer)
    x_pair_pos, _, _, pos_pair_kept = featurize_rows(pos_pair_rows, featurizer)
    x_pair_neg, _, _, neg_pair_kept = featurize_rows(neg_pair_rows, featurizer)
    pair_count = min(len(x_pair_pos), len(x_pair_neg))
    x_pair_pos = x_pair_pos[:pair_count]
    x_pair_neg = x_pair_neg[:pair_count]
    pair_families = [pair_family(row) for row in neg_pair_kept[:pair_count]]
    pair_classes = [pair_reaction_class(row) for row in neg_pair_kept[:pair_count]]
    pair_margin_values = np.array(
        [
            class_margins.get(reaction_class, family_margins.get(family, args.margin))
            for family, reaction_class in zip(pair_families, pair_classes)
        ],
        dtype=np.float32,
    )
    pair_weight_values = np.array(
        [
            family_weights.get(family, 1.0) * class_weights.get(reaction_class, 1.0)
            for family, reaction_class in zip(pair_families, pair_classes)
        ],
        dtype=np.float32,
    )
    if pair_count == 0:
        raise RuntimeError("No pairwise rows survived featurization")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeasibilityMLP(in_dim=x_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    if args.init_checkpoint:
        # Curriculum warm-start: load state_dict from a previous round's checkpoint.
        # We use strict=False so a dropout/buffer mismatch does not abort the run;
        # architecture (in_dim, hidden_dim) is still validated by parameter shape.
        init_ckpt = torch.load(args.init_checkpoint, map_location=device)
        state = init_ckpt.get("state_dict", init_ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[init-checkpoint] loaded from {args.init_checkpoint}; "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        else:
            print(f"[init-checkpoint] loaded state_dict from {args.init_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = None
    if args.lr_scheduler == "cosine":
        total_epochs = args.epochs
        warmup = max(0, args.warmup_epochs)
        if warmup > 0:
            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
            warmup_scheduler = LinearLR(
                optimizer, start_factor=1.0 / max(warmup, 1), total_iters=warmup
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=max(total_epochs - warmup, 1), eta_min=args.lr_min
            )
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup]
            )
        else:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=args.lr_min
            )
    neg = max(float((y_train == 0).sum()), 1.0)
    pos = max(float((y_train == 1).sum()), 1.0)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))

    supervised_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    pair_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_pair_pos, dtype=torch.float32),
            torch.tensor(x_pair_neg, dtype=torch.float32),
            torch.tensor(pair_margin_values, dtype=torch.float32),
            torch.tensor(pair_weight_values, dtype=torch.float32),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    history = []
    best_val = -1.0
    best_path = os.path.join(args.output_dir, "best_pairwise_reward_mlp.pt")
    pair_batches = list(pair_loader)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for step, (batch_x, batch_y) in enumerate(supervised_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pair_pos, pair_neg, pair_margin, pair_weight = pair_batches[step % len(pair_batches)]
            pair_pos = pair_pos.to(device)
            pair_neg = pair_neg.to(device)
            pair_margin = pair_margin.to(device)
            pair_weight = pair_weight.to(device)

            optimizer.zero_grad()
            supervised_loss = bce(model(batch_x), batch_y)
            pos_score = model(pair_pos)
            neg_score = model(pair_neg)
            pair_loss = torch.nn.functional.softplus(pair_margin - (pos_score - neg_score))
            pairwise_loss = (pair_loss * pair_weight).sum() / torch.clamp(pair_weight.sum(), min=1.0)
            loss = args.bce_weight * supervised_loss + args.pairwise_weight * pairwise_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            total += len(batch_y)

        val_scores = predict(model, x_val, device, args.batch_size)
        val_metrics = compute_metrics(y_val, val_scores)
        val_ranking = ranking_metrics(make_ranking_rows(val_kept, y_val, val_scores, args.checkpoint_group_by))
        current_lr = optimizer.param_groups[0]["lr"]
        val_key = checkpoint_metric_value(args.checkpoint_metric, val_metrics, val_ranking)
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(total, 1),
                "lr": current_lr,
                "val": val_metrics,
                "val_ranking": val_ranking,
                "checkpoint_metric": args.checkpoint_metric,
                "checkpoint_metric_value": val_key,
            }
        )
        if val_key == val_key and val_key > best_val:
            best_val = float(val_key)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": x_train.shape[1],
                    "hidden_dim": args.hidden_dim,
                    "feature_mode": args.feature_mode,
                    "n_bits": args.n_bits,
                    "fp_mode": args.fp_mode,
                    "include_descriptors": args.include_descriptors,
                    "epoch": epoch,
                    "best_val": best_val,
                    "checkpoint_metric": args.checkpoint_metric,
                    "checkpoint_group_by": args.checkpoint_group_by,
                },
                best_path,
            )
        if scheduler is not None:
            scheduler.step()

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    val_scores = predict(model, x_val, device, args.batch_size)
    test_scores = predict(model, x_test, device, args.batch_size)
    val_ranking = ranking_metrics(make_ranking_rows(val_kept, y_val, val_scores, args.checkpoint_group_by))
    test_ranking = ranking_metrics(make_ranking_rows(test_kept, y_test, test_scores, args.checkpoint_group_by))
    save_predictions(os.path.join(args.output_dir, "val_predictions.csv"), val_kept, y_val, val_scores)
    save_predictions(os.path.join(args.output_dir, "test_predictions.csv"), test_kept, y_test, test_scores)

    metrics = {
        "config": vars(args),
        "device": str(device),
        "counts": {
            "real_train_rows_featurized": len(train_kept),
            "pair_rows_requested": len(pos_pair_rows),
            "pair_rows_featurized": pair_count,
            "val_rows_featurized": len(val_kept),
            "test_rows_featurized": len(test_kept),
            "train_positive": int((y_train == 1).sum()),
            "train_negative": int((y_train == 0).sum()),
            "pair_family_counts": dict(Counter(pair_families)),
            "pair_class_counts": dict(Counter(pair_classes)),
        },
        "pair_family_margins": family_margins,
        "pair_family_weights": family_weights,
        "pair_class_margins": class_margins,
        "pair_class_weights": class_weights,
        "val": compute_metrics(y_val, val_scores),
        "test": compute_metrics(y_test, test_scores),
        "val_ranking_real": val_ranking,
        "test_ranking_real": test_ranking,
        "test_by_dataset": compute_group_metrics(test_kept, y_test, test_scores, "dataset"),
        "test_by_reaction_class": compute_group_metrics(test_kept, y_test, test_scores, "reaction_class"),
        "history": history,
        "best_checkpoint": best_path,
        "best_epoch": checkpoint.get("epoch"),
        "checkpoint_metric": args.checkpoint_metric,
        "best_checkpoint_metric_value": checkpoint.get("best_val"),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps({k: metrics[k] for k in ["device", "counts", "val", "test", "best_checkpoint"]}, indent=2))


if __name__ == "__main__":
    main()
