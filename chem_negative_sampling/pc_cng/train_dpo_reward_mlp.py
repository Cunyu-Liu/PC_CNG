"""Train a DPO-style reward MLP using Chemformer-scored candidate sets.

The model is still a lightweight reaction scorer, but the pairwise objective is
closer to reaction-LM preference tuning: observed products are chosen, generated
or real alternatives are rejected, and Chemformer conditional log-likelihoods
serve as the reference policy scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .evaluate_candidate_reranking import grouped_metrics, ranking_metrics
from .train_feasibility_mlp import (
    FeasibilityMLP,
    ReactionFeaturizer,
    compute_group_metrics,
    compute_metrics,
    featurize_rows,
    predict,
    save_predictions,
    set_seed,
)


def read_scored_candidates(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            reaction = row.get("candidate_reaction", "")
            if not reaction:
                continue
            try:
                ref_score = float(row.get("lm_score", "nan"))
            except ValueError:
                continue
            if ref_score != ref_score:
                continue
            rows.append(
                {
                    "group_id": row.get("group_id", ""),
                    "source_id": row.get("source_id", ""),
                    "reaction_smiles": reaction,
                    "label": int(row.get("label", 0) or 0),
                    "split": row.get("split", "unknown") or "unknown",
                    "dataset": row.get("dataset", ""),
                    "reaction_class": row.get("candidate_family", ""),
                    "candidate_source": row.get("candidate_source", ""),
                    "candidate_family": row.get("candidate_family", ""),
                    "ref_score": ref_score,
                    "sample_weight": 1.0,
                }
            )
    return rows


def source_allowed(row: Dict[str, object], pair_source: str) -> bool:
    if pair_source == "all":
        return True
    return str(row.get("candidate_source", "")) == pair_source


def build_pairs(
    rows: Sequence[Dict[str, object]],
    split: str,
    pair_source: str,
    max_negatives_per_positive: int | None,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], np.ndarray]:
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != split:
            continue
        if not source_allowed(row, pair_source):
            continue
        grouped[str(row.get("group_id", ""))].append(row)

    chosen: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []
    ref_deltas: List[float] = []
    for group_rows in grouped.values():
        positives = [row for row in group_rows if int(row.get("label", 0)) == 1]
        negatives = [row for row in group_rows if int(row.get("label", 0)) == 0]
        if not positives or not negatives:
            continue
        for pos in positives:
            negs = list(negatives)
            rng.shuffle(negs)
            if max_negatives_per_positive is not None:
                negs = negs[:max_negatives_per_positive]
            for neg in negs:
                chosen.append(pos)
                rejected.append(neg)
                ref_deltas.append(float(pos["ref_score"]) - float(neg["ref_score"]))
    return chosen, rejected, np.array(ref_deltas, dtype=np.float32)


def parse_key_float(items: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key] = float(value)
    return out


def apply_row_weights(rows: Sequence[Dict[str, object]], source_weights: Dict[str, float]) -> None:
    for row in rows:
        source = str(row.get("candidate_source", ""))
        row["sample_weight"] = float(row.get("sample_weight", 1.0)) * source_weights.get(source, 1.0)


def score_candidate_rows(
    rows: Sequence[Dict[str, object]],
    scores: np.ndarray,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row, score in zip(rows, scores.tolist()):
        item = dict(row)
        item["score"] = float(score)
        out.append(item)
    return out


def write_candidate_scores(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "group_id",
        "source_id",
        "reaction_smiles",
        "label",
        "split",
        "dataset",
        "candidate_source",
        "candidate_family",
        "ref_score",
        "score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_candidate_scores(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "overall": ranking_metrics(rows),
        "by_split": grouped_metrics(rows, "split"),
        "by_dataset": grouped_metrics(rows, "dataset"),
        "by_candidate_source": grouped_metrics(rows, "candidate_source"),
        "by_candidate_family": grouped_metrics(rows, "candidate_family"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-csv", required=True, help="Candidate CSV with lm_score column")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--n-bits", type=int, default=4096)
    parser.add_argument("--fp-mode", choices=["binary", "count", "binary_count"], default="binary")
    parser.add_argument("--include-descriptors", action="store_true")
    parser.add_argument("--pair-source", choices=["synthetic", "real", "all"], default="synthetic")
    parser.add_argument("--max-negatives-per-positive", type=int, default=None)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--dpo-weight", type=float, default=1.0)
    parser.add_argument("--dpo-beta", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--reference-scale", choices=["none", "standardize"], default="standardize")
    parser.add_argument("--source-weight", action="append", default=[], help="candidate_source=weight")
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = read_scored_candidates(args.scored_csv)
    source_weights = parse_key_float(args.source_weight)
    train_rows = [row for row in rows if row.get("split") == "train"]
    val_rows = [row for row in rows if row.get("split") == "val"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    apply_row_weights(train_rows, source_weights)

    chosen_rows, rejected_rows, ref_delta = build_pairs(
        rows=rows,
        split="train",
        pair_source=args.pair_source,
        max_negatives_per_positive=args.max_negatives_per_positive,
        seed=args.seed,
    )
    if not chosen_rows:
        raise RuntimeError("No preference pairs found")

    ref_mean = float(ref_delta.mean()) if len(ref_delta) else 0.0
    ref_std = float(ref_delta.std()) if len(ref_delta) > 1 else 1.0
    if ref_std < 1e-6:
        ref_std = 1.0
    ref_delta_train = ref_delta.copy()
    if args.reference_scale == "standardize":
        ref_delta_train = (ref_delta_train - ref_mean) / ref_std

    featurizer = ReactionFeaturizer(
        n_bits=args.n_bits,
        fp_mode=args.fp_mode,
        include_descriptors=args.include_descriptors,
    )
    x_train, y_train, w_train, train_kept = featurize_rows(train_rows, featurizer)
    x_val, y_val, _, val_kept = featurize_rows(val_rows, featurizer)
    x_test, y_test, _, test_kept = featurize_rows(test_rows, featurizer)
    x_chosen, _, _, chosen_kept = featurize_rows(chosen_rows, featurizer)
    x_rejected, _, _, rejected_kept = featurize_rows(rejected_rows, featurizer)
    pair_count = min(len(x_chosen), len(x_rejected), len(ref_delta_train))
    if pair_count == 0:
        raise RuntimeError("No preference pairs survived featurization")
    x_chosen = x_chosen[:pair_count]
    x_rejected = x_rejected[:pair_count]
    ref_delta_train = ref_delta_train[:pair_count]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeasibilityMLP(in_dim=x_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    neg = max(float(w_train[y_train == 0].sum()), 1.0)
    pos = max(float(w_train[y_train == 1].sum()), 1.0)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device), reduction="none")

    supervised_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(w_train, dtype=torch.float32),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    pair_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_chosen, dtype=torch.float32),
            torch.tensor(x_rejected, dtype=torch.float32),
            torch.tensor(ref_delta_train, dtype=torch.float32),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_val = -1.0
    best_path = os.path.join(args.output_dir, "best_pairwise_reward_mlp.pt")
    history = []
    pair_batches = list(pair_loader)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for step, (batch_x, batch_y, batch_w) in enumerate(supervised_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            pair_pos, pair_neg, batch_ref_delta = pair_batches[step % len(pair_batches)]
            pair_pos = pair_pos.to(device)
            pair_neg = pair_neg.to(device)
            batch_ref_delta = batch_ref_delta.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            supervised_loss = (bce(logits, batch_y) * batch_w).sum() / torch.clamp(batch_w.sum(), min=1.0)
            pos_score = model(pair_pos)
            neg_score = model(pair_neg)
            policy_delta = pos_score - neg_score
            pair_loss = torch.nn.functional.softplus(args.margin - policy_delta).mean()
            dpo_loss = -torch.nn.functional.logsigmoid(args.dpo_beta * (policy_delta - batch_ref_delta)).mean()
            loss = args.bce_weight * supervised_loss + args.pairwise_weight * pair_loss + args.dpo_weight * dpo_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            total += len(batch_y)

        val_scores = predict(model, x_val, device, args.batch_size)
        val_metrics = compute_metrics(y_val, val_scores) if len(y_val) else {}
        val_key = float(val_metrics.get("roc_auc", 0.0))
        history.append({"epoch": epoch, "loss": total_loss / max(total, 1), "val": val_metrics})
        if val_key == val_key and val_key > best_val:
            best_val = val_key
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": x_train.shape[1],
                    "hidden_dim": args.hidden_dim,
                    "n_bits": args.n_bits,
                    "fp_mode": args.fp_mode,
                    "include_descriptors": args.include_descriptors,
                    "epoch": epoch,
                    "best_val": best_val,
                    "training_objective": "dpo_reward_mlp",
                    "reference_score": "chemformer_conditional_loglikelihood",
                    "reference_delta_mean": ref_mean,
                    "reference_delta_std": ref_std,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    val_scores = predict(model, x_val, device, args.batch_size)
    test_scores = predict(model, x_test, device, args.batch_size)
    save_predictions(os.path.join(args.output_dir, "val_predictions.csv"), val_kept, y_val, val_scores)
    save_predictions(os.path.join(args.output_dir, "test_predictions.csv"), test_kept, y_test, test_scores)

    all_x, _, _, all_kept = featurize_rows(rows, featurizer)
    all_scores = predict(model, all_x, device, args.batch_size)
    candidate_scores = score_candidate_rows(all_kept, all_scores)
    write_candidate_scores(os.path.join(args.output_dir, "candidate_scores.csv"), candidate_scores)
    ranking = evaluate_candidate_scores(candidate_scores)
    with open(os.path.join(args.output_dir, "ranking_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(ranking, handle, indent=2, ensure_ascii=False)

    metrics = {
        "config": vars(args),
        "device": str(device),
        "counts": {
            "rows": len(rows),
            "train_rows_featurized": len(train_kept),
            "val_rows_featurized": len(val_kept),
            "test_rows_featurized": len(test_kept),
            "preference_pairs_requested": len(chosen_rows),
            "preference_pairs_featurized": pair_count,
            "train_positive": int((y_train == 1).sum()),
            "train_negative": int((y_train == 0).sum()),
            "candidate_source_counts": dict(Counter(str(row.get("candidate_source", "")) for row in rows)),
            "pair_candidate_family_counts": dict(Counter(str(row.get("candidate_family", "")) for row in rejected_kept[:pair_count])),
        },
        "reference_delta": {
            "mean": ref_mean,
            "std": ref_std,
            "min": float(ref_delta.min()) if len(ref_delta) else 0.0,
            "max": float(ref_delta.max()) if len(ref_delta) else 0.0,
        },
        "val": compute_metrics(y_val, val_scores),
        "test": compute_metrics(y_test, test_scores),
        "test_by_dataset": compute_group_metrics(test_kept, y_test, test_scores, "dataset"),
        "test_by_candidate_source": compute_group_metrics(test_kept, y_test, test_scores, "candidate_source"),
        "ranking": ranking,
        "history": history,
        "best_checkpoint": best_path,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps({k: metrics[k] for k in ["device", "counts", "reference_delta", "val", "test", "ranking"]}, indent=2))


if __name__ == "__main__":
    main()
