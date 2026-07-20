"""Train a reaction-center edit decoder over candidate anchors."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for train_reaction_center_edit_decoder.py") from exc

try:  # pragma: no cover
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None  # type: ignore
    roc_auc_score = None  # type: ignore

from .reaction_center_edit_decoder import FEATURE_NAMES


class EditDecoderMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.15, output_dim: int = 1):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        out = self.net(x)
        if self.output_dim == 1:
            return out.squeeze(-1)
        return out


def decoder_head_scores(model: nn.Module, x: torch.Tensor, head: str = "positive") -> torch.Tensor:
    logits = model(x)
    if logits.ndim == 1:
        return logits
    if head == "positive":
        return logits[:, 0]
    if head == "hard_negative":
        return logits[:, 1]
    raise ValueError(f"Unknown decoder head: {head}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_candidate_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in ["pair_id", "split", "is_true_anchor", *FEATURE_NAMES] if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing decoder dataset columns: {missing[:20]}")
        rows = list(reader)
        for row in rows:
            if "candidate_role" not in row or not row["candidate_role"]:
                row["candidate_role"] = "observed_positive" if int(row.get("is_true_anchor", 0) or 0) == 1 else "unannotated_candidate"
            row.setdefault("is_known_positive", "1" if row["candidate_role"] == "known_positive_alt" else "0")
            row.setdefault("is_hard_negative", "1" if row["candidate_role"] == "hard_negative" else "0")
            row.setdefault("hard_negative_weight", "1.0" if row["candidate_role"] == "hard_negative" else "0.0")
        return rows


def split_groups(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row.get("split", "train")][row["pair_id"]].append(row)
    return grouped


def valid_groups(groups: Dict[str, List[Dict[str, str]]]) -> List[List[Dict[str, str]]]:
    out = []
    for rows in groups.values():
        labels = [int(row["is_true_anchor"]) for row in rows]
        if sum(labels) == 1 and len(rows) >= 2:
            out.append(rows)
    return out


def row_features(row: Dict[str, str]) -> List[float]:
    return [float(row.get(name, 0.0) or 0.0) for name in FEATURE_NAMES]


def row_positive_target(row: Dict[str, str]) -> float:
    role = row.get("candidate_role", "")
    if int(row.get("is_true_anchor", 0) or 0) == 1:
        return 1.0
    if role == "known_positive_alt" or int(row.get("is_known_positive", 0) or 0) == 1:
        return 1.0
    return 0.0


def row_hard_target(row: Dict[str, str]) -> float:
    return 1.0 if int(row.get("is_hard_negative", 0) or 0) == 1 or row.get("candidate_role") == "hard_negative" else 0.0


def row_hard_weight(row: Dict[str, str]) -> float:
    try:
        value = float(row.get("hard_negative_weight", 1.0) or 1.0)
    except ValueError:
        value = 1.0
    return max(0.1, min(1.0, value))


def make_row_tensors(groups: Sequence[List[Dict[str, str]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = [row for group in groups for row in group]
    x = torch.tensor([row_features(row) for row in rows], dtype=torch.float32)
    positive_y = torch.tensor([row_positive_target(row) for row in rows], dtype=torch.float32)
    hard_y = torch.tensor([row_hard_target(row) for row in rows], dtype=torch.float32)
    hard_w = torch.tensor([row_hard_weight(row) if row_hard_target(row) else 1.0 for row in rows], dtype=torch.float32)
    return x, positive_y, hard_y, hard_w


def make_pair_tensors(groups: Sequence[List[Dict[str, str]]], seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    pos_features: List[List[float]] = []
    neg_features: List[List[float]] = []
    for group in groups:
        pos = [row for row in group if int(row["is_true_anchor"]) == 1][0]
        negs = [row for row in group if int(row["is_true_anchor"]) == 0]
        neg = rng.choice(negs)
        pos_features.append(row_features(pos))
        neg_features.append(row_features(neg))
    return torch.tensor(pos_features, dtype=torch.float32), torch.tensor(neg_features, dtype=torch.float32)


def make_positive_rank_tensors(groups: Sequence[List[Dict[str, str]]], seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    positive_features: List[List[float]] = []
    contrast_features: List[List[float]] = []
    for group in groups:
        positives = [row for row in group if row_positive_target(row) > 0.5]
        contrasts = [
            row
            for row in group
            if row_positive_target(row) < 0.5 and row.get("candidate_role") in {"hard_negative", "artifact", "unannotated_candidate"}
        ]
        if not positives or not contrasts:
            continue
        positive_features.append(row_features(rng.choice(positives)))
        contrast_features.append(row_features(rng.choice(contrasts)))
    return torch.tensor(positive_features, dtype=torch.float32), torch.tensor(contrast_features, dtype=torch.float32)


def make_hard_rank_tensors(groups: Sequence[List[Dict[str, str]]], seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    hard_features: List[List[float]] = []
    low_features: List[List[float]] = []
    for group in groups:
        hard_rows = [row for row in group if row_hard_target(row) > 0.5]
        low_rows = [
            row
            for row in group
            if row_hard_target(row) < 0.5 and row.get("candidate_role") in {"artifact", "observed_positive", "known_positive_alt"}
        ]
        if not hard_rows or not low_rows:
            continue
        hard_features.append(row_features(rng.choice(hard_rows)))
        low_features.append(row_features(rng.choice(low_rows)))
    return torch.tensor(hard_features, dtype=torch.float32), torch.tensor(low_features, dtype=torch.float32)


def evaluate(model: nn.Module, groups: Sequence[List[Dict[str, str]]], device: torch.device, head: str = "positive") -> Dict[str, float]:
    model.eval()
    top1 = 0
    reciprocal_rank = 0.0
    all_labels: List[int] = []
    all_scores: List[float] = []
    with torch.no_grad():
        for group in groups:
            x = torch.tensor([row_features(row) for row in group], dtype=torch.float32, device=device)
            scores = decoder_head_scores(model, x, head=head).detach().cpu().numpy()
            if head == "hard_negative":
                labels = np.array([int(row_hard_target(row)) for row in group], dtype=np.int64)
            else:
                labels = np.array([int(row_positive_target(row)) for row in group], dtype=np.int64)
            if labels.sum() == 0:
                continue
            true_index = int(np.where(labels == 1)[0][0])
            order = list(np.argsort(-scores))
            rank = order.index(true_index) + 1
            if rank == 1:
                top1 += 1
            reciprocal_rank += 1.0 / rank
            all_labels.extend(labels.tolist())
            all_scores.extend(scores.tolist())
    evaluated_groups = max(int(len([group for group in groups if any((row_hard_target(row) if head == "hard_negative" else row_positive_target(row)) > 0.5 for row in group)])), 1)
    metrics = {
        "groups": int(len(groups)),
        "evaluated_groups": evaluated_groups,
        "top1_accuracy": float(top1 / evaluated_groups),
        "mrr": float(reciprocal_rank / evaluated_groups),
    }
    if len(set(all_labels)) > 1 and roc_auc_score is not None and average_precision_score is not None:
        metrics["row_roc_auc"] = float(roc_auc_score(all_labels, all_scores))
        metrics["row_auprc"] = float(average_precision_score(all_labels, all_scores))
    return metrics


def save_group_predictions(path: str, model: nn.Module, groups: Sequence[List[Dict[str, str]]], device: torch.device) -> None:
    fieldnames = [
        "pair_id",
        "source_id",
        "split",
        "is_true_anchor",
        "score",
        "hard_negative_score",
        "candidate_role",
        "is_known_positive",
        "is_hard_negative",
        "fragment_map",
        "true_anchor_map",
        "candidate_anchor_map",
        "edit_action",
        "candidate_reaction",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        model.eval()
        with torch.no_grad():
            for group in groups:
                x = torch.tensor([row_features(row) for row in group], dtype=torch.float32, device=device)
                scores = decoder_head_scores(model, x, head="positive").detach().cpu().numpy().tolist()
                hard_scores = decoder_head_scores(model, x, head="hard_negative").detach().cpu().numpy().tolist()
                for row, score, hard_score in zip(group, scores, hard_scores):
                    out = {name: row.get(name, "") for name in fieldnames}
                    out["score"] = f"{float(score):.8f}"
                    out["hard_negative_score"] = f"{float(hard_score):.8f}"
                    writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--loss-mode", choices=["masked_hard_negative", "observed_anchor"], default="masked_hard_negative")
    parser.add_argument("--positive-bce-weight", type=float, default=1.0)
    parser.add_argument("--hard-bce-weight", type=float, default=1.0)
    parser.add_argument("--positive-rank-weight", type=float, default=0.5)
    parser.add_argument("--hard-rank-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_candidate_rows(args.input)
    grouped = split_groups(rows)
    train_groups = valid_groups(grouped.get("train", {}))
    val_groups = valid_groups(grouped.get("val", {}))
    test_groups = valid_groups(grouped.get("test", {}))
    if not train_groups:
        raise RuntimeError("No valid train candidate groups found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = 2 if args.loss_mode == "masked_hard_negative" else 1
    model = EditDecoderMLP(len(FEATURE_NAMES), hidden_dim=args.hidden_dim, dropout=args.dropout, output_dim=output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    row_x, positive_y, hard_y, hard_w = make_row_tensors(train_groups)
    positive_pos = max(float(positive_y.sum()), 1.0)
    positive_neg = max(float((positive_y == 0).sum()), 1.0)
    hard_pos = max(float(hard_y.sum()), 1.0)
    hard_neg = max(float((hard_y == 0).sum()), 1.0)
    positive_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_neg / positive_pos], device=device))
    hard_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hard_neg / hard_pos], device=device), reduction="none")
    history = []
    best_score = -1.0
    best_path = os.path.join(args.output_dir, "best_reaction_center_edit_decoder.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.loss_mode == "observed_anchor":
            pos_x, neg_x = make_pair_tensors(train_groups, seed=args.seed + epoch)
            loader = DataLoader(TensorDataset(pos_x, neg_x), batch_size=args.batch_size, shuffle=True)
        else:
            loader = DataLoader(
                TensorDataset(row_x, positive_y, hard_y, hard_w),
                batch_size=args.batch_size,
                shuffle=True,
            )
            positive_rank_x, positive_rank_y = make_positive_rank_tensors(train_groups, seed=args.seed + epoch)
            hard_rank_x, hard_rank_y = make_hard_rank_tensors(train_groups, seed=args.seed + epoch)
            positive_rank_batches = (
                list(DataLoader(TensorDataset(positive_rank_x, positive_rank_y), batch_size=args.batch_size, shuffle=True))
                if positive_rank_x.ndim == 2 and len(positive_rank_x) > 0
                else []
            )
            hard_rank_batches = (
                list(DataLoader(TensorDataset(hard_rank_x, hard_rank_y), batch_size=args.batch_size, shuffle=True))
                if hard_rank_x.ndim == 2 and len(hard_rank_x) > 0
                else []
            )
        total_loss = 0.0
        total = 0
        for step, batch in enumerate(loader):
            if args.loss_mode == "observed_anchor":
                batch_pos, batch_neg = batch
                batch_pos = batch_pos.to(device)
                batch_neg = batch_neg.to(device)
                optimizer.zero_grad()
                pos_score = decoder_head_scores(model, batch_pos, head="positive")
                neg_score = decoder_head_scores(model, batch_neg, head="positive")
                loss = torch.nn.functional.softplus(args.margin - (pos_score - neg_score)).mean()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_pos)
                total += len(batch_pos)
            else:
                batch_x, batch_positive_y, batch_hard_y, batch_hard_w = batch
                batch_x = batch_x.to(device)
                batch_positive_y = batch_positive_y.to(device)
                batch_hard_y = batch_hard_y.to(device)
                batch_hard_w = batch_hard_w.to(device)
                optimizer.zero_grad()
                positive_logits = decoder_head_scores(model, batch_x, head="positive")
                hard_logits = decoder_head_scores(model, batch_x, head="hard_negative")
                loss = args.positive_bce_weight * positive_bce(positive_logits, batch_positive_y)
                hard_loss = hard_bce(hard_logits, batch_hard_y)
                loss = loss + args.hard_bce_weight * (hard_loss * batch_hard_w).mean()

                if positive_rank_batches:
                    rank_pos, rank_neg = positive_rank_batches[step % len(positive_rank_batches)]
                    rank_pos = rank_pos.to(device)
                    rank_neg = rank_neg.to(device)
                    pos_score = decoder_head_scores(model, rank_pos, head="positive")
                    neg_score = decoder_head_scores(model, rank_neg, head="positive")
                    loss = loss + args.positive_rank_weight * torch.nn.functional.softplus(
                        args.margin - (pos_score - neg_score)
                    ).mean()

                if hard_rank_batches:
                    rank_hard, rank_low = hard_rank_batches[step % len(hard_rank_batches)]
                    rank_hard = rank_hard.to(device)
                    rank_low = rank_low.to(device)
                    hard_score = decoder_head_scores(model, rank_hard, head="hard_negative")
                    low_score = decoder_head_scores(model, rank_low, head="hard_negative")
                    loss = loss + args.hard_rank_weight * torch.nn.functional.softplus(
                        args.margin - (hard_score - low_score)
                    ).mean()

                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_x)
                total += len(batch_x)

        eval_groups = val_groups if val_groups else train_groups
        val_metrics = evaluate(model, eval_groups, device, head="positive")
        hard_val_metrics = evaluate(model, eval_groups, device, head="hard_negative") if args.loss_mode == "masked_hard_negative" else {}
        history.append({"epoch": epoch, "loss": total_loss / max(total, 1), "val": val_metrics, "hard_val": hard_val_metrics})
        score = val_metrics.get("top1_accuracy", 0.0) + 0.25 * val_metrics.get("mrr", 0.0)
        if hard_val_metrics:
            score += 0.5 * hard_val_metrics.get("top1_accuracy", 0.0) + 0.125 * hard_val_metrics.get("mrr", 0.0)
        if score > best_score:
            best_score = float(score)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_names": FEATURE_NAMES,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "output_dim": output_dim,
                    "loss_mode": args.loss_mode,
                    "input_dim": len(FEATURE_NAMES),
                    "epoch": epoch,
                    "best_score": best_score,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = {
        "config": vars(args),
        "device": str(device),
        "counts": {
            "rows": len(rows),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "test_groups": len(test_groups),
        },
        "train": evaluate(model, train_groups, device, head="positive"),
        "train_hard": evaluate(model, train_groups, device, head="hard_negative") if args.loss_mode == "masked_hard_negative" else {},
        "val": evaluate(model, val_groups, device, head="positive") if val_groups else {},
        "val_hard": evaluate(model, val_groups, device, head="hard_negative") if val_groups and args.loss_mode == "masked_hard_negative" else {},
        "test": evaluate(model, test_groups, device, head="positive") if test_groups else {},
        "test_hard": evaluate(model, test_groups, device, head="hard_negative") if test_groups and args.loss_mode == "masked_hard_negative" else {},
        "history": history,
        "best_checkpoint": best_path,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    if val_groups:
        save_group_predictions(os.path.join(args.output_dir, "val_predictions.csv"), model, val_groups, device)
    if test_groups:
        save_group_predictions(os.path.join(args.output_dir, "test_predictions.csv"), model, test_groups, device)
    print(json.dumps({k: metrics[k] for k in ["device", "counts", "train", "val", "test", "best_checkpoint"]}, indent=2))


if __name__ == "__main__":
    main()
