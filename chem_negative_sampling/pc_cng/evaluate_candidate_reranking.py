"""Evaluate downstream candidate product reranking.

This script turns binary feasibility models into candidate rerankers by scoring
sets of plausible outcomes that share the same reaction context. It supports
two evaluation sources:

- real CSV rows: observed positives and real negatives grouped by reactants.
- synthetic CSV rows: PC-CNG candidates grouped with their parent positive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for evaluate_candidate_reranking.py") from exc

from .train_feasibility_mlp import FeasibilityMLP, featurize_rows, make_reaction_featurizer, predict


def load_checkpoint(path: str, device: torch.device) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Older torch.
        return torch.load(path, map_location=device)


def checkpoint_path(model_dir: str) -> str:
    for name in ["best_feasibility_mlp.pt", "best_pairwise_reward_mlp.pt"]:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No supported checkpoint found in {model_dir}")


class CheckpointScorer:
    def __init__(self, model_dir: str, device: torch.device) -> None:
        self.model_dir = model_dir
        path = checkpoint_path(model_dir)
        checkpoint = load_checkpoint(path, device)
        self.feature_mode = str(checkpoint.get("feature_mode", "morgan"))
        self.n_bits = int(checkpoint.get("n_bits", 4096))
        self.fp_mode = str(checkpoint.get("fp_mode", "binary"))
        self.include_descriptors = bool(checkpoint.get("include_descriptors", False))
        self.hidden_dim = int(checkpoint.get("hidden_dim", 2048))
        self.input_dim = int(checkpoint.get("input_dim", self.n_bits * 3))
        self.featurizer = make_reaction_featurizer(
            feature_mode=self.feature_mode,
            n_bits=self.n_bits,
            fp_mode=self.fp_mode,
            include_descriptors=self.include_descriptors,
        )
        self.model = FeasibilityMLP(in_dim=self.input_dim, hidden_dim=self.hidden_dim, dropout=0.0).to(device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.device = device

    def score(self, rows: Sequence[Dict[str, object]], batch_size: int) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        x, _, _, kept = featurize_rows(rows, self.featurizer)
        if len(kept) == 0:
            return np.zeros((0,), dtype=np.float32), []
        return predict(self.model, x, self.device, batch_size), kept


def read_real_rows(paths: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label_type = row.get("label_type", "")
                if label_type not in {"positive", "real_negative"}:
                    continue
                reaction = row.get("reaction_smiles", "")
                if not reaction:
                    continue
                out = dict(row)
                out["_input_path"] = path
                rows.append(out)
    return rows


def build_real_candidate_rows(
    real_rows: Sequence[Dict[str, str]],
    group_by: str,
    candidate_scope: str,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in real_rows:
        dataset = row.get("source") or os.path.basename(row.get("_input_path", "real"))
        group_value = row.get(group_by) or row.get("reactants") or row.get("split_key") or row.get("source_id")
        if not group_value:
            continue
        if candidate_scope == "same_split":
            key = ("real", dataset, row.get("split", "unknown"), group_value)
        elif candidate_scope == "all_group":
            key = ("real", dataset, "all", group_value)
        else:
            raise ValueError(f"Unsupported candidate_scope: {candidate_scope}")
        grouped[key].append(row)

    out: List[Dict[str, object]] = []
    for key, rows in grouped.items():
        positives = [row for row in rows if row.get("label_type") == "positive"]
        negatives = [row for row in rows if row.get("label_type") == "real_negative"]
        if not positives or not negatives:
            continue
        group_id = "|".join(key)
        for row in rows:
            out.append(
                {
                    "group_id": group_id,
                    "source_id": row.get("source_id", ""),
                    "reaction_smiles": row.get("reaction_smiles", ""),
                    "label": 1 if row.get("label_type") == "positive" else 0,
                    "split": row.get("split", "unknown") if candidate_scope == "same_split" else "all",
                    "dataset": key[1],
                    "candidate_source": "real",
                    "candidate_family": row.get("label_type", ""),
                    "reaction_class": row.get("reaction_class", ""),
                }
            )
    return out


def positive_lookup(real_rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    for row in real_rows:
        if row.get("label_type") == "positive" and row.get("source_id"):
            lookup[row["source_id"]] = row
    return lookup


def build_synthetic_candidate_rows(
    synthetic_paths: Sequence[str],
    positives: Dict[str, Dict[str, str]],
    review_statuses: Sequence[str],
) -> List[Dict[str, object]]:
    allowed_status = set(review_statuses)
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for path in synthetic_paths:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source_id = row.get("source_id", "")
                if source_id not in positives:
                    continue
                status = row.get("review_status", "keep_synthetic_negative")
                if allowed_status and status not in allowed_status:
                    continue
                reaction = row.get("candidate_reaction") or row.get("reaction_smiles") or ""
                if not reaction:
                    continue
                row = dict(row)
                row["_candidate_reaction"] = reaction
                grouped[source_id].append(row)

    out: List[Dict[str, object]] = []
    for source_id, negatives in grouped.items():
        pos = positives[source_id]
        if not negatives:
            continue
        dataset = pos.get("source") or "real"
        split = pos.get("split", "unknown")
        group_id = f"synthetic|{dataset}|{split}|{source_id}"
        out.append(
            {
                "group_id": group_id,
                "source_id": source_id,
                "reaction_smiles": pos.get("reaction_smiles", ""),
                "label": 1,
                "split": split,
                "dataset": dataset,
                "candidate_source": "synthetic",
                "candidate_family": "observed_positive",
                "reaction_class": pos.get("reaction_class", ""),
            }
        )
        for row in negatives:
            out.append(
                {
                    "group_id": group_id,
                    "source_id": source_id,
                    "reaction_smiles": row["_candidate_reaction"],
                    "label": 0,
                    "split": split,
                    "dataset": dataset,
                    "candidate_source": "synthetic",
                    "candidate_family": row.get("action_family") or row.get("failure_type", "synthetic_negative"),
                    "reaction_class": pos.get("reaction_class", ""),
                }
            )
    return out


def score_rows(
    rows: Sequence[Dict[str, object]],
    model_dirs: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    if not rows:
        return []
    scores_by_key: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    row_by_key: Dict[Tuple[str, str], Dict[str, object]] = {}
    for model_dir in model_dirs:
        scorer = CheckpointScorer(model_dir, device)
        scores, kept = scorer.score(rows, batch_size=batch_size)
        for row, score in zip(kept, scores.tolist()):
            key = (str(row["group_id"]), str(row["reaction_smiles"]))
            scores_by_key[key].append(float(score))
            row_by_key[key] = row

    out: List[Dict[str, object]] = []
    for key, scores in scores_by_key.items():
        row = dict(row_by_key[key])
        row["score"] = float(np.mean(scores))
        row["score_min"] = float(np.min(scores))
        row["score_max"] = float(np.max(scores))
        row["models_scored"] = len(scores)
        out.append(row)
    return out


def dcg(labels: Sequence[int]) -> float:
    return sum((1.0 if label else 0.0) / math.log2(rank + 1) for rank, label in enumerate(labels, start=1))


def ranking_metrics(rows: Sequence[Dict[str, object]]) -> Dict[str, float | int]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)

    groups = 0
    top1 = 0.0
    top3 = 0.0
    mrr = 0.0
    ndcg = 0.0
    rank_sum = 0.0
    candidate_count = 0
    positive_count = 0
    random_top1 = 0.0
    for group_rows in grouped.values():
        labels = [int(row["label"]) for row in group_rows]
        if not any(labels) or all(labels):
            continue
        ranked = sorted(group_rows, key=lambda row: float(row["score"]), reverse=True)
        ranked_labels = [int(row["label"]) for row in ranked]
        first_positive_rank = next(rank for rank, label in enumerate(ranked_labels, start=1) if label == 1)
        ideal = sorted(ranked_labels, reverse=True)
        groups += 1
        top1 += 1.0 if ranked_labels[0] == 1 else 0.0
        top3 += 1.0 if any(ranked_labels[:3]) else 0.0
        mrr += 1.0 / first_positive_rank
        ndcg += dcg(ranked_labels) / max(dcg(ideal), 1e-12)
        rank_sum += first_positive_rank
        candidate_count += len(group_rows)
        positives = sum(labels)
        positive_count += positives
        random_top1 += positives / len(group_rows)

    if groups == 0:
        return {
            "groups": 0,
            "candidate_rows": 0,
            "top1": 0.0,
            "top3": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "mean_first_positive_rank": 0.0,
            "mean_candidates_per_group": 0.0,
            "mean_positives_per_group": 0.0,
            "random_top1_expected": 0.0,
        }
    return {
        "groups": groups,
        "candidate_rows": candidate_count,
        "top1": top1 / groups,
        "top3": top3 / groups,
        "mrr": mrr / groups,
        "ndcg": ndcg / groups,
        "mean_first_positive_rank": rank_sum / groups,
        "mean_candidates_per_group": candidate_count / groups,
        "mean_positives_per_group": positive_count / groups,
        "random_top1_expected": random_top1 / groups,
    }


def grouped_metrics(rows: Sequence[Dict[str, object]], field: str) -> Dict[str, Dict[str, float | int]]:
    values = sorted({str(row.get(field, "") or "unknown") for row in rows})
    out: Dict[str, Dict[str, float | int]] = {}
    for value in values:
        subset = [row for row in rows if str(row.get(field, "") or "unknown") == value]
        metrics = ranking_metrics(subset)
        if int(metrics["groups"]) > 0:
            out[value] = metrics
    return out


def write_predictions(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "group_id",
        "source_id",
        "split",
        "dataset",
        "candidate_source",
        "candidate_family",
        "reaction_class",
        "label",
        "score",
        "score_min",
        "score_max",
        "models_scored",
        "reaction_smiles",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", action="append", default=[])
    parser.add_argument("--synthetic-csv", action="append", default=[])
    parser.add_argument("--model-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-by", default="reactants")
    parser.add_argument("--candidate-scope", choices=["same_split", "all_group"], default="same_split")
    parser.add_argument("--review-status", action="append", default=["keep_synthetic_negative"])
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    real_rows = read_real_rows(args.real_csv)
    candidates: List[Dict[str, object]] = []
    if args.real_csv:
        candidates.extend(
            build_real_candidate_rows(
                real_rows=real_rows,
                group_by=args.group_by,
                candidate_scope=args.candidate_scope,
            )
        )
    if args.synthetic_csv:
        candidates.extend(
            build_synthetic_candidate_rows(
                synthetic_paths=args.synthetic_csv,
                positives=positive_lookup(real_rows),
                review_statuses=args.review_status,
            )
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    scored = score_rows(candidates, args.model_dir, batch_size=args.batch_size, device=device)
    predictions_path = os.path.join(args.output_dir, "candidate_scores.csv")
    write_predictions(predictions_path, scored)

    summary: Dict[str, object] = {
        "config": vars(args),
        "device": str(device),
        "candidate_rows_requested": len(candidates),
        "candidate_rows_scored": len(scored),
        "overall": ranking_metrics(scored),
        "by_split": grouped_metrics(scored, "split"),
        "by_dataset": grouped_metrics(scored, "dataset"),
        "by_candidate_source": grouped_metrics(scored, "candidate_source"),
        "by_candidate_family": grouped_metrics(scored, "candidate_family"),
        "predictions": predictions_path,
    }
    summary_path = os.path.join(args.output_dir, "ranking_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
