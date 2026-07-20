#!/usr/bin/env python3
"""Train or apply a frozen external-score calibration recipe.

The recipe is deliberately small and auditable: group-zscore the Chemformer
likelihood and PC-CNG scores, then fit a fixed-feature logistic scorer on a
predeclared split. Hyperparameters are CLI-configured and should be frozen
before scoring any held-out benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


Row = Dict[str, str]
FEATURE_NAMES = [
    "bias",
    "chemformer_group_z",
    "pc_cng_group_z",
    "pc_minus_chem_group_z",
    "chem_times_pc_group_z",
    "pc_cng_group_z_squared",
]


def parse_float(value: object) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_label(row: Mapping[str, str]) -> int:
    value = parse_float(row.get("label"))
    return int(value or 0)


def read_rows(path: str) -> List[Row]:
    with open(path, newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: str, rows: Sequence[Row], extra_fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows: Iterable[Row]) -> Dict[str, List[Row]]:
    groups: Dict[str, List[Row]] = defaultdict(list)
    for row in rows:
        groups[row.get("group_id", "")].append(row)
    return dict(groups)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(var) if var > 1e-12 else 1.0
    return mean, std


def build_raw_features(
    rows: Sequence[Row],
    primary_score: str,
    pc_score: str,
) -> Tuple[np.ndarray, List[int]]:
    feature_rows: List[List[float]] = []
    indices: List[int] = []
    for group in group_rows(rows).values():
        usable: List[Tuple[int, float, float]] = []
        for idx, row in enumerate(group):
            ext = parse_float(row.get(primary_score))
            pc = parse_float(row.get(pc_score))
            if ext is None or pc is None:
                continue
            usable.append((idx, ext, pc))
        ext_mean, ext_std = mean_std([ext for _, ext, _ in usable])
        pc_mean, pc_std = mean_std([pc for _, _, pc in usable])
        for local_idx, ext, pc in usable:
            ext_z = (ext - ext_mean) / ext_std
            pc_z = (pc - pc_mean) / pc_std
            feature_rows.append(
                [
                    1.0,
                    ext_z,
                    pc_z,
                    pc_z - ext_z,
                    ext_z * pc_z,
                    pc_z * pc_z,
                ]
            )
            indices.append(id(group[local_idx]))
    if not feature_rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=float), []
    return np.asarray(feature_rows, dtype=float), indices


def row_id_index(rows: Sequence[Row]) -> Dict[int, int]:
    return {id(row): idx for idx, row in enumerate(rows)}


def standardize_features(
    features: np.ndarray,
    train_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = features.copy()
    means = np.zeros(out.shape[1], dtype=float)
    stds = np.ones(out.shape[1], dtype=float)
    if out.shape[0] == 0:
        return out, means, stds
    for col in range(1, out.shape[1]):  # keep bias fixed at 1.0
        train_values = out[train_mask, col] if train_mask.any() else out[:, col]
        mean = float(train_values.mean()) if train_values.size else 0.0
        std = float(train_values.std()) if train_values.size and train_values.std() > 1e-12 else 1.0
        out[:, col] = (out[:, col] - mean) / std
        means[col] = mean
        stds[col] = std
    return out, means, stds


def apply_standardization(features: np.ndarray, means: Sequence[float], stds: Sequence[float]) -> np.ndarray:
    out = features.copy()
    for col in range(1, out.shape[1]):
        std = float(stds[col]) if abs(float(stds[col])) > 1e-12 else 1.0
        out[:, col] = (out[:, col] - float(means[col])) / std
    return out


def sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def balanced_weights(labels: np.ndarray) -> np.ndarray:
    positives = int(labels.sum())
    negatives = int(labels.shape[0] - positives)
    weights = np.ones(labels.shape[0], dtype=float)
    if positives and negatives:
        weights[labels == 1] = labels.shape[0] / (2.0 * positives)
        weights[labels == 0] = labels.shape[0] / (2.0 * negatives)
    return weights


def train_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    coef = np.zeros(features.shape[1], dtype=float)
    sample_weights = balanced_weights(labels)
    weight_sum = max(float(sample_weights.sum()), 1.0)
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        logits = features @ coef
        probs = sigmoid(logits)
        error = (probs - labels) * sample_weights
        grad = (features.T @ error) / weight_sum
        reg = coef.copy()
        reg[0] = 0.0
        grad += l2 * reg / max(features.shape[0], 1)
        coef -= lr * grad
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            eps = 1e-9
            loss_vec = -(labels * np.log(probs + eps) + (1.0 - labels) * np.log(1.0 - probs + eps))
            loss = float((loss_vec * sample_weights).sum() / weight_sum)
            history.append({"epoch": float(epoch), "weighted_logloss": loss})
    return coef, history


def train_pairwise(
    features: np.ndarray,
    row_indices: np.ndarray,
    rows: Sequence[Row],
    train_split: str,
    epochs: int,
    lr: float,
    l2: float,
) -> Tuple[np.ndarray, List[Dict[str, float]], int]:
    feature_by_row = {int(row_idx): pos for pos, row_idx in enumerate(row_indices)}
    id_to_row_index = row_id_index(rows)
    diffs: List[np.ndarray] = []
    for group in group_rows(rows).values():
        if not group or (group[0].get("split") or "unknown") != train_split:
            continue
        positives = []
        negatives = []
        for row in group:
            row_idx = id_to_row_index.get(id(row))
            if row_idx is None or row_idx not in feature_by_row:
                continue
            feature_idx = feature_by_row[row_idx]
            if parse_label(row) > 0:
                positives.append(features[feature_idx])
            else:
                negatives.append(features[feature_idx])
        for pos_features in positives:
            for neg_features in negatives:
                diffs.append(pos_features - neg_features)
    if not diffs:
        raise SystemExit(f"No train pairs found for split {train_split!r}")
    pair_features = np.asarray(diffs, dtype=float)
    coef = np.zeros(features.shape[1], dtype=float)
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        logits = pair_features @ coef
        # d/dx log(1 + exp(-x)) = -sigmoid(-x)
        scale = -sigmoid(-logits)
        grad = (pair_features.T @ scale) / max(pair_features.shape[0], 1)
        reg = coef.copy()
        reg[0] = 0.0
        grad += l2 * reg / max(pair_features.shape[0], 1)
        coef -= lr * grad
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            loss = float(np.logaddexp(0.0, -logits).mean())
            history.append({"epoch": float(epoch), "pairwise_logloss": loss})
    return coef, history, int(pair_features.shape[0])


def score_group(group: Sequence[Row], score_name: str) -> Optional[Dict[str, float]]:
    scored: List[Tuple[float, int]] = []
    for row in group:
        score = parse_float(row.get(score_name))
        if score is None:
            continue
        scored.append((score, parse_label(row)))
    if not scored:
        return None
    positives = sum(label > 0 for _, label in scored)
    negatives = sum(label <= 0 for _, label in scored)
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    labels = [label for _, label in ranked]
    first_positive_rank = next((idx + 1 for idx, label in enumerate(labels) if label > 0), None)
    if first_positive_rank is None:
        return None
    dcg = sum((1.0 if label > 0 else 0.0) / math.log2(idx + 2) for idx, label in enumerate(labels))
    ideal_labels = sorted(labels, reverse=True)
    idcg = sum((1.0 if label > 0 else 0.0) / math.log2(idx + 2) for idx, label in enumerate(ideal_labels))
    return {
        "candidate_rows": float(len(scored)),
        "positives": float(positives),
        "top1": 1.0 if labels[0] > 0 else 0.0,
        "top3": 1.0 if any(label > 0 for label in labels[:3]) else 0.0,
        "mrr": 1.0 / float(first_positive_rank),
        "ndcg": dcg / idcg if idcg > 0 else 0.0,
    }


def aggregate(items: Sequence[Dict[str, float]]) -> Dict[str, object]:
    if not items:
        return {"groups": 0, "candidate_rows": 0, "top1": None, "top3": None, "mrr": None, "ndcg": None}
    out: Dict[str, object] = {
        "groups": len(items),
        "candidate_rows": int(sum(item["candidate_rows"] for item in items)),
        "mean_candidates_per_group": sum(item["candidate_rows"] for item in items) / len(items),
    }
    for metric in ["top1", "top3", "mrr", "ndcg"]:
        out[metric] = sum(item[metric] for item in items) / len(items)
    return out


def evaluate(rows: Sequence[Row], score_name: str) -> Dict[str, object]:
    groups = group_rows(rows)
    by_split: Dict[str, object] = {}
    split_values = sorted({row.get("split", "unknown") or "unknown" for row in rows})
    for split in split_values:
        split_groups = [group for group in groups.values() if any((row.get("split") or "unknown") == split for row in group)]
        by_split[split] = aggregate(
            [metrics for group in split_groups if (metrics := score_group(group, score_name)) is not None]
        )
    return {
        "overall": aggregate([metrics for group in groups.values() if (metrics := score_group(group, score_name)) is not None]),
        "by_split": by_split,
    }


def markdown_table(summary: Mapping[str, object], score_names: Sequence[str]) -> str:
    metrics_by_score = dict(summary["metrics_by_score"])  # type: ignore[index]
    lines = [
        "# External Score Calibrator Training Summary",
        "",
        f"Input: `{summary['input_candidate_scores']}`",
        "",
        "| Score | Split | Groups | Top-1 | Top-3 | MRR | NDCG |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for score in score_names:
        payload = dict(metrics_by_score[score])
        for split, metrics_obj in dict(payload["by_split"]).items():
            metrics = dict(metrics_obj)
            if not metrics.get("groups"):
                continue
            lines.append(
                "| {score} | {split} | {groups} | {top1:.4f} | {top3:.4f} | {mrr:.4f} | {ndcg:.4f} |".format(
                    score=score,
                    split=split,
                    groups=metrics["groups"],
                    top1=metrics["top1"],
                    top3=metrics["top3"],
                    mrr=metrics["mrr"],
                    ndcg=metrics["ndcg"],
                )
            )
    lines.extend(["", f"Decision: `{summary['decision']}`", ""])
    lines.append(
        "Boundary: this model is trained only on the configured training split and is frozen before any held-out 5k scoring."
    )
    return "\n".join(lines) + "\n"


def run_train(args: argparse.Namespace) -> Dict[str, object]:
    rows = read_rows(args.train_candidate_scores)
    raw_features, feature_row_ids = build_raw_features(rows, args.primary_score, args.pc_score)
    id_to_index = row_id_index(rows)
    row_indices = np.asarray([id_to_index[row_id] for row_id in feature_row_ids], dtype=int)
    labels = np.asarray([parse_label(rows[idx]) for idx in row_indices], dtype=float)
    train_mask = np.asarray([(rows[idx].get("split") or "unknown") == args.train_split for idx in row_indices], dtype=bool)
    if not train_mask.any():
        raise SystemExit(f"No rows found for train split {args.train_split!r}")
    features, feature_means, feature_stds = standardize_features(raw_features, train_mask)
    if args.objective == "pointwise":
        coef, history = train_logistic(features[train_mask], labels[train_mask], args.epochs, args.learning_rate, args.l2)
        train_pair_count = None
    else:
        coef, history, train_pair_count = train_pairwise(
            features,
            row_indices,
            rows,
            args.train_split,
            args.epochs,
            args.learning_rate,
            args.l2,
        )
    scores = features @ coef
    score_name = args.score_name
    for idx, score in zip(row_indices, scores):
        rows[int(idx)][score_name] = f"{float(score):.10g}"

    score_columns = [args.primary_score, args.pc_score]
    for candidate in args.comparison_score_column:
        if candidate in rows[0] and candidate not in score_columns:
            score_columns.append(candidate)
    score_columns.append(score_name)

    metrics_by_score = {score: evaluate(rows, score) for score in score_columns}
    model = {
        "model_name": args.model_name,
        "score_name": score_name,
        "recipe": f"fixed_feature_group_zscore_{args.objective}_logistic_regression",
        "objective": args.objective,
        "feature_names": FEATURE_NAMES,
        "primary_score": args.primary_score,
        "pc_score": args.pc_score,
        "train_split": args.train_split,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "feature_means": [float(value) for value in feature_means],
        "feature_stds": [float(value) for value in feature_stds],
        "coefficients": [float(value) for value in coef],
        "training_rows": int(train_mask.sum()),
        "training_pairs": train_pair_count,
        "all_scored_rows": int(raw_features.shape[0]),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "external_score_calibrator_model.json")
    scored_path = os.path.join(args.output_dir, "candidate_scores_with_calibrator.csv")
    summary_path = os.path.join(args.output_dir, "external_score_calibrator_summary.json")
    md_path = os.path.join(args.output_dir, "external_score_calibrator_summary.md")
    with open(model_path, "w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_rows(scored_path, rows, [score_name])
    summary = {
        "input_candidate_scores": os.path.abspath(args.train_candidate_scores),
        "model_json": model_path,
        "scored_candidates_csv": scored_path,
        "score_name": score_name,
        "score_columns": score_columns,
        "metrics_by_score": metrics_by_score,
        "training_history": history,
        "decision": "frozen_preheldout_recipe_trained_on_existing_train_split_only",
        "notes": [
            "No held-out 5k rows or scores are used for training or model selection.",
            "Validation/test metrics on the repaired 25k file are audit evidence only.",
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_table(summary, score_columns))
    print(json.dumps({"model": model_path, "summary": summary_path, "markdown": md_path}, indent=2))
    return summary


def run_apply(args: argparse.Namespace) -> Dict[str, object]:
    rows = read_rows(args.apply_candidate_scores)
    with open(args.model_json, encoding="utf-8") as handle:
        model = json.load(handle)
    raw_features, feature_row_ids = build_raw_features(rows, model["primary_score"], model["pc_score"])
    features = apply_standardization(raw_features, model["feature_means"], model["feature_stds"])
    coef = np.asarray(model["coefficients"], dtype=float)
    id_to_index = row_id_index(rows)
    row_indices = [id_to_index[row_id] for row_id in feature_row_ids]
    scores = features @ coef
    score_name = args.score_name or model["score_name"]
    for idx, score in zip(row_indices, scores):
        rows[int(idx)][score_name] = f"{float(score):.10g}"
    os.makedirs(args.output_dir, exist_ok=True)
    scored_path = os.path.join(args.output_dir, "candidate_scores_with_calibrator.csv")
    summary_path = os.path.join(args.output_dir, "external_score_calibrator_apply_summary.json")
    md_path = os.path.join(args.output_dir, "external_score_calibrator_apply_summary.md")
    write_rows(scored_path, rows, [score_name])
    score_columns = [model["primary_score"], model["pc_score"], score_name]
    summary = {
        "input_candidate_scores": os.path.abspath(args.apply_candidate_scores),
        "model_json": os.path.abspath(args.model_json),
        "scored_candidates_csv": scored_path,
        "score_name": score_name,
        "metrics_by_score": {score: evaluate(rows, score) for score in score_columns},
        "decision": "applied_frozen_preheldout_recipe",
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_table(summary, score_columns))
    print(json.dumps({"summary": summary_path, "markdown": md_path}, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--train-candidate-scores", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--primary-score", default="chemformer_likelihood")
    train.add_argument("--pc-score", default="pc_cng")
    train.add_argument("--train-split", default="train")
    train.add_argument("--score-name", default="pc_cng_lr_calibrator_v1")
    train.add_argument("--model-name", default="pc_cng_lr_calibrator_v1")
    train.add_argument("--epochs", type=int, default=500)
    train.add_argument("--learning-rate", type=float, default=0.05)
    train.add_argument("--l2", type=float, default=1.0)
    train.add_argument("--objective", choices=["pointwise", "pairwise"], default="pointwise")
    train.add_argument("--comparison-score-column", action="append", default=[
        "hybrid_pc_cng_w0p00",
        "hybrid_pc_cng_w0p25",
        "hybrid_pc_cng_w0p50",
        "hybrid_pc_cng_w0p75",
        "hybrid_pc_cng_w1p00",
    ])

    apply = subparsers.add_parser("apply")
    apply.add_argument("--apply-candidate-scores", required=True)
    apply.add_argument("--model-json", required=True)
    apply.add_argument("--output-dir", required=True)
    apply.add_argument("--score-name", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "apply":
        run_apply(args)
    else:  # pragma: no cover
        raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
