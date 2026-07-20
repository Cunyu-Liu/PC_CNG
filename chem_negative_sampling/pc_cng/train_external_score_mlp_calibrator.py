#!/usr/bin/env python3
"""Train a frozen MLP score calibrator before held-out evaluation.

This script intentionally uses only existing score columns as inputs. It must
not use candidate_source, candidate_family, or any label-derived field as a
feature because those fields reveal how the benchmark candidate was created.
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
    "chemformer_group_z",
    "pc_cng_group_z",
    "pc_minus_chem_group_z",
    "chem_times_pc_group_z",
    "chemformer_rank01",
    "pc_cng_rank01",
    "chemformer_gap_to_top_z",
    "pc_cng_gap_to_top_z",
    "chemformer_group_minmax",
    "pc_cng_group_minmax",
    "log_group_size",
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
    return mean, math.sqrt(var) if var > 1e-12 else 1.0


def minmax(value: float, values: Sequence[float]) -> float:
    low = min(values)
    high = max(values)
    return (value - low) / (high - low) if high > low else 0.0


def rank01(values: Sequence[float]) -> List[float]:
    if len(values) <= 1:
        return [1.0 for _ in values]
    order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
    ranks = [0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    denom = max(len(values) - 1, 1)
    return [1.0 - rank / denom for rank in ranks]


def row_id_index(rows: Sequence[Row]) -> Dict[int, int]:
    return {id(row): idx for idx, row in enumerate(rows)}


def build_features(rows: Sequence[Row], primary_score: str, pc_score: str) -> Tuple[np.ndarray, np.ndarray]:
    features: List[List[float]] = []
    row_indices: List[int] = []
    global_index = row_id_index(rows)
    for group in group_rows(rows).values():
        usable: List[Tuple[Row, float, float]] = []
        for row in group:
            ext = parse_float(row.get(primary_score))
            pc = parse_float(row.get(pc_score))
            if ext is None or pc is None:
                continue
            usable.append((row, ext, pc))
        if not usable:
            continue
        ext_values = [ext for _, ext, _ in usable]
        pc_values = [pc for _, _, pc in usable]
        ext_mean, ext_std = mean_std(ext_values)
        pc_mean, pc_std = mean_std(pc_values)
        ext_ranks = rank01(ext_values)
        pc_ranks = rank01(pc_values)
        ext_top = max(ext_values)
        pc_top = max(pc_values)
        log_group_size = math.log1p(len(usable))
        for pos, (row, ext, pc) in enumerate(usable):
            ext_z = (ext - ext_mean) / ext_std
            pc_z = (pc - pc_mean) / pc_std
            features.append(
                [
                    ext_z,
                    pc_z,
                    pc_z - ext_z,
                    ext_z * pc_z,
                    ext_ranks[pos],
                    pc_ranks[pos],
                    (ext - ext_top) / ext_std,
                    (pc - pc_top) / pc_std,
                    minmax(ext, ext_values),
                    minmax(pc, pc_values),
                    log_group_size,
                ]
            )
            row_indices.append(global_index[id(row)])
    if not features:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=float), np.zeros((0,), dtype=int)
    return np.asarray(features, dtype=float), np.asarray(row_indices, dtype=int)


def standardize_train(features: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = features.copy()
    train_values = out[train_mask] if train_mask.any() else out
    means = train_values.mean(axis=0) if train_values.size else np.zeros(out.shape[1], dtype=float)
    stds = train_values.std(axis=0) if train_values.size else np.ones(out.shape[1], dtype=float)
    stds = np.where(stds > 1e-12, stds, 1.0)
    out = (out - means) / stds
    return out, means, stds


def apply_standardization(features: np.ndarray, means: Sequence[float], stds: Sequence[float]) -> np.ndarray:
    means_arr = np.asarray(means, dtype=float)
    stds_arr = np.asarray(stds, dtype=float)
    stds_arr = np.where(np.abs(stds_arr) > 1e-12, stds_arr, 1.0)
    return (features - means_arr) / stds_arr


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def softplus(values: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, values)


def build_train_pairs(rows: Sequence[Row], row_indices: np.ndarray, train_split: str) -> Tuple[np.ndarray, np.ndarray]:
    feature_by_row = {int(row_idx): feature_idx for feature_idx, row_idx in enumerate(row_indices)}
    id_to_row = row_id_index(rows)
    positives: List[int] = []
    negatives: List[int] = []
    for group in group_rows(rows).values():
        if not group or (group[0].get("split") or "unknown") != train_split:
            continue
        group_pos: List[int] = []
        group_neg: List[int] = []
        for row in group:
            row_idx = id_to_row.get(id(row))
            if row_idx is None or row_idx not in feature_by_row:
                continue
            feature_idx = feature_by_row[row_idx]
            if parse_label(row) > 0:
                group_pos.append(feature_idx)
            else:
                group_neg.append(feature_idx)
        for pos in group_pos:
            for neg in group_neg:
                positives.append(pos)
                negatives.append(neg)
    if not positives:
        raise SystemExit(f"No train pairs found for split {train_split!r}")
    return np.asarray(positives, dtype=int), np.asarray(negatives, dtype=int)


def init_model(input_dim: int, hidden_dim: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(0.0, 0.1, size=(input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim, dtype=float),
        "w2": rng.normal(0.0, 0.1, size=(hidden_dim,)),
        "b2": np.zeros(1, dtype=float),
    }


def forward(features: np.ndarray, model: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    hidden_pre = features @ model["w1"] + model["b1"]
    hidden = np.tanh(hidden_pre)
    scores = hidden @ model["w2"] + float(model["b2"][0])
    return hidden, scores


def train_pairwise_mlp(
    features: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    hidden_dim: int,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, float]]]:
    model = init_model(features.shape[1], hidden_dim, seed)
    adam_m = {key: np.zeros_like(value) for key, value in model.items()}
    adam_v = {key: np.zeros_like(value) for key, value in model.items()}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    history: List[Dict[str, float]] = []
    pair_count = max(len(pos_idx), 1)
    for epoch in range(1, epochs + 1):
        hidden, scores = forward(features, model)
        delta = scores[pos_idx] - scores[neg_idx]
        loss = float(softplus(-delta).mean())
        scale = -sigmoid(-delta) / pair_count
        grad_scores = np.zeros_like(scores)
        np.add.at(grad_scores, pos_idx, scale)
        np.add.at(grad_scores, neg_idx, -scale)

        grad_w2 = hidden.T @ grad_scores + l2 * model["w2"] / pair_count
        grad_b2 = np.asarray([grad_scores.sum()])
        grad_hidden = np.outer(grad_scores, model["w2"])
        grad_hidden_pre = grad_hidden * (1.0 - hidden * hidden)
        grad_w1 = features.T @ grad_hidden_pre + l2 * model["w1"] / pair_count
        grad_b1 = grad_hidden_pre.sum(axis=0)
        grads = {"w1": grad_w1, "b1": grad_b1, "w2": grad_w2, "b2": grad_b2}

        for key in model:
            adam_m[key] = beta1 * adam_m[key] + (1.0 - beta1) * grads[key]
            adam_v[key] = beta2 * adam_v[key] + (1.0 - beta2) * (grads[key] * grads[key])
            m_hat = adam_m[key] / (1.0 - beta1 ** epoch)
            v_hat = adam_v[key] / (1.0 - beta2 ** epoch)
            model[key] = model[key] - lr * m_hat / (np.sqrt(v_hat) + eps)

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            history.append({"epoch": float(epoch), "pairwise_logloss": loss})
    return model, history


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
    for split in sorted({row.get("split", "unknown") or "unknown" for row in rows}):
        split_groups = [group for group in groups.values() if (group[0].get("split") or "unknown") == split]
        by_split[split] = aggregate(
            [metrics for group in split_groups if (metrics := score_group(group, score_name)) is not None]
        )
    return {
        "overall": aggregate([metrics for group in groups.values() if (metrics := score_group(group, score_name)) is not None]),
        "by_split": by_split,
    }


def write_markdown(summary: Mapping[str, object], path: str, score_columns: Sequence[str]) -> None:
    metrics_by_score = dict(summary["metrics_by_score"])  # type: ignore[index]
    lines = [
        "# External Score MLP Calibrator Summary",
        "",
        f"Input: `{summary['input_candidate_scores']}`",
        "",
        "| Score | Split | Groups | Top-1 | Top-3 | MRR | NDCG |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for score in score_columns:
        by_split = dict(dict(metrics_by_score[score])["by_split"])
        for split, payload in by_split.items():
            metrics = dict(payload)
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
    lines.append("Boundary: trained only on the configured train split; held-out 5k scores are not used.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def serialize_model(model: Mapping[str, np.ndarray]) -> Dict[str, object]:
    return {key: value.tolist() for key, value in model.items()}


def deserialize_model(payload: Mapping[str, object]) -> Dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=float) for key, value in payload.items()}


def run_train(args: argparse.Namespace) -> Dict[str, object]:
    rows = read_rows(args.train_candidate_scores)
    raw_features, row_indices = build_features(rows, args.primary_score, args.pc_score)
    train_mask = np.asarray([(rows[int(idx)].get("split") or "unknown") == args.train_split for idx in row_indices], dtype=bool)
    features, means, stds = standardize_train(raw_features, train_mask)
    pos_idx, neg_idx = build_train_pairs(rows, row_indices, args.train_split)
    model, history = train_pairwise_mlp(
        features,
        pos_idx,
        neg_idx,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    _, scores = forward(features, model)
    for row_idx, score in zip(row_indices, scores):
        rows[int(row_idx)][args.score_name] = f"{float(score):.10g}"

    score_columns = [args.primary_score, args.pc_score]
    for candidate in args.comparison_score_column:
        if candidate in rows[0] and candidate not in score_columns:
            score_columns.append(candidate)
    score_columns.append(args.score_name)
    metrics_by_score = {score: evaluate(rows, score) for score in score_columns}

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "external_score_mlp_calibrator_model.json")
    scored_path = os.path.join(args.output_dir, "candidate_scores_with_mlp_calibrator.csv")
    summary_path = os.path.join(args.output_dir, "external_score_mlp_calibrator_summary.json")
    md_path = os.path.join(args.output_dir, "external_score_mlp_calibrator_summary.md")
    model_payload = {
        "model_name": args.model_name,
        "score_name": args.score_name,
        "recipe": "fixed_feature_group_score_pairwise_mlp",
        "primary_score": args.primary_score,
        "pc_score": args.pc_score,
        "feature_names": FEATURE_NAMES,
        "feature_means": means.tolist(),
        "feature_stds": stds.tolist(),
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "seed": args.seed,
        "train_split": args.train_split,
        "training_pairs": int(len(pos_idx)),
        "training_rows": int(train_mask.sum()),
        "parameters": serialize_model(model),
    }
    with open(model_path, "w", encoding="utf-8") as handle:
        json.dump(model_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_rows(scored_path, rows, [args.score_name])
    summary = {
        "input_candidate_scores": os.path.abspath(args.train_candidate_scores),
        "model_json": model_path,
        "scored_candidates_csv": scored_path,
        "score_name": args.score_name,
        "score_columns": score_columns,
        "metrics_by_score": metrics_by_score,
        "training_history": history,
        "decision": "frozen_preheldout_mlp_recipe_trained_on_existing_train_split_only",
        "notes": [
            "No held-out 5k rows or scores are used for training or model selection.",
            "Features are derived only from existing score columns within each group.",
            "candidate_source, candidate_family, and label-derived features are forbidden.",
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(summary, md_path, score_columns)
    print(json.dumps({"model": model_path, "summary": summary_path, "markdown": md_path}, indent=2))
    return summary


def run_apply(args: argparse.Namespace) -> Dict[str, object]:
    rows = read_rows(args.apply_candidate_scores)
    with open(args.model_json, encoding="utf-8") as handle:
        model_payload = json.load(handle)
    raw_features, row_indices = build_features(rows, model_payload["primary_score"], model_payload["pc_score"])
    features = apply_standardization(raw_features, model_payload["feature_means"], model_payload["feature_stds"])
    model = deserialize_model(model_payload["parameters"])
    _, scores = forward(features, model)
    score_name = args.score_name or model_payload["score_name"]
    for row_idx, score in zip(row_indices, scores):
        rows[int(row_idx)][score_name] = f"{float(score):.10g}"
    os.makedirs(args.output_dir, exist_ok=True)
    scored_path = os.path.join(args.output_dir, "candidate_scores_with_mlp_calibrator.csv")
    summary_path = os.path.join(args.output_dir, "external_score_mlp_calibrator_apply_summary.json")
    md_path = os.path.join(args.output_dir, "external_score_mlp_calibrator_apply_summary.md")
    write_rows(scored_path, rows, [score_name])
    score_columns = [model_payload["primary_score"], model_payload["pc_score"], score_name]
    summary = {
        "input_candidate_scores": os.path.abspath(args.apply_candidate_scores),
        "model_json": os.path.abspath(args.model_json),
        "scored_candidates_csv": scored_path,
        "score_name": score_name,
        "metrics_by_score": {score: evaluate(rows, score) for score in score_columns},
        "decision": "applied_frozen_preheldout_mlp_recipe",
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(summary, md_path, score_columns)
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
    train.add_argument("--score-name", default="pc_cng_mlp_calibrator_v1")
    train.add_argument("--model-name", default="pc_cng_mlp_calibrator_v1")
    train.add_argument("--hidden-dim", type=int, default=16)
    train.add_argument("--epochs", type=int, default=1000)
    train.add_argument("--learning-rate", type=float, default=0.01)
    train.add_argument("--l2", type=float, default=0.0001)
    train.add_argument("--seed", type=int, default=20260715)
    train.add_argument("--comparison-score-column", action="append", default=[
        "hybrid_pc_cng_w0p00",
        "hybrid_pc_cng_w0p25",
        "hybrid_pc_cng_w0p50",
        "hybrid_pc_cng_w0p75",
        "hybrid_pc_cng_w1p00",
        "pc_cng_pairwise_calibrator_v1",
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
