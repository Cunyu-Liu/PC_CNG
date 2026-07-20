#!/usr/bin/env python3
"""P2-04 MLP calibrator v2 (chemformer-aware).

Warm-starts from the P1-01 v1 calibrator and fine-tunes on chemformer beam
candidates (held-out full-beam 5k). Trains 10 MLPs (one per seed) with early
stopping on validation Top-1, then runs a paired t-test vs the Chemformer
likelihood baseline across the 10 seeds.

This script extends v1 by importing its primitives and adding:
  * ``--warm-start``   : path to v1 model JSON (loaded via ``deserialize_model``)
  * ``--candidates``   : full_candidates.csv with 59,300 chemformer beam rows
  * ``--seeds``        : comma-separated 10 seeds
  * ``--train-split-ratio`` : 80/20 train/val group split
  * Early stopping on val Top-1 (patience=20 epochs)
  * 10-seed paired t-test vs Chemformer LL baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .train_external_score_mlp_calibrator import (
    FEATURE_NAMES,
    Row,
    apply_standardization,
    build_features,
    deserialize_model,
    evaluate,
    forward,
    group_rows,
    init_model,
    parse_float,
    parse_label,
    read_rows,
    score_group,
    serialize_model,
    sigmoid,
    softplus,
    standardize_train,
    train_pairwise_mlp,
)


DEFAULT_SEEDS = (
    "20260710,20260711,20260712,20260713,20260714,"
    "20260715,20260716,20260717,20260718,20260719"
)
METRIC_NAMES = ["top1", "top3", "top5", "ndcg10"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to full_candidates.csv with 59,300 chemformer beam rows.",
    )
    parser.add_argument(
        "--warm-start",
        required=True,
        help="Path to v1 model JSON (warm-start initialization).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for v2 model + per-seed metrics + paired significance.",
    )
    parser.add_argument(
        "--seeds",
        default=DEFAULT_SEEDS,
        help="Comma-separated 10 seeds (default: 20260710..20260719).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Hidden layer size (default 32, doubled from v1's 16).",
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20,
        help="Stop if val Top-1 does not improve for N consecutive epochs.",
    )
    parser.add_argument(
        "--train-split-ratio",
        type=float,
        default=0.8,
        help="Fraction of groups used for training (rest is validation).",
    )
    parser.add_argument("--primary-score", default="chemformer_likelihood")
    parser.add_argument("--pc-score", default="pc_cng")
    parser.add_argument(
        "--score-name",
        default="pc_cng_mlp_calibrator_v2",
        help="Column name written for the v2 calibrated score.",
    )
    parser.add_argument(
        "--model-name",
        default="pc_cng_mlp_calibrator_v2_chemformer_aware",
    )
    return parser.parse_args(argv)


def warm_start_model(
    warm_start_path: str,
    hidden_dim: int,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Load v1 model JSON and adapt weights to v2 ``hidden_dim``.

    * If ``hidden_dim == v1.hidden_dim``, the v1 parameters are reused as-is.
    * If ``hidden_dim > v1.hidden_dim``, the extra hidden units are initialized
      with small Gaussian noise (matching v1's ``init_model`` scale).
    * If ``hidden_dim < v1.hidden_dim``, the v1 parameters are truncated.

    Returns ``(model_dict, v1_payload)`` where ``v1_payload`` is the full JSON
    payload (used for feature_means/feature_stds and metadata).
    """
    with open(warm_start_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    v1_params = deserialize_model(payload["parameters"])
    v1_hidden = int(payload.get("hidden_dim", v1_params["w1"].shape[1]))
    rng = np.random.default_rng(seed)

    if hidden_dim == v1_hidden:
        model = {key: value.copy() for key, value in v1_params.items()}
    elif hidden_dim > v1_hidden:
        extra = hidden_dim - v1_hidden
        w1_extra = rng.normal(0.0, 0.1, size=(v1_params["w1"].shape[0], extra))
        b1_extra = rng.normal(0.0, 0.1, size=(extra,))
        w2_extra = rng.normal(0.0, 0.1, size=(extra,))
        model = {
            "w1": np.concatenate([v1_params["w1"], w1_extra], axis=1),
            "b1": np.concatenate([v1_params["b1"], b1_extra], axis=0),
            "w2": np.concatenate([v1_params["w2"], w2_extra], axis=0),
            "b2": v1_params["b2"].copy(),
        }
    else:
        model = {
            "w1": v1_params["w1"][:, :hidden_dim].copy(),
            "b1": v1_params["b1"][:hidden_dim].copy(),
            "w2": v1_params["w2"][:hidden_dim].copy(),
            "b2": v1_params["b2"].copy(),
        }
    return model, payload


def _split_group_ids(
    group_ids: Sequence[str],
    train_ratio: float,
    seed: int,
) -> Tuple[set, set]:
    """Deterministic train/val split of group IDs using a seeded permutation."""
    sorted_ids = sorted(set(group_ids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(sorted_ids))
    n_train = int(round(len(sorted_ids) * train_ratio))
    train_ids = {sorted_ids[int(i)] for i in perm[:n_train]}
    val_ids = {sorted_ids[int(i)] for i in perm[n_train:]}
    return train_ids, val_ids


def _build_pairs_for_split(
    rows: Sequence[Row],
    row_indices: np.ndarray,
    split_group_ids: set,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (positive, negative) feature-index pairs for groups in the split."""
    feature_by_row = {int(idx): feat_idx for feat_idx, idx in enumerate(row_indices)}
    id_to_row = {id(row): idx for idx, row in enumerate(rows)}
    positives: List[int] = []
    negatives: List[int] = []
    for group_id, group in group_rows(rows).items():
        if group_id not in split_group_ids:
            continue
        group_pos: List[int] = []
        group_neg: List[int] = []
        for row in group:
            row_idx = id_to_row.get(id(row))
            if row_idx is None or row_idx not in feature_by_row:
                continue
            feat_idx = feature_by_row[row_idx]
            if parse_label(row) > 0:
                group_pos.append(feat_idx)
            else:
                group_neg.append(feat_idx)
        for pos in group_pos:
            for neg in group_neg:
                positives.append(pos)
                negatives.append(neg)
    if not positives:
        return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)
    return np.asarray(positives, dtype=int), np.asarray(negatives, dtype=int)


def _ranking_metrics_from_scores(
    scored: Sequence[Tuple[float, int]],
) -> Dict[str, float]:
    """Compute Top-1, Top-3, Top-5, NDCG@10 from (score, label) pairs."""
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    labels = [label for _, label in ranked]
    if not labels:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "ndcg10": 0.0}
    top1 = 1.0 if labels[0] > 0 else 0.0
    top3 = 1.0 if any(label > 0 for label in labels[:3]) else 0.0
    top5 = 1.0 if any(label > 0 for label in labels[:5]) else 0.0
    dcg = sum(
        (1.0 if label > 0 else 0.0) / math.log2(idx + 2)
        for idx, label in enumerate(labels[:10])
    )
    ideal_labels = sorted(labels, reverse=True)
    idcg = sum(
        (1.0 if label > 0 else 0.0) / math.log2(idx + 2)
        for idx, label in enumerate(ideal_labels[:10])
    )
    ndcg10 = dcg / idcg if idcg > 0 else 0.0
    return {"top1": top1, "top3": top3, "top5": top5, "ndcg10": ndcg10}


def _evaluate_split(
    rows: Sequence[Row],
    row_indices: np.ndarray,
    scores: np.ndarray,
    split_group_ids: set,
) -> Dict[str, float]:
    """Evaluate Top-1/Top-3/Top-5/NDCG@10 averaged over groups in the split."""
    score_by_row_idx: Dict[int, float] = {}
    for i, ridx in enumerate(row_indices):
        score_by_row_idx[int(ridx)] = float(scores[i])
    id_to_row = {id(row): idx for idx, row in enumerate(rows)}
    items: List[Dict[str, float]] = []
    for group_id, group in group_rows(rows).items():
        if group_id not in split_group_ids:
            continue
        scored: List[Tuple[float, int]] = []
        for row in group:
            row_idx = id_to_row.get(id(row))
            if row_idx is None:
                continue
            score = score_by_row_idx.get(row_idx)
            if score is None:
                continue
            scored.append((score, parse_label(row)))
        if not scored:
            continue
        positives = sum(label > 0 for _, label in scored)
        negatives = sum(label <= 0 for _, label in scored)
        if positives == 0 or negatives == 0:
            continue
        items.append(_ranking_metrics_from_scores(scored))
    if not items:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "ndcg10": 0.0, "groups": 0}
    out: Dict[str, float] = {"groups": float(len(items))}
    for metric in METRIC_NAMES:
        out[metric] = sum(it[metric] for it in items) / len(items)
    return out


def _evaluate_baseline_split(
    rows: Sequence[Row],
    split_group_ids: set,
    primary_score: str,
) -> Dict[str, float]:
    """Evaluate the Chemformer LL baseline on a split (no model needed)."""
    items: List[Dict[str, float]] = []
    for group_id, group in group_rows(rows).items():
        if group_id not in split_group_ids:
            continue
        scored: List[Tuple[float, int]] = []
        for row in group:
            score = parse_float(row.get(primary_score))
            if score is None:
                continue
            scored.append((score, parse_label(row)))
        if not scored:
            continue
        positives = sum(label > 0 for _, label in scored)
        negatives = sum(label <= 0 for _, label in scored)
        if positives == 0 or negatives == 0:
            continue
        items.append(_ranking_metrics_from_scores(scored))
    if not items:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "ndcg10": 0.0, "groups": 0}
    out: Dict[str, float] = {"groups": float(len(items))}
    for metric in METRIC_NAMES:
        out[metric] = sum(it[metric] for it in items) / len(items)
    return out


def train_with_early_stopping(
    features: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    initial_model: Dict[str, np.ndarray],
    epochs: int,
    lr: float,
    l2: float,
    val_eval_fn: Callable[[Dict[str, np.ndarray]], Dict[str, float]],
    patience: int,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, float]]]:
    """Fine-tune ``initial_model`` with Adam, early-stopping on val Top-1.

    The per-epoch Adam update mirrors v1's ``train_pairwise_mlp`` so that the
    warm-started weights continue to be optimized with the same dynamics. The
    difference is that we evaluate validation Top-1 every epoch and revert to
    the best model when no improvement is seen for ``patience`` consecutive
    epochs.
    """
    model = {key: value.copy() for key, value in initial_model.items()}
    adam_m = {key: np.zeros_like(value) for key, value in model.items()}
    adam_v = {key: np.zeros_like(value) for key, value in model.items()}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    pair_count = max(len(pos_idx), 1)

    history: List[Dict[str, float]] = []
    best_val_top1 = -1.0
    best_model = {key: value.copy() for key, value in model.items()}
    epochs_since_improve = 0

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

        val_metrics = val_eval_fn(model)
        val_top1 = float(val_metrics.get("top1", 0.0))
        history.append(
            {
                "epoch": float(epoch),
                "pairwise_logloss": loss,
                "val_top1": val_top1,
                "val_top3": float(val_metrics.get("top3", 0.0)),
                "val_top5": float(val_metrics.get("top5", 0.0)),
                "val_ndcg10": float(val_metrics.get("ndcg10", 0.0)),
            }
        )

        if val_top1 > best_val_top1 + 1e-9:
            best_val_top1 = val_top1
            best_model = {key: value.copy() for key, value in model.items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                break

    return best_model, history


def paired_significance_test(
    v2_metrics: Sequence[float],
    baseline_metrics: Sequence[float],
) -> Dict[str, float]:
    """Paired t-test of v2 vs baseline across ``n_seeds`` paired runs.

    Returns mean_delta, std_delta, t_stat, p_value, ci_low, ci_high, n_seeds.
    Uses ``scipy.stats.ttest_rel`` for the paired t-statistic and p-value, and
    the t-distribution for the 95% confidence interval on the mean delta.
    """
    from scipy import stats

    v2 = np.asarray(v2_metrics, dtype=float)
    base = np.asarray(baseline_metrics, dtype=float)
    n = int(min(len(v2), len(base)))
    if n == 0:
        return {
            "mean_delta": 0.0,
            "std_delta": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n_seeds": 0,
        }
    deltas = v2[:n] - base[:n]
    mean_delta = float(deltas.mean())
    std_delta = float(deltas.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and std_delta > 1e-12:
        result = stats.ttest_rel(v2[:n], base[:n])
        t_stat = float(result.statistic)
        p_value = float(result.pvalue)
        se = std_delta / math.sqrt(n)
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        ci_low = mean_delta - t_crit * se
        ci_high = mean_delta + t_crit * se
    else:
        t_stat = 0.0
        p_value = 1.0
        ci_low = mean_delta
        ci_high = mean_delta
    return {
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_seeds": n,
    }


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var) if var > 1e-12 else 0.0


def run_train_v2(args: argparse.Namespace) -> Dict[str, object]:
    """Train 10 warm-started MLPs with early stopping and paired t-test."""
    rows = read_rows(args.candidates)
    raw_features, row_indices = build_features(rows, args.primary_score, args.pc_score)
    if raw_features.shape[0] == 0:
        raise SystemExit("No features built from candidates CSV.")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) != 10:
        raise SystemExit(f"Expected 10 seeds, got {len(seeds)}: {seeds}")

    group_ids = sorted({row.get("group_id", "") for row in rows})
    if not group_ids:
        raise SystemExit("No group_id found in candidates CSV.")

    # Load v1 warm-start payload (for metadata; per-seed warm_start_model is
    # called inside the loop so the padding RNG is seed-specific).
    _, v1_payload = warm_start_model(args.warm_start, args.hidden_dim, seeds[0])
    v1_hidden_dim = int(v1_payload.get("hidden_dim", 16))

    score_name = args.score_name
    os.makedirs(args.output_dir, exist_ok=True)

    per_seed_metrics: List[Dict[str, object]] = []

    for seed_idx, seed in enumerate(seeds):
        train_ids, val_ids = _split_group_ids(group_ids, args.train_split_ratio, seed)

        # Re-fit standardization on this seed's train split (v1 primitive).
        train_mask = np.asarray(
            [
                (rows[int(idx)].get("group_id", "") in train_ids)
                for idx in row_indices
            ],
            dtype=bool,
        )
        if not train_mask.any():
            raise SystemExit(f"No training rows for seed {seed}.")
        features, feat_means, feat_stds = standardize_train(raw_features, train_mask)

        pos_idx, neg_idx = _build_pairs_for_split(rows, row_indices, train_ids)
        if len(pos_idx) == 0:
            raise SystemExit(f"No training pairs for seed {seed}.")

        warm_model, _ = warm_start_model(args.warm_start, args.hidden_dim, seed)

        def val_eval_fn(
            model: Dict[str, np.ndarray],
            _rows=rows,
            _row_indices=row_indices,
            _features=features,
            _val_ids=val_ids,
        ) -> Dict[str, float]:
            _, scores = forward(_features, model)
            return _evaluate_split(_rows, _row_indices, scores, _val_ids)

        best_model, history = train_with_early_stopping(
            features=features,
            pos_idx=pos_idx,
            neg_idx=neg_idx,
            initial_model=warm_model,
            epochs=args.epochs,
            lr=args.learning_rate,
            l2=args.l2,
            val_eval_fn=val_eval_fn,
            patience=args.early_stopping_patience,
        )

        _, final_scores = forward(features, best_model)
        train_metrics = _evaluate_split(rows, row_indices, final_scores, train_ids)
        val_metrics = _evaluate_split(rows, row_indices, final_scores, val_ids)
        baseline_val = _evaluate_baseline_split(rows, val_ids, args.primary_score)

        model_path = os.path.join(args.output_dir, f"model_seed{seed}.json")
        model_payload = {
            "model_name": args.model_name,
            "score_name": score_name,
            "recipe": "warm_started_v1_chemformer_aware_pairwise_mlp_with_early_stopping",
            "primary_score": args.primary_score,
            "pc_score": args.pc_score,
            "feature_names": FEATURE_NAMES,
            "feature_means": feat_means.tolist(),
            "feature_stds": feat_stds.tolist(),
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "seed": seed,
            "train_split_ratio": args.train_split_ratio,
            "early_stopping_patience": args.early_stopping_patience,
            "warm_start_path": os.path.abspath(args.warm_start),
            "v1_hidden_dim": v1_hidden_dim,
            "training_pairs": int(len(pos_idx)),
            "train_groups": int(len(train_ids)),
            "val_groups": int(len(val_ids)),
            "best_val_top1": float(val_metrics["top1"]),
            "epochs_trained": len(history),
            "parameters": serialize_model(best_model),
        }
        with open(model_path, "w", encoding="utf-8") as handle:
            json.dump(model_payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

        per_seed_metrics.append(
            {
                "seed": seed,
                "train_top1": train_metrics["top1"],
                "train_top3": train_metrics["top3"],
                "train_top5": train_metrics["top5"],
                "train_ndcg10": train_metrics["ndcg10"],
                "val_top1": val_metrics["top1"],
                "val_top3": val_metrics["top3"],
                "val_top5": val_metrics["top5"],
                "val_ndcg10": val_metrics["ndcg10"],
                "baseline_val_top1": baseline_val["top1"],
                "baseline_val_top3": baseline_val["top3"],
                "baseline_val_top5": baseline_val["top5"],
                "baseline_val_ndcg10": baseline_val["ndcg10"],
                "epochs_trained": len(history),
                "model_path": model_path,
            }
        )
        print(
            f"[seed {seed}] val_top1={val_metrics['top1']:.4f} "
            f"baseline_val_top1={baseline_val['top1']:.4f} "
            f"epochs={len(history)}"
        )

    metrics_summary: Dict[str, object] = {}
    for metric in METRIC_NAMES:
        v2_vals = [float(s[f"val_{metric}"]) for s in per_seed_metrics]
        base_vals = [float(s[f"baseline_val_{metric}"]) for s in per_seed_metrics]
        m_v2, s_v2 = _mean_std(v2_vals)
        m_base, s_base = _mean_std(base_vals)
        sig = paired_significance_test(v2_vals, base_vals)
        metrics_summary[metric] = {
            "v2_mean": m_v2,
            "v2_std": s_v2,
            "baseline_mean": m_base,
            "baseline_std": s_base,
            "delta_mean": m_v2 - m_base,
            "paired_test": sig,
        }

    paired_path = os.path.join(args.output_dir, "paired_significance.json")
    paired_payload = {
        "metric": "top1",
        "v2_score_name": score_name,
        "baseline_score_name": args.primary_score,
        "n_seeds": len(seeds),
        "seeds": seeds,
        **metrics_summary["top1"]["paired_test"],  # type: ignore[arg-type]
        "v2_mean_top1": metrics_summary["top1"]["v2_mean"],
        "v2_std_top1": metrics_summary["top1"]["v2_std"],
        "baseline_mean_top1": metrics_summary["top1"]["baseline_mean"],
        "baseline_std_top1": metrics_summary["top1"]["baseline_std"],
    }
    with open(paired_path, "w", encoding="utf-8") as handle:
        json.dump(paired_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    summary_path = os.path.join(args.output_dir, "summary.json")
    delta_top1_pp = (
        metrics_summary["top1"]["v2_mean"] - metrics_summary["top1"]["baseline_mean"]  # type: ignore[index]
    ) * 100.0
    decision = "GO" if delta_top1_pp >= 1.0 else "NO-GO"
    summary_payload = {
        "task": "p2_04_mlp_calibrator_v2_chemformer_aware",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "warm_start_path": os.path.abspath(args.warm_start),
        "candidates_csv": os.path.abspath(args.candidates),
        "train_split_ratio": args.train_split_ratio,
        "metrics": metrics_summary,
        "decision_threshold_pp": 1.0,
        "v2_top1_mean": metrics_summary["top1"]["v2_mean"],
        "baseline_top1_mean": metrics_summary["top1"]["baseline_mean"],
        "delta_top1_pp": delta_top1_pp,
        "decision": decision,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    recipe_path = os.path.join(args.output_dir, "v2_calibrator_recipe.json")
    recipe_payload = {
        "warm_start_path": os.path.abspath(args.warm_start),
        "warm_start_hidden_dim": v1_hidden_dim,
        "v2_hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "early_stopping_patience": args.early_stopping_patience,
        "train_split_ratio": args.train_split_ratio,
        "primary_score": args.primary_score,
        "pc_score": args.pc_score,
        "feature_names": FEATURE_NAMES,
        "n_train_groups_per_seed": int(round(len(group_ids) * args.train_split_ratio)),
        "n_val_groups_per_seed": int(round(len(group_ids) * (1.0 - args.train_split_ratio))),
        "n_seeds": len(seeds),
        "seeds": seeds,
    }
    with open(recipe_path, "w", encoding="utf-8") as handle:
        json.dump(recipe_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    csv_path = os.path.join(args.output_dir, "per_seed_metrics.csv")
    fieldnames = list(per_seed_metrics[0].keys()) if per_seed_metrics else []
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in per_seed_metrics:
            writer.writerow(row)

    sig_top1 = metrics_summary["top1"]["paired_test"]  # type: ignore[index]
    print("\n=== P2-04 v2 Calibrator Summary ===")
    print(
        f"V2 Top-1: {metrics_summary['top1']['v2_mean'] * 100:.2f} "  # type: ignore[index]
        f"+/- {metrics_summary['top1']['v2_std'] * 100:.2f} pp"  # type: ignore[index]
    )
    print(
        f"Baseline (Chemformer LL) Top-1: "
        f"{metrics_summary['top1']['baseline_mean'] * 100:.2f} "  # type: ignore[index]
        f"+/- {metrics_summary['top1']['baseline_std'] * 100:.2f} pp"  # type: ignore[index]
    )
    print(f"Delta: {delta_top1_pp:+.2f} pp")
    print(
        f"Paired t-stat: {sig_top1['t_stat']:.4f}, "  # type: ignore[index]
        f"p-value: {sig_top1['p_value']:.4g}"  # type: ignore[index]
    )
    print(
        f"95% CI: [{sig_top1['ci_low'] * 100:.2f}, "  # type: ignore[index]
        f"{sig_top1['ci_high'] * 100:.2f}] pp"  # type: ignore[index]
    )
    print(f"Decision: {decision} (threshold: +1.0 pp)")

    return {
        "summary_path": summary_path,
        "paired_significance_path": paired_path,
        "recipe_path": recipe_path,
        "per_seed_metrics_csv": csv_path,
        "metrics": metrics_summary,
        "decision": decision,
        "delta_top1_pp": delta_top1_pp,
    }


def main() -> None:
    args = parse_args()
    run_train_v2(args)


if __name__ == "__main__":
    main()
