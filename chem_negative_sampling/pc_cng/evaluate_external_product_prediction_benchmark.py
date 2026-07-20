"""Evaluate external product-prediction candidates with PC-CNG and LM scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .ranking_metrics import grouped_metrics, ranking_metrics


METRIC_NAMES = ["top1", "top3", "mrr", "ndcg"]
BASE_FIELDS = [
    "group_id",
    "source_id",
    "reactants",
    "agents",
    "candidate_product",
    "candidate_reaction",
    "label",
    "split",
    "dataset",
    "candidate_source",
    "candidate_family",
    "reaction_class",
    "pc_cng_score_status",
]


def sniff_dialect(path: str) -> csv.Dialect:
    with open(path, encoding="utf-8") as handle:
        sample = handle.read(4096)
    if "\t" in sample.splitlines()[0]:
        return csv.excel_tab
    return csv.excel


def row_key(row: Dict[str, object]) -> Tuple[str, str]:
    reaction = str(row.get("candidate_reaction") or row.get("reaction_smiles") or "")
    return str(row.get("group_id", "")), reaction


def read_candidate_rows(path: str) -> List[Dict[str, object]]:
    dialect = sniff_dialect(path)
    rows: List[Dict[str, object]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        for row in reader:
            reaction = row.get("candidate_reaction") or row.get("reaction_smiles") or ""
            if not row.get("group_id") or not reaction:
                continue
            out: Dict[str, object] = dict(row)
            out["candidate_reaction"] = reaction
            out["reaction_smiles"] = reaction
            out["label"] = int(row.get("label", 0) or 0)
            rows.append(out)
    return rows


def parse_external_score_spec(item: str) -> Tuple[str, str, str]:
    if "=" not in item:
        raise ValueError(f"Expected NAME=PATH[:COLUMN], got {item!r}")
    name, rest = item.split("=", 1)
    if not name:
        raise ValueError(f"Missing score name in {item!r}")
    if ":" in rest:
        path, column = rest.rsplit(":", 1)
    else:
        path, column = rest, "lm_score"
    if not path or not column:
        raise ValueError(f"Expected NAME=PATH[:COLUMN], got {item!r}")
    return name, path, column


def read_score_map(path: str, column: str) -> Tuple[Dict[Tuple[str, str], float], Dict[str, int]]:
    dialect = sniff_dialect(path)
    scores: Dict[Tuple[str, str], float] = {}
    stats = {"rows": 0, "usable_rows": 0, "missing_column": 0, "bad_score": 0}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        if column not in (reader.fieldnames or []):
            stats["missing_column"] = 1
            return scores, stats
        for row in reader:
            stats["rows"] += 1
            key = row_key(row)
            if not key[0] or not key[1]:
                continue
            try:
                value = float(row[column])
            except (TypeError, ValueError):
                stats["bad_score"] += 1
                continue
            if value != value:
                stats["bad_score"] += 1
                continue
            scores[key] = value
            stats["usable_rows"] += 1
    return scores, stats


def attach_score(
    rows: Sequence[Dict[str, object]],
    score_name: str,
    scores: Dict[Tuple[str, str], float],
) -> Dict[str, int]:
    missing = 0
    attached = 0
    for row in rows:
        value = scores.get(row_key(row))
        if value is None:
            missing += 1
            continue
        row[score_name] = float(value)
        attached += 1
    return {"attached": attached, "missing": missing}


def apply_missing_negative_score(
    rows: Sequence[Dict[str, object]],
    score_name: str,
    score_value: float,
    reason_field: str,
    reason: str,
) -> Dict[str, int]:
    filled = 0
    missing_positive = 0
    still_missing_negative = 0
    for row in rows:
        if is_finite_score(row, score_name):
            continue
        if int(row.get("label", 0) or 0) == 0:
            row[score_name] = float(score_value)
            row[reason_field] = reason
            filled += 1
        else:
            missing_positive += 1
    for row in rows:
        if int(row.get("label", 0) or 0) == 0 and not is_finite_score(row, score_name):
            still_missing_negative += 1
    return {
        "filled_negative_rows": filled,
        "missing_positive_rows": missing_positive,
        "still_missing_negative_rows": still_missing_negative,
        "score_value": float(score_value),
    }


def score_pc_cng_with_models(
    rows: Sequence[Dict[str, object]],
    model_dirs: Sequence[str],
    batch_size: int,
    device_name: str | None,
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, object]]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required when --model-dir is used") from exc

    from .evaluate_candidate_reranking import checkpoint_path, load_checkpoint
    from .train_feasibility_mlp import FeasibilityMLP, featurize_rows, make_reaction_featurizer, predict

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_rows = [
        {
            "group_id": row["group_id"],
            "source_id": row.get("source_id", ""),
            "reaction_smiles": row["candidate_reaction"],
            "label": row["label"],
            "split": row.get("split", "unknown"),
            "dataset": row.get("dataset", ""),
            "candidate_source": row.get("candidate_source", ""),
            "candidate_family": row.get("candidate_family", ""),
            "reaction_class": row.get("reaction_class", ""),
        }
        for row in rows
    ]
    checkpoints = [load_checkpoint(checkpoint_path(model_dir), device) for model_dir in model_dirs]
    if not checkpoints:
        return {}, {"device": str(device), "model_dirs": [], "scored_rows": 0}

    first = checkpoints[0]
    feature_mode = str(first.get("feature_mode", "morgan"))
    n_bits = int(first.get("n_bits", 4096))
    fp_mode = str(first.get("fp_mode", "binary"))
    include_descriptors = bool(first.get("include_descriptors", False))
    input_dim = int(first.get("input_dim", n_bits * 3))
    featurizer = make_reaction_featurizer(
        feature_mode=feature_mode,
        n_bits=n_bits,
        fp_mode=fp_mode,
        include_descriptors=include_descriptors,
    )
    features, _, _, kept = featurize_rows(model_rows, featurizer)
    if len(kept) == 0:
        return {}, {"device": str(device), "model_dirs": list(model_dirs), "scored_rows": 0}
    if features.shape[1] != input_dim:
        raise RuntimeError(f"Feature dimension mismatch: got {features.shape[1]}, checkpoint expects {input_dim}")

    scores_by_key: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for model_dir, checkpoint in zip(model_dirs, checkpoints):
        ckpt_feature_mode = str(checkpoint.get("feature_mode", "morgan"))
        ckpt_n_bits = int(checkpoint.get("n_bits", 4096))
        ckpt_fp_mode = str(checkpoint.get("fp_mode", "binary"))
        ckpt_include_descriptors = bool(checkpoint.get("include_descriptors", False))
        ckpt_input_dim = int(checkpoint.get("input_dim", input_dim))
        if (ckpt_feature_mode, ckpt_n_bits, ckpt_fp_mode, ckpt_include_descriptors, ckpt_input_dim) != (
            feature_mode,
            n_bits,
            fp_mode,
            include_descriptors,
            input_dim,
        ):
            raise RuntimeError(f"PC-CNG checkpoint feature config mismatch in {model_dir}")
        hidden_dim = int(checkpoint.get("hidden_dim", 2048))
        model = FeasibilityMLP(in_dim=input_dim, hidden_dim=hidden_dim, dropout=0.0).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model_scores = predict(model, features, device, batch_size)
        for row, score in zip(kept, model_scores.tolist()):
            scores_by_key[row_key(row)].append(float(score))

    scores = {key: sum(values) / len(values) for key, values in scores_by_key.items()}
    return scores, {
        "device": str(device),
        "model_dirs": list(model_dirs),
        "scored_rows": len(kept),
        "feature_reuse": "single featurization reused across all PC-CNG checkpoints",
        "feature_mode": feature_mode,
        "n_models": len(checkpoints),
    }


def is_finite_score(row: Dict[str, object], score_name: str) -> bool:
    value = row.get(score_name)
    if value is None:
        return False
    try:
        score = float(value)
    except (TypeError, ValueError):
        return False
    return score == score and math.isfinite(score)


def filter_complete_groups(rows: Sequence[Dict[str, object]], score_names: Sequence[str]) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    complete_rows = [row for row in rows if all(is_finite_score(row, score_name) for score_name in score_names)]
    missing_score_rows = len(rows) - len(complete_rows)

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in complete_rows:
        grouped[str(row["group_id"])].append(row)

    kept: List[Dict[str, object]] = []
    dropped_groups = 0
    dropped_rows = 0
    for group_rows in grouped.values():
        has_pos = any(int(row["label"]) == 1 for row in group_rows)
        has_neg = any(int(row["label"]) == 0 for row in group_rows)
        if has_pos and has_neg:
            kept.extend(group_rows)
        else:
            dropped_groups += 1
            dropped_rows += len(group_rows)
    return kept, {
        "kept_groups": len({str(row["group_id"]) for row in kept}),
        "kept_rows": len(kept),
        "candidate_rows_with_required_scores": len(complete_rows),
        "missing_score_rows": missing_score_rows,
        "dropped_groups_without_ranking_decision": dropped_groups,
        "dropped_rows_without_ranking_decision": dropped_rows,
    }


def normalize_scores(rows: Sequence[Dict[str, object]], score_name: str, output_name: str, mode: str) -> None:
    if mode == "none":
        for row in rows:
            row[output_name] = float(row[score_name])
        return

    if mode == "global_zscore":
        values = [float(row[score_name]) for row in rows]
        mean = sum(values) / max(len(values), 1)
        var = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
        std = math.sqrt(var) if var > 1e-12 else 1.0
        for row in rows:
            row[output_name] = (float(row[score_name]) - mean) / std
        return

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)

    for group_rows in grouped.values():
        values = [float(row[score_name]) for row in group_rows]
        if mode == "group_zscore":
            mean = sum(values) / max(len(values), 1)
            var = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
            std = math.sqrt(var) if var > 1e-12 else 1.0
            for row in group_rows:
                row[output_name] = (float(row[score_name]) - mean) / std
        elif mode == "group_minmax":
            low = min(values)
            high = max(values)
            denom = high - low if high > low else 1.0
            for row in group_rows:
                row[output_name] = (float(row[score_name]) - low) / denom
        else:
            raise ValueError(f"Unsupported normalization mode: {mode}")


def evaluate_score(rows: Sequence[Dict[str, object]], score_name: str) -> Dict[str, object]:
    scored = []
    for row in rows:
        if not is_finite_score(row, score_name):
            continue
        out = dict(row)
        out["score"] = float(row[score_name])
        scored.append(out)
    return {
        "rows": len(scored),
        "overall": ranking_metrics(scored),
        "by_split": grouped_metrics(scored, "split"),
        "by_dataset": grouped_metrics(scored, "dataset"),
        "by_candidate_source": grouped_metrics(scored, "candidate_source"),
        "by_candidate_family": grouped_metrics(scored, "candidate_family"),
        "by_reaction_class": grouped_metrics(scored, "reaction_class"),
    }


def add_hybrid_scores(
    rows: Sequence[Dict[str, object]],
    pc_norm: str,
    external_norm: str,
    weights: Sequence[float],
) -> List[Tuple[str, float]]:
    names: List[Tuple[str, float]] = []
    for weight in weights:
        name = f"hybrid_pc_cng_w{weight:.2f}".replace(".", "p")
        for row in rows:
            row[name] = weight * float(row[pc_norm]) + (1.0 - weight) * float(row[external_norm])
        names.append((name, weight))
    return names


def selection_value(metrics: Dict[str, object], split: str, metric: str) -> Tuple[float, float, float, float]:
    by_split = dict(metrics.get("by_split", {}))
    target = dict(by_split.get(split, {})) or dict(metrics.get("overall", {}))
    return (
        float(target.get(metric, 0.0)),
        float(target.get("mrr", 0.0)),
        float(target.get("ndcg", 0.0)),
        float(target.get("top3", 0.0)),
    )


def select_hybrid(
    hybrid_metrics: Dict[str, Dict[str, object]],
    weights: Dict[str, float],
    split: str,
    metric: str,
) -> Dict[str, object]:
    if not hybrid_metrics:
        return {}
    best_name = max(hybrid_metrics, key=lambda name: selection_value(hybrid_metrics[name], split, metric))
    return {
        "name": best_name,
        "pc_cng_weight": weights[best_name],
        "selection_split": split,
        "selection_metric": metric,
        "selection_value": selection_value(hybrid_metrics[best_name], split, metric)[0],
        "metrics": hybrid_metrics[best_name],
    }


def pct(metrics: Dict[str, object], metric: str) -> str:
    return f"{float(metrics.get(metric, 0.0)) * 100.0:.2f}"


def table_row(name: str, role: str, weight: str, metrics: Dict[str, object]) -> Dict[str, str]:
    overall = dict(metrics.get("overall", {}))
    by_split = dict(metrics.get("by_split", {}))
    val = dict(by_split.get("val", {}))
    test = dict(by_split.get("test", {}))
    return {
        "model": name,
        "role": role,
        "pc_cng_weight": weight,
        "groups": str(overall.get("groups", 0)),
        "overall_top1": pct(overall, "top1"),
        "overall_top3": pct(overall, "top3"),
        "overall_mrr": pct(overall, "mrr"),
        "overall_ndcg": pct(overall, "ndcg"),
        "val_top1": pct(val, "top1") if val else "",
        "test_top1": pct(test, "top1") if test else "",
        "test_mrr": pct(test, "mrr") if test else "",
        "test_ndcg": pct(test, "ndcg") if test else "",
    }


def write_csv(path: str, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fieldnames) + " |")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_scored_candidates(path: str, rows: Sequence[Dict[str, object]], score_names: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(BASE_FIELDS)
    for name in score_names:
        if name not in fieldnames:
            fieldnames.append(name)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def default_weights(weights: Sequence[float]) -> List[float]:
    if weights:
        return [float(weight) for weight in weights]
    return [0.0, 0.25, 0.5, 0.75, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--external-score", action="append", default=[], help="NAME=PATH[:COLUMN]")
    parser.add_argument("--primary-external-score", default=None)
    parser.add_argument("--model-dir", action="append", default=[])
    parser.add_argument("--pc-cng-score-csv", default=None)
    parser.add_argument("--pc-cng-score-column", default="score")
    parser.add_argument(
        "--pc-cng-invalid-negative-score",
        type=float,
        default=None,
        help="Assign this PC-CNG score to label=0 candidates that cannot be featurized/scored.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--normalization", choices=["none", "global_zscore", "group_zscore", "group_minmax"], default="group_zscore")
    parser.add_argument("--hybrid-weight", action="append", type=float, default=[])
    parser.add_argument("--selection-split", default="val")
    parser.add_argument("--selection-metric", choices=METRIC_NAMES, default="top1")
    parser.add_argument("--strict-complete-groups", dest="strict_complete_groups", action="store_true", default=True)
    parser.add_argument("--allow-incomplete-groups", dest="strict_complete_groups", action="store_false")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = read_candidate_rows(args.candidate_csv)
    external_stats: Dict[str, object] = {}
    external_names: List[str] = []
    for spec in args.external_score:
        name, path, column = parse_external_score_spec(spec)
        scores, stats = read_score_map(path, column)
        attach_stats = attach_score(rows, name, scores)
        external_stats[name] = {"path": path, "column": column, "read": stats, "attach": attach_stats}
        external_names.append(name)

    pc_cng_stats: Dict[str, object] = {}
    if args.model_dir:
        pc_scores, pc_cng_stats = score_pc_cng_with_models(rows, args.model_dir, args.batch_size, args.device)
        pc_cng_stats["attach"] = attach_score(rows, "pc_cng", pc_scores)
    elif args.pc_cng_score_csv:
        pc_scores, stats = read_score_map(args.pc_cng_score_csv, args.pc_cng_score_column)
        pc_cng_stats = {
            "path": args.pc_cng_score_csv,
            "column": args.pc_cng_score_column,
            "read": stats,
            "attach": attach_score(rows, "pc_cng", pc_scores),
        }
    if args.pc_cng_invalid_negative_score is not None:
        pc_cng_stats["invalid_negative_fill"] = apply_missing_negative_score(
            rows,
            score_name="pc_cng",
            score_value=args.pc_cng_invalid_negative_score,
            reason_field="pc_cng_score_status",
            reason="invalid_or_unfeaturizable_negative_penalty",
        )

    primary_external = args.primary_external_score or (external_names[0] if external_names else "")
    score_names = list(external_names)
    if any(is_finite_score(row, "pc_cng") for row in rows):
        score_names.append("pc_cng")

    eval_rows = list(rows)
    complete_filter = {}
    required_for_hybrid = [name for name in [primary_external, "pc_cng"] if name]
    if args.strict_complete_groups and required_for_hybrid:
        eval_rows, complete_filter = filter_complete_groups(eval_rows, required_for_hybrid)

    score_metrics: Dict[str, Dict[str, object]] = {}
    for name in score_names:
        score_metrics[name] = evaluate_score(eval_rows, name)

    hybrid_metrics: Dict[str, Dict[str, object]] = {}
    hybrid_weights: Dict[str, float] = {}
    hybrid_names: List[str] = []
    hybrid_filter: Dict[str, object] = {}
    if primary_external and "pc_cng" in score_names:
        if args.strict_complete_groups:
            hybrid_rows = eval_rows
            hybrid_filter = dict(complete_filter)
            hybrid_filter["same_rows_as_evaluation"] = True
        else:
            # Validity-aware evaluation intentionally allows method-specific
            # score coverage. Hybrids still require both scores, so compute
            # them on the shared scored subset without shrinking single-model
            # metrics such as the full Chemformer likelihood table.
            hybrid_rows, hybrid_filter = filter_complete_groups(eval_rows, required_for_hybrid)
            hybrid_filter["same_rows_as_evaluation"] = False

        normalize_scores(hybrid_rows, "pc_cng", "_pc_cng_norm", args.normalization)
        normalize_scores(hybrid_rows, primary_external, f"_{primary_external}_norm", args.normalization)
        for name, weight in add_hybrid_scores(
            hybrid_rows,
            pc_norm="_pc_cng_norm",
            external_norm=f"_{primary_external}_norm",
            weights=default_weights(args.hybrid_weight),
        ):
            hybrid_metrics[name] = evaluate_score(hybrid_rows, name)
            hybrid_weights[name] = weight
            hybrid_names.append(name)

    selected_hybrid = select_hybrid(hybrid_metrics, hybrid_weights, args.selection_split, args.selection_metric)
    scored_candidates_path = os.path.join(args.output_dir, "candidate_scores.csv")
    write_scored_candidates(scored_candidates_path, eval_rows, score_names + hybrid_names)

    table_rows = []
    for name in external_names:
        table_rows.append(table_row(name, "external_lm", "", score_metrics[name]))
    if "pc_cng" in score_metrics:
        table_rows.append(table_row("pc_cng", "pc_cng_pairwise", "1.00", score_metrics["pc_cng"]))
    for name in hybrid_names:
        table_rows.append(table_row(name, "hybrid", f"{hybrid_weights[name]:.2f}", hybrid_metrics[name]))
    table_fields = [
        "model",
        "role",
        "pc_cng_weight",
        "groups",
        "overall_top1",
        "overall_top3",
        "overall_mrr",
        "overall_ndcg",
        "val_top1",
        "test_top1",
        "test_mrr",
        "test_ndcg",
    ]
    write_csv(os.path.join(args.output_dir, "paper_table.csv"), table_rows, table_fields)
    write_markdown(os.path.join(args.output_dir, "paper_table.md"), table_rows, table_fields)

    summary = {
        "task": "strict_external_product_prediction_benchmark",
        "config": vars(args),
        "candidate_csv": args.candidate_csv,
        "candidate_rows_requested": len(rows),
        "candidate_rows_evaluated": len(eval_rows),
        "strict_complete_group_filter": complete_filter,
        "external_scores": external_stats,
        "pc_cng_score": pc_cng_stats,
        "primary_external_score": primary_external,
        "normalization": args.normalization,
        "hybrid_complete_group_filter": hybrid_filter,
        "score_metrics": score_metrics,
        "hybrid_metrics": hybrid_metrics,
        "selected_hybrid": selected_hybrid,
        "outputs": {
            "candidate_scores": scored_candidates_path,
            "paper_table_csv": os.path.join(args.output_dir, "paper_table.csv"),
            "paper_table_md": os.path.join(args.output_dir, "paper_table.md"),
        },
        "notes": [
            "When strict_complete_groups is enabled, evaluation first intersects candidate rows with required model scores, then keeps groups that still contain positive and negative candidates.",
            "If --pc-cng-invalid-negative-score is set, unfeaturizable label=0 candidates receive the configured low PC-CNG score and remain in the shared candidate set.",
            "Hybrid scores are computed after the configured score normalization within each group by default.",
            "The selected hybrid weight is chosen on the validation split when available; otherwise the overall metrics backstop selection.",
        ],
    }
    summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"summary": summary_path, "rows": len(eval_rows), "selected_hybrid": selected_hybrid.get("name")}, indent=2))


if __name__ == "__main__":
    main()
