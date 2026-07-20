"""Build manuscript-ready evidence tables for PC-CNG v3.

The project has multiple benchmark families with different scopes:

- Type-1 same-context candidate reranking.
- Chemformer-reference preference tuning.
- Science Advances-style low-positive protocols.
- Type-2 low-yield feasibility supervision.

This script collects the current JSON summaries into compact CSV/Markdown
tables while preserving the distinction between those scopes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence

from .evaluate_candidate_reranking import ranking_metrics


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def pct(stats: Dict[str, object]) -> str:
    return f"{float(stats.get('mean', 0.0)) * 100.0:.2f} +/- {float(stats.get('std', 0.0)) * 100.0:.2f}"


def pct_value(value: object) -> str:
    return f"{float(value or 0.0) * 100.0:.2f}"


def pct_mean_std(mean_value: object, std_value: object) -> str:
    mean_float = float(mean_value or 0.0)
    std_float = float(std_value or 0.0)
    if mean_float != mean_float:
        return "n/a"
    if std_float != std_float:
        std_float = 0.0
    return f"{mean_float * 100.0:.2f} +/- {std_float * 100.0:.2f}"


def val(stats: Dict[str, object]) -> float:
    return float(stats.get("mean", 0.0))


def summarize(values: Sequence[float]) -> Dict[str, object]:
    values = [float(value) for value in values if value == value]
    return {
        "mean": mean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def setting_metric(summary: Dict[str, object], setting: str, family: str, metric: str) -> Dict[str, object]:
    return dict(dict(dict(summary.get(setting, {})).get(family, {})).get(metric, {}))


def read_csv_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item: Dict[str, object] = dict(row)
            try:
                item["label"] = int(float(row.get("label", 0) or 0))
                item["score"] = float(row.get("score", 0.0) or 0.0)
            except ValueError:
                continue
            rows.append(item)
    return rows


def normalize_reaction_class(row: Dict[str, object]) -> str:
    reaction_class = str(row.get("reaction_class", "") or "").strip()
    if reaction_class:
        return reaction_class
    dataset = str(row.get("dataset", "") or "").strip()
    if dataset == "regiosqm20":
        return "RegioSQM20"
    return "unknown"


def summarize_grouped_metric(records: Sequence[Dict[str, object]], field: str, metric: str) -> str:
    return pct(summarize([float(dict(record.get(field, {})).get(metric, 0.0)) for record in records]))


def metric_at(payload: Dict[str, object], model: str, split: str, metric: str) -> float:
    score_metrics = dict(payload.get("score_metrics", {}))
    model_metrics = dict(score_metrics.get(model, {}))
    if split == "overall":
        return float(dict(model_metrics.get("overall", {})).get(metric, 0.0))
    return float(dict(dict(model_metrics.get("by_split", {})).get(split, {})).get(metric, 0.0))


def ranking_at(payload: Dict[str, object], split: str, metric: str) -> float:
    if split == "overall":
        return float(dict(payload.get("overall", {})).get(metric, 0.0))
    return float(dict(dict(payload.get("by_split", {})).get(split, {})).get(metric, 0.0))


def build_type1_main(root: Path) -> List[Dict[str, str]]:
    ablation = load_json(root / "results/type1_diverse_anchor_ablation_full/paper_summary/benchmark_with_ci.json")
    dpo = load_json(root / "results/type1_diverse_anchor_dpo_reference/paper_summary/benchmark_with_ci.json")
    rows: List[Dict[str, str]] = []
    ablation_summary = dict(ablation.get("summary", {}))
    dpo_summary = dict(dpo.get("summary", {}))

    rows.append(
        {
            "scope": "type1_diverse_anchor",
            "model": "pairwise_default",
            "overall_top1": pct(setting_metric(ablation_summary, "pairwise_default", "ranking", "top1")),
            "synthetic_top1": pct(setting_metric(ablation_summary, "pairwise_default", "synthetic_ranking", "top1")),
            "regio_top1": pct(setting_metric(ablation_summary, "pairwise_default", "regio_challenge", "top1")),
            "heteroatom_top1": pct(setting_metric(ablation_summary, "pairwise_default", "heteroatom_challenge", "top1")),
            "test_top1": pct(setting_metric(ablation_summary, "pairwise_default", "test_ranking", "top1")),
            "interpretation": "main type-1 boundary model",
        }
    )
    rows.append(
        {
            "scope": "type1_diverse_anchor",
            "model": "family_margin",
            "overall_top1": pct(setting_metric(ablation_summary, "family_margin", "ranking", "top1")),
            "synthetic_top1": pct(setting_metric(ablation_summary, "family_margin", "synthetic_ranking", "top1")),
            "regio_top1": pct(setting_metric(ablation_summary, "family_margin", "regio_challenge", "top1")),
            "heteroatom_top1": pct(setting_metric(ablation_summary, "family_margin", "heteroatom_challenge", "top1")),
            "test_top1": pct(setting_metric(ablation_summary, "family_margin", "test_ranking", "top1")),
            "interpretation": "no clear gain over default pairwise",
        }
    )
    rows.append(
        {
            "scope": "type1_diverse_anchor",
            "model": "frozen_chemformer_ll",
            "overall_top1": pct(setting_metric(dpo_summary, "pairwise_only_synth", "chemformer_reference", "top1")),
            "synthetic_top1": "28.33 +/- 0.00",
            "regio_top1": "",
            "heteroatom_top1": "",
            "test_top1": "",
            "interpretation": "external pretrained LM baseline",
        }
    )
    rows.append(
        {
            "scope": "type1_diverse_anchor",
            "model": "chemformer_ref_pairwise_only",
            "overall_top1": pct(setting_metric(dpo_summary, "pairwise_only_synth", "reward_model", "top1")),
            "synthetic_top1": pct(setting_metric(dpo_summary, "pairwise_only_synth", "synthetic_ranking", "top1")),
            "regio_top1": "",
            "heteroatom_top1": "",
            "test_top1": pct(setting_metric(dpo_summary, "pairwise_only_synth", "test_ranking", "top1")),
            "interpretation": "reference-policy supplement; below pure pairwise",
        }
    )
    return rows


def build_type2_table(root: Path) -> List[Dict[str, str]]:
    payload = load_json(root / "results/type2_low_yield_branch_full/summary.json")
    summary = dict(payload.get("summary", {}))
    rows: List[Dict[str, str]] = []
    for setting in sorted(summary):
        raw = dict(summary[setting])
        rows.append(
            {
                "setting": setting,
                "n": str(raw.get("n", 0)),
                "test_roc_auc": pct(dict(dict(raw.get("test", {})).get("roc_auc", {}))),
                "test_auprc": pct(dict(dict(raw.get("test", {})).get("auprc", {}))),
                "test_f1": pct(dict(dict(raw.get("test", {})).get("f1", {}))),
                "hitea_roc_auc": pct(dict(dict(raw.get("test_hitea", {})).get("roc_auc", {}))),
                "regiosqm20_roc_auc": pct(dict(dict(raw.get("test_regiosqm20", {})).get("roc_auc", {}))),
            }
        )
    return rows


def build_type1_dataset_table(root: Path) -> List[Dict[str, str]]:
    run_root = root / "results/type1_diverse_anchor_ablation_full"
    records: Dict[str, List[Dict[str, object]]] = {}
    for path in sorted(run_root.glob("pairwise_default_seed*/rerank_same_split/ranking_metrics.json")):
        payload = load_json(path)
        for dataset, metrics in dict(payload.get("by_dataset", {})).items():
            records.setdefault(str(dataset), []).append(dict(metrics))

    rows: List[Dict[str, str]] = []
    for dataset, values in sorted(records.items()):
        rows.append(
            {
                "dataset": dataset,
                "n_seeds": str(len(values)),
                "groups": f"{mean([float(record.get('groups', 0.0)) for record in values]):.1f}",
                "candidate_rows": f"{mean([float(record.get('candidate_rows', 0.0)) for record in values]):.1f}",
                "top1": pct(summarize([float(record.get("top1", 0.0)) for record in values])),
                "mrr": pct(summarize([float(record.get("mrr", 0.0)) for record in values])),
                "ndcg": pct(summarize([float(record.get("ndcg", 0.0)) for record in values])),
            }
        )
    return rows


def build_type1_reaction_class_table(root: Path, min_groups: int = 5) -> List[Dict[str, str]]:
    run_root = root / "results/type1_diverse_anchor_ablation_full"
    records: Dict[str, List[Dict[str, object]]] = {}
    group_counts: Dict[str, List[int]] = {}
    row_counts: Dict[str, List[int]] = {}
    for path in sorted(run_root.glob("pairwise_default_seed*/rerank_same_split/candidate_scores.csv")):
        rows = read_csv_rows(path)
        classes = sorted({normalize_reaction_class(row) for row in rows})
        for reaction_class in classes:
            subset = [row for row in rows if normalize_reaction_class(row) == reaction_class]
            metrics = ranking_metrics(subset)
            if int(metrics.get("groups", 0)) < min_groups:
                continue
            records.setdefault(reaction_class, []).append(metrics)
            group_counts.setdefault(reaction_class, []).append(int(metrics.get("groups", 0)))
            row_counts.setdefault(reaction_class, []).append(int(metrics.get("candidate_rows", 0)))

    rows_out: List[Dict[str, str]] = []
    for reaction_class, values in sorted(records.items()):
        rows_out.append(
            {
                "reaction_class": reaction_class,
                "n_seeds": str(len(values)),
                "groups": f"{mean(group_counts[reaction_class]):.1f}",
                "candidate_rows": f"{mean(row_counts[reaction_class]):.1f}",
                "top1": pct(summarize([float(record.get("top1", 0.0)) for record in values])),
                "mrr": pct(summarize([float(record.get("mrr", 0.0)) for record in values])),
                "ndcg": pct(summarize([float(record.get("ndcg", 0.0)) for record in values])),
            }
        )
    return rows_out


def build_type2_reaction_class_table(root: Path, setting: str = "low_yield_synth05") -> List[Dict[str, str]]:
    run_root = root / "results/type2_low_yield_branch_full"
    records: Dict[str, List[Dict[str, object]]] = {}
    for path in sorted(run_root.glob(f"{setting}_seed*/metrics.json")):
        payload = load_json(path)
        for reaction_class, metrics in dict(payload.get("test_by_reaction_class", {})).items():
            records.setdefault(str(reaction_class), []).append(dict(metrics))

    rows: List[Dict[str, str]] = []
    for reaction_class, values in sorted(records.items()):
        rows.append(
            {
                "setting": setting,
                "reaction_class": reaction_class,
                "n_seeds": str(len(values)),
                "n_rows": f"{mean([float(record.get('n', 0.0)) for record in values]):.1f}",
                "roc_auc": pct(summarize([float(record.get("roc_auc", 0.0)) for record in values])),
                "auprc": pct(summarize([float(record.get("auprc", 0.0)) for record in values])),
                "f1": pct(summarize([float(record.get("f1", 0.0)) for record in values])),
            }
        )
    return rows


def build_action_family_table(root: Path) -> List[Dict[str, str]]:
    payload = load_json(root / "results/type1_diverse_anchor_full/action_family_contribution/action_family_contribution.json")
    generation = dict(payload.get("generation", {}))
    rows: List[Dict[str, str]] = []
    for family in ["regio", "heteroatom", "tautomer", "low_yield_seed"]:
        raw = dict(generation.get(family, {}))
        rows.append(
            {
                "family": family,
                "total": str(raw.get("total", 0)),
                "keep": str(raw.get("keep_synthetic_negative", 0)),
                "keep_train": str(dict(raw.get("kept_split_counts", {})).get("train", 0)),
                "needs_review": str(raw.get("needs_review_or_downweight", 0)),
                "discard_known_positive": str(raw.get("discard_known_positive", 0)),
                "role": {
                    "regio": "type-1 reranking",
                    "heteroatom": "type-1 reranking",
                    "tautomer": "legacy type-1 reranking",
                    "low_yield_seed": "type-2 feasibility",
                }.get(family, ""),
            }
        )
    return rows


def build_external_product_bridge_table(root: Path) -> List[Dict[str, str]]:
    strict = load_json(root / "results/external_product_prediction_benchmark_20260711/benchmark/benchmark_summary.json")
    morgan = load_json(root / "results/external_product_prediction_benchmark_20260711/benchmark_validity_aware/benchmark_summary.json")
    graph = load_json(
        root
        / "results/type1_graph_stats_pairwise_full/graph_stats_ensemble5/external_validity_aware/benchmark_summary.json"
    )
    rows: List[Dict[str, str]] = []
    configs = [
        (
            "strict_intersection",
            "Chemformer likelihood",
            strict,
            "chemformer_likelihood",
            "shared scored-candidate intersection",
        ),
        (
            "strict_intersection",
            "PC-CNG Morgan pairwise",
            strict,
            "pc_cng",
            "pure learned scorer on shared intersection",
        ),
        (
            "validity_aware",
            "Chemformer likelihood",
            morgan,
            "chemformer_likelihood",
            "full beam candidate set",
        ),
        (
            "validity_aware",
            "PC-CNG Morgan 5-seed",
            morgan,
            "pc_cng",
            "main external product-selection branch",
        ),
        (
            "validity_aware",
            "PC-CNG graph-stats 5-seed",
            graph,
            "pc_cng",
            "architecture supplement",
        ),
    ]
    for scope, model, payload, metric_key, note in configs:
        if not payload:
            continue
        overall = dict(dict(payload.get("score_metrics", {})).get(metric_key, {})).get("overall", {})
        rows.append(
            {
                "scope": scope,
                "model": model,
                "groups": str(int(dict(overall).get("groups", 0))),
                "candidate_rows": str(int(payload.get("candidate_rows_evaluated", 0))),
                "overall_top1": pct_value(metric_at(payload, metric_key, "overall", "top1")),
                "overall_mrr": pct_value(metric_at(payload, metric_key, "overall", "mrr")),
                "test_top1": pct_value(metric_at(payload, metric_key, "test", "top1")),
                "test_mrr": pct_value(metric_at(payload, metric_key, "test", "mrr")),
                "note": note,
            }
        )
    return rows


def build_graph_stats_architecture_table(root: Path) -> List[Dict[str, str]]:
    morgan_same = load_json(
        root
        / "results/type1_diverse_anchor_ablation_full/pairwise_default_seed20260710/rerank_same_split/ranking_metrics.json"
    )
    graph_seed_same = load_json(
        root
        / "results/type1_graph_stats_pairwise_full/graph_stats_seed20260710/rerank_same_split/ranking_metrics.json"
    )
    graph_ensemble_same = load_json(
        root / "results/type1_graph_stats_pairwise_full/graph_stats_ensemble5/rerank_same_split/ranking_metrics.json"
    )
    morgan_external = load_json(
        root / "results/external_product_prediction_benchmark_20260711/benchmark_validity_aware/benchmark_summary.json"
    )
    graph_external = load_json(
        root
        / "results/type1_graph_stats_pairwise_full/graph_stats_ensemble5/external_validity_aware/benchmark_summary.json"
    )
    rows: List[Dict[str, str]] = []

    def add_row(model: str, same_payload: Dict[str, object], external_payload: Dict[str, object], note: str) -> None:
        if not same_payload:
            return
        external_top1 = ""
        external_mrr = ""
        if external_payload:
            external_top1 = pct_value(metric_at(external_payload, "pc_cng", "test", "top1"))
            external_mrr = pct_value(metric_at(external_payload, "pc_cng", "test", "mrr"))
        rows.append(
            {
                "model": model,
                "same_context_overall_top1": pct_value(ranking_at(same_payload, "overall", "top1")),
                "same_context_test_top1": pct_value(ranking_at(same_payload, "test", "top1")),
                "same_context_synthetic_top1": pct_value(
                    dict(dict(same_payload.get("by_candidate_source", {})).get("synthetic", {})).get("top1", 0.0)
                ),
                "external_validity_test_top1": external_top1,
                "external_validity_test_mrr": external_mrr,
                "note": note,
            }
        )

    add_row("Morgan seed20260710", morgan_same, morgan_external, "baseline seed; Morgan 5-seed remains external main")
    add_row("Graph-stats seed20260710", graph_seed_same, graph_external, "seed-matched graph-aware scorer")
    add_row("Graph-stats 5-seed ensemble", graph_ensemble_same, graph_external, "architecture supplement")
    return rows


def build_combined_feature_multiseed_table(root: Path) -> List[Dict[str, str]]:
    v2_summary = load_json(
        root / "results/type1_unreacted_substrate_supplement_v2_20260711/unreacted_v2_multiseed_summary/summary.json"
    )
    combined_summary = load_json(
        root / "results/type1_combined_feature_v2_20260712/combined_feature_v2_multiseed_summary/summary.json"
    )
    rows: List[Dict[str, str]] = []

    def get_stats(payload: Dict[str, object], scope: str, metric: str) -> Dict[str, float]:
        summary_list = list(payload.get("summary", []))
        for row in summary_list:
            if dict(row).get("scope") == scope:
                return {
                    "mean": float(dict(row).get(f"{metric}_mean", 0.0)),
                    "std": float(dict(row).get(f"{metric}_std", 0.0)),
                    "n_seeds": int(dict(row).get("n_seeds", 0)),
                }
        return {"mean": 0.0, "std": 0.0, "n_seeds": 0}

    for scope in ["original_regio_hitea", "expanded_curated"]:
        for metric in ["overall_top1", "test_top1", "val_top1", "overall_mrr", "overall_ndcg"]:
            v2 = get_stats(v2_summary, scope, metric)
            comb = get_stats(combined_summary, scope, metric)
            delta = comb["mean"] - v2["mean"]
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "v2_n_seeds": str(v2["n_seeds"]),
                    "combined_n_seeds": str(comb["n_seeds"]),
                    "v2_mean": pct_value(v2["mean"]),
                    "v2_std": pct_value(v2["std"]),
                    "combined_mean": pct_value(comb["mean"]),
                    "combined_std": pct_value(comb["std"]),
                    "delta_pp": f"{delta * 100.0:+.2f}",
                }
            )
    return rows


def build_combined_feature_paired_significance_table(root: Path) -> List[Dict[str, str]]:
    specs = [
        ("original_regio_hitea", root / "results/type1_combined_feature_v2_20260712/paired_significance_v2_vs_combined_original/summary.json"),
        ("expanded_curated", root / "results/type1_combined_feature_v2_20260712/paired_significance_v2_vs_combined_expanded/summary.json"),
    ]
    rows: List[Dict[str, str]] = []
    for scope, path in specs:
        payload = load_json(path)
        if not payload:
            continue
        ensemble = dict(payload.get("ensemble_summary", {}))
        for metric in ["top1", "mrr", "ndcg"]:
            stats = dict(ensemble.get(metric, {}))
            if not stats:
                continue
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "groups": str(int(stats.get("groups", 0) or 0)),
                    "v2_mean": pct_value(stats.get("baseline_mean", 0.0)),
                    "combined_mean": pct_value(stats.get("candidate_mean", 0.0)),
                    "delta": pct_value(stats.get("delta_mean", 0.0)),
                    "delta_ci95": f"[{pct_value(stats.get('delta_ci95_low', 0.0))}, {pct_value(stats.get('delta_ci95_high', 0.0))}]",
                    "paired_permutation_p": f"{float(stats.get('paired_permutation_p', 1.0) or 1.0):.4g}",
                    "sign_test_p": f"{float(stats.get('sign_test_p', 1.0) or 1.0):.4g}",
                    "combined_better_groups": str(int(stats.get("candidate_better_groups", 0) or 0)),
                    "v2_better_groups": str(int(stats.get("baseline_better_groups", 0) or 0)),
                    "tie_groups": str(int(stats.get("tie_groups", 0) or 0)),
                }
            )
    return rows


def build_reaction_class_gate_table(root: Path) -> List[Dict[str, str]]:
    baseline = load_json(root / "results/reaction_class_benchmark_20260711/reaction_class_benchmark.json")
    fallback = load_json(
        root / "results/type1_class_fallback_supplement_20260711/reaction_class_fallback_trained/reaction_class_benchmark.json"
    )
    partial = load_json(
        root / "results/type1_partial_product_supplement_20260711/reaction_class_partial_product_trained/reaction_class_benchmark.json"
    )
    unreacted = load_json(
        root / "results/type1_unreacted_substrate_supplement_v2_20260711/reaction_class_unreacted_trained/reaction_class_benchmark.json"
    )
    rows: List[Dict[str, str]] = []

    def class_map(payload: Dict[str, object], model: str) -> Dict[str, Dict[str, object]]:
        summaries = dict(payload.get("summaries", {}))
        model_payload = dict(summaries.get(model, {}))
        return {
            str(row.get("reaction_class", "")): dict(row)
            for row in list(model_payload.get("class_summary", []))
        }

    base_map = class_map(baseline, "morgan_seed20260710_same")
    fallback_map = class_map(fallback, "fallback_trained")
    partial_map = class_map(partial, "partial_product_trained")
    unreacted_map = class_map(unreacted, "unreacted_trained")
    classes = [
        "Alkylation",
        "Amide coupling",
        "Cabonylation",
        "Cu coupling",
        "Hydrogenation",
        "Ni coupling",
        "Rh coupling",
    ]
    for reaction_class in classes:
        base = base_map.get(reaction_class, {})
        after = fallback_map.get(reaction_class, {})
        partial_after = partial_map.get(reaction_class, {})
        unreacted_after = unreacted_map.get(reaction_class, {})
        rows.append(
            {
                "reaction_class": reaction_class,
                "base_groups": str(int(base.get("groups", 0) or 0)),
                "fallback_groups": str(int(after.get("groups", 0) or 0)),
                "partial_groups": str(int(partial_after.get("groups", 0) or 0)),
                "unreacted_groups": str(int(unreacted_after.get("groups", 0) or 0)),
                "base_top1": pct_value(base.get("top1", 0.0)),
                "fallback_top1": pct_value(after.get("top1", 0.0)),
                "fallback_mrr": pct_value(after.get("mrr", 0.0)),
                "partial_top1": pct_value(partial_after.get("top1", 0.0)),
                "partial_mrr": pct_value(partial_after.get("mrr", 0.0)),
                "unreacted_top1": pct_value(unreacted_after.get("top1", 0.0)),
                "unreacted_top1_tie_aware": pct_value(unreacted_after.get("top1_tie_aware", 0.0)),
                "unreacted_mrr": pct_value(unreacted_after.get("mrr", 0.0)),
                "unreacted_tie_only_errors": str(int(unreacted_after.get("tie_only_error_groups", 0) or 0)),
                "fallback_status": str(after.get("status", "missing")),
                "partial_status": str(partial_after.get("status", "missing")),
                "unreacted_status": str(unreacted_after.get("status", "missing")),
                "unreacted_tie_aware_status": str(unreacted_after.get("tie_aware_status", "missing")),
                "recommendation": str(
                    unreacted_after.get("recommendation")
                    or partial_after.get("recommendation")
                    or after.get("recommendation", "")
                ),
            }
        )
    return rows


def build_curated_weak_class_context_table(root: Path) -> List[Dict[str, str]]:
    v2_expanded = load_json(
        root / "results/type1_curated_weak_class_contexts_20260711/reaction_class_v2_model_expanded_curated_scope/reaction_class_benchmark.json"
    )
    curated_augmented = load_json(
        root / "results/type1_curated_weak_class_contexts_20260711/reaction_class_curated_augmented/reaction_class_benchmark.json"
    )
    classw050 = load_json(
        root / "results/type1_curated_weak_class_contexts_20260711/reaction_class_curated_classw050_rc_expanded/reaction_class_benchmark.json"
    )
    support = load_json(
        root / "results/type1_curated_weak_class_contexts_20260711/source_support_after_curated_fallback/source_support_audit.json"
    )

    def class_map(payload: Dict[str, object], model: str) -> Dict[str, Dict[str, object]]:
        summaries = dict(payload.get("summaries", {}))
        model_payload = dict(summaries.get(model, {}))
        return {
            str(row.get("reaction_class", "")): dict(row)
            for row in list(model_payload.get("class_summary", []))
        }

    def support_map(payload: Dict[str, object]) -> Dict[str, Dict[str, object]]:
        return {
            str(row.get("reaction_class", "")): dict(row)
            for row in list(payload.get("class_summary", []))
        }

    v2_map = class_map(v2_expanded, "v2_model_expanded_curated_scope")
    selected_model = "curated_classw050_rc_expanded" if classw050 else "curated_augmented"
    curated_map = class_map(classw050, selected_model) if classw050 else class_map(curated_augmented, selected_model)
    support_rows = support_map(support)
    rows: List[Dict[str, str]] = []
    for reaction_class in ["Amide coupling", "Cu coupling", "Hydrogenation", "Rh coupling", "Ni coupling"]:
        before = v2_map.get(reaction_class, {})
        after = curated_map.get(reaction_class, {})
        support_row = support_rows.get(reaction_class, {})
        before_top1 = float(before.get("top1", 0.0) or 0.0)
        after_top1 = float(after.get("top1", 0.0) or 0.0)
        rows.append(
            {
                "reaction_class": reaction_class,
                "support_status": str(support_row.get("status", "")),
                "positive_parent_reactions": str(int(support_row.get("positive_parent_reactions", 0) or 0)),
                "candidate_parent_reactions": str(int(support_row.get("candidate_parent_reactions", 0) or 0)),
                "v2_expanded_groups": str(int(before.get("groups", 0) or 0)),
                "v2_expanded_top1": pct_value(before.get("top1", 0.0)),
                "v2_expanded_mrr": pct_value(before.get("mrr", 0.0)),
                "selected_model": selected_model,
                "curated_augmented_groups": str(int(after.get("groups", 0) or 0)),
                "curated_augmented_top1": pct_value(after.get("top1", 0.0)),
                "curated_augmented_mrr": pct_value(after.get("mrr", 0.0)),
                "top1_delta_pp": f"{(after_top1 - before_top1) * 100.0:.2f}",
                "curated_status": str(after.get("status", "")),
                "recommendation": str(support_row.get("recommendation", "")),
            }
        )
    return rows


def build_curated_model_selection_table(root: Path) -> List[Dict[str, str]]:
    candidates = [
        (
            "v2_unreacted_seed20260710",
            "original_regio_hitea",
            root
            / "results/type1_unreacted_substrate_supplement_v2_20260711/unreacted_augmented_pairwise_seed20260710/rerank_same_split/ranking_metrics.json",
            "current v2 main scope",
        ),
        (
            "v2_unreacted_seed20260710",
            "expanded_curated",
            root / "results/type1_curated_weak_class_contexts_20260711/v2_model_expanded_curated_scope/ranking_metrics.json",
            "zero-shot on curated Amide/Cu contexts",
        ),
        (
            "curated_unweighted_seed20260711",
            "original_regio_hitea",
            root
            / "results/type1_curated_weak_class_contexts_20260711/curated_augmented_pairwise_seed20260711/rerank_original_scope/ranking_metrics.json",
            "unweighted curated branch original-scope sanity check",
        ),
        (
            "curated_unweighted_seed20260711",
            "expanded_curated",
            root
            / "results/type1_curated_weak_class_contexts_20260711/curated_augmented_pairwise_seed20260711/rerank_same_split/ranking_metrics.json",
            "unweighted curated branch",
        ),
        (
            "curated_classw050_seed20260711",
            "original_regio_hitea",
            root
            / "results/type1_curated_weak_class_contexts_20260711/curated_augmented_pairwise_classw050_rc_seed20260711/rerank_original_scope/ranking_metrics.json",
            "selected single-seed candidate; Amide/Cu pairwise weight 0.5",
        ),
        (
            "curated_classw050_seed20260711",
            "expanded_curated",
            root
            / "results/type1_curated_weak_class_contexts_20260711/curated_augmented_pairwise_classw050_rc_seed20260711/rerank_expanded_scope/ranking_metrics.json",
            "selected single-seed candidate; Amide/Cu pairwise weight 0.5",
        ),
    ]
    rows: List[Dict[str, str]] = []
    for model, scope, path, note in candidates:
        payload = load_json(path)
        if not payload:
            continue
        by_dataset = dict(payload.get("by_dataset", {}))
        rows.append(
            {
                "model": model,
                "scope": scope,
                "groups": str(int(dict(payload.get("overall", {})).get("groups", 0) or 0)),
                "overall_top1": pct_value(ranking_at(payload, "overall", "top1")),
                "test_top1": pct_value(ranking_at(payload, "test", "top1")),
                "val_top1": pct_value(ranking_at(payload, "val", "top1")),
                "regiosqm20_top1": pct_value(dict(by_dataset.get("regiosqm20", {})).get("top1", 0.0)),
                "hitea_top1": pct_value(dict(by_dataset.get("hitea_full", {})).get("top1", 0.0)),
                "curated_uspto_top1": pct_value(
                    dict(by_dataset.get("curated_uspto_openmolecules_rule", {})).get("top1", 0.0)
                ),
                "synthetic_top1": pct_value(dict(dict(payload.get("by_candidate_source", {})).get("synthetic", {})).get("top1", 0.0)),
                "note": note,
            }
        )
    return rows


def build_curated_multiseed_stability_table(root: Path) -> List[Dict[str, str]]:
    payload = load_json(
        root / "results/type1_curated_weak_class_contexts_20260711/classw050_rc_multiseed_summary/summary.json"
    )
    rows: List[Dict[str, str]] = []
    for row in list(payload.get("summary", [])):
        item = dict(row)
        rows.append(
            {
                "scope": str(item.get("scope", "")),
                "n_seeds": str(int(item.get("n_seeds", 0) or 0)),
                "overall_top1": pct_mean_std(item.get("overall_top1_mean", 0.0), item.get("overall_top1_std", 0.0)),
                "test_top1": pct_mean_std(item.get("test_top1_mean", 0.0), item.get("test_top1_std", 0.0)),
                "val_top1": pct_mean_std(item.get("val_top1_mean", 0.0), item.get("val_top1_std", 0.0)),
                "overall_mrr": pct_mean_std(item.get("overall_mrr_mean", 0.0), item.get("overall_mrr_std", 0.0)),
                "overall_ndcg": pct_mean_std(item.get("overall_ndcg_mean", 0.0), item.get("overall_ndcg_std", 0.0)),
                "regiosqm20_top1": pct_mean_std(item.get("regiosqm20_top1_mean", 0.0), item.get("regiosqm20_top1_std", 0.0)),
                "hitea_top1": pct_mean_std(item.get("hitea_top1_mean", 0.0), item.get("hitea_top1_std", 0.0)),
                "curated_uspto_top1": pct_mean_std(
                    item.get("curated_uspto_top1_mean", 0.0),
                    item.get("curated_uspto_top1_std", 0.0),
                ),
            }
        )
    return rows


def build_curated_vs_v2_multiseed_table(root: Path) -> List[Dict[str, str]]:
    curated = load_json(
        root / "results/type1_curated_weak_class_contexts_20260711/classw050_rc_multiseed_summary/summary.json"
    )
    v2 = load_json(
        root / "results/type1_unreacted_substrate_supplement_v2_20260711/unreacted_v2_multiseed_summary/summary.json"
    )

    def by_scope(payload: Dict[str, object]) -> Dict[str, Dict[str, object]]:
        return {str(row.get("scope", "")): dict(row) for row in list(payload.get("summary", []))}

    curated_by_scope = by_scope(curated)
    v2_by_scope = by_scope(v2)
    rows: List[Dict[str, str]] = []
    for scope in ["original_regio_hitea", "expanded_curated"]:
        cur = curated_by_scope.get(scope, {})
        base = v2_by_scope.get(scope, {})
        if not cur or not base:
            continue
        for metric in ["overall_top1", "test_top1", "regiosqm20_top1", "hitea_top1", "curated_uspto_top1"]:
            cur_mean = float(cur.get(f"{metric}_mean", 0.0) or 0.0)
            base_mean = float(base.get(f"{metric}_mean", 0.0) or 0.0)
            if cur_mean != cur_mean and base_mean != base_mean:
                continue
            delta = cur_mean - base_mean if cur_mean == cur_mean and base_mean == base_mean else float("nan")
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "v2_n_seeds": str(int(base.get("n_seeds", 0) or 0)),
                    "classw050_n_seeds": str(int(cur.get("n_seeds", 0) or 0)),
                    "v2": pct_mean_std(base.get(f"{metric}_mean", 0.0), base.get(f"{metric}_std", 0.0)),
                    "classw050": pct_mean_std(cur.get(f"{metric}_mean", 0.0), cur.get(f"{metric}_std", 0.0)),
                    "delta_pp": "n/a" if delta != delta else f"{delta * 100.0:.2f}",
                }
            )
    return rows


def build_curated_paired_significance_table(root: Path) -> List[Dict[str, str]]:
    specs = [
        (
            "original_regio_hitea",
            root
            / "results/type1_curated_weak_class_contexts_20260711/paired_significance_original_v2_vs_classw050_seed20260711/summary.json",
        ),
        (
            "expanded_curated",
            root
            / "results/type1_curated_weak_class_contexts_20260711/paired_significance_expanded_v2_vs_classw050_seed20260711/summary.json",
        ),
    ]
    rows: List[Dict[str, str]] = []
    for scope, path in specs:
        payload = load_json(path)
        if not payload:
            continue
        summary = dict(payload.get("summary", {}))
        for metric in ["top1", "mrr", "ndcg"]:
            stats = dict(summary.get(metric, {}))
            if not stats:
                continue
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "groups": str(int(stats.get("groups", 0) or 0)),
                    "baseline_mean": pct_value(stats.get("baseline_mean", 0.0)),
                    "candidate_mean": pct_value(stats.get("candidate_mean", 0.0)),
                    "delta": pct_value(stats.get("delta_mean", 0.0)),
                    "delta_ci95": f"[{pct_value(stats.get('delta_ci95_low', 0.0))}, {pct_value(stats.get('delta_ci95_high', 0.0))}]",
                    "paired_permutation_p": f"{float(stats.get('paired_permutation_p', 1.0) or 1.0):.4g}",
                    "sign_test_p": f"{float(stats.get('sign_test_p', 1.0) or 1.0):.4g}",
                    "candidate_better_groups": str(int(stats.get("candidate_better_groups", 0) or 0)),
                    "baseline_better_groups": str(int(stats.get("baseline_better_groups", 0) or 0)),
                    "tie_groups": str(int(stats.get("tie_groups", 0) or 0)),
                }
            )
    return rows


def build_curated_paired_significance_10seed_table(root: Path) -> List[Dict[str, str]]:
    specs = [
        (
            "original_regio_hitea",
            "group_ensemble",
            root
            / "results/type1_curated_weak_class_contexts_20260711/paired_significance_10seed/original_regio_hitea/summary.json",
        ),
        (
            "expanded_curated",
            "group_ensemble",
            root
            / "results/type1_curated_weak_class_contexts_20260711/paired_significance_10seed/expanded_curated/summary.json",
        ),
        (
            "original_regio_hitea",
            "seed_bootstrap",
            root
            / "results/type1_curated_weak_class_contexts_20260711/paired_significance_10seed/original_regio_hitea/summary.json",
        ),
        (
            "expanded_curated",
            "seed_bootstrap",
            root
            / "results/type1_curated_weak_class_contexts_20260711/paired_significance_10seed/expanded_curated/summary.json",
        ),
    ]
    rows: List[Dict[str, str]] = []
    for scope, level, path in specs:
        payload = load_json(path)
        if not payload:
            continue
        if level == "group_ensemble":
            summary = dict(payload.get("ensemble_summary", {}))
            for metric in ["top1", "mrr", "ndcg"]:
                stats = dict(summary.get(metric, {}))
                if not stats:
                    continue
                rows.append(
                    {
                        "scope": scope,
                        "level": level,
                        "metric": metric,
                        "groups": str(int(stats.get("groups", 0) or 0)),
                        "baseline_mean": pct_value(stats.get("baseline_mean", 0.0)),
                        "candidate_mean": pct_value(stats.get("candidate_mean", 0.0)),
                        "delta": pct_value(stats.get("delta_mean", 0.0)),
                        "delta_ci95": f"[{pct_value(stats.get('delta_ci95_low', 0.0))}, {pct_value(stats.get('delta_ci95_high', 0.0))}]",
                        "paired_permutation_p": f"{float(stats.get('paired_permutation_p', 1.0) or 1.0):.4g}",
                        "sign_test_p": f"{float(stats.get('sign_test_p', 1.0) or 1.0):.4g}",
                        "candidate_better_groups": str(int(stats.get("candidate_better_groups", 0) or 0)),
                        "baseline_better_groups": str(int(stats.get("baseline_better_groups", 0) or 0)),
                        "tie_groups": str(int(stats.get("tie_groups", 0) or 0)),
                    }
                )
        elif level == "seed_bootstrap":
            seed_bootstrap = dict(payload.get("seed_bootstrap", {}))
            for metric in ["top1", "mrr", "ndcg"]:
                stats = dict(seed_bootstrap.get(metric, {}))
                if not stats:
                    continue
                rows.append(
                    {
                        "scope": scope,
                        "level": level,
                        "metric": metric,
                        "groups": "",
                        "baseline_mean": "",
                        "candidate_mean": "",
                        "delta": pct_value(stats.get("mean", 0.0)),
                        "delta_ci95": f"[{pct_value(stats.get('ci95_low', 0.0))}, {pct_value(stats.get('ci95_high', 0.0))}]",
                        "paired_permutation_p": "",
                        "sign_test_p": "",
                        "candidate_better_groups": "",
                        "baseline_better_groups": "",
                        "tie_groups": "",
                    }
                )
    return rows


def build_source_support_audit_table(root: Path) -> List[Dict[str, str]]:
    audit = load_json(root / "results/reaction_class_source_support_audit_20260711/source_support_audit.json")
    unreacted = load_json(
        root / "results/type1_unreacted_substrate_supplement_v2_20260711/source_support_after_unreacted/source_support_audit.json"
    )
    curated = load_json(
        root / "results/type1_curated_weak_class_contexts_20260711/source_support_after_curated_fallback/source_support_audit.json"
    )
    rows: List[Dict[str, str]] = []
    class_rows = {
        str(row.get("reaction_class", "")): dict(row)
        for row in list(audit.get("class_summary", []))
    }
    class_rows.update(
        {
            str(row.get("reaction_class", "")): dict(row)
            for row in list(unreacted.get("class_summary", []))
        }
    )
    class_rows.update(
        {
            str(row.get("reaction_class", "")): dict(row)
            for row in list(curated.get("class_summary", []))
        }
    )
    classes = [
        "Alkylation",
        "Amide coupling",
        "Cabonylation",
        "Cu coupling",
        "Hydrogenation",
        "Ni coupling",
        "Rh coupling",
    ]
    for reaction_class in classes:
        raw = class_rows.get(reaction_class, {})
        if not raw:
            continue
        rows.append(
            {
                "reaction_class": reaction_class,
                "status": str(raw.get("status", "")),
                "positive_sources": str(int(raw.get("positive_sources", 0) or 0)),
                "positive_parent_reactions": str(int(raw.get("positive_parent_reactions", 0) or 0)),
                "candidate_sources": str(int(raw.get("candidate_sources", 0) or 0)),
                "candidate_parent_reactions": str(int(raw.get("candidate_parent_reactions", 0) or 0)),
                "source_group_deficit": str(int(raw.get("source_group_deficit", 0) or 0)),
                "molecular_parent_deficit": str(int(raw.get("molecular_parent_deficit", 0) or 0)),
                "positive_parent_coverage": pct_value(raw.get("coverage_of_positive_parent_reactions", 0.0)),
                "recommendation": str(raw.get("recommendation", "")),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/cunyuliu/pc_cng_research")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir or root / "results/manuscript_tables_pc_cng_v3")
    tables = {
        "main_type1_reranking": (
            build_type1_main(root),
            [
                "scope",
                "model",
                "overall_top1",
                "synthetic_top1",
                "regio_top1",
                "heteroatom_top1",
                "test_top1",
                "interpretation",
            ],
        ),
        "supp_type2_low_yield": (
            build_type2_table(root),
            ["setting", "n", "test_roc_auc", "test_auprc", "test_f1", "hitea_roc_auc", "regiosqm20_roc_auc"],
        ),
        "supp_action_family_generation": (
            build_action_family_table(root),
            ["family", "total", "keep", "keep_train", "needs_review", "discard_known_positive", "role"],
        ),
        "supp_type1_dataset": (
            build_type1_dataset_table(root),
            ["dataset", "n_seeds", "groups", "candidate_rows", "top1", "mrr", "ndcg"],
        ),
        "supp_type1_reaction_class": (
            build_type1_reaction_class_table(root),
            ["reaction_class", "n_seeds", "groups", "candidate_rows", "top1", "mrr", "ndcg"],
        ),
        "supp_type2_reaction_class": (
            build_type2_reaction_class_table(root),
            ["setting", "reaction_class", "n_seeds", "n_rows", "roc_auc", "auprc", "f1"],
        ),
        "main_external_product_bridge": (
            build_external_product_bridge_table(root),
            [
                "scope",
                "model",
                "groups",
                "candidate_rows",
                "overall_top1",
                "overall_mrr",
                "test_top1",
                "test_mrr",
                "note",
            ],
        ),
        "supp_graph_stats_architecture": (
            build_graph_stats_architecture_table(root),
            [
                "model",
                "same_context_overall_top1",
                "same_context_test_top1",
                "same_context_synthetic_top1",
                "external_validity_test_top1",
                "external_validity_test_mrr",
                "note",
            ],
        ),
        "supp_reaction_class_gate": (
            build_reaction_class_gate_table(root),
            [
                "reaction_class",
                "base_groups",
                "fallback_groups",
                "partial_groups",
                "unreacted_groups",
                "base_top1",
                "fallback_top1",
                "fallback_mrr",
                "partial_top1",
                "partial_mrr",
                "unreacted_top1",
                "unreacted_top1_tie_aware",
                "unreacted_mrr",
                "unreacted_tie_only_errors",
                "fallback_status",
                "partial_status",
                "unreacted_status",
                "unreacted_tie_aware_status",
                "recommendation",
            ],
        ),
        "supp_source_support_audit": (
            build_source_support_audit_table(root),
            [
                "reaction_class",
                "status",
                "positive_sources",
                "positive_parent_reactions",
                "candidate_sources",
                "candidate_parent_reactions",
                "source_group_deficit",
                "molecular_parent_deficit",
                "positive_parent_coverage",
                "recommendation",
            ],
        ),
        "supp_curated_weak_class_contexts": (
            build_curated_weak_class_context_table(root),
            [
                "reaction_class",
                "support_status",
                "positive_parent_reactions",
                "candidate_parent_reactions",
                "v2_expanded_groups",
                "v2_expanded_top1",
                "v2_expanded_mrr",
                "selected_model",
                "curated_augmented_groups",
                "curated_augmented_top1",
                "curated_augmented_mrr",
                "top1_delta_pp",
                "curated_status",
                "recommendation",
            ],
        ),
        "supp_curated_model_selection": (
            build_curated_model_selection_table(root),
            [
                "model",
                "scope",
                "groups",
                "overall_top1",
                "test_top1",
                "val_top1",
                "regiosqm20_top1",
                "hitea_top1",
                "curated_uspto_top1",
                "synthetic_top1",
                "note",
            ],
        ),
        "supp_curated_multiseed_stability": (
            build_curated_multiseed_stability_table(root),
            [
                "scope",
                "n_seeds",
                "overall_top1",
                "test_top1",
                "val_top1",
                "overall_mrr",
                "overall_ndcg",
                "regiosqm20_top1",
                "hitea_top1",
                "curated_uspto_top1",
            ],
        ),
        "supp_curated_vs_v2_multiseed": (
            build_curated_vs_v2_multiseed_table(root),
            [
                "scope",
                "metric",
                "v2_n_seeds",
                "classw050_n_seeds",
                "v2",
                "classw050",
                "delta_pp",
            ],
        ),
        "supp_curated_paired_significance": (
            build_curated_paired_significance_table(root),
            [
                "scope",
                "metric",
                "groups",
                "baseline_mean",
                "candidate_mean",
                "delta",
                "delta_ci95",
                "paired_permutation_p",
                "sign_test_p",
                "candidate_better_groups",
                "baseline_better_groups",
                "tie_groups",
            ],
        ),
        "supp_curated_paired_significance_10seed": (
            build_curated_paired_significance_10seed_table(root),
            [
                "scope",
                "level",
                "metric",
                "groups",
                "baseline_mean",
                "candidate_mean",
                "delta",
                "delta_ci95",
                "paired_permutation_p",
                "sign_test_p",
                "candidate_better_groups",
                "baseline_better_groups",
                "tie_groups",
            ],
        ),
        "supp_combined_feature_multiseed": (
            build_combined_feature_multiseed_table(root),
            [
                "scope",
                "metric",
                "v2_n_seeds",
                "combined_n_seeds",
                "v2_mean",
                "v2_std",
                "combined_mean",
                "combined_std",
                "delta_pp",
            ],
        ),
        "supp_combined_feature_paired_significance": (
            build_combined_feature_paired_significance_table(root),
            [
                "scope",
                "metric",
                "groups",
                "v2_mean",
                "combined_mean",
                "delta",
                "delta_ci95",
                "paired_permutation_p",
                "sign_test_p",
                "combined_better_groups",
                "v2_better_groups",
                "tie_groups",
            ],
        ),
    }

    manifest = {}
    for name, (rows, fields) in tables.items():
        csv_path = output_dir / f"{name}.csv"
        md_path = output_dir / f"{name}.md"
        write_csv(csv_path, rows, fields)
        write_markdown(md_path, rows, fields)
        manifest[name] = {"csv": str(csv_path), "markdown": str(md_path), "rows": len(rows)}

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "tables": manifest}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
