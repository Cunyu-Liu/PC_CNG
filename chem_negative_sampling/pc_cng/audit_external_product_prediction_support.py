"""Audit support and coverage for the external product-prediction bridge.

The external bridge has multiple denominators: source contexts, full candidate
rows, strict complete groups with all required scores, and validity-aware rows.
This audit makes those denominators explicit before claiming scale-up progress.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


def read_json(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sniff_dialect(path: str) -> csv.Dialect:
    with open(path, encoding="utf-8") as handle:
        sample = handle.read(4096)
    first = sample.splitlines()[0] if sample.splitlines() else ""
    if "\t" in first:
        return csv.excel_tab
    return csv.excel


def nonempty(value: object) -> str:
    return str(value or "").strip()


def group_field(row: Mapping[str, str]) -> str:
    return nonempty(row.get("group_id") or row.get("source_id") or row.get("context_id") or row.get("row_index"))


def summarize_csv(path: Optional[str]) -> Dict[str, object]:
    if not path or not os.path.exists(path):
        return {"path": path, "exists": False}

    rows = 0
    groups = set()
    row_split_counts: Counter[str] = Counter()
    row_dataset_counts: Counter[str] = Counter()
    row_candidate_source_counts: Counter[str] = Counter()
    group_splits: Dict[str, str] = {}
    group_datasets: Dict[str, str] = {}
    group_candidate_sources: Dict[str, Counter[str]] = defaultdict(Counter)

    dialect = sniff_dialect(path)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows += 1
            group = group_field(row)
            split = nonempty(row.get("split")) or "unknown"
            dataset = nonempty(row.get("dataset") or row.get("source")) or "unknown"
            candidate_source = nonempty(row.get("candidate_source")) or "unknown"
            row_split_counts[split] += 1
            row_dataset_counts[dataset] += 1
            row_candidate_source_counts[candidate_source] += 1
            if group:
                groups.add(group)
                group_splits.setdefault(group, split)
                group_datasets.setdefault(group, dataset)
                group_candidate_sources[group][candidate_source] += 1

    group_split_counts = Counter(group_splits.values())
    group_dataset_counts = Counter(group_datasets.values())
    groups_with_pc_cng = sum(1 for sources in group_candidate_sources.values() if any("pc_cng" in key for key in sources))
    groups_with_external_beam = sum(
        1
        for sources in group_candidate_sources.values()
        if any("beam" in key or "chemformer" in key for key in sources)
    )
    groups_with_observed = sum(
        1
        for sources in group_candidate_sources.values()
        if any("observed_positive" in key for key in sources)
    )
    return {
        "path": path,
        "exists": True,
        "fieldnames": fieldnames,
        "rows": rows,
        "groups": len(groups),
        "row_split_counts": dict(sorted(row_split_counts.items())),
        "group_split_counts": dict(sorted(group_split_counts.items())),
        "row_dataset_counts": dict(sorted(row_dataset_counts.items())),
        "group_dataset_counts": dict(sorted(group_dataset_counts.items())),
        "row_candidate_source_counts": dict(sorted(row_candidate_source_counts.items())),
        "groups_with_observed_positive": groups_with_observed,
        "groups_with_pc_cng_candidates": groups_with_pc_cng,
        "groups_with_external_beam_candidates": groups_with_external_beam,
    }


def metric_summary(summary: Mapping[str, object]) -> Dict[str, object]:
    score_metrics = summary.get("score_metrics")
    if not isinstance(score_metrics, dict):
        return {}
    out: Dict[str, object] = {}
    for name, payload in score_metrics.items():
        if not isinstance(payload, dict):
            continue
        row: Dict[str, object] = {}
        overall = payload.get("overall")
        if isinstance(overall, dict):
            row["overall_groups"] = overall.get("groups")
            row["overall_top1"] = overall.get("top1")
            row["overall_mrr"] = overall.get("mrr")
            row["overall_ndcg"] = overall.get("ndcg")
        by_split = payload.get("by_split")
        if isinstance(by_split, dict) and isinstance(by_split.get("test"), dict):
            test = by_split["test"]
            row["test_groups"] = test.get("groups")
            row["test_top1"] = test.get("top1")
            row["test_mrr"] = test.get("mrr")
            row["test_ndcg"] = test.get("ndcg")
        out[str(name)] = row
    return out


def best_method(metrics: Mapping[str, object], split_prefix: str) -> Dict[str, object]:
    best_name = ""
    best_top1 = -1.0
    for name, payload in metrics.items():
        if not isinstance(payload, dict):
            continue
        value = payload.get(f"{split_prefix}_top1")
        try:
            top1 = float(value)
        except (TypeError, ValueError):
            continue
        if top1 > best_top1:
            best_name = name
            best_top1 = top1
    return {"method": best_name, "top1": best_top1 if best_name else None}


def summarize_evaluation(summary_path: Optional[str]) -> Dict[str, object]:
    summary = read_json(summary_path)
    if not summary:
        return {"path": summary_path, "exists": False}
    strict_filter = summary.get("strict_complete_group_filter")
    if not isinstance(strict_filter, dict):
        strict_filter = {}
    pc_cng_score = summary.get("pc_cng_score")
    if not isinstance(pc_cng_score, dict):
        pc_cng_score = {}
    metrics = metric_summary(summary)
    return {
        "path": summary_path,
        "exists": True,
        "candidate_rows_requested": summary.get("candidate_rows_requested"),
        "candidate_rows_evaluated": summary.get("candidate_rows_evaluated"),
        "strict_complete_group_filter": strict_filter,
        "pc_cng_scored_rows": pc_cng_score.get("scored_rows"),
        "pc_cng_missing": (pc_cng_score.get("attach") or {}).get("missing")
        if isinstance(pc_cng_score.get("attach"), dict)
        else None,
        "metrics": metrics,
        "best_overall_top1": best_method(metrics, "overall"),
        "best_test_top1": best_method(metrics, "test"),
        "selected_hybrid": summary.get("selected_hybrid"),
    }


def decision_flags(
    context_groups: int,
    target_groups: int,
    strict_eval: Mapping[str, object],
) -> List[str]:
    flags: List[str] = []
    if context_groups < target_groups:
        flags.append("external_context_target_not_met")
    strict_filter = strict_eval.get("strict_complete_group_filter")
    kept_groups = None
    if isinstance(strict_filter, dict):
        kept_groups = strict_filter.get("kept_groups")
    try:
        strict_groups = int(kept_groups) if kept_groups is not None else 0
    except (TypeError, ValueError):
        strict_groups = 0
    if strict_groups < target_groups:
        flags.append("strict_complete_group_target_not_met")
    if strict_groups < max(1, int(0.25 * max(context_groups, 1))):
        flags.append("strict_pc_cng_score_coverage_low")
    return flags


def write_markdown(path: str, payload: Mapping[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    contexts = payload["contexts"]
    candidates = payload["full_candidates"]
    strict_eval = payload["strict_evaluation"]
    validity_eval = payload["validity_aware_evaluation"]
    lines = [
        "# External Product Prediction Support Audit",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Target groups | `{payload['target_groups']}` |",
        f"| Context groups | `{contexts.get('groups')}` |",
        f"| Context target deficit | `{payload['context_target_deficit']}` |",
        f"| Full candidate rows / groups | `{candidates.get('rows')}` / `{candidates.get('groups')}` |",
        f"| Strict evaluated rows | `{strict_eval.get('candidate_rows_evaluated')}` |",
        f"| Strict complete groups | `{(strict_eval.get('strict_complete_group_filter') or {}).get('kept_groups')}` |",
        f"| Validity-aware evaluated rows | `{validity_eval.get('candidate_rows_evaluated')}` |",
        f"| Decision flags | `{', '.join(payload['decision_flags'])}` |",
        "",
        "## Best Top-1",
        "",
        "| Scope | Best overall | Best test |",
        "|---|---|---|",
        f"| Strict | `{strict_eval.get('best_overall_top1')}` | `{strict_eval.get('best_test_top1')}` |",
        f"| Validity-aware | `{validity_eval.get('best_overall_top1')}` | `{validity_eval.get('best_test_top1')}` |",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_audit(
    contexts_csv: str,
    full_candidate_csv: str,
    strict_summary_json: str,
    validity_summary_json: str,
    output_dir: str,
    target_groups: int,
) -> Dict[str, object]:
    contexts = summarize_csv(contexts_csv)
    full_candidates = summarize_csv(full_candidate_csv)
    strict_eval = summarize_evaluation(strict_summary_json)
    validity_eval = summarize_evaluation(validity_summary_json)
    context_groups = int(contexts.get("groups") or 0)
    payload: Dict[str, object] = {
        "target_groups": target_groups,
        "contexts": contexts,
        "full_candidates": full_candidates,
        "strict_evaluation": strict_eval,
        "validity_aware_evaluation": validity_eval,
        "context_target_deficit": max(0, target_groups - context_groups),
        "decision_flags": decision_flags(context_groups, target_groups, strict_eval),
    }
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "external_product_prediction_support_audit.json")
    md_path = os.path.join(output_dir, "external_product_prediction_support_audit.md")
    payload["outputs"] = {"json": json_path, "summary_md": md_path}
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_markdown(md_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts-csv", required=True)
    parser.add_argument("--full-candidate-csv", required=True)
    parser.add_argument("--strict-summary-json", required=True)
    parser.add_argument("--validity-summary-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-groups", type=int, default=25000)
    args = parser.parse_args()

    payload = run_audit(
        contexts_csv=args.contexts_csv,
        full_candidate_csv=args.full_candidate_csv,
        strict_summary_json=args.strict_summary_json,
        validity_summary_json=args.validity_summary_json,
        output_dir=args.output_dir,
        target_groups=args.target_groups,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
