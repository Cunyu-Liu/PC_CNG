"""Generate PC-CNG negatives directly for external product-prediction contexts.

This CPU-only bridge fills the coverage gap where newly selected external
contexts do not yet have matching synthetic PC-CNG rows.  The output schema is
compatible with ``build_external_product_prediction_candidate_set --synthetic-csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Sequence

from .chem_utils import join_reaction
from .counterfactual import CounterfactualGenerator
from .reaction_lm_scorer import canonical_smiles


OUTPUT_FIELDS = [
    "source_id",
    "group_id",
    "context_row_index",
    "positive_reaction",
    "candidate_reaction",
    "task",
    "failure_type",
    "edit_action",
    "parent_reactants",
    "parent_product",
    "candidate_reactants",
    "candidate_product",
    "valid",
    "atom_balance",
    "locality",
    "closeness",
    "hard_score",
    "false_negative_risk",
    "passes_filter",
    "label",
    "provenance",
    "review_status",
    "action_family",
]


def product_key(product: str) -> str:
    return canonical_smiles(product) or (product or "").strip()


def read_context_rows(path: str) -> Iterable[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"source_id", "reactants", "observed_product"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Context CSV missing required columns: {missing}")
        for row in reader:
            yield dict(row)


def candidate_to_output(
    candidate: object,
    context: Dict[str, str],
    row_index: int,
    review_status: str,
) -> Dict[str, object]:
    data = candidate.to_dict()  # CounterfactualCandidate keeps the stable schema.
    data["group_id"] = context.get("group_id", "")
    data["context_row_index"] = row_index
    data["review_status"] = review_status
    data["action_family"] = data.get("failure_type", "pc_cng_negative")
    return data


def write_summary_md(path: str, summary: Dict[str, object]) -> None:
    lines = [
        "# External Context PC-CNG Candidate Generation",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Input contexts | `{summary['input_contexts']}` |",
        f"| Output CSV | `{summary['output_csv']}` |",
        f"| Processed contexts | `{summary['processed_contexts']}` |",
        f"| Contexts with candidates | `{summary['contexts_with_candidates']}` |",
        f"| Generated candidates | `{summary['generated_candidates']}` |",
        f"| Coverage | `{summary['context_coverage']}` |",
        f"| Avg candidates/context | `{summary['avg_candidates_per_context']}` |",
        f"| Selected tasks | `{', '.join(summary['selected_tasks'])}` |",
        f"| Max candidates/context | `{summary['max_candidates_per_context']}` |",
        f"| Include failed | `{summary['include_failed']}` |",
        f"| Include same product | `{summary['include_same_product']}` |",
        "",
        "## Failure Families",
        "",
    ]
    by_failure = dict(summary.get("by_failure_type", {}))
    if by_failure:
        lines.extend(["| Failure type | Count |", "|---|---:|"])
        for key, value in sorted(by_failure.items()):
            lines.append(f"| `{key}` | {value} |")
    else:
        lines.append("No candidates were generated.")
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def generate_for_contexts(
    context_csv: str,
    output_csv: str,
    summary_json: str,
    summary_md: str | None,
    tasks: Sequence[str],
    max_candidates_per_context: int,
    include_failed: bool,
    include_same_product: bool,
    review_status: str,
    limit: int | None,
    start_index: int,
) -> Dict[str, object]:
    generator = CounterfactualGenerator()
    selected_tasks = set(tasks)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(summary_json), exist_ok=True)
    if summary_md:
        os.makedirs(os.path.dirname(summary_md), exist_ok=True)

    processed = 0
    skipped = 0
    generated = 0
    contexts_with_candidates = 0
    by_task: Dict[str, int] = {}
    by_failure: Dict[str, int] = {}

    with open(output_csv, "w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row_index, context in enumerate(read_context_rows(context_csv)):
            if row_index < start_index:
                continue
            if limit is not None and processed >= limit:
                break

            reactants = (context.get("reactants") or "").strip()
            agents = (context.get("agents") or "").strip()
            observed_product = (context.get("observed_product") or "").strip()
            source_id = (context.get("source_id") or context.get("group_id") or f"context_{row_index:09d}").strip()
            if not reactants or not observed_product:
                skipped += 1
                continue

            processed += 1
            positive = join_reaction(reactants, observed_product, agents)
            observed_key = product_key(observed_product)
            seen_products = {observed_key}
            kept_for_context = 0
            try:
                candidates = generator.generate_for_reaction(
                    positive,
                    source_id=source_id,
                    include_failed=include_failed,
                )
            except Exception:
                skipped += 1
                continue

            for candidate in candidates:
                if candidate.task not in selected_tasks:
                    continue
                candidate_key = product_key(candidate.candidate_product)
                if not include_same_product and candidate_key == observed_key:
                    continue
                if candidate_key in seen_products:
                    continue
                seen_products.add(candidate_key)

                writer.writerow(candidate_to_output(candidate, context, row_index, review_status))
                generated += 1
                kept_for_context += 1
                by_task[candidate.task] = by_task.get(candidate.task, 0) + 1
                by_failure[candidate.failure_type] = by_failure.get(candidate.failure_type, 0) + 1
                if kept_for_context >= max_candidates_per_context:
                    break
            if kept_for_context:
                contexts_with_candidates += 1

    summary = {
        "task": "external_context_pc_cng_candidate_generation",
        "input_contexts": context_csv,
        "output_csv": output_csv,
        "summary_json": summary_json,
        "summary_md": summary_md,
        "processed_contexts": processed,
        "skipped_contexts": skipped,
        "contexts_with_candidates": contexts_with_candidates,
        "generated_candidates": generated,
        "context_coverage": contexts_with_candidates / processed if processed else 0.0,
        "avg_candidates_per_context": generated / processed if processed else 0.0,
        "avg_candidates_per_covered_context": generated / contexts_with_candidates if contexts_with_candidates else 0.0,
        "by_task": by_task,
        "by_failure_type": by_failure,
        "selected_tasks": list(tasks),
        "max_candidates_per_context": max_candidates_per_context,
        "include_failed": include_failed,
        "include_same_product": include_same_product,
        "review_status": review_status,
        "limit": limit,
        "start_index": start_index,
        "notes": [
            "CPU-only targeted generation for external product-selection contexts.",
            "Default task filter keeps forward_outcome candidates because retro_precursor rows usually keep the observed product and do not expand product-reranking alternatives.",
            "Rows are intended to be passed as an additional --synthetic-csv into build_external_product_prediction_candidate_set.",
        ],
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    if summary_md:
        write_summary_md(summary_md, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--summary-md", default=None)
    parser.add_argument("--task", action="append", dest="tasks", default=None)
    parser.add_argument("--max-candidates-per-context", type=int, default=8)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--include-same-product", action="store_true")
    parser.add_argument("--review-status", default="keep_synthetic_negative")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    tasks = args.tasks or ["forward_outcome"]
    summary = generate_for_contexts(
        context_csv=args.context_csv,
        output_csv=args.output,
        summary_json=args.summary,
        summary_md=args.summary_md,
        tasks=tasks,
        max_candidates_per_context=args.max_candidates_per_context,
        include_failed=args.include_failed,
        include_same_product=args.include_same_product,
        review_status=args.review_status,
        limit=args.limit,
        start_index=args.start_index,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
