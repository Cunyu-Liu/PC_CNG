"""Inspect top-ranked errors in candidate reranking outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def analyze(rows: List[Dict[str, str]], reaction_class: str, limit: int) -> Dict[str, object]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if reaction_class and row.get("reaction_class") != reaction_class:
            continue
        grouped[row["group_id"]].append(row)

    errors: List[Dict[str, object]] = []
    top_family_counts: Counter[str] = Counter()
    evaluated = 0
    for group_id, group_rows in sorted(grouped.items()):
        labels = [int(float(row.get("label", 0) or 0)) for row in group_rows]
        if not any(labels) or all(labels):
            continue
        evaluated += 1
        ranked = sorted(group_rows, key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
        if int(float(ranked[0].get("label", 0) or 0)) == 1:
            continue
        positive = next(row for row in ranked if int(float(row.get("label", 0) or 0)) == 1)
        top_family = ranked[0].get("candidate_family", "")
        top_family_counts[top_family] += 1
        errors.append(
            {
                "group_id": group_id,
                "source_id": ranked[0].get("source_id", ""),
                "split": ranked[0].get("split", ""),
                "top_family": top_family,
                "top_score": float(ranked[0].get("score", 0.0) or 0.0),
                "positive_score": float(positive.get("score", 0.0) or 0.0),
                "score_margin_top_minus_positive": float(ranked[0].get("score", 0.0) or 0.0)
                - float(positive.get("score", 0.0) or 0.0),
                "ranked_rows": [
                    {
                        "label": int(float(row.get("label", 0) or 0)),
                        "candidate_family": row.get("candidate_family", ""),
                        "score": float(row.get("score", 0.0) or 0.0),
                        "reaction_smiles": row.get("reaction_smiles", "")[:240],
                    }
                    for row in ranked[:limit]
                ],
            }
        )

    return {
        "reaction_class": reaction_class,
        "evaluated_groups": evaluated,
        "error_groups": len(errors),
        "top1": 1.0 - (len(errors) / evaluated if evaluated else 0.0),
        "top_error_family_counts": dict(top_family_counts),
        "errors": errors[:limit],
    }


def write_markdown(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        f"# Reranking error analysis: {payload.get('reaction_class', '')}",
        "",
        f"- evaluated_groups: {payload.get('evaluated_groups', 0)}",
        f"- error_groups: {payload.get('error_groups', 0)}",
        f"- top1: {float(payload.get('top1', 0.0)) * 100.0:.2f}",
        f"- top_error_family_counts: {json.dumps(payload.get('top_error_family_counts', {}), ensure_ascii=False)}",
        "",
    ]
    for index, error in enumerate(payload.get("errors", []), start=1):
        error = dict(error)
        lines.extend(
            [
                f"## Error {index}: {error.get('source_id', '')}",
                "",
                f"- split: {error.get('split', '')}",
                f"- top_family: {error.get('top_family', '')}",
                f"- top_score: {float(error.get('top_score', 0.0)):.6f}",
                f"- positive_score: {float(error.get('positive_score', 0.0)):.6f}",
                f"- margin_top_minus_positive: {float(error.get('score_margin_top_minus_positive', 0.0)):.6f}",
                "",
            ]
        )
        for row in list(error.get("ranked_rows", [])):
            row = dict(row)
            lines.append(
                f"  - label={row.get('label')} family={row.get('candidate_family')} "
                f"score={float(row.get('score', 0.0)):.6f} rxn={row.get('reaction_smiles', '')}"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-scores", required=True)
    parser.add_argument("--reaction-class", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    payload = analyze(read_rows(args.candidate_scores), args.reaction_class, args.limit)
    json_path = os.path.join(args.output_dir, "reranking_errors.json")
    md_path = os.path.join(args.output_dir, "reranking_errors.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    write_markdown(md_path, payload)
    print(json.dumps({"json": json_path, "markdown": md_path, **payload}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
