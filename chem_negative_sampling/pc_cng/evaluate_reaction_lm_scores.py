"""Evaluate reaction LM candidate scores with ranking metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

from .ranking_metrics import grouped_metrics, ranking_metrics


def read_scored_rows(path: str, score_column: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if score_column not in row:
                raise ValueError(f"Missing score column {score_column!r} in {path}")
            out: Dict[str, object] = dict(row)
            out["score"] = float(row[score_column])
            out["reaction_smiles"] = row.get("candidate_reaction", "")
            out["label"] = int(row.get("label", 0) or 0)
            rows.append(out)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Scored candidate CSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-column", default="lm_score")
    args = parser.parse_args()

    rows = read_scored_rows(args.input, args.score_column)
    summary = {
        "input": args.input,
        "score_column": args.score_column,
        "rows": len(rows),
        "overall": ranking_metrics(rows),
        "by_split": grouped_metrics(rows, "split"),
        "by_dataset": grouped_metrics(rows, "dataset"),
        "by_candidate_source": grouped_metrics(rows, "candidate_source"),
        "by_candidate_family": grouped_metrics(rows, "candidate_family"),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
