"""Run baseline comparison for retrosynthesis/reranking smoke experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List

from .baselines import (
    dora_alternate_center,
    load_positive_rows,
    pc_cng_rule,
    pu_reliable_negative,
    random_mismatch,
    template_perturbation,
)
from .reranker import LogisticReactionRanker, evaluate_ranking, split_by_source


BASELINES = {
    "random": random_mismatch,
    "template_perturbation": template_perturbation,
    "dora_alternate_center": dora_alternate_center,
    "pu_reliable_negative": pu_reliable_negative,
    "pc_cng_rule_mvp": pc_cng_rule,
}


def ranker_rows(positives: List[Dict[str, str]], negatives: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in positives:
        rows.append(
            {
                "source_id": row["source_id"],
                "reaction_smiles": row["reaction_smiles"],
                "label": 1,
                "task": "positive",
                "failure_type": "positive",
            }
        )
    for row in negatives:
        rows.append(
            {
                "source_id": row["source_id"],
                "reaction_smiles": row["candidate_reaction"],
                "label": 0,
                "task": row.get("task", "baseline"),
                "failure_type": row.get("failure_type", "negative"),
            }
        )
    return rows


def write_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["source_id", "reaction_smiles", "label", "task", "failure_type"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Positive reaction CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    positives = load_positive_rows(args.input, args.limit)
    results: Dict[str, object] = {"input": args.input, "positive_rows": len(positives), "baselines": {}}

    for baseline_name, generator in BASELINES.items():
        negatives = generator(positives)
        rows = ranker_rows(positives, negatives)
        train_rows, test_rows = split_by_source(rows)
        model = LogisticReactionRanker(epochs=args.epochs)
        model.fit(train_rows)
        metrics = evaluate_ranking(model, test_rows).to_dict()

        baseline_dir = os.path.join(args.output_dir, baseline_name)
        os.makedirs(baseline_dir, exist_ok=True)
        write_csv(os.path.join(baseline_dir, "ranker_dataset.csv"), rows)
        model.save_json(os.path.join(baseline_dir, "lightweight_ranker.json"))
        with open(os.path.join(baseline_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)

        results["baselines"][baseline_name] = {
            "negative_rows": len(negatives),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "metrics": metrics,
        }

    summary_path = os.path.join(args.output_dir, "experiment_matrix_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

