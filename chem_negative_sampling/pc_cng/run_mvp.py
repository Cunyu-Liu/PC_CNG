"""Run the PC-CNG MVP end to end.

Example:
    python3 -m pc_cng.run_mvp --output-dir results/pc_cng_mvp

With custom data:
    python3 -m pc_cng.run_mvp --input data/positives.csv --output-dir results/run1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List

from .counterfactual import CounterfactualGenerator
from .reranker import LogisticReactionRanker, evaluate_ranking, split_by_source


DEMO_REACTIONS = [
    {"source_id": "demo_001", "reaction_smiles": "CC(=O)Cl.CN>>CC(=O)NC"},
    {"source_id": "demo_002", "reaction_smiles": "CC(=O)Cl.OCC>>CC(=O)OCC"},
    {"source_id": "demo_003", "reaction_smiles": "O=C(Cl)c1ccccc1.CN>>CNC(=O)c1ccccc1"},
    {"source_id": "demo_004", "reaction_smiles": "CCBr.N>>CCN"},
    {"source_id": "demo_005", "reaction_smiles": "CCBr.O>>CCO"},
    {"source_id": "demo_006", "reaction_smiles": "c1ccccc1Br.O>>c1ccccc1O"},
    {"source_id": "demo_007", "reaction_smiles": "C=CCBr.N>>C=CCN"},
    {"source_id": "demo_008", "reaction_smiles": "CC(=O)O.CN>>CC(=O)NC"},
    {"source_id": "demo_009", "reaction_smiles": "COC(=O)C.N>>NC(=O)C"},
    {"source_id": "demo_010", "reaction_smiles": "CCCl.N>>CCN"},
]


def read_positive_reactions(path: str | None) -> List[Dict[str, str]]:
    if path is None:
        return list(DEMO_REACTIONS)

    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "reaction_smiles" not in (reader.fieldnames or []):
            raise ValueError("Input CSV must contain a reaction_smiles column")
        for index, row in enumerate(reader, start=1):
            reaction = (row.get("reaction_smiles") or "").strip()
            if not reaction:
                continue
            source_id = (row.get("source_id") or row.get("id") or f"row_{index:06d}").strip()
            rows.append({"source_id": source_id, "reaction_smiles": reaction})
    return rows


def write_csv(path: str, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_ranker_rows(positive_rows: List[Dict[str, str]], negative_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in positive_rows:
        rows.append(
            {
                "source_id": row["source_id"],
                "reaction_smiles": row["reaction_smiles"],
                "label": 1,
                "task": "positive",
                "failure_type": "positive",
            }
        )
    for row in negative_rows:
        rows.append(
            {
                "source_id": row["source_id"],
                "reaction_smiles": row["candidate_reaction"],
                "label": 0,
                "task": row["task"],
                "failure_type": row["failure_type"],
            }
        )
    return rows


def summarize_negatives(rows: List[Dict[str, object]]) -> Dict[str, object]:
    by_task: Dict[str, int] = {}
    by_failure: Dict[str, int] = {}
    hard_scores = []
    fn_risks = []
    for row in rows:
        task = str(row["task"])
        failure = str(row["failure_type"])
        by_task[task] = by_task.get(task, 0) + 1
        by_failure[failure] = by_failure.get(failure, 0) + 1
        hard_scores.append(float(row["hard_score"]))
        fn_risks.append(float(row["false_negative_risk"]))
    return {
        "total": len(rows),
        "by_task": by_task,
        "by_failure_type": by_failure,
        "avg_hard_score": sum(hard_scores) / len(hard_scores) if hard_scores else 0.0,
        "avg_false_negative_risk": sum(fn_risks) / len(fn_risks) if fn_risks else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="CSV with reaction_smiles and optional source_id")
    parser.add_argument("--output-dir", type=str, default="results/pc_cng_mvp")
    parser.add_argument("--include-failed", action="store_true", help="Keep candidates failing MVP filters")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    positives = read_positive_reactions(args.input)
    generator = CounterfactualGenerator()

    negative_dicts: List[Dict[str, object]] = []
    for row in positives:
        candidates = generator.generate_for_reaction(
            row["reaction_smiles"],
            source_id=row["source_id"],
            include_failed=args.include_failed,
        )
        negative_dicts.extend(candidate.to_dict() for candidate in candidates)

    positive_path = os.path.join(args.output_dir, "positive_reactions.csv")
    negative_path = os.path.join(args.output_dir, "synthetic_counterfactual_negatives.csv")
    ranker_data_path = os.path.join(args.output_dir, "ranker_dataset.csv")
    model_path = os.path.join(args.output_dir, "lightweight_ranker.json")
    metrics_path = os.path.join(args.output_dir, "mvp_metrics.json")

    write_csv(positive_path, positives, ["source_id", "reaction_smiles"])
    negative_fields = [
        "source_id",
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
    ]
    write_csv(negative_path, negative_dicts, negative_fields)

    ranker_rows = build_ranker_rows(positives, negative_dicts)
    write_csv(
        ranker_data_path,
        ranker_rows,
        ["source_id", "reaction_smiles", "label", "task", "failure_type"],
    )

    retro_rows = [row for row in ranker_rows if row["task"] in {"positive", "retro_precursor"}]
    train_rows, test_rows = split_by_source(retro_rows)
    ranker = LogisticReactionRanker(epochs=args.epochs)
    ranker.fit(train_rows)
    ranker.save_json(model_path)
    metrics = {
        "positive_count": len(positives),
        "negative_summary": summarize_negatives(negative_dicts),
        "retro_train_rows": len(train_rows),
        "retro_test_rows": len(test_rows),
        "retro_ranking_metrics": evaluate_ranking(ranker, test_rows).to_dict(),
        "outputs": {
            "positive_reactions": positive_path,
            "synthetic_counterfactual_negatives": negative_path,
            "ranker_dataset": ranker_data_path,
            "lightweight_ranker": model_path,
        },
        "notes": [
            "MVP metrics are smoke-test evidence only, not publishable claims.",
            "Synthetic rows must not be represented as real failed experiments.",
            "Scale-up requires RDKit atom mapping, real HTE/RegioSQM negatives, and stronger baselines.",
        ],
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

