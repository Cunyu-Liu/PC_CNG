"""Build Science Advances-style RegioSQM20 few-positive splits.

The Science Advances negative-data paper evaluates low-positive regimes with
RegioSQM20 positives plus many negatives. This builder creates comparable
local splits from our normalized RegioSQM20 CSV while keeping validation/test
reaction contexts out of the training negatives.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence


OUTPUT_FIELDS = [
    "source_id",
    "reaction_smiles",
    "reactants",
    "agents",
    "products",
    "label_type",
    "yield",
    "source",
    "split_key",
    "split",
    "reaction_class",
]


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: str, rows: Sequence[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in OUTPUT_FIELDS}
            writer.writerow(out)


def count_by(rows: Iterable[Dict[str, str]], key: str) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[row.get(key, "")] += 1
    return dict(counts)


def clone_with_split(row: Dict[str, str], split: str) -> Dict[str, str]:
    out = dict(row)
    out["split"] = split
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--setting", choices=["k_low", "k_high"], required=True)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--train-positives", type=int, default=None)
    parser.add_argument("--val-positives", type=int, default=165)
    parser.add_argument("--test-positives", type=int, default=164)
    parser.add_argument("--train-negatives", type=int, default=748)
    args = parser.parse_args()

    train_positive_target = args.train_positives
    if train_positive_target is None:
        train_positive_target = 22 if args.setting == "k_low" else 220

    rng = random.Random(args.seed)
    rows = read_rows(args.input)
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("label_type") not in {"positive", "real_negative"}:
            continue
        reactants = row.get("reactants", "")
        if reactants:
            grouped[reactants].append(row)

    eligible_groups = []
    for reactants, group_rows in grouped.items():
        labels = Counter(row.get("label_type", "") for row in group_rows)
        if labels["positive"] > 0 and labels["real_negative"] > 0:
            eligible_groups.append(reactants)
    eligible_groups = sorted(eligible_groups)
    rng.shuffle(eligible_groups)

    required = train_positive_target + args.val_positives + args.test_positives
    if len(eligible_groups) < required:
        raise RuntimeError(
            f"Not enough eligible RegioSQM20 groups: need {required}, found {len(eligible_groups)}"
        )

    train_groups = set(eligible_groups[:train_positive_target])
    val_start = train_positive_target
    val_groups = set(eligible_groups[val_start : val_start + args.val_positives])
    test_start = val_start + args.val_positives
    test_groups = set(eligible_groups[test_start : test_start + args.test_positives])
    heldout_groups = val_groups | test_groups

    train_positive_rows: List[Dict[str, str]] = []
    train_negative_pool: List[Dict[str, str]] = []
    val_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []

    for reactants, group_rows in grouped.items():
        if reactants in train_groups:
            train_positive_rows.extend(
                clone_with_split(row, "train") for row in group_rows if row.get("label_type") == "positive"
            )
            train_negative_pool.extend(
                clone_with_split(row, "train") for row in group_rows if row.get("label_type") == "real_negative"
            )
        elif reactants in val_groups:
            val_rows.extend(clone_with_split(row, "val") for row in group_rows)
        elif reactants in test_groups:
            test_rows.extend(clone_with_split(row, "test") for row in group_rows)
        elif reactants not in heldout_groups:
            train_negative_pool.extend(
                clone_with_split(row, "train") for row in group_rows if row.get("label_type") == "real_negative"
            )

    if len(train_negative_pool) > args.train_negatives:
        train_negative_rows = rng.sample(train_negative_pool, args.train_negatives)
    else:
        train_negative_rows = list(train_negative_pool)

    output_rows = train_positive_rows + train_negative_rows + val_rows + test_rows
    output_rows.sort(key=lambda row: (row.get("split", ""), row.get("reactants", ""), row.get("label_type", "")))

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_path = os.path.join(args.output_dir, "regiosqm20_science_advances_split.csv")
    train_positive_path = os.path.join(args.output_dir, "train_positives.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")
    write_rows(dataset_path, output_rows)
    write_rows(train_positive_path, train_positive_rows)

    summary = {
        "input": args.input,
        "setting": args.setting,
        "seed": args.seed,
        "eligible_groups": len(eligible_groups),
        "targets": {
            "train_positives": train_positive_target,
            "train_negatives": args.train_negatives,
            "val_positives": args.val_positives,
            "test_positives": args.test_positives,
        },
        "actual": {
            "rows": len(output_rows),
            "train_positive_rows": len(train_positive_rows),
            "train_negative_rows": len(train_negative_rows),
            "train_negative_pool": len(train_negative_pool),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "labels": count_by(output_rows, "label_type"),
            "splits": count_by(output_rows, "split"),
        },
        "paths": {
            "dataset": dataset_path,
            "train_positives": train_positive_path,
            "summary": summary_path,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
