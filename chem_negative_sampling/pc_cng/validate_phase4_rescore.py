"""Validate a repaired Phase-4 shuffled-parent rescore directory.

This checks provenance-sensitive file freshness and the class/feature contract
before an aggregation script is allowed to consume the outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path


SCENARIOS = (
    "author_lab",
    "condition_space",
    "random",
    "reaction_family",
    "scaffold",
    "time",
    "ni_coupling",
    "uspto_patent",
)


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def validate(base: Path, since: datetime) -> None:
    records_dir = base / "per_scenario_records"
    if not records_dir.is_dir():
        raise RuntimeError(f"missing records directory: {records_dir}")

    for scenario in SCENARIOS:
        path = records_dir / f"{scenario}__shuffled_parent__semi_hard.csv"
        if not path.is_file():
            raise RuntimeError(f"missing scenario output: {path.name}")
        mtime = datetime.fromtimestamp(path.stat().st_mtime, since.tzinfo)
        if mtime <= since:
            raise RuntimeError(
                f"stale copied output: {path.name} mtime={mtime.isoformat()}"
            )
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise RuntimeError(f"empty scenario output: {path.name}")

        labels = [int(row["label"]) for row in rows]
        positives = [_bool(row.get("is_positive", "")) for row in rows]
        if set(labels) != {0, 1}:
            raise RuntimeError(f"{path.name}: labels are not binary: {set(labels)}")
        # The fixed test pool is not required to be 1:1 balanced; the
        # rescore training builder enforces that invariant separately before
        # fitting.  Here we only require both classes to be represented.
        if sum(labels) == 0 or sum(labels) == len(labels):
            raise RuntimeError(f"{path.name}: test pool has only one class")
        if any(label != int(flag) for label, flag in zip(labels, positives)):
            raise RuntimeError(f"{path.name}: label/is_positive contract mismatch")
        scores = [float(row["score"]) for row in rows]
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError(f"{path.name}: non-finite score")
        print(
            f"{scenario}: rows={len(rows)} positives={sum(labels)} "
            f"score_min={min(scores):.6f} score_max={max(scores):.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_results", type=Path)
    parser.add_argument("--since", required=True)
    args = parser.parse_args()
    since = datetime.fromisoformat(args.since)
    validate(args.base_results, since)
    print("phase4 rescore validation: PASS")


if __name__ == "__main__":
    main()
