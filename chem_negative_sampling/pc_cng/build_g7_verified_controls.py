"""Build deterministic real HTE positive/negative controls for G7 v2.

The output is an unblinded coordinator artifact.  Reviewer-facing forms are
created separately by :mod:`pc_cng.p4_g7_sampling_v2`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


REQUIRED_COLUMNS = {
    "record_id",
    "source_publication",
    "measured_yield",
    "missing_measurement",
    "reported_zero",
    "reaction_family",
    "reaction_smiles",
    "experimental_group",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_rank(record_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _family_balanced_pick(
    rows: List[Dict[str, Any]],
    *,
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["reaction_family"])].append(row)
    for family_rows in by_family.values():
        family_rows.sort(key=lambda row: _stable_rank(str(row["record_id"]), seed))
    families = sorted(by_family)
    selected: List[Dict[str, Any]] = []
    cursor = 0
    while len(selected) < n and families:
        family = families[cursor % len(families)]
        pool = by_family[family]
        if pool:
            selected.append(pool.pop(0))
        if not pool:
            families.remove(family)
            if not families:
                break
            cursor %= len(families)
        else:
            cursor += 1
    if len(selected) < n:
        raise RuntimeError(f"insufficient eligible controls: requested {n}, found {len(selected)}")
    return selected


def build_controls(
    frame: pd.DataFrame,
    *,
    n_per_type: int = 10,
    positive_yield_min: float = 80.0,
    negative_yield_max: float = 0.0,
    seed: int = 20260729,
) -> List[Dict[str, Any]]:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"HTE table missing required columns: {missing}")
    if n_per_type <= 0:
        raise ValueError("n_per_type must be positive")

    rows = frame.to_dict("records")
    eligible = []
    seen_reactions = set()
    seen_groups = set()
    for row in rows:
        reaction = str(row.get("reaction_smiles", "")).strip()
        group = str(row.get("experimental_group", "")).strip()
        if (
            not reaction
            or ">" not in reaction
            or _truthy(row.get("missing_measurement"))
            or reaction in seen_reactions
            or not group
            or group in seen_groups
        ):
            continue
        try:
            measured_yield = float(row["measured_yield"])
        except (TypeError, ValueError):
            continue
        copied = dict(row)
        copied["_yield"] = measured_yield
        eligible.append(copied)
        seen_reactions.add(reaction)
        seen_groups.add(group)

    positives = [
        row for row in eligible if row["_yield"] >= positive_yield_min
    ]
    negatives = [
        row
        for row in eligible
        if row["_yield"] <= negative_yield_max
        and _truthy(row.get("reported_zero"))
    ]
    selected = {
        "positive_control": _family_balanced_pick(
            positives,
            n=n_per_type,
            seed=seed + 1,
        ),
        "obvious_negative_control": _family_balanced_pick(
            negatives,
            n=n_per_type,
            seed=seed + 2,
        ),
    }
    controls: List[Dict[str, Any]] = []
    for control_type, selected_rows in selected.items():
        for row in selected_rows:
            record_id = str(row["record_id"])
            controls.append(
                {
                    "control_id": (
                        "HTE-"
                        + hashlib.sha256(
                            f"{control_type}:{record_id}".encode("utf-8")
                        ).hexdigest()[:16]
                    ),
                    "control_type": control_type,
                    "reaction_smiles": str(row["reaction_smiles"]),
                    "experimental_provenance": (
                        f"{row['source_publication']}; record_id={record_id}; "
                        f"measured_yield={row['_yield']}"
                    ),
                    "verification_status": "INDEPENDENTLY_VERIFIED",
                    "reaction_family": str(row["reaction_family"]),
                }
            )
    return controls


def write_controls(
    controls: Sequence[Dict[str, Any]],
    *,
    output_csv: Path,
    manifest_path: Path,
    input_path: Path,
    parameters: Dict[str, Any],
) -> None:
    fields = [
        "control_id",
        "control_type",
        "reaction_smiles",
        "experimental_provenance",
        "verification_status",
        "reaction_family",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(controls)
    manifest = {
        "schema": "p4_g7_verified_hte_controls_v1",
        "scientific_status": "CONTROL_POOL_ONLY_NOT_EXPERT_EVIDENCE",
        "input": {
            "path": str(input_path.resolve()),
            "sha256": _sha256(input_path.resolve()),
        },
        "output": {
            "path": str(output_csv.resolve()),
            "sha256": _sha256(output_csv.resolve()),
            "n_rows": len(controls),
        },
        "selection_parameters": parameters,
        "control_counts": {
            control_type: sum(
                row["control_type"] == control_type for row in controls
            )
            for control_type in ("positive_control", "obvious_negative_control")
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--n-per-type", type=int, default=10)
    parser.add_argument("--positive-yield-min", type=float, default=80.0)
    parser.add_argument("--negative-yield-max", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input.suffix == ".parquet":
        frame = pd.read_parquet(args.input)
    else:
        frame = pd.read_csv(args.input)
    parameters = {
        "n_per_type": args.n_per_type,
        "positive_yield_min": args.positive_yield_min,
        "negative_yield_max": args.negative_yield_max,
        "negative_requires_reported_zero": True,
        "unique_reaction_and_experimental_group": True,
        "family_balanced_round_robin": True,
        "seed": args.seed,
    }
    controls = build_controls(
        frame,
        n_per_type=args.n_per_type,
        positive_yield_min=args.positive_yield_min,
        negative_yield_max=args.negative_yield_max,
        seed=args.seed,
    )
    write_controls(
        controls,
        output_csv=args.output,
        manifest_path=args.manifest,
        input_path=args.input,
        parameters=parameters,
    )
    print(
        json.dumps(
            {
                "n_controls": len(controls),
                "output": str(args.output),
                "manifest": str(args.manifest),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
