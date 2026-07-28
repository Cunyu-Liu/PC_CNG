#!/usr/bin/env python3
"""Verify that a G6 v3 primary inference was independently reconstructed.

The original a602a41 formal artifact serialized NumPy booleans as ``"True"``
or ``"False"`` because its writer used ``default=str``.  This verifier accepts
that one documented legacy representation, normalizes it to JSON booleans, and
requires every remaining key and value to match exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pc_cng.p4_g6_benchmark_v3 import validate_formal_analysis_plan

_LEGACY_NUMPY_BOOL_FIELDS = frozenset({"ci_all_positive"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_legacy_json_scalars(
    value: Any,
    *,
    field_name: str | None = None,
) -> Any:
    """Normalize only documented legacy NumPy booleans at known fields."""
    if isinstance(value, dict):
        return {
            key: canonicalize_legacy_json_scalars(item, field_name=key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            canonicalize_legacy_json_scalars(item, field_name=field_name)
            for item in value
        ]
    if field_name in _LEGACY_NUMPY_BOOL_FIELDS and value == "True":
        return True
    if field_name in _LEGACY_NUMPY_BOOL_FIELDS and value == "False":
        return False
    return value


def verify_reconstruction(
    formal_result: dict[str, Any],
    independent_inference: dict[str, Any],
    analysis_plan: dict[str, Any],
) -> dict[str, Any]:
    validate_formal_analysis_plan(analysis_plan)
    if formal_result.get("scientific_status") != "FORMAL":
        raise AssertionError("formal result does not carry scientific_status=FORMAL")
    embedded_plan = formal_result.get("analysis_plan")
    if embedded_plan != analysis_plan:
        raise AssertionError("formal result analysis plan differs from frozen plan")
    formal_primary = canonicalize_legacy_json_scalars(
        formal_result.get("primary_inference")
    )
    independent_primary = canonicalize_legacy_json_scalars(independent_inference)
    if formal_primary != independent_primary:
        raise AssertionError(
            "independent primary inference differs from the formal result"
        )
    comparisons = independent_primary.get("comparisons", [])
    expected = list(analysis_plan["primary_comparisons"])
    observed = [item.get("comparison") for item in comparisons]
    if observed != expected:
        raise AssertionError(
            f"comparison order differs from frozen plan: {observed!r} != {expected!r}"
        )
    return {
        "verified": True,
        "verification_scope": "complete primary_inference object",
        "legacy_normalization": {
            "accepted_fields": sorted(_LEGACY_NUMPY_BOOL_FIELDS),
            "accepted_values": ["True", "False"],
            "reason": "a602a41 formal writer serialized NumPy bool scalars via default=str",
        },
        "comparison_order": observed,
        "all_superiority_confirmed": all(
            bool(item["superiority_confirmed"]) for item in comparisons
        ),
        "baseline_selection": independent_primary.get("baseline_selection"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-result", type=Path, required=True)
    parser.add_argument("--independent-inference", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    formal_result = json.loads(args.formal_result.read_text())
    independent = json.loads(args.independent_inference.read_text())
    plan = json.loads(args.analysis_plan.read_text())
    result = verify_reconstruction(formal_result, independent, plan)
    result["artifact_hashes"] = {
        "formal_result": _sha256(args.formal_result),
        "independent_inference": _sha256(args.independent_inference),
        "analysis_plan": _sha256(args.analysis_plan),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
