"""Independent artifact verifier for the exploratory Phase 4 Union_v2 run."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


EXPECTED_SCENARIOS = {
    "author_lab",
    "condition_space",
    "ni_coupling",
    "random",
    "reaction_family",
    "scaffold",
    "time",
    "uspto_patent",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _signature(row: Mapping[str, str]) -> Tuple[str, int, bool, str, str]:
    return (
        str(row.get("reaction_smiles", "")),
        int(row.get("label", 0)),
        str(row.get("is_positive", "")).lower() in {"true", "1"},
        str(row.get("source", "?")),
        str(row.get("experimental_group", "default")),
    )


def _read_records(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def parse_source_mixtures(log_text: str) -> Dict[str, Dict[str, int]]:
    current: Optional[str] = None
    mixtures: Dict[str, Dict[str, int]] = {}
    scenario_re = re.compile(r"^\[union\] === Scenario: ([^=]+?) ===$")
    mixture_re = re.compile(r"^\s*\[union_v2\] train source mixture: (\{.*\})$")
    for line in log_text.splitlines():
        scenario_match = scenario_re.match(line.strip())
        if scenario_match:
            current = scenario_match.group(1).strip()
            continue
        mixture_match = mixture_re.match(line)
        if mixture_match and current:
            parsed = ast.literal_eval(mixture_match.group(1))
            mixtures[current] = {str(k): int(v) for k, v in parsed.items()}
    return mixtures


def verify(args: argparse.Namespace) -> Dict[str, Any]:
    result_path = args.base_results / "union_arm_results.json"
    records_dir = args.base_results / "per_scenario_records"
    result = json.loads(result_path.read_text())
    log_text = args.log.read_text(errors="replace")
    mixtures = parse_source_mixtures(log_text)
    scenarios = set(result.get("results", {}))

    checks: Dict[str, Any] = {
        "result_arm_is_union_v2": result.get("arm") == "learned_union_v2",
        "difficulty_match_requested": bool(
            result.get("difficulty_match_requested")
        ),
        "all_expected_scenarios_present": scenarios == EXPECTED_SCENARIOS,
        "log_confirms_learned_model_loaded": "[union] G8-C model loaded" in log_text,
        "log_has_no_checkpoint_load_failure": "failed to load checkpoint" not in log_text,
        "source_mixture_logged_for_all_scenarios": set(mixtures)
        == EXPECTED_SCENARIOS,
        "learned_source_selected_in_all_scenarios": all(
            mix.get("learned_structured", 0) > 0 for mix in mixtures.values()
        )
        if mixtures
        else False,
        "checkpoint_exists": args.checkpoint.exists(),
    }

    scenario_audit: Dict[str, Any] = {}
    for scenario in sorted(EXPECTED_SCENARIOS):
        union_path = (
            records_dir
            / f"{scenario}__learned_union_v2__semi_hard.csv"
        )
        base_path = (
            records_dir
            / f"{scenario}__shuffled_parent__semi_hard.csv"
        )
        audit: Dict[str, Any] = {
            "union_records_exist": union_path.exists(),
            "base_records_exist": base_path.exists(),
            "source_mixture": mixtures.get(scenario),
        }
        if union_path.exists() and base_path.exists():
            union_rows = _read_records(union_path)
            base_rows = _read_records(base_path)
            audit["n_union_records"] = len(union_rows)
            audit["n_base_records"] = len(base_rows)
            audit["record_alignment"] = (
                len(union_rows) == len(base_rows)
                and all(
                    _signature(union_row) == _signature(base_row)
                    for union_row, base_row in zip(union_rows, base_rows)
                )
            )
            audit["all_scores_finite"] = all(
                math.isfinite(float(row["score"])) for row in union_rows
            )
        scenario_result = result.get("results", {}).get(scenario, {})
        difficulty = scenario_result.get("train_difficulty", {})
        audit["n_train_negative"] = difficulty.get("n_negative")
        audit["matched_fraction"] = difficulty.get("matched_fraction")
        audit["difficulty_metadata_complete"] = (
            isinstance(difficulty.get("n_negative"), int)
            and difficulty.get("n_negative", 0) > 0
            and isinstance(difficulty.get("n_in_band"), int)
            and isinstance(difficulty.get("fallback_count"), int)
            and difficulty.get("matched_fraction") is not None
        )
        scenario_audit[scenario] = audit

    checks["all_record_sets_aligned"] = all(
        audit.get("record_alignment") for audit in scenario_audit.values()
    )
    checks["all_scores_finite"] = all(
        audit.get("all_scores_finite") for audit in scenario_audit.values()
    )
    checks["difficulty_metadata_complete"] = all(
        audit.get("difficulty_metadata_complete")
        for audit in scenario_audit.values()
    )
    checks["all_scenarios_have_multiple_selected_sources"] = all(
        len([count for count in (audit.get("source_mixture") or {}).values() if count > 0])
        >= 2
        for audit in scenario_audit.values()
    )

    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None

    verified = all(bool(value) for value in checks.values())
    out = {
        "schema_version": "phase4_union_v2_verification_v1",
        "verified": verified,
        "claim_scope": "exploratory development arm only",
        "checks": checks,
        "scenario_audit": scenario_audit,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256(args.checkpoint) if args.checkpoint.exists() else None,
        },
        "inputs": {
            "union_result": {
                "path": str(result_path),
                "sha256": _sha256(result_path),
            },
            "run_log": {
                "path": str(args.log),
                "sha256": _sha256(args.log),
            },
        },
        "git_commit_at_verification": git_commit,
    }
    args.output.write_text(json.dumps(out, indent=2))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    result = verify(build_parser().parse_args())
    print(json.dumps(result, indent=2))
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
