"""Fail-closed three-expert pilot builder for Phase F / G7.

The historical G7 builder allowed proxy "real negatives", blank reaction
contexts and two-reviewer pilots.  This v2 builder refuses all three.  It uses
Phase-D complete-case candidate caches, a frozen source-policy map and an
independently curated obvious-negative control file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA = "p4_g7_expert_pilot_v2"
DEFAULT_SEED = 20260729
DEFAULT_PER_STRATUM = 10
MIN_REVIEWERS = 3

STRATA = (
    "positive_control",
    "obvious_negative_control",
    "random_mismatch",
    "rule_pc_cng",
    "learned_structured",
    "shuffled_real",
    "uniform_union",
    "learned_source_gate",
)
SOURCE_STRATA = {
    "random_mismatch": "random_mismatch",
    "rule_pc_cng": "rule_pc_cng",
    "learned_structured": "learned_structured",
    "shuffled_real": "shuffled_real",
}
SOURCE_NAMES = (
    "random_mismatch",
    "shuffled_real",
    "similarity_retrieval",
    "template_perturbation",
    "rule_pc_cng",
    "learned_structured",
)
SCORING_DIMENSIONS = (
    "structural_validity",
    "mechanistic_plausibility",
    "plausible_competing_outcome",
    "likely_low_yield_or_failure",
    "expert_false_negative_risk_assessment",
    "confidence",
)
REASON_CODES = (
    "wrong_reaction_center",
    "unlikely_bond_change",
    "condition_mismatch",
    "chemoselectivity_issue",
    "regioselectivity_issue",
    "stereochemistry_issue",
    "plausible_side_product",
    "likely_feasible_alternative",
    "insufficient_information",
    "other",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha(repo_root: Optional[Path]) -> str:
    command = ["git"]
    if repo_root is not None:
        command.extend(["-C", str(repo_root)])
    command.extend(["rev-parse", "HEAD"])
    return subprocess.check_output(command, text=True).strip()


def _parse_reaction(reaction: str) -> Tuple[str, str, str]:
    if not reaction or ">" not in reaction:
        raise ValueError("expert item requires complete reaction context")
    parts = reaction.split(">")
    if len(parts) < 3:
        raise ValueError("reaction must encode reactants>conditions>product")
    reactants = parts[0].strip()
    conditions = ">".join(parts[1:-1]).strip()
    product = parts[-1].strip()
    if not reactants or not product:
        raise ValueError("reaction context has empty reactants or product")
    return reactants, conditions, product


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_candidate_caches(
    cache_paths: Sequence[Path],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    entries: List[Dict[str, Any]] = []
    artifacts: List[Dict[str, Any]] = []
    seen_groups = set()
    for cache_path in cache_paths:
        resolved = cache_path.resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"candidate cache must contain a list: {resolved}")
        scenario = resolved.stem
        artifacts.append(
            {
                "path": str(resolved),
                "sha256": _sha256(resolved),
                "n_entries": len(payload),
            }
        )
        for item in payload:
            group = str(item.get("group", "")).strip()
            candidates = item.get("candidates", {})
            if not group or group in seen_groups:
                continue
            if sorted(candidates) != sorted(SOURCE_NAMES):
                continue
            reaction = str(item.get("reaction_smiles", ""))
            _parse_reaction(reaction)
            copied = dict(item)
            copied["scenario"] = scenario
            entries.append(copied)
            seen_groups.add(group)
    if not entries:
        raise RuntimeError("no complete six-source candidate parents found")
    return entries, artifacts


def load_policy_maps(
    policy_paths: Sequence[Path],
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    selected: Dict[str, str] = {}
    artifacts: List[Dict[str, Any]] = []
    for policy_path in policy_paths:
        resolved = policy_path.resolve()
        rows = _read_csv(resolved)
        artifacts.append(
            {
                "path": str(resolved),
                "sha256": _sha256(resolved),
                "n_rows": len(rows),
            }
        )
        for row in rows:
            group = str(row.get("group", "")).strip()
            source = str(row.get("selected_source", "")).strip()
            if not group or source not in SOURCE_NAMES:
                raise ValueError(f"invalid policy-map row in {resolved}")
            if group in selected and selected[group] != source:
                raise ValueError(f"conflicting selected source for group {group}")
            selected[group] = source
    if not selected:
        raise RuntimeError("policy maps contain no selections")
    return selected, artifacts


def load_verified_controls(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = _read_csv(path)
    required = {
        "control_id",
        "control_type",
        "reaction_smiles",
        "experimental_provenance",
        "verification_status",
        "reaction_family",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            "control CSV must contain: " + ", ".join(sorted(required))
        )
    controls: List[Dict[str, Any]] = []
    for row in rows:
        control_type = row["control_type"].strip()
        if control_type not in {"positive_control", "obvious_negative_control"}:
            raise ValueError(f"invalid control_type: {control_type}")
        if row["verification_status"].strip() != "INDEPENDENTLY_VERIFIED":
            raise ValueError(
                f"control is not independently verified: {row['control_id']}"
            )
        if not row["experimental_provenance"].strip():
            raise ValueError(f"control lacks provenance: {row['control_id']}")
        _parse_reaction(row["reaction_smiles"])
        controls.append(dict(row))
    artifact = {
        "path": str(path.resolve()),
        "sha256": _sha256(path.resolve()),
        "n_rows": len(rows),
        "control_counts": dict(Counter(row["control_type"] for row in rows)),
    }
    return controls, artifact


def _expert_item(
    *,
    source_record: Mapping[str, Any],
    stratum: str,
    reaction_smiles: str,
    source: str,
    candidate_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    reactants, conditions, candidate_product = _parse_reaction(reaction_smiles)
    metadata = candidate_metadata or {}
    return {
        "blinded_id": "",
        "stratum": stratum,
        "group": str(source_record.get("group", source_record.get("control_id", ""))),
        "scenario": str(source_record.get("scenario", "external_control")),
        "reaction_family": str(source_record.get("family", source_record.get("reaction_family", ""))),
        "reactants": reactants,
        "conditions": conditions,
        "candidate_product": candidate_product,
        "candidate_reaction": reaction_smiles,
        "source": source,
        "false_negative_risk": metadata.get("false_negative_risk"),
        "positive_similarity": metadata.get("positive_similarity"),
        "experimental_provenance": source_record.get("experimental_provenance", ""),
        "verification_status": source_record.get("verification_status", ""),
    }


def build_pilot_items(
    entries: Sequence[Mapping[str, Any]],
    policy: Mapping[str, str],
    controls: Sequence[Mapping[str, Any]],
    *,
    per_stratum: int = DEFAULT_PER_STRATUM,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")
    rng = random.Random(seed)
    control_by_type: Dict[str, List[Mapping[str, Any]]] = {
        "positive_control": [],
        "obvious_negative_control": [],
    }
    for control in controls:
        control_by_type[str(control["control_type"])].append(control)
    for control_type, pool in control_by_type.items():
        if len(pool) < per_stratum:
            raise RuntimeError(
                f"{control_type} requires {per_stratum} independently verified "
                f"items; found {len(pool)}"
            )

    eligible = [entry for entry in entries if str(entry["group"]) in policy]
    required_source_parents = per_stratum * (len(STRATA) - 2)
    if len(eligible) < required_source_parents:
        raise RuntimeError(
            f"pilot requires {required_source_parents} unique complete-case "
            f"parents for source strata; found {len(eligible)}"
        )
    rng.shuffle(eligible)

    items: List[Dict[str, Any]] = []
    for control_type in ("positive_control", "obvious_negative_control"):
        pool = list(control_by_type[control_type])
        rng.shuffle(pool)
        for control in pool[:per_stratum]:
            items.append(
                _expert_item(
                    source_record=control,
                    stratum=control_type,
                    reaction_smiles=str(control["reaction_smiles"]),
                    source=control_type,
                )
            )

    cursor = 0
    source_strata = list(SOURCE_STRATA)
    for stratum in source_strata:
        source = SOURCE_STRATA[stratum]
        for entry in eligible[cursor : cursor + per_stratum]:
            candidate = entry["candidates"][source]
            items.append(
                _expert_item(
                    source_record=entry,
                    stratum=stratum,
                    reaction_smiles=str(candidate["negative_reaction"]),
                    source=source,
                    candidate_metadata=candidate,
                )
            )
        cursor += per_stratum

    for local_index, entry in enumerate(
        eligible[cursor : cursor + per_stratum]
    ):
        source = SOURCE_NAMES[local_index % len(SOURCE_NAMES)]
        candidate = entry["candidates"][source]
        items.append(
            _expert_item(
                source_record=entry,
                stratum="uniform_union",
                reaction_smiles=str(candidate["negative_reaction"]),
                source=source,
                candidate_metadata=candidate,
            )
        )
    cursor += per_stratum

    for entry in eligible[cursor : cursor + per_stratum]:
        source = policy[str(entry["group"])]
        candidate = entry["candidates"][source]
        items.append(
            _expert_item(
                source_record=entry,
                stratum="learned_source_gate",
                reaction_smiles=str(candidate["negative_reaction"]),
                source=source,
                candidate_metadata=candidate,
            )
        )

    counts = Counter(item["stratum"] for item in items)
    expected_total = per_stratum * len(STRATA)
    if len(items) != expected_total:
        raise RuntimeError(
            f"pilot must contain exactly {expected_total} items; found {len(items)}"
        )
    if any(counts[stratum] != per_stratum for stratum in STRATA):
        raise RuntimeError(f"pilot stratum contract failed: {dict(counts)}")
    groups = [item["group"] for item in items]
    if len(groups) != len(set(groups)):
        raise RuntimeError("pilot reuses a parent/control across strata")

    rng.shuffle(items)
    for index, item in enumerate(items, start=1):
        item["blinded_id"] = f"G7V2-{index:04d}"
    return items, {
        "n_total": len(items),
        "stratum_counts": dict(counts),
        "unique_parent_or_control_count": len(set(groups)),
        "seed": seed,
        "per_stratum": per_stratum,
    }


def write_pilot_package(
    items: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    n_reviewers: int,
    seed: int,
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    if n_reviewers < MIN_REVIEWERS:
        raise ValueError(f"Phase F pilot requires at least {MIN_REVIEWERS} reviewers")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    forms_dir = output_dir / "blinded_forms"
    forms_dir.mkdir(exist_ok=True)

    public_fields = [
        "blinded_id",
        "reactants",
        "conditions",
        "candidate_product",
        "reaction_family",
        *SCORING_DIMENSIONS,
        "reason_code",
        "notes",
    ]
    public_rows = []
    for item in items:
        row = {
            key: item[key]
            for key in (
                "blinded_id",
                "reactants",
                "conditions",
                "candidate_product",
                "reaction_family",
            )
        }
        row.update({dimension: "" for dimension in SCORING_DIMENSIONS})
        row["reason_code"] = ""
        row["notes"] = ""
        public_rows.append(row)
    _write_csv(output_dir / "samples_blinded.csv", public_rows, public_fields)

    form_artifacts = []
    for reviewer_index in range(1, n_reviewers + 1):
        rows = list(public_rows)
        random.Random(seed + reviewer_index).shuffle(rows)
        path = forms_dir / f"reviewer_{reviewer_index}.csv"
        _write_csv(path, rows, public_fields)
        form_artifacts.append(
            {"path": str(path), "sha256": _sha256(path), "n_rows": len(rows)}
        )

    key_path = output_dir / "sampling_key_unblinded.json"
    _write_json(
        key_path,
        {
            "schema": SCHEMA,
            "items": list(items),
            "reason_codes": REASON_CODES,
        },
    )
    instructions_path = output_dir / "reviewer_instructions.md"
    instructions_path.write_text(
        "# PC-CNG blinded expert pilot v2\n\n"
        "Score each dimension from 1 (strongly implausible/invalid) to 5 "
        "(strongly plausible/valid). Use one reason code when possible. "
        "Candidate source, model scores, risk estimates and observed labels "
        "are intentionally hidden. Reviewers must work independently and "
        "must not attempt to identify candidate sources.\n\n"
        "Reason codes: " + ", ".join(REASON_CODES) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": SCHEMA,
        "status": "PILOT_MATERIALS_READY_EXPERT_RESPONSES_PENDING",
        "created_at_utc": _utc_now(),
        "n_items": len(items),
        "n_reviewers": n_reviewers,
        "scoring_dimensions": SCORING_DIMENSIONS,
        "reason_codes": REASON_CODES,
        "blinding": {
            "source_hidden": True,
            "risk_hidden": True,
            "observed_label_hidden": True,
            "independent_random_order_per_reviewer": True,
        },
        "provenance": dict(provenance),
        "artifacts": {
            "sampling_key": {
                "path": str(key_path),
                "sha256": _sha256(key_path),
            },
            "reviewer_instructions": {
                "path": str(instructions_path),
                "sha256": _sha256(instructions_path),
            },
            "forms": form_artifacts,
        },
        "scientific_status": (
            "No expert evidence exists until independently completed forms "
            "are returned and analyzed."
        ),
    }
    manifest_path = output_dir / "pilot_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cache", action="append", type=Path, required=True)
    parser.add_argument("--policy-map", action="append", type=Path, required=True)
    parser.add_argument("--verified-controls", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-reviewers", type=int, default=3)
    parser.add_argument("--per-stratum", type=int, default=DEFAULT_PER_STRATUM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repo-root", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    entries, cache_artifacts = load_candidate_caches(args.candidate_cache)
    policy, policy_artifacts = load_policy_maps(args.policy_map)
    controls, control_artifact = load_verified_controls(args.verified_controls)
    items, sampling = build_pilot_items(
        entries,
        policy,
        controls,
        per_stratum=args.per_stratum,
        seed=args.seed,
    )
    manifest = write_pilot_package(
        items,
        output_dir=args.output_dir,
        n_reviewers=args.n_reviewers,
        seed=args.seed,
        provenance={
            "git_commit_sha": _git_sha(args.repo_root),
            "candidate_caches": cache_artifacts,
            "policy_maps": policy_artifacts,
            "verified_controls": control_artifact,
            "sampling": sampling,
        },
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
