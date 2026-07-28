"""Independent artifact-contract verifier for a formal G8-C run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch

from .g8c_action_schema import EditType


REQUIRED_FILES = (
    "formal_validation.json",
    "go_no_go.json",
    "model_checkpoint.pt",
    "train_log.json",
    "run_manifest.json",
    "environment.json",
    "input_hashes.json",
    "commands.log",
)
FORBIDDEN_FORMAL_FILES = (
    "comparison_results.csv",
    "pareto_frontier.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_result(result_dir: Path) -> Dict[str, Any]:
    errors = []
    for name in REQUIRED_FILES:
        if not (result_dir / name).is_file():
            errors.append(f"missing required artifact: {name}")
    for name in FORBIDDEN_FORMAL_FILES:
        if (result_dir / name).exists():
            errors.append(f"legacy formal evaluation artifact present: {name}")
    raw_dir = result_dir / "raw_predictions"
    if raw_dir.exists() and any(raw_dir.iterdir()):
        errors.append("legacy raw_predictions are non-empty")
    if errors:
        return {"verified": False, "errors": errors}

    result = json.loads((result_dir / "formal_validation.json").read_text())
    gate = json.loads((result_dir / "go_no_go.json").read_text())
    manifest = json.loads((result_dir / "run_manifest.json").read_text())
    thresholds = result["verdict"]["thresholds"]
    edit = result["edit_validation"]
    candidate = result["candidate_validation"]
    risk = result["risk_validation"]
    reward = result["reward_hacking_validation"]
    availability = result["risk_source_availability"]

    reconstructed_checks = {
        "edit_locus_accuracy": (
            edit["edit_locus_accuracy"]["value"]
            >= thresholds["edit_locus_accuracy_min"]
        ),
        "edit_type_accuracy": (
            edit["edit_type_accuracy"]["value"]
            >= thresholds["edit_type_accuracy_min"]
        ),
        "valid_edit_rate": (
            candidate["valid_edit_rate"]["value"]
            >= thresholds["valid_edit_rate_min"]
        ),
        "candidate_coverage": (
            candidate["candidate_coverage"]["value"]
            >= thresholds["candidate_coverage_min"]
        ),
        "fnr_calibration": (
            risk["ece_10_bin"] <= thresholds["fnr_ece_max"]
        ),
        "reward_log_ratio_bounded": (
            reward["max_abs_log_ratio"]
            <= thresholds["reward_max_abs_log_ratio_max"]
        ),
        "reward_policy_not_collapsed": (
            reward["mean_action_type_entropy"]
            >= thresholds["reward_action_type_entropy_min"]
        ),
        "reference_frozen": (
            reward["reference_hash_before"]
            == reward["reference_hash_after"]
            and bool(reward["reference_frozen"])
        ),
        "known_positive_collision_supervision": (
            availability.get("known_positive_collision", 0) > 0
        ),
        "observed_competing_product_supervision": (
            availability.get("observed_competing_product", 0) > 0
        ),
        "heldout_hte_outcome_supervision": (
            availability.get("heldout_hte_outcome", 0) > 0
        ),
    }
    if reconstructed_checks != result["verdict"]["checks"]:
        errors.append("stored verdict checks do not match independent rebuild")
    core_pass = all(reconstructed_checks.values())
    expert_available = availability.get("expert_label", 0) > 0
    if core_pass and expert_available:
        reconstructed_status = "FORMAL_SOURCE_EXPERT_PASS"
    elif core_pass:
        reconstructed_status = (
            "FORMAL_SOURCE_EXPERT_PARTIAL_EXPERT_LABELS_PENDING"
        )
    else:
        reconstructed_status = "FORMAL_SOURCE_EXPERT_NO_GO"
    if reconstructed_status != result["status"]:
        errors.append("formal status does not match independent rebuild")
    if gate.get("status") != reconstructed_status:
        errors.append("go_no_go status differs from formal validation status")
    if not result.get("sealed_test_untouched"):
        errors.append("sealed-test contract is not asserted")
    if result.get("tiny_self_built_mlp_evaluation") != "DISABLED":
        errors.append("legacy tiny MLP is not disabled")
    if manifest.get("formal_run") is not True:
        errors.append("run manifest is not formal")
    if manifest.get("formal_partition") != "v2_unseen_train_holdout":
        errors.append("run did not use the frozen v2 unseen holdout")

    stored_hashes = json.loads((result_dir / "input_hashes.json").read_text())
    input_hash_verification = {}
    for raw_path, expected in stored_hashes.items():
        path = Path(raw_path)
        if path.exists():
            actual = _sha256(path)
            input_hash_verification[raw_path] = actual == expected
            if actual != expected:
                errors.append(f"input hash mismatch: {raw_path}")
        else:
            input_hash_verification[raw_path] = None

    checkpoint = torch.load(
        result_dir / "model_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    expected_schema = {
        name: int(member) for name, member in EditType.__members__.items()
    }
    if checkpoint.get("action_schema") != expected_schema:
        errors.append("checkpoint action schema mismatch")
    if checkpoint.get("formal_validation_status") != reconstructed_status:
        errors.append("checkpoint status mismatch")

    return {
        "verified": not errors,
        "errors": errors,
        "reconstructed_status": reconstructed_status,
        "reconstructed_checks": reconstructed_checks,
        "input_hash_verification": input_hash_verification,
        "artifact_hashes": {
            name: _sha256(result_dir / name) for name in REQUIRED_FILES
        },
        "claim_boundary": (
            "source-expert credibility only; no source-superiority claim"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = verify_result(args.result_dir)
    output = args.output or args.result_dir / "independent_verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
