"""Fail-closed contracts for Phase-E external blind evaluation.

The contract deliberately separates three moments:

1. metadata-only registration of a candidate external dataset;
2. model/analysis freeze while labels remain under independent custody;
3. verification immediately before a one-shot formal evaluation.

This module never opens the sealed label artifact.  It accepts only a
custodian-produced receipt containing its digest and schema metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


CONTRACT_VERSION = "phase-e-v1"
CANDIDATE_STATUS = "CANDIDATE_METADATA_ONLY"
SEALED_STATUS = "SEALED_UNUSED_FOR_METHOD_DESIGN"

_KNOWN_DEVELOPMENT_IDS = {
    "hitea",
    "nicolit",
    "ni_coupling",
    "phase4_fixed_testset",
    "phase4_v41",
    "regiosqm20",
    "uspto",
}
_DEVELOPMENT_PATH_MARKERS = (
    "external/hitea",
    "ni_coupling_supplement",
    "phase4_fixed_testset",
    "phase4_v41",
    "regiosqm20",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _git_sha(repo_root: Optional[Path] = None) -> str:
    command = ["git"]
    if repo_root is not None:
        command.extend(["-C", str(repo_root)])
    command.extend(["rev-parse", "HEAD"])
    return subprocess.check_output(command, text=True).strip()


def _load_forbidden_index(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {
            "contract_version": CONTRACT_VERSION,
            "artifacts": [],
            "index_sha256": None,
        }
    index = _read_json(path)
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("forbidden artifact index must contain an artifacts list")
    expected = index.get("index_sha256")
    unsigned = dict(index)
    unsigned.pop("index_sha256", None)
    if expected != _canonical_digest(unsigned):
        raise ValueError("forbidden artifact index digest mismatch")
    return index


def _contamination_reasons(
    *,
    dataset_id: str,
    source_url: str,
    provider_checksums: Sequence[Mapping[str, Any]],
    upstream_datasets: Sequence[str],
    forbidden_index: Mapping[str, Any],
) -> List[str]:
    reasons: List[str] = []
    normalized_id = dataset_id.strip().lower()
    if normalized_id in _KNOWN_DEVELOPMENT_IDS:
        reasons.append(f"dataset id is already development data: {dataset_id}")

    searchable = " ".join(
        [dataset_id, source_url, *upstream_datasets]
    ).replace("\\", "/").lower()
    for marker in _DEVELOPMENT_PATH_MARKERS:
        if marker in searchable:
            reasons.append(f"development path/source marker detected: {marker}")

    forbidden_hashes = {
        str(item.get("sha256", "")).lower()
        for item in forbidden_index.get("artifacts", [])
        if item.get("sha256")
    }
    for item in provider_checksums:
        algorithm = str(item.get("algorithm", "")).lower()
        digest = str(item.get("digest", "")).lower()
        if algorithm == "sha256" and digest in forbidden_hashes:
            reasons.append(f"provider artifact hash was used in development: {digest}")
    return sorted(set(reasons))


def build_forbidden_index(
    artifacts: Sequence[Path],
    *,
    dataset_ids: Sequence[str],
) -> Dict[str, Any]:
    if dataset_ids and len(dataset_ids) != len(artifacts):
        raise ValueError("--dataset-id must be absent or match --artifact count")
    entries: List[Dict[str, Any]] = []
    for index, artifact in enumerate(artifacts):
        resolved = artifact.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        entries.append(
            {
                "dataset_id": (
                    dataset_ids[index] if dataset_ids else resolved.stem
                ),
                "path": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
        )
    unsigned = {
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": _utc_now(),
        "artifacts": entries,
    }
    return {**unsigned, "index_sha256": _canonical_digest(unsigned)}


def register_candidate(
    *,
    dataset_id: str,
    title: str,
    source_url: str,
    doi: str,
    license_name: str,
    reaction_family: str,
    provider_checksums: Sequence[Mapping[str, Any]],
    upstream_datasets: Sequence[str],
    rationale: str,
    forbidden_index: Optional[Path] = None,
) -> Dict[str, Any]:
    index = _load_forbidden_index(forbidden_index)
    reasons = _contamination_reasons(
        dataset_id=dataset_id,
        source_url=source_url,
        provider_checksums=provider_checksums,
        upstream_datasets=upstream_datasets,
        forbidden_index=index,
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "status": CANDIDATE_STATUS,
        "registered_at_utc": _utc_now(),
        "dataset": {
            "dataset_id": dataset_id,
            "title": title,
            "source_url": source_url,
            "doi": doi,
            "license": license_name,
            "reaction_family": reaction_family,
            "provider_checksums": list(provider_checksums),
            "upstream_datasets": list(upstream_datasets),
        },
        "rationale": rationale,
        "labels_downloaded_or_inspected_by_development_team": False,
        "contamination_audit": {
            "status": "PASS" if not reasons else "FAIL",
            "reasons": reasons,
            "forbidden_index_sha256": index.get("index_sha256"),
        },
    }


def _artifact_record(path: Path, *, role: str) -> Dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "role": role,
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def freeze_candidate(
    candidate: Mapping[str, Any],
    *,
    label_receipt_path: Path,
    model_artifact: Path,
    analysis_spec: Path,
    evaluation_pool_files: Sequence[Path],
    model_git_commit: str,
    forbidden_index: Optional[Path] = None,
) -> Dict[str, Any]:
    if candidate.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("candidate uses an unsupported contract version")
    if candidate.get("status") != CANDIDATE_STATUS:
        raise ValueError("only metadata-only candidates can be frozen")
    if candidate.get("labels_downloaded_or_inspected_by_development_team") is not False:
        raise ValueError("candidate does not assert label-free development")
    if candidate.get("contamination_audit", {}).get("status") != "PASS":
        raise ValueError("candidate contamination audit is not PASS")

    receipt = _read_json(label_receipt_path)
    required_receipt = {
        "dataset_id",
        "sealed_label_sha256",
        "label_schema_sha256",
        "n_rows",
        "custodian",
        "labels_never_exposed_to_development",
        "created_at_utc",
    }
    missing = sorted(required_receipt - set(receipt))
    if missing:
        raise ValueError(f"label receipt missing fields: {missing}")
    if receipt["dataset_id"] != candidate["dataset"]["dataset_id"]:
        raise ValueError("label receipt dataset_id does not match candidate")
    if receipt["labels_never_exposed_to_development"] is not True:
        raise ValueError("label receipt does not prove label-free development")
    if int(receipt["n_rows"]) <= 0:
        raise ValueError("label receipt n_rows must be positive")

    index = _load_forbidden_index(forbidden_index)
    reasons = _contamination_reasons(
        dataset_id=str(candidate["dataset"]["dataset_id"]),
        source_url=str(candidate["dataset"]["source_url"]),
        provider_checksums=candidate["dataset"].get("provider_checksums", []),
        upstream_datasets=candidate["dataset"].get("upstream_datasets", []),
        forbidden_index=index,
    )
    pool_artifacts = [
        _artifact_record(path, role="label_free_evaluation_pool")
        for path in evaluation_pool_files
    ]
    forbidden_hashes = {
        str(item.get("sha256", "")).lower()
        for item in index.get("artifacts", [])
    }
    for artifact in pool_artifacts:
        if artifact["sha256"].lower() in forbidden_hashes:
            reasons.append(
                "evaluation pool artifact was already used in development: "
                f"{artifact['path']}"
            )
    if reasons:
        raise ValueError("contamination audit failed: " + "; ".join(sorted(set(reasons))))

    model_record = _artifact_record(model_artifact, role="frozen_model")
    analysis_record = _artifact_record(analysis_spec, role="frozen_analysis_spec")
    label_receipt_record = _artifact_record(
        label_receipt_path,
        role="independent_label_receipt",
    )
    unsigned = {
        "contract_version": CONTRACT_VERSION,
        "status": SEALED_STATUS,
        "frozen_at_utc": _utc_now(),
        "dataset": dict(candidate["dataset"]),
        "labels_unseen_before_model_freeze": True,
        "model_and_analysis_frozen": True,
        "model_freeze": {
            **model_record,
            "git_commit": model_git_commit,
        },
        "analysis_freeze": analysis_record,
        "evaluation_pool_artifacts": pool_artifacts,
        "label_receipt": {
            **receipt,
            "receipt_artifact": label_receipt_record,
        },
        "contamination_audit": {
            "status": "PASS",
            "reasons": [],
            "forbidden_index_sha256": index.get("index_sha256"),
        },
        "formal_evaluation": {
            "run_count": 0,
            "one_shot_required": True,
            "results_must_be_fully_disclosed": True,
            "primary_endpoint_change_after_unseal_forbidden": True,
        },
    }
    return {**unsigned, "manifest_sha256": _canonical_digest(unsigned)}


def verify_sealed_manifest(
    manifest_path: Path,
    *,
    expected_pool_dir: Optional[Path] = None,
    expected_git_commit: Optional[str] = None,
) -> Dict[str, Any]:
    failures: List[str] = []
    manifest = _read_json(manifest_path)
    if manifest.get("contract_version") != CONTRACT_VERSION:
        failures.append("unsupported contract_version")
    if manifest.get("status") != SEALED_STATUS:
        failures.append("status is not SEALED_UNUSED_FOR_METHOD_DESIGN")
    if manifest.get("labels_unseen_before_model_freeze") is not True:
        failures.append("labels were not sealed through model freeze")
    if manifest.get("model_and_analysis_frozen") is not True:
        failures.append("model/analysis freeze is not asserted")
    if manifest.get("contamination_audit", {}).get("status") != "PASS":
        failures.append("contamination audit is not PASS")
    if manifest.get("formal_evaluation", {}).get("run_count") != 0:
        failures.append("formal evaluation was already consumed")

    unsigned = dict(manifest)
    expected_manifest_digest = unsigned.pop("manifest_sha256", None)
    if expected_manifest_digest != _canonical_digest(unsigned):
        failures.append("manifest digest mismatch")

    artifacts: List[Mapping[str, Any]] = []
    for key in ("model_freeze", "analysis_freeze"):
        value = manifest.get(key)
        if isinstance(value, dict):
            artifacts.append(value)
        else:
            failures.append(f"missing {key}")
    pool_artifacts = manifest.get("evaluation_pool_artifacts")
    if not isinstance(pool_artifacts, list) or not pool_artifacts:
        failures.append("evaluation pool artifact list is empty")
        pool_artifacts = []
    artifacts.extend(pool_artifacts)
    receipt_artifact = manifest.get("label_receipt", {}).get("receipt_artifact")
    if isinstance(receipt_artifact, dict):
        artifacts.append(receipt_artifact)
    else:
        failures.append("missing independent label receipt artifact")

    checked_artifacts: List[Dict[str, Any]] = []
    for artifact in artifacts:
        path = Path(str(artifact.get("path", "")))
        exists = path.is_file()
        digest_ok = exists and _sha256(path) == artifact.get("sha256")
        size_ok = exists and path.stat().st_size == artifact.get("size_bytes")
        checked_artifacts.append(
            {
                "role": artifact.get("role"),
                "path": str(path),
                "exists": exists,
                "sha256_matches": digest_ok,
                "size_matches": size_ok,
            }
        )
        if not exists or not digest_ok or not size_ok:
            failures.append(f"artifact verification failed: {path}")

    if expected_pool_dir is not None:
        expected = expected_pool_dir.resolve()
        for artifact in manifest.get("evaluation_pool_artifacts", []):
            path = Path(str(artifact.get("path", ""))).resolve()
            try:
                path.relative_to(expected)
            except ValueError:
                failures.append(f"evaluation artifact outside expected pool: {path}")

    frozen_commit = manifest.get("model_freeze", {}).get("git_commit")
    if expected_git_commit is not None and frozen_commit != expected_git_commit:
        failures.append("frozen model git commit does not match expected commit")

    receipt = manifest.get("label_receipt", {})
    if receipt.get("labels_never_exposed_to_development") is not True:
        failures.append("label custodian receipt is not fail-closed")
    if not receipt.get("sealed_label_sha256"):
        failures.append("sealed label digest is missing")

    return {
        "verified": not failures,
        "failures": failures,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "dataset_id": manifest.get("dataset", {}).get("dataset_id"),
        "checked_artifacts": checked_artifacts,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parse_checksum(values: Iterable[str]) -> List[Dict[str, str]]:
    checksums: List[Dict[str, str]] = []
    for value in values:
        parts = value.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "provider checksum must use algorithm:digest:file_name"
            )
        checksums.append(
            {"algorithm": parts[0], "digest": parts[1], "file_name": parts[2]}
        )
    return checksums


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index-known")
    index_parser.add_argument("--artifact", action="append", type=Path, required=True)
    index_parser.add_argument("--dataset-id", action="append", default=[])
    index_parser.add_argument("--output", type=Path, required=True)

    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--dataset-id", required=True)
    register_parser.add_argument("--title", required=True)
    register_parser.add_argument("--source-url", required=True)
    register_parser.add_argument("--doi", required=True)
    register_parser.add_argument("--license", dest="license_name", required=True)
    register_parser.add_argument("--reaction-family", required=True)
    register_parser.add_argument("--provider-checksum", action="append", default=[])
    register_parser.add_argument("--upstream-dataset", action="append", default=[])
    register_parser.add_argument("--rationale", required=True)
    register_parser.add_argument("--forbidden-index", type=Path)
    register_parser.add_argument("--output", type=Path, required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--candidate", type=Path, required=True)
    freeze_parser.add_argument("--label-receipt", type=Path, required=True)
    freeze_parser.add_argument("--model-artifact", type=Path, required=True)
    freeze_parser.add_argument("--analysis-spec", type=Path, required=True)
    freeze_parser.add_argument(
        "--evaluation-pool-file",
        action="append",
        type=Path,
        required=True,
    )
    freeze_parser.add_argument("--model-git-commit")
    freeze_parser.add_argument("--repo-root", type=Path)
    freeze_parser.add_argument("--forbidden-index", type=Path)
    freeze_parser.add_argument("--output", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--expected-pool-dir", type=Path)
    verify_parser.add_argument("--expected-git-commit")
    verify_parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "index-known":
        payload = build_forbidden_index(
            args.artifact,
            dataset_ids=args.dataset_id,
        )
        _write_json(args.output, payload)
        return 0
    if args.command == "register":
        payload = register_candidate(
            dataset_id=args.dataset_id,
            title=args.title,
            source_url=args.source_url,
            doi=args.doi,
            license_name=args.license_name,
            reaction_family=args.reaction_family,
            provider_checksums=_parse_checksum(args.provider_checksum),
            upstream_datasets=args.upstream_dataset,
            rationale=args.rationale,
            forbidden_index=args.forbidden_index,
        )
        _write_json(args.output, payload)
        return 0 if payload["contamination_audit"]["status"] == "PASS" else 2
    if args.command == "freeze":
        payload = freeze_candidate(
            _read_json(args.candidate),
            label_receipt_path=args.label_receipt,
            model_artifact=args.model_artifact,
            analysis_spec=args.analysis_spec,
            evaluation_pool_files=args.evaluation_pool_file,
            model_git_commit=(
                args.model_git_commit or _git_sha(args.repo_root)
            ),
            forbidden_index=args.forbidden_index,
        )
        _write_json(args.output, payload)
        return 0
    verification = verify_sealed_manifest(
        args.manifest,
        expected_pool_dir=args.expected_pool_dir,
        expected_git_commit=args.expected_git_commit,
    )
    if args.output:
        _write_json(args.output, verification)
    else:
        print(json.dumps(verification, indent=2))
    return 0 if verification["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
