import json
from pathlib import Path

import pytest

from pc_cng.phase_e_sealed_contract import (
    CANDIDATE_STATUS,
    SEALED_STATUS,
    build_forbidden_index,
    freeze_candidate,
    register_candidate,
    verify_sealed_manifest,
)


def _candidate(tmp_path: Path):
    known = tmp_path / "known.csv"
    known.write_text("x\nold\n")
    index = build_forbidden_index([known], dataset_ids=["old-development"])
    index_path = tmp_path / "forbidden.json"
    index_path.write_text(json.dumps(index))
    candidate = register_candidate(
        dataset_id="jacs2025_pdcn_external",
        title="Independent pharmaceutically relevant Pd C-N HTE",
        source_url="https://figshare.com/articles/dataset/28215923",
        doi="10.1021/jacs.5c00933",
        license_name="MIT",
        reaction_family="Buchwald-Hartwig C-N coupling",
        provider_checksums=[
            {"algorithm": "md5", "digest": "abc", "file_name": "data.csv"}
        ],
        upstream_datasets=["Ha et al. 2025 dedicated HTE"],
        rationale="Independent external HTE candidate",
        forbidden_index=index_path,
    )
    return candidate, index_path


def _freeze(tmp_path: Path):
    candidate, index_path = _candidate(tmp_path)
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    spec = tmp_path / "analysis.md"
    spec.write_text("primary endpoint: source-macro AUPRC\n")
    pool = tmp_path / "pool.json"
    pool.write_text('{"label_free": true}\n')
    receipt = tmp_path / "label_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "dataset_id": "jacs2025_pdcn_external",
                "sealed_label_sha256": "f" * 64,
                "label_schema_sha256": "e" * 64,
                "n_rows": 4204,
                "custodian": "independent-data-custodian",
                "labels_never_exposed_to_development": True,
                "created_at_utc": "2026-07-29T00:00:00+00:00",
            }
        )
    )
    manifest = freeze_candidate(
        candidate,
        label_receipt_path=receipt,
        model_artifact=model,
        analysis_spec=spec,
        evaluation_pool_files=[pool],
        model_git_commit="a" * 40,
        forbidden_index=index_path,
    )
    manifest_path = tmp_path / "sealed_test_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, model, pool


def test_metadata_registration_does_not_claim_labels():
    candidate = register_candidate(
        dataset_id="external_hte",
        title="External HTE",
        source_url="https://example.org/data",
        doi="10.0000/example",
        license_name="MIT",
        reaction_family="C-N coupling",
        provider_checksums=[],
        upstream_datasets=[],
        rationale="test",
    )
    assert candidate["status"] == CANDIDATE_STATUS
    assert candidate["labels_downloaded_or_inspected_by_development_team"] is False
    assert candidate["contamination_audit"]["status"] == "PASS"


@pytest.mark.parametrize(
    "dataset_id,source_url",
    [
        ("hitea", "https://example.org/new"),
        ("new-id", "/repo/external/HiTEA/data.csv"),
        ("regiosqm20", "https://example.org/new"),
        ("phase4_fixed_testset", "https://example.org/new"),
    ],
)
def test_known_development_data_is_rejected(dataset_id, source_url):
    candidate = register_candidate(
        dataset_id=dataset_id,
        title="Candidate",
        source_url=source_url,
        doi="10.0000/example",
        license_name="MIT",
        reaction_family="test",
        provider_checksums=[],
        upstream_datasets=[],
        rationale="test",
    )
    assert candidate["contamination_audit"]["status"] == "FAIL"


def test_freeze_requires_independent_label_receipt(tmp_path):
    candidate, _ = _candidate(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"dataset_id": candidate["dataset"]["dataset_id"]}))
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    spec = tmp_path / "spec.md"
    spec.write_text("frozen")
    pool = tmp_path / "pool.json"
    pool.write_text("{}")
    with pytest.raises(ValueError, match="label receipt missing fields"):
        freeze_candidate(
            candidate,
            label_receipt_path=receipt,
            model_artifact=model,
            analysis_spec=spec,
            evaluation_pool_files=[pool],
            model_git_commit="a" * 40,
        )


def test_frozen_contract_verifies_without_opening_labels(tmp_path):
    manifest_path, _, _ = _freeze(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == SEALED_STATUS
    assert "sealed_label_path" not in manifest["label_receipt"]
    assert manifest["label_receipt"]["sealed_label_sha256"] == "f" * 64
    verification = verify_sealed_manifest(
        manifest_path,
        expected_pool_dir=tmp_path,
        expected_git_commit="a" * 40,
    )
    assert verification["verified"] is True
    assert verification["failures"] == []


def test_tampering_fails_closed(tmp_path):
    manifest_path, model, _ = _freeze(tmp_path)
    model.write_bytes(b"tampered")
    verification = verify_sealed_manifest(manifest_path)
    assert verification["verified"] is False
    assert any("artifact verification failed" in item for item in verification["failures"])


def test_pool_must_remain_inside_expected_directory(tmp_path):
    manifest_path, _, _ = _freeze(tmp_path)
    outside = tmp_path / "other"
    outside.mkdir()
    verification = verify_sealed_manifest(
        manifest_path,
        expected_pool_dir=outside,
    )
    assert verification["verified"] is False
    assert any("outside expected pool" in item for item in verification["failures"])
