"""Tests for P4-G0 Claim Registry and Audit module.

Run with::

    python3 -m pytest chem_negative_sampling/tests/test_claim_registry.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure the pc_cng package is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "chem_negative_sampling") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "chem_negative_sampling"))

from pc_cng.audit.run_claim_audit import (
    ALLOWED_STATUSES,
    Claim,
    build_claim_registry,
    verify_claims,
    run_claim_audit,
    write_outputs,
)
from pc_cng.audit import run_claim_audit as audit_module


# ---------------------------------------------------------------------------
# Claim dataclass tests
# ---------------------------------------------------------------------------

class TestClaimDataclass:
    """Tests for the Claim dataclass."""

    def test_claim_creation(self):
        c = Claim(
            claim_id="TEST-01",
            claim_text="Test claim",
            claim_location="test",
            metric_name="test_metric",
            reported_value=0.5,
        )
        assert c.claim_id == "TEST-01"
        assert c.status == "UNVERIFIED"
        assert c.recomputed_value is None

    def test_claim_to_dict(self):
        c = Claim(
            claim_id="TEST-02",
            claim_text="Test",
            claim_location="test",
            metric_name="m",
            reported_value=1.0,
            status="VERIFIED",
        )
        d = c.to_dict()
        assert d["claim_id"] == "TEST-02"
        assert d["status"] == "VERIFIED"
        assert "claim_text" in d
        assert "metric_name" in d

    def test_all_required_fields_present(self):
        """Every claim must have all fields from the P4-G0 spec."""
        c = Claim(
            claim_id="TEST-03",
            claim_text="Test",
            claim_location="test",
            metric_name="m",
            reported_value=1.0,
        )
        d = c.to_dict()
        required = [
            "claim_id", "claim_text", "claim_location", "metric_name",
            "reported_value", "recomputed_value", "artifact_path",
            "implementation_path", "checkpoint_path", "split_manifest",
            "status", "reason", "required_action",
        ]
        for field in required:
            assert field in d, f"Missing required field: {field}"


# ---------------------------------------------------------------------------
# Claim registry tests
# ---------------------------------------------------------------------------

class TestClaimRegistry:
    """Tests for the claim registry builder."""

    def test_registry_not_empty(self):
        claims = build_claim_registry()
        assert len(claims) > 0, "Claim registry must not be empty"

    def test_all_claims_have_ids(self):
        claims = build_claim_registry()
        for c in claims:
            assert c.claim_id, f"Claim missing claim_id: {c}"

    def test_claim_ids_unique(self):
        claims = build_claim_registry()
        ids = [c.claim_id for c in claims]
        assert len(ids) == len(set(ids)), "Duplicate claim_ids found"

    def test_all_claims_have_metric_name(self):
        claims = build_claim_registry()
        for c in claims:
            assert c.metric_name, f"Claim {c.claim_id} missing metric_name"

    def test_all_claims_have_reported_value(self):
        claims = build_claim_registry()
        for c in claims:
            assert c.reported_value is not None, f"Claim {c.claim_id} missing reported_value"

    def test_registry_covers_headline_claims(self):
        """Registry must cover the 5 abstract headline claims."""
        claims = build_claim_registry()
        ids = {c.claim_id for c in claims}
        # Must include at least these critical claims
        assert "ABS-01" in ids, "Missing ABS-01 (MRR range)"
        assert "ABS-02" in ids, "Missing ABS-02 (delta vs GNN)"
        assert "ABS-03" in ids, "Missing ABS-03 (delta vs Chemformer)"
        assert "ABS-04" in ids, "Missing ABS-04 (LLM-judge kappa)"
        assert "ABS-05" in ids, "Missing ABS-05 (nine-dim score)"

    def test_registry_covers_architecture_claims(self):
        """Registry must cover architecture/method claims."""
        claims = build_claim_registry()
        ids = {c.claim_id for c in claims}
        assert "METH-01" in ids, "Missing METH-01 (backbone params)"
        assert "METH-02" in ids, "Missing METH-02 (LoRA targets)"
        assert "METH-03" in ids, "Missing METH-03 (LoRA config)"

    def test_registry_covers_p3_claims(self):
        """Registry must cover P3-01 through P3-08 claims."""
        claims = build_claim_registry()
        ids = {c.claim_id for c in claims}
        for i in range(1, 9):
            prefix = f"P3-0{i}"
            assert any(cid.startswith(prefix) for cid in ids), f"Missing {prefix} claims"


# ---------------------------------------------------------------------------
# Status validation tests
# ---------------------------------------------------------------------------

class TestStatusValidation:
    """Tests for claim status validation."""

    def test_allowed_statuses_complete(self):
        assert ALLOWED_STATUSES == {
            "VERIFIED",
            "PARTIALLY_VERIFIED",
            "MISLABELED",
            "UNVERIFIED",
            "INVALIDATED",
        }

    def test_default_status_is_unverified(self):
        c = Claim("T", "T", "T", "T", 0)
        assert c.status == "UNVERIFIED"

    def test_all_statuses_in_allowed_set(self):
        claims = build_claim_registry()
        # Before verification, all should be UNVERIFIED
        for c in claims:
            assert c.status in ALLOWED_STATUSES, f"Invalid status: {c.status}"


# ---------------------------------------------------------------------------
# Output file tests
# ---------------------------------------------------------------------------

class TestOutputFiles:
    """Tests for output file generation."""

    def test_write_outputs_creates_all_files(self, tmp_path):
        """write_outputs must create claim_registry.json, recomputed_metrics.csv,
        anomaly_report.md, and go_no_go.json."""
        claims = build_claim_registry()
        # Set some statuses for testing
        for i, c in enumerate(claims):
            c.status = ["VERIFIED", "PARTIALLY_VERIFIED", "MISLABELED",
                        "UNVERIFIED", "INVALIDATED"][i % 5]

        write_outputs(claims, tmp_path, Path("."))

        assert (tmp_path / "claim_registry.json").exists()
        assert (tmp_path / "recomputed_metrics.csv").exists()
        assert (tmp_path / "anomaly_report.md").exists()
        assert (tmp_path / "go_no_go.json").exists()

    def test_claim_registry_json_structure(self, tmp_path):
        """claim_registry.json must be a top-level LIST of claim dicts.

        The P4-G0 spec's acceptance verification command iterates this file
        directly with `for c in claims; c["status"]; c.get("claim_id")`, so it
        MUST be a JSON array, not a wrapped object.
        """
        claims = build_claim_registry()
        write_outputs(claims, tmp_path, Path("."))

        registry = json.loads((tmp_path / "claim_registry.json").read_text())
        assert isinstance(registry, list), "claim_registry.json must be a top-level list"
        assert len(registry) == len(claims)
        # Every entry must have the required fields per spec
        required_fields = [
            "claim_id", "claim_text", "claim_location", "metric_name",
            "reported_value", "recomputed_value", "artifact_path",
            "implementation_path", "checkpoint_path", "split_manifest",
            "status", "reason", "required_action",
        ]
        for entry in registry:
            for field in required_fields:
                assert field in entry, f"Missing field {field} in claim {entry.get('claim_id')}"
        # Phase metadata is emitted into go_no_go.json instead
        go = json.loads((tmp_path / "go_no_go.json").read_text())
        assert go["phase"] == "P4-G0"

    def test_go_no_go_json_structure(self, tmp_path):
        """go_no_go.json must have the required fields from the P4-G0 spec."""
        claims = build_claim_registry()
        write_outputs(claims, tmp_path, Path("."))

        go = json.loads((tmp_path / "go_no_go.json").read_text())
        assert go["phase"] == "P4-G0"
        assert go["status"] in {"GO", "PARTIAL_GO", "NO_GO", "DEFERRED"}
        assert "primary_metric" in go
        assert "predeclared_threshold" in go
        assert "evidence_paths" in go
        assert "limitations" in go
        assert "next_phase_allowed" in go

    def test_csv_has_header_and_rows(self, tmp_path):
        """recomputed_metrics.csv must have a header row and one row per claim."""
        claims = build_claim_registry()
        write_outputs(claims, tmp_path, Path("."))

        csv_text = (tmp_path / "recomputed_metrics.csv").read_text()
        lines = csv_text.strip().split("\n")
        assert len(lines) == len(claims) + 1  # header + rows
        assert "claim_id" in lines[0]
        assert "status" in lines[0]


# ---------------------------------------------------------------------------
# Verification logic tests
# ---------------------------------------------------------------------------

class TestVerificationLogic:
    """Tests for the verification dispatch logic."""

    def test_verify_claims_returns_list(self, tmp_path):
        """verify_claims must return a list of the same length."""
        claims = build_claim_registry()
        result = verify_claims(claims, tmp_path)
        assert isinstance(result, list)
        assert len(result) == len(claims)

    def test_all_claims_get_status_after_verify(self, tmp_path):
        """After verification, every claim must have a status in ALLOWED_STATUSES."""
        claims = build_claim_registry()
        verify_claims(claims, tmp_path)
        for c in claims:
            assert c.status in ALLOWED_STATUSES, (
                f"Claim {c.claim_id} has invalid status: {c.status}"
            )

    def test_unverified_claim_when_artifact_missing(self, tmp_path):
        """Claims should be UNVERIFIED when artifacts are missing."""
        claims = build_claim_registry()
        verify_claims(claims, tmp_path)  # tmp_path has no artifacts
        unverified = [c for c in claims if c.status == "UNVERIFIED"]
        # Most claims should be UNVERIFIED since no artifacts exist in tmp_path
        assert len(unverified) > 0


# ---------------------------------------------------------------------------
# Integration test (requires actual repo)
# ---------------------------------------------------------------------------

class TestIntegrationWithRepo:
    """Integration tests that require the actual pc_cng_research repo.

    These tests are skipped if the repo is not found at the expected location.
    """

    @pytest.fixture
    def repo_root(self):
        """Find the pc_cng_research repo root."""
        # Check common locations
        candidates = [
            Path(os.environ.get("PC_CNG_REPO", "")),
            Path.home() / "pc_cng_research",
            Path("/home/cunyuliu/pc_cng_research"),
            _REPO_ROOT / ".." / "pc_cng_research",
        ]
        for c in candidates:
            if c.exists() and (c / "results" / "pretrained_backbone_chemformer_lora_20260720").exists():
                return c
        pytest.skip("pc_cng_research repo not found")

    def test_full_audit_run(self, repo_root, tmp_path):
        """Run the full audit and verify outputs."""
        manuscript = repo_root / "docs" / "manuscript_v3_20260720.md"
        if not manuscript.exists():
            pytest.skip("manuscript not found")

        summary = run_claim_audit(manuscript, repo_root, tmp_path)
        assert summary["n_total"] > 0
        assert "n_verified" in summary

        # Check all output files exist
        assert (tmp_path / "claim_registry.json").exists()
        assert (tmp_path / "recomputed_metrics.csv").exists()
        assert (tmp_path / "anomaly_report.md").exists()
        assert (tmp_path / "go_no_go.json").exists()

    def test_go_no_go_decision_valid(self, repo_root, tmp_path):
        """The go_no_go.json must have a valid decision."""
        manuscript = repo_root / "docs" / "manuscript_v3_20260720.md"
        if not manuscript.exists():
            pytest.skip("manuscript not found")

        run_claim_audit(manuscript, repo_root, tmp_path)
        go = json.loads((tmp_path / "go_no_go.json").read_text())
        assert go["status"] in {"GO", "PARTIAL_GO", "NO_GO", "DEFERRED"}
