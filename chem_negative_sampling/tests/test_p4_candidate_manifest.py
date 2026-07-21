"""Tests for P4-G1 Benchmark Contract & Candidate Manifest.

Run with::

    python3 -m pytest chem_negative_sampling/tests/test_p4_candidate_manifest.py -v
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

# Ensure the pc_cng package is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "chem_negative_sampling") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "chem_negative_sampling"))

from pc_cng.build_p4_candidate_manifests import (
    MANIFEST_VERSION,
    REQUIRED_CANDIDATE_FIELDS,
    CANDIDATE_SOURCES,
    MANIFEST_SEED,
    _canonicalize_smiles,
    _strip_atom_map,
    _tanimoto_sim,
    _product_scaffold,
    _has_atom_map,
    _edit_distance,
    _corrupt_smiles,
    _compute_manifest_hash,
    _make_candidate,
    build_hte_feasibility_manifest,
    build_forward_candidates_manifest,
    build_retro_candidates_manifest,
    build_all_manifests,
)
from pc_cng.audit_p4_candidate_manifests import (
    audit_manifest,
    write_audit_report,
    EXPECTED_MANIFESTS,
)


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for SMILES helper functions."""

    def test_strip_atom_map_basic(self):
        # Atom-mapped SMILES has :NN: patterns
        mapped = "[CH3:1][OH:2]"
        unmapped = _strip_atom_map(mapped)
        assert ":1" not in unmapped
        assert ":2" not in unmapped

    def test_strip_atom_map_empty(self):
        assert _strip_atom_map("") == ""
        assert _strip_atom_map(None) == ""

    def test_canonicalize_smiles_ethanol(self):
        # RDKit may or may not be available in CI; skip if not
        if not _rdkit_available():
            pytest.skip("RDKit not available")
        canon = _canonicalize_smiles("CCO")
        assert canon == "CCO"

    def test_canonicalize_smiles_invalid(self):
        if not _rdkit_available():
            pytest.skip("RDKit not available")
        # Invalid SMILES returns empty string
        assert _canonicalize_smiles("XYZNOTASMILES") == ""

    def test_tanimoto_sim_identical(self):
        if not _rdkit_available():
            pytest.skip("RDKit not available")
        sim = _tanimoto_sim("CCO", "CCO")
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_tanimoto_sim_different(self):
        if not _rdkit_available():
            pytest.skip("RDKit not available")
        sim = _tanimoto_sim("CCO", "c1ccccc1")
        assert 0.0 <= sim < 1.0

    def test_product_scaffold_benzene(self):
        if not _rdkit_available():
            pytest.skip("RDKit not available")
        scaffold = _product_scaffold("c1ccccc1")
        # Benzene scaffold is benzene itself
        assert scaffold != ""

    def test_has_atom_map(self):
        assert _has_atom_map("[CH3:1][OH:2]") == "mapped"
        assert _has_atom_map("CCO") == "unmapped"
        assert _has_atom_map("") == "unknown"

    def test_edit_distance_identical(self):
        assert _edit_distance("CCO", "CCO") == 0

    def test_edit_distance_different(self):
        d = _edit_distance("CCO", "CCN")
        assert d == 1

    def test_edit_distance_empty(self):
        assert _edit_distance("", "CCO") == -1
        assert _edit_distance("CCO", "") == -1

    def test_corrupt_smiles_changes_input(self):
        import random
        rng = random.Random(42)
        original = "CCOc1ccccc1CC(=O)O"
        corrupted = _corrupt_smiles(original, rng)
        assert corrupted != original

    def test_compute_manifest_hash_deterministic(self):
        manifest = {
            "benchmark_name": "test",
            "manifest_version": "v1",
            "groups": [{"group_id": "g1"}],
        }
        h1 = _compute_manifest_hash(manifest)
        h2 = _compute_manifest_hash(manifest)
        assert h1 == h2
        # Hash should not depend on the manifest_hash field itself
        manifest_with_hash = dict(manifest)
        manifest_with_hash["manifest_hash"] = h1
        h3 = _compute_manifest_hash(manifest_with_hash)
        assert h1 == h3

    def test_compute_manifest_hash_changes_with_content(self):
        m1 = {"benchmark_name": "test1", "groups": []}
        m2 = {"benchmark_name": "test2", "groups": []}
        assert _compute_manifest_hash(m1) != _compute_manifest_hash(m2)


def _rdkit_available() -> bool:
    """Check whether RDKit is importable."""
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# _make_candidate tests
# ---------------------------------------------------------------------------

class TestMakeCandidate:
    """Tests for the _make_candidate helper."""

    def _make_train_products(self):
        return ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]

    def test_gold_candidate_has_gold_flag(self):
        c = _make_candidate(
            benchmark_name="test",
            group_id="g1",
            source_reaction_id="src1",
            parent_reaction_id="par1",
            experimental_group_id="exp1",
            split="test",
            candidate_smiles="CCO",
            candidate_source="gold",
            candidate_source_rank=0,
            gold=True,
            gold_smiles="CCO",
            reaction_family="alcohol",
            reaction_template="oxidation",
            train_products=self._make_train_products(),
            manifest_hash="abc123",
        )
        assert c["gold_candidate"] is True
        assert c["candidate_source"] == "gold"
        assert c["oracle_coverage"] == 1.0
        assert c["edit_type"] == "none"
        assert c["edit_distance"] == 0

    def test_non_gold_candidate_has_zero_oracle(self):
        c = _make_candidate(
            benchmark_name="test",
            group_id="g1",
            source_reaction_id="src1",
            parent_reaction_id="par1",
            experimental_group_id="exp1",
            split="test",
            candidate_smiles="CCN",
            candidate_source="random_mismatch",
            candidate_source_rank=1,
            gold=False,
            gold_smiles="CCO",
            reaction_family="alcohol",
            reaction_template="oxidation",
            train_products=self._make_train_products(),
            manifest_hash="abc123",
        )
        assert c["gold_candidate"] is False
        assert c["oracle_coverage"] == 0.0
        assert c["edit_type"] == "random_mismatch"
        assert c["edit_distance"] > 0

    def test_all_24_required_fields_populated(self):
        c = _make_candidate(
            benchmark_name="test",
            group_id="g1",
            source_reaction_id="src1",
            parent_reaction_id="par1",
            experimental_group_id="exp1",
            split="test",
            candidate_smiles="CCO",
            candidate_source="gold",
            candidate_source_rank=0,
            gold=True,
            gold_smiles="CCO",
            reaction_family="alcohol",
            reaction_template="oxidation",
            train_products=self._make_train_products(),
            manifest_hash="abc123",
        )
        for field in REQUIRED_CANDIDATE_FIELDS:
            assert field in c, f"Missing required field: {field}"
            assert c[field] is not None or isinstance(c[field], (bool, int, float, str)), (
                f"Field {field} has invalid value: {c[field]!r}"
            )

    def test_known_positive_collision_when_non_gold_matches_gold(self):
        """When a non-gold candidate canonical SMILES equals gold canonical."""
        c = _make_candidate(
            benchmark_name="test",
            group_id="g1",
            source_reaction_id="src1",
            parent_reaction_id="par1",
            experimental_group_id="exp1",
            split="test",
            candidate_smiles="CCO",  # same as gold
            candidate_source="tanimoto_retrieval",
            candidate_source_rank=3,
            gold=False,  # but not marked as gold
            gold_smiles="CCO",
            reaction_family="alcohol",
            reaction_template="oxidation",
            train_products=self._make_train_products(),
            manifest_hash="abc123",
        )
        assert c["known_positive_collision"] is True

    def test_candidate_id_format(self):
        c = _make_candidate(
            benchmark_name="test",
            group_id="g1",
            source_reaction_id="src1",
            parent_reaction_id="par1",
            experimental_group_id="exp1",
            split="test",
            candidate_smiles="CCO",
            candidate_source="random_corruption",
            candidate_source_rank=2,
            gold=False,
            gold_smiles="CCO",
            reaction_family="alcohol",
            reaction_template="oxidation",
            train_products=[],
            manifest_hash="abc123",
        )
        assert c["candidate_id"] == "g1_random_corruption_2"


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Tests for module-level constants."""

    def test_manifest_version_is_v1(self):
        assert MANIFEST_VERSION == "v1"

    def test_manifest_seed_is_fixed(self):
        assert MANIFEST_SEED == 20260721

    def test_eight_candidate_sources(self):
        assert len(CANDIDATE_SOURCES) == 8
        for src in ["gold", "random_mismatch", "random_corruption",
                    "tanimoto_retrieval", "template_perturbation",
                    "unconstrained_edit", "rule_pc_cng", "external_beam"]:
            assert src in CANDIDATE_SOURCES

    def test_24_required_candidate_fields(self):
        assert len(REQUIRED_CANDIDATE_FIELDS) == 24

    def test_required_fields_match_spec(self):
        expected = {
            "benchmark_name", "group_id", "source_reaction_id",
            "parent_reaction_id", "experimental_group_id", "gold_candidate",
            "candidate_id", "candidate_smiles", "candidate_source",
            "candidate_source_rank", "canonical_smiles", "atom_mapping_status",
            "reaction_family", "reaction_template", "product_scaffold",
            "edit_type", "edit_distance", "train_overlap",
            "known_positive_collision", "nearest_train_similarity",
            "split", "oracle_coverage", "manifest_version", "manifest_hash",
        }
        assert set(REQUIRED_CANDIDATE_FIELDS) == expected

    def test_expected_manifests_listed(self):
        assert set(EXPECTED_MANIFESTS) == {
            "hte_feasibility_v1.json",
            "fixed_forward_candidates_v1.json",
            "fixed_retro_candidates_v1.json",
        }


# ---------------------------------------------------------------------------
# Manifest structure tests (build small synthetic manifests)
# ---------------------------------------------------------------------------

class TestManifestStructure:
    """Tests that manifest structure matches the P4-G1 spec."""

    def _make_synthetic_manifest(self, n_groups=3) -> dict:
        """Build a small synthetic manifest for testing."""
        import random
        rng = random.Random(MANIFEST_SEED)
        groups = []
        for i in range(n_groups):
            gid = f"test_group_{i}"
            gold_smi = "CCO"
            candidates = [
                _make_candidate(
                    benchmark_name="test",
                    group_id=gid,
                    source_reaction_id=f"src_{i}",
                    parent_reaction_id=f"par_{i}",
                    experimental_group_id=f"exp_{i}",
                    split="test",
                    candidate_smiles=gold_smi,
                    candidate_source="gold",
                    candidate_source_rank=0,
                    gold=True,
                    gold_smiles=gold_smi,
                    reaction_family="alcohol",
                    reaction_template="oxidation",
                    train_products=["c1ccccc1"],
                    manifest_hash="PENDING",
                )
            ]
            # Add one non-gold candidate
            candidates.append(_make_candidate(
                benchmark_name="test",
                group_id=gid,
                source_reaction_id=f"src_{i}",
                parent_reaction_id=f"par_{i}",
                experimental_group_id=f"exp_{i}",
                split="test",
                candidate_smiles="CCN",
                candidate_source="random_mismatch",
                candidate_source_rank=1,
                gold=False,
                gold_smiles=gold_smi,
                reaction_family="alcohol",
                reaction_template="oxidation",
                train_products=["c1ccccc1"],
                manifest_hash="PENDING",
            ))
            groups.append({
                "group_id": gid,
                "source_reaction_id": f"src_{i}",
                "parent_reaction_id": f"par_{i}",
                "experimental_group_id": f"exp_{i}",
                "split": "test",
                "candidates": candidates,
            })

        manifest = {
            "benchmark_name": "test",
            "manifest_version": MANIFEST_VERSION,
            "manifest_hash": "",
            "groups": groups,
        }
        manifest["manifest_hash"] = _compute_manifest_hash(manifest)
        # Backfill
        for g in groups:
            for c in g["candidates"]:
                c["manifest_hash"] = manifest["manifest_hash"]
        return manifest

    def test_manifest_has_required_top_level_fields(self):
        m = self._make_synthetic_manifest()
        assert "benchmark_name" in m
        assert "manifest_version" in m
        assert "manifest_hash" in m
        assert "groups" in m
        assert m["manifest_version"] == MANIFEST_VERSION

    def test_manifest_hash_is_sha256_hex(self):
        m = self._make_synthetic_manifest()
        h = m["manifest_hash"]
        assert len(h) == 64
        # All hex characters
        int(h, 16)  # raises ValueError if not hex

    def test_each_group_has_exactly_one_gold(self):
        m = self._make_synthetic_manifest()
        for g in m["groups"]:
            golds = [c for c in g["candidates"] if c["gold_candidate"]]
            assert len(golds) == 1

    def test_candidate_ids_unique_within_group(self):
        m = self._make_synthetic_manifest()
        for g in m["groups"]:
            cids = [c["candidate_id"] for c in g["candidates"]]
            assert len(cids) == len(set(cids))

    def test_all_required_fields_in_each_candidate(self):
        m = self._make_synthetic_manifest()
        for g in m["groups"]:
            for c in g["candidates"]:
                for field in REQUIRED_CANDIDATE_FIELDS:
                    assert field in c, f"Missing field {field}"

    def test_manifest_hash_backfilled_to_candidates(self):
        m = self._make_synthetic_manifest()
        h = m["manifest_hash"]
        for g in m["groups"]:
            for c in g["candidates"]:
                assert c["manifest_hash"] == h

    def test_manifest_hash_recomputes_correctly(self):
        m = self._make_synthetic_manifest()
        recomputed = _compute_manifest_hash(m)
        assert recomputed == m["manifest_hash"]


# ---------------------------------------------------------------------------
# Audit module tests
# ---------------------------------------------------------------------------

class TestAuditModule:
    """Tests for the audit_manifest function."""

    def _make_valid_manifest(self) -> dict:
        """Build a manifest that passes audit."""
        import random
        rng = random.Random(MANIFEST_SEED)
        groups = []
        for i in range(5):
            gid = f"valid_group_{i}"
            gold_smi = "CCO"
            candidates = [
                _make_candidate(
                    benchmark_name="test",
                    group_id=gid, source_reaction_id=f"src_{i}",
                    parent_reaction_id=f"par_{i}", experimental_group_id=f"exp_{i}",
                    split="test",
                    candidate_smiles=gold_smi, candidate_source="gold",
                    candidate_source_rank=0, gold=True, gold_smiles=gold_smi,
                    reaction_family="alcohol", reaction_template="oxidation",
                    train_products=["c1ccccc1"], manifest_hash="PENDING",
                ),
                _make_candidate(
                    benchmark_name="test",
                    group_id=gid, source_reaction_id=f"src_{i}",
                    parent_reaction_id=f"par_{i}", experimental_group_id=f"exp_{i}",
                    split="test",
                    candidate_smiles="CCN", candidate_source="random_mismatch",
                    candidate_source_rank=1, gold=False, gold_smiles=gold_smi,
                    reaction_family="alcohol", reaction_template="oxidation",
                    train_products=["c1ccccc1"], manifest_hash="PENDING",
                ),
            ]
            groups.append({
                "group_id": gid, "source_reaction_id": f"src_{i}",
                "parent_reaction_id": f"par_{i}", "experimental_group_id": f"exp_{i}",
                "split": "test", "candidates": candidates,
            })
        m = {
            "benchmark_name": "test",
            "manifest_version": MANIFEST_VERSION,
            "manifest_hash": "",
            "groups": groups,
        }
        m["manifest_hash"] = _compute_manifest_hash(m)
        for g in groups:
            for c in g["candidates"]:
                c["manifest_hash"] = m["manifest_hash"]
        return m

    def test_valid_manifest_passes_audit(self):
        m = self._make_valid_manifest()
        findings = audit_manifest(m, "test_manifest.json")
        assert findings["n_groups"] == 5
        assert findings["n_gold"] == 5
        assert len(findings["errors"]) == 0
        assert findings["stats"]["parent_leakage_count"] == 0

    def test_manifest_with_zero_gold_fails(self):
        m = self._make_valid_manifest()
        # Set all candidates to non-gold
        for g in m["groups"]:
            for c in g["candidates"]:
                c["gold_candidate"] = False
        m["manifest_hash"] = _compute_manifest_hash(m)
        findings = audit_manifest(m, "test_zero_gold.json")
        assert len(findings["errors"]) > 0
        assert any("exactly 1 gold" in e for e in findings["errors"])

    def test_manifest_with_two_golds_fails(self):
        m = self._make_valid_manifest()
        # Make the second candidate also gold
        for g in m["groups"]:
            if len(g["candidates"]) > 1:
                g["candidates"][1]["gold_candidate"] = True
        m["manifest_hash"] = _compute_manifest_hash(m)
        findings = audit_manifest(m, "test_two_golds.json")
        assert len(findings["errors"]) > 0
        assert any("exactly 1 gold" in e for e in findings["errors"])

    def test_manifest_with_parent_leakage_fails(self):
        m = self._make_valid_manifest()
        # Make first group's parent_reaction_id same as second group's, but with different split
        if len(m["groups"]) >= 2:
            m["groups"][0]["parent_reaction_id"] = "shared_parent"
            m["groups"][0]["split"] = "train"
            for c in m["groups"][0]["candidates"]:
                c["parent_reaction_id"] = "shared_parent"
                c["split"] = "train"
            m["groups"][1]["parent_reaction_id"] = "shared_parent"
            m["groups"][1]["split"] = "test"
            for c in m["groups"][1]["candidates"]:
                c["parent_reaction_id"] = "shared_parent"
                c["split"] = "test"
        m["manifest_hash"] = _compute_manifest_hash(m)
        findings = audit_manifest(m, "test_leakage.json")
        assert findings["stats"]["parent_leakage_count"] >= 1
        assert any("Parent leakage" in e for e in findings["errors"])

    def test_manifest_with_missing_field_fails(self):
        m = self._make_valid_manifest()
        # Remove a field from first candidate
        del m["groups"][0]["candidates"][0]["reaction_template"]
        m["manifest_hash"] = _compute_manifest_hash(m)
        findings = audit_manifest(m, "test_missing_field.json")
        assert any("missing field" in e for e in findings["errors"])

    def test_manifest_with_duplicate_candidate_ids_fails(self):
        m = self._make_valid_manifest()
        # Make two candidates have the same ID
        if len(m["groups"][0]["candidates"]) >= 2:
            m["groups"][0]["candidates"][1]["candidate_id"] = m["groups"][0]["candidates"][0]["candidate_id"]
        m["manifest_hash"] = _compute_manifest_hash(m)
        findings = audit_manifest(m, "test_dup_cids.json")
        assert any("duplicate candidate_ids" in e for e in findings["errors"])

    def test_audit_reports_stats(self):
        m = self._make_valid_manifest()
        findings = audit_manifest(m, "test_stats.json")
        s = findings["stats"]
        assert "candidate_source_distribution" in s
        assert "split_distribution" in s
        assert "candidate_count_per_group" in s
        assert "known_positive_collisions" in s
        assert "train_overlaps" in s
        assert "nearest_train_similarity" in s
        assert "oracle_top1_coverage" in s
        assert "n_unique_scaffolds" in s
        assert "n_unique_parents" in s
        assert "parent_leakage_count" in s


# ---------------------------------------------------------------------------
# go_no_go.json tests
# ---------------------------------------------------------------------------

class TestGoNoGo:
    """Tests for the go_no_go.json writer."""

    def test_go_status_when_no_errors(self, tmp_path):
        import random
        rng = random.Random(MANIFEST_SEED)
        groups = []
        for i in range(3):
            gid = f"gn_group_{i}"
            candidates = [
                _make_candidate(
                    benchmark_name="test", group_id=gid, source_reaction_id=f"src_{i}",
                    parent_reaction_id=f"par_{i}", experimental_group_id=f"exp_{i}",
                    split="test", candidate_smiles="CCO", candidate_source="gold",
                    candidate_source_rank=0, gold=True, gold_smiles="CCO",
                    reaction_family="alcohol", reaction_template="oxidation",
                    train_products=[], manifest_hash="PENDING",
                ),
            ]
            groups.append({
                "group_id": gid, "source_reaction_id": f"src_{i}",
                "parent_reaction_id": f"par_{i}", "experimental_group_id": f"exp_{i}",
                "split": "test", "candidates": candidates,
            })
        m = {
            "benchmark_name": "test", "manifest_version": MANIFEST_VERSION,
            "manifest_hash": "", "groups": groups,
        }
        m["manifest_hash"] = _compute_manifest_hash(m)
        for g in groups:
            for c in g["candidates"]:
                c["manifest_hash"] = m["manifest_hash"]

        findings = [audit_manifest(m, "test.json")]
        go_no_go = write_audit_report(findings, tmp_path)

        assert go_no_go["phase"] == "P4-G1"
        assert go_no_go["status"] == "GO"
        assert go_no_go["next_phase_allowed"] is True
        # Files should exist
        assert (tmp_path / "go_no_go.json").exists()
        assert (tmp_path / "audit_report.md").exists()
        assert (tmp_path / "manifest_audit_details.json").exists()

    def test_no_go_status_when_errors_present(self, tmp_path):
        # Manifest with errors (no gold)
        m = {
            "benchmark_name": "test",
            "manifest_version": MANIFEST_VERSION,
            "manifest_hash": "",
            "groups": [
                {
                    "group_id": "g1",
                    "source_reaction_id": "s1",
                    "parent_reaction_id": "p1",
                    "experimental_group_id": "e1",
                    "split": "test",
                    "candidates": [
                        {"candidate_id": "c1", "gold_candidate": False},
                    ],
                },
            ],
        }
        m["manifest_hash"] = _compute_manifest_hash(m)
        findings = [audit_manifest(m, "bad.json")]
        go_no_go = write_audit_report(findings, tmp_path)
        assert go_no_go["status"] == "NO_GO"
        assert go_no_go["next_phase_allowed"] is False


# ---------------------------------------------------------------------------
# End-to-end build tests (require repo data; skipped if data missing)
# ---------------------------------------------------------------------------

class TestEndToEndBuild:
    """End-to-end build tests. These require the actual repo data on the
    remote server and will be skipped if data files are not present."""

    def _repo_root(self) -> Path:
        """Find the repo root by searching upward for data/processed/."""
        p = Path(__file__).resolve()
        for parent in [p.parent] + list(p.parents):
            if (parent / "data" / "processed").exists():
                return parent
        return _REPO_ROOT

    def test_build_hte_feasibility_manifest(self):
        repo_root = self._repo_root()
        csv_path = repo_root / "data" / "processed" / "hitea_full_normalized.csv"
        if not csv_path.exists():
            pytest.skip(f"HTEa data not found: {csv_path}")
        import random
        rng = random.Random(MANIFEST_SEED)
        m = build_hte_feasibility_manifest(repo_root, rng)
        assert m["benchmark_name"] == "P4-HTE-Feasibility"
        assert m["manifest_version"] == MANIFEST_VERSION
        assert len(m["manifest_hash"]) == 64
        assert len(m["groups"]) > 0
        # Every group must have exactly one gold
        for g in m["groups"]:
            golds = [c for c in g["candidates"] if c["gold_candidate"]]
            assert len(golds) == 1

    def test_build_forward_candidates_manifest(self):
        repo_root = self._repo_root()
        csv_path = repo_root / "data" / "processed" / "uspto_openmolecules_normalized.csv"
        if not csv_path.exists():
            pytest.skip(f"USPTO-OM data not found: {csv_path}")
        import random
        rng = random.Random(MANIFEST_SEED + 1)
        m = build_forward_candidates_manifest(repo_root, rng)
        assert m["benchmark_name"] == "P4-Fixed-Forward-Candidates"
        assert m["manifest_version"] == MANIFEST_VERSION
        assert len(m["manifest_hash"]) == 64
        assert len(m["groups"]) > 0

    def test_build_retro_candidates_manifest(self):
        repo_root = self._repo_root()
        csv_path = repo_root / "data" / "processed" / "uspto_openmolecules_normalized.csv"
        if not csv_path.exists():
            pytest.skip(f"USPTO-OM data not found: {csv_path}")
        import random
        rng = random.Random(MANIFEST_SEED + 2)
        m = build_retro_candidates_manifest(repo_root, rng)
        assert m["benchmark_name"] == "P4-Fixed-Retro-Candidates"
        assert m["manifest_version"] == MANIFEST_VERSION
        assert len(m["manifest_hash"]) == 64
        assert len(m["groups"]) > 0

    def test_build_all_manifests_writes_files(self, tmp_path):
        repo_root = self._repo_root()
        if not (repo_root / "data" / "processed" / "hitea_full_normalized.csv").exists():
            pytest.skip("Repo data not available")
        output_dir = tmp_path / "manifests"
        summary = build_all_manifests(repo_root, output_dir)
        assert "benchmarks" in summary
        for name in ["hte_feasibility", "fixed_forward", "fixed_retro"]:
            assert name in summary["benchmarks"]
            assert summary["benchmarks"][name]["n_groups"] > 0
        # Verify files exist
        assert (output_dir / "hte_feasibility_v1.json").exists()
        assert (output_dir / "fixed_forward_candidates_v1.json").exists()
        assert (output_dir / "fixed_retro_candidates_v1.json").exists()


# ---------------------------------------------------------------------------
# Spec acceptance test (the extra structural check from the spec)
# ---------------------------------------------------------------------------

class TestSpecAcceptanceCheck:
    """The spec lines 462-473 require an additional structural check:

    .. code-block:: python

        for path in Path("data/p4/manifests").glob("*.json"):
            data = json.load(open(path))
            assert data["manifest_hash"]
            assert data["groups"]
            for group in data["groups"]:
                assert sum(bool(c["gold_candidate"]) for c in group["candidates"]) == 1
                assert len({c["candidate_id"] for c in group["candidates"]}) == len(group["candidates"])
    """

    def test_spec_structural_check_passes_on_synthetic(self, tmp_path):
        """Verify the spec's exact structural check passes on a synthetic manifest."""
        # Build synthetic manifest
        import random
        rng = random.Random(MANIFEST_SEED)
        groups = []
        for i in range(3):
            gid = f"spec_group_{i}"
            gold_smi = "CCO"
            candidates = [
                _make_candidate(
                    benchmark_name="test", group_id=gid, source_reaction_id=f"src_{i}",
                    parent_reaction_id=f"par_{i}", experimental_group_id=f"exp_{i}",
                    split="test", candidate_smiles=gold_smi, candidate_source="gold",
                    candidate_source_rank=0, gold=True, gold_smiles=gold_smi,
                    reaction_family="alcohol", reaction_template="oxidation",
                    train_products=[], manifest_hash="PENDING",
                ),
                _make_candidate(
                    benchmark_name="test", group_id=gid, source_reaction_id=f"src_{i}",
                    parent_reaction_id=f"par_{i}", experimental_group_id=f"exp_{i}",
                    split="test", candidate_smiles="CCN", candidate_source="random_mismatch",
                    candidate_source_rank=1, gold=False, gold_smiles=gold_smi,
                    reaction_family="alcohol", reaction_template="oxidation",
                    train_products=[], manifest_hash="PENDING",
                ),
            ]
            groups.append({
                "group_id": gid, "source_reaction_id": f"src_{i}",
                "parent_reaction_id": f"par_{i}", "experimental_group_id": f"exp_{i}",
                "split": "test", "candidates": candidates,
            })
        m = {
            "benchmark_name": "test", "manifest_version": MANIFEST_VERSION,
            "manifest_hash": "", "groups": groups,
        }
        m["manifest_hash"] = _compute_manifest_hash(m)
        for g in groups:
            for c in g["candidates"]:
                c["manifest_hash"] = m["manifest_hash"]

        # Write to tmp_path and run the spec check
        manifest_path = tmp_path / "test_manifest_v1.json"
        manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")

        # Run the EXACT check from the spec
        data = json.load(open(manifest_path))
        assert data["manifest_hash"]
        assert data["groups"]
        for group in data["groups"]:
            assert sum(bool(c["gold_candidate"]) for c in group["candidates"]) == 1
            assert len({c["candidate_id"] for c in group["candidates"]}) == len(group["candidates"])
