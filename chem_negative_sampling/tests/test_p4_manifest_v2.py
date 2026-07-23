"""Tests for P4 v2 candidate manifest builder.

Run with::

    python3 -m pytest tests/test_p4_manifest_v2.py -v
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest
from rdkit import Chem

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pc_cng.build_p4_candidate_manifests_v2 import (  # noqa: E402
    RULES,
    V2_SEED,
    audit_v2_manifest,
    build_known_positive_pool,
    build_v2_manifest,
    generate_rule_pccng,
    generate_valid_corruption,
    generate_valid_unconstrained_edit,
)


def _canon(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles  # fixture v1 deliberately contains invalid SMILES
    return Chem.MolToSmiles(mol, isomericSmiles=True)


# ---------------------------------------------------------------------------
# Rule-based PC-CNG generator
# ---------------------------------------------------------------------------

class TestRulePCCNG:
    """Reaction-SMARTS interconversion rules."""

    def test_rules_compile(self):
        """All 17 rules must be defined."""
        assert len(RULES) == 17

    def test_acid_to_amide(self):
        smi, meta = generate_rule_pccng("CC(=O)O", frozenset(),
                                        random.Random(0))
        assert smi is not None
        assert Chem.MolFromSmiles(smi) is not None
        assert "N" in smi  # amide nitrogen introduced
        assert meta["rule"] is not None

    def test_halogen_swap(self):
        smi, _ = generate_rule_pccng("c1ccc(Br)cc1", frozenset(),
                                     random.Random(0))
        assert smi is not None
        assert "Cl" in smi or "Br" not in smi

    def test_nitro_to_amine(self):
        smi, _ = generate_rule_pccng("c1ccc([N+](=O)[O-])cc1", frozenset(),
                                     random.Random(0))
        assert smi is not None
        assert Chem.MolFromSmiles(smi) is not None

    def test_product_differs_from_gold(self):
        gold = "CC(=O)O"
        smi, _ = generate_rule_pccng(gold, frozenset(), random.Random(0))
        assert smi is not None
        assert _canon(smi) != _canon(gold)

    def test_atom_mapped_gold_accepted(self):
        """Atom-mapped gold (as in the HTEa data) must work."""
        smi, _ = generate_rule_pccng("[CH3:1][C:2](=[O:3])[OH:4]",
                                     frozenset(), random.Random(0))
        assert smi is not None
        assert Chem.MolFromSmiles(smi) is not None

    def test_collision_rejected_and_next_rule_tried(self):
        """Products in the known-positive pool must be rejected."""
        gold = "CC(=O)O"
        # Poison the pool with every reachable product; generator must
        # exhaust rules and return None rather than emit a collision.
        pool = set()
        for seed in range(50):
            smi, _ = generate_rule_pccng(gold, frozenset(),
                                         random.Random(seed))
            if smi:
                pool.add(_canon(smi))
        pool.add(_canon(gold))
        smi, meta = generate_rule_pccng(gold, frozenset(pool),
                                        random.Random(0))
        assert smi is None
        assert meta["rejected_collisions"] >= 1

    def test_no_applicable_rule_returns_none(self):
        """A molecule with no rule-matching group yields None."""
        smi, _ = generate_rule_pccng("CCCC", frozenset(), random.Random(0))
        assert smi is None

    def test_deterministic_per_seed(self):
        out1, _ = generate_rule_pccng("c1ccc(Br)cc1C(=O)O", frozenset(),
                                      random.Random(42))
        out2, _ = generate_rule_pccng("c1ccc(Br)cc1C(=O)O", frozenset(),
                                      random.Random(42))
        assert out1 == out2


# ---------------------------------------------------------------------------
# Valid corruption / unconstrained edit generators
# ---------------------------------------------------------------------------

_FIXTURES = [
    "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1CCC[C@H]1c1cccnc1",
    "c1ccc([N+](=O)[O-])cc1", "CC(C)C(=O)Nc1ccccc1", "OCc1ccccc1",
    "c1ccc(Br)cc1", "CCOC(=O)c1ccccc1", "C1CCCCC1",
]


class TestValidCorruption:
    def test_all_fixtures_produce_valid_smiles(self):
        for fixture in _FIXTURES:
            rng = random.Random(f"corr:{fixture}")
            smi, meta = generate_valid_corruption(fixture, frozenset(), rng)
            assert smi is not None, fixture
            assert Chem.MolFromSmiles(smi) is not None, fixture
            assert _canon(smi) != _canon(fixture)
            assert meta["attempts"] >= 1

    def test_deterministic(self):
        a, _ = generate_valid_corruption("CCO", frozenset(), random.Random(7))
        b, _ = generate_valid_corruption("CCO", frozenset(), random.Random(7))
        assert a == b

    def test_collision_avoided(self):
        gold = "CCO"
        pool = set()
        for seed in range(30):
            smi, _ = generate_valid_corruption(gold, frozenset(),
                                               random.Random(seed))
            if smi:
                pool.add(_canon(smi))
        rng = random.Random(0)
        smi, _ = generate_valid_corruption(gold, frozenset(pool), rng)
        assert smi is None or _canon(smi) not in pool


class TestValidUnconstrainedEdit:
    def test_all_fixtures_produce_valid_smiles(self):
        for fixture in _FIXTURES:
            rng = random.Random(f"edit:{fixture}")
            smi, meta = generate_valid_unconstrained_edit(
                fixture, frozenset(), rng)
            assert smi is not None, fixture
            assert Chem.MolFromSmiles(smi) is not None, fixture
            assert _canon(smi) != _canon(fixture)
            assert meta["attempts"] >= 1

    def test_edit_changes_heavy_atom_count(self):
        """Deletion edits should usually reduce heavy-atom count."""
        reduced = 0
        gold = "CC(=O)Oc1ccccc1C(=O)O"
        n_gold = Chem.MolFromSmiles(gold).GetNumHeavyAtoms()
        for seed in range(10):
            smi, _ = generate_valid_unconstrained_edit(
                gold, frozenset(), random.Random(seed))
            if Chem.MolFromSmiles(smi).GetNumHeavyAtoms() < n_gold:
                reduced += 1
        assert reduced >= 5


# ---------------------------------------------------------------------------
# v2 manifest assembly (fixture v1 manifest)
# ---------------------------------------------------------------------------

def _fixture_v1_group(gid: str, gold_smiles: str, split: str) -> dict:
    """Minimal v1-style group with all 8 candidates."""
    def cand(src, rank, smi, gold=False):
        return {
            "benchmark_name": "P4-HTE-Feasibility",
            "group_id": gid,
            "source_reaction_id": f"src_{gid}",
            "parent_reaction_id": gid,
            "experimental_group_id": gid,
            "gold_candidate": gold,
            "candidate_id": f"{gid}_{src}_{rank}",
            "candidate_smiles": smi,
            "candidate_source": src,
            "candidate_source_rank": rank,
            "canonical_smiles": _canon(smi),
            "atom_mapping_status": False,
            "reaction_family": "fam",
            "reaction_template": "HTEa",
            "product_scaffold": "",
            "edit_type": "none" if gold else src,
            "edit_distance": 0 if gold else 1,
            "train_overlap": False,
            "known_positive_collision": False,
            "nearest_train_similarity": 0.0,
            "split": split,
            "oracle_coverage": 1.0 if gold else 0.0,
            "manifest_version": "v1",
            "manifest_hash": "v1hash",
        }
    return {
        "group_id": gid,
        "source_reaction_id": f"src_{gid}",
        "parent_reaction_id": gid,
        "experimental_group_id": gid,
        "split": split,
        "candidates": [
            cand("gold", 0, gold_smiles, gold=True),
            cand("random_mismatch", 1, "CCOCC"),
            cand("random_corruption", 2, "INVALID###"),
            cand("tanimoto_retrieval", 3, "CCO"),
            cand("template_perturbation", 4, "CCC"),
            cand("unconstrained_edit", 5, "TRUNC("),
            cand("rule_pc_cng", 6, "INVALID###"),
            cand("external_beam", 7, "CCCC"),
        ],
    }


def _fixture_v1() -> dict:
    groups = [
        _fixture_v1_group("hte_g1", "[CH3:1][C:2](=[O:3])[OH:4]", "train"),
        _fixture_v1_group("hte_g2", "c1ccc(Br)cc1", "test"),
        _fixture_v1_group("hte_g3", "c1ccc([N+](=O)[O-])cc1", "val"),
    ]
    m = {"benchmark_name": "P4-HTE-Feasibility", "manifest_version": "v1",
         "manifest_hash": "", "groups": groups}
    from pc_cng.build_p4_candidate_manifests import _compute_manifest_hash
    m["manifest_hash"] = _compute_manifest_hash(m)
    return m


class TestBuildV2Manifest:
    def test_build_and_audit_fixture(self):
        v1 = _fixture_v1()
        pool = frozenset({_canon("CC(=O)O"), _canon("c1ccc(Br)cc1"),
                          _canon("c1ccc([N+](=O)[O-])cc1")})
        v2, report = build_v2_manifest(v1, pool, train_products=[],
                                       seed=V2_SEED)
        assert v2["manifest_version"] == "v2"
        assert v2["manifest_hash"] != v1["manifest_hash"]
        assert v2["derived_from"]["manifest_version"] == "v1"
        assert len(v2["groups"]) == 3

        audit = audit_v2_manifest(v1, v2)
        failed = [c for c in audit["checks"] if not c["passed"]]
        assert not failed, failed
        assert audit["all_passed"]
        for src, entry in audit["per_source_validity"].items():
            assert entry["valid_fraction"] == 1.0, src

    def test_regenerated_sources_changed_others_identical(self):
        v1 = _fixture_v1()
        v2, _ = build_v2_manifest(v1, frozenset(), train_products=[],
                                  seed=V2_SEED)
        for g1, g2 in zip(v1["groups"], v2["groups"]):
            c1 = {c["candidate_source"]: c for c in g1["candidates"]}
            c2 = {c["candidate_source"]: c for c in g2["candidates"]}
            for src in ("gold", "random_mismatch", "tanimoto_retrieval",
                        "template_perturbation", "external_beam"):
                assert c1[src]["candidate_smiles"] == \
                    c2[src]["candidate_smiles"], src
            for src in ("random_corruption", "unconstrained_edit",
                        "rule_pc_cng"):
                assert c2[src]["candidate_smiles"] not in (
                    "INVALID###", "TRUNC("), src
                assert Chem.MolFromSmiles(
                    c2[src]["candidate_smiles"]) is not None, src

    def test_rule_metadata_recorded(self):
        v1 = _fixture_v1()
        v2, report = build_v2_manifest(v1, frozenset(), train_products=[],
                                       seed=V2_SEED)
        for g in v2["groups"]:
            rec = next(c for c in g["candidates"]
                       if c["candidate_source"] == "rule_pc_cng")
            assert "pccng_rule" in rec
        assert report["rule_usage"]

    def test_deterministic_build(self):
        v1 = _fixture_v1()
        v2a, _ = build_v2_manifest(v1, frozenset(), train_products=[],
                                   seed=V2_SEED)
        v2b, _ = build_v2_manifest(v1, frozenset(), train_products=[],
                                   seed=V2_SEED)
        assert v2a["manifest_hash"] == v2b["manifest_hash"]

    def test_different_seed_different_candidates(self):
        v1 = _fixture_v1()
        v2a, _ = build_v2_manifest(v1, frozenset(), train_products=[],
                                   seed=1)
        v2b, _ = build_v2_manifest(v1, frozenset(), train_products=[],
                                   seed=2)
        assert v2a["manifest_hash"] != v2b["manifest_hash"]

    def test_regenerated_candidates_differ_from_all_siblings(self):
        """Even when the rule fallback fires, A6 must differ from A2."""
        # Gold with NO rule-applicable group -> rule fallback always fires.
        v1 = {"benchmark_name": "P4-HTE-Feasibility",
              "manifest_version": "v1", "manifest_hash": "h",
              "groups": [_fixture_v1_group("hte_g9", "c1ccncc1", "train")]}
        v2, report = build_v2_manifest(v1, frozenset(), train_products=[],
                                       seed=V2_SEED)
        g = v2["groups"][0]
        canons = {}
        for c in g["candidates"]:
            canon = c["canonical_smiles"]
            assert canon not in canons, (c["candidate_source"], canon)
            canons[canon] = c["candidate_source"]

    def test_candidates_have_24_fields(self):
        from pc_cng.build_p4_candidate_manifests import (
            REQUIRED_CANDIDATE_FIELDS)
        v1 = _fixture_v1()
        v2, _ = build_v2_manifest(v1, frozenset(), train_products=[],
                                  seed=V2_SEED)
        for g in v2["groups"]:
            for c in g["candidates"]:
                for field in REQUIRED_CANDIDATE_FIELDS:
                    assert field in c, field


# ---------------------------------------------------------------------------
# Audit failure detection
# ---------------------------------------------------------------------------

class TestAuditDetectsDefects:
    def test_detects_a6_equals_a2(self):
        """The v1 defect (A6 ≡ A2) must fail the audit."""
        v1 = _fixture_v1()
        # Sabotage: make rule_pc_cng identical to random_corruption.
        for g in v1["groups"]:
            for c in g["candidates"]:
                if c["candidate_source"] == "rule_pc_cng":
                    c["candidate_smiles"] = next(
                        x["candidate_smiles"] for x in g["candidates"]
                        if x["candidate_source"] == "random_corruption")
                    c["canonical_smiles"] = "dup"
                if c["candidate_source"] == "random_corruption":
                    c["canonical_smiles"] = "dup"
        audit = audit_v2_manifest(v1, v1)  # v1 audited as "v2"
        names = {c["name"]: c["passed"] for c in audit["checks"]}
        assert names["rule_pccng_differs_from_corruption"] is False
        assert names["all_candidates_parseable"] is False
        assert audit["all_passed"] is False

    def test_detects_unparseable(self):
        v1 = _fixture_v1()
        v2, _ = build_v2_manifest(v1, frozenset(), train_products=[],
                                  seed=V2_SEED)
        v2["groups"][0]["candidates"][2]["candidate_smiles"] = "junk(("
        audit = audit_v2_manifest(v1, v2)
        names = {c["name"]: c["passed"] for c in audit["checks"]}
        assert names["all_candidates_parseable"] is False

    def test_detects_copied_source_mutation(self):
        v1 = _fixture_v1()
        v2, _ = build_v2_manifest(v1, frozenset(), train_products=[],
                                  seed=V2_SEED)
        for c in v2["groups"][0]["candidates"]:
            if c["candidate_source"] == "random_mismatch":
                c["candidate_smiles"] = "CO"
        audit = audit_v2_manifest(v1, v2)
        names = {c["name"]: c["passed"] for c in audit["checks"]}
        assert names["copied_sources_byte_identical"] is False


# ---------------------------------------------------------------------------
# Known-positive pool
# ---------------------------------------------------------------------------

class TestKnownPositivePool:
    def test_pool_includes_csv_products_and_golds(self, tmp_path):
        csv_path = tmp_path / "hte.csv"
        csv_path.write_text(
            "products,split\n[CH3:1][OH:2],train\nc1ccccc1,test\n")
        v1 = _fixture_v1()
        pool = build_known_positive_pool(csv_path, v1)
        assert _canon("CO") in pool
        assert _canon("c1ccccc1") in pool
        # gold of group hte_g1 (atom-mapped acetic acid)
        assert _canon("CC(=O)O") in pool
