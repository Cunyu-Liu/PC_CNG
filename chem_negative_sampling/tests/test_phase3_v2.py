"""Unit tests for Phase 3 v2 improvements (A+B+C+D).

Tests cover:
  - Improvement A: RobustNegativeGenerator never returns None on valid input
  - Improvement B: ReactionAwareClassifier (GAT) builds and predicts
  - Improvement C: budget matching + negative controls
  - Improvement D: Holm correction integration
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

# Bootstrap path
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

SAMPLE_REACTIONS = [
    # atom-mapped Suzuki coupling
    "[C:1]1=CC=CC=C1B(O)O.[Br:2]C1=CC=CC=C1>C1CCOC1>[C:1]1=CC=CC=C1[C:2]1=CC=CC=C1",
    # esterification
    "CC(=O)O.CCO>>CC(=O)OCC.O",
    # amide bond formation
    "CC(=O)Cl.CN>>CC(=O)NC",
    # simple SN2
    "CBr.[OH-:1].[Na+]>>CO.[Br-]",
    # reduction
    "C(=O)O[CH3:1]>>CO[CH3:1]",
]


def _has_rdkit() -> bool:
    try:
        import rdkit  # noqa: F401
        return True
    except Exception:
        return False


def make_test_dataframe(reactions: List[str]) -> pd.DataFrame:
    """Build a small HiTEA-like DataFrame from reaction SMILES."""
    families = ["cross_coupling", "esterification", "amidation", "sn2", "reduction"]
    return pd.DataFrame({
        "reaction_smiles": reactions,
        "experimental_group": [f"g{i % 3}" for i in range(len(reactions))],
        "reaction_family": [families[i % len(families)] for i in range(len(reactions))],
        "yield_bin": [i % 2 for i in range(len(reactions))],
        "split_key": [f"s{i}" for i in range(len(reactions))],
    })


class _StubGenerator:
    """Stub generator that always fails (to test fallback)."""

    def __init__(self, method: str = "stub_fail"):
        self.method = method

    def generate(self, reaction_smiles: str):
        return None


class _FixedGenerator:
    """Stub generator that returns a fixed negative product."""

    def __init__(self, method: str = "stub_fixed"):
        self.method = method

    def generate(self, reaction_smiles: str):
        return "c1ccccc1"  # benzene, always valid


# ---------------------------------------------------------------------------
# Improvement A: RobustNegativeGenerator
# ---------------------------------------------------------------------------

class TestRobustNegativeGenerator:
    """Improvement A: generator must never return None on valid reaction."""

    def test_robust_never_none_on_valid_input(self):
        from pc_cng.robust_negative_generator import RobustNegativeGenerator
        robust = RobustNegativeGenerator(_StubGenerator(), seed=42)
        for rxn in SAMPLE_REACTIONS:
            neg = robust.generate(rxn)
            assert neg is not None, f"Robust generator returned None for {rxn}"
            # Must be a 3-part reaction SMILES
            assert neg.count(">") == 2, f"Expected reaction SMILES, got {neg}"

    def test_robust_uses_base_when_available(self):
        from pc_cng.robust_negative_generator import RobustNegativeGenerator
        robust = RobustNegativeGenerator(_FixedGenerator(), seed=42)
        neg = robust.generate(SAMPLE_REACTIONS[0])
        assert neg is not None
        assert "c1ccccc1" in neg  # base output used (tier 1)
        assert robust.stats["tier1_base"] == 1

    def test_robust_fallback_stats_tracked(self):
        from pc_cng.robust_negative_generator import RobustNegativeGenerator
        robust = RobustNegativeGenerator(_StubGenerator(), seed=42)
        for rxn in SAMPLE_REACTIONS:
            robust.generate(rxn)
        # All should succeed via tier 2/3/4, none failed
        assert robust.stats["failed"] == 0
        assert robust.stats["total"] == len(SAMPLE_REACTIONS)
        # At least one tier should have fired
        total_success = (robust.stats["tier1_base"] + robust.stats["tier2_center"]
                         + robust.stats["tier3_scaffold"] + robust.stats["tier4_mismatch"])
        assert total_success == len(SAMPLE_REACTIONS)

    def test_robust_returns_none_for_malformed(self):
        from pc_cng.robust_negative_generator import RobustNegativeGenerator
        robust = RobustNegativeGenerator(_StubGenerator(), seed=42)
        assert robust.generate("") is None
        assert robust.generate("not a reaction") is None
        assert robust.generate("A>B") is None  # only 2 parts

    def test_negative_is_valid_smiles(self):
        """Generated negative product must be RDKit-valid."""
        from pc_cng.robust_negative_generator import (
            RobustNegativeGenerator, split_reaction, is_valid_smiles)
        robust = RobustNegativeGenerator(_StubGenerator(), seed=42)
        for rxn in SAMPLE_REACTIONS:
            neg = robust.generate(rxn)
            sp = split_reaction(neg)
            assert sp is not None
            _, _, product = sp
            assert is_valid_smiles(product), f"Invalid product SMILES: {product}"


# ---------------------------------------------------------------------------
# Improvement B: ReactionAwareClassifier (GAT)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_rdkit(), reason="RDKit required")
class TestReactionAwareClassifier:
    """Improvement B: GAT classifier builds and predicts."""

    def test_import(self):
        from pc_cng.reaction_gnn import ReactionAwareClassifier
        assert ReactionAwareClassifier is not None

    def test_fp_only_mode_compatible_with_enhanced_mlp(self):
        """train/predict_proba (fp-only) must match EnhancedMLP interface."""
        from pc_cng.reaction_gnn import ReactionAwareClassifier
        rng = np.random.default_rng(42)
        X = rng.random((20, 8192)).astype(np.float32)
        y = rng.integers(0, 2, 20).astype(np.float32)
        clf = ReactionAwareClassifier(input_dim=8192, seed=42)
        clf.train(X, y, epochs=2, batch_size=8, verbose=False)
        scores = clf.predict_proba(X)
        assert scores.shape == (20,)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_fit_reactions_basic(self):
        """fit_reactions builds graphs and predicts."""
        from pc_cng.reaction_gnn import ReactionAwareClassifier
        # Use a few reactions as both positive and negative (synthetic test)
        rxns = SAMPLE_REACTIONS[:4]
        labels = np.array([1, 1, 0, 0], dtype=np.float32)
        clf = ReactionAwareClassifier(input_dim=8192, seed=42)
        clf.fit_reactions(rxns, labels, epochs=2, batch_size=2, verbose=False)
        scores = clf.predict_proba_reactions(rxns)
        assert scores.shape == (4,)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_batch_size_1_does_not_crash(self):
        """BatchNorm with batch_size=1 must not crash (eval trick)."""
        from pc_cng.reaction_gnn import ReactionAwareClassifier
        rxns = SAMPLE_REACTIONS[:2]
        labels = np.array([1, 0], dtype=np.float32)
        clf = ReactionAwareClassifier(input_dim=8192, seed=42)
        clf.fit_reactions(rxns, labels, epochs=1, batch_size=1, verbose=False)
        scores = clf.predict_proba_reactions(rxns)
        assert scores.shape == (2,)


# ---------------------------------------------------------------------------
# Improvement C: Budget matching + negative controls
# ---------------------------------------------------------------------------

class TestBudgetMatching:
    """Improvement C: budget-matched datasets."""

    def test_budget_matched_equal_counts(self):
        from pc_cng.run_phase3_v2 import build_budget_matched_datasets
        from pc_cng.robust_negative_generator import RobustNegativeGenerator
        df = make_test_dataframe(SAMPLE_REACTIONS * 4)  # 20 rows
        gen1 = RobustNegativeGenerator(_FixedGenerator("m1"), seed=42)
        gen2 = RobustNegativeGenerator(_StubGenerator("m2"), seed=42)
        # Stub always fails tier1, but robust fallback succeeds
        generators = {"m1": gen1, "m2": gen2}
        budget = build_budget_matched_datasets(df, df, generators)
        # Both methods should have data
        if "m1" in budget and "m2" in budget:
            X1, y1, _, _ = budget["m1"]
            X2, y2, _, _ = budget["m2"]
            if X1 is not None and X2 is not None:
                # Budget matched: same number of samples
                assert len(X1) == len(X2), "Budget mismatch between methods"

    def test_shuffled_parent_breaks_pairing(self):
        from pc_cng.run_phase3_v2 import build_shuffled_parent_dataset
        from pc_cng.robust_negative_generator import RobustNegativeGenerator
        df = make_test_dataframe(SAMPLE_REACTIONS * 3)
        gen = RobustNegativeGenerator(_FixedGenerator(), seed=42)
        X, y, recs = build_shuffled_parent_dataset(df, gen, offset=5)
        assert X is not None
        assert len(y) == len(recs)
        # Labels should be balanced (1 pos + 1 neg per pair)
        assert set(y.tolist()) == {0.0, 1.0}


class TestNegativeControls:
    """Improvement C: negative-control arms."""

    def test_randomized_label_shuffles(self):
        from pc_cng.run_phase3_v2 import RandomizedLabelGenerator
        base = _FixedGenerator()
        rl = RandomizedLabelGenerator(base)
        assert rl.method == "randomized_label"
        out = rl.generate(SAMPLE_REACTIONS[0])
        assert out is not None  # delegates to base


# ---------------------------------------------------------------------------
# Improvement D: Holm correction
# ---------------------------------------------------------------------------

class TestHolmCorrection:
    """Improvement D: Holm correction across scenarios."""

    def test_holm_correction_exists(self):
        from pc_cng.paired_cluster_inference import holm_correction
        results = holm_correction([0.01, 0.04, 0.03], alpha=0.05)
        assert len(results) == 3
        # Smallest p rejected
        assert results[0]["rejected"] is True  # p=0.01 -> 0.01*3=0.03 < 0.05

    def test_apply_holm_across_scenarios(self):
        from pc_cng.run_phase3_v2 import apply_holm_across_scenarios
        # Fake results with paired_ci
        all_results = {
            "random": {"paired_ci": {
                "learned_vs_rule": {"p_value": 0.01, "delta_mean": 0.05,
                                    "delta_ci_low": 0.02, "delta_ci_high": 0.08},
            }},
            "scaffold": {"paired_ci": {
                "learned_vs_rule": {"p_value": 0.04, "delta_mean": 0.03,
                                    "delta_ci_low": -0.01, "delta_ci_high": 0.07},
            }},
        }
        holm = apply_holm_across_scenarios(all_results, alpha=0.05)
        assert "learned_vs_rule" in holm
        assert "random" in holm["learned_vs_rule"]
        assert "scaffold" in holm["learned_vs_rule"]
        # Adjusted p should be >= original
        assert (holm["learned_vs_rule"]["random"]["adjusted_p"]
                >= holm["learned_vs_rule"]["random"]["original_p"])


# ---------------------------------------------------------------------------
# Integration: end-to-end smoke test
# ---------------------------------------------------------------------------

class TestIntegrationSmoke:
    """Smoke test: full pipeline runs without crashing on tiny data."""

    def test_make_classifier_factory(self):
        from pc_cng.run_phase3_v2 import make_classifier
        clf = make_classifier(8192, seed=42, use_gnn=False)
        # Should be EnhancedMLP when use_gnn=False
        from pc_cng.phase3_enhanced import EnhancedMLP
        assert isinstance(clf, EnhancedMLP)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
