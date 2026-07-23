"""Tests for P4-G5 risk-aware losses and the risk scorer.

Run with::

    python3 -m pytest chem_negative_sampling/tests/test_risk_aware_loss.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from pc_cng.training.train_risk_aware import (
    METHODS,
    compute_loss,
    hard_binary_loss,
    label_smoothing_loss,
    pu_nnpu_loss,
    risk_weighted_infonce_loss,
    risk_weighted_pairwise_loss,
)
from pc_cng.models.risk_aware_scorer import (
    MIN_WEIGHT,
    RISK_SIGNALS,
    WEIGHT_COMPONENTS,
    FalseNegativeRiskModel,
    atom_mapping_quality,
    build_observed_pool,
    chemical_validity_score,
    compute_sample_weights,
)


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

class TestHardBinary:
    def test_matches_bce(self):
        logits = torch.tensor([0.5, -1.0, 2.0])
        labels = torch.tensor([1.0, 0.0, 1.0])
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        assert torch.isclose(hard_binary_loss(logits, labels), expected)

    def test_gradient_flows(self):
        logits = torch.tensor([0.5, -1.0], requires_grad=True)
        labels = torch.tensor([1.0, 0.0])
        hard_binary_loss(logits, labels).backward()
        assert logits.grad is not None and logits.grad.abs().sum() > 0


class TestLabelSmoothing:
    def test_eps_bounds(self):
        logits = torch.zeros(2)
        labels = torch.tensor([1.0, 0.0])
        with pytest.raises(ValueError):
            label_smoothing_loss(logits, labels, eps=0.5)
        with pytest.raises(ValueError):
            label_smoothing_loss(logits, labels, eps=-0.1)

    def test_smoothed_targets(self):
        logits = torch.tensor([1.0, -1.0])
        labels = torch.tensor([1.0, 0.0])
        eps = 0.1
        smoothed = torch.tensor([0.9, 0.1])
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, smoothed)
        assert torch.isclose(label_smoothing_loss(logits, labels, eps=eps), expected)

    def test_dispatch(self):
        logits = torch.tensor([0.0, 0.0])
        labels = torch.tensor([1.0, 0.0])
        loss = compute_loss("label_smoothing", logits, labels)
        assert loss.item() > 0


class TestPuNnpu:
    def test_pi_bounds(self):
        logits = torch.tensor([1.0, -1.0])
        labels = torch.tensor([1.0, 0.0])
        is_unlabeled = torch.tensor([0.0, 1.0])
        with pytest.raises(ValueError):
            pu_nnpu_loss(logits, labels, is_unlabeled, pi=0.0)
        with pytest.raises(ValueError):
            pu_nnpu_loss(logits, labels, is_unlabeled, pi=1.0)

    def test_requires_p_and_u(self):
        logits = torch.tensor([1.0, 2.0])
        labels = torch.tensor([1.0, 1.0])
        is_unlabeled = torch.tensor([0.0, 0.0])
        with pytest.raises(ValueError):
            pu_nnpu_loss(logits, labels, is_unlabeled, pi=0.1)

    def test_nonnegative_clamp(self):
        # Very confident correct predictions make U-risk - pi*P-risk negative
        logits = torch.tensor([10.0, -10.0])
        labels = torch.tensor([1.0, 0.0])
        is_unlabeled = torch.tensor([0.0, 1.0])
        loss = pu_nnpu_loss(logits, labels, is_unlabeled, pi=0.5)
        assert loss.item() >= 0.0

    def test_dispatch_uses_labels_for_u_mask(self):
        logits = torch.tensor([1.0, -1.0, 0.5, -0.5])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = compute_loss("pu_nnpu", logits, labels, pu_prior=0.2)
        assert loss.item() >= 0.0


class TestRiskWeightedPairwise:
    def test_higher_pos_score_lower_loss(self):
        group_ids = ["g1", "g1"]
        labels = torch.tensor([1.0, 0.0])
        weights = torch.tensor([1.0, 1.0])
        good = risk_weighted_pairwise_loss(torch.tensor([3.0, -3.0]), labels, group_ids, weights)
        bad = risk_weighted_pairwise_loss(torch.tensor([-3.0, 3.0]), labels, group_ids, weights)
        assert good.item() < bad.item()

    def test_zero_weight_zeroes_loss(self):
        group_ids = ["g1", "g1"]
        labels = torch.tensor([1.0, 0.0])
        logits = torch.tensor([-3.0, 3.0])  # badly mis-ranked
        weights = torch.tensor([1.0, 0.0])  # negative fully down-weighted
        loss = risk_weighted_pairwise_loss(logits, labels, group_ids, weights)
        assert loss.item() == pytest.approx(0.0, abs=1e-8)

    def test_invalid_margin(self):
        with pytest.raises(ValueError):
            risk_weighted_pairwise_loss(
                torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]),
                ["g", "g"], torch.ones(2), margin=0.0,
            )

    def test_no_pairs_raises(self):
        with pytest.raises(ValueError):
            risk_weighted_pairwise_loss(
                torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]),
                ["g1", "g2"], torch.ones(2),
            )

    def test_multi_group(self):
        logits = torch.tensor([2.0, -1.0, 1.0, -2.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        group_ids = ["g1", "g1", "g2", "g2"]
        weights = torch.tensor([1.0, 0.5, 1.0, 0.5])
        loss = risk_weighted_pairwise_loss(logits, labels, group_ids, weights)
        assert loss.item() > 0


class TestRiskWeightedInfoNCE:
    def test_perfect_ranking_low_loss(self):
        logits = torch.tensor([5.0, -5.0, -5.0])
        labels = torch.tensor([1.0, 0.0, 0.0])
        group_ids = ["g", "g", "g"]
        weights = torch.ones(3)
        loss = risk_weighted_infonce_loss(logits, labels, group_ids, weights)
        assert loss.item() < 0.01

    def test_weight_reduces_negative_pull(self):
        logits = torch.tensor([0.0, 0.0])
        labels = torch.tensor([1.0, 0.0])
        group_ids = ["g", "g"]
        hi = risk_weighted_infonce_loss(logits, labels, group_ids, torch.tensor([1.0, 1.0]))
        lo = risk_weighted_infonce_loss(logits, labels, group_ids, torch.tensor([1.0, 0.01]))
        assert lo.item() < hi.item()

    def test_invalid_tau(self):
        with pytest.raises(ValueError):
            risk_weighted_infonce_loss(
                torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]),
                ["g", "g"], torch.ones(2), tau=0.0,
            )

    def test_incomplete_group_raises(self):
        with pytest.raises(ValueError):
            risk_weighted_infonce_loss(
                torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]),
                ["g1", "g2"], torch.ones(2),
            )

    def test_dispatch(self):
        logits = torch.tensor([1.0, -1.0])
        labels = torch.tensor([1.0, 0.0])
        loss = compute_loss(
            "risk_weighted_infonce", logits, labels,
            group_ids=["g", "g"], weights=torch.ones(2),
        )
        assert loss.item() > 0


class TestDispatch:
    def test_all_methods_present(self):
        assert set(METHODS) == {
            "hard_binary", "label_smoothing", "pu_nnpu",
            "risk_weighted_pairwise", "risk_weighted_infonce",
        }

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            compute_loss("not_a_method", torch.zeros(2), torch.zeros(2))


# ---------------------------------------------------------------------------
# Risk scorer unit pieces
# ---------------------------------------------------------------------------

class TestChemHelpers:
    def test_chemical_validity(self):
        assert chemical_validity_score("CCO") == 1.0
        assert chemical_validity_score("not_a_smiles###") == 0.0
        assert chemical_validity_score("") == 0.0

    def test_atom_mapping_quality(self):
        assert atom_mapping_quality("mapped") == 1.0
        assert atom_mapping_quality("unmapped") == 0.5
        assert atom_mapping_quality("garbage") == 0.0


def _feat(**overrides):
    base = {
        "database_exact_collision": 0.0,
        "template_collision": 0.0,
        "nearest_positive_similarity": 0.5,
        "nearest_negative_similarity": 0.5,
        "ensemble_mean": 0.5,
        "ensemble_variance": 0.1,
        "epistemic_uncertainty": 0.1,
        "aleatoric_uncertainty": 0.1,
        "reaction_family_support": 0.5,
        "edit_locality": 0.5,
        "edit_distance": 0.5,
        "atom_mapping_quality": 1.0,
        "experimental_support": 0.5,
        "chemical_validity": 1.0,
    }
    base.update(overrides)
    return base


class TestFalseNegativeRiskModel:
    def test_fit_separable(self):
        pos = [_feat(nearest_positive_similarity=0.9, ensemble_mean=0.9) for _ in range(30)]
        neg = [_feat(nearest_positive_similarity=0.1, ensemble_mean=0.1) for _ in range(30)]
        model = FalseNegativeRiskModel()
        metrics = model.fit(pos, neg, epochs=300, lr=0.5)
        assert metrics["train_auroc"] == pytest.approx(1.0)
        p_pos = model.predict_fnr([_feat(nearest_positive_similarity=0.9, ensemble_mean=0.9)])[0]
        p_neg = model.predict_fnr([_feat(nearest_positive_similarity=0.1, ensemble_mean=0.1)])[0]
        assert p_pos > 0.8 > p_neg

    def test_serialization_roundtrip(self):
        pos = [_feat(ensemble_mean=0.8) for _ in range(10)]
        neg = [_feat(ensemble_mean=0.2) for _ in range(10)]
        model = FalseNegativeRiskModel()
        model.fit(pos, neg, epochs=50)
        d = model.to_dict()
        model2 = FalseNegativeRiskModel.from_dict(d)
        f = [_feat(ensemble_mean=0.5)]
        assert model.predict_fnr(f) == pytest.approx(model2.predict_fnr(f))
        assert json.dumps(d)  # JSON-serializable

    def test_feature_names_order(self):
        model = FalseNegativeRiskModel()
        assert model.feature_names == RISK_SIGNALS


class TestSampleWeights:
    @staticmethod
    def _two_feats(**kw0):
        """Two candidates with distinct uncertainty (non-degenerate boundary)."""
        f0 = _feat(chemical_validity=1.0, reaction_family_support=1.0,
                   experimental_support=1.0, epistemic_uncertainty=0.8)
        f0.update(kw0)
        f1 = _feat(epistemic_uncertainty=0.2)
        return [f0, f1]

    def test_product_form(self):
        feats = self._two_feats()
        out = compute_sample_weights(feats, [0.25, 0.25])
        rec = out[0]  # highest-uncertainty item -> boundary normalised to 1.0
        assert rec["false_negative_risk"] == pytest.approx(0.25)
        assert rec["one_minus_fnr"] == pytest.approx(0.75)
        assert rec["boundary_value"] == pytest.approx(1.0)
        assert rec["data_support"] == pytest.approx(1.0)
        expected = 1.0 * 1.0 * 1.0 * 0.75
        assert rec["sample_weight"] == pytest.approx(expected)
        # min-max normalisation puts the low-uncertainty item at boundary 0
        assert out[1]["boundary_value"] == pytest.approx(0.0)
        assert out[1]["sample_weight"] == MIN_WEIGHT

    def test_known_positive_collision_override(self):
        feats = self._two_feats()
        out = compute_sample_weights(feats, [0.1, 0.1], known_positive_collision=[True, False])
        assert out[0]["false_negative_risk"] == 1.0
        assert out[0]["sample_weight"] == MIN_WEIGHT

    def test_ablation_neutralizes_component(self):
        feats = self._two_feats(chemical_validity=0.0)  # would zero item 0's weight
        full = compute_sample_weights(feats, [0.5, 0.5])[0]["sample_weight"]
        abl = compute_sample_weights(feats, [0.5, 0.5], ablate=["chemical_validity"])[0]["sample_weight"]
        assert full == MIN_WEIGHT
        assert abl > full

    def test_unknown_ablation_raises(self):
        with pytest.raises(ValueError):
            compute_sample_weights([_feat()], [0.5], ablate=["not_a_component"])

    def test_all_components_ablatable(self):
        for comp in WEIGHT_COMPONENTS:
            out = compute_sample_weights(self._two_feats(), [0.5, 0.5], ablate=[comp])
            assert out[0]["sample_weight"] >= MIN_WEIGHT


class TestObservedPool:
    def test_exclusion_and_labels(self, tmp_path):
        csv_path = tmp_path / "htea.csv"
        csv_path.write_text(
            "source_id,reaction_smiles,reactants,agents,products,label_type,yield,source,split_key,split,reaction_class\n"
            "r1,C>>CCO,,,CCO,exp,50,x,sk1,train,famA\n"
            "r2,C>>CCN,,,CCN,exp,0,x,sk2,train,famA\n"
            "r3,C>>CCC,,,CCC,exp,10,x,sk3,val,famB\n"
        )
        pool = build_observed_pool(csv_path, frozenset({"sk3"}), seed=1)
        assert pool.pos_smiles and pool.neg_smiles
        assert all("CCC" not in s for s in pool.pos_smiles + pool.neg_smiles)
        assert pool.family_counts.get("famA") == 2
        assert "famB" not in pool.family_counts
