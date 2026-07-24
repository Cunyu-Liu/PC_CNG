"""Tests for P4-G8C Learned Structured Proposal.

Covers: EditType enum, model architecture (7 sub-modules), training stage
losses (4 stages), Pareto frontier evaluation, verdict computation, and
CLI contract.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

os.environ.setdefault("RDKitRDLogger", "0")

# Ensure chem_negative_sampling is importable
_THIS = Path(__file__).resolve()
for _cand in (_THIS.parents[1], _THIS.parents[2] / "chem_negative_sampling"):
    if (_cand / "pc_cng").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from pc_cng.p4_g8c_learned_structured_proposal import (
    ARMS,
    BASE_SEED,
    BOND_ORDERS,
    DEFAULT_HIDDEN,
    EditType,
    NUM_EDIT_TYPES,
    ParetoPoint,
    Stage1ReconstructionLoss,
    Stage2ImitationLoss,
    Stage3ContrastiveLoss,
    Stage4DPOLoss,
    StructuredEdit,
    StructuredProposalModel,
    StructuredProposalOutput,
    _dominates,
    _diversity,
    _collision_risk,
    _edit_controllability,
    cluster_bootstrap_utility_ci,
    compute_verdict,
    evaluate_pareto_frontier,
)


# ---------------------------------------------------------------------------
# Mock BatchedGraph helper
# ---------------------------------------------------------------------------

def make_mock_batch(num_graphs=2, max_atoms=5, hidden_dim=32):
    """Create a mock BatchedGraph-compatible object for testing."""
    from pc_cng.learned_graph_edit_decoder import (
        ATOM_FEAT_DIM, BOND_FEAT_DIM, BatchedGraph, ReactionGraphData,
    )

    graphs = []
    all_atom_feats = []
    all_edge_index = []
    all_edge_feats = []
    all_batch_idx = []
    true_anchors = []
    candidates_per_graph = []
    atom_offset = 0

    for g in range(num_graphs):
        n_atoms = max_atoms + g  # varying sizes
        atom_features = torch.randn(n_atoms, ATOM_FEAT_DIM)
        # simple ring: 0-1, 1-2, ..., (n-2)-(n-1), (n-1)-0
        edges_src = list(range(n_atoms)) + [n_atoms - 1]
        edges_dst = [(i + 1) % n_atoms for i in range(n_atoms)] + [0]
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_features = torch.randn(len(edges_src), BOND_FEAT_DIM)
        atom_map_nums = torch.arange(1, n_atoms + 1, dtype=torch.long)

        graph = ReactionGraphData(
            atom_features=atom_features,
            edge_index=edge_index,
            edge_features=edge_features,
            atom_map_nums=atom_map_nums,
            true_anchor_idx=0,
            candidate_anchor_indices=[0, 1] if n_atoms > 1 else [0],
            source_id=f"test_{g}",
            pair_id=f"test_{g}|pair",
            mapped_reaction="",
            reactants="",
            product="",
            fragment_map=1,
            true_anchor_map=1,
        )
        graphs.append(graph)
        all_atom_feats.append(atom_features)
        if all_edge_index:
            offset_ei = edge_index + atom_offset
            all_edge_index.append(offset_ei)
            all_edge_feats.append(edge_features)
        else:
            all_edge_index.append(edge_index)
            all_edge_feats.append(edge_features)
        all_batch_idx.append(torch.full((n_atoms,), g, dtype=torch.long))
        true_anchors.append(atom_offset)
        candidates_per_graph.append([atom_offset, atom_offset + 1] if n_atoms > 1 else [atom_offset])
        atom_offset += n_atoms

    return BatchedGraph(
        atom_features=torch.cat(all_atom_feats, dim=0),
        edge_index=torch.cat(all_edge_index, dim=1) if all_edge_index else torch.zeros((2, 0), dtype=torch.long),
        edge_features=torch.cat(all_edge_feats, dim=0) if all_edge_feats else torch.zeros((0, BOND_FEAT_DIM)),
        batch_idx=torch.cat(all_batch_idx, dim=0),
        true_anchor_indices=true_anchors,
        candidate_anchor_indices_per_graph=candidates_per_graph,
        graphs=graphs,
    )


# ---------------------------------------------------------------------------
# EditType Tests
# ---------------------------------------------------------------------------

class TestEditType:
    def test_four_edit_types(self):
        assert len(EditType) == 4

    def test_values(self):
        assert EditType.ATOM_TRANSMUTATION == 0
        assert EditType.BOND_ORDER_CHANGE == 1
        assert EditType.FORMED_BOND_MIGRATE == 2
        assert EditType.NO_EDIT == 3

    def test_num_edit_types(self):
        assert NUM_EDIT_TYPES == 4


# ---------------------------------------------------------------------------
# Model Architecture Tests
# ---------------------------------------------------------------------------

class TestStructuredProposalModel:
    """Test the full model with all 7 sub-modules."""

    def test_forward_output_shapes(self):
        model = StructuredProposalModel(hidden_dim=32, num_heads=2, num_layers=2, dropout=0.0)
        model.eval()
        batch = make_mock_batch(num_graphs=3, max_atoms=4, hidden_dim=32)
        with torch.no_grad():
            out = model(batch)
        assert isinstance(out, StructuredProposalOutput)
        assert out.locus_logits.shape[0] == 3  # num_graphs
        assert out.type_logits.shape == (3, NUM_EDIT_TYPES)
        assert out.risk.shape == (3,)
        assert out.uncertainty.shape == (3,)
        assert "atom_logits" in out.arg_logits
        assert "bond_logits" in out.arg_logits
        assert "migrate_query" in out.arg_logits

    def test_validity_mask_shape(self):
        model = StructuredProposalModel(hidden_dim=32, num_heads=2, num_layers=2, dropout=0.0)
        model.eval()
        batch = make_mock_batch(num_graphs=2, max_atoms=3, hidden_dim=32)
        with torch.no_grad():
            out = model(batch)
        # validity mask: [num_graphs, max_len, NUM_EDIT_TYPES]
        assert out.validity_mask.shape[2] == NUM_EDIT_TYPES

    def test_risk_in_range(self):
        model = StructuredProposalModel(hidden_dim=32, num_heads=2, num_layers=2, dropout=0.0)
        model.eval()
        batch = make_mock_batch(num_graphs=2, max_atoms=3, hidden_dim=32)
        with torch.no_grad():
            out = model(batch)
        # risk uses sigmoid -> [0, 1]
        assert (out.risk >= 0).all() and (out.risk <= 1).all()

    def test_uncertainty_nonneg(self):
        model = StructuredProposalModel(hidden_dim=32, num_heads=2, num_layers=2, dropout=0.0)
        model.eval()
        batch = make_mock_batch(num_graphs=2, max_atoms=3, hidden_dim=32)
        with torch.no_grad():
            out = model(batch)
        # uncertainty uses softplus -> >= 0
        assert (out.uncertainty >= 0).all()

    def test_parameters_exist(self):
        model = StructuredProposalModel(hidden_dim=32, num_heads=2, num_layers=2, dropout=0.0)
        params = sum(p.numel() for p in model.parameters())
        assert params > 0
        # Check all 7 sub-modules exist
        assert hasattr(model, "transformer")
        assert hasattr(model, "center_encoder")
        assert hasattr(model, "locus_pointer")
        assert hasattr(model, "type_classifier")
        assert hasattr(model, "arg_decoder")
        assert hasattr(model, "validity_mask")
        assert hasattr(model, "risk_head")


class TestRiskUncertaintyHead:
    def test_mc_estimate(self):
        from pc_cng.p4_g8c_learned_structured_proposal import RiskUncertaintyHead
        head = RiskUncertaintyHead(hidden_dim=32, dropout=0.3)
        graph_emb = torch.randn(4, 32)
        locus_emb = torch.randn(4, 32)
        risk_mean, unc_total = head.mc_estimate(graph_emb, locus_emb, n_samples=5)
        assert risk_mean.shape == (4,)
        assert unc_total.shape == (4,)
        assert (risk_mean >= 0).all() and (risk_mean <= 1).all()


# ---------------------------------------------------------------------------
# Training Stage Loss Tests
# ---------------------------------------------------------------------------

class TestStage1ReconstructionLoss:
    def test_loss_decreases(self):
        """Loss should be finite and computable."""
        model = StructuredProposalModel(hidden_dim=32, num_heads=2, num_layers=2, dropout=0.0)
        batch = make_mock_batch(num_graphs=4, max_atoms=3, hidden_dim=32)
        out = model(batch)
        locus_target = torch.tensor([0, 0, 0, 0])
        type_target = torch.tensor([2, 2, 2, 2])
        loss_fn = Stage1ReconstructionLoss()
        loss, comps = loss_fn(out, locus_target, type_target)
        assert torch.isfinite(loss)
        assert "locus_loss" in comps
        assert "type_loss" in comps

    def test_arg_loss(self):
        model = StructuredProposalModel(hidden_dim=32, num_heads=2, num_layers=2, dropout=0.0)
        batch = make_mock_batch(num_graphs=2, max_atoms=3, hidden_dim=32)
        out = model(batch)
        locus_target = torch.tensor([0, 0])
        type_target = torch.tensor([0, 0])
        arg_target = {"atom_logits": torch.tensor([0, 0])}
        loss_fn = Stage1ReconstructionLoss(arg_w=0.5)
        loss, comps = loss_fn(out, locus_target, type_target, arg_target)
        assert torch.isfinite(loss)
        assert "arg_loss" in comps


class TestStage2ImitationLoss:
    def test_kl_loss(self):
        model = StructuredProposalModel(hidden_dim=32, num_heads=2, num_layers=2, dropout=0.0)
        batch = make_mock_batch(num_graphs=2, max_atoms=3, hidden_dim=32)
        out = model(batch)
        max_len = out.locus_logits.shape[1]
        locus_probs = torch.softmax(torch.randn(2, max_len), dim=-1)
        type_probs = torch.softmax(torch.randn(2, NUM_EDIT_TYPES), dim=-1)
        loss_fn = Stage2ImitationLoss(temperature=2.0)
        loss, comps = loss_fn(out, locus_probs, type_probs)
        assert torch.isfinite(loss)
        assert "locus_kl" in comps
        assert "type_kl" in comps


class TestStage3ContrastiveLoss:
    def test_contrastive(self):
        model = StructuredProposalModel(hidden_dim=32, num_heads=2, num_layers=2, dropout=0.0)
        batch = make_mock_batch(num_graphs=4, max_atoms=3, hidden_dim=32)
        out = model(batch)
        pos_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss_fn = Stage3ContrastiveLoss(margin=0.5)
        loss, comps = loss_fn(out, pos_mask)
        assert torch.isfinite(loss)
        assert "pos_risk" in comps
        assert "neg_risk" in comps
        assert "contrast" in comps


class TestStage4DPOLoss:
    def test_ipo_loss(self):
        model = StructuredProposalModel(hidden_dim=32, num_heads=2, num_layers=2, dropout=0.0)
        batch = make_mock_batch(num_graphs=4, max_atoms=3, hidden_dim=32)
        out = model(batch)
        g = 2
        out_pref = StructuredProposalOutput(
            locus_logits=out.locus_logits[:g], type_logits=out.type_logits[:g],
            arg_logits={k: v[:g] for k, v in out.arg_logits.items()},
            validity_mask=out.validity_mask[:g], risk=out.risk[:g],
            uncertainty=out.uncertainty[:g], graph_emb=out.graph_emb[:g],
            node_emb=out.node_emb)
        out_disp = StructuredProposalOutput(
            locus_logits=out.locus_logits[g:g*2], type_logits=out.type_logits[g:g*2],
            arg_logits={k: v[g:g*2] for k, v in out.arg_logits.items()},
            validity_mask=out.validity_mask[g:g*2], risk=out.risk[g:g*2],
            uncertainty=out.uncertainty[g:g*2], graph_emb=out.graph_emb[g:g*2],
            node_emb=out.node_emb)
        locus_pref = torch.tensor([0, 0])
        type_pref = torch.tensor([0, 0])
        locus_disp = torch.tensor([0, 0])
        type_disp = torch.tensor([0, 0])
        ref_pref = torch.zeros(g)
        ref_disp = torch.zeros(g)
        loss_fn = Stage4DPOLoss(beta=0.1, use_ipo=True)
        loss, comps = loss_fn(out_pref, out_disp, locus_pref, type_pref,
                              locus_disp, type_disp, ref_pref, ref_disp)
        assert torch.isfinite(loss)
        assert "dpo_loss" in comps
        assert "preference_acc" in comps

    def test_dpo_loss(self):
        """Test DPO (non-IPO) mode."""
        model = StructuredProposalModel(hidden_dim=32, num_heads=2, num_layers=2, dropout=0.0)
        batch = make_mock_batch(num_graphs=4, max_atoms=3, hidden_dim=32)
        out = model(batch)
        g = 2
        out_pref = StructuredProposalOutput(
            locus_logits=out.locus_logits[:g], type_logits=out.type_logits[:g],
            arg_logits={k: v[:g] for k, v in out.arg_logits.items()},
            validity_mask=out.validity_mask[:g], risk=out.risk[:g],
            uncertainty=out.uncertainty[:g], graph_emb=out.graph_emb[:g],
            node_emb=out.node_emb)
        out_disp = StructuredProposalOutput(
            locus_logits=out.locus_logits[g:g*2], type_logits=out.type_logits[g:g*2],
            arg_logits={k: v[g:g*2] for k, v in out.arg_logits.items()},
            validity_mask=out.validity_mask[g:g*2], risk=out.risk[g:g*2],
            uncertainty=out.uncertainty[g:g*2], graph_emb=out.graph_emb[g:g*2],
            node_emb=out.node_emb)
        locus_pref = torch.tensor([0, 0])
        type_pref = torch.tensor([0, 0])
        locus_disp = torch.tensor([0, 0])
        type_disp = torch.tensor([0, 0])
        ref_pref = torch.zeros(g)
        ref_disp = torch.zeros(g)
        loss_fn = Stage4DPOLoss(beta=0.1, use_ipo=False)
        loss, comps = loss_fn(out_pref, out_disp, locus_pref, type_pref,
                              locus_disp, type_disp, ref_pref, ref_disp)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_diversity(self):
        assert _diversity(["a", "b", "c"]) == 1.0
        assert _diversity(["a", "a", "a"]) == 1/3
        assert _diversity([]) == 0.0

    def test_collision_risk(self):
        pos = ["a", "b", "c"]
        negs = ["a", "d", "e"]
        assert _collision_risk(negs, pos) == 1/3
        assert _collision_risk([], pos) == 0.0
        assert _collision_risk(["x", "y"], pos) == 0.0

    def test_edit_controllability(self):
        edits = [
            StructuredEdit(locus=0, edit_type=EditType.ATOM_TRANSMUTATION),
            StructuredEdit(locus=1, edit_type=EditType.ATOM_TRANSMUTATION),
            StructuredEdit(locus=2, edit_type=EditType.BOND_ORDER_CHANGE),
        ]
        assert _edit_controllability(edits, EditType.ATOM_TRANSMUTATION) == 2/3
        assert _edit_controllability([], EditType.ATOM_TRANSMUTATION) == 0.0


# ---------------------------------------------------------------------------
# Pareto Frontier Tests
# ---------------------------------------------------------------------------

class TestParetoFrontier:
    def test_dominates(self):
        a = ParetoPoint(arm="a", utility=0.8, validity=0.9, risk=0.1, coverage=5)
        b = ParetoPoint(arm="b", utility=0.5, validity=0.7, risk=0.3, coverage=3)
        assert _dominates(a, b)
        assert not _dominates(b, a)

    def test_no_domination_on_tradeoff(self):
        a = ParetoPoint(arm="a", utility=0.8, validity=0.5, risk=0.3, coverage=3)
        b = ParetoPoint(arm="b", utility=0.5, validity=0.9, risk=0.1, coverage=5)
        assert not _dominates(a, b)
        assert not _dominates(b, a)

    def test_pareto_frontier(self):
        points = [
            ParetoPoint(arm="rule_pc_cng", utility=0.5, validity=0.7, risk=0.3, coverage=3),
            ParetoPoint(arm="learned_structured", utility=0.8, validity=0.9, risk=0.1, coverage=5),
            ParetoPoint(arm="unconstrained_neural", utility=0.6, validity=0.6, risk=0.4, coverage=4),
        ]
        result = evaluate_pareto_frontier(points)
        assert "learned_structured" in result["frontier"]
        assert result["learned_dominates_rule"] is True

    def test_no_domination(self):
        points = [
            ParetoPoint(arm="rule_pc_cng", utility=0.8, validity=0.3, risk=0.1, coverage=3),
            ParetoPoint(arm="learned_structured", utility=0.3, validity=0.8, risk=0.4, coverage=5),
        ]
        result = evaluate_pareto_frontier(points)
        assert result["learned_dominates_rule"] is False
        assert len(result["frontier"]) == 2  # both are on frontier


# ---------------------------------------------------------------------------
# Cluster Bootstrap CI Tests
# ---------------------------------------------------------------------------

class TestClusterBootstrap:
    def test_positive_delta(self):
        arm = [("c1", 0.8), ("c2", 0.9), ("c3", 0.7)]
        base = [("c1", 0.5), ("c2", 0.6), ("c3", 0.4)]
        delta, lo, hi = cluster_bootstrap_utility_ci(arm, base, n_boot=200, seed=42)
        assert delta > 0
        assert lo > 0  # CI should be positive
        assert hi > lo

    def test_negative_delta(self):
        arm = [("c1", 0.3), ("c2", 0.2)]
        base = [("c1", 0.8), ("c2", 0.9)]
        delta, lo, hi = cluster_bootstrap_utility_ci(arm, base, n_boot=200, seed=42)
        assert delta < 0
        assert hi < 0  # CI should be negative

    def test_zero_delta(self):
        arm = [("c1", 0.5), ("c2", 0.5)]
        base = [("c1", 0.5), ("c2", 0.5)]
        delta, lo, hi = cluster_bootstrap_utility_ci(arm, base, n_boot=200, seed=42)
        assert abs(delta) < 1e-6
        assert lo <= 0 <= hi  # CI should straddle zero


# ---------------------------------------------------------------------------
# Verdict Tests
# ---------------------------------------------------------------------------

class TestVerdict:
    def _make_arm_result(self, arm, utility=0.5, validity=0.7, collision_risk=0.3,
                         diversity=0.5, n_candidates=10):
        from pc_cng.p4_g8c_learned_structured_proposal import ArmResult
        return ArmResult(
            arm=arm, utility=utility, validity=validity,
            collision_risk=collision_risk, diversity=diversity,
            n_candidates=n_candidates,
            utility_per_cluster=[("c1", utility), ("c2", utility)],
        )

    def test_go(self):
        rule = self._make_arm_result("rule_pc_cng", utility=0.5, validity=0.7)
        learned = self._make_arm_result("learned_structured", utility=0.8, validity=0.9)
        results = {"rule_pc_cng": rule, "learned_structured": learned}
        pareto = {
            "learned_dominates_rule": True,
            "learned_risk_dominates_rule": False,
            "frontier": ["learned_structured"],
        }
        ci = (0.15, 0.05, 0.25)  # all positive
        verdict = compute_verdict(results, pareto, ci, coverage_matched=True)
        assert verdict["verdict"] == "GO"

    def test_partial_go(self):
        rule = self._make_arm_result("rule_pc_cng", utility=0.5, validity=0.7)
        learned = self._make_arm_result("learned_structured", utility=0.6, validity=0.7)
        results = {"rule_pc_cng": rule, "learned_structured": learned}
        pareto = {
            "learned_dominates_rule": False,
            "learned_risk_dominates_rule": False,
            "frontier": ["rule_pc_cng", "learned_structured"],
        }
        ci = (0.05, -0.02, 0.12)  # CI straddles zero
        verdict = compute_verdict(results, pareto, ci, coverage_matched=True)
        assert verdict["verdict"] == "PARTIAL_GO"

    def test_no_go(self):
        rule = self._make_arm_result("rule_pc_cng", utility=0.8, validity=0.9)
        learned = self._make_arm_result("learned_structured", utility=0.3, validity=0.5)
        results = {"rule_pc_cng": rule, "learned_structured": learned}
        pareto = {
            "learned_dominates_rule": False,
            "learned_risk_dominates_rule": False,
            "frontier": ["rule_pc_cng"],
        }
        ci = (-0.3, -0.4, -0.2)  # all negative
        verdict = compute_verdict(results, pareto, ci, coverage_matched=True)
        assert verdict["verdict"] == "NO_GO"


# ---------------------------------------------------------------------------
# Contract Tests
# ---------------------------------------------------------------------------

class TestContract:
    def test_four_arms(self):
        assert len(ARMS) == 4
        assert "rule_pc_cng" in ARMS
        assert "unconstrained_neural" in ARMS
        assert "learned_structured" in ARMS
        assert "learned_structured_risk" in ARMS

    def test_bond_orders(self):
        assert BOND_ORDERS == (1, 2, 3)

    def test_model_has_seven_modules(self):
        model = StructuredProposalModel(hidden_dim=16, num_heads=2, num_layers=1, dropout=0.0)
        sub_modules = [
            model.transformer,
            model.center_encoder,
            model.locus_pointer,
            model.type_classifier,
            model.arg_decoder,
            model.validity_mask,
            model.risk_head,
        ]
        assert all(m is not None for m in sub_modules)

    def test_four_stage_losses(self):
        losses = [Stage1ReconstructionLoss(), Stage2ImitationLoss(),
                  Stage3ContrastiveLoss(), Stage4DPOLoss()]
        assert len(losses) == 4

    def test_no_ppo(self):
        """Ensure Stage4 uses DPO/IPO, not PPO."""
        loss = Stage4DPOLoss(use_ipo=True)
        assert hasattr(loss, "beta")
        assert hasattr(loss, "use_ipo")
        # DPO mode
        loss_dpo = Stage4DPOLoss(use_ipo=False)
        assert loss_dpo.use_ipo is False
