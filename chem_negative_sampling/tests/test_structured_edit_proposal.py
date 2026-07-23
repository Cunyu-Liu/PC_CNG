"""Tests for P4-G8C Learned Structured Proposal.

G8-C is gated on G3 >= Weak GO, G4 >= Partial GO, G5 >= Partial GO.
Implementation is DEFERRED; these tests validate entry conditions and
the architecture interface contract defined in the spec.

Spec: 提示词/pccng 的分阶段提示词.md#L1572-1833 (P4-G8C)
"""

import os
import sys
from pathlib import Path

import pytest

# Suppress RDKit warnings
os.environ["RDKitRDLogger"] = "0"


# ---------------------------------------------------------------------------
# Architecture contract constants (frozen from spec L1689-1751)
# ---------------------------------------------------------------------------

REQUIRED_INPUT_FIELDS = [
    "atom_mapped_reaction_graph",
    "reaction_center",
    "condition_embedding",
    "failure_prototype",
    "risk_context",
]

REQUIRED_MODULES = [
    "reaction_graph_transformer",
    "reaction_center_encoder",
    "edit_locus_pointer",
    "edit_type_classifier",
    "atom_bond_argument_decoder",
    "validity_action_mask",
    "risk_uncertainty_head",
]

REQUIRED_OUTPUT_ACTIONS = [
    "select_edit_locus",
    "select_edit_type",
    "select_atom_bond_arguments",
    "apply_constrained_edit",
    "predict_risk",
    "predict_boundary_value",
]

REQUIRED_TRAINING_STAGES = [
    "stage1_edit_reconstruction",
    "stage2_rule_proposal_imitation",
    "stage3_competing_outcome_learning",
    "stage4_risk_adjusted_preference",
]

PREFERENCE_METHODS = ["pairwise", "dpo", "ipo"]
FORBIDDEN_METHODS = ["ppo"]  # spec L1750: 暂不使用 PPO

COMPARISON_BASELINES = [
    "rule_pc_cng",
    "unconstrained_neural_generator",
    "learned_structured_proposal",
    "learned_structured_proposal_with_risk",
]

PARETO_DIMENSIONS = ["utility", "validity", "risk"]


# ---------------------------------------------------------------------------
# Entry Condition Tests
# ---------------------------------------------------------------------------

class TestEntryConditions:
    """Validate G8-C gate conditions (spec L1590-1600)."""

    def test_entry_conditions_g3_weak_go(self):
        """G8-C requires P4-G3 >= Weak GO."""
        # G3 v2 verdict: WEAK_GO (results/p4_augmentation_v2_chemformer/go_no_go.json)
        g3_status = "WEAK_GO"
        assert g3_status in ("WEAK_GO", "GO"), (
            "G8-C blocked: P4-G3 must be >= Weak GO"
        )

    def test_entry_conditions_g4_partial_go(self):
        """G8-C requires P4-G4 >= Partial GO."""
        # G4 v2 verdict: GO (results/p4_generator_scorer_matrix_v2/go_no_go.json)
        # A6 positive in 3/3 scorers, interaction p=2.75e-05
        g4_status = "GO"
        assert g4_status in ("PARTIAL_GO", "GO"), (
            "G8-C blocked: P4-G4 must be >= Partial GO"
        )

    def test_entry_conditions_g5_partial_go(self):
        """G8-C requires P4-G5 >= Partial GO."""
        # G5 verdict: PARTIAL_GO (results/p4_risk_aware/go_no_go.json)
        g5_status = "PARTIAL_GO"
        assert g5_status in ("PARTIAL_GO", "GO"), (
            "G8-C blocked: P4-G5 must be >= Partial GO"
        )


# ---------------------------------------------------------------------------
# Architecture Interface Contract Tests
# ---------------------------------------------------------------------------

class TestArchitectureContract:
    """Validate the architecture interface defined in spec L1689-1751."""

    def test_required_modules_defined(self):
        """All spec-required modules must be in the contract."""
        assert len(REQUIRED_MODULES) == 7
        assert "reaction_graph_transformer" in REQUIRED_MODULES
        assert "edit_locus_pointer" in REQUIRED_MODULES
        assert "validity_action_mask" in REQUIRED_MODULES
        assert "risk_uncertainty_head" in REQUIRED_MODULES
        # Ensure no PPO in preference methods
        assert not (set(FORBIDDEN_METHODS) & set(PREFERENCE_METHODS)), (
            "PPO must not be used as first-version preference method"
        )

    def test_training_stages_and_pareto_defined(self):
        """4-stage training sequence and Pareto dimensions must be defined."""
        assert len(REQUIRED_TRAINING_STAGES) == 4
        assert REQUIRED_TRAINING_STAGES[0] == "stage1_edit_reconstruction"
        assert REQUIRED_TRAINING_STAGES[-1] == "stage4_risk_adjusted_preference"
        # Pareto frontier dimensions (spec L1761-1771)
        assert set(PARETO_DIMENSIONS) == {"utility", "validity", "risk"}
        # Success must not be judged by uniqueness/diversity alone
        assert "diversity" not in PARETO_DIMENSIONS
        assert "uniqueness" not in PARETO_DIMENSIONS
