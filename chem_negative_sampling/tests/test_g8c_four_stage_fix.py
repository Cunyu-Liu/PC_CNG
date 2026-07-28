"""Smoke tests for the G8-C four-stage training fix.

Tests that:
  1. ``extract_real_edit_targets`` works on a sample atom-mapped reaction.
  2. ``extract_rule_proposals`` works on a sample reaction.
  3. ``build_competing_outcome_pairs`` works on a small HTE sample.
  4. ``build_preference_pairs`` works on a small HTE sample.
  5. Stage 1-4 training runs without errors on a tiny dataset.

Run with::

    python3 -m pytest tests/test_g8c_four_stage_fix.py -v --tb=short -x
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Ensure the pc_cng package is importable.
_CNS_ROOT = Path(__file__).resolve().parents[1]
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from pc_cng.g8c_data_preparation import (  # noqa: E402
    assign_formal_pair_partition,
    build_competing_outcome_pairs,
    build_preference_pairs,
    extract_real_edit_targets,
    extract_rule_proposals,
    load_g8c_training_data,
)
from pc_cng.p4_g8c_learned_structured_proposal import (  # noqa: E402
    EditType,
    StructuredProposalModel,
    _state_dict_sha256,
    _frozen_hash_partition,
    _featurize_risk_safe,
    compute_action_logp,
    compute_logp,
    train_stage,
)
from pc_cng.reaction_boundary_generator import (  # noqa: E402
    ReactionBoundaryGenerator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A real atom-mapped HTE alkylation reaction (from the parquet sample).
SAMPLE_RXN = (
    "[F:1][c:2]1[c:11]([c:7]2[cH:6][c:4]([Br:5])[cH:3]1)[nH:10][n:9][cH:8]2."
    "[CH3:12]OS(OC)(=O)=O>>[CH3:12][n:10]1[c:11]([c:7]2[cH:8][n:9]1)"
    "[c:2]([F:1])[cH:3][c:4]([Br:5])[cH:6]2"
)

# A second atom-mapped reaction with a different product scaffold, so that
# competing-outcome pairing has something to work with.
SAMPLE_RXN_2 = (
    "[F:1][c:2]1[c:11]([c:7]2[cH:6][c:4]([Br:5])[cH:3]1)[nH:10][n:9][cH:8]2."
    "[CH3:12]OS(OC)(=O)=O>>[CH3:12][n:9]1[c:8][c:7]2[c:11]([cH:6][c:4]([Br:5])"
    "[cH:3]2)[n:10]1[F:1]"
)

HTE_PARQUET = Path("/home/cunyuliu/pc_cng_research/data/processed/p4_hte_normalized.parquet")


@pytest.fixture(scope="module")
def rule_generator():
    return ReactionBoundaryGenerator(
        max_candidates_per_reaction=4, allow_unmapped_fallback=False)


@pytest.fixture(scope="module")
def small_hite_df():
    """Load a small slice of the real HTE parquet (or synthesize one)."""
    if HTE_PARQUET.exists():
        df = pd.read_parquet(HTE_PARQUET)
        # Keep only atom-mapped, positive-yield rows from the train split.
        df = df[df["split"].astype(str) == "train"].head(60).copy()
        if len(df) >= 10:
            return df
    # Fallback synthetic dataframe (used only if the parquet is unavailable).
    rows = []
    for i in range(12):
        rxn = SAMPLE_RXN if i % 2 == 0 else SAMPLE_RXN_2
        yld = 50.0 if i % 2 == 0 else 5.0
        rows.append({
            "reaction_smiles": rxn,
            "reactant_1_smiles": "c12c(cc(cc1F)Br)cn[nH]2",
            "reactant_2_smiles": "S(OC)(OC)(=O)=O",
            "measured_yield": yld,
            "experimental_group": "TEST_GROUP",
            "products": "Cn1ncc2cc(Br)cc(F)c12",
            "split": "train",
            "yield_bin": 1 if yld > 25 else 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. extract_real_edit_targets
# ---------------------------------------------------------------------------

class TestExtractRealEditTargets:
    def test_returns_dict_with_required_keys(self):
        result = extract_real_edit_targets(SAMPLE_RXN)
        required = {"locus", "edit_type", "formed_bonds", "broken_bonds",
                    "changed_bonds", "reacting_atoms", "mapped"}
        assert required.issubset(result.keys())

    def test_mapped_reaction_has_formed_bond(self):
        result = extract_real_edit_targets(SAMPLE_RXN)
        assert result["mapped"] is True
        assert len(result["formed_bonds"]) > 0
        # A real newly formed bond is reconstruction target BOND_FORM.  It is
        # distinct from the generative FORMED_BOND_MIGRATE operation.
        assert result["edit_type"] == int(EditType.BOND_FORM)
        assert result["actions"][0]["partner_map"] > 0

    def test_locus_is_reacting_atom(self):
        result = extract_real_edit_targets(SAMPLE_RXN)
        assert result["locus"] in result["reacting_atoms"]

    def test_unmapped_reaction_is_not_applicable(self):
        result = extract_real_edit_targets("CC.O>>CCO")
        assert result["mapped"] is False
        assert result["edit_type"] == int(EditType.NOT_APPLICABLE)
        assert result["valid_for_formal"] is False
        assert result["actions"] == []
        assert result["locus"] == 0

    def test_risk_featurizer_keeps_parseable_unmapped_outcome(self):
        graph = _featurize_risk_safe("CC.O>>CCO")
        assert graph is not None
        assert graph.atom_features.shape[0] == 3


# ---------------------------------------------------------------------------
# 2. extract_rule_proposals
# ---------------------------------------------------------------------------

class TestExtractRuleProposals:
    def test_returns_list_of_dicts(self, rule_generator):
        proposals = extract_rule_proposals(SAMPLE_RXN, rule_generator)
        assert isinstance(proposals, list)
        for p in proposals:
            assert {"locus", "edit_type", "edit_action",
                    "candidate_product", "hard_score"}.issubset(p.keys())

    def test_proposal_edit_types_are_valid(self, rule_generator):
        proposals = extract_rule_proposals(SAMPLE_RXN, rule_generator)
        valid_types = {int(EditType.ATOM_TRANSMUTATION),
                       int(EditType.BOND_ORDER_CHANGE),
                       int(EditType.FORMED_BOND_MIGRATE),
                       int(EditType.NO_EDIT),
                       int(EditType.NOT_APPLICABLE)}
        for p in proposals:
            assert p["edit_type"] in valid_types

    def test_handles_unmapped_reaction_gracefully(self, rule_generator):
        # The generator may auto-map unmapped reactions via RXNMapper, so we
        # only assert that the function returns a list without crashing.
        proposals = extract_rule_proposals("CC.O>>CCO", rule_generator)
        assert isinstance(proposals, list)


# ---------------------------------------------------------------------------
# 3. build_competing_outcome_pairs
# ---------------------------------------------------------------------------

class TestBuildCompetingOutcomePairs:
    def test_returns_list_of_dicts(self, small_hite_df):
        pairs = build_competing_outcome_pairs(small_hite_df)
        assert isinstance(pairs, list)
        for p in pairs:
            assert {"reactants", "preferred_product", "preferred_yield",
                    "competing_product", "competing_yield",
                    "experimental_group", "reaction_smiles"}.issubset(p.keys())

    def test_preferred_yield_geq_competing(self, small_hite_df):
        pairs = build_competing_outcome_pairs(small_hite_df)
        for p in pairs:
            if p["preferred_product"] and p["competing_product"]:
                assert p["preferred_yield"] >= p["competing_yield"]

    def test_empty_df_returns_empty(self):
        assert build_competing_outcome_pairs(pd.DataFrame()) == []

    def test_context_partition_is_deterministic_and_group_safe(self):
        pairs = [
            {"context_key": f"context-{i}", "split": "train"}
            for i in range(100)
        ]
        first = assign_formal_pair_partition(pairs)
        second = assign_formal_pair_partition(list(reversed(pairs)))
        first_map = {
            pair["context_key"]: pair["formal_split"] for pair in first
        }
        second_map = {
            pair["context_key"]: pair["formal_split"] for pair in second
        }
        assert first_map == second_map
        assert {"train", "val"}.issubset(set(first_map.values()))

    def test_v2_holdout_is_order_invariant_and_group_safe(self):
        rows = [
            {"group": f"group-{i // 2}", "record": i}
            for i in range(100)
        ]
        train, holdout = _frozen_hash_partition(
            rows,
            lambda row: row["group"],
            namespace="phase_c_v2_test",
        )
        train_groups = {row["group"] for row in train}
        holdout_groups = {row["group"] for row in holdout}
        assert train_groups.isdisjoint(holdout_groups)
        reverse_train, reverse_holdout = _frozen_hash_partition(
            list(reversed(rows)),
            lambda row: row["group"],
            namespace="phase_c_v2_test",
        )
        assert {row["record"] for row in train} == {
            row["record"] for row in reverse_train
        }
        assert {row["record"] for row in holdout} == {
            row["record"] for row in reverse_holdout
        }


# ---------------------------------------------------------------------------
# 4. build_preference_pairs
# ---------------------------------------------------------------------------

class TestBuildPreferencePairs:
    def test_returns_list_of_dicts(self, small_hite_df):
        pairs = build_preference_pairs(small_hite_df, generator=None)
        assert isinstance(pairs, list)
        for p in pairs:
            assert {"reactants", "preferred_reaction", "dispreferred_reaction",
                    "preferred_yield", "dispreferred_yield",
                    "experimental_group"}.issubset(p.keys())

    def test_preferred_yield_gt_dispreferred(self, small_hite_df):
        pairs = build_preference_pairs(small_hite_df, generator=None)
        for p in pairs:
            assert p["preferred_yield"] > p["dispreferred_yield"]

    def test_empty_df_returns_empty(self):
        assert build_preference_pairs(pd.DataFrame(), generator=None) == []


# ---------------------------------------------------------------------------
# 5. Stage 1-4 training smoke test
# ---------------------------------------------------------------------------

class TestFourStageTraining:
    """End-to-end smoke test: run all 4 stages on a tiny dataset."""

    @pytest.fixture(scope="class")
    def device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @pytest.fixture(scope="class")
    def tiny_reactions(self, small_hite_df):
        """Pick atom-mapped reactions that featurize successfully."""
        from pc_cng.p4_g8c_learned_structured_proposal import _featurize_safe
        rxns = small_hite_df["reaction_smiles"].astype(str).tolist()
        good = []
        for r in rxns:
            if _featurize_safe(r) is not None:
                good.append(r)
            if len(good) >= 10:
                break
        assert len(good) >= 2, "need at least 2 featurizable reactions"
        return good

    @pytest.fixture(scope="class")
    def caches(self, tiny_reactions, rule_generator):
        edit_targets = {r: extract_real_edit_targets(r) for r in tiny_reactions}
        rule_proposals = {}
        for r in tiny_reactions:
            rule_proposals[r] = extract_rule_proposals(r, rule_generator)
        # Build pairs from a tiny dataframe so the caches are non-empty.
        import pandas as _pd
        rows = []
        for i, r in enumerate(tiny_reactions):
            rows.append({
                "reaction_smiles": r,
                "reactant_1_smiles": f"reactant_{i % 3}",
                "measured_yield": float(50 - i * 4),
                "experimental_group": "smoke",
                "products": f"product_{i}",
                "split": "train",
            })
        tiny_df = _pd.DataFrame(rows)
        competing = build_competing_outcome_pairs(tiny_df)
        preference = build_preference_pairs(tiny_df, generator=rule_generator)
        # If pairs are empty (too few distinct reactants), synthesize minimal
        # entries so the stage-3/4 code paths are exercised.
        if not competing:
            competing = [{
                "reactants": tiny_reactions[0],
                "preferred_product": "A",
                "preferred_yield": 50.0,
                "competing_product": "B",
                "competing_yield": 5.0,
                "experimental_group": "smoke",
                "reaction_smiles": tiny_reactions[0],
                "competing_reaction_smiles": tiny_reactions[-1],
            }]
        if not preference:
            preference = [{
                "reactants": tiny_reactions[0],
                "preferred_reaction": tiny_reactions[0],
                "dispreferred_reaction": tiny_reactions[-1],
                "preferred_yield": 50.0,
                "dispreferred_yield": 5.0,
                "experimental_group": "smoke",
            }]
        return {
            "edit_targets": edit_targets,
            "rule_proposals": rule_proposals,
            "competing_pairs": competing,
            "preference_pairs": preference,
            "risk_examples": [
                {
                    "reaction_smiles": tiny_reactions[0],
                    "risk_label": 1,
                    "risk_source": "known_positive_collision",
                },
                {
                    "reaction_smiles": tiny_reactions[-1],
                    "risk_label": 0,
                    "risk_source": "heldout_hte_outcome",
                },
            ],
        }

    def test_stage1_reconstruction(self, tiny_reactions, caches, device):
        model = StructuredProposalModel(hidden_dim=32, num_heads=2,
                                        num_layers=1, dropout=0.1)
        log = train_stage(
            model, stage=1,
            train_reactions=tiny_reactions,
            val_reactions=tiny_reactions[:2],
            rule_generator=ReactionBoundaryGenerator(),
            epochs=1, batch_size=4, lr=1e-3,
            device=device, seed=42,
            edit_targets_cache=caches["edit_targets"],
        )
        assert len(log) == 1
        assert log[0]["stage"] == 1

    def test_formal_run_fails_closed_without_real_pairs(self, tiny_reactions, device):
        """Formal mode must never silently re-create pseudo supervision."""
        model = StructuredProposalModel(hidden_dim=32, num_heads=2,
                                        num_layers=1, dropout=0.1)
        with pytest.raises(RuntimeError, match="requires an edit-target cache"):
            train_stage(
                model, stage=3,
                train_reactions=tiny_reactions,
                val_reactions=tiny_reactions[:2],
                rule_generator=ReactionBoundaryGenerator(),
                epochs=1, batch_size=4, lr=1e-3,
                device=device, seed=42,
                formal_run=True,
            )

    def test_formal_stage1_uses_joint_real_action_supervision(
            self, tiny_reactions, caches, device):
        model = StructuredProposalModel(hidden_dim=32, num_heads=2,
                                        num_layers=1, dropout=0.1)
        log = train_stage(
            model, stage=1,
            train_reactions=tiny_reactions,
            val_reactions=tiny_reactions[:2],
            rule_generator=ReactionBoundaryGenerator(),
            epochs=1, batch_size=4, lr=1e-3,
            device=device, seed=42,
            edit_targets_cache=caches["edit_targets"],
            rule_proposals_cache=caches["rule_proposals"],
            competing_pairs_cache=caches["competing_pairs"],
            preference_pairs_cache=caches["preference_pairs"],
            risk_examples_cache=caches["risk_examples"],
            formal_run=True,
        )
        assert log[-1]["formal_supervision"] is True
        assert "arg_loss" in log[-1]["components"]

    def test_formal_stage3_uses_prevalidated_real_edit_rehearsal(
            self, tiny_reactions, caches, device):
        model = StructuredProposalModel(hidden_dim=32, num_heads=2,
                                        num_layers=1, dropout=0.1)
        log = train_stage(
            model, stage=3,
            train_reactions=tiny_reactions,
            val_reactions=tiny_reactions[:2],
            rule_generator=ReactionBoundaryGenerator(),
            epochs=1, batch_size=4, lr=1e-3,
            device=device, seed=42,
            edit_targets_cache=caches["edit_targets"],
            rule_proposals_cache=caches["rule_proposals"],
            competing_pairs_cache=caches["competing_pairs"],
            preference_pairs_cache=caches["preference_pairs"],
            risk_examples_cache=caches["risk_examples"],
            formal_run=True,
        )
        assert log[-1]["formal_supervision"] is True
        assert (
            log[-1]["components"]["reconstruction_rehearsal_loss"]
            > 0.0
        )

    def test_stage2_imitation(self, tiny_reactions, caches, device):
        model = StructuredProposalModel(hidden_dim=32, num_heads=2,
                                        num_layers=1, dropout=0.1)
        log = train_stage(
            model, stage=2,
            train_reactions=tiny_reactions,
            val_reactions=tiny_reactions[:2],
            rule_generator=ReactionBoundaryGenerator(),
            epochs=1, batch_size=4, lr=1e-3,
            device=device, seed=42,
            edit_targets_cache=caches["edit_targets"],
            rule_proposals_cache=caches["rule_proposals"],
        )
        assert len(log) == 1
        assert log[0]["stage"] == 2

    def test_stage3_contrastive(self, tiny_reactions, caches, device):
        model = StructuredProposalModel(hidden_dim=32, num_heads=2,
                                        num_layers=1, dropout=0.1)
        log = train_stage(
            model, stage=3,
            train_reactions=tiny_reactions,
            val_reactions=tiny_reactions[:2],
            rule_generator=ReactionBoundaryGenerator(),
            epochs=1, batch_size=4, lr=1e-3,
            device=device, seed=42,
            competing_pairs_cache=caches["competing_pairs"],
        )
        assert len(log) == 1
        assert log[0]["stage"] == 3

    def test_stage4_dpo_with_reference(self, tiny_reactions, caches, device):
        import copy
        model = StructuredProposalModel(hidden_dim=32, num_heads=2,
                                        num_layers=1, dropout=0.1)
        # Reference model = frozen deep copy (simulates post-Stage-3 snapshot).
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        log = train_stage(
            model, stage=4,
            train_reactions=tiny_reactions,
            val_reactions=tiny_reactions[:2],
            rule_generator=ReactionBoundaryGenerator(),
            epochs=1, batch_size=4, lr=1e-3,
            device=device, seed=42,
            edit_targets_cache=caches["edit_targets"],
            preference_pairs_cache=caches["preference_pairs"],
            ref_model=ref_model,
        )
        assert len(log) == 1
        assert log[0]["stage"] == 4

    def test_compute_logp(self, tiny_reactions, caches, device):
        """compute_logp returns finite log-probabilities."""
        from pc_cng.p4_g8c_learned_structured_proposal import (
            _collate_reactions, StructuredProposalOutput)
        model = StructuredProposalModel(hidden_dim=32, num_heads=2,
                                        num_layers=1, dropout=0.1).to(device)
        batch, success = _collate_reactions(
            tiny_reactions[:2], device, map_unmapped=False)
        if batch is None:
            pytest.skip("featurization failed on tiny set")
        out = model(batch)
        loci = torch.tensor(
            [caches["edit_targets"][r]["locus"] for r in success],
            device=device, dtype=torch.long)
        types = torch.tensor(
            [caches["edit_targets"][r]["edit_type"] for r in success],
            device=device, dtype=torch.long)
        logp = compute_logp(out, loci, types)
        assert logp.shape == (len(success),)
        assert torch.isfinite(logp).all()

    def test_complete_action_logp_and_reference_hash(
            self, tiny_reactions, caches, device):
        from pc_cng.p4_g8c_learned_structured_proposal import (
            _collate_reactions,
            _primary_real_target,
        )
        model = StructuredProposalModel(hidden_dim=32, num_heads=2,
                                        num_layers=1, dropout=0.1).to(device)
        batch, success = _collate_reactions(
            tiny_reactions[:2], device, map_unmapped=False)
        if batch is None:
            pytest.skip("featurization failed on tiny set")
        targets = [
            _primary_real_target(
                reaction,
                graph,
                caches["edit_targets"],
            )
            for reaction, graph in zip(success, batch.graphs)
        ]
        if any(target is None for target in targets):
            pytest.skip("no complete real targets")
        loci = torch.tensor(
            [target["locus"] for target in targets],
            dtype=torch.long,
            device=device,
        )
        output = model(batch, locus_index=loci)
        before = _state_dict_sha256(model)
        logp = compute_action_logp(output, targets)
        after = _state_dict_sha256(model)
        assert logp.shape == (len(targets),)
        assert torch.isfinite(logp).all()
        assert before == after
