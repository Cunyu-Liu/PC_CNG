"""Unit tests for the v4 distance-based atom-balance criterion.

Covers BALANCE_DIST_SLACK=2 semantics used by
``run_phase4_fixed_testset.generate_union_candidates`` (pool eligibility)
and the ``balance_dist_slack`` mode of
``generate_structured_proposal_exhaustive`` (generator-side filtering).

Background: the v3 ratio-based criterion (atom_balance_score with
eps=0.011) silently killed ALL single-atom transmutation candidates on
large multi-component systems (L1 distance +2 on ~150 atoms exceeds the
ratio tolerance), starving the learned arm's semi_hard pool and giving
rule_pc_cng a pool-monopoly home advantage.  The distance-based
criterion is size-independent: slack=2 admits exactly one transmutation
while foreign products (shuffled / random mismatch) stay excluded.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pc_cng.chem_utils import atom_count_distance  # noqa: E402

try:  # pragma: no cover - optional heavy deps
    import torch
    from rdkit import Chem  # noqa: F401
    _DEPS_OK = True
except Exception:  # pragma: no cover
    torch = None
    _DEPS_OK = False

SLACK = 2


def _balanced(reactants: str, true_prod: str, cand_prod: str) -> bool:
    """Mirror of the v4 pool-eligibility predicate in _mk_candidate."""
    return atom_count_distance(reactants, cand_prod) <= \
        atom_count_distance(reactants, true_prod) + SLACK


def _balanced_norm(reactants: str, true_prod: str, cand_prod: str) -> bool:
    """v4.1 predicate: strip atom maps / bracket-H noise before counting."""
    from pc_cng.p4_g8c_learned_structured_proposal import _strip_atom_maps
    r = _strip_atom_maps(reactants)
    t = _strip_atom_maps(true_prod)
    c = _strip_atom_maps(cand_prod)
    return atom_count_distance(r, c) <= atom_count_distance(r, t) + SLACK


class TestDistanceBalanceCriterion(unittest.TestCase):
    """Predicate-level tests (no model required)."""

    # Mono-coupling of 1,4-dibromobenzene with phenylboronic acid.
    REACTANTS = "Brc1ccc(Br)cc1.OB(O)c1ccccc1"
    TRUE_PROD = "c1ccc(-c2ccc(Br)cc2)cc1"  # 4-bromobiphenyl

    def test_halogen_transmutation_admitted(self):
        # Br -> Cl on the product costs exactly L1 +2 (Br -1, Cl +1).
        cand = "c1ccc(-c2ccc(Cl)cc2)cc1"
        self.assertTrue(_balanced(self.REACTANTS, self.TRUE_PROD, cand))

    def test_connectivity_edit_admitted(self):
        # Bond migration (regioisomer) conserves atom counts: delta 0.
        cand = "c1ccc(-c2cccc(Br)c2)cc1"  # 3-bromobiphenyl
        self.assertTrue(_balanced(self.REACTANTS, self.TRUE_PROD, cand))

    def test_foreign_product_excluded(self):
        # Shuffled/random real products differ by far more than slack.
        for foreign in ("c1ccncc1", "OCc1ccccc1", "c1ccc2ccccc2c1"):
            self.assertFalse(
                _balanced(self.REACTANTS, self.TRUE_PROD, foreign),
                msg=f"foreign product {foreign} should be excluded")

    def test_size_independence(self):
        # The ratio-based eps=0.011 fails on large systems; distance=2
        # must admit the transmutation regardless of spectator size.
        big_reactants = (
            "Brc1ccc(Br)cc1.OB(O)c1ccccc1."
            "c1ccc(P(c2ccccc2)c2ccccc2)cc1.O=C([O-])[O-].[Na+].[Na+]"
        )
        cand = "c1ccc(-c2ccc(Cl)cc2)cc1"
        self.assertTrue(_balanced(big_reactants, self.TRUE_PROD, cand))

    def test_premapped_reaction_normalized(self):
        # HiTEA-style pre-mapped SMILES: bracket-H tokens ([CH3], [cH])
        # are counted by the regex tokenizer.  Without normalisation the
        # reactant-side H noise inflates d_neg and falsely excludes a
        # valid Br->Cl transmutation; with v4.1 normalisation it passes.
        mapped_reactants = (
            "[CH3:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.[Br:8][Br:9]")
        mapped_true = (
            "[CH3:1][c:2]1[cH:3][cH:4][c:5]([Br:8])[cH:6][cH:7]1")
        cand_plain = "Cc1ccc(Cl)cc1"
        # raw (unnormalised) predicate falsely rejects
        self.assertFalse(
            _balanced(mapped_reactants, mapped_true, cand_plain))
        # normalised predicate admits
        self.assertTrue(
            _balanced_norm(mapped_reactants, mapped_true, cand_plain))


@unittest.skipUnless(_DEPS_OK, "torch/rdkit unavailable")
class TestExhaustiveGeneratorDistSlack(unittest.TestCase):
    """Generator-level: distance mode emits transmutations that the
    legacy ratio mode starves (the v3 H1-failure root cause)."""

    RXN = ("Brc1ccc(Br)cc1.OB(O)c1ccccc1"
           ">>c1ccc(-c2ccc(Br)cc2)cc1")

    def _edits(self, **kwargs):
        from pc_cng.p4_g8c_learned_structured_proposal import (
            StructuredProposalModel,
            generate_structured_proposal_exhaustive,
        )
        torch.manual_seed(0)
        model = StructuredProposalModel(
            hidden_dim=32, num_heads=2, num_layers=1, dropout=0.0)
        return generate_structured_proposal_exhaustive(
            model, self.RXN, top_k=64, use_validity_mask=True,
            map_unmapped=False, require_atom_balance=True, **kwargs)

    def test_dist_slack_admits_transmutation(self):
        from pc_cng.p4_g8c_learned_structured_proposal import EditType
        edits = self._edits(balance_dist_slack=SLACK)
        self.assertTrue(edits, "generator returned no candidates")
        types = {e.edit_type for e in edits}
        self.assertIn(EditType.ATOM_TRANSMUTATION, types)
        # every emitted candidate must satisfy the distance criterion
        reactants = self.RXN.split(">")[0]
        true_prod = self.RXN.split(">")[-1]
        for e in edits:
            self.assertTrue(
                _balanced(reactants, true_prod, e.applied_product),
                msg=f"unbalanced candidate emitted: {e.applied_product}")

    def test_legacy_ratio_mode_starves_transmutation(self):
        # Documents the v3 failure mode on the SAME reaction: ratio eps
        # rejects the Br->Cl candidate (score drop 4/30 -> 6/30 > 0.011).
        from pc_cng.p4_g8c_learned_structured_proposal import EditType
        edits = self._edits(balance_eps=0.011)
        types = {e.edit_type for e in edits}
        self.assertNotIn(EditType.ATOM_TRANSMUTATION, types)


if __name__ == "__main__":
    unittest.main()
