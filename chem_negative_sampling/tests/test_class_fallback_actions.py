from __future__ import annotations

import importlib.util
import unittest

from pc_cng.hard_negative_actions import class_fallback_actions, partial_product_actions, unreacted_substrate_actions


class ClassFallbackActionsTest(unittest.TestCase):
    def test_class_fallback_generates_no_conversion_candidate(self) -> None:
        rows = class_fallback_actions(
            "CCBr.O=S([O-])c1cccn1.[Na+]>>c1cccn1",
            source_id="rxn1",
            known_positives=set(),
            max_candidates_per_reaction=4,
            min_product_similarity=0.0,
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(all(row.action_family == "class_fallback" for row in rows))
        self.assertTrue(all(row.candidate_reaction != row.positive_reaction for row in rows))
        self.assertTrue(any("CCBr" in row.candidate_product for row in rows))

    def test_unreacted_substrate_keeps_high_similarity_failed_substrate(self) -> None:
        rows = unreacted_substrate_actions(
            "CC(=O)c1ccccc1>>CC(O)c1ccccc1",
            source_id="rxn_hydrogenation",
            known_positives=set(),
            max_candidates_per_reaction=2,
            min_product_similarity=0.0,
            max_product_similarity=0.9999,
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(all(row.action_family == "unreacted_substrate" for row in rows))
        self.assertTrue(all(row.candidate_reaction != row.positive_reaction for row in rows))
        self.assertTrue(any("CC(=O)" in row.candidate_product for row in rows))

    @unittest.skipUnless(importlib.util.find_spec("rdkit") is not None, "RDKit is required")
    def test_partial_product_generates_mapped_product_fragment(self) -> None:
        rows = partial_product_actions(
            "[CH3:1][Br:2].[NH2:3][CH3:4]>>[CH3:1][NH:3][CH3:4]",
            source_id="rxn2",
            known_positives=set(),
            max_candidates_per_reaction=4,
            min_product_similarity=0.0,
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(all(row.action_family == "partial_product" for row in rows))
        self.assertTrue(all(row.candidate_reaction != row.positive_reaction for row in rows))
        self.assertTrue(any("partial_product" in row.edit_action for row in rows))


if __name__ == "__main__":
    unittest.main()
