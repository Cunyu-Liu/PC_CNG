from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("rdkit") is not None, "RDKit is required")
class GraphStatsFeaturizerTest(unittest.TestCase):
    def test_graph_stats_reaction_featurizer(self) -> None:
        from pc_cng.train_feasibility_mlp import make_reaction_featurizer

        featurizer = make_reaction_featurizer(feature_mode="graph_stats")
        features = featurizer.reaction_fp("CCO>>CC=O")
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], featurizer.output_dim)
        self.assertGreater(featurizer.output_dim, 100)
        self.assertIsNone(featurizer.reaction_fp(">>CC=O"))

    def test_morgan_and_graph_stats_have_distinct_dimensions(self) -> None:
        from pc_cng.train_feasibility_mlp import make_reaction_featurizer

        morgan = make_reaction_featurizer(feature_mode="morgan", n_bits=128)
        graph_stats = make_reaction_featurizer(feature_mode="graph_stats")
        self.assertNotEqual(morgan.output_dim, graph_stats.output_dim)

    def test_combined_featurizer_concatenates_morgan_and_graph_stats(self) -> None:
        from pc_cng.train_feasibility_mlp import make_reaction_featurizer

        morgan = make_reaction_featurizer(feature_mode="morgan", n_bits=128)
        graph_stats = make_reaction_featurizer(feature_mode="graph_stats")
        combined = make_reaction_featurizer(feature_mode="combined", n_bits=128)
        self.assertEqual(combined.output_dim, morgan.output_dim + graph_stats.output_dim)
        fp_m = morgan.reaction_fp("CCO>>CC=O")
        fp_g = graph_stats.reaction_fp("CCO>>CC=O")
        fp_c = combined.reaction_fp("CCO>>CC=O")
        self.assertIsNotNone(fp_m)
        self.assertIsNotNone(fp_g)
        self.assertIsNotNone(fp_c)
        import numpy as np
        expected = np.concatenate([fp_m, fp_g])
        self.assertEqual(fp_c.shape[0], expected.shape[0])
        self.assertTrue(np.allclose(fp_c, expected))
        self.assertIsNone(combined.reaction_fp(">>CC=O"))


if __name__ == "__main__":
    unittest.main()
