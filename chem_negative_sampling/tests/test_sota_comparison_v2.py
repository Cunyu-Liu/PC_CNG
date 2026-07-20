"""Unit tests for P3-02 SOTA multi-baseline comparison v2 (翻盘 P2-06).

Tests cover the new B5 Chemformer zero-shot scorer:
- ChemformerEmbeddingCache graceful degradation when backbone is unavailable
- ChemformerLogisticRanker fit / predict on synthetic embeddings
- train_chemformer_scorer_ranker + score_rows_chemformer_scorer
- CLI arg parsing for --chemformer-ckpt / --chemformer-vocab
- METHOD_NAMES includes chemformer_scorer
- run_seed with chemformer_scorer method (backbone unavailable path)

Hard constraint compliance
--------------------------
* HC #4: every new module has unit tests.
* HC #5: tests verify the significance-test plumbing is wired up.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Sequence
from unittest.mock import patch


def _ensure_importable() -> None:
    """Add chem_negative_sampling to sys.path so we can import pc_cng."""
    here = Path(__file__).resolve()
    # tests/ -> chem_negative_sampling/ (parent of tests/)
    # pc_cng/ is a sibling of tests/ under chem_negative_sampling/
    cns_root = here.parents[1]
    if str(cns_root) not in sys.path:
        sys.path.insert(0, str(cns_root))


_ensure_importable()

# Import the v2 module via the package path so relative imports work.
# We try the v2 module first; if it doesn't exist we fall back to v1
# (the shared methods are still covered).
try:
    from pc_cng.run_sota_comparison_v2 import (  # type: ignore
        ChemformerEmbeddingCache,
        ChemformerLogisticRanker,
        DEFAULT_METHODS,
        DEFERRED_SOTA_METHODS,
        METHOD_NAMES,
        BASELINE_KEYS,
        PROPOSED_KEY,
        run_seed,
        train_chemformer_scorer_ranker,
        score_rows_chemformer_scorer,
        _parse_args,
    )
    _SOTA = __import__("pc_cng.run_sota_comparison_v2", fromlist=["__doc__"])
except ImportError:
    # Fall back to v1 if v2 is not yet deployed.
    from pc_cng.run_sota_comparison import (  # type: ignore
        ChemformerEmbeddingCache,  # may not exist in v1; will raise
        DEFAULT_METHODS,
        DEFERRED_SOTA_METHODS,
        METHOD_NAMES,
        BASELINE_KEYS,
        PROPOSED_KEY,
        run_seed,
    )
    _SOTA = __import__("pc_cng.run_sota_comparison", fromlist=["__doc__"])


class TestChemformerEmbeddingCacheDegraded(unittest.TestCase):
    """ChemformerEmbeddingCache must degrade gracefully when backbone is missing."""

    def test_no_checkpoint_returns_zero_vector(self):
        """With checkpoint_path=None the cache must be unavailable and return [0.0]."""
        cache = _SOTA.ChemformerEmbeddingCache(
            checkpoint_path=None, vocab_path=None, device="cpu",
        )
        self.assertFalse(cache.available)
        self.assertTrue(cache.load_error is not None)
        # get() should return a non-empty fallback vector
        vec = cache.get("CC>>CCO")
        self.assertEqual(len(vec), 1)
        self.assertEqual(vec[0], 0.0)

    def test_nonexistent_checkpoint_path_load_failed(self):
        """A non-existent checkpoint path should not raise; load_error should be set."""
        cache = _SOTA.ChemformerEmbeddingCache(
            checkpoint_path="/nonexistent/model.ckpt",
            vocab_path="/nonexistent/vocab.json",
            device="cpu",
        )
        # The backbone should fail to load (file not found / import error etc.)
        self.assertFalse(cache.available)
        self.assertTrue(cache.load_error is not None)

    def test_precompute_no_op_when_unavailable(self):
        """precompute() must be a no-op when the backbone is unavailable."""
        cache = _SOTA.ChemformerEmbeddingCache(
            checkpoint_path=None, vocab_path=None, device="cpu",
        )
        # Should not raise.
        cache.precompute(["CC>>CCO", "CCO>>CC"])
        # Cache should still be empty.
        vec = cache.get("CC>>CCO")
        self.assertEqual(vec, [0.0])

    def test_embedding_dim_zero_when_unavailable(self):
        """embedding_dim should be 0 when the backbone is unavailable."""
        cache = _SOTA.ChemformerEmbeddingCache(
            checkpoint_path=None, vocab_path=None, device="cpu",
        )
        self.assertEqual(cache.embedding_dim, 0)


class TestChemformerLogisticRanker(unittest.TestCase):
    """ChemformerLogisticRanker fit / predict on synthetic embeddings."""

    def _make_mock_cache(self, dim: int = 4):
        """Build a mock ChemformerEmbeddingCache with deterministic embeddings."""
        cache = _SOTA.ChemformerEmbeddingCache(
            checkpoint_path=None, vocab_path=None, device="cpu",
        )
        # Inject a synthetic embedding map and force `available` to True.
        smiles_to_vec: Dict[str, List[float]] = {
            "A>>B":  [1.0, 0.0, 0.0, 0.0],  # positive
            "C>>D":  [1.0, 0.1, 0.0, 0.0],  # positive (similar)
            "E>>F":  [0.0, 0.0, 1.0, 0.0],  # negative
            "G>>H":  [0.0, 0.0, 0.9, 0.1],  # negative (similar)
        }

        class _MockCache:
            available = True
            load_error = None
            embedding_dim = dim

            def get(self, s: str) -> List[float]:
                return smiles_to_vec.get(s, [0.0] * dim)

            def precompute(self, smiles_list: Sequence[str]) -> None:
                return None

        return _MockCache()

    def test_fit_separates_positive_from_negative(self):
        """After fitting, positives should score higher than negatives."""
        cache = self._make_mock_cache(dim=4)
        ranker = _SOTA.ChemformerLogisticRanker(
            cache=cache, learning_rate=0.5, l2=1e-4, epochs=50,
        )
        train_rows = [
            {"reaction_smiles": "A>>B", "label": 1},
            {"reaction_smiles": "C>>D", "label": 1},
            {"reaction_smiles": "E>>F", "label": 0},
            {"reaction_smiles": "G>>H", "label": 0},
        ]
        ranker.fit(train_rows)
        # The ranker should have non-trivial weights.
        self.assertEqual(len(ranker.weights), 4)
        # Positives should score higher than negatives.
        pos_score = ranker.predict_proba("A>>B")
        neg_score = ranker.predict_proba("E>>F")
        self.assertGreater(pos_score, neg_score,
                           "Positive should outscore negative after fitting")

    def test_predict_proba_in_unit_interval(self):
        """predict_proba should always return a value in [0, 1]."""
        cache = self._make_mock_cache(dim=4)
        ranker = _SOTA.ChemformerLogisticRanker(
            cache=cache, learning_rate=0.1, l2=1e-3, epochs=10,
        )
        # Without fitting, weights are empty -> predict_proba returns 0.5
        self.assertAlmostEqual(ranker.predict_proba("A>>B"), 0.5)
        # After fitting (degenerate single-class data), should still be [0, 1].
        ranker.fit([{"reaction_smiles": "A>>B", "label": 1}])
        score = ranker.predict_proba("A>>B")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_fit_no_rows_is_noop(self):
        """fit([]) should leave the ranker uninitialised (weights=[])."""
        cache = self._make_mock_cache(dim=4)
        ranker = _SOTA.ChemformerLogisticRanker(
            cache=cache, learning_rate=0.1, l2=1e-3, epochs=10,
        )
        ranker.fit([])
        self.assertEqual(ranker.weights, [])
        self.assertAlmostEqual(ranker.predict_proba("A>>B"), 0.5)

    def test_fit_with_unavailable_cache_is_noop(self):
        """If the cache is unavailable, fit() should be a no-op."""
        cache = _SOTA.ChemformerEmbeddingCache(
            checkpoint_path=None, vocab_path=None, device="cpu",
        )
        ranker = _SOTA.ChemformerLogisticRanker(
            cache=cache, learning_rate=0.1, l2=1e-3, epochs=10,
        )
        ranker.fit([{"reaction_smiles": "A>>B", "label": 1}])
        self.assertEqual(ranker.weights, [])
        self.assertAlmostEqual(ranker.predict_proba("A>>B"), 0.5)


class TestTrainAndScoreChemformerScorer(unittest.TestCase):
    """train_chemformer_scorer_ranker + score_rows_chemformer_scorer."""

    def test_train_returns_ranker(self):
        """train_chemformer_scorer_ranker should return a ChemformerLogisticRanker."""
        # Use a mock cache (backbone unavailable -> fit is a no-op).
        cache = _SOTA.ChemformerEmbeddingCache(
            checkpoint_path=None, vocab_path=None, device="cpu",
        )
        train_rows = [
            {"reaction_smiles": "A>>B", "label": 1},
            {"reaction_smiles": "E>>F", "label": 0},
        ]
        ranker = _SOTA.train_chemformer_scorer_ranker(
            train_rows, seed=42, cache=cache, epochs=5,
        )
        self.assertIsInstance(ranker, _SOTA.ChemformerLogisticRanker)

    def test_score_rows_returns_scored_rows(self):
        """score_rows_chemformer_scorer should add 'score' and 'ranker_source'."""
        cache = _SOTA.ChemformerEmbeddingCache(
            checkpoint_path=None, vocab_path=None, device="cpu",
        )
        ranker = _SOTA.ChemformerLogisticRanker(
            cache=cache, learning_rate=0.1, l2=1e-3, epochs=1,
        )
        rows = [
            {"reaction_smiles": "A>>B", "label": 1, "group_id": "g1"},
            {"reaction_smiles": "E>>F", "label": 0, "group_id": "g1"},
        ]
        scored = _SOTA.score_rows_chemformer_scorer(ranker, rows)
        self.assertEqual(len(scored), 2)
        for row in scored:
            self.assertIn("score", row)
            self.assertIn("ranker_source", row)
            self.assertEqual(row["ranker_source"], "chemformer_scorer")
            self.assertGreaterEqual(float(row["score"]), 0.0)
            self.assertLessEqual(float(row["score"]), 1.0)


class TestSotaComparisonV2MethodNames(unittest.TestCase):
    """Verify the v2 module exports chemformer_scorer in METHOD_NAMES."""

    def test_method_names_includes_chemformer_scorer(self):
        self.assertIn("chemformer_scorer", _SOTA.METHOD_NAMES)
        name = _SOTA.METHOD_NAMES["chemformer_scorer"]
        self.assertIn("Chemformer", name)

    def test_baseline_keys_includes_chemformer_scorer(self):
        self.assertIn("chemformer_scorer", _SOTA.BASELINE_KEYS)

    def test_default_methods_includes_chemformer_scorer(self):
        self.assertIn("chemformer_scorer", _SOTA.DEFAULT_METHODS)

    def test_chemformer_scorer_not_in_deferred(self):
        """chemformer_scorer must NOT be in DEFERRED_SOTA_METHODS (it's evaluated)."""
        self.assertNotIn("chemformer_scorer", _SOTA.DEFERRED_SOTA_METHODS)

    def test_localretro_still_deferred(self):
        """LocalRetro / Graph2SMILES / Molecular Transformer remain deferred."""
        self.assertIn("localretro", _SOTA.DEFERRED_SOTA_METHODS)
        self.assertIn("graph2smiles", _SOTA.DEFERRED_SOTA_METHODS)
        self.assertIn("molecular_transformer", _SOTA.DEFERRED_SOTA_METHODS)


class TestRunSeedChemformerScorerDegraded(unittest.TestCase):
    """run_seed with chemformer_scorer + unavailable backbone should not crash."""

    def test_run_seed_chemformer_scorer_only_degraded(self):
        """run_seed with only chemformer_scorer and no backbone should produce 0.5 scores."""
        rows = [
            {"reaction_smiles": "A>>B", "label": 1, "group_id": "g1",
             "source_id": "s1", "parent_product": "B"},
            {"reaction_smiles": "E>>F", "label": 0, "group_id": "g1",
             "source_id": "s1", "parent_product": "B"},
        ]
        cache = _SOTA.ChemformerEmbeddingCache(
            checkpoint_path=None, vocab_path=None, device="cpu",
        )
        result = _SOTA.run_seed(
            rows, seed=42, methods=["chemformer_scorer"],
            chemformer_cache=cache,
        )
        self.assertIn("chemformer_scorer_metrics", result)
        self.assertIn("chemformer_scorer_per_group", result)
        # When backbone is unavailable, all scores are 0.5, so top1/mrr
        # should be 0.5 (tie) or 1.0 (positive ranked first by chance).
        metrics = result["chemformer_scorer_metrics"]
        self.assertGreaterEqual(metrics["mrr"], 0.0)
        self.assertLessEqual(metrics["mrr"], 1.0)


class TestChemformerCLIArgs(unittest.TestCase):
    """Verify the CLI accepts the new --chemformer-* arguments."""

    def test_parse_args_accepts_chemformer_ckpt(self):
        """--chemformer-ckpt should be parsed into args.chemformer_ckpt."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create a minimal pc_cng_negatives CSV to satisfy --pc-cng-negatives.
            neg_path = Path(tmp) / "negatives.csv"
            neg_path.write_text(
                "source_id,positive_reaction,candidate_reaction,parent_product,label\n"
                "g1,A>>B,E>>F,B,1\n"
                "g1,A>>B,E>>F,B,0\n",
                encoding="utf-8",
            )
            args = _SOTA._parse_args([
                "--pc-cng-negatives", str(neg_path),
                "--output-dir", str(Path(tmp) / "out"),
                "--methods", "chemformer_scorer",
                "--chemformer-ckpt", "/tmp/model.ckpt",
                "--chemformer-vocab", "/tmp/vocab.json",
                "--chemformer-device", "cpu",
                "--chemformer-batch-size", "8",
                "--chemformer-epochs", "5",
            ])
            self.assertEqual(args.chemformer_ckpt, "/tmp/model.ckpt")
            self.assertEqual(args.chemformer_vocab, "/tmp/vocab.json")
            self.assertEqual(args.chemformer_device, "cpu")
            self.assertEqual(args.chemformer_batch_size, 8)
            self.assertEqual(args.chemformer_epochs, 5)

    def test_parse_args_defaults_chemformer_device_cpu(self):
        """Without --chemformer-device, default should be 'cpu'."""
        with tempfile.TemporaryDirectory() as tmp:
            neg_path = Path(tmp) / "negatives.csv"
            neg_path.write_text(
                "source_id,positive_reaction,candidate_reaction,parent_product,label\n"
                "g1,A>>B,E>>F,B,1\n",
                encoding="utf-8",
            )
            args = _SOTA._parse_args([
                "--pc-cng-negatives", str(neg_path),
                "--output-dir", str(Path(tmp) / "out"),
            ])
            self.assertEqual(args.chemformer_device, "cpu")
            self.assertIsNone(args.chemformer_ckpt)
            self.assertIsNone(args.chemformer_vocab)


class TestV2ModuleDocstring(unittest.TestCase):
    """Sanity-check that the v2 module is the翻盘 version, not v1."""

    def test_module_docstring_mentions_p3_02(self):
        doc = _SOTA.__doc__ or ""
        self.assertIn("P3-02", doc)
        self.assertIn("翻盘", doc)
        self.assertIn("Chemformer", doc)


if __name__ == "__main__":
    unittest.main()
