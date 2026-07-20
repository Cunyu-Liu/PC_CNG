"""Tests for multiseed paired significance tool."""

from __future__ import annotations

import csv
import os
import tempfile
import unittest

from pc_cng.multiseed_paired_significance import (
    build_ensemble_scores,
    per_seed_group_metrics,
    seed_level_bootstrap,
)


def _write_score_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["group_id", "label", "score", "reaction_smiles", "source_id"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestBuildEnsembleScores(unittest.TestCase):
    def test_averages_across_two_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seed_a = os.path.join(tmp, "seed_a.csv")
            seed_b = os.path.join(tmp, "seed_b.csv")
            rows_a = [
                {"group_id": "g1", "source_id": "s1", "reaction_smiles": "A>>B",
                 "label": "1", "score": "0.9", "score_min": "0.9", "score_max": "0.9", "models_scored": "1",
                 "split": "test", "dataset": "d1", "candidate_source": "real",
                 "candidate_family": "positive", "reaction_class": ""},
                {"group_id": "g1", "source_id": "s2", "reaction_smiles": "A>>C",
                 "label": "0", "score": "0.3", "score_min": "0.3", "score_max": "0.3", "models_scored": "1",
                 "split": "test", "dataset": "d1", "candidate_source": "real",
                 "candidate_family": "real_negative", "reaction_class": ""},
            ]
            rows_b = [
                {"group_id": "g1", "source_id": "s1", "reaction_smiles": "A>>B",
                 "label": "1", "score": "0.7", "score_min": "0.7", "score_max": "0.7", "models_scored": "1",
                 "split": "test", "dataset": "d1", "candidate_source": "real",
                 "candidate_family": "positive", "reaction_class": ""},
                {"group_id": "g1", "source_id": "s2", "reaction_smiles": "A>>C",
                 "label": "0", "score": "0.5", "score_min": "0.5", "score_max": "0.5", "models_scored": "1",
                 "split": "test", "dataset": "d1", "candidate_source": "real",
                 "candidate_family": "real_negative", "reaction_class": ""},
            ]
            _write_score_csv(seed_a, rows_a)
            _write_score_csv(seed_b, rows_b)
            ensemble = build_ensemble_scores([seed_a, seed_b])
            by_key = {(r["group_id"], r["source_id"], r["reaction_smiles"]): r for r in ensemble}
            pos = by_key[("g1", "s1", "A>>B")]
            neg = by_key[("g1", "s2", "A>>C")]
            self.assertAlmostEqual(float(pos["score"]), 0.8, places=6)
            self.assertAlmostEqual(float(neg["score"]), 0.4, places=6)
            self.assertEqual(int(pos["models_scored"]), 2)
            self.assertAlmostEqual(float(pos["score_min"]), 0.7, places=6)
            self.assertAlmostEqual(float(pos["score_max"]), 0.9, places=6)


class TestPerSeedGroupMetrics(unittest.TestCase):
    def test_computes_per_seed_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seed_a = os.path.join(tmp, "seed_a.csv")
            seed_b = os.path.join(tmp, "seed_b.csv")
            rows_a = [
                {"group_id": "g1", "source_id": "s1", "reaction_smiles": "A>>B",
                 "label": "1", "score": "0.9", "score_min": "0.9", "score_max": "0.9", "models_scored": "1",
                 "split": "test", "dataset": "d1", "candidate_source": "real",
                 "candidate_family": "positive", "reaction_class": ""},
                {"group_id": "g1", "source_id": "s2", "reaction_smiles": "A>>C",
                 "label": "0", "score": "0.3", "score_min": "0.3", "score_max": "0.3", "models_scored": "1",
                 "split": "test", "dataset": "d1", "candidate_source": "real",
                 "candidate_family": "real_negative", "reaction_class": ""},
            ]
            rows_b = [
                {"group_id": "g1", "source_id": "s1", "reaction_smiles": "A>>B",
                 "label": "1", "score": "0.7", "score_min": "0.7", "score_max": "0.7", "models_scored": "1",
                 "split": "test", "dataset": "d1", "candidate_source": "real",
                 "candidate_family": "positive", "reaction_class": ""},
                {"group_id": "g1", "source_id": "s2", "reaction_smiles": "A>>C",
                 "label": "0", "score": "0.8", "score_min": "0.8", "score_max": "0.8", "models_scored": "1",
                 "split": "test", "dataset": "d1", "candidate_source": "real",
                 "candidate_family": "real_negative", "reaction_class": ""},
            ]
            _write_score_csv(seed_a, rows_a)
            _write_score_csv(seed_b, rows_b)
            metrics = per_seed_group_metrics([seed_a, seed_b])
            self.assertIn("0", metrics)
            self.assertIn("1", metrics)
            self.assertIn("g1", metrics["0"])
            self.assertAlmostEqual(metrics["0"]["g1"]["top1"], 1.0, places=6)
            self.assertAlmostEqual(metrics["1"]["g1"]["top1"], 0.0, places=6)


class TestSeedLevelBootstrap(unittest.TestCase):
    def test_delta_agrees_with_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_csvs = []
            cand_csvs = []
            for i in range(5):
                bp = os.path.join(tmp, f"base_{i}.csv")
                cp = os.path.join(tmp, f"cand_{i}.csv")
                base_rows = [
                    {"group_id": "g1", "source_id": "s1", "reaction_smiles": "A>>B",
                     "label": "1", "score": str(0.2 + 0.01 * i), "score_min": str(0.2 + 0.01 * i),
                     "score_max": str(0.2 + 0.01 * i), "models_scored": "1",
                     "split": "test", "dataset": "d1", "candidate_source": "real",
                     "candidate_family": "positive", "reaction_class": ""},
                    {"group_id": "g1", "source_id": "s2", "reaction_smiles": "A>>C",
                     "label": "0", "score": str(0.8 + 0.01 * i), "score_min": str(0.8 + 0.01 * i),
                     "score_max": str(0.8 + 0.01 * i), "models_scored": "1",
                     "split": "test", "dataset": "d1", "candidate_source": "real",
                     "candidate_family": "real_negative", "reaction_class": ""},
                ]
                cand_rows = [
                    {"group_id": "g1", "source_id": "s1", "reaction_smiles": "A>>B",
                     "label": "1", "score": str(0.9 + 0.01 * i), "score_min": str(0.9 + 0.01 * i),
                     "score_max": str(0.9 + 0.01 * i), "models_scored": "1",
                     "split": "test", "dataset": "d1", "candidate_source": "real",
                     "candidate_family": "positive", "reaction_class": ""},
                    {"group_id": "g1", "source_id": "s2", "reaction_smiles": "A>>C",
                     "label": "0", "score": str(0.3 + 0.01 * i), "score_min": str(0.3 + 0.01 * i),
                     "score_max": str(0.3 + 0.01 * i), "models_scored": "1",
                     "split": "test", "dataset": "d1", "candidate_source": "real",
                     "candidate_family": "real_negative", "reaction_class": ""},
                ]
                _write_score_csv(bp, base_rows)
                _write_score_csv(cp, cand_rows)
                base_csvs.append(bp)
                cand_csvs.append(cp)
            base_metrics = per_seed_group_metrics(base_csvs)
            cand_metrics = per_seed_group_metrics(cand_csvs)
            self.assertAlmostEqual(base_metrics["0"]["g1"]["top1"], 0.0, places=6)
            self.assertAlmostEqual(cand_metrics["0"]["g1"]["top1"], 1.0, places=6)
            result = seed_level_bootstrap(base_metrics, cand_metrics, iterations=200, seed=42)
            self.assertIn("top1", result)
            self.assertAlmostEqual(result["top1"]["mean"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
