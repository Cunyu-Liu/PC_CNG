"""Unit tests for the false-negative three-layer control (P1-08)."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from pc_cng.false_negative_three_layer_control import (
    DEFAULT_ENSEMBLE_STD_THRESHOLD,
    REVIEWED_FIELDS,
    SAMPLED_FOR_REVIEW_FIELDS,
    cohens_kappa,
    database_retrieval_layer,
    ensemble_agreement_layer,
    expert_review_layer,
    fleiss_kappa,
    rule_based_plausibility_check,
    run_three_layer_control,
    stratified_sample_for_review,
    verdict_to_binary,
    write_csv,
)


def _make_reviewed_row(**overrides) -> dict:
    base = {field: "" for field in REVIEWED_FIELDS}
    base.update({
        "source_id": "hitea_000000001",
        "candidate_reaction": "CCO>>CC=O",
        "candidate_reactants": "CCO",
        "candidate_product": "CC=O",
        "positive_reaction": "CCO.[O]>>CC=O",
        "parent_reactants": "CCO.[O]",
        "parent_product": "CC=O",
        "task": "retro_precursor",
        "failure_type": "retro_missing_reactant",
        "valid": "1",
        "atom_balance": "0.95",
        "false_negative_risk": "0.20",
        "review_status": "keep_synthetic_negative",
        "passes_filter": "True",
        "label": "0",
    })
    base.update(overrides)
    return base


def _write_seed_predictions(path: Path, source_id: str, scores: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "source_id", "dataset", "reaction_class", "label", "score", "reaction_smiles"])
        writer.writeheader()
        for s in scores:
            writer.writerow({
                "source_id": source_id, "dataset": "hitea",
                "reaction_class": "Alkylation", "label": "1",
                "score": f"{s:.6f}", "reaction_smiles": "CCO>>CC=O",
            })


class CohensKappaTest(unittest.TestCase):
    def test_perfect_agreement(self):
        kappa, info = cohens_kappa([1, 1, 2, 2, 1], [1, 1, 2, 2, 1])
        self.assertAlmostEqual(kappa, 1.0, places=6)
        self.assertAlmostEqual(info["observed_agreement"], 1.0)

    def test_no_agreement_better_than_chance_returns_low(self):
        # rater A always says 1, rater B always says 2 => kappa <= 0
        kappa, _ = cohens_kappa([1, 1, 1, 1], [2, 2, 2, 2])
        self.assertLessEqual(kappa, 0.0)

    def test_moderate_agreement_in_unit_interval(self):
        kappa, info = cohens_kappa([1, 1, 2, 2, 1, 2, 1, 2], [1, 2, 2, 2, 1, 2, 1, 1])
        self.assertTrue(-1.0 <= kappa <= 1.0)
        self.assertEqual(info["n"], 8)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cohens_kappa([1, 2], [1])


class FleissKappaTest(unittest.TestCase):
    def test_perfect_agreement(self):
        # 3 raters, all agree on each subject, 2 categories
        ratings = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        kappa, props = fleiss_kappa(ratings, n_categories=2)
        self.assertAlmostEqual(kappa, 1.0, places=6)

    def test_zero_agreement_negative_kappa(self):
        # 3 raters, all disagree on each subject
        ratings = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
        kappa, _ = fleiss_kappa(ratings, n_categories=3)
        self.assertLess(kappa, 0.0)

    def test_empty_returns_zero(self):
        kappa, props = fleiss_kappa([], n_categories=2)
        self.assertEqual(kappa, 0.0)
        self.assertEqual(len(props), 2)


class VerdictToBinaryTest(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(verdict_to_binary(1), 0)
        self.assertEqual(verdict_to_binary(2), 0)
        self.assertEqual(verdict_to_binary(3), 1)
        self.assertEqual(verdict_to_binary(4), 2)
        self.assertEqual(verdict_to_binary(5), 2)


class RuleBasedPlausibilityCheckTest(unittest.TestCase):
    def test_keep_clean_row(self):
        row = _make_reviewed_row()
        self.assertEqual(rule_based_plausibility_check(row), "keep")

    def test_exclude_invalid(self):
        row = _make_reviewed_row(valid="0")
        self.assertEqual(rule_based_plausibility_check(row), "exclude")

    def test_exclude_low_atom_balance(self):
        row = _make_reviewed_row(atom_balance="0.3")
        self.assertEqual(rule_based_plausibility_check(row), "exclude")

    def test_exclude_high_fnr_needs_review(self):
        row = _make_reviewed_row(
            review_status="needs_review_or_downweight", false_negative_risk="0.9")
        self.assertEqual(rule_based_plausibility_check(row), "exclude")

    def test_uncertain_middle(self):
        row = _make_reviewed_row(
            atom_balance="0.6", false_negative_risk="0.5",
            review_status="needs_review_or_downweight")
        self.assertEqual(rule_based_plausibility_check(row), "uncertain")


class EnsembleAgreementLayerTest(unittest.TestCase):
    def test_excludes_high_variance_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ens_dir = root / "ensemble"
            # 5 seeds: 4 agree (std~0), 1 disagrees => low std; keep
            for i, scores in enumerate([
                [0.1, 0.1, 0.1, 0.1, 0.1],   # low variance parent => keep
                [0.1, 0.9, 0.1, 0.9, 0.1],   # high variance parent => exclude
            ]):
                _write_seed_predictions(
                    ens_dir / f"seed{i}_20260710" / "test_predictions.csv",
                    f"parent_{i}", scores)
            rows = [
                _make_reviewed_row(source_id="parent_0"),
                _make_reviewed_row(source_id="parent_1"),
            ]
            kept, excluded, stats = ensemble_agreement_layer(
                rows, str(ens_dir), std_threshold=0.15)
            self.assertEqual(len(kept), 1)
            self.assertEqual(len(excluded), 1)
            self.assertEqual(stats["parent_coverage"], 2)
            self.assertEqual(excluded[0]["source_id"], "parent_1")

    def test_no_coverage_keeps_conservatively(self):
        with tempfile.TemporaryDirectory() as tmp:
            ens_dir = Path(tmp) / "ensemble"
            ens_dir.mkdir()
            (ens_dir / "seed0_20260710").mkdir()
            _write_seed_predictions(
                ens_dir / "seed0_20260710" / "test_predictions.csv",
                "parent_0", [0.1])
            rows = [_make_reviewed_row(source_id="parent_unknown")]
            kept, excluded, stats = ensemble_agreement_layer(
                rows, str(ens_dir), std_threshold=0.15)
            self.assertEqual(len(kept), 1)
            self.assertEqual(len(excluded), 0)
            self.assertEqual(stats["parent_no_coverage"], 1)


class DatabaseRetrievalLayerTest(unittest.TestCase):
    def test_exact_reaction_match_excludes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "db.csv"
            write_csv(db_path, [{
                "source_id": "uspto_1",
                "reaction_smiles": "CCO>>CC=O",
                "reactants": "CCO",
                "agents": "",
                "products": "CC=O",
                "label_type": "positive",
                "yield": "100",
                "source": "uspto",
                "split_key": "k1",
                "split": "train",
            }], ["source_id", "reaction_smiles", "reactants", "agents",
                 "products", "label_type", "yield", "source", "split_key", "split"])
            rows = [_make_reviewed_row(candidate_reaction="CCO>>CC=O")]
            kept, excluded, stats = database_retrieval_layer(
                rows, str(db_path), tanimoto_threshold=0.95,
                tanimoto_sample_size=100)
            self.assertEqual(len(excluded), 1)
            self.assertEqual(len(kept), 0)
            self.assertGreater(stats["exact_match_hits"], 0)

    def test_no_match_keeps(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "db.csv"
            write_csv(db_path, [{
                "source_id": "uspto_1", "reaction_smiles": "c1ccccc1>>c1ccccc1",
                "reactants": "c1ccccc1", "agents": "", "products": "c1ccccc1",
                "label_type": "positive", "yield": "100", "source": "uspto",
                "split_key": "k1", "split": "train",
            }], ["source_id", "reaction_smiles", "reactants", "agents",
                 "products", "label_type", "yield", "source", "split_key", "split"])
            rows = [_make_reviewed_row(candidate_reaction="CCO>>CC=O")]
            kept, excluded, stats = database_retrieval_layer(
                rows, str(db_path), tanimoto_threshold=0.95,
                tanimoto_sample_size=100)
            self.assertEqual(len(kept), 1)
            self.assertEqual(len(excluded), 0)


class ExpertReviewLayerTest(unittest.TestCase):
    def test_rule_based_fallback(self):
        rows = [
            _make_reviewed_row(),  # keep
            _make_reviewed_row(valid="0"),  # exclude
            _make_reviewed_row(atom_balance="0.6", false_negative_risk="0.5",
                               review_status="needs_review_or_downweight"),  # uncertain
        ]
        kept, excluded, stats = expert_review_layer(rows, expert_review_dir=None)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(excluded), 2)  # exclude + uncertain both excluded
        self.assertFalse(stats["expert_executed"])
        self.assertEqual(stats["fallback"], "rule_based_plausibility_check")

    def test_uses_expert_verdicts_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            rev_dir = Path(tmp) / "review"
            rev_dir.mkdir()
            ratings_path = rev_dir / "reviewer_ratings_raw.csv"
            with ratings_path.open("w", newline="", encoding="utf-8") as h:
                w = csv.DictWriter(h, fieldnames=["sample_id", "overall_verdict"])
                w.writeheader()
                w.writerow({"sample_id": "S0001", "overall_verdict": "5"})  # keep
                w.writerow({"sample_id": "S0002", "overall_verdict": "1"})  # exclude
            rows = [
                _make_reviewed_row(sample_id="S0001"),
                _make_reviewed_row(sample_id="S0002"),
            ]
            kept, excluded, stats = expert_review_layer(rows, str(rev_dir))
            self.assertTrue(stats["expert_executed"])
            self.assertEqual(len(kept), 1)
            self.assertEqual(len(excluded), 1)


class RunThreeLayerControlTest(unittest.TestCase):
    def test_end_to_end_high_confidence_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ens_dir = root / "ensemble"
            for i in range(3):
                _write_seed_predictions(
                    ens_dir / f"seed{i}_20260710" / "test_predictions.csv",
                    "parent_0", [0.1, 0.1, 0.1])
            db_path = root / "db.csv"
            write_csv(db_path, [{
                "source_id": "uspto_1", "reaction_smiles": "c1ccccc1>>c1ccccc1",
                "reactants": "c1ccccc1", "agents": "", "products": "c1ccccc1",
                "label_type": "positive", "yield": "100", "source": "uspto",
                "split_key": "k1", "split": "train",
            }], ["source_id", "reaction_smiles", "reactants", "agents",
                 "products", "label_type", "yield", "source", "split_key", "split"])
            rows = [
                _make_reviewed_row(source_id="parent_0"),  # passes all 3
                _make_reviewed_row(source_id="parent_0", valid="0"),  # L3 excludes
            ]
            high, summary = run_three_layer_control(
                rows, ensemble_dir=str(ens_dir), database_csv=str(db_path),
                expert_review_dir=None)
            self.assertEqual(len(high), 1)
            self.assertIn("go_no_go_verdict", summary)
            self.assertGreater(summary["high_confidence_rate"], 0.0)


class StratifiedSampleTest(unittest.TestCase):
    def test_sample_size_and_blinding(self):
        rows = []
        for i in range(200):
            rows.append(_make_reviewed_row(
                source_id=f"r{i}", failure_type="retro_missing_reactant"))
        for i in range(100):
            rows.append(_make_reviewed_row(
                source_id=f"s{i}", failure_type="regio_wrong_position"))
        controls = [{"reaction_smiles": "c1ccccc1>>c1ccccc1"} for _ in range(50)]
        sampled = stratified_sample_for_review(
            rows, n_samples=100, seed=42,
            control_rows=controls, n_controls=20)
        self.assertEqual(len(sampled), 120)  # 100 synthetic + 20 controls
        origins = {r["source_origin"] for r in sampled}
        self.assertEqual(origins, {"pc_cng_synthetic", "real_negative_control"})
        # verdict fields are blank (awaiting reviewers)
        self.assertTrue(all(r["overall_verdict"] == "" for r in sampled))
        # sample_ids unique & formatted
        ids = [r["sample_id"] for r in sampled]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertTrue(all(s.startswith("S") for s in ids))


if __name__ == "__main__":
    unittest.main()
