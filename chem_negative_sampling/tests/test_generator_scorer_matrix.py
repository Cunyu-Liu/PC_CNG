"""Tests for P4-G4 failure diagnostic matrix (run_p4_g4_diagnostic.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pc_cng.run_p4_g4_diagnostic import (
    MorganMLPScorer,
    build_cell_table,
    cell_effect_sizes,
    diagnose_hypotheses,
    difficulty_profile,
    interaction_anova,
    mixed_effects_model,
    morgan_fp,
    tanimoto_to_gold,
    write_summary_csv,
)


def _record(backbone: str, arm: str, seed: int, mrr: float) -> dict:
    keys = ["mrr", "top1", "top3", "ndcg", "auprc", "ece", "brier"]
    metrics = {k: mrr for k in keys}
    return {
        "backbone": backbone, "arm_id": arm,
        "arm_name": arm, "seed": seed,
        "trainable_parameters": 100, "total_parameters": 100,
        "wall_clock_seconds": 1.0,
        "n_train_examples": 788 if arm != "A0" else 394,
        "n_train_pos": 394, "n_train_neg": 0 if arm == "A0" else 394,
        "val_metrics": dict(metrics), "test_metrics": dict(metrics),
    }


def _synthetic_cells(scorers=("s1", "s2"), arms=("A0", "A1", "A6"), seeds=(1, 2, 3)):
    """Synthetic cells where A1 helps s1 but not s2 (interaction present)."""
    recs = []
    gains = {("s1", "A1"): 0.10, ("s1", "A6"): 0.05,
             ("s2", "A1"): -0.01, ("s2", "A6"): 0.08}
    for sc in scorers:
        for arm in arms:
            for sd in seeds:
                mrr = 0.40 + gains.get((sc, arm), 0.0) + (sd - 2) * 0.005
                recs.append(_record(sc, arm, sd, mrr))
    return recs


class TestMorganFeaturization:
    def test_valid_smiles(self):
        fp = morgan_fp("CCO")
        assert fp is not None
        assert fp.shape == (2048,)
        assert fp.sum() > 0

    def test_invalid_smiles_returns_none(self):
        assert morgan_fp("not_a_smiles[") is None

    def test_tanimoto_identical_is_one(self):
        assert tanimoto_to_gold("CCO", "CCO") == pytest.approx(1.0)

    def test_tanimoto_different_less_than_one(self):
        sim = tanimoto_to_gold("CCO", "c1ccccc1")
        assert sim is not None and 0.0 <= sim < 0.9

    def test_tanimoto_invalid_returns_none(self):
        assert tanimoto_to_gold("bad[", "CCO") is None


class TestMorganMLPScorer:
    def test_forward_shape(self):
        model = MorganMLPScorer()
        x = torch.randn(4, 2048)
        assert model(x).shape == (4,)

    def test_parameter_count(self):
        model = MorganMLPScorer()
        n = sum(p.numel() for p in model.parameters())
        # 2048*512 + 512 + 512*256 + 256 + 256*1 + 1 = 1,180,673
        assert n == 1_180_673

    def test_deterministic_in_eval(self):
        torch.manual_seed(0)
        model = MorganMLPScorer(dropout=0.0).eval()
        x = torch.randn(2, 2048)
        assert torch.allclose(model(x), model(x))


class TestCellTable:
    def test_delta_vs_a0(self):
        recs = [_record("s1", "A0", 1, 0.3), _record("s1", "A1", 1, 0.45)]
        rows = build_cell_table(recs)
        a1 = next(r for r in rows if r["source"] == "A1")
        assert a1["delta_vs_A0"] == pytest.approx(0.15)

    def test_missing_a0_gives_none(self):
        recs = [_record("s1", "A1", 1, 0.45)]
        rows = build_cell_table(recs)
        assert rows[0]["delta_vs_A0"] is None


class TestCellEffectSizes:
    def test_positive_effect(self):
        recs = [_record("s1", "A0", s, 0.30) for s in range(10)]
        recs += [_record("s1", "A1", s, 0.40) for s in range(10)]
        eff = cell_effect_sizes(recs)
        assert "s1" in eff and "A1" in eff["s1"]
        assert eff["s1"]["A1"]["pp_diff"] == pytest.approx(10.0, abs=0.01)
        assert eff["s1"]["A1"]["ci_low"] > 0

    def test_no_effect_ci_crosses_zero(self):
        recs = [_record("s1", "A0", s, 0.30 + 0.01 * (s % 3)) for s in range(10)]
        recs += [_record("s1", "A2", s, 0.30 + 0.01 * ((s + 1) % 3)) for s in range(10)]
        eff = cell_effect_sizes(recs)
        assert eff["s1"]["A2"]["ci_low"] < 0 < eff["s1"]["A2"]["ci_high"]


class TestInteractionModel:
    def test_anova_detects_interaction(self):
        recs = _synthetic_cells()
        rows = build_cell_table(recs)
        anova = interaction_anova(rows)
        assert "C(source):C(scorer)" in anova
        assert anova["n_obs"] == 2 * 2 * 3  # scorers × (A1,A6) × seeds

    def test_mixed_effects_runs(self):
        recs = _synthetic_cells()
        rows = build_cell_table(recs)
        mixed = mixed_effects_model(rows)
        assert "formula" in mixed


class TestDifficultyProfile:
    def test_profile_structure(self, tmp_path):
        manifest = {"groups": [{
            "group_id": "g0",
            "candidates": [
                {"candidate_smiles": "CCO", "candidate_source": "gold",
                 "gold_candidate": True},
                {"candidate_smiles": "CCO", "candidate_source": "tanimoto_retrieval",
                 "gold_candidate": False},
                {"candidate_smiles": "c1ccccc1", "candidate_source": "random_mismatch",
                 "gold_candidate": False},
            ]}]}
        path = tmp_path / "m.json"
        path.write_text(json.dumps(manifest))
        prof = difficulty_profile(path)
        assert prof["tanimoto_retrieval"]["mean"] == pytest.approx(1.0)
        assert prof["tanimoto_retrieval"]["valid_fraction"] == 1.0
        assert prof["random_mismatch"]["mean"] < 0.5
        for src in prof:
            assert {"n_total", "n_valid_smiles", "valid_fraction",
                    "mean", "std", "p10", "p50", "p90"} <= set(prof[src])

    def test_profile_tracks_invalid_smiles(self, tmp_path):
        """Unparseable SMILES must lower valid_fraction, not be silently dropped."""
        manifest = {"groups": [{
            "group_id": "g0",
            "candidates": [
                {"candidate_smiles": "CCO", "candidate_source": "gold",
                 "gold_candidate": True},
                {"candidate_smiles": "CCO", "candidate_source": "random_corruption",
                 "gold_candidate": False},
                {"candidate_smiles": "not_a_smiles[##", "candidate_source": "random_corruption",
                 "gold_candidate": False},
                {"candidate_smiles": "also_bad[", "candidate_source": "random_corruption",
                 "gold_candidate": False},
            ]}]}
        path = tmp_path / "m.json"
        path.write_text(json.dumps(manifest))
        prof = difficulty_profile(path)
        rc = prof["random_corruption"]
        assert rc["n_total"] == 3
        assert rc["n_valid_smiles"] == 1
        assert rc["valid_fraction"] == pytest.approx(1 / 3, abs=1e-3)


class TestHypothesisDiagnosis:
    def _effects(self):
        return {
            "chemformer": {
                "A1": {"pp_diff": 12.0, "ci_low": 0.05, "ci_high": 0.2,
                       "baseline_mean_mrr": 0.29, "cohens_d": 1.5},
                "A6": {"pp_diff": 15.0, "ci_low": 0.1, "ci_high": 0.2,
                       "baseline_mean_mrr": 0.29, "cohens_d": 2.0},
            },
            "gnn": {
                "A1": {"pp_diff": 4.0, "ci_low": 0.01, "ci_high": 0.08,
                       "baseline_mean_mrr": 0.41, "cohens_d": 1.0},
                "A6": {"pp_diff": -0.01, "ci_low": -0.02, "ci_high": 0.02,
                       "baseline_mean_mrr": 0.41, "cohens_d": 0.0},
            },
        }

    def test_h2_rejected_when_chemformer_weakest_baseline(self):
        anova = {"C(source):C(scorer)": {"p_value": 0.001}}
        difficulty = {"random_corruption": {"mean": 0.2}}
        dup = {"duplicated": True}
        h = diagnose_hypotheses(self._effects(), anova, difficulty, dup, True)
        assert h["H2_chemformer_simply_stronger"]["verdict"] == "REJECTED"

    def test_h5_confirmed_with_small_p(self):
        anova = {"C(source):C(scorer)": {"p_value": 0.001}}
        h = diagnose_hypotheses(self._effects(), anova, {}, {}, True)
        assert h["H5_source_x_scorer_interaction"]["verdict"] == "CONFIRMED"

    def test_h1_untestable(self):
        h = diagnose_hypotheses(self._effects(), {}, {}, {}, True)
        assert h["H1_pc_cng_intrinsically_valuable"]["verdict"] == "UNTESTABLE"

    def test_smiles_validity_flags_text_artifact_arms(self):
        """Majority-invalid sources must be flagged with their arms."""
        difficulty = {
            "random_corruption": {"mean": 0.8, "valid_fraction": 0.07},
            "unconstrained_edit": {"mean": 0.17, "valid_fraction": 0.05},
            "tanimoto_retrieval": {"mean": 0.15, "valid_fraction": 1.0},
        }
        h = diagnose_hypotheses(self._effects(), {}, difficulty, {}, True)
        sv = h["smiles_validity"]
        assert set(sv["invalid_sources_majority_unparseable"]) == {
            "random_corruption", "unconstrained_edit"}
        assert set(sv["text_artifact_arms"]) == {"A2", "A5"}

    def test_smiles_validity_empty_when_all_valid(self):
        difficulty = {"tanimoto_retrieval": {"mean": 0.15, "valid_fraction": 1.0}}
        h = diagnose_hypotheses(self._effects(), {}, difficulty, {}, True)
        assert h["smiles_validity"]["invalid_sources_majority_unparseable"] == {}
        assert h["smiles_validity"]["text_artifact_arms"] == []

    def test_a6_positive_scorer_count(self):
        h = diagnose_hypotheses(self._effects(), {}, {}, {}, True)
        assert h["n_scorers_with_a6_positive"] == 1
        assert h["pc_cng_a6_positive_ci_scorers"] == ["chemformer"]


class TestSummaryCSV:
    def test_roundtrip(self, tmp_path):
        recs = [_record("morgan_mlp", "A1", 1, 0.5)]
        path = tmp_path / "s.csv"
        write_summary_csv(recs, path)
        import csv
        rows = list(csv.DictReader(open(path)))
        assert len(rows) == 1
        assert rows[0]["backbone"] == "morgan_mlp"
        assert float(rows[0]["test_mrr"]) == pytest.approx(0.5)


class TestIntegration:
    @pytest.fixture
    def manifest_path(self):
        p = Path("data/p4/manifests/hte_feasibility_v1.json")
        if not p.exists():
            pytest.skip("v1 manifest not found")
        return p

    def test_difficulty_profile_real_manifest(self, manifest_path):
        prof = difficulty_profile(manifest_path)
        assert len(prof) >= 6
        # Tanimoto-retrieved negatives should be most similar to gold
        assert prof["tanimoto_retrieval"]["mean"] > prof["random_mismatch"]["mean"]

    def test_g3_dirs_present(self):
        for d in ("results/p4_augmentation_chemformer/summary.csv",
                  "results/p4_augmentation_gnn/summary.csv"):
            if not Path(d).exists():
                pytest.skip(f"{d} not present")
