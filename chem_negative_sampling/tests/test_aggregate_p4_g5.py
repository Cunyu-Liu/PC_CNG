"""Tests for P4-G5 aggregation and go/no-go gate (pc_cng.aggregate_p4_g5).

Fixture runs are synthetic but carry the full run-json schema produced by
run_p4_risk_aware.run_single, so verdict logic and contract files are
exercised end-to-end without GPU training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_CNS_ROOT = Path(__file__).resolve().parents[1]
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from pc_cng.aggregate_p4_g5 import (  # noqa: E402
    compute_go_no_go,
    load_ablation_runs,
    load_runs,
    main as aggregate_main,
    write_ablation_csv,
    write_phase_contract_files,
    write_summary_csv,
)

SEEDS = [101, 102, 103, 104]


def _run(method, seed, *, mrr=0.4, auprc=0.2, ece=0.3, ff_mrr=0.21,
         kp_rec=0.45, coll_hr=0.0, sel_risk=0.25, drop=0.01):
    return {
        "phase": "P4-G5",
        "method": method,
        "seed": seed,
        "ablate": [],
        "trainable_parameters": 180737,
        "n_train": 1000,
        "n_train_pos": 500,
        "pu_prior": 0.1,
        "best_epoch": 1,
        "val_mrr_per_epoch": [mrr, mrr + drop],
        "wall_clock_seconds": 40.0,
        "val_metrics": {"mrr": mrr, "top1": 0.2, "auprc": auprc,
                        "ece": ece, "brier": 0.2, "nll": 0.6},
        "test_metrics": {"mrr": mrr, "top1": 0.2, "auprc": auprc,
                         "ece": ece, "brier": 0.2, "nll": 0.6},
        "fixed_forward_test_mrr": ff_mrr,
        "stress": {
            "known_positive": {"n": 10, "recovery_top1": kp_rec, "mean_prob": 0.5},
            "near_positive": {"n": 10, "hard_reject_rate": 0.1, "fnr_corr": 0.3},
            "ood_family": {"n": 10, "ece": ece, "brier": 0.2, "nll": 0.6},
            "collision_sensitivity": {"n": 5, "hard_reject_rate": coll_hr,
                                      "mean_score": 0.1},
        },
        "selective": {"risk_at_0p8": sel_risk, "auc": 0.1,
                      "coverage": [0.8, 1.0], "risk": [sel_risk, sel_risk + 0.02]},
        "training_stability": {"max_val_mrr_drop": drop},
    }


def _runs(baseline_kw=None, **method_kw):
    """{method: [runs]} with baseline hard_binary plus challengers."""
    out = {"hard_binary": [_run("hard_binary", s, **(baseline_kw or {})) for s in SEEDS]}
    for method, kw in method_kw.items():
        out[method] = [_run(method, s, **kw) for s in SEEDS]
    return out


_RISK_MANIFEST = {
    "risk_model": {
        "feature_names": ["database_exact_collision", "ensemble_mean",
                          "epistemic_uncertainty"],
        "coef": [0.5, 0.2, 0.1],
    }
}

_RISK_MANIFEST_DOMINATED = {
    "risk_model": {
        "feature_names": ["database_exact_collision", "ensemble_mean",
                          "epistemic_uncertainty"],
        "coef": [0.05, 0.6, 0.35],
    }
}


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

class TestVerdict:
    def test_go_when_two_criteria_incl_external(self):
        runs = _runs(
            risk_weighted_pairwise=dict(auprc=0.30, ff_mrr=0.30, ece=0.20),
        )
        go = compute_go_no_go(runs, _RISK_MANIFEST)
        assert go["status"] == "GO"
        assert go["n_criteria_satisfied"] >= 2
        assert go["external_task_metric_satisfied"]
        assert go["next_phase_allowed"] is True

    def test_partial_go_calibration_only(self):
        # External metrics identical to baseline, ECE reduced >=20% with CI<0
        runs = _runs(
            risk_weighted_pairwise=dict(auprc=0.2, ff_mrr=0.21, ece=0.20),
        )
        go = compute_go_no_go(runs, _RISK_MANIFEST)
        assert go["status"] == "PARTIAL_GO"
        assert go["criteria"]["ece_relative_reduction_20"]["satisfied"]
        assert not go["external_task_metric_satisfied"]
        assert go["next_phase_allowed"] is True

    def test_no_go_simultaneous_degradation(self):
        # Challenger significantly worse on AUPRC and ECE, nothing satisfied
        runs = _runs(pu_nnpu=dict(auprc=0.15, ece=0.40, sel_risk=0.30))
        go = compute_go_no_go(runs, _RISK_MANIFEST)
        assert go["status"] == "NO_GO"
        assert any("simultaneous" in r for r in go["nogo_reasons"])
        assert go["next_phase_allowed"] is False

    def test_no_go_known_positive_severe_failure(self):
        runs = _runs(
            risk_weighted_pairwise=dict(auprc=0.30, ff_mrr=0.30, ece=0.20,
                                        kp_rec=0.10),
        )
        go = compute_go_no_go(runs, _RISK_MANIFEST)
        assert go["status"] == "NO_GO"
        assert any("known_positive" in r for r in go["nogo_reasons"])

    def test_no_go_self_score_dominance(self):
        runs = _runs(
            risk_weighted_pairwise=dict(auprc=0.30, ff_mrr=0.30, ece=0.20),
        )
        go = compute_go_no_go(runs, _RISK_MANIFEST_DOMINATED)
        assert go["status"] == "NO_GO"
        assert any("self_score" in r for r in go["nogo_reasons"])
        assert go["fnr_self_score_coef_share"] == pytest.approx(0.95 / 1.0, abs=1e-6)

    def test_no_go_identical_methods(self):
        runs = _runs(label_smoothing=dict())
        go = compute_go_no_go(runs, _RISK_MANIFEST)
        assert go["status"] == "NO_GO"
        assert go["n_criteria_satisfied"] == 0

    def test_missing_baseline(self):
        go = compute_go_no_go({"pu_nnpu": [_run("pu_nnpu", s) for s in SEEDS]}, None)
        assert go["status"] == "NO_GO"
        assert "baseline" in go["reason"]

    def test_go_no_go_schema_contract(self):
        runs = _runs(label_smoothing=dict())
        go = compute_go_no_go(runs, _RISK_MANIFEST)
        for key in ("phase", "status", "primary_metric", "predeclared_threshold",
                    "evidence_paths", "limitations", "next_phase_allowed"):
            assert key in go, f"missing contract key: {key}"
        assert go["phase"] == "P4-G5"
        assert go["status"] in ("GO", "PARTIAL_GO", "NO_GO", "DEFERRED")
        assert go["primary_metric"]["name"] == "hte_auprc"
        assert isinstance(go["limitations"], list)


# ---------------------------------------------------------------------------
# CSVs and contract files
# ---------------------------------------------------------------------------

@pytest.fixture()
def staging_dir(tmp_path):
    """A fake output_dir with main + ablation runs on disk."""
    out = tmp_path / "p4_risk_aware"
    for method in ("hard_binary", "risk_weighted_pairwise"):
        d = out / "runs" / method
        d.mkdir(parents=True)
        for s in SEEDS:
            with open(d / f"seed_{s}.json", "w") as f:
                json.dump(_run(method, s, ece=0.20 if method != "hard_binary" else 0.3), f)
    d = out / "ablation" / "chemical_validity"
    d.mkdir(parents=True)
    for s in SEEDS:
        with open(d / f"seed_{s}.json", "w") as f:
            json.dump(_run("risk_weighted_pairwise", s, auprc=0.19), f)
    return out


def test_load_runs_and_ablation(staging_dir):
    runs = load_runs(staging_dir)
    abl = load_ablation_runs(staging_dir)
    assert sorted(runs) == ["hard_binary", "risk_weighted_pairwise"]
    assert len(runs["hard_binary"]) == len(SEEDS)
    assert sorted(abl) == ["chemical_validity"]
    assert len(abl["chemical_validity"]) == len(SEEDS)


def test_summary_and_ablation_csv(staging_dir):
    runs = load_runs(staging_dir)
    abl = load_ablation_runs(staging_dir)
    write_summary_csv(runs, staging_dir / "summary.csv")
    write_ablation_csv(abl, runs["risk_weighted_pairwise"], staging_dir / "ablation.csv")
    summary = (staging_dir / "summary.csv").read_text().strip().splitlines()
    assert len(summary) == 1 + 2 * len(SEEDS)
    abl_lines = (staging_dir / "ablation.csv").read_text().strip().splitlines()
    assert len(abl_lines) == 2
    assert "delta_mrr_vs_full" in abl_lines[0]


def test_contract_files_written(staging_dir):
    runs = load_runs(staging_dir)
    abl = load_ablation_runs(staging_dir)
    go = compute_go_no_go(runs, _RISK_MANIFEST)
    write_phase_contract_files(staging_dir, runs, abl, go)

    for name in ("run_manifest.json", "environment.json", "input_hashes.json",
                 "commands.log", "go_no_go.json"):
        assert (staging_dir / name).exists(), name

    rm = json.load(open(staging_dir / "run_manifest.json"))
    assert rm["n_main_runs"] == 2 * len(SEEDS)
    assert rm["n_ablation_runs"] == len(SEEDS)
    assert rm["ablated_components"] == ["chemical_validity"]

    go2 = json.load(open(staging_dir / "go_no_go.json"))
    assert go2["evidence_paths"], "evidence_paths must be filled by contract writer"
    assert any("summary.csv" in p for p in go2["evidence_paths"])

    ih = json.load(open(staging_dir / "input_hashes.json"))
    assert ih["phase"] == "P4-G5"
    # Missing input files on tmp staging -> None, but keys must exist
    assert "manifest_v2" in ih["sha256"]
    assert "risk_artifacts" in ih["sha256"]


def test_full_main_pipeline(staging_dir, monkeypatch):
    """End-to-end: aggregate_main writes all outputs incl. go_no_go."""
    monkeypatch.setattr(sys, "argv", ["aggregate_p4_g5", "--output-dir", str(staging_dir)])
    aggregate_main()
    go = json.load(open(staging_dir / "go_no_go.json"))
    # Challenger has ECE reduced (0.2 vs 0.3) with same external metrics
    assert go["status"] == "PARTIAL_GO"
    assert (staging_dir / "summary.csv").exists()
    assert (staging_dir / "ablation.csv").exists()
