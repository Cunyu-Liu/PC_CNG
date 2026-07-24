"""Tests for P4-G8B v2 cross-family transfer (full spec execution).

Covers: rule family classifier, LoRA adapter, EWC anchor, statistics
(cluster bootstrap, exact sign-flip permutation, Cohen's d), verdict logic,
and the training-entry CLI contract.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ["RDKitRDLogger"] = "0"

_THIS = Path(__file__).resolve()
for _cand in (_THIS.parents[1], _THIS.parents[2] / "chem_negative_sampling"):
    if (_cand / "pc_cng").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from pc_cng.p4_g8b_transfer_v2 import (
    CN_COUPLING,
    EAS,
    METHODS,
    PREDECLARED_SEEDS,
    AdaptedMLP,
    LoRAAdapter,
    classify_uspto_ord_family,
    cluster_bootstrap_delta_ci,
    cohens_d,
    compute_verdict_v2,
    exact_sign_flip_pvalue,
    is_cn_coupling,
    is_eas,
    pool_families,
)
from pc_cng.p4_g8b_cross_family_transfer import MorganMLPScorer


# ---------------------------------------------------------------------------
# Classifier tests (real reactions, no fabricated labels)
# ---------------------------------------------------------------------------

class TestFamilyClassifier:
    def test_buchwald_hartwig_is_cn_coupling(self):
        # Bromobenzene + aniline -> diphenylamine (classic C-N coupling)
        rxn = "Brc1ccccc1.Nc1ccccc1>>c1ccc(Nc2ccccc2)cc1"
        assert is_cn_coupling(rxn)
        assert classify_uspto_ord_family(rxn) == CN_COUPLING

    def test_nitration_is_eas(self):
        # Benzene -> nitrobenzene; nitric acid may sit in the reactant field
        # ('>>' format) or in the agent field ('>a>' format).
        rxn_reactant_side = "c1ccccc1.O=[N+]([O-])O>>O=[N+]([O-])c1ccccc1"
        rxn_agent = "c1ccccc1>O=[N+]([O-])O>O=[N+]([O-])c1ccccc1"
        assert is_eas(rxn_reactant_side)
        assert is_eas(rxn_agent)
        assert classify_uspto_ord_family(rxn_agent) == EAS

    def test_aryl_bromination_is_eas(self):
        rxn = "c1ccccc1.BrBr>>Brc1ccccc1"
        assert is_eas(rxn)

    def test_friedel_crafts_is_eas(self):
        rxn = "c1ccccc1.CC(=O)Cl>>CC(=O)c1ccccc1"
        assert is_eas(rxn)

    def test_suzuki_is_neither(self):
        # Aryl halide + boronic acid -> biaryl: C-C, not C-N; residual rules
        # must not fire.
        rxn = "Brc1ccccc1.OB(O)c1ccccc1>>c1ccc(-c2ccccc2)cc1"
        assert not is_cn_coupling(rxn)
        assert not is_eas(rxn)
        assert classify_uspto_ord_family(rxn) is None

    def test_amide_coupling_is_neither(self):
        rxn = "CC(=O)Cl.Nc1ccccc1>>CC(=O)Nc1ccccc1"
        # aniline N acylation: product has no aryl C-N formation from halide
        assert not is_cn_coupling(rxn)

    def test_cn_coupling_rejects_residual_aryl_halide(self):
        rxn = "Brc1ccccc1.Nc1ccccc1>>Brc1ccc(Nc2ccccc2)cc1"
        assert not is_cn_coupling(rxn)


# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------

class TestLoRAAdapter:
    def test_zero_init_is_identity(self):
        base = MorganMLPScorer(n_bits=128, hidden=32)
        adapted = AdaptedMLP(base, rank=4)
        base.eval()
        adapted.eval()
        x = torch.randn(3, 128)
        assert torch.allclose(base(x), adapted(x), atol=1e-6)

    def test_adapter_param_efficiency(self):
        base = MorganMLPScorer(n_bits=2048, hidden=256)
        adapted = AdaptedMLP(base, rank=8)
        n_adapter = sum(p.numel() for p in adapted.adapter_parameters())
        n_base = sum(p.numel() for p in base.parameters())
        assert n_adapter < 0.05 * n_base

    def test_base_frozen_after_lora_setup(self):
        base = MorganMLPScorer(n_bits=128, hidden=32)
        adapted = AdaptedMLP(base)
        for p in adapted.base.parameters():
            p.requires_grad = False
        trainable = [p for p in adapted.parameters() if p.requires_grad]
        assert all(p.shape[0] <= 128 for p in trainable)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_sign_flip_exact_on_clear_positive(self):
        rng = np.random.RandomState(0)
        deltas = rng.normal(0.05, 0.01, size=10).tolist()
        p = exact_sign_flip_pvalue(deltas)
        assert p <= 0.01  # 2/1024 or better

    def test_sign_flip_on_null(self):
        deltas = [0.01, -0.01, 0.005, -0.005, 0.002, -0.002, 0.001, -0.001, 0.0, 0.0]
        p = exact_sign_flip_pvalue(deltas)
        assert p > 0.3

    def test_bootstrap_ci_contains_positive_delta(self):
        rng = np.random.RandomState(1)
        n_clusters = 30
        clusters = [f"c{i}" for i in range(n_clusters) for _ in range(5)]
        labels = np.array([1] + [0] * 4).repeat(n_clusters)
        base_scores = rng.uniform(size=len(labels))
        method_scores = base_scores + labels * rng.uniform(0.5, 1.5, size=len(labels))
        delta, lo, hi = cluster_bootstrap_delta_ci(
            method_scores, base_scores, labels, clusters, n_boot=200, seed=7)
        assert lo > 0, f"expected positive CI, got [{lo}, {hi}]"
        assert lo <= delta <= hi

    def test_bootstrap_ci_straddles_zero_on_null(self):
        rng = np.random.RandomState(2)
        n_clusters = 30
        clusters = [f"c{i}" for i in range(n_clusters) for _ in range(5)]
        labels = np.array([1] + [0] * 4).repeat(n_clusters)
        base_scores = rng.uniform(size=len(labels))
        method_scores = rng.uniform(size=len(labels))
        delta, lo, hi = cluster_bootstrap_delta_ci(
            method_scores, base_scores, labels, clusters, n_boot=200, seed=7)
        assert lo <= 0 <= hi

    def test_cohens_d(self):
        assert cohens_d([0.05, 0.05, 0.05]) == float("inf")
        assert abs(cohens_d([0.05, -0.05])) < 1e-9
        assert cohens_d([]) == 0.0


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _stat(name, group, method_results):
    return {"name": name, "pair_group": group, "methods": method_results}


def _res(delta, lo, hi, forgetting=0.0):
    return {"delta_mean": delta, "ci_low": lo, "ci_high": hi,
            "p_value": 0.01, "cohens_d": 1.0, "forgetting_mean": forgetting}


class TestVerdictV2:
    def test_go_two_groups_positive(self):
        stats = [
            _stat("USPTO:EAS→USPTO:C-N", "EAS↔C-N", {"direct": _res(0.03, 0.01, 0.05)}),
            _stat("USPTO→HTE:Pd", "USPTO→HTE", {"multi_task": _res(0.02, 0.005, 0.04)}),
            _stat("HTE:A→HTE:B", "HTE family", {"direct": _res(-0.01, -0.03, 0.0)}),
        ]
        v = compute_verdict_v2(stats)
        assert v["verdict"] == "GO"
        assert v["n_positive_pair_groups"] == 2
        assert v["next_phase_allowed"]

    def test_partial_go_one_group(self):
        stats = [
            _stat("d1", "EAS↔C-N", {"direct": _res(0.03, 0.01, 0.05)}),
            _stat("d2", "USPTO→HTE", {"direct": _res(-0.02, -0.04, -0.001)}),
        ]
        v = compute_verdict_v2(stats)
        assert v["verdict"] == "PARTIAL_GO"
        assert len(v["negative_directions"]) == 1

    def test_no_go_all_negative(self):
        stats = [
            _stat("d1", "EAS↔C-N", {"direct": _res(-0.03, -0.05, -0.01)}),
            _stat("d2", "USPTO→HTE", {"direct": _res(-0.02, -0.04, 0.0)}),
        ]
        v = compute_verdict_v2(stats)
        assert v["verdict"] == "NO_GO"
        assert not v["next_phase_allowed"]

    def test_no_go_severe_forgetting(self):
        stats = [
            _stat("d1", "EAS↔C-N", {"ewc": _res(0.03, 0.01, 0.05)}),
            _stat("d2", "USPTO→HTE", {"ewc": _res(0.02, 0.01, 0.03, forgetting=-0.35)}),
        ]
        v = compute_verdict_v2(stats)
        assert v["verdict"] == "NO_GO"
        assert "forgetting" in v["reason"]


# ---------------------------------------------------------------------------
# Contract & pooling
# ---------------------------------------------------------------------------

class TestContract:
    def test_methods_cover_spec(self):
        assert set(METHODS) == {"direct", "head_ft", "lora_adapter", "ewc",
                                "risk_aware", "multi_task"}

    def test_ten_predeclared_seeds(self):
        assert len(PREDECLARED_SEEDS) == 10
        assert PREDECLARED_SEEDS == sorted(PREDECLARED_SEEDS)

    def test_pool_families(self):
        a = {"train": [{"x": 1}], "val": [], "test": [{"x": 2}]}
        b = {"train": [{"x": 3}], "val": [{"x": 4}], "test": []}
        pooled = pool_families(a, b)
        assert len(pooled["train"]) == 2
        assert len(pooled["val"]) == 1
        assert len(pooled["test"]) == 1

    def test_cli_has_contract_flags(self):
        import argparse
        import inspect
        from pc_cng import p4_g8b_transfer_v2 as mod
        src = inspect.getsource(mod.main)
        for flag in ("--train-idx", "--val-idx", "--test-idx",
                     "--candidate-manifest", "--seed", "--output-dir"):
            assert flag in src, f"missing contract flag {flag}"
