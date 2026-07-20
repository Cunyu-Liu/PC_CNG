"""Unit tests for ``hte_eval.py`` (P3-05).

These tests run on CPU only and use synthetic SMILES / CSV data so they
do not require GPU or remote-server access.  They cover:
- Fingerprint computation (Morgan, reaction difference)
- Random negative generation (atom swap + bond deletion)
- PC-CNG negatives loader (gracefully handles missing file)
- Ranking metrics (Top-1 / MRR / NDCG@10)
- Leave-one-out evaluation end-to-end across all 3 strategies
- Family-cluster bootstrap CI
- CLI ``main()`` smoke test with a synthetic HTE CSV
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Make modules importable in both layouts.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

try:  # server layout: chem_negative_sampling/tests/test_hte_eval.py
    from evaluation.hte_eval import (  # type: ignore[import]
        EvaluationResult,
        SeedMetrics,
        build_arg_parser,
        compute_mrr,
        compute_ndcg_at_k,
        compute_top1,
        evaluate_leave_one_out,
        family_cluster_bootstrap_ci,
        generate_random_negatives,
        load_pc_cng_negatives,
        main,
        morgan_fingerprint,
        reaction_fingerprint,
        run_evaluation,
    )
except ImportError:  # flat local layout
    from hte_eval import (  # type: ignore[no-redef]
        EvaluationResult,
        SeedMetrics,
        build_arg_parser,
        compute_mrr,
        compute_ndcg_at_k,
        compute_top1,
        evaluate_leave_one_out,
        family_cluster_bootstrap_ci,
        generate_random_negatives,
        load_pc_cng_negatives,
        main,
        morgan_fingerprint,
        reaction_fingerprint,
        run_evaluation,
    )

# Force CPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

# A handful of small valid reaction SMILES for testing.  Each is a real
# organic transformation with a distinct product so the fingerprints differ.
POSITIVE_REACTIONS = [
    "CCO.CC(=O)O>>CC(=O)OCC",  # esterification -> ethyl acetate
    "CC(C)O.CC(=O)O>>CC(=O)OC(C)C",  # isopropyl acetate
    "CCO.OC(=O)C>>CC(=O)OCC",  # alt. esterification
    "CCCCO.CC(=O)O>>CC(=O)OCCCC",  # butyl acetate
    "CCO.CCC(=O)O>>CCC(=O)OCC",  # propionate ester
    "CCCO.CC(=O)O>>CC(=O)OCCC",  # propyl acetate
    "CCO.O=C(O)C>>O=C(O)COCC",  # glycolate ester
    "CC(C)(C)O.CC(=O)O>>CC(=O)OC(C)(C)C",  # tert-butyl acetate
]


@pytest.fixture
def small_reaction_pool() -> list:
    return list(POSITIVE_REACTIONS)


@pytest.fixture
def pc_cng_csv(tmp_path: Path) -> str:
    """A tiny PC-CNG negatives CSV with a reaction_smiles column."""
    path = tmp_path / "pc_cng_negs.csv"
    rows = [
        {"reaction_smiles": "CCO.CC(=O)O>>CC(=O)OCCC", "label": "negative"},
        {"reaction_smiles": "CCO.CC(=O)O>>CC(=O)OCC", "label": "negative"},
        {"reaction_smiles": "CCO.CC(=O)O>>CCC(=O)OCC", "label": "negative"},
        {"reaction_smiles": "CCO.CC(=O)O>>CC(=O)OC(C)C", "label": "negative"},
        {"reaction_smiles": "CCO.CC(=O)O>>CC(=O)OCCCC", "label": "negative"},
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["reaction_smiles", "label"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return str(path)


@pytest.fixture
def hte_csv(tmp_path: Path) -> str:
    """A synthetic HTEa CSV with 2 classes, each with 12 reactions."""
    path = tmp_path / "hte.csv"
    fields = [
        "source_id", "reaction_smiles", "reactants", "agents", "products",
        "label_type", "yield", "source", "split_key", "split", "reaction_class",
    ]
    rows = []
    # Class "Ester" with 12 reactions (using our POSITIVE_REACTIONS + variants)
    for i, rxn in enumerate(POSITIVE_REACTIONS):
        rows.append({
            "source_id": f"e{i}", "reaction_smiles": rxn,
            "reactants": rxn.split(">>")[0], "agents": "", "products": rxn.split(">>")[1],
            "label_type": "positive", "yield": str(50 + i), "source": "hitea",
            "split_key": "k", "split": "train", "reaction_class": "Ester",
        })
    # Add a few more Ester reactions to reach 12
    extra = [
        "CCO.C(=O)CO>>C(=O)COCC",
        "CCO.CCC(=O)O>>CCC(=O)OCC",
        "CCO.O=C(O)CC>>O=C(O)CCOCC",
        "CCO.CCCCC(=O)O>>CCCCC(=O)OCC",
    ]
    for i, rxn in enumerate(extra):
        rows.append({
            "source_id": f"e2{i}", "reaction_smiles": rxn,
            "reactants": rxn.split(">>")[0], "agents": "", "products": rxn.split(">>")[1],
            "label_type": "positive", "yield": str(60 + i), "source": "hitea",
            "split_key": "k", "split": "train", "reaction_class": "Ester",
        })

    # Class "Amine" with 12 reactions
    amine_rxns = [
        "CCN.CC(=O)O>>CC(=O)NCC",
        "CCN.CCC(=O)O>>CCC(=O)NCC",
        "CCN.O=C(O)C>>O=C(O)CNCC",
        "CCN.CCCC(=O)O>>CCCC(=O)NCC",
        "CCN.CCCCC(=O)O>>CCCCC(=O)NCC",
        "CCN.CCCCCC(=O)O>>CCCCCC(=O)NCC",
        "CC(C)N.CC(=O)O>>CC(=O)NC(C)C",
        "CC(C)N.CCC(=O)O>>CCC(=O)NC(C)C",
        "CC(C)N.O=C(O)C>>O=C(O)CNC(C)C",
        "CC(C)N.CCCC(=O)O>>CCCC(=O)NC(C)C",
        "CC(C)N.CCCCC(=O)O>>CCCCC(=O)NC(C)C",
        "CC(C)N.CCCCCC(=O)O>>CCCCCC(=O)NC(C)C",
    ]
    for i, rxn in enumerate(amine_rxns):
        rows.append({
            "source_id": f"m{i}", "reaction_smiles": rxn,
            "reactants": rxn.split(">>")[0], "agents": "", "products": rxn.split(">>")[1],
            "label_type": "positive", "yield": str(55 + i), "source": "hitea",
            "split_key": "k", "split": "train", "reaction_class": "Amine",
        })

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return str(path)


# ---------------------------------------------------------------------------
# Fingerprint tests
# ---------------------------------------------------------------------------


class TestMorganFingerprint:
    def test_valid_smiles(self):
        fp = morgan_fingerprint("CCO", radius=2, n_bits=2048)
        assert fp is not None
        assert fp.shape == (2048,)
        assert fp.dtype == np.uint8
        # At least one bit set for ethanol
        assert fp.sum() > 0

    def test_invalid_smiles(self):
        assert morgan_fingerprint("not_a_smiles") is None

    def test_empty_smiles(self):
        assert morgan_fingerprint("") is None

    def test_custom_n_bits(self):
        fp = morgan_fingerprint("CCO", radius=2, n_bits=1024)
        assert fp is not None
        assert fp.shape == (1024,)

    def test_custom_radius(self):
        # Use a larger molecule so different radii produce different FPs
        # (ethanol is too small: r=1 and r=3 give the same fingerprint)
        smi = "c1ccccc1C(=O)OCCN(C)C"
        fp1 = morgan_fingerprint(smi, radius=1, n_bits=2048)
        fp3 = morgan_fingerprint(smi, radius=3, n_bits=2048)
        # Different radius should usually produce different fingerprints
        assert not np.array_equal(fp1, fp3)

    def test_deterministic(self):
        fp1 = morgan_fingerprint("CCO", radius=2, n_bits=2048)
        fp2 = morgan_fingerprint("CCO", radius=2, n_bits=2048)
        assert np.array_equal(fp1, fp2)


class TestReactionFingerprint:
    def test_valid_reaction(self):
        fp = reaction_fingerprint("CCO.CC(=O)O>>CC(=O)OCC", radius=2, n_bits=2048)
        assert fp is not None
        assert fp.shape == (2048,)
        assert fp.sum() > 0

    def test_invalid_reaction_no_arrow(self):
        assert reaction_fingerprint("CCO.CC(=O)O") is None

    def test_empty_reaction(self):
        assert reaction_fingerprint("") is None

    def test_empty_reactants(self):
        assert reaction_fingerprint(">>CCO") is None

    def test_empty_products(self):
        assert reaction_fingerprint("CCO>>") is None

    def test_invalid_molecule(self):
        # One invalid molecule on reactant side -> None
        assert reaction_fingerprint("XYZ.ccc>>CCO") is None

    def test_deterministic(self):
        fp1 = reaction_fingerprint("CCO.CC(=O)O>>CC(=O)OCC")
        fp2 = reaction_fingerprint("CCO.CC(=O)O>>CC(=O)OCC")
        assert np.array_equal(fp1, fp2)

    def test_different_reactions_different_fps(self):
        fp1 = reaction_fingerprint("CCO.CC(=O)O>>CC(=O)OCC")
        fp2 = reaction_fingerprint("CCN.CC(=O)O>>CC(=O)NCC")
        assert not np.array_equal(fp1, fp2)


# ---------------------------------------------------------------------------
# Random negative generation tests
# ---------------------------------------------------------------------------


class TestGenerateRandomNegatives:
    def test_generates_requested_count(self, small_reaction_pool):
        negs = generate_random_negatives(small_reaction_pool, n_negatives=5, seed=42)
        assert len(negs) == 5

    def test_negatives_differ_from_positives(self, small_reaction_pool):
        negs = generate_random_negatives(small_reaction_pool, n_negatives=10, seed=42)
        for neg in negs:
            assert neg not in small_reaction_pool

    def test_negatives_have_arrow_format(self, small_reaction_pool):
        negs = generate_random_negatives(small_reaction_pool, n_negatives=5, seed=42)
        for neg in negs:
            assert ">>" in neg

    def test_negatives_preserve_reactants(self, small_reaction_pool):
        """Random perturbation should keep the reactant side unchanged."""
        negs = generate_random_negatives(small_reaction_pool, n_negatives=5, seed=42)
        for neg in negs:
            reactant_side = neg.split(">>")[0]
            # Should match one of the positive reactions' reactant side
            pos_reactants = [r.split(">>")[0] for r in small_reaction_pool]
            assert reactant_side in pos_reactants

    def test_zero_negatives(self, small_reaction_pool):
        negs = generate_random_negatives(small_reaction_pool, n_negatives=0, seed=42)
        assert negs == []

    def test_empty_positives(self):
        negs = generate_random_negatives([], n_negatives=5, seed=42)
        assert negs == []

    def test_deterministic_with_seed(self, small_reaction_pool):
        n1 = generate_random_negatives(small_reaction_pool, n_negatives=5, seed=42)
        n2 = generate_random_negatives(small_reaction_pool, n_negatives=5, seed=42)
        assert n1 == n2

    def test_different_seeds_yield_different_negatives(self, small_reaction_pool):
        n1 = generate_random_negatives(small_reaction_pool, n_negatives=5, seed=42)
        n2 = generate_random_negatives(small_reaction_pool, n_negatives=5, seed=43)
        # Highly unlikely they're identical
        assert n1 != n2

    def test_generated_negatives_parse_as_reactions(self, small_reaction_pool):
        """Each generated negative should have a parseable product side."""
        from rdkit import Chem
        negs = generate_random_negatives(small_reaction_pool, n_negatives=5, seed=42)
        for neg in negs:
            product_smi = neg.split(">>")[1]
            mol = Chem.MolFromSmiles(product_smi)
            assert mol is not None, f"Failed to parse product: {product_smi}"

    def test_returns_fewer_on_invalid_smiles(self):
        # Pool of invalid SMILES -> 0 negatives
        negs = generate_random_negatives(["XYZ>>ABC"], n_negatives=5, seed=42)
        assert negs == []


# ---------------------------------------------------------------------------
# PC-CNG negatives loader tests
# ---------------------------------------------------------------------------


class TestLoadPcCngNegatives:
    def test_loads_valid_csv(self, pc_cng_csv):
        negs = load_pc_cng_negatives(pc_cng_csv)
        assert len(negs) == 5
        assert all(">>" in n for n in negs)

    def test_missing_file_returns_empty(self):
        negs = load_pc_cng_negatives("/nonexistent/path/to/file.csv")
        assert negs == []

    def test_empty_path_returns_empty(self):
        negs = load_pc_cng_negatives("")
        assert negs == []

    def test_max_n_subsample(self, pc_cng_csv):
        negs = load_pc_cng_negatives(pc_cng_csv, max_n=2)
        assert len(negs) == 2

    def test_max_n_larger_than_pool(self, pc_cng_csv):
        negs = load_pc_cng_negatives(pc_cng_csv, max_n=1000)
        assert len(negs) == 5

    def test_missing_reaction_smiles_column(self, tmp_path: Path):
        path = tmp_path / "bad.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["foo", "bar"])
            writer.writeheader()
            writer.writerow({"foo": "1", "bar": "2"})
        negs = load_pc_cng_negatives(str(path))
        assert negs == []

    def test_handles_smiles_lowercase_column(self, tmp_path: Path):
        """Column lookup should be case-insensitive."""
        path = tmp_path / "lower.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["SMILES"])
            writer.writeheader()
            writer.writerow({"SMILES": "CCO>>CCO"})
        negs = load_pc_cng_negatives(str(path))
        assert negs == ["CCO>>CCO"]

    def test_skips_empty_smiles(self, tmp_path: Path):
        path = tmp_path / "gaps.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["reaction_smiles"])
            writer.writeheader()
            writer.writerow({"reaction_smiles": "CCO>>CCO"})
            writer.writerow({"reaction_smiles": ""})
            writer.writerow({"reaction_smiles": "CCN>>CCN"})
        negs = load_pc_cng_negatives(str(path))
        assert negs == ["CCO>>CCO", "CCN>>CCN"]


# ---------------------------------------------------------------------------
# Ranking metrics tests
# ---------------------------------------------------------------------------


class TestRankingMetrics:
    def test_top1_rank1(self):
        assert compute_top1(1) == 1.0

    def test_top1_rank_other(self):
        assert compute_top1(2) == 0.0
        assert compute_top1(10) == 0.0

    def test_mrr(self):
        assert compute_mrr(1) == 1.0
        assert compute_mrr(2) == 0.5
        assert compute_mrr(4) == 0.25
        assert compute_mrr(0) == 0.0

    def test_ndcg_at_k(self):
        # k=10
        assert compute_ndcg_at_k(1, k=10) == pytest.approx(1.0 / math.log2(2))
        assert compute_ndcg_at_k(2, k=10) == pytest.approx(1.0 / math.log2(3))
        # rank > k -> 0
        assert compute_ndcg_at_k(11, k=10) == 0.0
        # k=0 -> 0
        assert compute_ndcg_at_k(1, k=0) == 0.0
        # rank < 1 -> 0
        assert compute_ndcg_at_k(0, k=10) == 0.0

    def test_ndcg_decreasing_with_rank(self):
        vals = [compute_ndcg_at_k(r, k=10) for r in range(1, 11)]
        # Should be strictly decreasing
        for a, b in zip(vals, vals[1:]):
            assert a > b


# Math import used inside TestRankingMetrics.test_ndcg_at_k
import math  # noqa: E402


# ---------------------------------------------------------------------------
# evaluate_leave_one_out tests
# ---------------------------------------------------------------------------


class TestEvaluateLeaveOneOut:
    def test_random_strategy_returns_metrics(self, small_reaction_pool):
        m = evaluate_leave_one_out(
            small_reaction_pool,
            strategy="random",
            n_candidates=5,
            seed=42,
        )
        assert isinstance(m, SeedMetrics)
        assert m.strategy == "random"
        assert m.seed == 42
        assert 0.0 <= m.top1 <= 1.0
        assert 0.0 <= m.mrr <= 1.0
        assert 0.0 <= m.ndcg10 <= 1.0
        assert m.n_evaluated > 0

    def test_none_strategy_returns_metrics(self, small_reaction_pool):
        m = evaluate_leave_one_out(
            small_reaction_pool,
            strategy="none",
            n_candidates=5,
            seed=42,
        )
        assert isinstance(m, SeedMetrics)
        assert m.strategy == "none"
        assert m.n_evaluated > 0

    def test_pc_cng_strategy_with_negatives(self, small_reaction_pool):
        pc_cng_negs = [
            "CCO.CC(=O)O>>CC(=O)OCCC",  # different product
            "CCO.CC(=O)O>>CC(=O)OCCCC",
            "CCO.CC(=O)O>>CC(=O)OCCCCC",
            "CCO.CC(=O)O>>CC(=O)OCCCCCC",
            "CCO.CC(=O)O>>CC(=O)OCCCCCCC",
        ]
        m = evaluate_leave_one_out(
            small_reaction_pool,
            strategy="pc_cng",
            pc_cng_negatives=pc_cng_negs,
            n_candidates=5,
            seed=42,
        )
        assert isinstance(m, SeedMetrics)
        assert m.strategy == "pc_cng"
        assert m.n_evaluated > 0

    def test_pc_cng_strategy_without_negatives_returns_zero(self, small_reaction_pool):
        m = evaluate_leave_one_out(
            small_reaction_pool,
            strategy="pc_cng",
            pc_cng_negatives=[],
            n_candidates=5,
            seed=42,
        )
        assert m.n_evaluated == 0
        assert m.top1 == 0.0

    def test_invalid_strategy(self, small_reaction_pool):
        with pytest.raises(ValueError, match="Unknown strategy"):
            evaluate_leave_one_out(small_reaction_pool, strategy="bogus", seed=42)

    def test_invalid_n_candidates(self, small_reaction_pool):
        with pytest.raises(ValueError, match="n_candidates"):
            evaluate_leave_one_out(small_reaction_pool, strategy="random", n_candidates=0, seed=42)

    def test_too_few_reactions(self):
        m = evaluate_leave_one_out(["CCO>>CCN"], strategy="random", seed=42)
        assert m.n_evaluated == 0
        assert m.top1 == 0.0

    def test_deterministic_with_seed(self, small_reaction_pool):
        m1 = evaluate_leave_one_out(small_reaction_pool, strategy="random", seed=42)
        m2 = evaluate_leave_one_out(small_reaction_pool, strategy="random", seed=42)
        assert m1.top1 == m2.top1
        assert m1.mrr == m2.mrr
        assert m1.ndcg10 == m2.ndcg10

    def test_metrics_in_valid_range(self, small_reaction_pool):
        for strategy in ("random", "none"):
            m = evaluate_leave_one_out(
                small_reaction_pool, strategy=strategy,
                n_candidates=5, seed=42,
            )
            assert 0.0 <= m.top1 <= 1.0
            assert 0.0 <= m.mrr <= 1.0
            assert 0.0 <= m.ndcg10 <= 1.0


# ---------------------------------------------------------------------------
# family_cluster_bootstrap_ci tests
# ---------------------------------------------------------------------------


class TestFamilyClusterBootstrapCi:
    def test_basic_ci(self):
        ranks = [1, 2, 1, 3, 1, 2, 1, 4] * 5
        classes = (["Ester"] * 4 + ["Amine"] * 4) * 5
        lo, hi = family_cluster_bootstrap_ci(
            ranks, classes, n_bootstrap=200, seed=42, metric="mrr",
        )
        assert 0.0 <= lo <= hi <= 1.0

    def test_top1_metric(self):
        ranks = [1, 1, 1, 2, 1, 1, 1, 2] * 5
        classes = (["Ester"] * 4 + ["Amine"] * 4) * 5
        lo, hi = family_cluster_bootstrap_ci(
            ranks, classes, n_bootstrap=200, seed=42, metric="top1",
        )
        assert 0.0 <= lo <= hi <= 1.0

    def test_ndcg_metric(self):
        ranks = [1, 2, 3, 4] * 10
        classes = ["Ester"] * 20 + ["Amine"] * 20
        lo, hi = family_cluster_bootstrap_ci(
            ranks, classes, n_bootstrap=200, seed=42, metric="ndcg", k=10,
        )
        assert 0.0 <= lo <= hi <= 1.0

    def test_empty_inputs(self):
        lo, hi = family_cluster_bootstrap_ci([], [], n_bootstrap=100, metric="mrr")
        assert lo == 0.0 and hi == 0.0

    def test_misaligned_inputs(self):
        with pytest.raises(ValueError, match="must align"):
            family_cluster_bootstrap_ci([1, 2], ["A"], n_bootstrap=10, metric="mrr")

    def test_invalid_n_bootstrap(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            family_cluster_bootstrap_ci([1, 2], ["A", "B"], n_bootstrap=0, metric="mrr")

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            family_cluster_bootstrap_ci([1, 2], ["A", "B"], n_bootstrap=10, metric="bogus")

    def test_deterministic_with_seed(self):
        ranks = [1, 2, 1, 3, 1, 2]
        classes = ["A", "A", "A", "B", "B", "B"]
        lo1, hi1 = family_cluster_bootstrap_ci(ranks, classes, n_bootstrap=100, seed=42, metric="mrr")
        lo2, hi2 = family_cluster_bootstrap_ci(ranks, classes, n_bootstrap=100, seed=42, metric="mrr")
        assert lo1 == lo2 and hi1 == hi2


# ---------------------------------------------------------------------------
# run_evaluation tests
# ---------------------------------------------------------------------------


class TestRunEvaluation:
    def test_full_pipeline_without_pc_cng(self, hte_csv, tmp_path: Path):
        out_dir = tmp_path / "results"
        result = run_evaluation(
            hte_csv=hte_csv,
            pc_cng_negatives_csv=None,
            seeds=[42, 43],
            n_per_class=8,
            min_class_size=10,
            n_candidates=5,
            output_dir=str(out_dir),
            n_bootstrap=50,
        )
        assert isinstance(result, EvaluationResult)
        # 2 seeds * 3 strategies = 6 per_seed entries
        assert len(result.per_seed) == 6
        # All three strategies should be present in summary
        assert "pc_cng" in result.summary
        assert "random" in result.summary
        assert "none" in result.summary
        # PC-CNG should have n_seeds=0 (no negatives file provided)
        assert result.summary["pc_cng"]["n_seeds"] == 0
        # Random and none should have n_seeds=2
        assert result.summary["random"]["n_seeds"] == 2
        assert result.summary["none"]["n_seeds"] == 2
        # Output files should exist
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "summary.md").exists()

    def test_full_pipeline_with_pc_cng(self, hte_csv, pc_cng_csv, tmp_path: Path):
        out_dir = tmp_path / "results_with_pc"
        result = run_evaluation(
            hte_csv=hte_csv,
            pc_cng_negatives_csv=pc_cng_csv,
            seeds=[42],
            n_per_class=8,
            min_class_size=10,
            n_candidates=3,
            output_dir=str(out_dir),
            n_bootstrap=20,
        )
        # All 3 strategies should have n_seeds=1 now
        assert result.summary["pc_cng"]["n_seeds"] == 1
        assert result.summary["random"]["n_seeds"] == 1
        assert result.summary["none"]["n_seeds"] == 1

    def test_no_output_dir(self, hte_csv):
        result = run_evaluation(
            hte_csv=hte_csv,
            pc_cng_negatives_csv=None,
            seeds=[42],
            n_per_class=5,
            min_class_size=5,
            n_candidates=3,
            output_dir=None,
            n_bootstrap=10,
        )
        assert isinstance(result, EvaluationResult)

    def test_metrics_json_is_valid(self, hte_csv, tmp_path: Path):
        out_dir = tmp_path / "json_check"
        run_evaluation(
            hte_csv=hte_csv,
            pc_cng_negatives_csv=None,
            seeds=[42],
            n_per_class=5,
            min_class_size=5,
            n_candidates=3,
            output_dir=str(out_dir),
            n_bootstrap=10,
        )
        with open(out_dir / "metrics.json") as fh:
            data = json.load(fh)
        assert "meta" in data
        assert "per_seed" in data
        assert "summary" in data
        assert "bootstrap_ci" in data
        assert data["meta"]["n_classes"] >= 1

    def test_summary_md_has_table(self, hte_csv, tmp_path: Path):
        out_dir = tmp_path / "md_check"
        run_evaluation(
            hte_csv=hte_csv,
            pc_cng_negatives_csv=None,
            seeds=[42],
            n_per_class=5,
            min_class_size=5,
            n_candidates=3,
            output_dir=str(out_dir),
            n_bootstrap=10,
        )
        md = (out_dir / "summary.md").read_text()
        assert "P3-05" in md
        assert "Top-1" in md
        assert "Strategy" in md

    def test_raises_on_empty_classes(self, tmp_path: Path):
        """If no class has >= min_class_size, run_evaluation should raise."""
        empty_csv = tmp_path / "empty.csv"
        fields = ["source_id", "reaction_smiles", "reaction_class"]
        with open(empty_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
        with pytest.raises(ValueError, match="No HTE classes"):
            run_evaluation(
                hte_csv=str(empty_csv),
                pc_cng_negatives_csv=None,
                seeds=[42],
                min_class_size=10,
            )


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCli:
    def test_arg_parser_required_args(self):
        parser = build_arg_parser()
        # Missing required args
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_arg_parser_parses_seeds(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--hte-csv", "/tmp/x.csv",
            "--seeds", "1,2,3",
        ])
        assert args.hte_csv == "/tmp/x.csv"
        assert args.seeds == [1, 2, 3]

    def test_arg_parser_default_values(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--hte-csv", "/tmp/x.csv",
            "--seeds", "1",
        ])
        assert args.n_per_class == 50
        assert args.min_class_size == 20
        assert args.n_candidates == 10
        assert args.n_bootstrap == 1000
        assert args.radius == 2
        assert args.n_bits == 2048
        assert args.pc_cng_negatives is None
        assert args.output_dir is None

    def test_arg_parser_invalid_seed(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--hte-csv", "/tmp/x.csv",
                "--seeds", "1,not_a_number",
            ])

    def test_main_end_to_end(self, hte_csv, tmp_path: Path):
        """Smoke test: invoke main() with a synthetic CSV."""
        out_dir = tmp_path / "cli_out"
        rc = main([
            "--hte-csv", hte_csv,
            "--output-dir", str(out_dir),
            "--n-per-class", "5",
            "--min-class-size", "5",
            "--n-candidates", "3",
            "--seeds", "42",
            "--n-bootstrap", "10",
        ])
        assert rc == 0
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "summary.md").exists()

    def test_main_missing_hte_csv(self, tmp_path: Path):
        rc = main([
            "--hte-csv", str(tmp_path / "nonexistent.csv"),
            "--seeds", "42",
        ])
        assert rc != 0

    def test_main_with_pc_cng_csv(self, hte_csv, pc_cng_csv, tmp_path: Path):
        out_dir = tmp_path / "cli_pc_out"
        rc = main([
            "--hte-csv", hte_csv,
            "--pc-cng-negatives", pc_cng_csv,
            "--output-dir", str(out_dir),
            "--n-per-class", "5",
            "--min-class-size", "5",
            "--n-candidates", "3",
            "--seeds", "42",
            "--n-bootstrap", "10",
        ])
        assert rc == 0
