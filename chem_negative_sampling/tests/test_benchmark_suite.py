"""Unit tests for `benchmark_suite` (P3-08).

These tests run fully offline - no GPU, no remote server, no network.  Mock
results directories are created with the ``tmp_path`` fixture so every code
path can be exercised deterministically.

Run::

    cd /tmp/p3_files/p3_08 && PYTHONPATH=. python -m pytest test_benchmark_suite.py -v
"""

from __future__ import annotations

import json
import os
import sys

import pytest

# Make sure the chem_negative_sampling package is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_CNS_ROOT = os.path.dirname(_HERE)
if _CNS_ROOT not in sys.path:
    sys.path.insert(0, _CNS_ROOT)

from evaluation import benchmark_suite as bs  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _make_results_tree(tmp_path, with_p3=None, with_p2=None):
    """Create a mock results/ tree with the requested P3 / P2 subdirs."""
    if with_p3 is None:
        with_p3 = ["P3-01", "P3-02", "P3-03", "P3-04", "P3-05", "P3-06", "P3-07"]
    if with_p2 is None:
        with_p2 = ["P2-02"]
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    for p3_id in with_p3:
        sub = bs.P3_DIR_MAP[p3_id]
        (results / sub).mkdir(parents=True, exist_ok=True)
    for p2_id in with_p2:
        sub = bs.P2_DIR_MAP[p2_id]
        (results / sub).mkdir(parents=True, exist_ok=True)
    return results


# ---------------------------------------------------------------------------
# _safe_float / _read_json / _find_summary
# ---------------------------------------------------------------------------


def test_safe_float_handles_bad_inputs():
    assert bs._safe_float("0.5") == 0.5
    assert bs._safe_float(0.5) == 0.5
    assert bs._safe_float(None) is None
    assert bs._safe_float(None, default=0.0) == 0.0
    assert bs._safe_float("not a number", default=-1.0) == -1.0
    assert bs._safe_float([1, 2, 3]) is None


def test_safe_float_preserves_zero():
    """Regression: 0.0 must round-trip through _safe_float (not become default)."""
    assert bs._safe_float(0.0) == 0.0
    assert bs._safe_float(0) == 0.0


def test_first_present_respects_falsy_zero():
    """Regression: ``or``-chains silently swallow 0.0; _first_present must not."""
    d = {"test_top1_accuracy": 0.0, "top1_accuracy": 0.5}
    assert bs._first_present(d, "test_top1_accuracy", "top1_accuracy") == 0.0
    # missing keys -> None
    assert bs._first_present(d, "nonexistent") is None
    # None values skipped
    d2 = {"a": None, "b": 1}
    assert bs._first_present(d2, "a", "b") == 1
    # non-dict input
    assert bs._first_present(["x"], "a") is None
    assert bs._first_present(None, "a") is None


def test_pair_variant_mrr_flat_and_nested():
    flat = {"direct_mrr": 0.0, "head_finetune_mrr": 0.1, "full_finetune_mrr": 0.2}
    assert bs._pair_variant_mrr(flat, "direct") == 0.0  # falsy but present
    assert bs._pair_variant_mrr(flat, "head_finetune") == 0.1
    assert bs._pair_variant_mrr(flat, "full_finetune") == 0.2
    nested = {
        "variants": {
            "direct": {"mrr": 0.0},
            "head_finetune": {"mrr": 0.15},
            "full_finetune": {"mrr": None},  # should be skipped -> None
        }
    }
    assert bs._pair_variant_mrr(nested, "direct") == 0.0
    assert bs._pair_variant_mrr(nested, "head_finetune") == 0.15
    assert bs._pair_variant_mrr(nested, "full_finetune") is None
    # flat takes precedence over nested when both present
    both = {"direct_mrr": 0.3, "variants": {"direct": {"mrr": 0.9}}}
    assert bs._pair_variant_mrr(both, "direct") == 0.3
    # missing entirely
    assert bs._pair_variant_mrr({}, "direct") is None


def test_read_json_missing_and_corrupt(tmp_path):
    assert bs._read_json(str(tmp_path / "nope.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json ", encoding="utf-8")
    assert bs._read_json(str(bad)) is None
    good = tmp_path / "good.json"
    _write_json(str(good), {"k": 1})
    assert bs._read_json(str(good)) == {"k": 1}


def test_find_summary_prefers_summary_json(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    assert bs._find_summary(str(d)) is None
    _write_json(str(d / "summary.json"), {"a": 1})
    assert bs._find_summary(str(d)) == {"a": 1}
    # aggregate_summary.json as fallback
    d2 = tmp_path / "d2"
    d2.mkdir()
    _write_json(str(d2 / "aggregate_summary.json"), {"b": 2})
    assert bs._find_summary(str(d2)) == {"b": 2}


# ---------------------------------------------------------------------------
# load_p3_results
# ---------------------------------------------------------------------------


def test_load_p3_results_full_tree(tmp_path):
    results = _make_results_tree(tmp_path)
    _write_json(str(results / bs.P3_DIR_MAP["P3-01"] / "summary.json"), {"mean_mrr": 0.31})
    _write_json(str(results / bs.P3_DIR_MAP["P3-04"] / "summary.json"), {"test_top1_accuracy": 0.0, "status": "NO-GO"})
    out = bs.load_p3_results(str(results))
    assert set(out.keys()) == set(bs.P3_DIR_MAP.keys())
    assert out["P3-01"] == {"mean_mrr": 0.31}
    assert out["P3-04"]["status"] == "NO-GO"
    # Missing summary -> empty dict
    assert out["P3-02"] == {}


def test_load_p3_results_missing_directory(tmp_path, capsys):
    # results dir does not exist at all
    out = bs.load_p3_results(str(tmp_path / "nope"))
    assert out == {}
    # partial tree - some P3 subdirs missing
    results = _make_results_tree(tmp_path, with_p3=["P3-01"])
    out = bs.load_p3_results(str(results))
    assert out["P3-01"] == {}
    assert out["P3-02"] == {}  # missing dir -> empty dict
    err = capsys.readouterr().err
    assert "P3-02" in err


# ---------------------------------------------------------------------------
# compute_negative_quality
# ---------------------------------------------------------------------------


def test_compute_negative_quality_missing_file(tmp_path):
    res = bs.compute_negative_quality(str(tmp_path / "missing.csv"))
    assert res["status"] == "skipped"
    assert res["n_negatives"] == 0
    assert "not found" in res["notes"]


def test_compute_negative_quality_synthetic_csv(tmp_path):
    csv_path = tmp_path / "negs.csv"
    # Mix of valid (ethanol, benzene) and invalid SMILES + a duplicate.
    rows = [
        "smiles,id",
        "CCO,1",
        "c1ccccc1,2",
        "CCO,3",  # duplicate of 1 -> uniqueness < 1
        "not_a_smiles,4",
        "C1=1,5",  # likely invalid
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    res = bs.compute_negative_quality(str(csv_path))
    assert res["status"] in ("ok",)  # valid path even if rdkit absent
    assert res["n_negatives"] == 5
    # Uniqueness: 4 unique / 5 total
    assert res["uniqueness"] == pytest.approx(4 / 5)
    if bs._HAS_RDKIT:
        # validity must be a float in [0,1]
        assert res["validity"] is not None
        assert 0.0 <= res["validity"] <= 1.0
        # diversity may be None if too few valid fingerprints, but if present
        # it must be in [0,1]
        if res["diversity"] is not None:
            assert 0.0 <= res["diversity"] <= 1.0
    else:
        assert res["validity"] is None
        assert "RDKit unavailable" in res["notes"]


def test_compute_negative_quality_no_smiles_column(tmp_path):
    csv_path = tmp_path / "negs.csv"
    csv_path.write_text("foo,bar\n1,2\n3,4\n", encoding="utf-8")
    res = bs.compute_negative_quality(str(csv_path))
    assert res["n_negatives"] == 0
    assert res["status"] == "error"


def test_compute_negative_quality_empty_rows(tmp_path):
    csv_path = tmp_path / "negs.csv"
    csv_path.write_text("smiles\n\n\n", encoding="utf-8")
    res = bs.compute_negative_quality(str(csv_path))
    assert res["n_negatives"] == 0
    assert res["status"] == "error"


def test_compute_negative_quality_accepts_alternate_column(tmp_path):
    csv_path = tmp_path / "negs.csv"
    csv_path.write_text("negative_smiles\nCCO\nc1ccccc1\n", encoding="utf-8")
    res = bs.compute_negative_quality(str(csv_path))
    assert res["n_negatives"] == 2
    assert res["uniqueness"] == 1.0


# ---------------------------------------------------------------------------
# compute_downstream_metrics
# ---------------------------------------------------------------------------


def test_compute_downstream_metrics_full():
    p3 = {
        "P3-01": {"mean_mrr": 0.31},
        "P3-04": {"test_top1_accuracy": 0.0, "status": "NO-GO"},
        "P3-06": {"yield_rmse": 0.12},
    }
    res = bs.compute_downstream_metrics(p3)
    assert res["status"] == "ok"
    assert res["retrosynthesis"]["mrr"] == 0.31
    assert res["retrosynthesis"]["delta"] == pytest.approx(0.31 - bs.GNN_BASELINE_MRR)
    assert res["retrosynthesis"]["gnn_baseline_mrr"] == bs.GNN_BASELINE_MRR
    assert res["condition"]["top1_accuracy"] == 0.0
    assert res["condition"]["status_note"] == "NO-GO"
    assert res["yield"]["rmse"] == 0.12
    assert res["yield"]["available"] is True


def test_compute_downstream_metrics_missing_yield():
    p3 = {"P3-01": {"mrr": 0.30}, "P3-04": {"accuracy": 0.05}}
    res = bs.compute_downstream_metrics(p3)
    assert res["retrosynthesis"]["mrr"] == 0.30
    assert res["condition"]["top1_accuracy"] == 0.05
    assert res["yield"]["available"] is False
    assert any("P3-06" in n for n in res["notes"])


def test_compute_downstream_metrics_all_missing_skips():
    res = bs.compute_downstream_metrics({})
    assert res["status"] == "skipped"
    assert res["retrosynthesis"]["mrr"] is None
    assert res["condition"]["top1_accuracy"] is None


def test_compute_downstream_metrics_handles_alternate_keys():
    p3 = {
        "P3-01": {"aggregate_mrr": 0.42},
        "P3-04": {"top1_accuracy": 0.1},
        "P3-06": {"test_rmse": 0.3},
    }
    res = bs.compute_downstream_metrics(p3)
    assert res["retrosynthesis"]["mrr"] == 0.42
    assert res["condition"]["top1_accuracy"] == 0.1
    assert res["yield"]["rmse"] == 0.3


# ---------------------------------------------------------------------------
# compute_cross_dataset_metrics
# ---------------------------------------------------------------------------


def test_cross_dataset_missing_dir(tmp_path):
    res = bs.compute_cross_dataset_metrics(str(tmp_path / "nope"))
    assert res["status"] == "skipped"
    assert "not found" in res["notes"]


def test_cross_dataset_missing_summary(tmp_path):
    d = tmp_path / "p303"
    d.mkdir()
    res = bs.compute_cross_dataset_metrics(str(d))
    assert res["status"] == "skipped"
    assert "summary.json not found" in res["notes"]


def test_cross_dataset_with_pairs_flat_format(tmp_path):
    d = tmp_path / "p303"
    d.mkdir()
    pairs = [
        {"pair": "A->B", "direct_mrr": 0.20, "head_finetune_mrr": 0.25, "full_finetune_mrr": 0.28},
        {"pair": "C->D", "direct_mrr": 0.30, "head_finetune_mrr": 0.33, "full_finetune_mrr": 0.36},
    ]
    _write_json(str(d / "summary.json"), {"pairs": pairs})
    res = bs.compute_cross_dataset_metrics(str(d))
    assert res["status"] == "ok"
    assert res["n_pairs"] == 2
    # head deltas: (0.25-0.20) + (0.33-0.30) = 0.05+0.03 = 0.08 / 2 = 0.04
    assert res["mean_mrr_delta_vs_direct"]["head_finetune"] == pytest.approx(0.04)
    # full deltas: (0.28-0.20) + (0.36-0.30) = 0.08+0.06 = 0.14 / 2 = 0.07
    assert res["mean_mrr_delta_vs_direct"]["full_finetune"] == pytest.approx(0.07)


def test_cross_dataset_with_nested_variants_format(tmp_path):
    d = tmp_path / "p303"
    d.mkdir()
    pairs = [
        {
            "pair": "A->B",
            "variants": {
                "direct": {"mrr": 0.20},
                "head_finetune": {"mrr": 0.24},
                "full_finetune": {"mrr": 0.27},
            },
        },
    ]
    _write_json(str(d / "summary.json"), {"migration_pairs": pairs})
    res = bs.compute_cross_dataset_metrics(str(d))
    assert res["status"] == "ok"
    assert res["n_pairs"] == 1
    assert res["mean_mrr_delta_vs_direct"]["head_finetune"] == pytest.approx(0.04)
    assert res["mean_mrr_delta_vs_direct"]["full_finetune"] == pytest.approx(0.07)


def test_cross_dataset_empty_pairs_list(tmp_path):
    d = tmp_path / "p303"
    d.mkdir()
    _write_json(str(d / "summary.json"), {"pairs": []})
    res = bs.compute_cross_dataset_metrics(str(d))
    assert res["status"] == "ok"
    assert res["n_pairs"] == 0
    assert "no pairs" in res["notes"]


def test_cross_dataset_top_level_list(tmp_path):
    d = tmp_path / "p303"
    d.mkdir()
    pairs = [{"direct_mrr": 0.1, "head_finetune_mrr": 0.2, "full_finetune_mrr": 0.3}]
    _write_json(str(d / "summary.json"), pairs)
    res = bs.compute_cross_dataset_metrics(str(d))
    assert res["status"] == "ok"
    assert res["n_pairs"] == 1
    assert res["mean_mrr_delta_vs_direct"]["head_finetune"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# compute_efficiency_metrics
# ---------------------------------------------------------------------------


def test_efficiency_numpy_fallback_no_torch():
    """When torch is unavailable (or ckpt missing), numpy probe is used."""
    res = bs.compute_efficiency_metrics(backbone_ckpt=None, vocab_path=None, n_samples=50)
    assert res["status"] == "ok"
    assert res["mode"] == "numpy_random_probe"
    assert res["n_samples"] == 50
    assert res["latency_ms_per_reaction"] is not None
    assert res["latency_ms_per_reaction"] >= 0.0
    assert res["throughput_reactions_per_sec"] is not None
    assert res["memory_mb"] is not None
    assert "numpy" in res["notes"].lower() or "no backbone" in res["notes"].lower()


def test_efficiency_missing_checkpoint_falls_back(tmp_path):
    res = bs.compute_efficiency_metrics(
        backbone_ckpt=str(tmp_path / "nope.pt"), vocab_path=None, n_samples=20
    )
    assert res["status"] == "ok"
    assert res["mode"] == "numpy_random_probe"
    assert "numpy fallback" in res["notes"] or "not found" in res["notes"]


@pytest.mark.skipif(not bs._HAS_TORCH, reason="torch not installed")
def test_efficiency_torch_probe_with_fake_checkpoint(tmp_path):
    """When torch is available, build a fake state_dict checkpoint and verify
    the torch probe path executes end-to-end."""
    import torch

    ckpt = tmp_path / "fake.pt"
    state = {
        "embedding.weight": torch.randn(256, 1024),
        "layer.0.weight": torch.randn(1024, 1024),
    }
    torch.save(state, str(ckpt))
    res = bs.compute_efficiency_metrics(
        backbone_ckpt=str(ckpt), vocab_path=None, n_samples=10
    )
    assert res["status"] == "ok"
    assert res["mode"] == "torch_backbone_probe"
    assert res["latency_ms_per_reaction"] is not None
    assert res["memory_mb"] is not None
    assert res["checkpoint"] == str(ckpt)


# ---------------------------------------------------------------------------
# compute_plausibility_metrics
# ---------------------------------------------------------------------------


def test_plausibility_both_missing(tmp_path):
    res = bs.compute_plausibility_metrics(
        dft_dir=str(tmp_path / "no_dft"), llm_judge_dir=str(tmp_path / "no_llm")
    )
    assert res["status"] == "skipped"
    assert res["dft"]["validation_rate"] is None
    assert res["llm_judge"]["agreement_kappa"] is None
    assert "not found" in res["dft"]["notes"]
    assert "not found" in res["llm_judge"]["notes"]


def test_plausibility_with_summaries(tmp_path):
    dft = tmp_path / "dft"
    dft.mkdir()
    _write_json(str(dft / "summary.json"), {"validation_rate": 0.92})
    llm = tmp_path / "llm"
    llm.mkdir()
    _write_json(str(llm / "summary.json"), {"kappa": 0.646})
    res = bs.compute_plausibility_metrics(str(dft), str(llm))
    assert res["status"] == "ok"
    assert res["dft"]["validation_rate"] == pytest.approx(0.92)
    assert res["llm_judge"]["agreement_kappa"] == pytest.approx(0.646)


def test_plausibility_partial_summary_keys(tmp_path):
    dft = tmp_path / "dft"
    dft.mkdir()
    _write_json(str(dft / "summary.json"), {"dft_validation_rate": 0.5})
    llm = tmp_path / "llm"
    llm.mkdir()
    _write_json(str(llm / "summary.json"), {"cohen_kappa": 0.4})
    res = bs.compute_plausibility_metrics(str(dft), str(llm))
    assert res["dft"]["validation_rate"] == pytest.approx(0.5)
    assert res["llm_judge"]["agreement_kappa"] == pytest.approx(0.4)


def test_plausibility_summary_missing_keys(tmp_path):
    dft = tmp_path / "dft"
    dft.mkdir()
    _write_json(str(dft / "summary.json"), {"unrelated_key": 1})
    llm = tmp_path / "llm"
    llm.mkdir()
    _write_json(str(llm / "summary.json"), {})
    res = bs.compute_plausibility_metrics(str(dft), str(llm))
    assert res["status"] == "skipped"
    assert "missing" in res["dft"]["notes"]
    assert "missing" in res["llm_judge"]["notes"]


def test_plausibility_respects_falsy_zero_rates(tmp_path):
    """Regression: validation_rate=0.0 must not be swallowed by or-chains."""
    dft = tmp_path / "dft"
    dft.mkdir()
    _write_json(str(dft / "summary.json"), {"validation_rate": 0.0})
    llm = tmp_path / "llm"
    llm.mkdir()
    _write_json(str(llm / "summary.json"), {"kappa": 0.0})
    res = bs.compute_plausibility_metrics(str(dft), str(llm))
    assert res["status"] == "ok"
    assert res["dft"]["validation_rate"] == 0.0
    assert res["llm_judge"]["agreement_kappa"] == 0.0


def test_downstream_condition_zero_is_recorded():
    """Regression (P3-04 NO-GO): Top-1 accuracy of 0.0 must be recorded, not None."""
    p3 = {"P3-04": {"test_top1_accuracy": 0.0, "status": "NO-GO"}}
    res = bs.compute_downstream_metrics(p3)
    assert res["condition"]["top1_accuracy"] == 0.0
    assert res["condition"]["status_note"] == "NO-GO"
    # With only cond_acc=0.0 set, status should be "ok" (0.0 is a real value)
    assert res["status"] == "ok"


# ---------------------------------------------------------------------------
# compute_ablation_metrics
# ---------------------------------------------------------------------------


def test_ablation_deferred_when_no_results():
    res = bs.compute_ablation_metrics(None)
    assert res["status"] == "deferred_to_future_work"
    assert "deferred" in res["notes"].lower()
    assert "physicochemical_constraints" in res["components"]


def test_ablation_deferred_when_dir_empty(tmp_path):
    res = bs.compute_ablation_metrics(str(tmp_path))
    assert res["status"] == "deferred_to_future_work"


def test_ablation_loaded_from_existing_summary(tmp_path):
    ablation_dir = tmp_path / "ablation_p2_20260720"
    ablation_dir.mkdir()
    _write_json(
        str(ablation_dir / "summary.json"),
        {
            "ablations": [
                {"name": "no_physchem", "metric": "mrr", "value": 0.20},
                {"name": "no_counterfactual", "metric": "mrr", "value": 0.22},
                {"name": "no_neg_sampling", "metric": "mrr", "value": 0.18},
            ]
        },
    )
    res = bs.compute_ablation_metrics(str(tmp_path))
    assert res["status"] == "ok"
    assert len(res["ablations"]) == 3
    assert res["source"].endswith("summary.json")


def test_ablation_summary_present_but_empty(tmp_path):
    p = tmp_path / "ablation_summary.json"
    _write_json(str(p), {"ablations": []})
    res = bs.compute_ablation_metrics(str(tmp_path))
    assert res["status"] == "deferred_to_future_work"
    assert res["source"] is not None


# ---------------------------------------------------------------------------
# run_benchmark (end-to-end with mock data)
# ---------------------------------------------------------------------------


def test_run_benchmark_writes_outputs(tmp_path):
    results = _make_results_tree(tmp_path)
    # P3-01 summary
    _write_json(str(results / bs.P3_DIR_MAP["P3-01"] / "summary.json"), {"mean_mrr": 0.31})
    # P3-03 summary with cross-dataset pairs
    _write_json(
        str(results / bs.P3_DIR_MAP["P3-03"] / "summary.json"),
        {"pairs": [{"direct_mrr": 0.2, "head_finetune_mrr": 0.25, "full_finetune_mrr": 0.28}]},
    )
    # P3-04 NO-GO
    _write_json(
        str(results / bs.P3_DIR_MAP["P3-04"] / "summary.json"),
        {"test_top1_accuracy": 0.0, "status": "NO-GO"},
    )
    # P3-07 LLM judge
    _write_json(str(results / bs.P3_DIR_MAP["P3-07"] / "summary.json"), {"kappa": 0.646})
    # P2-02 DFT
    _write_json(str(results / bs.P2_DIR_MAP["P2-02"] / "summary.json"), {"validation_rate": 0.92})
    # PC-CNG negatives CSV
    neg_dir = results / "uspto_openmolecules_full_generation"
    neg_dir.mkdir(parents=True, exist_ok=True)
    (neg_dir / "pc_cng_synthetic_negatives_reviewed.csv").write_text(
        "smiles,id\nCCO,1\nc1ccccc1,2\nCCO,3\n", encoding="utf-8"
    )

    out_dir = tmp_path / "benchmark_out"
    metrics = bs.run_benchmark(
        results_dir=str(results),
        output_dir=str(out_dir),
        backbone_ckpt=None,
        vocab_path=None,
    )
    # metrics.json + report.md both written
    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "report.md").is_file()
    # All 6 dimensions present
    for dim in bs.DIMENSION_NAMES:
        assert dim in metrics, f"missing {dim}"
    # Metadata sanity
    assert metrics["p3_08_metadata"]["task_id"] == "P3-08"
    # Spot-check a few values propagated correctly
    assert metrics["dimension_2_downstream"]["retrosynthesis"]["mrr"] == 0.31
    assert metrics["dimension_5_plausibility"]["llm_judge"]["agreement_kappa"] == 0.646
    # Efficiency falls back to numpy (no backbone provided)
    assert metrics["dimension_4_efficiency"]["mode"] == "numpy_random_probe"
    # Ablation is deferred by default (no ablation summary in the tree)
    assert metrics["dimension_6_ablation"]["status"] == "deferred_to_future_work"
    # Report contains expected section headers
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "# P3-08 Comprehensive Benchmark Report" in report
    assert "Dimension 1" in report
    assert "Dimension 6" in report


def test_run_benchmark_subset_of_dimensions(tmp_path):
    results = _make_results_tree(tmp_path)
    out_dir = tmp_path / "out"
    metrics = bs.run_benchmark(
        results_dir=str(results),
        output_dir=str(out_dir),
        dimensions=["1", "4"],
    )
    assert "dimension_1_negative_quality" in metrics
    assert "dimension_4_efficiency" in metrics
    # Dimensions not requested should be absent
    assert "dimension_2_downstream" not in metrics
    assert "dimension_3_cross_dataset" not in metrics


def test_run_benchmark_empty_results_dir(tmp_path):
    """All dimensions should report skipped/error gracefully, not raise."""
    empty = tmp_path / "empty_results"
    empty.mkdir()
    out_dir = tmp_path / "out"
    metrics = bs.run_benchmark(
        results_dir=str(empty),
        output_dir=str(out_dir),
        backbone_ckpt=None,
        vocab_path=None,
    )
    assert metrics["dimension_1_negative_quality"]["status"] == "skipped"
    assert metrics["dimension_2_downstream"]["status"] == "skipped"
    assert metrics["dimension_3_cross_dataset"]["status"] == "skipped"
    assert metrics["dimension_4_efficiency"]["status"] == "ok"  # numpy fallback always works
    assert metrics["dimension_5_plausibility"]["status"] == "skipped"
    # Ablation is "deferred" not "skipped"
    assert metrics["dimension_6_ablation"]["status"] == "deferred_to_future_work"


# ---------------------------------------------------------------------------
# _normalize_dimensions / _fmt / _render_report
# ---------------------------------------------------------------------------


def test_normalize_dimensions_default_all():
    assert bs._normalize_dimensions(None) == set(bs.DIMENSION_NAMES)
    assert bs._normalize_dimensions([]) == set(bs.DIMENSION_NAMES)
    assert bs._normalize_dimensions(["all"]) == set(bs.DIMENSION_NAMES)


def test_normalize_dimensions_short_and_long_forms():
    s = bs._normalize_dimensions(["1", "dim2", "cross_dataset", "efficiency", "5", "6"])
    assert s == {
        "dimension_1_negative_quality",
        "dimension_2_downstream",
        "dimension_3_cross_dataset",
        "dimension_4_efficiency",
        "dimension_5_plausibility",
        "dimension_6_ablation",
    }


def test_normalize_dimensions_unknown_selector_falls_back_to_all():
    s = bs._normalize_dimensions(["totally_bogus"])
    # Unknown selector yields empty set -> fall back to all
    assert s == set(bs.DIMENSION_NAMES)


def test_fmt():
    assert bs._fmt(None) == "—"
    assert bs._fmt(0.5) == "0.5000"
    assert bs._fmt("text") == "text"
    assert bs._fmt(3) == "3"


def test_render_report_contains_all_sections():
    metrics = {
        "p3_08_metadata": {"task_id": "P3-08", "version": "v3", "date": "20260720"},
        "dimension_1_negative_quality": {"status": "ok", "source": "x.csv", "n_negatives": 10, "validity": 0.9, "uniqueness": 0.8, "diversity": 0.5, "notes": ""},
        "dimension_2_downstream": {"status": "ok", "retrosynthesis": {"mrr": 0.31, "gnn_baseline_mrr": 0.243, "delta": 0.067}, "condition": {"top1_accuracy": 0.0, "status_note": "NO-GO"}, "yield": {"rmse": None, "available": False}, "notes": ["P3-06 yield RMSE not available"]},
        "dimension_3_cross_dataset": {"status": "ok", "source": "p303", "n_pairs": 7, "mean_mrr_delta_vs_direct": {"head_finetune": 0.04, "full_finetune": 0.07}, "per_pair": [], "notes": ""},
        "dimension_4_efficiency": {"status": "ok", "n_samples": 100, "latency_ms_per_reaction": 0.5, "throughput_reactions_per_sec": 2000.0, "memory_mb": 10.0, "mode": "numpy_random_probe", "notes": ""},
        "dimension_5_plausibility": {"status": "ok", "dft": {"source": "dft", "validation_rate": 0.92, "notes": ""}, "llm_judge": {"source": "llm", "agreement_kappa": 0.646, "notes": ""}, "notes": []},
        "dimension_6_ablation": {"status": "deferred_to_future_work", "components": ["a", "b"], "ablations": [], "notes": "deferred", "source": None},
    }
    md = bs._render_report(metrics)
    for header in [
        "# P3-08 Comprehensive Benchmark Report",
        "## Dimension 1",
        "## Dimension 2",
        "## Dimension 3",
        "## Dimension 4",
        "## Dimension 5",
        "## Dimension 6",
    ]:
        assert header in md
    # Ablation entries table is rendered when ablations list is non-empty
    metrics["dimension_6_ablation"]["ablations"] = [{"name": "no_physchem", "metric": "mrr", "value": 0.2}]
    md2 = bs._render_report(metrics)
    assert "no_physchem" in md2


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------


def test_main_writes_outputs(tmp_path, monkeypatch):
    results = _make_results_tree(tmp_path, with_p3=[], with_p2=[])
    out_dir = tmp_path / "cli_out"
    rc = bs.main(
        [
            "--results-dir", str(results),
            "--output-dir", str(out_dir),
            "--dimensions", "1,4",
        ]
    )
    assert rc == 0
    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "report.md").is_file()


def test_main_dimensions_all_default(tmp_path):
    results = _make_results_tree(tmp_path, with_p3=[], with_p2=[])
    out_dir = tmp_path / "cli_out2"
    rc = bs.main(
        ["--results-dir", str(results), "--output-dir", str(out_dir)]
    )
    assert rc == 0
    with open(out_dir / "metrics.json", "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    # default dimensions=all -> all 6 present
    for dim in bs.DIMENSION_NAMES:
        assert dim in metrics


def test_build_parser_required_args():
    parser = bs._build_parser()
    # Missing required args should make argparse error (SystemExit).
    with pytest.raises(SystemExit):
        parser.parse_args([])


# ---------------------------------------------------------------------------
# Module-level docstring / constants sanity (lightweight smoke tests)
# ---------------------------------------------------------------------------


def test_module_docstring_mentions_p3_08_and_dimensions():
    doc = bs.__doc__ or ""
    assert "P3-08" in doc
    assert "6 dimensions" in doc.lower() or "6 Dimensions" in doc
    for word in ["Negative generation", "Downstream", "Cross-dataset", "efficiency", "plausibility", "Ablation"]:
        assert word.lower() in doc.lower()


def test_p3_dir_map_has_seven_entries():
    expected = {"P3-01", "P3-02", "P3-03", "P3-04", "P3-05", "P3-06", "P3-07"}
    assert set(bs.P3_DIR_MAP.keys()) == expected
    # All paths mention 20260720 per the spec
    for v in bs.P3_DIR_MAP.values():
        assert "20260720" in v


def test_dimension_names_in_order():
    assert bs.DIMENSION_NAMES == [
        "dimension_1_negative_quality",
        "dimension_2_downstream",
        "dimension_3_cross_dataset",
        "dimension_4_efficiency",
        "dimension_5_plausibility",
        "dimension_6_ablation",
    ]
