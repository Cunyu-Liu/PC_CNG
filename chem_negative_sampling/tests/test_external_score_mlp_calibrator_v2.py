"""Unit tests for P2-04 MLP calibrator v2 (chemformer-aware)."""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

import numpy as np
import pytest

from pc_cng.train_external_score_mlp_calibrator_v2 import (
    DEFAULT_SEEDS,
    paired_significance_test,
    parse_args,
    run_train_v2,
    train_with_early_stopping,
    warm_start_model,
)
from pc_cng.train_external_score_mlp_calibrator import (
    FEATURE_NAMES,
    build_features,
    init_model,
    read_rows,
    serialize_model,
    standardize_train,
)


def _write_synthetic_candidates(
    path: str,
    n_groups: int = 5,
    n_neg_per_group: int = 1,
) -> None:
    """Write a small synthetic candidates CSV for testing."""
    rows: List[Dict[str, str]] = []
    for g in range(n_groups):
        gid = f"g{g}"
        rows.append(
            {
                "group_id": gid,
                "source_id": f"s{g}_pos",
                "label": "1",
                "split": "test",
                "chemformer_likelihood": f"{0.6 + 0.05 * g:.4f}",
                "pc_cng": f"{0.7 + 0.03 * g:.4f}",
            }
        )
        for n in range(n_neg_per_group):
            rows.append(
                {
                    "group_id": gid,
                    "source_id": f"s{g}_neg{n}",
                    "label": "0",
                    "split": "test",
                    "chemformer_likelihood": f"{0.2 + 0.01 * n:.4f}",
                    "pc_cng": f"{0.15 + 0.01 * n:.4f}",
                }
            )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_synthetic_v1_model(
    path: str,
    hidden_dim: int = 16,
    seed: int = 42,
) -> None:
    """Write a synthetic v1 model JSON for warm-start testing."""
    model = init_model(len(FEATURE_NAMES), hidden_dim, seed)
    payload = {
        "model_name": "synthetic_v1",
        "score_name": "pc_cng_mlp_calibrator_v1",
        "recipe": "fixed_feature_group_score_pairwise_mlp",
        "primary_score": "chemformer_likelihood",
        "pc_score": "pc_cng",
        "feature_names": FEATURE_NAMES,
        "feature_means": [0.0] * len(FEATURE_NAMES),
        "feature_stds": [1.0] * len(FEATURE_NAMES),
        "hidden_dim": hidden_dim,
        "epochs": 100,
        "learning_rate": 0.01,
        "l2": 0.0001,
        "seed": seed,
        "train_split": "train",
        "training_pairs": 100,
        "training_rows": 100,
        "parameters": serialize_model(model),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _default_args(candidates: str, warm_start: str, output_dir: str, **overrides):
    argv = [
        "--candidates", candidates,
        "--warm-start", warm_start,
        "--output-dir", output_dir,
        "--seeds", "1,2,3,4,5,6,7,8,9,10",
        "--hidden-dim", "8",
        "--epochs", "20",
        "--learning-rate", "0.01",
        "--l2", "0.0001",
        "--early-stopping-patience", "5",
        "--train-split-ratio", "0.8",
    ]
    for key, value in overrides.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return parse_args(argv)


# ---------------------------------------------------------------------------
# Test 1: Module imports
# ---------------------------------------------------------------------------


def test_module_imports():
    """Verify required v2 functions are importable."""
    assert callable(parse_args)
    assert callable(run_train_v2)
    assert callable(warm_start_model)
    assert callable(train_with_early_stopping)
    assert callable(paired_significance_test)
    # v1 primitives re-exported/imported by v2
    assert len(FEATURE_NAMES) == 11
    assert DEFAULT_SEEDS.count(",") == 9  # 10 seeds


# ---------------------------------------------------------------------------
# Test 2: Warm-start loads v1 model correctly
# ---------------------------------------------------------------------------


def test_warm_start_loads_v1_model(tmp_path):
    """Warm-start loads v1 weights and adapts to v2 hidden_dim."""
    v1_path = tmp_path / "v1_model.json"
    _write_synthetic_v1_model(str(v1_path), hidden_dim=16, seed=42)

    # Same hidden_dim: reuse v1 weights directly.
    model_same, payload = warm_start_model(str(v1_path), hidden_dim=16, seed=99)
    assert model_same["w1"].shape == (11, 16)
    assert model_same["b1"].shape == (16,)
    assert model_same["w2"].shape == (16,)
    assert model_same["b2"].shape == (1,)
    assert payload["hidden_dim"] == 16
    assert payload["primary_score"] == "chemformer_likelihood"

    # Larger hidden_dim: pad with random init, first 16 cols match v1.
    model_big, _ = warm_start_model(str(v1_path), hidden_dim=32, seed=99)
    assert model_big["w1"].shape == (11, 32)
    assert model_big["b1"].shape == (32,)
    assert model_big["w2"].shape == (32,)
    np.testing.assert_allclose(model_big["w1"][:, :16], model_same["w1"])
    np.testing.assert_allclose(model_big["b1"][:16], model_same["b1"])
    np.testing.assert_allclose(model_big["w2"][:16], model_same["w2"])
    np.testing.assert_allclose(model_big["b2"], model_same["b2"])

    # Smaller hidden_dim: truncate.
    model_small, _ = warm_start_model(str(v1_path), hidden_dim=8, seed=99)
    assert model_small["w1"].shape == (11, 8)
    assert model_small["b1"].shape == (8,)
    assert model_small["w2"].shape == (8,)
    np.testing.assert_allclose(model_small["w1"], model_same["w1"][:, :8])
    np.testing.assert_allclose(model_small["b1"], model_same["b1"][:8])


# ---------------------------------------------------------------------------
# Test 3: Early stopping triggers when val Top-1 doesn't improve
# ---------------------------------------------------------------------------


def test_early_stopping_triggers(tmp_path):
    """Early stopping halts training after `patience` epochs without improvement."""
    csv_path = tmp_path / "cands.csv"
    _write_synthetic_candidates(str(csv_path), n_groups=10, n_neg_per_group=2)
    v1_path = tmp_path / "v1_model.json"
    _write_synthetic_v1_model(str(v1_path), hidden_dim=8, seed=42)

    from pc_cng.train_external_score_mlp_calibrator_v2 import (
        _build_pairs_for_split,
        _split_group_ids,
    )

    rows = read_rows(str(csv_path))
    raw_features, row_indices = build_features(rows, "chemformer_likelihood", "pc_cng")
    group_ids = sorted({r["group_id"] for r in rows})
    train_ids, val_ids = _split_group_ids(group_ids, 0.8, 42)
    train_mask = np.asarray(
        [rows[int(i)]["group_id"] in train_ids for i in row_indices], dtype=bool
    )
    features, _, _ = standardize_train(raw_features, train_mask)
    pos_idx, neg_idx = _build_pairs_for_split(rows, row_indices, train_ids)
    assert len(pos_idx) > 0

    # Constant val_eval_fn: val Top-1 never improves beyond the first epoch.
    call_count = [0]

    def const_val_eval(model):
        call_count[0] += 1
        return {"top1": 0.5, "top3": 0.5, "top5": 0.5, "ndcg10": 0.5}

    warm_model, _ = warm_start_model(str(v1_path), 8, 42)
    best_model, history = train_with_early_stopping(
        features=features,
        pos_idx=pos_idx,
        neg_idx=neg_idx,
        initial_model=warm_model,
        epochs=100,
        lr=0.01,
        l2=0.0001,
        val_eval_fn=const_val_eval,
        patience=5,
    )

    # Epoch 1 sets best=0.5; epochs 2..6 do not improve; break at epoch 6.
    assert len(history) <= 6
    assert len(history) < 100
    assert call_count[0] == len(history)
    # Best model should be a valid model dict.
    assert "w1" in best_model
    assert "b1" in best_model
    assert "w2" in best_model
    assert "b2" in best_model


# ---------------------------------------------------------------------------
# Test 4: 10-seed loop produces 10 model JSON files
# ---------------------------------------------------------------------------


def test_ten_seed_loop_produces_ten_models(tmp_path):
    """The 10-seed loop writes 10 model JSON files."""
    csv_path = tmp_path / "cands.csv"
    _write_synthetic_candidates(str(csv_path), n_groups=20, n_neg_per_group=3)
    v1_path = tmp_path / "v1_model.json"
    _write_synthetic_v1_model(str(v1_path), hidden_dim=8, seed=42)

    output_dir = tmp_path / "out"
    args = _default_args(str(csv_path), str(v1_path), str(output_dir))
    run_train_v2(args)

    model_files = sorted(output_dir.glob("model_seed*.json"))
    assert len(model_files) == 10
    seeds_seen = set()
    for path in model_files:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert "parameters" in payload
        assert payload["hidden_dim"] == 8
        assert payload["warm_start_path"] == os.path.abspath(str(v1_path))
        assert payload["recipe"] == "warm_started_v1_chemformer_aware_pairwise_mlp_with_early_stopping"
        seeds_seen.add(payload["seed"])
    assert seeds_seen == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}


# ---------------------------------------------------------------------------
# Test 5: paired_significance.json has required keys
# ---------------------------------------------------------------------------


def test_paired_significance_json_keys(tmp_path):
    """paired_significance.json contains all required keys."""
    csv_path = tmp_path / "cands.csv"
    _write_synthetic_candidates(str(csv_path), n_groups=20, n_neg_per_group=3)
    v1_path = tmp_path / "v1_model.json"
    _write_synthetic_v1_model(str(v1_path), hidden_dim=8, seed=42)

    output_dir = tmp_path / "out"
    args = _default_args(str(csv_path), str(v1_path), str(output_dir))
    run_train_v2(args)

    with open(output_dir / "paired_significance.json", encoding="utf-8") as handle:
        paired = json.load(handle)
    required = {"mean_delta", "std_delta", "t_stat", "p_value", "ci_low", "ci_high", "n_seeds"}
    assert required.issubset(paired.keys())
    assert paired["n_seeds"] == 10
    assert paired["metric"] == "top1"
    assert paired["v2_score_name"] == "pc_cng_mlp_calibrator_v2"
    assert paired["baseline_score_name"] == "chemformer_likelihood"
    assert paired["ci_low"] <= paired["mean_delta"] <= paired["ci_high"]


# ---------------------------------------------------------------------------
# Test 6: summary.json has Top-1/Top-3/Top-5/NDCG@10 with mean and std
# ---------------------------------------------------------------------------


def test_summary_json_has_all_metrics(tmp_path):
    """summary.json reports mean+std for Top-1/Top-3/Top-5/NDCG@10."""
    csv_path = tmp_path / "cands.csv"
    _write_synthetic_candidates(str(csv_path), n_groups=20, n_neg_per_group=3)
    v1_path = tmp_path / "v1_model.json"
    _write_synthetic_v1_model(str(v1_path), hidden_dim=8, seed=42)

    output_dir = tmp_path / "out"
    args = _default_args(str(csv_path), str(v1_path), str(output_dir))
    run_train_v2(args)

    with open(output_dir / "summary.json", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["n_seeds"] == 10
    assert summary["task"] == "p2_04_mlp_calibrator_v2_chemformer_aware"
    for metric in ["top1", "top3", "top5", "ndcg10"]:
        assert metric in summary["metrics"]
        block = summary["metrics"][metric]
        for key in ["v2_mean", "v2_std", "baseline_mean", "baseline_std", "delta_mean", "paired_test"]:
            assert key in block
        pt = block["paired_test"]
        for key in ["mean_delta", "std_delta", "t_stat", "p_value", "ci_low", "ci_high", "n_seeds"]:
            assert key in pt
        assert pt["n_seeds"] == 10


# ---------------------------------------------------------------------------
# Test 7: Synthetic small dataset (10 rows) end-to-end
# ---------------------------------------------------------------------------


def test_synthetic_ten_row_end_to_end(tmp_path):
    """End-to-end run on a 10-row synthetic dataset completes without error."""
    csv_path = tmp_path / "cands.csv"
    _write_synthetic_candidates(str(csv_path), n_groups=5, n_neg_per_group=1)
    assert sum(1 for _ in open(csv_path)) - 1 == 10  # 10 data rows

    v1_path = tmp_path / "v1_model.json"
    _write_synthetic_v1_model(str(v1_path), hidden_dim=8, seed=42)

    output_dir = tmp_path / "out"
    args = _default_args(str(csv_path), str(v1_path), str(output_dir), epochs=10)
    result = run_train_v2(args)

    assert os.path.exists(output_dir / "summary.json")
    assert os.path.exists(output_dir / "paired_significance.json")
    assert os.path.exists(output_dir / "v2_calibrator_recipe.json")
    assert os.path.exists(output_dir / "per_seed_metrics.csv")
    model_files = list(output_dir.glob("model_seed*.json"))
    assert len(model_files) == 10
    assert result["decision"] in {"GO", "NO-GO"}

    # per_seed_metrics.csv should have 10 data rows + 1 header
    with open(output_dir / "per_seed_metrics.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    for row in rows:
        for col in ["val_top1", "val_top3", "val_top5", "val_ndcg10",
                     "baseline_val_top1", "baseline_val_top3",
                     "baseline_val_top5", "baseline_val_ndcg10"]:
            assert col in row


# ---------------------------------------------------------------------------
# Test 8: paired_significance_test function (10-seed paired)
# ---------------------------------------------------------------------------


def test_paired_significance_function_basic():
    """paired_significance_test returns a valid paired t-test result."""
    v2 = [0.80, 0.82, 0.79, 0.81, 0.83, 0.78, 0.80, 0.81, 0.82, 0.80]
    base = [0.75, 0.76, 0.74, 0.75, 0.77, 0.73, 0.74, 0.75, 0.76, 0.74]
    result = paired_significance_test(v2, base)
    assert result["n_seeds"] == 10
    assert result["mean_delta"] > 0  # v2 > baseline
    assert result["t_stat"] > 0
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["ci_low"] < result["ci_high"]
    assert result["ci_low"] <= result["mean_delta"] <= result["ci_high"]


def test_paired_significance_function_identical():
    """When v2 == baseline, mean_delta=0 and p_value is large."""
    vals = [0.5] * 10
    result = paired_significance_test(vals, vals)
    assert result["n_seeds"] == 10
    assert abs(result["mean_delta"]) < 1e-9
    # std_delta=0 → degenerate case → p_value=1.0
    assert result["p_value"] == 1.0


def test_paired_significance_function_short():
    """Fewer than 2 samples → degenerate output, no crash."""
    result = paired_significance_test([0.8], [0.7])
    assert result["n_seeds"] == 1
    assert result["mean_delta"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Test 9: parse_args defaults
# ---------------------------------------------------------------------------


def test_parse_args_defaults(tmp_path):
    """parse_args applies v2 defaults (hidden_dim=32, lr=0.005, patience=20)."""
    argv = [
        "--candidates", str(tmp_path / "c.csv"),
        "--warm-start", str(tmp_path / "v.json"),
        "--output-dir", str(tmp_path / "o"),
    ]
    args = parse_args(argv)
    assert args.hidden_dim == 32
    assert args.epochs == 2000
    assert args.learning_rate == 0.005
    assert args.l2 == 0.0001
    assert args.early_stopping_patience == 20
    assert args.train_split_ratio == 0.8
    assert args.primary_score == "chemformer_likelihood"
    assert args.pc_score == "pc_cng"
    assert args.seeds == DEFAULT_SEEDS
    assert args.seeds.count(",") == 9  # 10 seeds


def test_parse_args_rejects_non_ten_seeds(tmp_path):
    """run_train_v2 raises when not exactly 10 seeds are provided."""
    csv_path = tmp_path / "cands.csv"
    _write_synthetic_candidates(str(csv_path), n_groups=10, n_neg_per_group=2)
    v1_path = tmp_path / "v1_model.json"
    _write_synthetic_v1_model(str(v1_path), hidden_dim=8, seed=42)

    output_dir = tmp_path / "out"
    args = _default_args(
        str(csv_path), str(v1_path), str(output_dir),
        seeds="1,2,3",  # only 3 seeds
    )
    with pytest.raises(SystemExit):
        run_train_v2(args)
