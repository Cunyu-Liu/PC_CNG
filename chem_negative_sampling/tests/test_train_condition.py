"""
Unit tests for train_condition.py (P3-04).

Covers main functions:
  - extract_product_from_reaction
  - featurize_product
  - load_or_create_idx (HC #9 split contract)
  - build_dataset
  - train_head (LogisticRegression per head)
  - paired_bootstrap_ci (HC #5 simple bootstrap)
  - family_cluster_bootstrap_ci (HC #5 family-cluster bootstrap)
  - train_one_seed (end-to-end one seed)
  - run_training (multi-seed + summary)

HC #4: coverage target >=80%.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Make chem_negative_sampling package importable.
_HERE = Path(__file__).resolve().parent
_CNS_ROOT = _HERE.parent
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from training.train_condition import (
    P2_08_BASELINE_TEST_TOP1,
    build_dataset,
    extract_product_from_reaction,
    family_cluster_bootstrap_ci,
    featurize_product,
    load_conditions,
    load_or_create_idx,
    main,
    paired_bootstrap_ci,
    run_training,
    train_head,
    train_one_seed,
)


def test_extract_product_from_reaction_double_arrow() -> None:
    """Standard R>>P reaction should yield product."""
    assert extract_product_from_reaction("CCO.CC(=O)O>>CC(=O)OCC") == "CC(=O)OCC"
    assert extract_product_from_reaction("A>>B") == "B"


def test_extract_product_from_reaction_single_arrow() -> None:
    """Three-part R>A>P reaction should yield product (last part)."""
    assert extract_product_from_reaction("A>[Pd].CCO>B") == "B"


def test_extract_product_from_reaction_no_arrow() -> None:
    """Bare SMILES should be returned as-is."""
    assert extract_product_from_reaction("CCO") == "CCO"


def test_extract_product_from_reaction_empty() -> None:
    """Empty input should yield empty string."""
    assert extract_product_from_reaction("") == ""


def test_extract_product_from_reaction_whitespace() -> None:
    """Whitespace around product should be stripped."""
    assert extract_product_from_reaction("A >> B ") == "B"


def test_featurize_product_shape_and_dtype() -> None:
    """Morgan fingerprint should be (2048,) uint8."""
    fp = featurize_product("CCO")
    assert fp.shape == (2048,)
    assert fp.dtype == np.uint8


def test_featurize_product_invalid_smiles() -> None:
    """Invalid SMILES should yield all-zero fingerprint."""
    fp = featurize_product("XXXXX_invalid")
    assert fp.shape == (2048,)
    assert int(fp.sum()) == 0


def test_featurize_product_empty() -> None:
    """Empty SMILES should yield all-zero fingerprint."""
    fp = featurize_product("")
    assert int(fp.sum()) == 0


def test_featurize_product_deterministic() -> None:
    """Same SMILES should produce same fingerprint."""
    fp1 = featurize_product("CCO")
    fp2 = featurize_product("CCO")
    assert np.array_equal(fp1, fp2)


def test_load_conditions(tmp_path) -> None:
    """load_conditions should return list of dicts from JSON file."""
    p = tmp_path / "c.json"
    data = [{"source_id": "1", "reaction_smiles": "A>>B"}]
    p.write_text(json.dumps(data))
    result = load_conditions(str(p))
    assert result == data


def test_load_or_create_idx_loads_existing(tmp_path) -> None:
    """Existing idx file should be loaded as-is."""
    idx_path = tmp_path / "idx.json"
    with open(idx_path, "w") as f:
        json.dump({"indices": [0, 5, 10]}, f)
    result = load_or_create_idx(str(idx_path), 100, "train", [])
    assert result == [0, 5, 10]


def test_load_or_create_idx_auto_creates_train(tmp_path) -> None:
    """Missing idx file should be auto-created with stratified split."""
    records = [
        {"reaction_smiles": f"A{i}>>B{i}", "source_id": str(i)}
        for i in range(100)
    ]
    train_path = tmp_path / "train.json"
    val_path = tmp_path / "val.json"
    test_path = tmp_path / "test.json"

    train_idx = load_or_create_idx(str(train_path), 100, "train", records)
    val_idx = load_or_create_idx(str(val_path), 100, "val", records)
    test_idx = load_or_create_idx(str(test_path), 100, "test", records)

    assert os.path.exists(train_path)
    assert os.path.exists(val_path)
    assert os.path.exists(test_path)

    # No overlap between splits.
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx).isdisjoint(set(test_idx))
    assert set(val_idx).isdisjoint(set(test_idx))

    # Roughly 80/10/10 split (cluster-based, may be slightly off).
    assert 70 <= len(train_idx) <= 90
    assert 5 <= len(val_idx) <= 20
    assert 5 <= len(test_idx) <= 20


def test_load_or_create_idx_stratified_by_reaction_smiles(tmp_path) -> None:
    """Records sharing the same reaction_smiles should not be split across sets."""
    records = []
    for rxn_smiles in ["A>>B", "C>>D", "E>>F", "G>>H", "I>>J"]:
        for j in range(10):  # 10 records per unique reaction
            records.append({"reaction_smiles": rxn_smiles, "source_id": f"{rxn_smiles}_{j}"})

    train_path = tmp_path / "train.json"
    val_path = tmp_path / "val.json"
    test_path = tmp_path / "test.json"

    train_idx = load_or_create_idx(str(train_path), 50, "train", records)
    val_idx = load_or_create_idx(str(val_path), 50, "val", records)
    test_idx = load_or_create_idx(str(test_path), 50, "test", records)

    # Verify: each reaction_smiles appears in only one split.
    train_rxns = {records[i]["reaction_smiles"] for i in train_idx}
    val_rxns = {records[i]["reaction_smiles"] for i in val_idx}
    test_rxns = {records[i]["reaction_smiles"] for i in test_idx}

    assert train_rxns.isdisjoint(val_rxns)
    assert train_rxns.isdisjoint(test_rxns)
    assert val_rxns.isdisjoint(test_rxns)


def test_load_or_create_idx_none_path() -> None:
    """None idx_path should still return valid indices (auto-create, no save)."""
    records = [
        {"reaction_smiles": f"A{i}>>B{i}", "source_id": str(i)}
        for i in range(20)
    ]
    train_idx = load_or_create_idx(None, 20, "train", records)
    assert len(train_idx) > 0
    assert all(0 <= i < 20 for i in train_idx)


def test_load_or_create_idx_invalid_split() -> None:
    """Invalid split name should raise ValueError."""
    try:
        load_or_create_idx(None, 10, "invalid_split", [])
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_build_dataset_basic() -> None:
    """build_dataset should produce X, y, label_maps with correct shapes."""
    records = [
        {"reaction_smiles": "A>>B", "catalyst": "[Pd]",
         "solvent": "CCO", "reagent": "c1ccccc1"},
        {"reaction_smiles": "C>>D", "catalyst": "[Cu]",
         "solvent": "CO", "reagent": "c1ccccc1"},
        {"reaction_smiles": "E>>F", "catalyst": "[Pd]",
         "solvent": "CCO", "reagent": "CC(=O)O"},
    ]
    X, y, label_maps = build_dataset(records, [0, 1, 2])
    assert X.shape == (3, 2048)
    assert "catalyst" in y
    assert "solvent" in y
    assert "reagent" in y
    assert len(y["catalyst"]) == 3
    assert len(y["solvent"]) == 3
    assert len(y["reagent"]) == 3
    assert len(label_maps["catalyst"]) == 2  # [Pd], [Cu]
    assert len(label_maps["solvent"]) == 2  # CCO, CO


def test_build_dataset_handles_missing_values() -> None:
    """Missing catalyst/solvent/reagent should default to 'none'."""
    records = [
        {"reaction_smiles": "A>>B"},  # no agents fields
    ]
    X, y, label_maps = build_dataset(records, [0])
    assert X.shape == (1, 2048)
    assert y["catalyst"][0] == 0  # 'none' is alphabetically first
    assert label_maps["catalyst"]["none"] == 0


def test_build_dataset_multi_smiles_label() -> None:
    """Multi-SMILES labels should take only the first SMILES."""
    records = [
        {"reaction_smiles": "A>>B", "catalyst": "[Pd].[Cu]",
         "solvent": "", "reagent": ""},
    ]
    _, y, label_maps = build_dataset(records, [0])
    # First SMILES is [Pd], so its label index should match [Pd] in label_map.
    pd_idx = label_maps["catalyst"]["[Pd]"]
    assert y["catalyst"][0] == pd_idx


def test_paired_bootstrap_ci_positive_difference() -> None:
    """System A clearly better than B: CI should exclude 0, p < 0.05."""
    a = [0.7, 0.8, 0.6, 0.75, 0.85, 0.72, 0.78, 0.65, 0.8, 0.77]
    b = [0.5] * 10
    mean_diff, ci_low, ci_high, p_value = paired_bootstrap_ci(
        a, b, n_iterations=1000, seed=42
    )
    assert mean_diff > 0
    assert ci_low > 0
    assert p_value < 0.05


def test_paired_bootstrap_ci_equal_systems() -> None:
    """Equal systems: mean_diff ~ 0, CI should include 0, p > 0.05."""
    a = [0.5, 0.5, 0.5, 0.5, 0.5]
    b = [0.5, 0.5, 0.5, 0.5, 0.5]
    mean_diff, ci_low, ci_high, p_value = paired_bootstrap_ci(
        a, b, n_iterations=1000, seed=42
    )
    assert abs(mean_diff) < 1e-6
    assert ci_low <= 0 <= ci_high
    assert p_value > 0.05


def test_paired_bootstrap_ci_length_mismatch_raises() -> None:
    """Length mismatch should raise ValueError."""
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.5, 0.6], [0.5], n_iterations=100)


def test_paired_bootstrap_ci_empty() -> None:
    """Empty input should return trivial CI."""
    md, lo, hi, p = paired_bootstrap_ci([], [], n_iterations=10)
    assert md == 0.0
    assert p == 1.0


def test_family_cluster_bootstrap_ci_basic() -> None:
    """Family-cluster bootstrap should produce sensible CI for clear win."""
    a = [0.7, 0.8, 0.75]
    b = [0.5, 0.5, 0.5]
    family_ids = [
        ["src1", "src2", "src1"],
        ["src3", "src4"],
        ["src5", "src5", "src6"],
    ]
    md, lo, hi, p = family_cluster_bootstrap_ci(
        a, b, family_ids, n_iterations=1000, seed=42
    )
    assert md > 0
    # CI should bracket the mean_diff.
    assert lo <= md <= hi


def test_family_cluster_bootstrap_ci_length_mismatch() -> None:
    """Length mismatch between a and family_ids should raise ValueError."""
    with pytest.raises(ValueError):
        family_cluster_bootstrap_ci(
            [0.5, 0.6], [0.5, 0.5], [["s1"]], n_iterations=100
        )


def test_family_cluster_bootstrap_ci_empty() -> None:
    """Empty inputs should return trivial CI."""
    md, lo, hi, p = family_cluster_bootstrap_ci([], [], [], n_iterations=10)
    assert md == 0.0
    assert p == 1.0


def test_train_head_small_synthetic() -> None:
    """train_head on small synthetic data should produce valid metrics."""
    rng = np.random.RandomState(42)
    n = 60
    X = rng.randint(0, 2, size=(n, 128)).astype(np.uint8)
    y = rng.randint(0, 3, size=n)
    X_train, X_val, X_test = X[:40], X[40:50], X[50:]
    y_train, y_val, y_test = y[:40], y[40:50], y[50:]
    metrics = train_head(
        X_train, y_train, X_val, y_val, X_test, y_test, seed=42
    )
    assert "train_top1" in metrics
    assert "val_top1" in metrics
    assert "test_top1" in metrics
    assert "train_top3" in metrics
    assert 0.0 <= metrics["test_top1"] <= 1.0
    assert 0.0 <= metrics["test_top3"] <= 1.0
    assert metrics["n_classes"] >= 1


def test_train_head_single_class() -> None:
    """Single class in training should not crash; DummyClassifier used."""
    X = np.zeros((10, 16), dtype=np.uint8)
    X[5:, :] = 1
    y = np.zeros(10, dtype=np.int64)  # all class 0
    metrics = train_head(X[:6], y[:6], X[6:8], y[6:8], X[8:], y[8:], seed=0)
    assert metrics["n_classes"] >= 1
    assert 0.0 <= metrics["test_top1"] <= 1.0


def test_train_one_seed_end_to_end() -> None:
    """train_one_seed should train all three heads and return metrics."""
    records = []
    rng = np.random.RandomState(0)
    for i in range(60):
        cat = ["[Pd]", "[Cu]", "[Fe]"][rng.randint(0, 3)]
        sol = ["CCO", "CO", "O"][rng.randint(0, 3)]
        reg = ["c1ccccc1", "CC(=O)O", ""][rng.randint(0, 3)]
        records.append({
            "source_id": f"src{i}",
            "reaction_smiles": f"R{i}>>P{i}",
            "catalyst": cat,
            "solvent": sol,
            "reagent": reg,
        })

    train_idx = list(range(40))
    val_idx = list(range(40, 50))
    test_idx = list(range(50, 60))

    m = train_one_seed(records, train_idx, val_idx, test_idx, seed=42)
    assert "catalyst" in m
    assert "solvent" in m
    assert "reagent" in m
    assert "test_family_ids" in m
    assert m["seed"] == 42
    assert len(m["test_family_ids"]) == 10
    # All heads should have valid metric ranges.
    for head in ["catalyst", "solvent", "reagent"]:
        assert 0.0 <= m[head]["test_top1"] <= 1.0


def test_run_training_with_auto_idx(tmp_path) -> None:
    """End-to-end run_training with auto-created idx files (HC #9)."""
    data_path = tmp_path / "conditions.json"
    records = []
    rng = np.random.RandomState(0)
    for i in range(80):
        cat = ["[Pd]", "[Cu]"][rng.randint(0, 2)]
        sol = ["CCO", "CO"][rng.randint(0, 2)]
        reg = ["c1ccccc1", ""][rng.randint(0, 2)]
        records.append({
            "source_id": f"src{i}",
            "reaction_smiles": f"R{i}>>P{i}",
            "catalyst": cat,
            "solvent": sol,
            "reagent": reg,
        })
    with open(data_path, "w") as f:
        json.dump(records, f)

    train_path = tmp_path / "train_idx.json"
    val_path = tmp_path / "val_idx.json"
    test_path = tmp_path / "test_idx.json"

    out_dir = tmp_path / "out"
    summary = run_training(
        data_path=str(data_path),
        train_idx_path=str(train_path),
        val_idx_path=str(val_path),
        test_idx_path=str(test_path),
        seeds=[42, 43],
        output_dir=str(out_dir),
    )

    # Per-seed JSON files exist.
    assert (out_dir / "metrics_seed_42.json").exists()
    assert (out_dir / "metrics_seed_43.json").exists()
    # Summary JSON exists.
    assert (out_dir / "summary.json").exists()

    # Auto-created idx files (HC #9).
    assert train_path.exists()
    assert val_path.exists()
    assert test_path.exists()

    # Summary structure.
    assert "catalyst_test_top1_mean" in summary
    assert "solvent_test_top1_mean" in summary
    assert "reagent_test_top1_mean" in summary
    assert "catalyst_paired_bootstrap_ci" in summary
    assert summary["catalyst_paired_bootstrap_ci"]["baseline"] == "P2-08 synthetic ~50%"
    assert summary["catalyst_paired_bootstrap_ci"]["method"] == "family_cluster_bootstrap"
    assert "catalyst_paired_bootstrap_ci_simple" in summary
    assert summary["n_seeds"] == 2


def test_run_training_no_idx_paths_creates_default_files(tmp_path) -> None:
    """When idx paths are None, run_training should still work."""
    data_path = tmp_path / "conditions.json"
    records = []
    rng = np.random.RandomState(0)
    for i in range(60):
        cat = ["[Pd]", "[Cu]"][rng.randint(0, 2)]
        records.append({
            "source_id": f"src{i}",
            "reaction_smiles": f"R{i}>>P{i}",
            "catalyst": cat,
            "solvent": "CCO",
            "reagent": "",
        })
    with open(data_path, "w") as f:
        json.dump(records, f)

    out_dir = tmp_path / "out_no_idx"
    summary = run_training(
        data_path=str(data_path),
        train_idx_path=None,
        val_idx_path=None,
        test_idx_path=None,
        seeds=[42],
        output_dir=str(out_dir),
    )
    assert summary["n_seeds"] == 1
    assert (out_dir / "summary.json").exists()


def test_main_cli(tmp_path, monkeypatch) -> None:
    """CLI main() should produce per-seed and summary JSON."""
    data_path = tmp_path / "conditions.json"
    records = []
    rng = np.random.RandomState(0)
    for i in range(60):
        cat = ["[Pd]", "[Cu]"][rng.randint(0, 2)]
        records.append({
            "source_id": f"src{i}",
            "reaction_smiles": f"R{i}>>P{i}",
            "catalyst": cat,
            "solvent": "CCO",
            "reagent": "",
        })
    with open(data_path, "w") as f:
        json.dump(records, f)

    train_path = tmp_path / "train_idx.json"
    val_path = tmp_path / "val_idx.json"
    test_path = tmp_path / "test_idx.json"
    out_dir = tmp_path / "out_cli"

    monkeypatch.setattr(
        "sys.argv",
        ["train_condition.py",
         "--data", str(data_path),
         "--train-idx", str(train_path),
         "--val-idx", str(val_path),
         "--test-idx", str(test_path),
         "--seeds", "42,43",
         "--output-dir", str(out_dir)],
    )
    main()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "metrics_seed_42.json").exists()


def test_p2_08_baseline_value() -> None:
    """P2-08 baseline should be 0.50 (synthetic conditions)."""
    assert P2_08_BASELINE_TEST_TOP1 == 0.50


def test_paired_bootstrap_ci_negative_difference() -> None:
    """System A clearly worse than B: mean_diff < 0, CI excludes 0."""
    a = [0.3, 0.2, 0.4, 0.25, 0.35, 0.28, 0.22, 0.35, 0.2, 0.23]
    b = [0.5] * 10
    md, lo, hi, p = paired_bootstrap_ci(a, b, n_iterations=1000, seed=42)
    assert md < 0
    assert hi < 0
    assert p < 0.05
