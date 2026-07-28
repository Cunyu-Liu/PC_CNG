from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "chem_negative_sampling"))

from pc_cng.run_phase4_fixed_testset import load_external_normalized_csv


def test_external_loader_preserves_frozen_train_test_and_adds_metadata(tmp_path):
    path = tmp_path / "external.csv"
    pd.DataFrame(
        [
            {"reaction_smiles": "CC>>CCO", "split": "train", "source_id": "a", "yield": 80},
            {"reaction_smiles": "CC>>CCN", "split": "train", "source_id": "b", "yield": 10},
            {"reaction_smiles": "CC>>CCC", "split": "test", "source_id": "c", "yield": 50},
        ]
    ).to_csv(path, index=False)

    out = load_external_normalized_csv(path, max_train=0, max_test=0)

    assert len(out["train"]) == 2
    assert len(out["test"]) == 1
    assert out["n_train_total"] == 2
    assert out["n_test_total"] == 1
    for key in ("experimental_group", "reaction_family", "split_key", "yield_bin"):
        assert key in out["train"].columns
    assert out["train"]["reaction_smiles"].notna().all()
    assert out["test"]["reaction_smiles"].notna().all()


def test_external_loader_rejects_missing_test_split(tmp_path):
    path = tmp_path / "train_only.csv"
    pd.DataFrame(
        [{"reaction_smiles": "CC>>CCO", "split": "train"}]
    ).to_csv(path, index=False)

    try:
        load_external_normalized_csv(path, max_train=0, max_test=0)
    except ValueError as exc:
        assert "non-empty train/test" in str(exc)
    else:
        raise AssertionError("train-only external data must fail closed")
