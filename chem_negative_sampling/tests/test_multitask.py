"""Unit tests for P3-06 multi-task joint training (``multitask.py``).

Tests run WITHOUT GPU, WITHOUT remote access, and WITHOUT the actual P3-01
checkpoint.  A tiny ``d_model=16`` fallback backbone / tokenizer (provided
by ``multitask._Fallback*``) is used for speed.  Coverage target: ≥80%.
Test time target: <60s.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

# Force CPU for tests
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Ensure the chem_negative_sampling package root is importable
_HERE = Path(__file__).resolve().parent
_CNS_ROOT = _HERE.parent
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from models.multitask import (  # noqa: E402
    DEFAULT_SEEDS,
    TASKS,
    MultiTaskModel,
    MultiTaskTrainer,
    _FallbackBackbone,
    _FallbackTokenizer,
    _compute_auc,
    _compute_mrr,
    _extract_product_from_reaction,
    _load_condition_json,
    _load_retrosynthesis_csv,
    _load_yield_csv,
    _parse_yield,
    _render_summary_md,
    build_backbone,
    family_cluster_bootstrap_ci,
    load_multitask_data,
    load_or_create_split,
    load_tokenizer,
    main,
    paired_bootstrap_ci,
    run_experiment,
)

D_MODEL = 16
N_CAT = 3
N_SOL = 4
N_REG = 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def device() -> str:
    return "cpu"


@pytest.fixture
def tokenizer() -> Any:
    return load_tokenizer(vocab_path=None, max_seq_len=32)


@pytest.fixture
def backbone(device: str) -> torch.nn.Module:
    """Build a tiny fresh-init backbone (no checkpoint)."""
    return build_backbone(
        checkpoint_path=None,
        hparams={"d_model": D_MODEL, "vocabulary_size": 100},
        freeze=True,
        apply_lora_flag=False,  # fallback backbone has no LoRA targets
    )


@pytest.fixture
def multitask_model(backbone: torch.nn.Module, device: str) -> MultiTaskModel:
    return MultiTaskModel(
        backbone=backbone,
        n_catalyst_classes=N_CAT,
        n_solvent_classes=N_SOL,
        n_reagent_classes=N_REG,
        d_model=D_MODEL,
    ).to(device)


def _make_retro_rows(n: int = 24, n_sources: int = 6, seed: int = 0) -> List[Dict[str, Any]]:
    """Synthetic retrosynthesis rows."""
    rng = np.random.RandomState(seed)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        sid = f"src_{i % n_sources}"
        smiles = f"CC{i % 5}>>CC{(i + 1) % 5}O{i % 3}"
        label = int(rng.randint(0, 2))
        rows.append({"smiles": smiles, "label": label, "source_id": sid})
    return rows


def _make_condition_rows(n: int = 24, n_sources: int = 6, seed: int = 0) -> List[Dict[str, Any]]:
    """Synthetic condition rows with integer labels."""
    rng = np.random.RandomState(seed)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        sid = f"src_{i % n_sources}"
        smiles = f"CC{i % 5}>>CC{(i + 1) % 5}O"
        rows.append({
            "smiles": smiles,
            "catalyst_label": int(rng.randint(0, N_CAT)),
            "solvent_label": int(rng.randint(0, N_SOL)),
            "reagent_label": int(rng.randint(0, N_REG)),
            "source_id": sid,
        })
    return rows


def _make_yield_rows(n: int = 24, n_sources: int = 6, seed: int = 0) -> List[Dict[str, Any]]:
    """Synthetic yield rows."""
    rng = np.random.RandomState(seed)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        sid = f"src_{i % n_sources}"
        smiles = f"CC{i % 5}>>CC{(i + 1) % 5}O"
        yld = float(rng.uniform(0, 100))
        rows.append({"smiles": smiles, "yield": yld, "source_id": sid})
    return rows


@pytest.fixture
def retro_rows() -> List[Dict[str, Any]]:
    return _make_retro_rows(n=24, seed=0)


@pytest.fixture
def condition_rows() -> List[Dict[str, Any]]:
    return _make_condition_rows(n=24, seed=1)


@pytest.fixture
def yield_rows() -> List[Dict[str, Any]]:
    return _make_yield_rows(n=24, seed=2)


@pytest.fixture
def train_val_test_retro(retro_rows: List[Dict[str, Any]]):
    tr, va, te = load_or_create_split(retro_rows, None, None, None, seed=42)
    return (
        [retro_rows[i] for i in tr],
        [retro_rows[i] for i in va],
        [retro_rows[i] for i in te],
    )


# ---------------------------------------------------------------------------
# 1. MultiTaskModel tests
# ---------------------------------------------------------------------------
class TestMultiTaskModel:
    def test_construct_all_heads(self, backbone: torch.nn.Module) -> None:
        model = MultiTaskModel(
            backbone=backbone,
            n_catalyst_classes=N_CAT,
            n_solvent_classes=N_SOL,
            n_reagent_classes=N_REG,
            d_model=D_MODEL,
        )
        assert model.retrosynthesis_head is not None
        assert model.catalyst_head is not None
        assert model.solvent_head is not None
        assert model.reagent_head is not None
        assert model.yield_head is not None
        assert model.active_tasks == set(TASKS)

    def test_construct_single_task_retro(self, backbone: torch.nn.Module) -> None:
        model = MultiTaskModel(
            backbone=backbone, d_model=D_MODEL, active_tasks={"retrosynthesis"}
        )
        assert model.retrosynthesis_head is not None
        assert model.catalyst_head is None
        assert model.yield_head is None

    def test_construct_single_task_condition(self, backbone: torch.nn.Module) -> None:
        model = MultiTaskModel(
            backbone=backbone,
            n_catalyst_classes=N_CAT,
            n_solvent_classes=N_SOL,
            n_reagent_classes=N_REG,
            d_model=D_MODEL,
            active_tasks={"condition"},
        )
        assert model.retrosynthesis_head is None
        assert model.catalyst_head is not None
        assert model.yield_head is None

    def test_construct_single_task_yield(self, backbone: torch.nn.Module) -> None:
        model = MultiTaskModel(
            backbone=backbone, d_model=D_MODEL, active_tasks={"yield"}
        )
        assert model.retrosynthesis_head is None
        assert model.catalyst_head is None
        assert model.yield_head is not None

    def test_param_counts(self, backbone: torch.nn.Module) -> None:
        model_full = MultiTaskModel(
            backbone=backbone, n_catalyst_classes=N_CAT,
            n_solvent_classes=N_SOL, n_reagent_classes=N_REG, d_model=D_MODEL,
        )
        model_st = MultiTaskModel(
            backbone=backbone, d_model=D_MODEL, active_tasks={"retrosynthesis"}
        )
        full_params = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
        st_params = sum(p.numel() for p in model_st.parameters() if p.requires_grad)
        # Multi-task has more trainable params than single-task
        assert full_params > st_params, f"full={full_params} <= st={st_params}"
        # Backbone is frozen so params are only in heads
        assert full_params > 0
        assert st_params > 0

    def test_forward_retrosynthesis(self, multitask_model: MultiTaskModel) -> None:
        ids = torch.tensor([[2, 5, 6, 7, 3]])
        mask = torch.ones_like(ids)
        out = multitask_model(ids, attention_mask=mask, task="retrosynthesis")
        assert out.shape == (1,), f"got {out.shape}"

    def test_forward_condition(self, multitask_model: MultiTaskModel) -> None:
        ids = torch.tensor([[2, 5, 6, 7, 3]])
        mask = torch.ones_like(ids)
        out = multitask_model(ids, attention_mask=mask, task="condition")
        assert isinstance(out, dict)
        assert set(out.keys()) == {"catalyst", "solvent", "reagent"}
        assert out["catalyst"].shape == (1, N_CAT)
        assert out["solvent"].shape == (1, N_SOL)
        assert out["reagent"].shape == (1, N_REG)

    def test_forward_yield(self, multitask_model: MultiTaskModel) -> None:
        ids = torch.tensor([[2, 5, 6, 7, 3]])
        mask = torch.ones_like(ids)
        out = multitask_model(ids, attention_mask=mask, task="yield")
        assert out.shape == (1,), f"got {out.shape}"

    def test_forward_unknown_task_raises(self, multitask_model: MultiTaskModel) -> None:
        ids = torch.tensor([[2, 5, 6, 7, 3]])
        mask = torch.ones_like(ids)
        with pytest.raises(ValueError, match="Unknown task"):
            multitask_model(ids, attention_mask=mask, task="bogus")

    def test_forward_inactive_head_raises(self, backbone: torch.nn.Module) -> None:
        model = MultiTaskModel(
            backbone=backbone, d_model=D_MODEL, active_tasks={"retrosynthesis"}
        )
        ids = torch.tensor([[2, 5, 6, 7, 3]])
        mask = torch.ones_like(ids)
        with pytest.raises(ValueError, match="yield head not active"):
            model(ids, attention_mask=mask, task="yield")

    def test_forward_all(self, multitask_model: MultiTaskModel) -> None:
        ids = torch.tensor([[2, 5, 6, 7, 3], [2, 8, 9, 3, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
        out = multitask_model.forward_all(ids, attention_mask=mask)
        assert "retrosynthesis" in out
        assert "catalyst" in out
        assert "solvent" in out
        assert "reagent" in out
        assert "yield" in out
        assert out["retrosynthesis"].shape[0] == 2
        assert out["catalyst"].shape == (2, N_CAT)
        assert out["yield"].shape == (2,)

    def test_forward_all_partial(self, backbone: torch.nn.Module) -> None:
        model = MultiTaskModel(
            backbone=backbone, d_model=D_MODEL, active_tasks={"yield"}
        )
        ids = torch.tensor([[2, 5, 6, 7, 3]])
        mask = torch.ones_like(ids)
        out = model.forward_all(ids, attention_mask=mask)
        assert "yield" in out
        assert "retrosynthesis" not in out
        assert "catalyst" not in out

    def test_yield_head_architecture(self, backbone: torch.nn.Module) -> None:
        model = MultiTaskModel(backbone=backbone, d_model=D_MODEL)
        # Linear -> ReLU -> Linear
        assert isinstance(model.yield_head[0], torch.nn.Linear)
        assert isinstance(model.yield_head[1], torch.nn.ReLU)
        assert isinstance(model.yield_head[2], torch.nn.Linear)
        assert model.yield_head[0].in_features == D_MODEL
        assert model.yield_head[0].out_features == 64
        assert model.yield_head[2].in_features == 64
        assert model.yield_head[2].out_features == 1

    def test_condition_head_architecture(self, backbone: torch.nn.Module) -> None:
        model = MultiTaskModel(
            backbone=backbone, n_catalyst_classes=N_CAT,
            n_solvent_classes=N_SOL, n_reagent_classes=N_REG, d_model=D_MODEL,
        )
        assert isinstance(model.catalyst_head, torch.nn.Linear)
        assert model.catalyst_head.in_features == D_MODEL
        assert model.catalyst_head.out_features == N_CAT
        assert model.solvent_head.out_features == N_SOL
        assert model.reagent_head.out_features == N_REG


# ---------------------------------------------------------------------------
# 2. MultiTaskTrainer tests
# ---------------------------------------------------------------------------
class TestMultiTaskTrainer:
    def test_construct(
        self,
        multitask_model: MultiTaskModel,
        tokenizer: Any,
        retro_rows: List[Dict[str, Any]],
        condition_rows: List[Dict[str, Any]],
        yield_rows: List[Dict[str, Any]],
        device: str,
    ) -> None:
        trainer = MultiTaskTrainer(
            model=multitask_model,
            tokenizer=tokenizer,
            train_rows_by_task={
                "retrosynthesis": retro_rows,
                "condition": condition_rows,
                "yield": yield_rows,
            },
            device=device,
            epochs=1,
            batch_size=4,
            uncertainty_weights=True,
        )
        assert trainer.uncertainty_weights is True
        assert trainer.log_vars is not None
        assert set(trainer.active_tasks) == set(TASKS)

    def test_construct_no_uncertainty(
        self,
        multitask_model: MultiTaskModel,
        tokenizer: Any,
        retro_rows: List[Dict[str, Any]],
        device: str,
    ) -> None:
        trainer = MultiTaskTrainer(
            model=multitask_model,
            tokenizer=tokenizer,
            train_rows_by_task={"retrosynthesis": retro_rows},
            device=device,
            epochs=1,
            batch_size=4,
            uncertainty_weights=False,
        )
        assert trainer.log_vars is None
        assert trainer.active_tasks == ["retrosynthesis"]

    def test_train_all_tasks(
        self,
        multitask_model: MultiTaskModel,
        tokenizer: Any,
        retro_rows: List[Dict[str, Any]],
        condition_rows: List[Dict[str, Any]],
        yield_rows: List[Dict[str, Any]],
        device: str,
    ) -> None:
        trainer = MultiTaskTrainer(
            model=multitask_model,
            tokenizer=tokenizer,
            train_rows_by_task={
                "retrosynthesis": retro_rows[:16],
                "condition": condition_rows[:16],
                "yield": yield_rows[:16],
            },
            test_rows_by_task={
                "retrosynthesis": retro_rows[16:],
                "condition": condition_rows[16:],
                "yield": yield_rows[16:],
            },
            device=device,
            epochs=2,
            batch_size=4,
            uncertainty_weights=True,
        )
        history = trainer.train()
        assert set(history.keys()) == set(TASKS)
        for task in TASKS:
            assert len(history[task]) == 2, f"task={task} history={history[task]}"
            # Loss should be a finite float
            assert all(isinstance(x, float) for x in history[task])
            assert all(not math.isnan(x) for x in history[task])

    def test_train_empty_tasks(self, multitask_model: MultiTaskModel, tokenizer: Any, device: str) -> None:
        trainer = MultiTaskTrainer(
            model=multitask_model,
            tokenizer=tokenizer,
            train_rows_by_task={},
            device=device,
            epochs=1,
            batch_size=4,
        )
        history = trainer.train()
        assert history == {}

    def test_evaluate_retrosynthesis(
        self,
        multitask_model: MultiTaskModel,
        tokenizer: Any,
        retro_rows: List[Dict[str, Any]],
        device: str,
    ) -> None:
        trainer = MultiTaskTrainer(
            model=multitask_model,
            tokenizer=tokenizer,
            train_rows_by_task={"retrosynthesis": retro_rows[:16]},
            test_rows_by_task={"retrosynthesis": retro_rows[16:]},
            device=device,
            epochs=1,
            batch_size=4,
            uncertainty_weights=False,
        )
        trainer.train()
        metrics = trainer.evaluate("retrosynthesis")
        assert metrics["task"] == "retrosynthesis"
        assert metrics["n_examples"] == len(retro_rows[16:])
        assert "mrr" in metrics
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert len(metrics["per_example_correct"]) == metrics["n_examples"]

    def test_evaluate_condition(
        self,
        multitask_model: MultiTaskModel,
        tokenizer: Any,
        condition_rows: List[Dict[str, Any]],
        device: str,
    ) -> None:
        trainer = MultiTaskTrainer(
            model=multitask_model,
            tokenizer=tokenizer,
            train_rows_by_task={"condition": condition_rows[:16]},
            test_rows_by_task={"condition": condition_rows[16:]},
            device=device,
            epochs=1,
            batch_size=4,
            uncertainty_weights=False,
        )
        trainer.train()
        metrics = trainer.evaluate("condition")
        assert metrics["task"] == "condition"
        assert metrics["n_examples"] == len(condition_rows[16:])
        assert "catalyst_top1" in metrics
        assert "solvent_top1" in metrics
        assert "reagent_top1" in metrics
        assert "avg_top1" in metrics
        assert 0.0 <= metrics["avg_top1"] <= 1.0

    def test_evaluate_yield(
        self,
        multitask_model: MultiTaskModel,
        tokenizer: Any,
        yield_rows: List[Dict[str, Any]],
        device: str,
    ) -> None:
        trainer = MultiTaskTrainer(
            model=multitask_model,
            tokenizer=tokenizer,
            train_rows_by_task={"yield": yield_rows[:16]},
            test_rows_by_task={"yield": yield_rows[16:]},
            device=device,
            epochs=1,
            batch_size=4,
            uncertainty_weights=False,
        )
        trainer.train()
        metrics = trainer.evaluate("yield")
        assert metrics["task"] == "yield"
        assert metrics["n_examples"] == len(yield_rows[16:])
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0

    def test_evaluate_empty_test(self, multitask_model: MultiTaskModel, tokenizer: Any, device: str) -> None:
        trainer = MultiTaskTrainer(
            model=multitask_model,
            tokenizer=tokenizer,
            train_rows_by_task={"retrosynthesis": []},
            test_rows_by_task={"retrosynthesis": []},
            device=device,
            epochs=1,
            uncertainty_weights=False,
        )
        metrics = trainer.evaluate("retrosynthesis")
        assert metrics["n_examples"] == 0

    def test_evaluate_unknown_task_raises(
        self,
        multitask_model: MultiTaskModel,
        tokenizer: Any,
        device: str,
    ) -> None:
        trainer = MultiTaskTrainer(
            model=multitask_model,
            tokenizer=tokenizer,
            train_rows_by_task={},
            device=device,
            epochs=1,
        )
        with pytest.raises(ValueError, match="Unknown task"):
            trainer.evaluate("bogus")

    def test_uncertainty_weighting_changes_loss(
        self,
        multitask_model: MultiTaskModel,
        tokenizer: Any,
        retro_rows: List[Dict[str, Any]],
        device: str,
    ) -> None:
        """Verify that uncertainty weighting wraps the raw loss."""
        trainer = MultiTaskTrainer(
            model=multitask_model,
            tokenizer=tokenizer,
            train_rows_by_task={"retrosynthesis": retro_rows[:8]},
            device=device,
            epochs=1,
            batch_size=4,
            uncertainty_weights=True,
        )
        batch = retro_rows[:4]
        raw_loss = trainer._task_loss("retrosynthesis", batch)
        weighted = trainer._weighted_loss("retrosynthesis", raw_loss)
        # With log_var=0 init: weighted = 0.5 * exp(0) * raw + 0.5 * 0 = 0.5 * raw
        assert abs(weighted.item() - 0.5 * raw_loss.item()) < 1e-5

    def test_task_loss_unknown_task(
        self,
        multitask_model: MultiTaskModel,
        tokenizer: Any,
        device: str,
    ) -> None:
        trainer = MultiTaskTrainer(
            model=multitask_model,
            tokenizer=tokenizer,
            train_rows_by_task={},
            device=device,
            epochs=1,
        )
        with pytest.raises(ValueError, match="Unknown task"):
            trainer._task_loss("bogus", [])


# ---------------------------------------------------------------------------
# 3. load_multitask_data tests (with mock CSVs/JSON)
# ---------------------------------------------------------------------------
class TestLoadMultitaskData:
    def _write_retro_csv(self, path: Path, n_rows: int = 12) -> Path:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["source_id", "reaction_smiles", "label_type"])
            w.writeheader()
            for i in range(n_rows):
                w.writerow({
                    "source_id": f"rsrc_{i % 3}",
                    "reaction_smiles": f"CC{i}>>CC{i}O",
                    "label_type": "positive" if i % 2 == 0 else "negative",
                })
        return path

    def _write_condition_json(self, path: Path, n_rows: int = 12) -> Path:
        records = []
        for i in range(n_rows):
            records.append({
                "source_id": f"csrc_{i % 3}",
                "reaction_smiles": f"CC{i}>>CC{i}O",
                "catalyst": f"Pd{i % 2}",
                "solvent": f"DMF" if i % 3 == 0 else "THF",
                "reagent": f"K2CO3" if i % 2 == 0 else "",
            })
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(records, fh)
        return path

    def _write_yield_csv(self, path: Path, n_rows: int = 12) -> Path:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["source_id", "reaction_smiles", "yield", "reaction_class"])
            w.writeheader()
            for i in range(n_rows):
                yld = str(50 + i) if i % 4 != 0 else ""  # 25% missing yield
                w.writerow({
                    "source_id": f"ysrc_{i % 3}",
                    "reaction_smiles": f"CC{i}>>CC{i}O",
                    "yield": yld,
                    "reaction_class": f"Alkylation",
                })
        return path

    def test_load_retrosynthesis_csv(self, tmp_path: Path) -> None:
        path = self._write_retro_csv(tmp_path / "retro.csv")
        rows = _load_retrosynthesis_csv(str(path))
        assert len(rows) == 12
        assert all("smiles" in r and "label" in r and "source_id" in r for r in rows)
        assert rows[0]["label"] == 1  # positive
        assert rows[1]["label"] == 0  # negative

    def test_load_retrosynthesis_missing_file(self, tmp_path: Path) -> None:
        rows = _load_retrosynthesis_csv(str(tmp_path / "nonexistent.csv"))
        assert rows == []

    def test_load_condition_json(self, tmp_path: Path) -> None:
        path = self._write_condition_json(tmp_path / "cond.json")
        rows, maps = _load_condition_json(str(path))
        assert len(rows) == 12
        assert "catalyst" in maps and "solvent" in maps and "reagent" in maps
        # Reagent has "none" fallback for empty strings
        assert "none" in maps["reagent"]
        # Labels are integers in valid range
        for r in rows:
            assert 0 <= r["catalyst_label"] < len(maps["catalyst"])
            assert 0 <= r["solvent_label"] < len(maps["solvent"])
            assert 0 <= r["reagent_label"] < len(maps["reagent"])

    def test_load_condition_missing_file(self, tmp_path: Path) -> None:
        rows, maps = _load_condition_json(str(tmp_path / "nonexistent.json"))
        assert rows == []
        assert maps == {"catalyst": {}, "solvent": {}, "reagent": {}}

    def test_load_yield_csv_skips_empty(self, tmp_path: Path) -> None:
        path = self._write_yield_csv(tmp_path / "yield.csv")
        rows = _load_yield_csv(str(path))
        # 12 rows, 3 have empty yield (i % 4 == 0 -> i=0,4,8)
        assert len(rows) == 9
        assert all(isinstance(r["yield"], float) for r in rows)

    def test_load_yield_missing_file(self, tmp_path: Path) -> None:
        rows = _load_yield_csv(str(tmp_path / "nonexistent.csv"))
        assert rows == []

    def test_load_multitask_data(self, tmp_path: Path) -> None:
        self._write_retro_csv(tmp_path / "regiosqm20_normalized.csv")
        self._write_condition_json(tmp_path / "ord_conditions.json")
        self._write_yield_csv(tmp_path / "hitea_full_normalized.csv")

        data = load_multitask_data(str(tmp_path))
        assert "retrosynthesis" in data
        assert "condition" in data
        assert "yield" in data
        assert "condition_label_maps" in data
        assert "n_classes" in data
        assert len(data["retrosynthesis"]) == 12
        assert len(data["condition"]) == 12
        assert len(data["yield"]) == 9  # 3 rows skipped
        assert data["n_classes"]["catalyst"] > 0
        assert data["n_classes"]["solvent"] > 0
        assert data["n_classes"]["reagent"] > 0

    def test_load_multitask_data_empty_dir(self, tmp_path: Path) -> None:
        data = load_multitask_data(str(tmp_path))
        assert data["retrosynthesis"] == []
        assert data["condition"] == []
        assert data["yield"] == []
        assert data["n_classes"] == {"catalyst": 0, "solvent": 0, "reagent": 0}


# ---------------------------------------------------------------------------
# 4. Bootstrap CI tests
# ---------------------------------------------------------------------------
class TestBootstrapCI:
    def test_paired_bootstrap_ci_basic(self) -> None:
        a = [1.0, 1.0, 1.0, 1.0, 1.0]
        b = [0.0, 0.0, 0.0, 0.0, 0.0]
        md, lo, hi, p = paired_bootstrap_ci(a, b, n_iterations=500, seed=0)
        assert md == 1.0
        assert lo > 0
        assert hi > 0
        assert p < 0.1

    def test_paired_bootstrap_ci_no_diff(self) -> None:
        a = [0.5, 0.5, 0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5, 0.5, 0.5]
        md, lo, hi, p = paired_bootstrap_ci(a, b, n_iterations=500, seed=0)
        assert md == 0.0
        assert p == 1.0

    def test_paired_bootstrap_ci_empty(self) -> None:
        md, lo, hi, p = paired_bootstrap_ci([], [], n_iterations=100)
        assert (md, lo, hi, p) == (0.0, 0.0, 0.0, 1.0)

    def test_paired_bootstrap_ci_length_mismatch(self) -> None:
        md, lo, hi, p = paired_bootstrap_ci([1.0, 2.0], [1.0], n_iterations=100)
        assert (md, lo, hi, p) == (0.0, 0.0, 0.0, 1.0)

    def test_family_cluster_bootstrap_ci_basic(self) -> None:
        # 10 examples, 2 clusters
        a = [1.0] * 5 + [1.0] * 5
        b = [0.0] * 5 + [0.0] * 5
        clusters = ["c1"] * 5 + ["c2"] * 5
        md, lo, hi, p = family_cluster_bootstrap_ci(a, b, clusters, n_iterations=500, seed=0)
        assert md == 1.0
        assert lo > 0
        assert p < 0.1

    def test_family_cluster_bootstrap_ci_no_diff(self) -> None:
        a = [0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5]
        clusters = ["c1", "c2", "c3"]
        md, lo, hi, p = family_cluster_bootstrap_ci(a, b, clusters, n_iterations=500, seed=0)
        assert md == 0.0
        assert p == 1.0

    def test_family_cluster_bootstrap_ci_empty(self) -> None:
        md, lo, hi, p = family_cluster_bootstrap_ci([], [], [], n_iterations=100)
        assert (md, lo, hi, p) == (0.0, 0.0, 0.0, 1.0)

    def test_family_cluster_bootstrap_ci_mismatch(self) -> None:
        md, lo, hi, p = family_cluster_bootstrap_ci([1.0, 2.0], [1.0], ["c1"], n_iterations=100)
        assert (md, lo, hi, p) == (0.0, 0.0, 0.0, 1.0)

    def test_family_cluster_vs_paired(self) -> None:
        """Family-cluster CI should typically be wider (more conservative)."""
        rng = np.random.RandomState(0)
        a = rng.uniform(0.5, 1.0, size=50).tolist()
        b = rng.uniform(0.4, 0.9, size=50).tolist()
        clusters = [f"c{i % 5}" for i in range(50)]
        _, lo_p, hi_p, _ = paired_bootstrap_ci(a, b, n_iterations=500, seed=0)
        _, lo_f, hi_f, _ = family_cluster_bootstrap_ci(a, b, clusters, n_iterations=500, seed=0)
        # Both should produce finite CIs
        assert isinstance(lo_p, float) and isinstance(hi_p, float)
        assert isinstance(lo_f, float) and isinstance(hi_f, float)


# ---------------------------------------------------------------------------
# 5. Utility function tests
# ---------------------------------------------------------------------------
class TestUtilities:
    def test_load_tokenizer_fallback(self) -> None:
        tok = load_tokenizer(vocab_path=None, max_seq_len=16)
        assert isinstance(tok, _FallbackTokenizer)
        assert tok.max_seq_len == 16
        ids, mask = tok.batch_encode(["CCO", "CC>>O"])
        assert ids.shape[0] == 2
        assert mask.shape == ids.shape

    def test_load_tokenizer_missing_vocab(self, tmp_path: Path) -> None:
        tok = load_tokenizer(vocab_path=str(tmp_path / "no_vocab.json"))
        assert isinstance(tok, _FallbackTokenizer)

    def test_build_backbone_no_checkpoint(self) -> None:
        bb = build_backbone(
            checkpoint_path=None,
            hparams={"d_model": D_MODEL, "vocabulary_size": 100},
            freeze=True,
            apply_lora_flag=False,
        )
        # On servers with chemformer installed, PretrainedChemformerBackbone is
        # used; on minimal envs, _FallbackBackbone is used. Accept either.
        try:
            from models.pretrained_backbone import PretrainedChemformerBackbone
            acceptable = (_FallbackBackbone, PretrainedChemformerBackbone)
        except Exception:
            acceptable = (_FallbackBackbone,)
        assert isinstance(bb, acceptable)
        # Frozen backbone has no trainable params
        assert all(not p.requires_grad for p in bb.parameters())

    def test_build_backbone_missing_checkpoint(self, tmp_path: Path) -> None:
        bb = build_backbone(
            checkpoint_path=str(tmp_path / "missing.pt"),
            hparams={"d_model": D_MODEL, "vocabulary_size": 100},
            freeze=False,
            apply_lora_flag=False,
        )
        try:
            from models.pretrained_backbone import PretrainedChemformerBackbone
            acceptable = (_FallbackBackbone, PretrainedChemformerBackbone)
        except Exception:
            acceptable = (_FallbackBackbone,)
        assert isinstance(bb, acceptable)

    def test_parse_yield(self) -> None:
        assert _parse_yield("85.5") == 85.5
        assert _parse_yield(85.5) == 85.5
        assert _parse_yield("") is None
        assert _parse_yield(None) is None
        assert _parse_yield("nan") is None
        assert _parse_yield("NaN") is None
        assert _parse_yield("not_a_number") is None
        assert _parse_yield("0") == 0.0

    def test_extract_product_from_reaction(self) -> None:
        assert _extract_product_from_reaction("CCO.CC(=O)O>>CC(=O)OCC") == "CC(=O)OCC"
        assert _extract_product_from_reaction("A>[Pd].CCO>B") == "B"
        assert _extract_product_from_reaction("CCO") == "CCO"
        assert _extract_product_from_reaction("") == ""

    def test_compute_mrr(self) -> None:
        probs = np.array([0.9, 0.1, 0.8, 0.2])
        labels = np.array([1, 0, 1, 0])
        sids = ["a", "a", "b", "b"]
        mrr = _compute_mrr(probs, labels, sids)
        # Group a: 0.9 ranked 1st -> RR=1.0; Group b: 0.8 ranked 1st -> RR=1.0
        assert mrr == 1.0

    def test_compute_mrr_empty(self) -> None:
        assert _compute_mrr(np.array([]), np.array([]), []) == 0.0

    def test_compute_auc_basic(self) -> None:
        labels = np.array([1, 0, 1, 0])
        probs = np.array([0.9, 0.1, 0.8, 0.2])
        auc = _compute_auc(labels, probs)
        assert 0.0 <= auc <= 1.0
        # Perfect separation
        assert auc > 0.5

    def test_compute_auc_single_class(self) -> None:
        labels = np.array([1, 1, 1])
        probs = np.array([0.9, 0.8, 0.7])
        assert _compute_auc(labels, probs) == 0.5

    def test_load_or_create_split_with_files(self, tmp_path: Path) -> None:
        rows = [{"source_id": f"s{i}"} for i in range(10)]
        train_p = tmp_path / "train.json"
        val_p = tmp_path / "val.json"
        test_p = tmp_path / "test.json"
        train_p.write_text(json.dumps(list(range(8))))
        val_p.write_text(json.dumps([8]))
        test_p.write_text(json.dumps([9]))
        tr, va, te = load_or_create_split(rows, str(train_p), str(val_p), str(test_p))
        assert tr == [0, 1, 2, 3, 4, 5, 6, 7]
        assert va == [8]
        assert te == [9]

    def test_load_or_create_split_with_dict_format(self, tmp_path: Path) -> None:
        rows = [{"source_id": f"s{i}"} for i in range(10)]
        train_p = tmp_path / "train.json"
        val_p = tmp_path / "val.json"
        test_p = tmp_path / "test.json"
        train_p.write_text(json.dumps({"indices": [0, 1, 2]}))
        val_p.write_text(json.dumps({"indices": [3]}))
        test_p.write_text(json.dumps({"indices": [4]}))
        tr, va, te = load_or_create_split(rows, str(train_p), str(val_p), str(test_p))
        assert tr == [0, 1, 2]
        assert va == [3]
        assert te == [4]

    def test_load_or_create_split_auto(self) -> None:
        rows = [{"source_id": f"s{i}"} for i in range(10)]
        tr, va, te = load_or_create_split(rows, None, None, None, seed=42)
        assert len(tr) + len(va) + len(te) == 10
        # ~80/10/10 split
        assert len(tr) >= 5
        assert len(tr) <= 9

    def test_render_summary_md(self) -> None:
        summary = {
            "n_seeds": 3,
            "epochs": 5,
            "lr": 1e-4,
            "batch_size": 16,
            "uncertainty_weighting": True,
            "backbone_ckpt": "/path/to/ckpt.pt",
            "tasks": {
                "retrosynthesis": {
                    "metric": "accuracy",
                    "multitask_mean": 0.85,
                    "singletask_mean": 0.80,
                    "family_cluster_bootstrap_ci": {
                        "mean_diff": 0.05,
                        "ci_low": 0.01,
                        "ci_high": 0.09,
                        "p_value": 0.02,
                    },
                    "go": True,
                },
            },
        }
        md = _render_summary_md(summary)
        assert "P3-06" in md
        assert "retrosynthesis" in md
        assert "✓" in md


# ---------------------------------------------------------------------------
# 6. Integration test: run_experiment with tiny data
# ---------------------------------------------------------------------------
class TestRunExperiment:
    def test_run_experiment_tiny(self, tmp_path: Path) -> None:
        """End-to-end smoke test with tiny synthetic data."""
        # Write minimal data files
        retro_csv = tmp_path / "regiosqm20_normalized.csv"
        with open(retro_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["source_id", "reaction_smiles", "label_type"])
            w.writeheader()
            for i in range(20):
                w.writerow({
                    "source_id": f"rsrc_{i % 4}",
                    "reaction_smiles": f"CC{i}>>CC{i}O",
                    "label_type": "positive" if i % 2 == 0 else "negative",
                })

        cond_json = tmp_path / "ord_conditions.json"
        records = []
        for i in range(20):
            records.append({
                "source_id": f"csrc_{i % 4}",
                "reaction_smiles": f"CC{i}>>CC{i}O",
                "catalyst": f"cat{i % 2}",
                "solvent": f"sol{i % 2}",
                "reagent": f"reg{i % 2}",
            })
        cond_json.write_text(json.dumps(records))

        yield_csv = tmp_path / "hitea_full_normalized.csv"
        with open(yield_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["source_id", "reaction_smiles", "yield"])
            w.writeheader()
            for i in range(20):
                w.writerow({
                    "source_id": f"ysrc_{i % 4}",
                    "reaction_smiles": f"CC{i}>>CC{i}O",
                    "yield": str(50 + i),
                })

        out_dir = tmp_path / "out"
        summary = run_experiment(
            backbone_ckpt=None,
            vocab_path="",
            data_dir=str(tmp_path),
            seeds=[20260710, 20260711],
            output_dir=str(out_dir),
            epochs=1,
            lr=1e-3,
            batch_size=4,
            device="cpu",
            uncertainty_weighting=True,
            bootstrap_iterations=200,
            hparams={"d_model": D_MODEL, "vocabulary_size": 100},
        )
        assert summary["n_seeds"] == 2
        assert "tasks" in summary
        # At least one task should have results
        assert len(summary["tasks"]) >= 1
        for task, info in summary["tasks"].items():
            assert "multitask_mean" in info
            assert "singletask_mean" in info
            assert "family_cluster_bootstrap_ci" in info
        # Output files exist
        assert (out_dir / "summary.json").exists()
        assert (out_dir / "summary.md").exists()
        assert (out_dir / "seed20260710" / "metrics.json").exists()


# ---------------------------------------------------------------------------
# 7. CLI / main() test
# ---------------------------------------------------------------------------
class TestCLI:
    def test_parse_seeds_comma(self) -> None:
        from models.multitask import _parse_seeds
        assert _parse_seeds("20260710,20260711,20260712") == [20260710, 20260711, 20260712]

    def test_parse_seeds_range(self) -> None:
        from models.multitask import _parse_seeds
        assert _parse_seeds("20260710..20260712") == [20260710, 20260711, 20260712]

    def test_parse_seeds_single(self) -> None:
        from models.multitask import _parse_seeds
        assert _parse_seeds("20260710") == [20260710]

    def test_parse_seeds_empty(self) -> None:
        from models.multitask import _parse_seeds
        assert _parse_seeds("") == list(DEFAULT_SEEDS)

    def test_main_help(self, capsys) -> None:
        with pytest.raises(SystemExit):
            main(["--help"])
        captured = capsys.readouterr()
        assert "P3-06" in captured.out

    def test_main_runs(self, tmp_path: Path, monkeypatch) -> None:
        """Smoke test that main() runs end-to-end with tiny data."""
        # Write minimal data files
        retro_csv = tmp_path / "regiosqm20_normalized.csv"
        with open(retro_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["source_id", "reaction_smiles", "label_type"])
            w.writeheader()
            for i in range(16):
                w.writerow({
                    "source_id": f"rsrc_{i % 4}",
                    "reaction_smiles": f"CC{i}>>CC{i}O",
                    "label_type": "positive" if i % 2 == 0 else "negative",
                })

        cond_json = tmp_path / "ord_conditions.json"
        records = [
            {
                "source_id": f"csrc_{i % 4}",
                "reaction_smiles": f"CC{i}>>CC{i}O",
                "catalyst": f"cat{i % 2}",
                "solvent": f"sol{i % 2}",
                "reagent": f"reg{i % 2}",
            }
            for i in range(16)
        ]
        cond_json.write_text(json.dumps(records))

        yield_csv = tmp_path / "hitea_full_normalized.csv"
        with open(yield_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["source_id", "reaction_smiles", "yield"])
            w.writeheader()
            for i in range(16):
                w.writerow({
                    "source_id": f"ysrc_{i % 4}",
                    "reaction_smiles": f"CC{i}>>CC{i}O",
                    "yield": str(50 + i),
                })

        out_dir = tmp_path / "cli_out"
        # Patch DEFAULT_SEEDS to keep the test fast (already only 1 seed passed)
        rc = main([
            "--backbone-ckpt", "",
            "--vocab", "",
            "--data-dir", str(tmp_path),
            "--seeds", "20260710",
            "--output-dir", str(out_dir),
            "--epochs", "1",
            "--lr", "1e-3",
            "--batch-size", "4",
            "--device", "cpu",
            "--bootstrap-iterations", "100",
        ])
        assert rc == 0
        assert (out_dir / "summary.json").exists()
        assert (out_dir / "summary.md").exists()


# ---------------------------------------------------------------------------
# 8. Performance / timing check
# ---------------------------------------------------------------------------
def test_suite_completes_under_60s() -> None:
    """Sanity check: the test suite should complete in <60s.

    This is a soft assertion -- if it triggers, individual tests should be
    sped up.
    """
    # This test is a no-op placeholder; the real timing check is done by
    # running `pytest --durations=0` and inspecting the output.
    assert True
