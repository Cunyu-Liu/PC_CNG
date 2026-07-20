"""Unit tests for P3-03 cross-dataset fine-tuning head.

Tests run WITHOUT GPU, WITHOUT remote access, and WITHOUT the actual P3-01
checkpoint.  A tiny ``d_model=16`` fallback backbone / scorer / tokenizer
(provided by ``finetune_head._Fallback*``) is used for speed.
"""

from __future__ import annotations

import csv
import json
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

from training.finetune_head import (  # noqa: E402
    VARIANTS,
    _FallbackScorer,
    build_head,
    evaluate,
    family_cluster_bootstrap_ci,
    few_shot_finetune,
    load_dataset,
    load_or_create_split,
    load_pretrained_scorer,
    load_tokenizer,
    main,
    paired_bootstrap_ci,
    run_pair,
    stratified_group_split,
)
from training import finetune_head as fh  # noqa: E402

D_MODEL = 16


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
def tiny_model(device: str) -> torch.nn.Module:
    """Build a tiny fresh-init scorer (no checkpoint)."""
    return load_pretrained_scorer(
        checkpoint_path=None,
        vocab_path="",
        device=device,
        hparams={"d_model": D_MODEL, "vocabulary_size": 100},
    )


def _make_csv(path: Path, n_rows: int = 24, n_sources: int = 6, seed: int = 0) -> Path:
    """Write a tiny normalized reaction CSV."""
    rng = np.random.RandomState(seed)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "reaction_smiles", "label_type"])
        writer.writeheader()
        for i in range(n_rows):
            sid = f"src_{i % n_sources}"
            # Vary the SMILES so the tokenizer produces distinct tokens
            smiles = f"CC{i % 5}>>CC{(i + 1) % 5}O{i % 3}"
            label_type = "positive" if (i % 2 == 0) else "negative"
            writer.writerow(
                {"source_id": sid, "reaction_smiles": smiles, "label_type": label_type}
            )
    return path


@pytest.fixture
def source_csv(tmp_path: Path) -> Path:
    return _make_csv(tmp_path / "source.csv", n_rows=24, n_sources=6, seed=1)


@pytest.fixture
def target_csv(tmp_path: Path) -> Path:
    return _make_csv(tmp_path / "target.csv", n_rows=24, n_sources=6, seed=2)


@pytest.fixture
def tiny_rows() -> List[Dict[str, Any]]:
    """Synthetic target rows for few-shot fine-tuning."""
    return [
        {"smiles": f"CC{i}>>CC{i+1}O", "label": i % 2, "source_id": f"src_{i % 4}"}
        for i in range(16)
    ]


# ---------------------------------------------------------------------------
# 1. build_head tests
# ---------------------------------------------------------------------------
class TestBuildHead:
    def test_returns_sequential(self) -> None:
        head = build_head(d_model=D_MODEL, n_classes=1)
        assert isinstance(head, torch.nn.Sequential)

    def test_architecture(self) -> None:
        head = build_head(d_model=D_MODEL, n_classes=1)
        # Linear -> ReLU -> Linear
        assert isinstance(head[0], torch.nn.Linear)
        assert isinstance(head[1], torch.nn.ReLU)
        assert isinstance(head[2], torch.nn.Linear)
        assert head[0].in_features == D_MODEL
        assert head[0].out_features == 128
        assert head[2].in_features == 128
        assert head[2].out_features == 1

    def test_param_count_under_100k_default(self) -> None:
        head = build_head(d_model=512, n_classes=1)
        n_params = sum(p.numel() for p in head.parameters())
        # 512*128 + 128 + 128*1 + 1 = 65793
        assert n_params <= 100_000, f"head has {n_params} params (>100K)"
        assert 60_000 <= n_params <= 70_000, f"head has {n_params} params (expected ~66K)"

    def test_param_count_tiny(self) -> None:
        head = build_head(d_model=D_MODEL, n_classes=1)
        n_params = sum(p.numel() for p in head.parameters())
        # 16*128 + 128 + 128*1 + 1 = 2305
        assert n_params <= 100_000
        assert n_params == 16 * 128 + 128 + 128 + 1

    def test_forward_shape(self) -> None:
        head = build_head(d_model=D_MODEL, n_classes=1)
        x = torch.randn(4, D_MODEL)
        out = head(x)
        assert out.shape == (4, 1)

    def test_multi_class(self) -> None:
        head = build_head(d_model=D_MODEL, n_classes=3)
        x = torch.randn(4, D_MODEL)
        out = head(x)
        assert out.shape == (4, 3)


# ---------------------------------------------------------------------------
# 2. load_pretrained_scorer tests
# ---------------------------------------------------------------------------
class TestLoadPretrainedScorer:
    def test_no_checkpoint_returns_model(self, device: str) -> None:
        model = load_pretrained_scorer(
            checkpoint_path=None, vocab_path="", device=device,
            hparams={"d_model": D_MODEL, "vocabulary_size": 100},
        )
        assert hasattr(model, "backbone")
        assert hasattr(model, "head")
        # Forward pass works
        ids = torch.tensor([[2, 5, 6, 7, 3]])
        mask = torch.ones_like(ids)
        out = model(ids, attention_mask=mask)
        assert out.shape[0] == 1

    def test_missing_checkpoint_path_returns_model(self, device: str, tmp_path: Path) -> None:
        missing = str(tmp_path / "does_not_exist.pt")
        model = load_pretrained_scorer(
            checkpoint_path=missing, vocab_path="", device=device,
            hparams={"d_model": D_MODEL, "vocabulary_size": 100},
        )
        assert model is not None

    def test_device_placement(self, device: str) -> None:
        model = load_pretrained_scorer(
            checkpoint_path=None, vocab_path="", device=device,
            hparams={"d_model": D_MODEL, "vocabulary_size": 100},
        )
        # All params on CPU
        for p in model.parameters():
            assert p.device.type == "cpu"


# ---------------------------------------------------------------------------
# 3. few_shot_finetune tests (3 variants)
# ---------------------------------------------------------------------------
class TestFewShotFinetune:
    def test_direct_returns_unchanged(self, tiny_model: torch.nn.Module, tiny_rows: List[Dict[str, Any]], tokenizer: Any, device: str) -> None:
        # Snapshot a parameter from the head
        head_params_before = [p.clone() for p in tiny_model.head.parameters()]
        out = few_shot_finetune(
            tiny_model, tiny_rows, "direct", tokenizer,
            n_epochs=2, lr=1e-3, device=device, batch_size=4,
        )
        assert out is tiny_model  # same object
        # Params unchanged
        for before, after in zip(head_params_before, tiny_model.head.parameters()):
            assert torch.allclose(before, after)

    def test_head_finetune_freezes_backbone(
        self, tiny_model: torch.nn.Module, tiny_rows: List[Dict[str, Any]], tokenizer: Any, device: str
    ) -> None:
        # Capture a backbone parameter
        bb_param = next(tiny_model.backbone.parameters())
        bb_before = bb_param.clone()
        head_before = [p.clone() for p in tiny_model.head.parameters()]

        out = few_shot_finetune(
            tiny_model, tiny_rows, "head_finetune", tokenizer,
            n_epochs=2, lr=1e-3, device=device, batch_size=4,
        )
        assert out is tiny_model
        # Backbone frozen
        for p in tiny_model.backbone.parameters():
            assert not p.requires_grad
        # Head trainable
        for p in tiny_model.head.parameters():
            assert p.requires_grad
        # Backbone params unchanged
        assert torch.allclose(bb_before, bb_param)
        # Head params changed
        head_changed = any(
            not torch.allclose(before, after)
            for before, after in zip(head_before, tiny_model.head.parameters())
        )
        assert head_changed, "head params should change during head_finetune"

    def test_full_finetune_unfreezes_all(
        self, tiny_model: torch.nn.Module, tiny_rows: List[Dict[str, Any]], tokenizer: Any, device: str
    ) -> None:
        bb_param = next(tiny_model.backbone.parameters())
        bb_before = bb_param.clone()
        head_before = [p.clone() for p in tiny_model.head.parameters()]

        out = few_shot_finetune(
            tiny_model, tiny_rows, "full_finetune", tokenizer,
            n_epochs=2, lr=1e-3, device=device, batch_size=4,
        )
        assert out is tiny_model
        # All params trainable
        for p in tiny_model.parameters():
            assert p.requires_grad
        # Both backbone and head changed
        assert not torch.allclose(bb_before, bb_param), "backbone params should change in full_finetune"
        head_changed = any(
            not torch.allclose(before, after)
            for before, after in zip(head_before, tiny_model.head.parameters())
        )
        assert head_changed

    def test_unknown_variant_raises(self, tiny_model: torch.nn.Module, tiny_rows: List[Dict[str, Any]], tokenizer: Any, device: str) -> None:
        with pytest.raises(ValueError):
            few_shot_finetune(
                tiny_model, tiny_rows, "bogus", tokenizer,
                n_epochs=1, lr=1e-3, device=device,
            )

    def test_empty_target_rows_returns_unchanged(
        self, tiny_model: torch.nn.Module, tokenizer: Any, device: str
    ) -> None:
        head_before = [p.clone() for p in tiny_model.head.parameters()]
        out = few_shot_finetune(
            tiny_model, [], "head_finetune", tokenizer,
            n_epochs=2, lr=1e-3, device=device,
        )
        assert out is tiny_model
        for before, after in zip(head_before, tiny_model.head.parameters()):
            assert torch.allclose(before, after)


# ---------------------------------------------------------------------------
# 4. evaluate tests
# ---------------------------------------------------------------------------
class TestEvaluate:
    def test_returns_required_keys(
        self, tiny_model: torch.nn.Module, tiny_rows: List[Dict[str, Any]], tokenizer: Any, device: str
    ) -> None:
        metrics = evaluate(tiny_model, tokenizer, tiny_rows, device=device, batch_size=4)
        for key in ("mrr", "accuracy", "auc", "n_examples"):
            assert key in metrics, f"missing key: {key}"
        assert metrics["n_examples"] == len(tiny_rows)

    def test_metrics_ranges(
        self, tiny_model: torch.nn.Module, tiny_rows: List[Dict[str, Any]], tokenizer: Any, device: str
    ) -> None:
        metrics = evaluate(tiny_model, tokenizer, tiny_rows, device=device, batch_size=4)
        assert 0.0 <= metrics["mrr"] <= 1.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["auc"] <= 1.0

    def test_per_example_arrays(
        self, tiny_model: torch.nn.Module, tiny_rows: List[Dict[str, Any]], tokenizer: Any, device: str
    ) -> None:
        metrics = evaluate(tiny_model, tokenizer, tiny_rows, device=device, batch_size=4)
        assert len(metrics["per_example_probs"]) == len(tiny_rows)
        assert len(metrics["per_example_labels"]) == len(tiny_rows)
        assert len(metrics["per_example_source_ids"]) == len(tiny_rows)
        assert len(metrics["per_example_correct"]) == len(tiny_rows)

    def test_empty_rows(
        self, tiny_model: torch.nn.Module, tokenizer: Any, device: str
    ) -> None:
        metrics = evaluate(tiny_model, tokenizer, [], device=device)
        assert metrics["n_examples"] == 0
        assert metrics["mrr"] == 0.0
        assert metrics["accuracy"] == 0.0
        assert metrics["auc"] == 0.5


# ---------------------------------------------------------------------------
# 5. paired_bootstrap_ci tests
# ---------------------------------------------------------------------------
class TestPairedBootstrapCI:
    def test_basic_returns_tuple(self) -> None:
        a = [0.8, 0.85, 0.82, 0.78, 0.83]
        b = [0.7, 0.72, 0.71, 0.69, 0.70]
        result = paired_bootstrap_ci(a, b, n_iterations=200, seed=42)
        assert len(result) == 4
        mean_diff, ci_low, ci_high, p_value = result
        assert mean_diff > 0  # a > b
        assert ci_low <= mean_diff <= ci_high
        assert 0.0 <= p_value <= 1.0

    def test_identical_metrics(self) -> None:
        a = [0.5, 0.6, 0.55, 0.5, 0.6]
        result = paired_bootstrap_ci(a, a, n_iterations=200, seed=42)
        mean_diff, ci_low, ci_high, p_value = result
        assert mean_diff == 0.0
        assert p_value == 1.0

    def test_empty_inputs(self) -> None:
        result = paired_bootstrap_ci([], [], n_iterations=100)
        assert result == (0.0, 0.0, 0.0, 1.0)

    def test_length_mismatch(self) -> None:
        result = paired_bootstrap_ci([0.5, 0.6], [0.5], n_iterations=100)
        assert result == (0.0, 0.0, 0.0, 1.0)

    def test_positive_ci_for_clear_improvement(self) -> None:
        # a clearly better than b
        rng = np.random.RandomState(0)
        a = list(rng.uniform(0.75, 0.85, size=20))
        b = list(rng.uniform(0.55, 0.65, size=20))
        mean_diff, ci_low, ci_high, p_value = paired_bootstrap_ci(a, b, n_iterations=500, seed=42)
        assert ci_low > 0, f"expected CI low > 0, got {ci_low}"
        assert p_value < 0.1


# ---------------------------------------------------------------------------
# 6. family_cluster_bootstrap_ci tests
# ---------------------------------------------------------------------------
class TestFamilyClusterBootstrapCI:
    def test_basic_returns_tuple(self) -> None:
        a = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        b = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        clusters = ["s1", "s1", "s2", "s2", "s3", "s3"]
        result = family_cluster_bootstrap_ci(a, b, clusters, n_iterations=200, seed=42)
        assert len(result) == 4
        mean_diff, ci_low, ci_high, p_value = result
        assert mean_diff > 0  # a > b
        assert ci_low <= mean_diff <= ci_high

    def test_identical_metrics(self) -> None:
        a = [0.0, 1.0, 0.0, 1.0]
        clusters = ["s1", "s1", "s2", "s2"]
        result = family_cluster_bootstrap_ci(a, a, clusters, n_iterations=200, seed=42)
        mean_diff, _, _, p_value = result
        assert mean_diff == 0.0
        assert p_value == 1.0

    def test_empty_inputs(self) -> None:
        result = family_cluster_bootstrap_ci([], [], [], n_iterations=100)
        assert result == (0.0, 0.0, 0.0, 1.0)

    def test_length_mismatch(self) -> None:
        result = family_cluster_bootstrap_ci([1.0, 0.0], [1.0], ["s1", "s2"], n_iterations=100)
        assert result == (0.0, 0.0, 0.0, 1.0)

    def test_single_cluster(self) -> None:
        a = [1.0, 1.0, 0.0]
        b = [0.0, 0.0, 1.0]
        clusters = ["s1", "s1", "s1"]
        result = family_cluster_bootstrap_ci(a, b, clusters, n_iterations=100, seed=42)
        mean_diff, ci_low, ci_high, _ = result
        assert mean_diff > 0
        # With a single cluster, CI should be wide (bootstrap of one cluster)
        assert ci_low <= mean_diff <= ci_high

    def test_clear_improvement(self) -> None:
        # a correct on all, b correct on none
        a = [1.0] * 20
        b = [0.0] * 20
        clusters = [f"s{i // 2}" for i in range(20)]  # 10 clusters of 2
        mean_diff, ci_low, ci_high, p_value = family_cluster_bootstrap_ci(
            a, b, clusters, n_iterations=300, seed=42
        )
        assert mean_diff == 1.0
        assert ci_low > 0
        assert p_value < 0.1


# ---------------------------------------------------------------------------
# 7. Data loading helpers
# ---------------------------------------------------------------------------
class TestDataHelpers:
    def test_load_dataset(self, source_csv: Path) -> None:
        rows = load_dataset(str(source_csv))
        assert len(rows) == 24
        assert all("smiles" in r and "label" in r and "source_id" in r for r in rows)
        # label_type='positive' (i%2==0) -> label=1
        assert any(r["label"] == 1 for r in rows)
        assert any(r["label"] == 0 for r in rows)

    def test_load_dataset_missing_file(self, tmp_path: Path) -> None:
        rows = load_dataset(str(tmp_path / "nope.csv"))
        assert rows == []

    def test_load_or_create_split_with_files(self, tmp_path: Path) -> None:
        rows = [{"source_id": f"s{i}"} for i in range(10)]
        train_path = tmp_path / "train.json"
        val_path = tmp_path / "val.json"
        test_path = tmp_path / "test.json"
        train_path.write_text(json.dumps([0, 1, 2, 3, 4, 5, 6, 7]))
        val_path.write_text(json.dumps([8]))
        test_path.write_text(json.dumps([9]))
        tr, va, te = load_or_create_split(rows, str(train_path), str(val_path), str(test_path))
        assert tr == [0, 1, 2, 3, 4, 5, 6, 7]
        assert va == [8]
        assert te == [9]

    def test_load_or_create_split_auto(self) -> None:
        rows = [{"source_id": f"s{i}"} for i in range(10)]
        tr, va, te = load_or_create_split(rows, None, None, None, seed=42)
        assert len(tr) + len(va) + len(te) == 10
        # Stratified by source_id: no overlap
        tr_sids = {rows[i]["source_id"] for i in tr}
        va_sids = {rows[i]["source_id"] for i in va}
        te_sids = {rows[i]["source_id"] for i in te}
        assert tr_sids.isdisjoint(va_sids)
        assert tr_sids.isdisjoint(te_sids)
        assert va_sids.isdisjoint(te_sids)

    def test_stratified_group_split(self) -> None:
        rows = [{"source_id": f"s{i}"} for i in range(10)]
        tr, te = stratified_group_split(rows, n_few_shot=0.1, seed=42)
        # 1 group (10%) for train, 9 groups for test
        tr_sids = {rows[i]["source_id"] for i in tr}
        te_sids = {rows[i]["source_id"] for i in te}
        assert len(tr_sids) == 1
        assert len(te_sids) == 9
        assert tr_sids.isdisjoint(te_sids)

    def test_stratified_group_split_no_train_fallback(self) -> None:
        rows = [{"source_id": f"s{i}"} for i in range(5)]
        # n_few_shot=0 would give 0 groups, but fallback ensures >=1 train
        tr, te = stratified_group_split(rows, n_few_shot=0.0, seed=42)
        assert len(tr) >= 1


# ---------------------------------------------------------------------------
# 8. run_pair tests (with fallback scorer / tiny d_model)
# ---------------------------------------------------------------------------
class TestRunPair:
    def test_run_pair_smoke(
        self,
        source_csv: Path,
        target_csv: Path,
        tmp_path: Path,
        device: str,
    ) -> None:
        """Smoke test: run_pair with fallback scorer, 2 seeds, 1 epoch."""
        start = time.time()
        output_dir = tmp_path / "out"
        summary = run_pair(
            source_name="uspto",
            target_name="ord",
            source_csv=str(source_csv),
            target_csv=str(target_csv),
            backbone_ckpt=None,  # fresh init
            vocab_path="",
            seeds=[20260710, 20260711],
            output_dir=str(output_dir),
            n_few_shot=0.2,
            epochs=1,
            lr=1e-3,
            device=device,
            train_idx_path=None,
            val_idx_path=None,
            test_idx_path=None,
            bootstrap_iterations=100,
        )
        elapsed = time.time() - start
        assert elapsed < 120.0, f"run_pair took {elapsed:.1f}s (>120s)"

        # Check summary structure
        assert summary["pair"] == "uspto_to_ord"
        assert summary["source"] == "uspto"
        assert summary["target"] == "ord"
        assert summary["n_seeds"] == 2
        assert summary["checkpoint_used"] is False
        for v in VARIANTS:
            assert v in summary["variants"]
            assert "mrr_mean" in summary["variants"][v]
            assert "mrr_per_seed" in summary["variants"][v]
            assert len(summary["variants"][v]["mrr_per_seed"]) == 2
        assert "paired_bootstrap_ci" in summary
        assert "head_finetune_vs_direct" in summary["paired_bootstrap_ci"]
        assert "full_finetune_vs_direct" in summary["paired_bootstrap_ci"]
        assert "go_no_go" in summary

        # Check files were written
        pair_dir = output_dir / "uspto_to_ord"
        assert (pair_dir / "summary.json").exists()
        assert (pair_dir / "summary.md").exists()
        # Per-seed metrics files
        for seed in [20260710, 20260711]:
            assert (pair_dir / f"seed{seed}" / "metrics.json").exists()

    def test_run_pair_summary_md_has_go(
        self,
        source_csv: Path,
        target_csv: Path,
        tmp_path: Path,
        device: str,
    ) -> None:
        output_dir = tmp_path / "out2"
        run_pair(
            source_name="hitea",
            target_name="uspto",
            source_csv=str(source_csv),
            target_csv=str(target_csv),
            backbone_ckpt=None,
            vocab_path="",
            seeds=[20260710],
            output_dir=str(output_dir),
            n_few_shot=0.3,
            epochs=1,
            lr=1e-3,
            device=device,
            bootstrap_iterations=50,
        )
        md = (output_dir / "hitea_to_uspto" / "summary.md").read_text(encoding="utf-8")
        assert "hitea_to_uspto" in md
        assert "head_finetune" in md
        assert "full_finetune" in md
        assert "GO" in md


# ---------------------------------------------------------------------------
# 9. CLI / parsing tests
# ---------------------------------------------------------------------------
class TestCLI:
    def test_parse_pairs_all(self) -> None:
        pairs = fh.parse_pairs("all")
        assert pairs == fh.MIGRATION_PAIRS
        assert len(pairs) == 7

    def test_parse_pairs_explicit(self) -> None:
        pairs = fh.parse_pairs("uspto->ord,ord->uspto")
        assert pairs == [("uspto", "ord"), ("ord", "uspto")]

    def test_parse_pairs_invalid(self) -> None:
        with pytest.raises(ValueError):
            fh.parse_pairs("uspto:ord")

    def test_parse_seeds_comma(self) -> None:
        seeds = fh.parse_seeds("20260710,20260711,20260712")
        assert seeds == [20260710, 20260711, 20260712]

    def test_parse_seeds_range(self) -> None:
        seeds = fh.parse_seeds("20260710..20260712")
        assert seeds == [20260710, 20260711, 20260712]

    def test_parse_seeds_single(self) -> None:
        seeds = fh.parse_seeds("20260710")
        assert seeds == [20260710]

    def test_main_skip_missing_csv(self, tmp_path: Path, monkeypatch) -> None:
        """main() should skip pairs whose CSVs are missing."""
        out_dir = tmp_path / "cli_out"
        rc = main(
            [
                "--pairs", "uspto->ord",
                "--data-dir", str(tmp_path / "nonexistent"),
                "--output-dir", str(out_dir),
                "--seeds", "20260710",
                "--device", "cpu",
                "--epochs", "1",
                "--bootstrap-iterations", "50",
            ]
        )
        assert rc == 0
        # No pair directories created (CSVs missing)
        assert not (out_dir / "uspto_to_ord").exists()

    def test_main_end_to_end(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """main() end-to-end with synthetic CSVs."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # Create CSVs matching DATASET_CSV_MAP
        _make_csv(data_dir / "uspto_openmolecules_normalized.csv", n_rows=16, n_sources=4, seed=10)
        _make_csv(data_dir / "ord_normalized.csv", n_rows=16, n_sources=4, seed=11)

        out_dir = tmp_path / "cli_out2"
        rc = main(
            [
                "--pairs", "uspto->ord",
                "--data-dir", str(data_dir),
                "--output-dir", str(out_dir),
                "--seeds", "20260710",
                "--device", "cpu",
                "--epochs", "1",
                "--n-few-shot", "0.25",
                "--bootstrap-iterations", "50",
                "--backbone-ckpt", "",
                "--vocab", "",
            ]
        )
        assert rc == 0
        assert (out_dir / "uspto_to_ord" / "summary.json").exists()
        assert (out_dir / "all_pairs_summary.json").exists()


# ---------------------------------------------------------------------------
# 10. Fallback classes sanity
# ---------------------------------------------------------------------------
class TestFallbackClasses:
    def test_fallback_backbone_forward(self) -> None:
        bb = fh._FallbackBackbone(d_model=D_MODEL, vocab_size=100)
        ids = torch.tensor([[2, 5, 6, 7, 3], [2, 5, 3, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        out = bb(ids, attention_mask=mask, pool=True)
        assert out.shape == (2, D_MODEL)

    def test_fallback_head_forward(self) -> None:
        head = fh._FallbackHead(d_model=D_MODEL)
        x = torch.randn(4, D_MODEL)
        out = head(x)
        assert out.shape == (4,)  # squeezed

    def test_fallback_scorer_forward(self) -> None:
        scorer = _FallbackScorer(d_model=D_MODEL, vocab_size=100)
        ids = torch.tensor([[2, 5, 6, 7, 3]])
        mask = torch.ones_like(ids)
        out = scorer(ids, attention_mask=mask)
        assert out.shape == (1,)

    def test_fallback_tokenizer_batch_encode(self) -> None:
        tok = fh._FallbackTokenizer(max_seq_len=16)
        smiles_list = ["CCO", "CC>>O", "CCN"]
        ids, mask = tok.batch_encode(smiles_list)
        assert ids.shape[0] == 3
        assert mask.shape[0] == 3
        assert ids.shape[1] == mask.shape[1]
        # Padding positions have pad_idx (0)
        assert (ids[mask == 0] == 0).all()

    def test_fallback_tokenizer_empty(self) -> None:
        tok = fh._FallbackTokenizer(max_seq_len=16)
        ids, mask = tok.batch_encode([])
        assert ids.shape[0] == 0


# ---------------------------------------------------------------------------
# 11. Metric helpers
# ---------------------------------------------------------------------------
class TestMetricHelpers:
    def test_compute_mrr(self) -> None:
        probs = np.array([0.9, 0.8, 0.7, 0.6])
        labels = np.array([0, 1, 0, 0])
        sids = ["s1", "s1", "s1", "s1"]
        mrr = fh._compute_mrr(probs, labels, sids)
        # positive is at rank 2 -> RR=0.5
        assert mrr == 0.5

    def test_compute_mrr_no_positive(self) -> None:
        probs = np.array([0.9, 0.8])
        labels = np.array([0, 0])
        sids = ["s1", "s1"]
        mrr = fh._compute_mrr(probs, labels, sids)
        assert mrr == 0.0

    def test_compute_auc(self) -> None:
        labels = np.array([1, 1, 0, 0])
        probs = np.array([0.9, 0.8, 0.3, 0.2])
        auc = fh._compute_auc(labels, probs)
        # perfect separation
        assert auc == 1.0

    def test_compute_auc_random(self) -> None:
        labels = np.array([1, 0, 1, 0])
        probs = np.array([0.4, 0.6, 0.5, 0.3])
        auc = fh._compute_auc(labels, probs)
        assert 0.0 <= auc <= 1.0

    def test_compute_auc_single_class(self) -> None:
        labels = np.array([1, 1, 1])
        probs = np.array([0.9, 0.5, 0.1])
        auc = fh._compute_auc(labels, probs)
        assert auc == 0.5


# ---------------------------------------------------------------------------
# 12. render summary md
# ---------------------------------------------------------------------------
class TestRenderSummaryMd:
    def test_render_summary_md_basic(self) -> None:
        summary = {
            "pair": "uspto_to_ord",
            "source": "uspto",
            "target": "ord",
            "n_seeds": 2,
            "n_few_shot": 0.1,
            "checkpoint_used": False,
            "variants": {
                v: {"mrr_mean": 0.5, "mrr_std": 0.1, "mrr_per_seed": [0.5, 0.5]}
                for v in VARIANTS
            },
            "paired_bootstrap_ci": {
                "head_finetune_vs_direct": {"mean_diff": 0.1, "ci_low": -0.05, "ci_high": 0.25, "p_value": 0.2},
                "full_finetune_vs_direct": {"mean_diff": 0.05, "ci_low": 0.01, "ci_high": 0.1, "p_value": 0.05},
            },
            "family_cluster_bootstrap_ci": {
                "head_finetune_vs_direct": {"mean_diff": 0.08, "ci_low": 0.0, "ci_high": 0.16, "p_value": 0.1},
                "full_finetune_vs_direct": {"mean_diff": 0.04, "ci_low": -0.02, "ci_high": 0.1, "p_value": 0.3},
            },
            "go_no_go": {"head_finetune_go": False, "full_finetune_go": True},
        }
        md = fh._render_summary_md(summary)
        assert "uspto_to_ord" in md
        assert "head_finetune" in md
        assert "full_finetune" in md
        assert "GO" in md
        assert "YES" in md or "NO" in md
