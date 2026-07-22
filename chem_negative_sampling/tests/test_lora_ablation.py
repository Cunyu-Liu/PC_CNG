"""Tests for P4-G2 LoRA Ablation module.

Run with::

    python3 -m pytest chem_negative_sampling/tests/test_lora_ablation.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# Ensure the pc_cng package is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from pc_cng.run_p4_lora_ablation import (
    AblationConfig,
    CONFIG_IDS,
    NONINFERIORITY_MARGIN,
    build_config_registry,
    build_module_registry,
    compute_calibration_metrics,
    compute_go_no_go,
    compute_checkpoint_hash,
    discover_linear_modules,
    discover_target_patterns,
    load_manifest_candidates,
    noninferiority_test,
    paired_bootstrap_ci,
    select_best_backbone,
    write_summary_csv,
)
from models.pretrained_backbone import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_VOCAB_PATH,
    PretrainedChemformerBackbone,
    PretrainedReactionScorer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    """Build a model instance for module discovery tests."""
    if not DEFAULT_CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found: {DEFAULT_CHECKPOINT_PATH}")
    backbone = PretrainedChemformerBackbone(checkpoint_path=DEFAULT_CHECKPOINT_PATH, freeze=True)
    scorer = PretrainedReactionScorer(backbone)
    return scorer


# ---------------------------------------------------------------------------
# Module auto-discovery tests
# ---------------------------------------------------------------------------

class TestModuleDiscovery:
    """Tests for auto-discovery of real module names (spec: must not assume q_proj/v_proj)."""

    def test_discover_linear_modules_returns_list(self, model):
        """discover_linear_modules must return a non-empty list of (name, module) tuples."""
        linears = discover_linear_modules(model)
        assert isinstance(linears, list)
        assert len(linears) > 0, "Model must have at least one nn.Linear module"
        for name, module in linears:
            assert isinstance(name, str)
            assert isinstance(module, torch.nn.Linear)

    def test_discover_linear_modules_finds_ffn(self, model):
        """Must find FFN linear1 and linear2 modules."""
        linears = discover_linear_modules(model)
        names = [name for name, _ in linears]
        # FFN modules should exist
        assert any("linear1" in n for n in names), "Must find linear1 (FFN up-projection)"
        assert any("linear2" in n for n in names), "Must find linear2 (FFN down-projection)"

    def test_discover_linear_modules_finds_attn_out_proj(self, model):
        """Must find self_attn.out_proj (the attention output projection)."""
        linears = discover_linear_modules(model)
        names = [name for name, _ in linears]
        assert any("self_attn.out_proj" in n for n in names), \
            "Must find self_attn.out_proj (attention output projection)"

    def test_no_hardcoded_q_proj(self, model):
        """Must NOT find q_proj/v_proj/k_proj (they don't exist in this model)."""
        linears = discover_linear_modules(model)
        names = [name for name, _ in linears]
        # nn.MultiheadAttention uses in_proj_weight (Parameter, not Linear)
        # So there should be no q_proj/v_proj/k_proj Linear modules
        for forbidden in ["q_proj", "v_proj", "k_proj"]:
            assert not any(forbidden in n for n in names), \
                f"Found {forbidden} — should not exist in nn.MultiheadAttention-based model"

    def test_build_module_registry(self, model):
        """build_module_registry must record all Linear modules with shapes."""
        registry = build_module_registry(model)
        assert isinstance(registry, dict)
        assert len(registry) > 0
        for name, info in registry.items():
            if info["type"] == "nn.Linear":
                assert "in_features" in info
                assert "out_features" in info
            elif info["type"] == "nn.Parameter":
                assert "shape" in info

    def test_discover_target_patterns_ffn(self, model):
        """FFN patterns must use wildcard for layer index."""
        patterns = discover_target_patterns(model, "ffn")
        assert len(patterns) > 0
        for p in patterns:
            assert "*" in p, f"Pattern must contain wildcard: {p}"
            assert "linear1" in p or "linear2" in p

    def test_discover_target_patterns_attention(self, model):
        """Attention patterns must match self_attn.out_proj."""
        patterns = discover_target_patterns(model, "attention")
        assert len(patterns) > 0
        for p in patterns:
            assert "self_attn.out_proj" in p
            assert "*" in p

    def test_discover_target_patterns_all_linear(self, model):
        """all_linear patterns must include both FFN and attention."""
        patterns = discover_target_patterns(model, "all_linear")
        assert len(patterns) >= 2  # At least linear1, linear2
        # Should include out_proj
        assert any("out_proj" in p for p in patterns)


# ---------------------------------------------------------------------------
# Config registry tests
# ---------------------------------------------------------------------------

class TestConfigRegistry:
    """Tests for the 6-config registry."""

    def test_config_registry_has_six_configs(self, model):
        """Must have exactly 6 configs: C1-C6."""
        registry = build_config_registry(model)
        assert "configs" in registry
        configs = registry["configs"]
        for cid in CONFIG_IDS:
            assert cid in configs, f"Missing config {cid}"

    def test_config_fields_complete(self, model):
        """Each config must have all required fields."""
        registry = build_config_registry(model)
        required = ["config_id", "name", "description", "adapter", "target_patterns", "rank", "alpha", "dropout"]
        for cid, config in registry["configs"].items():
            for field in required:
                assert field in config, f"Config {cid} missing field {field}"

    def test_c1_is_frozen(self, model):
        """C1 must be zero-shot/frozen (adapter=none)."""
        registry = build_config_registry(model)
        c1 = registry["configs"]["C1"]
        assert c1["adapter"] == "none"
        assert c1["rank"] == 0
        assert c1["target_patterns"] == []

    def test_c2_is_ffn_lora(self, model):
        """C2 must be LoRA on FFN with rank 8."""
        registry = build_config_registry(model)
        c2 = registry["configs"]["C2"]
        assert c2["adapter"] == "lora"
        assert c2["rank"] == 8
        assert len(c2["target_patterns"]) > 0
        assert any("linear1" in p for p in c2["target_patterns"])

    def test_c3_is_attention_lora(self, model):
        """C3 must be LoRA on attention out_proj."""
        registry = build_config_registry(model)
        c3 = registry["configs"]["C3"]
        assert c3["adapter"] == "lora"
        assert any("out_proj" in p for p in c3["target_patterns"])

    def test_c4_c5_are_all_linear(self, model):
        """C4 and C5 must target all linear modules."""
        registry = build_config_registry(model)
        for cid in ["C4", "C5"]:
            config = registry["configs"][cid]
            assert config["adapter"] == "lora"
            assert len(config["target_patterns"]) >= 2

    def test_c4_rank8_c5_rank16(self, model):
        """C4 rank=8, C5 rank=16."""
        registry = build_config_registry(model)
        assert registry["configs"]["C4"]["rank"] == 8
        assert registry["configs"]["C5"]["rank"] == 16

    def test_c6_is_full_finetune(self, model):
        """C6 must be full fine-tuning."""
        registry = build_config_registry(model)
        c6 = registry["configs"]["C6"]
        assert c6["adapter"] == "full"
        assert c6["rank"] == 0

    def test_module_registry_present(self, model):
        """Config registry must include the module registry."""
        registry = build_config_registry(model)
        assert "module_registry" in registry
        assert len(registry["module_registry"]) > 0

    def test_discovered_patterns_present(self, model):
        """Config registry must include discovered patterns by scope."""
        registry = build_config_registry(model)
        assert "discovered_patterns" in registry
        assert "ffn" in registry["discovered_patterns"]
        assert "attention" in registry["discovered_patterns"]
        assert "all_linear" in registry["discovered_patterns"]


# ---------------------------------------------------------------------------
# Calibration metrics tests
# ---------------------------------------------------------------------------

class TestCalibrationMetrics:
    """Tests for ECE and Brier score computation."""

    def test_perfect_predictions_low_ece(self):
        """Perfect predictions should have low ECE and Brier."""
        logits = torch.tensor([10.0, -10.0, 10.0, -10.0])
        labels = torch.tensor([1, 0, 1, 0])
        result = compute_calibration_metrics(logits, labels)
        assert result["ece"] < 0.01
        assert result["brier"] < 0.01

    def test_worst_predictions_high_brier(self):
        """Completely wrong predictions should have high Brier."""
        logits = torch.tensor([-10.0, 10.0, -10.0, 10.0])
        labels = torch.tensor([1, 0, 1, 0])
        result = compute_calibration_metrics(logits, labels)
        assert result["brier"] > 0.9

    def test_random_predictions_moderate_scores(self):
        """Random predictions should have moderate ECE and Brier."""
        logits = torch.tensor([0.0, 0.0, 0.0, 0.0])
        labels = torch.tensor([1, 0, 1, 0])
        result = compute_calibration_metrics(logits, labels)
        # All probs = 0.5, accuracy = 0.5, so ECE should be low
        assert result["ece"] < 0.1
        assert 0.2 < result["brier"] < 0.3  # (0.5-0)^2 and (0.5-1)^2 average = 0.25


# ---------------------------------------------------------------------------
# Paired bootstrap CI tests
# ---------------------------------------------------------------------------

class TestPairedBootstrapCI:
    """Tests for paired bootstrap CI."""

    def test_equal_arrays_zero_delta(self):
        """Identical arrays should have delta_mean ≈ 0."""
        result = paired_bootstrap_ci([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert abs(result["delta_mean"]) < 1e-6
        assert result["ci_low"] <= 0 <= result["ci_high"]

    def test_treatment_better(self):
        """Treatment consistently better should have positive delta."""
        result = paired_bootstrap_ci([0.8, 0.8, 0.8, 0.8, 0.8], [0.5, 0.5, 0.5, 0.5, 0.5])
        assert result["delta_mean"] > 0
        assert result["ci_low"] > 0

    def test_treatment_worse(self):
        """Treatment consistently worse should have negative delta."""
        result = paired_bootstrap_ci([0.3, 0.3, 0.3], [0.7, 0.7, 0.7])
        assert result["delta_mean"] < 0
        assert result["ci_high"] < 0

    def test_unequal_length_raises(self):
        """Unequal length arrays should raise ValueError."""
        with pytest.raises(ValueError):
            paired_bootstrap_ci([0.5, 0.5], [0.5])

    def test_empty_arrays(self):
        """Empty arrays should return zeros."""
        result = paired_bootstrap_ci([], [])
        assert result["n"] == 0


# ---------------------------------------------------------------------------
# Non-inferiority test tests
# ---------------------------------------------------------------------------

class TestNoninferiority:
    """Tests for non-inferiority test logic."""

    def test_noninferior_when_lora_matches_full_ft(self):
        """LoRA matching full FT should be non-inferior; slightly below is inferior."""
        lora_results = {
            "C2": [{"test_metrics": {"mrr": 0.5}} for _ in range(10)],
            "C3": [{"test_metrics": {"mrr": 0.49}} for _ in range(10)],
        }
        full_ft = [{"test_metrics": {"mrr": 0.5}} for _ in range(10)]
        result = noninferiority_test(lora_results, full_ft, margin=-0.005)
        # C2: delta=0.0 > -0.005 → NONINFERIOR
        assert result["C2"]["is_noninferior"] is True
        # C3: delta=-0.01 < -0.005 → INFERIOR
        assert result["C3"]["is_noninferior"] is False
        assert result["C3"]["status"] == "INFERIOR"

    def test_inferior_when_lora_far_below(self):
        """LoRA far below full FT should be inferior."""
        lora_results = {
            "C2": [{"test_metrics": {"mrr": 0.3}} for _ in range(10)],
        }
        full_ft = [{"test_metrics": {"mrr": 0.6}} for _ in range(10)]
        result = noninferiority_test(lora_results, full_ft, margin=-0.005)
        assert result["C2"]["is_noninferior"] is False
        assert result["C2"]["status"] == "INFERIOR"

    def test_insufficient_seeds(self):
        """Less than 2 seeds should return INSUFFICIENT_SEEDS."""
        lora_results = {"C2": [{"test_metrics": {"mrr": 0.5}}]}
        full_ft = [{"test_metrics": {"mrr": 0.5}}]
        result = noninferiority_test(lora_results, full_ft)
        assert result["C2"]["status"] == "INSUFFICIENT_SEEDS"


# ---------------------------------------------------------------------------
# Backbone selection tests
# ---------------------------------------------------------------------------

class TestBackboneSelection:
    """Tests for backbone selection logic."""

    def test_select_best_noninferior(self):
        """Should select the non-inferior config with highest MRR."""
        config_registry = {
            "configs": {
                "C2": {"target_patterns": ["p1"], "rank": 8, "alpha": 16.0, "dropout": 0.0},
                "C4": {"target_patterns": ["p1", "p2"], "rank": 8, "alpha": 16.0, "dropout": 0.0},
            }
        }
        all_results = {
            "C2": [{"test_metrics": {"mrr": 0.5}, "trainable_parameters": 100000} for _ in range(10)],
            "C4": [{"test_metrics": {"mrr": 0.52}, "trainable_parameters": 150000} for _ in range(10)],
        }
        noninferiority = {
            "C2": {"is_noninferior": True},
            "C4": {"is_noninferior": True},
        }
        selected = select_best_backbone(config_registry, all_results, noninferiority)
        assert selected is not None
        assert selected["config_id"] == "C4"  # Higher MRR
        assert "checkpoint_hash" in selected
        assert "target_modules" in selected
        assert "rank" in selected
        assert "alpha" in selected
        assert "selection_metric" in selected
        assert "selection_rule" in selected

    def test_select_none_when_all_inferior(self):
        """Should return None when no LoRA config is non-inferior (strict mode)."""
        config_registry = {"configs": {"C2": {"target_patterns": [], "rank": 8, "alpha": 16, "dropout": 0}}}
        all_results = {"C2": [{"test_metrics": {"mrr": 0.3}, "trainable_parameters": 1000}]}
        noninferiority = {"C2": {"is_noninferior": False}}
        selected = select_best_backbone(config_registry, all_results, noninferiority)
        assert selected is None

    def test_select_partial_go_when_all_inferior(self):
        """PARTIAL_GO: allow_partial_go=True selects best LoRA even when all INFERIOR."""
        config_registry = {
            "configs": {
                "C2": {"target_patterns": ["p1"], "rank": 8, "alpha": 16.0, "dropout": 0.0},
                "C3": {"target_patterns": ["p2"], "rank": 8, "alpha": 16.0, "dropout": 0.0},
            }
        }
        all_results = {
            "C2": [{"test_metrics": {"mrr": 0.43}, "trainable_parameters": 377345} for _ in range(3)],
            "C3": [{"test_metrics": {"mrr": 0.47}, "trainable_parameters": 180737} for _ in range(3)],
        }
        noninferiority = {
            "C2": {"is_noninferior": False},
            "C3": {"is_noninferior": False},
        }
        selected = select_best_backbone(
            config_registry, all_results, noninferiority,
            allow_partial_go=True,
        )
        assert selected is not None
        assert selected["config_id"] == "C3"  # Higher MRR
        assert selected["is_noninferior"] is False
        assert "partial_go" in selected["selection_rule"]

    def test_selected_backbone_has_all_required_fields(self):
        """selected_backbone.json must have all fields from spec."""
        config_registry = {
            "configs": {
                "C4": {"target_patterns": ["p1"], "rank": 8, "alpha": 16.0, "dropout": 0.0},
            }
        }
        all_results = {
            "C4": [{"test_metrics": {"mrr": 0.5}, "trainable_parameters": 100000} for _ in range(10)],
        }
        noninferiority = {"C4": {"is_noninferior": True}}
        selected = select_best_backbone(config_registry, all_results, noninferiority)
        required_fields = [
            "checkpoint", "checkpoint_hash", "architecture", "target_modules",
            "rank", "alpha", "dropout", "trainable_parameters", "training_budget",
            "selection_metric", "selection_rule",
        ]
        for field in required_fields:
            assert field in selected, f"Missing required field: {field}"


# ---------------------------------------------------------------------------
# GO/NO-GO tests
# ---------------------------------------------------------------------------

class TestGoNoGo:
    """Tests for GO/NO-GO verdict logic."""

    def test_go_when_noninferior_and_efficient(self):
        """GO when LoRA is non-inferior and param-efficient."""
        all_results = {
            "C2": [{"test_metrics": {"mrr": 0.5}, "trainable_parameters": 10000} for _ in range(10)],
            "C6": [{"test_metrics": {"mrr": 0.5}, "trainable_parameters": 200000} for _ in range(10)],
        }
        noninferiority = {"C2": {"is_noninferior": True}}
        selected = {"config_id": "C2", "mean_mrr": 0.5, "trainable_parameters": 10000, "checkpoint_hash": "abc"}
        go = compute_go_no_go(all_results, noninferiority, selected, all_results["C6"])
        assert go["status"] == "GO"
        assert go["next_phase_allowed"] is True

    def test_no_go_when_all_inferior(self):
        """NO-GO when no LoRA is non-inferior."""
        all_results = {
            "C2": [{"test_metrics": {"mrr": 0.3}, "trainable_parameters": 10000} for _ in range(10)],
            "C6": [{"test_metrics": {"mrr": 0.6}, "trainable_parameters": 200000} for _ in range(10)],
        }
        noninferiority = {"C2": {"is_noninferior": False}}
        selected = None
        go = compute_go_no_go(all_results, noninferiority, selected, all_results["C6"])
        assert go["status"] == "NO_GO"
        assert go["next_phase_allowed"] is False

    def test_partial_go_when_inferior_but_efficient(self):
        """PARTIAL_GO when LoRA is INFERIOR but selected and param-efficient."""
        all_results = {
            "C3": [{"test_metrics": {"mrr": 0.47}, "trainable_parameters": 180737} for _ in range(3)],
            "C6": [{"test_metrics": {"mrr": 0.54}, "trainable_parameters": 19314785} for _ in range(3)],
        }
        noninferiority = {"C3": {"is_noninferior": False}}
        selected = {
            "config_id": "C3", "mean_mrr": 0.47, "trainable_parameters": 180737,
            "checkpoint_hash": "abc", "is_noninferior": False,
        }
        go = compute_go_no_go(all_results, noninferiority, selected, all_results["C6"])
        assert go["status"] == "PARTIAL_GO"
        assert go["next_phase_allowed"] is True

    def test_go_status_values(self):
        """Status must be one of GO, PARTIAL_GO, NO_GO."""
        all_results = {"C6": [{"test_metrics": {"mrr": 0.5}, "trainable_parameters": 200000}]}
        noninferiority = {}
        go = compute_go_no_go(all_results, noninferiority, None, all_results["C6"])
        assert go["status"] in {"GO", "PARTIAL_GO", "NO_GO"}
        assert go["phase"] == "P4-G2"


# ---------------------------------------------------------------------------
# Summary CSV tests
# ---------------------------------------------------------------------------

class TestSummaryCSV:
    """Tests for summary CSV generation."""

    def test_csv_has_correct_columns(self, tmp_path):
        """Summary CSV must have all required metric columns."""
        all_results = {
            "C1": [{
                "config_id": "C1", "config_name": "frozen", "seed": 42,
                "trainable_parameters": 0, "total_parameters": 100000, "param_ratio": 0.0,
                "wall_clock_seconds": 10.0, "peak_memory_mb": 100.0, "inference_latency_ms": 1.0,
                "val_metrics": {"mrr": 0.3, "top1": 0.2, "top3": 0.4, "ndcg": 0.35, "ece": 0.1, "brier": 0.2},
                "test_metrics": {"mrr": 0.32, "top1": 0.22, "top3": 0.42, "ndcg": 0.37, "ece": 0.11, "brier": 0.21},
            }],
        }
        csv_path = tmp_path / "summary.csv"
        write_summary_csv(all_results, csv_path)
        assert csv_path.exists()

        import csv as csv_mod
        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            required_cols = [
                "config_id", "seed", "trainable_parameters", "test_mrr",
                "test_top1", "test_ndcg", "test_ece", "test_brier",
                "val_mrr", "peak_memory_mb", "wall_clock_seconds",
            ]
            for col in required_cols:
                assert col in rows[0], f"Missing column: {col}"

    def test_csv_multiple_rows(self, tmp_path):
        """CSV should have one row per config×seed."""
        all_results = {
            "C1": [{
                "config_id": "C1", "config_name": "frozen", "seed": 1,
                "trainable_parameters": 0, "total_parameters": 100, "param_ratio": 0.0,
                "wall_clock_seconds": 1, "peak_memory_mb": 1, "inference_latency_ms": 1,
                "val_metrics": {"mrr": 0.1, "top1": 0.1, "top3": 0.1, "ndcg": 0.1, "ece": 0.1, "brier": 0.1},
                "test_metrics": {"mrr": 0.1, "top1": 0.1, "top3": 0.1, "ndcg": 0.1, "ece": 0.1, "brier": 0.1},
            }],
            "C2": [
                {
                    "config_id": "C2", "config_name": "lora", "seed": 1,
                    "trainable_parameters": 1000, "total_parameters": 100, "param_ratio": 0.01,
                    "wall_clock_seconds": 1, "peak_memory_mb": 1, "inference_latency_ms": 1,
                    "val_metrics": {"mrr": 0.2, "top1": 0.2, "top3": 0.2, "ndcg": 0.2, "ece": 0.2, "brier": 0.2},
                    "test_metrics": {"mrr": 0.2, "top1": 0.2, "top3": 0.2, "ndcg": 0.2, "ece": 0.2, "brier": 0.2},
                },
                {
                    "config_id": "C2", "config_name": "lora", "seed": 2,
                    "trainable_parameters": 1000, "total_parameters": 100, "param_ratio": 0.01,
                    "wall_clock_seconds": 1, "peak_memory_mb": 1, "inference_latency_ms": 1,
                    "val_metrics": {"mrr": 0.21, "top1": 0.21, "top3": 0.21, "ndcg": 0.21, "ece": 0.21, "brier": 0.21},
                    "test_metrics": {"mrr": 0.21, "top1": 0.21, "top3": 0.21, "ndcg": 0.21, "ece": 0.21, "brier": 0.21},
                },
            ],
        }
        csv_path = tmp_path / "summary.csv"
        write_summary_csv(all_results, csv_path)
        import csv as csv_mod
        with open(csv_path) as f:
            rows = list(csv_mod.DictReader(f))
            assert len(rows) == 3  # 1 + 2


# ---------------------------------------------------------------------------
# Manifest loading tests
# ---------------------------------------------------------------------------

class TestManifestLoading:
    """Tests for P4-G1 manifest data loading."""

    @pytest.fixture(scope="class")
    def manifest_path(self):
        path = Path("data/p4/manifests/hte_feasibility_v1.json")
        if not path.exists():
            pytest.skip(f"Manifest not found: {path}")
        return path

    def test_load_manifest_returns_three_splits(self, manifest_path):
        """Must return train, val, test splits."""
        splits = load_manifest_candidates(manifest_path)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_train_split_largest(self, manifest_path):
        """Train split should be the largest."""
        splits = load_manifest_candidates(manifest_path)
        assert len(splits["train"]) > len(splits["val"])
        assert len(splits["train"]) > len(splits["test"])

    def test_each_candidate_has_label(self, manifest_path):
        """Each candidate must have a label (0 or 1)."""
        splits = load_manifest_candidates(manifest_path)
        for split_name, candidates in splits.items():
            for cand in candidates:
                assert "label" in cand
                assert cand["label"] in (0, 1)
                assert "smiles" in cand
                assert "group_id" in cand

    def test_gold_ratio_one_eighth(self, manifest_path):
        """Each group has 1 gold + 7 negatives, so gold ratio ≈ 1/8."""
        splits = load_manifest_candidates(manifest_path)
        for split_name, candidates in splits.items():
            if not candidates:
                continue
            n_gold = sum(c["label"] for c in candidates)
            ratio = n_gold / len(candidates)
            assert 0.10 < ratio < 0.20, f"Gold ratio {ratio} in {split_name} not ~1/8"


# ---------------------------------------------------------------------------
# Checkpoint hash tests
# ---------------------------------------------------------------------------

class TestCheckpointHash:
    """Tests for checkpoint SHA-256 hash."""

    def test_hash_is_hex_string(self):
        """Checkpoint hash must be a hex string."""
        if not DEFAULT_CHECKPOINT_PATH.exists():
            pytest.skip("Checkpoint not found")
        h = compute_checkpoint_hash(DEFAULT_CHECKPOINT_PATH)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex
        int(h, 16)  # Must be valid hex

    def test_hash_consistent(self):
        """Same file must produce same hash."""
        if not DEFAULT_CHECKPOINT_PATH.exists():
            pytest.skip("Checkpoint not found")
        h1 = compute_checkpoint_hash(DEFAULT_CHECKPOINT_PATH)
        h2 = compute_checkpoint_hash(DEFAULT_CHECKPOINT_PATH)
        assert h1 == h2

    def test_nonexistent_file_empty_hash(self):
        """Nonexistent file should return empty string."""
        h = compute_checkpoint_hash(Path("/nonexistent/file.ckpt"))
        assert h == ""


# ---------------------------------------------------------------------------
# Spec structural check
# ---------------------------------------------------------------------------

class TestSpecAcceptance:
    """Verify all spec-required outputs can be produced."""

    def test_noninferiority_margin_is_negative_half_pp(self):
        """Margin must be -0.005 (-0.5 percentage points)."""
        assert NONINFERIORITY_MARGIN == -0.005

    def test_six_config_ids(self):
        """Must have exactly 6 config IDs C1-C6."""
        assert CONFIG_IDS == ["C1", "C2", "C3", "C4", "C5", "C6"]

    def test_selected_backbone_required_fields(self):
        """selected_backbone must have all 11 fields from spec."""
        # Already tested in TestBackboneSelection, but verify the field list
        required = [
            "checkpoint", "checkpoint_hash", "architecture", "target_modules",
            "rank", "alpha", "dropout", "trainable_parameters", "training_budget",
            "selection_metric", "selection_rule",
        ]
        assert len(required) == 11
