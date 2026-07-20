"""Unit tests for the LoRA adapter (P3-01)."""

from __future__ import annotations

import os
import sys

import pytest
import torch
from torch import nn

# Force CPU for tests
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.adapter import (  # noqa: E402
    LoRALinear,
    apply_lora,
    count_total_parameters,
    count_trainable_parameters,
    freeze_non_lora_params,
    LoraConfig,
)
from models.pretrained_backbone import (  # noqa: E402
    PretrainedChemformerBackbone,
    PretrainedReactionScorer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def backbone_frozen():
    return PretrainedChemformerBackbone(checkpoint_path=None, freeze=True)


@pytest.fixture
def linear_layer():
    """A simple nn.Linear layer for isolated LoRA tests."""
    torch.manual_seed(0)
    return nn.Linear(64, 32)


# ---------------------------------------------------------------------------
# LoRALinear tests
# ---------------------------------------------------------------------------
class TestLoRALinear:
    def test_output_shape(self, linear_layer):
        lora = LoRALinear(linear_layer, r=8, alpha=16.0)
        x = torch.randn(4, 64)
        out = lora(x)
        assert out.shape == (4, 32)

    def test_identity_init(self, linear_layer):
        """At init, B=0 so LoRA output == base output."""
        lora = LoRALinear(linear_layer, r=8, alpha=16.0)
        x = torch.randn(4, 64)
        base_out = linear_layer(x)
        lora_out = lora(x)
        assert torch.allclose(base_out, lora_out, atol=1e-6)

    def test_base_is_frozen(self, linear_layer):
        lora = LoRALinear(linear_layer, r=8, alpha=16.0)
        for p in lora.base.parameters():
            assert not p.requires_grad

    def test_lora_params_trainable(self, linear_layer):
        lora = LoRALinear(linear_layer, r=8, alpha=16.0)
        assert lora.lora_A.weight.requires_grad
        assert lora.lora_B.weight.requires_grad

    def test_lora_A_shape(self, linear_layer):
        lora = LoRALinear(linear_layer, r=8, alpha=16.0)
        # A: (r, in_features) = (8, 64)
        assert lora.lora_A.weight.shape == (8, 64)

    def test_lora_B_shape(self, linear_layer):
        lora = LoRALinear(linear_layer, r=8, alpha=16.0)
        # B: (out_features, r) = (32, 8)
        assert lora.lora_B.weight.shape == (32, 8)

    def test_lora_B_zero_init(self, linear_layer):
        lora = LoRALinear(linear_layer, r=8, alpha=16.0)
        assert torch.all(lora.lora_B.weight == 0)

    def test_lora_A_nonzero_init(self, linear_layer):
        lora = LoRALinear(linear_layer, r=8, alpha=16.0)
        assert not torch.all(lora.lora_A.weight == 0)

    def test_scaling_factor(self, linear_layer):
        lora = LoRALinear(linear_layer, r=8, alpha=16.0)
        assert lora.scaling == pytest.approx(16.0 / 8.0)

    def test_gradient_flow(self, linear_layer):
        """After a backward pass, only LoRA params should have non-None grads."""
        lora = LoRALinear(linear_layer, r=8, alpha=16.0)
        x = torch.randn(4, 64)
        out = lora(x)
        loss = out.sum()
        loss.backward()
        # base.weight should not have a grad (frozen)
        assert lora.base.weight.grad is None
        # LoRA params should have grads
        assert lora.lora_A.weight.grad is not None
        assert lora.lora_B.weight.grad is not None

    def test_invalid_r(self, linear_layer):
        with pytest.raises(ValueError, match="r must be positive"):
            LoRALinear(linear_layer, r=0)

    def test_different_r_values(self, linear_layer):
        for r in [1, 4, 8, 16, 32]:
            lora = LoRALinear(linear_layer, r=r, alpha=2 * r)
            assert lora.lora_A.weight.shape == (r, 64)
            assert lora.lora_B.weight.shape == (32, r)


# ---------------------------------------------------------------------------
# apply_lora tests
# ---------------------------------------------------------------------------
class TestApplyLora:
    def test_apply_lora_to_backbone(self, backbone_frozen):
        n_replaced = apply_lora(
            backbone_frozen,
            r=8,
            alpha=16.0,
            target_patterns=["encoder_layers.*.linear1", "encoder_layers.*.linear2"],
        )
        # 6 layers × 2 linear layers = 12 replacements
        assert n_replaced == 12

    def test_apply_lora_default_patterns(self, backbone_frozen):
        n_replaced = apply_lora(backbone_frozen, r=8)
        # Default targets: linear1, linear2 only (6 layers × 2 = 12)
        # Attention projections are skipped to avoid breaking nn.MultiheadAttention
        assert n_replaced == 12

    def test_apply_lora_preserves_output_shape(self, backbone_frozen):
        apply_lora(backbone_frozen, r=8, target_patterns=["encoder_layers.*.linear1"])
        backbone_frozen.eval()
        ids = torch.tensor([[2, 6, 6, 9, 3]])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            out = backbone_frozen(ids, attention_mask=mask, pool=True)
        assert out.shape == (1, 512)

    def test_lora_replace_changes_module_type(self, backbone_frozen):
        apply_lora(
            backbone_frozen,
            r=8,
            target_patterns=["encoder_layers.0.linear1"],
        )
        assert isinstance(backbone_frozen.encoder_layers[0].linear1, LoRALinear)
        # Other layers should still be nn.Linear
        assert not isinstance(backbone_frozen.encoder_layers[1].linear1, LoRALinear)


# ---------------------------------------------------------------------------
# freeze_non_lora_params tests
# ---------------------------------------------------------------------------
class TestFreezeNonLora:
    def test_freeze_all_without_lora(self, backbone_frozen):
        """Without LoRA, freeze_non_lora_params should leave 0 trainable in backbone."""
        n = freeze_non_lora_params(backbone_frozen)
        assert n == 0

    def test_freeze_with_lora(self, backbone_frozen):
        apply_lora(
            backbone_frozen,
            r=8,
            target_patterns=["encoder_layers.*.linear1", "encoder_layers.*.linear2"],
        )
        n = freeze_non_lora_params(backbone_frozen)
        # 12 LoRA layers × 2 matrices (A, B) each
        # r=8, in=2048/512, out=512/2048
        # linear1: A=(8, 2048), B=(512, 8) -> 8*2048 + 512*8 = 16384 + 4096 = 20480 per layer
        # linear2: A=(8, 512), B=(2048, 8) -> 8*512 + 2048*8 = 4096 + 16384 = 20480 per layer
        # 12 layers × 20480 = 245760 trainable params
        assert n > 0
        assert n < count_total_parameters(backbone_frozen)

    def test_freeze_with_scorer(self, backbone_frozen):
        """Head params should remain trainable."""
        apply_lora(
            backbone_frozen,
            r=8,
            target_patterns=["encoder_layers.*.linear1"],
        )
        scorer = PretrainedReactionScorer(backbone_frozen)
        n = freeze_non_lora_params(scorer)
        # LoRA params + head params
        assert n > 0
        # Check head params are trainable
        head_trainable = sum(
            p.numel() for p in scorer.head.parameters() if p.requires_grad
        )
        assert head_trainable > 0

    def test_only_lora_and_head_trainable(self, backbone_frozen):
        apply_lora(backbone_frozen, r=8)
        scorer = PretrainedReactionScorer(backbone_frozen)
        freeze_non_lora_params(scorer)
        for name, p in scorer.named_parameters():
            is_trainable = bool(p.requires_grad)
            is_lora = "lora_A" in name or "lora_B" in name
            is_head = name.startswith("head.")
            if is_trainable:
                assert is_lora or is_head, f"Unexpected trainable param: {name}"
            else:
                assert not (is_lora or is_head), f"Unexpected frozen LoRA/head param: {name}"


# ---------------------------------------------------------------------------
# Parameter counting tests
# ---------------------------------------------------------------------------
class TestParamCounting:
    def test_count_total(self, backbone_frozen):
        total = count_total_parameters(backbone_frozen)
        assert total > 1_000_000

    def test_count_trainable_frozen(self, backbone_frozen):
        n = count_trainable_parameters(backbone_frozen)
        assert n == 0

    def test_count_trainable_with_lora(self, backbone_frozen):
        apply_lora(backbone_frozen, r=8, target_patterns=["encoder_layers.*.linear1"])
        freeze_non_lora_params(backbone_frozen)
        n = count_trainable_parameters(backbone_frozen)
        assert n > 0

    def test_lora_fraction_small(self, backbone_frozen):
        """LoRA trainable params should be < 5% of total."""
        apply_lora(backbone_frozen, r=8)
        freeze_non_lora_params(backbone_frozen)
        total = count_total_parameters(backbone_frozen)
        trainable = count_trainable_parameters(backbone_frozen)
        fraction = trainable / total
        assert fraction < 0.05, f"LoRA fraction {fraction:.4f} should be < 5%"


# ---------------------------------------------------------------------------
# LoraConfig tests
# ---------------------------------------------------------------------------
class TestLoraConfig:
    def test_default_config(self):
        cfg = LoraConfig()
        assert cfg.r == 8
        assert cfg.alpha == 16.0
        assert len(cfg.target_patterns) > 0

    def test_custom_config(self):
        cfg = LoraConfig(r=16, alpha=32.0, target_patterns=["encoder.*.linear1"])
        assert cfg.r == 16
        assert cfg.alpha == 32.0
        assert cfg.target_patterns == ["encoder.*.linear1"]
