"""LoRA adapter for parameter-efficient fine-tuning of Chemformer (P3-01).

Implements Low-Rank Adaptation (Hu et al. 2022) for ``nn.Linear`` layers in
the pretrained Chemformer encoder.  Only the LoRA matrices (``A``, ``B``) are
trainable; the pretrained weights are frozen.

Usage
-----
>>> from pc_cng.models.pretrained_backbone import PretrainedChemformerBackbone
>>> from pc_cng.models.adapter import apply_lora, freeze_non_lora_params
>>> backbone = PretrainedChemformerBackbone("model_sanitized.ckpt", freeze=True)
>>> n_replaced = apply_lora(backbone, r=8, target_patterns=["encoder_layers.*.linear1",
...                                                                  "encoder_layers.*.linear2"])
>>> n_trainable = freeze_non_lora_params(backbone)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

__all__ = [
    "LoRALinear",
    "apply_lora",
    "freeze_non_lora_params",
    "count_trainable_parameters",
    "LoraConfig",
]


@dataclass
class LoraConfig:
    """Configuration for a single LoRA injection pass."""

    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_patterns: List[str] = field(
        default_factory=lambda: [
            "encoder_layers.*.linear1",
            "encoder_layers.*.linear2",
        ]
    )
    bias: str = "none"  # "none" | "all" | "lora_only"


class LoRALinear(nn.Module):
    """``nn.Linear`` + low-rank update ``W x + (B A) x``.

    The pretrained ``base`` layer is frozen (``requires_grad=False``).  ``A``
    is initialised with a small Gaussian, ``B`` is zero-initialised so the
    initial output equals the base layer's output (identity init).
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA r must be positive, got {r}")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(r)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return base_out + self.scaling * lora_out

    @property
    def weight(self) -> torch.Tensor:
        """Delegate to base.weight for compatibility with nn.MultiheadAttention."""
        return self.base.weight

    @property
    def bias(self):
        """Delegate to base.bias for compatibility with nn.MultiheadAttention."""
        return self.base.bias

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}"


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    for pat in patterns:
        # Convert shell-style glob to regex: "*" -> ".*", "?" -> "."
        regex = "^" + re.escape(pat).replace(r"\*", ".*").replace(r"\?", ".") + "$"
        if re.match(regex, name):
            return True
    return False


def apply_lora(
    module: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_patterns: Optional[Sequence[str]] = None,
) -> int:
    """Replace target ``nn.Linear`` layers with :class:`LoRALinear`.

    Parameters
    ----------
    module:
        The root module to inject LoRA into (typically a
        :class:`PretrainedChemformerBackbone`).
    r, alpha, dropout:
        LoRA hyper-parameters.
    target_patterns:
        Glob patterns matched against ``module.named_modules()`` names.  When
        ``None`` the default targets the feed-forward and attention projections
        of the Chemformer encoder layers.

    Returns
    -------
    int
        Number of ``nn.Linear`` layers replaced with ``LoRALinear``.
    """
    if target_patterns is None:
        target_patterns = LoraConfig().target_patterns

    # Collect replacements first to avoid mutating during iteration
    replacements: List[Tuple[str, nn.Linear]] = []
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        # Match against the parent-stripped name (e.g. "encoder_layers.0.linear1")
        if _matches_any(name, target_patterns):
            replacements.append((name, child))

    n_replaced = 0
    for name, child in replacements:
        lora_layer = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
        # Navigate to the parent module and replace the attribute
        *parent_path, attr = name.split(".")
        parent = module
        for part in parent_path:
            parent = getattr(parent, part)
        # Handle nn.MultiheadAttention.in_proj_weight (Parameter, not Linear)
        # by leaving it as-is; only true nn.Linear layers get wrapped.
        if hasattr(parent, attr):
            setattr(parent, attr, lora_layer)
            n_replaced += 1
    return n_replaced


def freeze_non_lora_params(module: nn.Module) -> int:
    """Freeze everything except LoRA params + classification head params.

    Returns the number of trainable parameters.
    """
    for name, p in module.named_parameters():
        # LoRA params: "lora_A.weight" / "lora_B.weight"
        is_lora = "lora_A" in name or "lora_B" in name
        # Head params: anything under "head." (for PretrainedReactionScorer)
        is_head = name.startswith("head.")
        p.requires_grad = bool(is_lora or is_head)
    return count_trainable_parameters(module)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_total_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
