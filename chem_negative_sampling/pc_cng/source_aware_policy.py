"""Development-only reaction-conditioned negative-source policy.

This module implements the minimal Phase-D policy primitive described in
``提示词/pccng3.md``.  It is deliberately independent from the Phase-4
fixed-pool runner: no test-set labels, source ratios, or benchmark-specific
thresholds are embedded here.  A caller supplies a reaction representation,
per-source candidate statistics, and an availability mask, then receives a
probability over sources under a one-candidate budget.

This is a policy component, not evidence that adaptive source selection is
useful.  Training and evaluation must use a validation split and a sealed
test split separately.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GateTrainingConfig:
    """Frozen development configuration for supervised source-policy fitting."""

    epochs: int = 300
    learning_rate: float = 2e-3
    target_temperature: float = 0.20
    entropy_weight: float = 0.02
    source_dropout: float = 0.10
    seed: int = 20260729

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.target_temperature <= 0:
            raise ValueError("target_temperature must be positive")
        if self.entropy_weight < 0:
            raise ValueError("entropy_weight cannot be negative")
        if not 0 <= self.source_dropout < 1:
            raise ValueError("source_dropout must be in [0, 1)")


class SourceAwareSoftmaxGate(nn.Module):
    """Context-conditioned softmax gate over negative-sample sources.

    Parameters
    ----------
    reaction_dim:
        Width of the reaction/context representation.
    source_feature_dim:
        Width of the per-source candidate statistics.
    n_sources:
        Number of source experts.  The second dimension of ``source_features``
        and ``available`` must equal this value.
    hidden_dim:
        Width of the shared context and source scoring layers.
    temperature:
        Positive softmax temperature.  This is a model hyperparameter and
        must be frozen before sealed-test evaluation.
    """

    def __init__(
        self,
        reaction_dim: int,
        source_feature_dim: int,
        n_sources: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if reaction_dim <= 0 or source_feature_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if n_sources <= 0:
            raise ValueError("n_sources must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.reaction_dim = int(reaction_dim)
        self.source_feature_dim = int(source_feature_dim)
        self.n_sources = int(n_sources)
        self.temperature = float(temperature)

        self.context_encoder = nn.Sequential(
            nn.Linear(self.reaction_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.source_scorer = nn.Sequential(
            nn.Linear(hidden_dim + self.source_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        reaction_features: Tensor,
        source_features: Tensor,
        available: Optional[Tensor] = None,
    ) -> Tensor:
        """Return ``p(source | reaction, task, backbone)``.

        Shapes:
            ``reaction_features``: ``[batch, reaction_dim]``
            ``source_features``: ``[batch, n_sources, source_feature_dim]``
            ``available``: optional boolean ``[batch, n_sources]``
        """
        if reaction_features.ndim != 2:
            raise ValueError("reaction_features must have shape [batch, reaction_dim]")
        if source_features.ndim != 3:
            raise ValueError(
                "source_features must have shape [batch, n_sources, source_feature_dim]"
            )
        batch, n_sources, source_dim = source_features.shape
        if reaction_features.shape != (batch, self.reaction_dim):
            raise ValueError("reaction feature shape does not match gate configuration")
        if n_sources != self.n_sources or source_dim != self.source_feature_dim:
            raise ValueError("source feature shape does not match gate configuration")

        if available is None:
            available = torch.ones(
                (batch, n_sources), dtype=torch.bool, device=source_features.device
            )
        elif available.shape != (batch, n_sources):
            raise ValueError("available must have shape [batch, n_sources]")
        else:
            available = available.to(dtype=torch.bool, device=source_features.device)
        if not bool(available.any(dim=1).all()):
            raise ValueError("each reaction must have at least one available source")

        context = self.context_encoder(reaction_features)
        context = context.unsqueeze(1).expand(-1, n_sources, -1)
        logits = self.source_scorer(
            torch.cat([context, source_features], dim=-1)
        ).squeeze(-1)
        logits = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
        return torch.softmax(logits / self.temperature, dim=-1)

    @torch.no_grad()
    def select_source(
        self,
        reaction_features: Tensor,
        source_features: Tensor,
        available: Optional[Tensor] = None,
    ) -> Tensor:
        """Select exactly one available source per reaction.

        This deterministic argmax is appropriate for a frozen evaluation
        policy.  Stochastic sampling, if desired during development, should
        be implemented by the caller with an explicitly recorded RNG seed.
        """
        probs = self(reaction_features, source_features, available)
        return probs.argmax(dim=-1)


def source_names_from_indices(
    indices: Tensor, source_names: Sequence[str]
) -> Tuple[str, ...]:
    """Convert selected source indices to names with bounds checking."""
    names = tuple(source_names)
    if not names:
        raise ValueError("source_names cannot be empty")
    if indices.ndim != 1:
        raise ValueError("indices must have shape [batch]")
    if bool(((indices < 0) | (indices >= len(names))).any()):
        raise ValueError("source index is out of bounds")
    return tuple(names[int(i)] for i in indices.detach().cpu().tolist())


def masked_reward_targets(
    rewards: Tensor,
    available: Tensor,
    temperature: float,
) -> Tensor:
    """Turn cross-fitted source rewards into a masked soft target.

    ``rewards`` must be out-of-fold or validation-derived.  This helper has
    no access to test labels and intentionally does not rank sources on a
    benchmark test set.
    """
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [batch, n_sources]")
    if available.shape != rewards.shape:
        raise ValueError("available must match rewards")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    mask = available.to(dtype=torch.bool, device=rewards.device)
    if not bool(mask.any(dim=1).all()):
        raise ValueError("each reaction must have at least one available source")
    logits = rewards / float(temperature)
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(logits, dim=-1)


def _drop_available_sources(
    available: Tensor,
    dropout: float,
    generator: torch.Generator,
) -> Tensor:
    """Apply source dropout without ever removing every source in a row."""
    if dropout <= 0:
        return available
    keep = torch.rand(
        available.shape,
        generator=generator,
        device="cpu",
    ).to(available.device) >= dropout
    dropped = available & keep
    empty = ~dropped.any(dim=1)
    if bool(empty.any()):
        # Preserve the first originally available source for empty rows.
        first = available.to(torch.int64).argmax(dim=1)
        rows = torch.nonzero(empty, as_tuple=False).flatten()
        dropped[rows, first[rows]] = True
    return dropped


def train_source_gate(
    gate: SourceAwareSoftmaxGate,
    reaction_features: Tensor,
    source_features: Tensor,
    rewards: Tensor,
    available: Tensor,
    config: GateTrainingConfig,
) -> Dict[str, object]:
    """Fit a source gate from cross-fitted per-reaction source rewards.

    The objective is soft-label cross entropy plus an optional entropy bonus.
    Source dropout is applied only to the training mask.  The target is
    recomputed under the same mask, so unavailable sources never receive
    target mass.  All tensors and the gate must already be on the desired
    device; Phase-D formal runners are expected to require CUDA.
    """
    if reaction_features.ndim != 2 or source_features.ndim != 3:
        raise ValueError("invalid gate feature ranks")
    if rewards.shape != available.shape:
        raise ValueError("rewards and available must have the same shape")
    if source_features.shape[:2] != rewards.shape:
        raise ValueError("source feature/source reward dimensions do not match")
    if reaction_features.shape[0] != rewards.shape[0]:
        raise ValueError("reaction and reward batch sizes do not match")

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(config.seed)
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )
    history: List[Dict[str, float]] = []
    gate.train()
    for epoch in range(config.epochs):
        train_available = _drop_available_sources(
            available.to(dtype=torch.bool),
            config.source_dropout,
            cpu_generator,
        )
        target = masked_reward_targets(
            rewards,
            train_available,
            config.target_temperature,
        )
        probs = gate(reaction_features, source_features, train_available)
        log_probs = torch.log(probs.clamp_min(1e-9))
        cross_entropy = -(target * log_probs).sum(dim=1).mean()
        entropy = -(probs * log_probs).sum(dim=1).mean()
        loss = cross_entropy - config.entropy_weight * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch in {0, config.epochs - 1} or (epoch + 1) % 50 == 0:
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(loss.detach().cpu()),
                    "cross_entropy": float(cross_entropy.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                }
            )

    gate.eval()
    with torch.no_grad():
        final_probs = gate(reaction_features, source_features, available)
        final_target = masked_reward_targets(
            rewards,
            available,
            config.target_temperature,
        )
        agreement = float(
            (
                final_probs.argmax(dim=1)
                == final_target.argmax(dim=1)
            ).float().mean().cpu()
        )
        selected = final_probs.argmax(dim=1)
        counts = torch.bincount(
            selected.detach().cpu(),
            minlength=gate.n_sources,
        )
        entropy = float(
            (
                -(final_probs * torch.log(final_probs.clamp_min(1e-9)))
                .sum(dim=1)
                .mean()
            ).cpu()
        )
    return {
        "history": history,
        "target_argmax_agreement": agreement,
        "selection_counts": [int(v) for v in counts.tolist()],
        "mean_policy_entropy": entropy,
        "config": {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "target_temperature": config.target_temperature,
            "entropy_weight": config.entropy_weight,
            "source_dropout": config.source_dropout,
            "seed": config.seed,
        },
    }
