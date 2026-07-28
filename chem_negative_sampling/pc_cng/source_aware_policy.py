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

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


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
