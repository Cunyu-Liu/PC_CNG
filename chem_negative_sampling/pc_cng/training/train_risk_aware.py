"""P4-G5: Risk-aware training losses and trainer.

Five label-treatment methods for synthetic counterfactual negatives:

1. ``hard_binary``            — synthetic = absolute negative (G3 baseline)
2. ``label_smoothing``        — synthetic negative label smoothed to eps
3. ``pu_nnpu``                — gold = P, synthetic = U, non-negative PU loss
   with class prior pi estimated by the observed-data-calibrated risk model
4. ``risk_weighted_pairwise`` — within-group margin loss, per-negative weight
5. ``risk_weighted_infonce``  — within-group weighted InfoNCE (gold = anchor)

All losses share the Chemformer-LoRA C3 frozen config, the same 5-epoch
budget, and early stopping on val MRR (identical protocol to P4-G3).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

METHODS = [
    "hard_binary",
    "label_smoothing",
    "pu_nnpu",
    "risk_weighted_pairwise",
    "risk_weighted_infonce",
]

DEFAULT_LABEL_SMOOTHING_EPS = 0.1
DEFAULT_PAIRWISE_MARGIN = 1.0
DEFAULT_INFONCE_TAU = 1.0


# ---------------------------------------------------------------------------
# Losses (pure functions for testability)
# ---------------------------------------------------------------------------

def hard_binary_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """BCE with logits; synthetic negatives are absolute negatives."""
    return F.binary_cross_entropy_with_logits(logits, labels)


def label_smoothing_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eps: float = DEFAULT_LABEL_SMOOTHING_EPS,
) -> torch.Tensor:
    """BCE with asymmetric smoothing: positives -> 1-eps, negatives -> eps."""
    if not 0.0 <= eps < 0.5:
        raise ValueError(f"eps must be in [0, 0.5), got {eps}")
    smoothed = labels * (1.0 - eps) + (1.0 - labels) * eps
    return F.binary_cross_entropy_with_logits(logits, smoothed)


def pu_nnpu_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    is_unlabeled: torch.Tensor,
    pi: float,
) -> torch.Tensor:
    """Non-negative PU loss (Kiryo et al. 2017).

    P = gold observed (label 1); U = synthetic counterfactual (label 0 in
    ``labels`` but treated as unlabeled via ``is_unlabeled``).

        L = pi * E_P[l(g,1)] + max(0, E_U[l(g,0)] - pi * E_P[l(g,0)])

    pi = P(y=1 | U): estimated externally by the risk model (mean fnr).
    """
    if not 0.0 < pi < 1.0:
        raise ValueError(f"pi must be in (0, 1), got {pi}")
    zeros = torch.zeros_like(labels)
    ones = torch.ones_like(labels)
    loss_pos = F.binary_cross_entropy_with_logits(logits, ones, reduction="none")
    loss_neg = F.binary_cross_entropy_with_logits(logits, zeros, reduction="none")

    p_mask = ~is_unlabeled.bool()
    u_mask = is_unlabeled.bool()
    if not p_mask.any() or not u_mask.any():
        raise ValueError("PU loss needs at least one P and one U example")

    p_risk_pos = loss_pos[p_mask].mean()
    p_risk_neg = loss_neg[p_mask].mean()
    u_risk_neg = loss_neg[u_mask].mean()
    nn_term = torch.clamp(u_risk_neg - pi * p_risk_neg, min=0.0)
    return pi * p_risk_pos + nn_term


def risk_weighted_pairwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_ids: Sequence[str],
    weights: torch.Tensor,
    margin: float = DEFAULT_PAIRWISE_MARGIN,
) -> torch.Tensor:
    """Weighted margin ranking loss over (positive, negative) pairs within
    each group. ``weights`` is the per-example risk sample weight (only the
    negative side is used)."""
    if margin <= 0:
        raise ValueError(f"margin must be positive, got {margin}")
    total = logits.new_tensor(0.0)
    n_pairs = 0
    # Bucket indices per group (order-preserving)
    groups: Dict[str, List[int]] = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(g, []).append(i)
    for idxs in groups.values():
        pos = [i for i in idxs if labels[i].item() > 0.5]
        neg = [i for i in idxs if labels[i].item() <= 0.5]
        for pi in pos:
            for ni in neg:
                pair_loss = F.softplus(margin - (logits[pi] - logits[ni]))
                total = total + weights[ni] * pair_loss
                n_pairs += 1
    if n_pairs == 0:
        raise ValueError("no (pos, neg) pairs found in batch")
    return total / n_pairs


def risk_weighted_infonce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_ids: Sequence[str],
    weights: torch.Tensor,
    tau: float = DEFAULT_INFONCE_TAU,
) -> torch.Tensor:
    """Weighted InfoNCE within each group: gold is the anchor positive,
    synthetic negatives contribute exp(s/tau) scaled by their risk weight.

        L_g = -log( exp(s_pos/tau) / (exp(s_pos/tau) + sum_j w_j exp(s_j/tau)) )
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    groups: Dict[str, List[int]] = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(g, []).append(i)
    total = logits.new_tensor(0.0)
    n_groups = 0
    for idxs in groups.values():
        pos = [i for i in idxs if labels[i].item() > 0.5]
        neg = [i for i in idxs if labels[i].item() <= 0.5]
        if not pos or not neg:
            continue
        s_pos = logits[pos[0]] / tau
        s_neg = torch.stack([logits[j] for j in neg]) / tau
        w_neg = torch.stack([weights[j] for j in neg])
        denom = torch.exp(s_pos) + torch.sum(w_neg * torch.exp(s_neg))
        total = total + (-s_pos + torch.log(denom))
        n_groups += 1
    if n_groups == 0:
        raise ValueError("no complete group (pos+neg) found in batch")
    return total / n_groups


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def compute_loss(
    method: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_ids: Optional[Sequence[str]] = None,
    weights: Optional[torch.Tensor] = None,
    pu_prior: float = 0.1,
    label_smoothing_eps: float = DEFAULT_LABEL_SMOOTHING_EPS,
    pairwise_margin: float = DEFAULT_PAIRWISE_MARGIN,
    infonce_tau: float = DEFAULT_INFONCE_TAU,
) -> torch.Tensor:
    """Dispatch a P4-G5 loss by method name."""
    if method == "hard_binary":
        return hard_binary_loss(logits, labels)
    if method == "label_smoothing":
        return label_smoothing_loss(logits, labels, eps=label_smoothing_eps)
    if method == "pu_nnpu":
        is_unlabeled = (labels < 0.5).float()
        return pu_nnpu_loss(logits, labels, is_unlabeled, pi=pu_prior)
    if method == "risk_weighted_pairwise":
        assert group_ids is not None and weights is not None
        return risk_weighted_pairwise_loss(logits, labels, group_ids, weights, margin=pairwise_margin)
    if method == "risk_weighted_infonce":
        assert group_ids is not None and weights is not None
        return risk_weighted_infonce_loss(logits, labels, group_ids, weights, tau=infonce_tau)
    raise ValueError(f"unknown method: {method}")
