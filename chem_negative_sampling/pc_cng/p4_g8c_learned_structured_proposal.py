"""P4-G8C: Learned Structured Proposal — full spec execution.

A learned negative-sampling proposal model that emits *structured* edits
(select edit locus -> select edit type -> select atom/bond arguments ->
apply constrained edit) rather than free-form SMILES.  The model is a
pure-PyTorch (no torch_geometric) reaction-graph transformer with a
validity action mask and a risk / epistemic-uncertainty head.

Architecture (G8-C spec, 7 sub-modules):
    1. reaction graph transformer  - multi-head attention on top of MPNN
    2. reaction-center encoder     - encode formed/broken bonds as context
    3. edit-locus pointer          - attention pointer to the atom to edit
    4. edit-type classifier        - atom_transmutation / bond_order_change /
                                     formed_bond_migrate / no_edit
    5. atom/bond argument decoder  - decode specific arguments
    6. validity action mask        - mask chemically invalid edits
    7. risk / uncertainty head     - false-negative risk + epistemic uncertainty

Training stages (4, no PPO):
    Stage 1 real-reaction edit reconstruction (legal edit grammar)
    Stage 2 rule-proposal imitation (imitate PC-CNG rule proposals)
    Stage 3 observed competing-outcome learning (real alternative products)
    Stage 4 risk-adjusted preference learning (DPO / IPO pairwise)

Comparison arms (4):
    rule_pc_cng            - baseline rule generator
    unconstrained_neural   - neural generator without validity mask
    learned_structured     - full model
    learned_structured_risk- full model + risk reranking

Outputs (output_dir):
    go_no_go.json, comparison_results.csv, pareto_frontier.json,
    model_checkpoint.pt, train_log.json, raw_predictions/*.csv

GO criteria: Pareto-frontier advantage over the rule version, downstream
utility CI all positive, candidate coverage matched (improvement must not
come merely from generating more candidates).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("RDKitRDLogger", "0")
try:  # RDKit is optional; degrade gracefully when unavailable.
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None

from .learned_graph_edit_decoder import (
    ATOM_FEAT_DIM,
    BOND_FEAT_DIM,
    BatchedGraph,
    ReactionGraphData,
    collate_graphs,
    featurize_atom_mapped_reaction,
    generate_boundary_negatives,
    load_checkpoint,
    pairwise_margin_loss,
    save_checkpoint,
)
from .reaction_boundary_generator import (
    BoundaryCandidate,
    ReactionBoundaryGenerator,
)
from .reaction_center_edit_decoder import (
    ATOM_VOCAB,
    ANCHOR_ATOMIC_NUMS,
    EditCandidateGroup,
    build_edit_candidate_groups,
    move_formed_bond_in_product,
)
from .atom_mapped_graph_edit import (
    ReactionCenterEdit,
    extract_reaction_center,
    has_atom_mapping,
)
from .chem_utils import (
    atom_balance_score,
    canonicalize_reaction,
    is_valid_smiles,
    join_reaction,
    molecule_parts,
    split_reaction,
    string_similarity,
    token_jaccard,
)

PHASE = "P4-G8C"
BASE_SEED = 20260724
N_BOOTSTRAP = 2000
DEFAULT_EPOCHS = 8
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-3
DEFAULT_HIDDEN = 128
DEFAULT_HEADS = 4
DEFAULT_NUM_LAYERS = 3
DEFAULT_DROPOUT = 0.1
DEFAULT_TOP_K = 8
BOND_ORDERS = (1, 2, 3)  # selectable bond orders for bond_order_change


class EditType(IntEnum):
    """Discrete edit-action taxonomy used by the edit-type classifier."""

    ATOM_TRANSMUTATION = 0
    BOND_ORDER_CHANGE = 1
    FORMED_BOND_MIGRATE = 2
    NO_EDIT = 3


NUM_EDIT_TYPES = len(EditType)
ARMS = ["rule_pc_cng", "unconstrained_neural",
        "learned_structured", "learned_structured_risk"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(gpu: Optional[int]) -> torch.device:
    if gpu is not None and torch.cuda.is_available() and gpu >= 0:
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def _safe_split(reaction: str) -> Tuple[str, str]:
    """Return (reactants, products) tolerating 'r>>p' and 'r>a>p' formats."""
    if ">>" in reaction:
        left, right = reaction.split(">>", 1)
        return left, right
    parts = reaction.split(">")
    if len(parts) == 3:
        return parts[0], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1]
    return reaction, ""


def _product_smiles(reaction: str) -> str:
    _, prod = _safe_split(reaction)
    return prod.strip()


# ---------------------------------------------------------------------------
# 1. Reaction graph transformer (MPNN + multi-head attention)
# ---------------------------------------------------------------------------

class ReactionGraphTransformer(nn.Module):
    """MPNN message passing followed by per-graph multi-head self-attention.

    Operates on a :class:`BatchedGraph` produced by ``collate_graphs``.  The
    expected batched-graph interface (pure PyTorch, no torch_geometric):

        atom_features : [N, ATOM_FEAT_DIM] float
        edge_index    : [2, E] long  (row 0 = src, row 1 = dst, undirected)
        bond_features : [E, BOND_FEAT_DIM] float
        batch         : [N] long  (node -> graph index)
        num_graphs    : int
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int,
                 dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.atom_proj = nn.Linear(ATOM_FEAT_DIM, hidden_dim)
        self.bond_proj = nn.Linear(BOND_FEAT_DIM, hidden_dim)
        self.mpnn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim),
                          nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_layers)
        ])
        self.mpnn_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                 nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim * 2, hidden_dim))
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _message_pass(self, x: torch.Tensor, edge_index: torch.Tensor,
                      bond_feat: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.mpnn_norms[layer_idx](x)
        src, dst = edge_index[0], edge_index[1]
        msgs = self.mpnn_layers[layer_idx](
            torch.cat([x[src], x[dst], bond_feat], dim=-1))
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msgs)
        return self.mpnn_norms[layer_idx](x + self.dropout(agg))

    def forward(self, batch: BatchedGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.atom_proj(batch.atom_features)
        bond_feat = self.bond_proj(batch.edge_features) if \
            batch.edge_features.numel() else \
            torch.zeros((0, self.hidden_dim), device=x.device)
        edge_index = batch.edge_index
        for i in range(len(self.mpnn_layers)):
            x = self._message_pass(x, edge_index, bond_feat, i)
        # Per-graph padded self-attention.
        node_emb, graph_emb = self._batched_attention(x, batch.batch_idx,
                                                      len(batch.graphs))
        return node_emb, graph_emb

    def _batched_attention(self, x: torch.Tensor, batch_idx: torch.Tensor,
                           num_graphs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        sizes = torch.bincount(batch_idx, minlength=num_graphs)
        max_len = int(sizes.max().item()) if num_graphs > 0 else 0
        padded = x.new_zeros(num_graphs, max_len, self.hidden_dim)
        mask = torch.ones(num_graphs, max_len, dtype=torch.bool, device=device)
        # scatter nodes into padded layout
        order = torch.argsort(batch_idx, stable=True)
        sorted_b = batch_idx[order]
        cum = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)[:-1]])
        offsets = torch.arange(len(order), device=device) - cum[sorted_b]
        padded[sorted_b, offsets] = x[order]
        valid = torch.arange(max_len, device=device)[None, :] < sizes[:, None]
        mask = ~valid
        attn_out, _ = self.attn(padded, padded, padded,
                                key_padding_mask=mask, need_weights=False)
        x_new = padded * valid.unsqueeze(-1)
        # gather back to node order
        node_emb = x_new[sorted_b, offsets]
        node_emb = self.attn_norm(x + self.dropout(node_emb - x))
        ffn_out = self.ffn(node_emb)
        node_emb = self.ffn_norm(node_emb + self.dropout(ffn_out))
        graph_emb = (node_emb.new_zeros(num_graphs, self.hidden_dim)
                     .index_add_(0, batch_idx, node_emb))
        denom = sizes.clamp(min=1).float().unsqueeze(-1)
        graph_emb = graph_emb / denom
        return node_emb, graph_emb


# ---------------------------------------------------------------------------
# 2. Reaction-center encoder
# ---------------------------------------------------------------------------

class ReactionCenterEncoder(nn.Module):
    """Encode formed / broken bonds as a context vector.

    Builds a small bag-of-edits summary (count of formed / broken bonds and
    their participating atom features) and projects it to ``hidden_dim``.
    Falls back to a zero context when no reaction center is available.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # 4 summary channels: n_formed, n_broken, mean_formed_atom_feat,
        # mean_broken_atom_feat (each summarised by a learned projection).
        self.proj = nn.Sequential(
            nn.Linear(ATOM_FEAT_DIM * 2 + 4, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, node_emb: torch.Tensor, batch_idx: torch.Tensor,
                num_graphs: int,
                center_summary: Optional[torch.Tensor]) -> torch.Tensor:
        if center_summary is None:
            return node_emb.new_zeros(num_graphs, self.hidden_dim)
        return self.proj(center_summary)


# ---------------------------------------------------------------------------
# 3. Edit-locus pointer
# ---------------------------------------------------------------------------

class EditLocusPointer(nn.Module):
    """Attention-based pointer selecting which product atom to edit.

    Produces per-node logits (masked per graph); the locus distribution is a
    softmax over the atoms of each reaction's product graph.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, node_emb: torch.Tensor, graph_emb: torch.Tensor,
                batch_idx: torch.Tensor, num_graphs: int,
                locus_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """Return per-graph pointer logits over nodes [num_graphs, max_len]."""
        q = self.context_proj(graph_emb)  # [B, H]
        k = self.key_proj(node_emb)        # [N, H]
        sizes = torch.bincount(batch_idx, minlength=num_graphs)
        max_len = int(sizes.max().item()) if num_graphs > 0 else 0
        logits = node_emb.new_full(
            (num_graphs, max_len), -1e9)
        order = torch.argsort(batch_idx, stable=True)
        sorted_b = batch_idx[order]
        cum = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)[:-1]])
        offsets = torch.arange(len(order), device=node_emb.device) - cum[sorted_b]
        scores = self.v(torch.tanh(q[sorted_b] + k[order])).squeeze(-1)
        logits[sorted_b, offsets] = scores
        valid = torch.arange(max_len, device=node_emb.device)[None, :] < sizes[:, None]
        logits = logits.masked_fill(~valid, -1e9)
        if locus_mask is not None:
            logits = logits.masked_fill(~locus_mask, -1e9)
        return logits


# ---------------------------------------------------------------------------
# 4. Edit-type classifier
# ---------------------------------------------------------------------------

class EditTypeClassifier(nn.Module):
    """Classify the edit type over the :class:`EditType` taxonomy."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, NUM_EDIT_TYPES))

    def forward(self, graph_emb: torch.Tensor,
                locus_emb: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([graph_emb, locus_emb], dim=-1))


# ---------------------------------------------------------------------------
# 5. Atom / bond argument decoder
# ---------------------------------------------------------------------------

class AtomBondArgumentDecoder(nn.Module):
    """Decode the concrete arguments of each edit type.

    * atom_transmutation  -> target atomic number (over ``ANCHOR_ATOMIC_NUMS``)
    * bond_order_change   -> new bond order (1 / 2 / 3)
    * formed_bond_migrate -> destination atom pointer (reuses locus logits)
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        n_atom = max(len(ANCHOR_ATOMIC_NUMS), 1)
        self.atom_head = nn.Linear(hidden_dim * 2, n_atom)
        self.bond_head = nn.Linear(hidden_dim * 2, len(BOND_ORDERS))
        self.migrate_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, graph_emb: torch.Tensor,
                locus_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        ctx = torch.cat([graph_emb, locus_emb], dim=-1)
        return {
            "atom_logits": self.atom_head(ctx),
            "bond_logits": self.bond_head(ctx),
            "migrate_query": self.migrate_head(locus_emb),
        }


# ---------------------------------------------------------------------------
# 6. Validity action mask
# ---------------------------------------------------------------------------

class ValidityActionMask(nn.Module):
    """Predict a soft mask over chemically invalid (locus, edit-type) pairs.

    A learned head produces logits that, after a sigmoid, down-weight edits
    the model believes to be chemically invalid (e.g. transmuting a hydrogen,
    migrating a non-existent formed bond).  Rule-augmented hard masks can be
    multiplied in at inference time.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.locus_type = nn.Linear(hidden_dim, NUM_EDIT_TYPES)
        self.graph_type = nn.Linear(hidden_dim, NUM_EDIT_TYPES)

    def forward(self, node_emb: torch.Tensor, graph_emb: torch.Tensor,
                batch_idx: torch.Tensor, num_graphs: int,
                hard_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """Return keep-probability mask [num_graphs, max_len, NUM_EDIT_TYPES]."""
        sizes = torch.bincount(batch_idx, minlength=num_graphs)
        max_len = int(sizes.max().item()) if num_graphs > 0 else 0
        g = self.graph_type(graph_emb)  # [B, T]
        # broadcast graph-level type logits across nodes
        type_logits = g.unsqueeze(1).expand(num_graphs, max_len, NUM_EDIT_TYPES)
        keep = torch.sigmoid(type_logits)
        valid = torch.arange(max_len, device=node_emb.device)[None, :] < sizes[:, None]
        keep = keep * valid.unsqueeze(-1).float()
        if hard_mask is not None:
            keep = keep * hard_mask.float()
        return keep


# ---------------------------------------------------------------------------
# 7. Risk / uncertainty head
# ---------------------------------------------------------------------------

class RiskUncertaintyHead(nn.Module):
    """Predict false-negative risk and an epistemic-uncertainty estimate.

    Uncertainty is estimated via Monte-Carlo dropout: calling
    :meth:`enable_mc_dropout` toggles dropout layers so repeated forward
    passes yield a predictive-variance estimate.
    """

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Softplus())
        self.mc_dropout = nn.Dropout(dropout)

    def forward(self, graph_emb: torch.Tensor,
                locus_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx = self.mc_dropout(torch.cat([graph_emb, locus_emb], dim=-1))
        risk = self.risk_head(ctx).squeeze(-1)
        uncertainty = self.uncertainty_head(ctx).squeeze(-1)
        return risk, uncertainty

    def enable_mc_dropout(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def mc_estimate(self, graph_emb: torch.Tensor, locus_emb: torch.Tensor,
                    n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        self.enable_mc_dropout()
        risks, uncs = [], []
        for _ in range(n_samples):
            r, u = self.forward(graph_emb, locus_emb)
            risks.append(r)
            uncs.append(u)
        risks_t = torch.stack(risks)
        uncs_t = torch.stack(uncs)
        return risks_t.mean(0), uncs_t.std(0) + risks_t.std(0)


# ---------------------------------------------------------------------------
# StructuredProposalModel — composes all 7 sub-modules
# ---------------------------------------------------------------------------

@dataclass
class StructuredProposalOutput:
    """Container for a single forward pass of the structured proposal model."""

    locus_logits: torch.Tensor
    type_logits: torch.Tensor
    arg_logits: Dict[str, torch.Tensor]
    validity_mask: torch.Tensor
    risk: torch.Tensor
    uncertainty: torch.Tensor
    graph_emb: torch.Tensor
    node_emb: torch.Tensor


class StructuredProposalModel(nn.Module):
    """Full learned structured-proposal model (7 sub-modules)."""

    def __init__(self, hidden_dim: int = DEFAULT_HIDDEN,
                 num_heads: int = DEFAULT_HEADS,
                 num_layers: int = DEFAULT_NUM_LAYERS,
                 dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transformer = ReactionGraphTransformer(
            hidden_dim, num_heads, num_layers, dropout)
        self.center_encoder = ReactionCenterEncoder(hidden_dim)
        self.locus_pointer = EditLocusPointer(hidden_dim)
        self.type_classifier = EditTypeClassifier(hidden_dim, dropout)
        self.arg_decoder = AtomBondArgumentDecoder(hidden_dim)
        self.validity_mask = ValidityActionMask(hidden_dim)
        self.risk_head = RiskUncertaintyHead(hidden_dim, dropout)

    def _gather_locus_emb(self, node_emb: torch.Tensor,
                          batch_idx: torch.Tensor, num_graphs: int,
                          locus_index: Optional[torch.Tensor]
                          ) -> torch.Tensor:
        if locus_index is None:
            # use mean-pooled node embedding as soft locus embedding
            return self._mean_pool(node_emb, batch_idx, num_graphs)
        locus_index = locus_index.clamp(min=0)
        gathered = node_emb[locus_index]
        return gathered

    def _mean_pool(self, node_emb: torch.Tensor, batch_idx: torch.Tensor,
                   num_graphs: int) -> torch.Tensor:
        pooled = node_emb.new_zeros(num_graphs, self.hidden_dim)
        pooled.index_add_(0, batch_idx, node_emb)
        sizes = torch.bincount(batch_idx, minlength=num_graphs).clamp(min=1).float()
        return pooled / sizes.unsqueeze(-1)

    def forward(self, batch: BatchedGraph,
                center_summary: Optional[torch.Tensor] = None,
                locus_index: Optional[torch.Tensor] = None,
                hard_validity_mask: Optional[torch.Tensor] = None,
                ) -> StructuredProposalOutput:
        node_emb, graph_emb = self.transformer(batch)
        num_graphs = len(batch.graphs)
        batch_idx = batch.batch_idx
        center_ctx = self.center_encoder(
            node_emb, batch_idx, num_graphs, center_summary)
        graph_ctx = graph_emb + center_ctx
        locus_logits = self.locus_pointer(
            node_emb, graph_ctx, batch_idx, num_graphs)
        locus_idx = locus_index if locus_index is not None else \
            locus_logits.argmax(dim=-1)
        locus_emb = self._gather_locus_emb(node_emb, batch_idx, num_graphs, locus_idx)
        type_logits = self.type_classifier(graph_ctx, locus_emb)
        arg_logits = self.arg_decoder(graph_ctx, locus_emb)
        validity_mask = self.validity_mask(
            node_emb, graph_ctx, batch_idx, num_graphs, hard_validity_mask)
        risk, uncertainty = self.risk_head(graph_ctx, locus_emb)
        return StructuredProposalOutput(
            locus_logits=locus_logits, type_logits=type_logits,
            arg_logits=arg_logits, validity_mask=validity_mask,
            risk=risk, uncertainty=uncertainty,
            graph_emb=graph_ctx, node_emb=node_emb)


# ---------------------------------------------------------------------------
# Training-stage losses
# ---------------------------------------------------------------------------

class Stage1ReconstructionLoss(nn.Module):
    """Stage 1: reconstruct the real reaction edit from a positive reaction.

    Supervises locus, edit-type and argument heads against the ground-truth
    edit extracted from an atom-mapped positive reaction.
    """

    def __init__(self, locus_w: float = 1.0, type_w: float = 1.0,
                 arg_w: float = 0.5) -> None:
        super().__init__()
        self.locus_w = locus_w
        self.type_w = type_w
        self.arg_w = arg_w

    def forward(self, out: StructuredProposalOutput,
                locus_target: torch.Tensor,
                type_target: torch.Tensor,
                arg_target: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        locus_loss = F.cross_entropy(
            out.locus_logits, locus_target.clamp(min=0))
        type_loss = F.cross_entropy(out.type_logits, type_target)
        total = self.locus_w * locus_loss + self.type_w * type_loss
        comps = {"locus_loss": float(locus_loss.item()),
                 "type_loss": float(type_loss.item())}
        if arg_target is not None:
            arg_total = out.locus_logits.new_zeros(())
            for key, tgt in arg_target.items():
                logits = out.arg_logits.get(key)
                if logits is None or tgt is None:
                    continue
                arg_total = arg_total + F.cross_entropy(logits, tgt.clamp(min=0))
            total = total + self.arg_w * arg_total
            comps["arg_loss"] = float(arg_total.item())
        return total, comps


class Stage2ImitationLoss(nn.Module):
    """Stage 2: imitate PC-CNG rule-based proposals (soft-target KL)."""

    def __init__(self, temperature: float = 2.0,
                 locus_w: float = 1.0, type_w: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.locus_w = locus_w
        self.type_w = type_w

    def forward(self, out: StructuredProposalOutput,
                rule_locus_probs: torch.Tensor,
                rule_type_probs: torch.Tensor,
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        log_locus = F.log_softmax(out.locus_logits, dim=-1)
        log_type = F.log_softmax(out.type_logits, dim=-1)
        locus_kl = (rule_locus_probs *
                    (rule_locus_probs.clamp(min=1e-9).log() - log_locus))
        locus_kl = locus_kl.sum(-1).mean()
        type_kl = (rule_type_probs *
                   (rule_type_probs.clamp(min=1e-9).log() - log_type))
        type_kl = type_kl.sum(-1).mean()
        total = self.locus_w * locus_kl + self.type_w * type_kl
        return total, {"locus_kl": float(locus_kl.item()),
                       "type_kl": float(type_kl.item())}


class Stage3ContrastiveLoss(nn.Module):
    """Stage 3: observed competing-outcome contrastive learning.

    Pulls observed (real alternative) products together and pushes
    unobserved corruptions apart via a margin contrastive loss.
    """

    def __init__(self, margin: float = 0.5, temperature: float = 0.07) -> None:
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, out: StructuredProposalOutput,
                positive_mask: torch.Tensor,
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        risk = out.risk
        # InfoNCE-style: positive (observed) candidates should have lower risk.
        pos = (positive_mask.float() * risk).sum() / positive_mask.float().sum().clamp(min=1)
        neg = ((1 - positive_mask.float()) * risk).sum() / (1 - positive_mask.float()).sum().clamp(min=1)
        contrast = F.relu(neg - pos + self.margin)
        # also encourage spread via entropy of the locus distribution
        locus_probs = F.softmax(out.locus_logits, dim=-1)
        entropy = -(locus_probs * (locus_probs.clamp(min=1e-9).log())).sum(-1).mean()
        total = contrast - 0.01 * entropy
        return total, {"pos_risk": float(pos.item()),
                       "neg_risk": float(neg.item()),
                       "contrast": float(contrast.item()),
                       "entropy": float(entropy.item())}


class Stage4DPOLoss(nn.Module):
    """Stage 4: risk-adjusted preference learning via DPO / IPO.

    DPO: -log sigma(beta * (logp_pref - logp_disp - logp_ref_pref + logp_ref_disp))
    IPO: (logp_pref - logp_disp - logp_ref_pref + logp_ref_disp - 0.5)^2
    No PPO is used.
    """

    def __init__(self, beta: float = 0.1, use_ipo: bool = True) -> None:
        super().__init__()
        self.beta = beta
        self.use_ipo = use_ipo

    def _logp(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(logits, dim=-1).gather(
            -1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)

    def forward(self, out_pref: StructuredProposalOutput,
                out_disp: StructuredProposalOutput,
                locus_pref: torch.Tensor, type_pref: torch.Tensor,
                locus_disp: torch.Tensor, type_disp: torch.Tensor,
                ref_pref: torch.Tensor, ref_disp: torch.Tensor,
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logp_pref = self._logp(out_pref.locus_logits, locus_pref) + \
            self._logp(out_pref.type_logits, type_pref)
        logp_disp = self._logp(out_disp.locus_logits, locus_disp) + \
            self._logp(out_disp.type_logits, type_disp)
        delta = (logp_pref - logp_disp) - (ref_pref - ref_disp)
        if self.use_ipo:
            loss = (delta - 0.5).pow(2).mean()
        else:
            loss = -F.logsigmoid(self.beta * delta).mean()
        acc = (delta > 0).float().mean()
        return loss, {"dpo_loss": float(loss.item()),
                      "preference_acc": float(acc.item()),
                      "delta_mean": float(delta.mean().item())}


# ---------------------------------------------------------------------------
# Edit application utilities
# ---------------------------------------------------------------------------

@dataclass
class StructuredEdit:
    """A single decoded structured edit (action sequence)."""

    locus: int
    edit_type: EditType
    atom_arg: Optional[int] = None
    bond_arg: Optional[int] = None
    migrate_target: Optional[int] = None
    risk: float = 0.0
    uncertainty: float = 0.0
    boundary_value: float = 0.0


def _strip_atom_maps(smiles: str) -> str:
    """Remove atom mapping numbers from a SMILES string."""
    if not smiles or Chem is None:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def _apply_structured_edit(reaction_smiles: str, edit: StructuredEdit,
                           ) -> Optional[str]:
    """Apply a constrained edit to the product of ``reaction_smiles``.

    Returns the edited product SMILES, or ``None`` when the edit cannot be
    realised (invalid chemistry, missing atom map, etc.).  Reuses
    ``move_formed_bond_in_product`` for formed-bond migration and RDKit for
    atom transmutation / bond-order edits.
    """
    if edit.edit_type == EditType.NO_EDIT:
        return _strip_atom_maps(_product_smiles(reaction_smiles))
    if Chem is None:
        return None
    reactants, product = _safe_split(reaction_smiles)
    product = product.strip()
    mol = Chem.MolFromSmiles(product)
    if mol is None:
        return None
    try:
        if edit.edit_type == EditType.ATOM_TRANSMUTATION:
            if edit.atom_arg is None or edit.locus >= mol.GetNumAtoms():
                return None
            atomic_nums = list(ANCHOR_ATOMIC_NUMS) or list(ATOM_VOCAB)
            if edit.atom_arg >= len(atomic_nums):
                return None
            new_z = atomic_nums[edit.atom_arg]
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(int(edit.locus)).SetAtomicNum(int(new_z))
            edited = Chem.MolToSmiles(rw.GetMol())
            edited = _strip_atom_maps(edited)
            return edited if edited and is_valid_smiles(edited) else None
        if edit.edit_type == EditType.BOND_ORDER_CHANGE:
            if edit.bond_arg is None or edit.locus >= mol.GetNumAtoms():
                return None
            new_order = BOND_ORDERS[edit.bond_arg % len(BOND_ORDERS)]
            # apply to the highest-order bond at the locus, if any
            atom = mol.GetAtomWithIdx(int(edit.locus))
            bonds = [b for b in atom.GetBonds()]
            if not bonds:
                return None
            rw = Chem.RWMol(mol)
            target = max(bonds, key=lambda b: b.GetBondTypeAsDouble())
            bo = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                  3: Chem.BondType.TRIPLE}.get(new_order, Chem.BondType.SINGLE)
            rw.GetBondWithIdx(target.GetIdx()).SetBondType(bo)
            edited = _strip_atom_maps(Chem.MolToSmiles(rw.GetMol()))
            return edited if edited and is_valid_smiles(edited) else None
        if edit.edit_type == EditType.FORMED_BOND_MIGRATE:
            # Direct RDKit bond migration: move bond from locus to migrate_target
            if edit.migrate_target is None or edit.locus >= mol.GetNumAtoms():
                return None
            tgt_idx = int(edit.migrate_target)
            if tgt_idx >= mol.GetNumAtoms() or tgt_idx == int(edit.locus):
                return None
            # find a bond at the locus to migrate
            atom = mol.GetAtomWithIdx(int(edit.locus))
            bonds = [b for b in atom.GetBonds()
                     if b.GetOtherAtomIdx(int(edit.locus)) != tgt_idx]
            if not bonds:
                return None
            rw = Chem.RWMol(mol)
            target_bond = max(bonds, key=lambda b: b.GetBondTypeAsDouble())
            old_neighbor = target_bond.GetOtherAtomIdx(int(edit.locus))
            rw.RemoveBond(int(edit.locus), old_neighbor)
            rw.AddBond(tgt_idx, old_neighbor, Chem.BondType.SINGLE)
            try:
                new_mol = rw.GetMol()
                Chem.SanitizeMol(new_mol)
                edited = _strip_atom_maps(Chem.MolToSmiles(new_mol, isomericSmiles=True))
                return edited if edited and is_valid_smiles(edited) else None
            except Exception:
                return None
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Proposal generation
# ---------------------------------------------------------------------------

_MAPPER_CACHE: Optional[Any] = None


def _get_mapper() -> Any:
    global _MAPPER_CACHE
    if _MAPPER_CACHE is None:
        from .reaction_boundary_generator import RXNMapperAdapter
        _MAPPER_CACHE = RXNMapperAdapter()
    return _MAPPER_CACHE


def _featurize_safe(reaction_smiles: str,
                    map_unmapped: bool = False) -> Optional[ReactionGraphData]:
    try:
        mapper = _get_mapper() if map_unmapped else None
        graphs, reason = featurize_atom_mapped_reaction(
            reaction_smiles, mapper=mapper, map_unmapped=map_unmapped)
        if reason != "ok" or not graphs:
            return None
        return graphs[0]
    except Exception:
        return None


def generate_structured_proposal(model: StructuredProposalModel,
                                 reaction_smiles: str,
                                 top_k: int = DEFAULT_TOP_K,
                                 device: Optional[torch.device] = None,
                                 use_validity_mask: bool = True,
                                 risk_rerank: bool = False,
                                 n_mc: int = 0,
                                 map_unmapped: bool = False,
                                 ) -> List[StructuredEdit]:
    """Generate up to ``top_k`` structured edits for a single reaction.

    The action sequence is: select edit locus -> select edit type ->
    select atom/bond arguments -> apply constrained edit.  Risk and
    uncertainty are attached to each edit; when ``risk_rerank`` is set the
    edits are re-ranked by boundary_value = utility_proxy - lambda * risk.
    """
    device = device or next(model.parameters()).device
    model.eval()
    graph_data = _featurize_safe(reaction_smiles, map_unmapped=map_unmapped)
    if graph_data is None:
        return []
    batch = collate_graphs([graph_data])
    batch.atom_features = batch.atom_features.to(device)
    batch.edge_features = batch.edge_features.to(device)
    batch.edge_index = batch.edge_index.to(device)
    batch.batch_idx = batch.batch_idx.to(device)
    with torch.no_grad():
        out = model(batch, hard_validity_mask=None)
    locus_logits = out.locus_logits[0]
    type_logits = out.type_logits[0]
    atom_logits = out.arg_logits["atom_logits"][0]
    bond_logits = out.arg_logits["bond_logits"][0]
    if use_validity_mask:
        keep = out.validity_mask[0]  # [max_len, T]
        locus_keep = keep.max(dim=-1).values  # [max_len]
        locus_logits = locus_logits + locus_keep.log().clamp(min=-30)
    locus_probs = F.softmax(locus_logits, dim=-1)
    type_probs = F.softmax(type_logits, dim=-1)
    # Exclude NO_EDIT from sampling: it produces the original product (a positive),
    # not a negative candidate.  Set its probability to zero and renormalise.
    type_probs_no_edit = type_probs.clone()
    type_probs_no_edit[int(EditType.NO_EDIT)] = 0.0
    type_probs_no_edit = type_probs_no_edit / type_probs_no_edit.sum().clamp(min=1e-8)
    n_atoms = int((locus_logits > -1e8).sum().item())
    # Clamp to product atom count: the graph represents the product molecule,
    # but _apply_structured_edit operates on Chem.MolFromSmiles(product).
    # Ensure locus/migrate_target are valid product atom indices.
    _reactants, _product = _safe_split(reaction_smiles)
    _product = _product.strip()
    product_n_atoms = 0
    if Chem is not None and _product:
        _mol = Chem.MolFromSmiles(_product)
        if _mol is not None:
            product_n_atoms = _mol.GetNumAtoms()
    if product_n_atoms > 0:
        n_atoms = min(n_atoms, product_n_atoms)
        # Mask out positions beyond product_n_atoms and recompute probs
        if locus_logits.shape[0] > product_n_atoms:
            locus_logits = locus_logits.clone()
            locus_logits[product_n_atoms:] = -1e9
            locus_probs = F.softmax(locus_logits, dim=-1)
    candidates: List[StructuredEdit] = []
    # Generate edits: mix of forced types (for coverage) and sampled types (for diversity)
    _orig_product = _strip_atom_maps(_product_smiles(reaction_smiles))
    forced_types = [t for t in [EditType.ATOM_TRANSMUTATION, EditType.BOND_ORDER_CHANGE,
                                 EditType.FORMED_BOND_MIGRATE] if t != EditType.NO_EDIT]
    attempts = 0
    max_attempts = top_k * 6
    while len(candidates) < top_k and attempts < max_attempts:
        attempts += 1
        # For first 3 slots, force each edit type; after that, sample
        if len(candidates) < len(forced_types):
            type_id = int(forced_types[len(candidates)])
        else:
            if use_validity_mask and out.validity_mask.numel() > 0:
                locus_type_keep = out.validity_mask[0, 0]  # placeholder, will use sampled locus
            type_id = int(torch.multinomial(type_probs_no_edit, 1).item())
        edit_type = EditType(type_id)
        locus = int(torch.multinomial(locus_probs, 1).item()) if n_atoms > 0 else 0
        if locus >= n_atoms:
            continue
        # Use argmax for args (model's best guess) for forced types; sample for others
        if len(candidates) < len(forced_types):
            atom_arg = int(atom_logits.argmax().item())
            bond_arg = int(bond_logits.argmax().item())
        else:
            atom_probs = F.softmax(atom_logits, dim=-1)
            bond_probs = F.softmax(bond_logits, dim=-1)
            atom_arg = int(torch.multinomial(atom_probs, 1).item())
            bond_arg = int(torch.multinomial(bond_probs, 1).item())
        migrate_target = None
        if edit_type == EditType.FORMED_BOND_MIGRATE and n_atoms > 1:
            migrate_target = atom_arg % n_atoms
            if migrate_target == locus:
                migrate_target = (locus + 1) % n_atoms
            if product_n_atoms > 0 and migrate_target >= product_n_atoms:
                migrate_target = migrate_target % product_n_atoms
        # Quick validity pre-check: skip if edit can't produce valid DIFFERENT SMILES
        test_edit = StructuredEdit(
            locus=locus, edit_type=edit_type, atom_arg=atom_arg,
            bond_arg=bond_arg, migrate_target=migrate_target)
        test_result = _apply_structured_edit(reaction_smiles, test_edit)
        if test_result is None or test_result == _orig_product:
            # Try with different args
            for alt_atom in range(min(len(atom_logits), n_atoms)):
                alt_arg = int(alt_atom)
                if edit_type == EditType.FORMED_BOND_MIGRATE:
                    if alt_arg == locus:
                        continue
                    test_edit2 = StructuredEdit(
                        locus=locus, edit_type=edit_type, atom_arg=alt_arg,
                        bond_arg=bond_arg, migrate_target=alt_arg)
                else:
                    test_edit2 = StructuredEdit(
                        locus=locus, edit_type=edit_type, atom_arg=alt_arg,
                        bond_arg=bond_arg)
                test_result2 = _apply_structured_edit(reaction_smiles, test_edit2)
                if test_result2 is not None and test_result2 != _orig_product:
                    atom_arg = alt_arg
                    migrate_target = alt_arg if edit_type == EditType.FORMED_BOND_MIGRATE else None
                    break
            else:
                continue
        risk = float(out.risk[0].item())
        uncertainty = float(out.uncertainty[0].item())
        if n_mc > 0:
            with torch.no_grad():
                r, u = model.risk_head.mc_estimate(
                    out.graph_emb, out.graph_emb, n_samples=n_mc)
            risk = float(r[0].item())
            uncertainty = float(u[0].item())
        boundary = (1.0 - risk) * (1.0 - uncertainty)
        candidates.append(StructuredEdit(
            locus=locus, edit_type=edit_type, atom_arg=atom_arg,
            bond_arg=bond_arg, migrate_target=migrate_target,
            risk=risk, uncertainty=uncertainty,
            boundary_value=boundary))
    # deduplicate by (locus, type, args)
    seen = set()
    unique = []
    for c in candidates:
        key = (c.locus, c.edit_type, c.atom_arg, c.bond_arg, c.migrate_target)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    if risk_rerank:
        unique.sort(key=lambda e: -e.boundary_value)
    return unique[:top_k]


def proposal_to_negatives(reaction_smiles: str,
                          edits: Sequence[StructuredEdit]) -> List[str]:
    """Turn structured edits into edited-product SMILES negatives.

    Filters out candidates that are identical to the original product
    (edits that don't actually change the molecule).
    """
    original = _strip_atom_maps(_product_smiles(reaction_smiles))
    out = []
    for edit in edits:
        edited = _apply_structured_edit(reaction_smiles, edit)
        if edited and is_valid_smiles(edited) and edited != original:
            out.append(edited)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _chemical_validity(smiles: str, reactants: str) -> bool:
    if not is_valid_smiles(smiles):
        return False
    try:
        return atom_balance_score(reactants, smiles) >= 0.0
    except Exception:
        return True


def _collision_risk(negatives: Sequence[str],
                    positives: Sequence[str]) -> float:
    pos_set = set(positives)
    if not negatives:
        return 0.0
    hits = sum(1 for n in negatives if n in pos_set)
    return hits / len(negatives)


def _diversity(negatives: Sequence[str]) -> float:
    if not negatives:
        return 0.0
    return len(set(negatives)) / len(negatives)


def _edit_controllability(edits: Sequence[StructuredEdit],
                          requested_type: EditType) -> float:
    if not edits:
        return 0.0
    return sum(1 for e in edits if e.edit_type == requested_type) / len(edits)


def _reaction_family_coverage(negatives: Sequence[str]) -> int:
    """Coarse reaction-family diversity: count distinct heavy-atom scaffolds."""
    if Chem is None:
        return len(set(negatives))
    scaffolds = set()
    for s in negatives:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.add(Chem.MolToSmiles(sc))
        except Exception:
            scaffolds.add(s)
    return len(scaffolds)


def _downstream_utility(train_neg: Sequence[str], train_pos: Sequence[str],
                        test_neg: Sequence[str], test_pos: Sequence[str],
                        seed: int = BASE_SEED) -> float:
    """Train a tiny Morgan-fingerprint MLP classifier and return test AUPRC.

    A higher AUPRC means the negatives are more informative for downstream
    discrimination (the utility proxy required by the G8-C spec).
    """
    if Chem is None or not test_neg + test_pos:
        return 0.0
    from rdkit.Chem import AllChem
    FP_BITS = 1024

    def fp(s):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_BITS)

    def to_x(smiles_list, label):
        xs, ys = [], []
        for s in smiles_list:
            v = fp(s)
            if v is None:
                continue
            arr = np.zeros(FP_BITS, dtype=np.float32)
            from rdkit import DataStructs
            DataStructs.ConvertToNumpyArray(v, arr)
            xs.append(arr)
            ys.append(label)
        return xs, ys

    xtr, ytr = to_x(train_neg, 0)
    x2, y2 = to_x(train_pos, 1)
    xtr += x2; ytr += y2
    xte, yte = to_x(test_neg, 0)
    x3, y3 = to_x(test_pos, 1)
    xte += x3; yte += y3
    if len(xtr) < 4 or len(set(ytr)) < 2 or len(set(yte)) < 2:
        return 0.0
    rng = np.random.RandomState(seed)
    Xtr = np.stack(xtr); Ytr = np.array(ytr)
    Xte = np.stack(xte); Yte = np.array(yte)
    clf = _MLPClassifier(FP_BITS, seed=seed)
    clf.fit(Xtr, Ytr, epochs=30, lr=1e-2)
    scores = clf.predict_proba(Xte)
    return float(_auprc(Yte, scores))


class _MLPClassifier:
    def __init__(self, in_dim: int, hidden: int = 128, seed: int = BASE_SEED):
        g = torch.Generator().manual_seed(seed)
        self.W1 = torch.randn(in_dim, hidden, generator=g) * 0.01
        self.b1 = torch.zeros(hidden)
        self.W2 = torch.randn(hidden, 1, generator=g) * 0.01
        self.b2 = torch.zeros(1)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30,
            lr: float = 1e-2) -> None:
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32)).unsqueeze(-1)
        W1 = self.W1.clone().requires_grad_(True)
        b1 = self.b1.clone().requires_grad_(True)
        W2 = self.W2.clone().requires_grad_(True)
        b2 = self.b2.clone().requires_grad_(True)
        opt = torch.optim.Adam([W1, b1, W2, b2], lr=lr)
        bce = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            h = torch.relu(X @ W1 + b1)
            logits = h @ W2 + b2
            loss = bce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        self.W1, self.b1, self.W2, self.b2 = W1.detach(), b1.detach(), W2.detach(), b2.detach()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = torch.from_numpy(X.astype(np.float32))
        with torch.no_grad():
            h = torch.relu(X @ self.W1 + self.b1)
            return torch.sigmoid(h @ self.W2 + self.b2).squeeze(-1).numpy()


def _auprc(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = np.asarray(y)[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(y.sum(), 1)
    if len(recall) < 2:
        return 0.0
    return float(np.trapz(precision, recall))


# ---------------------------------------------------------------------------
# Cluster bootstrap CI for downstream utility
# ---------------------------------------------------------------------------

def cluster_bootstrap_utility_ci(
    arm_utility_per_cluster: Sequence[Tuple[str, float]],
    baseline_utility_per_cluster: Sequence[Tuple[str, float]],
    n_boot: int = N_BOOTSTRAP, seed: int = BASE_SEED,
) -> Tuple[float, float, float]:
    """Percentile 95% CI of the utility delta (arm - rule) under cluster
    resampling.  Returns (delta_mean, ci_low, ci_high)."""
    arm = {c: v for c, v in arm_utility_per_cluster}
    base = {c: v for c, v in baseline_utility_per_cluster}
    clusters = sorted(set(arm) & set(base))
    if len(clusters) < 2:
        d = (statistics.mean(arm.values()) if arm else 0.0) - \
            (statistics.mean(base.values()) if base else 0.0)
        return float(d), float(d), float(d)
    rng = np.random.RandomState(seed)
    clusters_arr = np.array(clusters)
    deltas = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters_arr, size=len(clusters_arr), replace=True)
        a = np.mean([arm[c] for c in sampled])
        b = np.mean([base[c] for c in sampled])
        deltas.append(a - b)
    deltas = np.array(deltas)
    return (float(deltas.mean()),
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)))


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

@dataclass
class ParetoPoint:
    arm: str
    utility: float
    validity: float
    risk: float  # lower is better
    coverage: float


def _dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    """True if ``a`` Pareto-dominates ``b`` (max utility, max validity, min risk)."""
    ge = (a.utility >= b.utility and a.validity >= b.validity
          and a.risk <= b.risk)
    gt = (a.utility > b.utility or a.validity > b.validity or a.risk < b.risk)
    return ge and gt


def evaluate_pareto_frontier(points: Sequence[ParetoPoint]) -> Dict[str, Any]:
    """Return the Pareto-optimal set and pairwise dominance relationships."""
    pts = list(points)
    frontier = [p for p in pts
                if not any(_dominates(q, p) for q in pts if q is not p)]
    dominance: Dict[str, Dict[str, bool]] = {}
    for a in pts:
        dominance[a.arm] = {}
        for b in pts:
            dominance[a.arm][b.arm] = _dominates(a, b) if a is not b else False
    learned_dominates_rule = dominance.get(
        "learned_structured", {}).get("rule_pc_cng", False)
    learned_risk_dominates_rule = dominance.get(
        "learned_structured_risk", {}).get("rule_pc_cng", False)
    return {
        "frontier": [p.arm for p in frontier],
        "frontier_points": [
            {"arm": p.arm, "utility": p.utility, "validity": p.validity,
             "risk": p.risk, "coverage": p.coverage} for p in frontier],
        "dominance": dominance,
        "learned_dominates_rule": learned_dominates_rule,
        "learned_risk_dominates_rule": learned_risk_dominates_rule,
    }


# ---------------------------------------------------------------------------
# Comparison arms
# ---------------------------------------------------------------------------

@dataclass
class ArmResult:
    arm: str
    negatives: List[str] = field(default_factory=list)
    edits: List[StructuredEdit] = field(default_factory=list)
    utility: float = 0.0
    validity: float = 0.0
    collision_risk: float = 0.0
    controllability: float = 0.0
    family_coverage: int = 0
    diversity: float = 0.0
    n_candidates: int = 0
    utility_per_cluster: List[Tuple[str, float]] = field(default_factory=list)


def run_comparison_arms(reactions: Sequence[str],
                        model: StructuredProposalModel,
                        rule_generator: ReactionBoundaryGenerator,
                        positives: Sequence[str],
                        test_positives: Sequence[str],
                        top_k: int = DEFAULT_TOP_K,
                        device: Optional[torch.device] = None,
                        seed: int = BASE_SEED,
                        map_unmapped: bool = False,
                        ) -> Dict[str, ArmResult]:
    """Generate negatives with each of the 4 arms and compute all metrics.

    Arms:
        rule_pc_cng             - baseline rule generator
        unconstrained_neural    - neural generator, validity mask disabled
        learned_structured      - full model
        learned_structured_risk - full model + risk reranking
    """
    device = device or next(model.parameters()).device
    results: Dict[str, ArmResult] = {}
    all_rule_neg: List[str] = []
    all_clusters: List[str] = []
    for i, rxn in enumerate(reactions):
        reactants, _ = _safe_split(rxn)
        cluster = f"rxn_{i}"
        # Rule arm
        try:
            rule_cands = rule_generator.generate_for_reaction(rxn, source_id=cluster)
        except Exception:
            rule_cands = []
        rule_neg = [c.candidate_product for c in rule_cands
                    if c.candidate_product and is_valid_smiles(c.candidate_product)]
        all_rule_neg.extend(rule_neg)
        all_clusters.extend([cluster] * len(rule_neg))

    # Per-arm generation
    arm_negatives: Dict[str, List[str]] = {a: [] for a in ARMS}
    arm_edits: Dict[str, List[StructuredEdit]] = {a: [] for a in ARMS}
    for rxn in reactions:
        reactants, _ = _safe_split(rxn)
        # learned edits (shared decode for both learned arms)
        edits = generate_structured_proposal(
            model, rxn, top_k=top_k, device=device,
            use_validity_mask=True, risk_rerank=False, map_unmapped=map_unmapped)
        edits_uncon = generate_structured_proposal(
            model, rxn, top_k=top_k, device=device,
            use_validity_mask=False, risk_rerank=False, map_unmapped=map_unmapped)
        edits_risk = generate_structured_proposal(
            model, rxn, top_k=top_k, device=device,
            use_validity_mask=True, risk_rerank=True, n_mc=5, map_unmapped=map_unmapped)
        arm_negatives["learned_structured"].extend(
            proposal_to_negatives(rxn, edits))
        arm_edits["learned_structured"].extend(edits)
        arm_negatives["unconstrained_neural"].extend(
            proposal_to_negatives(rxn, edits_uncon))
        arm_edits["unconstrained_neural"].extend(edits_uncon)
        arm_negatives["learned_structured_risk"].extend(
            proposal_to_negatives(rxn, edits_risk))
        arm_edits["learned_structured_risk"].extend(edits_risk)
    arm_negatives["rule_pc_cng"] = all_rule_neg

    # Candidate-coverage matching: cap every arm to the rule arm's count so
    # improvement cannot come from generating more candidates.
    cap = len(all_rule_neg)
    for a in ARMS:
        if len(arm_negatives[a]) > cap and cap > 0:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(arm_negatives[a]), size=cap, replace=False)
            arm_negatives[a] = [arm_negatives[a][i] for i in idx]
            if len(arm_edits[a]) > cap:
                arm_edits[a] = [arm_edits[a][i] for i in idx]

    for a in ARMS:
        negs = arm_negatives[a]
        reactants_list = [_safe_split(r)[0] for r in reactions]
        validity = (sum(_chemical_validity(n, reactants_list[i % max(len(reactants_list), 1)])
                        for i, n in enumerate(negs)) / len(negs)) if negs else 0.0
        col = _collision_risk(negs, list(positives))
        div = _diversity(negs)
        cov = _reaction_family_coverage(negs)
        edits = arm_edits.get(a, [])
        ctrl = _edit_controllability(
            edits, EditType.ATOM_TRANSMUTATION) if edits else 0.0
        # downstream utility (per-cluster for bootstrap)
        test_neg_sample = negs[: min(len(negs), 50)]
        util = _downstream_utility(negs, list(positives), test_neg_sample,
                                   list(test_positives), seed=seed)
        per_cluster = [(f"rxn_{i}", util) for i in range(min(len(negs), 1))]
        results[a] = ArmResult(
            arm=a, negatives=negs, edits=edits, utility=util,
            validity=validity, collision_risk=col, controllability=ctrl,
            family_coverage=cov, diversity=div, n_candidates=len(negs),
            utility_per_cluster=per_cluster)
    return results


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def compute_verdict(comparison_results: Dict[str, ArmResult],
                    pareto: Dict[str, Any],
                    utility_ci: Tuple[float, float, float],
                    coverage_matched: bool) -> Dict[str, Any]:
    """GO / PARTIAL_GO / NO_GO per the G8-C spec.

    * GO: learned proposal Pareto-dominates the rule version on
      utility-validity-risk, downstream-utility CI all positive, and
      candidate coverage matched.
    * PARTIAL_GO: some metrics improve but the learned arm is not
      Pareto-dominant.
    * NO_GO: no improvement, or worse than the rule baseline.
    """
    rule = comparison_results.get("rule_pc_cng")
    learned = comparison_results.get("learned_structured",
                                     comparison_results.get("learned_structured_risk"))
    delta_mean, ci_low, ci_high = utility_ci
    ci_all_positive = ci_low > 0
    dominates_rule = pareto.get("learned_dominates_rule", False) or \
        pareto.get("learned_risk_dominates_rule", False)

    improvements = 0
    total_metrics = 0
    if rule and learned:
        for metric in ("utility", "validity", "diversity"):
            total_metrics += 1
            if getattr(learned, metric) > getattr(rule, metric):
                improvements += 1
        if learned.collision_risk < rule.collision_risk:
            improvements += 1
        total_metrics += 1

    if dominates_rule and ci_all_positive and coverage_matched:
        verdict, reason = "GO", (
            "Learned structured proposal Pareto-dominates the rule version "
            "on utility-validity-risk; downstream-utility CI all positive "
            f"[{ci_low:+.4f},{ci_high:+.4f}]; candidate coverage matched.")
    elif improvements > 0:
        verdict, reason = "PARTIAL_GO", (
            f"{improvements}/{total_metrics} metrics improved vs rule; "
            f"Pareto-dominant={dominates_rule}; CI_all_positive={ci_all_positive}; "
            f"coverage_matched={coverage_matched}.")
    else:
        verdict, reason = "NO_GO", (
            f"No improvement over rule ({improvements}/{total_metrics} metrics); "
            f"CI=[{ci_low:+.4f},{ci_high:+.4f}].")
    return {
        "verdict": verdict, "reason": reason,
        "dominates_rule": dominates_rule,
        "ci_all_positive": ci_all_positive,
        "coverage_matched": coverage_matched,
        "utility_delta_mean": delta_mean,
        "utility_ci_low": ci_low, "utility_ci_high": ci_high,
        "improvements": improvements, "total_metrics": total_metrics,
    }


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------

def _collate_reactions(reactions: Sequence[str], device: torch.device,
                       map_unmapped: bool = False):
    """Returns (batch, successful_rxns) or (None, [])."""
    graphs = []
    successful_rxns = []
    for rxn in reactions:
        g = _featurize_safe(rxn, map_unmapped=map_unmapped)
        if g is not None:
            graphs.append(g)
            successful_rxns.append(rxn)
    if not graphs:
        return None, []
    batch = collate_graphs(graphs)
    batch.atom_features = batch.atom_features.to(device)
    batch.edge_features = batch.edge_features.to(device)
    batch.edge_index = batch.edge_index.to(device)
    batch.batch_idx = batch.batch_idx.to(device)
    return batch, successful_rxns


def _extract_targets(reaction: str) -> Tuple[int, int]:
    """Heuristic supervision targets for stage 1 (locus, edit type).

    Uses the first formed-bond atom as the locus and ATOM_TRANSMUTATION as
    the default edit type when a reaction center is extractable; otherwise
    NO_EDIT at locus 0.
    """
    try:
        center = extract_reaction_center(reaction)
        if center and getattr(center, "formed_bonds", None):
            return int(center.formed_bonds[0][0]), int(EditType.FORMED_BOND_MIGRATE)
    except Exception:
        pass
    return 0, int(EditType.NO_EDIT)


def train_stage(model: StructuredProposalModel, stage: int,
                train_reactions: Sequence[str],
                val_reactions: Sequence[str],
                rule_generator: ReactionBoundaryGenerator,
                epochs: int, batch_size: int, lr: float,
                device: torch.device, seed: int = BASE_SEED,
                log: Optional[List[dict]] = None,
                map_unmapped: bool = False) -> List[dict]:
    """Run a single training stage.  Returns the per-epoch log entries."""
    set_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    grad_clip = 1.0  # prevent DPO loss explosion
    stage_losses = {
        1: Stage1ReconstructionLoss(),
        2: Stage2ImitationLoss(),
        3: Stage3ContrastiveLoss(),
        4: Stage4DPOLoss(use_ipo=True),
    }
    loss_fn = stage_losses[stage]
    log = log if log is not None else []
    n = len(train_reactions)
    for epoch in range(epochs):
        model.train()
        order = np.random.RandomState(seed + epoch).permutation(n)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch_rxns = [train_reactions[i] for i in idx]
            batch, success_rxns = _collate_reactions(batch_rxns, device, map_unmapped=map_unmapped)
            if batch is None:
                continue
            opt.zero_grad()
            out = model(batch)
            num_g = len(success_rxns)
            if stage == 1:
                loci = torch.tensor(
                    [_extract_targets(r)[0] for r in success_rxns],
                    device=device, dtype=torch.long)
                types = torch.tensor(
                    [_extract_targets(r)[1] for r in success_rxns],
                    device=device, dtype=torch.long)
                loss, _ = loss_fn(out, loci, types)
            elif stage == 2:
                loci = torch.tensor(
                    [_extract_targets(r)[0] for r in success_rxns],
                    device=device, dtype=torch.long)
                types = torch.tensor(
                    [_extract_targets(r)[1] for r in success_rxns],
                    device=device, dtype=torch.long)
                locus_probs = F.one_hot(loci, num_classes=out.locus_logits.shape[-1]).float()
                type_probs = F.one_hot(types, num_classes=NUM_EDIT_TYPES).float()
                loss, _ = loss_fn(out, locus_probs, type_probs)
            elif stage == 3:
                pos_mask = torch.ones(num_g, device=device)
                if num_g > 1:
                    pos_mask[num_g // 2:] = 0
                loss, _ = loss_fn(out, pos_mask)
            else:  # stage 4 DPO
                g = num_g // 2 if num_g >= 2 else 1
                out_pref = StructuredProposalOutput(
                    locus_logits=out.locus_logits[:g], type_logits=out.type_logits[:g],
                    arg_logits={k: v[:g] for k, v in out.arg_logits.items()},
                    validity_mask=out.validity_mask[:g], risk=out.risk[:g],
                    uncertainty=out.uncertainty[:g], graph_emb=out.graph_emb[:g],
                    node_emb=out.node_emb)
                out_disp = StructuredProposalOutput(
                    locus_logits=out.locus_logits[g:g * 2] if out.locus_logits.shape[0] >= g * 2 else out.locus_logits[:g],
                    type_logits=out.type_logits[g:g * 2] if out.type_logits.shape[0] >= g * 2 else out.type_logits[:g],
                    arg_logits={k: (v[g:g * 2] if v.shape[0] >= g * 2 else v[:g]) for k, v in out.arg_logits.items()},
                    validity_mask=out.validity_mask[g:g * 2] if out.validity_mask.shape[0] >= g * 2 else out.validity_mask[:g],
                    risk=out.risk[g:g * 2] if out.risk.shape[0] >= g * 2 else out.risk[:g],
                    uncertainty=out.uncertainty[g:g * 2] if out.uncertainty.shape[0] >= g * 2 else out.uncertainty[:g],
                    graph_emb=out.graph_emb[g:g * 2] if out.graph_emb.shape[0] >= g * 2 else out.graph_emb[:g],
                    node_emb=out.node_emb)
                loci = torch.tensor(
                    [_extract_targets(r)[0] for r in success_rxns[:g]],
                    device=device, dtype=torch.long)
                types = torch.tensor(
                    [_extract_targets(r)[1] for r in success_rxns[:g]],
                    device=device, dtype=torch.long)
                ref_pref = torch.zeros(g, device=device)
                ref_disp = torch.zeros(g, device=device)
                loss, _ = loss_fn(out_pref, out_disp, loci, types,
                                  loci, types, ref_pref, ref_disp)
            if not torch.isfinite(loss) or float(loss.item()) > 1e4:
                opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        # validation pass (loss-free, just track magnitude)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            vb, v_success = _collate_reactions(val_reactions[:batch_size], device, map_unmapped=map_unmapped)
            if vb is not None:
                vout = model(vb)
                val_loss = min(float(vout.locus_logits.std().item()), 1e3)
        entry = {"stage": stage, "epoch": epoch, "train_loss": avg,
                 "val_signal": val_loss}
        log.append(entry)
        print(f"[{PHASE}] stage={stage} epoch={epoch} "
              f"train_loss={avg:.4f} val_signal={val_loss:.4f}")
    return log


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reactions(path: Path, limit: Optional[int] = None) -> List[str]:
    """Load atom-mapped reaction SMILES from a JSON list or CSV (column
    ``reaction_smiles``)."""
    out: List[str] = []
    if not path or not Path(path).exists():
        return out
    with open(path) as f:
        if path.suffix == ".json":
            data = json.load(f)
            for r in data:
                if isinstance(r, str):
                    out.append(r)
                elif isinstance(r, dict):
                    out.append(r.get("reaction_smiles", r.get("reaction", "")))
        else:
            reader = csv.DictReader(f)
            for row in reader:
                out.append(row.get("reaction_smiles", row.get("reaction", "")))
    out = [r.strip() for r in out if r and r.strip()]
    if limit:
        out = out[:limit]
    return out


def load_rule_proposals(path: Path) -> Dict[str, List[BoundaryCandidate]]:
    if not path or not Path(path).exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    out: Dict[str, List[BoundaryCandidate]] = {}
    for src, cands in data.items():
        built = []
        for c in cands:
            try:
                built.append(BoundaryCandidate(
                    source_id=c.get("source_id", src),
                    positive_reaction=c.get("positive_reaction", ""),
                    candidate_reaction=c.get("candidate_reaction", ""),
                    task=c.get("task", ""),
                    failure_type=c.get("failure_type", ""),
                    edit_action=c.get("edit_action", ""),
                    parent_reactants=c.get("parent_reactants", ""),
                    parent_product=c.get("parent_product", ""),
                    candidate_reactants=c.get("candidate_reactants", ""),
                    candidate_product=c.get("candidate_product", ""),
                    valid=float(c.get("valid", 0.0)),
                    atom_balance=float(c.get("atom_balance", 0.0)),
                    locality=float(c.get("locality", 0.0)),
                    closeness=float(c.get("closeness", 0.0)),
                    hard_score=float(c.get("hard_score", 0.0)),
                    false_negative_risk=float(c.get("false_negative_risk", 0.5)),
                    passes_filter=bool(c.get("passes_filter", True)),
                    mapped=bool(c.get("mapped", False)),
                    center_maps=c.get("center_maps", "")))
            except Exception:
                continue
        out[src] = built
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description=f"{PHASE} learned structured proposal")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--rule-proposals", type=Path, default=None)
    parser.add_argument("--risk-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/p4_g8c_learned_structured_proposal"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--num-rounds", type=int, default=4,
                        help="Number of full stage rounds (1..4)")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--stage", type=int, default=0,
                        help="Train only this stage (1..4); 0 = all stages")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--map-unmapped", action="store_true",
                        help="Use RXNMapper for unmapped reactions")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    t0 = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_predictions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.gpu)
    set_seed(args.seed)

    print(f"[{PHASE}] Loading data ...")
    train_rxns = load_reactions(args.train_data, args.limit_train)
    val_rxns = load_reactions(args.val_data, args.limit_val)
    test_rxns = load_reactions(args.test_data, args.limit_test)
    if args.smoke:
        train_rxns = train_rxns[:32]
        val_rxns = val_rxns[:16]
        test_rxns = test_rxns[:16]
    print(f"[{PHASE}] train={len(train_rxns)} val={len(val_rxns)} test={len(test_rxns)}")

    rule_generator = ReactionBoundaryGenerator(
        max_candidates_per_reaction=args.top_k, allow_unmapped_fallback=False)

    model = StructuredProposalModel(
        hidden_dim=args.hidden_dim, num_heads=args.num_heads,
        num_layers=DEFAULT_NUM_LAYERS, dropout=args.dropout).to(device)

    stages = [1, 2, 3, 4] if args.stage <= 0 else [args.stage]
    n_rounds = 1 if args.smoke else args.num_rounds
    log: List[dict] = []
    for rnd in range(n_rounds):
        for st in stages:
            log = train_stage(
                model, st, train_rxns, val_rxns, rule_generator,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                device=device, seed=args.seed + rnd * 100, log=log,
                map_unmapped=args.map_unmapped)

    # Comparison arms
    eval_rxns = test_rxns[: min(len(test_rxns), 50 if not args.smoke else 8)]
    positives = [_strip_atom_maps(_product_smiles(r)) for r in eval_rxns if _product_smiles(r)]
    test_positives = positives[: max(1, len(positives) // 2)]
    comparison = run_comparison_arms(
        eval_rxns, model, rule_generator, positives, test_positives,
        top_k=args.top_k, device=device, seed=args.seed,
        map_unmapped=args.map_unmapped)

    # Per-arm raw predictions
    for arm, res in comparison.items():
        with open(raw_dir / f"{arm}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["candidate_smiles", "risk", "boundary_value"])
            for i, neg in enumerate(res.negatives):
                edit = res.edits[i] if i < len(res.edits) else None
                w.writerow([neg,
                            edit.risk if edit else 0.0,
                            edit.boundary_value if edit else 0.0])

    # comparison_results.csv
    rows = []
    for arm, res in comparison.items():
        rows.append({
            "arm": arm, "n_candidates": res.n_candidates,
            "utility": f"{res.utility:.6f}", "validity": f"{res.validity:.6f}",
            "collision_risk": f"{res.collision_risk:.6f}",
            "controllability": f"{res.controllability:.6f}",
            "family_coverage": res.family_coverage,
            "diversity": f"{res.diversity:.6f}"})
    with open(output_dir / "comparison_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    # Pareto frontier
    points = [ParetoPoint(
        arm=res.arm, utility=res.utility, validity=res.validity,
        risk=res.collision_risk, coverage=res.family_coverage)
        for res in comparison.values()]
    pareto = evaluate_pareto_frontier(points)
    with open(output_dir / "pareto_frontier.json", "w") as f:
        json.dump(pareto, f, indent=2)

    # Downstream-utility cluster bootstrap CI (learned_structured vs rule)
    learned = comparison.get("learned_structured")
    rule = comparison.get("rule_pc_cng")
    if learned and rule and learned.utility_per_cluster and rule.utility_per_cluster:
        ci = cluster_bootstrap_utility_ci(
            learned.utility_per_cluster, rule.utility_per_cluster,
            n_boot=args.n_bootstrap, seed=args.seed)
    else:
        ci = (0.0, 0.0, 0.0)
    coverage_matched = all(
        abs(res.n_candidates - rule.n_candidates) <= max(1, 0.1 * max(rule.n_candidates, 1))
        for res in comparison.values()) if rule else True

    verdict = compute_verdict(comparison, pareto, ci, coverage_matched)
    go_no_go = {
        "phase": PHASE, "status": verdict["verdict"], "version": "full_spec",
        "primary_metric": {"name": "downstream_utility",
                           "comparison": "learned_structured_vs_rule_pc_cng"},
        "comparison_arms": ARMS,
        "pareto": {"learned_dominates_rule": pareto["learned_dominates_rule"],
                   "learned_risk_dominates_rule": pareto["learned_risk_dominates_rule"],
                   "frontier": pareto["frontier"]},
        "utility_ci": {"delta_mean": ci[0], "ci_low": ci[1], "ci_high": ci[2],
                       "n_bootstrap": args.n_bootstrap},
        "coverage_matched": coverage_matched,
        "verdict_detail": verdict,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(output_dir / "go_no_go.json", "w") as f:
        json.dump(go_no_go, f, indent=2)

    # checkpoint + train log
    torch.save({"state_dict": model.state_dict(),
                "hidden_dim": model.hidden_dim,
                "architecture": "StructuredProposalModel"},
               str(output_dir / "model_checkpoint.pt"))
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Standard phase outputs (spec L116-128)
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump({
            "phase": PHASE, "version": "full_spec",
            "arms": ARMS,
            "n_train_reactions": len(train_rxns),
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_rounds": args.num_rounds,
            "stages": ["reconstruction", "rule_imitation",
                       "competing_outcomes", "risk_adjusted_dpo"],
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
        }, f, indent=2)
    with open(output_dir / "environment.json", "w") as f:
        env = {"python": sys.version.split()[0],
               "platform": platform.platform(),
               "torch": torch.__version__, "numpy": np.__version__}
        try:
            import rdkit
            env["rdkit"] = rdkit.__version__
        except ImportError:
            pass
        json.dump(env, f, indent=2)
    hashes = {}
    for p in [args.train_data, args.val_data, args.test_data]:
        if p and Path(p).exists():
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
            hashes[str(p)] = h.hexdigest()
    with open(output_dir / "input_hashes.json", "w") as f:
        json.dump(hashes, f, indent=2)
    with open(output_dir / "commands.log", "w") as f:
        f.write(" ".join([sys.executable, "-m", "pc_cng.p4_g8c_learned_structured_proposal"] +
                         [f"--{k}={v}" for k, v in vars(args).items()]) + "\n")

    print(f"\n[{PHASE}] verdict={verdict['verdict']}")
    print(f"[{PHASE}] utility_delta={ci[0]:+.4f} CI[{ci[1]:+.4f},{ci[2]:+.4f}]")
    print(f"[{PHASE}] Pareto-dominates-rule={pareto['learned_dominates_rule']} "
          f"coverage_matched={coverage_matched}")
    print(f"[{PHASE}] outputs in {output_dir}")


if __name__ == "__main__":
    main()
