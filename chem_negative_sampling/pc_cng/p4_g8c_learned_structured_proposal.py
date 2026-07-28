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
    _build_product_graph,
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
    _candidate_anchor_atoms,
    _looks_like_transfer_fragment,
    build_edit_candidate_groups,
    move_formed_bond_in_product,
)
from .g8c_data_preparation import (
    DEFAULT_COLLISION_REVIEW,
    DEFAULT_EXPERT_FORMS,
    DEFAULT_HTE_PARQUET,
    extract_real_edit_targets,
    load_g8c_training_data,
)
from .g8c_action_schema import EditType, GENERATIVE_EDIT_TYPES, NUM_EDIT_TYPES
from .atom_mapped_graph_edit import (
    ReactionCenterEdit,
    extract_reaction_center,
    has_atom_mapping,
)
from .chem_utils import (
    atom_balance_score,
    atom_count_distance,
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
FORMAL_VALIDATION_THRESHOLDS = {
    "edit_locus_accuracy_min": 0.20,
    "edit_type_accuracy_min": 0.50,
    "valid_edit_rate_min": 0.95,
    "candidate_coverage_min": 0.80,
    "fnr_ece_max": 0.15,
    "reward_max_abs_log_ratio_max": 5.0,
    "reward_action_type_entropy_min": 0.50,
}


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
        # Reconstruction and proposal actions have different semantics.
        # A real product-forming BOND_FORM target must not compete in the same
        # head with rule-generator NOT_APPLICABLE / counterfactual edit types.
        self.reconstruction_locus_pointer = EditLocusPointer(hidden_dim)
        self.reconstruction_type_classifier = EditTypeClassifier(
            hidden_dim,
            dropout,
        )
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
        sizes = torch.bincount(batch_idx, minlength=num_graphs)
        offsets = torch.cat(
            [sizes.new_zeros(1), sizes.cumsum(0)[:-1]]
        )
        local = locus_index.to(device=node_emb.device, dtype=torch.long)
        local = torch.maximum(local, torch.zeros_like(local))
        local = torch.minimum(local, sizes.clamp(min=1) - 1)
        return node_emb[offsets + local]

    def _migration_logits(
        self,
        node_emb: torch.Tensor,
        batch_idx: torch.Tensor,
        query: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        sizes = torch.bincount(batch_idx, minlength=num_graphs)
        max_len = int(sizes.max().item()) if num_graphs else 0
        logits = node_emb.new_full((num_graphs, max_len), -1e9)
        order = torch.argsort(batch_idx, stable=True)
        sorted_batch = batch_idx[order]
        offsets = torch.arange(
            len(order),
            device=node_emb.device,
        ) - torch.cat([sizes.new_zeros(1), sizes.cumsum(0)[:-1]])[sorted_batch]
        scores = (
            node_emb[order] * query[sorted_batch]
        ).sum(-1) / math.sqrt(self.hidden_dim)
        logits[sorted_batch, offsets] = scores
        return logits

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
                action_head: str = "proposal",
                ) -> StructuredProposalOutput:
        node_emb, graph_emb = self.transformer(batch)
        num_graphs = len(batch.graphs)
        batch_idx = batch.batch_idx
        center_ctx = self.center_encoder(
            node_emb, batch_idx, num_graphs, center_summary)
        graph_ctx = graph_emb + center_ctx
        if action_head == "reconstruction":
            locus_pointer = self.reconstruction_locus_pointer
            type_classifier = self.reconstruction_type_classifier
        elif action_head == "proposal":
            locus_pointer = self.locus_pointer
            type_classifier = self.type_classifier
        else:
            raise ValueError(f"unknown action_head={action_head!r}")
        locus_logits = locus_pointer(
            node_emb, graph_ctx, batch_idx, num_graphs)
        locus_idx = locus_index if locus_index is not None else \
            locus_logits.argmax(dim=-1)
        locus_emb = self._gather_locus_emb(node_emb, batch_idx, num_graphs, locus_idx)
        type_logits = type_classifier(graph_ctx, locus_emb)
        arg_logits = self.arg_decoder(graph_ctx, locus_emb)
        arg_logits["migrate_logits"] = self._migration_logits(
            node_emb,
            batch_idx,
            arg_logits["migrate_query"],
            num_graphs,
        )
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
                if not bool((tgt != -100).any()):
                    continue
                arg_total = arg_total + F.cross_entropy(
                    logits,
                    tgt,
                    ignore_index=-100,
                )
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
                locus_supervision_mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        log_locus = F.log_softmax(out.locus_logits, dim=-1)
        log_type = F.log_softmax(out.type_logits, dim=-1)
        locus_kl = (rule_locus_probs *
                    (rule_locus_probs.clamp(min=1e-9).log() - log_locus))
        locus_per_example = locus_kl.sum(-1)
        if locus_supervision_mask is None:
            locus_kl = locus_per_example.mean()
        else:
            mask = locus_supervision_mask.float()
            locus_kl = (locus_per_example * mask).sum() / mask.sum().clamp(min=1)
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
    # Pre-applied edited product SMILES (map-stripped).  Set by the
    # deterministic exhaustive generator so callers do NOT need to re-apply
    # the edit via ``_apply_structured_edit`` (which is fragile when the
    # reaction is unmapped or the product has multiple parts - the graph
    # atom indices and the re-parsed product mol indices can disagree).
    applied_product: Optional[str] = None


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
    if edit.edit_type in {EditType.NOT_APPLICABLE, EditType.BOND_FORM}:
        return None
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
        if edit.edit_type == EditType.BOND_BREAK:
            if edit.locus >= mol.GetNumAtoms():
                return None
            atom = mol.GetAtomWithIdx(int(edit.locus))
            bonds = list(atom.GetBonds())
            if edit.migrate_target is not None:
                bonds = [
                    bond for bond in bonds
                    if bond.GetOtherAtomIdx(int(edit.locus))
                    == int(edit.migrate_target)
                ]
            if not bonds:
                return None
            target = bonds[0]
            rw = Chem.RWMol(mol)
            rw.RemoveBond(target.GetBeginAtomIdx(), target.GetEndAtomIdx())
            edited_mol = rw.GetMol()
            Chem.SanitizeMol(edited_mol)
            edited = _strip_atom_maps(Chem.MolToSmiles(edited_mol))
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
_MAP_RESULT_CACHE: Dict[str, Optional[str]] = {}


def _get_mapper() -> Any:
    global _MAPPER_CACHE
    if _MAPPER_CACHE is None:
        from .reaction_boundary_generator import RXNMapperAdapter
        _MAPPER_CACHE = RXNMapperAdapter()
    return _MAPPER_CACHE


def _map_cached(mapper: Any, reaction_smiles: str) -> Optional[str]:
    """map_reaction with a process-level result cache (RXNMapper is the
    runtime bottleneck when enumerating candidates for thousands of
    reactions; the same reaction is queried by both the union-candidate
    pool and the main-arm training generator)."""
    if reaction_smiles in _MAP_RESULT_CACHE:
        return _MAP_RESULT_CACHE[reaction_smiles]
    try:
        mapped = mapper.map_reaction(reaction_smiles)
    except Exception:
        mapped = None
    _MAP_RESULT_CACHE[reaction_smiles] = mapped
    return mapped


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


# ---------------------------------------------------------------------------
# Deterministic exhaustive proposal generation (v3, 2026-07-27)
# ---------------------------------------------------------------------------
#
# Root cause of the v2 learned-candidate collapse (0.13 valid edits/reaction
# on author_lab): the sampling decoder (1) forced argmax arguments -> duplicate
# edits killed by dedup, (2) required atom mapping + a non-empty formed-bond
# set via ``featurize_atom_mapped_reaction`` -> zero candidates for unmapped
# USPTO/NI reactions and for reactions without formed bonds, and (3) suffered
# graph/product atom-index misalignment on multi-part products.
#
# The exhaustive generator replaces sampling with deterministic, model-ranked
# enumeration of the chemically-sensible edit space:
#   * featurise the PRODUCT graph directly (no mapping / formed-bond
#     requirement), so any parseable reaction yields model scores;
#   * enumerate transmutation / bond-order / bond-migration edits on the SAME
#     RDKit mol used for featurisation (index alignment is trivial);
#   * keep edits that sanitise and differ from the original product;
#   * rank by the model's joint score (locus + type + argument logits) and
#     attach per-locus risk/uncertainty from the risk head.
#
# Each returned edit carries ``applied_product`` so callers never re-apply
# edits through the fragile string round-trip.

# Halogen / chalcogen / pnictogen swaps aligned with the rule generator's
# ATOM_TRANSMUTATIONS (chemically valence-plausible subset).
_EXHAUSTIVE_TRANSMUTATIONS: Tuple[Tuple[int, int], ...] = (
    (17, 35), (35, 17), (53, 35), (53, 17), (9, 17),
    (7, 8), (8, 7), (8, 16), (16, 8),
)

# Safety cap: products with more atoms than this are restricted to the
# model's top-scoring loci before enumeration (bonds are always enumerated;
# they are cheap).
_EXHAUSTIVE_MAX_LOCI = 48


def _sanitize_edited_mol(rw_mol) -> Optional[str]:
    """Sanitise an RWMol edit product; return map-stripped canonical SMILES."""
    try:
        mol = rw_mol.GetMol()
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        smi = _strip_atom_maps(smi)
        return smi if smi and is_valid_smiles(smi) else None
    except Exception:
        return None


def generate_structured_proposal_exhaustive(
        model: StructuredProposalModel,
        reaction_smiles: str,
        top_k: int = DEFAULT_TOP_K,
        device: Optional[torch.device] = None,
        use_validity_mask: bool = True,
        risk_rerank: bool = False,
        n_mc: int = 0,
        map_unmapped: bool = True,
        require_atom_balance: bool = False,
        balance_eps: float = 0.011,
        balance_dist_slack: Optional[int] = None,
        ) -> List[StructuredEdit]:
    """Deterministic, model-ranked enumeration of the structured edit space.

    Unlike :func:`generate_structured_proposal` (ancestral sampling), this
    generator guarantees coverage: every chemically-sensible edit is tried
    exactly once, valid unique products are kept, and the model only RANKS
    them.  Works for unmapped reactions and products without formed bonds.

    When ``require_atom_balance`` is set, candidates whose atom-balance
    vs the reactants degrades beyond the tolerance relative to the TRUE
    product are rejected.  Two tolerance modes:

      * ``balance_dist_slack`` set (preferred, v4): L1 atom-count distance
        criterion - candidate rejected when
        ``dist(reactants, cand) > dist(reactants, true) + slack``.
        slack=2 admits exactly one atom transmutation while foreign
        products stay excluded, independent of system size.
      * legacy ratio mode (``balance_dist_slack=None``): candidate
        rejected when its atom-balance SCORE falls more than
        ``balance_eps`` below the true product's score.  WARNING: on
        large multi-component systems the ratio tolerance silently kills
        ALL single transmutations (L1 +2 on ~150 atoms exceeds eps).

    This enforces stoichiometric constructibility relative to the
    observed outcome (leaving-group losses are tolerated because the
    reference is the true product, not exact equality): connectivity
    edits (bond migration / bond order) pass, product swaps do not.  The
    result is a boundary negative with no composition-mismatch shortcut -
    the signal a shuffled-parent control exploits.
    """
    device = device or next(model.parameters()).device
    model.eval()
    if Chem is None:
        return []

    # --- 1. Parse reaction (map opportunistically for richer features) -----
    rxn = reaction_smiles
    if map_unmapped and not has_atom_mapping(rxn):
        mapped = _map_cached(_get_mapper(), rxn)
        if mapped:
            rxn = mapped
    try:
        reactants, agents, product = split_reaction(rxn)
    except ValueError:
        return []
    mol = Chem.MolFromSmiles(product) if product else None
    if mol is None or mol.GetNumAtoms() == 0:
        return []
    # Mapped SMILES carry explicit bracket H counts ([CH3], [cH], [nH]).
    # Left as-is, every valence-changing edit (bond-order change, bond
    # migration) fails sanitisation because the explicit-H bookkeeping is
    # stale after the edit.  Reset H bookkeeping and refresh the property
    # cache so RDKit recomputes implicit Hs both for featurisation and for
    # post-edit sanitisation.  Atom map numbers are preserved.
    for _atom in mol.GetAtoms():
        _atom.SetNoImplicit(False)
        _atom.SetNumExplicitHs(0)
    mol.UpdatePropertyCache(strict=False)
    n_atoms = mol.GetNumAtoms()
    orig_product = _strip_atom_maps(_product_smiles(reaction_smiles))
    # Balance reference: the TRUE product's own score vs the ORIGINAL
    # reactants (string-based; leaving-group loss already priced in).
    # NORMALISATION (v4.1): strip atom maps from the reactants side too.
    # Datasets like HiTEA ship pre-mapped reactions whose bracket-H tokens
    # ([CH3], [cH], [NH]) are counted by the regex tokenizer, while the
    # (map-stripped) candidate products carry none - the asymmetric H
    # bookkeeping inflated d_neg and systematically excluded valid
    # boundary edits in the fixed-pool eligibility check.
    orig_reactants = _strip_atom_maps(reaction_smiles.split(">")[0])
    bal_floor = -1.0
    dist_ceiling: Optional[int] = None
    if require_atom_balance:
        if balance_dist_slack is not None:
            # Distance-based (v4): size-independent; slack=2 admits
            # exactly one atom transmutation.
            dist_ceiling = atom_count_distance(orig_reactants,
                                               orig_product) \
                + balance_dist_slack
        else:
            bal_floor = atom_balance_score(orig_reactants, orig_product) \
                - balance_eps

    # --- 2. Featurise the product graph directly ---------------------------
    map_to_idx = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()
                  if a.GetAtomMapNum()}
    atom_features, edge_index, edge_features, atom_map_nums = \
        _build_product_graph(mol, map_to_idx)
    graph = ReactionGraphData(
        atom_features=atom_features, edge_index=edge_index,
        edge_features=edge_features, atom_map_nums=atom_map_nums,
        true_anchor_idx=0,
        candidate_anchor_indices=list(range(n_atoms)),
        source_id="exhaustive", pair_id="exhaustive|0",
        mapped_reaction=rxn, reactants=reactants, product=product,
        fragment_map=0, true_anchor_map=0, atom_map_to_idx=map_to_idx)
    batch = collate_graphs([graph])
    batch.atom_features = batch.atom_features.to(device)
    batch.edge_features = batch.edge_features.to(device)
    batch.edge_index = batch.edge_index.to(device)
    batch.batch_idx = batch.batch_idx.to(device)
    with torch.no_grad():
        out = model(batch, hard_validity_mask=None)

    locus_logits = out.locus_logits[0][:n_atoms]
    type_logits = out.type_logits[0]
    atom_logits = out.arg_logits["atom_logits"][0]
    bond_logits = out.arg_logits["bond_logits"][0]
    migrate_logits = out.arg_logits["migrate_logits"][0][:n_atoms]
    if use_validity_mask and out.validity_mask.numel() > 0:
        keep = out.validity_mask[0][:n_atoms]  # [n_atoms, T]
        keep_log = keep.clamp(min=1e-3).log()
    else:
        keep_log = torch.zeros(n_atoms, NUM_EDIT_TYPES,
                               device=locus_logits.device)

    # Per-locus risk / uncertainty (single graph: expand graph context).
    with torch.no_grad():
        graph_ctx = out.graph_emb.expand(n_atoms, -1)
        locus_embs = out.node_emb[:n_atoms]
        risks_t, uncs_t = model.risk_head(graph_ctx, locus_embs)
        if n_mc > 0:
            risks_t, uncs_t = model.risk_head.mc_estimate(
                out.graph_emb.expand(n_atoms, -1), locus_embs,
                n_samples=n_mc)
    risks = risks_t.detach().cpu().numpy()
    uncs = uncs_t.detach().cpu().numpy()
    locus_np = locus_logits.detach().cpu().numpy()
    type_np = type_logits.detach().cpu().numpy()
    atom_np = atom_logits.detach().cpu().numpy()
    bond_np = bond_logits.detach().cpu().numpy()
    migrate_np = migrate_logits.detach().cpu().numpy()
    keep_np = keep_log.detach().cpu().numpy()

    anchor_vocab = sorted(ANCHOR_ATOMIC_NUMS)
    boundary_all = (1.0 - risks) * (1.0 / (1.0 + uncs))

    # --- 3. Locus restriction for very large products ----------------------
    if n_atoms > _EXHAUSTIVE_MAX_LOCI:
        top_loci = set(np.argsort(-locus_np)[:_EXHAUSTIVE_MAX_LOCI].tolist())
    else:
        top_loci = set(range(n_atoms))

    # --- 4. Enumerate edits on the SAME mol --------------------------------
    seen_products = {orig_product}
    scored: List[Tuple[float, float, StructuredEdit]] = []

    def _register(locus: int, edit_type: EditType, prod_smi: Optional[str],
                  atom_arg: Optional[int] = None,
                  bond_arg: Optional[int] = None,
                  migrate_target: Optional[int] = None,
                  extra_score: float = 0.0) -> None:
        if prod_smi is None or prod_smi in seen_products:
            return
        if require_atom_balance:
            if dist_ceiling is not None:
                if atom_count_distance(orig_reactants, prod_smi) \
                        > dist_ceiling:
                    return
            elif atom_balance_score(orig_reactants, prod_smi) < bal_floor:
                return
        seen_products.add(prod_smi)
        t = int(edit_type)
        joint = float(locus_np[locus]) + float(type_np[t]) + extra_score
        if t < keep_np.shape[1]:
            joint += float(keep_np[locus, t])
        bv = float(boundary_all[locus])
        scored.append((joint, bv, StructuredEdit(
            locus=int(locus), edit_type=edit_type, atom_arg=atom_arg,
            bond_arg=bond_arg, migrate_target=migrate_target,
            risk=float(risks[locus]), uncertainty=float(uncs[locus]),
            boundary_value=bv, applied_product=prod_smi)))

    # (a) atom transmutation
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in top_loci:
            continue
        z = atom.GetAtomicNum()
        for old_z, new_z in _EXHAUSTIVE_TRANSMUTATIONS:
            if z != old_z:
                continue
            rw = Chem.RWMol(mol)
            rw_atom = rw.GetAtomWithIdx(idx)
            rw_atom.SetAtomicNum(new_z)
            rw_atom.SetFormalCharge(0)
            prod = _sanitize_edited_mol(rw)
            arg = anchor_vocab.index(new_z) if new_z in anchor_vocab else None
            extra = float(atom_np[arg]) if arg is not None and \
                arg < len(atom_np) else 0.0
            _register(idx, EditType.ATOM_TRANSMUTATION, prod,
                      atom_arg=arg, extra_score=extra)

    # (b) bond order change (SINGLE <-> DOUBLE, non-aromatic)
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            continue
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            target, order_val = Chem.BondType.DOUBLE, 2
        elif bt == Chem.BondType.DOUBLE:
            target, order_val = Chem.BondType.SINGLE, 1
        else:
            continue
        b_idx, e_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if b_idx not in top_loci and e_idx not in top_loci:
            continue
        locus = b_idx if locus_np[b_idx] >= locus_np[e_idx] else e_idx
        rw = Chem.RWMol(mol)
        rw.GetBondWithIdx(bond.GetIdx()).SetBondType(target)
        prod = _sanitize_edited_mol(rw)
        barg = BOND_ORDERS.index(order_val) if order_val in BOND_ORDERS else 0
        extra = float(bond_np[barg]) if barg < len(bond_np) else 0.0
        _register(locus, EditType.BOND_ORDER_CHANGE, prod,
                  bond_arg=barg, extra_score=extra)

    # (c) formed-bond migration (substituent relocation -> regioisomer)
    for bond in mol.GetBonds():
        for frag_idx, anchor_idx in ((bond.GetBeginAtomIdx(),
                                      bond.GetEndAtomIdx()),
                                     (bond.GetEndAtomIdx(),
                                      bond.GetBeginAtomIdx())):
            if anchor_idx not in top_loci:
                continue
            frag_atom = mol.GetAtomWithIdx(frag_idx)
            anchor_atom = mol.GetAtomWithIdx(anchor_idx)
            if not _looks_like_transfer_fragment(frag_atom, anchor_atom):
                continue
            targets = _candidate_anchor_atoms(mol, frag_idx, anchor_idx, 5)
            for tgt_atom in (targets or [])[:8]:
                tgt_idx = tgt_atom.GetIdx()
                rw = Chem.RWMol(mol)
                rw.RemoveBond(frag_idx, anchor_idx)
                rw.AddBond(frag_idx, tgt_idx, Chem.BondType.SINGLE)
                prod = _sanitize_edited_mol(rw)
                _register(anchor_idx, EditType.FORMED_BOND_MIGRATE, prod,
                          migrate_target=tgt_idx,
                          extra_score=float(migrate_np[tgt_idx]))

    # (d) bond break (explicit Stage-1 grammar action)
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left not in top_loci and right not in top_loci:
            continue
        locus, partner = (
            (left, right)
            if locus_np[left] >= locus_np[right]
            else (right, left)
        )
        rw = Chem.RWMol(mol)
        rw.RemoveBond(left, right)
        prod = _sanitize_edited_mol(rw)
        extra = float(migrate_np[partner]) if partner < len(migrate_np) else 0.0
        _register(
            locus,
            EditType.BOND_BREAK,
            prod,
            migrate_target=partner,
            extra_score=extra,
        )

    # --- 5. Rank & return ---------------------------------------------------
    if risk_rerank:
        scored.sort(key=lambda item: (-item[1], -item[0]))
    else:
        scored.sort(key=lambda item: (-item[0], -item[1]))
    return [edit for _joint, _bv, edit in scored[:top_k]]


def generate_structured_proposal(model: StructuredProposalModel,
                                 reaction_smiles: str,
                                 top_k: int = DEFAULT_TOP_K,
                                 device: Optional[torch.device] = None,
                                 use_validity_mask: bool = True,
                                 risk_rerank: bool = False,
                                 n_mc: int = 0,
                                 map_unmapped: bool = True,
                                 exhaustive: bool = True,
                                 require_atom_balance: bool = True,
                                 ) -> List[StructuredEdit]:
    """Generate up to ``top_k`` structured edits for a single reaction.

    The action sequence is: select edit locus -> select edit type ->
    select atom/bond arguments -> apply constrained edit.  Risk and
    uncertainty are attached to each edit; when ``risk_rerank`` is set the
    edits are re-ranked by boundary_value = utility_proxy - lambda * risk.

    By default (``exhaustive=True``) this delegates to the deterministic
    enumerator :func:`generate_structured_proposal_exhaustive`, which fixes
    the v2 sampling-collapse (0.13 valid edits/reaction).  Pass
    ``exhaustive=False`` to use the legacy ancestral-sampling decoder.

    ``require_atom_balance=True`` (default) restricts the enumerator to
    stoichiometrically-constructible boundary candidates (relative to the
    true product's own balance); set False for unrestricted proposals.
    ``map_unmapped=True`` (default) opportunistically atom-maps unmapped
    reactions via RXNMapper for richer features; the enumerator also works
    without mapping (product-graph featurisation), just with a zeroed
    atom-map feature channel.
    """
    if exhaustive:
        return generate_structured_proposal_exhaustive(
            model, reaction_smiles, top_k=top_k, device=device,
            use_validity_mask=use_validity_mask, risk_rerank=risk_rerank,
            n_mc=n_mc, map_unmapped=map_unmapped,
            require_atom_balance=require_atom_balance)
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
        edited = getattr(edit, "applied_product", None)
        if not edited:
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


def _target_for_graph(
    target: Dict[str, Any],
    graph: ReactionGraphData,
    *,
    formal_run: bool,
) -> Optional[Dict[str, int]]:
    edit_type = int(target["edit_type"])
    locus_map = int(target.get("locus_map", target.get("locus", 0)))
    partner_map = int(target.get("partner_map", 0))
    type_only = edit_type in {
        int(EditType.NO_EDIT),
        int(EditType.NOT_APPLICABLE),
    }
    locus = graph.atom_map_to_idx.get(locus_map)
    if locus is None and not type_only:
        if formal_run:
            return None
        locus = 0
    partner = graph.atom_map_to_idx.get(partner_map, -100)
    atom_target = -100
    atomic_num = int(target.get("atom_target_atomic_num", -100))
    anchor_vocab = sorted(ANCHOR_ATOMIC_NUMS)
    if atomic_num in anchor_vocab:
        atom_target = anchor_vocab.index(atomic_num)
    return {
        "locus": int(locus or 0),
        "locus_supervised": int(not type_only and locus is not None),
        "edit_type": edit_type,
        "atom_target": int(atom_target),
        "bond_target": int(target.get("bond_order_index", -100)),
        "migrate_target": int(partner),
    }


def _real_actions_for_graph(
    reaction: str,
    graph: ReactionGraphData,
    edit_targets_cache: Dict[str, Dict[str, Any]],
    *,
    formal_run: bool,
) -> List[Dict[str, int]]:
    cached = edit_targets_cache.get(reaction)
    if cached is None or not cached.get("valid_for_formal", False):
        return []
    actions = cached.get("actions") or []
    converted = [
        _target_for_graph(action, graph, formal_run=formal_run)
        for action in actions
    ]
    return [target for target in converted if target is not None]


def _rule_distribution_for_graph(
    reaction: str,
    graph: ReactionGraphData,
    rule_proposals_cache: Dict[str, List[Dict[str, Any]]],
    *,
    formal_run: bool,
) -> Optional[Dict[str, Any]]:
    proposals = rule_proposals_cache.get(reaction)
    if not proposals:
        return None
    converted = [
        _target_for_graph(proposal, graph, formal_run=formal_run)
        for proposal in proposals
    ]
    usable = [
        (proposal, target)
        for proposal, target in zip(proposals, converted)
        if target is not None
    ]
    if not usable:
        return None
    weights = np.asarray(
        [max(float(proposal.get("hard_score", 0.0)), 1e-6)
         for proposal, _ in usable],
        dtype=np.float64,
    )
    weights = weights / weights.sum()
    return {"items": usable, "weights": weights}


def _collate_reaction_pairs(
    pairs: Sequence[Dict[str, Any]],
    preferred_key: str,
    competing_key: str,
    device: torch.device,
    *,
    map_unmapped: bool,
) -> Tuple[Optional[BatchedGraph], List[Dict[str, Any]], int]:
    kept: List[Dict[str, Any]] = []
    preferred_graphs: List[ReactionGraphData] = []
    competing_graphs: List[ReactionGraphData] = []
    for pair in pairs:
        preferred = _featurize_safe(
            str(pair[preferred_key]),
            map_unmapped=map_unmapped,
        )
        competing = _featurize_safe(
            str(pair[competing_key]),
            map_unmapped=map_unmapped,
        )
        if preferred is None or competing is None:
            continue
        preferred_graphs.append(preferred)
        competing_graphs.append(competing)
        kept.append(pair)
    if not kept:
        return None, [], 0
    batch = collate_graphs([*preferred_graphs, *competing_graphs])
    batch.atom_features = batch.atom_features.to(device)
    batch.edge_features = batch.edge_features.to(device)
    batch.edge_index = batch.edge_index.to(device)
    batch.batch_idx = batch.batch_idx.to(device)
    return batch, kept, len(kept)


def _batch_from_graphs(
    graphs: Sequence[ReactionGraphData],
    device: torch.device,
) -> BatchedGraph:
    batch = collate_graphs(list(graphs))
    batch.atom_features = batch.atom_features.to(device)
    batch.edge_features = batch.edge_features.to(device)
    batch.edge_index = batch.edge_index.to(device)
    batch.batch_idx = batch.batch_idx.to(device)
    return batch


def _primary_real_target(
    reaction: str,
    graph: ReactionGraphData,
    edit_targets_cache: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, int]]:
    targets = _real_actions_for_graph(
        reaction,
        graph,
        edit_targets_cache,
        formal_run=True,
    )
    return targets[0] if targets else None


def _collate_risk_examples(
    examples: Sequence[Dict[str, Any]],
    device: torch.device,
    *,
    map_unmapped: bool,
) -> Tuple[Optional[BatchedGraph], torch.Tensor]:
    graphs: List[ReactionGraphData] = []
    labels: List[float] = []
    for example in examples:
        graph = _featurize_risk_safe(str(example["reaction_smiles"]))
        if graph is None:
            continue
        graphs.append(graph)
        labels.append(float(example["risk_label"]))
    if not graphs:
        return None, torch.empty(0, device=device)
    return (
        _batch_from_graphs(graphs, device),
        torch.tensor(labels, device=device, dtype=torch.float32),
    )


def _featurize_risk_safe(
    reaction_smiles: str,
) -> Optional[ReactionGraphData]:
    """Featurize any parseable candidate product for FNR supervision.

    Formal edit reconstruction intentionally requires a mapped reaction with a
    valid edit grammar.  Candidate-level risk does not: known-positive
    collisions and measured HTE outcomes without a formed bond are still
    legitimate risk labels.  This product-graph path prevents silently
    discarding those labels.
    """
    if Chem is None:
        return None
    reactants, product = _safe_split(reaction_smiles)
    molecule = Chem.MolFromSmiles(product) if product else None
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    try:
        map_to_idx = {
            atom.GetAtomMapNum(): atom.GetIdx()
            for atom in molecule.GetAtoms()
            if atom.GetAtomMapNum()
        }
        atom_features, edge_index, edge_features, atom_map_nums = (
            _build_product_graph(molecule, map_to_idx)
        )
        return ReactionGraphData(
            atom_features=atom_features,
            edge_index=edge_index,
            edge_features=edge_features,
            atom_map_nums=atom_map_nums,
            true_anchor_idx=0,
            candidate_anchor_indices=list(range(molecule.GetNumAtoms())),
            source_id="formal_risk",
            pair_id="formal_risk|0",
            mapped_reaction=reaction_smiles,
            reactants=reactants,
            product=product,
            fragment_map=0,
            true_anchor_map=0,
            atom_map_to_idx=map_to_idx,
        )
    except Exception:
        return None


def compute_logp(out: StructuredProposalOutput, loci: torch.Tensor,
                 types: torch.Tensor,
                 graph_offset: int = 0) -> torch.Tensor:
    """Compute log-probability of (locus, edit_type) under model.

    graph_offset: index offset into the batch (for combined batches).
    """
    n = loci.shape[0]
    locus_logits = out.locus_logits[graph_offset:graph_offset + n]
    type_logits = out.type_logits[graph_offset:graph_offset + n]
    locus_logp = F.log_softmax(locus_logits, dim=-1).gather(
        -1, loci.clamp(min=0, max=locus_logits.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
    type_logp = F.log_softmax(type_logits, dim=-1).gather(
        -1, types.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return locus_logp + type_logp


def compute_action_logp(
    out: StructuredProposalOutput,
    targets: Sequence[Dict[str, int]],
    *,
    graph_offset: int = 0,
) -> torch.Tensor:
    """Log-probability of the complete supervised structured action.

    Unlike the legacy ``compute_logp`` helper, this includes the real action
    argument whenever one is observed: atom identity, bond order or partner
    atom.  Missing arguments are explicitly ignored instead of replaced by a
    pseudo-target.
    """
    if not targets:
        return out.locus_logits.new_empty(0)
    loci = torch.tensor(
        [target["locus"] for target in targets],
        device=out.locus_logits.device,
        dtype=torch.long,
    )
    types = torch.tensor(
        [target["edit_type"] for target in targets],
        device=out.type_logits.device,
        dtype=torch.long,
    )
    total = compute_logp(out, loci, types, graph_offset=graph_offset)
    for row, target in enumerate(targets):
        edit_type = EditType(int(target["edit_type"]))
        if edit_type == EditType.ATOM_TRANSMUTATION:
            key, argument = "atom_logits", int(target["atom_target"])
        elif edit_type == EditType.BOND_ORDER_CHANGE:
            key, argument = "bond_logits", int(target["bond_target"])
        elif edit_type in {
            EditType.BOND_FORM,
            EditType.BOND_BREAK,
            EditType.FORMED_BOND_MIGRATE,
        }:
            key, argument = "migrate_logits", int(target["migrate_target"])
        else:
            continue
        if argument < 0:
            continue
        logits = out.arg_logits[key][graph_offset + row]
        if argument >= logits.shape[-1]:
            raise RuntimeError(
                f"formal action argument {argument} exceeds {key} width "
                f"{logits.shape[-1]}"
            )
        total[row] = total[row] + F.log_softmax(logits, dim=-1)[argument]
    return total


def _resolve_targets(reaction: str,
                     edit_targets_cache: Optional[Dict[str, Dict[str, Any]]] = None,
                     rule_proposals_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                     prefer_rule: bool = False) -> Tuple[int, int]:
    """Resolve (locus, edit_type) for a reaction using REAL data.

    When ``prefer_rule`` and a rule-proposal cache hit exists, use the
    first rule proposal.  Otherwise use the cached real edit target,
    falling back to a fresh ``extract_real_edit_targets`` call.
    """
    if prefer_rule and rule_proposals_cache is not None:
        props = rule_proposals_cache.get(reaction)
        if props:
            p = props[0]
            return int(p["locus"]), int(p["edit_type"])
    if edit_targets_cache is not None:
        t = edit_targets_cache.get(reaction)
        if t is not None:
            return int(t["locus"]), int(t["edit_type"])
    t = extract_real_edit_targets(reaction)
    return int(t["locus"]), int(t["edit_type"])


def train_stage_formal(
    model: StructuredProposalModel,
    stage: int,
    train_reactions: Sequence[str],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    log: Optional[List[dict]],
    map_unmapped: bool,
    edit_targets_cache: Dict[str, Dict[str, Any]],
    rule_proposals_cache: Dict[str, List[Dict[str, Any]]],
    competing_pairs_cache: List[Dict[str, Any]],
    preference_pairs_cache: List[Dict[str, Any]],
    risk_examples_cache: List[Dict[str, Any]],
    ref_model: Optional[StructuredProposalModel],
) -> List[dict]:
    """Fail-closed formal training with no pseudo-label fallback branches."""
    required: Dict[int, Any] = {
        1: edit_targets_cache,
        2: rule_proposals_cache,
        3: competing_pairs_cache,
        4: preference_pairs_cache,
    }
    if stage not in required or not required[stage]:
        raise RuntimeError(
            f"formal G8-C stage {stage} requires non-empty real supervision cache"
        )
    if stage == 3 and not risk_examples_cache:
        raise RuntimeError(
            "formal G8-C Stage 3 requires candidate-level risk supervision"
        )
    if stage == 4 and ref_model is None:
        raise RuntimeError(
            "formal G8-C Stage 4 requires a frozen post-Stage-3 reference model"
        )

    set_seed(seed)
    model.to(device)
    if ref_model is not None:
        ref_model.to(device)
        ref_model.eval()
        for parameter in ref_model.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stage1_loss = Stage1ReconstructionLoss()
    stage2_loss = Stage2ImitationLoss()
    stage3_loss = Stage3ContrastiveLoss()
    log = log if log is not None else []
    if stage in {1, 2}:
        n_steps = len(train_reactions)
    elif stage == 3:
        n_steps = max(
            len(competing_pairs_cache),
            len(risk_examples_cache),
        )
    else:
        n_steps = len(preference_pairs_cache)
    risk_positive = sum(
        int(example["risk_label"]) == 1
        for example in risk_examples_cache
    )
    risk_negative = len(risk_examples_cache) - risk_positive
    rehearsal_pool: List[Tuple[ReactionGraphData, Dict[str, int]]] = []
    if stage != 1:
        for reaction in train_reactions:
            graph = _featurize_safe(
                str(reaction),
                map_unmapped=map_unmapped,
            )
            if graph is None:
                continue
            targets = _real_actions_for_graph(
                str(reaction),
                graph,
                edit_targets_cache,
                formal_run=True,
            )
            if targets:
                rehearsal_pool.append((graph, targets[0]))
        if not rehearsal_pool:
            raise RuntimeError(
                f"formal G8-C Stage {stage} has no prevalidated real-edit "
                "rehearsal examples"
            )

    for epoch in range(epochs):
        model.train()
        order = np.random.RandomState(seed + epoch).permutation(n_steps)
        epoch_loss = 0.0
        n_batches = 0
        component_totals: Dict[str, float] = {}
        for start in range(0, n_steps, batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad()

            if stage == 1:
                graphs: List[ReactionGraphData] = []
                targets: List[Dict[str, int]] = []
                for index in indices:
                    reaction = str(train_reactions[int(index)])
                    graph = _featurize_safe(
                        reaction,
                        map_unmapped=map_unmapped,
                    )
                    if graph is None:
                        continue
                    for target in _real_actions_for_graph(
                        reaction,
                        graph,
                        edit_targets_cache,
                        formal_run=True,
                    ):
                        graphs.append(graph)
                        targets.append(target)
                if not graphs:
                    continue
                batch = _batch_from_graphs(graphs, device)
                loci = torch.tensor(
                    [target["locus"] for target in targets],
                    device=device,
                    dtype=torch.long,
                )
                types = torch.tensor(
                    [target["edit_type"] for target in targets],
                    device=device,
                    dtype=torch.long,
                )
                output = model(
                    batch,
                    locus_index=loci,
                    action_head="reconstruction",
                )
                loss, components = stage1_loss(
                    output,
                    loci,
                    types,
                    {
                        "atom_logits": torch.tensor(
                            [target["atom_target"] for target in targets],
                            device=device,
                            dtype=torch.long,
                        ),
                        "bond_logits": torch.tensor(
                            [target["bond_target"] for target in targets],
                            device=device,
                            dtype=torch.long,
                        ),
                        "migrate_logits": torch.tensor(
                            [target["migrate_target"] for target in targets],
                            device=device,
                            dtype=torch.long,
                        ),
                    },
                )

            elif stage == 2:
                reactions = [str(train_reactions[int(index)]) for index in indices]
                batch, successful = _collate_reactions(
                    reactions,
                    device,
                    map_unmapped=map_unmapped,
                )
                if batch is None:
                    continue
                distributions = [
                    _rule_distribution_for_graph(
                        reaction,
                        graph,
                        rule_proposals_cache,
                        formal_run=True,
                    )
                    for reaction, graph in zip(successful, batch.graphs)
                ]
                if any(distribution is None for distribution in distributions):
                    raise RuntimeError(
                        "formal G8-C Stage 2 has an unresolvable rule target"
                    )
                max_locus = max(graph.atom_features.shape[0] for graph in batch.graphs)
                locus_probs = torch.zeros(
                    (len(distributions), max_locus),
                    device=device,
                )
                type_probs = torch.zeros(
                    (len(distributions), NUM_EDIT_TYPES),
                    device=device,
                )
                locus_mask = torch.zeros(len(distributions), device=device)
                chosen_loci: List[int] = []
                for row, distribution in enumerate(distributions):
                    assert distribution is not None
                    for weight, (_proposal, target) in zip(
                        distribution["weights"],
                        distribution["items"],
                    ):
                        type_probs[row, target["edit_type"]] += float(weight)
                        if target["locus_supervised"]:
                            locus_probs[row, target["locus"]] += float(weight)
                            locus_mask[row] = 1.0
                    chosen_loci.append(
                        int(locus_probs[row].argmax().item())
                        if locus_mask[row] else 0
                    )
                output = model(
                    batch,
                    locus_index=torch.tensor(
                        chosen_loci,
                        device=device,
                        dtype=torch.long,
                    ),
                )
                loss, components = stage2_loss(
                    output,
                    locus_probs,
                    type_probs,
                    locus_mask,
                )

            elif stage == 3:
                pairs = [
                    competing_pairs_cache[int(index) % len(competing_pairs_cache)]
                    for index in indices
                ]
                batch, kept_pairs, pair_count = _collate_reaction_pairs(
                    pairs,
                    "reaction_smiles",
                    "competing_reaction_smiles",
                    device,
                    map_unmapped=map_unmapped,
                )
                if batch is None or pair_count == 0:
                    continue
                output = model(batch)
                positive_mask = torch.cat(
                    [
                        torch.ones(pair_count, device=device),
                        torch.zeros(pair_count, device=device),
                    ]
                )
                contrastive, components = stage3_loss(output, positive_mask)
                risk_rows = [
                    risk_examples_cache[int(index) % len(risk_examples_cache)]
                    for index in indices
                ]
                risk_batch, risk_labels = _collate_risk_examples(
                    risk_rows,
                    device,
                    map_unmapped=map_unmapped,
                )
                if risk_batch is None:
                    raise RuntimeError(
                        "formal G8-C Stage 3 could not collate risk supervision"
                    )
                risk_output = model(risk_batch)
                per_example_weight = torch.where(
                    risk_labels > 0.5,
                    torch.full_like(
                        risk_labels,
                        len(risk_examples_cache)
                        / max(2.0 * risk_positive, 1.0),
                    ),
                    torch.full_like(
                        risk_labels,
                        len(risk_examples_cache)
                        / max(2.0 * risk_negative, 1.0),
                    ),
                )
                risk_bce = F.binary_cross_entropy(
                    risk_output.risk,
                    risk_labels,
                    weight=per_example_weight,
                )
                loss = contrastive + risk_bce
                components["risk_bce"] = float(risk_bce.item())
                components["same_context_pairs"] = float(len(kept_pairs))

            else:
                pairs = [
                    preference_pairs_cache[int(index) % len(preference_pairs_cache)]
                    for index in indices
                ]
                batch, kept_pairs, pair_count = _collate_reaction_pairs(
                    pairs,
                    "preferred_reaction",
                    "dispreferred_reaction",
                    device,
                    map_unmapped=map_unmapped,
                )
                if batch is None or pair_count == 0:
                    continue
                preferred_targets: List[Dict[str, int]] = []
                dispreferred_targets: List[Dict[str, int]] = []
                for row, pair in enumerate(kept_pairs):
                    preferred_target = _primary_real_target(
                        str(pair["preferred_reaction"]),
                        batch.graphs[row],
                        edit_targets_cache,
                    )
                    dispreferred_target = _primary_real_target(
                        str(pair["dispreferred_reaction"]),
                        batch.graphs[pair_count + row],
                        edit_targets_cache,
                    )
                    if preferred_target is None or dispreferred_target is None:
                        raise RuntimeError(
                            "formal G8-C Stage 4 preference lacks a real action target"
                        )
                    preferred_targets.append(preferred_target)
                    dispreferred_targets.append(dispreferred_target)
                pref_loci = torch.tensor(
                    [target["locus"] for target in preferred_targets],
                    device=device,
                    dtype=torch.long,
                )
                pref_types = torch.tensor(
                    [target["edit_type"] for target in preferred_targets],
                    device=device,
                    dtype=torch.long,
                )
                disp_loci = torch.tensor(
                    [target["locus"] for target in dispreferred_targets],
                    device=device,
                    dtype=torch.long,
                )
                disp_types = torch.tensor(
                    [target["edit_type"] for target in dispreferred_targets],
                    device=device,
                    dtype=torch.long,
                )
                all_loci = torch.cat([pref_loci, disp_loci])
                policy_output = model(batch, locus_index=all_loci)
                with torch.no_grad():
                    reference_output = ref_model(batch, locus_index=all_loci)
                policy_pref = compute_action_logp(
                    policy_output,
                    preferred_targets,
                )
                policy_disp = compute_action_logp(
                    policy_output,
                    dispreferred_targets,
                    graph_offset=pair_count,
                )
                reference_pref = compute_action_logp(
                    reference_output,
                    preferred_targets,
                )
                reference_disp = compute_action_logp(
                    reference_output,
                    dispreferred_targets,
                    graph_offset=pair_count,
                )
                delta = (
                    policy_pref - policy_disp
                    - reference_pref + reference_disp
                )
                yield_weights = torch.tensor(
                    [
                        0.5 + min(
                            abs(
                                float(pair.get("preferred_yield", 0.0))
                                - float(pair.get("dispreferred_yield", 0.0))
                            ) / 100.0,
                            1.0,
                        )
                        for pair in kept_pairs
                    ],
                    device=device,
                    dtype=torch.float32,
                )
                loss = ((delta - 0.5).pow(2) * yield_weights).mean()
                components = {
                    "dpo_loss": float(loss.item()),
                    "preference_acc": float((delta > 0).float().mean().item()),
                    "delta_mean": float(delta.mean().item()),
                    "max_abs_log_ratio": float(delta.abs().max().item()),
                    "mean_observed_yield_weight": float(yield_weights.mean().item()),
                }

            if stage != 1:
                rehearsal_graphs: List[ReactionGraphData] = []
                rehearsal_targets: List[Dict[str, int]] = []
                for position, raw_index in enumerate(indices):
                    graph, target = rehearsal_pool[
                        (int(raw_index) + epoch + position)
                        % len(rehearsal_pool)
                    ]
                    rehearsal_graphs.append(graph)
                    rehearsal_targets.append(target)
                rehearsal_batch = _batch_from_graphs(
                    rehearsal_graphs,
                    device,
                )
                rehearsal_loci = torch.tensor(
                    [target["locus"] for target in rehearsal_targets],
                    device=device,
                    dtype=torch.long,
                )
                rehearsal_types = torch.tensor(
                    [target["edit_type"] for target in rehearsal_targets],
                    device=device,
                    dtype=torch.long,
                )
                rehearsal_output = model(
                    rehearsal_batch,
                    locus_index=rehearsal_loci,
                    action_head="reconstruction",
                )
                rehearsal_loss, rehearsal_components = stage1_loss(
                    rehearsal_output,
                    rehearsal_loci,
                    rehearsal_types,
                    {
                        "atom_logits": torch.tensor(
                            [
                                target["atom_target"]
                                for target in rehearsal_targets
                            ],
                            device=device,
                            dtype=torch.long,
                        ),
                        "bond_logits": torch.tensor(
                            [
                                target["bond_target"]
                                for target in rehearsal_targets
                            ],
                            device=device,
                            dtype=torch.long,
                        ),
                        "migrate_logits": torch.tensor(
                            [
                                target["migrate_target"]
                                for target in rehearsal_targets
                            ],
                            device=device,
                            dtype=torch.long,
                        ),
                    },
                )
                loss = loss + 0.25 * rehearsal_loss
                components["reconstruction_rehearsal_loss"] = float(
                    rehearsal_loss.item()
                )
                for key, value in rehearsal_components.items():
                    components[f"rehearsal_{key}"] = value

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"formal G8-C Stage {stage} produced a non-finite loss"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
            for key, value in components.items():
                component_totals[key] = component_totals.get(key, 0.0) + float(value)

        if n_batches == 0:
            raise RuntimeError(
                f"formal G8-C Stage {stage} produced zero trainable batches"
            )
        entry = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": epoch_loss / n_batches,
            "n_batches": n_batches,
            "formal_supervision": True,
            "components": {
                key: value / n_batches
                for key, value in component_totals.items()
            },
        }
        log.append(entry)
        print(
            f"[{PHASE}] formal stage={stage} epoch={epoch} "
            f"train_loss={entry['train_loss']:.4f} batches={n_batches}"
        )
    return log


# REAL-DATA train_stage (phase-2 fix)
def train_stage(model: StructuredProposalModel, stage: int,
                train_reactions: Sequence[str],
                val_reactions: Sequence[str],
                rule_generator: ReactionBoundaryGenerator,
                epochs: int, batch_size: int, lr: float,
                device: torch.device, seed: int = BASE_SEED,
                log: Optional[List[dict]] = None,
                map_unmapped: bool = False,
                edit_targets_cache: Optional[Dict[str, Dict[str, Any]]] = None,
                rule_proposals_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                competing_pairs_cache: Optional[List[Dict[str, Any]]] = None,
                preference_pairs_cache: Optional[List[Dict[str, Any]]] = None,
                risk_examples_cache: Optional[List[Dict[str, Any]]] = None,
                ref_model: Optional[StructuredProposalModel] = None,
                formal_run: bool = False) -> List[dict]:
    """Run a single training stage.  Returns the per-epoch log entries.

    Stages 1-4 now use REAL supervision:
        * Stage 1: real edit targets from atom-mapped reaction centers.
        * Stage 2: real rule-generator proposals (imitation).
        * Stage 3: real HTE competing-outcome pairs (contrastive).
        * Stage 4: real DPO preference pairs with a frozen reference model.
    In exploratory/standalone mode, missing caches retain the historical
    fallback behaviour.  A formal run is fail-closed: the required real
    supervision and reference policy must be present, and a failed batch
    collation cannot silently turn into pseudo-supervision.
    """
    if formal_run:
        if edit_targets_cache is None:
            raise RuntimeError("formal G8-C requires an edit-target cache")
        if rule_proposals_cache is None:
            raise RuntimeError("formal G8-C requires a rule-proposal cache")
        if competing_pairs_cache is None:
            raise RuntimeError("formal G8-C requires a competing-pair cache")
        if preference_pairs_cache is None:
            raise RuntimeError("formal G8-C requires a preference-pair cache")
        if risk_examples_cache is None:
            raise RuntimeError("formal G8-C requires a risk-supervision cache")
        return train_stage_formal(
            model,
            stage,
            train_reactions,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed,
            log=log,
            map_unmapped=map_unmapped,
            edit_targets_cache=edit_targets_cache,
            rule_proposals_cache=rule_proposals_cache,
            competing_pairs_cache=competing_pairs_cache,
            preference_pairs_cache=preference_pairs_cache,
            risk_examples_cache=risk_examples_cache,
            ref_model=ref_model,
        )

    set_seed(seed)
    model.to(device)
    if ref_model is not None:
        ref_model.to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
        order = np.random.RandomState(seed + epoch).permutation(n) if n > 0 else []
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch_rxns = [train_reactions[i] for i in idx]
            opt.zero_grad()

            if stage == 1:
                # Stage 1: REAL edit targets from atom-mapped reaction centers.
                batch, success_rxns = _collate_reactions(
                    batch_rxns, device, map_unmapped=map_unmapped)
                if batch is None:
                    continue
                out = model(batch)
                pairs = [_resolve_targets(r, edit_targets_cache)
                         for r in success_rxns]
                if formal_run:
                    missing = [r for r in success_rxns
                               if edit_targets_cache is None or r not in edit_targets_cache]
                    if missing:
                        raise RuntimeError(
                            f"formal G8-C Stage 1 missing edit targets for {len(missing)} reactions")
                loci = torch.tensor([p[0] for p in pairs],
                                    device=device, dtype=torch.long)
                loci = loci.clamp(min=0, max=out.locus_logits.shape[-1] - 1)  # phase2-clamp-loci
                types = torch.tensor([p[1] for p in pairs],
                                     device=device, dtype=torch.long)
                loss, _ = loss_fn(out, loci, types)

            elif stage == 2:
                # Stage 2: imitate REAL rule-generator proposals.
                batch, success_rxns = _collate_reactions(
                    batch_rxns, device, map_unmapped=map_unmapped)
                if batch is None:
                    continue
                out = model(batch)
                pairs = [_resolve_targets(
                    r, edit_targets_cache, rule_proposals_cache,
                    prefer_rule=True) for r in success_rxns]
                if formal_run:
                    missing = [r for r in success_rxns
                               if rule_proposals_cache is None or r not in rule_proposals_cache]
                    if missing:
                        raise RuntimeError(
                            f"formal G8-C Stage 2 missing rule proposals for {len(missing)} reactions")
                loci = torch.tensor([p[0] for p in pairs],
                                    device=device, dtype=torch.long)
                loci = loci.clamp(min=0, max=out.locus_logits.shape[-1] - 1)  # phase2-clamp-loci
                types = torch.tensor([p[1] for p in pairs],
                                     device=device, dtype=torch.long)
                locus_probs = F.one_hot(
                    loci, num_classes=out.locus_logits.shape[-1]).float()
                type_probs = F.one_hot(types, num_classes=NUM_EDIT_TYPES).float()
                loss, _ = loss_fn(out, locus_probs, type_probs)

            elif stage == 3:
                # Stage 3: REAL competing-outcome pairs from HTE data.
                if competing_pairs_cache:
                    pairs = [competing_pairs_cache[int(i) % len(competing_pairs_cache)]
                             for i in idx]
                    pref_rxns = [p["reaction_smiles"] for p in pairs]
                    comp_rxns = [p.get("competing_reaction_smiles") or p["reaction_smiles"]
                                 for p in pairs]
                    pref_batch, pref_success = _collate_reactions(
                        pref_rxns, device, map_unmapped=map_unmapped)
                    comp_batch, comp_success = _collate_reactions(
                        comp_rxns, device, map_unmapped=map_unmapped)
                    if pref_batch is None or comp_batch is None:
                        if formal_run:
                            raise RuntimeError(
                                "formal G8-C Stage 3 could not collate a real competing-outcome pair")
                        batch, success_rxns = _collate_reactions(
                            batch_rxns, device, map_unmapped=map_unmapped)
                        if batch is None:
                            continue
                        out = model(batch)
                        num_g = len(success_rxns)
                        pos_mask = torch.ones(num_g, device=device)
                        if num_g > 1:
                            pos_mask[num_g // 2:] = 0
                        loss, _ = loss_fn(out, pos_mask)
                    else:
                        # Combine preferred and competing into a SINGLE batch
                        # so collate_graphs pads all to the same max_nodes.
                        combined_rxns = pref_rxns + comp_rxns
                        combined_batch, combined_success = _collate_reactions(
                            combined_rxns, device, map_unmapped=map_unmapped)
                        if combined_batch is None:
                            continue
                        out = model(combined_batch)
                        num_pref = len(pref_success)
                        num_comp = len(comp_success)
                        pos_mask = torch.cat([
                            torch.ones(num_pref, device=device),
                            torch.zeros(num_comp, device=device)])
                        loss, _ = loss_fn(out, pos_mask)
                else:
                    batch, success_rxns = _collate_reactions(
                        batch_rxns, device, map_unmapped=map_unmapped)
                    if batch is None:
                        continue
                    out = model(batch)
                    num_g = len(success_rxns)
                    pos_mask = torch.ones(num_g, device=device)
                    if num_g > 1:
                        pos_mask[num_g // 2:] = 0
                    loss, _ = loss_fn(out, pos_mask)

            else:  # stage 4 DPO
                if preference_pairs_cache:
                    pairs = [preference_pairs_cache[int(i) % len(preference_pairs_cache)]
                             for i in idx]
                    pref_rxns = [p["preferred_reaction"] for p in pairs]
                    disp_rxns = [p["dispreferred_reaction"] for p in pairs]
                    pref_batch, pref_success = _collate_reactions(
                        pref_rxns, device, map_unmapped=map_unmapped)
                    disp_batch, disp_success = _collate_reactions(
                        disp_rxns, device, map_unmapped=map_unmapped)
                    if pref_batch is None or disp_batch is None:
                        if formal_run:
                            raise RuntimeError(
                                "formal G8-C Stage 4 could not collate a real preference pair")
                        batch, success_rxns = _collate_reactions(
                            batch_rxns, device, map_unmapped=map_unmapped)
                        if batch is None:
                            continue
                        out = model(batch)
                        num_g = len(success_rxns)
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
                        pairs_t = [_resolve_targets(r, edit_targets_cache)
                                    for r in success_rxns[:g]]
                        loci = torch.tensor([p[0] for p in pairs_t],
                                            device=device, dtype=torch.long)
                        loci = loci.clamp(min=0, max=out.locus_logits.shape[-1] - 1)  # phase2-clamp-loci
                        types = torch.tensor([p[1] for p in pairs_t],
                                             device=device, dtype=torch.long)
                        ref_pref = torch.zeros(g, device=device)
                        ref_disp = torch.zeros(g, device=device)
                        loss, _ = loss_fn(out_pref, out_disp, loci, types,
                                          loci, types, ref_pref, ref_disp)
                    else:
                        # Combine preferred and dispreferred into SINGLE batch
                        # so collate_graphs pads all to same max_nodes
                        combined_rxns = pref_rxns + disp_rxns
                        combined_batch, combined_success = _collate_reactions(
                            combined_rxns, device, map_unmapped=map_unmapped)
                        if combined_batch is None:
                            continue
                        out_all = model(combined_batch)
                        num_pref = len(pref_success)
                        num_disp = len(disp_success)
                        g = min(num_pref, num_disp)
                        pref_pairs_t = [_resolve_targets(r, edit_targets_cache)
                                        for r in pref_success[:g]]
                        disp_pairs_t = [_resolve_targets(r, edit_targets_cache)
                                        for r in disp_success[:g]]
                        pref_loci = torch.tensor([p[0] for p in pref_pairs_t],
                                                 device=device, dtype=torch.long)
                        pref_types = torch.tensor([p[1] for p in pref_pairs_t],
                                                  device=device, dtype=torch.long)
                        disp_loci = torch.tensor([p[0] for p in disp_pairs_t],
                                                 device=device, dtype=torch.long)
                        disp_types = torch.tensor([p[1] for p in disp_pairs_t],
                                                  device=device, dtype=torch.long)
                        max_locus = out_all.locus_logits.shape[-1] - 1
                        pref_loci = pref_loci.clamp(min=0, max=max_locus)
                        disp_loci = disp_loci.clamp(min=0, max=max_locus)
                        if ref_model is not None:
                            with torch.no_grad():
                                ref_out_all = ref_model(combined_batch)
                            ref_pref = compute_logp(ref_out_all, pref_loci, pref_types,
                                                     graph_offset=0)
                            ref_disp = compute_logp(ref_out_all, disp_loci, disp_types,
                                                     graph_offset=num_pref)
                        else:
                            ref_pref = torch.zeros(g, device=device)
                            ref_disp = torch.zeros(g, device=device)
                        # Slice outputs from combined batch
                        out_pref = StructuredProposalOutput(
                            locus_logits=out_all.locus_logits[:g],
                            type_logits=out_all.type_logits[:g],
                            arg_logits={k: v[:g] for k, v in out_all.arg_logits.items()},
                            validity_mask=out_all.validity_mask[:g],
                            risk=out_all.risk[:g],
                            uncertainty=out_all.uncertainty[:g],
                            graph_emb=out_all.graph_emb[:g],
                            node_emb=out_all.node_emb)
                        out_disp = StructuredProposalOutput(
                            locus_logits=out_all.locus_logits[num_pref:num_pref + g],
                            type_logits=out_all.type_logits[num_pref:num_pref + g],
                            arg_logits={k: v[num_pref:num_pref + g] for k, v in out_all.arg_logits.items()},
                            validity_mask=out_all.validity_mask[num_pref:num_pref + g],
                            risk=out_all.risk[num_pref:num_pref + g],
                            uncertainty=out_all.uncertainty[num_pref:num_pref + g],
                            graph_emb=out_all.graph_emb[num_pref:num_pref + g],
                            node_emb=out_all.node_emb)
                        loss, _ = loss_fn(out_pref, out_disp, pref_loci, pref_types,
                                          disp_loci, disp_types, ref_pref, ref_disp)
                else:
                    batch, success_rxns = _collate_reactions(
                        batch_rxns, device, map_unmapped=map_unmapped)
                    if batch is None:
                        continue
                    out = model(batch)
                    num_g = len(success_rxns)
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
                    pairs_t = [_resolve_targets(r, edit_targets_cache)
                                for r in success_rxns[:g]]
                    loci = torch.tensor([p[0] for p in pairs_t],
                                        device=device, dtype=torch.long)
                    loci = loci.clamp(min=0, max=out.locus_logits.shape[-1] - 1)  # phase2-clamp-loci
                    types = torch.tensor([p[1] for p in pairs_t],
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
            vb, v_success = _collate_reactions(
                val_reactions[:batch_size], device, map_unmapped=map_unmapped)
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
# Formal validation (real supervision only; no self-built tiny MLP)
# ---------------------------------------------------------------------------

def _state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _wilson_interval(successes: int, total: int) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return (
        float((centre - half) / denominator),
        float((centre + half) / denominator),
    )


def _deterministic_subset(
    rows: Sequence[Any],
    limit: Optional[int],
    seed: int,
) -> List[Any]:
    rows = list(rows)
    if limit is None or len(rows) <= limit:
        return rows
    indices = np.random.RandomState(seed).choice(
        len(rows),
        size=limit,
        replace=False,
    )
    return [rows[int(index)] for index in sorted(indices)]


def _frozen_hash_partition(
    rows: Sequence[Any],
    key_fn,
    *,
    namespace: str,
    holdout_modulus: int = 5,
) -> Tuple[List[Any], List[Any]]:
    """Group-safe deterministic train/holdout partition."""
    train: List[Any] = []
    holdout: List[Any] = []
    for row in rows:
        key = str(key_fn(row))
        token = f"{namespace}|{key}".encode("utf-8")
        bucket = int(hashlib.sha256(token).hexdigest()[:8], 16)
        (holdout if bucket % holdout_modulus == 0 else train).append(row)
    return train, holdout


def _proportion_metric(successes: int, total: int) -> Dict[str, Any]:
    low, high = _wilson_interval(successes, total)
    return {
        "value": float(successes / total) if total else 0.0,
        "successes": int(successes),
        "n": int(total),
        "ci_method": "Wilson 95%",
        "ci_low": low,
        "ci_high": high,
    }


def evaluate_formal_edit_validation(
    model: StructuredProposalModel,
    reactions: Sequence[str],
    edit_targets_cache: Dict[str, Dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    limit: Optional[int],
    seed: int,
    map_unmapped: bool,
) -> Dict[str, Any]:
    eligible = [
        reaction
        for reaction in dict.fromkeys(reactions)
        if edit_targets_cache.get(reaction, {}).get("valid_for_formal")
        and edit_targets_cache.get(reaction, {}).get("actions")
    ]
    selected = _deterministic_subset(eligible, limit, seed)
    if not selected:
        raise RuntimeError("formal edit validation has no eligible real targets")

    locus_success = type_success = joint_success = 0
    locus_total = type_total = joint_total = 0
    argument_success = argument_total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch, successful = _collate_reactions(
                selected[start:start + batch_size],
                device,
                map_unmapped=map_unmapped,
            )
            if batch is None:
                continue
            output = model(batch, action_head="reconstruction")
            predicted_loci = output.locus_logits.argmax(-1)
            predicted_types = output.type_logits.argmax(-1)
            for row, reaction in enumerate(successful):
                targets = _real_actions_for_graph(
                    reaction,
                    batch.graphs[row],
                    edit_targets_cache,
                    formal_run=True,
                )
                if not targets:
                    continue
                pred_locus = int(predicted_loci[row].item())
                pred_type = int(predicted_types[row].item())
                locus_total += 1
                type_total += 1
                joint_total += 1
                locus_success += int(
                    any(pred_locus == target["locus"] for target in targets)
                )
                type_success += int(
                    any(pred_type == target["edit_type"] for target in targets)
                )
                matching = [
                    target for target in targets
                    if pred_locus == target["locus"]
                    and pred_type == target["edit_type"]
                ]
                joint_success += int(bool(matching))
                if not matching:
                    continue
                target = matching[0]
                edit_type = EditType(pred_type)
                if edit_type == EditType.ATOM_TRANSMUTATION:
                    key, expected = "atom_logits", target["atom_target"]
                elif edit_type == EditType.BOND_ORDER_CHANGE:
                    key, expected = "bond_logits", target["bond_target"]
                elif edit_type in {
                    EditType.BOND_FORM,
                    EditType.BOND_BREAK,
                    EditType.FORMED_BOND_MIGRATE,
                }:
                    key, expected = "migrate_logits", target["migrate_target"]
                else:
                    continue
                if expected < 0:
                    continue
                argument_total += 1
                prediction = int(output.arg_logits[key][row].argmax().item())
                argument_success += int(prediction == expected)

    if min(locus_total, type_total, joint_total) <= 0:
        raise RuntimeError("formal edit validation produced no scored reactions")
    return {
        "split": "validation",
        "selection": "deterministic_seeded_subset_before_metric_computation",
        "seed": seed,
        "n_eligible": len(eligible),
        "n_selected": len(selected),
        "edit_locus_accuracy": _proportion_metric(locus_success, locus_total),
        "edit_type_accuracy": _proportion_metric(type_success, type_total),
        "joint_locus_type_accuracy": _proportion_metric(
            joint_success,
            joint_total,
        ),
        "argument_accuracy_given_joint_match": _proportion_metric(
            argument_success,
            argument_total,
        ),
        "selected_reactions": selected,
    }


def evaluate_formal_candidate_generation(
    model: StructuredProposalModel,
    reactions: Sequence[str],
    *,
    device: torch.device,
    top_k: int,
    limit: Optional[int],
    seed: int,
    map_unmapped: bool,
) -> Dict[str, Any]:
    selected = _deterministic_subset(
        list(dict.fromkeys(reactions)),
        limit,
        seed,
    )
    if not selected:
        raise RuntimeError("formal candidate validation has no reactions")
    covered = 0
    generated = 0
    valid = 0
    for reaction in selected:
        edits = generate_structured_proposal_exhaustive(
            model,
            reaction,
            top_k=top_k,
            device=device,
            use_validity_mask=True,
            risk_rerank=True,
            n_mc=0,
            map_unmapped=map_unmapped,
            require_atom_balance=True,
            balance_dist_slack=2,
        )
        negatives = proposal_to_negatives(reaction, edits)
        generated += len(edits)
        valid += len(negatives)
        covered += int(bool(negatives))
    return {
        "split": "validation",
        "selection": "deterministic_seeded_subset_before_generation",
        "seed": seed,
        "top_k_budget": top_k,
        "n_reactions": len(selected),
        "n_generated_candidates": generated,
        "valid_edit_rate": _proportion_metric(valid, generated),
        "candidate_coverage": _proportion_metric(covered, len(selected)),
        "atom_balance_contract": "candidate_distance<=true_product_distance+2",
    }


def _expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(len(labels), 1)
    ece = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (
                (probabilities >= edges[index])
                & (probabilities <= edges[index + 1])
            )
        else:
            mask = (
                (probabilities >= edges[index])
                & (probabilities < edges[index + 1])
            )
        if not bool(mask.any()):
            continue
        ece += (
            float(mask.sum()) / total
            * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
        )
    return float(ece)


def evaluate_formal_risk_validation(
    model: StructuredProposalModel,
    examples: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    limit: Optional[int],
    seed: int,
    map_unmapped: bool,
) -> Dict[str, Any]:
    selected = _deterministic_subset(examples, limit, seed)
    labels: List[float] = []
    probabilities: List[float] = []
    sources: List[str] = []
    groups: List[str] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            rows = selected[start:start + batch_size]
            batch, batch_labels = _collate_risk_examples(
                rows,
                device,
                map_unmapped=map_unmapped,
            )
            if batch is None:
                continue
            output = model(batch)
            probabilities.extend(output.risk.detach().cpu().tolist())
            labels.extend(batch_labels.detach().cpu().tolist())
            # Formal main prefilters unfeaturizable examples; keep source and
            # experimental-group provenance aligned with predictions.
            sources.extend([str(row["risk_source"]) for row in rows])
            groups.extend([
                str(
                    row.get("experimental_group")
                    or row.get("record_id")
                    or row["reaction_smiles"]
                )
                for row in rows
            ])
    label_array = np.asarray(labels, dtype=np.float64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if len(label_array) == 0 or len(np.unique(label_array)) < 2:
        raise RuntimeError(
            "formal FNR validation requires featurizable examples from both classes"
        )
    if not (
        len(label_array)
        == len(probability_array)
        == len(sources)
        == len(groups)
    ):
        raise RuntimeError("formal FNR validation provenance alignment failed")

    calibration_mask = np.asarray([
        int(hashlib.sha256(
            f"phase_c_risk_calibration_v1|{group}".encode("utf-8")
        ).hexdigest()[:8], 16) % 2 == 0
        for group in groups
    ])
    evaluation_mask = ~calibration_mask
    if (
        calibration_mask.sum() == 0
        or evaluation_mask.sum() == 0
        or len(np.unique(label_array[calibration_mask])) < 2
        or len(np.unique(label_array[evaluation_mask])) < 2
    ):
        raise RuntimeError(
            "formal FNR calibration/evaluation group split lacks both classes"
        )
    clipped = np.clip(probability_array, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    temperatures = np.geomspace(0.5, 5.0, 91)

    def _nll(temp: float) -> float:
        calibrated = 1.0 / (
            1.0 + np.exp(-logits[calibration_mask] / temp)
        )
        labels_cal = label_array[calibration_mask]
        return float(-np.mean(
            labels_cal * np.log(np.clip(calibrated, 1e-9, 1.0))
            + (1.0 - labels_cal)
            * np.log(np.clip(1.0 - calibrated, 1e-9, 1.0))
        ))

    calibration_losses = np.asarray([_nll(float(t)) for t in temperatures])
    temperature = float(temperatures[int(calibration_losses.argmin())])
    evaluation_labels = label_array[evaluation_mask]
    raw_evaluation = probability_array[evaluation_mask]
    calibrated_evaluation = 1.0 / (
        1.0 + np.exp(-logits[evaluation_mask] / temperature)
    )
    evaluation_sources = [
        source
        for source, include in zip(sources, evaluation_mask.tolist())
        if include
    ]
    source_counts = {
        source: evaluation_sources.count(source)
        for source in sorted(set(evaluation_sources))
    }
    return {
        "split": "validation",
        "selection": (
            "deterministic_seeded_subset_then_frozen_experimental_group_"
            "calibration_evaluation_partition"
        ),
        "seed": seed,
        "n_selected": len(label_array),
        "n_calibration": int(calibration_mask.sum()),
        "n_evaluation": int(evaluation_mask.sum()),
        "evaluation_class_balance": float(evaluation_labels.mean()),
        "source_counts": source_counts,
        "temperature_scaling": {
            "partition": "phase_c_risk_calibration_v1",
            "grid": "91 log-spaced values from 0.5 to 5.0",
            "selected_temperature": temperature,
            "calibration_nll": float(calibration_losses.min()),
        },
        "raw_ece_10_bin": _expected_calibration_error(
            evaluation_labels,
            raw_evaluation,
        ),
        "ece_10_bin": _expected_calibration_error(
            evaluation_labels,
            calibrated_evaluation,
        ),
        "raw_brier": float(
            np.mean((raw_evaluation - evaluation_labels) ** 2)
        ),
        "brier": float(
            np.mean((calibrated_evaluation - evaluation_labels) ** 2)
        ),
        "auprc": _auprc(evaluation_labels, calibrated_evaluation),
    }


def evaluate_formal_reward_hacking(
    model: StructuredProposalModel,
    reference_model: StructuredProposalModel,
    pairs: Sequence[Dict[str, Any]],
    edit_targets_cache: Dict[str, Dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    limit: Optional[int],
    seed: int,
    map_unmapped: bool,
    reference_hash_before: str,
) -> Dict[str, Any]:
    selected = _deterministic_subset(pairs, limit, seed)
    deltas: List[float] = []
    entropies: List[float] = []
    model.eval()
    reference_model.eval()
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch, kept, pair_count = _collate_reaction_pairs(
                selected[start:start + batch_size],
                "preferred_reaction",
                "dispreferred_reaction",
                device,
                map_unmapped=map_unmapped,
            )
            if batch is None:
                continue
            preferred_targets: List[Dict[str, int]] = []
            dispreferred_targets: List[Dict[str, int]] = []
            for row, pair in enumerate(kept):
                preferred = _primary_real_target(
                    str(pair["preferred_reaction"]),
                    batch.graphs[row],
                    edit_targets_cache,
                )
                dispreferred = _primary_real_target(
                    str(pair["dispreferred_reaction"]),
                    batch.graphs[pair_count + row],
                    edit_targets_cache,
                )
                if preferred is None or dispreferred is None:
                    continue
                preferred_targets.append(preferred)
                dispreferred_targets.append(dispreferred)
            if len(preferred_targets) != pair_count:
                continue
            all_loci = torch.tensor(
                [
                    *[target["locus"] for target in preferred_targets],
                    *[target["locus"] for target in dispreferred_targets],
                ],
                device=device,
                dtype=torch.long,
            )
            policy = model(batch, locus_index=all_loci)
            reference = reference_model(batch, locus_index=all_loci)
            policy_delta = (
                compute_action_logp(policy, preferred_targets)
                - compute_action_logp(
                    policy,
                    dispreferred_targets,
                    graph_offset=pair_count,
                )
            )
            reference_delta = (
                compute_action_logp(reference, preferred_targets)
                - compute_action_logp(
                    reference,
                    dispreferred_targets,
                    graph_offset=pair_count,
                )
            )
            deltas.extend((policy_delta - reference_delta).cpu().tolist())
            type_probabilities = F.softmax(policy.type_logits, dim=-1)
            entropy = -(
                type_probabilities
                * type_probabilities.clamp(min=1e-9).log()
            ).sum(-1)
            entropies.extend(entropy.cpu().tolist())
    if not deltas:
        raise RuntimeError("formal reward-hacking validation has no usable pairs")
    reference_hash_after = _state_dict_sha256(reference_model)
    return {
        "split": "validation",
        "n_pairs": len(deltas),
        "mean_log_ratio": float(np.mean(deltas)),
        "max_abs_log_ratio": float(np.max(np.abs(deltas))),
        "preference_accuracy": float(np.mean(np.asarray(deltas) > 0.0)),
        "mean_action_type_entropy": float(np.mean(entropies)),
        "reference_hash_before": reference_hash_before,
        "reference_hash_after": reference_hash_after,
        "reference_frozen": reference_hash_before == reference_hash_after,
    }


def compute_formal_validation_verdict(
    edit_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    risk_metrics: Dict[str, Any],
    reward_metrics: Dict[str, Any],
    risk_source_availability: Dict[str, int],
) -> Dict[str, Any]:
    thresholds = dict(FORMAL_VALIDATION_THRESHOLDS)
    checks = {
        "edit_locus_accuracy": (
            edit_metrics["edit_locus_accuracy"]["value"]
            >= thresholds["edit_locus_accuracy_min"]
        ),
        "edit_type_accuracy": (
            edit_metrics["edit_type_accuracy"]["value"]
            >= thresholds["edit_type_accuracy_min"]
        ),
        "valid_edit_rate": (
            candidate_metrics["valid_edit_rate"]["value"]
            >= thresholds["valid_edit_rate_min"]
        ),
        "candidate_coverage": (
            candidate_metrics["candidate_coverage"]["value"]
            >= thresholds["candidate_coverage_min"]
        ),
        "fnr_calibration": (
            risk_metrics["ece_10_bin"] <= thresholds["fnr_ece_max"]
        ),
        "reward_log_ratio_bounded": (
            reward_metrics["max_abs_log_ratio"]
            <= thresholds["reward_max_abs_log_ratio_max"]
        ),
        "reward_policy_not_collapsed": (
            reward_metrics["mean_action_type_entropy"]
            >= thresholds["reward_action_type_entropy_min"]
        ),
        "reference_frozen": bool(reward_metrics["reference_frozen"]),
        "known_positive_collision_supervision": (
            risk_source_availability.get("known_positive_collision", 0) > 0
        ),
        "observed_competing_product_supervision": (
            risk_source_availability.get("observed_competing_product", 0) > 0
        ),
        "heldout_hte_outcome_supervision": (
            risk_source_availability.get("heldout_hte_outcome", 0) > 0
        ),
    }
    expert_available = risk_source_availability.get("expert_label", 0) > 0
    core_pass = all(checks.values())
    if core_pass and expert_available:
        status = "FORMAL_SOURCE_EXPERT_PASS"
    elif core_pass:
        status = "FORMAL_SOURCE_EXPERT_PARTIAL_EXPERT_LABELS_PENDING"
    else:
        status = "FORMAL_SOURCE_EXPERT_NO_GO"
    return {
        "status": status,
        "core_validation_pass": core_pass,
        "expert_labels_available": expert_available,
        "checks": checks,
        "thresholds": thresholds,
        "claim_boundary": (
            "credible source-expert validation only; no superiority claim "
            "against other negative sources"
        ),
        "formal_evaluation_contract": (
            "real validation supervision only; tiny self-built MLP disabled"
        ),
    }


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
    parser.add_argument("--train-data", type=Path, default=None)
    parser.add_argument("--val-data", type=Path, default=None)
    parser.add_argument("--test-data", type=Path, default=None)
    parser.add_argument(
        "--hte-parquet",
        type=Path,
        default=Path(DEFAULT_HTE_PARQUET),
    )
    parser.add_argument(
        "--collision-review",
        type=Path,
        default=Path(DEFAULT_COLLISION_REVIEW),
    )
    parser.add_argument(
        "--expert-form",
        action="append",
        dest="expert_forms",
        default=None,
        help="Completed expert-review CSV; repeat for multiple reviewers",
    )
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
    parser.add_argument("--formal-edit-val-limit", type=int, default=512)
    parser.add_argument("--formal-candidate-val-limit", type=int, default=128)
    parser.add_argument("--formal-risk-val-limit", type=int, default=2048)
    parser.add_argument("--formal-reward-val-limit", type=int, default=256)
    parser.add_argument(
        "--formal-partition",
        choices=("v1_source_split", "v2_unseen_train_holdout"),
        default="v1_source_split",
        help=(
            "v2 excludes the already-consumed source validation split and "
            "creates a new group-hash holdout from previously unevaluated "
            "source-training groups"
        ),
    )
    parser.add_argument(
        "--formal-run", action="store_true",
        help="Fail closed unless all real G8-C supervision caches and the "
             "post-Stage-3 reference policy are available; disables pseudo-label fallbacks",
    )
    args = parser.parse_args()

    if args.formal_run and args.smoke:
        raise ValueError("--formal-run cannot be combined with --smoke")
    if args.formal_run and args.stage != 0:
        raise ValueError(
            "--formal-run must execute the frozen Stage 1->2->3->4 sequence"
        )
    if not args.formal_run and (
        args.train_data is None
        or args.val_data is None
        or args.test_data is None
    ):
        raise ValueError(
            "exploratory mode requires --train-data, --val-data and --test-data"
        )

    t0 = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_predictions"
    if not args.formal_run:
        raw_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.gpu)
    if args.formal_run and device.type != "cuda":
        raise RuntimeError(
            "formal G8-C training requires an explicitly selected CUDA GPU"
        )
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
    # G8C real-data loading (phase-2 fix)
    # ---------------------------------------------------------------------------
    g8c_data = None
    try:
        g8c_data = load_g8c_training_data(
            hte_parquet_path=str(args.hte_parquet),
            generator=rule_generator,
            use_rule_generator=not args.smoke,
            max_rule_reactions=args.limit_train,
            formal=args.formal_run,
            collision_review_path=str(args.collision_review),
            expert_form_paths=tuple(args.expert_forms or DEFAULT_EXPERT_FORMS),
        )
        print(f"[{PHASE}] g8c data loaded: "
              f"edit_targets={len(g8c_data['edit_targets'])} "
              f"rule_proposals={len(g8c_data['rule_proposals'])} "
              f"competing_pairs={len(g8c_data['competing_pairs'])} "
              f"preference_pairs={len(g8c_data['preference_pairs'])} "
              f"risk_train={len(g8c_data['risk_supervision']['by_split']['train'])}")
        if args.formal_run:
            required_keys = (
                "edit_targets", "rule_proposals",
                "competing_pairs", "preference_pairs",
            )
            missing = [k for k in required_keys
                       if k not in g8c_data or not g8c_data[k]]
            if missing:
                raise RuntimeError(
                    "formal G8-C run requires non-empty caches: "
                    + ", ".join(missing))
    except Exception as exc:
        if args.formal_run:
            raise RuntimeError(
                f"formal G8-C data loading failed; refusing pseudo-supervision: {exc}") from exc
        print(f"[{PHASE}] WARNING: g8c data loading failed: {exc}; "
              "falling back to per-batch target extraction")
        g8c_data = None

    if args.formal_run:
        assert g8c_data is not None
        source_train_reactions = list(
            dict.fromkeys(g8c_data["reactions"]["train"])
        )
        source_val_reactions = list(
            dict.fromkeys(g8c_data["reactions"]["val"])
        )
        test_rxns = list(dict.fromkeys(g8c_data["reactions"]["test"]))
        if args.formal_partition == "v2_unseen_train_holdout":
            train_rxns, val_rxns = _frozen_hash_partition(
                source_train_reactions,
                lambda reaction: reaction,
                namespace="phase_c_v2_reaction_holdout_v1",
            )
            competing_train, competing_val = _frozen_hash_partition(
                g8c_data["competing_pairs_by_split"]["train"],
                lambda pair: pair["context_key"],
                namespace="phase_c_v2_pair_holdout_v1",
            )
            preference_train, val_preference_pairs = _frozen_hash_partition(
                g8c_data["preference_pairs_by_split"]["train"],
                lambda pair: pair["context_key"],
                namespace="phase_c_v2_pair_holdout_v1",
            )
            risk_train, risk_val = _frozen_hash_partition(
                g8c_data["risk_supervision"]["by_split"]["train"],
                lambda row: (
                    row.get("experimental_group")
                    or row.get("record_id")
                    or row["reaction_smiles"]
                ),
                namespace="phase_c_v2_risk_holdout_v1",
            )
            development_excluded = {
                "source_validation_reactions": len(source_val_reactions),
                "v1_validation_competing_pairs": len(
                    g8c_data["competing_pairs_by_split"]["val"]
                ),
                "v1_validation_preference_pairs": len(
                    g8c_data["preference_pairs_by_split"]["val"]
                ),
                "v1_validation_risk_examples": len(
                    g8c_data["risk_supervision"]["by_split"]["val"]
                ),
            }
        else:
            train_rxns = source_train_reactions
            val_rxns = source_val_reactions
            competing_train = g8c_data["competing_pairs"]
            competing_val = g8c_data["competing_pairs_by_split"]["val"]
            preference_train = g8c_data["preference_pairs"]
            val_preference_pairs = (
                g8c_data["preference_pairs_by_split"]["val"]
            )
            risk_train = g8c_data["risk_supervision"]["by_split"]["train"]
            risk_val = g8c_data["risk_supervision"]["by_split"]["val"]
            development_excluded = {}
        if args.limit_train is not None:
            train_rxns = train_rxns[:args.limit_train]
        if args.limit_val is not None:
            val_rxns = val_rxns[:args.limit_val]
        if args.limit_test is not None:
            test_rxns = test_rxns[:args.limit_test]
        missing_rule_targets = [
            reaction for reaction in train_rxns
            if reaction not in g8c_data["rule_proposals"]
        ]
        if missing_rule_targets:
            raise RuntimeError(
                "formal G8-C rule cache does not cover the frozen training "
                f"subset ({len(missing_rule_targets)} missing)"
            )
        risk_train = [
            row for row in risk_train
            if _featurize_risk_safe(str(row["reaction_smiles"])) is not None
        ]
        risk_val = [
            row for row in risk_val
            if _featurize_risk_safe(str(row["reaction_smiles"])) is not None
        ]
        if not risk_train or not risk_val:
            raise RuntimeError(
                "formal G8-C requires non-empty featurizable train and "
                "validation risk supervision"
            )
        if not val_preference_pairs:
            raise RuntimeError(
                "formal G8-C requires held-out validation preference pairs"
            )
        print(
            f"[{PHASE}] frozen HTE split: train={len(train_rxns)} "
            f"val={len(val_rxns)} sealed_test={len(test_rxns)} "
            f"risk_train={len(risk_train)} risk_val={len(risk_val)} "
            f"train_pairs={len(competing_train)} "
            f"val_preferences={len(val_preference_pairs)} "
            f"partition={args.formal_partition}"
        )
        formal_data_audit = {
            **g8c_data["data_audit"],
            "formal_partition": args.formal_partition,
            "n_model_train_reactions": len(train_rxns),
            "n_model_validation_reactions": len(val_rxns),
            "n_model_train_competing_pairs": len(competing_train),
            "n_model_validation_competing_pairs": len(competing_val),
            "n_model_train_preference_pairs": len(preference_train),
            "n_model_validation_preference_pairs": len(
                val_preference_pairs
            ),
            "n_model_train_risk_examples": len(risk_train),
            "n_model_validation_risk_examples": len(risk_val),
            "development_excluded": development_excluded,
        }
    else:
        risk_train = []
        risk_val = []
        val_preference_pairs = []
        competing_train = []
        preference_train = []
        formal_data_audit = {}

    model = StructuredProposalModel(
        hidden_dim=args.hidden_dim, num_heads=args.num_heads,
        num_layers=DEFAULT_NUM_LAYERS, dropout=args.dropout).to(device)

    stages = [1, 2, 3, 4] if args.stage <= 0 else [args.stage]
    n_rounds = 1 if args.smoke else args.num_rounds
    log: List[dict] = []
    # Build the frozen reference model for Stage-4 DPO only after Stage 3
    # has completed in the current round.  Snapshotting before the stage
    # loop would incorrectly use the pre-reconstruction policy.
    ref_model = None
    reference_hash_before = ""
    for rnd in range(n_rounds):
        for st in stages:
            if st == 4:
                import copy
                ref_model = copy.deepcopy(model)
                ref_model.eval()
                for p in ref_model.parameters():
                    p.requires_grad_(False)
                reference_hash_before = _state_dict_sha256(ref_model)
            log = train_stage(
                model, st, train_rxns, val_rxns, rule_generator,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                device=device, seed=args.seed + rnd * 100, log=log,
                map_unmapped=args.map_unmapped,
                edit_targets_cache=g8c_data['edit_targets'] if g8c_data else None,
                rule_proposals_cache=g8c_data['rule_proposals'] if g8c_data else None,
                competing_pairs_cache=(
                    competing_train if args.formal_run
                    else (g8c_data["competing_pairs"] if g8c_data else None)
                ),
                preference_pairs_cache=(
                    preference_train if args.formal_run
                    else (g8c_data["preference_pairs"] if g8c_data else None)
                ),
                risk_examples_cache=risk_train if args.formal_run else None,
                ref_model=ref_model if st == 4 else None,
                formal_run=args.formal_run)
            if args.formal_run and st == 4:
                assert ref_model is not None
                if _state_dict_sha256(ref_model) != reference_hash_before:
                    raise RuntimeError(
                        "formal Stage-4 mutated the frozen reference policy"
                    )

    if args.formal_run:
        assert g8c_data is not None
        assert ref_model is not None
        edit_validation = evaluate_formal_edit_validation(
            model,
            val_rxns,
            g8c_data["edit_targets"],
            device=device,
            batch_size=args.batch_size,
            limit=args.formal_edit_val_limit,
            seed=args.seed + 11,
            map_unmapped=args.map_unmapped,
        )
        candidate_validation = evaluate_formal_candidate_generation(
            model,
            val_rxns,
            device=device,
            top_k=args.top_k,
            limit=args.formal_candidate_val_limit,
            seed=args.seed + 23,
            map_unmapped=args.map_unmapped,
        )
        risk_validation = evaluate_formal_risk_validation(
            model,
            risk_val,
            device=device,
            batch_size=args.batch_size,
            limit=args.formal_risk_val_limit,
            seed=args.seed + 37,
            map_unmapped=args.map_unmapped,
        )
        reward_validation = evaluate_formal_reward_hacking(
            model,
            ref_model,
            val_preference_pairs,
            g8c_data["edit_targets"],
            device=device,
            batch_size=args.batch_size,
            limit=args.formal_reward_val_limit,
            seed=args.seed + 41,
            map_unmapped=args.map_unmapped,
            reference_hash_before=reference_hash_before,
        )
        verdict = compute_formal_validation_verdict(
            edit_validation,
            candidate_validation,
            risk_validation,
            reward_validation,
            g8c_data["risk_supervision"]["source_availability"],
        )
        formal_result = {
            "phase": PHASE,
            "version": "formal_source_expert_v1",
            "status": verdict["status"],
            "validation_only": True,
            "sealed_test_untouched": True,
            "tiny_self_built_mlp_evaluation": "DISABLED",
            "data_audit": formal_data_audit,
            "risk_source_availability": (
                g8c_data["risk_supervision"]["source_availability"]
            ),
            "edit_validation": edit_validation,
            "candidate_validation": candidate_validation,
            "risk_validation": risk_validation,
            "reward_hacking_validation": reward_validation,
            "verdict": verdict,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        with open(output_dir / "formal_validation.json", "w") as handle:
            json.dump(formal_result, handle, indent=2)
        with open(output_dir / "go_no_go.json", "w") as handle:
            json.dump({
                "phase": PHASE,
                "status": verdict["status"],
                "claim_boundary": verdict["claim_boundary"],
                "core_validation_pass": verdict["core_validation_pass"],
                "expert_labels_available": verdict["expert_labels_available"],
                "formal_validation_artifact": "formal_validation.json",
                "sealed_test_untouched": True,
            }, handle, indent=2)
        torch.save({
            "state_dict": model.state_dict(),
            "hidden_dim": model.hidden_dim,
            "architecture": "StructuredProposalModel",
            "action_schema": {
                name: int(member) for name, member in EditType.__members__.items()
            },
            "formal_validation_status": verdict["status"],
        }, str(output_dir / "model_checkpoint.pt"))
        with open(output_dir / "train_log.json", "w") as handle:
            json.dump(log, handle, indent=2)
        with open(output_dir / "run_manifest.json", "w") as handle:
            json.dump({
                "phase": PHASE,
                "version": "formal_source_expert_v1",
                "formal_run": True,
                "formal_partition": args.formal_partition,
                "gpu": str(device),
                "n_train_reactions": len(train_rxns),
                "n_validation_reactions": len(val_rxns),
                "n_sealed_test_reactions_not_evaluated": len(test_rxns),
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "num_rounds": n_rounds,
                "stages": [
                    "real_edit_reconstruction",
                    "actual_rule_action_imitation",
                    "same_context_competing_outcomes",
                    "risk_adjusted_real_action_preference_optimization",
                ],
                "formal_validation_thresholds": FORMAL_VALIDATION_THRESHOLDS,
                "seed": args.seed,
                "reference_hash": reference_hash_before,
            }, handle, indent=2)
        with open(output_dir / "environment.json", "w") as handle:
            json.dump({
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "cuda_device": str(device),
                "cuda_name": torch.cuda.get_device_name(device),
            }, handle, indent=2)
        hashes = {}
        for path in [
            args.hte_parquet,
            args.collision_review,
            *[Path(path) for path in (args.expert_forms or DEFAULT_EXPERT_FORMS)],
        ]:
            if path and Path(path).exists():
                digest = hashlib.sha256()
                with open(path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(8192), b""):
                        digest.update(chunk)
                hashes[str(path)] = digest.hexdigest()
        with open(output_dir / "input_hashes.json", "w") as handle:
            json.dump(hashes, handle, indent=2)
        with open(output_dir / "commands.log", "w") as handle:
            handle.write(" ".join(sys.argv) + "\n")
        print(f"\n[{PHASE}] formal_status={verdict['status']}")
        print(
            f"[{PHASE}] edit_locus="
            f"{edit_validation['edit_locus_accuracy']['value']:.4f} "
            f"edit_type={edit_validation['edit_type_accuracy']['value']:.4f} "
            f"valid={candidate_validation['valid_edit_rate']['value']:.4f} "
            f"coverage={candidate_validation['candidate_coverage']['value']:.4f} "
            f"fnr_ece={risk_validation['ece_10_bin']:.4f}"
        )
        print(f"[{PHASE}] outputs in {output_dir}")
        return

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
