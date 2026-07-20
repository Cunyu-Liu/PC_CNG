"""Learnable GNN-based reaction-center edit decoder (MPNN).

P1-05 deliverable: replaces the rule-based MLP in ``train_reaction_center_edit_decoder.py``
with a pure-PyTorch MPNN that ranks the observed reaction-center anchor above
plausible alternative anchors. At generation time, high-scoring non-observed
anchors are used as type-1 boundary negatives.

Implementation note: ``torch_geometric`` is NOT available in this environment,
so message passing is implemented with pure PyTorch (``index_add_`` for
scatter-sum, manual softmax for attention).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn

from .atom_mapped_graph_edit import extract_reaction_center, has_atom_mapping
from .chem_utils import join_reaction, molecule_parts, split_reaction
from .reaction_boundary_generator import RXNMapperAdapter
from .reaction_center_edit_decoder import (
    ANCHOR_ATOMIC_NUMS,
    ATOM_VOCAB,
    _candidate_anchor_atoms,
    _find_product_mol_with_maps,
    _looks_like_transfer_fragment,
    _map_to_idx,
    move_formed_bond_in_product,
)

try:  # pragma: no cover - environment dependent
    from rdkit import Chem, RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore


# ---------------------------------------------------------------------------
# Feature dimensions
# ---------------------------------------------------------------------------

ATOM_EMBED_VOCAB_SIZE: int = len(ATOM_VOCAB) + 1  # +1 for "other"
ATOM_NUM_FEATS: int = 6  # degree, formal_charge, total_h, is_aromatic, is_in_ring, mass_log
ATOM_FEAT_DIM: int = ATOM_EMBED_VOCAB_SIZE + ATOM_NUM_FEATS  # 11
BOND_FEAT_DIM: int = 6  # 4 bond-type one-hot + is_in_ring + is_conjugated


def _atom_feature_vector(atom) -> List[float]:
    """Build a fixed-length atom feature vector.

    Layout: [one-hot atomic_num over ATOM_VOCAB + "other" (11),
             degree, formal_charge, total_h, is_aromatic, is_in_ring, mass_log]
    Total: 17 dimensions.
    """
    atomic_num = atom.GetAtomicNum()
    one_hot = [0.0] * ATOM_EMBED_VOCAB_SIZE
    matched = False
    for idx, (_, number) in enumerate(ATOM_VOCAB):
        if atomic_num == number:
            one_hot[idx] = 1.0
            matched = True
            break
    if not matched:
        one_hot[-1] = 1.0  # "other"
    numerical = [
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        math.log1p(max(0.0, float(atom.GetMass()))),
    ]
    return one_hot + numerical


def _bond_feature_vector(bond) -> List[float]:
    """Build a fixed-length bond feature vector (6-d)."""
    bond_type = str(bond.GetBondType())
    type_onehot = {
        "SINGLE": [1.0, 0.0, 0.0, 0.0],
        "DOUBLE": [0.0, 1.0, 0.0, 0.0],
        "TRIPLE": [0.0, 0.0, 1.0, 0.0],
        "AROMATIC": [0.0, 0.0, 0.0, 1.0],
    }.get(bond_type, [1.0, 0.0, 0.0, 0.0])
    return type_onehot + [
        1.0 if bond.IsInRing() else 0.0,
        1.0 if bond.GetIsConjugated() else 0.0,
    ]


# ---------------------------------------------------------------------------
# Graph data structures
# ---------------------------------------------------------------------------


@dataclass
class ReactionGraphData:
    """Graph representation of a reaction's product molecule for one formed-bond group.

    All tensors are CPU tensors; callers should ``.to(device)`` them.
    """

    atom_features: torch.Tensor  # (N, ATOM_FEAT_DIM) float32
    edge_index: torch.Tensor  # (2, E) long - directed edges (both directions)
    edge_features: torch.Tensor  # (E, BOND_FEAT_DIM) float32
    atom_map_nums: torch.Tensor  # (N,) long - atom map numbers (0 if unmapped)
    true_anchor_idx: int  # index into atom_features
    candidate_anchor_indices: List[int]  # indices into atom_features (includes true_anchor)
    # metadata for negative generation
    source_id: str
    pair_id: str
    mapped_reaction: str
    reactants: str
    product: str
    fragment_map: int
    true_anchor_map: int
    atom_map_to_idx: Dict[int, int] = field(default_factory=dict)


@dataclass
class BatchedGraph:
    """Batched graph representation (concatenation of multiple graphs)."""

    atom_features: torch.Tensor  # (sum_N, F)
    edge_index: torch.Tensor  # (2, sum_E)
    edge_features: torch.Tensor  # (sum_E, F_bond)
    batch_idx: torch.Tensor  # (sum_N,) - graph index for each atom
    # per-graph metadata
    true_anchor_indices: List[int]  # absolute index in concatenated atoms
    candidate_anchor_indices_per_graph: List[List[int]]  # absolute indices
    graphs: List[ReactionGraphData]  # original graph data for post-processing


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------


def _build_product_graph(
    mol,
    atom_map_to_idx: Dict[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build graph tensors from an RDKit molecule.

    Returns (atom_features, edge_index, edge_features, atom_map_nums).
    """
    n_atoms = mol.GetNumAtoms()
    atom_feats: List[List[float]] = []
    atom_maps: List[int] = []
    for atom in mol.GetAtoms():
        atom_feats.append(_atom_feature_vector(atom))
        atom_maps.append(int(atom.GetAtomMapNum()))
    atom_features = torch.tensor(atom_feats, dtype=torch.float32)
    atom_map_nums = torch.tensor(atom_maps, dtype=torch.long)

    edges_src: List[int] = []
    edges_dst: List[int] = []
    edge_feats: List[List[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _bond_feature_vector(bond)
        # both directions
        edges_src.extend([i, j])
        edges_dst.extend([j, i])
        edge_feats.extend([bf, bf])
    if edges_src:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_features = torch.tensor(edge_feats, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_features = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float32)
    return atom_features, edge_index, edge_features, atom_map_nums


def featurize_atom_mapped_reaction(
    reaction_smiles: str,
    source_id: str = "",
    split: str = "train",
    label_type: str = "positive",
    mapper: Optional[RXNMapperAdapter] = None,
    map_unmapped: bool = False,
    max_anchor_distance: int = 6,
    max_candidates_per_pair: int = 8,
) -> Tuple[List[ReactionGraphData], str]:
    """Parse an atom-mapped reaction SMILES into per-formed-bond graph data.

    Returns (list_of_graph_data, reason). ``reason == "ok"`` on success.
    Each graph corresponds to one formed-bond group (fragment -> true anchor)
    and contains the candidate anchor indices.
    """
    if Chem is None:
        return [], "rdkit_unavailable"

    mapped_reaction = reaction_smiles
    if not has_atom_mapping(mapped_reaction):
        if not map_unmapped:
            return [], "missing_atom_mapping"
        mapper = mapper or RXNMapperAdapter()
        mapped = mapper.map_reaction(reaction_smiles)
        if not mapped or not has_atom_mapping(mapped):
            return [], "mapping_failed"
        mapped_reaction = mapped

    try:
        reactants, agents, product = split_reaction(mapped_reaction)
    except ValueError:
        return [], "invalid_reaction"

    edit = extract_reaction_center(mapped_reaction)
    if not edit.mapped or not edit.formed_bonds:
        return [], "no_formed_bond"

    graphs: List[ReactionGraphData] = []
    for formed_index, (left_map, right_map, _order) in enumerate(edit.formed_bonds):
        mol = _find_product_mol_with_maps(product, {left_map, right_map})
        if mol is None:
            continue
        mapping = _map_to_idx(mol)
        left_idx = mapping[left_map]
        right_idx = mapping[right_map]
        left_atom = mol.GetAtomWithIdx(left_idx)
        right_atom = mol.GetAtomWithIdx(right_idx)
        if _looks_like_transfer_fragment(left_atom, right_atom):
            fragment_map, true_anchor_map = left_map, right_map
            fragment_idx, true_anchor_idx = left_idx, right_idx
        elif _looks_like_transfer_fragment(right_atom, left_atom):
            fragment_map, true_anchor_map = right_map, left_map
            fragment_idx, true_anchor_idx = right_idx, left_idx
        else:
            continue

        # candidate anchors (true anchor + alternatives)
        candidate_atoms = [mol.GetAtomWithIdx(true_anchor_idx)]
        candidate_atoms.extend(
            _candidate_anchor_atoms(mol, fragment_idx, true_anchor_idx, max_anchor_distance)
        )
        seen: Set[int] = set()
        candidate_indices: List[int] = []
        candidate_maps: List[int] = []
        for cand in candidate_atoms:
            cand_map = int(cand.GetAtomMapNum())
            if not cand_map or cand_map in seen:
                continue
            seen.add(cand_map)
            candidate_indices.append(cand.GetIdx())
            candidate_maps.append(cand_map)
            if len(candidate_indices) >= max_candidates_per_pair + 1:
                break

        # must have at least one true + one candidate
        if true_anchor_idx not in candidate_indices:
            continue
        if len(candidate_indices) < 2:
            continue

        atom_features, edge_index, edge_features, atom_map_nums = _build_product_graph(mol, mapping)
        pair_id = f"{source_id}|formed{formed_index}|{fragment_map}->{true_anchor_map}"
        graphs.append(
            ReactionGraphData(
                atom_features=atom_features,
                edge_index=edge_index,
                edge_features=edge_features,
                atom_map_nums=atom_map_nums,
                true_anchor_idx=true_anchor_idx,
                candidate_anchor_indices=candidate_indices,
                source_id=source_id,
                pair_id=pair_id,
                mapped_reaction=mapped_reaction,
                reactants=reactants,
                product=product,
                fragment_map=fragment_map,
                true_anchor_map=true_anchor_map,
                atom_map_to_idx=dict(mapping),
            )
        )

    if not graphs:
        return [], "no_candidate_anchor"
    return graphs, "ok"


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def collate_graphs(graphs: List[ReactionGraphData]) -> BatchedGraph:
    """Batch a list of ReactionGraphData into a single concatenated graph.

    Edge indices are offset by each graph's atom offset.
    ``true_anchor_indices`` and ``candidate_anchor_indices_per_graph`` use
    absolute indices into the concatenated atom tensor.
    """
    if not graphs:
        return BatchedGraph(
            atom_features=torch.zeros((0, ATOM_FEAT_DIM), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_features=torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float32),
            batch_idx=torch.zeros((0,), dtype=torch.long),
            true_anchor_indices=[],
            candidate_anchor_indices_per_graph=[],
            graphs=[],
        )

    all_atom_feats: List[torch.Tensor] = []
    all_edge_index: List[torch.Tensor] = []
    all_edge_feats: List[torch.Tensor] = []
    all_batch_idx: List[torch.Tensor] = []
    true_anchors: List[int] = []
    candidates_per_graph: List[List[int]] = []

    atom_offset = 0
    for graph_idx, graph in enumerate(graphs):
        n_atoms = graph.atom_features.shape[0]
        all_atom_feats.append(graph.atom_features)
        all_batch_idx.append(torch.full((n_atoms,), graph_idx, dtype=torch.long))

        if graph.edge_index.shape[1] > 0:
            offset_edge = graph.edge_index + atom_offset
            all_edge_index.append(offset_edge)
            all_edge_feats.append(graph.edge_features)

        true_anchors.append(atom_offset + graph.true_anchor_idx)
        candidates_per_graph.append([atom_offset + idx for idx in graph.candidate_anchor_indices])
        atom_offset += n_atoms

    return BatchedGraph(
        atom_features=torch.cat(all_atom_feats, dim=0),
        edge_index=torch.cat(all_edge_index, dim=1) if all_edge_index else torch.zeros((2, 0), dtype=torch.long),
        edge_features=torch.cat(all_edge_feats, dim=0) if all_edge_feats else torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float32),
        batch_idx=torch.cat(all_batch_idx, dim=0),
        true_anchor_indices=true_anchors,
        candidate_anchor_indices_per_graph=candidates_per_graph,
        graphs=list(graphs),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class _MessageFunction(nn.Module):
    """MLP that computes a message from (h_i, h_j, e_ij)."""

    def __init__(self, hidden_dim: int, bond_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + bond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor, e_ij: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([h_i, h_j, e_ij], dim=-1))


class _GRUUpdate(nn.Module):
    """GRU-based hidden state update."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        return self.gru(message, h)


def _scatter_sum(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum scatter: aggregate ``values`` (E, D) by ``index`` (E,) into (dim_size, D)."""
    out = torch.zeros(dim_size, values.shape[-1], dtype=values.dtype, device=values.device)
    out.index_add_(0, index, values)
    return out


def _scatter_max(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Max scatter: aggregate ``values`` (E, D) by ``index`` (E,) into (dim_size, D).

    Uses non-in-place operations to preserve autograd.
    """
    init = torch.full((dim_size, values.shape[-1]), float("-inf"), dtype=values.dtype, device=values.device)
    out = init.scatter_reduce(0, index.unsqueeze(-1).expand_as(values), values, reduce="amax", include_self=True)
    # replace -inf (atoms with no incoming edges) with 0, non-in-place
    out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    return out


def _segment_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Segment-wise softmax over edges grouped by destination atom.

    ``scores``: (E, 1) or (E,) raw scores. ``index``: (E,) destination atom idx.
    """
    scores = scores.squeeze(-1) if scores.dim() == 2 and scores.shape[-1] == 1 else scores
    scores = scores - _scatter_max(scores.unsqueeze(-1), index, dim_size).squeeze(-1).index_select(0, index)
    exp = scores.exp()
    denom = _scatter_sum(exp.unsqueeze(-1), index, dim_size).squeeze(-1).index_select(0, index)
    denom = denom.clamp(min=1e-9)
    return exp / denom


class LearnedGraphEditDecoder(nn.Module):
    """GNN-based reaction-center anchor ranker (pure-PyTorch MPNN).

    Architecture:
      1. Atom encoder: atom features -> hidden embedding
      2. Bond encoder: bond features -> bond embedding
      3. MPNN: ``num_rounds`` of message passing
         - message = MLP_bond(h_i, h_j, e_ij)
         - aggregation = sum + max
         - update = GRU(h_i, aggregated_message)
      4. Global context: mean-pool of atom embeddings per graph
      5. Anchor scoring: MLP_anchor(h_atom, h_global) -> per-atom score

    Higher score = more likely to be the true reaction-center anchor.
    """

    def __init__(
        self,
        atom_feat_dim: int = ATOM_FEAT_DIM,
        bond_feat_dim: int = BOND_FEAT_DIM,
        hidden_dim: int = 128,
        num_rounds: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.atom_feat_dim = atom_feat_dim
        self.bond_feat_dim = bond_feat_dim
        self.hidden_dim = hidden_dim
        self.num_rounds = num_rounds

        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bond_encoder = nn.Sequential(
            nn.Linear(bond_feat_dim, hidden_dim),
            nn.ReLU(),
        )
        self.message_fn = nn.ModuleList(
            [_MessageFunction(hidden_dim, hidden_dim) for _ in range(num_rounds)]
        )
        self.update_fn = nn.ModuleList([_GRUUpdate(hidden_dim) for _ in range(num_rounds)])
        self.dropout = nn.Dropout(dropout)
        self.anchor_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch_idx: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            atom_features: (N, F_atom)
            edge_index: (2, E) directed
            edge_features: (E, F_bond)
            batch_idx: (N,) graph index per atom
            num_graphs: int - number of graphs in the batch

        Returns:
            per_atom_scores: (N,) float - anchor score for every atom
        """
        h = self.atom_encoder(atom_features)
        e = self.bond_encoder(edge_features) if edge_features.shape[0] > 0 else edge_features

        n_atoms = h.shape[0]
        if edge_index.shape[1] > 0:
            src = edge_index[0]
            dst = edge_index[1]
            for round_idx in range(self.num_rounds):
                h_src = h.index_select(0, src)
                h_dst = h.index_select(0, dst)
                messages = self.message_fn[round_idx](h_dst, h_src, e)
                # attention-weighted aggregation (softmax over edges per destination atom)
                attn_scores = messages.mean(dim=-1, keepdim=True)  # (E, 1)
                attn_weights = _segment_softmax(attn_scores, dst, n_atoms)  # (E,)
                weighted = messages * attn_weights.unsqueeze(-1)  # (E, hidden)
                agg_sum = _scatter_sum(weighted, dst, n_atoms)
                agg_max = _scatter_max(messages, dst, n_atoms)
                agg = agg_sum + agg_max
                h = self.update_fn[round_idx](h, agg)
                h = self.dropout(h)

        # global context per graph
        global_h = _scatter_mean(h, batch_idx, num_graphs)  # (num_graphs, hidden)
        # broadcast global to each atom
        global_per_atom = global_h.index_select(0, batch_idx)  # (N, hidden)
        # anchor score
        scores = self.anchor_scorer(torch.cat([h, global_per_atom], dim=-1)).squeeze(-1)
        return scores


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean scatter: average ``values`` (N, D) by ``index`` (N,) into (dim_size, D)."""
    total = _scatter_sum(values, index, dim_size)
    count = torch.zeros(dim_size, 1, dtype=values.dtype, device=values.device)
    count.index_add_(0, index, torch.ones(values.shape[0], 1, dtype=values.dtype, device=values.device))
    return total / count.clamp(min=1.0)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def pairwise_margin_loss(
    true_scores: torch.Tensor,
    candidate_scores: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Pairwise margin ranking loss.

    For each (true, candidate) pair:
      loss = max(0, margin - (score_true - score_candidate))

    Args:
        true_scores: (B,) scores of true anchors
        candidate_scores: (B, K) scores of candidate anchors (K candidates per graph)
        margin: required margin between true and candidate scores

    Returns:
        scalar loss (mean over all pairs)
    """
    if candidate_scores.numel() == 0:
        return torch.tensor(0.0, dtype=true_scores.dtype, device=true_scores.device)
    diff = true_scores.unsqueeze(-1) - candidate_scores  # (B, K)
    loss = torch.clamp(margin - diff, min=0.0)
    return loss.mean()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def predict_anchor_scores(
    model: LearnedGraphEditDecoder,
    graph: ReactionGraphData,
    device: torch.device,
) -> torch.Tensor:
    """Return per-candidate anchor scores for a single reaction graph.

    Returns a 1-D tensor of length ``len(graph.candidate_anchor_indices)``.
    """
    model.eval()
    batch_idx = torch.zeros(graph.atom_features.shape[0], dtype=torch.long)
    with torch.no_grad():
        all_scores = model(
            graph.atom_features.to(device),
            graph.edge_index.to(device),
            graph.edge_features.to(device),
            batch_idx.to(device),
            num_graphs=1,
        )
    candidate_indices = torch.tensor(graph.candidate_anchor_indices, dtype=torch.long, device=device)
    return all_scores.index_select(0, candidate_indices).cpu()


def predict_batch_anchor_scores(
    model: LearnedGraphEditDecoder,
    batched: BatchedGraph,
    device: torch.device,
) -> List[torch.Tensor]:
    """Return per-graph candidate anchor scores for a batched graph.

    Returns a list of 1-D tensors, one per graph.
    """
    model.eval()
    num_graphs = len(batched.graphs)
    if num_graphs == 0:
        return []
    with torch.no_grad():
        all_scores = model(
            batched.atom_features.to(device),
            batched.edge_index.to(device),
            batched.edge_features.to(device),
            batched.batch_idx.to(device),
            num_graphs=num_graphs,
        )
    all_scores = all_scores.cpu()
    results: List[torch.Tensor] = []
    for candidates in batched.candidate_anchor_indices_per_graph:
        if not candidates:
            results.append(torch.zeros((0,), dtype=torch.float32))
            continue
        idx = torch.tensor(candidates, dtype=torch.long)
        results.append(all_scores.index_select(0, idx))
    return results


# ---------------------------------------------------------------------------
# Negative generation
# ---------------------------------------------------------------------------


@dataclass
class GeneratedNegative:
    """A generated boundary negative reaction."""

    source_id: str
    pair_id: str
    positive_reaction: str
    candidate_reaction: str
    reactants: str
    parent_product: str
    candidate_product: str
    fragment_map: int
    true_anchor_map: int
    candidate_anchor_map: int
    decoder_score: float
    decoder_rank: int


def generate_boundary_negatives(
    model: LearnedGraphEditDecoder,
    reaction_smiles: str,
    source_id: str = "",
    top_k: int = 2,
    mapper: Optional[RXNMapperAdapter] = None,
    map_unmapped: bool = False,
    device: Optional[torch.device] = None,
    min_product_similarity: float = 0.0,
    max_anchor_distance: int = 6,
) -> Tuple[List[GeneratedNegative], str]:
    """Generate boundary negatives for a single reaction.

    Returns (negatives, reason). ``reason == "ok"`` on success.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs, reason = featurize_atom_mapped_reaction(
        reaction_smiles,
        source_id=source_id,
        mapper=mapper,
        map_unmapped=map_unmapped,
        max_anchor_distance=max_anchor_distance,
    )
    if reason != "ok":
        return [], reason

    negatives: List[GeneratedNegative] = []
    for graph in graphs:
        scores = predict_anchor_scores(model, graph, device)
        # rank candidates by score, exclude the true anchor
        ranked: List[Tuple[float, int]] = []
        for local_idx, cand_atom_idx in enumerate(graph.candidate_anchor_indices):
            if cand_atom_idx == graph.true_anchor_idx:
                continue
            ranked.append((float(scores[local_idx].item()), cand_atom_idx))
        ranked.sort(key=lambda x: x[0], reverse=True)

        for rank, (score, cand_atom_idx) in enumerate(ranked[:top_k], start=1):
            cand_map = int(graph.atom_map_nums[cand_atom_idx].item())
            if cand_map == 0:
                continue
            candidate_product = move_formed_bond_in_product(
                graph.product,
                graph.fragment_map,
                graph.true_anchor_map,
                cand_map,
            )
            if not candidate_product:
                continue
            candidate_reaction = join_reaction(graph.reactants, candidate_product, "")
            negatives.append(
                GeneratedNegative(
                    source_id=graph.source_id,
                    pair_id=graph.pair_id,
                    positive_reaction=graph.mapped_reaction,
                    candidate_reaction=candidate_reaction,
                    reactants=graph.reactants,
                    parent_product=graph.product,
                    candidate_product=candidate_product,
                    fragment_map=graph.fragment_map,
                    true_anchor_map=graph.true_anchor_map,
                    candidate_anchor_map=cand_map,
                    decoder_score=score,
                    decoder_rank=rank,
                )
            )
    return negatives, "ok"


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: LearnedGraphEditDecoder,
    path: str,
    extra: Optional[Dict] = None,
) -> None:
    """Save a model checkpoint with hyperparameters."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "atom_feat_dim": model.atom_feat_dim,
        "bond_feat_dim": model.bond_feat_dim,
        "hidden_dim": model.hidden_dim,
        "num_rounds": model.num_rounds,
        "architecture": "LearnedGraphEditDecoder",
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: torch.device) -> LearnedGraphEditDecoder:
    """Load a model from a checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LearnedGraphEditDecoder(
        atom_feat_dim=int(checkpoint.get("atom_feat_dim", ATOM_FEAT_DIM)),
        bond_feat_dim=int(checkpoint.get("bond_feat_dim", BOND_FEAT_DIM)),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        num_rounds=int(checkpoint.get("num_rounds", 3)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
