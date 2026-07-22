"""Pure-PyTorch GAT backbone for P4-G3 augmentation experiments.

Implements a GAT-style reaction encoder without torch_geometric dependency,
plus RDKit-based molecule-to-graph featurization.  Used as the second backbone
(different inductive bias from Chemformer transformer) per P4-G3 spec.

Architecture:
    atoms -> embedding -> GAT layers (multi-head attention over graph)
    -> global mean pool -> MLP -> reaction embedding -> ranking head
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RDKit molecule-to-graph featurization
# ---------------------------------------------------------------------------

# Atom features: atomic_num(6) + degree(4) + formal_charge(4) + hybridization(4)
# + aromatic(2) + num_H(4) + chirality(3) = 27
# Bond features: bond_type(4) + stereo(4) + conjugated(2) + ring(2) = 12

_ATOM_FEATURE_DIM = 27
_BOND_FEATURE_DIM = 12

def _one_hot(x: int, num_classes: int) -> List[float]:
    return [1.0 if i == x else 0.0 for i in range(num_classes)]


def atom_features(atom) -> List[float]:
    """Compute 27-d atom feature vector."""
    features = []
    # Atomic number (1-53, covering most organic atoms)
    features.extend(_one_hot(min(atom.GetAtomicNum() - 1, 52), 53)[:6])
    # Degree (0-3)
    features.extend(_one_hot(min(atom.GetDegree(), 3), 4))
    # Formal charge (-2 to +1)
    fc = atom.GetFormalCharge()
    fc_idx = min(max(fc + 2, 0), 3)
    features.extend(_one_hot(fc_idx, 4))
    # Hybridization
    hybrid_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0}  # S, SP, SP2, SP3, SP3D -> 4 classes
    hyb = atom.GetHybridization()
    features.extend(_one_hot(hybrid_map.get(int(hyb), 0), 4))
    # Aromatic
    features.extend(_one_hot(int(atom.GetIsAromatic()), 2))
    # Total number of Hs (0-3)
    features.extend(_one_hot(min(atom.GetTotalNumHs(), 3), 4))
    # Chirality
    features.extend(_one_hot(int(atom.GetChiralTag()), 3))
    return features


def bond_features(bond) -> List[float]:
    """Compute 12-d bond feature vector."""
    features = []
    # Bond type (single, double, triple, aromatic)
    bt_map = {1: 0, 2: 1, 3: 2, 12: 3}  # SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=12
    bt = int(bond.GetBondType())
    features.extend(_one_hot(bt_map.get(bt, 0), 4))
    # Stereo
    features.extend(_one_hot(min(int(bond.GetStereo()), 3), 4))
    # Conjugated
    features.extend(_one_hot(int(bond.GetIsConjugated()), 2))
    # In ring
    features.extend(_one_hot(int(bond.IsInRing()), 2))
    return features


def mol_to_graph(smiles: str) -> Optional[Dict]:
    """Convert SMILES string to graph dict with x, edge_index, edge_attr.

    Returns None if SMILES is invalid.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # Node features
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float32)  # [N, 27]

    # Edge features (undirected -> both directions)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # Forward edge
        edge_index.append([i, j])
        edge_attr.append(bf)
        # Backward edge
        edge_index.append([j, i])
        edge_attr.append(bf)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # [2, E]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # [E, 12]
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, _BOND_FEATURE_DIM), dtype=torch.float32)

    return {
        "x": x,                     # [N, 27]
        "edge_index": edge_index,   # [2, E]
        "edge_attr": edge_attr,     # [E, 12]
        "num_nodes": x.size(0),
    }


def collate_graphs(graphs: List[Dict]) -> Dict:
    """Collate a list of graph dicts into a batched graph.

    Returns dict with:
        x: [total_nodes, 27]
        edge_index: [2, total_edges]
        edge_attr: [total_edges, 12]
        batch: [total_nodes] — graph index for each node
        num_graphs: int
    """
    xs, edge_indices, edge_attrs, batch_ids = [], [], [], []
    node_offset = 0
    for i, g in enumerate(graphs):
        n = g["num_nodes"]
        xs.append(g["x"])
        edge_indices.append(g["edge_index"] + node_offset)
        edge_attrs.append(g["edge_attr"])
        batch_ids.extend([i] * n)
        node_offset += n

    return {
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "batch": torch.tensor(batch_ids, dtype=torch.long),
        "num_graphs": len(graphs),
    }


# ---------------------------------------------------------------------------
# Pure-PyTorch GAT layer
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """Single GAT layer (multi-head graph attention) without torch_geometric.

    Implements the GAT v1 mechanism:
        h_i' = ||_k=1^K sigma(sum_j alpha_ij^k W^k h_j)
    where alpha_ij are attention coefficients computed over neighbor pairs.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 edge_dim: int = 0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.head_dim = out_dim // heads

        # Linear transform: shared weight then reshape per head
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

        # Attention parameters: [heads, 2 * head_dim]
        self.att = nn.Parameter(torch.empty(1, heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.att)

        # Optional edge feature projection
        self.edge_dim = edge_dim
        if edge_dim > 0:
            self.lin_edge = nn.Linear(edge_dim, out_dim, bias=False)
        else:
            self.lin_edge = None

        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim] node features
            edge_index: [2, E] source and target node indices
            edge_attr: [E, edge_dim] edge features (optional)
            batch: [N] graph assignment for each node (unused, kept for API compat)

        Returns:
            [N, out_dim] updated node features
        """
        N = x.size(0)
        E = edge_index.size(1)
        H, D = self.heads, self.head_dim

        # Linear transform and reshape: [N, out_dim] -> [N, H, D]
        h = self.lin(x).view(N, H, D)

        # Edge feature projection
        if self.lin_edge is not None and edge_attr is not None and edge_attr.numel() > 0:
            edge_h = self.lin_edge(edge_attr).view(E, H, D)  # [E, H, D]
        else:
            edge_h = None

        # Compute attention scores
        src_idx = edge_index[0]  # [E]
        dst_idx = edge_index[1]  # [E]

        # Concatenate source and target features: [E, H, 2*D]
        h_src = h[src_idx]  # [E, H, D]
        h_dst = h[dst_idx]  # [E, H, D]
        h_pair = torch.cat([h_src, h_dst], dim=-1)  # [E, H, 2*D]

        # Attention: dot product with att vector
        # att: [1, H, 2*D], h_pair: [E, H, 2*D] -> edge_score: [E, H]
        edge_score = (h_pair * self.att).sum(dim=-1)

        # Add edge features if available
        if edge_h is not None:
            edge_score = edge_score + edge_h.sum(dim=-1)

        # LeakyReLU
        edge_score = F.leaky_relu(edge_score, negative_slope=self.negative_slope)

        # Softmax over neighbors (per destination node)
        # For numerical stability, subtract max per destination node
        # We use scatter_reduce for segment softmax
        edge_score_max = torch.full((N, H), float('-inf'), device=x.device, dtype=x.dtype)
        edge_score_max.scatter_reduce_(0, dst_idx.unsqueeze(-1).expand(E, H), edge_score, reduce='amax')
        edge_score = edge_score - edge_score_max[dst_idx]

        edge_exp = torch.exp(edge_score)  # [E, H]

        # Sum of exps per destination node
        exp_sum = torch.zeros(N, H, device=x.device, dtype=x.dtype)
        exp_sum.scatter_add_(0, dst_idx.unsqueeze(-1).expand(E, H), edge_exp)

        # Softmax: edge_exp / exp_sum[dst_idx]
        alpha = edge_exp / (exp_sum[dst_idx] + 1e-16)  # [E, H]

        if self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Aggregate: weighted sum of neighbor features
        messages = h[src_idx] * alpha.unsqueeze(-1)  # [E, H, D]

        # Scatter-add to destination nodes
        out = torch.zeros(N, H, D, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst_idx.view(E, 1, 1).expand(E, H, D), messages)

        # Concatenate heads and add bias
        out = out.reshape(N, H * D)  # [N, out_dim]
        out = out + self.bias

        return out


# ---------------------------------------------------------------------------
# GNN Reaction Encoder (pure PyTorch)
# ---------------------------------------------------------------------------

class GNNReactionEncoder(nn.Module):
    """GAT-based reaction encoder for candidate ranking.

    Different inductive bias from Chemformer (graph locality vs sequence):
    - Local message passing over molecular graphs
    - Permutation-invariant pooling
    - Atom/bond features from RDKit

    Input: molecular graphs (reactants + products)
    Output: reaction-level embedding for ranking/classification
    """

    def __init__(
        self,
        node_dim: int = _ATOM_FEATURE_DIM,
        edge_dim: int = _BOND_FEATURE_DIM,
        hidden_dim: int = 128,
        out_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        # Initial embeddings
        self.atom_embedding = nn.Linear(node_dim, hidden_dim)
        self.bond_embedding = nn.Linear(edge_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_out = hidden_dim
            self.gat_layers.append(
                GATLayer(hidden_dim, layer_out, heads=heads, dropout=dropout,
                         edge_dim=hidden_dim)
            )

        # Output projection
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode_molecules(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of molecular graphs.

        Args:
            x: [total_nodes, node_dim]
            edge_index: [2, total_edges]
            edge_attr: [total_edges, edge_dim]
            batch: [total_nodes] graph index for each node

        Returns:
            [num_graphs, out_dim] graph-level embeddings
        """
        # Move to model device
        device = self.atom_embedding.weight.device
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        batch = batch.to(device)

        # Embed atoms and bonds
        h = self.atom_embedding(x)  # [N, hidden_dim]
        edge_h = self.bond_embedding(edge_attr)  # [E, hidden_dim]

        # GAT layers
        for gat in self.gat_layers:
            h = h + gat(h, edge_index, edge_h, batch)  # Residual connection
            h = F.relu(h)

        # Global mean pooling
        num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
        pooled = torch.zeros(num_graphs, h.size(1), device=h.device, dtype=h.dtype)
        pooled.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
        counts = torch.zeros(num_graphs, 1, device=h.device, dtype=h.dtype)
        counts.scatter_add_(0, batch.unsqueeze(-1), torch.ones_like(batch.unsqueeze(-1), dtype=h.dtype))
        pooled = pooled / counts.clamp(min=1)

        return self.mlp(pooled)  # [num_graphs, out_dim]

    def forward(self, graphs: List[Dict]) -> torch.Tensor:
        """Forward pass on a list of molecular graphs.

        Args:
            graphs: List of graph dicts from mol_to_graph()

        Returns:
            [len(graphs), out_dim] embeddings
        """
        batched = collate_graphs(graphs)
        return self.encode_molecules(
            batched["x"], batched["edge_index"],
            batched["edge_attr"], batched["batch"],
        )


# ---------------------------------------------------------------------------
# Ranking head for GNN
# ---------------------------------------------------------------------------

class GNNRankingHead(nn.Module):
    """Ranking head on top of GNN reaction encoder.

    Scores each candidate reaction. For grouped ranking, we compute
    scores for all candidates in a group and rank by score.
    """

    def __init__(self, encoder_dim: int = 256, hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, reaction_embedding: torch.Tensor) -> torch.Tensor:
        """Score reaction embeddings.

        Args:
            reaction_embedding: [batch, encoder_dim]

        Returns:
            [batch] scores (higher = more likely positive)
        """
        return self.mlp(reaction_embedding).squeeze(-1)


# ---------------------------------------------------------------------------
# Complete GNN Reaction Scorer (encoder + ranking head)
# ---------------------------------------------------------------------------

class GNNReactionScorer(nn.Module):
    """Complete GNN-based reaction scorer for P4-G3.

    Combines GNNReactionEncoder + GNNRankingHead.
    """

    def __init__(
        self,
        node_dim: int = _ATOM_FEATURE_DIM,
        edge_dim: int = _BOND_FEATURE_DIM,
        hidden_dim: int = 128,
        encoder_out_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GNNReactionEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            out_dim=encoder_out_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
        self.head = GNNRankingHead(
            encoder_dim=encoder_out_dim,
            hidden_dim=128,
            dropout=dropout,
        )

    def forward(self, graphs: List[Dict]) -> torch.Tensor:
        """Score a list of molecular graphs.

        Args:
            graphs: List of graph dicts from mol_to_graph()

        Returns:
            [len(graphs)] scores
        """
        embeddings = self.encoder(graphs)  # [B, encoder_dim]
        scores = self.head(embeddings)  # [B]
        return scores

    def encode(self, graphs: List[Dict]) -> torch.Tensor:
        """Get reaction embeddings without scoring."""
        return self.encoder(graphs)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def build_gnn_scorer(
    hidden_dim: int = 128,
    encoder_out_dim: int = 256,
    num_layers: int = 3,
    heads: int = 4,
    dropout: float = 0.1,
    device: str = "cpu",
) -> GNNReactionScorer:
    """Build a GNN reaction scorer and move to device."""
    model = GNNReactionScorer(
        hidden_dim=hidden_dim,
        encoder_out_dim=encoder_out_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
    )
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
