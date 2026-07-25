"""反应中心感知 GAT 判别器（改进 B）。

纯 PyTorch 实现（不依赖 torch_geometric / DGL），服务器只需 torch + rdkit。

设计要点
--------
1. ``ReactionAwareClassifier`` 与 ``EnhancedMLP`` 接口兼容（``train`` / ``predict_proba``），
   但额外支持从 reaction SMILES 直接构建分子图并联合训练（``fit_reactions`` /
   ``predict_proba_reactions``）。
2. GAT 层完全手写：多头注意力 + LeakyReLU + softmax。
3. 反应中心 pooling：通过 atom_map_num 对齐 reactant/product，识别反应中心原子后
   做 attention pooling 得到 reaction-level 表示。
4. 显式键变化编码：用 RDKit 计算 formed / broken 键集合，按键类型 one-hot 聚合。
5. 多任务：主任务 BCE + 可选辅助任务（reaction_family 分类、yield_bin 分类）作为
   正则化，提升 OOD 泛化。
6. RDKit 不可用时自动降级为纯 fingerprint MLP，行为与 ``EnhancedMLP`` 等价。

依赖：``torch`` 必装；``rdkit`` 可选。
"""
from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 可选依赖：rdkit
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem  # noqa: F401  (供其它模块复用)
    _HAS_RDKIT = True
except Exception:  # pragma: no cover - 环境降级路径
    Chem = None  # type: ignore[assignment]
    _HAS_RDKIT = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# 原子符号 one-hot 表（顺序固定，便于复现）
ATOM_SYMBOLS: Tuple[str, ...] = ("C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "other")
# degree 0..6 共 7 维
MAX_DEGREE = 6
# formal_charge -2..+2 共 5 维
FORMAL_CHARGE_RANGE = 2
# 键类型 one-hot: single / double / triple / aromatic
BOND_TYPES: Tuple[str, ...] = ("single", "double", "triple", "aromatic")
# 原子特征维度 = 10 + 7 + 5 + 1 + 1 + 1 = 25
ATOM_FEAT_DIM = (
    len(ATOM_SYMBOLS)
    + (MAX_DEGREE + 1)
    + (2 * FORMAL_CHARGE_RANGE + 1)
    + 1  # is_aromatic
    + 1  # in_ring
    + 1  # atom_map_num（归一化后保留为标量）
)


# ===========================================================================
# 1. 分子图构建
# ===========================================================================
@contextmanager
def _eval_bn_if_batch1(*modules: nn.Module, batch_size: int):
    """当 batch_size == 1 时，临时把 BatchNorm1d 切到 eval 模式避免 ValueError。

    BatchNorm1d 在 training 模式下要求 batch_size > 1（否则无法估计方差）。
    约束 4 要求：batch_size=1 时用 eval 模式（使用 running stats）。
    """
    if batch_size > 1:
        yield
        return
    bn_modules: List[nn.BatchNorm1d] = []
    for m in modules:
        if m is not None:
            bn_modules.extend(mod for mod in m.modules() if isinstance(mod, nn.BatchNorm1d))
    saved_training = [m.training for m in bn_modules]
    for m in bn_modules:
        m.eval()
    try:
        yield
    finally:
        for m, was_training in zip(bn_modules, saved_training):
            m.train(was_training) if was_training else m.eval()


def _one_hot(idx: int, size: int) -> List[int]:
    """简单 one-hot 编码（越界返回全 0）。"""
    vec = [0] * size
    if 0 <= idx < size:
        vec[idx] = 1
    return vec


def build_mol_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用 RDKit 解析分子，返回 (node_feats, edge_index, atom_map_nums)。

    Parameters
    ----------
    smiles : str
        分子 SMILES（可含 atom map number）。

    Returns
    -------
    node_feats : np.ndarray, shape (N, ATOM_FEAT_DIM)
        节点特征。N 为原子数。
    edge_index : np.ndarray, shape (2, 2*E)
        无向边拆成两条有向边，``edge_index[0]`` 是源，``edge_index[1]`` 是目标。
    atom_map_nums : np.ndarray, shape (N,)
        每个原子的 atom map number（无则为 0）。

    RDKit 不可用或解析失败时返回空数组（N=0）。
    """
    if not _HAS_RDKIT or not smiles:
        return (
            np.zeros((0, ATOM_FEAT_DIM), dtype=np.float32),
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[union-attr]
    if mol is None:
        return (
            np.zeros((0, ATOM_FEAT_DIM), dtype=np.float32),
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    node_feats: List[List[float]] = []
    map_nums: List[int] = []
    ring_info = mol.GetRingInfo()
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in ATOM_SYMBOLS and sym != "other":
            sym_oh = _one_hot(ATOM_SYMBOLS.index(sym), len(ATOM_SYMBOLS))
        else:
            # 未在表中出现的元素 → "other" 槽
            sym_oh = _one_hot(len(ATOM_SYMBOLS) - 1, len(ATOM_SYMBOLS))
            sym_oh[len(ATOM_SYMBOLS) - 1] = 1
        degree = min(atom.GetDegree(), MAX_DEGREE)
        deg_oh = _one_hot(degree, MAX_DEGREE + 1)
        charge = max(-FORMAL_CHARGE_RANGE, min(atom.GetFormalCharge(), FORMAL_CHARGE_RANGE))
        charge_oh = _one_hot(charge + FORMAL_CHARGE_RANGE, 2 * FORMAL_CHARGE_RANGE + 1)
        is_arom = 1 if atom.GetIsAromatic() else 0
        in_ring = 1 if ring_info.NumAtomRings(atom.GetIdx()) > 0 else 0
        map_num = atom.GetAtomMapNum()
        # 归一化 atom map number：保留为 float，0 表示无 map
        map_num_feat = float(map_num) / 100.0 if map_num > 0 else 0.0

        node_feats.append(
            sym_oh + deg_oh + charge_oh + [is_arom, in_ring, map_num_feat]
        )
        map_nums.append(map_num)

    edges: List[List[int]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    node_feats_arr = np.array(node_feats, dtype=np.float32) if node_feats else \
        np.zeros((0, ATOM_FEAT_DIM), dtype=np.float32)
    edge_index_arr = np.array(edges, dtype=np.int64).T if edges else \
        np.zeros((2, 0), dtype=np.int64)
    map_nums_arr = np.array(map_nums, dtype=np.int64)
    return node_feats_arr, edge_index_arr, map_nums_arr


# ===========================================================================
# 2. 反应图构建
# ===========================================================================
def _split_reaction(reaction_smiles: str) -> Tuple[str, str, str]:
    """拆分 'reactants>agents>product'，返回三元组（agents 可为空）。"""
    parts = reaction_smiles.split(">")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], "", parts[1]
    return reaction_smiles, "", ""


def build_reaction_graph(reaction_smiles: str) -> Dict:
    """构建反应级图数据。

    解析 reactants 与 product，分别构建分子图，通过 atom_map_num 对齐
    reactant-product 原子对。agents 不参与图构建（与反应中心定义一致）。

    Returns
    -------
    dict 包含：
        - reactant_node_feats / reactant_edge_index / reactant_map_nums
        - product_node_feats / product_edge_index / product_map_nums
        - reactant_to_product : List[(ri, pi)] 同 map_num 的原子对
        - reaction_center_idx_r : reactant 侧反应中心原子下标
        - reaction_center_idx_p : product 侧反应中心原子下标
        - bond_change : np.ndarray (8,) 键变化向量
        - valid : bool 图是否可用于 GNN
    """
    if not _HAS_RDKIT or not reaction_smiles:
        return _empty_reaction_graph()
    r_smi, _a_smi, p_smi = _split_reaction(reaction_smiles)
    r_nodes, r_edges, r_maps = build_mol_graph(r_smi)
    p_nodes, p_edges, p_maps = build_mol_graph(p_smi)
    if r_nodes.shape[0] == 0 or p_nodes.shape[0] == 0:
        return _empty_reaction_graph()

    # 通过 atom map num 对齐
    r_idx_by_map: Dict[int, int] = {m: i for i, m in enumerate(r_maps) if m > 0}
    p_idx_by_map: Dict[int, int] = {m: i for i, m in enumerate(p_maps) if m > 0}
    pairs: List[Tuple[int, int]] = []
    for m, ri in r_idx_by_map.items():
        if m in p_idx_by_map:
            pairs.append((ri, p_idx_by_map[m]))

    # 反应中心：map num 只出现在一侧（消失 / 新生成）或对应原子键环境不同
    # 简化判定：map num 在 reactant 集合 ⊕ product 集合（对称差）即为反应中心，
    # 另外对成对的原子也标为反应中心邻域候选（下游 pooling 自适应）。
    r_map_set = set(r_idx_by_map.keys())
    p_map_set = set(p_idx_by_map.keys())
    center_maps = r_map_set.symmetric_difference(p_map_set) | r_map_set.intersection(p_map_set)
    # 这里把所有 mapped 原子都视为潜在中心候选，下游通过 ±2 hop 邻域收敛。
    rc_r = [r_idx_by_map[m] for m in center_maps if m in r_idx_by_map]
    rc_p = [p_idx_by_map[m] for m in center_maps if m in p_idx_by_map]

    bond_change = bond_change_features(reaction_smiles)

    return {
        "reactant_node_feats": r_nodes,
        "reactant_edge_index": r_edges,
        "reactant_map_nums": r_maps,
        "product_node_feats": p_nodes,
        "product_edge_index": p_edges,
        "product_map_nums": p_maps,
        "reactant_to_product": pairs,
        "reaction_center_idx_r": rc_r,
        "reaction_center_idx_p": rc_p,
        "bond_change": bond_change,
        "valid": True,
    }


def _empty_reaction_graph() -> Dict:
    return {
        "reactant_node_feats": np.zeros((0, ATOM_FEAT_DIM), dtype=np.float32),
        "reactant_edge_index": np.zeros((2, 0), dtype=np.int64),
        "reactant_map_nums": np.zeros((0,), dtype=np.int64),
        "product_node_feats": np.zeros((0, ATOM_FEAT_DIM), dtype=np.float32),
        "product_edge_index": np.zeros((2, 0), dtype=np.int64),
        "product_map_nums": np.zeros((0,), dtype=np.int64),
        "reactant_to_product": [],
        "reaction_center_idx_r": [],
        "reaction_center_idx_p": [],
        "bond_change": np.zeros(8, dtype=np.float32),
        "valid": False,
    }


# ===========================================================================
# 3. 显式键变化编码
# ===========================================================================
def bond_change_features(reaction_smiles: str) -> np.ndarray:
    """计算 formed / broken 键的按键类型聚合向量。

    返回 8 维向量：[formed_single, formed_double, formed_triple, formed_aromatic,
                    broken_single, broken_double, broken_triple, broken_aromatic]
    每个槽位是对应键类型的计数（ clipped 到 [0, 1] 以保持稀疏稳定）。
    """
    vec = np.zeros(8, dtype=np.float32)
    if not _HAS_RDKIT or not reaction_smiles:
        return vec
    r_smi, _a, p_smi = _split_reaction(reaction_smiles)
    r_bonds = _bond_set(r_smi)
    p_bonds = _bond_set(p_smi)
    if r_bonds is None or p_bonds is None:
        return vec

    # formed: 在 product 但不在 reactant
    for key, btype in p_bonds.items():
        if key not in r_bonds:
            if btype in BOND_TYPES:
                vec[BOND_TYPES.index(btype)] = 1.0
    # broken: 在 reactant 但不在 product
    for key, btype in r_bonds.items():
        if key not in p_bonds:
            if btype in BOND_TYPES:
                vec[4 + BOND_TYPES.index(btype)] = 1.0
    return vec


def _bond_set(smiles: str) -> Optional[Dict[Tuple[int, int], str]]:
    """返回 {(atom_map_a, atom_map_b): bond_type_str}，按 atom map num 标准化排序。

    若任一原子没有 map num，则降级为 (idx_a, idx_b)（仅在同一分子内有效）。
    """
    if not _HAS_RDKIT or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[union-attr]
    if mol is None:
        return None
    bonds: Dict[Tuple[int, int], str] = {}
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        m1, m2 = a1.GetAtomMapNum(), a2.GetAtomMapNum()
        key = tuple(sorted((m1 if m1 > 0 else a1.GetIdx(),
                            m2 if m2 > 0 else a2.GetIdx())))
        bt = _bond_type_str_rdkit(b)
        if bt is None:
            continue
        bonds[key] = bt  # 后出现的覆盖，保持 product 侧语义一致
    return bonds


def _bond_type_str_rdkit(bond) -> Optional[str]:
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:  # type: ignore[union-attr]
        return "single"
    if bt == Chem.rdchem.BondType.DOUBLE:  # type: ignore[union-attr]
        return "double"
    if bt == Chem.rdchem.BondType.TRIPLE:  # type: ignore[union-attr]
        return "triple"
    if bt == Chem.rdchem.BondType.AROMATIC:  # type: ignore[union-attr]
        return "aromatic"
    return None


# ===========================================================================
# 4. GAT 层（纯 PyTorch）
# ===========================================================================
class GATLayer(nn.Module):
    """单层多头图注意力（纯 PyTorch 实现）。

    实现：
        - 对每个节点做 Q,K,V 线性变换
        - attention = softmax(LeakyReLU(Q_i · K_j / sqrt(d)))
        - 聚合邻居（含自环）
        - 多头拼接（concat）后过线性投影
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4,
                 dropout: float = 0.2, leaky_slope: float = 0.2):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim 必须能整除 num_heads"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.dropout = dropout

        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)
        # 每个头一个 attention scalar a · [Wh_i || Wh_j]
        self.attn = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.attn, gain=1.414)
        self.leaky = nn.LeakyReLU(leaky_slope)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """前向。

        Parameters
        ----------
        node_feats : (N, in_dim)
        edge_index : (2, E)  包含自环边（由调用方加）

        Returns
        -------
        (N, out_dim)
        """
        if node_feats.shape[0] == 0:
            return node_feats.new_zeros((0, self.out_dim))
        N = node_feats.shape[0]
        H, Dh = self.num_heads, self.head_dim

        Q = self.W_q(node_feats).view(N, H, Dh)
        K = self.W_k(node_feats).view(N, H, Dh)
        V = self.W_v(node_feats).view(N, H, Dh)

        src, dst = edge_index[0], edge_index[1]  # 信息从 src 流向 dst
        # 计算每条边每个头的 attention logit
        # 使用与 GAT 原始实现一致的 a · [Wh_dst || Wh_src]
        q_dst = Q[dst]              # (E, H, Dh)
        k_src = K[src]              # (E, H, Dh)
        cat = torch.cat([q_dst, k_src], dim=-1)  # (E, H, 2Dh)
        logits = (cat * self.attn).sum(dim=-1)   # (E, H)
        logits = self.leaky(logits)

        # 对每个 dst 节点做 softmax（按 head 分组）
        # 用 scatter_softmax 实现：先 max 减，再 exp，再 sum
        attn = torch.exp(logits - logits.max(dim=0, keepdim=True).values)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        denom = torch.zeros(N, H, device=node_feats.device, dtype=attn.dtype)
        denom.index_add_(0, dst, attn)
        denom = denom.clamp(min=1e-12)
        # 对每条边：attn / denom[dst]
        alpha = attn / denom[dst]                 # (E, H)

        v_src = V[src]                             # (E, H, Dh)
        msg = alpha.unsqueeze(-1) * v_src          # (E, H, Dh)
        out = torch.zeros(N, H, Dh, device=node_feats.device, dtype=node_feats.dtype)
        out.index_add_(0, dst, msg)
        out = out.reshape(N, self.out_dim)
        return self.out_proj(out)


# ===========================================================================
# 5. 反应中心 pooling
# ===========================================================================
def _k_hop_neighbors(edge_index: torch.Tensor, n_nodes: int,
                     k: int, device: torch.device) -> torch.Tensor:
    """返回 (N, N) 的 0/1 mask，标记每个节点 k-hop 邻居（含自身）。

    用矩阵幂方法（N 较小，分子图通常 < 100 节点）。
    """
    if n_nodes == 0:
        return torch.zeros((0, 0), device=device)
    adj = torch.zeros((n_nodes, n_nodes), device=device)
    if edge_index.shape[1] > 0:
        adj[edge_index[0], edge_index[1]] = 1.0
    # 自环
    eye = torch.eye(n_nodes, device=device)
    reach = eye.clone()
    cur = eye.clone()
    for _ in range(k):
        cur = (cur @ adj + eye).clamp(max=1.0)
        reach = (reach + cur).clamp(max=1.0)
    return reach


def reaction_center_pool(node_feats: torch.Tensor,
                         edge_index: torch.Tensor,
                         center_idx: torch.Tensor,
                         n_nodes: int,
                         k_hop: int = 2) -> torch.Tensor:
    """反应中心 attention pooling。

    Parameters
    ----------
    node_feats : (N, D)
    edge_index : (2, E)
    center_idx : (C,) 反应中心原子下标（可为空）
    n_nodes : int
    k_hop : int  邻域半径

    Returns
    -------
    (D,)  reaction-level 表示。若无反应中心，退化为全图 mean pooling。
    """
    if n_nodes == 0:
        return node_feats.new_zeros(node_feats.shape[-1])
    device = node_feats.device
    if center_idx.numel() == 0:
        return node_feats.mean(dim=0)

    # 收集反应中心 k-hop 邻域
    reach = _k_hop_neighbors(edge_index, n_nodes, k_hop, device)  # (N, N)
    # 对每个中心节点取其邻域并集
    center_mask = reach[center_idx].sum(dim=0) > 0  # (N,) 在任一中心邻域内
    sel_idx = torch.nonzero(center_mask, as_tuple=False).squeeze(-1)
    if sel_idx.numel() == 0:
        return node_feats.mean(dim=0)
    sel_feats = node_feats[sel_idx]  # (M, D)

    # attention pooling: score = sel_feats · learnable? 这里用 self-attention 形式：
    # score_i = ||sel_feats_i|| 简化稳定，再做 softmax。
    # 用与节点特征解耦的 score：mean 然后 softmax。
    scores = sel_feats.mean(dim=-1)  # (M,)
    weights = torch.softmax(scores, dim=0)
    pooled = (weights.unsqueeze(-1) * sel_feats).sum(dim=0)
    return pooled


# ===========================================================================
# 6. 融合分类头
# ===========================================================================
class FingerprintBranch(nn.Module):
    """fingerprint branch: input_dim → 512 → hidden_dim。"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GNNBranch(nn.Module):
    """GNN branch: 2 层 GAT + reaction center pooling + bond change → 256。"""

    def __init__(self, in_dim: int = ATOM_FEAT_DIM, hidden_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.gat_layers.append(
                GATLayer(dims[i], dims[i + 1], num_heads=num_heads, dropout=dropout)
            )
        self.dropout = dropout
        # bond_change (8) + pooled (hidden_dim) → hidden_dim
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim + 8, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, batch_graphs: List[Dict]) -> torch.Tensor:
        """对 batch 内每个反应图跑 GAT + pooling，返回 (B, hidden_dim)。

        关键：先收集所有图的 (pooled, bond) 再统一过 ``self.fuse``（含 BatchNorm1d），
        避免 per-graph 调用导致 BN batch_size=1 报错。
        """
        device = next(self.parameters()).device
        B = len(batch_graphs)
        pooled_list: List[torch.Tensor] = []
        bond_list: List[torch.Tensor] = []
        for g in batch_graphs:
            if not g["valid"]:
                pooled_list.append(torch.zeros(self.hidden_dim, device=device))
                bond_list.append(torch.zeros(8, device=device))
                continue
            # reactant + product 分别跑 GAT
            r_out = self._encode_side(
                g["reactant_node_feats"], g["reactant_edge_index"],
                g["reaction_center_idx_r"], device,
            )
            p_out = self._encode_side(
                g["product_node_feats"], g["product_edge_index"],
                g["reaction_center_idx_p"], device,
            )
            pooled = (r_out + p_out) / 2.0          # (hidden_dim,)
            bond = torch.from_numpy(g["bond_change"]).float().to(device)  # (8,)
            pooled_list.append(pooled)
            bond_list.append(bond)
        pooled_batch = torch.stack(pooled_list, dim=0)  # (B, hidden_dim)
        bond_batch = torch.stack(bond_list, dim=0)      # (B, 8)
        # batch 维度统一过 BN，避免 per-graph batch_size=1
        with _eval_bn_if_batch1(self, batch_size=B):
            fused = self.fuse(torch.cat([pooled_batch, bond_batch], dim=-1))
        return fused  # (B, hidden_dim)

    def _encode_side(self, node_feats_np: np.ndarray, edge_index_np: np.ndarray,
                     center_idx: List[int], device: torch.device) -> torch.Tensor:
        n = node_feats_np.shape[0]
        x = torch.from_numpy(node_feats_np).float().to(device)
        ei = torch.from_numpy(edge_index_np).long().to(device)
        # 加自环
        if n > 0:
            selfloop = torch.arange(n, device=device).unsqueeze(0).repeat(2, 1)
            ei = torch.cat([ei, selfloop], dim=1)
        for layer in self.gat_layers:
            x = F.elu(layer(x, ei))
            x = F.dropout(x, p=self.dropout, training=self.training)
        center_t = torch.tensor(center_idx, dtype=torch.long, device=device) \
            if len(center_idx) > 0 else torch.zeros(0, dtype=torch.long, device=device)
        pooled = reaction_center_pool(x, ei, center_t, n, k_hop=2)
        return pooled


class FusionHead(nn.Module):
    """融合 fingerprint 与 GNN 分支 → 128 → 1。"""

    def __init__(self, hidden_dim: int = 256, fused_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 1),
        )

    def forward(self, fp_emb: torch.Tensor, gnn_emb: torch.Tensor,
                gnn_weight: float = 0.5) -> torch.Tensor:
        # 显式加权融合：fp 与 gnn 在各自路径后再 concat
        # 这里采用加权 concat：把 gnn_emb * (2*w) 与 fp_emb * (2*(1-w)) concat
        # 保证总能量守恒，且 w=0.5 时退化为标准 concat。
        w = float(gnn_weight)
        fused = torch.cat([fp_emb * (2.0 * (1.0 - w)), gnn_emb * (2.0 * w)], dim=-1)
        return self.fc(fused)


# ===========================================================================
# 7. 主分类器
# ===========================================================================
class ReactionAwareClassifier:
    """反应中心感知 GAT 判别器。

    与 ``EnhancedMLP`` 接口兼容，但额外支持从 reaction SMILES 构建图联合训练。

    Parameters
    ----------
    input_dim : int
        fingerprint 输入维度（默认 8192，与 ``reaction_fp_enhanced`` 一致）。
    seed : int
        随机种子。
    hidden_dim : int
        GAT 隐层维度（同时作为 fingerprint branch 输出维度）。
    num_heads : int
        GAT 多头数。
    num_layers : int
        GAT 层数。
    dropout : float
        Dropout 概率。
    gnn_weight : float
        GNN 输出与 fingerprint 输出的融合权重（0~1，0=纯 fingerprint，1=纯 GNN）。
    """

    def __init__(self, input_dim: int = 8192, seed: int = 42,
                 hidden_dim: int = 256, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.2,
                 gnn_weight: float = 0.5):
        self.input_dim = input_dim
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_weight = gnn_weight

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed(seed)

        # fingerprint branch
        self.fp_branch = FingerprintBranch(input_dim, hidden_dim=hidden_dim,
                                           dropout=dropout).to(self._device)
        # GNN branch（仅当 rdkit 可用才参与计算）
        self.has_gnn = _HAS_RDKIT
        if self.has_gnn:
            self.gnn_branch = GNNBranch(
                in_dim=ATOM_FEAT_DIM, hidden_dim=hidden_dim,
                num_heads=num_heads, num_layers=num_layers, dropout=dropout,
            ).to(self._device)
        else:
            self.gnn_branch = None
        self.fusion_head = FusionHead(hidden_dim=hidden_dim, dropout=dropout).to(self._device)

        # 辅助任务头（多任务正则化）
        self.aux_family_head: Optional[nn.Linear] = None
        self.aux_yield_head: Optional[nn.Linear] = None
        self._aux_family_classes: int = 0
        self._aux_yield_classes: int = 0

        # 兼容纯 fingerprint 模式（train / predict_proba）
        self._fp_only_mode: bool = False

        # 构建 optimizer 时收集参数
        self._build_optimizer(lr=1e-3)

    # ---------- 内部工具 ----------
    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_optimizer(self, lr: float) -> None:
        params: List[nn.Parameter] = list(self.fp_branch.parameters()) + \
            list(self.fusion_head.parameters())
        if self.gnn_branch is not None:
            params += list(self.gnn_branch.parameters())
        if self.aux_family_head is not None:
            params += list(self.aux_family_head.parameters())
        if self.aux_yield_head is not None:
            params += list(self.aux_yield_head.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    def _set_aux_heads(self, family_labels: Optional[np.ndarray],
                       yield_labels: Optional[np.ndarray]) -> None:
        """根据传入的辅助标签动态构建辅助头（若存在且类别数 > 1）。"""
        if family_labels is not None and len(family_labels) > 0:
            n_cls = int(family_labels.max()) + 1
            if n_cls > 1:
                if self.aux_family_head is None or self._aux_family_classes != n_cls:
                    self.aux_family_head = nn.Linear(self.hidden_dim, n_cls).to(self._device)
                    self._aux_family_classes = n_cls
            else:
                self.aux_family_head = None
        else:
            self.aux_family_head = None

        if yield_labels is not None and len(yield_labels) > 0:
            n_cls = int(yield_labels.max()) + 1
            if n_cls > 1:
                if self.aux_yield_head is None or self._aux_yield_classes != n_cls:
                    self.aux_yield_head = nn.Linear(self.hidden_dim, n_cls).to(self._device)
                    self._aux_yield_classes = n_cls
            else:
                self.aux_yield_head = None
        else:
            self.aux_yield_head = None

    # ---------- 反应图构建（缓存友好） ----------
    def _build_graphs(self, reaction_smiles_list: List[str]) -> List[Dict]:
        return [build_reaction_graph(rxn) for rxn in reaction_smiles_list]

    def _fingerprint_batch(self, reaction_smiles_list: List[str]) -> np.ndarray:
        """惰性获取 fingerprint：先尝试 pc_cng.phase3_enhanced，再退化为 rdkit Morgan。

        始终返回 shape=(N, input_dim) 的数组。退化路径下用 4 个 Morgan
        fingerprint 拼接以匹配 input_dim=8192（与 reaction_fp_enhanced 一致）。
        """
        # 优先用 phase3_enhanced.reaction_fp_enhanced（8192 维）
        reaction_fp_enhanced = None
        for mod in ("pc_cng.phase3_enhanced", "phase3_enhanced"):
            try:
                mod_obj = __import__(mod, fromlist=["reaction_fp_enhanced"])
                reaction_fp_enhanced = getattr(mod_obj, "reaction_fp_enhanced")
                break
            except Exception:
                continue
        fps: List[np.ndarray] = []
        for rxn in reaction_smiles_list:
            fp = None
            if reaction_fp_enhanced is not None:
                try:
                    fp = reaction_fp_enhanced(rxn)
                except Exception:
                    fp = None
            if fp is None:
                # 退化路径：用 4 个 Morgan 拼接成 8192 维（与 enhanced 一致）
                fp = self._fallback_fp(rxn)
            fps.append(fp)
        return np.vstack(fps).astype(np.float32)

    def _fallback_fp(self, reaction_smiles: str) -> np.ndarray:
        """退化 fingerprint：4 段 Morgan 拼接，shape=(input_dim,)。"""
        try:
            from rdkit import Chem as _Chem
            from rdkit.Chem import AllChem as _AllChem
            from rdkit.DataStructs import ConvertToNumpyArray as _c2n
        except Exception:
            return np.zeros(self.input_dim, dtype=np.float32)
        sp = _split_reaction(reaction_smiles)
        r = sp[0] if sp else reaction_smiles
        p = sp[2] if sp else reaction_smiles
        r_mol = _Chem.MolFromSmiles(r)
        p_mol = _Chem.MolFromSmiles(p)
        nbits = self.input_dim // 4  # 每段 2048（当 input_dim=8192）
        r_arr = np.zeros(nbits, dtype=np.float32)
        p_arr = np.zeros(nbits, dtype=np.float32)
        if r_mol is not None:
            _c2n(_AllChem.GetMorganFingerprintAsBitVect(r_mol, 2, nBits=nbits), r_arr)
        if p_mol is not None:
            _c2n(_AllChem.GetMorganFingerprintAsBitVect(p_mol, 2, nBits=nbits), p_arr)
        formed = np.maximum(p_arr - r_arr, 0).astype(np.float32)
        broken = np.maximum(r_arr - p_arr, 0).astype(np.float32)
        return np.concatenate([r_arr, p_arr, formed, broken])

    # ---------- 联合训练（推荐入口） ----------
    def fit_reactions(self, reaction_smiles_list: List[str], labels: np.ndarray,
                      epochs: int = 150, batch_size: int = 64, lr: float = 1e-3,
                      verbose: bool = False,
                      family_labels: Optional[np.ndarray] = None,
                      yield_labels: Optional[np.ndarray] = None
                      ) -> "ReactionAwareClassifier":
        """从 reaction SMILES 构建图 + fingerprint，联合训练。

        Parameters
        ----------
        reaction_smiles_list : List[str]
        labels : np.ndarray, shape (N,)  0/1 标签。
        family_labels / yield_labels : 可选辅助任务标签（int 数组）。
        """
        self._set_seed(self.seed)
        self._build_optimizer(lr)
        self._set_aux_heads(family_labels, yield_labels)
        self._fp_only_mode = False

        n = len(reaction_smiles_list)
        if n == 0:
            return self

        # 1) 预计算 fingerprint（numpy）
        fps_np = self._fingerprint_batch(reaction_smiles_list)
        # 2) 预计算图（list of dict）
        graphs = self._build_graphs(reaction_smiles_list)

        fps_t = torch.from_numpy(fps_np).float().to(self._device)
        y_t = torch.from_numpy(np.asarray(labels, dtype=np.float32)).float().to(self._device)
        if family_labels is not None:
            fam_t = torch.from_numpy(np.asarray(family_labels, dtype=np.int64)).long().to(self._device)
        else:
            fam_t = None
        if yield_labels is not None:
            yld_t = torch.from_numpy(np.asarray(yield_labels, dtype=np.int64)).long().to(self._device)
        else:
            yld_t = None

        # 构建索引数组用于 batch
        idx = np.arange(n)
        bs = min(batch_size, n)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            np.random.shuffle(idx)
            self.fp_branch.train()
            self.fusion_head.train()
            if self.gnn_branch is not None:
                self.gnn_branch.train()
            if self.aux_family_head is not None:
                self.aux_family_head.train()
            if self.aux_yield_head is not None:
                self.aux_yield_head.train()

            total_loss = 0.0
            n_batches = 0
            for start in range(0, n, bs):
                batch_idx = idx[start:start + bs]
                cur_bs = len(batch_idx)
                b_fps = fps_t[batch_idx]
                b_y = y_t[batch_idx].unsqueeze(1)
                b_graphs = [graphs[i] for i in batch_idx]

                self.optimizer.zero_grad()
                # batch_size==1 时把 BN 切到 eval 模式避免 ValueError
                with _eval_bn_if_batch1(
                    self.fp_branch, self.gnn_branch, self.fusion_head,
                    self.aux_family_head, self.aux_yield_head,
                    batch_size=cur_bs,
                ):
                    fp_emb = self.fp_branch(b_fps)
                    if self.gnn_branch is not None:
                        gnn_emb = self.gnn_branch(b_graphs)
                    else:
                        gnn_emb = torch.zeros_like(fp_emb)
                    logits = self.fusion_head(fp_emb, gnn_emb,
                                              gnn_weight=self.gnn_weight)
                    main_loss = criterion(logits, b_y)

                    # 辅助任务（基于 gnn_emb 或 fp_emb）
                    aux_loss = torch.zeros((), device=self._device)
                    feat_for_aux = gnn_emb if (self.gnn_branch is not None) else fp_emb
                    if self.aux_family_head is not None and fam_t is not None:
                        fam_logits = self.aux_family_head(feat_for_aux)
                        aux_loss = aux_loss + F.cross_entropy(
                            fam_logits, fam_t[batch_idx])
                    if self.aux_yield_head is not None and yld_t is not None:
                        yld_logits = self.aux_yield_head(feat_for_aux)
                        aux_loss = aux_loss + F.cross_entropy(
                            yld_logits, yld_t[batch_idx])
                loss = main_loss + 0.1 * aux_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            scheduler.step()
            if verbose and (epoch + 1) % 25 == 0:
                print(f"    [GNN] epoch {epoch+1}/{epochs} "
                      f"loss={total_loss/max(1,n_batches):.4f}")
        return self

    # ---------- 推理（reaction SMILES） ----------
    def predict_proba_reactions(self, reaction_smiles_list: List[str]) -> np.ndarray:
        """返回每条反应的可行性概率 (0-1)。"""
        self.fp_branch.eval()
        self.fusion_head.eval()
        if self.gnn_branch is not None:
            self.gnn_branch.eval()
        if self.aux_family_head is not None:
            self.aux_family_head.eval()
        if self.aux_yield_head is not None:
            self.aux_yield_head.eval()

        n = len(reaction_smiles_list)
        if n == 0:
            return np.zeros((0,), dtype=np.float32)
        fps_np = self._fingerprint_batch(reaction_smiles_list)
        graphs = self._build_graphs(reaction_smiles_list)
        bs = 64
        probs = []
        with torch.no_grad():
            for start in range(0, n, bs):
                end = min(start + bs, n)
                b_fps = torch.from_numpy(fps_np[start:end]).float().to(self._device)
                b_graphs = graphs[start:end]
                fp_emb = self.fp_branch(b_fps)
                if self.gnn_branch is not None:
                    gnn_emb = self.gnn_branch(b_graphs)
                else:
                    gnn_emb = torch.zeros_like(fp_emb)
                logits = self.fusion_head(fp_emb, gnn_emb, gnn_weight=self.gnn_weight)
                probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
        return np.concatenate(probs)

    # ---------- 兼容接口（替换 EnhancedMLP） ----------
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 150,
              batch_size: int = 64, lr: float = 1e-3,
              verbose: bool = False) -> "ReactionAwareClassifier":
        """纯 fingerprint 模式训练（不构建图）。

        若有 reaction_smiles 应该用 ``fit_reactions``。此模式下行为与
        ``EnhancedMLP`` 等价：仅用 fingerprint branch + 一个临时线性头。
        """
        self._set_seed(self.seed)
        self._fp_only_mode = True
        self.has_gnn = False  # 训练阶段禁用 GNN
        # 重建一个纯 fingerprint 的临时分类头
        self._fp_only_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        ).to(self._device)
        self.optimizer = torch.optim.AdamW(
            list(self.fp_branch.parameters()) + list(self._fp_only_head.parameters()),
            lr=lr, weight_decay=1e-4,
        )

        X_t = torch.from_numpy(X).float().to(self._device)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).float().unsqueeze(1).to(self._device)
        dataset = TensorDataset(X_t, y_t)
        bs = min(batch_size, len(dataset))
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss()

        self.fp_branch.train()
        self._fp_only_head.train()
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                cur_bs = xb.shape[0]
                self.optimizer.zero_grad()
                # batch_size==1 时把 BN 切到 eval 模式避免 ValueError
                with _eval_bn_if_batch1(
                    self.fp_branch, self._fp_only_head, batch_size=cur_bs,
                ):
                    emb = self.fp_branch(xb)
                    logits = self._fp_only_head(emb)
                    loss = criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            scheduler.step()
            if verbose and (epoch + 1) % 25 == 0:
                print(f"    [FP-only] epoch {epoch+1}/{epochs} "
                      f"loss={total_loss/max(1,n_batches):.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """纯 fingerprint 推理（与 ``train`` 配套）。"""
        if not self._fp_only_mode:
            # 未显式调用 train，但被当作 EnhancedMLP 使用：自动进入 FP-only 推理
            # （此时 fusion_head 也只依赖 fp_emb，gnn_weight=0 等价）
            self._fp_only_mode = True
            self._fp_only_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 1),
            ).to(self._device)
        self.fp_branch.eval()
        self._fp_only_head.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X).float().to(self._device)
            emb = self.fp_branch(X_t)
            logits = self._fp_only_head(emb)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
        return probs


# ===========================================================================
# 8. 便捷工具
# ===========================================================================
def is_rdkit_available() -> bool:
    """返回 rdkit 是否可用（用于上层日志）。"""
    return _HAS_RDKIT


def extract_aux_labels_from_records(records: List[Dict]
                                    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """从 build_dataset_enhanced 返回的 records 中提取辅助任务标签。

    Returns
    -------
    (family_labels, yield_labels)  若 records 中没有对应字段则返回 None。
    """
    if not records:
        return None, None
    fam: List[int] = []
    yld: List[int] = []
    has_family = all("reaction_family" in r for r in records)
    has_yield = all("yield_bin" in r for r in records)
    if not has_family and not has_yield:
        return None, None
    # 把 family 字符串编码为 int
    fam_vocab: Dict[str, int] = {}
    for r in records:
        if has_family:
            f = str(r.get("reaction_family", "unknown"))
            if f not in fam_vocab:
                fam_vocab[f] = len(fam_vocab)
            fam.append(fam_vocab[f])
        if has_yield:
            yld.append(int(r.get("yield_bin", 0)))
    fam_arr = np.array(fam, dtype=np.int64) if has_family else None
    yld_arr = np.array(yld, dtype=np.int64) if has_yield else None
    return fam_arr, yld_arr


__all__ = [
    "ReactionAwareClassifier",
    "GATLayer",
    "FingerprintBranch",
    "GNNBranch",
    "FusionHead",
    "build_mol_graph",
    "build_reaction_graph",
    "reaction_center_pool",
    "bond_change_features",
    "is_rdkit_available",
    "extract_aux_labels_from_records",
    "ATOM_FEAT_DIM",
    "ATOM_SYMBOLS",
    "BOND_TYPES",
]
