"""Enhanced Phase 3 components: reaction-level features + improved MLP.

Self-contained module (no circular imports) with:
  - make_negative_rxn: construct proper reaction SMILES for negatives
  - reaction_fp_enhanced: 4*n_bits fingerprint (concat + differential)
  - EnhancedMLP: 4-layer MLP with BatchNorm + Dropout + AdamW
  - build_dataset_enhanced: critical fix for negative fingerprinting

ROOT CAUSE FIX: reaction_fp("A.B>>C") returns [reactant_fp, product_fp]
(both halves non-zero), but reaction_fp("D") returns [product_fp, zeros]
(second half ALL ZEROS).  The classifier learns "second half zero = negative"
instead of chemistry.  make_negative_rxn fixes this by constructing
"reactants>agents>neg_product" for negatives.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# --- Self-contained constants and helpers (avoid circular import) ---
MORGAN_RADIUS = 2
MORGAN_BITS = 2048


def morgan_fingerprint(smiles: str, radius: int = MORGAN_RADIUS,
                       n_bits: int = MORGAN_BITS) -> Optional[np.ndarray]:
    """Morgan fingerprint as numpy array."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


# --- Enhanced functions ---

def make_negative_rxn(original_rxn: str, neg_product: str) -> str:
    """Construct a proper reaction SMILES for a negative candidate.

    Given the original reaction SMILES (e.g. "A.B>C>D") and a negative
    product SMILES, return a reaction SMILES that combines the original
    reactants/agents with the negative product (e.g. "A.B>C>neg_product").
    """
    if not original_rxn or not neg_product:
        return neg_product
    parts = original_rxn.split(">")
    if len(parts) == 3:
        return f"{parts[0]}>{parts[1]}>{neg_product}"
    elif len(parts) == 2:
        return f"{parts[0]}>>{neg_product}"
    return neg_product


def reaction_fp_enhanced(reaction_smiles: str, radius: int = MORGAN_RADIUS,
                         n_bits: int = MORGAN_BITS) -> Optional[np.ndarray]:
    """Enhanced reaction fingerprint: concat + differential (4*n_bits dim).

    Returns [reactant_fp, product_fp, formed_bits, broken_bits] where:
      - reactant_fp: Morgan FP of reactants (n_bits)
      - product_fp: Morgan FP of products (n_bits)
      - formed_bits: bits in product but not reactant (bonds formed)
      - broken_bits: bits in reactant but not product (bonds broken)
    """
    if not reaction_smiles:
        return None
    parts = reaction_smiles.split(">")
    if len(parts) == 3:
        r_fp = morgan_fingerprint(parts[0], radius, n_bits)
        p_fp = morgan_fingerprint(parts[2], radius, n_bits)
        if r_fp is None and p_fp is None:
            return None
        r_arr = r_fp if r_fp is not None else np.zeros(n_bits, dtype=np.float32)
        p_arr = p_fp if p_fp is not None else np.zeros(n_bits, dtype=np.float32)
        formed = np.maximum(p_arr - r_arr, 0).astype(np.float32)
        broken = np.maximum(r_arr - p_arr, 0).astype(np.float32)
        return np.concatenate([r_arr, p_arr, formed, broken])
    fp = morgan_fingerprint(reaction_smiles, radius, n_bits)
    if fp is None:
        return None
    zeros = np.zeros(n_bits, dtype=np.float32)
    return np.concatenate([fp, zeros, fp, zeros])


def reaction_fp_enhanced_dim() -> int:
    return 4 * MORGAN_BITS


class EnhancedMLP:
    """Enhanced 4-layer MLP with BatchNorm, Dropout, and AdamW."""
    def __init__(self, input_dim: int, seed: int = 42,
                 hidden_dims=(512, 256, 128), dropout=(0.3, 0.3, 0.2)):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        layers = []
        prev = input_dim
        for h, d in zip(hidden_dims, dropout):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(d))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)
        self.input_dim = input_dim
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self._device)

    def train(self, X, y, epochs=150, batch_size=64, lr=1e-3, verbose=False):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        X_t = torch.from_numpy(X).float().to(self._device)
        y_t = torch.from_numpy(y).float().unsqueeze(1).to(self._device)
        dataset = TensorDataset(X_t, y_t)
        bs = min(batch_size, len(dataset))
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss()
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            scheduler.step()
            if verbose and (epoch + 1) % 25 == 0:
                print(f"    epoch {epoch+1}/{epochs} loss={total_loss/max(1,n_batches):.4f}")
        return self

    def predict_proba(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X).float().to(self._device)
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
        return probs


def build_dataset_enhanced(rows, generator, fp_fn=None):
    """Build (X, y, records) with CRITICAL FIX: proper reaction SMILES for negatives.

    Each positive contributes one positive example (label=1) and one
    generated negative (label=0).  The negative is fingerprinted as a
    proper reaction SMILES (reactants>agents>neg_product), NOT as a
    standalone product SMILES (which gives [fp, zeros]).
    """
    if fp_fn is None:
        fp_fn = reaction_fp_enhanced
    fps_pos: List[np.ndarray] = []
    fps_neg: List[np.ndarray] = []
    records: List[Dict] = []
    n_invalid = 0
    n_neg_fail = 0
    for _, row in rows.iterrows():
        rxn = row.get("reaction_smiles")
        if not rxn or not isinstance(rxn, str):
            n_invalid += 1
            continue
        pos_fp = fp_fn(rxn)
        if pos_fp is None:
            n_invalid += 1
            continue
        neg_product = generator.generate(rxn)
        if neg_product is None:
            n_neg_fail += 1
            continue
        # CRITICAL FIX: construct proper reaction SMILES for negative
        neg_rxn = make_negative_rxn(rxn, neg_product)
        neg_fp = fp_fn(neg_rxn)
        if neg_fp is None:
            n_neg_fail += 1
            continue
        fps_pos.append(pos_fp)
        fps_neg.append(neg_fp)
        cluster_key = str(row.get("experimental_group",
                                  row.get("split_key", "default")))
        family = str(row.get("reaction_family", "unknown"))
        yield_bin = int(row.get("yield_bin", 0)) if "yield_bin" in row else 0
        records.append({
            "reaction_smiles": rxn,
            "negative_smiles": neg_product,
            "label": 1, "score": 0.0,
            "experimental_group": cluster_key,
            "reaction_family": family,
            "yield_bin": yield_bin,
            "method": generator.method,
            "is_positive": True,
        })
        records.append({
            "reaction_smiles": neg_rxn,
            "negative_smiles": neg_product,
            "label": 0, "score": 0.0,
            "experimental_group": cluster_key,
            "reaction_family": family,
            "yield_bin": yield_bin,
            "method": generator.method,
            "is_positive": False,
        })

    if not fps_pos:
        return None, np.array([]), []
    X = np.vstack(fps_pos + fps_neg)
    y = np.array([1] * len(fps_pos) + [0] * len(fps_neg), dtype=np.float32)
    return X, y, records
