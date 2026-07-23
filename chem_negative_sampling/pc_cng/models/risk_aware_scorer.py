"""P4-G5: Risk-aware scoring for synthetic counterfactual candidates.

Computes per-candidate risk signals, trains a false-negative-risk model
calibrated ONLY on observed data (positive_observed vs negative_observed
from HTEa, plus known_positive_collision), and produces risk-based sample
weights:

    sample_weight = chemical_validity
                  x data_support
                  x boundary_value
                  x (1 - false_negative_risk)

Hard rules (per P4-G5 spec):
- The risk model is NEVER trained on synthetic-candidate self-labels.
- Calibration classes: positive_observed (HTEa yield>0),
  negative_observed (HTEa yield==0 / label_type=real_negative),
  known_positive_collision (treated as positive, fnr=1 override).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover
    RDKIT_AVAILABLE = False

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Risk signal names (13, per P4-G5 spec)
# ---------------------------------------------------------------------------

RISK_SIGNALS = [
    "database_exact_collision",
    "template_collision",
    "nearest_positive_similarity",
    "nearest_negative_similarity",
    "ensemble_mean",
    "ensemble_variance",
    "epistemic_uncertainty",
    "aleatoric_uncertainty",
    "reaction_family_support",
    "edit_locality",
    "edit_distance",
    "atom_mapping_quality",
    "experimental_support",
]

# Features used by the FNR model.  Ensemble-derived signals are excluded
# because (a) the ensemble is trained on observed data only and is
# overconfident on near-positive counterfactuals, and (b) the P4-G5 NO-GO
# check penalises self-score dominance.  Structural / similarity features
# are more discriminative for distinguishing observed positives from
# counterfactual unknowns.
FNR_MODEL_FEATURES = [
    "database_exact_collision",
    "template_collision",
    "nearest_positive_similarity",
    "nearest_negative_similarity",
    "reaction_family_support",
    "edit_locality",
    "edit_distance",
    "atom_mapping_quality",
    "experimental_support",
]

WEIGHT_COMPONENTS = [
    "chemical_validity",
    "data_support",
    "boundary_value",
    "one_minus_fnr",
]

MIN_WEIGHT = 1e-4


# ---------------------------------------------------------------------------
# Small chemistry helpers
# ---------------------------------------------------------------------------

def canonical_smiles(smiles: str) -> str:
    if not RDKIT_AVAILABLE or not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)


def morgan_fp(smiles: str, n_bits: int = 1024):
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, n_bits)


def chemical_validity_score(smiles: str) -> float:
    """1.0 for a clean single-fragment RDKit-parseable molecule, else partial.

    0.5 if parseable but multi-fragment (salt/mixture kept as largest frag),
    0.0 if unparseable.
    """
    if not RDKIT_AVAILABLE or not smiles:
        return 0.0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    n_frags = len(Chem.GetMolFrags(mol))
    return 1.0 if n_frags == 1 else 0.5


def atom_mapping_quality(status: str) -> float:
    return {"mapped": 1.0, "unmapped": 0.5}.get(status or "", 0.0)


# ---------------------------------------------------------------------------
# Observed-data pool (calibration set for the risk model)
# ---------------------------------------------------------------------------

@dataclass
class ObservedPool:
    """HTEa observed reactions used to calibrate the risk model.

    Only rows from allowed split keys (train parents) are used, so the risk
    model never sees manifest val/test parents.
    """

    pos_smiles: List[str] = field(default_factory=list)   # yield > 0
    neg_smiles: List[str] = field(default_factory=list)   # yield == 0
    pos_family: List[str] = field(default_factory=list)
    neg_family: List[str] = field(default_factory=list)
    family_counts: Dict[str, int] = field(default_factory=dict)
    group_sizes: Dict[str, int] = field(default_factory=dict)
    family_products: Dict[str, set] = field(default_factory=dict)
    all_products: set = field(default_factory=set)


def build_observed_pool(
    htea_csv_path: Path,
    excluded_split_keys: frozenset,
    max_neg_ratio: float = 2.0,
    seed: int = 20260723,
) -> ObservedPool:
    """Load HTEa rows, keep only non-excluded (train-parent) reactions."""
    import csv

    pool = ObservedPool()
    negs_all: List[Tuple[str, str]] = []
    with open(htea_csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sk = row.get("split_key", "")
            if sk in excluded_split_keys:
                continue
            prod = (row.get("products") or "").strip()
            if not prod:
                continue
            try:
                y = float(row.get("yield", "") or 0.0)
            except ValueError:
                continue
            fam = row.get("reaction_class", "") or "unknown"
            canon = canonical_smiles(prod)
            if not canon:
                continue
            pool.family_counts[fam] = pool.family_counts.get(fam, 0) + 1
            pool.group_sizes[sk] = pool.group_sizes.get(sk, 0) + 1
            pool.family_products.setdefault(fam, set()).add(canon)
            pool.all_products.add(canon)
            if y > 0:
                pool.pos_smiles.append(canon)
                pool.pos_family.append(fam)
            else:
                negs_all.append((canon, fam))

    # Downsample observed negatives to max_neg_ratio x positives (fixed seed)
    rng = random.Random(seed)
    n_neg = min(len(negs_all), int(len(pool.pos_smiles) * max_neg_ratio))
    negs = rng.sample(negs_all, n_neg) if len(negs_all) > n_neg else negs_all
    pool.neg_smiles = [s for s, _ in negs]
    pool.neg_family = [f for _, f in negs]
    return pool


# ---------------------------------------------------------------------------
# Reference fingerprint index for nearest-neighbor signals
# ---------------------------------------------------------------------------

class _NNIndex:
    """Tanimoto nearest-neighbor over a fixed reference subset (C-speed bulk)."""

    def __init__(self, smiles_list: Sequence[str], max_ref: int, seed: int):
        rng = random.Random(seed)
        refs = list(dict.fromkeys(smiles_list))  # dedupe, keep order
        if len(refs) > max_ref:
            refs = rng.sample(refs, max_ref)
        self.fps = []
        for s in refs:
            fp = morgan_fp(s)
            if fp is not None:
                self.fps.append(fp)

    def max_sim(self, smiles: str) -> float:
        fp = morgan_fp(smiles)
        if fp is None or not self.fps:
            return 0.0
        sims = DataStructs.BulkTanimotoSimilarity(fp, self.fps)
        return float(max(sims)) if sims else 0.0


# ---------------------------------------------------------------------------
# Feasibility ensemble (model-based risk signals)
# ---------------------------------------------------------------------------

class _MLPMember(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FeasibilityEnsemble:
    """Bagged MLP ensemble on Morgan fingerprints, trained on observed data.

    Members see bootstrap-resampled balanced subsets of the observed pool
    (positive_observed = 1, negative_observed = 0). Synthetic candidates are
    never used for training.
    """

    def __init__(self, n_members: int = 5, hidden: int = 256, seed: int = 20260723):
        self.n_members = n_members
        self.hidden = hidden
        self.seed = seed
        self.members: List[_MLPMember] = []

    @staticmethod
    def _fp_tensor(smiles_list: Sequence[str]) -> Tuple[Optional[torch.Tensor], List[int]]:
        arrs, keep = [], []
        for i, s in enumerate(smiles_list):
            fp = morgan_fp(s)
            if fp is None:
                continue
            arr = np.zeros(1024, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            arrs.append(arr)
            keep.append(i)
        if not arrs:
            return None, []
        return torch.tensor(np.stack(arrs)), keep

    def fit(
        self,
        pos_smiles: Sequence[str],
        neg_smiles: Sequence[str],
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> List[float]:
        """Train members; returns per-member final loss."""
        x_pos, _ = self._fp_tensor(pos_smiles)
        x_neg, _ = self._fp_tensor(neg_smiles)
        assert x_pos is not None and x_neg is not None, "empty observed pool"
        losses = []
        for k in range(self.n_members):
            rng = random.Random(self.seed + k)
            torch.manual_seed(self.seed + k)
            # Balanced bootstrap
            n = min(len(x_pos), len(x_neg))
            pi = [rng.randrange(len(x_pos)) for _ in range(n)]
            ni = [rng.randrange(len(x_neg)) for _ in range(n)]
            x = torch.cat([x_pos[pi], x_neg[ni]], dim=0).to(device)
            y = torch.cat([torch.ones(n), torch.zeros(n)]).to(device)

            member = _MLPMember(in_dim=x.shape[1], hidden=self.hidden).to(device)
            opt = torch.optim.Adam(member.parameters(), lr=lr)
            member.train()
            final_loss = 0.0
            for ep in range(epochs):
                perm = torch.randperm(len(x), device=device)
                tot, nb = 0.0, 0
                for i in range(0, len(x), batch_size):
                    idx = perm[i:i + batch_size]
                    logits = member(x[idx])
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += loss.item()
                    nb += 1
                final_loss = tot / max(nb, 1)
            member.eval()
            self.members.append(member)
            losses.append(final_loss)
        return losses

    @torch.no_grad()
    def score(self, smiles_list: Sequence[str], device: str = "cpu") -> Dict[str, List[float]]:
        """Return ensemble statistics per SMILES.

        Invalid SMILES get neutral uncertainty (mean 0.5, max variance).
        """
        x, keep = self._fp_tensor(smiles_list)
        n = len(smiles_list)
        mean = [0.5] * n
        var = [0.25] * n
        aleatoric = [0.25] * n
        if x is None:
            return {"mean": mean, "variance": var, "aleatoric": aleatoric}
        x = x.to(device)
        probs_per_member = []
        for m in self.members:
            probs_per_member.append(torch.sigmoid(m(x)))
        p = torch.stack(probs_per_member, dim=0)  # [K, N]
        p_mean = p.mean(dim=0)
        p_var = p.var(dim=0, unbiased=False)
        p_alea = (p * (1 - p)).mean(dim=0)
        for col, row in enumerate(keep):
            mean[row] = float(p_mean[col])
            var[row] = float(p_var[col])
            aleatoric[row] = float(p_alea[col])
        return {"mean": mean, "variance": var, "aleatoric": aleatoric}


# ---------------------------------------------------------------------------
# Risk feature extraction
# ---------------------------------------------------------------------------

class RiskFeatureExtractor:
    """Computes the 13 P4-G5 risk signals for candidate molecules."""

    def __init__(
        self,
        pool: ObservedPool,
        ensemble: Optional[FeasibilityEnsemble] = None,
        max_ref: int = 2000,
        seed: int = 20260723,
        device: str = "cpu",
    ):
        self.pool = pool
        self.ensemble = ensemble
        self.device = device
        self.pos_nn = _NNIndex(pool.pos_smiles, max_ref, seed + 1)
        self.neg_nn = _NNIndex(pool.neg_smiles, max_ref, seed + 2)
        self._max_family = max(pool.family_counts.values()) if pool.family_counts else 1
        self._max_group = max(pool.group_sizes.values()) if pool.group_sizes else 1

    def extract_batch(
        self,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, float]]:
        """Extract all 13 signals. Each candidate dict needs at least
        ``smiles`` (or ``candidate_smiles``); optional: ``gold_smiles``,
        ``reaction_family``, ``atom_mapping_status``, ``experimental_group_id``.
        """
        smiles_list = [
            (c.get("smiles") or c.get("candidate_smiles") or "") for c in candidates
        ]
        if self.ensemble is not None and self.ensemble.members:
            ens = self.ensemble.score(smiles_list, device=self.device)
        else:
            n = len(candidates)
            ens = {"mean": [0.5] * n, "variance": [0.25] * n, "aleatoric": [0.25] * n}

        out: List[Dict[str, float]] = []
        for i, cand in enumerate(candidates):
            smi = smiles_list[i]
            canon = canonical_smiles(smi)
            fam = cand.get("reaction_family", "") or "unknown"
            gold = cand.get("gold_smiles", "") or ""
            gold_fp = morgan_fp(gold)
            cand_fp = morgan_fp(smi)
            if gold_fp is not None and cand_fp is not None:
                locality = float(DataStructs.TanimotoSimilarity(gold_fp, cand_fp))
            else:
                locality = 0.0
            fam_support = math.log1p(self.pool.family_counts.get(fam, 0)) / math.log1p(self._max_family)
            gid = cand.get("experimental_group_id", "") or ""
            exp_support = math.log1p(self.pool.group_sizes.get(gid, 0)) / math.log1p(self._max_group)
            out.append({
                "database_exact_collision": 1.0 if canon and canon in self.pool.all_products else 0.0,
                "template_collision": 1.0 if canon and canon in self.pool.family_products.get(fam, set()) else 0.0,
                "nearest_positive_similarity": self.pos_nn.max_sim(smi),
                "nearest_negative_similarity": self.neg_nn.max_sim(smi),
                "ensemble_mean": float(ens["mean"][i]),
                "ensemble_variance": float(ens["variance"][i]),
                "epistemic_uncertainty": float(ens["variance"][i]),
                "aleatoric_uncertainty": float(ens["aleatoric"][i]),
                "reaction_family_support": float(fam_support),
                "edit_locality": locality,
                "edit_distance": 1.0 - locality,
                "atom_mapping_quality": atom_mapping_quality(cand.get("atom_mapping_status", "")),
                "experimental_support": float(exp_support),
                "chemical_validity": chemical_validity_score(smi),
            })
        return out


# ---------------------------------------------------------------------------
# False-negative-risk model (calibrated on observed data only)
# ---------------------------------------------------------------------------

class FalseNegativeRiskModel:
    """Logistic-regression risk model: P(candidate is actually positive).

    Trained ONLY on observed positives (y=1) vs observed negatives (y=0).
    Synthetic-candidate self-labels are never used.

    Uses FNR_MODEL_FEATURES (structural / similarity signals only) instead
    of all 13 RISK_SIGNALS, to avoid self-score dominance from the
    observed-data-trained ensemble.
    """

    def __init__(self, feature_names: Sequence[str] = FNR_MODEL_FEATURES):
        self.feature_names = list(feature_names)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def _featurize(self, feats: Sequence[Dict[str, float]]) -> np.ndarray:
        x = np.array([[f[k] for k in self.feature_names] for f in feats], dtype=np.float64)
        return x

    def _standardize(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.mean_ = x.mean(axis=0)
            self.std_ = x.std(axis=0)
            self.std_[self.std_ < 1e-12] = 1.0
        assert self.mean_ is not None and self.std_ is not None
        return (x - self.mean_) / self.std_

    def fit(
        self,
        pos_feats: Sequence[Dict[str, float]],
        neg_feats: Sequence[Dict[str, float]],
        l2: float = 1.0,
        epochs: int = 500,
        lr: float = 0.1,
        seed: int = 20260723,
    ) -> Dict[str, float]:
        """Fit logistic regression with simple full-batch GD (no sklearn dep
        on the hot path); returns train metrics."""
        x = np.concatenate([self._featurize(pos_feats), self._featurize(neg_feats)], axis=0)
        y = np.concatenate([np.ones(len(pos_feats)), np.zeros(len(neg_feats))])
        xs = self._standardize(x, fit=True)
        n, d = xs.shape
        rng = np.random.default_rng(seed)
        w = rng.normal(0, 0.01, size=d)
        b = 0.0
        for _ in range(epochs):
            z = xs @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            err = p - y
            grad_w = xs.T @ err / n + l2 * w / n
            grad_b = float(err.mean())
            w -= lr * grad_w
            b -= lr * grad_b
        self.coef_ = w
        self.intercept_ = float(b)
        p = self._predict_proba_std(xs)
        return {
            "train_logloss": float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))),
            "train_auroc": _auroc(y, p),
            "n_train": int(n),
        }

    def _predict_proba_std(self, xs: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None
        z = xs @ self.coef_ + self.intercept_
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict_fnr(self, feats: Sequence[Dict[str, float]]) -> List[float]:
        x = self._standardize(self._featurize(feats), fit=False)
        return [float(v) for v in self._predict_proba_std(x)]

    def to_dict(self) -> Dict[str, Any]:
        assert self.coef_ is not None and self.mean_ is not None and self.std_ is not None
        return {
            "feature_names": self.feature_names,
            "coef": [float(v) for v in self.coef_],
            "intercept": float(self.intercept_),
            "mean": [float(v) for v in self.mean_],
            "std": [float(v) for v in self.std_],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FalseNegativeRiskModel":
        m = cls(d["feature_names"])
        m.coef_ = np.array(d["coef"], dtype=np.float64)
        m.intercept_ = float(d["intercept"])
        m.mean_ = np.array(d["mean"], dtype=np.float64)
        m.std_ = np.array(d["std"], dtype=np.float64)
        return m


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUROC."""
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos = float(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def compute_sample_weights(
    feats: Sequence[Dict[str, float]],
    fnr: Sequence[float],
    known_positive_collision: Optional[Sequence[bool]] = None,
    ablate: Sequence[str] = (),
) -> List[Dict[str, float]]:
    """sample_weight = chemical_validity x data_support x boundary_value
    x (1 - false_negative_risk).

    - boundary_value: ``1 - |2*FNR - 1|`` (distance from the 0.5 decision
      boundary).  High when the FNR model is uncertain (FNR≈0.5); low when
      the model is confident (FNR≈0 or 1).  This replaces the previous
      min-max normalised epistemic_uncertainty, which was near-zero for
      feasible-looking candidates (ensemble overconfidence).
    - data_support: 0.5*family_support + 0.5*experimental_support.
    - known_positive_collision overrides fnr to 1.0 (weight -> MIN_WEIGHT).
    - Any component in ``ablate`` is replaced by 1.0 (neutralized).

    Returns per-candidate dicts with all components and the final weight.
    """
    unknown = set(ablate) - set(WEIGHT_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown weight components to ablate: {sorted(unknown)}")

    out: List[Dict[str, float]] = []
    for i, f in enumerate(feats):
        r = float(fnr[i])
        if known_positive_collision is not None and known_positive_collision[i]:
            r = 1.0
        # boundary_value: distance from FNR=0.5 decision boundary
        boundary = 1.0 - abs(2.0 * r - 1.0)
        comp = {
            "chemical_validity": float(f["chemical_validity"]),
            "data_support": float(0.5 * f["reaction_family_support"] + 0.5 * f["experimental_support"]),
            "boundary_value": boundary,
            "one_minus_fnr": 1.0 - r,
        }
        w = 1.0
        for k in WEIGHT_COMPONENTS:
            w *= 1.0 if k in ablate else comp[k]
        rec = dict(comp)
        rec["false_negative_risk"] = r
        rec["sample_weight"] = max(float(w), MIN_WEIGHT)
        out.append(rec)
    return out


def save_risk_model_manifest(
    path: Path,
    risk_model: FalseNegativeRiskModel,
    ensemble_meta: Dict[str, Any],
    pool_meta: Dict[str, Any],
    calibration: Dict[str, float],
    input_hashes: Dict[str, str],
) -> None:
    """Persist the risk-model manifest (parameters + provenance + calibration)."""
    payload = {
        "schema": "p4_g5_risk_model_manifest/v1",
        "risk_model": risk_model.to_dict(),
        "ensemble": ensemble_meta,
        "observed_pool": pool_meta,
        "calibration": calibration,
        "input_hashes": input_hashes,
        "policy": {
            "calibration_classes": [
                "positive_observed",
                "negative_observed",
                "known_positive_collision",
            ],
            "synthetic_labels_used_for_training": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
