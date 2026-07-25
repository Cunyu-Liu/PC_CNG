"""Phase 3 external validation: OOD splits + NI Coupling, with paired CI.

Runs the full Phase 3 external-validation protocol:

1. Load the G8-C trained model from
   ``results/p4_g8c_phase2_full/model_checkpoint.pt`` if available;
   otherwise fall back to the rule generator only.
2. For each OOD split (from ``data/ood_splits/``) and for the NI Coupling
   dataset:
   * Generate negatives with up to three methods:
     - ``learned_structured``  (G8-C model, if checkpoint exists)
     - ``rule_pc_cng``         (rule-based ReactionBoundaryGenerator)
     - ``random_mismatch``     (random product rotation baseline)
   * Train a Morgan-fingerprint MLP (256-d hidden, 2 layers) on
     (train positives label=1 + train negatives label=0).
   * Evaluate on the test split: per-record AUPRC, macro-AUPRC across
     reaction families / yield bins.
   * Compute a paired cluster bootstrap CI for
     ``learned_structured`` vs ``rule_pc_cng`` (the primary contrast).
3. Persist everything under ``results/phase3_external_validation/``:
   - ``per_split_results.json``
   - ``per_split_records/{split}__{method}.csv``  (for audit)
   - ``paired_ci.json``
   - ``ood_generalization_summary.json``
   - ``run_manifest.json``

Usage::

    python3 -m pc_cng.run_phase3_external_validation --gpu 5
    python3 -m pc_cng.run_phase3_external_validation --splits random scaffold
    python3 -m pc_cng.run_phase3_external_validation --max-train 1000 --max-test 300

Design choices
--------------
* The downstream classifier is intentionally simple (Morgan FP + 2-layer
  MLP) so that runtime is dominated by negative generation, not training.
* One negative is generated per positive (1:1 ratio) to keep the
  comparison fair across methods.  Generation is capped by
  ``--max-train`` / ``--max-test`` for efficiency on a loaded server.
* Paired cluster bootstrap reuses ``paired_cluster_inference`` so the CI
  methodology matches G6/G8-C exactly.
* All compute runs at ``nice -n 19``; the MLP trains on CPU by default
  and only uses GPU when ``--gpu`` is explicitly passed and available.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap (so `python3 -m pc_cng.run_phase3_external_validation` works)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

# ---------------------------------------------------------------------------
# Optional RDKit import
# ---------------------------------------------------------------------------
os.environ.setdefault("RDKitRDLogger", "0")
try:  # pragma: no cover - import guard
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    _RDKIT_OK = True
except Exception:  # pragma: no cover
    Chem = None
    AllChem = None
    DataStructs = None
    _RDKIT_OK = False

# ---------------------------------------------------------------------------
# Project imports (deferred where they may pull heavy deps)
# ---------------------------------------------------------------------------
from pc_cng.paired_cluster_inference import (  # noqa: E402
    auprc_metric,
    family_macro_auprc_metric,
    macro_auprc_metric,
    paired_cluster_bootstrap,
)

# Enhanced Phase 3 components (critical fix + improved MLP)
from pc_cng.phase3_enhanced import (  # noqa: E402
    EnhancedMLP,
    reaction_fp_enhanced,
    make_negative_rxn,
    build_dataset_enhanced,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PARQUET = _REPO_ROOT / "data/processed/p4_hte_normalized.parquet"
DEFAULT_NI_CSV = _REPO_ROOT / "data/processed/ni_coupling_supplement.csv"
DEFAULT_OOD_DIR = _REPO_ROOT / "data/ood_splits"
DEFAULT_CHECKPOINT = _REPO_ROOT / "results/p4_g8c_phase2_full/model_checkpoint.pt"
DEFAULT_OUTPUT = _REPO_ROOT / "results/phase3_external_validation"

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
MLP_HIDDEN = 256
MLP_LAYERS = 2
MLP_EPOCHS = 20
MLP_BATCH = 256
MLP_LR = 1e-3
NEG_RATIO = 1  # one negative per positive

DEFAULT_MAX_TRAIN = 2000
DEFAULT_MAX_TEST = 500
DEFAULT_N_BOOTSTRAP = 1000  # reduced from 2000 for speed; still robust

# Methods
METHOD_LEARNED = "learned_structured"
METHOD_RULE = "rule_pc_cng"
METHOD_RANDOM = "random_mismatch"
ALL_METHODS = (METHOD_LEARNED, METHOD_RULE, METHOD_RANDOM)
BASELINE_METHODS = (METHOD_RULE, METHOD_RANDOM)


# ---------------------------------------------------------------------------
# Morgan fingerprint + MLP
# ---------------------------------------------------------------------------

def morgan_fingerprint(smiles: str, radius: int = MORGAN_RADIUS,
                       n_bits: int = MORGAN_BITS) -> Optional[np.ndarray]:
    """Return a binary Morgan fingerprint or ``None`` on failure."""
    if not _RDKIT_OK or not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def reaction_fp(reaction_smiles: str, radius: int = MORGAN_RADIUS,
                n_bits: int = MORGAN_BITS) -> Optional[np.ndarray]:
    """Fingerprint for a full reaction SMILES.

    Splits on ``>`` and concatenates reactant + product fingerprints into
    a fixed ``2*n_bits`` vector.  Missing sides are zero-padded so the
    dimension is always consistent.  Returns ``None`` only when *both*
    sides fail to parse.
    """
    if not reaction_smiles:
        return None
    parts = reaction_smiles.split(">")
    if len(parts) == 3:
        r_fp = morgan_fingerprint(parts[0], radius, n_bits)
        p_fp = morgan_fingerprint(parts[2], radius, n_bits)
        if r_fp is None and p_fp is None:
            return None
        # Always return a 2*n_bits vector; zero-pad missing sides.
        r_arr = r_fp if r_fp is not None else np.zeros(n_bits, dtype=np.float32)
        p_arr = p_fp if p_fp is not None else np.zeros(n_bits, dtype=np.float32)
        return np.concatenate([r_arr, p_arr])
    # Not a reaction SMILES — fingerprint as a molecule, zero-pad product side.
    fp = morgan_fingerprint(reaction_smiles, radius, n_bits)
    if fp is None:
        return None
    return np.concatenate([fp, np.zeros(n_bits, dtype=np.float32)])


def reaction_fp_dim() -> int:
    """Dimensionality of the reaction fingerprint (2*MORGAN_BITS)."""
    return 2 * MORGAN_BITS


class MorganMLP:
    """A tiny 2-layer MLP on top of Morgan fingerprints.

    Implemented in pure NumPy so we do not need to compete for GPU memory
    with the running G8-C training.  Trains with Adam-like updates.
    """

    def __init__(self, input_dim: int, hidden: int = MLP_HIDDEN,
                 n_layers: int = MLP_LAYERS, seed: int = 20260725):
        self.input_dim = input_dim
        self.hidden = hidden
        self.n_layers = n_layers
        rng = np.random.default_rng(seed)
        # Layer 1
        self.W1 = (rng.standard_normal((input_dim, hidden)).astype(np.float32)
                   * np.sqrt(2.0 / input_dim))
        self.b1 = np.zeros(hidden, dtype=np.float32)
        # Layer 2 (output)
        out_dim = 1
        self.W2 = (rng.standard_normal((hidden, out_dim)).astype(np.float32)
                   * np.sqrt(2.0 / hidden))
        self.b2 = np.zeros(out_dim, dtype=np.float32)
        # Adam state
        self._mW1 = np.zeros_like(self.W1); self._vW1 = np.zeros_like(self.W1)
        self._mb1 = np.zeros_like(self.b1); self._vb1 = np.zeros_like(self.b1)
        self._mW2 = np.zeros_like(self.W2); self._vW2 = np.zeros_like(self.W2)
        self._mb2 = np.zeros_like(self.b2); self._vb2 = np.zeros_like(self.b2)
        self._t = 0
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._eps = 1e-8

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        p = 1.0 / (1.0 + np.exp(-z2))  # sigmoid
        return a1, p

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, p = self._forward(X)
        return p.ravel()

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = MLP_EPOCHS,
              batch_size: int = MLP_BATCH, lr: float = MLP_LR,
              verbose: bool = False) -> List[float]:
        """Train with mini-batch Adam.  Returns per-epoch losses."""
        n = len(X)
        losses: List[float] = []
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_shuf = X[perm]
            y_shuf = y[perm]
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                xb = X_shuf[start:start + batch_size]
                yb = y_shuf[start:start + batch_size]
                loss = self._train_step(xb, yb, lr)
                epoch_loss += loss
                n_batches += 1
            avg = epoch_loss / max(1, n_batches)
            losses.append(avg)
            if verbose and (epoch + 1) % 5 == 0:
                print(f"    [MLP] epoch {epoch+1}/{epochs}  loss={avg:.4f}")
        return losses

    def _train_step(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        n = len(X)
        a1, p = self._forward(X)
        p = p.ravel()
        # Binary cross-entropy loss
        eps = 1e-7
        p_clip = np.clip(p, eps, 1 - eps)
        loss = -np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip))
        # Gradient
        dz2 = (p - y).reshape(-1, 1) / n  # [n,1]
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (a1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        # Adam update
        self._t += 1
        self._adam_update(self.W1, dW1, self._mW1, self._vW1, lr)
        self._adam_update(self.b1, db1, self._mb1, self._vb1, lr)
        self._adam_update(self.W2, dW2, self._mW2, self._vW2, lr)
        self._adam_update(self.b2, db2, self._mb2, self._vb2, lr)
        return float(loss)

    def _adam_update(self, param, grad, m, v, lr):
        m *= self._beta1
        m += (1 - self._beta1) * grad
        v *= self._beta2
        v += (1 - self._beta2) * (grad * grad)
        m_hat = m / (1 - self._beta1 ** self._t)
        v_hat = v / (1 - self._beta2 ** self._t)
        param -= lr * m_hat / (np.sqrt(v_hat) + self._eps)


# ---------------------------------------------------------------------------
# Negative generation
# ---------------------------------------------------------------------------

class _NoOpMapper:
    """A no-op atom-mapping adapter that returns reactions unchanged.

    Used so ``ReactionBoundaryGenerator`` does not attempt to load the
    heavy ``RXNMapper`` model — we rely on ``allow_unmapped_fallback``
    instead.  This keeps the Phase 3 validation fast on a loaded server.
    """

    available = False

    def map_reaction(self, reaction_smiles: str) -> Optional[str]:
        return reaction_smiles if reaction_smiles else None


class NegativeGenerator:
    """Wraps the three negative-generation methods behind a uniform API.

    Each method returns a list of negative reaction SMILES (one per input
    reaction when possible).  The wrapper caches results so re-runs are
    cheap and so the learned / rule methods share the same input set.
    """

    def __init__(self, method: str, model=None, top_k: int = 1,
                 device=None, seed: int = 20260725):
        self.method = method
        self.model = model
        self.top_k = top_k
        self.device = device
        self.seed = seed
        self._cache: Dict[str, Optional[str]] = {}
        if method == METHOD_RULE:
            from pc_cng.reaction_boundary_generator import ReactionBoundaryGenerator
            # allow_unmapped_fallback=True so we don't depend on the slow
            # RXNMapper; the generator will still apply local graph edits.
            self._rule_gen = ReactionBoundaryGenerator(
                allow_unmapped_fallback=True,
                mapper=_NoOpMapper(),
            )
        elif method == METHOD_RANDOM:
            self._rng = random.Random(seed)
        elif method == METHOD_LEARNED:
            if model is None:
                raise ValueError("learned_structured requires a loaded model.")
        else:
            raise ValueError(f"Unknown method: {method}")

    def generate(self, reaction_smiles: str) -> Optional[str]:
        """Generate a single negative reaction SMILES."""
        if reaction_smiles in self._cache:
            return self._cache[reaction_smiles]
        neg: Optional[str] = None
        if self.method == METHOD_LEARNED:
            neg = self._generate_learned(reaction_smiles)
        elif self.method == METHOD_RULE:
            neg = self._generate_rule(reaction_smiles)
        elif self.method == METHOD_RANDOM:
            neg = self._generate_random(reaction_smiles)
        self._cache[reaction_smiles] = neg
        return neg

    def _generate_learned(self, reaction_smiles: str) -> Optional[str]:
        """Use the G8-C StructuredProposalModel to propose a negative."""
        try:
            from pc_cng.p4_g8c_learned_structured_proposal import (
                generate_structured_proposal,
                _apply_structured_edit,
                _safe_split,
            )
            edits = generate_structured_proposal(
                self.model,
                reaction_smiles,
                top_k=max(1, self.top_k),
                device=self.device,
                use_validity_mask=True,
                risk_rerank=False,
            )
            # Reconstruct the full negative reaction from the edited product.
            parts = reaction_smiles.split(">")
            if len(parts) != 3:
                return None
            reactants, agents = parts[0], parts[1]
            for edit in edits:
                edited_product = _apply_structured_edit(reaction_smiles, edit)
                if edited_product and isinstance(edited_product, str):
                    return f"{reactants}>{agents}>{edited_product}"
            return None
        except Exception as exc:
            return None

    def _generate_rule(self, reaction_smiles: str) -> Optional[str]:
        """Use the rule-based ReactionBoundaryGenerator."""
        try:
            candidates = self._rule_gen.generate_for_reaction(
                reaction_smiles, source_id="phase3", include_failed=False,
            )
            if candidates:
                # BoundaryCandidate.candidate_reaction is the full negative rxn
                c = candidates[0]
                neg = getattr(c, "candidate_reaction", None)
                if neg and isinstance(neg, str):
                    return neg
            return None
        except Exception:
            return None

    def _generate_random(self, reaction_smiles: str) -> Optional[str]:
        """Random product mismatch: replace the product with a simple valid molecule.

        Uses a fixed pool of common simple molecules so the result is always
        a syntactically valid (but chemically nonsensical) reaction.  This is
        the ``obvious negative`` baseline.
        """
        try:
            parts = reaction_smiles.split(">")
            if len(parts) != 3:
                return None
            reactants, agents, product = parts[0], parts[1], parts[2]
            # Pool of simple, guaranteed-valid molecules to use as fake products.
            pool = ["c1ccccc1", "C", "CC", "CCC", "CCO", "C1CCCCC1", "CC(=O)O"]
            if not hasattr(self, "_pool_idx"):
                self._pool_idx = 0
            fake_product = pool[self._pool_idx % len(pool)]
            self._pool_idx += 1
            return f"{reactants}>{agents}>{fake_product}"
        except Exception:
            return None


def load_g8c_model(checkpoint_path: Path, device=None):
    """Load the G8-C StructuredProposalModel checkpoint.

    Returns ``(model, device)`` on success or ``(None, error_msg)`` on failure.

    The training script (p4_g8c_learned_structured_proposal) saves a
    StructuredProposalModel checkpoint with keys:
      state_dict, hidden_dim, architecture="StructuredProposalModel"

    The older LearnedGraphEditDecoder.load_checkpoint expects a different
    architecture, so we load directly here.
    """
    import torch
    if not checkpoint_path.exists():
        return None, f"checkpoint not found: {checkpoint_path}"
    try:
        checkpoint = torch.load(
            str(checkpoint_path), map_location=device, weights_only=False)
        arch = checkpoint.get("architecture", "StructuredProposalModel")
        if arch == "StructuredProposalModel":
            from pc_cng.p4_g8c_learned_structured_proposal import (
                StructuredProposalModel,
                DEFAULT_HIDDEN,
                DEFAULT_HEADS,
                DEFAULT_NUM_LAYERS,
                DEFAULT_DROPOUT,
            )
            hidden_dim = int(checkpoint.get("hidden_dim", DEFAULT_HIDDEN))
            model = StructuredProposalModel(
                hidden_dim=hidden_dim,
                num_heads=DEFAULT_HEADS,
                num_layers=DEFAULT_NUM_LAYERS,
                dropout=DEFAULT_DROPOUT,
            )
            model.load_state_dict(checkpoint["state_dict"])
            if device is not None:
                model = model.to(device)
            model.eval()
            return model, device
        else:
            # Fallback: try LearnedGraphEditDecoder for legacy checkpoints
            from pc_cng.learned_graph_edit_decoder import load_checkpoint
            model, meta = load_checkpoint(str(checkpoint_path), device=device)
            return model, device
    except Exception as exc:
        return None, f"failed to load checkpoint: {exc}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hitea_split(parquet_path: Path, ood_dir: Path, split_name: str,
                     max_train: int, max_test: int) -> Dict:
    """Load the HiTEA parquet and return train/val/test reaction rows."""
    import pandas as pd
    df = pd.read_parquet(parquet_path).reset_index(drop=True)

    train_path = ood_dir / f"{split_name}_train_idx.json"
    test_path = ood_dir / f"{split_name}_test_idx.json"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"OOD split '{split_name}' not found in {ood_dir}. "
            "Run `python3 -m pc_cng.ood_splits` first."
        )
    with open(train_path) as fh:
        train_idx = json.load(fh)
    with open(test_path) as fh:
        test_idx = json.load(fh)

    def _select(rows, idx, cap):
        sub = rows.iloc[idx]
        if cap and len(sub) > cap:
            sub = sub.sample(n=cap, random_state=20260725)
        return sub

    train_rows = _select(df, train_idx, max_train)
    test_rows = _select(df, test_idx, max_test)
    return {
        "train": train_rows,
        "test": test_rows,
        "n_train_total": len(train_idx),
        "n_test_total": len(test_idx),
    }


def load_ni_coupling(ni_csv: Path, max_train: int, max_test: int) -> Dict:
    """Load NI Coupling data.

    The CSV's built-in test split is tiny (2 records), so we re-split the
    1,684 train records into an internal 80/20 train/test partition.  This
    gives a proper external-validation evaluation on Ni-catalysed coupling
    reactions that were never seen by the PC-CNG training pipeline.
    """
    import pandas as pd
    df = pd.read_csv(ni_csv)
    # The CSV has split=train (1684), val (2), test (2).  We re-split the
    # train pool into 80/20 for a meaningful external evaluation.
    train_pool = df[df["split"] == "train"].copy()
    if len(train_pool) < 20:
        # Fallback: use the whole dataset
        train_pool = df.copy()
    n_test_target = max(20, int(len(train_pool) * 0.20))
    test_rows = train_pool.sample(n=n_test_target, random_state=20260725)
    train_rows = train_pool.drop(test_rows.index)
    if max_train and len(train_rows) > max_train:
        train_rows = train_rows.sample(n=max_train, random_state=20260725)
    if max_test and len(test_rows) > max_test:
        test_rows = test_rows.sample(n=max_test, random_state=20260725)
    return {
        "train": train_rows,
        "test": test_rows,
        "n_train_total": len(train_pool),
        "n_test_total": n_test_target,
    }


def load_uspto_patent(uspto_csv: Path, ood_dir: Path,
                      max_train: int, max_test: int) -> Dict:
    """Load USPTO patent-disjoint split.

    The patent split indices (patent_{train,val,test}_idx.json) refer to row
    positions in the USPTO CSV.  This is a true patent-disjoint OOD split:
    train and test reactions come from completely different patents.

    USPTO records are all positive (yield 25-150%), so negatives are
    generated by the NegativeGenerator (same as other splits).
    """
    import pandas as pd
    df = pd.read_csv(uspto_csv).reset_index(drop=True)
    train_path = ood_dir / "patent_train_idx.json"
    test_path = ood_dir / "patent_test_idx.json"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Patent split not found in {ood_dir}. "
            "Run /tmp/create_patent_split.py first."
        )
    with open(train_path) as fh:
        train_idx = json.load(fh)
    with open(test_path) as fh:
        test_idx = json.load(fh)
    train_rows = df.iloc[train_idx].copy()
    test_rows = df.iloc[test_idx].copy()
    if max_train and len(train_rows) > max_train:
        train_rows = train_rows.sample(n=max_train, random_state=20260725)
    if max_test and len(test_rows) > max_test:
        test_rows = test_rows.sample(n=max_test, random_state=20260725)
    return {
        "train": train_rows,
        "test": test_rows,
        "n_train_total": len(train_idx),
        "n_test_total": len(test_idx),
    }


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

def build_dataset(rows, generator: NegativeGenerator,
                  fp_fn=reaction_fp) -> Tuple[Optional[np.ndarray], np.ndarray, List[Dict]]:
    """Build (X, y, records) where positives are real and negatives are generated.

    Each positive contributes one positive example (label=1) and one
    generated negative (label=0).  Records carry cluster_key metadata for
    paired bootstrap.
    """
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
        neg_rxn = generator.generate(rxn)
        if neg_rxn is None:
            n_neg_fail += 1
            continue
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
            "negative_smiles": neg_rxn,
            "label": 1,
            "score": 0.0,
            "experimental_group": cluster_key,
            "reaction_family": family,
            "yield_bin": yield_bin,
            "method": generator.method,
            "is_positive": True,
        })
        records.append({
            "reaction_smiles": neg_rxn,
            "negative_smiles": neg_rxn,
            "label": 0,
            "score": 0.0,
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


def evaluate_method(train_rows, test_rows, generator: NegativeGenerator,
                    seed: int) -> Dict:
    """Train Morgan-MLP on train positives/negatives, evaluate on test."""
    t0 = time.time()
    # CRITICAL: use build_dataset_enhanced which constructs proper reaction
    # SMILES for negatives (reactants>agents>neg_product), fixing the
    # "second half = zeros" bug. Uses enhanced 4*n_bits fingerprint.
    X_train, y_train, _ = build_dataset_enhanced(train_rows, generator)
    if X_train is None or len(X_train) < 10:
        return {"error": "insufficient training data", "n_train": 0}

    X_test, y_test, test_records = build_dataset_enhanced(test_rows, generator)
    if X_test is None or len(X_test) < 4:
        return {"error": "insufficient test data", "n_train": len(X_train)}

    mlp = EnhancedMLP(input_dim=X_train.shape[1], seed=seed)
    mlp.train(X_train, y_train, epochs=MLP_EPOCHS, batch_size=MLP_BATCH,
              lr=MLP_LR, verbose=False)

    scores = mlp.predict_proba(X_test)
    for i, rec in enumerate(test_records):
        rec["score"] = float(scores[i])

    # Metrics
    auprc = auprc_metric(test_records)
    macro = macro_auprc_metric(test_records, bin_key="yield_bin")
    fam_macro = family_macro_auprc_metric(test_records, family_key="reaction_family")

    elapsed = time.time() - t0
    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "auprc": float(auprc),
        "macro_auprc": float(macro),
        "family_macro_auprc": float(fam_macro),
        "elapsed_sec": float(elapsed),
        "test_records": test_records,
    }


def run_split(split_name: str, split_data: Dict, methods: Sequence[str],
              model, device, seed: int, n_bootstrap: int = DEFAULT_N_BOOTSTRAP) -> Dict:
    """Run all methods on one split and compute paired CI."""
    print(f"\n[phase3] === Split: {split_name} ===")
    print(f"  train={len(split_data['train'])}  test={len(split_data['test'])}")

    results: Dict[str, Dict] = {}
    test_records_by_method: Dict[str, List[Dict]] = {}

    for method in methods:
        print(f"  [{method}] generating negatives + training MLP ...")
        if method == METHOD_LEARNED and model is None:
            print(f"    SKIP: no G8-C model available")
            results[method] = {"error": "model_not_available", "skipped": True}
            continue
        try:
            if method == METHOD_LEARNED:
                gen = NegativeGenerator(method, model=model, device=device, seed=seed)
            else:
                gen = NegativeGenerator(method, seed=seed)
            res = evaluate_method(split_data["train"], split_data["test"], gen, seed)
            results[method] = res
            if "test_records" in res:
                test_records_by_method[method] = res["test_records"]
                print(f"    AUPRC={res['auprc']:.4f}  macro={res['macro_auprc']:.4f}  "
                      f"fam_macro={res['family_macro_auprc']:.4f}  "
                      f"({res['elapsed_sec']:.1f}s)")
            else:
                print(f"    ERROR: {res.get('error', 'unknown')}")
        except Exception as exc:
            results[method] = {"error": str(exc)}
            print(f"    ERROR: {exc}")

    # Paired cluster bootstrap CI for ALL available method pairs.
    # Priority order: learned_vs_rule, learned_vs_random, rule_vs_random.
    # Previously only learned_vs_rule was computed -> empty {} when G8-C
    # training was still running.  Now rule_vs_random is always available
    # so the baseline comparison (the user's "rule_pc_cng vs learned_structured"
    # task at minimum needs rule_vs_random as the floor) is captured.
    paired_ci: Dict = {}
    # Ordered challenger-vs-baseline pairs to compute (challenger, baseline).
    _pairs = [
        (METHOD_LEARNED, METHOD_RULE),
        (METHOD_LEARNED, METHOD_RANDOM),
        (METHOD_RULE, METHOD_RANDOM),  # always available with current baselines
    ]
    for challenger, baseline in _pairs:
        if challenger not in test_records_by_method:
            continue
        if baseline not in test_records_by_method:
            continue
        pair_key = f"{challenger}_vs_{baseline}"
        print(f"  [paired CI] {pair_key} ...")
        try:
            ch_records = list(test_records_by_method[challenger])
            bl_records = list(test_records_by_method[baseline])
            # Records differ in negatives but share the same positive test
            # reactions.  Align by trimming to the same length (positives are
            # the leading block in build_dataset, so the first N positives
            # match).  This is an exploratory approximation; a fully rigorous
            # comparison would re-score on a shared candidate set.
            n = min(len(ch_records), len(bl_records))
            ch_records = ch_records[:n]
            bl_records = bl_records[:n]
            ci = paired_cluster_bootstrap(
                ch_records, bl_records,
                metric_fn=auprc_metric,
                cluster_key="experimental_group",
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            paired_ci[pair_key] = ci
            print(f"    delta={ci['delta_mean']:.4f}  "
                  f"CI=[{ci['delta_ci_low']:.4f}, {ci['delta_ci_high']:.4f}]  "
                  f"all_positive={ci['ci_all_positive']}  "
                  f"p={ci['p_value']:.4f}")
        except Exception as exc:
            paired_ci[pair_key] = {"error": str(exc)}
            print(f"    ERROR ({pair_key}): {exc}")

    return {
        "split_name": split_name,
        "n_train": len(split_data["train"]),
        "n_test": len(split_data["test"]),
        "n_train_total": split_data.get("n_train_total", len(split_data["train"])),
        "n_test_total": split_data.get("n_test_total", len(split_data["test"])),
        "methods": {m: {k: v for k, v in r.items() if k != "test_records"}
                     for m, r in results.items()},
        "paired_ci": paired_ci,
        "test_records": test_records_by_method,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3 external validation (OOD + NI Coupling)."
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--ni-csv", type=Path, default=DEFAULT_NI_CSV)
    parser.add_argument("--ood-dir", type=Path, default=DEFAULT_OOD_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index (default: CPU).  Sharing is allowed.")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="OOD splits to evaluate (default: all in ood-dir)")
    parser.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN)
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--no-ni", action="store_true",
                        help="Skip NI Coupling dataset")
    parser.add_argument("--uspto-csv", type=Path,
                        default=Path("/home/cunyuliu/pc_cng_research/data/processed/uspto_openmolecules_normalized.csv"))
    parser.add_argument("--no-uspto", action="store_true",
                        help="Skip USPTO patent-split dataset")
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS),
                        choices=list(ALL_METHODS))
    args = parser.parse_args(argv)

    n_bootstrap = args.n_bootstrap

    t_start = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    records_dir = args.output / "per_split_records"
    records_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase3] output      : {args.output}")
    print(f"[phase3] parquet     : {args.parquet}")
    print(f"[phase3] ni_csv      : {args.ni_csv}")
    print(f"[phase3] ood_dir     : {args.ood_dir}")
    print(f"[phase3] checkpoint  : {args.checkpoint}")
    print(f"[phase3] max_train   : {args.max_train}")
    print(f"[phase3] max_test    : {args.max_test}")
    print(f"[phase3] n_bootstrap : {args.n_bootstrap}")
    print(f"[phase3] methods     : {args.methods}")

    # ---- Device ----
    device = None
    if args.gpu is not None:
        try:
            import torch
            if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
                device = torch.device(f"cuda:{args.gpu}")
                free, total = torch.cuda.mem_get_info(args.gpu)
                print(f"[phase3] GPU {args.gpu}: {free/1e9:.1f}/{total/1e9:.1f} GB free")
            else:
                print(f"[phase3] GPU {args.gpu} not available; using CPU")
        except Exception:
            print("[phase3] PyTorch/CUDA unavailable; using CPU")
    if device is None:
        print("[phase3] using CPU for any torch ops")

    # ---- Load G8-C model (optional) ----
    model, model_err = load_g8c_model(args.checkpoint, device=device)
    if model is not None:
        print(f"[phase3] G8-C model loaded from {args.checkpoint}")
        methods = [m for m in args.methods if m != METHOD_LEARNED] + [METHOD_LEARNED]
    else:
        print(f"[phase3] G8-C model NOT loaded: {model_err}")
        print(f"[phase3] Will run with baseline methods only: "
              f"{[m for m in args.methods if m != METHOD_LEARNED]}")
        methods = [m for m in args.methods if m != METHOD_LEARNED]

    # ---- Discover OOD splits ----
    if args.splits:
        split_names = list(args.splits)
    else:
        split_names = []
        for path in sorted(args.ood_dir.glob("*_metadata.json")):
            name = path.name.replace("_metadata.json", "")
            if name == "splits_manifest":
                continue
            # 'patent' split uses USPTO indices, not HiTEA; it is evaluated
            # separately as the uspto_patent external dataset below.
            if name == "patent":
                continue
            split_names.append(name)
    if not split_names:
        print("[phase3] WARNING: no OOD splits found; running NI Coupling only")

    # ---- Run each OOD split ----
    all_results: Dict[str, Dict] = {}
    for split_name in split_names:
        try:
            split_data = load_hitea_split(args.parquet, args.ood_dir, split_name,
                                          args.max_train, args.max_test)
        except Exception as exc:
            print(f"[phase3] ERROR loading split '{split_name}': {exc}")
            all_results[split_name] = {"error": str(exc)}
            continue
        res = run_split(split_name, split_data, methods, model, device,
                        args.seed, n_bootstrap)
        all_results[split_name] = res
        # Persist per-split records for audit
        for method, recs in res.get("test_records", {}).items():
            rec_path = records_dir / f"{split_name}__{method}.csv"
            with open(rec_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=[
                    "reaction_smiles", "negative_smiles", "label", "score",
                    "experimental_group", "reaction_family", "yield_bin",
                    "method", "is_positive",
                ])
                writer.writeheader()
                for r in recs:
                    writer.writerow({k: r.get(k, "") for k in r})

    # ---- NI Coupling ----
    if not args.no_ni and args.ni_csv.exists():
        print(f"\n[phase3] === NI Coupling ===")
        try:
            ni_data = load_ni_coupling(args.ni_csv, args.max_train, args.max_test)
            # NI Coupling uses reaction_smiles directly; add reaction_family col
            ni_data["train"]["reaction_family"] = "NI_COUPLING"
            ni_data["test"]["reaction_family"] = "NI_COUPLING"
            ni_data["train"]["experimental_group"] = ni_data["train"]["split_key"]
            ni_data["test"]["experimental_group"] = ni_data["test"]["split_key"]
            ni_data["train"]["yield_bin"] = 0
            ni_data["test"]["yield_bin"] = 0
            ni_res = run_split("ni_coupling", ni_data, methods, model, device,
                               args.seed, n_bootstrap)
            all_results["ni_coupling"] = ni_res
            for method, recs in ni_res.get("test_records", {}).items():
                rec_path = records_dir / f"ni_coupling__{method}.csv"
                with open(rec_path, "w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=[
                        "reaction_smiles", "negative_smiles", "label", "score",
                        "experimental_group", "reaction_family", "yield_bin",
                        "method", "is_positive",
                    ])
                    writer.writeheader()
                    for r in recs:
                        writer.writerow({k: r.get(k, "") for k in r})
        except Exception as exc:
            print(f"[phase3] ERROR on NI Coupling: {exc}")
            all_results["ni_coupling"] = {"error": str(exc)}

    # ---- USPTO Patent Split ----
    if not args.no_uspto and args.uspto_csv.exists():
        print(f"\n[phase3] === USPTO Patent Split ===")
        try:
            uspto_data = load_uspto_patent(args.uspto_csv, args.ood_dir,
                                           args.max_train, args.max_test)
            uspto_data["train"]["reaction_family"] = "USPTO_PATENT"
            uspto_data["test"]["reaction_family"] = "USPTO_PATENT"
            uspto_data["train"]["experimental_group"] = uspto_data["train"]["split_key"]
            uspto_data["test"]["experimental_group"] = uspto_data["test"]["split_key"]
            uspto_data["train"]["yield_bin"] = 0
            uspto_data["test"]["yield_bin"] = 0
            uspto_res = run_split("uspto_patent", uspto_data, methods, model, device,
                                  args.seed, n_bootstrap)
            all_results["uspto_patent"] = uspto_res
            for method, recs in uspto_res.get("test_records", {}).items():
                rec_path = records_dir / f"uspto_patent__{method}.csv"
                with open(rec_path, "w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=[
                        "reaction_smiles", "negative_smiles", "label", "score",
                        "experimental_group", "reaction_family", "yield_bin",
                        "method", "is_positive",
                    ])
                    writer.writeheader()
                    for r in recs:
                        writer.writerow({k: r.get(k, "") for k in r})
        except Exception as exc:
            print(f"[phase3] ERROR on USPTO Patent: {exc}")
            all_results["uspto_patent"] = {"error": str(exc)}

    # ---- Persist results ----
    # Strip test_records from the summary JSON (they go to CSV)
    summary = {}
    for name, res in all_results.items():
        if "error" in res and "methods" not in res:
            summary[name] = {"error": res["error"]}
            continue
        clean = {k: v for k, v in res.items() if k != "test_records"}
        summary[name] = clean

    with open(args.output / "per_split_results.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # Paired CI summary
    paired_ci_summary = {}
    for name, res in all_results.items():
        if isinstance(res, dict) and res.get("paired_ci"):
            paired_ci_summary[name] = res["paired_ci"]
    with open(args.output / "paired_ci.json", "w") as fh:
        json.dump(paired_ci_summary, fh, indent=2, default=str)

    # OOD generalization summary
    ood_summary = build_ood_generalization_summary(all_results)
    with open(args.output / "ood_generalization_summary.json", "w") as fh:
        json.dump(ood_summary, fh, indent=2, default=str)

    # Run manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_sec": time.time() - t_start,
        "parquet": str(args.parquet),
        "ni_csv": str(args.ni_csv),
        "ood_dir": str(args.ood_dir),
        "checkpoint": str(args.checkpoint),
        "model_loaded": model is not None,
        "model_error": model_err,
        "methods": methods,
        "splits": list(all_results.keys()),
        "max_train": args.max_train,
        "max_test": args.max_test,
        "n_bootstrap": args.n_bootstrap,
        "gpu": args.gpu,
        "seed": args.seed,
    }
    with open(args.output / "run_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    # ---- Console summary ----
    print("\n" + "=" * 70)
    print("[phase3] === External Validation Summary ===")
    print("=" * 70)
    for name, res in all_results.items():
        if "error" in res and "methods" not in res:
            print(f"  {name:20s}  ERROR: {res['error']}")
            continue
        methods_res = res.get("methods", {})
        for method, mr in methods_res.items():
            if "error" in mr:
                print(f"  {name:20s}  {method:20s}  ERROR: {mr['error']}")
            else:
                print(f"  {name:20s}  {method:20s}  "
                      f"AUPRC={mr.get('auprc', 0):.4f}  "
                      f"macro={mr.get('macro_auprc', 0):.4f}")
        if res.get("paired_ci"):
            ci = res["paired_ci"]
            # ci may be a dict of pairs {pair_key: {delta_mean, ...}} (new)
            # or a single dict (legacy).
            if isinstance(ci, dict) and "delta_mean" not in ci:
                for pair_key, pci in ci.items():
                    if isinstance(pci, dict) and "delta_mean" in pci:
                        print(f"  {'':20s}  {pair_key:20s}  "
                              f"delta={pci['delta_mean']:.4f}  "
                              f"CI=[{pci['delta_ci_low']:.4f}, {pci['delta_ci_high']:.4f}]  "
                              f"sig={pci.get('ci_all_positive', False)}  "
                              f"p={pci.get('p_value', 0):.4f}")
                    elif isinstance(pci, dict) and pci.get("error"):
                        print(f"  {'':20s}  {pair_key:20s}  ERROR: {pci['error']}")
            elif isinstance(ci, dict) and "delta_mean" in ci:
                print(f"  {'':20s}  {'paired CI':20s}  "
                      f"delta={ci['delta_mean']:.4f}  "
                      f"CI=[{ci['delta_ci_low']:.4f}, {ci['delta_ci_high']:.4f}]  "
                      f"sig={ci.get('ci_all_positive', False)}")
    print(f"\n[phase3] total elapsed: {time.time() - t_start:.1f}s")
    print(f"[phase3] results saved to: {args.output}")
    return 0


def build_ood_generalization_summary(all_results: Dict[str, Dict]) -> Dict:
    """Aggregate OOD generalization metrics across splits."""
    summary: Dict[str, Any] = {
        "ood_splits": {},
        "ni_coupling": None,
        "overall": {},
    }
    # Per-split: compute the AUPRC drop from random (in-distribution) to each OOD split
    random_auprc = None
    for name, res in all_results.items():
        if "methods" not in res:
            continue
        for method, mr in res.get("methods", {}).items():
            if "auprc" not in mr:
                continue
            if name == "random" and method == METHOD_RULE:
                random_auprc = mr["auprc"]
    for name, res in all_results.items():
        if "methods" not in res:
            continue
        split_summary: Dict[str, Any] = {}
        for method, mr in res.get("methods", {}).items():
            if "auprc" in mr:
                split_summary[method] = {
                    "auprc": mr["auprc"],
                    "macro_auprc": mr.get("macro_auprc", 0),
                    "family_macro_auprc": mr.get("family_macro_auprc", 0),
                }
                if random_auprc is not None and name != "random":
                    split_summary[method]["auprc_drop_vs_random"] = (
                        random_auprc - mr["auprc"]
                    )
        if split_summary:
            if name == "ni_coupling":
                summary["ni_coupling"] = split_summary
            else:
                summary["ood_splits"][name] = split_summary
    # Overall: average AUPRC drop across OOD splits for rule_pc_cng
    drops = []
    for name, ss in summary["ood_splits"].items():
        if name == "random":
            continue
        for method, vals in ss.items():
            if "auprc_drop_vs_random" in vals:
                drops.append((method, vals["auprc_drop_vs_random"]))
    if drops:
        by_method: Dict[str, List[float]] = defaultdict(list)
        for m, d in drops:
            by_method[m].append(d)
        summary["overall"]["mean_auprc_drop_by_method"] = {
            m: float(np.mean(ds)) for m, ds in by_method.items()
        }
    summary["overall"]["n_splits"] = len(summary["ood_splits"])
    summary["overall"]["random_reference_auprc"] = random_auprc
    return summary


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
