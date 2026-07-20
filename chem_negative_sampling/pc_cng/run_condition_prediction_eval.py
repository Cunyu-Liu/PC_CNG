"""P2-08: Reaction Condition Prediction downstream evaluation (L8).

Goal
----
Fix L8 - add condition prediction as a new downstream task.  Evaluate
whether PC-CNG boundary negatives help condition prediction versus a
no-augmentation baseline.

DEGRADATION PATH (Section 26.1)
-------------------------------
The USPTO OpenMolecules ``agents`` column is empty for all 530,238 rows
in ``data/processed/uspto_openmolecules_normalized.csv`` (no explicit
catalyst/solvent/reagent labels).  An explicit USPTO-condition dataset
search across ``data/`` returned no matches.  Following the documented
degradation path:

1. HiTEA yield data is available but is a yield-regression task, not a
   condition-prediction task.
2. Therefore we generate a *synthetic* condition prediction dataset by
   deriving condition labels from the reactant SMILES via RDKit
   metal-atom detection.  The 10 condition classes are:

       0. Pd            5. Pt/Au
       1. Cu            6. Zn/Mg
       2. Ni            7. Li/Na/K
       3. Fe            8. B/Si
       4. Ru/Rh/Ir      9. Organic (no metals detected)

   Each USPTO reaction is assigned exactly one label based on the
   first matching class in its reactants.  This is explicitly tagged as
   ``"synthetic_condition_from_metals"`` in every output manifest.

EXPERIMENTAL DESIGN
-------------------
* ``baseline``: MLP classifier trained on Morgan fingerprints of the
  reactants → predict condition class (cross-entropy loss).
* ``treatment``: Same MLP, trained on (real positives) + (PC-CNG
  synthetic negatives, assigned the *parent positive's* condition
  label).  The PC-CNG negatives provide additional training examples
  with the same condition label as their parent positive, testing
  whether augmented training improves generalization.  This mirrors
  the v2 data-augmentation recipe used by ``run_cross_dataset_transfer_eval``.

METRICS
-------
Top-1, Top-3, Top-5, Top-10 accuracy + MRR + NDCG@10.

SIGNIFICANCE
------------
10-seed paired t-test (``scipy.stats.ttest_rel``) on the per-seed
metric deltas (treatment - baseline) for seeds 20260710..20260719.

OUTPUTS
-------
* ``summary.json``                  - per-seed metrics + mean ± std
* ``paired_significance.json``      - paired t-test on each metric
* ``per_seed_metrics.csv``          - one row per (seed, condition)
* ``go_no_go_decision.json``        - GO / NO-GO decision
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

# RDKit / torch are imported lazily inside functions so that unit tests
# which only exercise pure-python helpers do not require the heavy stack.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SEEDS: List[int] = [
    20260710, 20260711, 20260712, 20260713, 20260714,
    20260715, 20260716, 20260717, 20260718, 20260719,
]

DATASET_TAG = "synthetic_condition_from_metals"
FALLBACK_REASON = (
    "USPTO OpenMolecules normalized CSV has empty `agents` column for all "
    "rows; no separate USPTO condition dataset available. Deriving "
    "synthetic condition labels from reactant SMILES via RDKit metal-atom "
    "detection (degradation path Section 26.1)."
)

# Each entry is (class_name, set_of_atomic_numbers).
# Order matters: earlier classes take priority when multiple metals are
# present (Pd wins over Cu, etc.).  This mirrors a common chemist heuristic
# that the catalytic transition metal dominates the reaction class.
CONDITION_CLASSES: List[Tuple[str, set]] = [
    ("Pd", {46}),
    ("Cu", {29}),
    ("Ni", {28}),
    ("Fe", {26}),
    ("Ru_Rh_Ir", {44, 45, 77}),
    ("Pt_Au", {78, 79}),
    ("Zn_Mg", {30, 12}),
    ("Li_Na_K", {3, 11, 19}),
    ("B_Si", {5, 14}),
    ("Organic", set()),  # no metals detected
]

assert len(CONDITION_CLASSES) == 10, "Need exactly 10 classes for Top-10"

CLASS_NAME_TO_IDX: Dict[str, int] = {
    name: idx for idx, (name, _) in enumerate(CONDITION_CLASSES)
}
IDX_TO_CLASS_NAME: Dict[int, str] = {
    idx: name for idx, (name, _) in enumerate(CONDITION_CLASSES)
}


# ---------------------------------------------------------------------------
# Seed util
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python / NumPy / PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_seeds(raw: str) -> List[int]:
    """Parse a comma-separated seed string into a list of ints."""
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Condition-label extraction
# ---------------------------------------------------------------------------

def detect_condition_class(reactants_smiles: str) -> int:
    """Return the condition-class index for ``reactants_smiles``.

    The label is derived by scanning every molecule in the reactant side
    for the presence of metal atoms (by atomic number) defined in
    :data:`CONDITION_CLASSES`.  The first matching class wins.  If no
    metals are detected, the last class (``"Organic"``) is returned.

    Errors during SMILES parsing fall through to the ``"Organic"`` class
    so that malformed rows are not silently dropped from training.
    """
    try:
        from rdkit import Chem
    except Exception:
        # Without RDKit we cannot detect metals - fall through to Organic.
        return len(CONDITION_CLASSES) - 1

    if not reactants_smiles or not reactants_smiles.strip():
        return len(CONDITION_CLASSES) - 1

    atomic_nums: set = set()
    for part in reactants_smiles.split("."):
        part = part.strip()
        if not part:
            continue
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            atomic_nums.add(atom.GetAtomicNum())

    # Walk classes in priority order.  "Organic" (empty set) is last.
    for idx, (_, nums) in enumerate(CONDITION_CLASSES):
        if not nums:
            continue
        if atomic_nums & nums:
            return idx
    return len(CONDITION_CLASSES) - 1


def generate_synthetic_condition_dataset(
    source_csv: str,
    output_csv: str,
    limit: int | None = None,
) -> Dict[str, object]:
    """Derive a synthetic condition-prediction dataset from USPTO reactions.

    Reads the normalized USPTO CSV (columns: source_id, reaction_smiles,
    reactants, agents, products, label_type, yield, source, split_key,
    split) and writes a new CSV with the schema::

        source_id, reactants, products, condition_label, condition_idx, split

    The original ``split`` column is preserved so train/val/test
    partitions are honored.
    """
    rows_written = 0
    class_counts: Dict[str, int] = defaultdict(int)
    skipped_parse = 0
    skipped_empty = 0

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(source_csv, newline="", encoding="utf-8") as fin, \
         open(output_csv, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        fieldnames = [
            "source_id", "reactants", "products",
            "condition_label", "condition_idx", "split",
        ]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            if row.get("label_type", "") not in ("positive", ""):
                # Only positive reactions carry meaningful condition labels.
                continue
            reactants = (row.get("reactants") or "").strip()
            products = (row.get("products") or "").strip()
            if not reactants:
                skipped_empty += 1
                continue
            idx = detect_condition_class(reactants)
            name = IDX_TO_CLASS_NAME[idx]
            writer.writerow({
                "source_id": row.get("source_id", ""),
                "reactants": reactants,
                "products": products,
                "condition_label": name,
                "condition_idx": idx,
                "split": row.get("split", ""),
            })
            class_counts[name] += 1
            rows_written += 1
            if limit is not None and rows_written >= limit:
                break

    return {
        "source_csv": source_csv,
        "output_csv": output_csv,
        "rows_written": rows_written,
        "class_counts": dict(class_counts),
        "skipped_empty_reactants": skipped_empty,
        "dataset_tag": DATASET_TAG,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_condition_dataset(csv_path: str) -> List[Dict[str, object]]:
    """Read a synthetic-condition CSV produced by
    :func:`generate_synthetic_condition_dataset`.
    """
    rows: List[Dict[str, object]] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "source_id": row.get("source_id", ""),
                "reactants": row.get("reactants", ""),
                "products": row.get("products", ""),
                "condition_label": row.get("condition_label", ""),
                "condition_idx": int(row.get("condition_idx", 0)),
                "split": row.get("split", ""),
            })
    return rows


def read_pc_cng_negatives(csv_path: str) -> List[Dict[str, object]]:
    """Load PC-CNG synthetic negatives with parent lookup.

    Returns a list of dicts with keys ``source_id``, ``candidate_reactants``,
    ``parent_reaction``, ``review_status``.  Only rows whose
    ``review_status`` is ``keep_synthetic_negative`` (or empty) are kept;
    rows explicitly flagged as ``needs_review_or_downweight`` are dropped
    to mirror the v2 reviewed-negatives recipe.
    """
    rows: List[Dict[str, object]] = []
    if not csv_path or not os.path.exists(csv_path):
        return rows
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            review = (row.get("review_status") or "").strip()
            if review and review not in ("keep_synthetic_negative", ""):
                continue
            source_id = (row.get("source_id") or "").strip()
            candidate = (row.get("candidate_reactants") or "").strip()
            if not source_id or not candidate:
                continue
            rows.append({
                "source_id": source_id,
                "candidate_reactants": candidate,
                "parent_reaction": (row.get("positive_reaction") or "").strip(),
                "review_status": review,
            })
    return rows


def build_treatment_rows(
    real_rows: Sequence[Dict[str, object]],
    pc_cng_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Augment ``real_rows`` with PC-CNG negatives.

    Each PC-CNG negative inherits its *parent positive's* condition label,
    so the model sees extra training examples with the same target class.
    PC-CNG rows whose parent is not in ``real_rows`` are dropped (they
    cannot be safely assigned a label).
    """
    parent_lookup: Dict[str, Dict[str, object]] = {
        str(r["source_id"]): r for r in real_rows
    }
    augmented: List[Dict[str, object]] = []
    for neg in pc_cng_rows:
        parent = parent_lookup.get(str(neg["source_id"]))
        if parent is None:
            continue
        augmented.append({
            "source_id": f"{neg['source_id']}_pccng",
            "reactants": neg["candidate_reactants"],
            "products": parent.get("products", ""),
            "condition_label": parent["condition_label"],
            "condition_idx": parent["condition_idx"],
            "split": parent.get("split", "train"),
        })
    return list(real_rows) + augmented


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------

class ReactantFeaturizer:
    """Morgan-fingerprint featurizer for the reactant side of a reaction.

    Output dimension is ``n_bits`` (binary) or ``n_bits * 2`` (binary+count).
    """

    def __init__(self, n_bits: int = 2048, radius: int = 2) -> None:
        self.n_bits = n_bits
        self.radius = radius
        try:
            from rdkit.Chem import rdFingerprintGenerator
            self._gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius, fpSize=n_bits,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("RDKit is required for featurization") from exc

    def featurize(self, reactants_smiles: str) -> np.ndarray | None:
        try:
            from rdkit import Chem, DataStructs
        except Exception:  # pragma: no cover
            return None
        arr = np.zeros((self.n_bits,), dtype=np.float32)
        seen_any = False
        for part in (reactants_smiles or "").split("."):
            part = part.strip()
            if not part:
                continue
            mol = Chem.MolFromSmiles(part)
            if mol is None:
                continue
            fp = self._gen.GetFingerprint(mol)
            tmp = np.zeros((self.n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, tmp)
            arr = np.maximum(arr, tmp.astype(np.float32))
            seen_any = True
        if not seen_any:
            return None
        return arr


def featurize_rows(
    rows: Sequence[Dict[str, object]],
    featurizer: ReactantFeaturizer,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """Return ``(X, y, kept_rows)`` where each kept row produced a valid
    fingerprint.
    """
    feats: List[np.ndarray] = []
    labels: List[int] = []
    kept: List[Dict[str, object]] = []
    for row in rows:
        vec = featurizer.featurize(str(row.get("reactants", "")))
        if vec is None:
            continue
        feats.append(vec)
        labels.append(int(row["condition_idx"]))
        kept.append(row)
    if not feats:
        return (
            np.zeros((0, featurizer.n_bits), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
        )
    return (
        np.stack(feats, axis=0).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        kept,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_mlp(in_dim: int, hidden_dim: int, num_classes: int):
    """Build a small MLP classifier."""
    import torch
    from torch import nn

    class ConditionMLP(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, num_classes: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim // 2, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    return ConditionMLP(in_dim, hidden_dim, num_classes)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """Compute Top-K accuracy + MRR + NDCG@10.

    ``y_true``: 1-D int array of true class indices, shape ``(N,)``.
    ``y_pred_proba``: 2-D float array of class probabilities, shape
    ``(N, num_classes)``.
    """
    if len(y_true) == 0:
        return {
            "n_samples": 0,
            "top1": 0.0, "top3": 0.0, "top5": 0.0, "top10": 0.0,
            "mrr": 0.0, "ndcg_at_10": 0.0,
        }

    # Sort classes by predicted prob (descending).
    order = np.argsort(-y_pred_proba, axis=1)  # (N, num_classes)
    num_classes = y_pred_proba.shape[1]

    metrics: Dict[str, float] = {"n_samples": float(len(y_true))}
    for k in ks:
        topk = order[:, :k]
        hits = np.any(topk == y_true.reshape(-1, 1), axis=1)
        metrics[f"top{k}"] = float(np.mean(hits.astype(np.float32)))

    # MRR: 1 / rank of true class.
    true_ranks = np.argmax(order == y_true.reshape(-1, 1), axis=1) + 1
    metrics["mrr"] = float(np.mean(1.0 / true_ranks.astype(np.float32)))

    # NDCG@10: binary relevance (1 if true class, 0 otherwise).
    ndcg_scores: List[float] = []
    for i in range(len(y_true)):
        ranked_labels = (order[i, :10] == y_true[i]).astype(np.float32)
        dcg = float(np.sum(ranked_labels / np.log2(np.arange(2, 2 + len(ranked_labels)))))
        ideal_hits = 1.0
        idcg = float(ideal_hits / math.log2(2.0))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    metrics["ndcg_at_10"] = float(np.mean(ndcg_scores))

    return metrics


def dcg_at_k(labels: Sequence[int], k: int) -> float:
    """Discounted Cumulative Gain for a binary-relevance label list."""
    total = 0.0
    for rank, label in enumerate(labels[:k], start=1):
        if label:
            total += 1.0 / math.log2(rank + 1)
    return total


def ndcg_at_k(ranked_labels: Sequence[int], k: int = 10) -> float:
    """Normalized DCG@k for binary relevance (at most one positive)."""
    if not any(ranked_labels):
        return 0.0
    dcg = dcg_at_k(ranked_labels, k)
    idcg = 1.0 / math.log2(2.0)  # ideal: single positive at rank 1
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(ranked_labels: Sequence[int]) -> float:
    """Reciprocal rank of the first positive label (0 if none)."""
    for rank, label in enumerate(ranked_labels, start=1):
        if label:
            return 1.0 / rank
    return 0.0


def topk_accuracy(ranked_labels: Sequence[int], k: int) -> float:
    """1.0 if any of the top-k labels is positive, else 0.0."""
    return 1.0 if any(ranked_labels[:k]) else 0.0


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_condition_model(
    train_rows: Sequence[Dict[str, object]],
    val_rows: Sequence[Dict[str, object]],
    featurizer: ReactantFeaturizer,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    device_name: str | None = None,
) -> Dict[str, object]:
    """Train a ConditionMLP and return a payload with the trained model
    and a featurizer reference for downstream evaluation.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)

    x_train, y_train, _ = featurize_rows(train_rows, featurizer)
    x_val, y_val, _ = featurize_rows(val_rows, featurizer)
    if len(x_train) == 0:
        raise RuntimeError("No training rows survived featurization")

    num_classes = len(CONDITION_CLASSES)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = _build_mlp(x_train.shape[1], hidden_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    best_val_acc = -1.0
    best_state: Dict[str, object] | None = None
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        if len(x_val):
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(x_val, dtype=torch.float32, device=device))
                preds = logits.argmax(dim=1).cpu().numpy()
            val_acc = float(np.mean(preds == y_val))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {
        "model": model,
        "featurizer": featurizer,
        "device": device,
        "best_val_acc": best_val_acc if best_val_acc >= 0 else None,
        "train_count": len(x_train),
        "val_count": len(x_val),
    }


def evaluate_condition_model(
    trained: Dict[str, object],
    eval_rows: Sequence[Dict[str, object]],
    batch_size: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """Evaluate a trained model on ``eval_rows``.

    Returns ``(aggregate_metrics, per_row_records)``.
    """
    import torch

    featurizer: ReactantFeaturizer = trained["featurizer"]  # type: ignore[assignment]
    model = trained["model"]
    device = trained["device"]

    x_eval, y_eval, kept = featurize_rows(eval_rows, featurizer)
    if len(x_eval) == 0:
        return (
            {"n_samples": 0, "top1": 0.0, "top3": 0.0, "top5": 0.0, "top10": 0.0,
             "mrr": 0.0, "ndcg_at_10": 0.0},
            [],
        )

    model.eval()
    with torch.no_grad():
        logits_list: List[torch.Tensor] = []
        for start in range(0, len(x_eval), batch_size):
            chunk = torch.tensor(
                x_eval[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            logits_list.append(model(chunk))
        logits = torch.cat(logits_list, dim=0)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    y_true = np.asarray(y_eval, dtype=np.int64)
    metrics = compute_classification_metrics(y_true, probs)

    per_row: List[Dict[str, object]] = []
    order = np.argsort(-probs, axis=1)
    for i, row in enumerate(kept):
        ranked = order[i].tolist()
        true_idx = int(y_true[i])
        ranked_labels = [1 if int(c) == true_idx else 0 for c in ranked]
        per_row.append({
            "source_id": row.get("source_id", ""),
            "true_label": IDX_TO_CLASS_NAME[true_idx],
            "true_idx": true_idx,
            "top1_hit": int(ranked_labels[0]),
            "top3_hit": int(any(ranked_labels[:3])),
            "top5_hit": int(any(ranked_labels[:5])),
            "top10_hit": int(any(ranked_labels[:10])),
            "reciprocal_rank": reciprocal_rank(ranked_labels),
            "ndcg_at_10": ndcg_at_k(ranked_labels, k=10),
        })
    return metrics, per_row


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------

def run_single_seed(
    seed: int,
    train_rows: Sequence[Dict[str, object]],
    val_rows: Sequence[Dict[str, object]],
    test_rows: Sequence[Dict[str, object]],
    pc_cng_rows: Sequence[Dict[str, object]],
    n_bits: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    device_name: str | None,
) -> Dict[str, object]:
    """Train baseline + treatment for one seed, evaluate on test."""
    featurizer = ReactantFeaturizer(n_bits=n_bits)

    # Baseline: real positives only.
    baseline_trained = train_condition_model(
        train_rows=train_rows,
        val_rows=val_rows,
        featurizer=featurizer,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        device_name=device_name,
    )
    baseline_metrics, baseline_per_row = evaluate_condition_model(
        baseline_trained, test_rows, batch_size,
    )

    # Treatment: real positives + PC-CNG negatives (inheriting parent's label).
    treatment_train = build_treatment_rows(train_rows, pc_cng_rows)
    treatment_trained = train_condition_model(
        train_rows=treatment_train,
        val_rows=val_rows,
        featurizer=featurizer,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        device_name=device_name,
    )
    treatment_metrics, treatment_per_row = evaluate_condition_model(
        treatment_trained, test_rows, batch_size,
    )

    return {
        "seed": seed,
        "baseline_metrics": baseline_metrics,
        "treatment_metrics": treatment_metrics,
        "baseline_per_row": baseline_per_row,
        "treatment_per_row": treatment_per_row,
        "baseline_train_count": baseline_trained["train_count"],
        "treatment_train_count": treatment_trained["train_count"],
        "best_val_acc_baseline": baseline_trained["best_val_acc"],
        "best_val_acc_treatment": treatment_trained["best_val_acc"],
    }


# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------

def paired_ttest(
    baseline_vals: Sequence[float],
    treatment_vals: Sequence[float],
) -> Dict[str, object]:
    """Two-sided paired t-test on aligned per-seed values.

    Falls back to a degenerate result if ``scipy`` is unavailable or if
    the input has fewer than 2 paired samples.
    """
    n = min(len(baseline_vals), len(treatment_vals))
    if n < 2:
        return {
            "n": n,
            "t_stat": float("nan"),
            "p_value": 1.0,
            "mean_delta": 0.0,
            "std_delta": 0.0,
            "df": n - 1,
        }
    b = np.asarray(baseline_vals[:n], dtype=np.float64)
    t = np.asarray(treatment_vals[:n], dtype=np.float64)
    deltas = t - b
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
    # Degenerate case: zero variance in deltas (e.g. identical inputs).
    # scipy.stats.ttest_rel returns NaN here; we surface a deterministic
    # p-value instead (1.0 if no effect, 0.0 if a constant non-zero effect).
    if std_delta == 0.0:
        t_stat = float("inf") if mean_delta > 0 else (
            float("-inf") if mean_delta < 0 else float("nan")
        )
        p_value = 0.0 if mean_delta != 0.0 else 1.0
        return {
            "n": int(n),
            "t_stat": t_stat,
            "p_value": p_value,
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "df": int(n - 1),
        }
    try:
        from scipy import stats
        result = stats.ttest_rel(t, b)
        t_stat = float(result.statistic)
        p_value = float(result.pvalue)
        # scipy can still return NaN in pathological cases - fall back.
        if not (p_value == p_value):  # NaN check
            raise ValueError("NaN p-value from scipy")
    except Exception:
        # Manual fallback: paired t-statistic.
        se = std_delta / math.sqrt(n) if n > 1 else 0.0
        t_stat = float(mean_delta / se) if se > 0 else float("nan")
        try:
            p_value = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0)))))
        except Exception:
            p_value = 1.0
    return {
        "n": int(n),
        "t_stat": t_stat,
        "p_value": p_value,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "df": int(n - 1),
    }


def aggregate_seed_metrics(
    seed_results: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    """Return ``{metric: {mean, std, baseline_vals, treatment_vals, deltas}}``
    for each metric in Top-1/3/5/10, MRR, NDCG@10.
    """
    metric_keys = ["top1", "top3", "top5", "top10", "mrr", "ndcg_at_10"]
    out: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        b_vals = [float(r["baseline_metrics"].get(key, 0.0)) for r in seed_results]
        t_vals = [float(r["treatment_metrics"].get(key, 0.0)) for r in seed_results]
        deltas = [t - b for t, b in zip(t_vals, b_vals)]
        out[key] = {
            "baseline_mean": float(np.mean(b_vals)) if b_vals else 0.0,
            "baseline_std": float(np.std(b_vals, ddof=1)) if len(b_vals) > 1 else 0.0,
            "treatment_mean": float(np.mean(t_vals)) if t_vals else 0.0,
            "treatment_std": float(np.std(t_vals, ddof=1)) if len(t_vals) > 1 else 0.0,
            "delta_mean": float(np.mean(deltas)) if deltas else 0.0,
            "delta_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "baseline_vals": b_vals,
            "treatment_vals": t_vals,
            "deltas": deltas,
        }
    return out


def build_paired_significance(
    seed_results: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """Run a paired t-test on each metric across seeds."""
    agg = aggregate_seed_metrics(seed_results)
    sig: Dict[str, object] = {}
    for key, stats in agg.items():
        ttest = paired_ttest(stats["baseline_vals"], stats["treatment_vals"])
        sig[key] = {
            "n_seeds": len(seed_results),
            "baseline_mean": stats["baseline_mean"],
            "treatment_mean": stats["treatment_mean"],
            "delta_mean": stats["delta_mean"],
            "delta_std": stats["delta_std"],
            "delta_mean_pp": stats["delta_mean"] * 100.0,
            "t_stat": ttest["t_stat"],
            "p_value": ttest["p_value"],
            "df": ttest["df"],
            "per_seed_baseline": stats["baseline_vals"],
            "per_seed_treatment": stats["treatment_vals"],
            "per_seed_delta": stats["deltas"],
        }
    return sig


def build_go_no_go_decision(
    significance: Dict[str, object],
    primary_metric: str = "top1",
    delta_threshold_pp: float = 1.0,
    p_threshold: float = 0.05,
) -> Dict[str, object]:
    """Apply the GO / NO-GO decision rule on the primary metric.

    GO requires:
      * ``delta_mean_pp > delta_threshold_pp``
      * ``p_value < p_threshold``
      * treatment mean >= baseline mean
    """
    primary = significance[primary_metric]
    delta_pp = float(primary["delta_mean_pp"])
    p_value = float(primary["p_value"])
    treatment_better = float(primary["treatment_mean"]) >= float(primary["baseline_mean"])
    passes = (
        delta_pp > delta_threshold_pp
        and p_value < p_threshold
        and treatment_better
    )
    decision = "GO (write to main table)" if passes else "NO-GO (downgrade to supplementary)"
    return {
        "decision": decision,
        "primary_metric": primary_metric,
        "delta_mean_pp": delta_pp,
        "p_value": p_value,
        "treatment_better": treatment_better,
        "criteria": {
            "delta_threshold_pp": delta_threshold_pp,
            "p_threshold": p_threshold,
            "delta_passes": delta_pp > delta_threshold_pp,
            "p_passes": p_value < p_threshold,
            "treatment_better_passes": treatment_better,
        },
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_per_seed_csv(path: str, seed_results: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "seed", "condition",
        "baseline_train_count", "treatment_train_count",
        "baseline_top1", "treatment_top1", "delta_top1",
        "baseline_top3", "treatment_top3", "delta_top3",
        "baseline_top5", "treatment_top5", "delta_top5",
        "baseline_top10", "treatment_top10", "delta_top10",
        "baseline_mrr", "treatment_mrr", "delta_mrr",
        "baseline_ndcg_at_10", "treatment_ndcg_at_10", "delta_ndcg_at_10",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in seed_results:
            b = r["baseline_metrics"]
            t = r["treatment_metrics"]
            writer.writerow({
                "seed": r["seed"],
                "condition": "all",
                "baseline_train_count": r["baseline_train_count"],
                "treatment_train_count": r["treatment_train_count"],
                "baseline_top1": b.get("top1", 0.0),
                "treatment_top1": t.get("top1", 0.0),
                "delta_top1": t.get("top1", 0.0) - b.get("top1", 0.0),
                "baseline_top3": b.get("top3", 0.0),
                "treatment_top3": t.get("top3", 0.0),
                "delta_top3": t.get("top3", 0.0) - b.get("top3", 0.0),
                "baseline_top5": b.get("top5", 0.0),
                "treatment_top5": t.get("top5", 0.0),
                "delta_top5": t.get("top5", 0.0) - b.get("top5", 0.0),
                "baseline_top10": b.get("top10", 0.0),
                "treatment_top10": t.get("top10", 0.0),
                "delta_top10": t.get("top10", 0.0) - b.get("top10", 0.0),
                "baseline_mrr": b.get("mrr", 0.0),
                "treatment_mrr": t.get("mrr", 0.0),
                "delta_mrr": t.get("mrr", 0.0) - b.get("mrr", 0.0),
                "baseline_ndcg_at_10": b.get("ndcg_at_10", 0.0),
                "treatment_ndcg_at_10": t.get("ndcg_at_10", 0.0),
                "delta_ndcg_at_10": t.get("ndcg_at_10", 0.0) - b.get("ndcg_at_10", 0.0),
            })


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="P2-08 Reaction Condition Prediction downstream evaluation",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="USPTO condition dataset CSV (normalized USPTO OpenMolecules CSV "
             "with columns source_id, reactants, products, split, ...).",
    )
    parser.add_argument(
        "--pc-cng-negatives", required=True,
        help="PC-CNG synthetic negatives CSV (reviewed).",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds (default: 20260710..20260719).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of USPTO rows used (smoke test).",
    )
    parser.add_argument(
        "--n-bits", type=int, default=2048,
        help="Morgan fingerprint bit size.",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=512,
        help="Hidden layer size for the MLP.",
    )
    parser.add_argument(
        "--device", default=None,
        help="torch device override (e.g. cuda:0, cpu).",
    )
    parser.add_argument(
        "--condition-cache", default=None,
        help="Optional path to a cached synthetic-condition CSV. If absent, "
             "one is generated next to --output-dir.",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    os.makedirs(args.output_dir, exist_ok=True)

    # Suppress RDKit logging flood.
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Step 1: Build / load the synthetic condition dataset
    # ------------------------------------------------------------------
    if args.condition_cache and os.path.exists(args.condition_cache):
        condition_csv = args.condition_cache
        print(f"Reusing cached condition dataset: {condition_csv}")
    else:
        condition_csv = os.path.join(args.output_dir, "synthetic_condition_dataset.csv")
        print(f"Generating synthetic condition dataset from {args.dataset}")
        gen_stats = generate_synthetic_condition_dataset(
            source_csv=args.dataset,
            output_csv=condition_csv,
            limit=args.limit,
        )
        _write_json(os.path.join(args.output_dir, "dataset_generation_summary.json"), gen_stats)
        print(
            f"Wrote {gen_stats['rows_written']} rows; class distribution: "
            f"{gen_stats['class_counts']}"
        )

    all_rows = read_condition_dataset(condition_csv)
    if not all_rows:
        raise RuntimeError(f"No rows loaded from {condition_csv}")

    # Use USPTO split column if present; otherwise fall back to a 70/15/15
    # random split keyed on the first seed (deterministic across seeds).
    splits: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    for row in all_rows:
        split = str(row.get("split", "")).strip().lower()
        if split in splits:
            splits[split].append(row)
        else:
            splits["train"].append(row)
    if not splits["train"] and not splits["val"] and not splits["test"]:
        raise RuntimeError("All splits empty after read")
    if not splits["test"]:
        # Fall back: re-split deterministically.
        if not splits["val"]:
            rng = random.Random(20260710)
            rng.shuffle(all_rows)
            n_total = len(all_rows)
            n_train = max(1, int(n_total * 0.7))
            n_val = max(1, int(n_total * 0.15))
            splits["train"] = all_rows[:n_train]
            splits["val"] = all_rows[n_train:n_train + n_val]
            splits["test"] = all_rows[n_train + n_val:]
        else:
            # Treat val as test if no explicit test split.
            splits["test"] = splits["val"]
            splits["val"] = []

    print(
        f"Loaded {len(all_rows)} rows | train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Step 2: Load PC-CNG negatives
    # ------------------------------------------------------------------
    pc_cng_rows = read_pc_cng_negatives(args.pc_cng_negatives)
    parent_ids_in_train = {str(r["source_id"]) for r in splits["train"]}
    pc_cng_in_train = [r for r in pc_cng_rows if str(r["source_id"]) in parent_ids_in_train]
    print(
        f"Loaded {len(pc_cng_rows)} PC-CNG negatives "
        f"({len(pc_cng_in_train)} match train parent IDs)",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Step 3: Run 10 seeds
    # ------------------------------------------------------------------
    seed_results: List[Dict[str, object]] = []
    for seed in seeds:
        t0 = time.time()
        print(f"\n=== Seed {seed} ===", flush=True)
        result = run_single_seed(
            seed=seed,
            train_rows=splits["train"],
            val_rows=splits["val"],
            test_rows=splits["test"],
            pc_cng_rows=pc_cng_in_train,
            n_bits=args.n_bits,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            device_name=args.device,
        )
        elapsed = time.time() - t0
        b_top1 = result["baseline_metrics"].get("top1", 0.0)
        t_top1 = result["treatment_metrics"].get("top1", 0.0)
        print(
            f"  baseline top1={b_top1:.4f}  treatment top1={t_top1:.4f}  "
            f"delta={(t_top1 - b_top1) * 100:.2f}pp  ({elapsed:.1f}s)",
            flush=True,
        )
        seed_results.append(result)

    # ------------------------------------------------------------------
    # Step 4: Aggregate + significance + decision
    # ------------------------------------------------------------------
    significance = build_paired_significance(seed_results)
    decision = build_go_no_go_decision(significance, primary_metric="top1")

    summary = {
        "task": "P2-08 Reaction Condition Prediction downstream (L8)",
        "dataset_tag": DATASET_TAG,
        "fallback_reason": FALLBACK_REASON,
        "dataset_csv": args.dataset,
        "condition_dataset_csv": condition_csv,
        "pc_cng_negatives_csv": args.pc_cng_negatives,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "n_bits": args.n_bits,
            "hidden_dim": args.hidden_dim,
            "limit": args.limit,
            "device": args.device,
        },
        "class_distribution": dict(
            defaultdict(int, {k: 0 for k, _ in CONDITION_CLASSES}),
        ),
        "splits": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "pc_cng_negatives": {
            "total_loaded": len(pc_cng_rows),
            "matching_train_parents": len(pc_cng_in_train),
        },
        "per_seed": [
            {
                "seed": r["seed"],
                "baseline_metrics": r["baseline_metrics"],
                "treatment_metrics": r["treatment_metrics"],
                "baseline_train_count": r["baseline_train_count"],
                "treatment_train_count": r["treatment_train_count"],
                "best_val_acc_baseline": r["best_val_acc_baseline"],
                "best_val_acc_treatment": r["best_val_acc_treatment"],
            }
            for r in seed_results
        ],
        "aggregate": aggregate_seed_metrics(seed_results),
    }

    # Populate class distribution from the actual loaded rows.
    class_dist: Dict[str, int] = defaultdict(int)
    for row in all_rows:
        class_dist[str(row["condition_label"])] += 1
    summary["class_distribution"] = dict(class_dist)

    _write_json(os.path.join(args.output_dir, "summary.json"), summary)
    _write_json(os.path.join(args.output_dir, "paired_significance.json"), significance)
    _write_json(os.path.join(args.output_dir, "go_no_go_decision.json"), decision)
    _write_per_seed_csv(os.path.join(args.output_dir, "per_seed_metrics.csv"), seed_results)

    # ------------------------------------------------------------------
    # Step 5: Print final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("P2-08 Reaction Condition Prediction — Summary")
    print("=" * 70)
    print(f"Dataset tag:       {DATASET_TAG}")
    print(f"Seeds:             {len(seeds)}")
    print(f"Train/Val/Test:    {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"PC-CNG in train:   {len(pc_cng_in_train)}")
    for key in ["top1", "top3", "top5", "top10", "mrr", "ndcg_at_10"]:
        s = significance[key]
        print(
            f"  {key:<12} baseline={s['baseline_mean']:.4f}  "
            f"treatment={s['treatment_mean']:.4f}  "
            f"Δ={s['delta_mean_pp']:+.2f}pp  p={s['p_value']:.4g}"
        )
    print(f"Go/No-Go:          {decision['decision']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
