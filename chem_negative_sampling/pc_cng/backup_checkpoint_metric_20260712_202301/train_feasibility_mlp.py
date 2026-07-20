"""Train a full-data reaction feasibility/reranking MLP.

This model is stronger than the lightweight stdlib reranker: it uses RDKit
reaction fingerprints and a PyTorch MLP. Real val/test metrics are reported
only on real labeled reactions. Synthetic PC-CNG negatives are used only for
training and only when their parent positive reaction belongs to the train
split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
except Exception as exc:  # pragma: no cover
    raise RuntimeError("RDKit is required for train_feasibility_mlp.py") from exc

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for train_feasibility_mlp.py") from exc

from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from .chem_utils import atom_tokens, molecule_parts, split_reaction

RDLogger.DisableLog("rdApp.*")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mols_from_smiles(smiles: str):
    mols = []
    for part in molecule_parts(smiles):
        mol = Chem.MolFromSmiles(part)
        if mol is not None:
            mols.append(mol)
    return mols


class ReactionFeaturizer:
    def __init__(
        self,
        n_bits: int = 2048,
        radius: int = 2,
        fp_mode: str = "binary",
        include_descriptors: bool = False,
    ):
        self.n_bits = n_bits
        self.fp_mode = fp_mode
        self.include_descriptors = include_descriptors
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp_width = n_bits * 2 if fp_mode == "binary_count" else n_bits
        self.output_dim = fp_width * 3 + (32 if include_descriptors else 0)

    def molset_fp(self, smiles: str) -> np.ndarray:
        binary_arr = np.zeros((self.n_bits,), dtype=np.float32)
        count_arr = np.zeros((self.n_bits,), dtype=np.float32)
        for mol in mols_from_smiles(smiles):
            bitvect = self.generator.GetFingerprint(mol)
            tmp = np.zeros((self.n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bitvect, tmp)
            binary_arr = np.maximum(binary_arr, tmp.astype(np.float32))

            countvect = self.generator.GetCountFingerprint(mol)
            for bit, value in countvect.GetNonzeroElements().items():
                count_arr[int(bit)] += float(value)

        if self.fp_mode == "binary":
            return binary_arr
        if self.fp_mode == "count":
            return np.log1p(count_arr).astype(np.float32)
        if self.fp_mode == "binary_count":
            return np.concatenate([binary_arr, np.log1p(count_arr).astype(np.float32)])
        raise ValueError(f"Unsupported fp_mode: {self.fp_mode}")

    def molset_descriptors(self, smiles: str) -> np.ndarray:
        values = np.zeros((8,), dtype=np.float32)
        for mol in mols_from_smiles(smiles):
            atoms = list(mol.GetAtoms())
            values += np.array(
                [
                    mol.GetNumHeavyAtoms(),
                    Descriptors.MolWt(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.RingCount(mol),
                    sum(1 for atom in atoms if atom.GetIsAromatic()),
                    sum(1 for atom in atoms if atom.GetAtomicNum() not in {1, 6}),
                    sum(atom.GetFormalCharge() for atom in atoms),
                ],
                dtype=np.float32,
            )
        # Heavy-tailed descriptors are easier for the MLP after mild compression.
        signed = np.sign(values) * np.log1p(np.abs(values))
        return signed.astype(np.float32)

    def reaction_fp(self, reaction_smiles: str) -> np.ndarray | None:
        try:
            reactants, _, products = split_reaction(reaction_smiles)
        except ValueError:
            return None
        reactant_fp = self.molset_fp(reactants)
        product_fp = self.molset_fp(products)
        if reactant_fp.sum() == 0 or product_fp.sum() == 0:
            return None
        xor_fp = np.abs(product_fp - reactant_fp)
        features = [reactant_fp, product_fp, xor_fp]
        if self.include_descriptors:
            reactant_desc = self.molset_descriptors(reactants)
            product_desc = self.molset_descriptors(products)
            features.extend(
                [
                    reactant_desc,
                    product_desc,
                    product_desc - reactant_desc,
                    np.abs(product_desc - reactant_desc),
                ]
            )
        return np.concatenate(features).astype(np.float32)


class GraphStatsReactionFeaturizer:
    """RDKit-only graph-pair encoder for reactant/product comparison.

    This is a lightweight architecture upgrade over raw Morgan fingerprints:
    each side is encoded with interpretable graph statistics, then pooled as
    reactant, product, signed difference, and absolute difference.
    """

    ATOMIC_NUMBERS = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    DEGREE_BINS = [0, 1, 2, 3, 4, 5]
    RING_BINS = [3, 4, 5, 6, 7, 8]

    def __init__(self) -> None:
        side_dim = len(self.ATOMIC_NUMBERS) + 1 + len(self.DEGREE_BINS) + 5 + 4 + len(self.RING_BINS) + 8
        self.output_dim = side_dim * 4 + 4
        self.n_bits = self.output_dim
        self.fp_mode = "graph_stats"
        self.include_descriptors = True

    def molset_graph_stats(self, smiles: str) -> np.ndarray | None:
        values = np.zeros((self.output_dim // 4 - 1,), dtype=np.float32)
        valid_parts = 0
        for mol in mols_from_smiles(smiles):
            valid_parts += 1
            atoms = list(mol.GetAtoms())
            atom_counts = np.zeros((len(self.ATOMIC_NUMBERS) + 1,), dtype=np.float32)
            degree_counts = np.zeros((len(self.DEGREE_BINS),), dtype=np.float32)
            charge_aromatic = np.zeros((5,), dtype=np.float32)
            bond_counts = np.zeros((4,), dtype=np.float32)
            ring_counts = np.zeros((len(self.RING_BINS),), dtype=np.float32)
            descriptors = np.array(
                [
                    mol.GetNumHeavyAtoms(),
                    Descriptors.MolWt(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.RingCount(mol),
                    sum(1 for atom in atoms if atom.GetIsAromatic()),
                    sum(1 for atom in atoms if atom.GetAtomicNum() not in {1, 6}),
                    sum(abs(atom.GetFormalCharge()) for atom in atoms),
                ],
                dtype=np.float32,
            )
            atomic_index = {num: idx for idx, num in enumerate(self.ATOMIC_NUMBERS)}
            degree_index = {degree: idx for idx, degree in enumerate(self.DEGREE_BINS)}
            ring_index = {size: idx for idx, size in enumerate(self.RING_BINS)}
            for atom in atoms:
                atom_counts[atomic_index.get(atom.GetAtomicNum(), len(self.ATOMIC_NUMBERS))] += 1.0
                degree_counts[degree_index.get(min(atom.GetDegree(), 5), len(self.DEGREE_BINS) - 1)] += 1.0
                charge = atom.GetFormalCharge()
                charge_aromatic[0] += max(float(charge), 0.0)
                charge_aromatic[1] += max(float(-charge), 0.0)
                charge_aromatic[2] += 1.0 if atom.GetIsAromatic() else 0.0
                charge_aromatic[3] += 1.0 if atom.IsInRing() else 0.0
                charge_aromatic[4] += float(atom.GetTotalNumHs())
            for bond in mol.GetBonds():
                bond_type = bond.GetBondType()
                if bond_type == Chem.BondType.SINGLE:
                    bond_counts[0] += 1.0
                elif bond_type == Chem.BondType.DOUBLE:
                    bond_counts[1] += 1.0
                elif bond_type == Chem.BondType.TRIPLE:
                    bond_counts[2] += 1.0
                elif bond_type == Chem.BondType.AROMATIC:
                    bond_counts[3] += 1.0
            for ring in mol.GetRingInfo().AtomRings():
                size = len(ring)
                ring_counts[ring_index.get(size, len(self.RING_BINS) - 1)] += 1.0
            values += np.concatenate(
                [
                    atom_counts,
                    degree_counts,
                    charge_aromatic,
                    bond_counts,
                    ring_counts,
                    descriptors,
                ]
            ).astype(np.float32)
        if valid_parts == 0:
            return None
        return (np.sign(values) * np.log1p(np.abs(values))).astype(np.float32)

    def reaction_fp(self, reaction_smiles: str) -> np.ndarray | None:
        try:
            reactants, _, products = split_reaction(reaction_smiles)
        except ValueError:
            return None
        reactant_stats = self.molset_graph_stats(reactants)
        product_stats = self.molset_graph_stats(products)
        if reactant_stats is None or product_stats is None:
            return None
        diff = product_stats - reactant_stats
        flags = np.array(
            [
                1.0,
                float(len(molecule_parts(reactants))),
                float(len(molecule_parts(products))),
                float(len(atom_tokens(products))),
            ],
            dtype=np.float32,
        )
        flags = np.sign(flags) * np.log1p(np.abs(flags))
        return np.concatenate([reactant_stats, product_stats, diff, np.abs(diff), flags]).astype(np.float32)


class CombinedReactionFeaturizer:
    """Combined Morgan + graph-stats reaction featurizer.

    Concatenates Morgan fingerprint features and graph-statistics features to
    capture both local substructure patterns (Morgan) and global structural
    properties (graph stats), with the goal of improving held-out generalization.
    """

    def __init__(
        self,
        n_bits: int = 2048,
        radius: int = 2,
        fp_mode: str = "binary",
        include_descriptors: bool = False,
    ) -> None:
        self.morgan = ReactionFeaturizer(
            n_bits=n_bits, radius=radius, fp_mode=fp_mode, include_descriptors=include_descriptors
        )
        self.graph_stats = GraphStatsReactionFeaturizer()
        self.output_dim = self.morgan.output_dim + self.graph_stats.output_dim
        self.n_bits = n_bits
        self.fp_mode = fp_mode
        self.include_descriptors = include_descriptors

    def reaction_fp(self, reaction_smiles: str) -> np.ndarray | None:
        morgan_fp = self.morgan.reaction_fp(reaction_smiles)
        graph_fp = self.graph_stats.reaction_fp(reaction_smiles)
        if morgan_fp is None or graph_fp is None:
            return None
        return np.concatenate([morgan_fp, graph_fp]).astype(np.float32)


def make_reaction_featurizer(
    feature_mode: str = "morgan",
    n_bits: int = 2048,
    fp_mode: str = "binary",
    include_descriptors: bool = False,
):
    if feature_mode == "morgan":
        return ReactionFeaturizer(n_bits=n_bits, fp_mode=fp_mode, include_descriptors=include_descriptors)
    if feature_mode == "graph_stats":
        return GraphStatsReactionFeaturizer()
    if feature_mode == "combined":
        return CombinedReactionFeaturizer(
            n_bits=n_bits, fp_mode=fp_mode, include_descriptors=include_descriptors
        )
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def parse_key_value(items: Sequence[str]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        values[key] = float(value)
    return values


def parse_key_int(items: Sequence[str]) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        values[key] = int(value)
    return values


def read_real_rows(path: str) -> Tuple[List[Dict[str, object]], Dict[str, str]]:
    rows: List[Dict[str, object]] = []
    source_split: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_type = row.get("label_type", "")
            if label_type not in {"positive", "real_negative"}:
                continue
            label = 1 if label_type == "positive" else 0
            split = row.get("split") or "train"
            source_id = row.get("source_id") or ""
            reaction = row.get("reaction_smiles") or ""
            if not source_id or not reaction:
                continue
            source_split[source_id] = split
            rows.append(
                {
                    "source_id": source_id,
                    "reaction_smiles": reaction,
                    "label": label,
                    "split": split,
                    "dataset": row.get("source", os.path.basename(path)),
                    "reaction_class": row.get("reaction_class", ""),
                    "origin": "real",
                    "sample_weight": 1.0,
                }
            )
    return rows, source_split


def read_synthetic_rows(
    path: str,
    source_split: Dict[str, str],
    max_rows: int | None = None,
    allowed_families: set[str] | None = None,
    excluded_families: set[str] | None = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path or not os.path.exists(path):
        return rows
    allowed_families = allowed_families or set()
    excluded_families = excluded_families or set()
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if max_rows is not None and len(rows) >= max_rows:
                break
            source_id = row.get("source_id") or ""
            if source_split.get(source_id) != "train":
                continue
            if row.get("review_status", "keep_synthetic_negative") != "keep_synthetic_negative":
                continue
            reaction = row.get("candidate_reaction") or row.get("reaction_smiles") or ""
            if not reaction:
                continue
            action_family = row.get("action_family", "")
            if allowed_families and action_family not in allowed_families:
                continue
            if excluded_families and action_family in excluded_families:
                continue
            rows.append(
                {
                    "source_id": source_id,
                    "reaction_smiles": reaction,
                    "label": 0,
                    "split": "train",
                    "dataset": row.get("provenance", "pc_cng_synthetic"),
                    "reaction_class": row.get("reaction_class", "") or "synthetic",
                    "origin": "synthetic",
                    "sample_weight": 1.0,
                    "action_family": action_family,
                    "failure_type": row.get("failure_type", ""),
                }
            )
    return rows


def subsample_train_rows(
    rows: Sequence[Dict[str, object]],
    max_per_dataset: Dict[str, int],
    seed: int,
) -> List[Dict[str, object]]:
    if not max_per_dataset:
        return list(rows)
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, object]]] = {}
    passthrough: List[Dict[str, object]] = []
    for row in rows:
        dataset = str(row.get("dataset", ""))
        limit = max_per_dataset.get(dataset)
        if limit is None:
            passthrough.append(row)
        else:
            grouped.setdefault(dataset, []).append(row)
    out = list(passthrough)
    for dataset, dataset_rows in grouped.items():
        limit = max_per_dataset[dataset]
        if len(dataset_rows) > limit:
            out.extend(rng.sample(dataset_rows, limit))
        else:
            out.extend(dataset_rows)
    rng.shuffle(out)
    return out


def apply_sample_weights(
    rows: Sequence[Dict[str, object]],
    dataset_weights: Dict[str, float],
    origin_weights: Dict[str, float],
) -> None:
    for row in rows:
        dataset = str(row.get("dataset", ""))
        origin = str(row.get("origin", ""))
        weight = float(row.get("sample_weight", 1.0))
        weight *= dataset_weights.get(dataset, 1.0)
        weight *= origin_weights.get(origin, 1.0)
        row["sample_weight"] = weight


def featurize_rows(
    rows: Sequence[Dict[str, object]],
    featurizer: ReactionFeaturizer,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
    features = []
    labels = []
    weights = []
    kept = []
    for row in rows:
        fp = featurizer.reaction_fp(str(row["reaction_smiles"]))
        if fp is None:
            continue
        features.append(fp)
        labels.append(float(row["label"]))
        weights.append(float(row.get("sample_weight", 1.0)))
        kept.append(row)
    if not features:
        output_dim = int(getattr(featurizer, "output_dim", getattr(featurizer, "n_bits", 0) * 3))
        return (
            np.zeros((0, output_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            kept,
        )
    return np.stack(features), np.array(labels, dtype=np.float32), np.array(weights, dtype=np.float32), kept


class FeasibilityMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 1024, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(np.mean(y_true)),
        "pred_positive_rate": float(np.mean(y_pred)),
        "n": int(len(y_true)),
    }
    if len(set(y_true.tolist())) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def compute_group_metrics(
    rows: Sequence[Dict[str, object]],
    y_true: np.ndarray,
    y_score: np.ndarray,
    field: str,
    min_count: int = 20,
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        key = str(row.get(field, "") or "unknown")
        grouped.setdefault(key, []).append(idx)

    out: Dict[str, Dict[str, float]] = {}
    for key, indices in sorted(grouped.items()):
        if len(indices) < min_count:
            continue
        idx = np.array(indices, dtype=np.int64)
        out[key] = compute_metrics(y_true[idx], y_score[idx])
    return out


def save_predictions(path: str, rows: Sequence[Dict[str, object]], y_true: np.ndarray, y_score: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_id", "dataset", "reaction_class", "label", "score", "reaction_smiles"],
        )
        writer.writeheader()
        for row, label, score in zip(rows, y_true.tolist(), y_score.tolist()):
            writer.writerow(
                {
                    "source_id": row.get("source_id", ""),
                    "dataset": row.get("dataset", ""),
                    "reaction_class": row.get("reaction_class", ""),
                    "label": int(label),
                    "score": f"{float(score):.8f}",
                    "reaction_smiles": row.get("reaction_smiles", ""),
                }
            )


def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            scores.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", action="append", required=True)
    parser.add_argument("--synthetic-csv", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--fp-mode", choices=["binary", "count", "binary_count"], default="binary")
    parser.add_argument("--include-descriptors", action="store_true")
    parser.add_argument("--max-synthetic", type=int, default=None)
    parser.add_argument(
        "--synthetic-family",
        action="append",
        default=[],
        help="Only keep reviewed synthetic rows from this action_family. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-synthetic-family",
        action="append",
        default=[],
        help="Exclude reviewed synthetic rows from this action_family. Can be repeated.",
    )
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument(
        "--dataset-weight",
        action="append",
        default=[],
        help="Training weight override as dataset=weight, e.g. uspto_openmolecules_yield25to150=0.05",
    )
    parser.add_argument(
        "--origin-weight",
        action="append",
        default=[],
        help="Training weight override as origin=weight, e.g. synthetic=0.2",
    )
    parser.add_argument(
        "--max-train-per-dataset",
        action="append",
        default=[],
        help="Optional train subsampling as dataset=max_rows.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_weights = parse_key_value(args.dataset_weight)
    origin_weights = parse_key_value(args.origin_weight)
    max_train_per_dataset = parse_key_int(args.max_train_per_dataset)
    synthetic_families = set(args.synthetic_family)
    excluded_synthetic_families = set(args.exclude_synthetic_family)
    os.makedirs(args.output_dir, exist_ok=True)
    real_rows: List[Dict[str, object]] = []
    source_split: Dict[str, str] = {}
    for path in args.real_csv:
        rows, split_map = read_real_rows(path)
        real_rows.extend(rows)
        source_split.update(split_map)

    synthetic_rows: List[Dict[str, object]] = []
    remaining = args.max_synthetic
    for path in args.synthetic_csv:
        rows = read_synthetic_rows(
            path,
            source_split,
            remaining,
            allowed_families=synthetic_families,
            excluded_families=excluded_synthetic_families,
        )
        synthetic_rows.extend(rows)
        if remaining is not None:
            remaining = max(0, remaining - len(rows))
            if remaining == 0:
                break

    train_rows = [row for row in real_rows if row["split"] == "train"] + synthetic_rows
    apply_sample_weights(train_rows, dataset_weights=dataset_weights, origin_weights=origin_weights)
    train_rows = subsample_train_rows(train_rows, max_per_dataset=max_train_per_dataset, seed=args.seed)
    val_rows = [row for row in real_rows if row["split"] == "val"]
    test_rows = [row for row in real_rows if row["split"] == "test"]

    featurizer = ReactionFeaturizer(
        n_bits=args.n_bits,
        fp_mode=args.fp_mode,
        include_descriptors=args.include_descriptors,
    )
    x_train, y_train, w_train, train_kept = featurize_rows(train_rows, featurizer)
    x_val, y_val, _, val_kept = featurize_rows(val_rows, featurizer)
    x_test, y_test, _, test_kept = featurize_rows(test_rows, featurizer)
    if len(x_train) == 0:
        raise RuntimeError("No train rows survived RDKit featurization")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeasibilityMLP(in_dim=x_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    neg = max(float(w_train[y_train == 0].sum()), 1.0)
    pos = max(float(w_train[y_train == 1].sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device), reduction="none")
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(w_train, dtype=torch.float32),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    history = []
    best_val = -1.0
    best_path = os.path.join(args.output_dir, "best_feasibility_mlp.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for batch_x, batch_y, batch_w in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad()
            per_row_loss = criterion(model(batch_x), batch_y)
            loss = (per_row_loss * batch_w).sum() / torch.clamp(batch_w.sum(), min=1.0)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            total += len(batch_y)
        val_scores = predict(model, x_val, device, args.batch_size)
        val_metrics = compute_metrics(y_val, val_scores) if len(y_val) else {}
        val_key = val_metrics.get("roc_auc", val_metrics.get("accuracy", 0.0))
        record = {"epoch": epoch, "loss": total_loss / max(total, 1), "val": val_metrics}
        history.append(record)
        if val_key == val_key and val_key > best_val:
            best_val = float(val_key)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "n_bits": args.n_bits,
                    "fp_mode": args.fp_mode,
                    "include_descriptors": args.include_descriptors,
                    "hidden_dim": args.hidden_dim,
                    "input_dim": x_train.shape[1],
                    "epoch": epoch,
                    "best_val": best_val,
                },
                best_path,
            )

    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    val_scores = predict(model, x_val, device, args.batch_size) if len(y_val) else np.zeros((0,), dtype=np.float32)
    test_scores = predict(model, x_test, device, args.batch_size) if len(y_test) else np.zeros((0,), dtype=np.float32)
    if len(y_val):
        save_predictions(os.path.join(args.output_dir, "val_predictions.csv"), val_kept, y_val, val_scores)
    if len(y_test):
        save_predictions(os.path.join(args.output_dir, "test_predictions.csv"), test_kept, y_test, test_scores)

    metrics = {
        "config": {
            "real_csv": args.real_csv,
            "synthetic_csv": args.synthetic_csv,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "n_bits": args.n_bits,
            "fp_mode": args.fp_mode,
            "include_descriptors": args.include_descriptors,
            "max_synthetic": args.max_synthetic,
            "synthetic_family": sorted(synthetic_families),
            "exclude_synthetic_family": sorted(excluded_synthetic_families),
            "seed": args.seed,
            "dataset_weight": dataset_weights,
            "origin_weight": origin_weights,
            "max_train_per_dataset": max_train_per_dataset,
        },
        "device": str(device),
        "counts": {
            "real_rows": len(real_rows),
            "synthetic_rows": len(synthetic_rows),
            "train_rows_requested": len(train_rows),
            "train_rows_featurized": len(train_kept),
            "val_rows_featurized": len(val_kept),
            "test_rows_featurized": len(test_kept),
            "train_positive": int((y_train == 1).sum()),
            "train_negative": int((y_train == 0).sum()),
            "weighted_train_positive": float(w_train[y_train == 1].sum()),
            "weighted_train_negative": float(w_train[y_train == 0].sum()),
            "synthetic_family_counts": {
                family: sum(1 for row in synthetic_rows if str(row.get("action_family", "")) == family)
                for family in sorted({str(row.get("action_family", "")) for row in synthetic_rows})
            },
        },
        "val": compute_metrics(y_val, val_scores) if len(y_val) else {},
        "test": compute_metrics(y_test, test_scores) if len(y_test) else {},
        "val_by_dataset": compute_group_metrics(val_kept, y_val, val_scores, "dataset") if len(y_val) else {},
        "test_by_dataset": compute_group_metrics(test_kept, y_test, test_scores, "dataset") if len(y_test) else {},
        "test_by_reaction_class": compute_group_metrics(test_kept, y_test, test_scores, "reaction_class") if len(y_test) else {},
        "history": history,
        "best_checkpoint": best_path,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps({k: metrics[k] for k in ["device", "counts", "val", "test", "best_checkpoint"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
