"""Tiny dependency-free reranker for MVP validation.

This is not the final model. It is a sanity-check learner that verifies whether
PC-CNG negatives can provide a usable training signal for route/reaction
ranking before we scale to GNN/Transformer models.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from typing import Dict, Iterable, List, Sequence, Tuple

from .chem_utils import (
    atom_balance_score,
    atom_count_distance,
    atom_tokens,
    is_valid_reaction,
    molecule_parts,
    split_reaction,
    string_similarity,
    token_jaccard,
)


FEATURE_NAMES = [
    "bias",
    "valid",
    "atom_balance",
    "reactant_product_token_jaccard",
    "reactant_product_string_similarity",
    "one_reactant",
    "multi_reactant",
    "product_reused_as_reactant",
    "has_leaving_group",
    "normalized_atom_distance",
]


def featurize_reaction(reaction_smiles: str) -> List[float]:
    try:
        reactants, _, products = split_reaction(reaction_smiles)
    except ValueError:
        return [1.0] + [0.0] * (len(FEATURE_NAMES) - 1)

    reactant_parts = molecule_parts(reactants)
    product_parts = molecule_parts(products)
    reactant_set = set(reactant_parts)
    product_set = set(product_parts)
    total_atoms = max(len(atom_tokens(reactants)) + len(atom_tokens(products)), 1)
    atom_distance = atom_count_distance(reactants, products) / total_atoms

    return [
        1.0,
        1.0 if is_valid_reaction(reaction_smiles) else 0.0,
        atom_balance_score(reactants, products),
        token_jaccard(reactants, products),
        string_similarity(reactants, products),
        1.0 if len(reactant_parts) == 1 else 0.0,
        1.0 if len(reactant_parts) > 1 else 0.0,
        1.0 if reactant_set & product_set else 0.0,
        1.0 if ("Cl" in reactants or "Br" in reactants) else 0.0,
        max(0.0, min(1.0, atom_distance)),
    ]


@dataclass
class RankerMetrics:
    accuracy: float
    top1: float
    mrr: float
    ndcg: float
    groups: int

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "accuracy": self.accuracy,
            "top1": self.top1,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "groups": self.groups,
        }


class LogisticReactionRanker:
    """Minimal logistic model trained by SGD."""

    def __init__(self, learning_rate: float = 0.2, l2: float = 1e-4, epochs: int = 300):
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.weights = [0.0 for _ in FEATURE_NAMES]

    def fit(self, rows: Sequence[Dict[str, str | int | float]]) -> None:
        if not rows:
            return
        for _ in range(self.epochs):
            for row in rows:
                x = featurize_reaction(str(row["reaction_smiles"]))
                y = float(row["label"])
                pred = self.predict_proba_from_features(x)
                error = pred - y
                for i, value in enumerate(x):
                    grad = error * value + self.l2 * self.weights[i]
                    self.weights[i] -= self.learning_rate * grad

    def predict_proba(self, reaction_smiles: str) -> float:
        return self.predict_proba_from_features(featurize_reaction(reaction_smiles))

    def predict_proba_from_features(self, features: Sequence[float]) -> float:
        z = sum(weight * value for weight, value in zip(self.weights, features))
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature_names": FEATURE_NAMES,
            "weights": self.weights,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "epochs": self.epochs,
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


def evaluate_binary(model: LogisticReactionRanker, rows: Sequence[Dict[str, str | int | float]]) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        pred = 1 if model.predict_proba(str(row["reaction_smiles"])) >= 0.5 else 0
        correct += int(pred == int(row["label"]))
    return correct / len(rows)


def evaluate_ranking(model: LogisticReactionRanker, rows: Sequence[Dict[str, str | int | float]]) -> RankerMetrics:
    grouped: Dict[str, List[Dict[str, str | int | float]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_id"])].append(row)

    top1_total = 0.0
    mrr_total = 0.0
    ndcg_total = 0.0
    group_count = 0

    for group_rows in grouped.values():
        if not any(int(row["label"]) == 1 for row in group_rows):
            continue
        ranked = sorted(
            group_rows,
            key=lambda row: model.predict_proba(str(row["reaction_smiles"])),
            reverse=True,
        )
        group_count += 1
        for rank, row in enumerate(ranked, start=1):
            if int(row["label"]) == 1:
                top1_total += 1.0 if rank == 1 else 0.0
                mrr_total += 1.0 / rank
                ndcg_total += 1.0 / math.log2(rank + 1)
                break

    if group_count == 0:
        return RankerMetrics(accuracy=evaluate_binary(model, rows), top1=0.0, mrr=0.0, ndcg=0.0, groups=0)

    return RankerMetrics(
        accuracy=evaluate_binary(model, rows),
        top1=top1_total / group_count,
        mrr=mrr_total / group_count,
        ndcg=ndcg_total / group_count,
        groups=group_count,
    )


def split_by_source(
    rows: Sequence[Dict[str, str | int | float]], train_fraction: float = 0.7
) -> Tuple[List[Dict[str, str | int | float]], List[Dict[str, str | int | float]]]:
    source_ids = sorted({str(row["source_id"]) for row in rows})
    cutoff = max(1, int(len(source_ids) * train_fraction))
    train_ids = set(source_ids[:cutoff])
    train_rows = [row for row in rows if str(row["source_id"]) in train_ids]
    test_rows = [row for row in rows if str(row["source_id"]) not in train_ids]
    if not test_rows:
        return list(rows), list(rows)
    return train_rows, test_rows

