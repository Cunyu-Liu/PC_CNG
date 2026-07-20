"""Failure prototype calibrator (P1-06).

Implements a prototype-network based failure direction learner. Each real
negative sample is auto-labeled with one of 10 chemistry-grounded failure
types and used to learn a prototype vector per failure type. At inference,
synthetic negatives can be steered toward a target failure type by optimizing
their embedding toward the corresponding prototype.

Design:
  * Feature encoder: MLP (input_dim -> hidden -> embedding_dim)
  * Prototypes: one per failure type (num_failure_types x embedding_dim)
  * Classification: softmax(-||z - p_k||^2 / temperature)
  * Loss: cross-entropy + lambda * triplet(anch=pos proto, pos=same proto, neg=other proto)
  * Control: gradient of ||z - p_target||^2 w.r.t. input features

The feature vector comes from ``pc_cng.reranker.featurize_reaction`` so the
calibrator stays consistent with the rest of the PC-CNG pipeline.
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .chem_utils import (
    atom_balance_score,
    canonicalize_smiles,
    is_valid_smiles,
    molecule_parts,
    split_reaction,
    string_similarity,
    token_jaccard,
)
from .reranker import FEATURE_NAMES as RERANKER_FEATURE_NAMES


FAILURE_TYPES: List[str] = [
    "wrong_anchor",
    "broken_atom_balance",
    "invalid_valence",
    "fragment_misalignment",
    "wrong_bond_type",
    "aromaticity_violation",
    "stereochemistry_loss",
    "over_reaction",
    "under_reaction",
    "side_product",
]

FAILURE_TYPE_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(FAILURE_TYPES)}
NUM_FAILURE_TYPES: int = len(FAILURE_TYPES)

# Atom-balance threshold below which a reaction is flagged as unbalanced.
ATOM_BALANCE_THRESHOLD: float = 0.95
# Bond-change thresholds for over/under reaction detection.
OVER_REACTION_BOND_THRESHOLD: int = 2
UNIFORM_ENTROPY: float = math.log(NUM_FAILURE_TYPES)  # ln(10) ~ 2.303

try:  # pragma: no cover - environment dependent
    from rdkit import Chem, RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore


def _safe_mol(smiles: str):
    if Chem is None or not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _stereo_center_count(mol) -> int:
    """Count stereo centers in an RDKit molecule (0 if unavailable)."""
    if mol is None:
        return 0
    try:
        from rdkit.Chem import FindPotentialStereoCasters  # type: ignore

        info = FindPotentialStereoCasters(mol)
        return len(info)
    except Exception:
        try:
            from rdkit.Chem.FindMolChiralCenters import FindMolChiralCenters  # type: ignore

            centers = FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
            return len(centers)
        except Exception:
            return 0


def _aromatic_atom_count(mol) -> int:
    if mol is None:
        return 0
    try:
        return sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    except Exception:
        return 0


def _bond_change_count(product_smiles: str, expected_smiles: str) -> int:
    """Approximate number of changed bonds between two molecules.

    Falls back to a token-distance heuristic when RDKit is unavailable.
    """
    prod_mol = _safe_mol(product_smiles)
    exp_mol = _safe_mol(expected_smiles)
    if prod_mol is None or exp_mol is None:
        # Conservative heuristic: Levenshtein-derived difference.
        return max(0, len(product_smiles) - len(expected_smiles)) if product_smiles != expected_smiles else 0

    try:
        from rdkit.Chem import rdFMCS  # type: ignore

        mcs = rdFMCS.FindMCS([prod_mol, exp_mol], timeout=2)
        if mcs.smartsString == "" or mcs.numBonds is None:
            return max(prod_mol.GetNumBonds(), exp_mol.GetNumBonds())
        common = int(mcs.numBonds)
        return max(prod_mol.GetNumBonds(), exp_mol.GetNumBonds()) - common
    except Exception:
        return max(0, prod_mol.GetNumBonds() - exp_mol.GetNumBonds())


def label_failure_type(reaction_smiles: str, expected_outcomes: Optional[Sequence[str]] = None) -> str:
    """Automatically label a negative sample with its failure type.

    The labeling is intentionally heuristic and conservative. RDKit operations
    are wrapped so that failures default to ``invalid_valence``.

    Args:
        reaction_smiles: The (negative) reaction SMILES to label.
        expected_outcomes: Optional list of expected product SMILES (e.g. the
            positive counterpart). When provided, the candidate product is
            compared against these to detect wrong-anchor / side-product cases.

    Returns:
        One of ``FAILURE_TYPES``.
    """
    if not reaction_smiles:
        return "invalid_valence"

    try:
        reactants, _agents, products = split_reaction(reaction_smiles)
    except ValueError:
        return "invalid_valence"

    # 1. Validity / valence check.
    prod_mol = _safe_mol(products)
    if prod_mol is None:
        return "invalid_valence"
    if Chem is not None:
        try:
            Chem.SanitizeMol(prod_mol)
        except Exception:
            return "invalid_valence"

    reactant_parts = molecule_parts(reactants)
    product_parts = molecule_parts(products)

    # 3. Under-reaction: product set equals reactant set (no edit happened).
    if reactant_parts and product_parts and set(reactant_parts) == set(product_parts):
        return "under_reaction"

    expected = [e for e in (expected_outcomes or []) if e]

    if not expected:
        # Without a reference outcome: only the atom-balance signal is
        # informative. Default to the generic side_product bucket otherwise.
        balance = atom_balance_score(reactants, products)
        if balance < ATOM_BALANCE_THRESHOLD:
            return "broken_atom_balance"
        return "side_product"

    # Canonicalize expected outcomes once.
    expected_canon = []
    for exp in expected:
        canon = canonicalize_smiles(exp) or exp
        expected_canon.append(canon)

    prod_canon = canonicalize_smiles(products) or products

    # 4. If product matches an expected outcome exactly, it is NOT a structural
    #    failure of the product itself (e.g. a real low-yield reaction whose
    #    product is correct). Skip the regio-isomer (wrong_anchor) check and
    #    fall through to the atom-balance / other checks.
    is_exact_match = prod_canon in expected_canon

    # 5. Wrong anchor: same molecular formula as an expected outcome but a
    #    different structure (regio-isomer / stereoisomer). This is checked
    #    BEFORE atom-balance because reaction SMILES routinely omit byproducts
    #    (e.g. succinimide), so atom_balance is low for both candidate and
    #    expected. The formula match is the cleaner discriminator.
    if not is_exact_match:
        prod_formula = ""
        exp_formulas = []
        if Chem is not None:
            try:
                from rdkit.Chem import rdMolDescriptors  # type: ignore

                if prod_mol is not None:
                    prod_formula = rdMolDescriptors.CalcMolFormula(prod_mol)
                for exp in expected_canon:
                    exp_mol = _safe_mol(exp)
                    if exp_mol is not None:
                        exp_formulas.append(rdMolDescriptors.CalcMolFormula(exp_mol))
            except Exception:
                pass
        if prod_formula and exp_formulas and prod_formula in exp_formulas:
            return "wrong_anchor"

    # 6. Atom balance (now that the regio-isomer case has been handled).
    balance = atom_balance_score(reactants, products)
    if balance < ATOM_BALANCE_THRESHOLD:
        return "broken_atom_balance"

    # 7. Fragment misalignment: very different atom count -> fragments differ.
    prod_atom_count = prod_mol.GetNumAtoms() if prod_mol is not None else 0
    expected_atom_counts = []
    for exp in expected_canon:
        exp_mol = _safe_mol(exp)
        if exp_mol is not None:
            expected_atom_counts.append(exp_mol.GetNumAtoms())
    if expected_atom_counts:
        ref_count = max(expected_atom_counts)
        if abs(prod_atom_count - ref_count) > max(2, 0.2 * ref_count):
            return "fragment_misalignment"

    # 8. Stereochemistry loss.
    exp_stereo = max((_stereo_center_count(_safe_mol(e)) for e in expected_canon), default=0)
    prod_stereo = _stereo_center_count(prod_mol)
    if exp_stereo > 0 and prod_stereo < exp_stereo:
        return "stereochemistry_loss"

    # 9. Aromaticity violation.
    exp_arom = max((_aromatic_atom_count(_safe_mol(e)) for e in expected_canon), default=0)
    prod_arom = _aromatic_atom_count(prod_mol)
    if exp_arom > 0 and prod_arom == 0:
        return "aromaticity_violation"

    # 10. Over-reaction: many bonds differ from the expected product.
    max_bond_changes = 0
    for exp in expected_canon:
        changes = _bond_change_count(prod_canon, exp)
        max_bond_changes = max(max_bond_changes, changes)
    if max_bond_changes > OVER_REACTION_BOND_THRESHOLD:
        return "over_reaction"

    # 11. Wrong bond type: a single bond changed but atom counts match.
    if max_bond_changes == 1:
        return "wrong_bond_type"

    # Fallback bucket.
    return "side_product"


def extract_failure_type_labels(csv_path: str) -> Tuple[List[str], List[str]]:
    """Load a normalized CSV and return (reaction_smiles_list, failure_type_list).

    Only rows whose ``label_type`` marks a real negative are labeled. Dataset
    sources are handled specially:

      * ``regiosqm20`` real negatives are alternative-outcome regio-isomers
        and are labeled ``wrong_anchor`` (the defining failure mode of the
        RegioSQM20 alternative-outcome dataset).
      * ``hitea_full`` real negatives are low-yield reactions; the general
        ``label_failure_type`` heuristic is applied, and generic
        ``side_product`` fallbacks are overridden to ``under_reaction`` to
        reflect the low-yield semantics.
      * Other sources use the generic heuristic.
    """
    reactions: List[str] = []
    failure_types: List[str] = []

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_type = str(row.get("label_type", "")).strip().lower()
            if label_type not in {"real_negative", "negative", "neg", "failed", "failure"}:
                continue
            reaction = str(row.get("reaction_smiles", "")).strip()
            if not reaction:
                continue
            products = str(row.get("products", "")).strip()
            source = str(row.get("source", "")).strip().lower()

            expected_outcomes = [products] if products else []

            if source.startswith("regiosqm"):
                failure = "wrong_anchor"
            else:
                failure = label_failure_type(reaction, expected_outcomes)
                if failure == "side_product" and source.startswith("hitea"):
                    failure = "under_reaction"

            reactions.append(reaction)
            failure_types.append(failure)

    return reactions, failure_types


def _features_to_tensor(features: Sequence[Sequence[float]], device=None) -> torch.Tensor:
    return torch.tensor(features, dtype=torch.float32, device=device)


class FailurePrototypeCalibrator(nn.Module):
    """Prototype network for failure direction calibration."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_failure_types: int = NUM_FAILURE_TYPES,
        temperature: float = 0.5,
        triplet_weight: float = 0.3,
    ) -> None:
        super().__init__()
        if num_failure_types <= 0:
            raise ValueError("num_failure_types must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.num_failure_types = int(num_failure_types)
        self.temperature = float(temperature)
        self.triplet_weight = float(triplet_weight)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        )
        # Small random init so gradients can flow to the input features even
        # before ``init_prototypes`` is called (otherwise all-zero prototypes
        # would zero out the control-generation gradient).
        self.prototypes = nn.Parameter(
            torch.randn(self.num_failure_types, self.embedding_dim) * 0.1
        )
        self._initialized = False

    # ------------------------------------------------------------------ utils
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return F.normalize(z, p=2, dim=-1)

    def init_prototypes(self, features: torch.Tensor, labels: Sequence[int]) -> None:
        """Initialize prototype vectors as the mean embedding of each class."""
        self.eval()
        with torch.no_grad():
            z = self.encode(features)
            counts = torch.zeros(self.num_failure_types, device=z.device)
            sums = torch.zeros(self.num_failure_types, self.embedding_dim, device=z.device)
            for emb, label in zip(z, labels):
                idx = int(label)
                if 0 <= idx < self.num_failure_types:
                    sums[idx] += emb
                    counts[idx] += 1
            for k in range(self.num_failure_types):
                if counts[k] > 0:
                    self.prototypes.data[k] = sums[k] / counts[k]
                else:
                    # Random small init for empty classes.
                    self.prototypes.data[k] = torch.randn(self.embedding_dim) * 0.01
            # Re-normalize prototypes.
            self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=-1)
        self._initialized = True

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (classification_logits, embeddings).

        Logits are computed as ``-||z - p_k||^2 / temperature`` so that
        ``softmax(logits)`` gives the prototype-assignment distribution.
        """
        z = self.encode(x)
        # Squared L2 distance to each prototype: (B, K)
        diff = z.unsqueeze(1) - self.prototypes.unsqueeze(0)
        dist_sq = (diff * diff).sum(dim=-1)
        logits = -dist_sq / max(self.temperature, 1e-6)
        return logits, z

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted failure type indices in ``[0, num_failure_types)``."""
        logits, _ = self.forward(x)
        return logits.argmax(dim=-1)

    def failure_type_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probability distribution over failure types."""
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)

    def control_generation(self, x: torch.Tensor, target_type: int) -> torch.Tensor:
        """Return a loss whose gradient pushes ``x`` toward ``target_type``.

        The caller is expected to set ``x.requires_grad_(True)`` and call
        ``loss.backward()`` on the returned tensor. The gradient on ``x`` will
        move the embedding toward the target prototype.
        """
        if not (0 <= int(target_type) < self.num_failure_types):
            raise ValueError(f"target_type out of range: {target_type}")
        z = self.encode(x)
        target = self.prototypes[int(target_type)].unsqueeze(0).expand(z.size(0), -1)
        # Negative cosine similarity (embeddings are L2-normalized so this is
        # equivalent to half the squared distance).
        loss = (1.0 - (z * target).sum(dim=-1)).mean()
        return loss

    # --------------------------------------------------------------- persist
    def to_checkpoint(self) -> Dict[str, object]:
        return {
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "num_failure_types": self.num_failure_types,
            "temperature": self.temperature,
            "triplet_weight": self.triplet_weight,
            "feature_names": list(RERANKER_FEATURE_NAMES),
            "failure_types": list(FAILURE_TYPES),
            "initialized": self._initialized,
        }

    @classmethod
    def from_checkpoint(cls, path: str, map_location=None) -> "FailurePrototypeCalibrator":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(
            input_dim=int(blob["input_dim"]),
            hidden_dim=int(blob["hidden_dim"]),
            embedding_dim=int(blob["embedding_dim"]),
            num_failure_types=int(blob["num_failure_types"]),
            temperature=float(blob["temperature"]),
            triplet_weight=float(blob["triplet_weight"]),
        )
        model.load_state_dict(blob["state_dict"])
        model._initialized = bool(blob.get("initialized", True))
        return model


def _triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Per-sample triplet loss.

    For each sample, the positive prototype is its label's prototype and the
    negative prototype is the nearest *other* prototype.
    """
    if embeddings.size(0) == 0:
        return embeddings.new_zeros(())
    pos_proto = prototypes[labels]  # (B, D)
    dist_pos = (1.0 - (embeddings * pos_proto).sum(dim=-1))  # cosine distance
    # Distance to all prototypes, then mask out the positive.
    sim_all = embeddings @ prototypes.t()  # (B, K)
    mask = F.one_hot(labels, num_classes=prototypes.size(0)).bool()
    sim_all = sim_all.masked_fill(mask, -2.0)
    dist_neg = 1.0 - sim_all.max(dim=-1).values
    loss = F.relu(dist_pos - dist_neg + margin)
    return loss.mean()


def train_calibrator(
    model: FailurePrototypeCalibrator,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[str] = None,
    triplet_margin: float = 0.2,
    verbose: bool = False,
) -> Dict[str, object]:
    """Train the calibrator and return a history dict.

    The history contains per-epoch ``train_loss``, ``train_acc``,
    ``val_loss`` and ``val_acc``. The final entry also records
    ``best_val_acc`` and ``best_epoch``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    train_features = train_features.to(device).float()
    train_labels = train_labels.to(device).long()

    # Initialize prototypes before optimisation so they start at class means.
    if not model._initialized and train_features.size(0) > 0:
        model.init_prototypes(train_features, train_labels.tolist())

    if val_features is not None and val_labels is not None:
        val_features = val_features.to(device).float()
        val_labels = val_labels.to(device).long()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_acc = -1.0
    best_epoch = -1
    n_train = train_features.size(0)
    if n_train == 0:
        history["best_val_acc"] = 0.0
        history["best_epoch"] = -1
        return history

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        correct = 0
        seen = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb = train_features[idx]
            yb = train_labels[idx]
            logits, z = model(xb)
            ce = F.cross_entropy(logits, yb)
            tri = _triplet_loss(z, yb, model.prototypes, margin=triplet_margin)
            loss = ce + model.triplet_weight * tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=-1) == yb).sum().item()
            seen += xb.size(0)
        train_loss = total_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if val_features is not None and val_features.size(0) > 0:
            model.eval()
            with torch.no_grad():
                vlogits, _ = model(val_features)
                vloss = F.cross_entropy(vlogits, val_labels).item()
                vacc = (vlogits.argmax(dim=-1) == val_labels).float().mean().item()
            history["val_loss"].append(vloss)
            history["val_acc"].append(vacc)
            if vacc > best_val_acc:
                best_val_acc = vacc
                best_epoch = epoch
        else:
            history["val_loss"].append(float("nan"))
            history["val_acc"].append(float("nan"))
            if train_acc > best_val_acc:
                best_val_acc = train_acc
                best_epoch = epoch

        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
            print(
                f"epoch {epoch + 1:3d}/{epochs} train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} val_acc={history['val_acc'][-1]}"
            )

    history["best_val_acc"] = best_val_acc
    history["best_epoch"] = best_epoch
    return history


def evaluate_controllability(
    model: FailurePrototypeCalibrator,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    min_per_class: int = 10,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Evaluate failure type controllability.

    Metrics:
      * ``classification_accuracy``: overall accuracy on the test set.
      * ``per_class_accuracy``: dict failure_type -> accuracy.
      * ``confusion``: dict "true,pred" -> count (for error analysis).
      * ``mean_entropy``: mean per-sample Shannon entropy of the predicted
        distribution (natural log; uniform = ln(num_failure_types)).
      * ``normalized_entropy``: mean_entropy / ln(num_failure_types) in [0, 1].
      * ``aggregate_entropy``: entropy of the predicted-class distribution
        across the test set.
      * ``target_hit_rate``: dict failure_type -> hit rate when steering
        toward that prototype. Classes with fewer than ``min_per_class``
        samples are skipped (value = None).
      * ``uniform_entropy``: ln(num_failure_types), for reference.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    test_features = test_features.to(device).float()
    test_labels = test_labels.to(device).long()

    report: Dict[str, object] = {}
    if test_features.size(0) == 0:
        report["classification_accuracy"] = 0.0
        report["per_class_accuracy"] = {}
        report["confusion"] = {}
        report["mean_entropy"] = 0.0
        report["normalized_entropy"] = 0.0
        report["aggregate_entropy"] = 0.0
        report["target_hit_rate"] = {}
        report["uniform_entropy"] = UNIFORM_ENTROPY
        return report

    with torch.no_grad():
        logits, _ = model(test_features)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

    # Classification accuracy.
    correct = (preds == test_labels).float().mean().item()
    report["classification_accuracy"] = correct

    # Per-class accuracy + confusion matrix.
    per_class: Dict[str, float] = {}
    confusion: Dict[str, int] = {}
    class_counts = torch.bincount(test_labels, minlength=model.num_failure_types)
    pred_counts = torch.bincount(preds, minlength=model.num_failure_types)
    for k in range(model.num_failure_types):
        name = FAILURE_TYPES[k] if k < len(FAILURE_TYPES) else str(k)
        if int(class_counts[k]) > 0:
            mask = test_labels == k
            acc = (preds[mask] == k).float().mean().item()
            per_class[name] = acc
        else:
            per_class[name] = float("nan")
    for t, p in zip(test_labels.tolist(), preds.tolist()):
        key = f"{FAILURE_TYPES[t] if t < len(FAILURE_TYPES) else t},{FAILURE_TYPES[p] if p < len(FAILURE_TYPES) else p}"
        confusion[key] = confusion.get(key, 0) + 1
    report["per_class_accuracy"] = per_class
    report["confusion"] = confusion

    # Entropy metrics.
    eps = 1e-12
    sample_entropy = -(probs * (probs + eps).log()).sum(dim=-1)
    mean_entropy = float(sample_entropy.mean().item())
    report["mean_entropy"] = mean_entropy
    report["normalized_entropy"] = mean_entropy / UNIFORM_ENTROPY

    # Aggregate distribution entropy.
    aggregate = pred_counts.float() / max(int(pred_counts.sum()), 1)
    aggregate = aggregate[aggregate > 0]
    aggregate_entropy = float(-(aggregate * aggregate.log()).sum().item())
    report["aggregate_entropy"] = aggregate_entropy
    report["uniform_entropy"] = UNIFORM_ENTROPY

    # Target hit rate: steer each sample's embedding toward target prototype,
    # then measure how many get classified as the target. We emulate the
    # control by projecting features in embedding space toward the target
    # prototype and re-classifying.
    target_hit_rate: Dict[str, Optional[float]] = {}
    with torch.no_grad():
        z = model.encode(test_features)
        for k in range(model.num_failure_types):
            name = FAILURE_TYPES[k] if k < len(FAILURE_TYPES) else str(k)
            if int(class_counts[k]) < min_per_class:
                target_hit_rate[name] = None
                continue
            target_proto = model.prototypes[k]
            # Steer halfway toward the target prototype.
            steered = 0.5 * z + 0.5 * target_proto.unsqueeze(0)
            steered = F.normalize(steered, p=2, dim=-1)
            diff = steered.unsqueeze(1) - model.prototypes.unsqueeze(0)
            dist_sq = (diff * diff).sum(dim=-1)
            new_logits = -dist_sq / max(model.temperature, 1e-6)
            new_preds = new_logits.argmax(dim=-1)
            hit = (new_preds == k).float().mean().item()
            target_hit_rate[name] = hit
    report["target_hit_rate"] = target_hit_rate

    return report


def write_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


__all__ = [
    "FAILURE_TYPES",
    "FAILURE_TYPE_TO_IDX",
    "NUM_FAILURE_TYPES",
    "UNIFORM_ENTROPY",
    "ATOM_BALANCE_THRESHOLD",
    "FailurePrototypeCalibrator",
    "label_failure_type",
    "extract_failure_type_labels",
    "train_calibrator",
    "evaluate_controllability",
    "write_json",
]
