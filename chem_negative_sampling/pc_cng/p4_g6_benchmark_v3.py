"""Formal G6 v3 benchmark primitives.

This module replaces the product-only G6 v2 proxies with a formal benchmark
contract.  It is intentionally separate from v2 so historical artifacts remain
auditable.  A formal run must use the frozen Chemformer checkpoint on CUDA.

Design invariants
-----------------
* One frozen pretrained reaction encoder is shared by T1--T5.
* Every input is ``reactants | catalyst | solvent | reagent | product`` plus
  explicit temperature/time features; no task may consume product-only input.
* T2 is a cumulative-link ordinal model and T4 is trained with a pairwise loss.
* Source-comparison arms contain the same positive parents and exactly one
  negative per parent.  They differ only by negative source.
* The module never chooses a baseline from test data.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from rdkit import Chem
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, ndcg_score
from torch import nn
import torch.nn.functional as F

# ``pc_cng`` and ``models`` are installed as sibling top-level packages from
# ``chem_negative_sampling``.  Import the backbone through its packaged name
# so the documented ``python -m pc_cng.run_p4_g6_v3`` entrypoint works both
# from the source checkout and from a wheel installation.
from models.pretrained_backbone import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_VOCAB_PATH,
    ChemformerTokenizer,
    PretrainedChemformerBackbone,
)


FORMAL_SCHEMA_VERSION = "g6_v3_corrected_reanalysis_20260729"
PRIMARY_ENDPOINT = "T5_condition_feasibility_source_macro_auprc"
T1_PRIMARY_LOW_YIELD_THRESHOLD = 10.0
T5_FEASIBILITY_THRESHOLD = 50.0
ORDINAL_BIN_EDGES = (0.0, 10.0, 30.0, 50.0, 70.0, float("inf"))
FORMAL_MAX_SEQ_LEN = 512
PRE_REGISTERED_NONINFERIORITY_MARGIN = 0.02
PRE_REGISTERED_PRIMARY_COMPARISONS = (
    "pc_cng_vs_random",
    "pc_cng_vs_template_rule",
    "union_vs_pc_cng",
)
CANDIDATE_INTEGRITY_CONTRACT = {
    "exclude_parent_positive_collisions": True,
    "selection_order": "lowest source rank then candidate_id among non-colliding candidates",
    "cross_source_duplicates": "retain and report",
    "analysis_status": "CORRECTED_REANALYSIS_TEST_OUTCOMES_PREVIOUSLY_OBSERVED",
}
METRIC_IMPLEMENTATION_CONTRACT = {
    "T1_and_T5_auprc": "sklearn.metrics.average_precision_score with threshold-level tie handling",
    "T3_spearman": "scipy.stats.spearmanr with average ranks for ties",
    "T4_ndcg": "sklearn.metrics.ndcg_score with ignore_ties=false",
}
DETERMINISM_CONTRACT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "torch_deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
}
SOURCE_ALIASES = {
    "pc_cng": "rule_pc_cng",
    "random": "random_mismatch",
    "template_rule": "template_perturbation",
}


@dataclass(frozen=True)
class FormalAnalysisPlan:
    """Frozen choices that must be written before a formal test is inspected."""

    schema_version: str = FORMAL_SCHEMA_VERSION
    primary_endpoint: str = PRIMARY_ENDPOINT
    t1_low_yield_threshold: float = T1_PRIMARY_LOW_YIELD_THRESHOLD
    t5_feasibility_threshold: float = T5_FEASIBILITY_THRESHOLD
    ordinal_bin_edges: tuple[float, ...] = ORDINAL_BIN_EDGES
    primary_comparisons: tuple[str, ...] = PRE_REGISTERED_PRIMARY_COMPARISONS
    noninferiority_margin: float = PRE_REGISTERED_NONINFERIORITY_MARGIN
    source_aliases: Mapping[str, str] | None = None
    n_seeds: int = 5
    n_bootstrap: int = 2000
    n_permutations: int = 10000
    max_seq_len: int = FORMAL_MAX_SEQ_LEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "primary_endpoint": self.primary_endpoint,
            "t1_low_yield_threshold": self.t1_low_yield_threshold,
            "t5_feasibility_threshold": self.t5_feasibility_threshold,
            "ordinal_bin_edges": list(self.ordinal_bin_edges),
            "primary_comparisons": list(self.primary_comparisons),
            "noninferiority_margin": self.noninferiority_margin,
            "source_aliases": dict(self.source_aliases or SOURCE_ALIASES),
            "n_seeds": self.n_seeds,
            "n_bootstrap": self.n_bootstrap,
            "n_permutations": self.n_permutations,
            "max_seq_len": self.max_seq_len,
            "candidate_integrity_contract": dict(CANDIDATE_INTEGRITY_CONTRACT),
            "metric_implementation_contract": dict(METRIC_IMPLEMENTATION_CONTRACT),
            "determinism_contract": dict(DETERMINISM_CONTRACT),
        }


def validate_formal_analysis_plan(plan: Mapping[str, Any]) -> None:
    """Fail closed when an analysis plan changes the formal contract."""
    required = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "t1_low_yield_threshold": T1_PRIMARY_LOW_YIELD_THRESHOLD,
        "t5_feasibility_threshold": T5_FEASIBILITY_THRESHOLD,
        "noninferiority_margin": PRE_REGISTERED_NONINFERIORITY_MARGIN,
        "max_seq_len": FORMAL_MAX_SEQ_LEN,
    }
    for key, expected in required.items():
        if plan.get(key) != expected:
            raise ValueError(f"formal analysis plan mismatch for {key}: {plan.get(key)!r} != {expected!r}")
    if tuple(plan.get("primary_comparisons", ())) != PRE_REGISTERED_PRIMARY_COMPARISONS:
        raise ValueError("formal analysis plan must retain the preregistered primary comparison order")
    if tuple(float(x) for x in plan.get("ordinal_bin_edges", ())) != ORDINAL_BIN_EDGES:
        raise ValueError("formal analysis plan must retain ordinal bin edges")
    aliases = plan.get("source_aliases", {})
    if dict(aliases) != SOURCE_ALIASES:
        raise ValueError("formal analysis plan source aliases differ from the frozen contract")
    if dict(plan.get("candidate_integrity_contract", {})) != CANDIDATE_INTEGRITY_CONTRACT:
        raise ValueError("formal analysis plan candidate integrity contract differs from the corrected contract")
    if dict(plan.get("metric_implementation_contract", {})) != METRIC_IMPLEMENTATION_CONTRACT:
        raise ValueError("formal analysis plan metric implementation contract differs from the corrected contract")
    if dict(plan.get("determinism_contract", {})) != DETERMINISM_CONTRACT:
        raise ValueError("formal analysis plan determinism contract differs from the corrected contract")


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ".".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def _canonical_structure(smiles: Any) -> str:
    """Canonicalize a product for label-integrity checks, removing atom maps."""
    text = _text(smiles)
    if not text:
        return ""
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return text
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True)


def normalized_condition_fields(record: Mapping[str, Any]) -> dict[str, str]:
    """Return condition fields without silently collapsing their identities."""
    return {
        "reactants": _text(record.get("reactants")),
        "catalysts": _text(record.get("catalysts", record.get("catalyst", ""))),
        "solvents": _text(record.get("solvents", record.get("solvent", ""))),
        "reagents": _text(record.get("reagents", record.get("reagent", ""))),
        "products": _text(record.get("products")),
    }


def validate_reaction_condition_records(records: Sequence[Mapping[str, Any]], *, formal: bool) -> dict[str, float]:
    """Validate the formal reaction/condition input contract.

    Catalyst, solvent and reagent may be genuinely absent for a reaction, but
    the fields must exist in the normalized representation and temperature/time
    must be observed often enough for a condition-feasibility claim.
    """
    if not records:
        raise ValueError("empty HTE record collection")
    required = ("record_id", "reactants", "products", "measured_yield", "split", "experimental_group")
    for i, record in enumerate(records):
        missing = [k for k in required if k not in record]
        if missing:
            raise ValueError(f"record {i} missing required fields: {missing}")
        fields = normalized_condition_fields(record)
        if not fields["reactants"] or not fields["products"]:
            raise ValueError(f"record {record.get('record_id')} lacks reactants or products")
    n = len(records)
    temp = sum(record.get("temperature") not in (None, "") for record in records) / n
    time = sum(record.get("reaction_time_hrs") not in (None, "") for record in records) / n
    availability = {
        "temperature_fraction": temp,
        "reaction_time_fraction": time,
        "catalyst_fraction": sum(bool(normalized_condition_fields(r)["catalysts"]) for r in records) / n,
        "solvent_fraction": sum(bool(normalized_condition_fields(r)["solvents"]) for r in records) / n,
        "reagent_fraction": sum(bool(normalized_condition_fields(r)["reagents"]) for r in records) / n,
    }
    if formal and min(temp, time) < 0.80:
        raise ValueError(f"formal T5 requires >=80% temperature/time coverage, got {availability}")
    return availability


def context_token_visibility(
    records: Sequence[Mapping[str, Any]],
    tokenizer: ChemformerTokenizer,
    *,
    max_seq_len: int,
) -> dict[str, int]:
    """Audit whether every reaction-context segment fits without truncation."""
    any_truncated = 0
    product_partially_visible = 0
    product_not_visible = 0
    max_total_tokens = 0
    separator_count = 4
    for record in records:
        fields = normalized_condition_fields(record)
        lengths = [
            len(tokenizer.encode(segment, add_special=False))
            for segment in (
                fields["reactants"],
                fields["catalysts"],
                fields["solvents"],
                fields["reagents"],
                fields["products"],
            )
        ]
        total_tokens = 2 + separator_count + sum(lengths)
        max_total_tokens = max(max_total_tokens, total_tokens)
        if total_tokens > max_seq_len:
            any_truncated += 1
        product_start = 1 + separator_count + sum(lengths[:4])
        retained_product = max(
            0,
            min(lengths[4], max_seq_len - 1 - product_start),
        )
        if retained_product == 0 and lengths[4] > 0:
            product_not_visible += 1
        elif retained_product < lengths[4]:
            product_partially_visible += 1
    return {
        "n_records": len(records),
        "max_seq_len": int(max_seq_len),
        "max_total_tokens": int(max_total_tokens),
        "any_truncated": int(any_truncated),
        "product_partially_visible": int(product_partially_visible),
        "product_not_visible": int(product_not_visible),
        "all_segments_fully_visible": int(any_truncated == 0),
    }


def validate_cluster_contract(
    records: Sequence[Mapping[str, Any]],
    *,
    cluster_key: str = "experimental_group",
    endpoint_threshold: float = T5_FEASIBILITY_THRESHOLD,
    formal: bool,
) -> dict[str, float]:
    """Check that the pre-existing experimental clusters support paired inference.

    The cluster key is frozen before model fitting and must describe an actual
    plate/experimental group, not a post-hoc grouping engineered to improve a
    confidence interval.  Formal inference requires enough held-out clusters
    and endpoint variation for a cluster-level resampling design.
    """
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        value = str(record.get(cluster_key, "")).strip()
        if not value:
            raise ValueError(f"record {record.get('record_id', '')} lacks required cluster key {cluster_key}")
        clusters[value].append(record)
    evaluable = sum(
        len({int(float(item.get("measured_yield", 0.0)) >= endpoint_threshold) for item in group}) == 2
        for group in clusters.values()
    )
    result = {
        "cluster_count": float(len(clusters)),
        "endpoint_evaluable_cluster_count": float(evaluable),
        "cluster_key_is_preexisting_metadata": 1.0,
    }
    if formal and (len(clusters) < 20 or evaluable < 10):
        raise ValueError(
            f"formal paired-cluster inference requires >=20 {cluster_key} clusters and >=10 endpoint-evaluable clusters, got {result}"
        )
    return result


def partition_context_complete_records(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Partition records without manufacturing missing reaction context.

    Exclusions are returned to the caller for an immutable artifact.  This is
    preferable to silently replacing missing reactants with a zero vector, and
    the formal runner records the count, split and identifiers before training.
    """
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for record in records:
        fields = normalized_condition_fields(record)
        reasons = []
        if not fields["reactants"]:
            reasons.append("missing_reactants")
        if not fields["products"]:
            reasons.append("missing_products")
        if reasons:
            excluded.append({
                "record_id": str(record.get("record_id", "")),
                "split": str(record.get("split", "")),
                "reason": ";".join(reasons),
            })
        else:
            included.append(dict(record))
    return included, excluded


class ConditionNormalizer:
    """Train-split-only normalizer for explicit temperature/time features."""

    def __init__(self) -> None:
        self.temperature_mean = 0.0
        self.temperature_std = 1.0
        self.log_time_mean = 0.0
        self.log_time_std = 1.0

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
            return out if math.isfinite(out) else default
        except (TypeError, ValueError):
            return default

    def fit(self, train_records: Sequence[Mapping[str, Any]]) -> "ConditionNormalizer":
        temperatures = np.asarray([self._number(r.get("temperature")) for r in train_records], dtype=np.float32)
        log_times = np.asarray([math.log1p(max(0.0, self._number(r.get("reaction_time_hrs")))) for r in train_records], dtype=np.float32)
        self.temperature_mean = float(temperatures.mean())
        self.temperature_std = float(max(temperatures.std(), 1e-6))
        self.log_time_mean = float(log_times.mean())
        self.log_time_std = float(max(log_times.std(), 1e-6))
        return self

    def vector(self, record: Mapping[str, Any]) -> np.ndarray:
        fields = normalized_condition_fields(record)
        temperature = self._number(record.get("temperature"))
        log_time = math.log1p(max(0.0, self._number(record.get("reaction_time_hrs"))))
        return np.asarray([
            float(bool(fields["catalysts"])),
            float(bool(fields["solvents"])),
            float(bool(fields["reagents"])),
            (temperature - self.temperature_mean) / self.temperature_std,
            (log_time - self.log_time_mean) / self.log_time_std,
            min(4.0, float(fields["catalysts"].count(".") + 1)) if fields["catalysts"] else 0.0,
            min(4.0, float(fields["reagents"].count(".") + 1)) if fields["reagents"] else 0.0,
        ], dtype=np.float32)


class SharedPretrainedReactionEncoder(nn.Module):
    """One frozen Chemformer encoder shared by all G6 task heads.

    Context segments are joined with Chemformer ``<SEP>`` token IDs rather
    than an unknown ``>>`` character.  Thus reactants, catalyst, solvent,
    reagent and candidate product are visible jointly to the pretrained model.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        vocab_path: str | Path = DEFAULT_VOCAB_PATH,
        *,
        max_seq_len: int = FORMAL_MAX_SEQ_LEN,
        device: str | torch.device = "cuda",
        formal: bool = True,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        checkpoint = Path(checkpoint_path)
        vocab = Path(vocab_path)
        if formal:
            if self.device.type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("formal G6 v3 benchmark requires CUDA; refusing CPU fallback")
            if not checkpoint.is_file() or not vocab.is_file():
                raise FileNotFoundError("formal G6 v3 benchmark requires local pretrained Chemformer checkpoint and vocabulary")
        self.tokenizer = ChemformerTokenizer(vocab, max_seq_len=max_seq_len)
        self.sep_idx = self.tokenizer.token_to_id.get("<SEP>", 5)
        self.backbone = PretrainedChemformerBackbone(checkpoint_path=checkpoint if checkpoint.is_file() else None, freeze=True)
        self.backbone.eval().to(self.device)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.condition_dim = 7
        self.output_dim = int(self.backbone.hparams["d_model"]) + self.condition_dim

    def _context_ids(self, record: Mapping[str, Any]) -> list[int]:
        fields = normalized_condition_fields(record)
        segments = (fields["reactants"], fields["catalysts"], fields["solvents"], fields["reagents"], fields["products"])
        ids = [self.tokenizer.bos_idx]
        for index, segment in enumerate(segments):
            ids.extend(self.tokenizer.encode(segment, add_special=False))
            if index != len(segments) - 1:
                ids.append(self.sep_idx)
        ids.append(self.tokenizer.eos_idx)
        if len(ids) > self.tokenizer.max_seq_len:
            ids = ids[: self.tokenizer.max_seq_len - 1] + [self.tokenizer.eos_idx]
        return ids

    def encode_records(
        self,
        records: Sequence[Mapping[str, Any]],
        normalizer: ConditionNormalizer,
        *,
        batch_size: int = 16,
    ) -> torch.Tensor:
        if not records:
            return torch.empty((0, self.output_dim), dtype=torch.float32)
        all_outputs: list[torch.Tensor] = []
        self.backbone.eval()
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                batch = records[start:start + batch_size]
                encoded = [self._context_ids(record) for record in batch]
                width = max(len(ids) for ids in encoded)
                ids = torch.full((len(batch), width), self.tokenizer.pad_idx, dtype=torch.long, device=self.device)
                mask = torch.zeros((len(batch), width), dtype=torch.long, device=self.device)
                for row, item in enumerate(encoded):
                    ids[row, :len(item)] = torch.tensor(item, dtype=torch.long, device=self.device)
                    mask[row, :len(item)] = 1
                pooled = self.backbone(ids, attention_mask=mask, pool=True)
                condition = torch.tensor(np.stack([normalizer.vector(record) for record in batch]), dtype=torch.float32, device=self.device)
                all_outputs.append(torch.cat([pooled, condition], dim=1).cpu())
        return torch.cat(all_outputs, dim=0)


def ordinal_bin(value: float) -> int:
    for index in range(len(ORDINAL_BIN_EDGES) - 1):
        if ORDINAL_BIN_EDGES[index] <= value < ORDINAL_BIN_EDGES[index + 1]:
            return index
    return len(ORDINAL_BIN_EDGES) - 2


class CumulativeLinkHead(nn.Module):
    """True proportional-odds cumulative-link ordinal head."""

    def __init__(self, in_dim: int, n_classes: int = 5) -> None:
        super().__init__()
        self.score_layer = nn.Linear(in_dim, 1)
        self.threshold_start = nn.Parameter(torch.tensor(-1.0))
        self.threshold_deltas = nn.Parameter(torch.zeros(n_classes - 2))
        self.n_classes = n_classes

    def thresholds(self) -> torch.Tensor:
        if self.n_classes == 2:
            return self.threshold_start.reshape(1)
        increments = F.softplus(self.threshold_deltas) + 1e-4
        return torch.cat([self.threshold_start.reshape(1), self.threshold_start + torch.cumsum(increments, dim=0)])

    def cumulative_probabilities(self, features: torch.Tensor) -> torch.Tensor:
        latent = self.score_layer(features)
        return torch.sigmoid(latent - self.thresholds().reshape(1, -1))

    def class_probabilities(self, features: torch.Tensor) -> torch.Tensor:
        greater = self.cumulative_probabilities(features)
        batch = features.shape[0]
        one = torch.ones((batch, 1), dtype=features.dtype, device=features.device)
        zero = torch.zeros((batch, 1), dtype=features.dtype, device=features.device)
        return torch.cat([one - greater[:, :1], greater[:, :-1] - greater[:, 1:], greater[:, -1:]], dim=1).clamp_min(1e-7)

    def loss(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cumulative_targets = torch.stack([(targets > level).float() for level in range(self.n_classes - 1)], dim=1)
        return F.binary_cross_entropy_with_logits(self.score_layer(features) - self.thresholds().reshape(1, -1), cumulative_targets)


class G6MultiTaskHeads(nn.Module):
    """Five genuine task heads over the same cached pretrained encoder output."""

    def __init__(self, in_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        def trunk(out_dim: int = 1) -> nn.Sequential:
            return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(0.10), nn.Linear(hidden_dim, out_dim))
        self.t1_low_yield = trunk()
        self.t2_ordinal = CumulativeLinkHead(in_dim)
        self.t3_regression = trunk()
        self.t4_rank = trunk()
        self.t5_feasibility = trunk()

    def predict(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "T1": torch.sigmoid(self.t1_low_yield(features).squeeze(1)),
            "T2": self.t2_ordinal.class_probabilities(features),
            "T3": self.t3_regression(features).squeeze(1).clamp(0.0, 100.0),
            "T4": self.t4_rank(features).squeeze(1),
            "T5": torch.sigmoid(self.t5_feasibility(features).squeeze(1)),
        }


def _pair_indices(records: Sequence[Mapping[str, Any]], *, max_pairs: int, seed: int) -> list[tuple[int, int, float]]:
    by_plate: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_plate[str(record.get("plate_id") or record.get("experimental_group") or "ungrouped")].append(index)
    pairs: list[tuple[int, int, float]] = []
    rng = random.Random(seed)
    for plate, indices in sorted(by_plate.items()):
        ordered = sorted(indices, key=lambda i: float(records[i].get("measured_yield", 0.0)))
        local = [(ordered[high], ordered[low], 1.0) for high in range(1, len(ordered)) for low in range(high)
                 if float(records[ordered[high]].get("measured_yield", 0.0)) > float(records[ordered[low]].get("measured_yield", 0.0))]
        rng.shuffle(local)
        pairs.extend(local[:max_pairs])
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def train_multitask_heads(
    features: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    *,
    device: str | torch.device,
    seed: int,
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_rank_pairs: int = 4096,
) -> tuple[G6MultiTaskHeads, dict[str, float]]:
    """Train all five heads on GPU from one shared encoder feature matrix."""
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("G6 v3 training requires CUDA; refusing CPU fallback")
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    device_obj = torch.device(device)
    x = features.to(device_obj)
    yields = torch.tensor([float(r.get("measured_yield", 0.0)) for r in records], dtype=torch.float32, device=device_obj)
    t1 = (yields < T1_PRIMARY_LOW_YIELD_THRESHOLD).float()
    t2 = torch.tensor([ordinal_bin(float(v)) for v in yields.cpu().tolist()], dtype=torch.long, device=device_obj)
    t5 = (yields >= T5_FEASIBILITY_THRESHOLD).float()
    model = G6MultiTaskHeads(x.shape[1]).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rank_pairs = _pair_indices(records, max_pairs=max_rank_pairs, seed=seed)
    indices = np.arange(len(records))
    history: dict[str, float] = {}
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(indices)
        model.train()
        loss_total = 0.0
        batches = 0
        for start in range(0, len(indices), batch_size):
            idx = torch.tensor(indices[start:start + batch_size], dtype=torch.long, device=device_obj)
            batch_x = x.index_select(0, idx)
            prediction = model.predict(batch_x)
            loss = (
                F.binary_cross_entropy(prediction["T1"], t1.index_select(0, idx))
                + model.t2_ordinal.loss(batch_x, t2.index_select(0, idx))
                + F.smooth_l1_loss(prediction["T3"] / 100.0, yields.index_select(0, idx) / 100.0)
                + F.binary_cross_entropy(prediction["T5"], t5.index_select(0, idx))
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach().cpu())
            batches += 1
        if rank_pairs:
            sample = rank_pairs[:min(len(rank_pairs), max_rank_pairs)]
            high = torch.tensor([p[0] for p in sample], dtype=torch.long, device=device_obj)
            low = torch.tensor([p[1] for p in sample], dtype=torch.long, device=device_obj)
            model.train()
            rank_loss = F.softplus(-(model.predict(x.index_select(0, high))["T4"] - model.predict(x.index_select(0, low))["T4"])).mean()
            optimizer.zero_grad(set_to_none=True)
            rank_loss.backward()
            optimizer.step()
        else:
            rank_loss = torch.tensor(0.0)
        history = {"loss": loss_total / max(1, batches), "rank_loss": float(rank_loss.detach().cpu()), "epochs": float(epoch + 1), "n_rank_pairs": float(len(rank_pairs))}
    return model.eval(), history


def predict_multitask(model: G6MultiTaskHeads, features: torch.Tensor, *, device: str | torch.device) -> dict[str, np.ndarray]:
    device_obj = torch.device(device)
    model.eval()
    with torch.no_grad():
        pred = model.predict(features.to(device_obj))
    return {name: value.detach().cpu().numpy() for name, value in pred.items()}


def load_matched_source_arms(
    hte_records: Sequence[Mapping[str, Any]],
    manifest_path: str | Path,
    *,
    positive_threshold: float = T5_FEASIBILITY_THRESHOLD,
    seed: int = 20260728,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Create matched source arms using one candidate per shared parent.

    ``pc_cng`` maps to the existing ``rule_pc_cng`` source because no learned
    generator candidates are present in this frozen manifest.  The mapping is
    explicit in the analysis plan and output; it must not be renamed as learned.
    """
    with Path(manifest_path).open() as handle:
        manifest = json.load(handle)
    contexts = {str(r.get("record_id")): dict(r) for r in hte_records if r.get("split") == "train"}
    available: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    integrity_audit: dict[str, Any] = {
        "parent_positive_collisions_skipped_by_source": {
            source: 0 for source in SOURCE_ALIASES
        },
        "empty_candidates_skipped_by_source": {
            source: 0 for source in SOURCE_ALIASES
        },
        "parents_without_noncolliding_candidate_by_source": {
            source: 0 for source in SOURCE_ALIASES
        },
    }
    for group in manifest.get("groups", []):
        if group.get("split") != "train":
            continue
        parent_id = str(group.get("source_reaction_id", ""))
        parent = contexts.get(parent_id)
        if parent is None or float(parent.get("measured_yield", 0.0)) < positive_threshold:
            continue
        positive_structure = _canonical_structure(parent.get("products", ""))
        by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for candidate in group.get("candidates", []):
            by_source[str(candidate.get("candidate_source", ""))].append(candidate)
        for arm_source, manifest_source in SOURCE_ALIASES.items():
            candidates = sorted(
                by_source.get(manifest_source, []),
                key=lambda c: (
                    int(c.get("candidate_source_rank", 0)),
                    str(c.get("candidate_id", "")),
                ),
            )
            eligible: list[Mapping[str, Any]] = []
            for candidate in candidates:
                candidate_product = str(
                    candidate.get("canonical_smiles")
                    or candidate.get("candidate_smiles")
                    or ""
                )
                candidate_structure = _canonical_structure(candidate_product)
                if not candidate_structure:
                    integrity_audit["empty_candidates_skipped_by_source"][arm_source] += 1
                    continue
                if candidate_structure == positive_structure:
                    integrity_audit["parent_positive_collisions_skipped_by_source"][arm_source] += 1
                    continue
                eligible.append(candidate)
            if not eligible:
                integrity_audit["parents_without_noncolliding_candidate_by_source"][arm_source] += 1
                continue
            chosen = eligible[0]
            record = dict(parent)
            record.update({
                "candidate_id": str(chosen.get("candidate_id", "")),
                "parent_record_id": parent_id,
                "products": str(chosen.get("canonical_smiles") or chosen.get("candidate_smiles") or ""),
                "measured_yield": 0.0,
                "synthetic_negative": True,
                "negative_source": arm_source,
                "manifest_candidate_source": manifest_source,
                "edit_type": chosen.get("edit_type", ""),
            })
            available[parent_id][arm_source] = record
    common = sorted(parent for parent, values in available.items() if set(values) == set(SOURCE_ALIASES))
    if len(common) < 20:
        raise RuntimeError(f"only {len(common)} matched source parents; formal source comparison requires >=20")
    positives = []
    for parent_id in common:
        positive = dict(contexts[parent_id])
        positive.update({"parent_record_id": parent_id, "synthetic_negative": False, "negative_source": "positive"})
        positives.append(positive)
    arms: dict[str, list[dict[str, Any]]] = {"positive_only": list(positives)}
    for source in SOURCE_ALIASES:
        arms[source] = list(positives) + [available[parent][source] for parent in common]
    union_negatives = []
    for parent in common:
        source_names = tuple(SOURCE_ALIASES)
        source = source_names[(stable_int(f"{seed}:{parent}") % len(source_names))]
        union_negatives.append(available[parent][source])
    arms["union"] = list(positives) + union_negatives
    audit = assert_matched_source_arms(arms)
    chosen_by_parent = {
        parent: {
            source: _canonical_structure(available[parent][source]["products"])
            for source in SOURCE_ALIASES
        }
        for parent in common
    }
    integrity_audit["cross_source_duplicate_parent_count"] = sum(
        len(set(products.values())) < len(products)
        for products in chosen_by_parent.values()
    )
    integrity_audit["candidate_integrity_contract"] = dict(
        CANDIDATE_INTEGRITY_CONTRACT
    )
    audit.update({
        "n_matched_parents": len(common),
        "source_aliases": dict(SOURCE_ALIASES),
        "positive_threshold": positive_threshold,
        "candidate_integrity": integrity_audit,
    })
    return arms, audit


def assert_matched_source_arms(arms: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Verify that confirmatory source arms vary only negative source."""
    comparison_arms = ("pc_cng", "random", "template_rule", "union")
    missing = [arm for arm in comparison_arms if arm not in arms]
    if missing:
        raise AssertionError(f"missing matched comparison arms: {missing}")
    parent_sets: dict[str, set[str]] = {}
    counts: dict[str, dict[str, int]] = {}
    for arm in comparison_arms:
        records = list(arms[arm])
        positives = [r for r in records if not r.get("synthetic_negative")]
        negatives = [r for r in records if r.get("synthetic_negative")]
        pos_parent = {str(r.get("parent_record_id")) for r in positives}
        neg_parent = {str(r.get("parent_record_id")) for r in negatives}
        if pos_parent != neg_parent or len(positives) != len(negatives):
            raise AssertionError(f"{arm} violates one-positive/one-negative matched budget")
        positives_by_parent = {
            str(record.get("parent_record_id")): _canonical_structure(
                record.get("products", "")
            )
            for record in positives
        }
        for negative in negatives:
            parent_id = str(negative.get("parent_record_id"))
            if _canonical_structure(negative.get("products", "")) == positives_by_parent[parent_id]:
                raise AssertionError(
                    f"{arm} contains a parent-positive collision for {parent_id}"
                )
        parent_sets[arm] = pos_parent
        counts[arm] = {"n_positive": len(positives), "n_negative": len(negatives), "n_total": len(records)}
    reference = parent_sets[comparison_arms[0]]
    for arm in comparison_arms[1:]:
        if parent_sets[arm] != reference:
            raise AssertionError(f"{arm} differs in parent reactions from {comparison_arms[0]}")
    return {"comparison_arms": list(comparison_arms), "parent_count": len(reference), "counts": counts, "budget_matched": True}


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    label_values = np.asarray(labels, dtype=int)
    score_values = np.asarray(scores, dtype=float)
    if label_values.size == 0 or int(label_values.sum()) == 0:
        return 0.0
    return float(average_precision_score(label_values, score_values))


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return 0.0
    statistic = float(
        spearmanr(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        ).statistic
    )
    if not np.isfinite(statistic):
        return 0.0
    return statistic


def _ndcg(yields: Sequence[float], scores: Sequence[float]) -> float:
    if len(yields) < 2:
        return 0.0
    relevance = np.asarray(yields, dtype=float).reshape(1, -1)
    prediction = np.asarray(scores, dtype=float).reshape(1, -1)
    if float(relevance.max()) <= 0:
        return 0.0
    return float(ndcg_score(relevance, prediction, ignore_ties=False))


def source_macro_auprc(records: Sequence[Mapping[str, Any]], source_key: str = "source_publication") -> float:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(source_key, "unknown"))].append(record)
    values = []
    for group in groups.values():
        labels = [int(r["label"]) for r in group]
        if len(set(labels)) == 2:
            values.append(_average_precision(labels, [float(r["score"]) for r in group]))
    if values:
        return float(np.mean(values))
    return _average_precision([int(r["label"]) for r in records], [float(r["score"]) for r in records])


def source_macro_auprc_diagnostics(
    records: Sequence[Mapping[str, Any]], source_key: str = "source_publication"
) -> dict[str, float]:
    """Describe the scope of a source-macro endpoint without changing it.

    A source-macro AUPRC only represents cross-source generalization when two
    or more source-publication slices have both endpoint classes.  The current
    HiTEA release has a single publication; the formal result must state that
    its primary endpoint is therefore a single-source external HTE estimate,
    not evidence of cross-publication replication.
    """
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(source_key, "unknown"))].append(record)
    evaluable = sum(
        len({int(item["label"]) for item in group}) == 2
        for group in groups.values()
    )
    return {
        "source_publication_slices_total": float(len(groups)),
        "source_publication_slices_evaluable": float(evaluable),
        "source_macro_has_cross_publication_replication": float(evaluable >= 2),
    }


def fit_temperature_scaler(
    probabilities: np.ndarray,
    labels: Sequence[int],
    *,
    device: str | torch.device,
    max_steps: int = 50,
) -> float:
    """Fit one post-hoc temperature on the validation split, on GPU only.

    Calibration is selected strictly on validation labels and subsequently
    applied unchanged to the sealed test records.  The transformation is
    monotonic and cannot improve the ranking-based primary AUPRC by itself.
    """
    device_obj = torch.device(device)
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal G6 v3 calibration requires CUDA; refusing CPU fallback")
    scores = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    targets = np.asarray(labels, dtype=np.float32)
    if scores.shape[0] != targets.shape[0] or len(np.unique(targets)) < 2:
        return 1.0
    logits = torch.as_tensor(np.log(scores / (1.0 - scores)), device=device_obj)
    target_tensor = torch.as_tensor(targets, device=device_obj)
    log_temperature = torch.zeros((), device=device_obj, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_temperature], lr=0.20, max_iter=max_steps, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimiser.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, target_tensor)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).cpu())


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply a validation-fitted binary temperature without touching labels."""
    scores = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    logits = np.log(scores / (1.0 - scores))
    return 1.0 / (1.0 + np.exp(-logits / max(float(temperature), 0.05)))


def metric_records(task: str, test_records: Sequence[Mapping[str, Any]], predictions: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Attach task-specific scores and labels without mutating input records."""
    rows: list[dict[str, Any]] = []
    if task == "T2":
        probs = predictions["T2"]
        for record, probability in zip(test_records, probs):
            out = dict(record)
            out["score"] = float(np.argmax(probability))
            out["ordinal_probs"] = [float(x) for x in probability]
            out["label"] = ordinal_bin(float(record.get("measured_yield", 0.0)))
            rows.append(out)
        return rows
    key = task
    for record, score in zip(test_records, predictions[key]):
        out = dict(record)
        out["score"] = float(score)
        value = float(record.get("measured_yield", 0.0))
        if task == "T1":
            out["label"] = int(value < T1_PRIMARY_LOW_YIELD_THRESHOLD)
        elif task == "T5":
            out["label"] = int(value >= T5_FEASIBILITY_THRESHOLD)
        rows.append(out)
    return rows


def evaluate_secondary_metrics(test_records: Sequence[Mapping[str, Any]], predictions: Mapping[str, np.ndarray]) -> dict[str, float]:
    t1 = metric_records("T1", test_records, predictions)
    t2 = metric_records("T2", test_records, predictions)
    t3 = metric_records("T3", test_records, predictions)
    t4 = metric_records("T4", test_records, predictions)
    t5 = metric_records("T5", test_records, predictions)
    yields = np.asarray([float(r.get("measured_yield", 0.0)) for r in test_records])
    t3_scores = np.asarray([float(r["score"]) for r in t3])
    plates: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(t4):
        plates[str(record.get("plate_id") or record.get("experimental_group") or "ungrouped")].append(index)
    plate_ndcg = [_ndcg([yields[i] for i in ids], [float(t4[i]["score"]) for i in ids]) for ids in plates.values() if len(ids) >= 2]
    t5_labels = np.asarray([int(r["label"]) for r in t5], dtype=float)
    t5_scores = np.clip(np.asarray([float(r["score"]) for r in t5]), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (t5_scores >= low) & (t5_scores < high if high < 1 else t5_scores <= high)
        if mask.any():
            ece += float(mask.mean() * abs(t5_scores[mask].mean() - t5_labels[mask].mean()))
    confidence = np.maximum(t5_scores, 1.0 - t5_scores)
    keep = confidence >= np.quantile(confidence, 0.20)
    selective_risk = float(np.mean((t5_scores[keep] >= 0.5) != t5_labels[keep])) if keep.any() else 1.0
    source_scope = source_macro_auprc_diagnostics(t5)
    return {
        PRIMARY_ENDPOINT: source_macro_auprc(t5),
        "T1_low_yield_auprc_lt10": _average_precision([int(r["label"]) for r in t1], [float(r["score"]) for r in t1]),
        "T2_ordinal_mae": float(np.mean(np.abs(np.asarray([int(r["label"]) for r in t2]) - np.asarray([int(r["score"]) for r in t2])))),
        "T3_yield_mae": float(np.mean(np.abs(t3_scores - yields))),
        "T3_yield_spearman": _spearman(t3_scores, yields),
        "T4_plate_ndcg": float(np.mean(plate_ndcg)) if plate_ndcg else 0.0,
        "T5_ece": ece,
        "T5_selective_risk_at_80pct_coverage": selective_risk,
        **source_scope,
    }
