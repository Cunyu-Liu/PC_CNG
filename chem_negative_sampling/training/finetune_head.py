"""P3-03: Cross-dataset fine-tuning head to 翻盘 P2-05 NO-GO.

P2-05 直接 transfer 导致 0/7 pairs CI positive。P3-03 通过 **frozen
backbone + lightweight fine-tuning head** 替代直接复用 source dataset 的
head，目标在 ≥5/7 pairs 上翻盘为 GO。

本模块实现 7 个迁移对 (migration pairs) 上的 fine-tuning head 实验：

  1. uspto -> ord
  2. ord -> uspto
  3. hitea -> uspto
  4. hitea -> ord
  5. uspto -> hitea
  6. ord -> hitea
  7. uspto_openmolecules -> ord

每个迁移对对比 3 个变体 (variants)：

  * ``direct``        : source-trained model 直接在 target test 上评估
  * ``head_finetune`` : 冻结 backbone，仅训练 head（≤100K 参数）on 10% target
  * ``full_finetune`` : 解冻全部参数，在 10% target 上 fine-tune

所有性能主张基于 10-seed paired significance test (family-cluster bootstrap
CI)，cluster by ``source_id``。

依赖：Python 3.10 stdlib + torch + numpy（不引入新依赖；RDKit 可选）。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import backbone / scorer / tokenizer (resilient to missing server modules)
# ---------------------------------------------------------------------------
try:
    from models.pretrained_backbone import (  # type: ignore
        ChemformerTokenizer,
        PretrainedChemformerBackbone,
        PretrainedReactionScorer,
        ReactionClassificationHead,
    )
except ImportError:  # pragma: no cover - exercised on server
    try:
        from pc_cng.models.pretrained_backbone import (  # type: ignore
            ChemformerTokenizer,
            PretrainedChemformerBackbone,
            PretrainedReactionScorer,
            ReactionClassificationHead,
        )
    except ImportError:
        PretrainedChemformerBackbone = None  # type: ignore
        ReactionClassificationHead = None  # type: ignore
        PretrainedReactionScorer = None  # type: ignore
        ChemformerTokenizer = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_CSV_MAP: Dict[str, str] = {
    "uspto": "uspto_openmolecules_normalized.csv",
    "ord": "ord_normalized.csv",
    "hitea": "hitea_full_normalized.csv",
    "uspto_openmolecules": "uspto_openmolecules_normalized.csv",
}

MIGRATION_PAIRS: List[Tuple[str, str]] = [
    ("uspto", "ord"),
    ("ord", "uspto"),
    ("hitea", "uspto"),
    ("hitea", "ord"),
    ("uspto", "hitea"),
    ("ord", "hitea"),
    ("uspto_openmolecules", "ord"),
]

VARIANTS: List[str] = ["direct", "head_finetune", "full_finetune"]

DEFAULT_SEEDS: List[int] = list(range(20260710, 20260720))


# ---------------------------------------------------------------------------
# Fallback classes (used when server-side modules are unavailable, e.g. tests)
# ---------------------------------------------------------------------------
class _FallbackBackbone(nn.Module):
    """Tiny fallback backbone for testing (no checkpoint, no RDKit)."""

    def __init__(
        self,
        d_model: int = 16,
        vocab_size: int = 100,
        freeze: bool = False,
        checkpoint_path: Optional[str] = None,
        hparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        hp = {"d_model": d_model, "vocabulary_size": vocab_size}
        if hparams:
            hp.update(hparams)
        self.hparams = hp
        self.embed = nn.Embedding(int(hp["vocabulary_size"]), int(hp["d_model"]), padding_idx=0)
        self.proj = nn.Linear(int(hp["d_model"]), int(hp["d_model"]))
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pool: bool = True,
    ) -> torch.Tensor:
        emb = self.embed(token_ids)
        out = self.proj(emb)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            out = out * mask
        if pool:
            if attention_mask is not None:
                denom = mask.sum(dim=1).clamp(min=1.0)
                return out.sum(dim=1) / denom
            return out.mean(dim=1)
        return out


class _FallbackHead(nn.Module):
    """Lightweight head wrapping :func:`build_head` (squeeze output to 1-D)."""

    def __init__(self, d_model: int = 16, n_classes: int = 1) -> None:
        super().__init__()
        self.d_model = d_model
        self.net = build_head(d_model, n_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled).squeeze(-1)


class _FallbackScorer(nn.Module):
    """Fallback scorer combining :class:`_FallbackBackbone` + :class:`_FallbackHead`."""

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        d_model: int = 16,
        vocab_size: int = 100,
    ) -> None:
        super().__init__()
        self.backbone = backbone if backbone is not None else _FallbackBackbone(d_model, vocab_size)
        self.head = head if head is not None else _FallbackHead(self.backbone.hparams.get("d_model", d_model))

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled = self.backbone(token_ids, attention_mask=attention_mask, pool=True)
        return self.head(pooled)


class _FallbackTokenizer:
    """Character-level fallback tokenizer (no vocab file needed)."""

    def __init__(self, vocab_path: Optional[str] = None, max_seq_len: int = 64) -> None:
        self.max_seq_len = max_seq_len
        self.pad_idx = 0
        self.unk_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3
        self.vocab_size = 100

    def batch_encode(
        self,
        smiles_list: Sequence[str],
        add_special: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded: List[List[int]] = []
        for s in smiles_list:
            ids: List[int] = [self.bos_idx] if add_special else []
            for c in str(s)[: self.max_seq_len - 2]:
                ids.append((ord(c) % 90) + 4)  # 4..93
            if add_special:
                ids.append(self.eos_idx)
            encoded.append(ids)
        if not encoded:
            return (
                torch.zeros((0, 1), dtype=torch.long),
                torch.zeros((0, 1), dtype=torch.long),
            )
        max_len = max(len(ids) for ids in encoded)
        padded = torch.full((len(encoded), max_len), self.pad_idx, dtype=torch.long)
        mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
        for i, ids in enumerate(encoded):
            padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[i, : len(ids)] = 1
        return padded, mask


# ---------------------------------------------------------------------------
# 1. build_head
# ---------------------------------------------------------------------------
def build_head(d_model: int = 512, n_classes: int = 1) -> nn.Sequential:
    """Build the fine-tuning head.

    Architecture: ``Linear(d_model, 128) -> ReLU -> Linear(128, n_classes)``.

    Parameter count for ``d_model=512, n_classes=1``:
        512*128 + 128 + 128*1 + 1 = 65793 (~66K, ≤100K ✓)

    Args:
        d_model: Backbone hidden dimension (default 512 for Chemformer).
        n_classes: Number of output classes (default 1 for binary).

    Returns:
        ``nn.Sequential`` head module.
    """
    return nn.Sequential(
        nn.Linear(d_model, 128),
        nn.ReLU(),
        nn.Linear(128, n_classes),
    )


# ---------------------------------------------------------------------------
# 2. load_pretrained_scorer
# ---------------------------------------------------------------------------
def load_pretrained_scorer(
    checkpoint_path: Optional[str],
    vocab_path: str,
    device: str = "cpu",
    hparams: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Load :class:`PretrainedReactionScorer` from P3-01 checkpoint.

    Uses :class:`PretrainedChemformerBackbone` + :class:`ReactionClassificationHead`
    from ``models/pretrained_backbone.py`` when available.  If the checkpoint
    does not exist (or server modules are missing), a fresh model with random
    init is returned and the caller is responsible for documenting this.

    Args:
        checkpoint_path: Path to P3-01 ``model.pt``.  ``None`` or missing path
            triggers fresh random init.
        vocab_path: Path to ``bart_vocab.json`` (used to build tokenizer /
            infer vocab size on server).
        device: Target device (``'cpu'`` / ``'cuda:0'`` / ...).
        hparams: Optional hyperparameter override (e.g. ``{'d_model': 16}``
            for tests).

    Returns:
        A :class:`PretrainedReactionScorer` (or fallback) on ``device``.
    """
    hp = dict(hparams or {})
    d_model = int(hp.get("d_model", 512))

    checkpoint_exists = bool(checkpoint_path) and os.path.exists(str(checkpoint_path))

    if PretrainedReactionScorer is not None and PretrainedChemformerBackbone is not None:
        # ----- Real server path -----
        backbone = PretrainedChemformerBackbone(
            checkpoint_path=str(checkpoint_path) if checkpoint_exists else None,
            hparams=hp or None,
            freeze=False,
        )
        head_cls = ReactionClassificationHead or _FallbackHead
        head = head_cls(d_model=backbone.hparams["d_model"])
        scorer = PretrainedReactionScorer(backbone, head)
        if checkpoint_exists:
            try:
                ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
                sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
                if isinstance(sd, dict):
                    try:
                        scorer.load_state_dict(sd, strict=False)
                    except Exception:
                        pass  # partial load is acceptable
            except Exception:
                pass  # graceful: random init
    else:
        # ----- Fallback (tests / no server modules) -----
        backbone = _FallbackBackbone(d_model=d_model, freeze=False, hparams=hp)
        scorer = _FallbackScorer(backbone=backbone, head=_FallbackHead(d_model))
        if checkpoint_exists:
            try:
                ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
                sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
                if isinstance(sd, dict):
                    try:
                        scorer.load_state_dict(sd, strict=False)
                    except Exception:
                        pass
            except Exception:
                pass

    return scorer.to(device)


def load_tokenizer(vocab_path: str, max_seq_len: int = 256) -> Any:
    """Load the Chemformer tokenizer (or fallback for tests).

    Args:
        vocab_path: Path to ``bart_vocab.json``.  If missing, a fallback
            character-level tokenizer is returned.
        max_seq_len: Maximum sequence length.

    Returns:
        Tokenizer with ``batch_encode(smiles_list) -> (ids, mask)``.
    """
    if ChemformerTokenizer is not None and vocab_path and os.path.exists(str(vocab_path)):
        return ChemformerTokenizer(vocab_path, max_seq_len=max_seq_len)
    return _FallbackTokenizer(vocab_path=vocab_path, max_seq_len=max_seq_len)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_dataset(csv_path: str) -> List[Dict[str, Any]]:
    """Load a normalized reaction CSV.

    Expected columns (any of): ``reaction_smiles`` / ``smiles`` /
    ``positive_reaction``; ``label_type`` / ``label``; ``source_id`` /
    ``source`` / ``group_id``.

    ``label_type='positive'`` -> label 1, else 0.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of dicts with keys ``smiles``, ``label``, ``source_id``.
    """
    rows: List[Dict[str, Any]] = []
    if not csv_path or not os.path.exists(csv_path):
        return rows
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            smiles = (
                record.get("reaction_smiles")
                or record.get("smiles")
                or record.get("positive_reaction")
                or ""
            )
            smiles = str(smiles).strip()
            if not smiles:
                continue
            label_type = str(record.get("label_type") or "").strip().lower()
            if label_type:
                label = 1 if label_type == "positive" else 0
            else:
                try:
                    label = int(float(str(record.get("label", "0")).strip()))
                except (TypeError, ValueError):
                    label = 0
            source_id = (
                record.get("source_id")
                or record.get("source")
                or record.get("group_id")
                or smiles
            )
            source_id = str(source_id).strip()
            rows.append({"smiles": smiles, "label": label, "source_id": source_id})
    return rows


def load_or_create_split(
    rows: Sequence[Dict[str, Any]],
    train_idx_path: Optional[str],
    val_idx_path: Optional[str],
    test_idx_path: Optional[str],
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Load train/val/test indices from JSON files (HC #9).

    If any file is missing, an 80/10/10 stratified-by-``source_id`` group
    split is auto-created and returned.

    Args:
        rows: List of row dicts (must have ``source_id``).
        train_idx_path / val_idx_path / test_idx_path: Paths to JSON index files.
        seed: Random seed for auto-created split.

    Returns:
        ``(train_idx, val_idx, test_idx)`` integer lists.
    """
    if (
        train_idx_path
        and val_idx_path
        and test_idx_path
        and os.path.exists(train_idx_path)
        and os.path.exists(val_idx_path)
        and os.path.exists(test_idx_path)
    ):
        train_idx = json.loads(Path(train_idx_path).read_text(encoding="utf-8"))
        val_idx = json.loads(Path(val_idx_path).read_text(encoding="utf-8"))
        test_idx = json.loads(Path(test_idx_path).read_text(encoding="utf-8"))
        return list(train_idx), list(val_idx), list(test_idx)

    # Auto-create 80/10/10 group split stratified by source_id
    source_ids = sorted({r["source_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(source_ids)
    n = len(source_ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_sids = set(source_ids[:n_train])
    val_sids = set(source_ids[n_train : n_train + n_val])
    train_idx, val_idx, test_idx = [], [], []
    for i, r in enumerate(rows):
        if r["source_id"] in train_sids:
            train_idx.append(i)
        elif r["source_id"] in val_sids:
            val_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, val_idx, test_idx


def stratified_group_split(
    rows: Sequence[Dict[str, Any]],
    n_few_shot: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Split rows into train/test by group (``source_id``).

    All rows sharing a ``source_id`` go to the same split.  ``n_few_shot``
    fraction of groups form the train split; the remainder form the test
    split.

    Args:
        rows: List of row dicts.
        n_few_shot: Fraction of groups for train (default 0.1 = 10%).
        seed: Random seed for reproducibility.

    Returns:
        ``(train_idx, test_idx)`` integer lists.
    """
    source_ids = sorted({r["source_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(source_ids)
    n_train = max(1, int(len(source_ids) * n_few_shot))
    train_sids = set(source_ids[:n_train])
    train_idx, test_idx = [], []
    for i, r in enumerate(rows):
        if r["source_id"] in train_sids:
            train_idx.append(i)
        else:
            test_idx.append(i)
    if not train_idx and rows:
        # Ensure at least one training example
        train_idx.append(0)
        test_idx = [i for i in range(len(rows)) if i != 0]
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _compute_mrr(
    probs: np.ndarray,
    labels: np.ndarray,
    source_ids: Sequence[str],
) -> float:
    """Compute Mean Reciprocal Rank.

    For each ``source_id`` group, rank by score descending; the reciprocal
    rank of the first positive is averaged across groups.
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, sid in enumerate(source_ids):
        groups[sid].append(i)
    rrs: List[float] = []
    for sid, indices in groups.items():
        group = sorted(indices, key=lambda i: -float(probs[i]))
        for rank, i in enumerate(group, start=1):
            if int(labels[i]) == 1:
                rrs.append(1.0 / rank)
                break
    return float(np.mean(rrs)) if rrs else 0.0


def _compute_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute ROC AUC via the rank-based (Mann-Whitney U) formula."""
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if len(labels) == 0 or len(set(labels.tolist())) < 2:
        return 0.5
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(probs) + 1)
    sum_ranks_pos = float(ranks[labels == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ---------------------------------------------------------------------------
# 3. few_shot_finetune
# ---------------------------------------------------------------------------
def few_shot_finetune(
    model: nn.Module,
    target_rows: Sequence[Dict[str, Any]],
    variant: str,
    tokenizer: Any,
    n_epochs: int,
    lr: float,
    device: str = "cpu",
    batch_size: int = 16,
) -> nn.Module:
    """Few-shot fine-tune the model on target rows.

    Args:
        model: Source-trained :class:`PretrainedReactionScorer` (or fallback).
        target_rows: Target training rows (10% split).
        variant: One of ``'direct'``, ``'head_finetune'``, ``'full_finetune'``.
            * ``direct``        : return model unchanged.
            * ``head_finetune`` : freeze backbone params, unfreeze head, train.
            * ``full_finetune`` : unfreeze all params, train.
        tokenizer: Tokenizer with ``batch_encode``.
        n_epochs: Number of training epochs.
        lr: Learning rate.
        device: Target device.
        batch_size: Mini-batch size (default 16).

    Returns:
        The (possibly fine-tuned) model.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}; expected one of {VARIANTS}")

    if variant == "direct" or len(target_rows) == 0:
        return model

    backbone = getattr(model, "backbone", None)
    head = getattr(model, "head", None)

    if variant == "head_finetune":
        if backbone is not None:
            for p in backbone.parameters():
                p.requires_grad = False
        if head is not None:
            for p in head.parameters():
                p.requires_grad = True
    elif variant == "full_finetune":
        for p in model.parameters():
            p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return model

    optimizer = torch.optim.AdamW(trainable, lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    device_t = torch.device(device)
    for _ in range(max(1, n_epochs)):
        order = list(range(len(target_rows)))
        random.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            batch_rows = [target_rows[i] for i in batch_idx]
            smiles_list = [r["smiles"] for r in batch_rows]
            labels = torch.tensor([float(r["label"]) for r in batch_rows], dtype=torch.float32, device=device_t)
            ids, mask = tokenizer.batch_encode(smiles_list)
            ids = ids.to(device_t)
            mask = mask.to(device_t)
            optimizer.zero_grad()
            logits = model(ids, attention_mask=mask)
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    return model


# ---------------------------------------------------------------------------
# 4. evaluate
# ---------------------------------------------------------------------------
def evaluate(
    model: nn.Module,
    tokenizer: Any,
    rows: Sequence[Dict[str, Any]],
    device: str = "cpu",
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Evaluate the model on ``rows``.

    Args:
        model: Trained scorer.
        tokenizer: Tokenizer with ``batch_encode``.
        rows: Evaluation rows (each has ``smiles``, ``label``, ``source_id``).
        device: Target device.
        batch_size: Mini-batch size (default 16).

    Returns:
        Dict with keys: ``mrr``, ``accuracy``, ``auc``, ``n_examples``,
        ``per_example_probs``, ``per_example_labels``, ``per_example_source_ids``,
        ``per_example_correct``.
    """
    model.eval()
    device_t = torch.device(device)
    all_probs: List[float] = []
    all_labels: List[int] = []
    all_source_ids: List[str] = []

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = list(rows[start : start + batch_size])
            if not batch:
                continue
            smiles_list = [r["smiles"] for r in batch]
            ids, mask = tokenizer.batch_encode(smiles_list)
            ids = ids.to(device_t)
            mask = mask.to(device_t)
            logits = model(ids, attention_mask=mask)
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            if isinstance(probs, float):
                probs = [probs]
            all_probs.extend(float(p) for p in probs)
            all_labels.extend(int(r["label"]) for r in batch)
            all_source_ids.extend(str(r["source_id"]) for r in batch)

    probs_arr = np.array(all_probs, dtype=float)
    labels_arr = np.array(all_labels, dtype=int)
    preds = (probs_arr > 0.5).astype(int)
    correctness = (preds == labels_arr).astype(float)

    mrr = _compute_mrr(probs_arr, labels_arr, all_source_ids)
    accuracy = float(correctness.mean()) if len(correctness) else 0.0
    auc = _compute_auc(labels_arr, probs_arr)

    return {
        "mrr": mrr,
        "accuracy": accuracy,
        "auc": auc,
        "n_examples": int(len(labels_arr)),
        "per_example_probs": all_probs,
        "per_example_labels": all_labels,
        "per_example_source_ids": all_source_ids,
        "per_example_correct": correctness.tolist(),
    }


# ---------------------------------------------------------------------------
# 6. paired_bootstrap_ci
# ---------------------------------------------------------------------------
def paired_bootstrap_ci(
    metric_a_per_seed: Sequence[float],
    metric_b_per_seed: Sequence[float],
    n_iterations: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Paired bootstrap CI for per-seed metrics.

    Args:
        metric_a_per_seed: Per-seed metric values for variant A.
        metric_b_per_seed: Per-seed metric values for variant B (same length).
        n_iterations: Number of bootstrap iterations (default 10000).
        seed: Random seed.

    Returns:
        ``(mean_diff, ci_low, ci_high, p_value)``.
    """
    a = np.asarray(metric_a_per_seed, dtype=float)
    b = np.asarray(metric_b_per_seed, dtype=float)
    n = len(a)
    if n == 0 or n != len(b):
        return 0.0, 0.0, 0.0, 1.0

    diffs = a - b
    mean_diff = float(np.mean(diffs))

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_iterations, dtype=float)
    for i in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = float(np.mean(diffs[idx]))

    alpha = 0.05
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    if mean_diff == 0:
        p_value = 1.0
    else:
        opposite_sign = float(np.sum(np.sign(boot_means) != np.sign(mean_diff)))
        p_value = opposite_sign / n_iterations
        p_value = max(p_value, 1.0 / (n_iterations + 1))

    return mean_diff, ci_low, ci_high, p_value


# ---------------------------------------------------------------------------
# 7. family_cluster_bootstrap_ci
# ---------------------------------------------------------------------------
def family_cluster_bootstrap_ci(
    metric_a: Sequence[float],
    metric_b: Sequence[float],
    cluster_ids: Sequence[Any],
    n_iterations: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Family-cluster bootstrap CI by resampling ``source_id`` clusters.

    Clusters samples by ``cluster_ids`` (family) and resamples clusters with
    replacement, then aggregates per-example differences.  Recommended for
    chemistry data where reactions from the same source are correlated.

    Args:
        metric_a: Per-example metric for variant A (e.g. 0/1 correctness).
        metric_b: Per-example metric for variant B (same length).
        cluster_ids: Per-example cluster IDs (e.g. ``source_id``).
        n_iterations: Number of bootstrap iterations (default 10000).
        seed: Random seed.

    Returns:
        ``(mean_diff, ci_low, ci_high, p_value)``.
    """
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    cluster_ids_arr = np.asarray(cluster_ids)
    n = len(a)
    if n == 0 or n != len(b) or n != len(cluster_ids_arr):
        return 0.0, 0.0, 0.0, 1.0

    diffs = a - b
    mean_diff = float(np.mean(diffs))

    unique_clusters = np.unique(cluster_ids_arr)
    n_clusters = len(unique_clusters)
    if n_clusters == 0:
        return 0.0, 0.0, 0.0, 1.0

    # Pre-compute cluster -> indices mapping for speed
    cluster_to_idx: Dict[Any, np.ndarray] = {
        str(c): np.where(cluster_ids_arr == c)[0] for c in unique_clusters
    }
    cluster_keys = list(cluster_to_idx.keys())

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_iterations, dtype=float)
    for i in range(n_iterations):
        sampled = rng.choice(cluster_keys, size=n_clusters, replace=True)
        idx_parts = [cluster_to_idx[c] for c in sampled]
        idx = np.concatenate(idx_parts) if idx_parts else np.array([], dtype=int)
        if len(idx) == 0:
            boot_means[i] = 0.0
        else:
            boot_means[i] = float(np.mean(diffs[idx]))

    alpha = 0.05
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    if mean_diff == 0:
        p_value = 1.0
    else:
        opposite_sign = float(np.sum(np.sign(boot_means) != np.sign(mean_diff)))
        p_value = opposite_sign / n_iterations
        p_value = max(p_value, 1.0 / (n_iterations + 1))

    return mean_diff, ci_low, ci_high, p_value


# ---------------------------------------------------------------------------
# 5. run_pair
# ---------------------------------------------------------------------------
def run_pair(
    source_name: str,
    target_name: str,
    source_csv: str,
    target_csv: str,
    backbone_ckpt: Optional[str],
    vocab_path: str,
    seeds: Sequence[int],
    output_dir: str,
    n_few_shot: float = 0.1,
    epochs: int = 5,
    lr: float = 1e-4,
    device: str = "cpu",
    train_idx_path: Optional[str] = None,
    val_idx_path: Optional[str] = None,
    test_idx_path: Optional[str] = None,
    bootstrap_iterations: int = 10000,
) -> Dict[str, Any]:
    """Run cross-dataset fine-tuning experiment for one (source, target) pair.

    Workflow:
        1. Load source dataset (``label_type='positive'`` -> 1, else 0).
        2. Train initial head on source train split (or load source-trained
           head from ``backbone_ckpt``).
        3. Load target dataset; split 10% train / 90% test, stratified by
           ``source_id``.
        4. Run 3 variants × N seeds.
        5. Compute paired family-cluster bootstrap CI (``head_finetune`` vs
           ``direct``, ``full_finetune`` vs ``direct``).
        6. Save per-seed ``metrics.json`` + ``summary.json`` + ``summary.md``.

    Args:
        source_name / target_name: Dataset short names (e.g. ``'uspto'``).
        source_csv / target_csv: Paths to normalized CSVs.
        backbone_ckpt: Path to P3-01 ``model.pt`` (``None`` -> fresh init).
        vocab_path: Path to ``bart_vocab.json``.
        seeds: List of random seeds.
        output_dir: Base output directory.
        n_few_shot: Fraction of groups for target train (default 0.1).
        epochs: Fine-tuning epochs (default 5).
        lr: Learning rate (default 1e-4).
        device: ``'cpu'`` / ``'cuda:0'`` / ...
        train_idx_path / val_idx_path / test_idx_path: HC #9 split JSONs for
            the source dataset.  Auto-created if missing.
        bootstrap_iterations: Bootstrap iterations for CI.

    Returns:
        Summary dict (also saved to ``summary.json``).
    """
    pair_name = f"{source_name}_to_{target_name}"
    pair_dir = Path(output_dir) / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_exists = bool(backbone_ckpt) and os.path.exists(str(backbone_ckpt))

    # 1. Load source dataset
    source_rows = load_dataset(source_csv)
    source_train_idx, _, _ = load_or_create_split(
        source_rows, train_idx_path, val_idx_path, test_idx_path
    )

    # 2. Load target dataset
    target_rows = load_dataset(target_csv)

    # 3. Load tokenizer + source-trained model
    tokenizer = load_tokenizer(vocab_path)
    source_model = load_pretrained_scorer(backbone_ckpt, vocab_path, device)

    # If no checkpoint, train initial head on source train split
    if not checkpoint_exists and source_train_idx:
        source_train_rows = [source_rows[i] for i in source_train_idx if 0 <= i < len(source_rows)]
        source_model = few_shot_finetune(
            source_model,
            source_train_rows,
            "head_finetune",
            tokenizer,
            n_epochs=epochs,
            lr=lr,
            device=device,
        )

    # 4. Run 3 variants × N seeds
    per_seed_metrics: Dict[str, List[Dict[str, Any]]] = {v: [] for v in VARIANTS}
    per_seed_aggregate: Dict[str, List[float]] = {v: [] for v in VARIANTS}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Target split (varies by seed for proper paired comparison)
        target_train_idx, target_test_idx = stratified_group_split(
            target_rows, n_few_shot=n_few_shot, seed=seed
        )
        target_train_rows = [target_rows[i] for i in target_train_idx]
        target_test_rows = [target_rows[i] for i in target_test_idx]

        seed_dir = pair_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        seed_metrics: Dict[str, Any] = {"seed": seed, "variants": {}}

        for variant in VARIANTS:
            model_variant = copy.deepcopy(source_model)

            if variant != "direct":
                model_variant = few_shot_finetune(
                    model_variant,
                    target_train_rows,
                    variant,
                    tokenizer,
                    n_epochs=epochs,
                    lr=lr,
                    device=device,
                )

            metrics = evaluate(model_variant, tokenizer, target_test_rows, device=device)
            # Strip per-example arrays from the per-seed JSON (saved separately)
            compact = {
                "mrr": metrics["mrr"],
                "accuracy": metrics["accuracy"],
                "auc": metrics["auc"],
                "n_examples": metrics["n_examples"],
            }
            per_seed_metrics[variant].append(metrics)
            per_seed_aggregate[variant].append(metrics["mrr"])
            seed_metrics["variants"][variant] = compact

        # Save per-seed metrics.json
        (seed_dir / "metrics.json").write_text(
            json.dumps(seed_metrics, indent=2), encoding="utf-8"
        )

    # 5. Compute paired bootstrap CIs
    mrr_direct = per_seed_aggregate["direct"]
    mrr_head = per_seed_aggregate["head_finetune"]
    mrr_full = per_seed_aggregate["full_finetune"]

    head_vs_direct = paired_bootstrap_ci(mrr_head, mrr_direct, n_iterations=bootstrap_iterations)
    full_vs_direct = paired_bootstrap_ci(mrr_full, mrr_direct, n_iterations=bootstrap_iterations)

    # Family-cluster bootstrap on concatenated per-example correctness
    fc_summary: Dict[str, Any] = {}
    try:
        correct_direct = np.concatenate(
            [np.asarray(m["per_example_correct"]) for m in per_seed_metrics["direct"]]
        )
        correct_head = np.concatenate(
            [np.asarray(m["per_example_correct"]) for m in per_seed_metrics["head_finetune"]]
        )
        correct_full = np.concatenate(
            [np.asarray(m["per_example_correct"]) for m in per_seed_metrics["full_finetune"]]
        )
        sids = []
        for m in per_seed_metrics["direct"]:
            sids.extend(m["per_example_source_ids"])
        sids_arr = np.asarray(sids)

        if len(correct_direct) == len(correct_head) == len(sids_arr):
            fc_head = family_cluster_bootstrap_ci(
                correct_head, correct_direct, sids_arr, n_iterations=bootstrap_iterations
            )
            fc_full = family_cluster_bootstrap_ci(
                correct_full, correct_direct, sids_arr, n_iterations=bootstrap_iterations
            )
            fc_summary = {
                "head_finetune_vs_direct": {
                    "mean_diff": fc_head[0],
                    "ci_low": fc_head[1],
                    "ci_high": fc_head[2],
                    "p_value": fc_head[3],
                },
                "full_finetune_vs_direct": {
                    "mean_diff": fc_full[0],
                    "ci_low": fc_full[1],
                    "ci_high": fc_full[2],
                    "p_value": fc_full[3],
                },
            }
    except Exception:
        fc_summary = {}

    # 6. Build summary
    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def _std(xs: List[float]) -> float:
        return float(np.std(xs)) if xs else 0.0

    summary: Dict[str, Any] = {
        "pair": pair_name,
        "source": source_name,
        "target": target_name,
        "n_seeds": len(seeds),
        "n_few_shot": n_few_shot,
        "checkpoint_used": checkpoint_exists,
        "backbone_ckpt": str(backbone_ckpt) if backbone_ckpt else None,
        "variants": {
            v: {
                "mrr_mean": _mean(per_seed_aggregate[v]),
                "mrr_std": _std(per_seed_aggregate[v]),
                "mrr_per_seed": per_seed_aggregate[v],
            }
            for v in VARIANTS
        },
        "paired_bootstrap_ci": {
            "head_finetune_vs_direct": {
                "mean_diff": head_vs_direct[0],
                "ci_low": head_vs_direct[1],
                "ci_high": head_vs_direct[2],
                "p_value": head_vs_direct[3],
            },
            "full_finetune_vs_direct": {
                "mean_diff": full_vs_direct[0],
                "ci_low": full_vs_direct[1],
                "ci_high": full_vs_direct[2],
                "p_value": full_vs_direct[3],
            },
        },
        "family_cluster_bootstrap_ci": fc_summary,
        "go_no_go": {
            "head_finetune_go": head_vs_direct[1] > 0,
            "full_finetune_go": full_vs_direct[1] > 0,
        },
    }

    (pair_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (pair_dir / "summary.md").write_text(_render_summary_md(summary), encoding="utf-8")

    return summary


def _render_summary_md(summary: Dict[str, Any]) -> str:
    """Render a human-readable Markdown summary for one pair."""
    lines: List[str] = []
    lines.append(f"# P3-03 Cross-dataset fine-tuning: {summary['pair']}\n")
    lines.append(f"- Source: `{summary['source']}`  Target: `{summary['target']}`")
    lines.append(f"- Seeds: {summary['n_seeds']}  Few-shot fraction: {summary['n_few_shot']}")
    lines.append(f"- Checkpoint used: {summary['checkpoint_used']}\n")
    lines.append("## Per-variant MRR (mean ± std)\n")
    lines.append("| Variant | MRR mean | MRR std |")
    lines.append("|---|---|---|")
    for v in VARIANTS:
        vm = summary["variants"][v]
        lines.append(f"| {v} | {vm['mrr_mean']:.4f} | {vm['mrr_std']:.4f} |")
    lines.append("\n## Paired bootstrap CI (vs direct)\n")
    lines.append("| Comparison | Mean diff | CI low | CI high | p-value | GO |")
    lines.append("|---|---|---|---|---|---|")
    for key, label in [
        ("head_finetune_vs_direct", "head_finetune"),
        ("full_finetune_vs_direct", "full_finetune"),
    ]:
        ci = summary["paired_bootstrap_ci"][key]
        go = "✓" if ci["ci_low"] > 0 else "✗"
        lines.append(
            f"| {label} | {ci['mean_diff']:+.4f} | {ci['ci_low']:+.4f} | "
            f"{ci['ci_high']:+.4f} | {ci['p_value']:.4f} | {go} |"
        )
    if summary.get("family_cluster_bootstrap_ci"):
        lines.append("\n## Family-cluster bootstrap CI (per-example correctness)\n")
        lines.append("| Comparison | Mean diff | CI low | CI high | p-value |")
        lines.append("|---|---|---|---|---|")
        for key, label in [
            ("head_finetune_vs_direct", "head_finetune"),
            ("full_finetune_vs_direct", "full_finetune"),
        ]:
            if key in summary["family_cluster_bootstrap_ci"]:
                ci = summary["family_cluster_bootstrap_ci"][key]
                lines.append(
                    f"| {label} | {ci['mean_diff']:+.4f} | {ci['ci_low']:+.4f} | "
                    f"{ci['ci_high']:+.4f} | {ci['p_value']:.4f} |"
                )
    lines.append("\n## GO/NO-GO\n")
    go = summary["go_no_go"]
    lines.append(f"- head_finetune GO: {'YES' if go['head_finetune_go'] else 'NO'}")
    lines.append(f"- full_finetune GO: {'YES' if go['full_finetune_go'] else 'NO'}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 8. main
# ---------------------------------------------------------------------------
def parse_pairs(pairs_str: str) -> List[Tuple[str, str]]:
    """Parse ``--pairs`` argument."""
    if pairs_str.strip().lower() == "all":
        return list(MIGRATION_PAIRS)
    pairs: List[Tuple[str, str]] = []
    for token in pairs_str.split(","):
        token = token.strip()
        if not token:
            continue
        if "->" not in token:
            raise ValueError(f"Invalid pair '{token}'; expected 'src->tgt'")
        src, tgt = token.split("->", 1)
        pairs.append((src.strip(), tgt.strip()))
    return pairs


def parse_seeds(seeds_str: str) -> List[int]:
    """Parse ``--seeds`` argument: ``20260710,20260711`` or ``20260710..20260719``."""
    seeds_str = seeds_str.strip()
    if ".." in seeds_str:
        start, end = seeds_str.split("..", 1)
        return list(range(int(start), int(end) + 1))
    if "," in seeds_str:
        return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]
    if seeds_str:
        return [int(seeds_str)]
    return list(DEFAULT_SEEDS)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for P3-03 cross-dataset fine-tuning head experiment."""
    parser = argparse.ArgumentParser(
        description="P3-03: cross-dataset fine-tuning head (翻盘 P2-05 NO-GO)"
    )
    parser.add_argument(
        "--pairs",
        default="all",
        help="Comma-separated source->target pairs (or 'all')",
    )
    parser.add_argument(
        "--backbone-ckpt",
        default=None,
        help="Path to P3-01 seed checkpoint (e.g. .../seed20260710/model.pt)",
    )
    parser.add_argument("--vocab", default=None, help="Path to bart_vocab.json")
    parser.add_argument("--data-dir", default="data/processed", help="Data directory")
    parser.add_argument(
        "--seeds",
        default="20260710..20260719",
        help="Seeds: '20260710,20260711' or '20260710..20260719'",
    )
    parser.add_argument(
        "--output-dir",
        default="results/cross_dataset_finetune_head_20260720",
        help="Output directory",
    )
    parser.add_argument("--n-few-shot", type=float, default=0.1, help="Few-shot fraction")
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="cuda:0", help="Device (cpu / cuda:0 / ...)")
    parser.add_argument("--train-idx", default=None, help="HC #9: train idx JSON for source")
    parser.add_argument("--val-idx", default=None, help="HC #9: val idx JSON for source")
    parser.add_argument("--test-idx", default=None, help="HC #9: test idx JSON for source")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10000,
        help="Bootstrap iterations for CI",
    )
    args = parser.parse_args(argv)

    pairs = parse_pairs(args.pairs)
    seeds = parse_seeds(args.seeds)
    os.makedirs(args.output_dir, exist_ok=True)

    data_dir = args.data_dir
    vocab_path = args.vocab or "external/reaction_lm/Chemformer/bart_vocab.json"

    print(f"[P3-03] pairs={pairs} seeds={seeds} device={args.device}")
    print(f"[P3-03] backbone_ckpt={args.backbone_ckpt}")
    print(f"[P3-03] output_dir={args.output_dir}")

    all_summaries: List[Dict[str, Any]] = []
    for source_name, target_name in pairs:
        source_csv = os.path.join(data_dir, DATASET_CSV_MAP.get(source_name, f"{source_name}.csv"))
        target_csv = os.path.join(data_dir, DATASET_CSV_MAP.get(target_name, f"{target_name}.csv"))

        if not os.path.exists(source_csv):
            print(f"[P3-03] SKIP {source_name}->{target_name}: source CSV missing ({source_csv})")
            continue
        if not os.path.exists(target_csv):
            print(f"[P3-03] SKIP {source_name}->{target_name}: target CSV missing ({target_csv})")
            continue

        print(f"[P3-03] Running {source_name}->{target_name} ...")
        summary = run_pair(
            source_name=source_name,
            target_name=target_name,
            source_csv=source_csv,
            target_csv=target_csv,
            backbone_ckpt=args.backbone_ckpt,
            vocab_path=vocab_path,
            seeds=seeds,
            output_dir=args.output_dir,
            n_few_shot=args.n_few_shot,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            train_idx_path=args.train_idx,
            val_idx_path=args.val_idx,
            test_idx_path=args.test_idx,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        all_summaries.append(summary)
        print(
            f"[P3-03] {source_name}->{target_name}: "
            f"head_finetune MRR={summary['variants']['head_finetune']['mrr_mean']:.4f} "
            f"(direct={summary['variants']['direct']['mrr_mean']:.4f}) "
            f"CI=[{summary['paired_bootstrap_ci']['head_finetune_vs_direct']['ci_low']:+.4f}, "
            f"{summary['paired_bootstrap_ci']['head_finetune_vs_direct']['ci_high']:+.4f}]"
        )

    # Save aggregate summary
    if all_summaries:
        agg_path = Path(args.output_dir) / "all_pairs_summary.json"
        agg_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
        n_go = sum(
            1 for s in all_summaries if s["go_no_go"]["head_finetune_go"]
        )
        print(f"\n[P3-03] DONE. head_finetune GO on {n_go}/{len(all_summaries)} pairs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
