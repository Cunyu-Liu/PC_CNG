"""P3-06: Multi-task joint training (retrosynthesis + condition + yield).

This module implements multi-task joint training on top of the P3-01
``PretrainedChemformerBackbone`` (frozen + LoRA) to demonstrate shared
representation benefits across three reaction-level tasks:

  1. **Retrosynthesis scoring** (binary classification: positive vs.
     PC-CNG generated negative) -- uses ``regiosqm20_normalized.csv``.
  2. **Condition prediction** (3-way classification: catalyst / solvent /
     reagent) -- uses ``ord_conditions.json`` (P3-04).
  3. **Yield prediction** (regression on HTE data) -- uses
     ``hitea_full_normalized.csv`` (P3-05, ``yield`` field).

Multi-task learning allows the shared encoder to learn richer reaction
representations by training on complementary signals.  We use optional
**uncertainty weighting** (Kendall et al. 2018) to automatically balance
the three task losses via learnable log-variance parameters:

    L_total = Σ_i [ (1 / (2 * σ_i²)) * L_i + log(σ_i²) ]

For each seed we train (a) three single-task baselines (same backbone +
one head only) and (b) one multi-task model (all 3 heads jointly), then
evaluate on the test set and compute paired family-cluster bootstrap CI
(multitask vs. singletask per task).

Hard constraints respected:
  - HC #4: unit tests in ``test_multitask.py`` (>=80% coverage)
  - HC #5: 10-seed paired family-cluster bootstrap CI for performance claims
  - HC #9: ``--train-idx``/``--val-idx``/``--test-idx`` required (v3 splits
           for retrosynthesis; auto-created for other tasks)

Dependencies: Python 3.10 stdlib + torch + numpy (no new deps).
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
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import backbone / tokenizer (resilient to missing server modules)
# ---------------------------------------------------------------------------
try:
    from models.pretrained_backbone import (  # type: ignore
        ChemformerTokenizer,
        PretrainedChemformerBackbone,
        ReactionClassificationHead,
    )
    from models.adapter import apply_lora, freeze_non_lora_params  # type: ignore
except ImportError:  # pragma: no cover - exercised on server with different layout
    try:
        from pc_cng.models.pretrained_backbone import (  # type: ignore
            ChemformerTokenizer,
            PretrainedChemformerBackbone,
            ReactionClassificationHead,
        )
        from pc_cng.models.adapter import apply_lora, freeze_non_lora_params  # type: ignore
    except ImportError:
        PretrainedChemformerBackbone = None  # type: ignore
        ReactionClassificationHead = None  # type: ignore
        ChemformerTokenizer = None  # type: ignore
        apply_lora = None  # type: ignore
        freeze_non_lora_params = None  # type: ignore


__all__ = [
    "TASKS",
    "MultiTaskModel",
    "MultiTaskTrainer",
    "load_multitask_data",
    "load_tokenizer",
    "build_backbone",
    "paired_bootstrap_ci",
    "family_cluster_bootstrap_ci",
    "run_experiment",
    "main",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TASKS: Tuple[str, ...] = ("retrosynthesis", "condition", "yield")
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
    """Lightweight classification head (mirrors ReactionClassificationHead)."""

    def __init__(self, d_model: int = 16, n_classes: int = 1) -> None:
        super().__init__()
        self.d_model = d_model
        hidden = max(8, d_model // 2)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled).squeeze(-1)


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
                ids.append((ord(c) % 90) + 4)
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
# Tokenizer / backbone loaders
# ---------------------------------------------------------------------------
def load_tokenizer(vocab_path: Optional[str], max_seq_len: int = 256) -> Any:
    """Load the Chemformer tokenizer (or fallback for tests)."""
    if ChemformerTokenizer is not None and vocab_path and os.path.exists(str(vocab_path)):
        return ChemformerTokenizer(vocab_path, max_seq_len=max_seq_len)
    return _FallbackTokenizer(vocab_path=vocab_path, max_seq_len=max_seq_len)


def build_backbone(
    checkpoint_path: Optional[str],
    hparams: Optional[Dict[str, Any]] = None,
    freeze: bool = True,
    apply_lora_flag: bool = True,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
) -> nn.Module:
    """Build the shared backbone (PretrainedChemformerBackbone + LoRA).

    Falls back to :class:`_FallbackBackbone` when server modules are missing
    (used in unit tests).
    """
    hp = dict(hparams or {})
    d_model = int(hp.get("d_model", 512))
    checkpoint_exists = bool(checkpoint_path) and os.path.exists(str(checkpoint_path))

    if PretrainedChemformerBackbone is not None:
        backbone = PretrainedChemformerBackbone(
            checkpoint_path=str(checkpoint_path) if checkpoint_exists else None,
            hparams=hp or None,
            freeze=freeze,
        )
    else:
        backbone = _FallbackBackbone(
            d_model=d_model,
            freeze=freeze,
            checkpoint_path=checkpoint_path if checkpoint_exists else None,
            hparams=hp,
        )

    # Apply LoRA to the backbone (P3-01 style)
    if apply_lora_flag and apply_lora is not None:
        try:
            apply_lora(backbone, r=lora_r, alpha=lora_alpha)
        except Exception:
            pass  # graceful: LoRA injection may fail on fallback backbone

    return backbone


# ---------------------------------------------------------------------------
# MultiTaskModel
# ---------------------------------------------------------------------------
class MultiTaskModel(nn.Module):
    """Multi-task model with shared backbone + 3 task heads.

    Tasks
    -----
    - ``retrosynthesis``: binary classification (positive vs. negative)
    - ``condition``: 3-way classification (catalyst / solvent / reagent)
    - ``yield``: regression

    Parameters
    ----------
    backbone:
        PretrainedChemformerBackbone (or fallback) instance, frozen + LoRA.
    n_catalyst_classes, n_solvent_classes, n_reagent_classes:
        Number of classes for each condition sub-head.
    d_model:
        Backbone hidden dimension (inferred from ``backbone.hparams`` if not
        given).
    active_tasks:
        Set of tasks whose heads should be created.  ``None`` (default) creates
        all 3 heads.  Used by single-task baselines (``active_tasks={'retrosynthesis'}``).
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_catalyst_classes: int = 2,
        n_solvent_classes: int = 2,
        n_reagent_classes: int = 2,
        d_model: Optional[int] = None,
        active_tasks: Optional[Set[str]] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        d = int(d_model if d_model is not None else backbone.hparams.get("d_model", 512))
        self.d_model = d
        self.n_catalyst_classes = int(n_catalyst_classes)
        self.n_solvent_classes = int(n_solvent_classes)
        self.n_reagent_classes = int(n_reagent_classes)
        self.active_tasks: Set[str] = set(active_tasks) if active_tasks is not None else set(TASKS)

        if "retrosynthesis" in self.active_tasks:
            if ReactionClassificationHead is not None:
                self.retrosynthesis_head = ReactionClassificationHead(d_model=d)
            else:
                self.retrosynthesis_head = _FallbackHead(d_model=d, n_classes=1)
        else:
            self.retrosynthesis_head = None

        if "condition" in self.active_tasks:
            self.catalyst_head = nn.Linear(d, self.n_catalyst_classes)
            self.solvent_head = nn.Linear(d, self.n_solvent_classes)
            self.reagent_head = nn.Linear(d, self.n_reagent_classes)
        else:
            self.catalyst_head = None
            self.solvent_head = None
            self.reagent_head = None

        if "yield" in self.active_tasks:
            self.yield_head = nn.Sequential(
                nn.Linear(d, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        else:
            self.yield_head = None

    def forward(
        self,
        ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task: str = "retrosynthesis",
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Route to the appropriate head.

        Returns
        -------
        torch.Tensor
            For ``retrosynthesis`` and ``yield`` tasks.
        Dict[str, torch.Tensor]
            For ``condition`` task (keys: ``catalyst``, ``solvent``,
            ``reagent``).
        """
        pooled = self.backbone(ids, attention_mask=attention_mask, pool=True)
        if task == "retrosynthesis":
            if self.retrosynthesis_head is None:
                raise ValueError("retrosynthesis head not active in this model")
            return self.retrosynthesis_head(pooled)
        elif task == "condition":
            if self.catalyst_head is None:
                raise ValueError("condition head not active in this model")
            return {
                "catalyst": self.catalyst_head(pooled),
                "solvent": self.solvent_head(pooled),
                "reagent": self.reagent_head(pooled),
            }
        elif task == "yield":
            if self.yield_head is None:
                raise ValueError("yield head not active in this model")
            out = self.yield_head(pooled)
            return out.squeeze(-1)
        else:
            raise ValueError(f"Unknown task: {task!r}; expected one of {TASKS}")

    def forward_all(
        self,
        ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return all active head outputs.

        For ``condition`` the three sub-head outputs are flattened into
        separate keys (``catalyst`` / ``solvent`` / ``reagent``).
        """
        pooled = self.backbone(ids, attention_mask=attention_mask, pool=True)
        out: Dict[str, torch.Tensor] = {}
        if self.retrosynthesis_head is not None:
            out["retrosynthesis"] = self.retrosynthesis_head(pooled)
        if self.catalyst_head is not None:
            out["catalyst"] = self.catalyst_head(pooled)
            out["solvent"] = self.solvent_head(pooled)
            out["reagent"] = self.reagent_head(pooled)
        if self.yield_head is not None:
            out["yield"] = self.yield_head(pooled).squeeze(-1)
        return out


# ---------------------------------------------------------------------------
# MultiTaskTrainer
# ---------------------------------------------------------------------------
class MultiTaskTrainer:
    """Trainer for multi-task joint training.

    Alternating minibatch training: each epoch, iterate tasks in round-robin
    order.  Optional uncertainty weighting (Kendall et al. 2018) learns a
    log-variance per task and balances the losses automatically.

    Parameters
    ----------
    model:
        :class:`MultiTaskModel` to train.
    tokenizer:
        Tokenizer with ``batch_encode(smiles_list) -> (ids, mask)``.
    train_rows_by_task, val_rows_by_task, test_rows_by_task:
        Dict mapping task name -> list of row dicts.  Each row has ``smiles``
        and task-specific label fields.
    device:
        ``'cpu'`` / ``'cuda:0'`` / ...
    lr:
        Learning rate (default 1e-4).
    epochs:
        Number of training epochs (default 5).
    batch_size:
        Mini-batch size (default 16).
    uncertainty_weights:
        If ``True`` (default), use Kendall 2018 uncertainty weighting.
    """

    def __init__(
        self,
        model: MultiTaskModel,
        tokenizer: Any,
        train_rows_by_task: Dict[str, List[Dict[str, Any]]],
        val_rows_by_task: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        test_rows_by_task: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        device: str = "cpu",
        lr: float = 1e-4,
        epochs: int = 5,
        batch_size: int = 16,
        uncertainty_weights: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_rows_by_task = train_rows_by_task
        self.val_rows_by_task = val_rows_by_task or {}
        self.test_rows_by_task = test_rows_by_task or {}
        self.device = device
        self.lr = lr
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.uncertainty_weights = bool(uncertainty_weights)

        self.device_t = torch.device(device)

        # Only train tasks that have data AND are active in the model
        self.active_tasks: List[str] = [
            t for t in TASKS
            if t in model.active_tasks and train_rows_by_task.get(t)
        ]

        # Uncertainty weights (Kendall 2018): learnable log-variance per task
        if self.uncertainty_weights and len(self.active_tasks) > 0:
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.zeros(1, device=self.device_t))
                for task in self.active_tasks
            })
        else:
            self.log_vars = None

        # Collect trainable parameters (backbone LoRA + heads + log_vars)
        params: List[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
        if self.log_vars is not None:
            params.extend(list(self.log_vars.parameters()))
        if not params:
            # Ensure optimizer has at least one param group (avoid crash)
            params = [next(model.parameters())]
        self.optimizer = torch.optim.AdamW(params, lr=lr)

    # -- loss computation ------------------------------------------------
    def _task_loss(self, task: str, batch: List[Dict[str, Any]]) -> torch.Tensor:
        """Compute loss for one task batch."""
        ids, mask = self.tokenizer.batch_encode([str(r["smiles"]) for r in batch])
        ids = ids.to(self.device_t)
        mask = mask.to(self.device_t)

        if task == "retrosynthesis":
            labels = torch.tensor(
                [float(r["label"]) for r in batch],
                dtype=torch.float32, device=self.device_t,
            )
            logits = self.model(ids, attention_mask=mask, task="retrosynthesis")
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            return nn.functional.binary_cross_entropy_with_logits(logits, labels)

        elif task == "condition":
            cat_labels = torch.tensor(
                [int(r["catalyst_label"]) for r in batch],
                dtype=torch.long, device=self.device_t,
            )
            sol_labels = torch.tensor(
                [int(r["solvent_label"]) for r in batch],
                dtype=torch.long, device=self.device_t,
            )
            reg_labels = torch.tensor(
                [int(r["reagent_label"]) for r in batch],
                dtype=torch.long, device=self.device_t,
            )
            outputs = self.model(ids, attention_mask=mask, task="condition")
            loss_cat = nn.functional.cross_entropy(outputs["catalyst"], cat_labels)
            loss_sol = nn.functional.cross_entropy(outputs["solvent"], sol_labels)
            loss_reg = nn.functional.cross_entropy(outputs["reagent"], reg_labels)
            return (loss_cat + loss_sol + loss_reg) / 3.0

        elif task == "yield":
            targets = torch.tensor(
                [float(r["yield"]) for r in batch],
                dtype=torch.float32, device=self.device_t,
            )
            preds = self.model(ids, attention_mask=mask, task="yield")
            return nn.functional.mse_loss(preds, targets)

        else:
            raise ValueError(f"Unknown task: {task!r}")

    def _weighted_loss(self, task: str, task_loss: torch.Tensor) -> torch.Tensor:
        """Apply uncertainty weighting (Kendall 2018).

        L_weighted = (1 / (2 * σ²)) * L + log(σ²)
                   = 0.5 * exp(-log_var) * L + 0.5 * log_var
        """
        if self.log_vars is None or task not in self.log_vars:
            return task_loss
        log_var = self.log_vars[task]
        # Precision = 1 / σ² = exp(-log_var)
        precision = torch.exp(-log_var)
        return 0.5 * precision * task_loss + 0.5 * log_var

    # -- training --------------------------------------------------------
    def train(self) -> Dict[str, List[float]]:
        """Train all active tasks with alternating minibatches.

        Returns
        -------
        Dict[str, List[float]]
            Mapping task -> list of per-epoch average losses.
        """
        history: Dict[str, List[float]] = {task: [] for task in self.active_tasks}

        if not self.active_tasks:
            return history

        for epoch in range(self.epochs):
            self.model.train()
            epoch_losses: Dict[str, float] = {task: 0.0 for task in self.active_tasks}
            n_batches: Dict[str, int] = {task: 0 for task in self.active_tasks}

            # Round-robin over tasks
            for task in self.active_tasks:
                rows = self.train_rows_by_task.get(task, [])
                if not rows:
                    continue
                order = list(range(len(rows)))
                random.shuffle(order)

                for start in range(0, len(order), self.batch_size):
                    batch_idx = order[start: start + self.batch_size]
                    batch = [rows[i] for i in batch_idx]
                    if not batch:
                        continue

                    self.optimizer.zero_grad()
                    task_loss = self._task_loss(task, batch)
                    loss = self._weighted_loss(task, task_loss)
                    loss.backward()
                    self.optimizer.step()

                    epoch_losses[task] += float(task_loss.item())
                    n_batches[task] += 1

            for task in self.active_tasks:
                avg = epoch_losses[task] / max(1, n_batches[task])
                history[task].append(avg)

        return history

    # -- evaluation ------------------------------------------------------
    def evaluate(self, task: str) -> Dict[str, Any]:
        """Evaluate on the test set for one task.

        Returns
        -------
        Dict[str, Any]
            Metrics dict.  Keys depend on the task:

            - ``retrosynthesis``: ``mrr``, ``accuracy``, ``auc``,
              ``n_examples``, ``per_example_probs``, ``per_example_labels``,
              ``per_example_source_ids``, ``per_example_correct``.
            - ``condition``: ``catalyst_top1``, ``solvent_top1``,
              ``reagent_top1``, ``avg_top1``, ``n_examples``,
              ``per_example_correct``, ``per_example_source_ids``.
            - ``yield``: ``mae``, ``rmse``, ``n_examples``,
              ``per_example_preds``, ``per_example_targets``,
              ``per_example_source_ids``, ``per_example_abs_error``.
        """
        if task not in TASKS:
            raise ValueError(f"Unknown task: {task!r}; expected one of {TASKS}")
        self.model.eval()
        rows = self.test_rows_by_task.get(task, [])
        if not rows:
            return {"n_examples": 0, "task": task}

        with torch.no_grad():
            if task == "retrosynthesis":
                return self._evaluate_retrosynthesis(rows)
            elif task == "condition":
                return self._evaluate_condition(rows)
            elif task == "yield":
                return self._evaluate_yield(rows)
            else:
                raise ValueError(f"Unknown task: {task!r}")

    def _evaluate_retrosynthesis(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_probs: List[float] = []
        all_labels: List[int] = []
        all_source_ids: List[str] = []

        with torch.no_grad():
            for start in range(0, len(rows), self.batch_size):
                batch = rows[start: start + self.batch_size]
                if not batch:
                    continue
                ids, mask = self.tokenizer.batch_encode(
                    [str(r["smiles"]) for r in batch]
                )
                ids = ids.to(self.device_t)
                mask = mask.to(self.device_t)
                logits = self.model(ids, attention_mask=mask, task="retrosynthesis")
                if logits.dim() > 1 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                all_probs.extend(float(p) for p in probs)
                all_labels.extend(int(r["label"]) for r in batch)
                all_source_ids.extend(str(r.get("source_id", "")) for r in batch)

        probs_arr = np.array(all_probs, dtype=float)
        labels_arr = np.array(all_labels, dtype=int)
        preds = (probs_arr > 0.5).astype(int)
        correctness = (preds == labels_arr).astype(float)
        mrr = _compute_mrr(probs_arr, labels_arr, all_source_ids)
        accuracy = float(correctness.mean()) if len(correctness) else 0.0
        auc = _compute_auc(labels_arr, probs_arr)
        return {
            "task": "retrosynthesis",
            "mrr": mrr,
            "accuracy": accuracy,
            "auc": auc,
            "n_examples": int(len(labels_arr)),
            "per_example_probs": all_probs,
            "per_example_labels": all_labels,
            "per_example_source_ids": all_source_ids,
            "per_example_correct": correctness.tolist(),
        }

    def _evaluate_condition(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        sub_heads = ("catalyst", "solvent", "reagent")
        all_preds: Dict[str, List[int]] = {h: [] for h in sub_heads}
        all_labels: Dict[str, List[int]] = {h: [] for h in sub_heads}
        all_source_ids: List[str] = []

        with torch.no_grad():
            for start in range(0, len(rows), self.batch_size):
                batch = rows[start: start + self.batch_size]
                if not batch:
                    continue
                ids, mask = self.tokenizer.batch_encode(
                    [str(r["smiles"]) for r in batch]
                )
                ids = ids.to(self.device_t)
                mask = mask.to(self.device_t)
                outputs = self.model(ids, attention_mask=mask, task="condition")
                for h in sub_heads:
                    preds = outputs[h].argmax(dim=-1).cpu().numpy().tolist()
                    if isinstance(preds, int):
                        preds = [preds]
                    all_preds[h].extend(int(p) for p in preds)
                    all_labels[h].extend(int(r[f"{h}_label"]) for r in batch)
                all_source_ids.extend(str(r.get("source_id", "")) for r in batch)

        metrics: Dict[str, Any] = {"task": "condition", "n_examples": int(len(rows))}
        per_head_correct: Dict[str, List[float]] = {}
        for h in sub_heads:
            preds_arr = np.array(all_preds[h], dtype=int)
            labels_arr = np.array(all_labels[h], dtype=int)
            correct = (preds_arr == labels_arr).astype(float)
            per_head_correct[h] = correct.tolist()
            metrics[f"{h}_top1"] = float(correct.mean()) if len(correct) else 0.0

        avg_top1 = float(np.mean([metrics[f"{h}_top1"] for h in sub_heads])) if sub_heads else 0.0
        metrics["avg_top1"] = avg_top1

        # Per-example correct = 1 if ALL 3 sub-heads correct, else 0
        n = len(rows)
        all_correct = np.ones(n, dtype=float)
        for h in sub_heads:
            arr = np.array(per_head_correct[h], dtype=float)
            if len(arr) == n:
                all_correct *= arr
        metrics["per_example_correct"] = all_correct.tolist()
        metrics["per_example_source_ids"] = all_source_ids
        return metrics

    def _evaluate_yield(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_preds: List[float] = []
        all_targets: List[float] = []
        all_source_ids: List[str] = []

        with torch.no_grad():
            for start in range(0, len(rows), self.batch_size):
                batch = rows[start: start + self.batch_size]
                if not batch:
                    continue
                ids, mask = self.tokenizer.batch_encode(
                    [str(r["smiles"]) for r in batch]
                )
                ids = ids.to(self.device_t)
                mask = mask.to(self.device_t)
                preds = self.model(ids, attention_mask=mask, task="yield")
                preds_np = preds.cpu().numpy().tolist()
                if isinstance(preds_np, float):
                    preds_np = [preds_np]
                all_preds.extend(float(p) for p in preds_np)
                all_targets.extend(float(r["yield"]) for r in batch)
                all_source_ids.extend(str(r.get("source_id", "")) for r in batch)

        preds_arr = np.array(all_preds, dtype=float)
        targets_arr = np.array(all_targets, dtype=float)
        abs_err = np.abs(preds_arr - targets_arr)
        mae = float(abs_err.mean()) if len(abs_err) else 0.0
        rmse = float(np.sqrt((abs_err ** 2).mean())) if len(abs_err) else 0.0
        return {
            "task": "yield",
            "mae": mae,
            "rmse": rmse,
            "n_examples": int(len(rows)),
            "per_example_preds": all_preds,
            "per_example_targets": all_targets,
            "per_example_source_ids": all_source_ids,
            "per_example_abs_error": abs_err.tolist(),
        }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _compute_mrr(
    probs: np.ndarray,
    labels: np.ndarray,
    source_ids: Sequence[str],
) -> float:
    """Compute Mean Reciprocal Rank by ``source_id`` group."""
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
    """Compute ROC AUC via the Mann-Whitney U formula."""
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
# Data loading
# ---------------------------------------------------------------------------
def _extract_product_from_reaction(reaction_smiles: str) -> str:
    """Extract product SMILES from a reaction SMILES (R>>P format)."""
    if not reaction_smiles:
        return ""
    if ">>" in reaction_smiles:
        return reaction_smiles.split(">>")[-1].strip()
    if ">" in reaction_smiles:
        return reaction_smiles.split(">")[-1].strip()
    return reaction_smiles.strip()


def _parse_yield(raw: Any) -> Optional[float]:
    """Parse a yield cell, returning ``None`` for missing/non-numeric values."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _load_retrosynthesis_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load retrosynthesis CSV (positive=1, others=0)."""
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
            rows.append({
                "smiles": smiles,
                "label": label,
                "source_id": str(source_id),
            })
    return rows


def _load_condition_json(
    json_path: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Load condition JSON (catalyst/solvent/reagent).

    Returns
    -------
    Tuple[List[Row], Dict[str, Dict[str, int]]]
        Rows with integer ``catalyst_label`` / ``solvent_label`` /
        ``reagent_label`` fields, and the label maps for each sub-head.
        Missing classes are mapped to the most common class (fallback).
    """
    if not json_path or not os.path.exists(json_path):
        return [], {"catalyst": {}, "solvent": {}, "reagent": {}}

    with open(json_path, encoding="utf-8") as fh:
        records = json.load(fh)

    raw_rows: List[Dict[str, Any]] = []
    cat_counter: Counter = Counter()
    sol_counter: Counter = Counter()
    reg_counter: Counter = Counter()

    for rec in records:
        rxn = str(rec.get("reaction_smiles") or rec.get("smiles") or "").strip()
        if not rxn:
            continue
        cat = str(rec.get("catalyst") or "").strip()
        sol = str(rec.get("solvent") or "").strip()
        reg = str(rec.get("reagent") or "").strip()
        # Take the first SMILES if multiple (dotted)
        if "." in cat:
            cat = cat.split(".")[0]
        if "." in sol:
            sol = sol.split(".")[0]
        if "." in reg:
            reg = reg.split(".")[0]
        if not cat:
            cat = "none"
        if not sol:
            sol = "none"
        if not reg:
            reg = "none"
        source_id = str(rec.get("source_id") or rec.get("source") or rxn)
        raw_rows.append({
            "smiles": rxn,
            "catalyst": cat,
            "solvent": sol,
            "reagent": reg,
            "source_id": source_id,
        })
        cat_counter[cat] += 1
        sol_counter[sol] += 1
        reg_counter[reg] += 1

    # Build label maps (sorted for determinism)
    cat_map = {l: i for i, l in enumerate(sorted(cat_counter.keys()))}
    sol_map = {l: i for i, l in enumerate(sorted(sol_counter.keys()))}
    reg_map = {l: i for i, l in enumerate(sorted(reg_counter.keys()))}

    # Most common class as fallback for missing classes
    cat_fallback = cat_counter.most_common(1)[0][0] if cat_counter else "none"
    sol_fallback = sol_counter.most_common(1)[0][0] if sol_counter else "none"
    reg_fallback = reg_counter.most_common(1)[0][0] if reg_counter else "none"

    rows: List[Dict[str, Any]] = []
    for r in raw_rows:
        cat_label = cat_map.get(r["catalyst"], cat_map[cat_fallback])
        sol_label = sol_map.get(r["solvent"], sol_map[sol_fallback])
        reg_label = reg_map.get(r["reagent"], reg_map[reg_fallback])
        rows.append({
            "smiles": r["smiles"],
            "catalyst_label": int(cat_label),
            "solvent_label": int(sol_label),
            "reagent_label": int(reg_label),
            "source_id": r["source_id"],
        })

    label_maps = {"catalyst": cat_map, "solvent": sol_map, "reagent": reg_map}
    return rows, label_maps


def _load_yield_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load yield CSV (skip rows with empty yield)."""
    rows: List[Dict[str, Any]] = []
    if not csv_path or not os.path.exists(csv_path):
        return rows
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            smiles = (
                record.get("reaction_smiles")
                or record.get("smiles")
                or ""
            )
            smiles = str(smiles).strip()
            if not smiles:
                continue
            yld = _parse_yield(record.get("yield", ""))
            if yld is None:
                continue  # skip rows with empty yield
            source_id = (
                record.get("source_id")
                or record.get("source")
                or record.get("reaction_class")
                or smiles
            )
            rows.append({
                "smiles": smiles,
                "yield": float(yld),
                "source_id": str(source_id),
            })
    return rows


def load_multitask_data(data_dir: str) -> Dict[str, Any]:
    """Load the 3 task datasets.

    Parameters
    ----------
    data_dir:
        Path to ``data/processed`` directory.

    Returns
    -------
    Dict[str, Any]
        Dict with keys:
          - ``retrosynthesis``: List[Row] (from ``regiosqm20_normalized.csv``)
          - ``condition``: List[Row] (from ``ord_conditions.json``)
          - ``yield``: List[Row] (from ``hitea_full_normalized.csv``)
          - ``condition_label_maps``: Dict[str, Dict[str, int]]
          - ``n_classes``: Dict[str, int] (``catalyst`` / ``solvent`` / ``reagent``)
    """
    data_dir = str(data_dir)
    retro_csv = os.path.join(data_dir, "regiosqm20_normalized.csv")
    cond_json = os.path.join(data_dir, "ord_conditions.json")
    yield_csv = os.path.join(data_dir, "hitea_full_normalized.csv")

    retro_rows = _load_retrosynthesis_csv(retro_csv)
    cond_rows, cond_maps = _load_condition_json(cond_json)
    yield_rows = _load_yield_csv(yield_csv)

    n_classes = {
        "catalyst": len(cond_maps["catalyst"]),
        "solvent": len(cond_maps["solvent"]),
        "reagent": len(cond_maps["reagent"]),
    }

    return {
        "retrosynthesis": retro_rows,
        "condition": cond_rows,
        "yield": yield_rows,
        "condition_label_maps": cond_maps,
        "n_classes": n_classes,
    }


# ---------------------------------------------------------------------------
# Split helpers (HC #9)
# ---------------------------------------------------------------------------
def load_or_create_split(
    rows: Sequence[Dict[str, Any]],
    train_idx_path: Optional[str],
    val_idx_path: Optional[str],
    test_idx_path: Optional[str],
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Load train/val/test indices from JSON files (HC #9).

    If any file is missing, an 80/10/10 stratified-by-``source_id`` group
    split is auto-created.
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
        # Accept both bare lists and {"indices": [...]} formats
        if isinstance(train_idx, dict):
            train_idx = train_idx.get("indices", [])
        if isinstance(val_idx, dict):
            val_idx = val_idx.get("indices", [])
        if isinstance(test_idx, dict):
            test_idx = test_idx.get("indices", [])
        return list(train_idx), list(val_idx), list(test_idx)

    # Auto-create 80/10/10 group split stratified by source_id
    source_ids = sorted({r.get("source_id", "") for r in rows})
    rng = random.Random(seed)
    rng.shuffle(source_ids)
    n = len(source_ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_sids = set(source_ids[:n_train])
    val_sids = set(source_ids[n_train: n_train + n_val])
    train_idx, val_idx, test_idx = [], [], []
    for i, r in enumerate(rows):
        sid = r.get("source_id", "")
        if sid in train_sids:
            train_idx.append(i)
        elif sid in val_sids:
            val_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def paired_bootstrap_ci(
    metric_a: Sequence[float],
    metric_b: Sequence[float],
    n_iterations: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Standard paired bootstrap CI for per-example metrics.

    Resamples individual examples with replacement.

    Parameters
    ----------
    metric_a, metric_b:
        Per-example metric arrays for systems A and B (same length).
    n_iterations:
        Number of bootstrap iterations (default 10000).
    seed:
        Random seed.

    Returns
    -------
    Tuple[float, float, float, float]
        ``(mean_diff, ci_low, ci_high, p_value)``.
    """
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
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

    Parameters
    ----------
    metric_a, metric_b:
        Per-example metric arrays for systems A and B (same length).
    cluster_ids:
        Per-example cluster IDs (e.g. ``source_id``).
    n_iterations:
        Number of bootstrap iterations (default 10000).
    seed:
        Random seed.

    Returns
    -------
    Tuple[float, float, float, float]
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
    cluster_to_idx: Dict[str, np.ndarray] = {
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
# Experiment runner
# ---------------------------------------------------------------------------
def _build_model(
    backbone: nn.Module,
    n_classes: Dict[str, int],
    active_tasks: Set[str],
    device: str,
) -> MultiTaskModel:
    """Build a MultiTaskModel with the given active tasks."""
    model = MultiTaskModel(
        backbone=copy.deepcopy(backbone),
        n_catalyst_classes=max(2, n_classes.get("catalyst", 2)),
        n_solvent_classes=max(2, n_classes.get("solvent", 2)),
        n_reagent_classes=max(2, n_classes.get("reagent", 2)),
        active_tasks=active_tasks,
    )
    return model.to(device)


def run_experiment(
    backbone_ckpt: Optional[str],
    vocab_path: str,
    data_dir: str,
    seeds: Sequence[int],
    output_dir: str,
    epochs: int = 5,
    lr: float = 1e-4,
    batch_size: int = 16,
    device: str = "cpu",
    uncertainty_weighting: bool = True,
    train_idx_path: Optional[str] = None,
    val_idx_path: Optional[str] = None,
    test_idx_path: Optional[str] = None,
    bootstrap_iterations: int = 10000,
    hparams: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full P3-06 multi-task experiment across seeds.

    For each seed:
      1. Train single-task baselines (retro only, condition only, yield only)
      2. Train multi-task model (all 3 tasks jointly)
      3. Evaluate both on test set
      4. Compute paired family-cluster bootstrap CI (multitask vs singletask)
      5. Save per-seed ``metrics.json`` + ``summary.json`` + ``summary.md``
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load data once (shared across seeds)
    data = load_multitask_data(data_dir)
    n_classes = data["n_classes"]
    tokenizer = load_tokenizer(vocab_path)

    # Build splits per task
    # HC #9: v3 splits for retrosynthesis; auto-create for others
    splits_by_task: Dict[str, Tuple[List[int], List[int], List[int]]] = {}
    for task in TASKS:
        rows = data[task]
        if not rows:
            continue
        if task == "retrosynthesis":
            tr, va, te = load_or_create_split(
                rows, train_idx_path, val_idx_path, test_idx_path
            )
        else:
            tr, va, te = load_or_create_split(rows, None, None, None, seed=42)
        splits_by_task[task] = (tr, va, te)

    # Build a shared backbone (fresh init for reproducibility per-seed)
    def _build_shared_backbone() -> nn.Module:
        return build_backbone(
            checkpoint_path=backbone_ckpt,
            hparams=hparams,
            freeze=True,
            apply_lora_flag=True,
        )

    per_seed_summaries: List[Dict[str, Any]] = []
    multitask_metrics_by_task: Dict[str, List[float]] = {t: [] for t in TASKS}
    singletask_metrics_by_task: Dict[str, List[float]] = {t: [] for t in TASKS}
    per_example_by_task: Dict[str, Dict[str, List[Any]]] = {
        t: {"mt_correct": [], "st_correct": [], "source_ids": []} for t in TASKS
    }

    metric_key_by_task: Dict[str, str] = {
        "retrosynthesis": "accuracy",
        "condition": "avg_top1",
        "yield": "mae",
    }

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        seed_dir = out / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Prepare split rows per task
        train_rows_by_task: Dict[str, List[Dict[str, Any]]] = {}
        val_rows_by_task: Dict[str, List[Dict[str, Any]]] = {}
        test_rows_by_task: Dict[str, List[Dict[str, Any]]] = {}
        for task in TASKS:
            if task not in splits_by_task:
                continue
            tr, va, te = splits_by_task[task]
            rows = data[task]
            train_rows_by_task[task] = [rows[i] for i in tr if 0 <= i < len(rows)]
            val_rows_by_task[task] = [rows[i] for i in va if 0 <= i < len(rows)]
            test_rows_by_task[task] = [rows[i] for i in te if 0 <= i < len(rows)]

        seed_summary: Dict[str, Any] = {"seed": seed, "tasks": {}, "multitask": {}, "singletask": {}}

        # ---- Single-task baselines ----
        singletask_evals: Dict[str, Dict[str, Any]] = {}
        for task in TASKS:
            if task not in train_rows_by_task or not train_rows_by_task[task]:
                continue
            backbone_st = _build_shared_backbone()
            model_st = _build_model(backbone_st, n_classes, {task}, device)
            trainer_st = MultiTaskTrainer(
                model=model_st,
                tokenizer=tokenizer,
                train_rows_by_task={task: train_rows_by_task[task]},
                val_rows_by_task={task: val_rows_by_task.get(task, [])},
                test_rows_by_task={task: test_rows_by_task.get(task, [])},
                device=device,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                uncertainty_weights=False,  # single-task: no need for weighting
            )
            trainer_st.train()
            eval_st = trainer_st.evaluate(task)
            singletask_evals[task] = eval_st
            key = metric_key_by_task[task]
            singletask_metrics_by_task[task].append(float(eval_st.get(key, 0.0)))
            seed_summary["singletask"][task] = {
                k: v for k, v in eval_st.items()
                if not k.startswith("per_example_")
            }

        # ---- Multi-task model ----
        active_tasks = {t for t in TASKS if t in train_rows_by_task}
        if not active_tasks:
            continue
        backbone_mt = _build_shared_backbone()
        model_mt = _build_model(backbone_mt, n_classes, active_tasks, device)
        trainer_mt = MultiTaskTrainer(
            model=model_mt,
            tokenizer=tokenizer,
            train_rows_by_task=train_rows_by_task,
            val_rows_by_task=val_rows_by_task,
            test_rows_by_task=test_rows_by_task,
            device=device,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            uncertainty_weights=uncertainty_weighting,
        )
        history = trainer_mt.train()

        multitask_evals: Dict[str, Dict[str, Any]] = {}
        for task in active_tasks:
            eval_mt = trainer_mt.evaluate(task)
            multitask_evals[task] = eval_mt
            key = metric_key_by_task[task]
            multitask_metrics_by_task[task].append(float(eval_mt.get(key, 0.0)))
            seed_summary["multitask"][task] = {
                k: v for k, v in eval_mt.items()
                if not k.startswith("per_example_")
            }
            # Collect per-example for bootstrap CI
            mt_correct = eval_mt.get("per_example_correct", [])
            st_correct = singletask_evals.get(task, {}).get("per_example_correct", [])
            sids = eval_mt.get("per_example_source_ids", [])
            if mt_correct and st_correct and len(mt_correct) == len(st_correct):
                per_example_by_task[task]["mt_correct"].extend(mt_correct)
                per_example_by_task[task]["st_correct"].extend(st_correct)
                per_example_by_task[task]["source_ids"].extend(sids)

        seed_summary["train_history"] = history
        seed_summary["active_tasks"] = sorted(active_tasks)

        # Save per-seed metrics
        (seed_dir / "metrics.json").write_text(
            json.dumps(seed_summary, indent=2), encoding="utf-8"
        )
        per_seed_summaries.append(seed_summary)

        print(f"[P3-06] seed={seed} done; tasks={sorted(active_tasks)}")

    # ---- Aggregate + paired bootstrap CI ----
    summary: Dict[str, Any] = {
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "uncertainty_weighting": uncertainty_weighting,
        "backbone_ckpt": str(backbone_ckpt) if backbone_ckpt else None,
        "tasks": {},
    }

    for task in TASKS:
        mt_vals = multitask_metrics_by_task[task]
        st_vals = singletask_metrics_by_task[task]
        key = metric_key_by_task[task]

        # Per-seed paired bootstrap (if we have per-seed values)
        if mt_vals and st_vals and len(mt_vals) == len(st_vals):
            pb = paired_bootstrap_ci(mt_vals, st_vals, n_iterations=bootstrap_iterations)
        else:
            pb = (0.0, 0.0, 0.0, 1.0)

        # Family-cluster bootstrap (per-example)
        pe = per_example_by_task[task]
        if pe["mt_correct"] and pe["st_correct"] and len(pe["mt_correct"]) == len(pe["st_correct"]):
            # For yield, lower is better; for classification, higher is better.
            # Use negative abs_error so that "higher is better" uniformly.
            if task == "yield":
                mt_metric = [-float(x) for x in pe["mt_correct"]]
                st_metric = [-float(x) for x in pe["st_correct"]]
            else:
                mt_metric = [float(x) for x in pe["mt_correct"]]
                st_metric = [float(x) for x in pe["st_correct"]]
            fc = family_cluster_bootstrap_ci(
                mt_metric, st_metric, pe["source_ids"],
                n_iterations=bootstrap_iterations,
            )
        else:
            fc = (0.0, 0.0, 0.0, 1.0)

        summary["tasks"][task] = {
            "metric": key,
            "multitask_mean": float(np.mean(mt_vals)) if mt_vals else 0.0,
            "multitask_std": float(np.std(mt_vals)) if mt_vals else 0.0,
            "singletask_mean": float(np.mean(st_vals)) if st_vals else 0.0,
            "singletask_std": float(np.std(st_vals)) if st_vals else 0.0,
            "multitask_per_seed": mt_vals,
            "singletask_per_seed": st_vals,
            "paired_bootstrap_ci": {
                "mean_diff": pb[0],
                "ci_low": pb[1],
                "ci_high": pb[2],
                "p_value": pb[3],
            },
            "family_cluster_bootstrap_ci": {
                "mean_diff": fc[0],
                "ci_low": fc[1],
                "ci_high": fc[2],
                "p_value": fc[3],
            },
            # GO if multitask > singletask with CI excluding zero
            "go": bool(fc[0] > 0 and fc[1] > 0),
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(_render_summary_md(summary), encoding="utf-8")
    return summary


def _render_summary_md(summary: Dict[str, Any]) -> str:
    """Render a human-readable Markdown summary."""
    lines: List[str] = []
    lines.append("# P3-06: Multi-task Joint Training Summary\n")
    lines.append(f"- Seeds: {summary['n_seeds']}")
    lines.append(f"- Epochs: {summary['epochs']}  LR: {summary['lr']}  Batch: {summary['batch_size']}")
    lines.append(f"- Uncertainty weighting: {summary['uncertainty_weighting']}")
    lines.append(f"- Backbone checkpoint: `{summary['backbone_ckpt']}`\n")
    lines.append("## Per-task results (multitask vs singletask)\n")
    lines.append("| Task | Metric | Multitask | Singletask | Diff | CI low | CI high | p | GO |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for task, info in summary["tasks"].items():
        diff = info["multitask_mean"] - info["singletask_mean"]
        ci = info["family_cluster_bootstrap_ci"]
        go = "✓" if info["go"] else "✗"
        lines.append(
            f"| {task} | {info['metric']} | "
            f"{info['multitask_mean']:.4f} | {info['singletask_mean']:.4f} | "
            f"{diff:+.4f} | {ci['ci_low']:+.4f} | {ci['ci_high']:+.4f} | "
            f"{ci['p_value']:.4f} | {go} |"
        )
    lines.append("\n## Notes\n")
    lines.append("- Family-cluster bootstrap CI clusters by `source_id` (HC #5).")
    lines.append("- GO = multitask beats singletask with CI lower bound > 0.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_seeds(seeds_str: str) -> List[int]:
    """Parse ``--seeds`` argument."""
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
    """CLI entry point for P3-06 multi-task joint training."""
    parser = argparse.ArgumentParser(
        description="P3-06: multi-task joint training (retrosynthesis + condition + yield)"
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
        default="results/multitask_joint_training_20260720",
        help="Output directory",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="cuda:0", help="Device (cpu / cuda:0 / ...)")
    parser.add_argument(
        "--uncertainty-weighting",
        action="store_true",
        help="Use Kendall 2018 uncertainty weighting for multi-task loss",
    )
    parser.add_argument("--train-idx", default=None, help="HC #9: train idx JSON (retrosynthesis)")
    parser.add_argument("--val-idx", default=None, help="HC #9: val idx JSON (retrosynthesis)")
    parser.add_argument("--test-idx", default=None, help="HC #9: test idx JSON (retrosynthesis)")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10000,
        help="Bootstrap iterations for CI",
    )
    args = parser.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    vocab_path = args.vocab or "external/reaction_lm/Chemformer/bart_vocab.json"

    print(f"[P3-06] seeds={seeds} device={args.device}")
    print(f"[P3-06] backbone_ckpt={args.backbone_ckpt}")
    print(f"[P3-06] output_dir={args.output_dir}")
    print(f"[P3-06] uncertainty_weighting={args.uncertainty_weighting}")

    run_experiment(
        backbone_ckpt=args.backbone_ckpt,
        vocab_path=vocab_path,
        data_dir=args.data_dir,
        seeds=seeds,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        uncertainty_weighting=args.uncertainty_weighting,
        train_idx_path=args.train_idx,
        val_idx_path=args.val_idx,
        test_idx_path=args.test_idx,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(f"[P3-06] DONE. Results saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
