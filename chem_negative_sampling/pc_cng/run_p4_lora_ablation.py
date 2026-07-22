"""P4-G2: Chemformer-LoRA Ablation & Backbone Configuration Freeze.

CLI entry point::

    python3 -m pc_cng.run_p4_lora_ablation \
        --manifest data/p4/manifests/hte_feasibility_v1.json \
        --output-dir results/p4_lora_ablation \
        --device cuda:0 \
        --stage smoke

Stages:
    smoke      — 1 seed, all 6 configs, short epochs
    screening  — 3 seeds, passing configs, full epochs
    final      — 10 seeds, passing configs, full epochs
    full       — run all three stages sequentially

The script auto-discovers real ``nn.Linear`` module names from the model
(never assumes ``q_proj``/``v_proj``).  All configs use the same checkpoint,
candidate manifest, splits, training steps, batch budget, and eval code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Imports from existing project code
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from models.pretrained_backbone import (  # noqa: E402
    CHEMFORMER_HPARAMS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_VOCAB_PATH,
    ChemformerTokenizer,
    PretrainedChemformerBackbone,
    PretrainedReactionScorer,
    ReactionClassificationHead,
)
from models.adapter import (  # noqa: E402
    apply_lora,
    freeze_non_lora_params,
    count_trainable_parameters,
    count_total_parameters,
)
from pc_cng.ranking_metrics import ranking_metrics, grouped_metrics  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASE = "P4-G2"
NONINFERIORITY_MARGIN = -0.005  # -0.5 percentage points in MRR
DEFAULT_SEEDS_SMOKE = [20260721]
DEFAULT_SEEDS_SCREENING = [20260721, 20260722, 20260723]
DEFAULT_SEEDS_FINAL = list(range(20260721, 20260731))  # 10 seeds
DEFAULT_EPOCHS_SMOKE = 2
DEFAULT_EPOCHS_FULL = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_MAX_SEQ_LEN = 256

# Config IDs
CONFIG_IDS = ["C1", "C2", "C3", "C4", "C5", "C6"]


# ---------------------------------------------------------------------------
# Module auto-discovery (spec: "首先从代码自动读取真实 module names")
# ---------------------------------------------------------------------------

def discover_linear_modules(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """Return all (qualified_name, module) pairs where module is nn.Linear.

    This is the single source of truth for LoRA target selection.  We never
    hard-code names like ``q_proj`` or ``v_proj`` — we read them from the
    actual model instance.
    """
    found = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            found.append((name, module))
    return found


def build_module_registry(model: nn.Module) -> Dict[str, Any]:
    """Build a registry of all Linear modules with their shapes."""
    linears = discover_linear_modules(model)
    registry = {}
    for name, module in linears:
        registry[name] = {
            "type": "nn.Linear",
            "in_features": module.in_features,
            "out_features": module.out_features,
            "has_bias": module.bias is not None,
        }
    # Also record non-Linear attention parameters (e.g. in_proj_weight)
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
                registry[f"{name}.in_proj_weight"] = {
                    "type": "nn.Parameter",
                    "shape": list(module.in_proj_weight.shape),
                    "note": "Q/K/V combined projection (not nn.Linear, not LoRA-able by current adapter)",
                }
    return registry


def discover_target_patterns(model: nn.Module, scope: str) -> List[str]:
    """Auto-generate glob patterns for Linear modules matching a scope.

    Scopes:
        "ffn"        — feed-forward linear1 + linear2
        "attention"  — self_attn.out_proj
        "all_linear" — all nn.Linear in encoder_layers

    Patterns use * wildcard for layer index so they match all layers.
    """
    linears = discover_linear_modules(model)
    all_names = [name for name, _ in linears]

    if scope == "ffn":
        # Find linear1, linear2 patterns
        patterns = set()
        for name in all_names:
            if name.endswith(".linear1") or name.endswith(".linear2"):
                # Generalize layer index to wildcard
                parts = name.split(".")
                if len(parts) >= 2 and parts[-2].isdigit():
                    parts[-2] = "*"
                    patterns.add(".".join(parts))
        return sorted(patterns)
    elif scope == "attention":
        patterns = set()
        for name in all_names:
            if "self_attn.out_proj" in name:
                parts = name.split(".")
                if len(parts) >= 2 and parts[-3].isdigit():
                    parts[-3] = "*"
                    patterns.add(".".join(parts))
        return sorted(patterns)
    elif scope == "all_linear":
        # All nn.Linear in encoder_layers (handle backbone. prefix)
        patterns = set()
        for name in all_names:
            if "encoder_layers." in name:
                parts = name.split(".")
                # Find the digit part (layer index) and replace with wildcard
                for i, part in enumerate(parts):
                    if part.isdigit():
                        parts[i] = "*"
                        break
                patterns.add(".".join(parts))
        return sorted(patterns)
    else:
        return []


# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """One LoRA ablation configuration."""
    config_id: str
    name: str
    description: str
    adapter: str  # "none" | "lora" | "full"
    target_patterns: List[str] = field(default_factory=list)
    rank: int = 0
    alpha: float = 0.0
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def build_config_registry(model: nn.Module) -> Dict[str, Any]:
    """Build the 6-config registry using auto-discovered module names.

    Returns a dict with:
        - configs: {config_id: AblationConfig.to_dict()}
        - module_registry: all discovered Linear modules
        - discovered_patterns: patterns by scope
    """
    ffn_patterns = discover_target_patterns(model, "ffn")
    attn_patterns = discover_target_patterns(model, "attention")
    all_patterns = discover_target_patterns(model, "all_linear")
    module_registry = build_module_registry(model)

    configs = {
        "C1": AblationConfig(
            config_id="C1",
            name="zero_shot_frozen",
            description="Zero-shot / frozen backbone (no training)",
            adapter="none",
            target_patterns=[],
            rank=0,
            alpha=0.0,
            dropout=0.0,
        ),
        "C2": AblationConfig(
            config_id="C2",
            name="lora_baseline_ffn",
            description="LoRA on FFN (linear1+linear2), rank 8 — current baseline",
            adapter="lora",
            target_patterns=ffn_patterns,
            rank=8,
            alpha=16.0,
            dropout=0.0,
        ),
        "C3": AblationConfig(
            config_id="C3",
            name="lora_attention",
            description="LoRA on attention output projection (self_attn.out_proj), rank 8",
            adapter="lora",
            target_patterns=attn_patterns,
            rank=8,
            alpha=16.0,
            dropout=0.0,
        ),
        "C4": AblationConfig(
            config_id="C4",
            name="lora_all_linear_r8",
            description="LoRA on all encoder Linear modules, rank 8",
            adapter="lora",
            target_patterns=all_patterns,
            rank=8,
            alpha=16.0,
            dropout=0.0,
        ),
        "C5": AblationConfig(
            config_id="C5",
            name="lora_all_linear_r16",
            description="LoRA on all encoder Linear modules, rank 16",
            adapter="lora",
            target_patterns=all_patterns,
            rank=16,
            alpha=32.0,
            dropout=0.0,
        ),
        "C6": AblationConfig(
            config_id="C6",
            name="full_finetune",
            description="Full fine-tuning (all parameters trainable)",
            adapter="full",
            target_patterns=[],
            rank=0,
            alpha=0.0,
            dropout=0.0,
        ),
    }

    return {
        "configs": {cid: c.to_dict() for cid, c in configs.items()},
        "module_registry": module_registry,
        "discovered_patterns": {
            "ffn": ffn_patterns,
            "attention": attn_patterns,
            "all_linear": all_patterns,
        },
    }


# ---------------------------------------------------------------------------
# Manifest data loading
# ---------------------------------------------------------------------------

def load_manifest_candidates(manifest_path: Path) -> Dict[str, List[dict]]:
    """Load P4-G1 manifest and split candidates by split field.

    Returns {"train": [...], "val": [...], "test": [...]}.
    Each candidate dict is augmented with:
        - label: int (1 if gold_candidate else 0)
        - smiles: candidate_smiles (atom-mapped, stripped of atom maps for tokenizer)
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    splits = {"train": [], "val": [], "test": []}
    for group in manifest.get("groups", []):
        gid = group["group_id"]
        for cand in group.get("candidates", []):
            split = cand.get("split", "train")
            if split not in splits:
                continue
            # Strip atom map numbers from SMILES for tokenizer
            smiles = cand.get("candidate_smiles", "")
            # Remove atom map: patterns like :NN:
            import re
            smiles_clean = re.sub(r":\d+", "", smiles)
            entry = {
                "group_id": gid,
                "candidate_id": cand.get("candidate_id", ""),
                "smiles": smiles_clean,
                "raw_smiles": smiles,
                "label": 1 if cand.get("gold_candidate", False) else 0,
                "gold_candidate": cand.get("gold_candidate", False),
                "candidate_source": cand.get("candidate_source", ""),
                "split": split,
                "reaction_family": cand.get("reaction_family", ""),
                "reaction_template": cand.get("reaction_template", ""),
                "product_scaffold": cand.get("product_scaffold", ""),
                "edit_type": cand.get("edit_type", ""),
            }
            splits[split].append(entry)

    return splits


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(
    config: AblationConfig,
    checkpoint_path: Path,
    vocab_path: Path,
    device: str = "cpu",
) -> Tuple[PretrainedReactionScorer, ChemformerTokenizer, int]:
    """Build model per config. Returns (model, tokenizer, trainable_params)."""
    backbone = PretrainedChemformerBackbone(
        checkpoint_path=checkpoint_path,
        freeze=True,
    )

    if config.adapter == "lora":
        # Patterns were discovered from scorer (PretrainedReactionScorer) where
        # backbone is accessed as self.backbone, so names include "backbone." prefix.
        # apply_lora is called on backbone directly, so strip the prefix.
        stripped_patterns = []
        for pat in config.target_patterns:
            if pat.startswith("backbone."):
                stripped_patterns.append(pat[len("backbone."):])
            else:
                stripped_patterns.append(pat)
        n_replaced = apply_lora(
            backbone,
            r=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            target_patterns=stripped_patterns,
        )
        if n_replaced == 0:
            print(f"  WARNING: apply_lora replaced 0 modules! patterns={stripped_patterns}")
        trainable = freeze_non_lora_params(backbone)
    elif config.adapter == "full":
        # Unfreeze all backbone params
        for p in backbone.parameters():
            p.requires_grad = True
        trainable = count_trainable_parameters(backbone)
    else:  # "none"
        trainable = 0

    head = ReactionClassificationHead(d_model=backbone.hparams.get("d_model", 512))
    model = PretrainedReactionScorer(backbone, head)
    model.to(device)

    # Count trainable params including head
    trainable = count_trainable_parameters(model)
    tokenizer = ChemformerTokenizer(vocab_path)

    return model, tokenizer, trainable


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: PretrainedReactionScorer,
    tokenizer: ChemformerTokenizer,
    train_data: List[dict],
    val_data: List[dict],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    log_every: int = 50,
) -> Dict[str, Any]:
    """Train the model. Returns history and best state."""
    import random as _rng

    # Prepare training rows: label = gold_candidate
    # Use all candidates (gold + negatives) for BCE training
    train_smiles = [r["smiles"] for r in train_data]
    train_labels = torch.tensor([r["label"] for r in train_data], dtype=torch.float32)

    val_smiles = [r["smiles"] for r in val_data]
    val_labels = torch.tensor([r["label"] for r in val_data], dtype=torch.float32)

    # Class weighting
    n_pos = int(train_labels.sum().item())
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor(max(1.0, n_neg / max(n_pos, 1)), dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    history = []
    best_val_loss = float("inf")
    best_state = None
    n_train = len(train_smiles)

    for epoch in range(epochs):
        model.train()
        # Shuffle indices
        rng = _rng.Random(epoch * 1000 + 42)
        indices = list(range(n_train))
        rng.shuffle(indices)

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_smiles = [train_smiles[i] for i in batch_idx]
            batch_labels = train_labels[batch_idx].to(device)

            token_ids, attn_mask = tokenizer.batch_encode(batch_smiles)
            token_ids = token_ids.to(device)
            attn_mask = attn_mask.to(device)

            logits = model(token_ids, attn_mask)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for start in range(0, len(val_smiles), batch_size):
                batch_smiles = val_smiles[start:start + batch_size]
                batch_labels = val_labels[start:start + batch_size].to(device)
                token_ids, attn_mask = tokenizer.batch_encode(batch_smiles)
                token_ids = token_ids.to(device)
                attn_mask = attn_mask.to(device)
                logits = model(token_ids, attn_mask)
                loss = nn.BCEWithLogitsLoss()(logits, batch_labels)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best_val_loss": best_val_loss}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_calibration_metrics(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> Dict[str, float]:
    """Compute ECE (Expected Calibration Error) and Brier score."""
    probs = torch.sigmoid(logits)
    labels = labels.float()

    # ECE
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        else:
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        count = mask.sum().item()
        if count > 0:
            acc = labels[mask].mean().item()
            conf = probs[mask].mean().item()
            ece += (count / n) * abs(acc - conf)

    # Brier score
    brier = ((probs - labels) ** 2).mean().item()

    return {"ece": round(ece, 6), "brier": round(brier, 6)}


def evaluate_ranking(
    model: PretrainedReactionScorer,
    tokenizer: ChemformerTokenizer,
    candidates: List[dict],
    device: str,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Evaluate model on candidates using grouped ranking metrics.

    Returns:
        - ranking: {top1, top3, mrr, ndcg, ...}
        - calibration: {ece, brier}
        - raw_predictions: list of {group_id, candidate_id, label, score, ...}
        - inference_latency_ms: average ms per candidate
    """
    model.eval()
    all_logits = []
    all_labels = []
    raw_predictions = []

    # Measure inference latency
    n_candidates = len(candidates)
    latency_start = time.time()

    with torch.no_grad():
        for start in range(0, n_candidates, batch_size):
            batch = candidates[start:start + batch_size]
            batch_smiles = [r["smiles"] for r in batch]
            token_ids, attn_mask = tokenizer.batch_encode(batch_smiles)
            token_ids = token_ids.to(device)
            attn_mask = attn_mask.to(device)
            logits = model(token_ids, attn_mask)
            probs = torch.sigmoid(logits)

            for i, cand in enumerate(batch):
                score = probs[i].item()
                logit = logits[i].item()
                label = cand["label"]
                all_logits.append(logit)
                all_labels.append(label)
                raw_predictions.append({
                    "group_id": cand["group_id"],
                    "candidate_id": cand["candidate_id"],
                    "label": label,
                    "score": score,
                    "logit": logit,
                    "candidate_source": cand.get("candidate_source", ""),
                    "reaction_family": cand.get("reaction_family", ""),
                    "edit_type": cand.get("edit_type", ""),
                })

    latency_end = time.time()
    inference_latency_ms = ((latency_end - latency_start) / max(n_candidates, 1)) * 1000

    # Grouped ranking metrics
    ranking = ranking_metrics(raw_predictions)

    # Calibration metrics
    logits_tensor = torch.tensor(all_logits)
    labels_tensor = torch.tensor(all_labels)
    calibration = compute_calibration_metrics(logits_tensor, labels_tensor)

    return {
        "ranking": ranking,
        "calibration": calibration,
        "raw_predictions": raw_predictions,
        "inference_latency_ms": round(inference_latency_ms, 4),
        "n_candidates": n_candidates,
    }


# ---------------------------------------------------------------------------
# Run one config × seed
# ---------------------------------------------------------------------------

def run_config_seed(
    config: AblationConfig,
    seed: int,
    train_data: List[dict],
    val_data: List[dict],
    test_data: List[dict],
    checkpoint_path: Path,
    vocab_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run one config with one seed. Returns full metrics + predictions."""
    set_seed(seed)

    wall_start = time.time()

    # Build model (this initializes CUDA context via model.to(device))
    model, tokenizer, trainable_params = build_model(config, checkpoint_path, vocab_path, device)
    total_params = count_total_parameters(model)

    # Track peak memory (reset after model build so CUDA context is initialized)
    peak_memory_mb = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    # Train (skip for C1 frozen)
    if config.adapter == "none":
        training_result = {"history": [], "best_val_loss": 0.0}
    else:
        training_result = train_model(
            model, tokenizer, train_data, val_data,
            epochs=epochs, batch_size=batch_size, lr=lr, device=device,
        )

    # Evaluate on val and test
    val_eval = evaluate_ranking(model, tokenizer, val_data, device, batch_size)
    test_eval = evaluate_ranking(model, tokenizer, test_data, device, batch_size)

    wall_end = time.time()
    wall_clock_seconds = wall_end - wall_start

    # Peak memory after training + eval
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    result = {
        "config_id": config.config_id,
        "config_name": config.name,
        "seed": seed,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "param_ratio": round(trainable_params / max(total_params, 1), 6),
        "wall_clock_seconds": round(wall_clock_seconds, 2),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "inference_latency_ms": test_eval["inference_latency_ms"],
        "val_metrics": {
            "mrr": val_eval["ranking"]["mrr"],
            "top1": val_eval["ranking"]["top1"],
            "top3": val_eval["ranking"]["top3"],
            "ndcg": val_eval["ranking"]["ndcg"],
            "ece": val_eval["calibration"]["ece"],
            "brier": val_eval["calibration"]["brier"],
        },
        "test_metrics": {
            "mrr": test_eval["ranking"]["mrr"],
            "top1": test_eval["ranking"]["top1"],
            "top3": test_eval["ranking"]["top3"],
            "ndcg": test_eval["ranking"]["ndcg"],
            "ece": test_eval["calibration"]["ece"],
            "brier": test_eval["calibration"]["brier"],
        },
        "training_history": training_result["history"],
        "best_val_loss": training_result["best_val_loss"],
    }

    # Save raw predictions
    pred_dir = output_dir / "raw_predictions" / config.config_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"seed{seed}_predictions.json"
    with open(pred_path, "w") as f:
        json.dump({
            "config_id": config.config_id,
            "seed": seed,
            "val_predictions": val_eval["raw_predictions"],
            "test_predictions": test_eval["raw_predictions"],
        }, f, indent=2)

    # Save per-seed metrics
    metrics_dir = output_dir / "per_seed" / config.config_id
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"seed{seed}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Non-inferiority test
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    treatment_scores: List[float],
    baseline_scores: List[float],
    n_bootstrap: int = 10000,
    seed: int = 20260721,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired bootstrap CI of (treatment - baseline) differences."""
    import numpy as np
    if len(treatment_scores) != len(baseline_scores):
        raise ValueError(f"Score arrays must be equal length: {len(treatment_scores)} vs {len(baseline_scores)}")
    n = len(treatment_scores)
    if n == 0:
        return {"delta_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "n": 0}

    deltas = np.array(treatment_scores) - np.array(baseline_scores)
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        np.mean(rng.choice(deltas, size=n, replace=True)) for _ in range(n_bootstrap)
    ])
    ci_low = float(np.quantile(boot_means, alpha / 2))
    ci_high = float(np.quantile(boot_means, 1 - alpha / 2))
    delta_mean = float(np.mean(deltas))
    p_value = float(np.mean(boot_means <= 0))

    return {
        "delta_mean": round(delta_mean, 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "p_value": round(p_value, 6),
        "n": n,
    }


def noninferiority_test(
    lora_results_by_config: Dict[str, List[dict]],
    full_ft_results: List[dict],
    margin: float = NONINFERIORITY_MARGIN,
    metric_key: str = "mrr",
    split: str = "test",
) -> Dict[str, Any]:
    """Test each LoRA config for non-inferiority vs full fine-tuning.

    GO condition: ci_low of (LoRA - FullFT) MRR difference > margin.
    Margin is -0.005 (i.e., LoRA must not be more than 0.5pp worse).
    """
    full_ft_scores = [r[f"{split}_metrics"][metric_key] for r in full_ft_results]

    results = {}
    for config_id, seed_results in lora_results_by_config.items():
        if len(seed_results) < 2 or len(full_ft_results) < 2:
            # Need at least 2 seeds for bootstrap
            results[config_id] = {
                "status": "INSUFFICIENT_SEEDS",
                "n_lora_seeds": len(seed_results),
                "n_full_ft_seeds": len(full_ft_results),
            }
            continue

        # Match seeds (use min length)
        min_n = min(len(seed_results), len(full_ft_results))
        lora_scores = [r[f"{split}_metrics"][metric_key] for r in seed_results[:min_n]]
        full_scores = full_ft_scores[:min_n]

        ci_result = paired_bootstrap_ci(lora_scores, full_scores)
        is_noninferior = ci_result["ci_low"] > margin

        results[config_id] = {
            "status": "NONINFERIOR" if is_noninferior else "INFERIOR",
            "metric": metric_key,
            "split": split,
            "margin": margin,
            "lora_mean_mrr": round(statistics.mean(lora_scores), 6) if lora_scores else 0,
            "full_ft_mean_mrr": round(statistics.mean(full_scores), 6) if full_scores else 0,
            "delta_mean": ci_result["delta_mean"],
            "ci_low": ci_result["ci_low"],
            "ci_high": ci_result["ci_high"],
            "p_value": ci_result["p_value"],
            "n_seeds": min_n,
            "is_noninferior": is_noninferior,
        }

    return results


# ---------------------------------------------------------------------------
# Backbone selection
# ---------------------------------------------------------------------------

def select_best_backbone(
    config_registry: dict,
    all_results: Dict[str, List[dict]],
    noninferiority: dict,
    metric_key: str = "mrr",
    split: str = "test",
    allow_partial_go: bool = False,
) -> Optional[dict]:
    """Select the best LoRA config with fewest trainable params.

    Selection rule:
    1. Among configs that are NONINFERIOR, pick the one with the highest mean MRR.
    2. If tie, pick fewer trainable parameters.
    3. If no LoRA config is non-inferior:
       - When allow_partial_go=False, return None (NO-GO).
       - When allow_partial_go=True, pick the LoRA config with the highest mean
         MRR anyway (PARTIAL_GO).  The spec allows PARTIAL_GO when "LoRA is
         slightly below full fine-tuning but efficiency advantage is clear".
    """
    candidates = []
    fallback_candidates = []
    for config_id, results in all_results.items():
        if config_id == "C6" or config_id == "C1":
            continue  # Skip full FT and frozen
        if len(results) == 0:
            continue
        mean_mrr = statistics.mean([r[f"{split}_metrics"][metric_key] for r in results])
        mean_params = statistics.mean([r["trainable_parameters"] for r in results])
        ni = noninferiority.get(config_id, {})
        entry = {
            "config_id": config_id,
            "mean_mrr": mean_mrr,
            "mean_trainable_params": mean_params,
            "n_seeds": len(results),
            "is_noninferior": ni.get("is_noninferior", False),
        }
        fallback_candidates.append(entry)
        if ni.get("is_noninferior", False):
            candidates.append(entry)

    # PARTIAL_GO fallback: no non-inferior config, but select best LoRA anyway
    if not candidates and allow_partial_go and fallback_candidates:
        candidates = fallback_candidates
        selection_rule = "highest_mean_mrr_among_all_lora_then_fewest_params_partial_go"
    elif candidates:
        selection_rule = "highest_mean_mrr_among_noninferior_then_fewest_params"
    else:
        return None

    # Sort by MRR desc, then params asc
    candidates.sort(key=lambda x: (-x["mean_mrr"], x["mean_trainable_params"]))
    best = candidates[0]
    config = config_registry["configs"][best["config_id"]]

    return {
        "config_id": best["config_id"],
        "checkpoint": str(DEFAULT_CHECKPOINT_PATH),
        "checkpoint_hash": compute_checkpoint_hash(DEFAULT_CHECKPOINT_PATH),
        "architecture": "PretrainedChemformerBackbone (encoder-only, 6 layers, d=512, 8 heads)",
        "target_modules": config["target_patterns"],
        "rank": config["rank"],
        "alpha": config["alpha"],
        "dropout": config["dropout"],
        "trainable_parameters": int(best["mean_trainable_params"]),
        "training_budget": {
            "epochs": DEFAULT_EPOCHS_FULL,
            "batch_size": DEFAULT_BATCH_SIZE,
            "lr": DEFAULT_LR,
        },
        "selection_metric": f"{split}_{metric_key}",
        "selection_rule": selection_rule,
        "mean_mrr": round(best["mean_mrr"], 6),
        "n_seeds": best["n_seeds"],
        "is_noninferior": best["is_noninferior"],
    }


def compute_checkpoint_hash(checkpoint_path: Path) -> str:
    """Compute SHA-256 of checkpoint file."""
    if not checkpoint_path.exists():
        return ""
    h = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(all_results: Dict[str, List[dict]], output_path: Path) -> None:
    """Write summary.csv with per-config, per-seed metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for config_id, seed_results in all_results.items():
        for r in seed_results:
            rows.append({
                "config_id": r["config_id"],
                "config_name": r["config_name"],
                "seed": r["seed"],
                "trainable_parameters": r["trainable_parameters"],
                "total_parameters": r["total_parameters"],
                "param_ratio": r["param_ratio"],
                "wall_clock_seconds": r["wall_clock_seconds"],
                "peak_memory_mb": r["peak_memory_mb"],
                "inference_latency_ms": r["inference_latency_ms"],
                "val_mrr": r["val_metrics"]["mrr"],
                "val_top1": r["val_metrics"]["top1"],
                "val_top3": r["val_metrics"]["top3"],
                "val_ndcg": r["val_metrics"]["ndcg"],
                "val_ece": r["val_metrics"]["ece"],
                "val_brier": r["val_metrics"]["brier"],
                "test_mrr": r["test_metrics"]["mrr"],
                "test_top1": r["test_metrics"]["top1"],
                "test_top3": r["test_metrics"]["top3"],
                "test_ndcg": r["test_metrics"]["ndcg"],
                "test_ece": r["test_metrics"]["ece"],
                "test_brier": r["test_metrics"]["brier"],
            })

    if not rows:
        return

    fields = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# GO/NO-GO
# ---------------------------------------------------------------------------

def compute_go_no_go(
    all_results: Dict[str, List[dict]],
    noninferiority: dict,
    selected_backbone: Optional[dict],
    full_ft_results: List[dict],
) -> dict:
    """Compute GO/NO-GO verdict per spec.

    GO:
    - At least one LoRA config is non-inferior;
    - trainable parameters of selected config < full FT / 10;
    - checkpoint and config frozen.
    PARTIAL GO:
    - LoRA slightly below full FT but efficiency advantage clear.
    NO-GO:
    - Full FT significantly better than all parameter-efficient methods.
    """
    # Check if any LoRA config is non-inferior
    any_noninferior = any(
        ni.get("is_noninferior", False)
        for cid, ni in noninferiority.items()
        if cid not in ("C1", "C6")
    )

    # Check param efficiency
    param_efficient = False
    if selected_backbone and full_ft_results:
        selected_params = selected_backbone["trainable_parameters"]
        full_ft_params = full_ft_results[0]["trainable_parameters"]
        param_efficient = selected_params * 10 <= full_ft_params

    # Check config frozen
    config_frozen = selected_backbone is not None and "checkpoint_hash" in selected_backbone

    if any_noninferior and param_efficient and config_frozen:
        status = "GO"
    elif selected_backbone and not any_noninferior and param_efficient:
        # LoRA slightly below full FT but efficiency advantage is clear (≥10x)
        status = "PARTIAL_GO"
    else:
        status = "NO_GO"

    # Seed variance for selected config
    seed_variance = {}
    if selected_backbone:
        config_id = selected_backbone["config_id"]
        if config_id in all_results and len(all_results[config_id]) > 1:
            mrrs = [r["test_metrics"]["mrr"] for r in all_results[config_id]]
            seed_variance = {
                "mrr_mean": round(statistics.mean(mrrs), 6),
                "mrr_std": round(statistics.stdev(mrrs), 6) if len(mrrs) > 1 else 0,
                "mrr_min": round(min(mrrs), 6),
                "mrr_max": round(max(mrrs), 6),
            }

    return {
        "phase": PHASE,
        "status": status,
        "primary_metric": {
            "selected_config": selected_backbone["config_id"] if selected_backbone else None,
            "selected_mrr": selected_backbone["mean_mrr"] if selected_backbone else None,
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "any_noninferior": any_noninferior,
            "param_efficient": param_efficient,
            "config_frozen": config_frozen,
            "seed_variance": seed_variance,
        },
        "predeclared_threshold": {
            "primary_metric": "MRR",
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "param_efficiency_ratio": 10,
            "min_seeds": 10,
        },
        "evidence_paths": [
            "results/p4_lora_ablation/config_registry.json",
            "results/p4_lora_ablation/summary.csv",
            "results/p4_lora_ablation/noninferiority.json",
            "results/p4_lora_ablation/selected_backbone.json",
        ],
        "next_phase_allowed": status in ("GO", "PARTIAL_GO"),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="P4-G2 LoRA Ablation")
    parser.add_argument("--manifest", type=Path, default=Path("data/p4/manifests/hte_feasibility_v1.json"),
                        help="Path to P4-G1 manifest JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("results/p4_lora_ablation"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--stage", type=str, default="full",
                        choices=["smoke", "screening", "final", "full"],
                        help="Which stage to run")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config IDs to run (default: all)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (overrides stage defaults)")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[P4-G2] Stage: {args.stage}")
    print(f"[P4-G2] Manifest: {args.manifest}")
    print(f"[P4-G2] Output: {output_dir}")
    print(f"[P4-G2] Device: {args.device}")

    # Load manifest data
    print("[P4-G2] Loading manifest data...")
    splits = load_manifest_candidates(args.manifest)
    train_data = splits["train"]
    val_data = splits["val"]
    test_data = splits["test"]
    print(f"  Train: {len(train_data)} candidates ({sum(r['label'] for r in train_data)} gold)")
    print(f"  Val:   {len(val_data)} candidates ({sum(r['label'] for r in val_data)} gold)")
    print(f"  Test:  {len(test_data)} candidates ({sum(r['label'] for r in test_data)} gold)")

    # Build config registry (auto-discover module names)
    print("[P4-G2] Auto-discovering module names...")
    # Build a temporary model to discover modules
    backbone = PretrainedChemformerBackbone(checkpoint_path=args.checkpoint, freeze=True)
    scorer = PretrainedReactionScorer(backbone)
    config_registry = build_config_registry(scorer)

    # Save config registry
    reg_path = output_dir / "config_registry.json"
    with open(reg_path, "w") as f:
        json.dump(config_registry, f, indent=2)
    print(f"  Config registry saved: {reg_path}")
    print(f"  Discovered Linear modules: {len(config_registry['module_registry'])}")
    for scope, patterns in config_registry["discovered_patterns"].items():
        print(f"  {scope} patterns: {patterns}")

    # Determine configs and seeds
    if args.configs:
        config_ids = args.configs.split(",")
    else:
        config_ids = CONFIG_IDS

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    elif args.stage == "smoke":
        seeds = DEFAULT_SEEDS_SMOKE
        epochs = args.epochs or DEFAULT_EPOCHS_SMOKE
    elif args.stage == "screening":
        seeds = DEFAULT_SEEDS_SCREENING
        epochs = args.epochs or DEFAULT_EPOCHS_FULL
    elif args.stage == "final":
        seeds = DEFAULT_SEEDS_FINAL
        epochs = args.epochs or DEFAULT_EPOCHS_FULL
    else:  # full
        seeds = DEFAULT_SEEDS_SMOKE
        epochs = args.epochs or DEFAULT_EPOCHS_SMOKE

    print(f"[P4-G2] Configs: {config_ids}")
    print(f"[P4-G2] Seeds: {seeds}")
    print(f"[P4-G2] Epochs: {epochs}")

    # Run all config × seed combinations
    all_results: Dict[str, List[dict]] = {}
    for config_id in config_ids:
        config = AblationConfig(**config_registry["configs"][config_id])
        all_results[config_id] = []

        for seed in seeds:
            print(f"\n[P4-G2] Running {config_id} ({config.name}) seed={seed}...")
            result = run_config_seed(
                config=config,
                seed=seed,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                checkpoint_path=args.checkpoint,
                vocab_path=args.vocab,
                epochs=epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                output_dir=output_dir,
            )
            all_results[config_id].append(result)

            print(f"  trainable_params: {result['trainable_parameters']}")
            print(f"  test MRR: {result['test_metrics']['mrr']:.4f}  Top1: {result['test_metrics']['top1']:.4f}  NDCG: {result['test_metrics']['ndcg']:.4f}")
            print(f"  wall_clock: {result['wall_clock_seconds']:.1f}s  peak_mem: {result['peak_memory_mb']:.0f}MB")

    # Write summary CSV
    write_summary_csv(all_results, output_dir / "summary.csv")
    print(f"\n[P4-G2] Summary CSV: {output_dir / 'summary.csv'}")

    # Non-inferiority test (only if we have C6 full FT results)
    noninferiority = {}
    if "C6" in all_results and len(all_results["C6"]) > 0:
        lora_results = {k: v for k, v in all_results.items() if k not in ("C1", "C6")}
        full_ft_results = all_results.get("C6", [])
        noninferiority = noninferiority_test(lora_results, full_ft_results)

        ni_path = output_dir / "noninferiority.json"
        with open(ni_path, "w") as f:
            json.dump(noninferiority, f, indent=2)
        print(f"[P4-G2] Non-inferiority: {ni_path}")
        for cid, ni in noninferiority.items():
            print(f"  {cid}: {ni.get('status', '?')} (delta_mean={ni.get('delta_mean', '?')}, ci_low={ni.get('ci_low', '?')})")

    # Select best backbone.
    # For final/full stage: strict (only non-inferior configs).
    # For screening stage: allow PARTIAL_GO fallback (best LoRA even if all
    #   INFERIOR), because per spec we skip 10-seed when no config passes
    #   screening, and must still freeze a backbone for the PARTIAL_GO verdict.
    selected_backbone = None
    if noninferiority:
        allow_partial = args.stage in ("screening", "final", "full")
        selected_backbone = select_best_backbone(
            config_registry, all_results, noninferiority,
            allow_partial_go=allow_partial,
        )
        if selected_backbone:
            sb_path = output_dir / "selected_backbone.json"
            with open(sb_path, "w") as f:
                json.dump(selected_backbone, f, indent=2)
            print(f"[P4-G2] Selected backbone: {sb_path}")
            print(f"  Config: {selected_backbone['config_id']}")
            print(f"  MRR: {selected_backbone['mean_mrr']}")
            print(f"  Trainable params: {selected_backbone['trainable_parameters']}")
            print(f"  Non-inferior: {selected_backbone.get('is_noninferior', False)}")
        else:
            print("[P4-G2] No LoRA config available for selection (NO-GO)")

    # GO/NO-GO
    go_no_go = compute_go_no_go(
        all_results, noninferiority, selected_backbone,
        all_results.get("C6", []),
    )
    if go_no_go["status"] == "PARTIAL_GO":
        go_no_go["partial_go_conditions"] = {
            "reason": "LoRA slightly below full fine-tuning but efficiency advantage clear (>=10x fewer params)",
            "requirement": "Formal augmentation main results MUST simultaneously report full fine-tuning sensitivity",
            "selected_config": selected_backbone["config_id"] if selected_backbone else None,
            "param_efficiency_ratio": (
                all_results["C6"][0]["trainable_parameters"] / selected_backbone["trainable_parameters"]
                if selected_backbone and "C6" in all_results and all_results["C6"] else None
            ),
        }
    go_path = output_dir / "go_no_go.json"
    with open(go_path, "w") as f:
        json.dump(go_no_go, f, indent=2)
    print(f"\n[P4-G2] GO/NO-GO: {go_path}")
    print(f"  Status: {go_no_go['status']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
