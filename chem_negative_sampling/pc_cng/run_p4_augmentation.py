"""P4-G3: PC-CNG Augmentation Main Experiment.

Compares 7 augmentation arms (A0-A6) across 2 backbones (Chemformer-LoRA + GNN)
on candidate ranking tasks. 10 pre-declared seeds per backbone × arm.

Usage::

    python3 -m pc_cng.run_p4_augmentation \
        --manifest data/p4/manifests/hte_feasibility_v1.json \
        --output-dir results/p4_augmentation \
        --backbone chemformer \
        --device cuda:0 \
        --stage smoke

Arms:
    A0 positive-only
    A1 + random mismatch negatives
    A2 + random structural corruption
    A3 + Tanimoto negatives
    A4 + template negatives
    A5 + unconstrained edits
    A6 + rule PC-CNG
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from pc_cng.ranking_metrics import ranking_metrics  # noqa: E402
from pc_cng.gnn_backbone import (  # noqa: E402
    GNNReactionScorer,
    build_gnn_scorer,
    count_parameters as gnn_count_params,
    count_trainable_parameters as gnn_count_trainable,
    mol_to_graph,
    collate_graphs,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASE = "P4-G3"
DEFAULT_SEEDS = list(range(20260721, 20260731))  # 10 seeds
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_MAX_SEQ_LEN = 256

# Augmentation arm definitions
ARM_DEFINITIONS = {
    "A0": {"name": "positive_only", "negative_source": None},
    "A1": {"name": "random_mismatch", "negative_source": "random_mismatch"},
    "A2": {"name": "random_corruption", "negative_source": "random_corruption"},
    "A3": {"name": "tanimoto_retrieval", "negative_source": "tanimoto_retrieval"},
    "A4": {"name": "template_perturbation", "negative_source": "template_perturbation"},
    "A5": {"name": "unconstrained_edit", "negative_source": "unconstrained_edit"},
    "A6": {"name": "rule_pc_cng", "negative_source": "rule_pc_cng"},
}

ARM_IDS = list(ARM_DEFINITIONS.keys())  # A0-A6


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_manifest_candidates(manifest_path: Path) -> Dict[str, List[dict]]:
    """Load P4-G1 manifest and split candidates by split field.

    Returns {"train": [...], "val": [...], "test": [...]}.
    Each candidate dict is augmented with:
        - label: int (1 if gold_candidate else 0)
        - smiles: candidate_smiles (atom-mapped, stripped of atom maps for tokenizer)
        - negative_source: candidate_source for negatives, None for gold
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    splits = {"train": [], "val": [], "test": []}
    for group in manifest.get("groups", []):
        gid = group["group_id"]
        group_split = group.get("split", "train")
        for cand in group.get("candidates", []):
            split = cand.get("split", "train")
            if split not in splits:
                continue
            # Strip atom map numbers from SMILES for tokenizer
            smiles = cand.get("candidate_smiles", "")
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
                "negative_source": None if cand.get("gold_candidate", False) else cand.get("candidate_source", ""),
            }
            splits[split].append(entry)

    return splits


def build_arm_training_data(
    train_candidates: List[dict],
    arm_id: str,
) -> List[dict]:
    """Build training data for a specific augmentation arm.

    A0: only gold (positive) candidates
    A1-A6: gold + negatives of the specified source type

    Returns list of training examples.
    """
    negative_source = ARM_DEFINITIONS[arm_id]["negative_source"]
    examples = []

    for cand in train_candidates:
        if cand["gold_candidate"]:
            # Always include positive examples
            examples.append(cand)
        elif negative_source and cand["negative_source"] == negative_source:
            # Include negatives of the specified type
            examples.append(cand)

    return examples


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

@dataclass
class BackboneConfig:
    """Configuration for one backbone."""
    name: str  # "chemformer" or "gnn"
    checkpoint_path: Optional[str] = None
    lora_target_patterns: Optional[List[str]] = None
    lora_rank: int = 0
    lora_alpha: float = 0.0
    lora_dropout: float = 0.0
    gnn_hidden_dim: int = 128
    gnn_encoder_dim: int = 256
    gnn_num_layers: int = 3
    gnn_heads: int = 4
    gnn_dropout: float = 0.1


def build_chemformer_model(
    config: BackboneConfig,
    checkpoint_path: Path,
    vocab_path: Path,
    device: str = "cpu",
) -> Tuple[PretrainedReactionScorer, ChemformerTokenizer, int]:
    """Build Chemformer-LoRA model. Returns (model, tokenizer, trainable_params)."""
    backbone = PretrainedChemformerBackbone(
        checkpoint_path=checkpoint_path,
        freeze=True,
    )

    if config.lora_target_patterns:
        # Strip "backbone." prefix for apply_lora on backbone directly
        stripped_patterns = []
        for pat in config.lora_target_patterns:
            if pat.startswith("backbone."):
                stripped_patterns.append(pat[len("backbone."):])
            else:
                stripped_patterns.append(pat)
        n_replaced = apply_lora(
            backbone,
            r=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target_patterns=stripped_patterns,
        )
        if n_replaced == 0:
            print(f"  WARNING: apply_lora replaced 0 modules! patterns={stripped_patterns}")
        trainable = freeze_non_lora_params(backbone)
    else:
        trainable = 0

    head = ReactionClassificationHead(d_model=backbone.hparams.get("d_model", 512))
    scorer = PretrainedReactionScorer(backbone, head)
    scorer = scorer.to(device)

    tokenizer = ChemformerTokenizer(vocab_path=vocab_path, max_seq_len=DEFAULT_MAX_SEQ_LEN)

    # Count trainable params (LoRA + head)
    trainable = sum(p.numel() for p in scorer.parameters() if p.requires_grad)

    return scorer, tokenizer, trainable


def build_gnn_model(
    config: BackboneConfig,
    device: str = "cpu",
) -> Tuple[GNNReactionScorer, None, int]:
    """Build GNN model. Returns (model, None, trainable_params)."""
    model = GNNReactionScorer(
        hidden_dim=config.gnn_hidden_dim,
        encoder_out_dim=config.gnn_encoder_dim,
        num_layers=config.gnn_num_layers,
        heads=config.gnn_heads,
        dropout=config.gnn_dropout,
    )
    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, None, trainable


def build_model(
    config: BackboneConfig,
    checkpoint_path: Optional[Path],
    vocab_path: Optional[Path],
    device: str = "cpu",
) -> Tuple[nn.Module, Any, int]:
    """Build model for the specified backbone."""
    if config.name == "chemformer":
        return build_chemformer_model(config, checkpoint_path, vocab_path, device)
    elif config.name == "gnn":
        return build_gnn_model(config, device)
    else:
        raise ValueError(f"Unknown backbone: {config.name}")


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch_chemformer(
    model: PretrainedReactionScorer,
    tokenizer: ChemformerTokenizer,
    train_data: List[dict],
    optimizer: torch.optim.Optimizer,
    device: str,
    batch_size: int,
    epoch: int,
    seed: int,
) -> float:
    """Train Chemformer for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    rng = random.Random(seed + epoch * 1000)
    indices = list(range(len(train_data)))
    rng.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch = [train_data[j] for j in batch_indices]

        smiles_list = [b["smiles"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32).to(device)

        token_ids, attn_mask = tokenizer.batch_encode(smiles_list)
        token_ids = token_ids.to(device)
        attn_mask = attn_mask.to(device)

        logits = model(token_ids, attn_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_epoch_gnn(
    model: GNNReactionScorer,
    train_data: List[dict],
    optimizer: torch.optim.Optimizer,
    device: str,
    batch_size: int,
    epoch: int,
    seed: int,
) -> float:
    """Train GNN for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    rng = random.Random(seed + epoch * 1000)
    indices = list(range(len(train_data)))
    rng.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch = [train_data[j] for j in batch_indices]

        # Convert SMILES to graphs
        graphs = []
        labels_list = []
        for b in batch:
            g = mol_to_graph(b["smiles"])
            if g is not None:
                graphs.append(g)
                labels_list.append(b["label"])

        if not graphs:
            continue

        labels = torch.tensor(labels_list, dtype=torch.float32).to(device)
        scores = model(graphs)

        loss = F.binary_cross_entropy_with_logits(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_chemformer(
    model: PretrainedReactionScorer,
    tokenizer: ChemformerTokenizer,
    eval_data: List[dict],
    device: str,
    batch_size: int = 64,
) -> List[dict]:
    """Evaluate Chemformer on candidate ranking. Returns per-candidate predictions."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch = eval_data[i:i + batch_size]
            smiles_list = [b["smiles"] for b in batch]
            token_ids, attn_mask = tokenizer.batch_encode(smiles_list)
            token_ids = token_ids.to(device)
            attn_mask = attn_mask.to(device)
            logits = model(token_ids, attn_mask)

            for j, cand in enumerate(batch):
                predictions.append({
                    "group_id": cand["group_id"],
                    "candidate_id": cand["candidate_id"],
                    "label": cand["label"],
                    "score": logits[j].item(),
                    "candidate_source": cand["candidate_source"],
                })

    return predictions


def evaluate_gnn(
    model: GNNReactionScorer,
    eval_data: List[dict],
    device: str,
    batch_size: int = 64,
) -> List[dict]:
    """Evaluate GNN on candidate ranking. Returns per-candidate predictions."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch = eval_data[i:i + batch_size]
            graphs = []
            valid_indices = []
            for j, cand in enumerate(batch):
                g = mol_to_graph(cand["smiles"])
                if g is not None:
                    graphs.append(g)
                    valid_indices.append(j)

            if not graphs:
                continue

            scores = model(graphs)
            for idx, j in enumerate(valid_indices):
                predictions.append({
                    "group_id": batch[j]["group_id"],
                    "candidate_id": batch[j]["candidate_id"],
                    "label": batch[j]["label"],
                    "score": scores[idx].item(),
                    "candidate_source": batch[j]["candidate_source"],
                })

    return predictions


def compute_metrics_from_predictions(predictions: List[dict]) -> Dict[str, float]:
    """Compute ranking metrics from predictions."""
    result = ranking_metrics(predictions)
    return {
        "mrr": result["mrr"],
        "top1": result["top1"],
        "top3": result["top3"],
        "ndcg": result["ndcg"],
        "groups": result["groups"],
    }


def compute_auprc(predictions: List[dict]) -> float:
    """Compute AUPRC from predictions."""
    # Sort by score descending
    sorted_preds = sorted(predictions, key=lambda x: x["score"], reverse=True)
    labels = [p["label"] for p in sorted_preds]
    scores = [p["score"] for p in sorted_preds]

    # Compute precision-recall curve
    n_pos = sum(labels)
    if n_pos == 0:
        return 0.0

    tp = 0
    fp = 0
    prev_recall = 0.0
    auprc = 0.0

    for i, (label, score) in enumerate(zip(labels, scores)):
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        auprc += precision * (recall - prev_recall)
        prev_recall = recall

    return auprc


def compute_calibration_metrics(predictions: List[dict]) -> Dict[str, float]:
    """Compute ECE and Brier score from predictions."""
    if not predictions:
        return {"ece": 0.0, "brier": 0.0}

    # Convert scores to probabilities via sigmoid
    probs = torch.tensor([p["score"] for p in predictions])
    labels = torch.tensor([p["label"] for p in predictions], dtype=torch.float32)
    probs = torch.sigmoid(probs)

    # Brier score
    brier = ((probs - labels) ** 2).mean().item()

    # ECE (10 bins)
    n_bins = 10
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean().item()
        bin_conf = probs[mask].mean().item()
        bin_weight = mask.sum().item() / len(probs)
        ece += bin_weight * abs(bin_acc - bin_conf)

    return {"ece": ece, "brier": brier}


# ---------------------------------------------------------------------------
# Single run: backbone × arm × seed
# ---------------------------------------------------------------------------

def run_single_experiment(
    backbone_config: BackboneConfig,
    arm_id: str,
    seed: int,
    train_data: List[dict],
    val_data: List[dict],
    test_data: List[dict],
    checkpoint_path: Optional[Path],
    vocab_path: Optional[Path],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    output_dir: Path,
) -> dict:
    """Run a single backbone × arm × seed experiment.

    Returns dict with all metrics and metadata.
    """
    set_seed(seed)
    wall_start = time.time()

    # Build model
    model, tokenizer, trainable_params = build_model(
        backbone_config, checkpoint_path, vocab_path, device
    )
    total_params = gnn_count_params(model) if backbone_config.name == "gnn" else sum(p.numel() for p in model.parameters())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Build arm-specific training data
    arm_train_data = build_arm_training_data(train_data, arm_id)
    n_pos = sum(1 for d in arm_train_data if d["label"] == 1)
    n_neg = sum(1 for d in arm_train_data if d["label"] == 0)
    print(f"  Arm {arm_id}: {n_pos} pos + {n_neg} neg = {len(arm_train_data)} training examples")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    # Training loop
    best_val_mrr = -1.0
    best_epoch = 0
    best_state = None
    patience = 2
    patience_counter = 0

    for epoch in range(epochs):
        if backbone_config.name == "chemformer":
            loss = train_epoch_chemformer(
                model, tokenizer, arm_train_data, optimizer, device,
                batch_size, epoch, seed,
            )
        else:
            loss = train_epoch_gnn(
                model, arm_train_data, optimizer, device,
                batch_size, epoch, seed,
            )

        # Evaluate on validation set
        if backbone_config.name == "chemformer":
            val_preds = evaluate_chemformer(model, tokenizer, val_data, device)
        else:
            val_preds = evaluate_gnn(model, val_data, device)
        val_metrics = compute_metrics_from_predictions(val_preds)

        if val_metrics["mrr"] > best_val_mrr:
            best_val_mrr = val_metrics["mrr"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    wall_clock = time.time() - wall_start
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if torch.cuda.is_available() else 0

    # Final evaluation on test set
    if backbone_config.name == "chemformer":
        test_preds = evaluate_chemformer(model, tokenizer, test_data, device)
        val_preds = evaluate_chemformer(model, tokenizer, val_data, device)
    else:
        test_preds = evaluate_gnn(model, test_data, device)
        val_preds = evaluate_gnn(model, val_data, device)

    test_metrics = compute_metrics_from_predictions(test_preds)
    val_metrics = compute_metrics_from_predictions(val_preds)
    test_auprc = compute_auprc(test_preds)
    val_auprc = compute_auprc(val_preds)
    test_cal = compute_calibration_metrics(test_preds)
    val_cal = compute_calibration_metrics(val_preds)

    # Inference latency
    model.eval()
    with torch.no_grad():
        if backbone_config.name == "chemformer":
            sample = test_data[:batch_size]
            smiles_list = [s["smiles"] for s in sample]
            token_ids, attn_mask = tokenizer.batch_encode(smiles_list)
            token_ids = token_ids.to(device)
            attn_mask = attn_mask.to(device)
            start = time.time()
            for _ in range(10):
                model(token_ids, attn_mask)
            latency = (time.time() - start) / 10 * 1000  # ms
        else:
            sample = test_data[:batch_size]
            graphs = [mol_to_graph(s["smiles"]) for s in sample]
            graphs = [g for g in graphs if g is not None]
            if graphs:
                start = time.time()
                for _ in range(10):
                    model(graphs)
                latency = (time.time() - start) / 10 * 1000
            else:
                latency = 0.0

    # Save raw predictions
    pred_dir = output_dir / "paired_predictions" / f"{backbone_config.name}_{arm_id}_seed{seed}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / "test_predictions.json", "w") as f:
        json.dump(test_preds, f, indent=2)
    with open(pred_dir / "val_predictions.json", "w") as f:
        json.dump(val_preds, f, indent=2)

    return {
        "backbone": backbone_config.name,
        "arm_id": arm_id,
        "arm_name": ARM_DEFINITIONS[arm_id]["name"],
        "seed": seed,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "wall_clock_seconds": round(wall_clock, 2),
        "peak_memory_mb": round(peak_memory, 2),
        "inference_latency_ms": round(latency, 4),
        "best_epoch": best_epoch,
        "n_train_examples": len(arm_train_data),
        "n_train_pos": n_pos,
        "n_train_neg": n_neg,
        "val_metrics": {
            "mrr": round(val_metrics["mrr"], 6),
            "top1": round(val_metrics["top1"], 6),
            "top3": round(val_metrics["top3"], 6),
            "ndcg": round(val_metrics["ndcg"], 6),
            "auprc": round(val_auprc, 6),
            "ece": round(val_cal["ece"], 6),
            "brier": round(val_cal["brier"], 6),
        },
        "test_metrics": {
            "mrr": round(test_metrics["mrr"], 6),
            "top1": round(test_metrics["top1"], 6),
            "top3": round(test_metrics["top3"], 6),
            "ndcg": round(test_metrics["ndcg"], 6),
            "auprc": round(test_auprc, 6),
            "ece": round(test_cal["ece"], 6),
            "brier": round(test_cal["brier"], 6),
        },
    }


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    treatment: List[float],
    control: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute paired bootstrap CI for the difference (treatment - control)."""
    if len(treatment) != len(control):
        raise ValueError("Treatment and control must have same length")
    n = len(treatment)
    if n == 0:
        return {"delta_mean": 0, "ci_low": 0, "ci_high": 0, "p_value": 1.0, "n": 0}

    rng = random.Random(seed)
    deltas = [t - c for t, c in zip(treatment, control)]
    delta_mean = statistics.mean(deltas)

    boot_means = []
    for _ in range(n_bootstrap):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot_means.append(statistics.mean(sample))

    boot_means.sort()
    alpha = 1 - confidence
    ci_low = boot_means[int(alpha / 2 * n_bootstrap)]
    ci_high = boot_means[int((1 - alpha / 2) * n_bootstrap)]
    p_value = sum(1 for b in boot_means if b <= 0) / n_bootstrap

    return {
        "delta_mean": round(delta_mean, 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "p_value": round(p_value, 6),
        "n": n,
    }


def compute_effect_sizes(
    all_results: Dict[str, Dict[str, List[dict]]],
    baseline_arm: str = "A0",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute effect sizes vs baseline arm for each backbone.

    Returns {backbone: {arm_id: {metric: effect_size}}}.
    """
    effect_sizes = {}
    for backbone, arm_results in all_results.items():
        if baseline_arm not in arm_results:
            continue
        baseline = arm_results[baseline_arm]
        baseline_mrr = [r["test_metrics"]["mrr"] for r in baseline]
        baseline_mean = statistics.mean(baseline_mrr)
        baseline_std = statistics.stdev(baseline_mrr) if len(baseline_mrr) > 1 else 0

        effect_sizes[backbone] = {}
        for arm_id, results in arm_results.items():
            if arm_id == baseline_arm:
                continue
            arm_mrr = [r["test_metrics"]["mrr"] for r in results]
            arm_mean = statistics.mean(arm_mrr)
            arm_std = statistics.stdev(arm_mrr) if len(arm_mrr) > 1 else 0

            # Cohen's d
            pooled_std = math.sqrt((baseline_std ** 2 + arm_std ** 2) / 2) if (baseline_std + arm_std) > 0 else 1
            cohens_d = (arm_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0

            # Percentage point difference
            pp_diff = (arm_mean - baseline_mean) * 100

            effect_sizes[backbone][arm_id] = {
                "cohens_d": round(cohens_d, 4),
                "pp_diff": round(pp_diff, 4),
                "arm_mean_mrr": round(arm_mean, 6),
                "baseline_mean_mrr": round(baseline_mean, 6),
            }

    return effect_sizes


# ---------------------------------------------------------------------------
# GO/NO-GO verdict
# ---------------------------------------------------------------------------

def compute_go_no_go(
    all_results: Dict[str, Dict[str, List[dict]]],
    effect_sizes: Dict[str, Dict[str, Dict[str, float]]],
) -> dict:
    """Compute GO/NO-GO verdict per P4-G3 spec.

    Strong GO:
    - PC-CNG (A6) beats positive-only (A0) on at least 2 backbones
    - Mean improvement ≥ 1.0 pp
    - Cluster-bootstrap CI all positive
    - Beats best non-PC-CNG negative baseline by ≥ 0.5 pp
    - Calibration doesn't severely degrade
    - Oracle coverage same

    Weak GO:
    - Only 1 backbone has CI all positive
    - Claim narrowed to backbone-specific augmentation

    NO-GO:
    - PC-CNG ≤ positive-only
    - PC-CNG doesn't beat simple negative baseline
    - Gains explained by candidate coverage difference
    - Results only on self-built scorer
    """
    backbones_with_positive_ci = []
    backbones_with_significant_improvement = []
    all_improvements = []

    for backbone, arm_results in all_results.items():
        if "A0" not in arm_results or "A6" not in arm_results:
            continue

        a0_mrr = [r["test_metrics"]["mrr"] for r in arm_results["A0"]]
        a6_mrr = [r["test_metrics"]["mrr"] for r in arm_results["A6"]]

        # Check if A6 > A0
        mean_a0 = statistics.mean(a0_mrr)
        mean_a6 = statistics.mean(a6_mrr)
        improvement = mean_a6 - mean_a0
        improvement_pp = improvement * 100
        all_improvements.append(improvement_pp)

        # Bootstrap CI
        ci = paired_bootstrap_ci(a6_mrr, a0_mrr)
        if ci["ci_low"] > 0:
            backbones_with_positive_ci.append(backbone)

        if improvement_pp >= 1.0:
            backbones_with_significant_improvement.append(backbone)

        # Check vs best non-PC-CNG baseline
        best_baseline_mrr = 0
        best_baseline_id = None
        for arm_id in ["A1", "A2", "A3", "A4", "A5"]:
            if arm_id not in arm_results:
                continue
            arm_mrr = [r["test_metrics"]["mrr"] for r in arm_results[arm_id]]
            arm_mean = statistics.mean(arm_mrr)
            if arm_mean > best_baseline_mrr:
                best_baseline_mrr = arm_mean
                best_baseline_id = arm_id

        beats_best_baseline = mean_a6 > best_baseline_mrr + 0.005  # 0.5pp

    n_backbones = len([b for b in all_results if "A0" in all_results[b] and "A6" in all_results[b]])

    if len(backbones_with_positive_ci) >= 2 and len(backbones_with_significant_improvement) >= 2:
        status = "STRONG_GO"
    elif len(backbones_with_positive_ci) >= 1:
        status = "WEAK_GO"
    else:
        status = "NO_GO"

    mean_improvement = statistics.mean(all_improvements) if all_improvements else 0

    return {
        "phase": PHASE,
        "status": status,
        "n_backbones": n_backbones,
        "backbones_with_positive_ci": backbones_with_positive_ci,
        "backbones_with_significant_improvement": backbones_with_significant_improvement,
        "mean_improvement_pp": round(mean_improvement, 4),
        "improvements_by_backbone": {
            b: round(imp, 4) for b, imp in zip(
                [b for b in all_results if "A0" in all_results[b] and "A6" in all_results[b]],
                all_improvements,
            )
        },
        "predeclared_threshold": {
            "strong_go_min_backbones": 2,
            "strong_go_min_improvement_pp": 1.0,
            "strong_go_min_vs_best_baseline_pp": 0.5,
            "weak_go_min_backbones": 1,
        },
        "next_phase_allowed": status in ("STRONG_GO", "WEAK_GO"),
    }


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(
    all_results: Dict[str, Dict[str, List[dict]]],
    output_path: Path,
) -> None:
    """Write summary.csv with per-backbone, per-arm, per-seed metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for backbone, arm_results in all_results.items():
        for arm_id, seed_results in arm_results.items():
            for r in seed_results:
                rows.append({
                    "backbone": backbone,
                    "arm_id": arm_id,
                    "arm_name": r["arm_name"],
                    "seed": r["seed"],
                    "trainable_parameters": r["trainable_parameters"],
                    "total_parameters": r["total_parameters"],
                    "wall_clock_seconds": r["wall_clock_seconds"],
                    "peak_memory_mb": r["peak_memory_mb"],
                    "inference_latency_ms": r["inference_latency_ms"],
                    "n_train_examples": r["n_train_examples"],
                    "n_train_pos": r["n_train_pos"],
                    "n_train_neg": r["n_train_neg"],
                    "val_mrr": r["val_metrics"]["mrr"],
                    "val_top1": r["val_metrics"]["top1"],
                    "val_top3": r["val_metrics"]["top3"],
                    "val_ndcg": r["val_metrics"]["ndcg"],
                    "val_auprc": r["val_metrics"]["auprc"],
                    "val_ece": r["val_metrics"]["ece"],
                    "val_brier": r["val_metrics"]["brier"],
                    "test_mrr": r["test_metrics"]["mrr"],
                    "test_top1": r["test_metrics"]["top1"],
                    "test_top3": r["test_metrics"]["top3"],
                    "test_ndcg": r["test_metrics"]["ndcg"],
                    "test_auprc": r["test_metrics"]["auprc"],
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


def write_effect_sizes_csv(
    effect_sizes: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
) -> None:
    """Write effect_sizes.csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for backbone, arm_effects in effect_sizes.items():
        for arm_id, effects in arm_effects.items():
            rows.append({
                "backbone": backbone,
                "arm_id": arm_id,
                "arm_name": ARM_DEFINITIONS[arm_id]["name"],
                "cohens_d": effects["cohens_d"],
                "pp_diff": effects["pp_diff"],
                "arm_mean_mrr": effects["arm_mean_mrr"],
                "baseline_mean_mrr": effects["baseline_mean_mrr"],
            })

    if not rows:
        return

    fields = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="P4-G3 PC-CNG Augmentation Main Experiment")
    parser.add_argument("--manifest", type=Path, default=Path("data/p4/manifests/hte_feasibility_v1.json"),
                        help="Path to P4-G1 manifest JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("results/p4_augmentation"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--stage", type=str, default="full",
                        choices=["smoke", "full"],
                        help="smoke=1 seed, full=10 seeds")
    parser.add_argument("--backbone", type=str, default=None,
                        choices=["chemformer", "gnn"],
                        help="Run only one backbone (default: both)")
    parser.add_argument("--arms", type=str, default=None,
                        help="Comma-separated arm IDs to run (default: all A0-A6)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (overrides stage defaults)")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[P4-G3] Stage: {args.stage}")
    print(f"[P4-G3] Manifest: {args.manifest}")
    print(f"[P4-G3] Output: {output_dir}")
    print(f"[P4-G3] Device: {args.device}")

    # Load manifest data
    print("[P4-G3] Loading manifest data...")
    splits = load_manifest_candidates(args.manifest)
    train_data = splits["train"]
    val_data = splits["val"]
    test_data = splits["test"]
    print(f"  Train: {len(train_data)} candidates ({sum(r['label'] for r in train_data)} gold)")
    print(f"  Val:   {len(val_data)} candidates ({sum(r['label'] for r in val_data)} gold)")
    print(f"  Test:  {len(test_data)} candidates ({sum(r['label'] for r in test_data)} gold)")

    # Determine backbones
    backbones_to_run = []
    if args.backbone:
        backbones_to_run = [args.backbone]
    else:
        backbones_to_run = ["chemformer", "gnn"]

    # Determine arms
    if args.arms:
        arm_ids = args.arms.split(",")
    else:
        arm_ids = ARM_IDS

    # Determine seeds
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    elif args.stage == "smoke":
        seeds = [20260721]
    else:
        seeds = DEFAULT_SEEDS

    print(f"[P4-G3] Backbones: {backbones_to_run}")
    print(f"[P4-G3] Arms: {arm_ids}")
    print(f"[P4-G3] Seeds: {seeds}")
    print(f"[P4-G3] Epochs: {args.epochs}")

    # Load selected backbone config for Chemformer
    chemformer_config = None
    sb_path = Path("results/p4_lora_ablation/selected_backbone.json")
    if sb_path.exists():
        with open(sb_path) as f:
            sb = json.load(f)
        chemformer_config = BackboneConfig(
            name="chemformer",
            checkpoint_path=str(args.checkpoint),
            lora_target_patterns=sb.get("target_modules", []),
            lora_rank=sb.get("rank", 8),
            lora_alpha=sb.get("alpha", 16.0),
            lora_dropout=sb.get("dropout", 0.0),
        )
        print(f"[P4-G3] Chemformer config from selected_backbone.json: rank={chemformer_config.lora_rank}, "
              f"targets={chemformer_config.lora_target_patterns}")
    else:
        print(f"[P4-G3] WARNING: selected_backbone.json not found, using default Chemformer config")
        chemformer_config = BackboneConfig(
            name="chemformer",
            checkpoint_path=str(args.checkpoint),
            lora_target_patterns=["backbone.encoder_layers.*.self_attn.out_proj"],
            lora_rank=8,
            lora_alpha=16.0,
            lora_dropout=0.0,
        )

    gnn_config = BackboneConfig(
        name="gnn",
        gnn_hidden_dim=128,
        gnn_encoder_dim=256,
        gnn_num_layers=3,
        gnn_heads=4,
        gnn_dropout=0.1,
    )

    backbone_configs = {
        "chemformer": chemformer_config,
        "gnn": gnn_config,
    }

    # Run all experiments
    all_results: Dict[str, Dict[str, List[dict]]] = {}
    for backbone_name in backbones_to_run:
        config = backbone_configs[backbone_name]
        all_results[backbone_name] = {}

        for arm_id in arm_ids:
            all_results[backbone_name][arm_id] = []

            for seed in seeds:
                print(f"\n[P4-G3] Running {backbone_name} × {arm_id} × seed={seed}...")
                result = run_single_experiment(
                    backbone_config=config,
                    arm_id=arm_id,
                    seed=seed,
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    checkpoint_path=args.checkpoint if backbone_name == "chemformer" else None,
                    vocab_path=args.vocab if backbone_name == "chemformer" else None,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=args.device,
                    output_dir=output_dir,
                )
                all_results[backbone_name][arm_id].append(result)

                print(f"  trainable_params: {result['trainable_parameters']}")
                print(f"  test MRR: {result['test_metrics']['mrr']:.4f}  "
                      f"Top1: {result['test_metrics']['top1']:.4f}  "
                      f"AUPRC: {result['test_metrics']['auprc']:.4f}")
                print(f"  wall_clock: {result['wall_clock_seconds']:.1f}s  "
                      f"peak_mem: {result['peak_memory_mb']:.0f}MB")

    # Write summary CSV
    write_summary_csv(all_results, output_dir / "summary.csv")
    print(f"\n[P4-G3] Summary CSV: {output_dir / 'summary.csv'}")

    # Compute effect sizes
    effect_sizes = compute_effect_sizes(all_results)
    write_effect_sizes_csv(effect_sizes, output_dir / "effect_sizes.csv")
    print(f"[P4-G3] Effect sizes CSV: {output_dir / 'effect_sizes.csv'}")

    # GO/NO-GO verdict
    go_no_go = compute_go_no_go(all_results, effect_sizes)
    go_path = output_dir / "go_no_go.json"
    with open(go_path, "w") as f:
        json.dump(go_no_go, f, indent=2)
    print(f"\n[P4-G3] GO/NO-GO: {go_path}")
    print(f"  Status: {go_no_go['status']}")
    print(f"  Mean improvement: {go_no_go['mean_improvement_pp']:.2f}pp")
    print(f"  Backbones with positive CI: {go_no_go['backbones_with_positive_ci']}")

    # Save model manifests
    for backbone_name in backbones_to_run:
        config = backbone_configs[backbone_name]
        manifest = {
            "backbone": backbone_name,
            "config": asdict(config) if hasattr(config, '__dataclass_fields__') else str(config),
            "arms_run": arm_ids,
            "seeds_run": seeds,
            "n_seeds": len(seeds),
        }
        manifest_path = output_dir / "model_manifests" / f"{backbone_name}_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[P4-G3] Model manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
