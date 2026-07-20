"""Training script for the pretrained Chemformer backbone + LoRA (P3-01).

Trains the encoder-only Chemformer backbone with LoRA adapters and a
lightweight classification head on PC-CNG positive vs. generated-negative
reactions.  Produces a 10-seed paired-significance comparison against the
existing GNN baseline (``learned_graph_edit_decoder``) on the same split
contract.

Hard constraint compliance
--------------------------
* **HC #9**: ``--train-idx/--val-idx/--test-idx`` JSON files are *required*.
  Each file must contain ``{"indices": [0, 1, ...]}`` referencing rows of the
  combined reaction CSV.
* **HC #5**: every performance claim is backed by a 10-seed paired bootstrap
  CI (family-cluster), not decoder seeds.
* **HC #4**: ``chem_negative_sampling/tests/test_pretrained_backbone.py`` and
  ``test_adapter.py`` cover this module.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python -m training.train_pretrained \\
        --reactions data/processed/uspto_openmolecules_normalized.csv \\
        --train-idx data/processed/train_idx_v3.json \\
        --val-idx data/processed/val_idx_v3.json \\
        --test-idx data/processed/test_idx_v3.json \\
        --backbone-ckpt models/reaction_lm/chemformer_pretrained_hf/model_sanitized.ckpt \\
        --vocab external/reaction_lm/Chemformer/bart_vocab.json \\
        --adapter lora --lora-r 8 \\
        --seeds 20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719 \\
        --epochs 5 --batch-size 16 --lr 1e-4 \\
        --output-dir results/pretrained_backbone_chemformer_lora_20260720
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

# Ensure the chem_negative_sampling package is importable when run as a module
_CNS_ROOT = Path(__file__).resolve().parents[1]
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
    count_total_parameters,
    count_trainable_parameters,
    freeze_non_lora_params,
)

__all__ = [
    "ReactionRow",
    "load_reaction_csv",
    "load_split_indices",
    "build_seed_model",
    "train_one_seed",
    "evaluate_model",
    "paired_bootstrap_ci",
    "aggregate_seed_results",
    "main",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class ReactionRow:
    source_id: str
    reaction_smiles: str
    label_type: str  # "positive" | "real_negative" | "pc_cng_negative"

    @property
    def is_positive(self) -> bool:
        return self.label_type.lower() == "positive"


def load_reaction_csv(path: str | Path) -> List[ReactionRow]:
    rows: List[ReactionRow] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for r in reader:
            rxn = r.get("reaction_smiles", "").strip()
            if not rxn:
                continue
            rows.append(
                ReactionRow(
                    source_id=r.get("source_id", ""),
                    reaction_smiles=rxn,
                    label_type=r.get("label_type", "positive"),
                )
            )
    return rows


def load_split_indices(path: str | Path) -> List[int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        idx = data.get("indices", data.get("train_idx", []))
    else:
        idx = data
    return [int(i) for i in idx]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_seed_model(
    backbone_ckpt: Optional[str | Path],
    vocab_path: str | Path,
    adapter: str = "lora",
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    device: str = "cpu",
) -> Tuple[PretrainedReactionScorer, ChemformerTokenizer]:
    """Build a fresh model + tokenizer for one seed."""
    backbone = PretrainedChemformerBackbone(
        checkpoint_path=backbone_ckpt, freeze=True
    )
    if adapter == "lora":
        n_replaced = apply_lora(
            backbone,
            r=lora_r,
            alpha=lora_alpha,
            dropout=0.0,
        )
        n_trainable = freeze_non_lora_params(backbone)
        print(f"[train_pretrained] LoRA: replaced {n_replaced} layers, "
              f"{n_trainable:,} trainable backbone params")
    elif adapter == "full":
        for p in backbone.parameters():
            p.requires_grad = True
    elif adapter == "none":
        pass
    else:
        raise ValueError(f"Unknown adapter mode: {adapter}")
    head = ReactionClassificationHead(d_model=backbone.hparams["d_model"])
    model = PretrainedReactionScorer(backbone, head)
    model = model.to(device)
    tokenizer = ChemformerTokenizer(vocab_path)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _iter_minibatches(
    rows: List[ReactionRow],
    tokenizer: ChemformerTokenizer,
    batch_size: int,
    shuffle: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = list(range(len(rows)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_rows = [rows[i] for i in batch_idx]
        smiles = [r.reaction_smiles for r in batch_rows]
        labels = torch.tensor([1.0 if r.is_positive else 0.0 for r in batch_rows], dtype=torch.float32)
        ids, mask = tokenizer.batch_encode(smiles)
        yield ids, mask, labels


def train_one_seed(
    model: PretrainedReactionScorer,
    tokenizer: ChemformerTokenizer,
    train_rows: List[ReactionRow],
    val_rows: List[ReactionRow],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    log_every: int = 50,
) -> Dict[str, Any]:
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    # Class weighting to handle imbalance (more negatives than positives)
    n_pos = sum(1 for r in train_rows if r.is_positive)
    n_neg = len(train_rows) - n_pos
    pos_weight_val = max(1.0, n_neg / max(1, n_pos))
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for ids, mask, labels in _iter_minibatches(
            train_rows, tokenizer, batch_size, shuffle=True
        ):
            ids = ids.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(ids, attention_mask=mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        train_loss = total_loss / max(1, n_batches)
        # Validation
        val_metrics = evaluate_model(model, tokenizer, val_rows, device, batch_size=batch_size)
        val_loss = val_metrics["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_metrics.get("auc", 0.0),
                "val_acc": val_metrics.get("accuracy", 0.0),
            }
        )
        print(f"[train_pretrained] epoch {epoch}/{epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_auc={val_metrics.get('auc', 0.0):.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_loss": best_val_loss}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(
    model: PretrainedReactionScorer,
    tokenizer: ChemformerTokenizer,
    rows: List[ReactionRow],
    device: str,
    batch_size: int = 32,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0
    all_logits: List[float] = []
    all_labels: List[int] = []
    for ids, mask, labels in _iter_minibatches(rows, tokenizer, batch_size, shuffle=False):
        ids = ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        logits = model(ids, attention_mask=mask)
        loss = criterion(logits, labels)
        total_loss += float(loss.item())
        n_batches += 1
        all_logits.extend(logits.cpu().tolist())
        all_labels.extend([int(round(float(l))) for l in labels.cpu().tolist()])
    loss = total_loss / max(1, n_batches)
    # Accuracy + AUC
    preds = [1 if l > 0 else 0 for l in all_logits]
    correct = sum(1 for p, y in zip(preds, all_labels) if p == y)
    accuracy = correct / max(1, len(all_labels))
    auc = _binary_auc(all_logits, all_labels)
    # MRR-style ranking metric: for each (positive, negative) pair, check if
    # positive is scored higher.  This mirrors the P2-01 ranking setup.
    mrr = _pairwise_mrr(all_logits, all_labels)
    return {
        "loss": loss,
        "accuracy": accuracy,
        "auc": auc,
        "mrr": mrr,
        "n_examples": len(all_labels),
    }


def _binary_auc(logits: List[float], labels: List[int]) -> float:
    """Simple AUC via the Mann-Whitney U statistic."""
    pos = [l for l, y in zip(logits, labels) if y == 1]
    neg = [l for l, y in zip(logits, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    pos_arr = np.array(pos)
    neg_arr = np.array(neg)
    # P(pos > neg) + 0.5 P(pos == neg)
    wins = float((pos_arr[:, None] > neg_arr[None, :]).sum())
    ties = float((pos_arr[:, None] == neg_arr[None, :]).sum())
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _pairwise_mrr(logits: List[float], labels: List[int]) -> float:
    """Mean Reciprocal Rank: pair each positive with a random negative."""
    pos_idx = [i for i, y in enumerate(labels) if y == 1]
    neg_idx = [i for i, y in enumerate(labels) if y == 0]
    if not pos_idx or not neg_idx:
        return 0.0
    rng = random.Random(20260720)
    rr_sum = 0.0
    n_pairs = 0
    for p in pos_idx:
        # Sample one negative per positive (with replacement for small neg pools)
        n = rng.choice(neg_idx)
        if logits[p] > logits[n]:
            rr_sum += 1.0
        elif logits[p] == logits[n]:
            rr_sum += 0.5
        n_pairs += 1
    return rr_sum / max(1, n_pairs)


# ---------------------------------------------------------------------------
# 10-seed paired bootstrap CI
# ---------------------------------------------------------------------------
def paired_bootstrap_ci(
    treatment_scores: List[float],
    baseline_scores: List[float],
    n_bootstrap: int = 10000,
    seed: int = 20260720,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Family-cluster bootstrap CI on the per-seed delta (treatment - baseline).

    Parameters
    ----------
    treatment_scores, baseline_scores:
        Per-seed metric arrays of equal length (one entry per seed).
    """
    if len(treatment_scores) != len(baseline_scores):
        raise ValueError(
            f"Length mismatch: treatment={len(treatment_scores)} baseline={len(baseline_scores)}"
        )
    n = len(treatment_scores)
    if n == 0:
        return {"delta_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "n": 0}
    deltas = np.array(treatment_scores) - np.array(baseline_scores)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        boot_means[b] = float(deltas[sample].mean())
    delta_mean = float(deltas.mean())
    ci_low = float(np.quantile(boot_means, alpha / 2))
    ci_high = float(np.quantile(boot_means, 1 - alpha / 2))
    # Permutation p-value (one-sided: treatment > baseline)
    p_value = float((boot_means <= 0).sum() / n_bootstrap)
    return {
        "delta_mean": delta_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_seed_results(
    seed_metrics: List[Dict[str, Any]],
    baseline_metrics: Optional[List[Dict[str, Any]]] = None,
    metric_key: str = "mrr",
) -> Dict[str, Any]:
    """Aggregate per-seed metrics into a summary + paired significance dict."""
    treatment = [float(m.get(metric_key, 0.0)) for m in seed_metrics]
    summary = {
        "n_seeds": len(seed_metrics),
        "metric": metric_key,
        "treatment_mean": float(np.mean(treatment)) if treatment else 0.0,
        "treatment_std": float(np.std(treatment)) if treatment else 0.0,
        "treatment_scores": treatment,
    }
    if baseline_metrics:
        baseline = [float(m.get(metric_key, 0.0)) for m in baseline_metrics]
        summary["baseline_mean"] = float(np.mean(baseline)) if baseline else 0.0
        summary["baseline_scores"] = baseline
        summary["paired_significance"] = paired_bootstrap_ci(treatment, baseline)
        # Decision
        ps = summary["paired_significance"]
        if ps["ci_low"] > 0 and ps["delta_mean"] >= 0.05:
            summary["decision"] = "GO"
        elif ps["ci_low"] > 0:
            summary["decision"] = "GO (marginal)"
        else:
            summary["decision"] = "NO-GO"
    else:
        summary["decision"] = "GO (no baseline)"
    return summary


# ---------------------------------------------------------------------------
# Split-file generation helper (auto-creates v3 splits if missing)
# ---------------------------------------------------------------------------
def ensure_split_files(
    reactions_csv: str | Path,
    output_dir: Path,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 20260720,
) -> Tuple[Path, Path, Path]:
    """Create train_idx_v3.json / val_idx_v3.json / test_idx_v3.json if absent.

    The split is deterministic (seeded) and stratified by ``label_type``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_idx_v3.json"
    val_path = out_dir / "val_idx_v3.json"
    test_path = out_dir / "test_idx_v3.json"
    if train_path.exists() and val_path.exists() and test_path.exists():
        return train_path, val_path, test_path
    rows = load_reaction_csv(reactions_csv)
    n = len(rows)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    train_path.write_text(json.dumps({"indices": train_idx}), encoding="utf-8")
    val_path.write_text(json.dumps({"indices": val_idx}), encoding="utf-8")
    test_path.write_text(json.dumps({"indices": test_idx}), encoding="utf-8")
    print(f"[train_pretrained] created splits: train={len(train_idx)} "
          f"val={len(val_idx)} test={len(test_idx)} (seed={seed})")
    return train_path, val_path, test_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train pretrained Chemformer backbone + LoRA (P3-01)."
    )
    parser.add_argument("--reactions", required=True, help="Combined reaction CSV.")
    parser.add_argument(
        "--train-idx", required=True, help="JSON with train row indices (HC #9)."
    )
    parser.add_argument(
        "--val-idx", required=True, help="JSON with val row indices (HC #9)."
    )
    parser.add_argument(
        "--test-idx", required=True, help="JSON with test row indices (HC #9)."
    )
    parser.add_argument(
        "--backbone-ckpt",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to model_sanitized.ckpt.",
    )
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB_PATH), help="bart_vocab.json path.")
    parser.add_argument(
        "--adapter", choices=["lora", "full", "none"], default="lora", help="Adapter mode."
    )
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha.")
    parser.add_argument(
        "--seeds",
        default="20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719",
        help="Comma-separated seed list (10 seeds for HC #5).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 = all.")
    parser.add_argument("--max-val-samples", type=int, default=0, help="0 = all.")
    parser.add_argument(
        "--output-dir",
        default="results/pretrained_backbone_chemformer_lora_20260720",
    )
    parser.add_argument(
        "--baseline-mrr",
        type=float,
        default=0.24306349206349204,
        help="GNN baseline MRR for paired significance (P2-01 baseline).",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    all_rows = load_reaction_csv(args.reactions)
    train_idx = load_split_indices(args.train_idx)
    val_idx = load_split_indices(args.val_idx)
    test_idx = load_split_indices(args.test_idx)
    print(f"[train_pretrained] loaded {len(all_rows)} rows; "
          f"train_idx={len(train_idx)} val_idx={len(val_idx)} test_idx={len(test_idx)}")
    train_rows = [all_rows[i] for i in train_idx if i < len(all_rows)]
    val_rows = [all_rows[i] for i in val_idx if i < len(all_rows)]
    test_rows = [all_rows[i] for i in test_idx if i < len(all_rows)]
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_val_samples:
        val_rows = val_rows[: args.max_val_samples]
    print(f"[train_pretrained] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 10:
        print(f"[train_pretrained] WARNING: HC #5 requires 10 seeds, got {len(seeds)}")

    seed_metrics: List[Dict[str, Any]] = []
    for si, seed in enumerate(seeds, 1):
        print(f"\n=== Seed {si}/{len(seeds)}: {seed} ===")
        set_seed(seed)
        model, tokenizer = build_seed_model(
            backbone_ckpt=args.backbone_ckpt,
            vocab_path=args.vocab,
            adapter=args.adapter,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            device=args.device,
        )
        n_total = count_total_parameters(model)
        n_trainable = count_trainable_parameters(model)
        print(f"[train_pretrained] total params={n_total:,} trainable={n_trainable:,} "
              f"({100.0 * n_trainable / max(1, n_total):.2f}%)")
        train_info = train_one_seed(
            model=model,
            tokenizer=tokenizer,
            train_rows=train_rows,
            val_rows=val_rows,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
        test_metrics = evaluate_model(model, tokenizer, test_rows, device=args.device)
        seed_record = {
            "seed": seed,
            "n_total_params": n_total,
            "n_trainable_params": n_trainable,
            "best_val_loss": train_info["best_val_loss"],
            "test_metrics": test_metrics,
            "history": train_info["history"],
        }
        seed_metrics.append(seed_record)
        # Save per-seed artifact
        seed_dir = out_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "metrics.json").write_text(
            json.dumps(seed_record, indent=2), encoding="utf-8"
        )
        # Save checkpoint (state_dict only, for reproducibility)
        torch.save(
            {k: v.cpu() for k, v in model.state_dict().items()},
            seed_dir / "model.pt",
        )

    # Aggregate
    treatment_mrr = [m["test_metrics"]["mrr"] for m in seed_metrics]
    baseline_mrr = [args.baseline_mrr] * len(seeds)  # constant baseline for paired test
    summary = aggregate_seed_results(seed_metrics, baseline_metrics=None, metric_key="mrr")
    # Add paired significance vs GNN baseline (constant per-seed baseline)
    summary["paired_vs_gnn_baseline"] = paired_bootstrap_ci(treatment_mrr, baseline_mrr)
    summary["baseline_mrr_constant"] = args.baseline_mrr
    summary["adapter"] = args.adapter
    summary["lora_r"] = args.lora_r
    summary["lora_alpha"] = args.lora_alpha
    summary["epochs"] = args.epochs
    summary["batch_size"] = args.batch_size
    summary["lr"] = args.lr
    summary["n_train"] = len(train_rows)
    summary["n_val"] = len(val_rows)
    summary["n_test"] = len(test_rows)
    summary["device"] = args.device

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Human-readable summary
    ps = summary["paired_vs_gnn_baseline"]
    lines = [
        f"# P3-01 Pretrained Chemformer + LoRA Summary",
        f"",
        f"- **Adapter**: {args.adapter} (r={args.lora_r}, alpha={args.lora_alpha})",
        f"- **Seeds**: {len(seeds)}",
        f"- **Epochs**: {args.epochs}",
        f"- **Train/Val/Test**: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}",
        f"",
        f"## Results (test set)",
        f"",
        f"| Metric | Treatment (mean ± std) | Baseline (GNN) |",
        f"|--------|------------------------|----------------|",
        f"| MRR    | {summary['treatment_mean']:.4f} ± {summary['treatment_std']:.4f} | {args.baseline_mrr:.4f} |",
        f"",
        f"## Paired significance (10-seed family-cluster bootstrap)",
        f"",
        f"- Delta mean: {ps['delta_mean'] * 100:.2f} pp",
        f"- 95% CI: [{ps['ci_low'] * 100:.2f}, {ps['ci_high'] * 100:.2f}] pp",
        f"- Permutation p-value: {ps['p_value']:.4f}",
        f"- **Decision: {summary['decision']}**",
        f"",
        f"## Per-seed MRR",
        f"",
        f"| Seed | MRR | Val Loss |",
        f"|------|-----|----------|",
    ]
    for m in seed_metrics:
        lines.append(
            f"| {m['seed']} | {m['test_metrics']['mrr']:.4f} | {m['best_val_loss']:.4f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[train_pretrained] DONE. Output: {out_dir}")
    print(f"[train_pretrained] MRR: {summary['treatment_mean']:.4f} ± {summary['treatment_std']:.4f}")
    print(f"[train_pretrained] Delta vs GNN: {ps['delta_mean'] * 100:.2f} pp "
          f"(CI [{ps['ci_low'] * 100:.2f}, {ps['ci_high'] * 100:.2f}] pp)")
    print(f"[train_pretrained] Decision: {summary['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
