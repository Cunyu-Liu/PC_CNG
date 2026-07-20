"""P1-03 (part 1): Calibration error metrics for the 10-seed v2 reranker.

Reports Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and
Brier score for the existing ``type1_unreacted_substrate_supplement_v2`` 10-seed
models.  Each seed directory contains a ``test_predictions.csv`` with the
sigmoid probability (``score``) and the ground-truth ``label``; this script
aggregates per-seed metrics and reports the 10-seed mean with a bootstrap 95%
CI.

The calibration metrics are computed on the reranker's softmax/sigmoid
probability against the binary feasibility label, which is exactly the E4/E6
gap called out in Section 9 of the PC-CNG research notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .paired_reranking_significance import bootstrap_ci, mean, percentile


def load_predictions(path: str) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """Load a ``test_predictions.csv`` produced by train_*_mlp.py.

    Returns ``(scores, labels, raw_rows)`` where ``scores`` are float
    probabilities in [0, 1] and ``labels`` are 0/1 ints.
    """
    scores: List[float] = []
    labels: List[int] = []
    rows: List[Dict[str, object]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "score" not in (reader.fieldnames or []) or "label" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain 'score' and 'label' columns")
        for row in reader:
            try:
                score = float(row["score"])
                label = int(float(row["label"]))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            scores.append(score)
            labels.append(label)
            rows.append(dict(row))
    return np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int64), rows


def _bin_mask(scores: np.ndarray, bin_index: int, n_bins: int) -> np.ndarray:
    """Return boolean mask for ``scores`` falling into ``bin_index``.

    The last bin is inclusive on both ends so that ``score == 1.0`` is counted.
    """
    low = bin_index / n_bins
    high = (bin_index + 1) / n_bins
    if bin_index == n_bins - 1:
        return (scores >= low) & (scores <= high)
    return (scores >= low) & (scores < high)


def compute_ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error.

    ECE = sum_b (n_b / N) * |acc_b - conf_b|, where ``acc_b`` is the empirical
    accuracy in bin ``b`` and ``conf_b`` is the mean predicted probability.
    """
    if len(scores) == 0:
        return 0.0
    n = len(scores)
    ece = 0.0
    for b in range(n_bins):
        mask = _bin_mask(scores, b, n_bins)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(scores[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += (count / n) * abs(accuracy - confidence)
    return float(ece)


def compute_mce(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Maximum Calibration Error: max_b |acc_b - conf_b| over non-empty bins."""
    if len(scores) == 0:
        return 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = _bin_mask(scores, b, n_bins)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(scores[mask].mean())
        accuracy = float(labels[mask].mean())
        mce = max(mce, abs(accuracy - confidence))
    return float(mce)


def compute_brier_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """Brier score: mean((score - label)^2)."""
    if len(scores) == 0:
        return 0.0
    return float(np.mean((scores - labels.astype(np.float64)) ** 2))


def compute_all_calibration(
    scores: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Dict[str, float | int]:
    """Compute ECE, MCE, Brier, plus per-bin diagnostics."""
    ece = compute_ece(scores, labels, n_bins)
    mce = compute_mce(scores, labels, n_bins)
    brier = compute_brier_score(scores, labels)
    bins: List[Dict[str, float | int]] = []
    for b in range(n_bins):
        mask = _bin_mask(scores, b, n_bins)
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin": b, "low": b / n_bins, "high": (b + 1) / n_bins, "count": 0})
            continue
        bins.append(
            {
                "bin": b,
                "low": b / n_bins,
                "high": (b + 1) / n_bins,
                "count": count,
                "confidence": float(scores[mask].mean()),
                "accuracy": float(labels[mask].mean()),
                "gap": float(abs(float(labels[mask].mean()) - float(scores[mask].mean()))),
            }
        )
    return {
        "n": int(len(scores)),
        "n_bins": n_bins,
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "mean_score": float(scores.mean()) if len(scores) else 0.0,
        "positive_rate": float(labels.mean()) if len(labels) else 0.0,
        "bins": bins,
    }


def discover_seed_dirs(model_dir: str, seeds: Sequence[int]) -> Dict[int, str]:
    """Locate per-seed prediction directories under ``model_dir``.

    Looks for the ``unreacted_augmented_pairwise_seed{seed}`` pattern used by
    the v2 multiseed run; if not found, falls back to ``seed{seed}``.
    """
    out: Dict[int, str] = {}
    for seed in seeds:
        candidates = [
            os.path.join(model_dir, f"unreacted_augmented_pairwise_seed{seed}"),
            os.path.join(model_dir, f"seed{seed}"),
            os.path.join(model_dir, str(seed)),
        ]
        for path in candidates:
            if os.path.isdir(path):
                out[seed] = path
                break
    return out


def evaluate_seed(model_dir: str, seed: int, n_bins: int) -> Dict[str, object]:
    """Compute calibration metrics for a single seed's test predictions."""
    pred_path = os.path.join(model_dir, "test_predictions.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"test_predictions.csv not found under {model_dir}")
    scores, labels, _ = load_predictions(pred_path)
    metrics = compute_all_calibration(scores, labels, n_bins)
    metrics["seed"] = seed
    metrics["model_dir"] = model_dir
    metrics["pred_path"] = pred_path
    return metrics


def aggregate_seeds(per_seed: List[Dict[str, object]], bootstrap_iterations: int, seed: int) -> Dict[str, object]:
    """Aggregate per-seed metrics into 10-seed mean with bootstrap 95% CI."""
    def values(key: str) -> List[float]:
        return [float(rec[key]) for rec in per_seed if key in rec]

    out: Dict[str, object] = {"n_seeds": len(per_seed)}
    for key in ["ece", "mce", "brier"]:
        v = values(key)
        if not v:
            out[key] = {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "per_seed": []}
            continue
        ci_low, ci_high = bootstrap_ci(v, bootstrap_iterations, seed)
        out[key] = {
            "mean": mean(v),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "per_seed": v,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="P1-03 calibration error evaluation")
    parser.add_argument("--model-dir", required=True, help="Root containing per-seed subdirs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seeds",
        default="20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719",
    )
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    seed_dirs = discover_seed_dirs(args.model_dir, seeds)
    if not seed_dirs:
        raise FileNotFoundError(
            f"No per-seed directories found under {args.model_dir} for seeds {seeds}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    per_seed: List[Dict[str, object]] = []
    for seed in seeds:
        if seed not in seed_dirs:
            print(f"[calibration] seed={seed} directory not found, skipping")
            continue
        metrics = evaluate_seed(seed_dirs[seed], seed, args.n_bins)
        per_seed.append(metrics)
        print(
            f"[calibration] seed={seed} ECE={metrics['ece']:.4f} "
            f"MCE={metrics['mce']:.4f} Brier={metrics['brier']:.4f} "
            f"n={metrics['n']}"
        )

    aggregate = aggregate_seeds(per_seed, args.bootstrap_iterations, 20260719)

    payload = {
        "task": "calibration_error_10seed",
        "model_dir": args.model_dir,
        "seeds": [int(s) for s in seeds],
        "seeds_evaluated": [int(rec["seed"]) for rec in per_seed],
        "n_bins": args.n_bins,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "metric_definitions": {
            "ece": "Expected Calibration Error: sum_b (n_b/N) * |acc_b - conf_b|",
            "mce": "Maximum Calibration Error: max_b |acc_b - conf_b|",
            "brier": "Brier score: mean((score - label)^2)",
        },
    }
    with open(os.path.join(args.output_dir, "calibration_error_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    # Also write a flat CSV for quick inspection.
    with open(os.path.join(args.output_dir, "per_seed_calibration.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "n", "ece", "mce", "brier", "mean_score", "positive_rate"])
        writer.writeheader()
        for rec in per_seed:
            writer.writerow(
                {
                    "seed": rec["seed"],
                    "n": rec["n"],
                    "ece": f"{rec['ece']:.8f}",
                    "mce": f"{rec['mce']:.8f}",
                    "brier": f"{rec['brier']:.8f}",
                    "mean_score": f"{rec['mean_score']:.8f}",
                    "positive_rate": f"{rec['positive_rate']:.8f}",
                }
            )
    print(
        f"[calibration] aggregate ECE={aggregate['ece']['mean']:.4f} "
        f"MCE={aggregate['mce']['mean']:.4f} Brier={aggregate['brier']['mean']:.4f}"
    )
    print(json.dumps({"summary_path": os.path.join(args.output_dir, "calibration_error_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
