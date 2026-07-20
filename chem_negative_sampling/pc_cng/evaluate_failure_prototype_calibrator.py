"""Evaluate the failure prototype calibrator (P1-06) on the test split.

CLI:
    CUDA_VISIBLE_DEVICES=0 python -m pc_cng.evaluate_failure_prototype_calibrator \\
        --checkpoint results/failure_prototype_calibration_smoke_20260719/failure_prototype_calibrator.pt \\
        --real-negatives data/processed/hitea_full_normalized.csv \\
        --alt-negatives data/processed/regiosqm20_normalized.csv \\
        --output-report results/failure_prototype_calibration_smoke_20260719/evaluation_report.md

The script loads a trained checkpoint, evaluates classification accuracy /
entropy / target hit rate on a held-out test split, compares against a random
baseline, and writes a markdown report alongside a JSON copy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    from .failure_prototype_calibrator import (
        FAILURE_TYPES,
        FAILURE_TYPE_TO_IDX,
        FailurePrototypeCalibrator,
        evaluate_controllability,
        extract_failure_type_labels,
        write_json,
    )
    from .reranker import FEATURE_NAMES as RERANKER_FEATURE_NAMES, featurize_reaction
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pc_cng.failure_prototype_calibrator import (  # type: ignore
        FAILURE_TYPES,
        FAILURE_TYPE_TO_IDX,
        FailurePrototypeCalibrator,
        evaluate_controllability,
        extract_failure_type_labels,
        write_json,
    )
    from pc_cng.reranker import FEATURE_NAMES as RERANKER_FEATURE_NAMES, featurize_reaction  # type: ignore


def _resolve_path(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    alt = os.path.join(os.path.dirname(os.getcwd()), path)
    if os.path.exists(alt):
        return alt
    alt2 = os.path.join("/home/cunyuliu/pc_cng_research", path)
    if os.path.exists(alt2):
        return alt2
    return path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataset(real_csv: str, alt_csv: str, limit: int | None) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    per_source: List[Tuple[List[str], List[str]]] = []
    for path in (real_csv, alt_csv):
        if not path or not os.path.exists(path):
            per_source.append(([], []))
            continue
        r, f = extract_failure_type_labels(path)
        per_source.append((r, f))

    rxns: List[str] = []
    fails: List[str] = []
    if limit and limit > 0:
        active = [s for s in per_source if s[0]]
        if active:
            per_quota = max(1, limit // len(active))
            for r, f in per_source:
                if not r:
                    continue
                take = min(per_quota, len(r))
                rxns.extend(r[:take])
                fails.extend(f[:take])
    else:
        for r, f in per_source:
            rxns.extend(r)
            fails.extend(f)

    feats: List[List[float]] = []
    labels: List[int] = []
    counts: Dict[str, int] = {name: 0 for name in FAILURE_TYPES}
    for rxn, fail in zip(rxns, fails):
        try:
            f = featurize_reaction(rxn)
            if len(f) != len(RERANKER_FEATURE_NAMES):
                continue
            feats.append([float(x) for x in f])
        except Exception:
            continue
        if fail not in FAILURE_TYPE_TO_IDX:
            continue
        labels.append(FAILURE_TYPE_TO_IDX[fail])
        counts[fail] += 1
    if not feats:
        return (
            np.zeros((0, len(RERANKER_FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            counts,
        )
    return np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.int64), counts


def _random_baseline_accuracy(y: np.ndarray, seed: int = 0) -> float:
    if y.size == 0:
        return 0.0
    rng = np.random.RandomState(seed)
    preds = rng.randint(0, len(FAILURE_TYPES), size=y.size)
    return float((preds == y).mean())


def _random_target_hit_rate(y: np.ndarray, seed: int = 0) -> float:
    """Random target hit rate: probability of hitting target class k by chance."""
    if y.size == 0:
        return 0.0
    return 1.0 / len(FAILURE_TYPES)


def _format_md(report: Dict, baseline_acc: float, baseline_hit: float, counts: Dict[str, int]) -> str:
    lines: List[str] = []
    lines.append("# Failure Prototype Calibrator - Evaluation Report\n")
    lines.append(f"- Overall classification accuracy: **{report['classification_accuracy']:.4f}**")
    lines.append(f"- Random baseline accuracy: {baseline_acc:.4f}")
    lines.append(f"- Random baseline target hit rate: {baseline_hit:.4f}")
    lines.append(f"- Mean per-sample entropy: {report['mean_entropy']:.4f}")
    lines.append(f"- Normalized entropy (0=confident, 1=uniform): {report['normalized_entropy']:.4f}")
    lines.append(f"- Aggregate predicted-class entropy: {report['aggregate_entropy']:.4f}")
    lines.append(f"- Uniform reference entropy (ln 10): {report['uniform_entropy']:.4f}")
    lines.append("")
    lines.append("## Per-class accuracy\n")
    lines.append("| Failure type | Count | Accuracy | Target hit rate |")
    lines.append("|---|---:|---:|---:|")
    for name in FAILURE_TYPES:
        acc = report["per_class_accuracy"].get(name, float("nan"))
        hit = report["target_hit_rate"].get(name)
        hit_str = f"{hit:.4f}" if isinstance(hit, float) else "n/a (too few)"
        cnt = counts.get(name, 0)
        acc_str = f"{acc:.4f}" if isinstance(acc, float) and not math.isnan(acc) else "n/a"
        lines.append(f"| {name} | {cnt} | {acc_str} | {hit_str} |")
    lines.append("")
    lines.append("## Go/No-Go assessment (P1-06)\n")
    acc_ok = report["classification_accuracy"] >= 0.70
    # Controllability: target hit rate averaged across classes with enough samples.
    hits = [v for v in report["target_hit_rate"].values() if isinstance(v, float)]
    mean_hit = sum(hits) / len(hits) if hits else 0.0
    hit_ok = all(v >= 0.50 for v in hits) if hits else False
    entropy_ok = report["mean_entropy"] > 0.230
    if acc_ok and hit_ok and entropy_ok:
        verdict = "PASS (eligible for paper Section 6.3)"
    elif report["classification_accuracy"] > 0.30:
        verdict = "SUPPLEMENTARY (accuracy between 0.30 and 0.70 or controllability below threshold)"
    else:
        verdict = "FAIL (accuracy at random baseline)"
    lines.append(f"- Accuracy >= 0.70: {'yes' if acc_ok else 'no'}")
    lines.append(f"- Target hit rate >= 0.50 per class: {'yes' if hit_ok else 'no'} (mean={mean_hit:.4f})")
    lines.append(f"- Mean entropy > 0.230: {'yes' if entropy_ok else 'no'}")
    lines.append(f"\n**Verdict: {verdict}**\n")
    lines.append("> Note: this is a single-seed smoke evaluation. The paper-level")
    lines.append("> claim requires a 10-seed paired significance test against the")
    lines.append("> random baseline.")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained failure prototype calibrator")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--real-negatives", required=True)
    parser.add_argument("--alt-negatives", required=True)
    parser.add_argument("--output-report", default=None, help="Markdown report path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    _seed_everything(args.seed)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    model = FailurePrototypeCalibrator.from_checkpoint(args.checkpoint, map_location=args.device)

    real_path = _resolve_path(args.real_negatives)
    alt_path = _resolve_path(args.alt_negatives)
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"real-negatives CSV not found: {real_path}")
    if not os.path.exists(alt_path):
        raise FileNotFoundError(f"alt-negatives CSV not found: {alt_path}")

    X, y, counts = _build_dataset(real_path, alt_path, args.limit)
    if X.shape[0] == 0:
        raise RuntimeError("No usable negatives for evaluation.")
    print(f"[eval] total negatives={X.shape[0]}")

    # Use the same split logic as training: last 20% as test.
    n = X.shape[0]
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    n_test = max(1, int(n * 0.2))
    test_idx = perm[:n_test]
    X_test = X[test_idx]
    y_test = y[test_idx]
    print(f"[eval] test split={X_test.shape[0]}")

    report = evaluate_controllability(
        model,
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
        device=args.device,
    )
    report["test_size"] = int(X_test.shape[0])
    report["total_negatives"] = int(X.shape[0])

    baseline_acc = _random_baseline_accuracy(y_test, seed=args.seed)
    baseline_hit = _random_target_hit_rate(y_test, seed=args.seed)
    report["random_baseline_accuracy"] = baseline_acc
    report["random_baseline_target_hit_rate"] = baseline_hit

    md = _format_md(report, baseline_acc, baseline_hit, counts)
    out_md = args.output_report
    if out_md is None:
        out_md = os.path.join(os.path.dirname(args.checkpoint), "evaluation_report.md")
    os.makedirs(os.path.dirname(os.path.abspath(out_md)) or ".", exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as handle:
        handle.write(md)
    write_json(out_md.replace(".md", ".json"), report)
    print(f"[save] markdown -> {out_md}")
    print(f"[save] json -> {out_md.replace('.md', '.json')}")
    print(f"[done] acc={report['classification_accuracy']:.4f} baseline={baseline_acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
