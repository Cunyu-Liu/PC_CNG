"""Train the failure prototype calibrator (P1-06).

CLI:
    CUDA_VISIBLE_DEVICES=0 python -m pc_cng.train_failure_prototype_calibrator \\
        --real-negatives data/processed/hitea_full_normalized.csv \\
        --alt-negatives data/processed/regiosqm20_normalized.csv \\
        --output-dir results/failure_prototype_calibration_20260719 \\
        --epochs 50 --batch-size 64 --seed 20260719

Outputs:
    * ``failure_prototype_calibrator.pt`` - model checkpoint
    * ``train_summary.json`` - training history + final metrics
    * ``failure_type_distribution.json`` - per-class sample count + accuracy
    * ``controllability_report.json`` - entropy + target_hit_rate per class
"""

from __future__ import annotations

import argparse
import json
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
        train_calibrator,
        write_json,
    )
    from .reranker import FEATURE_NAMES as RERANKER_FEATURE_NAMES, featurize_reaction
except ImportError:  # allow direct invocation as a script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pc_cng.failure_prototype_calibrator import (  # type: ignore
        FAILURE_TYPES,
        FAILURE_TYPE_TO_IDX,
        FailurePrototypeCalibrator,
        evaluate_controllability,
        extract_failure_type_labels,
        train_calibrator,
        write_json,
    )
    from pc_cng.reranker import FEATURE_NAMES as RERANKER_FEATURE_NAMES, featurize_reaction  # type: ignore


def _resolve_path(path: str) -> str:
    """Resolve a data path relative to cwd, falling back to a sibling
    ``../data/processed`` layout used in this repository."""
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
    return path  # let the caller raise a clear error


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _featurize_batch(reactions: List[str]) -> Tuple[List[List[float]], List[int]]:
    feats: List[List[float]] = []
    bad_idx: List[int] = []
    for i, rxn in enumerate(reactions):
        try:
            f = featurize_reaction(rxn)
            if len(f) != len(RERANKER_FEATURE_NAMES):
                bad_idx.append(i)
                continue
            feats.append([float(x) for x in f])
        except Exception:
            bad_idx.append(i)
    return feats, bad_idx


def _build_dataset(
    real_negatives_csv: str,
    alt_negatives_csv: str,
    limit: int | None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    rxn_list: List[str] = []
    label_list: List[str] = []

    per_source: List[Tuple[List[str], List[str]]] = []
    for path in (real_negatives_csv, alt_negatives_csv):
        if not path or not os.path.exists(path):
            per_source.append(([], []))
            continue
        rxns, fails = extract_failure_type_labels(path)
        per_source.append((rxns, fails))

    # When a limit is set, sample evenly across sources so that the smoke test
    # includes the alt-outcome (wrong_anchor) negatives rather than only the
    # first (HITEA) source.
    if limit is not None and limit > 0:
        active = [s for s in per_source if s[0]]
        if active:
            per_quota = max(1, limit // len(active))
            for rxns, fails in per_source:
                if not rxns:
                    continue
                take = min(per_quota, len(rxns))
                rxn_list.extend(rxns[:take])
                label_list.extend(fails[:take])
    else:
        for rxns, fails in per_source:
            rxn_list.extend(rxns)
            label_list.extend(fails)

    feats, bad_idx = _featurize_batch(rxn_list)
    bad_set = set(bad_idx)
    kept_feats: List[List[float]] = []
    kept_labels: List[int] = []
    counts: Dict[str, int] = {name: 0 for name in FAILURE_TYPES}
    for i, (f, lname) in enumerate(zip(feats, label_list)):
        if i in bad_set:
            continue
        if lname not in FAILURE_TYPE_TO_IDX:
            continue
        kept_feats.append(f)
        kept_labels.append(FAILURE_TYPE_TO_IDX[lname])
        counts[lname] += 1

    if not kept_feats:
        return (
            np.zeros((0, len(RERANKER_FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            counts,
        )
    X = np.asarray(kept_feats, dtype=np.float32)
    y = np.asarray(kept_labels, dtype=np.int64)
    return X, y, counts


def _split(
    X: np.ndarray, y: np.ndarray, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train / val / test. Test = 20%, val = ``val_ratio`` of rest."""
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    if n <= 2:
        # Too small to split meaningfully: keep all in train.
        return X[perm], y[perm], X[:0], y[:0], X[:0], y[:0]
    n_test = max(1, int(n * 0.2))
    n_val = max(1, int((n - n_test) * val_ratio))
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    if train_idx.size == 0:
        train_idx = perm
        val_idx = perm[:0]
        test_idx = perm[:0]
    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--real-negatives", required=True, help="HITEA normalized CSV path")
    parser.add_argument("--alt-negatives", required=True, help="RegioSQM20 normalized CSV path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--triplet-weight", type=float, default=0.3)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None, help="Cap number of negatives (smoke test)")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    _seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    real_path = _resolve_path(args.real_negatives)
    alt_path = _resolve_path(args.alt_negatives)
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"real-negatives CSV not found: {real_path}")
    if not os.path.exists(alt_path):
        raise FileNotFoundError(f"alt-negatives CSV not found: {alt_path}")

    print(f"[load] real_negatives={real_path}")
    print(f"[load] alt_negatives={alt_path}")

    X, y, counts = _build_dataset(real_path, alt_path, args.limit)
    print(f"[load] total negatives={X.shape[0]} feature_dim={X.shape[1] if X.ndim == 2 else 0}")
    print(f"[load] failure-type distribution: {json.dumps(counts)}")

    if X.shape[0] == 0:
        raise RuntimeError("No usable negatives after featurization.")

    X_train, y_train, X_val, y_val, X_test, y_test = _split(
        X, y, val_ratio=args.val_ratio, seed=args.seed
    )
    print(
        f"[split] train={X_train.shape[0]} val={X_val.shape[0]} test={X_test.shape[0]}"
    )

    train_features = torch.from_numpy(X_train)
    train_labels = torch.from_numpy(y_train)
    val_features = torch.from_numpy(X_val) if X_val.size else None
    val_labels = torch.from_numpy(y_val) if y_val.size else None
    test_features = torch.from_numpy(X_test) if X_test.size else None
    test_labels = torch.from_numpy(y_test) if y_test.size else None

    model = FailurePrototypeCalibrator(
        input_dim=X.shape[1],
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_failure_types=len(FAILURE_TYPES),
        temperature=args.temperature,
        triplet_weight=args.triplet_weight,
    )

    history = train_calibrator(
        model=model,
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        triplet_margin=args.triplet_margin,
        verbose=not args.quiet,
    )

    checkpoint_path = os.path.join(args.output_dir, "failure_prototype_calibrator.pt")
    torch.save(model.to_checkpoint(), checkpoint_path)
    print(f"[save] checkpoint -> {checkpoint_path}")

    # Evaluation on the test split (fall back to val when test empty).
    eval_features = test_features if test_features is not None and test_features.size(0) > 0 else val_features
    eval_labels = test_labels if test_labels is not None and test_labels.size(0) > 0 else val_labels
    if eval_features is None or eval_features.size(0) == 0:
        eval_features = train_features
        eval_labels = train_labels

    report = evaluate_controllability(
        model,
        eval_features,
        eval_labels,
        device=args.device,
    )

    # Per-class sample count + accuracy.
    per_class_summary: Dict[str, Dict[str, object]] = {}
    label_tensor = eval_labels
    class_counts = torch.bincount(label_tensor, minlength=len(FAILURE_TYPES)).tolist()
    for k, name in enumerate(FAILURE_TYPES):
        per_class_summary[name] = {
            "count": int(class_counts[k]) if k < len(class_counts) else 0,
            "accuracy": report["per_class_accuracy"].get(name, float("nan")),
            "target_hit_rate": report["target_hit_rate"].get(name),
        }

    summary = {
        "args": vars(args),
        "feature_names": list(RERANKER_FEATURE_NAMES),
        "failure_types": list(FAILURE_TYPES),
        "total_negatives": int(X.shape[0]),
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "test_size": int(X_test.shape[0]),
        "failure_type_distribution": counts,
        "history": history,
        "final_metrics": {
            "classification_accuracy": report["classification_accuracy"],
            "mean_entropy": report["mean_entropy"],
            "normalized_entropy": report["normalized_entropy"],
            "aggregate_entropy": report["aggregate_entropy"],
            "uniform_entropy": report["uniform_entropy"],
            "best_val_acc": history.get("best_val_acc"),
            "best_epoch": history.get("best_epoch"),
        },
    }

    write_json(os.path.join(args.output_dir, "train_summary.json"), summary)
    write_json(
        os.path.join(args.output_dir, "failure_type_distribution.json"),
        {"per_class": per_class_summary, "total_counts": counts},
    )
    write_json(os.path.join(args.output_dir, "controllability_report.json"), report)

    print(
        f"[done] acc={report['classification_accuracy']:.4f} "
        f"mean_entropy={report['mean_entropy']:.4f} "
        f"uniform={report['uniform_entropy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
