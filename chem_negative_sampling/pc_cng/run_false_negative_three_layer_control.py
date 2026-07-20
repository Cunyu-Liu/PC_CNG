"""CLI runner for the three-layer false-negative control (P1-08).

Usage::

    python3 -m pc_cng.run_false_negative_three_layer_control \
        --ensemble-models results/type1_unreacted_substrate_supplement_v2_20260711 \
        --database data/processed/uspto_openmolecules_normalized.csv \
        --expert-review results/expert_review_20260719 \
        --output-dir results/false_negative_three_layer_20260719

The runner also materialises the double-blind review sample
(``sampled_for_review.csv``) inside the ``--expert-review`` directory so the
protocol's sampling step is reproducible from a single command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from .false_negative_three_layer_control import (
    DEFAULT_CONTROL_SIZE,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    REVIEWED_FIELDS,
    SAMPLED_FOR_REVIEW_FIELDS,
    read_csv,
    run_three_layer_control,
    stratified_sample_for_review,
    write_csv,
)


DEFAULT_INPUT_PATHS = [
    "results/hitea_full_generation/pc_cng_synthetic_negatives_reviewed.csv",
    "results/regiosqm20_full/pc_cng_synthetic_negatives_reviewed.csv",
]


def _load_reviewed_negatives(input_paths: List[str], root: str) -> List[dict]:
    rows: List[dict] = []
    for rel in input_paths:
        path = rel if os.path.isabs(rel) else os.path.join(root, rel)
        if not os.path.isfile(path):
            print(f"[warn] reviewed-negatives file not found, skipping: {path}", file=sys.stderr)
            continue
        file_rows, _ = read_csv(path)
        rows.extend(file_rows)
        print(f"[info] loaded {len(file_rows):>8d} rows from {path}")
    return rows


def _load_real_negative_controls(database_csv: str, n: int, seed: int) -> List[dict]:
    """Sample ``n`` real positive reactions from the database to serve as
    double-blind controls (true_label=1 => real reaction).  These let us
    measure whether reviewers can recognise genuinely-occurring reactions
    when blinded against the sample origin.
    """
    if not database_csv or not os.path.isfile(database_csv):
        return []
    rows, _ = read_csv(database_csv)
    positives = [r for r in rows if r.get("label_type", "") == "positive"]
    import random
    rng = random.Random(seed)
    if len(positives) > n:
        positives = rng.sample(positives, n)
    return positives


def _generate_review_sample(
    rows: List[dict],
    expert_review_dir: str,
    database_csv: str,
    sample_size: int,
    control_size: int,
    seed: int,
) -> str:
    os.makedirs(expert_review_dir, exist_ok=True)
    controls = _load_real_negative_controls(database_csv, control_size, seed)
    sampled = stratified_sample_for_review(
        rows, n_samples=sample_size, seed=seed,
        control_rows=controls, n_controls=control_size,
    )
    out_path = os.path.join(expert_review_dir, "sampled_for_review.csv")
    write_csv(out_path, sampled, SAMPLED_FOR_REVIEW_FIELDS)
    print(f"[info] wrote {len(sampled)} sampled-for-review rows to {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=",".join(DEFAULT_INPUT_PATHS),
        help="Comma-separated reviewed-negatives CSV paths (default: "
             "hitea_full_generation + regiosqm20_full)",
    )
    parser.add_argument(
        "--ensemble-models", required=True,
        help="Ensemble models dir with seed subdirs containing test_predictions.csv",
    )
    parser.add_argument(
        "--database", required=True,
        help="USPTO normalized CSV (positives database for Layer 2 retrieval)",
    )
    parser.add_argument(
        "--expert-review", default="results/expert_review_20260719",
        help="Expert review dir (output: sampled_for_review.csv; input: "
             "reviewer_ratings_raw.csv if present)",
    )
    parser.add_argument(
        "--output-dir", default="results/false_negative_three_layer_20260719",
        help="Output dir for high_confidence_negatives.csv + three_layer_summary.json",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--control-size", type=int, default=DEFAULT_CONTROL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ensemble-std-threshold", type=float, default=0.15)
    parser.add_argument("--tanimoto-threshold", type=float, default=0.95)
    parser.add_argument("--tanimoto-sample-size", type=int, default=10000)
    parser.add_argument(
        "--root", default=os.getcwd(),
        help="Repo root for resolving default relative input paths",
    )
    parser.add_argument(
        "--skip-review-sample", action="store_true",
        help="Do not (re)generate sampled_for_review.csv (use existing)",
    )
    args = parser.parse_args()

    input_paths = [p.strip() for p in args.input.split(",") if p.strip()]
    rows = _load_reviewed_negatives(input_paths, args.root)
    if not rows:
        print("[error] no reviewed negatives loaded; aborting", file=sys.stderr)
        return 2
    print(f"[info] total reviewed negatives loaded: {len(rows)}")

    if not args.skip_review_sample:
        _generate_review_sample(
            rows, args.expert_review, args.database,
            args.sample_size, args.control_size, args.seed,
        )

    high_confidence, summary = run_three_layer_control(
        rows,
        ensemble_dir=args.ensemble_models,
        database_csv=args.database,
        expert_review_dir=args.expert_review,
        ensemble_std_threshold=args.ensemble_std_threshold,
        tanimoto_threshold=args.tanimoto_threshold,
        tanimoto_sample_size=args.tanimoto_sample_size,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "high_confidence_negatives.csv")
    out_fields = REVIEWED_FIELDS + [
        "layer1_ensemble_std", "layer1_verdict",
        "layer2_verdict", "layer2_hit_reason",
        "layer3_verdict", "layer3_source",
    ]
    write_csv(out_csv, high_confidence, out_fields)
    print(f"[info] wrote {len(high_confidence)} high-confidence negatives to {out_csv}")

    summary_path = os.path.join(args.output_dir, "three_layer_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"[info] wrote summary to {summary_path}")

    rate = summary.get("high_confidence_rate", 0.0)
    verdict = summary.get("go_no_go_verdict", "UNKNOWN")
    print(f"[result] high-confidence rate: {rate:.4f} ({len(high_confidence)}/{len(rows)})")
    print(f"[result] Go/No-Go verdict: {verdict}")
    return 0 if verdict == "GO" else 0  # No-Go is not a CLI failure (still reports)


if __name__ == "__main__":
    raise SystemExit(main())
