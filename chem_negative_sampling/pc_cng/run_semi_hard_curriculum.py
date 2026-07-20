"""CLI runner for the semi-hard curriculum experiment (P1-07).

Runs a 4-round curriculum (warm-started) and a matched one-shot baseline,
then produces ``comparison.json`` via :mod:`pc_cng.evaluate_semi_hard_curriculum`.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=0 python -m pc_cng.run_semi_hard_curriculum \\
        --real-csv data/processed/regiosqm20_normalized.csv \\
        --synthetic-csv results/v2_boundary_generation/regiosqm20_boundary_negatives_reviewed.csv \\
        --output-dir results/semi_hard_curriculum_4round_20260719 \\
        --rounds "[0.10,0.35]" "[0.25,0.55]" "[0.40,0.70]" "[0.50,0.80]" \\
        --epochs-per-round 10 --overlap 0.2 \\
        --pairwise-weight 1.0 --margin 0.5 \\
        --feature-mode morgan --seed 20260719

For smoke tests where the actual feasibility distribution does not span the
suggested windows, use ``--quantile-rounds 4`` instead of ``--rounds`` so each
round gets an equal-size slice of the data.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from .evaluate_semi_hard_curriculum import (
    compare_curriculum_vs_one_shot,
    write_markdown_report,
)
from .semi_hard_curriculum import SemiHardCurriculum


def parse_rounds(round_args: Optional[List[str]]) -> Optional[List[Tuple[float, float]]]:
    if not round_args:
        return None
    parsed: List[Tuple[float, float]] = []
    for item in round_args:
        # Accept either "[0.10,0.35]" or "0.10,0.35" or "0.10 0.35"
        text = item.strip()
        try:
            obj = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            obj = [p.strip() for p in text.replace(" ", ",").split(",") if p.strip()]
        if not isinstance(obj, (list, tuple)) or len(obj) != 2:
            raise ValueError(f"Invalid round spec {item!r}; expected [low, high]")
        parsed.append((float(obj[0]), float(obj[1])))
    return parsed


def build_base_train_args(args: argparse.Namespace) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "epochs": args.epochs_per_round,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "dropout": args.dropout,
        "feature_mode": args.feature_mode,
        "n_bits": args.n_bits,
        "fp_mode": args.fp_mode,
        "pairwise_weight": args.pairwise_weight,
        "bce_weight": args.bce_weight,
        "margin": args.margin,
        "seed": args.seed,
        "checkpoint_metric": args.checkpoint_metric,
        "checkpoint_group_by": args.checkpoint_group_by,
    }
    if args.include_descriptors:
        base["include_descriptors"] = True
    if args.lr_scheduler != "none":
        base["lr_scheduler"] = args.lr_scheduler
        base["lr_min"] = args.lr_min
        base["warmup_epochs"] = args.warmup_epochs
    if args.family_margin:
        base["family_margin"] = list(args.family_margin)
    if args.family_weight:
        base["family_weight"] = list(args.family_weight)
    if args.class_margin:
        base["class_margin"] = list(args.class_margin)
    if args.class_weight:
        base["class_weight"] = list(args.class_weight)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real-csv", required=True, help="Real positives/negatives CSV with train/val/test splits.")
    parser.add_argument("--synthetic-csv", required=True, action="append",
                        help="Boundary negatives CSV with feasibility/hard_score column. Can be repeated.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", nargs="+", default=None,
                        help="Curriculum feasibility ranges, e.g. \"[0.10,0.35]\" \"[0.25,0.55]\" ...")
    parser.add_argument("--quantile-rounds", type=int, default=0,
                        help="If >0 and --rounds is empty, split feasibility distribution into N quantile rounds.")
    parser.add_argument("--epochs-per-round", type=int, default=10)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--min-round-size", type=int, default=8)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--feature-mode", choices=["morgan", "graph_stats", "combined"], default="morgan")
    parser.add_argument("--n-bits", type=int, default=4096)
    parser.add_argument("--fp-mode", choices=["binary", "count", "binary_count"], default="binary")
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--include-descriptors", action="store_true")
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--family-margin", action="append", default=[])
    parser.add_argument("--family-weight", action="append", default=[])
    parser.add_argument("--class-margin", action="append", default=[])
    parser.add_argument("--class-weight", action="append", default=[])
    parser.add_argument("--checkpoint-metric",
                        choices=["val_roc_auc", "val_auprc", "val_f1", "val_top1",
                                 "val_top3", "val_mrr", "val_ndcg"],
                        default="val_roc_auc")
    parser.add_argument("--checkpoint-group-by", default="reactants")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip one-shot baseline (use when only curriculum is needed).")
    parser.add_argument("--skip-curriculum", action="store_true",
                        help="Skip curriculum training (use when only baseline is needed).")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rounds_spec = parse_rounds(args.rounds)

    curriculum = SemiHardCurriculum(
        rounds=rounds_spec,
        epochs_per_round=args.epochs_per_round,
        overlap=args.overlap,
        quantile_rounds=args.quantile_rounds,
        min_round_size=args.min_round_size,
        seed=args.seed,
    )
    base_train_args = build_base_train_args(args)

    # We treat multiple --synthetic-csv as additive (train_pairwise_reward_mlp
    # also accepts multiple). For curriculum, we run rounds on the first CSV
    # (the boundary negatives). Additional CSVs are appended to every round so
    # extra negatives are shared across rounds (e.g. type-1 supplement).
    primary_synthetic = args.synthetic_csv[0]
    # For one-shot, we pass all CSVs; for curriculum rounds, the controller
    # writes its own per-round CSV from the primary file. If extra CSVs are
    # supplied, they are passed through to every round via the base_train_args
    # (handled by appending to the subprocess command). To keep this simple
    # and match the spec, we restrict the curriculum to the primary CSV and
    # warn if extra CSVs are provided.
    if len(args.synthetic_csv) > 1:
        sys.stderr.write(
            "[run_semi_hard_curriculum] WARNING: multiple --synthetic-csv given; "
            "only the first is used for curriculum round selection. "
            "Extras are ignored in this version.\n"
        )

    curr_summary: Optional[Dict[str, Any]] = None
    one_shot_summary: Optional[Dict[str, Any]] = None
    curr_dir = os.path.join(args.output_dir, "curriculum")
    one_shot_dir = os.path.join(args.output_dir, "one_shot")

    if not args.skip_curriculum:
        print(f"[run_semi_hard_curriculum] running curriculum in {curr_dir}")
        curr_summary = curriculum.run_curriculum(
            real_csv=args.real_csv,
            synthetic_csv=primary_synthetic,
            output_dir=curr_dir,
            base_train_args=base_train_args,
        )
        print(json.dumps({
            "curriculum_final_test_top1": curr_summary["final_test_top1"],
            "curriculum_total_epochs": curr_summary["total_epochs"],
            "curriculum_num_rounds": curr_summary["num_rounds"],
        }, indent=2))

    if not args.skip_baseline:
        print(f"[run_semi_hard_curriculum] running one-shot baseline in {one_shot_dir}")
        total_epochs = args.epochs_per_round * (len(rounds_spec) if rounds_spec else max(curriculum.quantile_rounds, 4))
        one_shot_summary = curriculum.run_one_shot_baseline(
            real_csv=args.real_csv,
            synthetic_csv=primary_synthetic,
            output_dir=one_shot_dir,
            base_train_args=base_train_args,
            total_epochs=total_epochs,
        )
        print(json.dumps({
            "one_shot_final_test_top1": one_shot_summary["final_test_top1"],
            "one_shot_total_epochs": one_shot_summary["total_epochs"],
        }, indent=2))

    if curr_summary and one_shot_summary:
        print("[run_semi_hard_curriculum] building comparison.json + markdown report")
        comparison = compare_curriculum_vs_one_shot(
            curriculum_dir=curr_dir,
            one_shot_dir=one_shot_dir,
            output_dir=args.output_dir,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        write_markdown_report(
            curriculum_summary=curr_summary,
            one_shot_summary=one_shot_summary,
            comparison=comparison,
            output_path=os.path.join(args.output_dir, "comparison_report.md"),
        )
        go_nogo = comparison.get("go_nogo_decision", "unknown")
        print(f"[run_semi_hard_curriculum] Go/No-Go decision: {go_nogo}")
        print(f"[run_semi_hard_curriculum] done. See {args.output_dir}")


if __name__ == "__main__":
    main()
