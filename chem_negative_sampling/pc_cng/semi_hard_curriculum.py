"""Semi-hard curriculum controller for boundary negative training.

Implements P1-07 of the PC-CNG research roadmap: a 4-round curriculum that
progressively introduces harder boundary negatives (by feasibility / hard_score)
rather than the one-shot "all negatives at once" baseline. Used to test the H3
hypothesis in paper Section 6.6: "Semi-hard curriculum vs top-k hardest in
boundary negative training".

The training itself is delegated to ``train_pairwise_reward_mlp`` (subprocess)
so the curriculum controller only needs to:
  1. Read boundary negatives and their feasibility scores.
  2. Partition them into rounds (fixed feasibility ranges or quantile-based).
  3. For each round, write a filtered synthetic negatives CSV, then call
     ``train_pairwise_reward_mlp`` with ``--init-checkpoint`` pointing to the
     previous round's best checkpoint (warm-start).
  4. Run the one-shot baseline (all negatives, same total epochs) for paired
     comparison.

Feasibility score source
------------------------
Boundary negatives produced by ``reaction_boundary_generator`` carry a
``hard_score`` column (from ``validator.CounterfactualValidator.score``), which
is the semi-hard target score in [0, 1]. We use ``hard_score`` as the
feasibility score by default; the column name ``feasibility`` is also accepted
if present. If neither column exists we fall back to a uniform random score
(smoke-test only; a warning is emitted).
"""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

FEASIBILITY_COLUMNS: Tuple[str, ...] = ("feasibility", "hard_score")


@dataclass
class CurriculumRound:
    """Configuration + observed outcome for one curriculum round."""

    round_idx: int
    feasibility_range: Tuple[float, float]
    num_negatives: int
    epochs: int
    output_dir: str = ""
    best_checkpoint: str = ""
    best_metric_value: float = float("nan")
    final_test_top1: float = float("nan")
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_idx": self.round_idx,
            "feasibility_range": list(self.feasibility_range),
            "num_negatives": self.num_negatives,
            "epochs": self.epochs,
            "output_dir": self.output_dir,
            "best_checkpoint": self.best_checkpoint,
            "best_metric_value": self.best_metric_value,
            "final_test_top1": self.final_test_top1,
            "history": self.history,
        }


def _parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def load_negatives_with_feasibility(
    synthetic_csv: str,
    feasibility_col: Optional[str] = None,
    random_fallback_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Read a synthetic boundary negatives CSV and attach a ``feasibility`` key.

    The returned dicts preserve all original columns. A normalized
    ``feasibility`` float field is added (in [0, 1]). If no feasibility column
    is found and ``random_fallback_seed`` is provided, a uniform random value
    is used (smoke-test path). If no column and no fallback, raises
    ``ValueError``.
    """
    rows: List[Dict[str, Any]] = []
    col_name: Optional[str] = feasibility_col
    if col_name is None:
        with open(synthetic_csv, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for candidate in FEASIBILITY_COLUMNS:
                if candidate in (reader.fieldnames or []):
                    col_name = candidate
                    break

    rng = random.Random(random_fallback_seed) if random_fallback_seed is not None else None
    used_fallback = False
    with open(synthetic_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        for row in reader:
            out = dict(row)
            if col_name is not None and col_name in row:
                out["feasibility"] = _parse_float(row.get(col_name), 0.0)
            elif rng is not None:
                out["feasibility"] = rng.random()
                used_fallback = True
            else:
                raise ValueError(
                    f"No feasibility column found in {synthetic_csv}; "
                    f"tried {FEASIBILITY_COLUMNS}. Pass random_fallback_seed for smoke test."
                )
            rows.append(out)
    if used_fallback:
        sys.stderr.write(
            f"[semi_hard_curriculum] WARNING: no feasibility column in "
            f"{synthetic_csv}; used uniform random fallback "
            f"(seed={random_fallback_seed}). Smoke-test only.\n"
        )
    return rows


def write_negatives_csv(negatives: Sequence[Dict[str, Any]], path: str) -> None:
    """Write a list of negative-row dicts to CSV, preserving all columns."""
    if not negatives:
        # Still write an empty file with a placeholder header so downstream
        # tools do not crash on a missing file.
        with open(path, "w", newline="", encoding="utf-8") as handle:
            handle.write("source_id,candidate_reaction,label\n")
        return
    fieldnames = list(negatives[0].keys())
    # Ensure required columns are present even if some rows are missing them.
    required = ("source_id", "candidate_reaction", "review_status")
    for col in required:
        if col not in fieldnames:
            fieldnames.append(col)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in negatives:
            writer.writerow(row)


class SemiHardCurriculum:
    """Semi-hard curriculum controller for boundary negative training.

    Instead of using all negatives at once (one-shot), progressively introduces
    harder negatives across multiple rounds. Each round warm-starts from the
    previous round's best checkpoint to avoid catastrophic forgetting.

    Args:
        rounds: List of (feasibility_low, feasibility_high) tuples. If empty
            and ``quantile_rounds`` is set, rounds are computed as equal-size
            quantiles of the feasibility distribution.
        epochs_per_round: epochs to train in each round.
        overlap: 0.0-1.0, fraction of previous round's negatives carried over.
        quantile_rounds: if ``rounds`` is empty, split the feasibility
            distribution into this many equal-size quantile rounds. Ignored
            when ``rounds`` is non-empty.
        min_round_size: minimum number of negatives per round. If a round
            has fewer negatives, the feasibility window is expanded toward the
            global median until the threshold is met (or all negatives are
            exhausted).
        seed: RNG seed for the overlap sampling and fallback.
    """

    def __init__(
        self,
        rounds: Optional[List[Tuple[float, float]]] = None,
        epochs_per_round: int = 10,
        overlap: float = 0.2,
        quantile_rounds: int = 0,
        min_round_size: int = 8,
        seed: int = 20260719,
    ) -> None:
        self.rounds_spec = list(rounds) if rounds else []
        self.epochs_per_round = int(epochs_per_round)
        self.overlap = max(0.0, min(1.0, float(overlap)))
        self.quantile_rounds = int(quantile_rounds)
        self.min_round_size = int(min_round_size)
        self.seed = int(seed)
        # populated during run_curriculum:
        self._prev_round_negatives: List[Dict[str, Any]] = []
        self._round_records: List[CurriculumRound] = []

    # ------------------------------------------------------------------
    # Round construction
    # ------------------------------------------------------------------
    def _resolve_rounds(self, all_negatives: Sequence[Dict[str, Any]]) -> List[Tuple[float, float]]:
        if self.rounds_spec:
            return [(float(lo), float(hi)) for lo, hi in self.rounds_spec]
        if self.quantile_rounds and self.quantile_rounds >= 2 and len(all_negatives) >= self.quantile_rounds:
            values = sorted(float(r["feasibility"]) for r in all_negatives)
            n = len(values)
            bounds = [values[min(int(i * n / self.quantile_rounds), n - 1)] for i in range(self.quantile_rounds + 1)]
            bounds[-1] = values[-1] + 1e-9
            ranges: List[Tuple[float, float]] = []
            for i in range(self.quantile_rounds):
                lo = bounds[i]
                hi = bounds[i + 1]
                # ensure non-empty windows
                if hi <= lo:
                    hi = lo + 1e-6
                ranges.append((lo, hi))
            return ranges
        # default: 4 equal quantiles
        if not all_negatives:
            return [(0.0, 1.0)]
        values = sorted(float(r["feasibility"]) for r in all_negatives)
        n = len(values)
        k = 4
        bounds = [values[min(int(i * n / k), n - 1)] for i in range(k + 1)]
        bounds[-1] = values[-1] + 1e-9
        return [(bounds[i], bounds[i + 1]) for i in range(k)]

    def select_negatives_for_round(
        self,
        all_negatives: Sequence[Dict[str, Any]],
        round_idx: int,
        resolved_rounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> List[Dict[str, Any]]:
        """Select negatives for given round based on feasibility range + overlap.

        For round 0, only negatives in ``rounds[0]`` are selected. For later
        rounds, the round's window is used plus ``overlap`` fraction of the
        previous round's negatives (random sample, seeded).
        """
        if resolved_rounds is None:
            resolved_rounds = self._resolve_rounds(all_negatives)
        if round_idx < 0 or round_idx >= len(resolved_rounds):
            raise IndexError(f"round_idx {round_idx} out of range (0..{len(resolved_rounds) - 1})")
        lo, hi = resolved_rounds[round_idx]

        def in_window(row: Dict[str, Any]) -> bool:
            v = float(row["feasibility"])
            return lo <= v < hi

        # If the window is too small, expand symmetrically until we hit min_round_size.
        selected = [r for r in all_negatives if in_window(r)]
        if len(selected) < self.min_round_size and all_negatives:
            values = sorted(float(r["feasibility"]) for r in all_negatives)
            global_lo = values[0]
            global_hi = values[-1] + 1e-9
            expand = 0.05
            while len(selected) < self.min_round_size and (lo > global_lo or hi < global_hi):
                lo = max(global_lo, lo - expand)
                hi = min(global_hi, hi + expand)
                selected = [r for r in all_negatives if lo <= float(r["feasibility"]) < hi]
                if lo <= global_lo and hi >= global_hi:
                    break

        # Overlap from previous round (dedup by source_id + candidate_reaction).
        carried: List[Dict[str, Any]] = []
        if round_idx > 0 and self._prev_round_negatives and self.overlap > 0.0:
            rng = random.Random(self.seed + round_idx * 17)
            k = max(1, int(round(len(self._prev_round_negatives) * self.overlap)))
            carried = rng.sample(self._prev_round_negatives, min(k, len(self._prev_round_negatives)))
        seen = {(r.get("source_id"), r.get("candidate_reaction")) for r in selected}
        merged = list(selected)
        for r in carried:
            key = (r.get("source_id"), r.get("candidate_reaction"))
            if key not in seen:
                merged.append(r)
                seen.add(key)
        return merged

    # ------------------------------------------------------------------
    # Subprocess runner
    # ------------------------------------------------------------------
    def _build_train_command(
        self,
        real_csv: str,
        synthetic_csv: str,
        output_dir: str,
        base_args: Dict[str, Any],
        init_checkpoint: Optional[str] = None,
        epochs_override: Optional[int] = None,
    ) -> List[str]:
        py = os.environ.get("PC_CNG_PYTHON", sys.executable)
        cmd: List[str] = [
            py, "-m", "pc_cng.train_pairwise_reward_mlp",
            "--real-csv", real_csv,
            "--synthetic-csv", synthetic_csv,
            "--output-dir", output_dir,
        ]
        int_keys = ("epochs", "batch_size", "hidden_dim", "lr", "dropout", "margin",
                    "pairwise_weight", "bce_weight", "n_bits", "seed", "warmup_epochs", "lr_min")
        str_keys = ("feature_mode", "fp_mode", "lr_scheduler", "checkpoint_metric",
                    "checkpoint_group_by")
        flag_keys = ("include_descriptors",)
        for key in int_keys:
            if key in base_args and base_args[key] is not None:
                cmd += [f"--{key.replace('_', '-')}", str(base_args[key])]
        for key in str_keys:
            if key in base_args and base_args[key] is not None:
                cmd += [f"--{key.replace('_', '-')}", str(base_args[key])]
        for key in flag_keys:
            if base_args.get(key):
                cmd.append(f"--{key.replace('_', '-')}")
        # family-margin / family-weight / class-margin / class-weight (lists of "k=v")
        for key in ("family_margin", "family_weight", "class_margin", "class_weight"):
            for item in base_args.get(key, []) or []:
                cmd += [f"--{key.replace('_', '-')}", str(item)]
        if epochs_override is not None:
            # Replace any --epochs already in cmd
            if "--epochs" in cmd:
                i = cmd.index("--epochs")
                cmd[i + 1] = str(epochs_override)
            else:
                cmd += ["--epochs", str(epochs_override)]
        if init_checkpoint:
            cmd += ["--init-checkpoint", init_checkpoint]
        return cmd

    def _run_subprocess(self, cmd: Sequence[str], log_path: str) -> int:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log:
            log.write("CMD: " + " ".join(cmd) + "\n\n")
            log.flush()
            result = subprocess.run(
                list(cmd),
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=os.environ.get("PC_CNG_CWD", "."),
            )
        return result.returncode

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def run_curriculum(
        self,
        real_csv: str,
        synthetic_csv: str,
        output_dir: str,
        base_train_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the full 4-round curriculum training.

        Returns a dict with per-round CurriculumRound records, the final test
        Top-1 (from the last round's best checkpoint), and the path to the
        final best checkpoint (for downstream evaluation).
        """
        os.makedirs(output_dir, exist_ok=True)
        all_negatives = load_negatives_with_feasibility(
            synthetic_csv, random_fallback_seed=self.seed
        )
        if not all_negatives:
            raise RuntimeError(f"No negatives loaded from {synthetic_csv}")
        resolved_rounds = self._resolve_rounds(all_negatives)
        self._round_records = []
        self._prev_round_negatives = []

        init_ckpt: Optional[str] = None
        history: List[Dict[str, Any]] = []
        final_test_top1 = float("nan")
        final_best_ckpt = ""

        for round_idx, (lo, hi) in enumerate(resolved_rounds):
            round_negatives = self.select_negatives_for_round(
                all_negatives, round_idx, resolved_rounds=resolved_rounds
            )
            round_dir = os.path.join(output_dir, f"round_{round_idx}")
            os.makedirs(round_dir, exist_ok=True)
            round_csv = os.path.join(round_dir, "synthetic_negatives_round.csv")
            write_negatives_csv(round_negatives, round_csv)

            cmd = self._build_train_command(
                real_csv=real_csv,
                synthetic_csv=round_csv,
                output_dir=round_dir,
                base_args=dict(base_train_args),
                init_checkpoint=init_ckpt,
                epochs_override=self.epochs_per_round,
            )
            log_path = os.path.join(round_dir, "train.log")
            rc = self._run_subprocess(cmd, log_path)
            if rc != 0:
                raise RuntimeError(
                    f"Round {round_idx} training failed (rc={rc}); see {log_path}"
                )
            metrics_path = os.path.join(round_dir, "metrics.json")
            if not os.path.exists(metrics_path):
                raise RuntimeError(f"Round {round_idx} produced no metrics.json in {round_dir}")
            with open(metrics_path, "r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            round_best_ckpt = metrics.get("best_checkpoint", os.path.join(round_dir, "best_pairwise_reward_mlp.pt"))
            round_metric_value = float(metrics.get("best_checkpoint_metric_value", float("nan")))
            test_ranking = metrics.get("test_ranking_real", {}) or {}
            round_test_top1 = float(test_ranking.get("top1", float("nan")))
            round_history = metrics.get("history", []) or []

            record = CurriculumRound(
                round_idx=round_idx,
                feasibility_range=(lo, hi),
                num_negatives=len(round_negatives),
                epochs=self.epochs_per_round,
                output_dir=round_dir,
                best_checkpoint=round_best_ckpt,
                best_metric_value=round_metric_value,
                final_test_top1=round_test_top1,
                history=round_history,
            )
            self._round_records.append(record)
            history.append({
                "round_idx": round_idx,
                "feasibility_range": [lo, hi],
                "num_negatives": len(round_negatives),
                "epochs": self.epochs_per_round,
                "best_metric_value": round_metric_value,
                "test_top1": round_test_top1,
            })
            init_ckpt = round_best_ckpt
            final_test_top1 = round_test_top1
            final_best_ckpt = round_best_ckpt
            # carry over this round's negatives for overlap in next round
            self._prev_round_negatives = list(round_negatives)

        summary = {
            "mode": "curriculum",
            "rounds": [r.to_dict() for r in self._round_records],
            "history": history,
            "final_test_top1": final_test_top1,
            "final_best_checkpoint": final_best_ckpt,
            "epochs_per_round": self.epochs_per_round,
            "total_epochs": self.epochs_per_round * len(resolved_rounds),
            "overlap": self.overlap,
            "seed": self.seed,
            "num_rounds": len(resolved_rounds),
            "total_negatives": len(all_negatives),
        }
        with open(os.path.join(output_dir, "curriculum_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        # Also dump a per-round per-epoch CSV for easy plotting.
        with open(os.path.join(output_dir, "curriculum_history.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["round_idx", "epoch", "loss", "lr",
                            "val_roc_auc", "val_top1", "checkpoint_metric_value"],
            )
            writer.writeheader()
            for r in self._round_records:
                for h in r.history:
                    val = h.get("val", {}) or {}
                    val_rank = h.get("val_ranking", {}) or {}
                    writer.writerow({
                        "round_idx": r.round_idx,
                        "epoch": h.get("epoch"),
                        "loss": h.get("loss"),
                        "lr": h.get("lr"),
                        "val_roc_auc": val.get("roc_auc"),
                        "val_top1": val_rank.get("top1"),
                        "checkpoint_metric_value": h.get("checkpoint_metric_value"),
                    })
        return summary

    def run_one_shot_baseline(
        self,
        real_csv: str,
        synthetic_csv: str,
        output_dir: str,
        base_train_args: Dict[str, Any],
        total_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """One-shot baseline: all negatives at once, trained for ``total_epochs``.

        ``total_epochs`` defaults to ``epochs_per_round * num_rounds`` so the
        comparison is fair (same total compute).
        """
        os.makedirs(output_dir, exist_ok=True)
        all_negatives = load_negatives_with_feasibility(
            synthetic_csv, random_fallback_seed=self.seed
        )
        resolved_rounds = self._resolve_rounds(all_negatives)
        n_rounds = max(1, len(resolved_rounds))
        if total_epochs is None:
            total_epochs = self.epochs_per_round * n_rounds

        cmd = self._build_train_command(
            real_csv=real_csv,
            synthetic_csv=synthetic_csv,
            output_dir=output_dir,
            base_args=dict(base_train_args),
            init_checkpoint=None,
            epochs_override=int(total_epochs),
        )
        log_path = os.path.join(output_dir, "train.log")
        rc = self._run_subprocess(cmd, log_path)
        if rc != 0:
            raise RuntimeError(f"One-shot baseline failed (rc={rc}); see {log_path}")
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        test_ranking = metrics.get("test_ranking_real", {}) or {}
        summary = {
            "mode": "one_shot",
            "total_epochs": int(total_epochs),
            "total_negatives": len(all_negatives),
            "final_test_top1": float(test_ranking.get("top1", float("nan"))),
            "best_checkpoint": metrics.get("best_checkpoint"),
            "best_metric_value": float(metrics.get("best_checkpoint_metric_value", float("nan"))),
            "test_metrics": metrics.get("test", {}),
            "test_ranking_real": test_ranking,
            "history": metrics.get("history", []) or [],
            "seed": self.seed,
        }
        with open(os.path.join(output_dir, "one_shot_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        return summary
