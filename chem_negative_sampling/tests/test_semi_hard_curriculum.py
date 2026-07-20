"""Unit tests for the semi-hard curriculum controller (P1-07).

Tests run on CPU and use tiny synthetic fixtures so they finish in seconds.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest
from typing import Any, Dict, List

# Ensure the repo root is on sys.path so `pc_cng` and `tests` import cleanly
# regardless of where pytest is invoked from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pc_cng.semi_hard_curriculum import (  # noqa: E402
    CurriculumRound,
    SemiHardCurriculum,
    load_negatives_with_feasibility,
    write_negatives_csv,
)


def _make_neg_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write a tiny boundary negatives CSV with hard_score + required cols."""
    if not rows:
        rows = [{
            "source_id": "src_1",
            "positive_reaction": "CC(=O)O.CCO>>CC(=O)OCC",
            "candidate_reaction": "CC(=O)O.CCO>>CCO",
            "review_status": "keep_synthetic_negative",
            "hard_score": 0.6,
            "action_family": "center_atom:Br->Cl",
            "failure_type": "forward_outcome",
        }]
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# Two real reaction contexts (so ranking_groups have both pos+neg per group).
_R1_REACT = "CC(=O)O.CCO"          # acetic acid + ethanol
_R1_POS_PROD = "CC(=O)OCC"         # ethyl acetate (positive outcome)
_R1_NEG_PROD = "CCO"               # ethanol (no reaction - negative outcome)
_R2_REACT = "c1ccccc1.BrBr"        # benzene + bromine
_R2_POS_PROD = "Brc1ccccc1"        # bromobenzene (positive outcome)
_R2_NEG_PROD = "c1ccccc1"          # benzene (no reaction - negative outcome)

# Four reaction rows cycled through to fill train/val/test:
# (reactants, product, label_type)
_FIXTURE_REACTIONS = [
    (_R1_REACT, _R1_POS_PROD, "positive"),
    (_R1_REACT, _R1_NEG_PROD, "real_negative"),
    (_R2_REACT, _R2_POS_PROD, "positive"),
    (_R2_REACT, _R2_NEG_PROD, "real_negative"),
]

# Pool of valid SMILES products for synthetic negatives (so featurize_rows
# does not drop them for having all-zero fingerprints).  All of these parse
# to non-trivial Morgan fingerprints.
_NEG_PRODUCT_POOL = [
    "CCO", "CC(=O)O", "C", "CC", "CCC", "CCCO", "CCCCO",
    "c1ccccc1", "Cc1ccccc1", "Brc1ccccc1", "Clc1ccccc1", "c1ccncc1",
    "C1CCCCC1", "CC(=O)C", "OCCO", "CCOCC", "CC(C)O", "CCN", "CCNC",
    "C1COCCO1", "CC(=O)N", "CC(=O)NC", "OC(=O)CC", "O=C(O)CC",
    "CC(=O)OC", "CC(=O)OCC", "CCOC(=O)C", "Brc1ccncc1", "Cc1cccnc1",
    "Clc1cccnc1", "c1ccccc1O", "Cc1ccccc1O", "Brc1ccccc1O",
    "CC(=O)Oc1ccccc1", "OC(=O)c1ccccc1", "CCOc1ccccc1",
    "c1ccccc1.N", "c1ccccc1.CCO", "c1ccccc1.CC(=O)O", "Brc1ccccc1.CCO",
]


def _make_real_csv(path: str, n_train: int = 8, n_val: int = 4, n_test: int = 4) -> None:
    """Write a tiny real positives/negatives CSV with valid SMILES.

    Cycle through 4 reaction rows (2 positive, 2 negative, spanning 2 distinct
    reactant sets so ranking_groups has both pos+neg per group).  Source IDs
    are assigned sequentially so the odd-indexed train rows (src_1, src_3, ...)
    are positives that synthetic negatives can pair against.
    """
    rows: List[Dict[str, Any]] = []
    idx = 0
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        for i in range(n):
            idx += 1
            reactants, product, label_type = _FIXTURE_REACTIONS[i % 4]
            rows.append({
                "source_id": f"src_{idx}",
                "reaction_smiles": f"{reactants}>>{product}",
                "reactants": reactants,
                "agents": "",
                "products": product,
                "label_type": label_type,
                "yield": "",
                "source": "test_fixture",
                "split_key": f"k{idx}",
                "split": split,
            })
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_smoke_neg_rows(n: int, seed: int = 0) -> List[Dict[str, Any]]:
    """Build ``n`` synthetic negatives with valid candidate reactions.

    Source IDs cycle through the train-positive source IDs (src_1, src_3,
    src_5, src_7 for the default 8-row train set).  Candidate reactions use
    the matching reactants paired with a varied valid product from
    ``_NEG_PRODUCT_POOL`` so each row featurizes to a non-zero fingerprint.
    Hard scores span [0.1, 0.9] so the 4-round curriculum has coverage.
    """
    train_pos_ids = ["src_1", "src_3", "src_5", "src_7"]
    train_pos_reactants = [_R1_REACT, _R2_REACT, _R1_REACT, _R2_REACT]
    train_pos_products = [_R1_POS_PROD, _R2_POS_PROD, _R1_POS_PROD, _R2_POS_PROD]
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        slot = i % len(train_pos_ids)
        sid = train_pos_ids[slot]
        reactants = train_pos_reactants[slot]
        pos_product = train_pos_products[slot]
        neg_product = _NEG_PRODUCT_POOL[i % len(_NEG_PRODUCT_POOL)]
        candidate = f"{reactants}>>{neg_product}"
        # Spread hard_score across [0.1, 0.9] so 4-round curriculum has coverage.
        hard_score = 0.1 + 0.8 * (i / max(n - 1, 1))
        rows.append({
            "source_id": sid,
            "positive_reaction": f"{reactants}>>{pos_product}",
            "candidate_reaction": candidate,
            "review_status": "keep_synthetic_negative",
            "hard_score": f"{hard_score:.4f}",
            "action_family": "center_atom:Br->Cl",
            "failure_type": "forward_outcome",
            "task": "forward_outcome",
        })
    return rows


class TestSemiHardCurriculum(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="semi_hard_test_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # 1. CurriculumRound dataclass
    # ------------------------------------------------------------------
    def test_curriculum_round_init(self) -> None:
        r = CurriculumRound(
            round_idx=2,
            feasibility_range=(0.25, 0.55),
            num_negatives=42,
            epochs=10,
        )
        self.assertEqual(r.round_idx, 2)
        self.assertEqual(r.feasibility_range, (0.25, 0.55))
        self.assertEqual(r.num_negatives, 42)
        self.assertEqual(r.epochs, 10)
        d = r.to_dict()
        self.assertEqual(d["feasibility_range"], [0.25, 0.55])
        self.assertEqual(d["num_negatives"], 42)
        # default fields exist
        self.assertEqual(d["best_checkpoint"], "")
        self.assertTrue(isinstance(d["history"], list))

    # ------------------------------------------------------------------
    # 2. select_negatives_for_round respects feasibility range
    # ------------------------------------------------------------------
    def test_select_negatives_feasibility_range(self) -> None:
        negatives = [
            {"source_id": f"s{i}", "candidate_reaction": f"A>>C{i}",
             "hard_score": v, "feasibility": v,
             "review_status": "keep_synthetic_negative"}
            for i, v in enumerate([0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
        ]
        c = SemiHardCurriculum(
            rounds=[(0.10, 0.35), (0.25, 0.55), (0.40, 0.70), (0.50, 0.80)],
            epochs_per_round=1, overlap=0.0, min_round_size=1, seed=1,
        )
        r0 = c.select_negatives_for_round(negatives, 0)
        r1 = c.select_negatives_for_round(negatives, 1)
        r2 = c.select_negatives_for_round(negatives, 2)
        r3 = c.select_negatives_for_round(negatives, 3)
        # Round 0: feasibility in [0.10, 0.35) -> 0.15, 0.20, 0.30
        self.assertEqual(len(r0), 3)
        self.assertEqual({r["source_id"] for r in r0}, {"s0", "s1", "s2"})
        # Round 1: [0.25, 0.55) -> 0.30, 0.40, 0.50
        self.assertEqual(len(r1), 3)
        self.assertEqual({r["source_id"] for r in r1}, {"s2", "s3", "s4"})
        # Round 2: [0.40, 0.70) -> 0.40, 0.50, 0.60
        self.assertEqual({r["source_id"] for r in r2}, {"s3", "s4", "s5"})
        # Round 3: [0.50, 0.80) -> 0.50, 0.60, 0.70
        self.assertEqual({r["source_id"] for r in r3}, {"s4", "s5", "s6"})

    # ------------------------------------------------------------------
    # 3. overlap is carried over from previous round
    # ------------------------------------------------------------------
    def test_select_negatives_overlap(self) -> None:
        negatives = [
            {"source_id": f"s{i}", "candidate_reaction": f"A>>C{i}",
             "hard_score": v, "feasibility": v,
             "review_status": "keep_synthetic_negative"}
            for i, v in enumerate([0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
        ]
        c = SemiHardCurriculum(
            rounds=[(0.10, 0.35), (0.30, 0.55)],
            epochs_per_round=1, overlap=1.0, min_round_size=1, seed=2,
        )
        # Round 0 -> [0.10,0.35): s0,s1,s2 (3 negatives)
        r0 = c.select_negatives_for_round(negatives, 0)
        self.assertEqual(len(r0), 3)
        c._prev_round_negatives = list(r0)  # simulate curriculum state
        # Round 1 -> [0.30,0.55): s2,s3,s4 (0.30,0.40,0.50); overlap=1.0 adds s0,s1 from prev
        r1 = c.select_negatives_for_round(negatives, 1)
        ids = {r["source_id"] for r in r1}
        self.assertIn("s2", ids)  # in current window
        self.assertIn("s3", ids)
        self.assertIn("s4", ids)
        self.assertIn("s0", ids)  # carried over
        self.assertIn("s1", ids)  # carried over

    # ------------------------------------------------------------------
    # 4. 4-round curriculum runs end-to-end (smoke; tiny data)
    # ------------------------------------------------------------------
    def test_curriculum_4rounds_smoke(self) -> None:
        # Use hard_score spanning [0.1, 0.9] so 4-round quantile curriculum has coverage.
        neg_rows = _make_smoke_neg_rows(40)
        neg_csv = os.path.join(self.tmp, "negatives.csv")
        _make_neg_csv(neg_csv, neg_rows)
        real_csv = os.path.join(self.tmp, "real.csv")
        _make_real_csv(real_csv, n_train=8, n_val=4, n_test=4)

        out_dir = os.path.join(self.tmp, "curriculum_out")
        c = SemiHardCurriculum(
            rounds=None, epochs_per_round=1, overlap=0.2,
            quantile_rounds=4, min_round_size=4, seed=42,
        )
        base_args = {
            "epochs": 1, "batch_size": 4, "hidden_dim": 16, "lr": 1e-3,
            "dropout": 0.0, "feature_mode": "morgan", "n_bits": 64,
            "fp_mode": "binary", "pairwise_weight": 1.0, "bce_weight": 1.0,
            "margin": 0.1, "seed": 42,
            "checkpoint_metric": "val_roc_auc", "checkpoint_group_by": "reactants",
        }
        # Force CPU for tests
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            summary = c.run_curriculum(
                real_csv=real_csv,
                synthetic_csv=neg_csv,
                output_dir=out_dir,
                base_train_args=base_args,
            )
        finally:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        self.assertEqual(summary["num_rounds"], 4)
        self.assertEqual(summary["epochs_per_round"], 1)
        self.assertEqual(summary["total_epochs"], 4)
        self.assertEqual(len(summary["rounds"]), 4)
        # each round should have a checkpoint saved
        for r in summary["rounds"]:
            self.assertTrue(os.path.exists(r["best_checkpoint"]),
                            f"Missing checkpoint for round {r['round_idx']}: {r['best_checkpoint']}")
        # curriculum_summary.json written
        self.assertTrue(os.path.exists(os.path.join(out_dir, "curriculum_summary.json")))
        # curriculum_history.csv written
        self.assertTrue(os.path.exists(os.path.join(out_dir, "curriculum_history.csv")))

    # ------------------------------------------------------------------
    # 5. one-shot baseline runs
    # ------------------------------------------------------------------
    def test_one_shot_baseline_runs(self) -> None:
        neg_rows = _make_smoke_neg_rows(20)
        neg_csv = os.path.join(self.tmp, "negatives.csv")
        _make_neg_csv(neg_csv, neg_rows)
        real_csv = os.path.join(self.tmp, "real.csv")
        _make_real_csv(real_csv, n_train=8, n_val=4, n_test=4)

        out_dir = os.path.join(self.tmp, "one_shot_out")
        c = SemiHardCurriculum(
            rounds=None, epochs_per_round=1, overlap=0.0,
            quantile_rounds=4, min_round_size=4, seed=42,
        )
        base_args = {
            "epochs": 1, "batch_size": 4, "hidden_dim": 16, "lr": 1e-3,
            "dropout": 0.0, "feature_mode": "morgan", "n_bits": 64,
            "fp_mode": "binary", "pairwise_weight": 1.0, "bce_weight": 1.0,
            "margin": 0.1, "seed": 42,
            "checkpoint_metric": "val_roc_auc", "checkpoint_group_by": "reactants",
        }
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            summary = c.run_one_shot_baseline(
                real_csv=real_csv,
                synthetic_csv=neg_csv,
                output_dir=out_dir,
                base_train_args=base_args,
                total_epochs=4,
            )
        finally:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        self.assertEqual(summary["mode"], "one_shot")
        self.assertEqual(summary["total_epochs"], 4)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "one_shot_summary.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "metrics.json")))

    # ------------------------------------------------------------------
    # 6. comparison.json format is correct
    # ------------------------------------------------------------------
    def test_comparison_output_format(self) -> None:
        from pc_cng.evaluate_semi_hard_curriculum import compare_curriculum_vs_one_shot
        # Run a tiny curriculum + one-shot first.
        neg_rows = _make_smoke_neg_rows(24)
        neg_csv = os.path.join(self.tmp, "negatives.csv")
        _make_neg_csv(neg_csv, neg_rows)
        real_csv = os.path.join(self.tmp, "real.csv")
        _make_real_csv(real_csv, n_train=8, n_val=4, n_test=4)
        c = SemiHardCurriculum(
            rounds=None, epochs_per_round=1, overlap=0.2,
            quantile_rounds=4, min_round_size=4, seed=7,
        )
        base_args = {
            "epochs": 1, "batch_size": 4, "hidden_dim": 16, "lr": 1e-3,
            "dropout": 0.0, "feature_mode": "morgan", "n_bits": 64,
            "fp_mode": "binary", "pairwise_weight": 1.0, "bce_weight": 1.0,
            "margin": 0.1, "seed": 7,
            "checkpoint_metric": "val_roc_auc", "checkpoint_group_by": "reactants",
        }
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            c.run_curriculum(real_csv, neg_csv, os.path.join(self.tmp, "curr"), base_args)
            c.run_one_shot_baseline(real_csv, neg_csv, os.path.join(self.tmp, "one"),
                                    base_args, total_epochs=4)
        finally:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        comp = compare_curriculum_vs_one_shot(
            curriculum_dir=os.path.join(self.tmp, "curr"),
            one_shot_dir=os.path.join(self.tmp, "one"),
            output_dir=self.tmp,
            bootstrap_iterations=200, seed=7,
        )
        for key in [
            "curriculum_test_top1_group_mean", "one_shot_test_top1_group_mean",
            "group_mean_diff_top1", "group_mean_diff_pp",
            "curriculum_test_top1_model", "one_shot_test_top1_model",
            "model_diff_pp", "n_paired_groups",
            "bootstrap_ci_low", "bootstrap_ci_high",
            "permutation_p_value", "sign_test_p_value",
            "ci_fully_positive", "pass_threshold_pp",
            "go_nogo_decision",
        ]:
            self.assertIn(key, comp, f"Missing key {key} in comparison")
        self.assertIn(comp["go_nogo_decision"], {"pass", "supplementary", "fail", "unknown"})
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "comparison.json")))

    # ------------------------------------------------------------------
    # 7. paired diff calculation is correct
    # ------------------------------------------------------------------
    def test_paired_diff_calculation(self) -> None:
        from pc_cng.evaluate_semi_hard_curriculum import paired_diffs
        baseline = {
            "g1": [{"label": 1, "score": 0.9}, {"label": 0, "score": 0.4}],
            "g2": [{"label": 1, "score": 0.3}, {"label": 0, "score": 0.5}],  # baseline loses g2
            "g3": [{"label": 1, "score": 0.7}, {"label": 0, "score": 0.6}],
        }
        candidate = {
            "g1": [{"label": 1, "score": 0.95}, {"label": 0, "score": 0.1}],  # still wins
            "g2": [{"label": 1, "score": 0.8}, {"label": 0, "score": 0.2}],   # now wins
            "g3": [{"label": 1, "score": 0.55}, {"label": 0, "score": 0.6}],  # now loses
        }
        diffs = paired_diffs(baseline, candidate)
        d_by_group = {d["group_id"]: d for d in diffs}
        self.assertEqual(set(d_by_group), {"g1", "g2", "g3"})
        # g1: both win -> diff 0
        self.assertAlmostEqual(d_by_group["g1"]["diff_top1"], 0.0)
        # g2: baseline loses (0), candidate wins (1) -> diff +1
        self.assertAlmostEqual(d_by_group["g2"]["diff_top1"], 1.0)
        # g3: baseline wins (1), candidate loses (0) -> diff -1
        self.assertAlmostEqual(d_by_group["g3"]["diff_top1"], -1.0)

    # ------------------------------------------------------------------
    # 8. checkpoint per round is saved
    # ------------------------------------------------------------------
    def test_checkpoint_per_round_saved(self) -> None:
        neg_rows = _make_smoke_neg_rows(32)
        neg_csv = os.path.join(self.tmp, "negatives.csv")
        _make_neg_csv(neg_csv, neg_rows)
        real_csv = os.path.join(self.tmp, "real.csv")
        _make_real_csv(real_csv, n_train=8, n_val=4, n_test=4)
        out_dir = os.path.join(self.tmp, "curr")
        c = SemiHardCurriculum(
            rounds=None, epochs_per_round=1, overlap=0.2,
            quantile_rounds=4, min_round_size=4, seed=99,
        )
        base_args = {
            "epochs": 1, "batch_size": 4, "hidden_dim": 16, "lr": 1e-3,
            "dropout": 0.0, "feature_mode": "morgan", "n_bits": 64,
            "fp_mode": "binary", "pairwise_weight": 1.0, "bce_weight": 1.0,
            "margin": 0.1, "seed": 99,
            "checkpoint_metric": "val_roc_auc", "checkpoint_group_by": "reactants",
        }
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            summary = c.run_curriculum(real_csv, neg_csv, out_dir, base_args)
        finally:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        for r in summary["rounds"]:
            ckpt_path = r["best_checkpoint"]
            self.assertTrue(os.path.exists(ckpt_path),
                            f"Round {r['round_idx']} checkpoint missing: {ckpt_path}")
            # checkpoint should be a .pt file
            self.assertTrue(ckpt_path.endswith(".pt"))

    # ------------------------------------------------------------------
    # 9. feasibility score extraction from synthetic negatives
    # ------------------------------------------------------------------
    def test_feasibility_score_extraction(self) -> None:
        rows = [
            {"source_id": "s1", "candidate_reaction": "A>>C1",
             "review_status": "keep_synthetic_negative",
             "hard_score": "0.45"},
            {"source_id": "s2", "candidate_reaction": "A>>C2",
             "review_status": "keep_synthetic_negative",
             "hard_score": "0.72"},
            {"source_id": "s3", "candidate_reaction": "A>>C3",
             "review_status": "keep_synthetic_negative",
             "hard_score": "0.30"},
        ]
        path = os.path.join(self.tmp, "negs.csv")
        _make_neg_csv(path, rows)
        loaded = load_negatives_with_feasibility(path)
        self.assertEqual(len(loaded), 3)
        feas = [r["feasibility"] for r in loaded]
        self.assertAlmostEqual(feas[0], 0.45)
        self.assertAlmostEqual(feas[1], 0.72)
        self.assertAlmostEqual(feas[2], 0.30)
        # original columns preserved
        for r in loaded:
            self.assertIn("hard_score", r)
            self.assertIn("review_status", r)

    def test_feasibility_random_fallback(self) -> None:
        """No hard_score column -> use random fallback when seed provided."""
        rows = [
            {"source_id": "s1", "candidate_reaction": "A>>C1",
             "review_status": "keep_synthetic_negative"},
            {"source_id": "s2", "candidate_reaction": "A>>C2",
             "review_status": "keep_synthetic_negative"},
        ]
        path = os.path.join(self.tmp, "negs_no_hs.csv")
        _make_neg_csv(path, rows)
        loaded = load_negatives_with_feasibility(path, random_fallback_seed=123)
        self.assertEqual(len(loaded), 2)
        for r in loaded:
            self.assertIn("feasibility", r)
            self.assertGreaterEqual(r["feasibility"], 0.0)
            self.assertLess(r["feasibility"], 1.0)


if __name__ == "__main__":
    unittest.main()
