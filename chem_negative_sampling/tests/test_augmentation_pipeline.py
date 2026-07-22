"""Tests for P4-G3 Augmentation Pipeline.

Run with::

    python3 -m pytest chem_negative_sampling/tests/test_augmentation_pipeline.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# Ensure the pc_cng package is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from pc_cng.run_p4_augmentation import (
    ARM_DEFINITIONS,
    ARM_IDS,
    BackboneConfig,
    build_arm_training_data,
    build_gnn_model,
    compute_auprc,
    compute_calibration_metrics,
    compute_effect_sizes,
    compute_go_no_go,
    compute_metrics_from_predictions,
    load_manifest_candidates,
    paired_bootstrap_ci,
    run_single_experiment,
    set_seed,
    write_summary_csv,
)
from pc_cng.gnn_backbone import (
    GNNReactionEncoder,
    GNNReactionScorer,
    GNNRankingHead,
    GATLayer,
    atom_features,
    bond_features,
    build_gnn_scorer,
    collate_graphs,
    count_parameters,
    count_trainable_parameters,
    mol_to_graph,
)


# ---------------------------------------------------------------------------
# GNN backbone tests
# ---------------------------------------------------------------------------

class TestGNNBackbone:
    """Tests for the pure-PyTorch GNN backbone."""

    def test_atom_features_dim(self):
        """Atom features should be 27-dimensional."""
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        atom = mol.GetAtomWithIdx(0)
        feats = atom_features(atom)
        assert len(feats) == 27

    def test_bond_features_dim(self):
        """Bond features should be 12-dimensional."""
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        bond = mol.GetBondWithIdx(0)
        feats = bond_features(bond)
        assert len(feats) == 12

    def test_mol_to_graph_valid(self):
        """Valid SMILES should produce a graph dict."""
        g = mol_to_graph("CCO")
        assert g is not None
        assert "x" in g
        assert "edge_index" in g
        assert "edge_attr" in g
        assert g["x"].shape[1] == 27
        assert g["edge_attr"].shape[1] == 12
        assert g["num_nodes"] == 3

    def test_mol_to_graph_invalid(self):
        """Invalid SMILES should return None."""
        g = mol_to_graph("not_a_smiles_###")
        assert g is None

    def test_collate_graphs(self):
        """Collate should concatenate graphs with correct batch indices."""
        g1 = mol_to_graph("CCO")
        g2 = mol_to_graph("c1ccccc1")
        batched = collate_graphs([g1, g2])
        assert batched["x"].shape[0] == g1["num_nodes"] + g2["num_nodes"]
        assert batched["batch"].max().item() == 1
        assert batched["num_graphs"] == 2

    def test_gat_layer_forward(self):
        """GATLayer should produce correct output shape."""
        layer = GATLayer(in_dim=27, out_dim=64, heads=4, edge_dim=12)
        g = mol_to_graph("CCO")
        batched = collate_graphs([g])
        out = layer(batched["x"], batched["edge_index"], batched["edge_attr"], batched["batch"])
        assert out.shape == (3, 64)

    def test_gnn_encoder_forward(self):
        """GNNReactionEncoder should produce [num_graphs, out_dim] output."""
        encoder = GNNReactionEncoder(hidden_dim=64, out_dim=128, num_layers=2, heads=2)
        g1 = mol_to_graph("CCO")
        g2 = mol_to_graph("c1ccccc1")
        out = encoder([g1, g2])
        assert out.shape == (2, 128)

    def test_gnn_scorer_forward(self):
        """GNNReactionScorer should produce scalar scores."""
        scorer = GNNReactionScorer(hidden_dim=64, encoder_out_dim=128, num_layers=2, heads=2)
        g = mol_to_graph("CCO")
        score = scorer([g])
        assert score.shape == (1,)

    def test_gnn_scorer_batch(self):
        """GNNReactionScorer should handle batches."""
        scorer = GNNReactionScorer(hidden_dim=64, encoder_out_dim=128, num_layers=2, heads=2)
        graphs = [mol_to_graph(s) for s in ["CCO", "c1ccccc1", "CC(=O)O"]]
        graphs = [g for g in graphs if g is not None]
        scores = scorer(graphs)
        assert scores.shape == (3,)

    def test_count_parameters(self):
        """Parameter counting should work."""
        model = GNNReactionScorer(hidden_dim=64, encoder_out_dim=128, num_layers=2, heads=2)
        total = count_parameters(model)
        trainable = count_trainable_parameters(model)
        assert total > 0
        assert trainable == total  # All params trainable by default

    def test_gnn_deterministic(self):
        """GNN should be deterministic with same seed and eval mode."""
        set_seed(42)
        model1 = GNNReactionScorer(hidden_dim=64, encoder_out_dim=128, num_layers=2, heads=2, dropout=0.0)
        model1.eval()
        set_seed(42)
        model2 = GNNReactionScorer(hidden_dim=64, encoder_out_dim=128, num_layers=2, heads=2, dropout=0.0)
        model2.eval()
        g = mol_to_graph("CCO")
        with torch.no_grad():
            out1 = model1([g])
            out2 = model2([g])
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Arm definition tests
# ---------------------------------------------------------------------------

class TestArmDefinitions:
    """Tests for augmentation arm definitions."""

    def test_seven_arms(self):
        """Must have exactly 7 arms (A0-A6)."""
        assert len(ARM_IDS) == 7
        assert ARM_IDS == ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]

    def test_a0_is_positive_only(self):
        """A0 must have no negative source."""
        assert ARM_DEFINITIONS["A0"]["negative_source"] is None

    def test_a6_is_rule_pccng(self):
        """A6 must be rule_pc_cng."""
        assert ARM_DEFINITIONS["A6"]["negative_source"] == "rule_pc_cng"

    def test_all_arms_have_names(self):
        """Each arm must have a name."""
        for arm_id in ARM_IDS:
            assert "name" in ARM_DEFINITIONS[arm_id]


# ---------------------------------------------------------------------------
# Training data building tests
# ---------------------------------------------------------------------------

class TestArmTrainingData:
    """Tests for arm-specific training data construction."""

    def _make_candidates(self, n_gold: int = 10, n_neg_per_type: int = 10) -> list:
        """Create synthetic candidate data."""
        candidates = []
        for i in range(n_gold):
            candidates.append({
                "group_id": f"g{i}",
                "candidate_id": f"gold_{i}",
                "smiles": "CCO",
                "label": 1,
                "gold_candidate": True,
                "candidate_source": "gold",
                "negative_source": None,
            })
        for source in ["random_mismatch", "random_corruption", "tanimoto_retrieval",
                       "template_perturbation", "unconstrained_edit", "rule_pc_cng"]:
            for i in range(n_neg_per_type):
                candidates.append({
                    "group_id": f"g{i}",
                    "candidate_id": f"{source}_{i}",
                    "smiles": "CCC",
                    "label": 0,
                    "gold_candidate": False,
                    "candidate_source": source,
                    "negative_source": source,
                })
        return candidates

    def test_a0_only_gold(self):
        """A0 should only include gold candidates."""
        candidates = self._make_candidates()
        arm_data = build_arm_training_data(candidates, "A0")
        assert all(d["label"] == 1 for d in arm_data)
        assert len(arm_data) == 10

    def test_a1_includes_mismatch(self):
        """A1 should include gold + random_mismatch negatives."""
        candidates = self._make_candidates()
        arm_data = build_arm_training_data(candidates, "A1")
        labels = [d["label"] for d in arm_data]
        assert labels.count(1) == 10  # gold
        assert labels.count(0) == 10  # mismatch negatives
        sources = set(d["negative_source"] for d in arm_data if d["label"] == 0)
        assert sources == {"random_mismatch"}

    def test_a6_includes_rule_pccng(self):
        """A6 should include gold + rule_pc_cng negatives."""
        candidates = self._make_candidates()
        arm_data = build_arm_training_data(candidates, "A6")
        labels = [d["label"] for d in arm_data]
        assert labels.count(1) == 10
        assert labels.count(0) == 10
        sources = set(d["negative_source"] for d in arm_data if d["label"] == 0)
        assert sources == {"rule_pc_cng"}


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for ranking and calibration metrics."""

    def test_perfect_ranking_mrr(self):
        """Perfect ranking should have MRR=1."""
        preds = [
            {"group_id": "g1", "label": 1, "score": 10.0},
            {"group_id": "g1", "label": 0, "score": 1.0},
            {"group_id": "g1", "label": 0, "score": 0.5},
        ]
        metrics = compute_metrics_from_predictions(preds)
        assert metrics["mrr"] == 1.0
        assert metrics["top1"] == 1.0

    def test_worst_ranking_mrr(self):
        """Worst ranking should have low MRR."""
        preds = [
            {"group_id": "g1", "label": 0, "score": 10.0},
            {"group_id": "g1", "label": 0, "score": 5.0},
            {"group_id": "g1", "label": 1, "score": 1.0},
        ]
        metrics = compute_metrics_from_predictions(preds)
        assert metrics["mrr"] == 1.0 / 3  # 1/3
        assert metrics["top1"] == 0.0

    def test_auprc_perfect(self):
        """Perfect predictions should have AUPRC=1."""
        preds = [
            {"group_id": "g1", "label": 1, "score": 10.0},
            {"group_id": "g1", "label": 1, "score": 9.0},
            {"group_id": "g1", "label": 0, "score": 1.0},
            {"group_id": "g1", "label": 0, "score": 0.0},
        ]
        auprc = compute_auprc(preds)
        assert auprc == pytest.approx(1.0, abs=0.01)

    def test_calibration_perfect(self):
        """Perfect calibration should have low ECE."""
        preds = [
            {"group_id": "g1", "label": 1, "score": 10.0},   # sigmoid ≈ 1
            {"group_id": "g1", "label": 0, "score": -10.0},  # sigmoid ≈ 0
        ]
        cal = compute_calibration_metrics(preds)
        assert cal["ece"] < 0.01
        assert cal["brier"] < 0.01


# ---------------------------------------------------------------------------
# Bootstrap CI tests
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    """Tests for paired bootstrap CI."""

    def test_positive_difference(self):
        """Consistently positive difference should have CI > 0."""
        treatment = [0.6, 0.65, 0.7, 0.62, 0.68]
        control = [0.5, 0.55, 0.6, 0.52, 0.58]
        ci = paired_bootstrap_ci(treatment, control)
        assert ci["delta_mean"] > 0
        assert ci["ci_low"] > 0

    def test_no_difference(self):
        """No difference should have CI spanning 0."""
        values = [0.5, 0.55, 0.6, 0.52, 0.58]
        ci = paired_bootstrap_ci(values, values)
        assert abs(ci["delta_mean"]) < 1e-6

    def test_negative_difference(self):
        """Consistently negative difference should have CI < 0."""
        treatment = [0.4, 0.45, 0.35, 0.42, 0.38]
        control = [0.5, 0.55, 0.6, 0.52, 0.58]
        ci = paired_bootstrap_ci(treatment, control)
        assert ci["delta_mean"] < 0
        assert ci["ci_high"] < 0


# ---------------------------------------------------------------------------
# Effect sizes tests
# ---------------------------------------------------------------------------

class TestEffectSizes:
    """Tests for effect size computation."""

    def test_effect_sizes_vs_baseline(self):
        """Effect sizes should be computed relative to A0."""
        all_results = {
            "chemformer": {
                "A0": [{"test_metrics": {"mrr": 0.3}}],
                "A6": [{"test_metrics": {"mrr": 0.5}}],
            }
        }
        effects = compute_effect_sizes(all_results)
        assert "chemformer" in effects
        assert "A6" in effects["chemformer"]
        assert effects["chemformer"]["A6"]["pp_diff"] == pytest.approx(20.0, abs=0.1)


# ---------------------------------------------------------------------------
# GO/NO-GO tests
# ---------------------------------------------------------------------------

class TestGoNoGo:
    """Tests for GO/NO-GO verdict computation."""

    def test_strong_go(self):
        """Strong GO when ≥2 backbones with positive CI and ≥1pp improvement."""
        all_results = {
            "chemformer": {
                "A0": [{"test_metrics": {"mrr": 0.3}} for _ in range(10)],
                "A6": [{"test_metrics": {"mrr": 0.5}} for _ in range(10)],
            },
            "gnn": {
                "A0": [{"test_metrics": {"mrr": 0.25}} for _ in range(10)],
                "A6": [{"test_metrics": {"mrr": 0.45}} for _ in range(10)],
            },
        }
        effects = compute_effect_sizes(all_results)
        go = compute_go_no_go(all_results, effects)
        assert go["status"] == "STRONG_GO"
        assert go["next_phase_allowed"] is True

    def test_weak_go(self):
        """Weak GO when only 1 backbone with positive CI."""
        all_results = {
            "chemformer": {
                "A0": [{"test_metrics": {"mrr": 0.3}} for _ in range(10)],
                "A6": [{"test_metrics": {"mrr": 0.5}} for _ in range(10)],
            },
            "gnn": {
                "A0": [{"test_metrics": {"mrr": 0.4}} for _ in range(10)],
                "A6": [{"test_metrics": {"mrr": 0.38}} for _ in range(10)],
            },
        }
        effects = compute_effect_sizes(all_results)
        go = compute_go_no_go(all_results, effects)
        assert go["status"] == "WEAK_GO"
        assert go["next_phase_allowed"] is True

    def test_no_go(self):
        """NO-GO when A6 ≤ A0."""
        all_results = {
            "chemformer": {
                "A0": [{"test_metrics": {"mrr": 0.5}} for _ in range(10)],
                "A6": [{"test_metrics": {"mrr": 0.3}} for _ in range(10)],
            },
            "gnn": {
                "A0": [{"test_metrics": {"mrr": 0.4}} for _ in range(10)],
                "A6": [{"test_metrics": {"mrr": 0.35}} for _ in range(10)],
            },
        }
        effects = compute_effect_sizes(all_results)
        go = compute_go_no_go(all_results, effects)
        assert go["status"] == "NO_GO"
        assert go["next_phase_allowed"] is False


# ---------------------------------------------------------------------------
# BackboneConfig tests
# ---------------------------------------------------------------------------

class TestBackboneConfig:
    """Tests for backbone configuration."""

    def test_chemformer_config(self):
        """Chemformer config should have required fields."""
        config = BackboneConfig(
            name="chemformer",
            checkpoint_path="/path/to/ckpt",
            lora_target_patterns=["backbone.encoder_layers.*.self_attn.out_proj"],
            lora_rank=8,
            lora_alpha=16.0,
            lora_dropout=0.0,
        )
        assert config.name == "chemformer"
        assert config.lora_rank == 8

    def test_gnn_config(self):
        """GNN config should have required fields."""
        config = BackboneConfig(
            name="gnn",
            gnn_hidden_dim=128,
            gnn_encoder_dim=256,
            gnn_num_layers=3,
            gnn_heads=4,
            gnn_dropout=0.1,
        )
        assert config.name == "gnn"
        assert config.gnn_hidden_dim == 128


# ---------------------------------------------------------------------------
# Summary CSV tests
# ---------------------------------------------------------------------------

class TestSummaryCSV:
    """Tests for summary CSV output."""

    def test_write_summary_csv(self, tmp_path):
        """Summary CSV should have correct headers and data."""
        all_results = {
            "chemformer": {
                "A0": [{
                    "arm_name": "positive_only",
                    "seed": 42,
                    "trainable_parameters": 180737,
                    "total_parameters": 19314937,
                    "wall_clock_seconds": 10.0,
                    "peak_memory_mb": 365.0,
                    "inference_latency_ms": 5.0,
                    "n_train_examples": 394,
                    "n_train_pos": 394,
                    "n_train_neg": 0,
                    "val_metrics": {"mrr": 0.2, "top1": 0.05, "top3": 0.2, "ndcg": 0.4, "auprc": 0.1, "ece": 0.3, "brier": 0.2},
                    "test_metrics": {"mrr": 0.25, "top1": 0.06, "top3": 0.22, "ndcg": 0.42, "auprc": 0.11, "ece": 0.31, "brier": 0.21},
                }],
            },
        }
        csv_path = tmp_path / "summary.csv"
        write_summary_csv(all_results, csv_path)
        assert csv_path.exists()
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 row
        assert "backbone" in lines[0]
        assert "chemformer" in lines[1]


# ---------------------------------------------------------------------------
# Integration test (manifest loading)
# ---------------------------------------------------------------------------

class TestManifestIntegration:
    """Integration tests with real manifest data."""

    @pytest.fixture
    def manifest_path(self):
        path = Path("data/p4/manifests/hte_feasibility_v1.json")
        if not path.exists():
            pytest.skip(f"Manifest not found: {path}")
        return path

    def test_load_manifest(self, manifest_path):
        """Manifest should load with correct structure."""
        splits = load_manifest_candidates(manifest_path)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert len(splits["train"]) > 0
        assert len(splits["val"]) > 0
        assert len(splits["test"]) > 0

    def test_manifest_has_all_negative_types(self, manifest_path):
        """Manifest should contain all expected negative types."""
        splits = load_manifest_candidates(manifest_path)
        neg_sources = set()
        for split in splits.values():
            for cand in split:
                if cand["negative_source"]:
                    neg_sources.add(cand["negative_source"])
        # Manifest may include additional types like external_beam
        expected_subset = {"random_mismatch", "random_corruption", "tanimoto_retrieval",
                           "template_perturbation", "unconstrained_edit", "rule_pc_cng"}
        assert expected_subset.issubset(neg_sources), \
            f"Missing negative types: {expected_subset - neg_sources}"

    def test_manifest_has_gold_per_group(self, manifest_path):
        """Each group should have exactly 1 gold candidate."""
        splits = load_manifest_candidates(manifest_path)
        from collections import Counter
        gold_counts = Counter()
        for split in splits.values():
            for cand in split:
                if cand["gold_candidate"]:
                    gold_counts[cand["group_id"]] += 1
        for gid, count in gold_counts.items():
            assert count == 1, f"Group {gid} has {count} gold candidates"


class TestManifestIntegrity:
    """Tests for aggregate_p4_g3.detect_arm_duplication (A2/A6 v1-manifest finding)."""

    def _write_manifest(self, tmp_path: Path, groups: list) -> Path:
        path = tmp_path / "manifest.json"
        with open(path, "w") as f:
            json.dump({"groups": groups}, f)
        return path

    def _group(self, gid: str, rc_smiles: str, pc_smiles: str) -> dict:
        return {
            "group_id": gid,
            "candidates": [
                {"candidate_id": f"{gid}_gold", "candidate_smiles": "CCO",
                 "candidate_source": "gold", "gold_candidate": True},
                {"candidate_id": f"{gid}_rc", "candidate_smiles": rc_smiles,
                 "candidate_source": "random_corruption", "gold_candidate": False},
                {"candidate_id": f"{gid}_pc", "candidate_smiles": pc_smiles,
                 "candidate_source": "rule_pc_cng", "gold_candidate": False},
            ],
        }

    def test_detects_full_duplication(self, tmp_path):
        from pc_cng.aggregate_p4_g3 import detect_arm_duplication
        groups = [self._group(f"g{i}", "CCC", "CCC") for i in range(3)]
        path = self._write_manifest(tmp_path, groups)
        dup = detect_arm_duplication(path)
        assert dup["duplicated"] is True
        assert dup["n_groups_checked"] == 3
        assert dup["n_groups_identical_smiles"] == 3

    def test_detects_no_duplication(self, tmp_path):
        from pc_cng.aggregate_p4_g3 import detect_arm_duplication
        groups = [self._group(f"g{i}", "CCC", "CCN") for i in range(3)]
        path = self._write_manifest(tmp_path, groups)
        dup = detect_arm_duplication(path)
        assert dup["duplicated"] is False
        assert dup["n_groups_identical_smiles"] == 0

    def test_partial_duplication_not_flagged(self, tmp_path):
        from pc_cng.aggregate_p4_g3 import detect_arm_duplication
        groups = [self._group("g0", "CCC", "CCC"), self._group("g1", "CCC", "CCN")]
        path = self._write_manifest(tmp_path, groups)
        dup = detect_arm_duplication(path)
        assert dup["duplicated"] is False
        assert dup["n_groups_identical_smiles"] == 1

    def test_real_v1_manifest_documents_known_duplication(self):
        """Frozen v1 manifest: rule_pc_cng SMILES duplicate random_corruption.

        This is a documented P4-G1 construction finding (not fixed in v1,
        which is immutable). The test locks the known state so the A6-vs-A2
        equivalence is explicit and any future v2 manifest must differ.
        """
        from pc_cng.aggregate_p4_g3 import detect_arm_duplication
        path = Path("data/p4/manifests/hte_feasibility_v1.json")
        if not path.exists():
            pytest.skip("v1 manifest not found")
        dup = detect_arm_duplication(path)
        assert dup["n_groups_checked"] == 500
        assert dup["duplicated"] is True


class TestSummaryMerging:
    """Tests for aggregate_p4_g3.load_summary_csv / merge_results."""

    def _row(self, backbone: str, arm: str, seed: int, mrr: float) -> dict:
        r = {
            "backbone": backbone, "arm_id": arm, "arm_name": arm,
            "seed": str(seed), "trainable_parameters": "100",
            "total_parameters": "1000", "wall_clock_seconds": "1.0",
            "peak_memory_mb": "10.0", "inference_latency_ms": "1.0",
            "n_train_examples": "10", "n_train_pos": "5", "n_train_neg": "5",
        }
        for k in ["mrr", "top1", "top3", "ndcg", "auprc", "ece", "brier"]:
            r[f"val_{k}"] = str(mrr)
            r[f"test_{k}"] = str(mrr)
        return r

    def test_load_summary_csv_roundtrip(self, tmp_path):
        import csv
        from pc_cng.aggregate_p4_g3 import load_summary_csv
        rows = [self._row("chemformer", "A0", 1, 0.3),
                self._row("chemformer", "A6", 1, 0.45),
                self._row("gnn", "A0", 1, 0.4)]
        path = tmp_path / "summary.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        res = load_summary_csv(path)
        assert set(res.keys()) == {"chemformer", "gnn"}
        assert len(res["chemformer"]) == 2
        assert res["chemformer"]["A6"][0]["test_metrics"]["mrr"] == pytest.approx(0.45)

    def test_merge_results_combines_backbones(self):
        from pc_cng.aggregate_p4_g3 import merge_results
        a = {"chemformer": {"A0": [{"seed": 1}]}}
        b = {"gnn": {"A0": [{"seed": 1}]}}
        merged = merge_results(a, b)
        assert set(merged.keys()) == {"chemformer", "gnn"}
        assert merged["chemformer"]["A0"][0]["seed"] == 1
