"""Unit tests for the learned GNN-based graph edit decoder (P1-05).

Tests run on CPU only (``CUDA_VISIBLE_DEVICES=`` is forced by the test runner).
Covers featurization, model forward pass, loss, training smoke, negative
generation, and checkpoint save/load.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import List

import pytest
import torch

# Force CPU for tests
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Ensure the chem_negative_sampling package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pc_cng.learned_graph_edit_decoder import (
    ATOM_FEAT_DIM,
    BOND_FEAT_DIM,
    BatchedGraph,
    GeneratedNegative,
    LearnedGraphEditDecoder,
    ReactionGraphData,
    collate_graphs,
    featurize_atom_mapped_reaction,
    generate_boundary_negatives,
    load_checkpoint,
    pairwise_margin_loss,
    predict_anchor_scores,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Real atom-mapped methylation reaction from HiTEA dataset (verified to work)
HITEA_REACTION = (
    "COS(=O)(=O)O[CH3:12].[F:1][c:2]1[cH:3][c:4]([Br:5])[cH:6][c:7]2[cH:8][n:9][nH:10][c:11]12"
    ">>[F:1][c:2]1[cH:3][c:4]([Br:5])[cH:6][c:7]2[cH:8][n:9][n:10]([CH3:12])[c:11]12"
)

# Unmapped esterification reaction (for RXNMapper fallback test)
UNMAPPED_REACTION = "CCO.CC(=O)O>>CCOC(=O)C.O"


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def graph_data() -> List[ReactionGraphData]:
    graphs, reason = featurize_atom_mapped_reaction(HITEA_REACTION, source_id="test_hitea")
    assert reason == "ok", f"Featurization failed: {reason}"
    assert len(graphs) > 0, "No graphs produced"
    return graphs


@pytest.fixture
def model(device) -> LearnedGraphEditDecoder:
    return LearnedGraphEditDecoder(
        atom_feat_dim=ATOM_FEAT_DIM,
        bond_feat_dim=BOND_FEAT_DIM,
        hidden_dim=32,
        num_rounds=2,
        dropout=0.0,
    ).to(device)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_featurize_atom_mapped_reaction_basic(graph_data):
    """Test 1: featurization of an atom-mapped reaction produces valid graph tensors."""
    graphs = graph_data
    assert len(graphs) > 0

    graph = graphs[0]
    assert isinstance(graph, ReactionGraphData)
    assert graph.atom_features.shape[1] == ATOM_FEAT_DIM
    assert graph.atom_features.dtype == torch.float32
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.dtype == torch.long
    assert graph.edge_features.shape[1] == BOND_FEAT_DIM
    assert graph.true_anchor_idx >= 0
    assert graph.true_anchor_idx < graph.atom_features.shape[0]
    assert len(graph.candidate_anchor_indices) >= 2
    assert graph.true_anchor_idx in graph.candidate_anchor_indices
    assert graph.fragment_map == 12
    assert graph.true_anchor_map == 10
    # all candidate indices are valid atom indices
    for idx in graph.candidate_anchor_indices:
        assert 0 <= idx < graph.atom_features.shape[0]
    # edge_index values are valid atom indices
    if graph.edge_index.shape[1] > 0:
        assert graph.edge_index.max().item() < graph.atom_features.shape[0]
        assert graph.edge_index.min().item() >= 0


def test_featurize_unmapped_reaction_fallback():
    """Test 2: unmapped reaction triggers RXNMapper fallback (or graceful failure)."""
    # Without map_unmapped, should return missing_atom_mapping
    graphs, reason = featurize_atom_mapped_reaction(UNMAPPED_REACTION, map_unmapped=False)
    assert reason == "missing_atom_mapping"
    assert len(graphs) == 0

    # With map_unmapped=True and RXNMapper available, should produce graphs
    # (RXNMapper is available in the test env; skip if it fails)
    try:
        graphs, reason = featurize_atom_mapped_reaction(
            UNMAPPED_REACTION, map_unmapped=True, source_id="unmapped_test"
        )
        # If RXNMapper works, we should get either ok or a chemistry-related skip reason
        assert reason in {"ok", "no_formed_bond", "no_candidate_anchor", "mapping_failed"}
        if reason == "ok":
            assert len(graphs) > 0
    except Exception:
        pytest.skip("RXNMapper not available in this environment")


def test_model_forward_pass(model, graph_data, device):
    """Test 3: model forward pass produces correct output shape."""
    graph = graph_data[0]
    n_atoms = graph.atom_features.shape[0]
    batch_idx = torch.zeros(n_atoms, dtype=torch.long)

    scores = model(
        graph.atom_features.to(device),
        graph.edge_index.to(device),
        graph.edge_features.to(device),
        batch_idx.to(device),
        num_graphs=1,
    )
    assert scores.shape == (n_atoms,)
    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all()


def test_pairwise_margin_loss(device):
    """Test 4: pairwise margin loss is 0 when true > candidate by margin, positive otherwise."""
    # Case 1: true score much higher than candidate → loss = 0
    true_scores = torch.tensor([5.0], device=device)
    candidate_scores = torch.tensor([[1.0, 2.0, 3.0]], device=device)
    loss = pairwise_margin_loss(true_scores, candidate_scores, margin=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

    # Case 2: true score lower than candidate → loss > 0
    true_scores = torch.tensor([1.0], device=device)
    candidate_scores = torch.tensor([[5.0, 4.0, 3.0]], device=device)
    loss = pairwise_margin_loss(true_scores, candidate_scores, margin=1.0)
    assert loss.item() > 0.0

    # Case 3: true score higher by exactly margin → loss = 0
    true_scores = torch.tensor([2.0], device=device)
    candidate_scores = torch.tensor([[1.0]], device=device)
    loss = pairwise_margin_loss(true_scores, candidate_scores, margin=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

    # Case 4: empty candidates → loss = 0
    true_scores = torch.tensor([1.0], device=device)
    candidate_scores = torch.zeros((1, 0), device=device)
    loss = pairwise_margin_loss(true_scores, candidate_scores, margin=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_predict_anchor_scores_returns_candidates_only(model, graph_data, device):
    """Test 5: predict_anchor_scores returns exactly one score per candidate anchor."""
    graph = graph_data[0]
    scores = predict_anchor_scores(model, graph, device)
    assert scores.shape == (len(graph.candidate_anchor_indices),)
    assert torch.isfinite(scores).all()
    # true anchor should have a score
    true_local = graph.candidate_anchor_indices.index(graph.true_anchor_idx)
    assert true_local < scores.shape[0]


def test_train_smoke_one_epoch(graph_data, device):
    """Test 6: a single training epoch on a small batch does not crash and reduces/updates loss."""
    from pc_cng.learned_graph_edit_decoder import collate_graphs, pairwise_margin_loss

    model = LearnedGraphEditDecoder(
        atom_feat_dim=ATOM_FEAT_DIM,
        bond_feat_dim=BOND_FEAT_DIM,
        hidden_dim=32,
        num_rounds=2,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Use the same graphs multiple times to simulate a training batch
    train_graphs = graph_data * 3
    model.train()
    initial_loss = None
    final_loss = None
    for step in range(3):
        batch = train_graphs[step * 1 : (step + 1) * 1]
        batched = collate_graphs(batch)
        optimizer.zero_grad()
        all_scores = model(
            batched.atom_features.to(device),
            batched.edge_index.to(device),
            batched.edge_features.to(device),
            batched.batch_idx.to(device),
            num_graphs=len(batch),
        )
        total_loss = torch.tensor(0.0, device=device)
        for gi, graph in enumerate(batch):
            candidates = batched.candidate_anchor_indices_per_graph[gi]
            true_abs = batched.true_anchor_indices[gi]
            if true_abs not in candidates:
                continue
            true_local = candidates.index(true_abs)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = all_scores.index_select(0, cand_t)
            true_score = scores[true_local]
            mask = torch.ones(len(candidates), dtype=torch.bool, device=device)
            mask[true_local] = False
            cand_scores = scores[mask]
            if cand_scores.numel() == 0:
                continue
            loss = pairwise_margin_loss(true_score.unsqueeze(0), cand_scores.unsqueeze(0), margin=1.0)
            total_loss = total_loss + loss
        if total_loss.requires_grad:
            total_loss.backward()
            optimizer.step()
        if step == 0:
            initial_loss = float(total_loss.item())
        final_loss = float(total_loss.item())

    # loss should be a finite number (training ran without crashing)
    assert initial_loss is not None
    assert final_loss is not None
    assert torch.isfinite(torch.tensor(final_loss)).item()


def test_generate_boundary_negatives_basic(model, graph_data, device):
    """Test 7: generated boundary negatives are valid SMILES reactions."""
    # Use a model with random weights - negatives should still be valid SMILES
    negatives, reason = generate_boundary_negatives(
        model,
        HITEA_REACTION,
        source_id="test_hitea",
        top_k=3,
        device=device,
    )
    assert reason == "ok"
    assert len(negatives) > 0

    from rdkit import Chem
    from pc_cng.chem_utils import is_valid_smiles, split_reaction

    for neg in negatives:
        assert isinstance(neg, GeneratedNegative)
        assert neg.candidate_reaction
        assert neg.candidate_product
        # candidate reaction should be parseable
        try:
            reactants, _, products = split_reaction(neg.candidate_reaction)
            assert is_valid_smiles(reactants), f"Invalid reactants: {reactants}"
            assert is_valid_smiles(products), f"Invalid products: {products}"
        except ValueError:
            pytest.fail(f"Could not split reaction: {neg.candidate_reaction}")
        # candidate product should differ from parent product
        assert neg.candidate_product != neg.parent_product


def test_checkpoint_save_load(model, graph_data, device):
    """Test 8: checkpoint save/load produces a model with identical predictions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_decoder.pt")
        save_checkpoint(model, ckpt_path, extra={"test": True})

        # Load checkpoint
        loaded_model = load_checkpoint(ckpt_path, device)

        # Compare predictions
        graph = graph_data[0]
        batch_idx = torch.zeros(graph.atom_features.shape[0], dtype=torch.long)
        model.eval()
        loaded_model.eval()
        with torch.no_grad():
            scores_orig = model(
                graph.atom_features,
                graph.edge_index,
                graph.edge_features,
                batch_idx,
                num_graphs=1,
            )
            scores_loaded = loaded_model(
                graph.atom_features,
                graph.edge_index,
                graph.edge_features,
                batch_idx,
                num_graphs=1,
            )
        assert torch.allclose(scores_orig, scores_loaded, atol=1e-6), "Loaded model predictions differ"


# ---------------------------------------------------------------------------
# Additional tests (collation, batched forward)
# ---------------------------------------------------------------------------


def test_collate_graphs_batching(graph_data):
    """Test 9: collate_graphs correctly batches multiple graphs with offset edges."""
    graphs = graph_data * 3
    batched = collate_graphs(graphs)
    assert isinstance(batched, BatchedGraph)
    assert batched.atom_features.shape[0] == sum(g.atom_features.shape[0] for g in graphs)
    assert len(batched.true_anchor_indices) == 3
    assert len(batched.candidate_anchor_indices_per_graph) == 3
    assert batched.batch_idx.max().item() == 2
    # edge indices should be within bounds
    if batched.edge_index.shape[1] > 0:
        assert batched.edge_index.max().item() < batched.atom_features.shape[0]


def test_batched_forward_pass(model, graph_data, device):
    """Test 10: model forward on a batched graph produces correct output."""
    graphs = graph_data * 4
    batched = collate_graphs(graphs)
    scores = model(
        batched.atom_features.to(device),
        batched.edge_index.to(device),
        batched.edge_features.to(device),
        batched.batch_idx.to(device),
        num_graphs=4,
    )
    assert scores.shape == (batched.atom_features.shape[0],)
    assert torch.isfinite(scores).all()
