import pytest
import torch

from pc_cng.source_aware_policy import SourceAwareSoftmaxGate, source_names_from_indices


def test_gate_returns_budget_one_distribution_and_respects_mask():
    gate = SourceAwareSoftmaxGate(
        reaction_dim=4, source_feature_dim=3, n_sources=3, hidden_dim=8
    )
    reaction = torch.randn(2, 4)
    source = torch.randn(2, 3, 3)
    available = torch.tensor([[True, False, True], [False, True, True]])

    probs = gate(reaction, source, available)

    assert probs.shape == (2, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2))
    assert torch.all(probs[0, 1] == 0)
    assert torch.all(probs[1, 0] == 0)
    chosen = gate.select_source(reaction, source, available)
    assert chosen.shape == (2,)
    assert bool(available[torch.arange(2), chosen].all())


def test_gate_rejects_no_available_source():
    gate = SourceAwareSoftmaxGate(2, 2, 2, hidden_dim=4)
    with pytest.raises(ValueError, match="at least one available source"):
        gate(torch.randn(1, 2), torch.randn(1, 2, 2), torch.tensor([[False, False]]))


def test_source_name_conversion_is_checked():
    assert source_names_from_indices(torch.tensor([1, 0]), ["rule", "learned"]) == (
        "learned",
        "rule",
    )
    with pytest.raises(ValueError, match="out of bounds"):
        source_names_from_indices(torch.tensor([2]), ["rule", "learned"])
