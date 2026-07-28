import pytest
import torch

from pc_cng.source_aware_policy import (
    GateTrainingConfig,
    SourceAwareSoftmaxGate,
    masked_reward_targets,
    source_names_from_indices,
    train_source_gate,
)


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


def test_masked_reward_targets_exclude_unavailable_sources():
    rewards = torch.tensor([[0.1, 10.0, 0.8], [0.7, 0.2, -3.0]])
    available = torch.tensor([[True, False, True], [True, True, False]])
    targets = masked_reward_targets(rewards, available, temperature=0.2)
    assert torch.allclose(targets.sum(dim=1), torch.ones(2))
    assert targets[0, 1] == 0
    assert targets[1, 2] == 0
    assert targets[0, 2] > targets[0, 0]
    assert targets[1, 0] > targets[1, 1]


def test_train_source_gate_learns_context_dependent_rewards():
    torch.manual_seed(7)
    n = 80
    reaction = torch.zeros(n, 2)
    reaction[: n // 2, 0] = 1.0
    reaction[n // 2 :, 1] = 1.0
    source = torch.zeros(n, 2, 2)
    source[:, 0, 0] = 1.0
    source[:, 1, 1] = 1.0
    rewards = torch.zeros(n, 2)
    rewards[: n // 2, 0] = 1.0
    rewards[n // 2 :, 1] = 1.0
    available = torch.ones(n, 2, dtype=torch.bool)
    gate = SourceAwareSoftmaxGate(
        reaction_dim=2,
        source_feature_dim=2,
        n_sources=2,
        hidden_dim=16,
    )
    audit = train_source_gate(
        gate,
        reaction,
        source,
        rewards,
        available,
        GateTrainingConfig(
            epochs=120,
            learning_rate=1e-2,
            target_temperature=0.1,
            entropy_weight=0.0,
            source_dropout=0.1,
            seed=7,
        ),
    )
    selected = gate.select_source(reaction, source, available)
    expected = torch.cat(
        [torch.zeros(n // 2), torch.ones(n // 2)]
    ).long()
    assert (selected == expected).float().mean() > 0.95
    assert audit["target_argmax_agreement"] > 0.95
