import json

import numpy as np
import pytest
import torch

import pc_cng.run_phase_d_source_policy as phase_d


def _fake_examples(n=6):
    examples = []
    for index in range(n):
        candidates = {}
        for source_index, source in enumerate(phase_d.SOURCE_NAMES):
            candidates[source] = {
                "negative_reaction": f"C>>N{index}{source_index}",
                "negative_product": f"N{index}{source_index}",
                "negative_fp": np.full(8, source_index, dtype=np.float32),
                "false_negative_risk": 0.1,
                "positive_similarity": 0.5,
                "source_features": [0.0] * phase_d.SOURCE_FEATURE_DIM,
            }
        examples.append(
            {
                "reaction_smiles": f"C>>C{index}",
                "true_product": f"C{index}",
                "family": "test",
                "group": f"g{index}",
                "reaction_features": np.zeros(
                    phase_d.REACTION_FEATURE_DIM, dtype=np.float32
                ),
                "candidates": candidates,
            }
        )
    return examples


def test_training_dataset_enforces_one_negative_per_parent(monkeypatch):
    monkeypatch.setattr(
        phase_d,
        "_normalise_reaction_fp",
        lambda _: np.zeros(8, dtype=np.float32),
    )
    examples = _fake_examples()
    selections = [
        phase_d.SOURCE_NAMES[index % len(phase_d.SOURCE_NAMES)]
        for index in range(len(examples))
    ]
    X, y, records, audit = phase_d._build_training_dataset(
        examples,
        selections,
        arm="test",
        seed=7,
    )
    assert X.shape == (2 * len(examples), 8)
    assert len(y) == len(records) == 2 * len(examples)
    assert audit["n_positive"] == audit["n_negative"] == len(examples)
    assert audit["budget_exact"] is True
    assert sum(audit["source_counts"].values()) == len(examples)


def test_policy_ablations_mask_frozen_feature_groups():
    n = 3
    reaction = torch.ones(n, phase_d.REACTION_FEATURE_DIM)
    source = torch.ones(n, len(phase_d.SOURCE_NAMES), phase_d.SOURCE_FEATURE_DIM)
    available = torch.ones(n, len(phase_d.SOURCE_NAMES), dtype=torch.bool)
    hardness = np.full((n, len(phase_d.SOURCE_NAMES)), 0.7, dtype=np.float32)

    r, s, a, rewards = phase_d._policy_variant_inputs(
        reaction, source, available, hardness, "gate_no_fnr"
    )
    assert torch.all(
        s[:, :, phase_d._source_feature_index("false_negative_risk")] == 0
    )
    assert torch.all(
        s[:, :, phase_d._source_feature_index("fnr_uncertainty")] == 0
    )
    assert torch.allclose(rewards, torch.full_like(rewards, 0.7))
    assert torch.equal(r, reaction)
    assert torch.equal(a, available)

    _, _, no_learned, _ = phase_d._policy_variant_inputs(
        reaction,
        source,
        available,
        hardness,
        "gate_no_learned_source",
    )
    assert not bool(
        no_learned[:, phase_d.SOURCE_NAMES.index(phase_d.SOURCE_LEARNED)].any()
    )


def test_formal_pool_contract_is_fail_closed(tmp_path):
    with pytest.raises(RuntimeError, match="sealed_test_manifest"):
        phase_d._formal_pool_contract(tmp_path)
    manifest = {
        "status": "SEALED_UNUSED_FOR_METHOD_DESIGN",
        "labels_unseen_before_model_freeze": True,
        "model_and_analysis_frozen": True,
    }
    (tmp_path / "sealed_test_manifest.json").write_text(json.dumps(manifest))
    assert phase_d._formal_pool_contract(tmp_path) == manifest
    manifest["labels_unseen_before_model_freeze"] = False
    (tmp_path / "sealed_test_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="labels_unseen_before_model_freeze"):
        phase_d._formal_pool_contract(tmp_path)


def test_source_macro_chance_uses_slice_prevalence():
    records = [
        {"is_positive": True, "source": "real"},
        {"is_positive": True, "source": "real"},
        {"is_positive": False, "source": "a"},
        {"is_positive": False, "source": "a"},
        {"is_positive": False, "source": "b"},
    ]
    expected = ((2 / 4) + (2 / 3)) / 2
    assert phase_d._source_macro_chance(records) == pytest.approx(expected)
