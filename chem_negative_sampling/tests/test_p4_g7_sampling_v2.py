import csv
import json

import pytest

from pc_cng.p4_g7_sampling_v2 import (
    SCORING_DIMENSIONS,
    SOURCE_NAMES,
    build_pilot_items,
    load_candidate_caches,
    load_policy_maps,
    load_verified_controls,
    write_pilot_package,
)


def _reaction(index, product):
    return f"C{index}>cat.solv>{product}"


def _write_inputs(tmp_path, n_entries=70, n_controls=10):
    tmp_path.mkdir(parents=True, exist_ok=True)
    entries = []
    for index in range(n_entries):
        candidates = {}
        for source_index, source in enumerate(SOURCE_NAMES):
            candidates[source] = {
                "source": source,
                "negative_reaction": _reaction(
                    index,
                    f"N{index}{source_index}",
                ),
                "negative_product": f"N{index}{source_index}",
                "false_negative_risk": 0.1 + source_index * 0.01,
                "positive_similarity": 0.5,
            }
        entries.append(
            {
                "group": f"g{index}",
                "family": "test_family",
                "reaction_smiles": _reaction(index, f"P{index}"),
                "true_product": f"P{index}",
                "candidates": candidates,
            }
        )
    cache = tmp_path / "random.json"
    cache.write_text(json.dumps(entries))

    policy = tmp_path / "random__mlp.csv"
    with open(policy, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group", "selected_source"],
        )
        writer.writeheader()
        for index in range(n_entries):
            writer.writerow(
                {
                    "group": f"g{index}",
                    "selected_source": SOURCE_NAMES[index % len(SOURCE_NAMES)],
                }
            )

    controls = tmp_path / "controls.csv"
    fields = [
        "control_id",
        "control_type",
        "reaction_smiles",
        "experimental_provenance",
        "verification_status",
        "reaction_family",
    ]
    with open(controls, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for control_type in ("positive_control", "obvious_negative_control"):
            for index in range(n_controls):
                writer.writerow(
                    {
                        "control_id": f"{control_type}-{index}",
                        "control_type": control_type,
                        "reaction_smiles": _reaction(
                            f"c{index}",
                            f"{control_type}{index}",
                        ),
                        "experimental_provenance": "independent experiment",
                        "verification_status": "INDEPENDENTLY_VERIFIED",
                        "reaction_family": "control_family",
                    }
                )
    return cache, policy, controls


def test_v2_builds_exact_balanced_80_item_three_reviewer_package(tmp_path):
    cache, policy_path, controls_path = _write_inputs(tmp_path)
    entries, _ = load_candidate_caches([cache])
    policy, _ = load_policy_maps([policy_path])
    controls, _ = load_verified_controls(controls_path)
    items, summary = build_pilot_items(
        entries,
        policy,
        controls,
        per_stratum=10,
        seed=7,
    )
    assert len(items) == 80
    assert set(summary["stratum_counts"].values()) == {10}
    assert summary["unique_parent_or_control_count"] == 80

    output = tmp_path / "pilot"
    manifest = write_pilot_package(
        items,
        output_dir=output,
        n_reviewers=3,
        seed=7,
        provenance={"test": True},
    )
    assert manifest["n_reviewers"] == 3
    assert len(manifest["artifacts"]["forms"]) == 3
    form_rows = list(
        csv.DictReader(open(output / "blinded_forms" / "reviewer_1.csv"))
    )
    assert len(form_rows) == 80
    assert "source" not in form_rows[0]
    assert "false_negative_risk" not in form_rows[0]
    assert "expert_false_negative_risk_assessment" in form_rows[0]
    assert set(SCORING_DIMENSIONS).issubset(form_rows[0])


def test_proxy_or_unverified_controls_fail_closed(tmp_path):
    cache, policy_path, controls_path = _write_inputs(tmp_path)
    rows = list(csv.DictReader(open(controls_path)))
    rows[0]["verification_status"] = "PROXY"
    with open(controls_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="not independently verified"):
        load_verified_controls(controls_path)

    entries, _ = load_candidate_caches([cache])
    policy, _ = load_policy_maps([policy_path])
    verified_path = _write_inputs(tmp_path / "second")[2]
    controls, _ = load_verified_controls(verified_path)
    with pytest.raises(RuntimeError, match="requires 10"):
        build_pilot_items(
            entries,
            policy,
            controls[:5],
            per_stratum=10,
        )


def test_blank_reaction_context_and_two_reviewers_are_rejected(tmp_path):
    cache, policy_path, controls_path = _write_inputs(tmp_path)
    payload = json.loads(cache.read_text())
    payload[0]["reaction_smiles"] = ""
    cache.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="complete reaction context"):
        load_candidate_caches([cache])

    cache, policy_path, controls_path = _write_inputs(tmp_path / "valid")
    entries, _ = load_candidate_caches([cache])
    policy, _ = load_policy_maps([policy_path])
    controls, _ = load_verified_controls(controls_path)
    items, _ = build_pilot_items(entries, policy, controls)
    with pytest.raises(ValueError, match="at least 3 reviewers"):
        write_pilot_package(
            items,
            output_dir=tmp_path / "pilot",
            n_reviewers=2,
            seed=7,
            provenance={},
        )
