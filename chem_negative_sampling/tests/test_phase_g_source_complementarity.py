import csv
import json

from pc_cng.analyze_phase_g_source_complementarity import analyze


SOURCES = [
    "random_mismatch",
    "shuffled_real",
    "similarity_retrieval",
    "template_perturbation",
    "rule_pc_cng",
    "learned_structured",
]


def test_development_analysis_stays_exploratory(tmp_path):
    cache_dir = tmp_path / "candidate_cache"
    policy_dir = tmp_path / "policy_maps"
    cache_dir.mkdir()
    policy_dir.mkdir()
    cache = []
    rows = []
    for index in range(12):
        candidates = {}
        selected = SOURCES[index % len(SOURCES)]
        row = {
            "group": f"g{index}",
            "reaction_family": f"f{index % 2}",
            "selected_source": selected,
        }
        for source_index, source in enumerate(SOURCES):
            candidates[source] = {
                "false_negative_risk": 0.1 + 0.01 * source_index,
                "positive_similarity": 0.2 + 0.02 * source_index,
                "boundary_closeness": 0.3 + 0.01 * source_index,
            }
            row[f"prob::{source}"] = 0.6 if source == selected else 0.08
            row[f"oof_hardness::{source}"] = 0.4 + 0.01 * source_index
            row[f"reward::{source}"] = 0.3 + 0.01 * source_index
        rows.append(row)
        cache.append(
            {
                "group": f"g{index}",
                "candidates": candidates,
            }
        )
    cache_path = cache_dir / "random.json"
    cache_path.write_text(json.dumps(cache))
    policy_path = policy_dir / "random__mlp.csv"
    with open(policy_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "scenario_audits": {
            "random": {"candidate_cache": str(cache_path)}
        },
        "results": [
            {
                "scenario": "random",
                "backbone": "mlp",
                "policy_map": str(policy_path),
                "arms": {
                    "learned_source_gate": {"source_macro_auprc": 0.7},
                    "gate_no_learned_source": {"source_macro_auprc": 0.68},
                    "gate_no_shuffled_real": {"source_macro_auprc": 0.69},
                },
            }
        ],
    }
    (tmp_path / "phase_d_results.json").write_text(json.dumps(result))
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"source_names": SOURCES, "mode": "development"})
    )

    report = analyze(tmp_path)
    assert report["status"] == "EXPLORATORY_SOURCE_COMPLEMENTARITY_ONLY"
    assert report["mechanism_exit_met"] is False
    cell = report["cells"][0]
    assert cell["n_parents"] == 12
    assert set(cell["selection_counts"]) == set(SOURCES)
    assert abs(
        cell["leave_one_source_out"]["learned_structured"][
            "full_gate_minus_ablation"
        ]
        - 0.02
    ) < 1e-12
