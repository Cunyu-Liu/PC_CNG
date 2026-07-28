import numpy as np

from pc_cng.analyze_phase4_mechanism_continuous import (
    attach_within_group_hardness,
    fit_cluster_bootstrap_spline,
)


def test_within_group_hardness_is_tie_aware():
    rows = [
        {"dataset": "d", "scorer": "s", "group_id": "g", "score": 0.1},
        {"dataset": "d", "scorer": "s", "group_id": "g", "score": 0.5},
        {"dataset": "d", "scorer": "s", "group_id": "g", "score": 0.5},
        {"dataset": "d", "scorer": "s", "group_id": "g", "score": 0.9},
    ]
    attach_within_group_hardness(rows)
    assert rows[0]["negative_hardness"] == 0.125
    assert rows[1]["negative_hardness"] == rows[2]["negative_hardness"] == 0.5
    assert rows[3]["negative_hardness"] == 0.875


def test_constant_feature_is_fail_closed():
    rows = []
    for group in range(10):
        for index in range(5):
            rows.append(
                {
                    "group_id": str(group),
                    "label": 0,
                    "false_negative_risk": 0.5,
                    "negative_hardness": index / 5,
                }
            )
    result = fit_cluster_bootstrap_spline(
        rows, "false_negative_risk", n_bootstrap=50, seed=1
    )
    assert result["status"] == "UNAVAILABLE_INSUFFICIENT_VARIATION"


def test_monotonic_signal_is_estimable():
    rng = np.random.default_rng(4)
    rows = []
    for group in range(20):
        for index in range(5):
            x = (group * 5 + index) / 99
            rows.append(
                {
                    "group_id": str(group),
                    "label": 0,
                    "positive_similarity": x,
                    "negative_hardness": float(
                        np.clip(0.1 + 0.8 * x + rng.normal(0, 0.01), 0, 1)
                    ),
                }
            )
    result = fit_cluster_bootstrap_spline(
        rows, "positive_similarity", n_bootstrap=50, seed=2
    )
    assert result["status"] == "ESTIMATED"
    assert result["end_delta"] > 0.5
