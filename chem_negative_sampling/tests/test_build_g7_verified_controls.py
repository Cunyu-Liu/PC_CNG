import pandas as pd
import pytest

from pc_cng.build_g7_verified_controls import build_controls


def _frame(n=40):
    rows = []
    for index in range(n):
        is_positive = index % 2 == 0
        rows.append(
            {
                "record_id": f"r{index}",
                "source_publication": "independent HTE publication",
                "measured_yield": 90.0 if is_positive else 0.0,
                "missing_measurement": False,
                "reported_zero": not is_positive,
                "reaction_family": f"family_{index % 4}",
                "reaction_smiles": f"C{index}>cat.solv>P{index}",
                "experimental_group": f"g{index}",
            }
        )
    return pd.DataFrame(rows)


def test_builds_balanced_real_controls():
    controls = build_controls(_frame(), n_per_type=10, seed=7)
    assert len(controls) == 20
    assert sum(row["control_type"] == "positive_control" for row in controls) == 10
    assert (
        sum(row["control_type"] == "obvious_negative_control" for row in controls)
        == 10
    )
    assert all(
        row["verification_status"] == "INDEPENDENTLY_VERIFIED"
        for row in controls
    )
    assert len({row["reaction_smiles"] for row in controls}) == 20


def test_negative_control_requires_reported_zero():
    frame = _frame()
    frame.loc[frame["measured_yield"] == 0, "reported_zero"] = False
    with pytest.raises(RuntimeError, match="insufficient eligible controls"):
        build_controls(frame, n_per_type=10)
