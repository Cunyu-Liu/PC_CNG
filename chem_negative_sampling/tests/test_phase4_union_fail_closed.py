from pathlib import Path

import pytest

from pc_cng.run_phase4_union_arm import require_learned_model


def test_union_fails_closed_when_learned_model_is_missing():
    with pytest.raises(SystemExit, match="refusing to degrade"):
        require_learned_model(
            None,
            "incompatible action schema",
            Path("/tmp/old_checkpoint.pt"),
        )


def test_union_accepts_loaded_learned_model():
    require_learned_model(object(), None, Path("/tmp/model.pt"))
