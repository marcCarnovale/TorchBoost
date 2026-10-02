from dataclasses import asdict

import numpy as np
import pytest

from experiments.deep_local_physics import dataset, run, study_config
from torchboost.adaptive.data import Preprocessor


def test_local_change_leaves_other_regions_identical_and_roles_disjoint():
    data = dataset(211, 1000, 3)
    fingerprints = set()
    for x, a, b, changed in data.values():
        np.testing.assert_array_equal(a[~changed], b[~changed])
        assert np.any(a[changed] != b[changed])
        fingerprints.add(x[0].tobytes())
    assert len(fingerprints) == 5
    assert not np.array_equal(data["audit"][0], dataset(223, 1000, 3)["audit"][0])


def test_all_source_arms_share_generic_regularization_and_plasticity():
    direct = study_config(211, 6, 64, 64, "direct", 0.1, 0.02)
    for kind in ("capacitor", "rlc"):
        cfg = study_config(211, 6, 64, 64, kind, 0.1, 0.02)
        assert asdict(direct.regularizers) == asdict(cfg.regularizers)
        assert asdict(direct.native.plasticity) == asdict(cfg.native.plasticity)
        a, b = asdict(direct.native.physics), asdict(cfg.native.physics)
        a.pop("mode")
        b.pop("mode")
        assert a == b
    span = direct.native.physics.thaw_temperature - direct.native.physics.ambient_temperature
    assert direct.native.plasticity.thermal_softening * span == pytest.approx(1.0)
    assert direct.native.plasticity.yield_threshold / direct.native.plasticity.stiffness == pytest.approx(0.02)


def test_native_development_smoke_does_not_prepare_or_score_audit(tmp_path, monkeypatch):
    original = Preprocessor.split

    def guarded(self, x, y, weight=None):
        assert len(x) != 2400, "development tried to prepare audit predictions"
        return original(self, x, y, weight)

    monkeypatch.setattr(Preprocessor, "split", guarded)
    result = run(211, 3, 1000, 8, 16, 0.5, 0.1, 0.02, "stationary", tmp_path / "smoke.json")
    assert result["status"] == "completed"
    assert result["audit"] is None
    assert len(result["arms"]) == 5
    # Zero drive during declared source warmup must reproduce plasticity-only.
    scores = [result["arms"][name]["ranking_trajectory_nll"] for name in ("plastic", "direct", "capacitor", "rlc")]
    assert np.ptp(scores) == 0.0


def test_confirmation_cannot_run_without_frozen_protocol(tmp_path):
    with pytest.raises(ValueError, match="protocol"):
        run(223, 3, 1000, 8, 16, 0.5, 0.1, 0.02, "stationary", tmp_path / "bad.json", True)
    assert not (tmp_path / "bad.json").exists()


def test_confirmation_binds_source_settings_and_scores_only_winner_and_reference(tmp_path):
    development = tmp_path / "development.json"
    args = (3, 300, 2, 2, 0.5, 0.1, 0.02, "recurring")
    run(211, *args, development, arms=("none", "plastic"))
    result = run(223, *args, tmp_path / "confirmation.json", True,
                 arms=("none", "plastic"), protocol=development)
    assert result["status"] == "completed"
    assert set(result["audit"]) == {"none", result["ranking_winner"]}
    assert len(result["audit"]["none"]["trajectory"]) == 3
    assert result["development_seed"] == 211
    with pytest.raises(ValueError, match="fresh seed"):
        run(211, *args, tmp_path / "reused.json", True,
            arms=("none", "plastic"), protocol=development)
