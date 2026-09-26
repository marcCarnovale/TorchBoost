from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.thermal_exposure import (
    exposure_distance,
    exposure_matched,
    thermal_exposure,
)


def history():
    return [
        {"step": step, "nodes": {"a": {"capacity": 1.0, "temperature": temp}},
         "thermal_after": temp - 1.0, "energy_error": 0.0}
        for step, temp in enumerate([1.1, 1.2])
    ]


def test_thermal_dose_has_declared_discrete_integral():
    cfg = SimpleNamespace(dt=0.5, ambient_temperature=1.0, thaw_temperature=1.2)
    out = thermal_exposure(history(), cfg)
    assert out["mean_thaw_exposure"] == pytest.approx(0.75)
    assert out["energy_exposure"] == pytest.approx(0.15)
    assert out["peak_thaw_units"] == pytest.approx(1.0)
    assert out["duration"] == pytest.approx(1.0)


def test_normalized_exposure_is_temperature_unit_invariant():
    cfg = SimpleNamespace(dt=0.5, ambient_temperature=1.0, thaw_temperature=1.2)
    base = thermal_exposure(history(), cfg)
    changed = deepcopy(history())
    for row in changed:
        row["nodes"]["a"]["temperature"] *= 10.0
        row["thermal_after"] *= 10.0
    scaled = thermal_exposure(changed, SimpleNamespace(dt=0.5, ambient_temperature=10.0, thaw_temperature=12.0))
    assert scaled["mean_thaw_exposure"] == pytest.approx(base["mean_thaw_exposure"])
    assert scaled["peak_thaw_units"] == pytest.approx(base["peak_thaw_units"])


def test_equal_peak_is_not_equal_dose():
    target = {"peak_thaw_units": 2.0, "mean_thaw_exposure": 5.0}
    candidate = {"peak_thaw_units": 2.0, "mean_thaw_exposure": 10.0}
    assert not exposure_matched(candidate, target)
    assert exposure_distance(candidate, target) > 0
    assert exposure_matched(target, target)
    assert exposure_distance(target, target) == 0


def test_replayed_history_is_rejected():
    rows = history()
    rows.append(deepcopy(rows[-1]))
    with pytest.raises(ValueError, match="increasing"):
        thermal_exposure(rows, SimpleNamespace(dt=0.5, ambient_temperature=1.0, thaw_temperature=1.2))
