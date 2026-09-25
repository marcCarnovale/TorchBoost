import math
from copy import deepcopy

import numpy as np
import pytest

from experiments.direct_feedback_control import DirectFeedbackController
from torchboost.adaptive.config import PhysicsConfig
from torchboost.adaptive.physics import PhysicalController


def make_controller(count=1, max_temperature=5.0):
    cfg = PhysicsConfig(
        mode="cooling", topology_normalization=True,
        initial_temperature=1.0, ambient_temperature=1.0,
        thaw_temperature=1.2, max_temperature=max_temperature,
        total_heat_capacity=0.2, cooling_time=20.0,
        charge_gain=1.0, max_injection=0.03, smoothing=0.8, dt=0.2,
    )
    controller = DirectFeedbackController(cfg)
    controller.synchronize({f"0:{i}": 0 for i in range(count)})
    return controller


def test_direct_replay_is_an_exact_noop_and_returns_detached_record():
    p = make_controller()
    p.advance(1.0, {}, 0)
    first = p.advance(2.0, {}, 1)
    before = p.state_dict()
    repeated = p.advance(2.0, {}, 1)
    assert repeated == first
    assert p.state_dict() == before
    repeated["nodes"]["0:0"]["temperature"] = -999
    assert p.state_dict() == before


@pytest.mark.parametrize("loss,step", [(3.0, 1), (3.0, 0), (float("nan"), 2), (float("inf"), 2)])
def test_invalid_steps_and_losses_do_not_heat(loss, step):
    p = make_controller()
    p.advance(1.0, {}, 0)
    p.advance(2.0, {}, 1)
    before = p.state_dict()
    with pytest.raises(ValueError):
        p.advance(loss, {}, step)
    assert p.state_dict() == before


def test_unsynchronized_controller_does_not_record_external_heat():
    p = make_controller()
    p.nodes.clear()
    p.reference = 1.0
    with pytest.raises(ValueError):
        p.advance(2.0, {}, 0)
    assert p.direct_heat_total == 0.0


@pytest.mark.parametrize("count", [1, 7, 63])
@pytest.mark.parametrize("maximum", [1.01, 5.0])
def test_external_heat_and_venting_close_whole_step_energy(count, maximum):
    p = make_controller(count, maximum)
    p.advance(1.0, {}, 0)
    before = p.thermal_energy() + p.electrical_energy()
    r = p.advance(2.0, {}, 1)
    assert r["thermal_before"] == pytest.approx(0.0)
    assert r["external_heat"] == pytest.approx(0.03)
    assert sum(n["direct_heat"] for n in r["nodes"].values()) == pytest.approx(0.03)
    expected = before + r["external_heat"] - r["cooling_energy"] - r["vented_energy"]
    assert p.thermal_energy() + p.electrical_energy() == pytest.approx(expected, abs=1e-13)
    assert abs(r["energy_error"]) < 1e-13
    expected_temperature = min(maximum, 1.0 + 0.03 / 0.2 * math.exp(-0.2 / 20.0))
    assert all(n["temperature"] == pytest.approx(expected_temperature) for n in p.nodes.values())


def test_checkpoint_restores_source_total_and_continuation_exactly():
    p = make_controller(7)
    p.advance(1.0, {}, 0)
    p.advance(2.0, {}, 1)
    value = p.state_dict()
    q = make_controller(7)
    q.load_state_dict(value)
    assert q.direct_heat_total == p.direct_heat_total
    assert q.advance(2.0, {}, 1) == p.advance(2.0, {}, 1)
    assert q.advance(2.5, {}, 2) == p.advance(2.5, {}, 2)
    assert q.state_dict() == p.state_dict()


def test_zero_surprise_matches_plain_cooling():
    p = make_controller(7)
    q = PhysicalController(deepcopy(p.config))
    q.synchronize({f"0:{i}": 0 for i in range(7)})
    for step, loss in enumerate([2.0, 1.0, 0.5]):
        direct = p.advance(loss, {}, step)
        cooling = q.advance(loss, {}, step)
        assert p.nodes == q.nodes
        assert direct["thermal_after"] == cooling["thermal_after"]
        assert direct["external_heat"] == 0.0


def test_uniform_heating_is_topology_invariant():
    temperatures = []
    for count in [1, 7, 63]:
        p = make_controller(count)
        p.advance(1.0, {}, 0)
        p.advance(2.0, {}, 1)
        temperatures.append(np.mean([n["temperature"] for n in p.nodes.values()]))
    assert np.max(temperatures) - np.min(temperatures) < 1e-13


def test_direct_feedback_rejects_an_electrical_reservoir():
    with pytest.raises(ValueError, match="cooling"):
        DirectFeedbackController(PhysicsConfig(mode="capacitor"))
