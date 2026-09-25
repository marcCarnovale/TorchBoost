from dataclasses import replace

import numpy as np
import pytest
from experiments.direct_feedback_control import DirectFeedbackController

from torchboost.adaptive.config import PhysicsConfig
from torchboost.adaptive.physics import PhysicalController


LOSSES = (0.7, 0.9, 0.8, 1.1, 1.05, 1.4, 1.2, 1.5)


def cfg(mode):
    return PhysicsConfig(
        mode=mode,
        source_normalization="adaptive_energy",
        topology_normalization=True,
        capacitance=1.0,
        discharge_time=5.0,
        inductive_time=2.0,
        cooling_time=20.0,
        total_heat_capacity=0.2,
        initial_temperature=1.0,
        ambient_temperature=1.0,
        thaw_temperature=1.2,
        max_temperature=10.0,
        charge_gain=0.25,
        max_injection=0.50,
        max_charge=100.0,
        smoothing=0.8,
        dt=0.2,
    )


def run(controller, losses=LOSSES):
    controller.synchronize({"0:0": 0, "0:1": 0, "0:2": 0})
    return [controller.advance(float(loss), {}, step) for step, loss in enumerate(losses)]


@pytest.mark.parametrize("mode", ["capacitor", "rlc"])
def test_adaptive_energy_circuit_is_positive_affine_loss_invariant(mode):
    base = run(PhysicalController(cfg(mode)))
    transformed = run(PhysicalController(cfg(mode)), [13.0 * x + 7.0 for x in LOSSES])
    np.testing.assert_allclose([x["source_signal"] for x in base],
                               [x["source_signal"] for x in transformed], rtol=0, atol=2e-14)
    np.testing.assert_allclose([x["source_work"] for x in base],
                               [x["source_work"] for x in transformed], rtol=0, atol=2e-14)
    np.testing.assert_allclose([x["thermal_after"] for x in base],
                               [x["thermal_after"] for x in transformed], rtol=0, atol=2e-14)


def test_adaptive_energy_direct_is_positive_affine_loss_invariant():
    base_cfg = replace(cfg("capacitor"), mode="cooling")
    base = run(DirectFeedbackController(base_cfg))
    transformed = run(DirectFeedbackController(base_cfg), [13.0 * x + 7.0 for x in LOSSES])
    np.testing.assert_allclose([x["source_signal"] for x in base],
                               [x["source_signal"] for x in transformed], rtol=0, atol=2e-14)
    np.testing.assert_allclose([x["external_heat"] for x in base],
                               [x["external_heat"] for x in transformed], rtol=0, atol=2e-14)
    np.testing.assert_allclose([x["thermal_after"] for x in base],
                               [x["thermal_after"] for x in transformed], rtol=0, atol=2e-14)


def test_adaptive_energy_gives_circuit_and_direct_same_requested_source_budget():
    circuit = run(PhysicalController(cfg("capacitor")))
    direct = run(DirectFeedbackController(replace(cfg("capacitor"), mode="cooling")))
    np.testing.assert_allclose([x["source_signal"] for x in circuit],
                               [x["source_signal"] for x in direct], rtol=0, atol=2e-14)
    np.testing.assert_allclose([x["source_work"] for x in circuit],
                               [x["external_heat"] for x in direct], rtol=0, atol=2e-14)


def test_adaptive_source_checkpoint_restores_scale_and_continuation():
    p = PhysicalController(cfg("capacitor"))
    p.synchronize({"0:0": 0})
    for step, loss in enumerate(LOSSES[:4]):
        p.advance(loss, {}, step)
    q = PhysicalController(cfg("capacitor"))
    q.load_state_dict(p.state_dict())
    assert q.surprise_scale == pytest.approx(p.surprise_scale)
    for step, loss in enumerate(LOSSES[4:], start=4):
        assert q.advance(loss, {}, step) == p.advance(loss, {}, step)


@pytest.mark.parametrize("kind", ["capacitor", "direct"])
def test_adaptive_energy_is_thermal_scale_invariant(kind):
    def normalized_history(total_heat_capacity, thaw):
        base = cfg("capacitor")
        base.total_heat_capacity = total_heat_capacity
        base.thaw_temperature = thaw
        base.max_temperature = 1.0 + 20.0 * (thaw - 1.0)
        base.__post_init__()
        controller = (PhysicalController(base) if kind == "capacitor" else
                      DirectFeedbackController(replace(base, mode="cooling")))
        rows = run(controller)
        span = thaw - base.ambient_temperature
        return [(row["thermal_after"] / (total_heat_capacity * span), row["source_signal"])
                for row in rows]

    a = normalized_history(0.2, 1.2)
    b = normalized_history(0.4, 1.4)
    np.testing.assert_allclose(a, b, rtol=0, atol=3e-14)
