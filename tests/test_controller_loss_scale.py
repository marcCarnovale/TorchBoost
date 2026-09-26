"""Characterize source scaling; this does not endorse cross-task dose mismatch."""
from dataclasses import replace

import pytest

from experiments.direct_feedback_control import DirectFeedbackController
from torchboost.adaptive.config import PhysicsConfig
from torchboost.adaptive.physics import PhysicalController


def source_heat(kind, scale=1.0, offset=0.0):
    cfg = PhysicsConfig(
        mode="capacitor", topology_normalization=True,
        capacitance=1.0, discharge_time=5.0, cooling_time=32.0,
        total_heat_capacity=10.0, dt=0.2,
        initial_temperature=1.0, ambient_temperature=1.0,
        thaw_temperature=1.2, max_temperature=1000.0,
        charge_gain=0.1, max_injection=100.0, max_charge=1000.0,
        smoothing=0.85,
    )
    model = (PhysicalController(cfg) if kind == "capacitor" else
             DirectFeedbackController(replace(cfg, mode="cooling")))
    model.synchronize({"0:0": 0})
    losses = (1.0, 1.1, 1.0, 1.2, 1.15, 1.3, 1.1, 1.25)
    for step, loss in enumerate(losses):
        model.advance(scale * loss + offset, {}, step)
    field = "resistor_heat" if kind == "capacitor" else "external_heat"
    return sum(row[field] for row in model.history)


def test_existing_sources_have_quadratic_vs_linear_loss_unit_scaling():
    cap_ratio = source_heat("capacitor", 10.0) / source_heat("capacitor")
    direct_ratio = source_heat("direct", 10.0) / source_heat("direct")
    assert cap_ratio == pytest.approx(100.0, rel=1e-12)
    assert direct_ratio == pytest.approx(10.0, rel=1e-12)


@pytest.mark.parametrize("kind", ["capacitor", "direct"])
def test_constant_loss_offset_does_not_change_source_heat(kind):
    assert source_heat(kind, offset=7.0) == pytest.approx(source_heat(kind), rel=1e-12)
