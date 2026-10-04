from dataclasses import replace

import numpy as np
import pytest

from experiments.normalized_energy_controller import (
    EnergySource,
    InnovationSignal,
    NormalizedEnergyController,
    add_capacitor_energy,
)
from torchboost.adaptive.config import PhysicsConfig


def controller(mode, count=7, maximum=5.0, rate=0.4):
    cfg = PhysicsConfig(
        mode=mode, topology_normalization=True, initial_temperature=1.0,
        ambient_temperature=1.0, thaw_temperature=1.2, max_temperature=maximum,
        total_heat_capacity=0.2, capacitance=1.0, max_charge=5.0,
        discharge_time=0.8, inductive_time=0.3, cooling_time=8.0, dt=0.2,
    )
    c = NormalizedEnergyController(cfg, source=EnergySource(rate=rate, warmup=2, threshold=0.3))
    c.synchronize({f"0:{i}": 0 for i in range(count)})
    return c


def losses():
    return [1.0, 1.01, 0.98, 1.02, 0.99, 1.2, 1.4, 1.45, 1.1, 1.0, 1.8, 1.9]


@pytest.mark.parametrize("mode", ["cooling", "capacitor", "rlc"])
@pytest.mark.parametrize("scale,offset", [(0.01, 7.0), (10.0, -3.0)])
def test_loss_unit_changes_preserve_entire_physical_trajectory(mode, scale, offset):
    a, b = controller(mode), controller(mode)
    for i, loss in enumerate(losses()):
        x = a.advance(loss, {}, i)
        y = b.advance(scale * loss + offset, {}, i)
        for field in ("source_work", "external_heat", "thermal_after", "electrical_after", "source_drive"):
            assert x[field] == pytest.approx(y[field], rel=1e-9, abs=1e-12)
        assert [s["temperature"] for s in a.nodes.values()] == pytest.approx(
            [s["temperature"] for s in b.nodes.values()], rel=1e-10)


@pytest.mark.parametrize("mode", ["cooling", "capacitor", "rlc"])
@pytest.mark.parametrize("count", [1, 7, 63])
def test_whole_step_energy_ledger_with_venting(mode, count):
    c = controller(mode, count, maximum=1.005)
    for i, loss in enumerate(losses()):
        before = c.electrical_energy() + c.thermal_energy()
        row = c.advance(loss, {}, i)
        expected = before + row["source_work"] + row["external_heat"] - row["cooling_energy"] - row["vented_energy"]
        assert c.electrical_energy() + c.thermal_energy() == pytest.approx(expected, abs=1e-12)
        assert abs(row["energy_error"]) < 1e-12
    assert c.source_energy_total > 0


@pytest.mark.parametrize("mode", ["cooling", "capacitor", "rlc"])
def test_replay_checkpoint_and_invalid_input_do_not_mutate(mode):
    a = controller(mode)
    for i, loss in enumerate(losses()):
        a.advance(loss, {}, i)
    state = a.state_dict()
    replay = a.advance(losses()[-1], {}, len(losses()) - 1)
    assert a.state_dict() == state
    replay["source_energy_total"] = -1
    for bad_loss, bad_step in [(1.0, 0), (2.0, len(losses()) - 1), (float("nan"), 99)]:
        with pytest.raises(ValueError):
            a.advance(bad_loss, {}, bad_step)
        assert a.state_dict() == state
    b = controller(mode)
    b.load_state_dict(state)
    assert a.advance(2.0, {}, 99) == b.advance(2.0, {}, 99)
    wrong = controller(mode, rate=9.0)
    with pytest.raises(ValueError, match="configuration"):
        wrong.load_state_dict(state)


def test_source_energy_is_shared_but_thermal_response_is_not_forced_equal():
    rows = []
    for mode in ("cooling", "capacitor", "rlc"):
        c = controller(mode)
        for i, loss in enumerate(losses()):
            c.advance(loss, {}, i)
        rows.append((c.source_energy_total, c.thermal_energy()))
    assert np.ptp([r[0] for r in rows]) < 1e-12
    assert np.ptp([r[1] for r in rows]) > 1e-4


def test_signed_capacitor_source_preserves_energy_and_rejects_excess():
    for sign in (-1, 1):
        q, energy = add_capacitor_energy(sign * 0.5, 2.0, 0.1, 2.0)
        assert np.sign(q) == sign
        assert energy == pytest.approx(0.1)
        q, energy = add_capacitor_energy(sign * 0.5, 2.0, 999, 1.0)
        assert q == sign
        assert energy == pytest.approx((1 - 0.25) / 4)
        assert add_capacitor_energy(sign * 3.0, 2.0, 0.1, 2.0) == (sign * 3.0, 0.0)


def test_constant_signal_cannot_manufacture_heat():
    c = controller("capacitor")
    for i in range(100):
        c.advance(8.0, {}, i)
    assert c.source_energy_total == 0
    assert c.thermal_energy() == 0


def test_warmup_and_zero_variance_fallback_are_explicit():
    signal = InnovationSignal(EnergySource(warmup=3, threshold=0.5))
    for _ in range(3):
        assert signal.observe(1.0)["source_drive"] == 0
    assert signal.observe(2.0)["normalized_surprise"] == 1.0


def test_temperature_unit_change_with_inverse_capacity_preserves_energy_and_thaw_units():
    a = controller("rlc")
    cfg = replace(a.config, ambient_temperature=10, initial_temperature=10,
                  thaw_temperature=12, max_temperature=50, total_heat_capacity=0.02)
    b = NormalizedEnergyController(cfg, source=a.source)
    b.synchronize({f"0:{i}": 0 for i in range(7)})
    for i, loss in enumerate(losses()):
        x, y = a.advance(loss, {}, i), b.advance(loss, {}, i)
        assert x["thermal_after"] == pytest.approx(y["thermal_after"], abs=1e-12)
        assert [s["temperature"] * 10 for s in a.nodes.values()] == pytest.approx(
            [s["temperature"] for s in b.nodes.values()])


@pytest.mark.parametrize("mode", ["cooling", "capacitor", "rlc"])
def test_topology_count_does_not_change_system_thermal_response(mode):
    traces = []
    for count in (1, 7, 63):
        c = controller(mode, count)
        trace = []
        for i, loss in enumerate(losses()):
            row = c.advance(loss, {}, i)
            trace.append((row["source_energy_total"], row["thermal_after"], row["electrical_after"]))
        traces.append(trace)
    np.testing.assert_allclose(traces[0], traces[1], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(traces[0], traces[2], rtol=1e-10, atol=1e-12)
