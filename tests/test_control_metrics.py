import math

import pytest
import torch

from torchboost import CapacitorController, PerformanceTracker, SplitMetricsCollector
from torchboost.trees import RoutingTrace


def test_capacitor_injection_only_on_regression_and_residual_heating():
    controller = CapacitorController(2, cooling_rate=0, reference_decay=0)
    controller.observe_validation(1.0)
    assert controller.charge == 0
    controller.observe_validation(1.2)
    assert controller.charge > 0
    state = controller.advance(torch.ones(2))
    assert (state["heat"] > 0).all()
    old = controller.charge.clone()
    controller.observe_validation(.9)
    assert controller.last_injected_charge == 0
    state = controller.advance(torch.ones(2))
    assert controller.charge < old and state["heat"].sum() > 0


def test_parallel_ohms_law_and_energy_budget():
    controller = CapacitorController(2, cooling_rate=0, capacitance=1.)
    controller.observe_validation(1.)
    controller.observe_validation(1.4)
    energy = controller.charge.square()/(2*controller.capacitance)
    r = torch.tensor([1., 4.], dtype=torch.float64)
    state = controller.advance(r, dt=.5)
    torch.testing.assert_close(state["voltage"].expand(2), state["current"]*r)
    torch.testing.assert_close(state["power"], state["current"].square()*r)
    torch.testing.assert_close(state["heat"].sum()+controller.charge.square()/(2*controller.capacitance), energy)
    torch.testing.assert_close(state["heat"][0], 4*state["heat"][1])


@pytest.mark.parametrize("law", ["linear", "radiative"])
def test_cooling_separate_from_heating_and_passive(law):
    controller = CapacitorController(3, cooling_law=law)
    hot = controller.temperature.clone()
    result = controller.advance(torch.ones(3), dt=100.)
    assert result["heat"].sum() == 0
    assert (controller.temperature < hot).all()
    assert (controller.temperature >= controller.ambient).all()
    torch.testing.assert_close((hot-controller.temperature)*controller.heat_capacity, result["cooling"])


def test_capacitor_state_restore_and_future_trajectory():
    first, second = CapacitorController(2), CapacitorController(2)
    first.observe_validation(1.)
    first.observe_validation(1.2)
    first.advance(torch.ones(2))
    second.load_state_dict(first.state_dict())
    a, b = first.advance(torch.ones(2)), second.advance(torch.ones(2))
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


def test_charge_saturation_is_logged():
    c = CapacitorController(1, max_charge=1.)
    c.observe_validation(0.)
    c.observe_validation(5.)
    assert c.charge == 1 and c.last_rejected_charge > 0


def test_information_gain_not_confused_with_entropy():
    y = torch.tensor([0., 0., 1., 1.])
    left = torch.tensor([[1., .5], [1., .5], [0., .5], [0., .5]])
    collector = SplitMetricsCollector(2, 3)
    trace = RoutingTrace(torch.ones(4, 2), left, torch.ones(4, 1))
    collector.observe_batch(trace, y, torch.ones(4))
    a, b = collector.finish_epoch(1)
    assert abs(a.information_gain-math.log(2)) < 1e-10
    assert a.routing_entropy == 0
    assert abs(b.information_gain) < 1e-10
    assert abs(b.routing_entropy-math.log(2)) < 1e-10
    assert a.effective_samples == 4


def test_cancelling_updates_count_against_stability():
    collector = SplitMetricsCollector(1, 0)
    zero, one = torch.zeros(1, 2), torch.ones(1, 2)
    b = torch.zeros(1)
    collector.observe_update(zero, b, one, b, one, b)
    collector.observe_update(one, b, zero, b, one, b)
    observation = collector.finish_epoch(1)[0]
    assert abs(observation.update_path_length-2*math.sqrt(2)) < 1e-10


def test_tracker_is_bounded_serializable_and_node_specific():
    collector = SplitMetricsCollector(2, 4)
    tracker = PerformanceTracker(history_length=2)
    for epoch in range(1, 5):
        tracker.record(collector.finish_epoch(epoch))
    assert set(tracker.snapshot()) == {"stage:4/node:0", "stage:4/node:1"}
    assert all(len(history) == 2 for history in tracker.history.values())
    restored = PerformanceTracker()
    restored.load_state_dict(tracker.state_dict())
    assert restored.state_dict() == tracker.state_dict()
    with pytest.raises(ValueError):
        tracker.record(collector.finish_epoch(4))
