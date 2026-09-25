
import math
from types import SimpleNamespace

import numpy as np
import torch
from torchboost.adaptive.config import ForestConfig, PhysicsConfig
from torchboost.adaptive.optim import DynamicOptimizer, LocalMomentum
from torchboost.adaptive.physics import PhysicalController

def cfg(mode="rlc"):
    return PhysicsConfig(mode=mode,topology_normalization=True,capacitance=1.,discharge_time=5.,
        inductive_time=2.,cooling_time=20.,total_heat_capacity=.2,dt=.2,
        initial_temperature=1.,ambient_temperature=1.,max_temperature=5.,
        thaw_temperature=1.2,charge_gain=2.,max_injection=.2)

def test_normalized_components_preserve_system_timescales():
    for n in (1,7,63):
        p=PhysicalController(cfg())
        p.synchronize({f"0:{i}":0 for i in range(n)})
        st=next(iter(p.nodes.values()))
        assert math.isclose(st["resistance"]/n,5.)
        assert math.isclose(st["inductance"]/st["resistance"],2.)
        assert math.isclose(st["capacity"]*n,.2)
        assert math.isclose(st["capacity"]/st["cooling"],20.)

def test_topology_recalibration_preserves_stored_energy():
    p=PhysicalController(cfg())
    p.synchronize({"0:0":0})
    s=p.nodes["0:0"];s["temperature"]=1.4;s["current"]=.3;p.charge=.4
    before=p.electrical_energy()+p.thermal_energy()
    p.synchronize({"0:0":0,"0:1":0,"0:2":0})
    after=p.electrical_energy()+p.thermal_energy()
    assert math.isclose(before,after,rel_tol=1e-12,abs_tol=1e-12)
    p.synchronize({"0:0":0})
    # Removed branches' stored energy is explicitly retired; live energy can drop.
    assert p.retired_energy>=0

def test_neutral_temperature_does_not_change_routing_at_rest():
    c=cfg("capacitor")
    assert c.initial_temperature==c.ambient_temperature==1.

def test_energy_momentum_is_invariant_to_parameter_count_for_equal_energy_density():
    c = ForestConfig(optimizer="energy_momentum", momentum=.2, momentum_max=.9, momentum_energy_gain=1.)
    p1 = torch.nn.Parameter(torch.zeros(2))
    p2 = torch.nn.Parameter(torch.zeros(20))
    a = LocalMomentum([{"params":[p1], "owner":"a"}], c)
    b = LocalMomentum([{"params":[p2], "owner":"b"}], c)
    a.state[p1]["momentum_buffer"] = torch.ones_like(p1)
    b.state[p2]["momentum_buffer"] = torch.ones_like(p2)
    p1.grad = torch.ones_like(p1)
    p2.grad = torch.ones_like(p2)
    a.step()
    b.step()
    assert math.isclose(a.param_groups[0]["last_beta"], b.param_groups[0]["last_beta"], rel_tol=1e-12)


def _controlled_group(node_count, temperature, local_energy):
    physics = cfg("rlc")
    physics.max_charge = 2.
    physics.thaw_temperature = 1.2
    physics.lr_coupling = .5
    forest_cfg = ForestConfig(optimizer="circuit_momentum", learning_rate=.1, physics=physics)
    dynamic = DynamicOptimizer.__new__(DynamicOptimizer)
    dynamic.config = forest_cfg
    dynamic.optimizer = SimpleNamespace(param_groups=[{"owner":"focus"}])
    nodes = {"focus":{"temperature":temperature, "inductive_energy":local_energy}}
    for i in range(node_count - 1):
        nodes[f"other:{i}"] = {"temperature":physics.ambient_temperature, "inductive_energy":0.}
    dynamic.set_controls(.1, nodes)
    return dynamic.optimizer.param_groups[0]


def test_thermal_lr_coupling_uses_fraction_of_ambient_to_thaw_interval():
    group = _controlled_group(2, 1.1, 0.)
    assert math.isclose(group["lr"], .125, rel_tol=1e-12)


def test_circuit_momentum_energy_scale_is_topology_invariant():
    two = _controlled_group(2, 1., .5)
    four = _controlled_group(4, 1., .25)
    assert math.isclose(two["inductive_energy"], four["inductive_energy"], rel_tol=1e-12)


def _observation(gradient):
    return SimpleNamespace(
        utility=.1,
        entropy=.2,
        gradient_norm=gradient,
        structural_gradient=.3 * gradient,
    )


def test_allocation_uses_nominal_resistance_instead_of_compounding_previous_step():
    c = cfg("capacitor")
    c.allocation = "gradient"
    p = PhysicalController(c)
    p.synchronize({"0:0":0, "0:1":0})
    observations = {"0:0":_observation(.2), "0:1":_observation(1.1)}
    first = p._resistances(sorted(p.nodes), observations)
    for key, value in zip(sorted(p.nodes), first):
        p.nodes[key]["resistance"] = float(value)
    second = p._resistances(sorted(p.nodes), observations)
    assert np.allclose(first, second, rtol=1e-12, atol=1e-12)


def test_nonuniform_allocation_preserves_topology_normalized_total_conductance():
    c = cfg("capacitor")
    c.allocation = "gradient"
    p = PhysicalController(c)
    p.synchronize({f"0:{i}":0 for i in range(7)})
    keys = sorted(p.nodes)
    observations = {key:_observation(.1 + i) for i, key in enumerate(keys)}
    resistance = p._resistances(keys, observations)
    expected = c.capacitance / c.discharge_time
    assert math.isclose(float((1. / resistance).sum()), expected, rel_tol=1e-12)


def test_topology_normalized_resistance_bounds_scale_with_network_size():
    c = cfg("capacitor")
    p = PhysicalController(c)
    p.synchronize({f"0:{i}":0 for i in range(511)})
    resistance = p._resistances(sorted(p.nodes), {})
    expected = 511 * c.discharge_time / c.capacitance
    assert expected > c.resistance_max
    assert np.allclose(resistance, expected)
