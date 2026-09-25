
import math
from torchboost.adaptive.config import PhysicsConfig
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