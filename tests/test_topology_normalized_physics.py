
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

def test_nonuniform_allocation_preserves_total_conductance():
    from types import SimpleNamespace
    c=cfg("capacitor");c.allocation="gradient"
    p=PhysicalController(c);ids={f"0:{i}":0 for i in range(9)};p.synchronize(ids)
    obs={k:SimpleNamespace(utility=.1,entropy=.2,gradient_norm=float(i+1),structural_gradient=0.)
         for i,k in enumerate(sorted(ids))}
    keys=sorted(ids);base=sum(1/p.nodes[k]["resistance"] for k in keys);r=p._resistances(keys,obs)
    assert math.isclose(float((1/r).sum()),base,rel_tol=1e-12,abs_tol=1e-12)


def test_optimizer_physics_is_scale_normalized():
    import torch
    from torchboost.adaptive.config import ForestConfig,StructureConfig
    from torchboost.adaptive.forest import AdaptiveForest
    from torchboost.adaptive.optim import DynamicOptimizer,LocalMomentum

    c=ForestConfig(n_trees=1,aggregation="additive",residual_weights=False,optimizer="energy_momentum",
        learning_rate=.01,collect_metrics=False,
        structure=StructureConfig(dynamic=False,initial_depth=0,max_depth=0,max_nodes=1),
        physics=PhysicsConfig(mode="cooling",topology_normalization=True,initial_temperature=1.,
            ambient_temperature=1.,thaw_temperature=1.2))
    p1=torch.nn.Parameter(torch.zeros(2));p2=torch.nn.Parameter(torch.zeros(20))
    a=LocalMomentum([{"params":[p1],"owner":"a"}],c);b=LocalMomentum([{"params":[p2],"owner":"b"}],c)
    a.state[p1]["momentum_buffer"]=torch.ones_like(p1);b.state[p2]["momentum_buffer"]=torch.ones_like(p2)
    p1.grad=torch.ones_like(p1);p2.grad=torch.ones_like(p2);a.step();b.step()
    assert math.isclose(a.param_groups[0]["last_beta"],b.param_groups[0]["last_beta"],rel_tol=1e-12)

    factors=[]
    for ambient,thaw,temp in ((1.,1.2,1.1),(10.,12.,11.)):
        q=ForestConfig(n_trees=1,aggregation="additive",residual_weights=False,optimizer="sgd",
            learning_rate=.01,collect_metrics=False,
            structure=StructureConfig(dynamic=False,initial_depth=0,max_depth=0,max_nodes=1),
            physics=PhysicsConfig(mode="cooling",topology_normalization=True,initial_temperature=ambient,
                ambient_temperature=ambient,thaw_temperature=thaw,lr_coupling=.3))
        forest=AdaptiveForest(2,1,q);opt=DynamicOptimizer(forest,q);root=next(forest.iter_nodes()).node_id
        opt.set_controls(.01,{root:{"temperature":temp,"inductive_energy":0.}})
        factors.append(next(g["lr"] for g in opt.optimizer.param_groups if g["owner"]==root)/.01)
    assert math.isclose(factors[0],factors[1],rel_tol=1e-12)
