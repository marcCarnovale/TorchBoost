import math
import torch
from types import SimpleNamespace
from torchboost.adaptive.config import ForestConfig,PhysicsConfig,StructureConfig
from torchboost.adaptive.forest import AdaptiveForest
from torchboost.adaptive.optim import DynamicOptimizer,LocalMomentum
from torchboost.adaptive.physics import PhysicalController

def base_cfg(n=1,optimizer="sgd",ambient=1.,thaw=1.2,lr_coupling=0.):
    mode="rlc" if optimizer=="circuit_momentum" else "cooling"
    return ForestConfig(n_trees=n,aggregation="additive",residual_weights=False,optimizer=optimizer,
        learning_rate=.01,collect_metrics=False,
        structure=StructureConfig(dynamic=False,initial_depth=0,max_depth=0,max_nodes=1),
        physics=PhysicsConfig(mode=mode,topology_normalization=True,initial_temperature=ambient,
            ambient_temperature=ambient,thaw_temperature=thaw,lr_coupling=lr_coupling))

def test_energy_momentum_is_parameter_count_normalized():
    cfg=base_cfg(optimizer="energy_momentum")
    p1=torch.nn.Parameter(torch.zeros(2));p2=torch.nn.Parameter(torch.zeros(20))
    a=LocalMomentum([{"params":[p1],"owner":"a"}],cfg);b=LocalMomentum([{"params":[p2],"owner":"b"}],cfg)
    a.state[p1]["momentum_buffer"]=torch.ones_like(p1);b.state[p2]["momentum_buffer"]=torch.ones_like(p2)
    p1.grad=torch.ones_like(p1);p2.grad=torch.ones_like(p2);a.step();b.step()
    assert math.isclose(a.param_groups[0]["last_beta"],b.param_groups[0]["last_beta"],rel_tol=1e-12)

def test_thermal_lr_uses_fraction_of_thaw_scale():
    factors=[]
    for ambient,thaw,temp in ((1.,1.2,1.1),(10.,12.,11.)):
        cfg=base_cfg(ambient=ambient,thaw=thaw,lr_coupling=.3);forest=AdaptiveForest(2,1,cfg);opt=DynamicOptimizer(forest,cfg)
        root=next(forest.iter_nodes()).node_id
        opt.set_controls(.01,{root:{"temperature":temp,"inductive_energy":0.}})
        factors.append(next(g["lr"] for g in opt.optimizer.param_groups if g["owner"]==root)/.01)
    assert math.isclose(factors[0],factors[1],rel_tol=1e-12)

def test_circuit_momentum_preserves_fixed_system_energy_scale():
    for n in (1,7):
        cfg=base_cfg(n,optimizer="circuit_momentum");forest=AdaptiveForest(2,1,cfg);opt=DynamicOptimizer(forest,cfg)
        roots=[node.node_id for node in forest.iter_nodes()]
        physical={key:{"temperature":1.,"inductive_energy":2./n} for key in roots};opt.set_controls(.01,physical)
        vals=[g["inductive_energy"] for g in opt.optimizer.param_groups if g["owner"] in physical]
        assert vals and all(math.isclose(v,2.,rel_tol=1e-12) for v in vals)

def test_allocation_preserves_topology_normalized_total_conductance():
    cfg=PhysicsConfig(mode="capacitor",allocation="gradient",topology_normalization=True,
        capacitance=1.,discharge_time=5.,initial_temperature=1.,ambient_temperature=1.)
    p=PhysicalController(cfg);ids={f"0:{i}":0 for i in range(9)};p.synchronize(ids)
    obs={k:SimpleNamespace(utility=.1,entropy=.2,gradient_norm=float(i+1),structural_gradient=0.)
         for i,k in enumerate(sorted(ids))}
    keys=sorted(ids);base=sum(1/p.nodes[k]["resistance"] for k in keys);r=p._resistances(keys,obs)
    assert math.isclose(float((1/r).sum()),base,rel_tol=1e-12,abs_tol=1e-12)
