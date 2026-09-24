from copy import deepcopy
from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from torchboost.adaptive.operating_scales import OperatingScales
from torchboost.adaptive.physics import PhysicalController
from torchboost.adaptive.plasticity import PlasticityModule
from torchboost.adaptive.config import ForestConfig,StructureConfig,PhysicsConfig,PlasticityConfig
from torchboost.adaptive.contracts import Observation
from torchboost.adaptive.forest import AdaptiveForest
from torchboost.adaptive.data import Preprocessor
from torchboost.adaptive.objectives import Objective
from torchboost.adaptive.scaling import EpochSampler,StepBudgetTrainer,migrate_single_tree,seed_learned_anchors
from torchboost.adaptive.single_tree import SingleTreeClassifier,SingleTreeRegressor,SingleTreeConfig

torch.set_num_threads(1)

@pytest.mark.parametrize('mode',['capacitor','rlc','cooling'])
@pytest.mark.parametrize('n',[1,7,63,255])
def test_topology_normalization(mode,n):
    base=None
    curves=[]
    for size in [1,n]:
        cfg=OperatingScales().physics(size,mode=mode)
        p=PhysicalController(cfg);p.synchronize({str(i):0 for i in range(size)})
        states=[]
        for step,loss in enumerate([1.,1.2,1.1,1.3,.9]*8):
            r=p.advance(loss,{},step)
            assert abs(r['energy_error'])<1e-10
            states.append([r['charge'],sum(z['current'] for z in r['nodes'].values()),
                           np.mean([z['temperature'] for z in r['nodes'].values()])])
        curves.append(states)
    np.testing.assert_allclose(curves[0],curves[1],rtol=2e-12,atol=2e-12)

def obs(step,utility=.2,occupancy=.5):
    return Observation('n',0,0,step,0,occupancy,1.,.1,utility,utility,1.,0.,.1,.1*step,1.,2.,1.,False,True,1.,1)

@pytest.mark.parametrize('mode',['plastic','full'])
def test_guard_requires_repeated_harm(mode):
    c=PlasticityConfig(mode=mode,stiffness=10.,yield_threshold=.01,
        release_policy='persistent_harm',release_patience=3,minimum_occupancy=.1)
    module=PlasticityModule(c);p={'n':{'weight':torch.nn.Parameter(torch.ones(3))}};module.synchronize(p)
    for step in range(4):
        event=module.advance(p,{'n':obs(step)}, {'n':1.},step,progress=.1)['events'][0]
        assert event['flow_fraction']==0 and event['damage']==0 and event['blocked_release']
    for step in range(4,7):
        event=module.advance(p,{'n':obs(step,-.2)},{'n':1.},step,progress=.1)['events'][0]
        assert event['release_allowed']==(step==6)
    assert event['flow_fraction']>0

def test_no_traffic_does_not_earn_release():
    c=PlasticityConfig(mode='full',release_policy='persistent_harm',release_patience=1,stiffness=10.,yield_threshold=.01)
    m=PlasticityModule(c);p={'n':{'v':torch.nn.Parameter(torch.ones(2))}};m.synchronize(p)
    e=m.advance(p,{'n':obs(0,-1.,0.)},{'n':1.},0,progress=.1)['events'][0]
    assert not e['release_allowed'] and e['damage']==0

def test_legacy_stress_law_still_releases():
    c=PlasticityConfig(mode='plastic',stiffness=10.,yield_threshold=.01)
    m=PlasticityModule(c);p={'n':{'v':torch.nn.Parameter(torch.ones(2))}};m.synchronize(p)
    assert m.advance(p,{'n':obs(0)},{'n':1.},0,progress=.1)['events'][0]['flow_fraction']>0

def test_sampler_is_nested_passes_and_resumable():
    a=EpochSampler(17,5);first=a.take(17);assert sorted(first.tolist())==list(range(17))
    a.take(3);s=a.state_dict();future=a.take(37)
    b=EpochSampler(17,7);b.load_state_dict(s);assert torch.equal(b.take(37),future)
    assert a.exposures==57


def small():
    rng=np.random.default_rng(4);x=rng.normal(size=(120,4)).astype('float32');y=(x[:,0]*x[:,1]>0).astype(int)
    pre=Preprocessor();pre.fit(x[:70],y[:70],classification=True,weights=np.ones(70))
    return pre,[pre.split(x[a:b],y[a:b]) for a,b in [(0,70),(70,95),(95,120)]]

def cfg(mode='none'):
    return ForestConfig(n_trees=1,aggregation='mean',residual_weights=False,epochs=4,batch_size=16,
        execution='forest_packed',record_diagnostics=False,collect_metrics=True,
        structure=StructureConfig(max_depth=2,initial_depth=0,max_nodes=31,structural_gate=False,complexity=0,gate_bimodality=0,allocation_regularization=0),
        physics=OperatingScales().physics(7,mode=mode),
        plasticity=PlasticityConfig(mode='full' if mode=='rlc' else 'none',release_policy='persistent_harm'),
        random_state=2)

@pytest.mark.parametrize('mode',['none','rlc'])
def test_resume_preserves_steps_curves_and_parameters(mode):
    pre,splits=small();c=cfg(mode)
    def make():
        return StepBudgetTrainer(AdaptiveForest(4,1,c),Objective('binary',1),c,torch.Generator().manual_seed(3),updates_per_block=2)
    a=make();a.fit_steps(*splits)
    b=make();b.fit_steps(*splits,blocks=2);state=b.state_dict()
    d=make();d.load_state_dict(state);d.fit_steps(*splits)
    for k,v in a.model.state_dict().items():assert torch.equal(v,d.model.state_dict()[k]),k
    assert a.examples_seen==128 and d.optimizer_steps==8
    assert [r['selection_loss'] for r in a.history]==[r['selection_loss'] for r in d.history]

@pytest.mark.parametrize('readout',['leaf','residual'])
@pytest.mark.parametrize('task',['binary','multiclass','regression'])
def test_predictor_import_and_fresh_anchors(readout,task):
    rng=np.random.default_rng(9);x=rng.normal(size=(80,4))
    if task=='regression':y=x[:,0]+.2*x[:,1];cls=SingleTreeRegressor
    else:y=((x[:,0]>0).astype(int) if task=='binary' else np.digitize(x[:,0],[-.3,.3]));cls=SingleTreeClassifier
    est=cls(SingleTreeConfig(depth=2,epochs=2,batch_size=32,readout=readout)).fit(x[:60],y[:60],eval_set=(x[60:],y[60:]))
    c=cfg('rlc')
    trainer,pre=migrate_single_tree(est,c,updates_per_block=2)
    with torch.no_grad():np.testing.assert_allclose(est.model_(pre.transform_x(x)).numpy(),trainer.model(pre.transform_x(x)).numpy(),atol=2e-6)
    assert float(trainer.plastic.penalty({n.node_id:n.parameters_for_plasticity() for n in trainer.model.iter_nodes()}))==0.
    assert all(v['evidence']==0. for v in trainer.plastic.states.values())
    trainer.fit_steps(pre.split(x[:40],y[:40]),pre.split(x[40:60],y[40:60]),pre.split(x[60:],y[60:]))
    assert trainer.optimizer_steps==8

@pytest.mark.parametrize('n',[0,-1,1.2])
def test_bad_topology(n):
    with pytest.raises(ValueError):OperatingScales().physics(n)
