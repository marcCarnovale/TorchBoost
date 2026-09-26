from copy import deepcopy
import math
import numpy as np
import pytest
import torch

from torchboost.adaptive import (AdaptiveForest, AdaptiveForestClassifier, ForestConfig,
                                 PhysicsConfig, StructureConfig, ScheduleConfig)
from torchboost.adaptive.long_horizon import LongHorizonPlan, summarize_trajectory

torch.set_num_threads(1)

@pytest.mark.parametrize('arity', [2, 3, 4])
@pytest.mark.parametrize('head', ['shared', 'specialized'])
@pytest.mark.parametrize('trace', [False, True])
def test_forest_packed_output_gradient(arity, head, trace):
    cfg = ForestConfig(n_trees=3, head_mode=head, structure=StructureConfig(arity=arity,max_depth=2,max_nodes=100))
    reference = AdaptiveForest(5,3,cfg).double()
    gen = torch.Generator().manual_seed(74)
    with torch.no_grad():
        for node in reference.iter_nodes():
            node.value.normal_(generator=gen)
            if node.structural is not None:
                node.structural.fill_(.63)
    candidate = deepcopy(reference)
    candidate.config.execution = 'forest_packed'
    x = torch.randn(17,5,generator=gen,dtype=torch.float64)
    kwargs = {'trace': trace, 'disabled_refinements': frozenset({'1:1'}),
              'disabled_nodes': frozenset({'0:2'}), 'disabled_tree_ids': frozenset({2})}
    a,b = reference(x,**kwargs),candidate(x,**kwargs)
    aa,bb = (a[0],b[0]) if trace else (a,b)
    torch.testing.assert_close(aa,bb,rtol=1e-10,atol=1e-10)
    aa.square().sum().backward();bb.square().sum().backward()
    for (name,p),(other,q) in zip(reference.named_parameters(),candidate.named_parameters()):
        assert name == other
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-9,atol=1e-9)
    if trace:
        assert set(a[1].nodes) == set(b[1].nodes)
        for key, ob in a[1].nodes.items():
            for field in ('reach','output','refinement'):
                torch.testing.assert_close(getattr(ob,field),getattr(b[1].nodes[key],field),rtol=1e-10,atol=1e-10)


def test_growth_prune_hard_and_masks():
    cfg = ForestConfig(n_trees=3, aggregation='additive', interaction_groups=((0,1),(2,3)),
                       structure=StructureConfig(dynamic=True,max_depth=3,initial_depth=1))
    a = AdaptiveForest(4,2,cfg)
    gen = torch.Generator().manual_seed(13)
    for tree in a.trees:
        tree.grow(tree.get(tree.root_id).children_ids[0], generator=gen,arity=3)
    b=deepcopy(a);b.config.execution='forest_packed'
    x=torch.randn(23,4,generator=gen)
    with torch.no_grad():
        for node in a.iter_nodes():node.value.normal_(generator=gen)
    b.load_state_dict(a.state_dict())
    for hard in (False,True):
        a.eval();b.eval();torch.testing.assert_close(a(x,hard=hard),b(x,hard=hard))
    for tree in a.trees:tree.prune(tree.root_id)
    for tree in b.trees:tree.prune(tree.root_id)
    torch.testing.assert_close(a(x),b(x))


def test_diagnostics_resume_and_best_does_not_stop(tmp_path):
    rng=np.random.default_rng(31);x=rng.normal(size=(96,4));y=(x[:,0]+x[:,1]>0).astype(int)
    cfg=ForestConfig(n_trees=2,epochs=12,execution='forest_packed',record_diagnostics=True,compact_history=True,
                     batch_size=20,collect_metrics=False,
                     structure=StructureConfig(max_depth=1,complexity=0,gate_bimodality=0,allocation_regularization=0))
    common={'control_set':(x[60:78],y[60:78]),'eval_set':(x[78:],y[78:])}
    full=AdaptiveForestClassifier(cfg).fit(x[:60],y[:60],**common)
    split=AdaptiveForestClassifier(cfg).fit(x[:60],y[:60],stop_epoch=5,**common)
    path=tmp_path/'state.pt';split.save(path)
    resumed=AdaptiveForestClassifier.load(path)
    resumed.resume_fit(x[:60],y[:60],**common)
    assert resumed.trainer_.epoch == 12
    assert resumed.history_[-1]['optimizer_steps'] == 36
    assert resumed.history_[-1]['examples_seen'] == 720
    for aa,bb in zip(full.trainer_.model.parameters(),resumed.trainer_.model.parameters()):
        torch.testing.assert_close(aa,bb,rtol=0,atol=0)
    for key in ('train_loss','train_errors','train_margin_min','optimizer_steps'):
        assert [r[key] for r in full.history_] == [r[key] for r in resumed.history_]
    for m in (full,split,resumed):m.trainer_.close()


def test_unlocked_plan_and_no_claim_detector():
    p=LongHorizonPlan(epochs=32,checkpoints=(8,32))
    cfg=ForestConfig(epochs=32,record_diagnostics=True)
    p.validate_config(cfg)
    cfg.plasticity.terminal_lock=True
    with pytest.raises(ValueError):p.validate_config(cfg)
    cfg.plasticity.terminal_lock=False
    cfg.schedules={'learning_rate':ScheduleConfig(kind='linear',low=.03,high=0)}
    with pytest.raises(ValueError):p.validate_config(cfg)
    rows=[dict(epoch=i,optimizer_steps=(i+1)*2,examples_seen=(i+1)*10,train_errors=0 if i>=2 else 1,
               selection_score=1 if i<10 else 2 if i<22 else .5) for i in range(32)]
    result=summarize_trajectory(rows,'binary',p)
    assert result['first_sustained_interpolation_epoch']==3
    assert result['double_descent_established'] is False
    assert result['nonmonotone_selection_screen'] is True

@pytest.mark.parametrize('trace',[False,True])
def test_fused_dropout_rng_and_state_purity(trace):
    cfg=ForestConfig(n_trees=3,feature_dropout=.2,tree_dropout=.2,
        structure=StructureConfig(max_depth=2))
    a=AdaptiveForest(4,2,cfg)
    with torch.no_grad():
        for n in a.iter_nodes():n.value.normal_()
    b=deepcopy(a);b.config.execution='forest_packed'
    x=torch.randn(13,4)
    a.train();b.train()
    ga=torch.Generator().manual_seed(1);gb=torch.Generator().manual_seed(1)
    ya,yb=a(x,generator=ga,trace=trace),b(x,generator=gb,trace=trace)
    torch.testing.assert_close(ya[0] if trace else ya,yb[0] if trace else yb)
    assert torch.equal(ga.get_state(),gb.get_state())
    before=deepcopy(b.state_dict());b.eval();b(x)
    for k,v in before.items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)


def test_diagnostics_do_not_change_training():
    rng=np.random.default_rng(122);x=rng.normal(size=(72,4));y=(x[:,0]>0).astype(int)
    cfg=ForestConfig(n_trees=2,epochs=5,collect_metrics=False,structure=StructureConfig(max_depth=1))
    a=AdaptiveForestClassifier(cfg).fit(x[:48],y[:48],eval_set=(x[48:],y[48:]))
    other=deepcopy(cfg);other.record_diagnostics=True;other.compact_history=True
    b=AdaptiveForestClassifier(other).fit(x[:48],y[:48],eval_set=(x[48:],y[48:]))
    for p,q in zip(a.trainer_.model.parameters(),b.trainer_.model.parameters()):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    a.trainer_.close();b.trainer_.close()


def test_coupled_fused_resume(tmp_path):
    from torchboost.adaptive import PlasticityConfig
    rng=np.random.default_rng(122);x=rng.normal(size=(90,4));y=(x[:,0]+x[:,1]>0).astype(int)
    cfg=ForestConfig(n_trees=2,epochs=10,record_diagnostics=True,execution='forest_packed',
        structure=StructureConfig(dynamic=True,initial_depth=1,max_depth=2,grow_every=2,
                                  cycle_epochs=8,initial_dormant_fraction=0),
        physics=PhysicsConfig(mode='rlc'),plasticity=PlasticityConfig(mode='full'))
    common={'control_set':(x[50:70],y[50:70]),'eval_set':(x[70:],y[70:])}
    a=AdaptiveForestClassifier(cfg).fit(x[:50],y[:50],**common)
    b=AdaptiveForestClassifier(cfg).fit(x[:50],y[:50],stop_epoch=4,**common)
    p=tmp_path/'coupled.pt';b.save(p);c=AdaptiveForestClassifier.load(p)
    c.resume_fit(x[:50],y[:50],**common)
    for u,v in zip(a.trainer_.model.parameters(),c.trainer_.model.parameters()):
        torch.testing.assert_close(u,v,rtol=0,atol=0)
    assert a.trainer_.model.topology_version==c.trainer_.model.topology_version
    assert a.history_[-1]['optimizer_steps']==c.history_[-1]['optimizer_steps']
    for m in (a,b,c):m.trainer_.close()


def test_regression_trajectory_metrics():
    from torchboost.adaptive import AdaptiveForestRegressor
    rng=np.random.default_rng(55);x=rng.normal(size=(60,3));y=np.stack((x[:,0],x[:,1]),axis=1)
    cfg=ForestConfig(n_trees=2,epochs=4,execution='forest_packed',record_diagnostics=True,
                     structure=StructureConfig(max_depth=1),collect_metrics=False)
    m=AdaptiveForestRegressor(cfg).fit(x[:40],y[:40],eval_set=(x[40:],y[40:]))
    result=summarize_trajectory(m.history_,'regression',LongHorizonPlan(4,(4,)))
    assert result['epochs_completed']==4
    assert m.history_[-1]['train_mse']>=0
    m.trainer_.close()
