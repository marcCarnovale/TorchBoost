from copy import deepcopy
import numpy as np
import pytest
import torch
from torchboost.adaptive.unified_progressive import UnifiedConfig,UnifiedProgressiveClassifier,UnifiedProgressiveRegressor
from torchboost.adaptive.newton_builder import derivatives,BuilderConfig,build_tree
from torchboost.adaptive.progressive_regularizers import Regularizers,allocation,penalties
from torchboost.adaptive.config import PhysicsConfig,PlasticityConfig,OnlineConfig,ScheduleConfig
from torchboost.adaptive.contracts import StructuralAction
from torchboost.adaptive.data import DataSplit
from torchboost.adaptive.objectives import Objective


def data(task='binary'):
    r=np.random.default_rng(819);x=r.normal(size=(400,5)).astype('float32')
    if task=='binary':y=(x[:,0]+x[:,1]*x[:,2]>0).astype(int)
    elif task=='multiclass':y=np.argmax(np.stack([x[:,0],x[:,1],-x[:,0]-x[:,1]],1),1)
    else:y=np.stack([x[:,0]+x[:,1]**2,2*x[:,2]],1)
    return x,y


def fit(task='binary',cfg=None,stop=None):
    x,y=data(task);cls=UnifiedProgressiveRegressor if task=='regression' else UnifiedProgressiveClassifier
    c=cfg or UnifiedConfig(n_trees=3,updates_per_stage=4,depth=2)
    return cls(c).fit(x[:220],y[:220],control_set=(x[220:300],y[220:300]),eval_set=(x[300:],y[300:]),stop_stages=stop)


@pytest.mark.parametrize('task',['binary','multiclass','regression'])
def test_full_newton_derivatives_match_autograd(task):
    _,y=data(task);k=1 if task=='binary' else 3 if task=='multiclass' else 2
    z=torch.randn(3,k,dtype=torch.double);yy=torch.tensor(y[:3]);obj=Objective(task,k)
    g,h=derivatives(z.numpy(),y[:3],task)
    for i in range(3):
        f=lambda v:obj.loss(v[None],yy[i:i+1])[0]
        assert np.allclose(torch.autograd.functional.jacobian(f,z[i]),g[i],atol=1e-10)
        assert np.allclose(torch.autograd.functional.hessian(f,z[i]),h[i],atol=1e-10)


@pytest.mark.parametrize('readout',['leaf','residual'])
def test_hard_proposal_leaf_equivalence(readout):
    x,y=data();ds=DataSplit(torch.from_numpy(x),torch.from_numpy(y),torch.ones(len(x)))
    t,r=build_tree(ds,torch.zeros(len(x),1),'binary',0,UnifiedConfig().native,BuilderConfig(depth=2,readout=readout),torch.ones(5),torch.Generator())
    out,tr=t(ds.x,hard=True,trace=True)
    for item in r['leaves']:
        reached=tr[item['node_id']].reach>0
        assert torch.allclose(out[reached],torch.tensor(item['value']).float().expand(reached.sum(),-1),atol=1e-6)


@pytest.mark.parametrize('kw',[{'min_child_weight':1e10},{'split_cost':1e10}])
def test_builder_constraints_act_on_actual_splits(kw):
    x,y=data();ds=DataSplit(torch.from_numpy(x),torch.from_numpy(y),torch.ones(len(x)))
    t,r=build_tree(ds,torch.zeros(len(x),1),'binary',0,UnifiedConfig().native,BuilderConfig(**kw),torch.ones(5),torch.Generator())
    assert len(t.nodes)==1 and not r['splits']


def test_constant_prior_is_not_counted_twice():
    x=np.zeros((100,3),dtype='float32');y=np.array([0]*80+[1]*20)
    m=UnifiedProgressiveClassifier(UnifiedConfig(n_trees=2,depth=0,updates_per_stage=1)).fit(x,y,eval_set=(x,y))
    assert m.n_estimators_==0
    assert np.allclose(m.predict_proba(x)[:,1],.2,atol=1e-7)


@pytest.mark.parametrize('task',['binary','multiclass','regression'])
def test_literal_additive_predictor_and_reload(task,tmp_path):
    m=fit(task);x,_=data(task);xx=m.preprocessor_.transform_x(x[:20])
    with torch.no_grad():
        manual=m.model_.bias.expand(20,-1).clone()
        for rate,t in zip(m.model_.stage_rates,m.model_.trees):manual+=rate*t(xx,hard=getattr(t,'force_hard',False))[0]
        torch.testing.assert_close(m.model_(xx),manual,atol=1e-7,rtol=1e-6)
    path=tmp_path/'m.pt';m.save(path);other=type(m).load(path)
    assert np.array_equal(m.decision_function(x),other.decision_function(x))


@pytest.mark.parametrize('adaptive',[False,True])
def test_exact_stage_resume_with_stateful_controls(adaptive,tmp_path):
    c=UnifiedConfig(n_trees=3,updates_per_stage=8,depth=1,row_subsample=.7,feature_subsample=.8)
    if adaptive:
        c.native.physics=PhysicsConfig(mode='rlc',heat_capacity=.01,cooling=.001)
        c.native.plasticity=PlasticityConfig(mode='full',stiffness=.2)
        c.native.online=OnlineConfig(enabled=True,concurrent=True,interval=2,window=2)
        c.anchor_min_passes=0.;c.anchor_min_updates=1;c.anchor_require_utility=False
    full=fit(cfg=deepcopy(c));part=fit(cfg=deepcopy(c),stop=1)
    p=tmp_path/'resume.pt';part.save(p);part=type(part).load(p)
    x,y=data();part.continue_fit(x[:220],y[:220],control_set=(x[220:300],y[220:300]),eval_set=(x[300:],y[300:]))
    a,b=full.trainer_.model.state_dict(),part.trainer_.model.state_dict()
    assert a.keys()==b.keys()
    for k in a:assert torch.equal(a[k],b[k]),k
    assert np.array_equal(full.predict_proba(x),part.predict_proba(x))


def test_strict_boosting_preserves_old_parameters_and_moments():
    m=fit(cfg=UnifiedConfig(n_trees=3,updates_per_stage=4,depth=1,age_decay=0.),stop=1)
    t=m.trainer_.model.trees[0];before={k:v.clone() for k,v in t.state_dict().items()}
    p=t.get(t.root_id).value;moment=m.trainer_.optimizer.optimizer.state[p]['exp_avg'].clone()
    x,y=data();m.continue_fit(x[:220],y[:220],control_set=(x[220:300],y[220:300]),eval_set=(x[300:],y[300:]))
    for k,v in before.items():assert torch.equal(v,t.state_dict()[k])
    assert torch.equal(moment,m.trainer_.optimizer.optimizer.state[p]['exp_avg'])
    assert m.trainer_.cache_hits>0


def test_real_row_pool_and_permanent_feature_masks():
    c=UnifiedConfig(n_trees=3,updates_per_stage=4,depth=2,row_subsample=.5,feature_subsample=.4)
    m=fit(cfg=c);x,_=data();xx=m.preprocessor_.transform_x(x[:30])
    for r in m.trainer_.sampling_history:
        assert len(r['rows'])==110 and len(set(r['rows']))==110 and len(r['features'])==2
    for t in m.trainer_.model.trees:
        other=xx.clone();other[:,t.feature_mask==0]+=100
        assert torch.equal(t(xx)[0],t(other)[0])


def test_interaction_masks_persist_during_joint_updates():
    c=UnifiedConfig(n_trees=3,updates_per_stage=4,depth=2);c.native.interaction_groups=((0,1),(2,3))
    m=fit(cfg=c)
    for t in m.trainer_.model.trees:assert tuple(torch.where(t.feature_mask)[0].tolist())==c.native.interaction_groups[t.tree_id%2]


def test_learned_budget_normalization_and_nonzero_gradients():
    m=fit();t=m.trainer_.model.trees[0];c=Regularizers(hierarchy=.3,allocation='learned')
    assert torch.allclose(sum(allocation(t,c).values()),torch.tensor(1.))
    penalty=sum(penalties(m.trainer_.model,None,None,c).values());penalty.backward()
    assert t.depth_logits.grad.abs().sum()>0
    assert any(n.allocation_logit.grad is not None and n.allocation_logit.grad.abs()>0 for n in t.nodes.values())


@pytest.mark.parametrize('field',['leaf_l2','hierarchy','tree_l2','route_balance','child_penalty','feature_l1'])
def test_each_regularizer_changes_live_computation(field):
    m=fit();model=m.trainer_.model;model.requires_grad_(True)
    c=Regularizers(**{field:.2});c.min_child_fraction=.49
    x,_=data();xx=m.preprocessor_.transform_x(x[:50]);z,tr=model(xx,trace=True)
    reg=sum(penalties(model,torch.ones(len(xx)),tr,c).values())
    assert float(reg.detach())>0
    model.zero_grad();reg.backward()
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)>0


def test_zero_penalty_and_no_dilution_when_zero_member_added():
    m=fit();model=m.trainer_.model;c=Regularizers(leaf_l2=.1,hierarchy=.2)
    assert float(sum(penalties(model,None,None,Regularizers()).values()))==0.
    before=sum(penalties(model,None,None,c).values()).detach()
    t=deepcopy(model.trees[0]);t.tree_id=99
    for n in t.nodes.values():n.value.data.zero_()
    model.append(t,.1,m.trainer_.optimizer.optimizer)
    assert torch.equal(before,sum(penalties(model,None,None,c).values()).detach())


@pytest.mark.parametrize('head',['shared','specialized'])
def test_input_dependent_heads_learn_and_serialize(head,tmp_path):
    m=fit('multiclass',UnifiedConfig(n_trees=3,updates_per_stage=5,depth=1,head_mode=head))
    assert m.trainer_.model.attention_weight.abs().sum()>0
    p=tmp_path/'heads.pt';m.save(p);other=type(m).load(p);x,_=data('multiclass')
    assert np.array_equal(m.predict_proba(x),other.predict_proba(x))


def test_monotonic_feature_penalties_and_separate_dropout_are_live():
    c=UnifiedConfig(n_trees=2,updates_per_stage=5,depth=1)
    c.native.monotonicity=((0,0,1),);c.native.monotonicity_penalty=.2
    c.native.feature_penalties=(.1,0,0,0,0);c.native.feature_dropout=.1;c.native.tree_dropout=.1
    m=fit(cfg=c);assert m.trainer_.term_totals['native']>0;assert m.trainer_.fixed_cache is None
    before=deepcopy(m.trainer_.model.state_dict());x,_=data()
    for _ in range(3):m.decision_function(x)
    for k,v in before.items():assert torch.equal(v,m.trainer_.model.state_dict()[k])


def test_anchor_maturity_not_awarded_by_tree_age():
    c=UnifiedConfig(n_trees=2,updates_per_stage=6,depth=1,anchor_min_passes=100.)
    c.native.plasticity=PlasticityConfig(mode='anchor',stiffness=.1)
    m=fit(cfg=c);assert not m.trainer_.admitted
    c.anchor_min_passes=0.;c.anchor_min_updates=1;c.anchor_require_utility=False
    mm=fit(cfg=c);assert mm.trainer_.admitted
    assert all(not a.requires_grad for s in mm.trainer_.plastic.states.values() for a in s['anchors'].values())


def test_thermal_reopening_changes_state_without_reinitializing_weights():
    c=UnifiedConfig(n_trees=3,updates_per_stage=4,depth=1,active_window=1)
    c.native.physics=PhysicsConfig(mode='capacitor',heat_capacity=.001,cooling=0.,charge_gain=100.,max_injection=1.,thaw_temperature=1.1)
    m=fit(cfg=c);t=m.trainer_;old=t.model.trees[0]
    for n in old.nodes.values():n.set_frozen(True)
    before={k:p.detach().clone() for k,p in old.named_parameters()};t.physical.reference=-10.
    x,y=data();ctl=m.preprocessor_.split(x[220:300],y[220:300]);t._observe_and_control(ctl,t.tick+10)
    assert any(e['event']=='thermal_thaw' for e in t.events)
    assert any(not n.frozen for n in old.nodes.values())
    for k,p in old.named_parameters():assert torch.equal(before[k],p)
    assert abs(t.physical.history[-1]['energy_error'])<1e-8


def test_dynamic_growth_actual_deletion_and_optimizer_state_ownership():
    c=UnifiedConfig(n_trees=2,updates_per_stage=12,depth=1)
    c.native.structure.dynamic=True;c.native.structure.structural_gate=True;c.native.structure.max_depth=3
    c.native.structure.grow_every=1;c.native.structure.prune_every=3;c.native.structure.cycle_epochs=8
    c.native.structure.initial_dormant_fraction=0.;c.native.structure.prune_tolerance=.1
    m=fit(cfg=c);t=m.trainer_;assert any(e['event']=='grow' for e in t.events)
    n=next(n for n in t.model.iter_nodes() if not n.is_leaf);tree=t.model.get_tree(n.tree_id)
    ids=set(tree.descendants(n.node_id));before=t.model.tensor_bytes()
    for k in ids:tree.get(k).value.data.zero_()
    x,y=data();ctl=m.preprocessor_.split(x[220:300],y[220:300])
    t._apply_structure([StructuralAction('prune',n.node_id,t.model.topology_version)],ctl,t.tick+1)
    assert ids.isdisjoint(t.model.node_map()) and ids.isdisjoint(t.collector.state)
    assert t.model.tensor_bytes()<before;t.optimizer.assert_ownership(t.model)


def test_online_trials_complete_with_sufficient_observation_window():
    c=UnifiedConfig(n_trees=2,updates_per_stage=16,depth=1,anchor_min_passes=0.,anchor_min_updates=1,anchor_require_utility=False)
    c.native.plasticity=PlasticityConfig(mode='anchor',stiffness=.2)
    c.native.online=OnlineConfig(enabled=True,interval=2,window=2,cooldown=1,concurrent=True,exploration=1.,deformation_source='parameters')
    m=fit(cfg=c);assert m.trainer_.scheduler.counts.sum()>0


@pytest.mark.parametrize('task',['binary','multiclass','regression'])
def test_refit_acceptance_never_worsens_declared_objective(task):
    c=UnifiedConfig(n_trees=2,updates_per_stage=4,depth=1,refit_every=1);c.regularizers.hierarchy=.05
    m=fit(task,c)
    assert m.trainer_.refit_history
    for r in m.trainer_.refit_history:
        assert r['objective_after']<=r['objective_before']+1e-7
        assert r['solve_residual']<1e-7


def test_schedules_and_age_learning_rates():
    c=UnifiedConfig(n_trees=3,updates_per_stage=4,depth=1,age_decay=.2)
    c.native.schedules={'learning_rate':ScheduleConfig('linear',.02,.005)}
    c.regularizers.hierarchy=.1;c.regularizer_schedule=ScheduleConfig('oscillatory',0,2,cycles=2)
    m=fit(cfg=c);rates={g['owner']:g['lr'] for g in m.trainer_.optimizer.optimizer.param_groups}
    assert rates['2:0']>rates['1:0']>rates['0:0']
    assert m.trainer_.optimizer.base_lr==pytest.approx(.005)
    assert m.trainer_.term_totals['hierarchy']>0


@pytest.mark.parametrize('kw',[{'row_subsample':0},{'feature_subsample':2},{'age_decay':float('nan')},{'n_trees':0},{'newton_l2':-1}])
def test_invalid_config_rejected(kw):
    with pytest.raises(ValueError):UnifiedConfig(**kw)


def test_evidence_growth_policy_uses_leaf_reducibility_metrics():
    c=UnifiedConfig(n_trees=1,updates_per_stage=8,depth=1)
    c.native.collect_metrics=True
    c.native.structure.dynamic=True
    c.native.structure.initial_depth=0
    c.native.structure.max_depth=2
    c.native.structure.max_nodes=15
    c.native.structure.grow_every=1
    c.native.structure.prune_every=20
    c.native.structure.initial_dormant_fraction=0.
    c.native.structure.growth_policy="evidence"
    m=fit(cfg=c)
    latest=m.trainer_.tracker.latest()
    assert latest
    assert all(o.exploration_score>=0 and o.budget_score>=0 for o in latest.values())
    assert any(e["event"]=="grow" for e in m.trainer_.events)
