from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification, make_regression
from torchboost.adaptive import AdaptiveForest, ForestConfig, StructureConfig
from torchboost.adaptive.single_tree import (SingleTreeConfig, PackedSingleTree, SingleTreeClassifier,
    SingleTreeRegressor, refit_readout)
from torchboost.adaptive.single_tree_diagnostics import diagnose_single_tree

torch.set_num_threads(1)

def native(depth, arity, outputs):
    cfg = SingleTreeConfig(depth=depth, arity=arity, epochs=4).forest_config()
    f = AdaptiveForest(5, outputs, cfg).double()
    g = torch.Generator().manual_seed(91)
    with torch.no_grad():
        for n in f.iter_nodes():
            n.value.copy_(torch.randn(n.value.shape, generator=g, dtype=n.value.dtype))
    return f

@pytest.mark.parametrize('depth,arity', [(0,2),(1,2),(3,2),(2,3),(2,4)])
@pytest.mark.parametrize('outputs', [1,3])
@pytest.mark.parametrize('readout', ['residual','leaf'])
def test_packing_soft_hard_native_roundtrip(depth, arity, outputs, readout):
    f = native(depth, arity, outputs)
    p = PackedSingleTree(f, readout=readout)
    x = torch.randn(13, 5, dtype=torch.float64)
    torch.testing.assert_close(p(x), f(x), atol=1e-11, rtol=1e-11)
    restored = p.to_native()
    torch.testing.assert_close(p(x), restored(x), atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(p(x, hard=True), restored(x, hard=True), atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(p.routing(x)[0][:, p.n_internal:].sum(1), torch.ones(len(x), dtype=x.dtype))

@pytest.mark.parametrize('depth,arity', [(1,2),(3,2),(2,3)])
def test_gradient_equivalence(depth, arity):
    f = native(depth, arity, 3)
    p = PackedSingleTree(f)
    x = torch.randn(11, 5, dtype=torch.float64)
    f(x).square().mean().backward(); p(x).square().mean().backward()
    nodes = list(f.trees[0].nodes.values())
    torch.testing.assert_close(torch.stack([n.routing_weight.grad for n in nodes if not n.is_leaf]), p.routing_weight.grad)
    torch.testing.assert_close(torch.stack([n.value.grad for n in nodes]), p.values.grad)
    torch.testing.assert_close(f.bias.grad, p.bias.grad)

@pytest.mark.parametrize('outputs', [1,3])
def test_one_tree_attention_is_a_noop(outputs):
    cfg = SingleTreeConfig(depth=2).forest_config(); cfg.aggregation='attention'; cfg.head_mode='specialized'
    f = AdaptiveForest(5, outputs, cfg)
    x = torch.randn(20, 5)
    with torch.no_grad():
        for n in f.iter_nodes(): n.value.normal_()
    before = f(x).detach()
    f(x).square().sum().backward()
    assert torch.count_nonzero(f.attention_weight.grad) == 0
    with torch.no_grad(): f.attention_weight.normal_(0,100); f.attention_bias.fill_(1000)
    torch.testing.assert_close(f(x), before, rtol=0, atol=0)

@pytest.mark.parametrize('classification', [True,False])
@pytest.mark.parametrize('readout', ['residual','leaf'])
def test_training_serialization_refit_and_diagnostics(tmp_path, classification, readout):
    if classification:
        x,y=make_classification(n_samples=100,n_features=5,n_informative=3,random_state=42)
        cls=SingleTreeClassifier
    else:
        x,y=make_regression(n_samples=100,n_features=5,n_targets=2,random_state=42)
        cls=SingleTreeRegressor
    c=SingleTreeConfig(depth=2, epochs=12, readout=readout, refit_every=8, refit_iterations=6, logit_penalty=.001)
    m=cls(c).fit(x[:70],y[:70],eval_set=(x[70:],y[70:]))
    states={k:v.clone() for k,v in m.model_.state_dict().items()}
    diag=diagnose_single_tree(m,x[:70],y[:70],x[70:],y[70:])
    assert diag['trees']==1 and diag['leaves']==4
    for k,v in m.model_.state_dict().items(): torch.testing.assert_close(v,states[k],rtol=0,atol=0)
    assert m.refits_[0]['after'] <= m.refits_[0]['before']
    path=tmp_path/'m.pt'; m.save(path); re=cls.load(path)
    np.testing.assert_array_equal(m.predict(x), re.predict(x))
    torch.testing.assert_close(m.model_(m.preprocessor_.transform_x(x)), m.native_model()(m.preprocessor_.transform_x(x)))

@pytest.mark.parametrize('change',[dict(depth=-1),dict(arity=1),dict(temperature=0),dict(final_learning_rate_ratio=0),dict(depth=13),dict(initializer='cart',readout='residual')])
def test_invalid_config(change):
    with pytest.raises(ValueError): SingleTreeConfig(**change)

@pytest.mark.parametrize('readout',['residual','leaf'])
def test_determinism(readout):
    x,y=make_classification(n_samples=80,n_features=5,random_state=5)
    c=SingleTreeConfig(depth=2,epochs=8,readout=readout)
    a=SingleTreeClassifier(c).fit(x,y);b=SingleTreeClassifier(c).fit(x,y)
    np.testing.assert_array_equal(a.predict_proba(x),b.predict_proba(x))

def test_cart_trains_and_rank_bound():
    x,y=make_classification(n_samples=100,n_features=5,random_state=42)
    m=SingleTreeClassifier(SingleTreeConfig(depth=2,epochs=8,readout='leaf',initializer='cart')).fit(x[:70],y[:70],eval_set=(x[70:],y[70:]))
    d=diagnose_single_tree(m,x[:70],y[:70],x[70:],y[70:])
    assert d['centered_leaf_design_rank_rtol_1e6'] <= 3

def test_multiple_trees_rejected():
    f=AdaptiveForest(5,1,ForestConfig(n_trees=2))
    with pytest.raises(ValueError): PackedSingleTree(f)

@pytest.mark.parametrize('depth',[0,1,4,7])
def test_balanced_initialization_weighted_flow_and_label_independence(depth):
    from torchboost.adaptive.single_tree import balanced_initialize
    from torchboost.adaptive.data import Preprocessor
    rng=np.random.default_rng(123)
    x=rng.normal(size=(101,5)); y=(x[:,0]>0).astype(int); weights=rng.uniform(.1,3,size=len(y))
    pp=Preprocessor(); pp.fit(x,y,classification=True,weights=weights)
    train=pp.split(x,y,weights)
    a=PackedSingleTree(native(depth,2,1)).float(); b=deepcopy(a)
    balanced_initialize(a,train)
    train.y=1-train.y
    balanced_initialize(b,train)
    mass,_=a.routing(train.x)
    occupancy=(mass[:,a.n_internal:]*train.weight[:,None]).sum(0)/train.weight.sum()
    torch.testing.assert_close(occupancy,torch.full_like(occupancy,1/a.n_leaves),atol=2e-7,rtol=2e-5)
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)


def test_routing_telemetry_does_not_change_learning():
    x,y=make_classification(n_samples=100,n_features=5,n_informative=3,random_state=14)
    a=SingleTreeClassifier(SingleTreeConfig(depth=3,epochs=12,initializer='balanced')).fit(x[:70],y[:70],eval_set=(x[70:],y[70:]))
    b=SingleTreeClassifier(SingleTreeConfig(depth=3,epochs=12,initializer='balanced',routing_diagnostics_every=4)).fit(x[:70],y[:70],eval_set=(x[70:],y[70:]))
    for k,v in a.model_.state_dict().items():torch.testing.assert_close(v,b.model_.state_dict()[k],atol=0,rtol=0)
    assert b.history_[0]['routing']['effective_leaves']==pytest.approx(8,abs=1e-5)
    assert b.history_[-1]['routing']['levels'][0]['depth']==0


def test_single_pass_balance_matches_original_values_and_gradients():
    import math
    from torchboost.adaptive.single_tree import _fit_objective,_center_logits
    from torchboost.adaptive.objectives import Objective
    c=SingleTreeConfig(depth=3,route_balance=.1,logit_penalty=.01)
    a=PackedSingleTree(native(3,2,3)); b=deepcopy(a)
    x=torch.randn(41,5,dtype=torch.float64);y=torch.arange(41)%3;w=torch.rand(41,dtype=torch.float64)+.1;o=Objective('multiclass',3)
    l=_fit_objective(a,x,y,w,o,c);l.backward()
    z=b(x);old=o.weighted_loss(z,y,w)+c.logit_penalty*(_center_logits(z,o.task).square().mean(1)*w).sum()/w.sum()
    mass,p=b.routing(x);reach=mass[:,:b.n_internal]*w[:,None]
    use=(reach[...,None]*p).sum(0)/reach.sum(0).clamp_min(1e-12)[:,None]
    kl=-use.clamp_min(1e-8).log().mean(1)-math.log(b.arity)
    old=old+c.route_balance*(kl*reach.sum(0)/w.sum()).sum()/b.depth;old.backward()
    torch.testing.assert_close(l,old,atol=1e-12,rtol=1e-12)
    for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa.grad,pb.grad,atol=1e-12,rtol=1e-12)


def test_positive_tail_replays_schedule_and_preserves_boundary_state():
    from torchboost.adaptive.single_tree import model_state_digest
    x,y=make_classification(n_samples=100,n_features=5,n_informative=3,random_state=7)
    c=SingleTreeConfig(depth=3,epochs=12,random_state=3)
    short=SingleTreeClassifier(c).fit(x[:70],y[:70],eval_set=(x[70:],y[70:]))
    long=SingleTreeClassifier(replace(c,epochs=36,schedule_epochs=12)).fit(x[:70],y[:70],eval_set=(x[70:],y[70:]))
    short.model_.load_state_dict(short.last_state_)
    boundary=next(h for h in long.history_ if h['epoch']==12)
    assert boundary['schedule_boundary_state_sha256']==model_state_digest(short.model_)
    for a,b in zip(short.history_,long.history_):
        for k in a:assert a[k]==b[k],k
    assert all(h['learning_rate']==c.learning_rate*c.final_learning_rate_ratio for h in long.history_ if h['epoch']>=12)
    assert len(long.history_)>len(short.history_)


@pytest.mark.parametrize('value',[0,-1,5.5,20])
def test_invalid_schedule_horizon(value):
    with pytest.raises(ValueError):SingleTreeConfig(epochs=12,schedule_epochs=value)


def test_unfitted_probabilities_use_standard_fitted_guard():
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):SingleTreeClassifier().predict_proba(np.ones((2,3)))
