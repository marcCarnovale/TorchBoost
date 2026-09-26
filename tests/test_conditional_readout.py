from copy import deepcopy
import numpy as np
import torch
from torchboost.adaptive.config import ForestConfig,StructureConfig
from torchboost.adaptive.forest import AdaptiveForest
from torchboost.adaptive.data import DataSplit
from experiments.conditional_readout import basis,readout,ridge_statistics,apply_ridge

def tree():
    c=ForestConfig(n_trees=1,aggregation='mean',residual_weights=False,
        structure=StructureConfig(initial_depth=0,max_depth=2,max_nodes=15,structural_gate=False))
    m=AdaptiveForest(3,1,c)
    with torch.no_grad():
        m.bias.fill_(.3)
        for i,n in enumerate(m.iter_nodes()):n.value.fill_(i*.07)
    return m

def test_linear_design_exactly_represents_native_predictor():
    m=tree();x=torch.randn(37,3)
    torch.testing.assert_close(basis(m,x)@readout(m),m(x),atol=2e-6,rtol=2e-6)

def test_refit_reduces_its_declared_objective_and_does_not_change_routing():
    m=tree();x=torch.randn(90,3);y=x[:,0,None]+.1*x[:,1,None];w=torch.ones(90);d=DataSplit(x,y,w)
    stats=ridge_statistics(m,d,batch=13);n,delta=apply_ridge(m,stats,.01)
    before=.5*float((m(x)-y).square().mean().detach())
    after=.5*float((n(x)-y).square().mean().detach())+.5*.01*delta**2
    assert after<=before+1e-6
    for a,b in zip(m.iter_nodes(),n.iter_nodes()):
        if a.routing_weight is not None:assert torch.equal(a.routing_weight,b.routing_weight)

def test_large_penalty_is_a_noop_limit():
    m=tree();x=torch.randn(20,3);d=DataSplit(x,torch.randn(20,1),torch.ones(20));s=ridge_statistics(m,d)
    other,_=apply_ridge(m,s,1e12);torch.testing.assert_close(m(x),other(x),rtol=1e-6,atol=1e-6)

def test_batching_statistics_is_invariant():
    m=tree();x=torch.randn(23,3);d=DataSplit(x,torch.randn(23,1),torch.linspace(.1,2,23))
    a=ridge_statistics(m,d,7);b=ridge_statistics(m,d,23)
    for x,y in zip(a,b):np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-7)
