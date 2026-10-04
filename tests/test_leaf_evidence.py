import numpy as np
import torch

from torchboost.adaptive.leaf_evidence import leaf_evidence


def test_reducible_signal_earns_more_budget_than_noise():
    r=np.random.default_rng(8);n=800
    x=torch.tensor(r.normal(size=(n,6)),dtype=torch.float32)
    p=1/(1+np.exp(-2*x[:,0].numpy()))
    structured=torch.tensor(r.binomial(1,p));noise=torch.tensor(r.binomial(1,.5,size=n))
    z=torch.zeros(n,1);w=torch.ones(n);reach=torch.ones(n)
    a=leaf_evidence(x,z,structured,w,reach,"binary");b=leaf_evidence(x,z,noise,w,reach,"binary")
    assert a.reducible_loss>b.reducible_loss
    assert a.explainable_fraction>b.explainable_fraction
    assert a.budget_score>b.budget_score


def test_tiny_leaf_does_not_spend_without_evidence():
    r=np.random.default_rng(9);n=500
    x=torch.tensor(r.normal(size=(n,4)),dtype=torch.float32);y=(x[:,0]>0).long()
    z=torch.zeros(n,1);w=torch.ones(n);reach=torch.zeros(n);reach[:8]=1
    tiny=leaf_evidence(x,z,y,w,reach,"binary")
    assert tiny.effective_n<12
    assert tiny.budget_score==0.
