import numpy as np
import torch
from torchboost.adaptive.progressive import (
    ProgressiveConfig, ProgressiveSum, ProgressiveTreeClassifier,
    _contribution_count_stats,
)

def test_learned_tree_rates_are_parameters_and_keep_append_values():
    m=ProgressiveSum(torch.zeros(1),learn_rates=True)
    m.append(torch.nn.Linear(2,1,bias=False),.4)
    m.append(torch.nn.Linear(2,1,bias=False),.2)
    assert isinstance(m.rates,torch.nn.Parameter)
    assert m.rates.requires_grad
    assert torch.allclose(m.rates.detach(),torch.tensor([.4,.2]))

def test_effective_count_is_scale_free_and_has_expected_limits():
    c=torch.tensor([3.,0.,0.])
    ne,ent=_contribution_count_stats(c)
    assert torch.allclose(ne,torch.tensor(1.))
    assert torch.allclose(ent,torch.tensor(1.))
    c=torch.ones(5)
    ne,ent=_contribution_count_stats(c)
    assert torch.allclose(ne,torch.tensor(5.))
    assert torch.allclose(ent,torch.tensor(5.))

def test_progressive_classifier_learns_rates_and_records_soft_count():
    r=np.random.default_rng(2);x=r.normal(size=(180,5)).astype("float32")
    y=(x[:,0]+.35*x[:,1]>.1).astype(int)
    cfg=ProgressiveConfig(n_trees=3,depth=2,stage_updates=6,batch_size=64,
        cart_value_updates=2,patience_stages=4,learn_tree_rates=True,
        tree_rate_l2=1e-4,tree_count_pressure=1e-4,tree_count_start_stage=2,
        random_state=3)
    m=ProgressiveTreeClassifier(cfg).fit(x[:140],y[:140],eval_set=(x[140:],y[140:]))
    assert isinstance(m.model_.rates,torch.nn.Parameter)
    assert m.history_
    for h in m.history_:
        assert 0 <= h["effective_tree_count"] <= h["trees"]+1e-5
        assert len(h["tree_contribution_rms"]) == h["trees"]
