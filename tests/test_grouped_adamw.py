from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import pytest
import torch
from grouped_adamw import install_grouped_adamw

@pytest.mark.parametrize('different_lr',[False,True])
def test_grouped_adamw_matches_distinct_local_groups(different_lr):
    gen=torch.Generator().manual_seed(732)
    a=[torch.nn.Parameter(torch.randn(3,5,generator=gen,dtype=torch.float64)) for _ in range(9)]
    b=[torch.nn.Parameter(p.detach().clone()) for p in a]
    def make(ps):
        return torch.optim.AdamW([dict(params=[p],owner=str(i),names=[str(i)],lr=.01*(i+1) if different_lr else .01) for i,p in enumerate(ps)],weight_decay=.001)
    oa,ob=make(a),make(b);install_grouped_adamw(ob)
    original=ob.param_groups
    for step in range(12):
        for index,(p,q) in enumerate(zip(a,b)):
            # Include intermittent unused parameters, as in conditional routing.
            grad=torch.randn(p.shape,generator=gen,dtype=p.dtype)
            p.grad=grad if (step+index)%3 else None
            q.grad=grad.clone() if p.grad is not None else None
        oa.step();ob.step()
        assert ob.param_groups is original
        assert [g['owner'] for g in ob.param_groups]==[str(i) for i in range(9)]
        for p,q in zip(a,b):torch.testing.assert_close(p,q,atol=1e-12,rtol=1e-12)
    for p,q in zip(a,b):
        for k,v in oa.state[p].items():torch.testing.assert_close(v,ob.state[q][k],atol=1e-12,rtol=1e-12)
    # New and deleted state retain ordinary optimizer ownership semantics.
    removed=b.pop();ob.state.pop(removed);ob.param_groups.pop()
    added=torch.nn.Parameter(torch.zeros(3,5,dtype=torch.float64))
    ob.add_param_group(dict(params=[added],owner='new',names=['new']))
    added.grad=torch.ones_like(added);ob.step()
    assert removed not in ob.state and added in ob.state
    assert ob.param_groups[-1]['owner']=='new'

def test_grouped_adamw_restores_groups_on_exception():
    p=torch.nn.Parameter(torch.zeros(2));o=torch.optim.AdamW([dict(params=[p],owner='node',names=['p'])])
    install_grouped_adamw(o);original=o.param_groups
    def fail():raise RuntimeError('deliberate closure failure')
    with pytest.raises(RuntimeError):o.step(fail)
    assert o.param_groups is original
