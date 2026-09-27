import numpy as np
import torch
from torch import nn

from experiments.higgs_endpoint_calibration import MLP_BATCH, MLP_EPOCHS, NTRAIN
from torchboost.adaptive.architecture_corners import CompositionalTreeNetwork


class MiniMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(4,8),nn.ReLU(),nn.Dropout(.1),
            nn.Linear(8,8),nn.ReLU(),nn.Dropout(.1),
            nn.Linear(8,1),
        )
    def forward(self,x):
        return self.net(x).squeeze(1)


def test_endpoint_contract_matches_canonical_higgs_mlp_budget():
    assert NTRAIN==500_000
    assert MLP_EPOCHS==20
    assert MLP_BATCH==4096


def test_depth_zero_corner_matches_mlp_forward_and_one_lockstep_update():
    torch.manual_seed(41)
    ref=MiniMLP()
    corner=CompositionalTreeNetwork.from_mlp(ref,max_tree_depth=2,seed=41)
    x=torch.randn(96,4);y=(x[:,0]+x[:,1]>0).float()
    ref_opt=torch.optim.AdamW(ref.parameters(),lr=1e-3,weight_decay=1e-5)
    corner_params=[p for p in corner.parameters() if p.requires_grad]
    corner_opt=torch.optim.AdamW(corner_params,lr=1e-3,weight_decay=1e-5)
    loss=nn.BCEWithLogitsLoss()

    ref.train();corner.train()
    state=torch.get_rng_state()
    zr=ref(x)
    torch.set_rng_state(state)
    zc=corner(x)
    assert torch.allclose(zr,zc,atol=2e-6,rtol=2e-6)

    lr=loss(zr,y);lc=loss(zc,y)
    ref_opt.zero_grad(set_to_none=True);corner_opt.zero_grad(set_to_none=True)
    lr.backward();lc.backward()
    torch.nn.utils.clip_grad_norm_(ref.parameters(),10.)
    torch.nn.utils.clip_grad_norm_(corner_params,10.)
    ref_opt.step();corner_opt.step()

    ref.eval();corner.eval()
    with torch.no_grad():
        assert torch.allclose(ref(x),corner(x),atol=3e-6,rtol=3e-6)
    assert sum(p.numel() for p in ref.parameters()) == sum(
        p.numel() for p in corner_params
    )
