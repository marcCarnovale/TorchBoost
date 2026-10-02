import math

import torch
from torch import nn

from experiments.higgs_residual_scale_sweep import build_fixed
from torchboost.adaptive.architecture_corners import CompositionalTreeNetwork


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(4,8),nn.ReLU(),nn.Dropout(.1),
            nn.Linear(8,8),nn.ReLU(),nn.Dropout(.1),
            nn.Linear(8,1),
        )
    def forward(self,x):
        return self.net(x).squeeze(1)


def anchor():
    torch.manual_seed(31)
    return CompositionalTreeNetwork.from_mlp(
        TinyMLP().eval(),max_tree_depth=2,seed=59
    ).eval()


def test_default_scale_matches_historical_sigmoid_minus_two():
    model=anchor()
    model.layers[0].grow_one_level()
    expected=1.0/(1.0+math.exp(2.0))
    assert abs(float(model.layers[0].architecture_gate)-expected)<1e-7


def test_fixed_scale_setter_is_exact_and_can_freeze_gate():
    model=anchor()
    model.layers[0].grow_one_level()
    model.layers[0].set_architecture_scale(.24,learnable=False)
    assert abs(float(model.layers[0].architecture_gate)-.24)<1e-7
    assert not model.layers[0].architecture_logit.requires_grad


def test_scale_sweep_models_preserve_anchor_function_at_birth():
    base=anchor()
    x=torch.randn(128,4)
    with torch.no_grad():
        expected=base(x)
        for scale in (.06,1.0/(1.0+math.exp(2.0)),.24,.5):
            model=build_fixed(base,scale).eval()
            assert torch.allclose(model(x),expected,atol=2e-6,rtol=2e-6)
