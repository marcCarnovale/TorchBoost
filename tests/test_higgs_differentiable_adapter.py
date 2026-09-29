import math

import torch
from torch import nn

from experiments.higgs_differentiable_adapter import (
    INITIAL_SCALE,
    build_adapter,
    partition,
)
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
    torch.manual_seed(41)
    return CompositionalTreeNetwork.from_mlp(
        TinyMLP().eval(),max_tree_depth=2,seed=71
    ).eval()


def test_adapter_is_function_preserving_and_perturbative():
    base=anchor();learned=build_adapter(base,learn_scales=True).eval()
    x=torch.randn(128,4)
    with torch.no_grad():
        assert torch.allclose(learned(x),base(x),atol=2e-6,rtol=2e-6)
    assert abs(INITIAL_SCALE-1/(1+math.exp(2)))<1e-12
    assert all(
        abs(float(layer.architecture_gate)-INITIAL_SCALE)<1e-7
        for layer in learned.layers
    )


def test_adapter_freezes_inherited_backbone_and_head():
    model=build_adapter(anchor(),learn_scales=True)
    assert not model.head.weight.requires_grad
    assert not model.head.bias.requires_grad
    for layer in model.layers:
        assert not layer.weight.requires_grad
        assert not layer.bias.requires_grad
        assert not layer.root.value.requires_grad
        assert not layer.root.linear_value.requires_grad


def test_learned_adapter_partition_is_only_residuals_and_five_scales():
    model=build_adapter(anchor(),learn_scales=True)
    residual,scales=partition(model)
    assert len(scales)==len(model.layers)
    assert residual
    trainable={name for name,p in model.named_parameters() if p.requires_grad}
    assert all(
        name.endswith("architecture_logit")
        or ".routing_weight" in name
        or ".routing_bias" in name
        or ".value" in name
        or ".linear_value" in name
        for name in trainable
    )


def test_fixed_adapter_has_no_trainable_scale_parameters():
    model=build_adapter(anchor(),learn_scales=False)
    residual,scales=partition(model)
    assert residual
    assert scales==[]
