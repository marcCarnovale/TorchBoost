import torch
from torch import nn

from experiments.higgs_differentiable_architecture import (
    build_supernet,
    parameter_partition,
)
from torchboost.adaptive.architecture_corners import CompositionalTreeNetwork
from torchboost.adaptive.architecture_regularization import (
    ArchitectureRegularization,
    architecture_state,
    differentiable_architecture_penalty,
)


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


def make_anchor():
    torch.manual_seed(23)
    ref=TinyMLP().eval()
    return CompositionalTreeNetwork.from_mlp(ref,max_tree_depth=2,seed=47).eval()


def test_supernet_growth_is_function_preserving_with_nonzero_soft_gates():
    anchor=make_anchor()
    supernet=build_supernet(anchor).eval()
    x=torch.randn(128,4)
    with torch.no_grad():
        expected=anchor(x)
        actual=supernet(x)
    assert torch.allclose(actual,expected,atol=2e-6,rtol=2e-6)
    state=architecture_state(supernet)
    assert all(abs(row["gate"] - 1.0) < 1e-7 for row in state["layers"])


def test_architecture_gate_receives_predictive_gradient_after_residual_moves():
    model=build_supernet(make_anchor()).train()
    # Make one zero-at-birth child nonzero, simulating one predictive warmup.
    layer=model.layers[-1]
    tree=layer.forest.trees[0]
    child=tree.get(layer.root.children_ids[0])
    with torch.no_grad():
        child.value.fill_(0.05)
    x=torch.randn(64,4)
    y=(x[:,0]>0).float()
    loss=nn.BCEWithLogitsLoss()(model(x),y)
    loss.backward()
    assert layer.architecture_logit.grad is not None
    assert torch.isfinite(layer.architecture_logit.grad)
    assert layer.architecture_logit.grad.abs()>0


def test_parameter_partition_separates_architecture_from_predictive_weights():
    model=build_supernet(make_anchor())
    weights,architecture=parameter_partition(model)
    assert len(architecture)==len(model.layers)
    assert all(p.requires_grad for p in architecture)
    assert not {id(p) for p in weights} & {id(p) for p in architecture}


def test_architecture_penalty_is_differentiable_and_charges_open_gates():
    model=build_supernet(make_anchor())
    cfg=ArchitectureRegularization(
        gate_l1=1e-3,gate_entropy=1e-4,residual_l2=1e-5,routing_l1=1e-5
    )
    penalty,parts=differentiable_architecture_penalty(model,cfg)
    assert penalty.requires_grad
    assert float(parts["gate_l1_raw"])>0
    penalty.backward()
    grads=[layer.architecture_logit.grad for layer in model.layers]
    assert all(g is not None and torch.isfinite(g) for g in grads)


def test_complexity_penalty_does_not_change_zero_residual_function():
    model=build_supernet(make_anchor()).eval()
    x=torch.randn(96,4)
    with torch.no_grad():
        before=model(x)
        for layer in model.layers:
            layer.architecture_logit.add_(1.75)
        after=model(x)
    # At birth every residual branch is exactly zero, so architecture scales
    # may start at unit strength without perturbing the calibrated MLP predictor.
    assert torch.allclose(before,after,atol=2e-6,rtol=2e-6)
