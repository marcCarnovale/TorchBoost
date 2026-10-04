import torch
from torch import nn

from experiments.higgs_mlp_architecture_search import mutate
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
    torch.manual_seed(17)
    ref=TinyMLP().eval()
    return CompositionalTreeNetwork.from_mlp(ref,max_tree_depth=2,seed=31).eval()


def test_first_search_mutations_are_function_preserving_at_birth():
    base=anchor()
    x=torch.randn(128,4)
    with torch.no_grad():
        expected=base(x)
        for variant in ("control","last_residual","last_full","all_residual"):
            candidate=mutate(base,variant).eval()
            actual=candidate(x)
            assert torch.allclose(actual,expected,atol=2e-6,rtol=2e-6),variant


def test_residual_only_mutation_trains_only_new_specialist_packet():
    model=mutate(anchor(),"last_residual")
    assert not model.head.weight.requires_grad
    assert not model.layers[0].weight.requires_grad
    layer=model.layers[-1]
    root=layer.root
    assert not root.value.requires_grad
    assert not root.linear_value.requires_grad
    assert root.routing_weight.requires_grad
    assert root.routing_bias.requires_grad
    children=[layer.forest.trees[0].get(i) for i in root.children_ids]
    assert children
    assert all(c.value.requires_grad and c.linear_value.requires_grad for c in children)


def test_full_mutation_keeps_end_to_end_anchor_trainable():
    model=mutate(anchor(),"last_full")
    assert model.head.weight.requires_grad
    assert model.layers[0].weight.requires_grad
    layer=model.layers[-1]
    assert not layer.weight.requires_grad
    assert layer.root.value.requires_grad
    assert layer.root.linear_value.requires_grad
