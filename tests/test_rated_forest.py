import torch

from torchboost.adaptive.progressive import ProgressiveSum, _new_packed
from torchboost.adaptive.rated_forest import (
    RatedAdaptiveForest,
    materialize_progressive_sum,
)
from torchboost.adaptive.training import model_snapshot, restore_model


def make_progressive(seed=7):
    torch.manual_seed(seed)
    bias = torch.tensor([0.13])
    model = ProgressiveSum(bias, learn_rates=True)
    for j, rate in enumerate((0.25, -0.4, 0.8)):
        tree, _ = _new_packed(5, 1, type("Cfg", (), {
            "depth": 3, "learning_rate": .01, "routing_temperature": 1.,
            "readout": "residual", "batch_size": 32, "cart_strength": 4.,
        })(), seed + j)
        with torch.no_grad():
            tree.values.normal_(generator=torch.Generator().manual_seed(seed + 20 + j))
            tree.routing_weight.normal_(generator=torch.Generator().manual_seed(seed + 40 + j))
            tree.routing_bias.normal_(generator=torch.Generator().manual_seed(seed + 60 + j))
            tree.bias.fill_(0.03 * (j + 1))
        model.append(tree, rate)
    return model


def test_materialization_is_prediction_exact_and_rates_remain_trainable():
    progressive = make_progressive()
    native = materialize_progressive_sum(progressive, learn_rates=True)
    x = torch.randn(37, 5, generator=torch.Generator().manual_seed(101))
    assert torch.allclose(progressive(x), native(x), atol=2e-6, rtol=2e-6)
    assert native.rates.requires_grad
    assert torch.allclose(native.rates, progressive.rates)


def test_rated_native_snapshot_restore_keeps_family_rates_and_predictions():
    progressive = make_progressive(11)
    native = materialize_progressive_sum(progressive)
    snap = model_snapshot(native)
    restored = restore_model(snap, native.input_dim, native.output_dim, native.config)
    assert isinstance(restored, RatedAdaptiveForest)
    x = torch.randn(29, 5, generator=torch.Generator().manual_seed(404))
    assert torch.allclose(native(x), restored(x), atol=1e-7, rtol=1e-7)
    assert restored.rates.requires_grad


def test_effective_count_is_invariant_to_tree_rate_rescaling_pair():
    progressive = make_progressive(19)
    native = materialize_progressive_sum(progressive)
    x = torch.randn(31, 5, generator=torch.Generator().manual_seed(505))
    before = native.effective_tree_counts(x)
    with torch.no_grad():
        # Scale the entire first tree output, including all residual values.
        for node in native.trees[0].nodes.values():
            node.value.mul_(2)
        native.rates[0].div_(2)
    after = native.effective_tree_counts(x)
    assert abs(before["participation"] - after["participation"]) < 1e-6
    assert abs(before["entropy"] - after["entropy"]) < 1e-6
