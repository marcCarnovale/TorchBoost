"""Characterize the preserved baseline, including known defects (not endorsements)."""
import copy

import pytest
import torch

from torchboost import SoftTree, TorchBoostModel, train_torchboost


@pytest.mark.parametrize("task, outputs", [("regression", 1), ("binary_classification", 1),
    ("multiclass_classification", 3), ("multitarget", 3)])
def test_legacy_shapes_and_eval_state_roundtrip(task, outputs):
    model = TorchBoostModel(2, 4, 2, task_type=task, num_classes=3).eval()
    x = torch.randn(8, 4)
    result = model(x)
    assert result.shape == (8, outputs)
    state = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    torch.testing.assert_close(result, model(x), atol=0, rtol=0)


def test_disconnected_pruning_alpha_is_characterized():
    tree = SoftTree(2, 3).eval()
    x = torch.randn(4, 2)
    before = tree(x).detach()
    penalty = tree.pruning_regularization().item()
    with torch.no_grad():
        tree.alpha.fill_(100.)
    torch.testing.assert_close(before, tree(x))
    assert tree.pruning_regularization().item() < penalty
    # Lower penalty is NOT evidence that any topology was removed.
    assert len(tree.leaf_values) == 8


def test_dead_controls_are_characterized():
    model = TorchBoostModel(2, 2, 2, dropout_rate=0).eval()
    x = torch.randn(6, 2)
    before, reg = model(x).detach(), model.temperature_regularization().detach()
    model.hardening_rate = 999
    model.use_hessian = True
    model.temp_reg_weight = 999
    torch.testing.assert_close(before, model(x))
    torch.testing.assert_close(reg, model.temperature_regularization())


def test_unconstrained_residual_weights_are_characterized():
    model = TorchBoostModel(2, 2, 2).eval()
    with torch.no_grad():
        model.residual_weights.fill_(2.)
    assert (model.residual_weights > 1).all()


def test_missing_value_attention_bug_is_characterized():
    model = TorchBoostModel(2, 2, 2).eval()
    x = torch.tensor([[float("nan"), 1.], [0., 2.]])
    assert torch.isfinite(model.trees[0](x)).all()
    assert torch.isnan(model(x)[0]).all()


def test_legacy_regularizer_gradients_and_weight_proxy():
    model = TorchBoostModel(2, 2, 2).eval()
    model.regularization().backward()
    assert model.trees[0].leaf_values.grad.abs().sum() > 0
    assert model.trees[0].alpha.grad.abs().sum() > 0
    assert abs(model.feature_importance().sum()-1) < 1e-6


def test_legacy_diversity_uses_second_forward(monkeypatch):
    model = TorchBoostModel(2, 2, 2, dropout_rate=0)
    counts = [0]
    hook = model.trees[0].register_forward_hook(lambda *args: counts.__setitem__(0, counts[0]+1))
    x, y = torch.randn(10, 2), torch.randn(10)
    train_torchboost(model, x[:7], y[:7], x[7:], y[7:], epochs=1)
    hook.remove()
    assert counts[0] == 3  # prediction, separate diversity pass, validation


def test_legacy_snapshot_mismatch_is_characterized(monkeypatch):
    model = TorchBoostModel(2, 2, 2, dropout_rate=0)
    evaluated = []
    original = model.harden_splits_cosine
    def capture(**kwargs):
        evaluated.append(model.trees[0].temperature.detach().clone())
        original(**kwargs)
    monkeypatch.setattr(model, "harden_splits_cosine", capture)
    x, y = torch.randn(12, 2), torch.randn(12)
    train_torchboost(model, x[:8], y[:8], x[8:], y[8:], epochs=1)
    assert not torch.equal(evaluated[0], model.trees[0].temperature)


def test_legacy_backoff_reversal_is_characterized():
    model = TorchBoostModel(2, 2, 2)
    model.harden_splits(1, 10, val_loss=1., prev_val_loss=1., method="cosine")
    baseline = model.trees[0].temperature.detach().clone()
    model.harden_splits(1, 10, val_loss=2., prev_val_loss=1., method="cosine")
    assert model.trees[0].temperature < baseline  # historical bug; new control fixes direction
