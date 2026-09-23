import copy

import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from torchboost import BinaryLogisticObjective, BinarySoftTree, StagewiseBinaryClassifier
from torchboost.export import predict_exported_proba


@pytest.fixture
def data():
    x, y = make_classification(n_samples=240, n_features=5, n_informative=4,
                               n_redundant=0, random_state=7)
    return train_test_split(x, y, stratify=y, random_state=13)


def small(**kwargs):
    return StagewiseBinaryClassifier(**{"n_estimators": 4, "max_depth": 2,
        "epochs_per_stage": 3, "batch_size": 64, "collect_metrics": True, **kwargs})


def test_logistic_derivatives_match_autograd():
    score = torch.tensor([-15., -2., 0., 1., 18.], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0., 1., 0., 1., 0.], dtype=torch.float64)
    first = torch.autograd.grad(BinaryLogisticObjective.loss(score, target).sum(), score, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), score)[0]
    g, h = BinaryLogisticObjective.derivatives(score, target)
    torch.testing.assert_close(g, first)
    torch.testing.assert_close(h, second, atol=1e-15, rtol=1e-9)
    assert h[2] == .25


@pytest.mark.parametrize("depth", [1, 2, 5])
def test_soft_and_hard_routing_mass(depth):
    tree = BinarySoftTree(5, depth)
    x = torch.randn(11, 5)
    for hard in [False, True]:
        trace = tree.routing(x, hard=hard)
        assert trace.reach.shape == (11, 2**depth-1)
        assert (trace.leaves >= 0).all()
        torch.testing.assert_close(trace.leaves.sum(1), torch.ones(11))
        if hard:
            assert ((trace.leaves == 0) | (trace.leaves == 1)).all()


def test_actual_gate_weights_receive_gradients():
    tree = BinarySoftTree(3, 3)
    tree(torch.randn(17, 3)).square().sum().backward()
    assert tree.weights.grad.abs().sum() > 0
    assert "temperature" not in dict(tree.named_parameters())


def test_temperature_validation_and_purity():
    tree = BinarySoftTree(3, 2)
    state = copy.deepcopy(tree.state_dict())
    x = torch.randn(7, 3)
    torch.testing.assert_close(tree(x), tree(x))
    for key in state:
        torch.testing.assert_close(state[key], tree.state_dict()[key])
    for invalid in [0, -1, float("nan"), torch.ones(2)]:
        with pytest.raises(ValueError):
            tree.set_temperature(invalid)


def test_coupled_soft_leaf_solution_is_stationary():
    tree = BinarySoftTree(2, 2).double()
    x = torch.randn(23, 2, dtype=torch.float64)
    g, h, w = torch.randn(23, dtype=torch.float64), torch.rand(23, dtype=torch.float64)+.1, torch.rand(23, dtype=torch.float64)
    tree.solve_leaves(x, g, h, w, .02, 7)
    f = tree(x)
    objective = (w*(g*f+.5*h*f*f)).sum()/w.sum()+.01*tree.leaf_values.square().sum()
    gradient = torch.autograd.grad(objective, tree.leaf_values)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-10, rtol=0)


def test_cart_warm_start_matches_hard_cart():
    from sklearn.tree import DecisionTreeRegressor
    x = torch.linspace(-2, 2, 31)[:, None]
    y = (x[:, 0] > 0).float()
    tree = BinarySoftTree(1, 3)
    tree.initialize_cart(x, y, torch.ones(31), min_samples_leaf=2, seed=0)
    baseline = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, random_state=0).fit(x, y)
    np.testing.assert_allclose(tree(x, hard=True).detach().numpy(), baseline.predict(x))


def test_stagewise_logloss_descends_and_generalizes(data):
    x, xv, y, yv = data
    model = small(n_estimators=8).fit(x, y)
    losses = [r["train_loss"] for r in model.history_ if r["accepted"]]
    assert np.all(np.diff(losses) < 0)
    assert roc_auc_score(yv, model.predict_proba(xv)[:, 1]) > .7
    assert log_loss(yv, model.predict_proba(xv)) < .69
    assert all(not p.requires_grad for p in model.model_.parameters())


def test_prefix_immutability(data):
    x, _, y, _ = data
    a, b = small(n_estimators=1).fit(x, y), small(n_estimators=3).fit(x, y)
    for key, value in a.model_.trees[0].state_dict().items():
        torch.testing.assert_close(value, b.model_.trees[0].state_dict()[key], rtol=0, atol=0)


def test_reproducible_without_mutating_torch_rng(data):
    x, _, y, _ = data
    before = torch.random.get_rng_state().clone()
    a, b = small().fit(x, y), small().fit(x, y)
    torch.testing.assert_close(before, torch.random.get_rng_state())
    np.testing.assert_array_equal(a.predict_proba(x), b.predict_proba(x))


def test_weight_scale_invariance(data):
    x, _, y, _ = data
    weight = np.where(y == 1, 2., 1.)
    a, b = small().fit(x, y, weight), small().fit(x, y, 100*weight)
    np.testing.assert_allclose(a.predict_proba(x), b.predict_proba(x), atol=1e-7)


def test_missing_values_entire_column_and_preprocessing_train_only(data):
    x, xv, y, _ = data
    x = x.copy()
    xv = xv.copy()
    x[:, 0] = np.nan
    x[::3, 1] = np.nan
    xv[:, 0] = np.nan
    m = small().fit(x, y)
    mean = m.center_.copy()
    assert np.isfinite(m.predict_proba(xv)).all()
    m.predict_proba(xv*100)
    np.testing.assert_array_equal(m.center_, mean)
    assert m.center_[0] == 0


@pytest.mark.parametrize("controller", [None, {"injection_gain": 2., "cooling_law": "radiative"}])
def test_checkpoint_roundtrip(data, tmp_path, controller):
    x, xv, y, yv = data
    m = small(controller=controller).fit(x, y, control_set=(xv, yv) if controller else None)
    path = tmp_path/"model.pt"
    m.save(path)
    restored = StagewiseBinaryClassifier.load(path)
    np.testing.assert_array_equal(m.predict_proba(xv), restored.predict_proba(xv))
    assert restored.tracker_.state_dict() == m.tracker_.state_dict()
    assert restored.control_history_ == m.control_history_


def test_hard_export_roundtrip(data, tmp_path):
    x, xv, y, _ = data
    model = small().fit(x, y)
    path = tmp_path/"model.json"
    model.export_json(path)
    np.testing.assert_allclose(predict_exported_proba(path, xv), model.predict_proba(xv, hard=True), atol=1e-7)


def test_best_checkpoint_is_the_evaluated_model(data):
    x, xv, y, yv = data
    m = small(n_estimators=6).fit(x, y, eval_set=(xv, yv))
    assert abs(log_loss(yv, m.predict_proba(xv))-m.best_validation_loss_) < 1e-7
    assert m.n_estimators_ <= 6


def test_sklearn_clone_and_non_numeric_labels(data):
    x, xv, y, _ = data
    labels = np.where(y == 1, "yes", "no")
    m = clone(small()).fit(x, labels)
    assert set(m.predict(xv)) <= {"yes", "no"}
    assert m.predict_proba(xv).shape == (len(xv), 2)


@pytest.mark.parametrize("weights", [[-1]*180, [0]*180, [1, 2], [float('nan')]*180])
def test_invalid_weights(data, weights):
    x, _, y, _ = data
    with pytest.raises(ValueError):
        small().fit(x, y, weights)


@pytest.mark.parametrize("option", [{"init": "fake"}, {"max_depth": 25}, {"leaf_l2": 0},
                                    {"batch_size": 0}, {"lr": float('nan')}])
def test_invalid_configuration(data, option):
    x, _, y, _ = data
    with pytest.raises(ValueError):
        small(**option).fit(x, y)


def test_controller_requires_explicit_control_split(data):
    x, _, y, _ = data
    with pytest.raises(ValueError, match="control_set"):
        small(controller={}).fit(x, y)


def test_controller_metrics_keep_split_identity(data):
    x, xv, y, yv = data
    m = small(controller={}, n_estimators=2).fit(x, y, control_set=(xv, yv))
    assert len(m.control_history_) == 6
    snapshot = m.tracker_.snapshot()
    assert "stage:0/node:0" in snapshot and "stage:1/node:2" in snapshot
    before = copy.deepcopy(m.model_.state_dict())
    m.predict_proba(xv)
    m.predict_proba(xv)
    for key in before:
        torch.testing.assert_close(before[key], m.model_.state_dict()[key], atol=0, rtol=0)
