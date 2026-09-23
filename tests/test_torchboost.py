import torch

from torchboost import SoftTree, TorchBoostModel, train_torchboost


def test_soft_tree_routing_is_a_probability_distribution():
    torch.manual_seed(7)
    depth = 3
    tree = SoftTree(input_dim=4, depth=depth, output_dim=2**depth, dropout_rate=0)
    with torch.no_grad():
        tree.leaf_values.copy_(torch.eye(2**depth))
    probabilities = tree(torch.randn(11, 4))
    assert probabilities.shape == (11, 2**depth)
    assert torch.all(probabilities >= 0)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(11), atol=1e-6)


def test_multiclass_forward_returns_logits_and_predict_proba_normalizes():
    torch.manual_seed(7)
    model = TorchBoostModel(
        num_trees=3,
        input_dim=4,
        tree_depth=2,
        task_type="multiclass_classification",
        num_classes=3,
        dropout_rate=0,
    )
    inputs = torch.randn(5, 4)
    logits = model(inputs)
    probabilities = model.predict_proba(inputs)
    assert logits.shape == (5, 3)
    assert probabilities.shape == (5, 3)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(5), atol=1e-6)


def test_multitarget_output_shape_honors_num_classes():
    model = TorchBoostModel(
        num_trees=2,
        input_dim=3,
        tree_depth=2,
        task_type="multitarget",
        num_classes=4,
        dropout_rate=0,
    )
    assert model(torch.randn(6, 3)).shape == (6, 4)


def test_tree_dropout_keeps_backward_graph_when_all_masks_drop(monkeypatch):
    model = TorchBoostModel(1, 2, 1, task_type="regression", dropout_rate=0.9)
    model.train()
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.zeros(*args, **kwargs))
    output = model(torch.randn(4, 2))
    output.sum().backward()


def test_invalid_tree_configuration_is_rejected():
    try:
        SoftTree(input_dim=2, depth=0)
    except ValueError as error:
        assert "depth" in str(error)
    else:
        raise AssertionError("depth=0 should be rejected")


def test_training_accepts_column_vector_regression_targets(recwarn):
    torch.manual_seed(7)
    model = TorchBoostModel(2, 3, 2, task_type="regression", dropout_rate=0)
    inputs = torch.randn(12, 3)
    targets = torch.randn(12, 1)
    train_torchboost(
        model,
        inputs[:8],
        targets[:8],
        inputs[8:],
        targets[8:],
        epochs=1,
        early_stopping=False,
    )
    assert not [warning for warning in recwarn if "target size" in str(warning.message)]
