import json

import numpy as np
import pytest
import torch
from torch import nn

from torchboost.adaptive.architecture_corners import (
    CompositionalTreeNetwork,
    ObliviousSoftForest,
    canonical_catboost_corner,
    canonical_mlp_corner,
)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 7),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.Linear(7, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def test_mlp_is_exact_depth_zero_tree_network_corner_and_growth_is_noop():
    torch.manual_seed(11)
    mlp = TinyMLP().eval()
    corner = CompositionalTreeNetwork.from_mlp(
        mlp, max_tree_depth=2, seed=11
    ).eval()
    x = torch.randn(64, 5)
    with torch.no_grad():
        expected = mlp(x)
        actual = corner(x)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert corner.grow_one_level() == 2
    with torch.no_grad():
        grown = corner(x)
    assert torch.allclose(grown, expected, atol=1e-6, rtol=1e-6)


def test_canonical_mlp_corner_has_exact_trainable_parameter_count():
    model = CompositionalTreeNetwork(
        21, (300, 300, 300, 300, 300), dropout=.1, max_tree_depth=3
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 368_101
    spec = canonical_mlp_corner()
    assert spec["hidden_widths"] == [300] * 5
    assert spec["tree_depth"] == 0
    assert spec["epochs"] == 20


def test_numerical_catboost_json_embeds_exactly(tmp_path):
    catboost = pytest.importorskip("catboost")
    rng = np.random.default_rng(23)
    x = rng.normal(size=(600, 6)).astype("float32")
    y = (x[:, 0] + .7 * x[:, 1] - .4 * x[:, 2] * x[:, 3] > 0).astype(int)
    model = catboost.CatBoostClassifier(
        iterations=7,
        depth=4,
        learning_rate=.08,
        loss_function="Logloss",
        verbose=False,
        random_seed=23,
        allow_writing_files=False,
    ).fit(x, y)
    path = tmp_path / "catboost.json"
    model.save_model(path, format="json")
    payload = json.loads(path.read_text())
    corner = ObliviousSoftForest.from_catboost_json(payload).eval()
    expected = model.predict(x[:128], prediction_type="RawFormulaVal")
    with torch.no_grad():
        actual = corner(torch.from_numpy(x[:128]))[:, 0].numpy()
    assert np.max(np.abs(actual - expected)) < 2e-6
    spec = canonical_catboost_corner()
    assert spec["iterations"] == 1536
    assert spec["depth"] == 10
    assert spec["grow_policy"] == "SymmetricTree"
    assert spec["border_count"] == 254
