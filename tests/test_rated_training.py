import torch

from torchboost.adaptive.data import DataSplit
from torchboost.adaptive.objectives import Objective
from torchboost.adaptive.progressive import ProgressiveSum, _new_packed
from torchboost.adaptive.progressive_regularizers import Regularizers
from torchboost.adaptive.rated_forest import materialize_progressive_sum
from torchboost.adaptive.rated_training import RatedJointTrainer, RatedRegularizers


def tiny_rated():
    cfg = type("Cfg", (), {
        "depth": 2, "learning_rate": .01, "routing_temperature": 1.,
        "readout": "residual", "batch_size": 16, "cart_strength": 4.,
    })()
    p = ProgressiveSum(torch.tensor([0.0]), learn_rates=True)
    for j, rate in enumerate((0.4, 0.6)):
        tree, _ = _new_packed(3, 1, cfg, 10 + j)
        with torch.no_grad():
            tree.values.normal_(generator=torch.Generator().manual_seed(30 + j))
        p.append(tree, rate)
    return materialize_progressive_sum(p)


def test_rated_trainer_requests_trace_and_regularizes_front_coefficients():
    model = tiny_rated()
    model.config.epochs = 1
    model.config.batch_size = 16
    objective = Objective("binary", 1)
    reg = RatedRegularizers(
        Regularizers(leaf_l2=1e-4, hierarchy=1e-4, tree_l2=1e-4),
        rate_l2=1e-4,
        count_pressure=1e-4,
    )
    trainer = RatedJointTrainer(
        model, objective, model.config, torch.Generator().manual_seed(7),
        rated_regularizers=reg,
    )
    assert trainer._needs_trace({}) is True
    x = torch.randn(32, 3, generator=torch.Generator().manual_seed(9))
    y = torch.randint(0, 2, (32,), generator=torch.Generator().manual_seed(11))
    split = DataSplit(x, y, torch.ones(32))
    before = model.rates.detach().clone()
    trainer.fit(split, split, split)
    assert torch.isfinite(model.rates).all()
    assert not torch.equal(before, model.rates)
    trainer.close()
