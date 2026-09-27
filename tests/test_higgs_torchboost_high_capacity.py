import numpy as np

from experiments.higgs_torchboost_high_capacity import high_capacity_schedule
from torchboost.adaptive.progressive import RollingBoostClassifier, RollingBoostConfig


def test_high_capacity_schedule_scales_capacity_and_compute():
    a = high_capacity_schedule(500_000)
    b = high_capacity_schedule(1_000_000)
    c = high_capacity_schedule(3_000_000)
    assert (a["n_trees"], b["n_trees"], c["n_trees"]) == (64, 96, 160)
    assert (a["depth"], b["depth"], c["depth"]) == (6, 7, 8)
    assert a["planned_total_optimizer_passes"] >= 8.0
    assert b["planned_total_optimizer_passes"] >= 8.0
    assert c["planned_total_optimizer_passes"] >= 8.0
    assert c["cart_sample_size"] < 3_000_000


def test_rolling_boost_sampled_cart_and_cached_joint_window():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(800, 7)).astype("float32")
    y = (x[:, 0] + 0.6 * x[:, 1] - 0.3 * x[:, 2] > 0).astype(int)
    cfg = RollingBoostConfig(
        n_trees=4,
        depth=3,
        stage_updates=3,
        batch_size=64,
        cart_value_updates=1,
        active_window=2,
        joint_updates=1,
        joint_every=2,
        cart_sample_size=96,
        patience_stages=5,
        random_state=17,
    )
    model = RollingBoostClassifier(cfg).fit(
        x[:600], y[:600], eval_set=(x[600:], y[600:])
    )
    assert model.n_estimators_ >= 1
    assert any(row["joint_refined"] for row in model.history_)
    assert np.isfinite(model.predict_proba(x[600:])).all()
