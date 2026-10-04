import json

from experiments.higgs_hybrid_benchmark import load_proxy, run, subset


def test_higgs_smoke_has_low_and_all_conditions_and_sealed_audit(tmp_path):
    result = run(seed=17, smoke=True)
    assert result["status"] == "completed"
    assert set(result["conditions"]) == {"low", "all"}
    assert result["conditions"]["low"]["features"] == 21
    assert result["conditions"]["all"]["features"] == 28
    for condition in result["conditions"].values():
        assert set(condition["audit"]) == {"torchboost", "catboost", "mlp"}
        for family in condition["audit"].values():
            assert 0 <= family["auc"] <= 1
            assert family["nll"] > 0
    path = tmp_path / "r.json"
    path.write_text(json.dumps(result, allow_nan=False))
    assert json.loads(path.read_text())["status"] == "completed"


def test_low_level_condition_is_literal_first_21_columns():
    _, splits, _ = load_proxy(seed=3, smoke=True)
    low = subset(splits, "low")
    for role in splits:
        assert low[role].x.shape[1] == 21
        assert (low[role].x == splits[role].x[:, :21]).all()
