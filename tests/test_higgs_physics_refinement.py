from experiments.higgs_physics_refinement import run


def test_higgs_physics_smoke_preserves_protocol_and_native_identity():
    result = run(seed=23, epochs=1, smoke=True, arms=("none", "plastic", "direct"))
    assert result["status"] == "completed"
    assert result["warm"]["native_identity_max_abs"] < 5e-6
    assert result["audit"] is not None
    assert set(result["audit"]) == {"none", "plastic", "direct"}
    assert set(result["arms"]) == {"none", "plastic", "direct"}
    assert result["generic_regularizers"]["count_pressure"] > 0
    assert result["source"]["rate"] > 0
