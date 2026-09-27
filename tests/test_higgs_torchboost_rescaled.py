from experiments.higgs_torchboost_rescaled import corrected_schedule


def test_corrected_higgs_schedule_scales_exposure_and_capacity():
    schedules = {n: corrected_schedule(n) for n in (500_000, 1_000_000, 3_000_000)}
    assert [schedules[n]["n_trees"] for n in schedules] == [40, 56, 64]
    assert [schedules[n]["depth"] for n in schedules] == [6, 6, 7]
    assert [schedules[n]["stage_updates"] for n in schedules] == [13, 18, 46]
    for n, schedule in schedules.items():
        assert 2.0 <= schedule["planned_presentations_per_row"] < 2.2
        assert schedule["cart_value_updates"] < schedule["stage_updates"]


def test_corrected_higgs_schedule_is_restricted_to_declared_scales():
    try:
        corrected_schedule(2_000_000)
    except ValueError:
        pass
    else:
        raise AssertionError("undeclared scale must not silently invent a schedule")
