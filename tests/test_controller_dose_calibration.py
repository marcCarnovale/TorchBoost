import math

import pytest

from experiments.calibrate_controller_dose import refinement_gain


def test_refinement_uses_two_thermal_moments_not_predictive_scores():
    candidate = {"peak_thaw_units": 2.0, "mean_thaw_exposure": 8.0, "ranking_nll": -999}
    target = {"peak_thaw_units": 1.0, "mean_thaw_exposure": 2.0}
    gain = refinement_gain(0.1, candidate, target)
    assert gain == pytest.approx(0.1 / math.sqrt(8.0))
    candidate["ranking_nll"] = 999
    assert refinement_gain(0.1, candidate, target) == gain


def test_refinement_rejects_zero_exposure():
    with pytest.raises(ValueError, match="positive"):
        refinement_gain(1.0, {"peak_thaw_units": 0, "mean_thaw_exposure": 1},
                        {"peak_thaw_units": 1, "mean_thaw_exposure": 1})
