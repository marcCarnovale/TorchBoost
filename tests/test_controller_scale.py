"""Independently verify thermal scale calculations; no predictive claims."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import numpy as np
import pytest
from dataclasses import replace
from calibrate_controller import required_charge, scripted_run
from torchboost.adaptive.config import PhysicsConfig

@pytest.mark.parametrize('mode',['capacitor','rlc'])
@pytest.mark.parametrize('nodes',[8,64,256])
def test_required_charge_achieves_declared_target(mode,nodes):
    result=required_charge(PhysicsConfig(mode=mode),nodes,.25)
    assert result['achieved_rise_with_diagnostic_caps']==pytest.approx(.25,abs=1e-12)
    assert abs(result['energy_account_error'])<1e-10
    assert not result['attainable_from_empty_under_original_caps']

def test_cooling_only_is_matched_under_proportional_capacity_scaling():
    base=PhysicsConfig(mode='cooling')
    a=scripted_run(base,8,steps=40)
    b=scripted_run(replace(base,heat_capacity=.001,cooling=.0002),8,steps=40)
    np.testing.assert_allclose([r['mean_temperature'] for r in a['history']],
                               [r['mean_temperature'] for r in b['history']],atol=1e-12)
    assert a['total_resistive_heat']==b['total_resistive_heat']==0

def test_calibration_rejects_unmodeled_heterogeneity():
    with pytest.raises(ValueError):required_charge(PhysicsConfig(mode='rlc',heterogeneity=.2),8,.25)
