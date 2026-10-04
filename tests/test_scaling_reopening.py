import pytest
from experiments.reopening_check import run_check

@pytest.mark.parametrize('mode',['cooling','capacitor','rlc'])
def test_real_loss_regression_reopens_frozen_nodes_only_with_heat(mode):
    r=run_check(mode)
    assert r['thaw_events']==(0 if mode=='cooling' else 7)
    assert max(abs(h['energy_error']) for h in r['history'])<1e-9
    assert r['history'][1]['control_loss']>r['history'][0]['control_loss']
