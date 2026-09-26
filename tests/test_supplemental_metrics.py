import numpy as np
from experiments.supplemental_metrics import binary_expected_risk,original_unit_errors


def test_expected_risk_equals_entropy_at_true_probabilities():
    p=np.array([.1,.2,.4,.5,.8,.9])
    r=binary_expected_risk(p,p)
    assert abs(r['excess_expected_log_loss'])<1e-14
    assert binary_expected_risk(p,np.ones(6)*.5)['excess_expected_log_loss']>0


def test_original_logprice_errors_in_actual_units():
    r=original_unit_errors('diamonds',None,np.log([100.,200.]),np.log([110.,190.]))
    np.testing.assert_allclose(r['original_rmse'],10,rtol=1e-10)
    np.testing.assert_allclose(r['original_mae'],10,rtol=1e-10)
