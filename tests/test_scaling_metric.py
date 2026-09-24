from types import SimpleNamespace
import numpy as np
import pytest
from experiments.scale_study import metric

def test_one_dimensional_regression_predictions_do_not_broadcast():
    p=SimpleNamespace(task='regression',target_scale=np.array([2.]))
    y=np.array([[1.],[3.],[9.]])
    assert metric(y[:,0],y,p)==0.
    assert metric(y+1,y,p)==2.
    assert metric(y[:,0]+1,y,p)==2.

def test_multioutput_units_and_mismatch():
    p=SimpleNamespace(task='regression',target_scale=np.array([2.,4.]))
    a=np.ones((3,2));b=np.zeros((3,2))
    assert metric(a,b,p)==pytest.approx(np.sqrt(10.))
    with pytest.raises(ValueError):metric(np.zeros(3),b,p)

def test_binary_nll_equivalent_for_columns():
    p=SimpleNamespace(task='binary')
    y=np.array([0,1,1,0]);z=np.array([-1.,1.,1.,-1.])
    assert metric(z,y,p)==pytest.approx(np.log1p(np.exp(-1)))
    assert metric(z[:,None],y[:,None],p)==metric(z,y,p)
