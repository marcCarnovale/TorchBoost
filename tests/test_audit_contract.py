import numpy as np
import pandas as pd
import pytest
from experiments.audit_autotune import metrics, feature_slices
from torchboost.adaptive.autotune import make_search_folds


def test_classification_metrics_and_confidence_errors():
    y=np.array([0,1,0,1]);p=np.array([[.8,.2],[.3,.7],[.05,.95],[.1,.9]])
    result,loss=metrics(y,p,True,np.array([0,1]))
    np.testing.assert_allclose(result['primary'],-np.log([.8,.7,.05,.9]).mean())
    assert result['accuracy']==.75
    assert result['confident_errors']==1
    assert result['top_1pct_loss_share']==pytest.approx(loss.max()/loss.sum())


def test_regression_metrics_are_target_scale_not_training_normalized():
    y=np.array([10.,20.,30.]);p=np.array([11.,18.,33.])
    result,loss=metrics(y,p,False)
    np.testing.assert_allclose(loss,[1,4,9])
    assert result['primary']==pytest.approx(np.sqrt(14/3))
    assert result['mae']==2


def test_probability_mixture_bound():
    a=np.array([[.05,.95],[.8,.2],[.2,.8]])
    b=np.array([[.7,.3],[.1,.9],[.4,.6]])
    y=np.array([0,1,1]);w=.4
    mixture=metrics(y,w*a+(1-w)*b,True,np.arange(2))[0]['primary']
    bound=w*metrics(y,a,True,np.arange(2))[0]['primary']+(1-w)*metrics(y,b,True,np.arange(2))[0]['primary']
    assert mixture<=bound


def test_regression_diversity_decomposition():
    rng=np.random.default_rng(17);pred=rng.normal(size=(3,40));y=rng.normal(size=40);w=np.array([.2,.3,.5])
    mse=np.mean((w@pred-y)**2)
    weighted=np.sum(w*np.mean((pred-y)**2,axis=1))
    diversity=.5*sum(w[i]*w[j]*np.mean((pred[i]-pred[j])**2) for i in range(3) for j in range(3))
    np.testing.assert_allclose(mse,weighted-diversity)


def test_slices_are_fitted_only_on_training_features():
    train=pd.DataFrame({'x':[0.,1.,2.],'cat':['a','b','a']})
    audit=pd.DataFrame({'x':[1.,8.,np.nan],'cat':['a','q','b']})
    r=feature_slices(train,audit)
    assert r['has_missing'].tolist()==[False,False,True]
    assert r['unseen_category'].tolist()==[False,True,False]
    assert r['outside_fit_marginal_1_99pct'].tolist()==[False,True,False]


def test_forward_folds_do_not_use_future_stop_or_rank():
    times=np.repeat(np.arange(60),3);X=np.ones((180,2));y=np.zeros(180)
    for f in make_search_folds(X,y,classification=False,times=times):
        assert max(times[f.fit])<min(times[f.stop])
        assert max(times[f.stop])<min(times[f.rank])
