import numpy as np
import pandas as pd
import pytest
from torchboost.adaptive.autotune import FeatureMap,TreeCandidate
from torchboost.adaptive.stable_features import StableFeatureMap,StableMappedTree
from torchboost.adaptive.single_tree import SingleTreeConfig

@pytest.mark.parametrize('mode',['raw','quantile','ple'])
def test_guard_preserves_entire_fitting_matrix(mode):
    X=pd.DataFrame({'constant':[3.]*8,'all_missing':[np.nan]*8,'varied':np.arange(8.),
                    'partial':[1.,2.,np.nan,3.,4.,np.nan,5.,6.],'single_cat':['a']*8,'cat':['a','b']*4})
    old=FeatureMap(mode).fit_transform(X);new=StableFeatureMap(mode).fit_transform(X)
    np.testing.assert_array_equal(new,old)

@pytest.mark.parametrize('mode',['raw','quantile','ple'])
def test_constant_changes_do_not_create_untrained_signals(mode):
    fit=pd.DataFrame({'constant':[3.]*8,'empty':[np.nan]*8,'varied':np.arange(8.),'single':['a']*8})
    m=StableFeatureMap(mode).fit(fit)
    a=pd.DataFrame({'constant':[3.],'empty':[np.nan],'varied':[2.],'single':['a']})
    b=pd.DataFrame({'constant':[1e200],'empty':[17.],'varied':[2.],'single':['new']})
    np.testing.assert_array_equal(m.transform(a),m.transform(b))
    assert m.protected_coordinates_>=3


def test_guard_refit_does_not_keep_old_mask():
    m=StableFeatureMap().fit(pd.DataFrame({'x':[3.]*4}))
    m.fit(pd.DataFrame({'x':[1.,2.,3.,4.]}))
    assert not m.constant_numeric_[0]
    assert m.transform(pd.DataFrame({'x':[1.]}))[0,0]!=m.transform(pd.DataFrame({'x':[4.]}))[0,0]


def test_guard_still_rejects_infinity():
    m=StableFeatureMap().fit(np.ones((5,2)))
    with pytest.raises(ValueError):m.transform([[np.inf,1.]])


def test_guarded_tree_prediction_invariance():
    rng=np.random.default_rng(8);X=pd.DataFrame({'x':rng.normal(size=120),'constant':np.zeros(120)})
    y=(X.x.to_numpy()>0).astype(int)
    c=TreeCandidate('tiny',SingleTreeConfig(depth=2,epochs=8,random_state=7,evaluate_every=1))
    tree=StableMappedTree(c,classification=True).fit(X.iloc[:80],y[:80],eval_set=(X.iloc[80:100],y[80:100]))
    test=X.iloc[100:].copy();shift=test.copy();shift['constant']=1e6
    np.testing.assert_array_equal(tree.predict_proba(test),tree.predict_proba(shift))
