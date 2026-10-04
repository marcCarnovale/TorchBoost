import numpy as np
from experiments.reference_mixture import ReferenceMixture
from torchboost.adaptive.specialist_forest import SpecialistForest
from tests.test_autotune import fitted_members


def test_same_probability_blender_on_both_sides():
    x,y,m=fitted_members(True)
    a=ReferenceMixture(m,classification=True).fit_blend(x[140:],y[140:])
    b=SpecialistForest(m).fit_blend(x[140:],y[140:])
    np.testing.assert_array_equal(a.weights_,b.weights_)
    np.testing.assert_array_equal(a.predict_proba(x),b.predict_proba(x))
    assert a.diagnostics(x,y)['estimators']==3


def test_same_regression_blender_on_both_sides():
    x,y,m=fitted_members(False)
    a=ReferenceMixture(m,classification=False).fit_blend(x[140:],y[140:])
    b=SpecialistForest(m).fit_blend(x[140:],y[140:])
    np.testing.assert_array_equal(a.weights_,b.weights_)
    np.testing.assert_array_equal(a.predict(x),b.predict(x))
