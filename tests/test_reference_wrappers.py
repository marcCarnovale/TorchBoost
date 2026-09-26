"""Sanity checks for actually installed benchmark implementations."""
import numpy as np
import pandas as pd
import pytest
import torch
from experiments.reference_models import make_reference, specifications, primary

torch.set_num_threads(1)

@pytest.mark.parametrize('family',['xgboost','lightgbm','catboost','mlp'])
@pytest.mark.parametrize('classification',[False,True])
def test_reference_finite_mixed_schema(family,classification):
    rng=np.random.default_rng(21)
    X=pd.DataFrame({'x':rng.normal(size=120),'category':rng.choice(['a','b','c'],120)})
    y=(X.x.to_numpy()>0).astype(int) if classification else X.x.to_numpy()*2+rng.normal(size=120)*.1
    X.loc[3,'x']=np.nan
    X.loc[119,'category']='unseen'
    params=specifications(family)[0]
    if family=='mlp':params={**params,'epochs':8,'width':16}
    model=make_reference(family,params,classification,31).fit(X.iloc[:80],y[:80],eval_set=(X.iloc[80:100],y[80:100]))
    prediction=model.predict(X.iloc[100:])
    assert prediction.shape==(20,)
    assert np.isfinite(prediction).all()
    assert np.isfinite(primary(model,X.iloc[100:],y[100:],classification))
    if classification:
        probabilities=model.predict_proba(X.iloc[100:])
        np.testing.assert_allclose(probabilities.sum(1),1,atol=1e-7)
