
import numpy as np

from torchboost.adaptive.unified_progressive import (
    UnifiedConfig,
    UnifiedProgressiveClassifier,
)


def dataset(n=1800,d=8,seed=4):
    r=np.random.default_rng(seed);x=r.normal(size=(n,d)).astype("float32")
    c=x[:,0]>0;z=np.where(c,1.2*x[:,2]-.8*x[:,3],-x[:,2]+.9*x[:,4])
    p=1/(1+np.exp(-z));y=r.binomial(1,p).astype(int);return x,y

def test_crossfit_proposal_records_oof_selection():
    x,y=dataset()
    c=UnifiedConfig(n_trees=1,updates_per_stage=1,depth=2,linear_values=True,
        proposal_mode="linear_model_tree",proposal_folds=3,proposal_candidates=4,
        samples_per_parameter=3.,min_samples_leaf=20)
    m=UnifiedProgressiveClassifier(c).fit(x[:1200],y[:1200],eval_set=(x[1200:],y[1200:]))
    r=m.trainer_.proposal_history[0]
    assert r["resolved_design"]["crossfit_folds"]==3
    assert all(s["selection"]=="crossfit" for s in r["splits"])

def test_auto_complexity_reduces_small_data_structure_and_regularizes_nodes():
    x,y=dataset(n=900,d=12)
    c=UnifiedConfig(n_trees=1,updates_per_stage=1,depth=4,linear_values=True,
        proposal_mode="linear_model_tree",auto_complexity=True,min_samples_leaf=5)
    m=UnifiedProgressiveClassifier(c).fit(x[:600],y[:600],eval_set=(x[600:],y[600:]))
    r=m.trainer_.proposal_history[0]["resolved_design"]
    assert r["min_samples"]>=2*(12+1)
    assert r["depth"]<4
    assert r["samples_per_parameter"]>=2

def test_auto_complexity_uses_crossfit_when_data_are_large_enough():
    x,y=dataset(n=2400,d=6)
    c=UnifiedConfig(n_trees=1,updates_per_stage=1,depth=2,linear_values=True,
        proposal_mode="linear_model_tree",auto_complexity=True)
    m=UnifiedProgressiveClassifier(c).fit(x[:1800],y[:1800],eval_set=(x[1800:],y[1800:]))
    assert m.trainer_.proposal_history[0]["resolved_design"]["crossfit_folds"]==3