
import numpy as np
from sklearn.linear_model import LogisticRegression
from torchboost.adaptive.oof_forest import OOFForest,OOFForestConfig
def test_oof_forest_keeps_best_and_weights():
    r=np.random.default_rng(1);x=r.normal(size=(120,4));y=(x[:,0]+.2*x[:,1]>0).astype(int)
    f=OOFForest(lambda seed:LogisticRegression(C=1+seed%3,max_iter=100),OOFForestConfig(n_candidates=4,n_members=2,folds=3,random_state=2))
    f.fit(x,y);assert len(f.models_)==2;assert abs(f.weights_.sum()-1)<1e-12;assert f.predict_proba(x).shape==(120,2)