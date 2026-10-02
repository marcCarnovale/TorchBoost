
"""Out-of-fold selected/weighted forests for fitted TorchBoost estimators."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import log_loss,mean_squared_error

@dataclass
class OOFForestConfig:
    n_candidates:int=12
    n_members:int=5
    folds:int=5
    inner_bag_fraction:float=.8
    outer_bag_fraction:float=1.
    weighting:str="softmax"  # uniform / inverse / softmax
    weight_temperature:float=.02
    random_state:int=0

class OOFForest:
    """Select candidates by OOF performance, then refit only retained members.

    `factory(seed)` must return a fresh sklearn-like estimator. Inner bags are
    candidate-specific training subsamples within each fold. The outer bag is a
    candidate-specific refit subsample after OOF selection.
    """
    def __init__(self,factory,config=None,classification=True):
        self.factory=factory;self.config=config or OOFForestConfig();self.classification=classification
    def _loss(self,y,p):
        return log_loss(y,p,labels=self.classes_) if self.classification else mean_squared_error(y,p)**.5
    def fit(self,X,y):
        X=np.asarray(X);y=np.asarray(y);c=self.config;rng=np.random.default_rng(c.random_state)
        self.classes_=np.unique(y) if self.classification else None
        splitter=(StratifiedKFold(c.folds,shuffle=True,random_state=c.random_state) if self.classification else KFold(c.folds,shuffle=True,random_state=c.random_state))
        rows=[]
        for j in range(c.n_candidates):
            pred=np.zeros((len(y),len(self.classes_))) if self.classification else np.zeros(len(y))
            for fold,(tr,va) in enumerate(splitter.split(X,y if self.classification else None)):
                bag=rng.choice(tr,max(2,int(len(tr)*c.inner_bag_fraction)),replace=False)
                m=self.factory(c.random_state+1009*j+37*fold).fit(X[bag],y[bag])
                pred[va]=m.predict_proba(X[va]) if self.classification else m.predict(X[va])
            score=self._loss(y,pred);rows.append((score,j))
        rows.sort();keep=rows[:min(c.n_members,len(rows))];self.oof_scores_=[s for s,_ in keep]
        if c.weighting=="uniform":w=np.ones(len(keep))
        elif c.weighting=="inverse":w=1/(np.asarray(self.oof_scores_)+1e-12)
        elif c.weighting=="softmax":
            z=-np.asarray(self.oof_scores_)/max(c.weight_temperature,1e-8);z-=z.max();w=np.exp(z)
        else:raise ValueError("invalid weighting")
        self.weights_=w/w.sum();self.models_=[]
        for score,j in keep:
            bag=rng.choice(len(y),max(2,int(len(y)*c.outer_bag_fraction)),replace=False)
            self.models_.append(self.factory(c.random_state+1009*j+99991).fit(X[bag],y[bag]))
        return self
    def predict_proba(self,X):
        if not self.classification:raise AttributeError("regression forest has no predict_proba")
        return sum(w*m.predict_proba(X) for w,m in zip(self.weights_,self.models_))
    def predict(self,X):
        if self.classification:return self.classes_[self.predict_proba(X).argmax(1)]
        return sum(w*m.predict(X) for w,m in zip(self.weights_,self.models_))