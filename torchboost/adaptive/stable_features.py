"""Guard unidentifiable feature directions without changing fitting coordinates.

The original benchmark's FeatureMap remains loadable. This separately versioned
map pins every encoded coordinate constant during fit to its fitted reference.
No outcome labels, stopping examples, or audit examples are used by the guard.
"""
from copy import deepcopy
import numpy as np
from .autotune import FeatureMap, MappedTree
from .single_tree import SingleTreeClassifier, SingleTreeRegressor


class StableFeatureMap(FeatureMap):
    def fit(self,X,y=None):
        # Refitting an existing object must not reuse an old mask.
        for name in ('constant_numeric_','constant_encoded_','encoded_reference_'):
            if hasattr(self,name):delattr(self,name)
        super().fit(X,y)
        frame=self._frame(X);numeric,_=self._numeric(frame)
        self.constant_numeric_=np.ptp(numeric,axis=0)==0
        encoded=super().transform(X)
        self.constant_encoded_=np.ptp(encoded,axis=0)==0
        self.encoded_reference_=encoded[0].copy()
        self.protected_coordinates_=int(self.constant_encoded_.sum())
        return self

    def transform(self,X):
        frame=self._frame(X)
        if hasattr(self,'constant_numeric_'):
            raw=frame.loc[:,self.numeric_].to_numpy(dtype=float,na_value=np.nan)
            if np.isinf(raw).any():raise ValueError('infinite features are not supported')
            # Clamp before scaling to avoid overflow in constant/all-missing columns;
            # preserve missingness here, then pin constant indicator columns below.
            for index,name in enumerate(self.numeric_):
                if self.constant_numeric_[index]:
                    frame[name]=frame[name].where(frame[name].isna(),self.median_[index])
        encoded=super().transform(frame)
        if hasattr(self,'constant_encoded_'):
            encoded[:,self.constant_encoded_]=self.encoded_reference_[self.constant_encoded_]
        return encoded


class StableMappedTree(MappedTree):
    """Matched tree fit using only the constant-coordinate feature guard."""
    def fit(self,X,y,*,eval_set,sample_weight=None):
        self.encoder_=StableFeatureMap(self.candidate.representation,self.candidate.bins,
            random_state=self.candidate.tree.random_state).fit(X)
        estimator=SingleTreeClassifier if self.classification else SingleTreeRegressor
        self.tree_=estimator(deepcopy(self.candidate.tree)).fit(self.encoder_.transform(X),y,sample_weight,
            eval_set=(self.encoder_.transform(eval_set[0]),eval_set[1],*eval_set[2:]))
        self.n_features_in_=self.encoder_.n_features_in_
        if self.classification:self.classes_=self.tree_.classes_.copy()
        return self
