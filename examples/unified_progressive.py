"""Small executable example, with disjoint role partitions."""
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from torchboost.adaptive.unified_progressive import UnifiedConfig,UnifiedProgressiveClassifier
from torchboost.adaptive.config import PlasticityConfig

torch.set_num_threads(1)
x,y=make_classification(n_samples=1200,n_features=12,n_informative=8,random_state=67)
cfg=UnifiedConfig(n_trees=16,updates_per_stage=16,depth=3,row_subsample=.8,feature_subsample=.8,age_decay=.2)
cfg.regularizers.hierarchy=.02
# Change this to "none" for the matched no-memory comparison.
cfg.native.plasticity=PlasticityConfig(mode='anchor',stiffness=.3)
model=UnifiedProgressiveClassifier(cfg).fit(x[:660],y[:660],control_set=(x[660:780],y[660:780]),eval_set=(x[780:960],y[780:960]))
print('Selected trees:',model.n_estimators_)
print('Untouched example audit loss:',log_loss(y[960:],model.predict_proba(x[960:])))
model.save('unified_example.pt')
reloaded=UnifiedProgressiveClassifier.load('unified_example.pt')
assert (model.predict_proba(x[960:])==reloaded.predict_proba(x[960:])).all()
