import numpy as np
import torch
from torchboost.adaptive.progressive import ProgressiveConfig,ProgressiveTreeClassifier

def data(n=400,seed=1):
 r=np.random.default_rng(seed);x=r.normal(size=(n,6)).astype('float32');y=(x[:,0]+.7*x[:,1]>0).astype(int);return x,y

def test_progressive_literal_sum_and_age_learning_rates():
 x,y=data();m=ProgressiveTreeClassifier(ProgressiveConfig(n_trees=3,depth=2,stage_updates=12,batch_size=128,learning_rate=.02,new_tree_shrinkage=.3,old_tree_lr_decay=.2,cart_value_updates=2,patience_stages=4,random_state=4)).fit(x[:300],y[:300],eval_set=(x[300:],y[300:]))
 assert m.n_estimators_>=1
 row=m.history_[min(1,len(m.history_)-1)]
 if row['trees']>1:
  assert row['tree_lrs'][-1] > row['tree_lrs'][0]
 with torch.no_grad():
  xx=m.preprocessor_.transform_x(x[300:310]);manual=m.model_.bias.expand(len(xx),-1).clone()
  for rate,tree in zip(m.model_.rates,m.model_.trees):manual+=rate*tree(xx)
  assert torch.allclose(manual,m.model_(xx))

def test_zero_old_lr_is_strict_stagewise_after_addition():
 x,y=data(seed=2);cfg=ProgressiveConfig(n_trees=2,depth=2,stage_updates=8,batch_size=128,old_tree_lr_decay=0.,cart_value_updates=0,patience_stages=3,random_state=2)
 m=ProgressiveTreeClassifier(cfg).fit(x[:300],y[:300],eval_set=(x[300:],y[300:]))
 if len(m.history_)==2: assert m.history_[1]['tree_lrs'][0]==0.
from torchboost.adaptive.progressive import RollingBoostConfig,RollingBoostClassifier

def test_rolling_boost_caches_old_scores_and_jointly_refines_window():
 x,y=data(seed=8);m=RollingBoostClassifier(RollingBoostConfig(n_trees=5,depth=2,stage_updates=4,joint_updates=2,joint_every=2,active_window=2,batch_size=128,patience_stages=6,random_state=8)).fit(x[:300],y[:300],eval_set=(x[300:],y[300:]))
 assert m.n_estimators_>=1
 assert any(r['joint_refined'] for r in m.history_[1:])
 assert np.isfinite(m.predict_proba(x[300:])).all()
