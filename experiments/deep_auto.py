
import os,time,json,numpy as np
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1")
from pathlib import Path
from sklearn.metrics import log_loss
from torchboost.adaptive.unified_progressive import UnifiedConfig,UnifiedProgressiveClassifier,default_native
from torchboost.adaptive.config import StructureConfig
from torchboost.adaptive.progressive_regularizers import Regularizers
from experiments.crossfit_scaling import make
nfit=12000;seed=71;total=nfit+5000;x,y=make(total,16,seed);r=np.random.default_rng(seed+9);p=r.permutation(total)
tr=p[:nfit];se=p[nfit:nfit+2000];au=p[nfit+2000:]
native=default_native();native.learning_rate=.01;native.batch_size=256;native.structure=StructureConfig(dynamic=False,initial_depth=0,max_depth=6,max_nodes=511)
rows=[]
for name,kw in [("deep_auto",{"auto_complexity":True,"proposal_candidates":1})]:
 c=UnifiedConfig(n_trees=1,updates_per_stage=512,depth=4,bins=8,min_samples_leaf=12,linear_values=True,linear_l2=8.,
  proposal_mode="linear_model_tree",cart_strength=7.,warm_value_updates=8,gate_release="oblique",checkpoint_every=32,
  regularizers=Regularizers(leaf_l2=1e-5,hierarchy=3e-4,linear_value_l2=2e-5),native=native,random_state=seed,**kw)
 t=time.time();m=UnifiedProgressiveClassifier(c).fit(x[tr],y[tr],eval_set=(x[se],y[se]))
 rows.append({"name":name,"selection":m.best_score_,"audit":log_loss(y[au],m.predict_proba(x[au])),
  "best_step":m.trainer_.best_epoch,"seconds":time.time()-t,"design":m.trainer_.proposal_history[0]["resolved_design"],"splits":len(m.trainer_.proposal_history[0]["splits"])})
 print(rows[-1],flush=True)
Path("deep_auto_result.json").write_text(json.dumps(rows,indent=2))