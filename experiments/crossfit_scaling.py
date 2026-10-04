
import os,json,time,argparse,numpy as np
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1")
from sklearn.metrics import log_loss
from torchboost.adaptive.unified_progressive import UnifiedConfig,UnifiedProgressiveClassifier,default_native
from torchboost.adaptive.config import StructureConfig
from torchboost.adaptive.progressive_regularizers import Regularizers

def make(n,d,seed):
    r=np.random.default_rng(seed);x=r.normal(size=(n,d)).astype("float32")
    bits=(x[:,:4]>0).astype(int);ctx=sum(bits[:,j]*(1<<j) for j in range(4))
    coef=r.normal(size=(16,d));coef[:,:4]=0
    raw=np.array([coef[c]@xx for c,xx in zip(ctx,x)]);raw=raw/np.std(raw)*1.15
    p=1/(1+np.exp(-raw));y=r.binomial(1,p).astype(int);return x,y

def fit_variant(x,y,tr,se,au,variant,seed):
    n=default_native();n.learning_rate=.012;n.batch_size=256
    n.structure=StructureConfig(dynamic=False,initial_depth=0,max_depth=5,max_nodes=255)
    kw={}
    if variant=="crossfit":kw=dict(proposal_folds=3,proposal_candidates=3,samples_per_parameter=4.,proposal_crossfit_max_rows=6000,proposal_screen_max_rows=2500)
    elif variant=="auto":kw=dict(auto_complexity=True,proposal_candidates=3)
    c=UnifiedConfig(n_trees=1,updates_per_stage=16,depth=2,bins=6,min_samples_leaf=12,
        linear_values=True,linear_l2=8.,proposal_mode="linear_model_tree",cart_strength=7.,
        warm_value_updates=8,gate_release="oblique",checkpoint_every=4,
        regularizers=Regularizers(leaf_l2=1e-5,hierarchy=2e-4,linear_value_l2=1e-5),
        native=n,random_state=seed,**kw)
    t=time.time();m=UnifiedProgressiveClassifier(c).fit(x[tr],y[tr],eval_set=(x[se],y[se]))
    return {"variant":variant,"selection":m.best_score_,"audit":log_loss(y[au],m.predict_proba(x[au])),
            "best_step":m.trainer_.best_epoch,"seconds":time.time()-t,
            "design":m.trainer_.proposal_history[0]["resolved_design"],
            "splits":len(m.trainer_.proposal_history[0]["splits"])}

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--n",type=int);a.add_argument("--seed",type=int,default=71);a.add_argument("--out",required=True);z=a.parse_args()
    total=z.n+5000;x,y=make(total,16,z.seed);r=np.random.default_rng(z.seed+9);p=r.permutation(total)
    tr=p[:z.n];se=p[z.n:z.n+2000];au=p[z.n+2000:]
    rows=[fit_variant(x,y,tr,se,au,v,z.seed) for v in ("insample","crossfit","auto")]
    try:
        from catboost import CatBoostClassifier
        t=time.time();m=CatBoostClassifier(iterations=256,depth=8,learning_rate=.05,l2_leaf_reg=20,verbose=False,random_seed=z.seed,thread_count=1).fit(x[tr],y[tr])
        rows.append({"variant":"catboost","audit":log_loss(y[au],m.predict_proba(x[au])),"seconds":time.time()-t})
    except Exception as e:rows.append({"variant":"catboost","error":repr(e)})
    Path(z.out).write_text(json.dumps(rows,indent=2));print(json.dumps(rows))