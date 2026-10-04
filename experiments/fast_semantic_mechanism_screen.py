"""Fast mechanism triage with known Bayes NLL.

Purpose: find regimes where a TorchBoost semantic mechanism has enough leverage
to matter materially, before spending compute on robust replication.

One quick arm per synthetic geometry. Each problem exposes exact Bayes
probabilities. We compare a small CatBoost baseline to four semantic experts
trained directly on the target, plus an equal-weight differentiable mixture.
Primary metric: fraction of CatBoost -> Bayes NLL gap recovered.

No real datasets and no HIGGS data.
"""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import log_loss,roc_auc_score
from sklearn.preprocessing import StandardScaler
import experiments.synthetic_cat_corner_controller_v6 as v6

N_TRAIN=3000
N_TEST=12000
P=24
PASSES=18
BATCH=256
LR=1e-3
REGIMES=("axis","oblique","ridge","interaction","regional_mix")

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))

def latent(regime,seed):
    rng=np.random.default_rng(seed); d={"regime":regime}
    if regime in ("oblique","ridge","regional_mix"):
        w=rng.normal(size=P); w/=np.linalg.norm(w); d["w"]=w.astype("float32")
    if regime=="oblique":
        v=rng.normal(size=P); v/=np.linalg.norm(v); d["v"]=v.astype("float32")
    if regime=="interaction":
        a=rng.normal(size=P); a/=np.linalg.norm(a); b=rng.normal(size=P); b/=np.linalg.norm(b)
        d["a"]=a.astype("float32"); d["b"]=b.astype("float32")
    return d

def sample(problem,n,seed):
    rng=np.random.default_rng(seed); x=rng.normal(size=(n,P)).astype("float32"); r=problem["regime"]
    if r=="axis":
        z=2.4*(x[:,0]>.2)-2.0*(x[:,1]<-.4)+1.7*((x[:,2]>.1)&(x[:,3]>.2))-.6
    elif r=="oblique":
        z=3.0*(x@problem["w"])+1.6*np.sin(1.8*(x@problem["v"]))+.9*(x[:,0]*x[:,1])
    elif r=="ridge":
        t=x@problem["w"]; z=3.4*np.sin(1.5*t)+1.2*t-.3*t*t
    elif r=="interaction":
        u=x@problem["a"]; v=x@problem["b"]; z=3.2*np.tanh(1.4*u*v)+1.0*u*v
    elif r=="regional_mix":
        t=x@problem["w"]; axis=2.6*(x[:,0]>.15)-2.2*(x[:,1]<-.25)
        smooth=3.2*np.sin(1.35*t); z=np.where(x[:,2]>0,axis,smooth)
    else: raise ValueError(r)
    # Irreducible noise is entirely Bernoulli; Bayes probabilities are known.
    prob=sigmoid(z); y=rng.binomial(1,prob).astype("float32")
    return x,y,prob

def metrics(y,p):
    p=np.clip(p,1e-7,1-1e-7)
    return {"nll":float(log_loss(y,p,labels=[0,1])),"auc":float(roc_auc_score(y,p))}

def train_expert(model,x,y,seed):
    torch.manual_seed(seed); model.train(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5)
    xt=torch.from_numpy(x); yt=torch.from_numpy(y); gen=torch.Generator().manual_seed(seed+9)
    best=(1e9,None)
    for _ in range(PASSES):
        order=torch.randperm(len(x),generator=gen)
        for s in range(0,len(order),BATCH):
            idx=order[s:s+BATCH]; opt.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(xt[idx]),yt[idx]); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
        with torch.no_grad():
            score=float(nn.functional.binary_cross_entropy_with_logits(model(xt),yt))
        if score<best[0]: best=(score,{k:v.detach().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); return model

@torch.no_grad()
def pred(model,x):
    model.eval(); out=[]
    for s in range(0,len(x),2048): out.append(torch.sigmoid(model(torch.from_numpy(x[s:s+2048]))).numpy())
    return np.concatenate(out)

class Mixture(nn.Module):
    def __init__(self,p):
        super().__init__(); self.experts=v6.build_experts(p,1234); self.logits=nn.Parameter(torch.zeros(4))
    def forward(self,x):
        vals=torch.stack([e(x) for e in self.experts],dim=1)
        w=torch.softmax(self.logits,dim=0)
        return (vals*w[None,:]).sum(1)

def run(regime,seed,out):
    torch.set_num_threads(4); start=time.perf_counter()
    problem=latent(regime,seed+41); x,y,bp_train=sample(problem,N_TRAIN,seed+1001); qx,qy,bp=sample(problem,N_TEST,seed+500001)
    sc=StandardScaler().fit(x); x=sc.transform(x).astype("float32"); qx=sc.transform(qx).astype("float32")
    # intentionally modest CatBoost: enough to be strong, cheap enough for triage
    from catboost import CatBoostClassifier
    cat=CatBoostClassifier(iterations=300,depth=7,learning_rate=.06,l2_leaf_reg=8,loss_function="Logloss",verbose=False,random_seed=seed,thread_count=4)
    cat.fit(x,y); cp=cat.predict_proba(qx)[:,1]
    bayes=metrics(qy,bp); cm=metrics(qy,cp); gap=cm["nll"]-bayes["nll"]
    makers={
      "axis_soft":lambda:v6.AxisSoftExpert(P),
      "oblique":lambda:v6.ObliqueExpert(P),
      "affine":lambda:v6.AffineExpert(P),
      "interaction":lambda:v6.InteractionExpert(P),
      "mixture":lambda:Mixture(P),
    }
    results={}
    for i,(name,maker) in enumerate(makers.items()):
        model=train_expert(maker(),x,y,seed+2000+i*100); pm=metrics(qy,pred(model,qx))
        recovered=(cm["nll"]-pm["nll"])/gap if gap>1e-8 else 0.0
        row={"ranking":pm,"delta_nll_vs_catboost":pm["nll"]-cm["nll"],"delta_auc_vs_catboost":pm["auc"]-cm["auc"],
             "catboost_to_bayes_gap_recovered":float(recovered)}
        if name=="mixture": row["weights"]={k:float(z) for k,z in zip(v6.EXPERT_NAMES,torch.softmax(model.logits,dim=0).detach())}
        results[name]=row
    result={"study":"fast_semantic_mechanism_bayes_gap_v10","regime":regime,"seed":seed,"train_rows":N_TRAIN,"test_rows":N_TEST,
      "passes":PASSES,"catboost":{"trees":int(cat.tree_count_),"ranking":cm},"bayes":bayes,"catboost_to_bayes_nll_gap":gap,
      "mechanisms":results,"seconds":time.perf_counter()-start}
    Path(out).parent.mkdir(parents=True,exist_ok=True); Path(out).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--regime",choices=REGIMES,required=True); p.add_argument("--seed",type=int,default=733); p.add_argument("--out",required=True)
    a=p.parse_args(); run(a.regime,a.seed,a.out)
