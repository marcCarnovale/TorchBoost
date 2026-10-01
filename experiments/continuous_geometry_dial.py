"""Fast continuous geometry-dial benchmark.

Interpolates the true logit between an axis-aligned piecewise mechanism and a
low-rank multiplicative interaction:
    z_lambda = normalized[(1-lambda) z_axis + lambda z_interaction]
for lambda in {0,.2,.4,.6,.8,1}.

Exact Bayes probabilities are known at every lambda. Compare CatBoost, MLP,
bilinear specialist, and a differentiable two-expert mixture. The mixture learns
both expert functions and its gate end-to-end. We ask whether learned bilinear
weight increases with the true interaction fraction while retaining substantial
CatBoost->Bayes gap recovery.

Synthetic only; no HIGGS or real benchmark data.
"""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss,roc_auc_score
from sklearn.preprocessing import StandardScaler
import experiments.fast_specialist_redesign as spec
import experiments.synthetic_cat_corner_controller_v6 as v6

P=24
N_TRAIN=3000
N_TEST=12000
PASSES=24
BATCH=256
LR=1.2e-3

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))

def problem(seed):
    rng=np.random.default_rng(seed)
    a=rng.normal(size=P); a/=np.linalg.norm(a)
    b=rng.normal(size=P); b/=np.linalg.norm(b)
    return a.astype("float32"),b.astype("float32")

def components(x,a,b):
    axis=2.4*(x[:,0]>.2)-2.0*(x[:,1]<-.4)+1.7*((x[:,2]>.1)&(x[:,3]>.2))-.6
    u=x@a; v=x@b
    interaction=3.2*np.tanh(1.4*u*v)+1.0*u*v
    # Normalize latent mechanisms to comparable population scale so lambda is
    # a meaningful geometry fraction rather than an amplitude artifact.
    axis=(axis-axis.mean())/(axis.std()+1e-8)
    interaction=(interaction-interaction.mean())/(interaction.std()+1e-8)
    return axis,interaction

def sample(a,b,lam,n,seed):
    rng=np.random.default_rng(seed); x=rng.normal(size=(n,P)).astype("float32")
    za,zi=components(x,a,b)
    z=2.6*((1-lam)*za+lam*zi)
    p=sigmoid(z); y=rng.binomial(1,p).astype("float32")
    return x,y,p

def metrics(y,p):
    p=np.clip(p,1e-7,1-1e-7)
    return {"nll":float(log_loss(y,p,labels=[0,1])),"auc":float(roc_auc_score(y,p))}

class AxisMLP(nn.Module):
    # MLP is the flexible non-specialist comparator, not claimed axis-specific.
    def __init__(self,p): super().__init__(); self.net=nn.Sequential(nn.Linear(p,96),nn.GELU(),nn.Linear(96,48),nn.GELU(),nn.Linear(48,1))
    def forward(self,x): return self.net(x).squeeze(1)

class TwoExpertMixture(nn.Module):
    def __init__(self,p):
        super().__init__(); self.mlp=AxisMLP(p); self.bilinear=spec.BilinearLowRank(p)
        self.gate=nn.Parameter(torch.zeros(2))
    def forward(self,x,temp=1.0):
        w=torch.softmax(self.gate/temp,dim=0)
        return w[0]*self.mlp(x)+w[1]*self.bilinear(x)
    def weights(self):
        return torch.softmax(self.gate,dim=0)

def train(model,x,y,seed):
    torch.manual_seed(seed); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5)
    xt=torch.from_numpy(x); yt=torch.from_numpy(y); gen=torch.Generator().manual_seed(seed+19)
    best=(1e9,None)
    for epoch in range(PASSES):
        order=torch.randperm(len(x),generator=gen); model.train()
        temp=max(.25,1.8*(1-epoch/(PASSES-1))+.25*(epoch/(PASSES-1)))
        for s in range(0,len(order),BATCH):
            idx=order[s:s+BATCH]; opt.zero_grad(set_to_none=True)
            logits=model(xt[idx],temp) if isinstance(model,TwoExpertMixture) else model(xt[idx])
            loss=nn.functional.binary_cross_entropy_with_logits(logits,yt[idx])
            if isinstance(model,TwoExpertMixture) and epoch<8:
                w=torch.softmax(model.gate/temp,dim=0); entropy=-(w*torch.log(w.clamp_min(1e-8))).sum()
                loss=loss-.004*entropy
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
        with torch.no_grad():
            logits=model(xt) if not isinstance(model,TwoExpertMixture) else model(xt,.25)
            score=float(nn.functional.binary_cross_entropy_with_logits(logits,yt))
        if score<best[0]: best=(score,{k:v.detach().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); return model

@torch.no_grad()
def predict(model,x):
    model.eval(); xt=torch.from_numpy(x)
    logits=model(xt,.25) if isinstance(model,TwoExpertMixture) else model(xt)
    return torch.sigmoid(logits).numpy()

def run(lam,seed,out):
    torch.set_num_threads(4); start=time.perf_counter(); a,b=problem(seed+41)
    x,y,_=sample(a,b,lam,N_TRAIN,seed+1001); qx,qy,bp=sample(a,b,lam,N_TEST,seed+500001)
    sc=StandardScaler().fit(x); x=sc.transform(x).astype("float32"); qx=sc.transform(qx).astype("float32")
    cat=CatBoostClassifier(iterations=300,depth=7,learning_rate=.06,l2_leaf_reg=8,loss_function="Logloss",verbose=False,random_seed=seed,thread_count=4)
    cat.fit(x,y); cm=metrics(qy,cat.predict_proba(qx)[:,1]); bayes=metrics(qy,bp); gap=cm["nll"]-bayes["nll"]
    makers={"mlp":lambda:AxisMLP(P),"bilinear":lambda:spec.BilinearLowRank(P),"mixture":lambda:TwoExpertMixture(P)}
    rows={}
    for i,(name,maker) in enumerate(makers.items()):
        m=train(maker(),x,y,seed+2000+100*i); mm=metrics(qy,predict(m,qx))
        row={"ranking":mm,"delta_nll_vs_catboost":mm["nll"]-cm["nll"],"delta_auc_vs_catboost":mm["auc"]-cm["auc"],
          "gap_recovered":float((cm["nll"]-mm["nll"])/gap) if gap>1e-8 else 0.0}
        if isinstance(m,TwoExpertMixture):
            w=m.weights().detach().numpy(); row["weights"]={"mlp":float(w[0]),"bilinear":float(w[1])}
        rows[name]=row
    result={"study":"continuous_axis_interaction_geometry_dial_v12","lambda_interaction":lam,"seed":seed,
      "train_rows":N_TRAIN,"test_rows":N_TEST,"catboost":cm,"bayes":bayes,"catboost_to_bayes_nll_gap":gap,
      "models":rows,"seconds":time.perf_counter()-start}
    Path(out).parent.mkdir(parents=True,exist_ok=True); Path(out).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)); print(json.dumps(result,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--lambda-interaction",type=float,required=True); p.add_argument("--seed",type=int,default=733); p.add_argument("--out",required=True)
    a=p.parse_args(); run(a.lambda_interaction,a.seed,a.out)
