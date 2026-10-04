"""Fast specialist redesign screen.

Only oblique and interaction regimes. Compare compact specialist variants against
the already-strong MLP and CatBoost, using exact Bayes-gap recovery. Designed to
finish quickly and reveal whether the specialist parameterization or merely the
generic MLP is doing the work.
"""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import experiments.fast_semantic_mechanism_screen as base

P=base.P

class ObliqueRidgeBank(nn.Module):
    """Projection bank with learnable 1D nonlinear basis per projection."""
    def __init__(self,p,k=32,harmonics=3):
        super().__init__(); self.proj=nn.Linear(p,k); self.lin=nn.Linear(p,1)
        self.sin=nn.Parameter(torch.zeros(k,harmonics)); self.cos=nn.Parameter(torch.zeros(k,harmonics))
        self.poly=nn.Parameter(torch.zeros(k,2)); nn.init.normal_(self.sin,std=.03); nn.init.normal_(self.cos,std=.03)
    def forward(self,x):
        z=self.proj(x); out=self.lin(x).squeeze(1)
        for h in range(1,self.sin.shape[1]+1):
            out=out+(torch.sin(h*z)*self.sin[:,h-1]+torch.cos(h*z)*self.cos[:,h-1]).sum(1)/math.sqrt(z.shape[1])
        out=out+(torch.tanh(z)*self.poly[:,0]+torch.tanh(z).square()*self.poly[:,1]).sum(1)/math.sqrt(z.shape[1])
        return out

class BilinearLowRank(nn.Module):
    """Explicit low-rank bilinear quadratic form plus nonlinear saturation."""
    def __init__(self,p,k=24):
        super().__init__(); self.a=nn.Linear(p,k,bias=False); self.b=nn.Linear(p,k,bias=False)
        self.raw=nn.Parameter(torch.zeros(k)); self.sat=nn.Parameter(torch.zeros(k)); self.lin=nn.Linear(p,1)
        nn.init.normal_(self.raw,std=.04); nn.init.normal_(self.sat,std=.04)
    def forward(self,x):
        z=self.a(x)*self.b(x)
        return self.lin(x).squeeze(1)+(z*self.raw+torch.tanh(1.5*z)*self.sat).sum(1)/math.sqrt(z.shape[1])

class FactorizedQuadratic(nn.Module):
    """Symmetric low-rank quadratic + ridge nonlinearities."""
    def __init__(self,p,k=32):
        super().__init__(); self.u=nn.Linear(p,k,bias=False); self.q=nn.Parameter(torch.zeros(k))
        self.t=nn.Parameter(torch.zeros(k)); self.lin=nn.Linear(p,1)
        nn.init.normal_(self.q,std=.03); nn.init.normal_(self.t,std=.03)
    def forward(self,x):
        z=self.u(x)
        return self.lin(x).squeeze(1)+(z.square()*self.q+torch.tanh(z)*self.t).sum(1)/math.sqrt(z.shape[1])

def train(model,x,y,seed,passes=28,lr=1.5e-3):
    torch.manual_seed(seed); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    xt=torch.from_numpy(x); yt=torch.from_numpy(y); gen=torch.Generator().manual_seed(seed+9)
    best=(1e9,None)
    for _ in range(passes):
        order=torch.randperm(len(x),generator=gen); model.train()
        for s in range(0,len(order),256):
            idx=order[s:s+256]; opt.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(xt[idx]),yt[idx]); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
        with torch.no_grad(): score=float(nn.functional.binary_cross_entropy_with_logits(model(xt),yt))
        if score<best[0]: best=(score,{k:v.detach().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); return model

@torch.no_grad()
def pred(model,x):
    model.eval(); return torch.sigmoid(model(torch.from_numpy(x))).numpy()

def run(regime,seed,out):
    torch.set_num_threads(4); start=time.perf_counter(); problem=base.latent(regime,seed+41)
    x,y,_=base.sample(problem,3000,seed+1001); qx,qy,bp=base.sample(problem,12000,seed+500001)
    sc=StandardScaler().fit(x); x=sc.transform(x).astype("float32"); qx=sc.transform(qx).astype("float32")
    cat=CatBoostClassifier(iterations=300,depth=7,learning_rate=.06,l2_leaf_reg=8,loss_function="Logloss",verbose=False,random_seed=seed,thread_count=4)
    cat.fit(x,y); cm=base.metrics(qy,cat.predict_proba(qx)[:,1]); bayes=base.metrics(qy,bp); gap=cm["nll"]-bayes["nll"]
    makers={"oblique_ridge_bank":lambda:ObliqueRidgeBank(P),"bilinear_low_rank":lambda:BilinearLowRank(P),
      "factorized_quadratic":lambda:FactorizedQuadratic(P),"mlp":lambda:__import__("experiments.synthetic_cat_corner_controller_v6",fromlist=["AffineExpert"]).AffineExpert(P)}
    rows={}
    for i,(name,maker) in enumerate(makers.items()):
        m=train(maker(),x,y,seed+2000+100*i); mm=base.metrics(qy,pred(m,qx))
        rows[name]={"ranking":mm,"delta_nll_vs_catboost":mm["nll"]-cm["nll"],"delta_auc_vs_catboost":mm["auc"]-cm["auc"],
          "gap_recovered":float((cm["nll"]-mm["nll"])/gap)}
    result={"study":"fast_specialist_redesign_v11","regime":regime,"seed":seed,"catboost":cm,"bayes":bayes,
      "catboost_to_bayes_nll_gap":gap,"variants":rows,"seconds":time.perf_counter()-start}
    Path(out).parent.mkdir(parents=True,exist_ok=True); Path(out).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)); print(json.dumps(result,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--regime",choices=["oblique","interaction"],required=True); p.add_argument("--seed",type=int,default=733); p.add_argument("--out",required=True)
    a=p.parse_args(); run(a.regime,a.seed,a.out)
