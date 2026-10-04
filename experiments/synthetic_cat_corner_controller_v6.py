"""Synthetic v6: aggressive differentiable explore/exploit around CatBoost.

Semantic residual experts:
- axis_soft: differentiable axis-aligned soft stumps;
- oblique: learned dense projections with nonlinear gates;
- affine: smooth MLP/affine residual;
- interaction: low-rank multiplicative feature interactions.

All experts train jointly for five reshuffled full passes. Architecture variables
have a separate faster optimizer. Early passes use high-temperature softmax plus
entropy reward and an exploration floor; later passes anneal temperature and
entropy so the controller can exploit a preferred expert or mixture.

The mixed residual is normalized in prediction space and multiplied by a learned
RMS-logit radius, preventing raw expert scale from masquerading as architecture
movement. Synthetic only: no real benchmark or HIGGS audit data.
"""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import experiments.synthetic_cat_corner_controller as v3
import experiments.synthetic_cat_corner_controller_v4 as v4

PASSES=5
BATCH=256
MODEL_LR=8e-4
ARCH_LR=1.2e-2
MIN_RADIUS=0.005
MAX_RADIUS=0.80
EPS=1e-6
TEMPS=(2.5,1.5,0.8,0.4,0.20)
ENTROPY=(0.020,0.012,0.006,0.002,0.0)
FLOORS=(0.12,0.08,0.04,0.015,0.005)

class AxisSoftExpert(nn.Module):
    def __init__(self,p,k=48):
        super().__init__()
        self.feature_logits=nn.Parameter(torch.zeros(k,p))
        self.threshold=nn.Parameter(torch.zeros(k))
        self.left=nn.Parameter(torch.zeros(k))
        self.right=nn.Parameter(torch.zeros(k))
        self.log_temp=nn.Parameter(torch.full((k,),math.log(0.7)))
        nn.init.normal_(self.feature_logits,std=.15)
        nn.init.normal_(self.left,std=.03); nn.init.normal_(self.right,std=.03)
    def forward(self,x):
        # Differentiable feature choice; annealing occurs at architecture mixer,
        # while each stump itself may learn a sparse axis preference.
        w=torch.softmax(self.feature_logits,dim=1)
        score=x@w.T-self.threshold
        q=torch.sigmoid(score/torch.exp(self.log_temp).clamp(.08,3.0))
        return ((1-q)*self.left+q*self.right).sum(dim=1)/math.sqrt(len(self.left))

class ObliqueExpert(nn.Module):
    def __init__(self,p,k=48):
        super().__init__()
        self.proj=nn.Linear(p,k)
        self.value=nn.Parameter(torch.empty(k))
        nn.init.normal_(self.value,std=.04)
    def forward(self,x):
        return (torch.tanh(self.proj(x))*self.value).sum(dim=1)/math.sqrt(len(self.value))

class AffineExpert(nn.Module):
    def __init__(self,p):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(p,96),nn.GELU(),nn.Linear(96,48),nn.GELU(),nn.Linear(48,1))
    def forward(self,x): return self.net(x).squeeze(1)

class InteractionExpert(nn.Module):
    def __init__(self,p,k=32):
        super().__init__()
        self.a=nn.Linear(p,k,bias=False); self.b=nn.Linear(p,k,bias=False)
        self.value=nn.Parameter(torch.empty(k)); self.linear=nn.Linear(p,1)
        nn.init.normal_(self.value,std=.04)
    def forward(self,x):
        z=torch.tanh(self.a(x))*torch.tanh(self.b(x))
        return self.linear(x).squeeze(1)+(z*self.value).sum(dim=1)/math.sqrt(len(self.value))

EXPERT_NAMES=("axis_soft","oblique","affine","interaction")

def build_experts(p,seed):
    torch.manual_seed(seed)
    return nn.ModuleList([AxisSoftExpert(p),ObliqueExpert(p),AffineExpert(p),InteractionExpert(p)])

def radius_from_logit(z):
    return MIN_RADIUS+(MAX_RADIUS-MIN_RADIUS)*torch.sigmoid(z)

def inv_radius(r):
    q=float(np.clip((r-MIN_RADIUS)/(MAX_RADIUS-MIN_RADIUS),1e-4,1-1e-4))
    return math.log(q/(1-q))

def mixture_weights(logits,temp,floor):
    w=torch.softmax(logits/temp,dim=0)
    if floor>0:
        w=(1-len(w)*floor)*w+floor
    return w/w.sum()

def basis(experts,x):
    vals=[]
    for expert in experts:
        raw=expert(x)
        scale=torch.sqrt(torch.mean(raw.square())+EPS)
        vals.append(torch.tanh(raw/scale))
    return torch.stack(vals,dim=1)

def normalized_mix(experts,x,w):
    b=basis(experts,x)
    m=(b*w[None,:]).sum(dim=1)
    return m/torch.sqrt(torch.mean(m.square())+EPS)

def train_controller(x,y,base_logits,prior,seed):
    experts=build_experts(x.shape[1],seed)
    arch_logits=nn.Parameter(torch.zeros(len(EXPERT_NAMES)))
    radius_logit=nn.Parameter(torch.tensor(inv_radius(prior),dtype=torch.float32))
    model_params=list(experts.parameters())
    model_opt=torch.optim.AdamW(model_params,lr=MODEL_LR,weight_decay=1e-5)
    arch_opt=torch.optim.AdamW([arch_logits,radius_logit],lr=ARCH_LR,weight_decay=0.0)
    loss_fn=nn.BCEWithLogitsLoss()
    xt=torch.from_numpy(x); yt=torch.from_numpy(y); bt=torch.from_numpy(base_logits.astype("float32"))
    gen=torch.Generator().manual_seed(seed+7001); history=[]
    for epoch in range(PASSES):
        order=torch.randperm(len(x),generator=gen)
        temp,ent,floor=TEMPS[epoch],ENTROPY[epoch],FLOORS[epoch]
        experts.train()
        for start in range(0,len(order),BATCH):
            idx=order[start:start+BATCH]
            model_opt.zero_grad(set_to_none=True); arch_opt.zero_grad(set_to_none=True)
            w=mixture_weights(arch_logits,temp,floor)
            m=normalized_mix(experts,xt[idx],w)
            r=radius_from_logit(radius_logit)
            empirical=loss_fn(bt[idx]+r*m,yt[idx])
            entropy=-(w*torch.log(w.clamp_min(1e-8))).sum()
            # Entropy reward early; vanishes by pass five for exploitation.
            loss=empirical-ent*entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_params,10.0)
            torch.nn.utils.clip_grad_norm_([arch_logits,radius_logit],5.0)
            model_opt.step(); arch_opt.step()
        with torch.no_grad():
            w=mixture_weights(arch_logits,TEMPS[epoch],FLOORS[epoch])
            history.append({"pass":epoch+1,"temperature":temp,"entropy_reward":ent,
                "exploration_floor":floor,"radius":float(radius_from_logit(radius_logit)),
                "weights":{k:float(z) for k,z in zip(EXPERT_NAMES,w)},
                "architecture_logits":{k:float(z) for k,z in zip(EXPERT_NAMES,arch_logits)}})
    # Final exploit weights use final low temperature/floor.
    w=mixture_weights(arch_logits,TEMPS[-1],FLOORS[-1]).detach().numpy()
    return experts,w,float(radius_from_logit(radius_logit).detach()),history

@torch.no_grad()
def calibration(experts,w,xfit,xeval):
    # Dataset-level normalization for stable function-space radius semantics.
    def raw_matrix(x):
        cols=[]
        for e in experts:
            e.eval(); raw=v3.residual_values(e,x)
            s=max(v4.rms(raw),EPS); cols.append(np.tanh(raw/s))
        return np.stack(cols,axis=1)
    bf=raw_matrix(xfit); be=raw_matrix(xeval)
    mf=bf@w; me=be@w; s=max(v4.rms(mf),EPS)
    return mf/s,me/s

def outer_evidence(x,y,prior,seed):
    skf=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed+501); out=[]
    for fold,(tr,va) in enumerate(skf.split(x,y)):
        tx,ty=x[tr],y[tr]; vx,vy=x[va],y[va]
        oof,trees=v3.oof_cat_logits(tx,ty,seed+10000+fold*100)
        experts,w,r,h=train_controller(tx,ty,oof,prior,seed+20000+fold*1000)
        _,u=calibration(experts,w,tx,vx)
        cat=v3.cat_model(seed+30000+fold); cat.fit(tx,ty,verbose=False)
        bp=cat.predict_proba(vx)[:,1]; bm=v3.metrics(vy,bp); perturb=r*u
        hm=v3.metrics(vy,v3._sigmoid(v3._logit(bp)+perturb))
        out.append({"fold":fold,"inner_oof_catboost_trees":trees,"anchor_trees":int(cat.tree_count_),
          "learned_radius":r,"mixture_weights":{k:float(z) for k,z in zip(EXPERT_NAMES,w)},
          "pass_history":h,"realized_rms_logit_delta":v4.rms(perturb),
          "realized_abs_max_logit_delta":float(np.max(np.abs(perturb))),
          "anchor":bm,"hybrid":hm,"delta_nll":hm["nll"]-bm["nll"],"delta_auc":hm["auc"]-bm["auc"]})
    return out

def run(regime,n,seed,out):
    torch.set_num_threads(4); started=time.perf_counter()
    problem=v3.latent_problem(regime,seed+41)
    x,y=v3.sample_problem(problem,n,seed+1001); qx,qy=v3.sample_problem(problem,max(12000,n),seed+500001)
    sc=StandardScaler().fit(x); x=sc.transform(x).astype("float32"); qx=sc.transform(qx).astype("float32")
    stats=v3.regime_stats(x,y); prior=v4.rms_radius_prior(stats)
    outer=outer_evidence(x,y,prior,seed)
    oof,trees=v3.oof_cat_logits(x,y,seed+60000)
    experts,w,r,h=train_controller(x,y,oof,prior,seed+70000)
    _,u=calibration(experts,w,x,qx)
    cat=v3.cat_model(seed+90000); cat.fit(x,y,verbose=False)
    bp=cat.predict_proba(qx)[:,1]; perturb=r*u; hp=v3._sigmoid(v3._logit(bp)+perturb)
    bm=v3.metrics(qy,bp); hm=v3.metrics(qy,hp)
    result={"study":"synthetic_cat_corner_semantic_explore_exploit_v6","regime":regime,"seed":seed,
      "train_rows":int(n),"ranking_rows":int(len(qy)),"passes":PASSES,"presentations_per_row":PASSES,
      "batch_size":BATCH,"model_lr":MODEL_LR,"architecture_lr":ARCH_LR,
      "temperature_schedule":TEMPS,"entropy_schedule":ENTROPY,"exploration_floor_schedule":FLOORS,
      "same_latent_problem_train_and_ranking":True,"train_only_stats":stats,
      "controller":{"prior_rms_logit_radius":prior,"minimum_radius":MIN_RADIUS,"maximum_radius":MAX_RADIUS,
        "expert_names":EXPERT_NAMES,"exact_catboost_fallback_allowed":False,"outer_fold_evidence":outer,
        "full_oof_catboost_retained_trees":trees,"learned_radius":r,
        "mixture_weights":{k:float(z) for k,z in zip(EXPERT_NAMES,w)},"pass_history":h,
        "ranking_realized_rms_logit_delta":v4.rms(perturb),
        "ranking_realized_abs_max_logit_delta":float(np.max(np.abs(perturb)))},
      "catboost":{"retained_trees":int(cat.tree_count_),"ranking":bm},"hybrid":{"ranking":hm},
      "deltas":{"hybrid_minus_catboost_nll":hm["nll"]-bm["nll"],"hybrid_minus_catboost_auc":hm["auc"]-bm["auc"]},
      "seconds":time.perf_counter()-started}
    Path(out).parent.mkdir(parents=True,exist_ok=True); Path(out).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--regime",choices=["axis_sparse","oblique_dense","mixed"],required=True)
    p.add_argument("--n",type=int,required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--out",required=True)
    a=p.parse_args(); run(a.regime,a.n,a.seed,a.out)
