"""Synthetic v8: matched-anchor nested architecture search.

Fixes v7's objective mismatch. Architecture/radius are NEVER learned against OOF
CatBoost logits and then applied to a different full-fit anchor.

For each outer fold and each of two inner rotations:
  A: fit CatBoost anchor + residual expert weights on inner-fit rows;
  B: freeze both and learn only architecture mixture/radius on unseen inner-select rows;
  C: evaluate the resulting correction on untouched outer-evidence rows using
     the SAME CatBoost anchor and SAME residual experts.

The six outer/rotation evidence cells are then aggregated into a consensus
mixture/radius. Final CatBoost and expert weights are refit on all training rows,
but architecture variables are frozen to that nested consensus; they are not
allowed to relearn an OOF-specific correction.

Synthetic only. No real benchmark or HIGGS shadow data.
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
import experiments.synthetic_cat_corner_controller_v6 as v6

MODEL_PASSES=8
ARCH_UPDATES=240
MODEL_BATCH=256
ARCH_BATCH=64
MODEL_LR=8e-4
ARCH_LR=8e-3
MIN_RADIUS=0.005
MAX_RADIUS=0.60
EPS=1e-6

def train_experts(experts,x,y,anchor_logits,seed,passes=MODEL_PASSES):
    params=list(experts.parameters()); opt=torch.optim.AdamW(params,lr=MODEL_LR,weight_decay=1e-5)
    xt=torch.from_numpy(x); yt=torch.from_numpy(y); bt=torch.from_numpy(anchor_logits.astype("float32"))
    gen=torch.Generator().manual_seed(seed+11)
    # broad fixed mixture while experts learn useful residual directions
    w=torch.full((len(v6.EXPERT_NAMES),),1/len(v6.EXPERT_NAMES))
    radius=torch.tensor(0.08)
    for _ in range(passes):
        order=torch.randperm(len(x),generator=gen)
        for s in range(0,len(order),MODEL_BATCH):
            idx=order[s:s+MODEL_BATCH]; opt.zero_grad(set_to_none=True)
            m=v6.normalized_mix(experts,xt[idx],w)
            loss=nn.functional.binary_cross_entropy_with_logits(bt[idx]+radius*m,yt[idx])
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,10.0); opt.step()

def learn_architecture(experts,x,y,anchor_logits,prior,seed):
    for p in experts.parameters(): p.requires_grad_(False)
    logits=nn.Parameter(torch.zeros(len(v6.EXPERT_NAMES)))
    rlogit=nn.Parameter(torch.tensor(v6.inv_radius(prior),dtype=torch.float32))
    opt=torch.optim.AdamW([logits,rlogit],lr=ARCH_LR)
    xt=torch.from_numpy(x); yt=torch.from_numpy(y); bt=torch.from_numpy(anchor_logits.astype("float32"))
    gen=torch.Generator().manual_seed(seed+29); history=[]
    for step in range(ARCH_UPDATES):
        idx=torch.randint(0,len(x),(min(ARCH_BATCH,len(x)),),generator=gen)
        # exploration first half, exploitation second half
        frac=step/max(ARCH_UPDATES-1,1)
        temp=max(0.18,2.0*(1-frac)+0.18*frac)
        floor=max(0.002,0.08*(1-frac))
        ent=0.012*max(0.0,1-2*frac)
        opt.zero_grad(set_to_none=True)
        w=v6.mixture_weights(logits,temp,floor); m=v6.normalized_mix(experts,xt[idx],w)
        r=v6.radius_from_logit(rlogit)
        empirical=nn.functional.binary_cross_entropy_with_logits(bt[idx]+r*m,yt[idx])
        entropy=-(w*torch.log(w.clamp_min(1e-8))).sum()
        (empirical-ent*entropy).backward()
        torch.nn.utils.clip_grad_norm_([logits,rlogit],5.0); opt.step()
        if step in (0,59,119,179,239):
            history.append({"update":step+1,"temperature":temp,"floor":floor,"entropy_reward":ent,
                "radius":float(v6.radius_from_logit(rlogit).detach()),
                "weights":{k:float(z) for k,z in zip(v6.EXPERT_NAMES,v6.mixture_weights(logits,temp,floor).detach())}})
    w=v6.mixture_weights(logits,0.18,0.002).detach().numpy()
    r=float(v6.radius_from_logit(rlogit).detach())
    for p in experts.parameters(): p.requires_grad_(True)
    return w,r,history

def fit_anchor(x,y,seed):
    m=v3.cat_model(seed); m.fit(x,y,verbose=False); return m

def evidence_cell(x,y,fit_idx,select_idx,evidence_idx,prior,seed,outer,rotation):
    fx,fy=x[fit_idx],y[fit_idx]; sx,sy=x[select_idx],y[select_idx]; ex,ey=x[evidence_idx],y[evidence_idx]
    anchor=fit_anchor(fx,fy,seed+1000)
    fl=v3._logit(anchor.predict_proba(fx)[:,1]); sl=v3._logit(anchor.predict_proba(sx)[:,1])
    experts=v6.build_experts(x.shape[1],seed+2000)
    train_experts(experts,fx,fy,fl,seed+3000)
    w,r,h=learn_architecture(experts,sx,sy,sl,prior,seed+4000)
    _,unit=v6.calibration(experts,w,fx,ex)
    ep=anchor.predict_proba(ex)[:,1]; perturb=r*unit
    hp=v3._sigmoid(v3._logit(ep)+perturb)
    bm=v3.metrics(ey,ep); hm=v3.metrics(ey,hp)
    return {"outer_fold":outer,"rotation":rotation,"fit_rows":int(len(fit_idx)),"select_rows":int(len(select_idx)),
      "evidence_rows":int(len(evidence_idx)),"anchor_trees":int(anchor.tree_count_),
      "weights":{k:float(z) for k,z in zip(v6.EXPERT_NAMES,w)},"radius":r,"architecture_history":h,
      "realized_rms_logit_delta":v4.rms(perturb),"anchor":bm,"hybrid":hm,
      "delta_nll":hm["nll"]-bm["nll"],"delta_auc":hm["auc"]-bm["auc"]}

def nested_search(x,y,prior,seed):
    outer=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed+501); cells=[]
    for ofold,(otr,ova) in enumerate(outer.split(x,y)):
        inner=StratifiedKFold(n_splits=2,shuffle=True,random_state=seed+900+ofold)
        a,b=next(inner.split(x[otr],y[otr]))
        A=otr[a]; B=otr[b]
        cells.append(evidence_cell(x,y,A,B,ova,prior,seed+ofold*10000,ofold,0))
        cells.append(evidence_cell(x,y,B,A,ova,prior,seed+ofold*10000+5000,ofold,1))
    # Evidence-weighted consensus: harmful cells receive no extra authority;
    # beneficial cells get smoothly more weight, but every cell contributes.
    dn=np.array([c["delta_nll"] for c in cells]); scale=max(float(np.std(dn)),1e-4)
    ew=np.exp(np.clip(-dn/scale,-3,3)); ew=ew/ew.sum()
    W=np.array([[c["weights"][k] for k in v6.EXPERT_NAMES] for c in cells])
    w=(ew[:,None]*W).sum(axis=0); w=w/w.sum()
    radii=np.array([c["radius"] for c in cells])
    # Geometric consensus prevents one runaway fold from dominating radius.
    r=float(np.exp(np.sum(ew*np.log(np.clip(radii,MIN_RADIUS,MAX_RADIUS)))))
    # If nested evidence is net harmful, contract rather than zero-collapse.
    mean_delta=float(np.mean(dn))
    if mean_delta>=0: r=max(MIN_RADIUS,min(r,0.02))
    return cells,w,r,{"mean_delta_nll":mean_delta,"median_delta_nll":float(np.median(dn)),
      "improving_cells":int(np.sum(dn<0)),"evidence_weights":[float(z) for z in ew]}

def run(regime,n,seed,out):
    torch.set_num_threads(4); started=time.perf_counter()
    problem=v3.latent_problem(regime,seed+41)
    x,y=v3.sample_problem(problem,n,seed+1001); qx,qy=v3.sample_problem(problem,max(12000,n),seed+500001)
    sc=StandardScaler().fit(x); x=sc.transform(x).astype("float32"); qx=sc.transform(qx).astype("float32")
    stats=v3.regime_stats(x,y); prior=v4.rms_radius_prior(stats)
    cells,w,r,summary=nested_search(x,y,prior,seed)
    # Final refit uses the SAME anchor semantics: CatBoost fitted on exactly the
    # rows whose residual experts are trained. Architecture is frozen.
    cat=fit_anchor(x,y,seed+90000); logits=v3._logit(cat.predict_proba(x)[:,1])
    experts=v6.build_experts(x.shape[1],seed+91000); train_experts(experts,x,y,logits,seed+92000,passes=12)
    _,unit=v6.calibration(experts,w,x,qx)
    bp=cat.predict_proba(qx)[:,1]; perturb=r*unit; hp=v3._sigmoid(v3._logit(bp)+perturb)
    bm=v3.metrics(qy,bp); hm=v3.metrics(qy,hp)
    result={"study":"synthetic_cat_corner_matched_anchor_nested_v8","regime":regime,"seed":seed,
      "train_rows":int(n),"ranking_rows":int(len(qy)),"same_latent_problem_train_and_ranking":True,
      "train_only_stats":stats,"controller":{"prior_radius":prior,"nested_cells":cells,"nested_summary":summary,
      "consensus_weights":{k:float(z) for k,z in zip(v6.EXPERT_NAMES,w)},"consensus_radius":r,
      "final_architecture_relearned":False,"matched_anchor_contract":True,
      "ranking_realized_rms_logit_delta":v4.rms(perturb),"ranking_realized_abs_max_logit_delta":float(np.max(np.abs(perturb)))},
      "catboost":{"retained_trees":int(cat.tree_count_),"ranking":bm},"hybrid":{"ranking":hm},
      "deltas":{"hybrid_minus_catboost_nll":hm["nll"]-bm["nll"],"hybrid_minus_catboost_auc":hm["auc"]-bm["auc"]},
      "seconds":time.perf_counter()-started}
    Path(out).parent.mkdir(parents=True,exist_ok=True); Path(out).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--regime",choices=["axis_sparse","oblique_dense","mixed"],required=True)
    p.add_argument("--n",type=int,required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--out",required=True)
    a=p.parse_args(); run(a.regime,a.n,a.seed,a.out)
