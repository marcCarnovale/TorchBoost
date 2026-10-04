"""Synthetic v5: differentiable CatBoost-corner exploration.

Three TorchBoost residual directions, softmax-remixed continuously. A learned
prediction-space radius expands/contracts differentiably. Training uses exactly
five reshuffled full passes: every fit row is presented once per pass. The
residual mixture is RMS-normalized in logit space so raw network scale cannot
fake exploration distance. Synthetic only; no HIGGS or real benchmark data.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import experiments.synthetic_cat_corner_controller as v3
import experiments.synthetic_cat_corner_controller_v4 as v4

PASSES=5
BATCH=256
MIN_RADIUS=0.005
MAX_RADIUS=0.60
EPS=1e-6

def logit_radius(raw):
    x=(raw-MIN_RADIUS)/(MAX_RADIUS-MIN_RADIUS)
    x=float(np.clip(x,1e-4,1-1e-4))
    return math.log(x/(1-x))

def batch_unit(models, xb):
    pieces=[]
    for model in models:
        raw=model(xb)
        scale=torch.sqrt(torch.mean(raw.square())+EPS)
        pieces.append(torch.tanh(raw/scale))
    return torch.stack(pieces,dim=1)

def train_controller(x,y,base_logits,prior,seed):
    names=list(v3.DIRECTIONS)
    models=[v3.build_residual(x.shape[1],name,seed+101*i) for i,name in enumerate(names)]
    mix_logits=torch.nn.Parameter(torch.zeros(len(names)))
    radius_logit=torch.nn.Parameter(torch.tensor(logit_radius(prior),dtype=torch.float32))
    params=[p for m in models for p in m.parameters()]+[mix_logits,radius_logit]
    opt=torch.optim.AdamW(params,lr=8e-4,weight_decay=1e-5)
    loss_fn=torch.nn.BCEWithLogitsLoss()
    xt=torch.from_numpy(x); yt=torch.from_numpy(y)
    bt=torch.from_numpy(base_logits.astype("float32"))
    gen=torch.Generator().manual_seed(seed+7001)
    history=[]
    for epoch in range(PASSES):
        order=torch.randperm(len(x),generator=gen)
        for m in models: m.train()
        for start in range(0,len(order),BATCH):
            idx=order[start:start+BATCH]
            opt.zero_grad(set_to_none=True)
            basis=batch_unit(models,xt[idx])
            weights=torch.softmax(mix_logits,dim=0)
            mixed=(basis*weights[None,:]).sum(dim=1)
            mixed=mixed/torch.sqrt(torch.mean(mixed.square())+EPS)
            radius=MIN_RADIUS+(MAX_RADIUS-MIN_RADIUS)*torch.sigmoid(radius_logit)
            pred=bt[idx]+radius*mixed
            loss=loss_fn(pred,yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,10.0)
            opt.step()
        history.append({
            "pass":epoch+1,
            "radius":float((MIN_RADIUS+(MAX_RADIUS-MIN_RADIUS)*torch.sigmoid(radius_logit)).detach()),
            "weights":[float(z) for z in torch.softmax(mix_logits,dim=0).detach()],
        })
    return models,torch.softmax(mix_logits,dim=0).detach().numpy(),float(
        (MIN_RADIUS+(MAX_RADIUS-MIN_RADIUS)*torch.sigmoid(radius_logit)).detach()
    ),history

@torch.no_grad()
def calibrated_unit(models,weights,xfit,xeval):
    fit=[]; ev=[]
    for model in models:
        model.eval()
        rf=v3.residual_values(model,xfit); re=v3.residual_values(model,xeval)
        scale=max(v4.rms(rf),EPS)
        fit.append(np.tanh(rf/scale)); ev.append(np.tanh(re/scale))
    mf=sum(float(w)*z for w,z in zip(weights,fit))
    me=sum(float(w)*z for w,z in zip(weights,ev))
    scale=max(v4.rms(mf),EPS)
    return mf/scale,me/scale

def outer_evidence(x,y,prior,seed):
    skf=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed+501)
    records=[]
    for fold,(tr,va) in enumerate(skf.split(x,y)):
        tx,ty=x[tr],y[tr]; vx,vy=x[va],y[va]
        oof,trees=v3.oof_cat_logits(tx,ty,seed+10000+fold*100)
        models,w,r,history=train_controller(tx,ty,oof,prior,seed+20000+fold*1000)
        _,unit=calibrated_unit(models,w,tx,vx)
        anchor=v3.cat_model(seed+30000+fold); anchor.fit(tx,ty,verbose=False)
        ap=anchor.predict_proba(vx)[:,1]; al=v3._logit(ap)
        am=v3.metrics(vy,ap); perturb=r*unit
        hm=v3.metrics(vy,v3._sigmoid(al+perturb))
        records.append({
            "fold":fold,"inner_oof_catboost_trees":trees,
            "anchor_trees":int(anchor.tree_count_),
            "learned_radius":r,
            "mixture_weights":{k:float(z) for k,z in zip(v3.DIRECTIONS,w)},
            "pass_history":history,
            "realized_rms_logit_delta":v4.rms(perturb),
            "realized_abs_max_logit_delta":float(np.max(np.abs(perturb))),
            "anchor":am,"hybrid":hm,
            "delta_nll":hm["nll"]-am["nll"],
            "delta_auc":hm["auc"]-am["auc"],
        })
    return records

def run(regime,n,seed,out):
    torch.set_num_threads(4); started=time.perf_counter()
    problem=v3.latent_problem(regime,seed+41)
    x,y=v3.sample_problem(problem,n,seed+1001)
    qx,qy=v3.sample_problem(problem,max(12000,n),seed+500001)
    scaler=StandardScaler().fit(x)
    x=scaler.transform(x).astype("float32"); qx=scaler.transform(qx).astype("float32")
    stats=v3.regime_stats(x,y); prior=v4.rms_radius_prior(stats)
    outer=outer_evidence(x,y,prior,seed)
    full_oof,full_trees=v3.oof_cat_logits(x,y,seed+60000)
    models,w,r,history=train_controller(x,y,full_oof,prior,seed+70000)
    _,unit=calibrated_unit(models,w,x,qx)
    cat=v3.cat_model(seed+90000); cat.fit(x,y,verbose=False)
    bp=cat.predict_proba(qx)[:,1]; perturb=r*unit
    hp=v3._sigmoid(v3._logit(bp)+perturb)
    bm=v3.metrics(qy,bp); hm=v3.metrics(qy,hp)
    result={
      "study":"synthetic_cat_corner_differentiable_remix_v5",
      "regime":regime,"seed":seed,"train_rows":int(n),"ranking_rows":int(len(qy)),
      "passes":PASSES,"presentations_per_row":PASSES,"batch_size":BATCH,
      "same_latent_problem_train_and_ranking":True,"train_only_stats":stats,
      "controller":{
        "prior_rms_logit_radius":prior,"minimum_radius":MIN_RADIUS,
        "maximum_radius":MAX_RADIUS,"exact_catboost_fallback_allowed":False,
        "directions":v3.DIRECTIONS,"outer_fold_evidence":outer,
        "full_oof_catboost_retained_trees":full_trees,
        "learned_radius":r,
        "mixture_weights":{k:float(z) for k,z in zip(v3.DIRECTIONS,w)},
        "pass_history":history,
        "ranking_realized_rms_logit_delta":v4.rms(perturb),
        "ranking_realized_abs_max_logit_delta":float(np.max(np.abs(perturb))),
      },
      "catboost":{"retained_trees":int(cat.tree_count_),"ranking":bm},
      "hybrid":{"ranking":hm},
      "deltas":{"hybrid_minus_catboost_nll":hm["nll"]-bm["nll"],
                "hybrid_minus_catboost_auc":hm["auc"]-bm["auc"]},
      "seconds":time.perf_counter()-started,
    }
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--regime",choices=["axis_sparse","oblique_dense","mixed"],required=True)
    p.add_argument("--n",type=int,required=True); p.add_argument("--seed",type=int,required=True)
    p.add_argument("--out",required=True); a=p.parse_args(); run(a.regime,a.n,a.seed,a.out)
