"""Synthetic v9: preserve the actual nested correction functions.

v8 transferred architecture hyperparameters but retrained expert functions, and
that refit drift weakened the nested-to-final relationship. v9 retains the six
actual nested correction functions.

For each outer fold:
- two inner rotations fit matched CatBoost anchors + residual experts and learn
  architecture on disjoint selection rows;
- both frozen correction functions predict the untouched outer fold; their
  average is that fold's honest OOF correction teacher;
- a separate CatBoost transfer anchor is fit on the whole outer-training fold,
  giving an OOF anchor closer to final full-fit CatBoost semantics.

Across all rows, a single bounded scalar beta is fit on OOF transfer-anchor
logits + honest correction teachers. Final evaluation uses full-fit CatBoost
plus beta times the mean of the six PRESERVED correction functions evaluated on
ranking rows. No expert-function refit occurs.

This deliberately tests function preservation before any distillation/compression.
Synthetic only; no real benchmark or HIGGS shadow data.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import experiments.synthetic_cat_corner_controller as v3
import experiments.synthetic_cat_corner_controller_v4 as v4
import experiments.synthetic_cat_corner_controller_v6 as v6
import experiments.synthetic_cat_corner_controller_v8 as v8

BETA_MIN=0.05
BETA_MAX=2.0
BETA_STEPS=400
EPS=1e-8

def fit_cell_function(x,y,fit_idx,select_idx,evidence_idx,prior,seed,outer,rotation):
    fx,fy=x[fit_idx],y[fit_idx]; sx,sy=x[select_idx],y[select_idx]
    anchor=v8.fit_anchor(fx,fy,seed+1000)
    fl=v3._logit(anchor.predict_proba(fx)[:,1])
    sl=v3._logit(anchor.predict_proba(sx)[:,1])
    experts=v6.build_experts(x.shape[1],seed+2000)
    v8.train_experts(experts,fx,fy,fl,seed+3000)
    w,r,h=v8.learn_architecture(experts,sx,sy,sl,prior,seed+4000)

    # Freeze the discovered function. Calibration uses only its fit rows.
    fit_basis=[]
    for expert in experts:
        raw=v3.residual_values(expert,fx)
        scale=max(v4.rms(raw),EPS)
        fit_basis.append(np.tanh(raw/scale))
    fit_basis=np.stack(fit_basis,axis=1)
    mixed_fit=fit_basis@w
    mix_scale=max(v4.rms(mixed_fit),EPS)

    def correction(z):
        cols=[]
        for expert in experts:
            raw_fit=v3.residual_values(expert,fx)
            scale=max(v4.rms(raw_fit),EPS)
            raw=v3.residual_values(expert,z)
            cols.append(np.tanh(raw/scale))
        basis=np.stack(cols,axis=1)
        return r*(basis@w)/mix_scale

    ev_delta=correction(x[evidence_idx])
    ep=anchor.predict_proba(x[evidence_idx])[:,1]
    bm=v3.metrics(y[evidence_idx],ep)
    hm=v3.metrics(y[evidence_idx],v3._sigmoid(v3._logit(ep)+ev_delta))
    meta={"outer_fold":outer,"rotation":rotation,"fit_rows":int(len(fit_idx)),
      "select_rows":int(len(select_idx)),"evidence_rows":int(len(evidence_idx)),
      "anchor_trees":int(anchor.tree_count_),"weights":{k:float(z) for k,z in zip(v6.EXPERT_NAMES,w)},
      "radius":float(r),"architecture_history":h,"realized_rms_logit_delta":v4.rms(ev_delta),
      "anchor":bm,"hybrid":hm,"delta_nll":hm["nll"]-bm["nll"],"delta_auc":hm["auc"]-bm["auc"]}
    return correction,ev_delta,meta

def calibrate_beta(y,anchor_logits,teacher_delta):
    yt=torch.from_numpy(y.astype("float32"))
    at=torch.from_numpy(anchor_logits.astype("float32"))
    dt=torch.from_numpy(teacher_delta.astype("float32"))
    raw=torch.nn.Parameter(torch.tensor(0.0))
    opt=torch.optim.Adam([raw],lr=0.05)
    loss_fn=torch.nn.BCEWithLogitsLoss()
    hist=[]
    for step in range(BETA_STEPS):
        opt.zero_grad(set_to_none=True)
        beta=BETA_MIN+(BETA_MAX-BETA_MIN)*torch.sigmoid(raw)
        loss=loss_fn(at+beta*dt,yt); loss.backward(); opt.step()
        if step in (0,49,99,199,399):
            hist.append({"step":step+1,"beta":float(beta.detach()),"oof_nll":float(loss.detach())})
    beta=float((BETA_MIN+(BETA_MAX-BETA_MIN)*torch.sigmoid(raw)).detach())
    base=v3.metrics(y,v3._sigmoid(anchor_logits))
    hybrid=v3.metrics(y,v3._sigmoid(anchor_logits+beta*teacher_delta))
    return beta,hist,base,hybrid

def build_preserved_ensemble(x,y,qx,prior,seed):
    outer=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed+501)
    oof_delta=np.zeros(len(y),dtype="float64")
    oof_anchor_logits=np.zeros(len(y),dtype="float64")
    ranking_functions=[]; cells=[]; fold_meta=[]

    for ofold,(otr,ova) in enumerate(outer.split(x,y)):
        inner=StratifiedKFold(n_splits=2,shuffle=True,random_state=seed+900+ofold)
        a,b=next(inner.split(x[otr],y[otr])); A=otr[a]; B=otr[b]
        f0,d0,m0=fit_cell_function(x,y,A,B,ova,prior,seed+ofold*10000,ofold,0)
        f1,d1,m1=fit_cell_function(x,y,B,A,ova,prior,seed+ofold*10000+5000,ofold,1)
        cells.extend([m0,m1]); ranking_functions.extend([f0,f1])
        oof_delta[ova]=0.5*(d0+d1)

        # Transfer anchor is trained on all rows except evidence fold, making its
        # semantics closer to the final full-data anchor than either inner anchor.
        transfer=v8.fit_anchor(x[otr],y[otr],seed+70000+ofold)
        oof_anchor_logits[ova]=v3._logit(transfer.predict_proba(x[ova])[:,1])
        fold_meta.append({"outer_fold":ofold,"transfer_anchor_trees":int(transfer.tree_count_),
          "teacher_rms":v4.rms(oof_delta[ova]),"teacher_abs_max":float(np.max(np.abs(oof_delta[ova])))})

    beta,beta_history,oof_base,oof_hybrid=calibrate_beta(y,oof_anchor_logits,oof_delta)
    ranking_delta=np.mean(np.stack([fn(qx) for fn in ranking_functions],axis=0),axis=0)
    return ranking_delta,beta,{"cells":cells,"outer_folds":fold_meta,
      "oof_teacher_rms":v4.rms(oof_delta),"oof_teacher_abs_max":float(np.max(np.abs(oof_delta))),
      "beta":beta,"beta_history":beta_history,"oof_transfer_anchor":oof_base,
      "oof_transfer_hybrid":oof_hybrid,"oof_delta_nll":oof_hybrid["nll"]-oof_base["nll"],
      "oof_delta_auc":oof_hybrid["auc"]-oof_base["auc"]}

def run(regime,n,seed,out):
    torch.set_num_threads(4); started=time.perf_counter()
    problem=v3.latent_problem(regime,seed+41)
    x,y=v3.sample_problem(problem,n,seed+1001)
    qx,qy=v3.sample_problem(problem,max(12000,n),seed+500001)
    sc=StandardScaler().fit(x); x=sc.transform(x).astype("float32"); qx=sc.transform(qx).astype("float32")
    stats=v3.regime_stats(x,y); prior=v4.rms_radius_prior(stats)
    ranking_teacher,beta,controller=build_preserved_ensemble(x,y,qx,prior,seed)

    cat=v8.fit_anchor(x,y,seed+90000)
    bp=cat.predict_proba(qx)[:,1]
    perturb=beta*ranking_teacher
    hp=v3._sigmoid(v3._logit(bp)+perturb)
    bm=v3.metrics(qy,bp); hm=v3.metrics(qy,hp)
    result={"study":"synthetic_cat_corner_preserved_function_ensemble_v9","regime":regime,"seed":seed,
      "train_rows":int(n),"ranking_rows":int(len(qy)),"same_latent_problem_train_and_ranking":True,
      "train_only_stats":stats,"controller":controller,
      "function_transfer":{"expert_functions_refit":False,"nested_functions_preserved":True,
        "ensemble_members":6,"transfer_beta":beta,
        "ranking_teacher_rms_before_beta":v4.rms(ranking_teacher),
        "ranking_realized_rms_logit_delta":v4.rms(perturb),
        "ranking_realized_abs_max_logit_delta":float(np.max(np.abs(perturb)))},
      "catboost":{"retained_trees":int(cat.tree_count_),"ranking":bm},"hybrid":{"ranking":hm},
      "deltas":{"hybrid_minus_catboost_nll":hm["nll"]-bm["nll"],"hybrid_minus_catboost_auc":hm["auc"]-bm["auc"]},
      "seconds":time.perf_counter()-started}
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--regime",choices=["axis_sparse","oblique_dense","mixed"],required=True)
    p.add_argument("--n",type=int,required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--out",required=True)
    a=p.parse_args(); run(a.regime,a.n,a.seed,a.out)
