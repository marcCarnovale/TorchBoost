"""Synthetic v7: adaptive bi-level explore/exploit around CatBoost.

This replaces a fixed epoch count with an architecture-update budget and an
adaptive controller. Expert parameters and architecture variables are optimized
on different rotating row streams. The streams swap roles every cycle pair, so
all rows eventually serve both fitting and architecture selection without a
permanent validation split.

Exploration is stateful rather than tied to epoch number:
- reheat when the preferred expert changes or mixture weights jump;
- cool when a winner persists and mixture drift falls;
- once >= TARGET_ARCH_UPDATES are accumulated and the architecture is stable,
  enter a short low-temperature exploitation phase;
- continue up to MAX_ARCH_UPDATES if evidence is still moving.

Semantic experts and prediction-space normalization come from v6.
Synthetic only: no real benchmark or HIGGS audit data.
"""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import experiments.synthetic_cat_corner_controller as v3
import experiments.synthetic_cat_corner_controller_v4 as v4
import experiments.synthetic_cat_corner_controller_v6 as v6

MODEL_BATCH=256
ARCH_BATCH=64
MODEL_LR=8e-4
ARCH_LR=1.2e-2
TARGET_ARCH_UPDATES=320
MAX_ARCH_UPDATES=640
MIN_CYCLES=6
MAX_CYCLES=40
EXPLOIT_CYCLES=2
START_TEMP=2.5
MIN_TEMP=0.12
START_ENTROPY=0.020
START_FLOOR=0.10
MIN_FLOOR=0.002
EPS=1e-6

def freeze(module,flag):
    for p in module.parameters():
        p.requires_grad_(not flag)

def empirical_batch(experts,arch_logits,radius_logit,xb,yb,bb,temp,floor):
    w=v6.mixture_weights(arch_logits,temp,floor)
    m=v6.normalized_mix(experts,xb,w)
    r=v6.radius_from_logit(radius_logit)
    loss=nn.functional.binary_cross_entropy_with_logits(bb+r*m,yb)
    return loss,w,r

def adaptive_train(x,y,base_logits,prior,seed):
    experts=v6.build_experts(x.shape[1],seed)
    arch_logits=nn.Parameter(torch.zeros(len(v6.EXPERT_NAMES)))
    radius_logit=nn.Parameter(torch.tensor(v6.inv_radius(prior),dtype=torch.float32))
    model_params=list(experts.parameters())
    model_opt=torch.optim.AdamW(model_params,lr=MODEL_LR,weight_decay=1e-5)
    arch_opt=torch.optim.AdamW([arch_logits,radius_logit],lr=ARCH_LR,weight_decay=0.0)
    xt=torch.from_numpy(x); yt=torch.from_numpy(y); bt=torch.from_numpy(base_logits.astype("float32"))
    gen=torch.Generator().manual_seed(seed+7001)
    temp=START_TEMP; entropy_coef=START_ENTROPY; floor=START_FLOOR
    arch_updates=0; history=[]; prev_w=None; prev_winner=None; winner_streak=0
    exploit_left=None; pair_a=None; pair_b=None

    for cycle in range(1,MAX_CYCLES+1):
        # New random role partition every two cycles; roles swap on the second.
        if cycle%2==1 or pair_a is None:
            perm=torch.randperm(len(x),generator=gen)
            cut=len(x)//2; pair_a=perm[:cut]; pair_b=perm[cut:]
        model_idx,arch_idx=(pair_a,pair_b) if cycle%2==1 else (pair_b,pair_a)

        # Expert update stream: architecture is fixed/detached.
        experts.train(); order=model_idx[torch.randperm(len(model_idx),generator=gen)]
        for start in range(0,len(order),MODEL_BATCH):
            idx=order[start:start+MODEL_BATCH]
            model_opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                w=v6.mixture_weights(arch_logits,temp,floor)
                r=v6.radius_from_logit(radius_logit)
            m=v6.normalized_mix(experts,xt[idx],w)
            loss=nn.functional.binary_cross_entropy_with_logits(bt[idx]+r*m,yt[idx])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model_params,10.0); model_opt.step()

        # Architecture stream: freeze experts so only mixture/radius move.
        freeze(experts,True); order=arch_idx[torch.randperm(len(arch_idx),generator=gen)]
        arch_losses=[]
        for start in range(0,len(order),ARCH_BATCH):
            idx=order[start:start+ARCH_BATCH]
            arch_opt.zero_grad(set_to_none=True)
            empirical,w,r=empirical_batch(experts,arch_logits,radius_logit,xt[idx],yt[idx],bt[idx],temp,floor)
            entropy=-(w*torch.log(w.clamp_min(1e-8))).sum()
            loss=empirical-entropy_coef*entropy
            loss.backward(); torch.nn.utils.clip_grad_norm_([arch_logits,radius_logit],5.0); arch_opt.step()
            arch_updates+=1; arch_losses.append(float(empirical.detach()))
            if arch_updates>=MAX_ARCH_UPDATES: break
        freeze(experts,False)

        with torch.no_grad():
            w=v6.mixture_weights(arch_logits,temp,floor).detach().cpu().numpy()
            r=float(v6.radius_from_logit(radius_logit).detach())
        winner=int(np.argmax(w)); drift=float(np.abs(w-prev_w).sum()) if prev_w is not None else float("nan")
        if prev_winner==winner: winner_streak+=1
        else: winner_streak=1
        changed=prev_winner is not None and winner!=prev_winner
        mean_arch=float(np.mean(arch_losses)) if arch_losses else None
        drift_json=float(drift) if np.isfinite(drift) else None
        history.append({"cycle":cycle,"architecture_updates":arch_updates,"temperature":temp,
            "entropy_reward":entropy_coef,"exploration_floor":floor,"radius":r,
            "weights":{k:float(z) for k,z in zip(v6.EXPERT_NAMES,w)},
            "winner":v6.EXPERT_NAMES[winner],"winner_streak":winner_streak,
            "weight_l1_drift":drift_json,"architecture_stream_loss":mean_arch,
            "role_model_rows":int(len(model_idx)),"role_arch_rows":int(len(arch_idx))})

        # If exploitation was triggered but the winner breaks, re-open search.
        if exploit_left is not None:
            if changed or (np.isfinite(drift) and drift>0.12):
                exploit_left=None
                temp=min(START_TEMP,max(0.6,temp*1.8)); entropy_coef=max(0.006,entropy_coef)
                floor=min(0.08,max(0.02,floor*2.0))
            else:
                exploit_left-=1
                if exploit_left<=0: break
        else:
            # Reheat unstable search; cool stable search.
            if changed or (np.isfinite(drift) and drift>0.22):
                temp=min(START_TEMP,temp*1.45)
                entropy_coef=min(START_ENTROPY,max(0.004,entropy_coef*1.35))
                floor=min(START_FLOOR,max(0.015,floor*1.35))
            elif winner_streak>=2 and np.isfinite(drift) and drift<0.12:
                temp=max(MIN_TEMP,temp*0.72)
                entropy_coef=max(0.0,entropy_coef*0.70)
                floor=max(MIN_FLOOR,floor*0.70)
            else:
                temp=max(0.35,temp*0.94)
                entropy_coef=max(0.001,entropy_coef*0.92)
                floor=max(0.008,floor*0.92)

            # Stable target reached: force a short pure-exploitation tail.
            recent=history[-3:]
            stable=(cycle>=MIN_CYCLES and arch_updates>=TARGET_ARCH_UPDATES and
                    winner_streak>=3 and len(recent)==3 and
                    all(h["weight_l1_drift"] is not None and h["weight_l1_drift"]<0.06 for h in recent))
            if stable:
                exploit_left=EXPLOIT_CYCLES
                temp=MIN_TEMP; entropy_coef=0.0; floor=MIN_FLOOR

        prev_w=w.copy(); prev_winner=winner
        if arch_updates>=MAX_ARCH_UPDATES: break

    with torch.no_grad():
        w=v6.mixture_weights(arch_logits,MIN_TEMP,MIN_FLOOR).detach().cpu().numpy()
        r=float(v6.radius_from_logit(radius_logit).detach())
    return experts,w,r,history,arch_updates

def run(regime,n,seed,out):
    torch.set_num_threads(4); started=time.perf_counter()
    problem=v3.latent_problem(regime,seed+41)
    x,y=v3.sample_problem(problem,n,seed+1001); qx,qy=v3.sample_problem(problem,max(12000,n),seed+500001)
    sc=StandardScaler().fit(x); x=sc.transform(x).astype("float32"); qx=sc.transform(qx).astype("float32")
    stats=v3.regime_stats(x,y); prior=v4.rms_radius_prior(stats)
    # Full-data OOF anchor preserves honest residual targets; fresh synthetic ranking remains untouched.
    oof,trees=v3.oof_cat_logits(x,y,seed+60000)
    experts,w,r,h,updates=adaptive_train(x,y,oof,prior,seed+70000)
    _,u=v6.calibration(experts,w,x,qx)
    cat=v3.cat_model(seed+90000); cat.fit(x,y,verbose=False)
    bp=cat.predict_proba(qx)[:,1]; perturb=r*u; hp=v3._sigmoid(v3._logit(bp)+perturb)
    bm=v3.metrics(qy,bp); hm=v3.metrics(qy,hp)
    result={"study":"synthetic_cat_corner_adaptive_bilevel_v7","regime":regime,"seed":seed,
      "train_rows":int(n),"ranking_rows":int(len(qy)),"model_batch":MODEL_BATCH,"architecture_batch":ARCH_BATCH,
      "target_architecture_updates":TARGET_ARCH_UPDATES,"max_architecture_updates":MAX_ARCH_UPDATES,
      "actual_architecture_updates":updates,"cycles":len(h),"model_lr":MODEL_LR,"architecture_lr":ARCH_LR,
      "same_latent_problem_train_and_ranking":True,"train_only_stats":stats,
      "controller":{"prior_rms_logit_radius":prior,"minimum_radius":v6.MIN_RADIUS,"maximum_radius":v6.MAX_RADIUS,
        "expert_names":v6.EXPERT_NAMES,"exact_catboost_fallback_allowed":False,
        "adaptive_schedule":"reheat on winner/drift instability; cool on persistent low-drift winner; 2-cycle exploit tail",
        "role_rotation":"disjoint half-streams swap expert/architecture roles every cycle",
        "full_oof_catboost_retained_trees":trees,"learned_radius":r,
        "mixture_weights":{k:float(z) for k,z in zip(v6.EXPERT_NAMES,w)},"history":h,
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
