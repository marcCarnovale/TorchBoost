"""Synthetic calibration for a data-adaptive CatBoost-corner prior.

This study intentionally does NOT touch any external benchmark dataset or the
locked HIGGS shadow audit. It asks whether TorchBoost can begin from a strong
tree anchor and release residual capacity only when cross-fitted evidence says
that the extra mechanism generalizes.

Protocol:
- generate synthetic binary data with a fresh ranking sample;
- compute cheap train-only regime statistics;
- map them to an explicit residual-capacity prior;
- form 3-fold out-of-fold CatBoost logits for every training row;
- learn a differentiable TorchBoost residual and one global release gate from
  OOF anchor logits, so no permanent selection split is consumed;
- fit the CatBoost anchor on all training rows;
- evaluate CatBoost and CatBoost+TorchBoost on the untouched synthetic ranking
  sample.

The explicit prior is deliberately simple and auditable. The experiment is a
calibration study, not yet a claim that the hand-written mapping is optimal.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from experiments.higgs_hybrid_benchmark import MLP
from torchboost.adaptive.architecture_corners import CompositionalTreeNetwork


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return np.log(p)-np.log1p(-p)


def metrics(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    return {"nll":float(log_loss(y,p,labels=[0,1])),"auc":float(roc_auc_score(y,p))}


def generate(regime,n,seed,p=24):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,p)).astype("float32")
    if regime=="axis_sparse":
        z=(1.8*(x[:,0]>.25)-1.5*(x[:,1]<-.35)+1.3*((x[:,2]>0)&(x[:,3]>.1))
           +.7*np.tanh(2*x[:,4])-.25)
    elif regime=="oblique_dense":
        w=rng.normal(size=p);w/=np.linalg.norm(w)
        v=rng.normal(size=p);v/=np.linalg.norm(v)
        z=2.2*(x@w)+1.15*np.sin(1.4*(x@v))+.45*(x[:,0]*x[:,1])-.15
    elif regime=="mixed":
        w=rng.normal(size=p);w/=np.linalg.norm(w)
        z=(1.25*(x[:,0]>.2)-1.05*(x[:,1]<-.4)+1.45*(x@w)
           +.65*np.tanh(x[:,2]*x[:,3])-.2)
    else:
        raise ValueError(regime)
    # Keep Bayes error nontrivial and identical in spirit across regimes.
    z=z+rng.normal(scale=.7,size=n)
    prob=_sigmoid(z)
    y=rng.binomial(1,prob).astype("float32")
    return x,y


def regime_stats(x,y):
    n,p=x.shape
    yc=y-y.mean()
    corrs=[]
    for j in range(p):
        xc=x[:,j]-x[:,j].mean()
        den=np.sqrt(np.sum(xc*xc)*np.sum(yc*yc))
        corrs.append(0.0 if den==0 else abs(float(np.sum(xc*yc)/den)))
    corrs=np.sort(np.asarray(corrs))[::-1]
    total=float(corrs.sum()+1e-12)
    top4=float(corrs[:min(4,p)].sum()/total)
    # Fraction of entries near zero is useful for genuinely sparse design
    # matrices while remaining harmless for dense Gaussian synthetic data.
    value_sparsity=float(np.mean(np.abs(x)<1e-8))
    imbalance=float(abs(y.mean()-.5)*2)
    return {
        "n":int(n),"p":int(p),"log10_n":float(np.log10(max(n,1))),
        "top4_marginal_signal_share":top4,
        "value_sparsity":value_sparsity,
        "class_imbalance":imbalance,
    }


def residual_prior(stats):
    """Return prior mean gate in [0.03, 0.70].

    Larger n earns more exploratory residual capacity. Concentrated marginal
    signal, sparse values, and imbalance pull the model toward the tree anchor.
    This mapping is fixed before seeing experiment results.
    """
    n=stats["n"]
    size=(math.log10(max(n,300))-math.log10(1200))/(math.log10(50000)-math.log10(1200))
    size=float(np.clip(size,0,1))
    tree_evidence=(.55*stats["top4_marginal_signal_share"]
                   +.25*stats["value_sparsity"]
                   +.20*stats["class_imbalance"])
    mean=.08+.48*size-.28*tree_evidence
    return float(np.clip(mean,.03,.70))


def build_residual(p,seed):
    torch.manual_seed(seed)
    base=MLP(p,96,3,.05)
    model=CompositionalTreeNetwork.from_mlp(base,max_tree_depth=2,seed=seed+17)
    # Expose one level of tree-like residual flexibility immediately, but keep
    # the entire residual contribution behind the learned global release gate.
    for layer in model.layers:
        layer.grow_one_level()
    return model


@torch.no_grad()
def residual_values(model,x,batch=2048):
    model.eval();out=[]
    for s in range(0,len(x),batch):
        out.append(model(torch.from_numpy(x[s:s+batch])).cpu().numpy())
    return np.concatenate(out)


def cat_model(seed):
    return CatBoostClassifier(
        iterations=400,depth=7,learning_rate=.05,l2_leaf_reg=8,
        loss_function="Logloss",verbose=False,random_seed=seed,thread_count=4,
    )


def oof_cat_logits(x,y,seed):
    skf=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed+101)
    out=np.zeros(len(y),dtype="float32")
    retained=[]
    for fold,(tr,va) in enumerate(skf.split(x,y)):
        m=cat_model(seed+1000+fold)
        m.fit(x[tr],y[tr],eval_set=(x[va],y[va]),early_stopping_rounds=50,verbose=False)
        out[va]=_logit(m.predict_proba(x[va])[:,1]).astype("float32")
        retained.append(int(m.tree_count_))
    return out,retained


def train_release(x,y,base_logits,prior_mean,seed,epochs=24,batch=256):
    model=build_residual(x.shape[1],seed)
    gate_logit=torch.nn.Parameter(torch.tensor(math.log(prior_mean/(1-prior_mean)),dtype=torch.float32))
    params=list(model.parameters())+[gate_logit]
    opt=torch.optim.AdamW(params,lr=8e-4,weight_decay=1e-5)
    loss_fn=torch.nn.BCEWithLogitsLoss()
    rng=torch.Generator().manual_seed(seed+7001)
    xt=torch.from_numpy(x);yt=torch.from_numpy(y);bt=torch.from_numpy(base_logits)
    prior_logit=float(math.log(prior_mean/(1-prior_mean)))
    # Stronger shrinkage on tiny n, vanishing gradually as evidence grows.
    prior_strength=float(0.08*math.sqrt(2000/max(len(x),2000)))
    best=(float("inf"),None,None,0)
    for epoch in range(epochs):
        order=torch.randperm(len(x),generator=rng)
        model.train()
        for s in range(0,len(order),batch):
            idx=order[s:s+batch]
            opt.zero_grad(set_to_none=True)
            gate=torch.sigmoid(gate_logit)
            logits=bt[idx]+gate*model(xt[idx])
            empirical=loss_fn(logits,yt[idx])
            penalty=prior_strength*(gate_logit-prior_logit).pow(2)
            loss=empirical+penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,10.)
            opt.step()
        with torch.no_grad():
            gate=float(torch.sigmoid(gate_logit))
            pp=torch.sigmoid(bt+gate*model(xt)).numpy()
            score=metrics(y,pp)["nll"]
        if score<best[0]:
            best=(score,{k:v.detach().clone() for k,v in model.state_dict().items()},
                  float(gate_logit.detach()),epoch+1)
    model.load_state_dict(best[1])
    gate=float(torch.sigmoid(torch.tensor(best[2])))
    return model,gate,best[3],prior_strength,best[0]


def run(regime,n,seed,out):
    torch.set_num_threads(4);started=time.perf_counter()
    x,y=generate(regime,n,seed)
    qx,qy=generate(regime,max(12000,n),seed+500000)
    scaler=StandardScaler().fit(x)
    x=scaler.transform(x).astype("float32");qx=scaler.transform(qx).astype("float32")
    stats=regime_stats(x,y);prior=residual_prior(stats)
    oof_logits,fold_trees=oof_cat_logits(x,y,seed)
    residual,gate,best_epoch,prior_strength,oof_nll=train_release(
        x,y,oof_logits,prior,seed+3000
    )
    cat=cat_model(seed+9000);cat.fit(x,y,verbose=False)
    base_p=cat.predict_proba(qx)[:,1];base_logits=_logit(base_p)
    rv=residual_values(residual,qx)
    hybrid_p=_sigmoid(base_logits+gate*rv)
    base_m=metrics(qy,base_p);hybrid_m=metrics(qy,hybrid_p)
    result={
        "study":"synthetic_cat_corner_progressive_release_v1",
        "regime":regime,"seed":seed,"train_rows":int(n),"ranking_rows":int(len(qy)),
        "features":int(x.shape[1]),"train_only_stats":stats,
        "controller":{
            "prior_residual_gate":prior,
            "learned_residual_gate":gate,
            "gate_delta":gate-prior,
            "prior_strength":prior_strength,
            "best_epoch":best_epoch,
            "oof_training_nll_at_best":oof_nll,
            "oof_catboost_retained_trees":fold_trees,
        },
        "catboost":{"retained_trees":int(cat.tree_count_),"ranking":base_m},
        "hybrid":{"ranking":hybrid_m},
        "deltas":{
            "hybrid_minus_catboost_nll":hybrid_m["nll"]-base_m["nll"],
            "hybrid_minus_catboost_auc":hybrid_m["auc"]-base_m["auc"],
        },
        "seconds":time.perf_counter()-started,
    }
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--regime",choices=["axis_sparse","oblique_dense","mixed"],required=True)
    p.add_argument("--n",type=int,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args();run(a.regime,a.n,a.seed,a.out)
