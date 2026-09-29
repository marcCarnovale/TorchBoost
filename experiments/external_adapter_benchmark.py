"""External numerical-classification benchmark for the residual-adapter mechanism.

This study is separate from the frozen HIGGS shadow protocol. It asks whether
the mechanism discovered on HIGGS transfers across small/medium public tabular
datasets without per-dataset architecture tuning.

For each dataset and seed:
- stratified 60/20/20 train/selection/ranking split;
- size-based MLP capacity fixed before results;
- MLP selected by selection NLL;
- frozen-backbone residual adapter at fixed scale sigmoid(-2);
- identical adapter with five held-out learned residual scales;
- CatBoost reference trained on train and selected by its ordinary fit;
- report ranking NLL/AUC for all four models.

No dataset-specific hyperparameter search is performed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from experiments.higgs_hybrid_benchmark import MLP
from torchboost.adaptive.architecture_corners import CompositionalTreeNetwork
from torchboost.adaptive.architecture_regularization import architecture_state

DATASETS={
    "phoneme":44127,
    "bioresponse":45019,
    "bank-marketing":44126,
    "magic-telescope":44125,
    "default-credit":45020,
    "electricity":44120,
    "miniboone":44128,
}
INITIAL_SCALE=1.0/(1.0+math.exp(2.0))


def metrics(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    return {"nll":float(log_loss(y,p,labels=[0,1])),"auc":float(roc_auc_score(y,p))}


def capacity(n):
    if n<10_000:
        return 128,3
    if n<50_000:
        return 192,4
    return 256,4


def load_dataset(name):
    did=DATASETS[name]
    bunch=fetch_openml(data_id=did,as_frame=True,parser="auto")
    x=bunch.data.copy()
    # Curated numerical benchmark IDs should already be numeric; coercion plus
    # median imputation makes missing-value handling explicit and reproducible.
    x=x.apply(pd.to_numeric,errors="coerce")
    y=LabelEncoder().fit_transform(np.asarray(bunch.target).astype(str))
    if len(np.unique(y))!=2:
        raise ValueError(f"{name} is not binary after loading")
    imp=SimpleImputer(strategy="median")
    x=imp.fit_transform(x).astype("float32")
    return x,y.astype("float32"),did


def splits(x,y,seed):
    tx,rx,ty,ry=train_test_split(
        x,y,test_size=.40,random_state=seed,stratify=y
    )
    sx,qx,sy,qy=train_test_split(
        rx,ry,test_size=.50,random_state=seed+1,stratify=ry
    )
    return (tx,ty),(sx,sy),(qx,qy)


@torch.no_grad()
def probability(model,x,batch=4096):
    model.eval();rows=[]
    for start in range(0,len(x),batch):
        rows.append(torch.sigmoid(model(torch.from_numpy(x[start:start+batch]))).numpy())
    return np.concatenate(rows)


def train_anchor(model,train_x,train_y,sel_x,sel_y,*,epochs,batch,seed):
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=1e-5)
    loss_fn=torch.nn.BCEWithLogitsLoss()
    rng=torch.Generator().manual_seed(seed)
    best=(float("inf"),None,0)
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(train_x),generator=rng)
        for start in range(0,len(order),batch):
            idx=order[start:start+batch].numpy()
            xb=torch.from_numpy(train_x[idx]);yb=torch.from_numpy(train_y[idx])
            opt.zero_grad(set_to_none=True);loss=loss_fn(model(xb),yb)
            loss.backward();torch.nn.utils.clip_grad_norm_(params,10.);opt.step()
        score=metrics(sel_y,probability(model,sel_x))["nll"]
        if score<best[0]:
            best=(score,{k:v.detach().clone() for k,v in model.state_dict().items()},epoch+1)
    model.load_state_dict(best[1])
    return best[2]


def build_adapter(anchor,learn_scales):
    model=deepcopy(anchor)
    for p in model.parameters(): p.requires_grad_(False)
    for layer in model.layers:
        layer.grow_one_level()
        layer.set_architecture_scale(INITIAL_SCALE,learnable=learn_scales)
        tree=layer.forest.trees[0];root=layer.root
        root.value.requires_grad_(False);root.linear_value.requires_grad_(False)
        if root.routing_weight is not None: root.routing_weight.requires_grad_(True)
        if root.routing_bias is not None: root.routing_bias.requires_grad_(True)
        for cid in root.children_ids:
            child=tree.get(cid)
            child.value.requires_grad_(True)
            if child.linear_value is not None: child.linear_value.requires_grad_(True)
            child.allocation_logit.requires_grad_(False)
    return model


def partition(model):
    residual=[];scales=[]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        (scales if name.endswith("architecture_logit") else residual).append(p)
    return residual,scales


def train_adapter(model,train_x,train_y,sel_x,sel_y,*,epochs,warmup,batch,seed,learn_scales):
    residual,scales=partition(model)
    ropt=torch.optim.AdamW(residual,lr=1e-3,weight_decay=1e-5)
    sopt=torch.optim.Adam(scales,lr=1e-2) if learn_scales else None
    loss_fn=torch.nn.BCEWithLogitsLoss()
    trng=torch.Generator().manual_seed(seed);srng=torch.Generator().manual_seed(seed+97)
    best=(float("inf"),None,0,None);scale_updates=0
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(train_x),generator=trng)
        sorder=torch.randperm(len(sel_x),generator=srng);cursor=0
        for bi,start in enumerate(range(0,len(order),batch)):
            idx=order[start:start+batch].numpy()
            xb=torch.from_numpy(train_x[idx]);yb=torch.from_numpy(train_y[idx])
            ropt.zero_grad(set_to_none=True)
            if sopt is not None: sopt.zero_grad(set_to_none=True)
            loss=loss_fn(model(xb),yb);loss.backward()
            torch.nn.utils.clip_grad_norm_(residual,10.);ropt.step()
            if learn_scales and epoch>=warmup and (bi+1)%2==0:
                if cursor+batch>len(sorder):
                    sorder=torch.randperm(len(sel_x),generator=srng);cursor=0
                si=sorder[cursor:cursor+batch].numpy();cursor+=batch
                sx=torch.from_numpy(sel_x[si]);sy=torch.from_numpy(sel_y[si])
                ropt.zero_grad(set_to_none=True);sopt.zero_grad(set_to_none=True)
                sloss=loss_fn(model(sx),sy);sloss.backward()
                torch.nn.utils.clip_grad_norm_(scales,2.);sopt.step();scale_updates+=1
        score=metrics(sel_y,probability(model,sel_x))["nll"]
        if score<best[0]:
            best=(score,{k:v.detach().clone() for k,v in model.state_dict().items()},epoch+1,architecture_state(model))
    model.load_state_dict(best[1])
    return best[2],best[3],scale_updates


def run(name,seed,out):
    torch.set_num_threads(4);started=time.perf_counter()
    x,y,did=load_dataset(name);(tx,ty),(sx,sy),(qx,qy)=splits(x,y,seed)
    scaler=StandardScaler().fit(tx)
    tx=scaler.transform(tx).astype("float32");sx=scaler.transform(sx).astype("float32");qx=scaler.transform(qx).astype("float32")
    width,depth=capacity(len(x));batch=min(256,max(32,len(tx)//8))
    anchor_epochs=40;adapter_epochs=12;warmup=6

    torch.manual_seed(seed+305)
    reference=MLP(tx.shape[1],width,depth,.1)
    canonical_rng=torch.get_rng_state()
    anchor=CompositionalTreeNetwork.from_mlp(reference,max_tree_depth=2,seed=seed+1200)
    torch.set_rng_state(canonical_rng)
    anchor_best=train_anchor(anchor,tx,ty,sx,sy,epochs=anchor_epochs,batch=batch,seed=seed+9001)

    fixed=build_adapter(anchor,False);learned=build_adapter(anchor,True)
    fixed_best,fixed_arch,_=train_adapter(fixed,tx,ty,sx,sy,epochs=adapter_epochs,warmup=0,batch=batch,seed=seed+19001,learn_scales=False)
    learned_best,learned_arch,scale_updates=train_adapter(learned,tx,ty,sx,sy,epochs=adapter_epochs,warmup=warmup,batch=batch,seed=seed+19001,learn_scales=True)

    cat=CatBoostClassifier(iterations=800,depth=8,learning_rate=.05,l2_leaf_reg=10,loss_function="Logloss",verbose=False,random_seed=seed,thread_count=4)
    cat.fit(tx,ty,eval_set=(sx,sy),early_stopping_rounds=80,verbose=False)

    anchor_m=metrics(qy,probability(anchor,qx))
    fixed_m=metrics(qy,probability(fixed,qx))
    learned_m=metrics(qy,probability(learned,qx))
    cat_m=metrics(qy,cat.predict_proba(qx)[:,1])
    result={
        "dataset":name,"openml_id":did,"seed":seed,"rows":len(x),"features":x.shape[1],
        "split_rows":{"train":len(tx),"selection":len(sx),"ranking":len(qx)},
        "mlp":{"width":width,"depth":depth,"best_epoch":anchor_best,"ranking":anchor_m},
        "fixed_adapter":{"best_epoch":fixed_best,"ranking":fixed_m},
        "learned_adapter":{"best_epoch":learned_best,"ranking":learned_m,"architecture":learned_arch,"scale_updates":scale_updates},
        "catboost":{"trees":cat.tree_count_,"ranking":cat_m},
        "deltas":{
            "learned_minus_mlp_nll":learned_m["nll"]-anchor_m["nll"],
            "learned_minus_mlp_auc":learned_m["auc"]-anchor_m["auc"],
            "learned_minus_fixed_nll":learned_m["nll"]-fixed_m["nll"],
            "learned_minus_fixed_auc":learned_m["auc"]-fixed_m["auc"],
            "learned_minus_catboost_nll":learned_m["nll"]-cat_m["nll"],
            "learned_minus_catboost_auc":learned_m["auc"]-cat_m["auc"],
        },
        "seconds":time.perf_counter()-started,
    }
    Path(out).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",choices=sorted(DATASETS),required=True)
    p.add_argument("--seed",type=int,required=True);p.add_argument("--out",required=True)
    a=p.parse_args();run(a.dataset,a.seed,a.out)
