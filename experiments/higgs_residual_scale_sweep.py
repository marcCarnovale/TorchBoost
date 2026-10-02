"""Fixed residual-scale controls around the historical perturbative optimum.

This is a diagnostic companion to differentiable architecture discovery.
It does NOT learn architecture scales and never opens the shadow audit.

The canonical 500k MLP anchor is trained exactly, then every hidden layer is
expanded by one zero-output residual tree level.  Architecture scales are fixed
at several values around the historical sigmoid(-2) = 0.1192029 point.  Each
arm receives the same two full-data epochs as an unchanged MLP continuation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from experiments.higgs_canonical_scaling import (
    LOW_FEATURES,
    arrays,
    fixed_splits,
    materialize,
    metrics,
)
from experiments.higgs_hybrid_benchmark import MLP
from experiments.higgs_differentiable_architecture import (
    BATCH,
    probability,
    train_anchor,
)
from torchboost.adaptive.architecture_corners import CompositionalTreeNetwork

NTRAIN=500_000
SEED=509
EXTRA_EPOCHS=2
SCALES=(0.06, 1.0/(1.0+math.exp(2.0)), 0.24, 0.5)


def train_fixed(model,train_x,train_y,selection_x,selection_y,*,epochs,seed):
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=1e-5)
    loss_fn=torch.nn.BCEWithLogitsLoss()
    rng=torch.Generator().manual_seed(seed)
    best=(float("inf"),None,0)
    history=[]
    started=time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order=torch.randperm(len(train_x),generator=rng)
        for start in range(0,len(order),BATCH):
            idx=order[start:start+BATCH].numpy()
            xb=torch.from_numpy(train_x[idx]);yb=torch.from_numpy(train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(xb),yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,10.)
            opt.step()
        sel=metrics(selection_y,probability(model,selection_x))
        history.append({"epoch":epoch+1,**sel})
        if sel["nll"]<best[0]:
            best=(sel["nll"],{k:v.detach().clone() for k,v in model.state_dict().items()},epoch+1)
    model.load_state_dict(best[1])
    return {"history":history,"best_epoch":best[2],"seconds":time.perf_counter()-started}


def build_fixed(anchor,scale):
    model=deepcopy(anchor)
    for layer in model.layers:
        layer.grow_one_level()
        layer.set_architecture_scale(scale,learnable=False)
    return model


def run(csv_gz,cache,out,seed=SEED):
    torch.set_num_threads(4)
    x_path,y_path,source=materialize(Path(csv_gz),Path(cache))
    x,y=arrays(x_path,y_path)
    splits=fixed_splits(x,y,NTRAIN)
    family_seed=seed+NTRAIN%10007

    scaler=StandardScaler().fit(splits["train"][0])
    train_x=scaler.transform(splits["train"][0]).astype("float32")
    selection_x=scaler.transform(splits["selection"][0]).astype("float32")
    ranking_x=scaler.transform(splits["ranking"][0]).astype("float32")
    train_y=np.asarray(splits["train"][1],dtype="float32")

    torch.manual_seed(family_seed+305)
    reference=MLP(LOW_FEATURES,300,5,.1)
    canonical_rng=torch.get_rng_state()
    anchor=CompositionalTreeNetwork.from_mlp(reference,max_tree_depth=3,seed=family_seed+1200)
    torch.set_rng_state(canonical_rng)
    anchor_training=train_anchor(
        anchor,train_x,train_y,selection_x,splits["selection"][1],family_seed
    )
    anchor_selection=metrics(splits["selection"][1],probability(anchor,selection_x))
    anchor_ranking=metrics(splits["ranking"][1],probability(anchor,ranking_x))

    order_seed=family_seed+19001
    control=deepcopy(anchor)
    control_training=train_fixed(
        control,train_x,train_y,selection_x,splits["selection"][1],
        epochs=EXTRA_EPOCHS,seed=order_seed
    )
    control_selection=metrics(splits["selection"][1],probability(control,selection_x))
    control_ranking=metrics(splits["ranking"][1],probability(control,ranking_x))

    rows={}
    probe=selection_x[:8192]
    for scale in SCALES:
        model=build_fixed(anchor,scale)
        birth=float(np.max(np.abs(probability(model,probe)-probability(anchor,probe))))
        if birth>3e-6:
            raise RuntimeError(f"scale {scale} changed birth function: {birth}")
        trained=train_fixed(
            model,train_x,train_y,selection_x,splits["selection"][1],
            epochs=EXTRA_EPOCHS,seed=order_seed
        )
        sel=metrics(splits["selection"][1],probability(model,selection_x))
        rank=metrics(splits["ranking"][1],probability(model,ranking_x))
        rows[f"{scale:.9f}"]={
            "scale":scale,
            "birth_max_probability_diff":birth,
            "selection":sel,
            "ranking":rank,
            "delta_selection_nll_vs_control":sel["nll"]-control_selection["nll"],
            "delta_ranking_nll_vs_control":rank["nll"]-control_ranking["nll"],
            "delta_ranking_auc_vs_control":rank["auc"]-control_ranking["auc"],
            **trained,
        }
        print(json.dumps({"scale":scale,**rows[f"{scale:.9f}"]}),flush=True)

    result={
        "status":"completed",
        "source":source,
        "seed":seed,
        "family_seed":family_seed,
        "ntrain":NTRAIN,
        "audit_opened":False,
        "shadow_audit_opened":False,
        "protocol":"experiments/higgs_shadow_protocol.json",
        "anchor":{"selection":anchor_selection,"ranking":anchor_ranking,**anchor_training},
        "control":{"selection":control_selection,"ranking":control_ranking,**control_training},
        "fixed_scales":rows,
    }
    Path(out).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv-gz",required=True)
    p.add_argument("--cache",default="/tmp/higgs-cache")
    p.add_argument("--out",required=True)
    p.add_argument("--seed",type=int,default=SEED)
    a=p.parse_args()
    answer=run(a.csv_gz,a.cache,a.out,a.seed)
    print(json.dumps(answer,indent=2,sort_keys=True,allow_nan=False))
