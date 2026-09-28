"""Calibrate TorchBoost's exact MLP and CatBoost HIGGS endpoints.

No audit split is opened here.  Both jobs use the canonical 500k train,
selection and ranking slices.  The MLP job trains the ordinary canonical MLP
and its depth-zero compositional TorchBoost embedding in lockstep with identical
initial coefficients, minibatches and dropout RNG states.  The CatBoost job
fits the frozen canonical CatBoost configuration, imports its JSON into
ObliviousSoftForest, and verifies raw-logit and ranking-metric equivalence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

from experiments.higgs_canonical_scaling import (
    LOW_FEATURES,
    arrays,
    fixed_splits,
    materialize,
    metrics,
)
from experiments.higgs_hybrid_benchmark import MLP
from torchboost.adaptive.architecture_corners import (
    CompositionalTreeNetwork,
    ObliviousSoftForest,
)

NTRAIN = 500_000
SEED = 509
MLP_EPOCHS = 20
MLP_BATCH = 4096


@torch.no_grad()
def _probability(model, x, batch=8192):
    model.eval()
    out=[]
    for start in range(0,len(x),batch):
        out.append(torch.sigmoid(model(torch.from_numpy(x[start:start+batch]))).numpy())
    return np.concatenate(out)


def calibrate_mlp(splits, seed):
    scaler=StandardScaler().fit(splits["train"][0])
    train_x=scaler.transform(splits["train"][0]).astype("float32")
    selection_x=scaler.transform(splits["selection"][0]).astype("float32")
    ranking_x=scaler.transform(splits["ranking"][0]).astype("float32")
    train_y=np.asarray(splits["train"][1],dtype="float32")

    torch.manual_seed(seed+305)
    reference=MLP(LOW_FEATURES,300,5,.1)
    # The frozen canonical MLP begins training from the RNG state immediately
    # after its own construction. Building the TorchBoost embedding allocates
    # modules and consumes RNG, so preserve/restore that state or calibration
    # would use a different dropout trajectory despite identical weights.
    canonical_training_rng=torch.get_rng_state()
    corner=CompositionalTreeNetwork.from_mlp(reference,max_tree_depth=3,seed=seed+1200)
    torch.set_rng_state(canonical_training_rng)
    ref_opt=torch.optim.AdamW(reference.parameters(),lr=1e-3,weight_decay=1e-5)
    corner_opt=torch.optim.AdamW(
        [p for p in corner.parameters() if p.requires_grad],lr=1e-3,weight_decay=1e-5
    )
    loss_fn=torch.nn.BCEWithLogitsLoss()
    generator=torch.Generator().manual_seed(seed+9001)

    reference.eval();corner.eval()
    probe=torch.from_numpy(train_x[:4096])
    with torch.no_grad():
        initial_diff=float((reference(probe)-corner(probe)).abs().max())
    if initial_diff>2e-6:
        raise RuntimeError(f"MLP endpoint initial mismatch: {initial_diff}")

    best_ref=(float("inf"),None,0)
    best_corner=(float("inf"),None,0)
    history=[]
    started=time.perf_counter()
    for epoch in range(MLP_EPOCHS):
        reference.train();corner.train()
        order=torch.randperm(len(train_x),generator=generator)
        max_batch_logit_diff=0.
        max_batch_loss_diff=0.
        for start in range(0,len(order),MLP_BATCH):
            idx=order[start:start+MLP_BATCH].numpy()
            xb=torch.from_numpy(train_x[idx]);yb=torch.from_numpy(train_y[idx])
            ref_opt.zero_grad(set_to_none=True);corner_opt.zero_grad(set_to_none=True)

            rng=torch.get_rng_state()
            zr=reference(xb)
            torch.set_rng_state(rng)
            zc=corner(xb)
            max_batch_logit_diff=max(
                max_batch_logit_diff,float((zr.detach()-zc.detach()).abs().max())
            )
            lr=loss_fn(zr,yb);lc=loss_fn(zc,yb)
            max_batch_loss_diff=max(max_batch_loss_diff,abs(float(lr.detach()-lc.detach())))
            lr.backward();lc.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(),10.)
            torch.nn.utils.clip_grad_norm_(
                [p for p in corner.parameters() if p.requires_grad],10.
            )
            ref_opt.step();corner_opt.step()

        pr=_probability(reference,selection_x)
        pc=_probability(corner,selection_x)
        mr=metrics(splits["selection"][1],pr)
        mc=metrics(splits["selection"][1],pc)
        prediction_diff=float(np.max(np.abs(pr-pc)))
        history.append({
            "epoch":epoch+1,
            "reference_selection":mr,
            "corner_selection":mc,
            "max_selection_probability_diff":prediction_diff,
            "max_batch_logit_diff":max_batch_logit_diff,
            "max_batch_loss_diff":max_batch_loss_diff,
        })
        print(json.dumps(history[-1]),flush=True)
        if mr["nll"]<best_ref[0]:
            best_ref=(mr["nll"],{k:v.detach().clone() for k,v in reference.state_dict().items()},epoch+1)
        if mc["nll"]<best_corner[0]:
            best_corner=(mc["nll"],{k:v.detach().clone() for k,v in corner.state_dict().items()},epoch+1)

    reference.load_state_dict(best_ref[1]);corner.load_state_dict(best_corner[1])
    rr=_probability(reference,ranking_x);rc=_probability(corner,ranking_x)
    return {
        "initial_max_logit_diff":initial_diff,
        "reference_best_epoch":best_ref[2],
        "corner_best_epoch":best_corner[2],
        "reference_ranking":metrics(splits["ranking"][1],rr),
        "corner_ranking":metrics(splits["ranking"][1],rc),
        "max_ranking_probability_diff":float(np.max(np.abs(rr-rc))),
        "reference_trainable_parameters":sum(p.numel() for p in reference.parameters() if p.requires_grad),
        "corner_trainable_parameters":sum(p.numel() for p in corner.parameters() if p.requires_grad),
        "history":history,
        "seconds":time.perf_counter()-started,
    }


def calibrate_catboost(splits, seed, model_path):
    started=time.perf_counter()
    model=CatBoostClassifier(
        iterations=1536,depth=10,learning_rate=.05,l2_leaf_reg=20,
        loss_function="Logloss",eval_metric="Logloss",verbose=False,
        random_seed=seed,thread_count=4,od_type="Iter",od_wait=100,
        use_best_model=True,allow_writing_files=False,
    ).fit(splits["train"][0],splits["train"][1],eval_set=splits["selection"])
    model.save_model(model_path,format="json")
    payload=json.loads(Path(model_path).read_text())
    corner=ObliviousSoftForest.from_catboost_json(payload).eval()

    x=np.asarray(splits["ranking"][0],dtype="float32")
    expected=np.asarray(model.predict(x,prediction_type="RawFormulaVal"),dtype=float)
    chunks=[]
    with torch.no_grad():
        for start in range(0,len(x),8192):
            chunks.append(corner(torch.from_numpy(x[start:start+8192]))[:,0].numpy())
    actual=np.concatenate(chunks)
    pe=1/(1+np.exp(-np.clip(expected,-40,40)))
    pa=1/(1+np.exp(-np.clip(actual,-40,40)))
    max_logit=float(np.max(np.abs(expected-actual)))
    max_prob=float(np.max(np.abs(pe-pa)))
    if max_logit>5e-5:
        raise RuntimeError(f"CatBoost endpoint import mismatch: {max_logit}")
    return {
        "retained_trees":int(model.tree_count_),
        "reference_ranking":metrics(splits["ranking"][1],pe),
        "corner_ranking":metrics(splits["ranking"][1],pa),
        "max_ranking_logit_diff":max_logit,
        "max_ranking_probability_diff":max_prob,
        "json_model":str(model_path),
        "seconds":time.perf_counter()-started,
    }


def run(csv_gz,cache,endpoint,out,model_out=None,seed=SEED):
    torch.set_num_threads(4)
    x_path,y_path,source=materialize(Path(csv_gz),Path(cache))
    x,y=arrays(x_path,y_path)
    splits=fixed_splits(x,y,NTRAIN)
    family_seed=seed+NTRAIN%10007
    result={
        "status":"running","endpoint":endpoint,"ntrain":NTRAIN,"seed":seed,
        "source":source,"audit_opened":False,"shadow_audit_opened":False,
        "protocol":"experiments/higgs_shadow_protocol.json",
    }
    Path(out).write_text(json.dumps(result,indent=2,sort_keys=True))
    if endpoint=="mlp":
        result["calibration"]=calibrate_mlp(splits,family_seed)
    elif endpoint=="catboost":
        if not model_out: raise ValueError("--model-out required for catboost")
        result["calibration"]=calibrate_catboost(splits,family_seed,Path(model_out))
    else:
        raise ValueError(endpoint)
    result["status"]="completed"
    Path(out).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv-gz",required=True)
    p.add_argument("--cache",default="/tmp/higgs-cache")
    p.add_argument("--endpoint",choices=("mlp","catboost"),required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--model-out")
    p.add_argument("--seed",type=int,default=SEED)
    a=p.parse_args()
    answer=run(a.csv_gz,a.cache,a.endpoint,a.out,a.model_out,a.seed)
    print(json.dumps(answer,indent=2,sort_keys=True,allow_nan=False))
