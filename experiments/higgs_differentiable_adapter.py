"""Faithful differentiable analogue of the winning frozen-backbone adapter.

The inherited canonical MLP affine packets and output head are frozen.
All hidden layers receive one zero-at-birth residual tree refinement.

TRAIN updates only residual packets and routing.
SELECTION updates only the five positive residual scales.
RANKING is evaluation only. The shadow audit is never opened.

Three matched arms isolate the source of improvement:
- fixed scales at sigmoid(-2);
- train-updated scales with the same trainable parameters and update cadence;
- held-out scales updated only from SELECTION.

This separates the value of held-out architecture allocation from merely
granting the adapter five additional trainable supervised parameters.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from experiments.higgs_canonical_scaling import (
    LOW_FEATURES, arrays, fixed_splits, materialize, metrics,
)
from experiments.higgs_differentiable_architecture import (
    BATCH, probability, train_anchor,
)
from experiments.higgs_hybrid_benchmark import MLP
from torchboost.adaptive.architecture_corners import CompositionalTreeNetwork
from torchboost.adaptive.architecture_regularization import architecture_state
from torchboost.adaptive.residual_adapter import (
    grow_frozen_backbone_adapter,
    partition_adapter_parameters,
)

NTRAIN=500_000
SEED=509
ADAPTER_EPOCHS=4
SCALE_WARMUP_EPOCHS=2
ARCH_EVERY=2
INITIAL_SCALE=1.0/(1.0+math.exp(2.0))


def build_adapter(anchor, *, learn_scales):
    return grow_frozen_backbone_adapter(anchor, learn_scales=learn_scales)


def partition(model):
    return partition_adapter_parameters(model)


def train_adapter(
    model,train_x,train_y,selection_x,selection_y,*,
    epochs,seed,scale_source,warmup_epochs,
    checkpoint_x=None,checkpoint_y=None,
):
    if checkpoint_x is None:
        checkpoint_x=selection_x
    if checkpoint_y is None:
        checkpoint_y=selection_y
    if scale_source not in {"none", "train", "selection"}:
        raise ValueError("scale_source must be none, train, or selection")
    learn_scales = scale_source != "none"
    residual_params,scale_params=partition(model)
    residual_opt=torch.optim.AdamW(residual_params,lr=1e-3,weight_decay=1e-5)
    scale_opt=(
        torch.optim.Adam(scale_params,lr=1e-2,weight_decay=0.)
        if learn_scales else None
    )
    loss_fn=torch.nn.BCEWithLogitsLoss()
    train_rng=torch.Generator().manual_seed(seed)
    selection_rng=torch.Generator().manual_seed(seed+97)
    best=(float("inf"),None,0,None)
    history=[]
    train_examples=0;selection_examples=0;scale_updates=0
    started=time.perf_counter()

    for epoch in range(epochs):
        model.train()
        order=torch.randperm(len(train_x),generator=train_rng)
        sorder=torch.randperm(len(selection_x),generator=selection_rng)
        scursor=0
        for bi,start in enumerate(range(0,len(order),BATCH)):
            idx=order[start:start+BATCH].numpy()
            xb=torch.from_numpy(train_x[idx]);yb=torch.from_numpy(train_y[idx])
            residual_opt.zero_grad(set_to_none=True)
            if scale_opt is not None: scale_opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(xb),yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual_params,10.)
            residual_opt.step()
            train_examples+=len(idx)

            scheduled_scale_update = (
                learn_scales and epoch>=warmup_epochs and (bi+1)%ARCH_EVERY==0
            )
            if scheduled_scale_update and scale_source == "train":
                torch.nn.utils.clip_grad_norm_(scale_params,2.)
                scale_opt.step()
                scale_updates+=1
            elif scheduled_scale_update and scale_source == "selection":
                if scursor+BATCH>len(sorder):
                    sorder=torch.randperm(len(selection_x),generator=selection_rng)
                    scursor=0
                sidx=sorder[scursor:scursor+BATCH].numpy();scursor+=BATCH
                sx=torch.from_numpy(selection_x[sidx])
                sy=torch.from_numpy(np.asarray(selection_y[sidx],dtype="float32"))
                residual_opt.zero_grad(set_to_none=True);scale_opt.zero_grad(set_to_none=True)
                # Held-out architecture arm: only the scale parameters receive
                # gradients from SELECTION. Residual packets remain TRAIN-only.
                sloss=loss_fn(model(sx),sy)
                sloss.backward()
                torch.nn.utils.clip_grad_norm_(scale_params,2.)
                scale_opt.step()
                selection_examples+=len(sidx);scale_updates+=1

        sel=metrics(checkpoint_y,probability(model,checkpoint_x))
        state=architecture_state(model)
        checkpoint_eligible = scale_source == "none" or scale_updates > 0
        row={
            "epoch":epoch+1,
            "selection":sel,
            "architecture":state,
            "scale_updates_so_far":scale_updates,
            "checkpoint_eligible":checkpoint_eligible,
        }
        history.append(row)
        print(json.dumps({"scale_source":scale_source,**row}),flush=True)
        if checkpoint_eligible and sel["nll"]<best[0]:
            best=(
                sel["nll"],
                {k:v.detach().clone() for k,v in model.state_dict().items()},
                epoch+1,state,
            )

    if best[1] is None:
        raise RuntimeError(
            f"{scale_source} arm never reached an eligible checkpoint; "
            "increase epochs or reduce scale warmup"
        )
    model.load_state_dict(best[1])
    return {
        "best_epoch":best[2],"best_architecture":best[3],"history":history,
        "seconds":time.perf_counter()-started,
        "train_examples_seen":train_examples,
        "selection_examples_seen_by_scales":selection_examples,
        "scale_updates":scale_updates,
    }


def run(csv_gz,cache,out,checkpoint_dir,seed=SEED):
    torch.set_num_threads(4)
    x_path,y_path,source=materialize(Path(csv_gz),Path(cache))
    x,y=arrays(x_path,y_path);splits=fixed_splits(x,y,NTRAIN)
    family_seed=seed+NTRAIN%10007
    scaler=StandardScaler().fit(splits["train"][0])
    train_x=scaler.transform(splits["train"][0]).astype("float32")
    selection_x=scaler.transform(splits["selection"][0]).astype("float32")
    selection_y=np.asarray(splits["selection"][1],dtype="float32")
    selection_mid=len(selection_x)//2
    architecture_x=selection_x[:selection_mid]
    architecture_y=selection_y[:selection_mid]
    checkpoint_x=selection_x[selection_mid:]
    checkpoint_y=selection_y[selection_mid:]
    ranking_x=scaler.transform(splits["ranking"][0]).astype("float32")
    train_y=np.asarray(splits["train"][1],dtype="float32")

    torch.manual_seed(family_seed+305)
    reference=MLP(LOW_FEATURES,300,5,.1)
    canonical_rng=torch.get_rng_state()
    anchor=CompositionalTreeNetwork.from_mlp(
        reference,max_tree_depth=3,seed=family_seed+1200
    )
    torch.set_rng_state(canonical_rng)
    anchor_training=train_anchor(
        anchor,train_x,train_y,selection_x,splits["selection"][1],family_seed
    )
    anchor_sel=metrics(splits["selection"][1],probability(anchor,selection_x))
    anchor_rank=metrics(splits["ranking"][1],probability(anchor,ranking_x))

    fixed=build_adapter(anchor,learn_scales=False)
    train_scale=build_adapter(anchor,learn_scales=True)
    learned=build_adapter(anchor,learn_scales=True)
    probe=architecture_x[:8192]
    for name,model in (("fixed",fixed),("train_scale",train_scale),("learned",learned)):
        diff=float(np.max(np.abs(probability(model,probe)-probability(anchor,probe))))
        if diff>3e-6: raise RuntimeError(f"{name} adapter changed birth function: {diff}")

    training_seed=family_seed+19001
    fixed_training=train_adapter(
        fixed,train_x,train_y,architecture_x,architecture_y,
        epochs=ADAPTER_EPOCHS,seed=training_seed,scale_source="none",warmup_epochs=0,
        checkpoint_x=checkpoint_x,checkpoint_y=checkpoint_y,
    )
    train_scale_training=train_adapter(
        train_scale,train_x,train_y,architecture_x,architecture_y,
        epochs=ADAPTER_EPOCHS,seed=training_seed,scale_source="train",
        warmup_epochs=SCALE_WARMUP_EPOCHS,
        checkpoint_x=checkpoint_x,checkpoint_y=checkpoint_y,
    )
    learned_training=train_adapter(
        learned,train_x,train_y,architecture_x,architecture_y,
        epochs=ADAPTER_EPOCHS,seed=training_seed,scale_source="selection",
        warmup_epochs=SCALE_WARMUP_EPOCHS,
        checkpoint_x=checkpoint_x,checkpoint_y=checkpoint_y,
    )

    fixed_sel=metrics(checkpoint_y,probability(fixed,checkpoint_x))
    fixed_rank=metrics(splits["ranking"][1],probability(fixed,ranking_x))
    train_scale_sel=metrics(checkpoint_y,probability(train_scale,checkpoint_x))
    train_scale_rank=metrics(splits["ranking"][1],probability(train_scale,ranking_x))
    learned_sel=metrics(checkpoint_y,probability(learned,checkpoint_x))
    learned_rank=metrics(splits["ranking"][1],probability(learned,ranking_x))

    checkpoint_dir=Path(checkpoint_dir);checkpoint_dir.mkdir(parents=True,exist_ok=True)
    torch.save({"state":fixed.state_dict()},checkpoint_dir/"fixed-adapter.pt")
    torch.save(
        {"state":train_scale.state_dict(),"architecture":architecture_state(train_scale)},
        checkpoint_dir/"train-scale-adapter.pt"
    )
    torch.save(
        {"state":learned.state_dict(),"architecture":architecture_state(learned)},
        checkpoint_dir/"heldout-scale-adapter.pt"
    )
    result={
        "status":"completed","source":source,"seed":seed,"family_seed":family_seed,
        "ntrain":NTRAIN,"audit_opened":False,"shadow_audit_opened":False,
        "protocol":"research/higgs_scale_source_control_protocol.md",
        "shadow_protocol":"experiments/higgs_shadow_protocol.json",
        "selection_partition":{
            "architecture_rows":len(architecture_x),
            "checkpoint_rows":len(checkpoint_x),
            "rule":"first_half_architecture_second_half_checkpoint",
        },
        "initial_scale":INITIAL_SCALE,
        "anchor":{"selection":anchor_sel,"ranking":anchor_rank,**anchor_training},
        "fixed_adapter":{"checkpoint_selection":fixed_sel,"ranking":fixed_rank,**fixed_training},
        "train_scale_adapter":{
            "checkpoint_selection":train_scale_sel,"ranking":train_scale_rank,
            "architecture":architecture_state(train_scale),**train_scale_training
        },
        "learned_scale_adapter":{
            "checkpoint_selection":learned_sel,"ranking":learned_rank,
            "architecture":architecture_state(learned),**learned_training
        },
        "deltas":{
            "heldout_checkpoint_nll_vs_fixed":learned_sel["nll"]-fixed_sel["nll"],
            "heldout_ranking_nll_vs_fixed":learned_rank["nll"]-fixed_rank["nll"],
            "heldout_ranking_auc_vs_fixed":learned_rank["auc"]-fixed_rank["auc"],
            "heldout_ranking_nll_vs_train_scale":learned_rank["nll"]-train_scale_rank["nll"],
            "heldout_ranking_auc_vs_train_scale":learned_rank["auc"]-train_scale_rank["auc"],
            "heldout_ranking_nll_vs_anchor":learned_rank["nll"]-anchor_rank["nll"],
            "heldout_ranking_auc_vs_anchor":learned_rank["auc"]-anchor_rank["auc"],
        },
    }
    Path(out).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv-gz",required=True)
    p.add_argument("--cache",default="/tmp/higgs-cache")
    p.add_argument("--out",required=True)
    p.add_argument("--checkpoint-dir",default="research-results/differentiable-adapter-checkpoints")
    p.add_argument("--seed",type=int,default=SEED)
    a=p.parse_args()
    answer=run(a.csv_gz,a.cache,a.out,a.checkpoint_dir,a.seed)
    print(json.dumps(answer,indent=2,sort_keys=True,allow_nan=False))
