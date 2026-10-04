"""500k canonical-HIGGS mechanism screen for the unified TorchBoost engine.

This screen never opens either audit.  Architecture choices may use selection
and ranking only.  The fresh shadow audit is locked in
experiments/higgs_shadow_protocol.json and remains untouched until a mechanism
configuration is fixed.

The variants isolate four questions:
1. Does native histogram-Newton construction improve the backbone?
2. Does releasing hard proposals into differentiable oblique gates help?
3. Do affine residual packets add useful representation capacity?
4. Does a convex global refit of stage coefficients restore missing ensemble
   coordination?

All variants use the same 500k training rows and canonical selection/ranking
slices.  No result from this file may be reported as shadow-audit performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.higgs_canonical_scaling import arrays, fixed_splits, materialize, metrics
from torchboost.adaptive.config import StructureConfig
from torchboost.adaptive.progressive_regularizers import Regularizers
from torchboost.adaptive.unified_progressive import (
    UnifiedConfig,
    UnifiedProgressiveClassifier,
    default_native,
)

NTRAIN=500_000
TREES=48
DEPTH=6
UPDATES=32
BATCH=2048
VARIANTS=(
    "hist_hard",
    "hist_oblique",
    "hist_oblique_rate",
    "affine_oblique",
    "affine_oblique_rate",
)


def config_for(variant: str, seed: int) -> UnifiedConfig:
    if variant not in VARIANTS:
        raise ValueError(variant)
    affine=variant.startswith("affine")
    oblique=variant!="hist_hard"
    native=default_native()
    native.batch_size=BATCH
    native.learning_rate=.01
    native.weight_decay=1e-5
    native.structure=StructureConfig(
        dynamic=False,
        initial_depth=0,
        max_depth=DEPTH,
        max_nodes=255,
        structural_gate=False,
        complexity=0.,
        allocation_regularization=0.,
    )
    return UnifiedConfig(
        n_trees=TREES,
        updates_per_stage=UPDATES,
        depth=DEPTH,
        bins=24,
        min_samples_leaf=20,
        min_child_weight=2.,
        newton_l2=2.,
        split_cost=0.,
        max_delta=4.,
        shrinkage=.30,
        age_decay=.80,
        active_window=6,
        row_subsample=.20,
        feature_subsample=.90,
        cart_strength=7.,
        warm_value_updates=8,
        gate_release="oblique" if oblique else "hard",
        readout="residual",
        linear_values=affine,
        linear_l2=10.,
        auto_complexity=True,
        proposal_mode="hist_newton",
        anchor_min_passes=.5,
        anchor_min_updates=16,
        max_cache_bytes=256*1024*1024,
        checkpoint_every=16,
        regularizers=Regularizers(
            leaf_l2=1e-6,
            hierarchy=2e-4,
            linear_value_l2=1e-5 if affine else 0.,
            feature_l1=1e-7 if affine else 0.,
        ),
        native=native,
        random_state=seed,
    )


@torch.no_grad()
def effective_counts(model, x):
    tx=model.preprocessor_.transform_x(x)
    pieces=[]
    for tree,rate in zip(model.model_.trees,model.model_.stage_rates):
        out=torch.cat([
            tree(batch,hard=getattr(tree,"force_hard",False))[0][:,0]
            for batch in tx.split(2048)
        ])
        pieces.append((rate*out).square().mean().sqrt())
    c=torch.stack(pieces).clamp_min(0.)
    total=c.sum()
    if float(total)<=1e-12:
        return {"participation":0.,"entropy":0.}
    p=c/total
    return {
        "participation":float(total.square()/c.square().sum().clamp_min(1e-12)),
        "entropy":float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum())),
    }


def run(csv_gz,cache,variant,seed=509,out=None):
    torch.set_num_threads(4)
    x_path,y_path,source=materialize(Path(csv_gz),Path(cache))
    x,y=arrays(x_path,y_path)
    splits=fixed_splits(x,y,NTRAIN)
    cfg=config_for(variant,seed+NTRAIN%10007)
    started=time.perf_counter()
    model=UnifiedProgressiveClassifier(cfg).fit(
        splits["train"][0],
        splits["train"][1],
        eval_set=splits["selection"],
    )
    rate_refit=None
    if variant.endswith("_rate"):
        model.refit_stage_rates(
            splits["train"][0],
            splits["train"][1],
            eval_set=splits["selection"],
            l2=1e-5,
            max_iter=30,
        )
        rate_refit=model.rate_refit_
    ranking=metrics(
        splits["ranking"][1],
        model.predict_proba(splits["ranking"][0])[:,1],
    )
    selection=metrics(
        splits["selection"][1],
        model.predict_proba(splits["selection"][0])[:,1],
    )
    result={
        "status":"completed",
        "variant":variant,
        "seed":seed,
        "ntrain":NTRAIN,
        "source":source,
        "audit_opened":False,
        "shadow_audit_opened":False,
        "protocol":"experiments/higgs_shadow_protocol.json",
        "schedule":{
            "requested_trees":TREES,
            "depth":DEPTH,
            "updates_per_stage":UPDATES,
            "batch_size":BATCH,
            "row_subsample":.20,
            "planned_optimizer_presentations":TREES*UPDATES*BATCH,
            "planned_optimizer_passes":TREES*UPDATES*BATCH/NTRAIN,
            "proposal":"histogram Newton",
            "affine_residual_values":variant.startswith("affine"),
            "gate_release":"hard" if variant=="hist_hard" else "oblique",
            "global_stage_rate_refit":variant.endswith("_rate"),
        },
        "retained_trees":int(model.n_estimators_),
        "parameters":int(sum(p.numel() for p in model.model_.parameters())),
        "effective_tree_count":effective_counts(model,splits["ranking"][0]),
        "selection":selection,
        "ranking":ranking,
        "best_selection_nll":float(model.best_score_),
        "rate_refit":rate_refit,
        "seconds":time.perf_counter()-started,
        "history_tail":model.history_[-8:],
    }
    if out:
        Path(out).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv-gz",required=True)
    p.add_argument("--cache",default="/tmp/higgs-cache")
    p.add_argument("--variant",choices=VARIANTS,required=True)
    p.add_argument("--seed",type=int,default=509)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    answer=run(a.csv_gz,a.cache,a.variant,a.seed,a.out)
    print(json.dumps(answer,indent=2,sort_keys=True,allow_nan=False))
