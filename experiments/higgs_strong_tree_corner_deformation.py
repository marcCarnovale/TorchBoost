"""500k HIGGS: exact strong tree corner + zero-at-birth TorchBoost deformation.

The canonical CatBoost model is fitted on TRAIN, exported, and imported exactly
into ObliviousSoftForest.  The imported forest is the inherited TorchBoost tree
corner.  It is frozen for this first strong-corner experiment.

One UniversalSoftTree residual is initialized to contribute exactly zero.
TRAIN updates only that residual.  SELECTION chooses among epoch 0 (the exact
strong tree corner) and later residual checkpoints.  RANKING is evaluation
only.  Neither the legacy audit nor the fresh shadow audit is opened.

This is not an external expert mixture: the final predictor is an additive
TorchBoost computation consisting of an exact imported tree corner plus one
trainable differentiable tree residual.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

from experiments.higgs_canonical_scaling import arrays,fixed_splits,materialize,metrics
from experiments.universal_soft_tree_deformation import UniversalSoftTree
from torchboost.adaptive.architecture_corners import ObliviousSoftForest

NTRAIN=500_000
SEED=509
EPOCHS=5
BATCH=4096
LR=1e-3
RESIDUAL_DEPTH=4
RANK=4

@torch.no_grad()
def residual_probability(model,x,base_logits,batch=8192):
    model.eval();out=[]
    for start in range(0,len(x),batch):
        xb=torch.from_numpy(x[start:start+batch])
        bb=torch.from_numpy(base_logits[start:start+batch])
        out.append(torch.sigmoid(bb+model(xb)).numpy())
    return np.concatenate(out)

@torch.no_grad()
def imported_logits(model,x,batch=8192):
    model.eval();out=[]
    for start in range(0,len(x),batch):
        out.append(model(torch.from_numpy(np.asarray(x[start:start+batch],dtype="float32")))[:,0].numpy())
    return np.concatenate(out)

def train_residual(model,train_x,train_y,train_base,selection_x,selection_y,selection_base,seed):
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=LR,weight_decay=1e-6)
    loss_fn=nn.BCEWithLogitsLoss()
    xt=torch.from_numpy(train_x);yt=torch.from_numpy(train_y);bt=torch.from_numpy(train_base)
    sxt=torch.from_numpy(selection_x);syt=torch.from_numpy(np.asarray(selection_y,dtype="float32"))
    sbt=torch.from_numpy(selection_base)
    gen=torch.Generator().manual_seed(seed+19001)

    with torch.no_grad():
        base_sel=float(loss_fn(sbt,syt))
    best=(base_sel,{k:v.detach().clone() for k,v in model.state_dict().items()},0,model.structure_summary())
    history=[{"epoch":0,"selection_nll":base_sel,"structure":best[3]}]
    started=time.perf_counter()
    train_examples=0

    for epoch in range(EPOCHS):
        model.train();order=torch.randperm(len(train_x),generator=gen);acc={}
        for start in range(0,len(order),BATCH):
            idx=order[start:start+BATCH]
            progress=(epoch+start/max(1,len(order)))/EPOCHS
            opt.zero_grad(set_to_none=True)
            z=bt[idx]+model(xt[idx])
            pred=loss_fn(z,yt[idx])
            reg,terms=model.complexity(progress)
            loss=pred+reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,10.)
            opt.step();train_examples+=len(idx)
            for k,v in terms.items():acc[k]=acc.get(k,0.)+float(v.detach())

        model.eval()
        with torch.no_grad(): sel=float(loss_fn(sbt+model(sxt),syt))
        state=model.structure_summary()
        row={"epoch":epoch+1,"selection_nll":sel,"structure":state,
             "regularization_terms":{k:v/max(1,len(range(0,len(order),BATCH))) for k,v in acc.items()}}
        history.append(row);print(json.dumps(row),flush=True)
        if sel<best[0]:
            best=(sel,{k:v.detach().clone() for k,v in model.state_dict().items()},epoch+1,state)

    model.load_state_dict(best[1])
    return {"baseline_selection_nll":base_sel,"best_selection_nll":best[0],
            "best_epoch":best[2],"best_structure":best[3],"history":history,
            "train_examples_seen":train_examples,"seconds":time.perf_counter()-started}

def run(csv_gz,cache,out,model_out,seed=SEED):
    torch.set_num_threads(4)
    x_path,y_path,source=materialize(Path(csv_gz),Path(cache))
    x,y=arrays(x_path,y_path);splits=fixed_splits(x,y,NTRAIN)
    family_seed=seed+NTRAIN%10007

    started=time.perf_counter()
    cat=CatBoostClassifier(
        iterations=1536,depth=10,learning_rate=.05,l2_leaf_reg=20,
        loss_function="Logloss",eval_metric="Logloss",verbose=False,
        random_seed=family_seed,thread_count=4,od_type="Iter",od_wait=100,
        use_best_model=True,allow_writing_files=False,
    ).fit(splits["train"][0],splits["train"][1],eval_set=splits["selection"])
    cat.save_model(model_out,format="json")
    corner=ObliviousSoftForest.from_catboost_json(json.loads(Path(model_out).read_text())).eval()
    for p in corner.parameters(): p.requires_grad_(False)

    probe=np.asarray(splits["selection"][0][:8192],dtype="float32")
    reference=np.asarray(cat.predict(probe,prediction_type="RawFormulaVal"),dtype="float32")
    imported=imported_logits(corner,probe)
    endpoint_max_logit_diff=float(np.max(np.abs(reference-imported)))
    if endpoint_max_logit_diff>5e-5:
        raise RuntimeError(f"strong tree corner import mismatch: {endpoint_max_logit_diff}")

    scaler=StandardScaler().fit(splits["train"][0])
    train_x=scaler.transform(splits["train"][0]).astype("float32")
    selection_x=scaler.transform(splits["selection"][0]).astype("float32")
    ranking_x=scaler.transform(splits["ranking"][0]).astype("float32")
    train_y=np.asarray(splits["train"][1],dtype="float32")

    # Cache the exact inherited function. CatBoost raw logits are numerically
    # equivalent to the imported TorchBoost corner, verified above.
    train_base=np.asarray(cat.predict(splits["train"][0],prediction_type="RawFormulaVal"),dtype="float32")
    selection_base=np.asarray(cat.predict(splits["selection"][0],prediction_type="RawFormulaVal"),dtype="float32")
    ranking_base=np.asarray(cat.predict(splits["ranking"][0],prediction_type="RawFormulaVal"),dtype="float32")

    torch.manual_seed(family_seed+22000)
    residual=UniversalSoftTree(train_x.shape[1],max_depth=RESIDUAL_DEPTH,rank=RANK)
    with torch.no_grad():
        residual.value_bias.zero_();residual.affine.zero_();residual.igain.zero_();residual.branch_logit.fill_(-.75)

    training=train_residual(
        residual,train_x,train_y,train_base,
        selection_x,splits["selection"][1],selection_base,family_seed,
    )

    base_selection=metrics(splits["selection"][1],1/(1+np.exp(-np.clip(selection_base,-40,40))))
    base_ranking=metrics(splits["ranking"][1],1/(1+np.exp(-np.clip(ranking_base,-40,40)))
    )
    deformed_selection=metrics(
        splits["selection"][1],residual_probability(residual,selection_x,selection_base)
    )
    deformed_ranking=metrics(
        splits["ranking"][1],residual_probability(residual,ranking_x,ranking_base)
    )

    result={
      "status":"completed","study":"higgs_strong_tree_corner_deformation_v1",
      "source":source,"seed":seed,"family_seed":family_seed,"ntrain":NTRAIN,
      "audit_opened":False,"shadow_audit_opened":False,
      "protocol":"experiments/higgs_shadow_protocol.json",
      "corner":{"family":"canonical_catboost_imported_as_oblivious_soft_forest",
        "retained_trees":int(cat.tree_count_),"endpoint_max_logit_diff":endpoint_max_logit_diff,
        "selection":base_selection,"ranking":base_ranking},
      "deformation":{"residual_depth":RESIDUAL_DEPTH,"interaction_rank":RANK,
        "trainable_parameters":int(sum(p.numel() for p in residual.parameters() if p.requires_grad)),
        "selection":deformed_selection,"ranking":deformed_ranking,
        "delta_selection_nll_vs_corner":deformed_selection["nll"]-base_selection["nll"],
        "delta_ranking_nll_vs_corner":deformed_ranking["nll"]-base_ranking["nll"],
        "delta_ranking_auc_vs_corner":deformed_ranking["auc"]-base_ranking["auc"],
        **training},
      "seconds_total":time.perf_counter()-started,
    }
    Path(out).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--csv-gz",required=True);p.add_argument("--cache",default="/tmp/higgs-cache")
    p.add_argument("--out",required=True);p.add_argument("--model-out",required=True);p.add_argument("--seed",type=int,default=SEED)
    a=p.parse_args();run(a.csv_gz,a.cache,a.out,a.model_out,a.seed)
