"""Baseline-preserving universal tree deformation.

A generic hard CART proposal is the inherited tree computation.  One
differentiable soft-tree residual is initialized to contribute exactly zero, so
training begins at the tree solution rather than merely near it.  The residual
may learn affine, oblique and low-rank interaction corrections; its structure
pays the same complexity rent in every regime.

The selection checkpoint includes epoch 0.  Consequently a deformation that
cannot beat the inherited tree on held-out selection data is rejected rather
than being allowed to damage the tree corner.

Synthetic mechanism test only.  This is a prerequisite for, not evidence of,
real-data competitiveness.
"""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import experiments.fast_semantic_mechanism_screen as base
from experiments.universal_soft_tree_deformation import UniversalSoftTree

REGIMES=("axis","oblique","ridge","interaction","regional_mix")
N_TRAIN=4000
N_TEST=12000
FIT_FRAC=.75
PASSES=48
BATCH=256
LR=1.5e-3
BASE_DEPTH=6
RESIDUAL_DEPTH=4
RANK=4

class PreservingDeformation(nn.Module):
    def __init__(self,p):
        super().__init__()
        self.residual=UniversalSoftTree(p,max_depth=RESIDUAL_DEPTH,rank=RANK)
        # The residual must be exactly zero at birth while retaining useful
        # first derivatives for affine and interaction coordinates.
        with torch.no_grad():
            self.residual.value_bias.zero_()
            self.residual.affine.zero_()
            self.residual.igain.zero_()
            self.residual.branch_logit.fill_(-.75)
        self.base_delta=nn.Parameter(torch.zeros(()))

    def forward(self,x,base_logit):
        return (1.+self.base_delta)*base_logit+self.residual(x)

    def complexity(self,progress):
        reg,terms=self.residual.complexity(progress)
        scale=2e-3*self.base_delta.square()
        terms={**terms,"base_scale":scale}
        return reg+scale,terms

@torch.no_grad()
def logit_from_proba(p):
    p=np.clip(np.asarray(p,dtype=np.float32),1e-5,1-1e-5)
    return np.log(p)-np.log1p(-p)

def train(model,fx,fy,fb,sx,sy,sb,seed):
    torch.manual_seed(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-6)
    xt=torch.from_numpy(fx);yt=torch.from_numpy(fy);bt=torch.from_numpy(fb)
    sxt=torch.from_numpy(sx);syt=torch.from_numpy(sy);sbt=torch.from_numpy(sb)
    gen=torch.Generator().manual_seed(seed+19)

    # Epoch zero is a real candidate: the exact inherited hard-tree predictor.
    with torch.no_grad():
        baseline_sel=float(nn.functional.binary_cross_entropy_with_logits(sbt,syt))
    best=(baseline_sel,{k:v.detach().clone() for k,v in model.state_dict().items()},0)
    history=[{"epoch":0,"selection_nll":baseline_sel,"structure":model.residual.structure_summary()}]

    for epoch in range(PASSES):
        model.train();order=torch.randperm(len(fx),generator=gen);acc={}
        for start in range(0,len(order),BATCH):
            idx=order[start:start+BATCH]
            progress=(epoch+start/max(1,len(order)))/PASSES
            opt.zero_grad(set_to_none=True)
            pred=model(xt[idx],bt[idx])
            loss0=nn.functional.binary_cross_entropy_with_logits(pred,yt[idx])
            reg,terms=model.complexity(progress)
            loss=loss0+reg;loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
            for k,v in terms.items():acc[k]=acc.get(k,0.)+float(v.detach())
        model.eval()
        with torch.no_grad():
            sel=float(nn.functional.binary_cross_entropy_with_logits(model(sxt,sbt),syt))
        if sel<best[0]:
            best=(sel,{k:v.detach().clone() for k,v in model.state_dict().items()},epoch+1)
        if epoch in (0,3,7,15,23,31,39,47):
            history.append({"epoch":epoch+1,"selection_nll":sel,
                "regularization_terms":{k:v/max(1,math.ceil(len(fx)/BATCH)) for k,v in acc.items()},
                "base_scale":float((1+model.base_delta).detach()),
                "structure":model.residual.structure_summary()})
    model.load_state_dict(best[1])
    return model,best[0],best[2],baseline_sel,history

@torch.no_grad()
def predict(model,x,b):
    out=[];model.eval()
    for start in range(0,len(x),2048):
        out.append(torch.sigmoid(model(torch.from_numpy(x[start:start+2048]),
                                      torch.from_numpy(b[start:start+2048]))).numpy())
    return np.concatenate(out)

def run(regime,seed,out):
    torch.set_num_threads(4);started=time.perf_counter()
    problem=base.latent(regime,seed+41)
    x,y,_=base.sample(problem,N_TRAIN,seed+1001)
    qx,qy,bayes_p=base.sample(problem,N_TEST,seed+500001)
    fit_idx,sel_idx=train_test_split(np.arange(len(y)),train_size=FIT_FRAC,stratify=y,random_state=seed+77)
    sc=StandardScaler().fit(x[fit_idx]);x=sc.transform(x).astype("float32");qx=sc.transform(qx).astype("float32")
    fx,fy=x[fit_idx],y[fit_idx];sx,sy=x[sel_idx],y[sel_idx]

    cart=DecisionTreeClassifier(max_depth=BASE_DEPTH,min_samples_leaf=20,random_state=seed+2000).fit(fx,fy)
    fb=logit_from_proba(cart.predict_proba(fx)[:,1]).astype("float32")
    sb=logit_from_proba(cart.predict_proba(sx)[:,1]).astype("float32")
    qb=logit_from_proba(cart.predict_proba(qx)[:,1]).astype("float32")
    cart_m=base.metrics(qy,1/(1+np.exp(-qb)))

    torch.manual_seed(seed+2000)
    model=PreservingDeformation(base.P)
    model,best_sel,best_epoch,base_sel,hist=train(model,fx,fy,fb,sx,sy,sb,seed+3000)
    pm=base.metrics(qy,predict(model,qx,qb))

    cat=CatBoostClassifier(iterations=300,depth=7,learning_rate=.06,l2_leaf_reg=8,
        loss_function="Logloss",verbose=False,random_seed=seed,thread_count=4,allow_writing_files=False)
    cat.fit(fx,fy);cm=base.metrics(qy,cat.predict_proba(qx)[:,1])
    bayes=base.metrics(qy,bayes_p);gap=cm["nll"]-bayes["nll"]

    answer={"study":"baseline_preserving_universal_deformation_v6","regime":regime,"seed":seed,
      "same_policy_across_regimes":True,"fit_rows":len(fit_idx),"selection_rows":len(sel_idx),"test_rows":len(qy),
      "base_tree":{"depth":BASE_DEPTH,"nodes":int(cart.tree_.node_count),"leaves":int(cart.tree_.n_leaves),
                   "selection_nll":base_sel,"ranking":cart_m},
      "deformation":{"residual_depth":RESIDUAL_DEPTH,"parameters":int(sum(p.numel() for p in model.parameters())),
        "best_epoch":best_epoch,"best_selection_nll":best_sel,"ranking":pm,
        "delta_nll_vs_base_tree":pm["nll"]-cart_m["nll"],"delta_auc_vs_base_tree":pm["auc"]-cart_m["auc"],
        "delta_nll_vs_catboost":pm["nll"]-cm["nll"],"delta_auc_vs_catboost":pm["auc"]-cm["auc"],
        "gap_recovered_vs_catboost":float((cm["nll"]-pm["nll"])/gap) if gap>1e-8 else 0.,
        "base_scale":float((1+model.base_delta).detach()),"structure":model.residual.structure_summary()},
      "catboost":{"ranking":cm,"trees":int(cat.tree_count_)},"bayes":bayes,"history":hist,
      "seconds":time.perf_counter()-started}
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(answer,indent=2,sort_keys=True,allow_nan=False))
    print(json.dumps(answer,indent=2,sort_keys=True,allow_nan=False))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--regime",choices=REGIMES,required=True);p.add_argument("--seed",type=int,default=733);p.add_argument("--out",required=True)
    a=p.parse_args();run(a.regime,a.seed,a.out)
