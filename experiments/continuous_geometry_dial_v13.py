"""Continuous geometry dial v13: identifiable held-out architecture weights.

Fixes v12's non-identifiable joint gate. Base experts are trained independently,
then frozen. Their centered logits are calibrated to unit RMS on a disjoint
selection split. The gate is learned only on that held-out split, so branch
rescaling cannot be traded against gate weight.

Experts: CatBoost endpoint, MLP, bilinear specialist.
Final prediction preserves each expert's calibration/intercept while mixing
normalized centered logit deviations with a learned common radius.

Synthetic only; exact Bayes NLL known.
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
import experiments.continuous_geometry_dial as dial
import experiments.fast_specialist_redesign as spec

EPS=1e-8
TRAIN_FRAC=.72
EXPERT_PASSES=28
GATE_STEPS=500

def logit(p):
    p=np.clip(p,1e-6,1-1e-6); return np.log(p)-np.log1p(-p)

def train_nn(model,x,y,seed):
    torch.manual_seed(seed); opt=torch.optim.AdamW(model.parameters(),lr=1.3e-3,weight_decay=1e-5)
    xt=torch.from_numpy(x); yt=torch.from_numpy(y); gen=torch.Generator().manual_seed(seed+19)
    best=(1e9,None)
    for _ in range(EXPERT_PASSES):
        order=torch.randperm(len(x),generator=gen); model.train()
        for s in range(0,len(order),256):
            idx=order[s:s+256]; opt.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(xt[idx]),yt[idx]); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
        with torch.no_grad(): score=float(nn.functional.binary_cross_entropy_with_logits(model(xt),yt))
        if score<best[0]: best=(score,{k:v.detach().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); return model

@torch.no_grad()
def nn_logits(model,x):
    model.eval(); return model(torch.from_numpy(x)).numpy()

def calibrate(sel_logits):
    # Shape [n, experts]. Keep each expert's selection intercept but normalize
    # centered variation, making mixture weights comparable and identifiable.
    center=sel_logits.mean(axis=0)
    centered=sel_logits-center[None,:]
    scale=np.sqrt(np.mean(centered*centered,axis=0))+EPS
    return center,scale

def transformed(raw,center,scale):
    return (raw-center[None,:])/scale[None,:]

def learn_gate(y,sel_logits):
    center,scale=calibrate(sel_logits); z=transformed(sel_logits,center,scale)
    zt=torch.from_numpy(z.astype("float32")); yt=torch.from_numpy(y.astype("float32"))
    centers=torch.from_numpy(center.astype("float32"))
    gate=nn.Parameter(torch.zeros(z.shape[1])); radius_raw=nn.Parameter(torch.tensor(0.0)); bias=nn.Parameter(torch.tensor(0.0))
    opt=torch.optim.Adam([gate,radius_raw,bias],lr=.03)
    history=[]
    for step in range(GATE_STEPS):
        frac=step/(GATE_STEPS-1); temp=max(.15,1.5*(1-frac)+.15*frac)
        w=torch.softmax(gate/temp,dim=0)
        radius=.15+2.85*torch.sigmoid(radius_raw)
        # Weighted endpoint intercept + normalized shape mixture.
        pred=bias+(w*centers).sum()+radius*(zt*w[None,:]).sum(1)
        loss=nn.functional.binary_cross_entropy_with_logits(pred,yt)
        # mild entropy only in first quarter
        if frac<.25:
            ent=-(w*torch.log(w.clamp_min(1e-8))).sum(); loss=loss-.002*(1-4*frac)*ent
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step in (0,99,249,499):
            history.append({"step":step+1,"temperature":temp,"loss":float(loss.detach()),
              "weights":[float(v) for v in w.detach()],"radius":float(radius.detach()),"bias":float(bias.detach())})
    w=torch.softmax(gate/.15,dim=0).detach().numpy()
    radius=float((.15+2.85*torch.sigmoid(radius_raw)).detach()); b=float(bias.detach())
    return center,scale,w,radius,b,history

def mix_predict(raw,center,scale,w,radius,bias):
    z=transformed(raw,center,scale)
    logits=bias+float(np.dot(w,center))+radius*(z@w)
    return dial.sigmoid(logits)

def run(lam,seed,out):
    torch.set_num_threads(4); start=time.perf_counter(); a,b=dial.problem(seed+41)
    x,y,_=dial.sample(a,b,lam,dial.N_TRAIN,seed+1001); qx,qy,bp=dial.sample(a,b,lam,dial.N_TEST,seed+500001)
    fit_idx,sel_idx=train_test_split(np.arange(len(y)),train_size=TRAIN_FRAC,stratify=y,random_state=seed+77)
    sc=StandardScaler().fit(x[fit_idx]); x=sc.transform(x).astype("float32"); qx=sc.transform(qx).astype("float32")
    fx,fy=x[fit_idx],y[fit_idx]; sx,sy=x[sel_idx],y[sel_idx]

    cat=CatBoostClassifier(iterations=300,depth=7,learning_rate=.06,l2_leaf_reg=8,loss_function="Logloss",verbose=False,random_seed=seed,thread_count=4)
    cat.fit(fx,fy)
    mlp=train_nn(dial.AxisMLP(dial.P),fx,fy,seed+2000)
    bil=train_nn(spec.BilinearLowRank(dial.P),fx,fy,seed+3000)

    def raw_logits(z):
        return np.stack([logit(cat.predict_proba(z)[:,1]),nn_logits(mlp,z),nn_logits(bil,z)],axis=1)

    sl=raw_logits(sx); center,scale,w,radius,bias,h=learn_gate(sy,sl)
    qp=mix_predict(raw_logits(qx),center,scale,w,radius,bias)
    # Comparators trained on the same fit subset for a clean architecture test.
    cp=cat.predict_proba(qx)[:,1]; mp=dial.predict(mlp,qx); bpred=dial.predict(bil,qx)
    bayes=dial.metrics(qy,bp); cm=dial.metrics(qy,cp); mm=dial.metrics(qy,mp); bm=dial.metrics(qy,bpred); hm=dial.metrics(qy,qp)
    gap=cm["nll"]-bayes["nll"]
    result={"study":"continuous_geometry_dial_identifiable_gate_v13","lambda_interaction":lam,"seed":seed,
      "fit_rows":int(len(fit_idx)),"selection_rows":int(len(sel_idx)),"test_rows":len(qy),
      "bayes":bayes,"catboost":cm,"mlp":mm,"bilinear":bm,"catboost_to_bayes_nll_gap":gap,
      "mixture":{"ranking":hm,"weights":{"catboost":float(w[0]),"mlp":float(w[1]),"bilinear":float(w[2])},
        "radius":radius,"bias":bias,"expert_logit_centers":[float(v) for v in center],
        "expert_centered_logit_rms":[float(v) for v in scale],"gate_history":h,
        "gap_recovered":float((cm["nll"]-hm["nll"])/gap) if gap>1e-8 else 0.0,
        "delta_nll_vs_catboost":hm["nll"]-cm["nll"],"delta_auc_vs_catboost":hm["auc"]-cm["auc"]},
      "seconds":time.perf_counter()-start}
    Path(out).parent.mkdir(parents=True,exist_ok=True); Path(out).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)); print(json.dumps(result,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--lambda-interaction",type=float,required=True); p.add_argument("--seed",type=int,default=733); p.add_argument("--out",required=True)
    a=p.parse_args(); run(a.lambda_interaction,a.seed,a.out)
