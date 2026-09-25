"""Scientific-law / deep-double-descent development studies.

Learn a central-force acceleration map across many orbital states and test
whether the learned vector field recovers radial direction and inverse-square
scaling. These are post-hoc diagnostics, not a symbolic-law-discovery claim.
"""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from torchboost.adaptive.unified_progressive import UnifiedConfig,UnifiedProgressiveRegressor,default_native
from torchboost.adaptive.config import StructureConfig
from torchboost.adaptive.progressive_regularizers import Regularizers

def central_force_data(n,seed,noise=0.,radius=(.55,2.0)):
    r=np.random.default_rng(seed);rad=np.exp(r.uniform(np.log(radius[0]),np.log(radius[1]),n))
    angle=r.uniform(-np.pi,np.pi,n);mu=r.uniform(.7,1.3,n);x=rad*np.cos(angle);y=rad*np.sin(angle)
    speed=np.sqrt(mu/rad)*r.uniform(.65,1.25,n)
    vx=-speed*np.sin(angle)+r.normal(0,.03,n);vy=speed*np.cos(angle)+r.normal(0,.03,n)
    X=np.c_[x,y,vx,vy,mu].astype("float32");A=np.c_[-mu*x/rad**3,-mu*y/rad**3].astype("float32")
    if noise:A=A+r.normal(0,noise*np.std(A,axis=0),A.shape).astype("float32")
    return X,A

def diagnostics(model,X,true):
    pred=model.predict(X);rmse=float(mean_squared_error(true,pred)**.5);pos=X[:,:2]
    rad=np.linalg.norm(pos,axis=1);mag=np.linalg.norm(pred,axis=1);unit=pred/np.maximum(mag[:,None],1e-12)
    alignment=float(np.mean(np.sum((-pos/rad[:,None])*unit,axis=1)));good=mag>1e-8
    slope=np.polyfit(np.log(rad[good]),np.log(mag[good]/X[good,4]),1)[0]
    return {"rmse":rmse,"radial_alignment":alignment,"inverse_power_exponent":float(-slope)}

def model(seed,depth,updates):
    native=default_native();native.learning_rate=.008;native.batch_size=256
    native.structure=StructureConfig(dynamic=False,initial_depth=0,max_depth=max(depth,1),max_nodes=max(31,2**(depth+1)-1))
    return UnifiedProgressiveRegressor(UnifiedConfig(n_trees=1,updates_per_stage=updates,depth=depth,bins=10,
        min_samples_leaf=16,linear_values=True,linear_l2=5.,proposal_mode="hist_newton",warm_value_updates=8,
        gate_release="oblique",checkpoint_every=max(8,updates//16),auto_complexity=True,proposal_candidates=2,
        regularizers=Regularizers(leaf_l2=1e-5,hierarchy=3e-4,linear_value_l2=2e-5),native=native,random_state=seed))

def run(seed,nfit,depth,updates,noise):
    X,y=central_force_data(nfit+7000,seed,noise);Xood,yood=central_force_data(3500,seed+991,0.,radius=(2.05,2.8))
    p=np.random.default_rng(seed+17).permutation(len(X));tr=p[:nfit];sel=p[nfit:nfit+2000];audit=p[nfit+2000:]
    Z=X[audit];rad=np.linalg.norm(Z[:,:2],axis=1);mu=Z[:,4];clean=np.c_[-mu*Z[:,0]/rad**3,-mu*Z[:,1]/rad**3]
    t=time.time();m=model(seed,depth,updates).fit(X[tr],y[tr],eval_set=(X[sel],y[sel]))
    out={"seed":seed,"nfit":nfit,"depth":depth,"updates":updates,"noise":noise,
         "torchboost":diagnostics(m,Z,clean),"torchboost_ood":diagnostics(m,Xood,yood),
         "best_step":m.trainer_.best_epoch,"seconds":time.time()-t}
    cb=CatBoostRegressor(iterations=512,depth=8,learning_rate=.05,l2_leaf_reg=20,loss_function="MultiRMSE",
                         verbose=False,random_seed=seed,thread_count=1).fit(X[tr],y[tr])
    out["catboost"]=diagnostics(cb,Z,clean);out["catboost_ood"]=diagnostics(cb,Xood,yood);return out

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--seed",type=int,default=31);a.add_argument("--nfit",type=int,default=12000)
    a.add_argument("--depth",type=int,default=4);a.add_argument("--updates",type=int,default=512)
    a.add_argument("--noise",type=float,default=.05);a.add_argument("--out",required=True);z=a.parse_args()
    result=run(z.seed,z.nfit,z.depth,z.updates,z.noise);Path(z.out).write_text(json.dumps(result,indent=2));print(json.dumps(result))
