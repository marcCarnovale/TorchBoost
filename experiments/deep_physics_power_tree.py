"""Long-horizon adaptive-control study on the CatBoost-ratchet power-tree task."""
import argparse,json,time,numpy as np
from pathlib import Path
from sklearn.metrics import log_loss
from torchboost.adaptive.unified_progressive import UnifiedConfig,UnifiedProgressiveClassifier,default_native
from torchboost.adaptive.config import StructureConfig,PhysicsConfig,PlasticityConfig,OnlineConfig
from torchboost.adaptive.progressive_regularizers import Regularizers

def make_dataset(seed):
    nfit=12000;nselection=2000;naudit=3000
    rng=np.random.default_rng(seed);X=rng.normal(size=(nfit+nselection+naudit,16)).astype("float32")
    bits=(X[:,:4]>0).astype(int);context=sum(bits[:,j]*(1<<j) for j in range(4))
    coefficient=rng.normal(size=(16,16));coefficient[:,:4]=0
    raw=np.array([coefficient[k]@row for k,row in zip(context,X)]);raw=raw/np.std(raw)*1.15
    p=1/(1+np.exp(-raw));return X,rng.binomial(1,p).astype(int)

def config(kind,seed,updates):
    native=default_native();native.learning_rate=.01;native.batch_size=256;native.observation_every=4;native.control_sample_size=256
    native.structure=StructureConfig(dynamic=(kind=="full"),initial_depth=0,max_depth=6,max_nodes=511,grow_every=24,
        prune_every=48,grow_per_event=1,growth_policy="hybrid" if kind=="full" else "best_first",
        structural_gate=True,complexity=1e-4,allocation_regularization=1e-4)
    if kind in ("plastic","cap","rlc","full"):
        native.plasticity=PlasticityConfig(mode="full",stiffness=.004,yield_threshold=.2,mobility=.04,work_hardening=.12,
            thermal_softening=.18 if kind in ("cap","rlc","full") else 0.,consolidation_rate=.04,
            release_policy="persistent_harm",release_patience=5)
    if kind in ("cap","rlc","full"):
        native.physics=PhysicsConfig(mode="capacitor" if kind=="cap" else "rlc",topology_normalization=True,
            capacitance=1.,discharge_time=5.,inductive_time=2.,cooling_time=32.,total_heat_capacity=.08,dt=.2,
            initial_temperature=1.,ambient_temperature=1.,max_temperature=2.5,thaw_temperature=1.08,
            charge_gain=3.,max_injection=.2,max_charge=1.5,smoothing=.85,lr_coupling=.025)
    if kind=="full":
        native.online=OnlineConfig(enabled=True,interval=16,window=8,cooldown=8,max_trials=4,defer_structure_for_trials=True)
    return UnifiedConfig(n_trees=1,updates_per_stage=updates,depth=4,bins=8,min_samples_leaf=12,linear_values=True,
        linear_l2=8.,proposal_mode="linear_model_tree",cart_strength=7.,warm_value_updates=8,gate_release="oblique",
        checkpoint_every=32,auto_complexity=True,proposal_candidates=1,
        regularizers=Regularizers(leaf_l2=1e-5,hierarchy=3e-4,linear_value_l2=2e-5),native=native,random_state=seed)

def run(kind,seed,updates):
    X,y=make_dataset(seed);p=np.random.default_rng(seed+9).permutation(len(X))
    tr=p[:12000];control=p[12000:13800];selection=p[13800:15800];audit=p[15800:]
    t=time.time();m=UnifiedProgressiveClassifier(config(kind,seed,updates)).fit(
        X[tr],y[tr],control_set=(X[control],y[control]),eval_set=(X[selection],y[selection]))
    h=m.trainer_.history[-1]
    return {"kind":kind,"seed":seed,"updates":updates,"selection":m.best_score_,
        "audit":log_loss(y[audit],m.predict_proba(X[audit])),"last_audit":log_loss(y[audit],m.predict_proba(X[audit],last=True)),
        "best_step":m.trainer_.best_epoch,"anchors":h.get("admitted_anchors",0),"events":h.get("event_counts",{}),
        "injection":h.get("cumulative_injection",0.),"max_temperature":h.get("max_temperature_seen",1.),"seconds":time.time()-t}

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--kind",choices=["none","plastic","cap","rlc","full"],required=True)
    a.add_argument("--seed",type=int,default=71);a.add_argument("--updates",type=int,default=512);a.add_argument("--out",required=True)
    z=a.parse_args();r=run(z.kind,z.seed,z.updates);Path(z.out).write_text(json.dumps(r,indent=2));print(json.dumps(r))
