
import os,time,json,numpy as np
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1")
from sklearn.metrics import log_loss
from torchboost.adaptive.unified_progressive import UnifiedConfig,UnifiedProgressiveClassifier,default_native
from torchboost.adaptive.config import PhysicsConfig,PlasticityConfig,OnlineConfig,StructureConfig
from torchboost.adaptive.progressive_regularizers import Regularizers

D=24
def domain(n,which,seed,shock=0.):
    r=np.random.default_rng(seed);x=r.normal(size=(n,D)).astype("float32")
    if which=="A":
        z=x[:,0]+.6*x[:,1]-.35*x[:,2];c=z>0
        raw=np.where(c,1.2*x[:,6]-.8*x[:,7]+.5*x[:,8],-.9*x[:,6]+1.05*x[:,9]-.4*x[:,10])
    else:
        z=.4*x[:,0]+.8*x[:,1]-.2*x[:,2]+.55*x[:,3];c=z>0
        raw=np.where(c,.65*x[:,6]-1.1*x[:,7]+.75*x[:,11],-1.1*x[:,6]+.7*x[:,9]+.65*x[:,12])
    raw=raw/np.std(raw)*1.3;p=1/(1+np.exp(-raw));y=r.binomial(1,p).astype(int)
    if shock:
        flip=r.random(n)<shock;y=np.where(flip,1-y,y)
    return x,y

AUD={k:domain(2200,k,9000+i) for i,k in enumerate(("A","B"))}
def cfg(kind,seed,stages,updates):
    n=default_native();n.learning_rate=.012;n.batch_size=256;n.observation_every=4;n.control_sample_size=192
    n.structure=StructureConfig(dynamic=(kind=="full"),initial_depth=0,max_depth=6,max_nodes=511,
        grow_every=12,prune_every=24,grow_per_event=1,structural_gate=True,complexity=1e-4,allocation_regularization=1e-4)
    if kind in ("plastic","pulse","cap","rlc","full"):
        n.plasticity=PlasticityConfig(mode="full",stiffness=.006,yield_threshold=.22,mobility=.05,
            work_hardening=.12,thermal_softening=.2 if kind in ("cap","rlc","full") else 0.,
            consolidation_rate=.05,release_policy="persistent_harm",release_patience=5)
    if kind=="pulse":
        n.physics=PhysicsConfig(mode="cooling",topology_normalization=True,capacitance=1.,discharge_time=5.,
            inductive_time=2.,cooling_time=24.,total_heat_capacity=.06,dt=.2,
            initial_temperature=1.,ambient_temperature=1.,max_temperature=3.,thaw_temperature=1.08)
    if kind in ("cap","rlc","full"):
        mode="capacitor" if kind=="cap" else "rlc"
        n.physics=PhysicsConfig(mode=mode,topology_normalization=True,capacitance=1.,discharge_time=5.,
            inductive_time=2.,cooling_time=24.,total_heat_capacity=.06,dt=.2,
            initial_temperature=1.,ambient_temperature=1.,max_temperature=3.,thaw_temperature=1.08,
            charge_gain=4.,max_injection=.25,max_charge=1.5,smoothing=.8,lr_coupling=.04)
    if kind=="full":
        n.online=OnlineConfig(enabled=True,interval=8,window=4,cooldown=4,max_trials=4,defer_structure_for_trials=True)
    return UnifiedConfig(n_trees=stages,updates_per_stage=updates,depth=3,bins=12,min_samples_leaf=20,
        shrinkage=.65,age_decay=.15,active_window=2,row_subsample=.9,feature_subsample=.9,
        cart_strength=6.,warm_value_updates=8,gate_release="oblique",readout="residual",
        linear_values=True,linear_l2=5.,proposal_mode="hist_newton",checkpoint_every=12,
        anchor_min_passes=.5,anchor_min_updates=24,
        regularizers=Regularizers(leaf_l2=1e-5,hierarchy=2e-4,linear_value_l2=1e-5,feature_l1=1e-6),
        native=n,random_state=seed)

def run(kind,seed,sequence,updates=48,shock_cycle=None):
    stages=len(sequence);m=UnifiedProgressiveClassifier(cfg(kind,seed,stages,updates));trajectory=[]
    for stage,name in enumerate(sequence):
        shock=.35 if shock_cycle==stage else 0.
        x,y=domain(1600,name,seed*1000+stage*31+1,shock)
        xc,yc=domain(650,name,seed*1000+stage*31+2,shock)
        xs,ys=domain(650,name,seed*1000+stage*31+3,shock)
        if stage==0:m.fit(x,y,control_set=(xc,yc),eval_set=(xs,ys),stop_stages=1)
        else:
            if kind=="pulse":
                # Matched simple control: a fixed non-electrical heat pulse on each regime change.
                m.trainer_.physical.synchronize({n.node_id:n.tree_id for n in m.trainer_.model.iter_nodes()})
                for n in m.trainer_.model.iter_nodes():
                    n.set_temperature(2.0)
                    st=m.trainer_.physical.nodes.get(n.node_id)
                    if st is not None: st["temperature"]=2.0
            m.continue_fit(x,y,control_set=(xc,yc),eval_set=(xs,ys),stop_stages=stage+1,allow_domain_shift=True)
        trajectory.append({"stage":stage,"domain":name,
            "A":log_loss(AUD["A"][1],m.predict_proba(AUD["A"][0],last=True)),
            "B":log_loss(AUD["B"][1],m.predict_proba(AUD["B"][0],last=True)),
            "selection":float(m.trainer_.history[-1]["selection_loss"])})
    h=m.trainer_.history[-1]
    return {"kind":kind,"seed":seed,"updates":updates,"sequence":sequence,"shock_cycle":shock_cycle,
        "trajectory":trajectory,"current_regret_proxy":float(sum(r[r["domain"]] for r in trajectory)),
        "events":h.get("event_counts",{}),"anchors":h.get("admitted_anchors",0),
        "injection":h.get("cumulative_injection",0.),"max_temperature":h.get("max_temperature_seen",1.)}

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--suite",choices=["recurring","shock"]);ap.add_argument("--seed",type=int);ap.add_argument("--kind");ap.add_argument("--updates",type=int,default=48);ap.add_argument("--out",required=True);a=ap.parse_args()
    seq=["A","B","A","B","A"] if a.suite=="recurring" else ["A","A","A"]
    shock=1 if a.suite=="shock" else None
    t=time.time();r=run(a.kind,a.seed,seq,a.updates,shock);r["seconds"]=time.time()-t
    Path(a.out).write_text(json.dumps(r,indent=2));print(json.dumps(r))