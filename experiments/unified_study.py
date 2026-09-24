"""Component addition, validation tuning, ablation and locked audit.

No audit labels enter fitting, checkpoint choice, config ranking, or follow-up
selection. Every registered job has a source hash, explicit budget, and result.
"""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,json,time,traceback,os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss,mean_squared_error
from torchboost.adaptive.unified_progressive import UnifiedConfig,UnifiedProgressiveClassifier,UnifiedProgressiveRegressor
from torchboost.adaptive.config import PhysicsConfig,PlasticityConfig,OnlineConfig,ScheduleConfig
from torchboost.adaptive.data import Preprocessor
from torchboost.adaptive.objectives import Objective
from torchboost.adaptive.training import restore_model
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/current/study';DATA=ROOT/'data/current'


def source_hashes():
    paths=list((ROOT/'torchboost/adaptive').glob('*.py'))+[Path(__file__)]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def make_data():
    DATA.mkdir(parents=True,exist_ok=True)
    def split(n,groups=None):
        rng=np.random.default_rng(9047)
        a=np.arange(n) if groups is None else np.unique(groups)
        a=rng.permutation(a);parts=np.split(a,[int(len(a)*.55),int(len(a)*.65),int(len(a)*.75),int(len(a)*.85)])
        return {k:(v if groups is None else np.flatnonzero(np.isin(groups,v))) for k,v in zip(['fit','control','selection','rank','audit'],parts)}
    r=np.random.default_rng(472051);n=24000;d=12;x=r.normal(size=(n,d)).astype('float32')
    context=sum((x[:,j]>0).astype(int)*2**j for j in range(5));coef=r.normal(size=(32,5));coef/=np.linalg.norm(coef,axis=1)[:,None]
    logits=1.6*(coef[context]*x[:,5:10]).sum(1)+.3*np.sin(x[:,10]*x[:,11]);p=1/(1+np.exp(-logits));y=r.binomial(1,p)
    q,_=np.linalg.qr(r.normal(size=(d,d)));manifest={}
    for name,xx in [('axis32',x),('oblique32',(x@q).astype('float32'))]:
        roles=split(n);np.savez_compressed(DATA/f'{name}.npz',X=xx,y=y,probability=p,**roles)
        manifest[name]={'kind':'synthetic binary','n':n,'features':d,'seed':472051,'formula':'32 conditional linear contexts, plus smooth interaction, Bernoulli labels; orthogonal pair shares labels','roles':{k:len(v) for k,v in roles.items()}}
    path=Path('/opt/pyvenv/lib/python3.13/site-packages/plotnine/data/diamonds.csv');frame=pd.read_csv(path)
    rows=np.random.default_rng(48721).permutation(len(frame))[:24000];frame=frame.iloc[rows].reset_index(drop=True)
    features=frame.drop(columns='price');groups=pd.util.hash_pandas_object(features,index=False).to_numpy();roles=split(len(frame),groups)
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    numeric=list(features.select_dtypes(include=np.number).columns);cat=[c for c in features if c not in numeric]
    enc=ColumnTransformer([('numeric','passthrough',numeric),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])
    enc.fit(features.iloc[roles['fit']]);xx=enc.transform(features).astype('float32')
    np.savez_compressed(DATA/'diamonds.npz',X=xx,y=np.log(frame.price.to_numpy()).astype('float32'),source_rows=rows,**roles)
    manifest['diamonds']={'kind':'real log-price regression','n':len(frame),'features':xx.shape[1],'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source':str(path),'encoder_fit':'fit rows only','split':'identical predictor profiles kept together','limitations':'reused public task; not independent task confirmation','roles':{k:len(v) for k,v in roles.items()}}
    from sklearn.datasets import load_digits
    digits=load_digits();roles=split(len(digits.target));np.savez_compressed(DATA/'digits.npz',X=digits.data.astype('float32'),y=digits.target,**roles)
    manifest['digits']={'kind':'reused multiclass control','n':len(digits.target),'features':64,'roles':{k:len(v) for k,v in roles.items()}}
    x=r.normal(size=(6000,5)).astype('float32');y=2*x[:,0]+.3*np.sin(x[:,1])+r.normal(0,.5,len(x));roles=split(len(x));np.savez_compressed(DATA/'monotone.npz',X=x,y=y.astype('float32'),**roles)
    manifest['monotone']={'kind':'synthetic monotone regression probe','n':len(x),'features':5,'known_monotonic_feature':0,'roles':{k:len(v) for k,v in roles.items()}}
    for name in manifest:manifest[name]['sha256']=hashlib.sha256((DATA/f'{name}.npz').read_bytes()).hexdigest()
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2))


def config(name,seed,trees=12,updates=16):
    c=UnifiedConfig(n_trees=trees,updates_per_stage=updates,depth=2,random_state=seed)
    c.native.observation_every=4;c.native.control_sample_size=64;c.native.structure.cycle_epochs=16
    if name=='hard':c.gate_release='hard';c.age_decay=0.
    if name=='strict':c.age_decay=0.
    if name=='joint':c.age_decay=1.;c.active_window=trees
    if name=='rows':c.row_subsample=.7
    if name=='features':c.feature_subsample=.7
    if name=='leaf':c.regularizers.leaf_l2=.02
    if name in ['hierarchy','learned','cyclic','core','core_memory','all']:c.regularizers.hierarchy=.03
    if name in ['learned','all']:c.regularizers.allocation='learned'
    if name=='cyclic':c.regularizer_schedule=ScheduleConfig('oscillatory',0.,2.,cycles=2.)
    if name=='feature_penalty':c.regularizers.feature_l1=.001
    if name=='tree_penalty':c.regularizers.tree_l2=.01
    if name=='balance':c.regularizers.route_balance=.005
    if name=='child':c.regularizers.child_penalty=.1;c.regularizers.min_child_fraction=.15
    if name=='support':c.min_child_weight=10.;c.split_cost=.1
    if name=='feature_dropout':c.native.feature_dropout=.05
    if name=='tree_dropout':c.native.tree_dropout=.1
    if name=='diversity':c.native.diversity=.005
    if name=='newton':c.refit_every=4;c.refit_damping=.02
    if name in ['core','core_memory','all']:
        c.row_subsample=.8;c.feature_subsample=.8;c.newton_l2=5.;c.regularizers.leaf_l2=.005;c.split_cost=.1
    if name in ['cooling','capacitor','rlc','circuit','rlc_fixed','all']:
        mode='cooling' if name=='cooling' else 'capacitor' if name=='capacitor' else 'rlc'
        c.native.physics=PhysicsConfig(mode=mode,initial_temperature=1.,ambient_temperature=.7,max_temperature=4.,
            capacitance=1.,resistance=6.,inductance=6.,dt=1.,heat_capacity=.002,cooling=.00002,charge_gain=15.,max_injection=.25,thaw_temperature=1.2,hierarchical=True)
    if name in ['memory','plastic','damage','online','core_memory','all']:
        mode='anchor' if name in ['memory','online','core_memory'] else 'plastic' if name=='plastic' else 'full'
        c.native.plasticity=PlasticityConfig(mode=mode,stiffness=.3,yield_threshold=.003,mobility=.03,work_hardening=.2,
            thermal_softening=.2 if name=='all' else 0.,damage_rate=.02,evidence_threshold=2.,minimum_utility=1e-5,consolidation_rate=.03)
        c.anchor_min_passes=.25;c.anchor_min_updates=2
    if name in ['online','all']:
        c.native.online=OnlineConfig(enabled=True,interval=4,window=2,cooldown=2,exploration=.3,deformation_source='parameters',defer_structure_for_trials=True)
    if name in ['dynamic','all']:
        c.native.structure.dynamic=True;c.native.structure.structural_gate=True;c.native.structure.max_depth=3
        c.native.structure.grow_every=4;c.native.structure.prune_every=8;c.native.structure.initial_dormant_fraction=0.
        c.native.structure.complexity=.001;c.native.structure.gate_bimodality=.001;c.native.structure.allocation_regularization=.001
        c.native.structure.trial_relaxation=True
    if name in ['momentum','energy','circuit','rlc_fixed']:
        c.native.optimizer={'momentum':'momentum','energy':'energy_momentum','circuit':'circuit_momentum','rlc_fixed':'momentum'}[name]
        c.native.learning_rate=.1;c.native.momentum=.7;c.native.momentum_max=.95;c.native.momentum_energy_gain=10.
    if name in ['shared','specialized']:c.head_mode=name
    if name=='leaf_readout':c.readout='leaf'
    if name=='monotonic':c.native.monotonicity=((0,0,1),);c.native.monotonicity_penalty=.1
    if name=='interaction':c.native.interaction_groups=((0,1,2),(3,4,5),(6,7,8),(9,10,11))
    if name=='feature_specific':c.native.feature_penalties=(.001,)*12
    c.__post_init__();return c

COMPONENTS=['base','hard','strict','joint','rows','features','leaf','hierarchy','learned','cyclic','feature_penalty',
 'tree_penalty','balance','child','support','feature_dropout','tree_dropout','diversity','newton','cooling','capacitor','rlc',
 'memory','plastic','damage','online','dynamic','core','core_memory','all','momentum','energy','circuit','rlc_fixed','leaf_readout']


def job(ds,name,seed,c,block='components',cap=2048):
    return dict(id=f'{block}_{ds}_{seed}_{name}',dataset=ds,name=name,seed=seed,config=asdict(c),block=block,
        fit_cap=cap,control_cap=512,selection_cap=512,rank_cap=1024)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    for sub in ['runs','models','protocols']:(OUT/sub).mkdir(exist_ok=True)
    jobs=[job(ds,name,seed,config(name,seed)) for ds in ['axis32','oblique32','diamonds'] for seed in [173,291] for name in COMPONENTS]
    for ds,names in [('digits',['base','shared','specialized']),('monotone',['base','monotonic']),('axis32',['interaction','feature_specific'])]:
        for seed in [173,291]:
            for name in names:jobs.append(job(ds,name,seed,config(name,seed,trees=8,updates=16),'targeted'))
    protocol={'block':'components','source_hashes':source_hashes(),'jobs':jobs,'audit_accessed':False,
        'roles':'disjoint fitting/control/checkpoint-selection/config-ranking/audit','limitations':'development comparisons, reused real tasks, bounded searches, no equal-compute claim','seeds':[173,291]}
    (OUT/'protocols/components.json').write_text(json.dumps(protocol,indent=2));print('prepared',len(jobs))


def metric(reg,y,p):return float(np.sqrt(mean_squared_error(y,p))) if reg else float(log_loss(y,p,labels=np.arange(p.shape[1])))


def fit_job(j):
    torch.set_num_threads(1);path=OUT/'runs'/f"{j['id']}.json"
    if path.exists():return j['id'],'exists'
    start=time.perf_counter();r={**j}
    try:
        z=np.load(DATA/f"{j['dataset']}.npz");x,y=z['X'],z['y'];roles={k:z[k][:j[k+'_cap']] for k in ['fit','control','selection','rank']}
        reg=j['dataset'] in ['diamonds','monotone'];cls=UnifiedProgressiveRegressor if reg else UnifiedProgressiveClassifier
        c=UnifiedConfig(**j['config']);m=cls(c).fit(x[roles['fit']],y[roles['fit']],control_set=(x[roles['control']],y[roles['control']]),eval_set=(x[roles['selection']],y[roles['selection']]))
        p=m.predict(x[roles['rank']]) if reg else m.predict_proba(x[roles['rank']]);t=m.trainer_;s=t.best_epoch
        ph=t.physical.history;pe=t.plastic.events
        activation=lambda before:dict(charge=sum(v['injected_charge'] for v in ph if not before or v['step']<=s),heat=sum(v['resistor_heat'] for v in ph if not before or v['step']<=s),
            flow=sum(v.get('flow_fraction',0)>0 for v in pe if not before or v['step']<=s),damage=sum(v.get('damage',0)>0 for v in pe if not before or v['step']<=s))
        r.update(status='success',rank_loss=metric(reg,y[roles['rank']],p),selected_loss=m.best_score_,selected_trees=m.n_estimators_,
          best_step=s,history=t.history,proposals=t.proposal_history,events=t.events,refits=t.refit_history,
          whole_run=activation(False),before_selected=activation(True),admitted=len(t.admitted),online_completed=int(t.scheduler.counts.sum()),
          max_energy_error=max([abs(v['energy_error']) for v in ph] or [0.]),optimizer_updates=t.optimizer_steps,examples_seen=t.examples_seen,
          selected_parameters=sum(v.numel() for v in m.model_.parameters()))
        torch.save({'snapshot':t.best_snapshot,'native':asdict(t.config),'preprocessor':m.preprocessor_.state_dict(),'task':m.objective_.task,'output_dim':m.objective_.output_dim},OUT/'models'/f"{j['id']}.pt")
        np.savez_compressed(OUT/'models'/f"{j['id']}_rank.npz",prediction=p,y=y[roles['rank']],indices=roles['rank'])
    except Exception:r.update(status='failed',error=traceback.format_exc())
    r['seconds']=time.perf_counter()-start;path.write_text(json.dumps(r,indent=2));return j['id'],r['status']


def run(block,workers=3):
    p=json.loads((OUT/'protocols'/f'{block}.json').read_text())
    if p['source_hashes']!=source_hashes():raise ValueError('source differs from registered protocol')
    todo=[j for j in p['jobs'] if not (OUT/'runs'/f"{j['id']}.json").exists()]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures=[ex.submit(fit_job,j) for j in todo]
        for i,f in enumerate(as_completed(futures),1):print(i,len(todo),*f.result(),flush=True)


def summary():
    rows=[]
    for f in (OUT/'runs').glob('*.json'):
        r=json.loads(f.read_text())
        if r['status']=='success':rows.append({k:r[k] for k in ['id','block','dataset','name','seed','rank_loss','selected_trees','seconds','best_step','optimizer_updates']})
    d=pd.DataFrame(rows);d.to_csv(OUT/'development_runs.csv',index=False)
    s=d.groupby(['block','dataset','name']).agg(rank_loss=('rank_loss','mean'),sd=('rank_loss','std'),count=('rank_loss','size'),trees=('selected_trees','mean')).reset_index()
    s.to_csv(OUT/'development_summary.csv',index=False);print(s.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['data','prepare','run','summary']);p.add_argument('--block',default='components');p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if a.action=='data':make_data()
    elif a.action=='prepare':prepare()
    elif a.action=='run':run(a.block,a.workers)
    else:summary()
