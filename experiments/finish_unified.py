"""Finish registered development blocks and evaluate locked choices exactly once.

This file does not select configurations using audit labels. Failed/missing runs
are explicitly retained. Final artifacts state limitations and incomplete work.
"""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import json,hashlib,time,traceback,os
import numpy as np
import pandas as pd
import torch
import joblib
from unified_study import ROOT,OUT,DATA,source_hashes,run,job,metric
from torchboost.adaptive.unified_progressive import UnifiedConfig
from torchboost.adaptive.data import Preprocessor
from torchboost.adaptive.objectives import Objective
from torchboost.adaptive.training import restore_model


def results(block):
    records=[]
    for p in (OUT/'runs').glob('*.json'):
        d=json.loads(p.read_text())
        if d['block']==block:records.append(d)
    return records


def choose(block,ds):
    rows=[r for r in results(block) if r['dataset']==ds and r['status']=='success']
    groups={}
    for r in rows:groups.setdefault(r['name'],[]).append(r)
    groups={k:v for k,v in groups.items() if len(v)==2}
    if not groups:raise RuntimeError(f'no complete two-seed candidate in {block} {ds}')
    name=min(groups,key=lambda k:np.mean([r['rank_loss'] for r in groups[k]]))
    return groups[name]


def register(block,jobs):
    path=OUT/'protocols'/f'{block}.json'
    if not path.exists():path.write_text(json.dumps({'block':block,'source_hashes':source_hashes(),'jobs':jobs,'audit_accessed':False,'design':'adaptive development on rank data; not independent confirmation'},indent=2))


def main():
    # Resume registered components; completed result files are not silently rerun.
    run('components',3)
    J=[]
    for ds in ['axis32','oblique32','diamonds']:
        chosen=choose('components',ds)
        for seed in [173,291]:
            original=next(r for r in chosen if r['seed']==seed)
            for idx in range(6):
                c=UnifiedConfig(**deepcopy(original['config']));c.n_trees=32;c.updates_per_stage=16
                c.depth=[2,3,3,4,3,3][idx];c.native.structure.max_depth=max(c.native.structure.max_depth,c.depth)
                c.shrinkage=[.2,.2,.1,.1,.2,.2][idx];c.newton_l2=[1.,1.,5.,5.,10.,1.][idx]
                if idx==4:c.regularizers.hierarchy=max(c.regularizers.hierarchy,.03)
                if idx==5:c.age_decay=0.;c.head_mode='none'
                if original['name']=='joint':c.active_window=c.n_trees
                c.__post_init__();J.append(job(ds,f'tune{idx}',seed,c,'tuning'))
    register('tuning',J);run('tuning',3)
    J=[]
    # Matched removal and additions from each tuned procedure. Each dependency
    # removal is explicit rather than pretending incompatible states can persist.
    for ds in ['axis32','oblique32','diamonds']:
        chosen=choose('tuning',ds)
        for seed in [173,291]:
            r=next(r for r in chosen if r['seed']==seed)
            for name in ['full','no_hierarchy','no_sampling','no_physics','no_memory','no_online','no_structure','no_age','with_memory','with_capacitor']:
                c=UnifiedConfig(**deepcopy(r['config']))
                if name=='no_hierarchy':c.regularizers.hierarchy=0.;c.regularizers.allocation='fixed'
                if name=='no_sampling':c.row_subsample=1.;c.feature_subsample=1.
                if name=='no_physics':c.native.physics.mode='none';c.native.physics.spark_probability=0.;c.native.physics.transfer_fraction=0.;c.native.plasticity.thermal_softening=0.;c.native.optimizer='momentum' if c.native.optimizer=='circuit_momentum' else c.native.optimizer
                if name=='no_memory':c.native.plasticity.mode='none';c.native.online.enabled=False
                if name=='no_online':c.native.online.enabled=False
                if name=='no_structure':c.native.structure.dynamic=False
                if name=='no_age':c.age_decay=1.
                if name=='with_memory':
                    from torchboost.adaptive.config import PlasticityConfig
                    c.native.plasticity=PlasticityConfig(mode='anchor',stiffness=.3,evidence_threshold=2.)
                if name=='with_capacitor':
                    from unified_study import config
                    c.native.physics=deepcopy(config('capacitor',seed).native.physics)
                c.__post_init__();J.append(job(ds,name,seed,c,'ablations'))
    register('ablations',J);run('ablations',3)
    # A 64-stage extension is a new explicitly budgeted fit. It is not mislabeled
    # as exact continuation or as an equal-compute baseline comparison.
    J=[]
    for ds in ['axis32','oblique32','diamonds']:
        chosen=choose('tuning',ds)
        for r in chosen:
            c=UnifiedConfig(**deepcopy(r['config']));c.n_trees=64;c.updates_per_stage=16
            if c.age_decay==1:c.active_window=max(c.active_window,32)
            J.append(job(ds,'64_stages',r['seed'],c,'capacity',cap=8192))
    register('capacity',J);run('capacity',3)
    lock=[]
    for ds in ['axis32','oblique32','diamonds']:
        for block in ['components','tuning','ablations','capacity']:
            for r in choose(block,ds):lock.append({'id':r['id'],'dataset':ds,'seed':r['seed'],'block':block,'rank_loss':r['rank_loss'],'model_sha256':hashlib.sha256((OUT/'models'/f"{r['id']}.pt").read_bytes()).hexdigest()})
    # Ablation scores are also recorded, but never used to revise this manifest.
    lockfile=OUT/'locked_choices.json'
    if lockfile.exists():
        if json.loads(lockfile.read_text())!=lock:raise ValueError('locked choices changed')
    else:lockfile.write_text(json.dumps(lock,indent=2))
    audit=[];torch.set_num_threads(1)
    for r in lock:
        z=np.load(DATA/f"{r['dataset']}.npz");ids=z['audit'];x,y=z['X'][ids],z['y'][ids]
        s=torch.load(OUT/'models'/f"{r['id']}.pt",map_location='cpu',weights_only=False)
        pre=Preprocessor();pre.load_state_dict(s['preprocessor']);cfg=__import__('torchboost.adaptive.config',fromlist=['ForestConfig']).ForestConfig(**s['native'])
        model=restore_model(s['snapshot'],len(pre.mean),s['output_dim'],cfg);model.eval();xx=pre.transform_x(x)
        with torch.no_grad():raw=torch.cat([model(b) for b in xx.split(1024)])
        reg=s['task']=='regression';pred=pre.inverse_target(raw.numpy()) if reg else Objective(s['task'],s['output_dim']).response(raw).numpy()
        if reg and pred.shape[1]==1:pred=pred[:,0]
        audit.append({**r,'audit_loss':metric(reg,y,pred),'selected_trees':len(model.trees),'parameters':sum(p.numel() for p in model.parameters())})
        np.savez_compressed(OUT/'models'/f"{r['id']}_audit.npz",prediction=pred,y=y,indices=ids)
    pd.DataFrame(audit).to_csv(OUT/'audit_selected.csv',index=False)
    # Reference selection is locked by rank, not by looking for an audit winner.
    refs=[];refroot=ROOT/'results/current/references'
    for p in refroot.glob('*.json'):
        r=json.loads(p.read_text())
        if r.get('status')=='success':refs.append(r)
    refchoices=[]
    for ds in ['axis32','oblique32','diamonds']:
        for family in ['xgb','lgb','cat']:
            groups={}
            for r in refs:
                if r['dataset']==ds and r['family']==family:groups.setdefault(r['index'],[]).append(r)
            groups={k:v for k,v in groups.items() if len(v)==2}
            if groups:
                key=min(groups,key=lambda k:np.mean([r['rank_loss'] for r in groups[k]]));refchoices.extend(groups[key])
    (refroot/'locked_choices.json').write_text(json.dumps(refchoices,indent=2));refaudit=[]
    for r in refchoices:
        z=np.load(DATA/f"{r['dataset']}.npz");ids=z['audit'];x,y=z['X'][ids],z['y'][ids]
        m=joblib.load(refroot/(r['id']+'.joblib'));reg=r['dataset']=='diamonds';p=m.predict(x) if reg else m.predict_proba(x)
        refaudit.append({**r,'audit_loss':metric(reg,y,p)})
    pd.DataFrame(refaudit).to_csv(refroot/'audit_selected.csv',index=False)

if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'results/current/finish_error.txt').write_text(traceback.format_exc());raise
