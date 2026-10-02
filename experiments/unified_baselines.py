from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import os,time,json,hashlib,traceback,joblib
import numpy as np
from sklearn.metrics import log_loss,mean_squared_error
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor,early_stopping,log_evaluation
from catboost import CatBoostClassifier,CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/current';OUT=ROOT/'results/current/references'

def jobs():
    result=[]
    for ds in ['axis32','oblique32','diamonds']:
      for seed in [173,291]:
        for family in ['xgb','lgb','cat']:
          for i in range(8 if family=='xgb' else 4):result.append(dict(id=f'{ds}_{seed}_{family}{i}',dataset=ds,seed=seed,family=family,index=i))
    return result

def run_one(j):
    p=OUT/(j['id']+'.json')
    if p.exists():return j['id'],'exists'
    t=time.perf_counter();r=j.copy()
    try:
        z=np.load(D/(j['dataset']+'.npz'));x,y=z['X'],z['y'];a=z['fit'][:2048];b=z['selection'][:512];c=z['rank'][:1024]
        reg=j['dataset']=='diamonds';i=j['index'];seed=j['seed']
        if j['family']=='xgb':
            params=dict(max_depth=[3,6][i//4],learning_rate=[.03,.1][(i//2)%2],min_child_weight=[1,10][i%2],reg_lambda=5.,n_estimators=512,subsample=.8,colsample_bytree=.8,n_jobs=1,random_state=seed,tree_method='hist',early_stopping_rounds=40,eval_metric='rmse' if reg else 'logloss')
            model=(XGBRegressor if reg else XGBClassifier)(**params);model.fit(x[a],y[a],eval_set=[(x[b],y[b])],verbose=False)
        elif j['family']=='lgb':
            params=dict(num_leaves=[15,31][i//2],reg_lambda=[0.,5.][i%2],learning_rate=.05,n_estimators=512,colsample_bytree=.8,min_child_samples=10,n_jobs=1,random_state=seed,verbosity=-1)
            model=(LGBMRegressor if reg else LGBMClassifier)(**params);model.fit(x[a],y[a],eval_set=[(x[b],y[b])],callbacks=[early_stopping(40,verbose=False),log_evaluation(0)])
        else:
            params=dict(depth=[4,6][i//2],l2_leaf_reg=[3.,10.][i%2],learning_rate=.05,iterations=512,thread_count=1,random_seed=seed,verbose=False,allow_writing_files=False)
            model=(CatBoostRegressor if reg else CatBoostClassifier)(**params);model.fit(x[a],y[a],eval_set=(x[b],y[b]),early_stopping_rounds=40)
        prediction=model.predict(x[c]) if reg else model.predict_proba(x[c]);loss=float(np.sqrt(mean_squared_error(y[c],prediction))) if reg else float(log_loss(y[c],prediction,labels=[0,1]))
        joblib.dump(model,OUT/(j['id']+'.joblib'));np.savez_compressed(OUT/(j['id']+'_rank.npz'),prediction=prediction,y=y[c],indices=c)
        r.update(status='success',rank_loss=loss,params=params)
    except Exception:r.update(status='failed',error=traceback.format_exc())
    r['seconds']=time.perf_counter()-t;p.write_text(json.dumps(r,indent=2));return j['id'],r['status']

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);J=jobs()
    (OUT/'protocol.json').write_text(json.dumps({'jobs':J,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'fit_rows':2048,'selection_rows':512,'rank_rows':1024,'audit_accessed':False,'comparison':'same data roles; unequal fitting budgets and search sizes'},indent=2))
    with ProcessPoolExecutor(max_workers=1) as ex:
        for i,f in enumerate(as_completed([ex.submit(run_one,j) for j in J]),1):print(i,len(J),*f.result(),flush=True)
