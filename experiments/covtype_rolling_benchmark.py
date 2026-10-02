"""Selection-screened TorchBoost rolling forests on the large Covertype 1-vs-2 task.

Train/control/selection/ranking/audit are disjoint.  Control is used only for
within-fit prefix selection, selection chooses the predeclared architecture,
ranking is a held-out confirmation split, and audit is untouched until after
the architecture is fixed.  CatBoost depth is selected on the same selection
split from a predeclared bounded grid.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier

from torchboost.adaptive.progressive import RollingBoostClassifier, RollingBoostConfig


def dataset(seed):
    x, y = fetch_covtype(return_X_y=True)
    mask = np.isin(y, (1, 2))
    x = np.asarray(x[mask], dtype="float32")
    y = (np.asarray(y[mask]) == 2).astype(int)
    ntrain, ncontrol, nselection, nranking, naudit = 80000, 10000, 10000, 10000, 20000
    total = ntrain + ncontrol + nselection + nranking + naudit
    idx = np.random.default_rng(seed).permutation(len(x))[:total]
    x, y = x[idx], y[idx]
    a=ntrain; b=a+ncontrol; c=b+nselection; d=c+nranking
    return x, y, {
        "train": slice(0,a), "control": slice(a,b), "selection": slice(b,c),
        "ranking": slice(c,d), "audit": slice(d,total)
    }


def rolling_cfg(seed, name):
    common = dict(
        batch_size=512, learning_rate=0.012, new_tree_shrinkage=0.35,
        old_tree_lr_decay=0.65, weight_decay=1e-5, cart_strength=7.0,
        leaf_l2=1e-5, depth_shrinkage=2e-4, row_subsample=0.85,
        feature_subsample=0.85, cart_value_updates=6, patience_stages=100,
        min_improvement=-1e6, active_window=4, joint_updates=4, joint_every=4,
        random_state=seed,
    )
    if name == "r16d4":
        return RollingBoostConfig(n_trees=16, depth=4, stage_updates=16, **common)
    if name == "r32d4":
        return RollingBoostConfig(n_trees=32, depth=4, stage_updates=16, **common)
    if name == "r24d5":
        return RollingBoostConfig(n_trees=24, depth=5, stage_updates=16, **common)
    raise ValueError(name)


def run(seed=59):
    x,y,s=dataset(seed)
    candidates=[]
    models={}
    for name in ("r16d4","r32d4","r24d5"):
        start=time.time()
        m=RollingBoostClassifier(rolling_cfg(seed,name)).fit(
            x[s["train"]], y[s["train"]],
            eval_set=(x[s["control"]],y[s["control"]]),
        )
        row={
            "name":name,
            "selection":float(log_loss(y[s["selection"]],m.predict_proba(x[s["selection"]]))),
            "ranking":float(log_loss(y[s["ranking"]],m.predict_proba(x[s["ranking"]]))),
            "trees":int(m.n_estimators_),
            "seconds":time.time()-start,
        }
        candidates.append(row);models[name]=m
    chosen=min(candidates,key=lambda z:z["selection"])
    tb=models[chosen["name"]]
    tb_audit=float(log_loss(y[s["audit"]],tb.predict_proba(x[s["audit"]])))

    cb_rows=[];cb_models={}
    for depth in (6,8,10):
        start=time.time()
        m=CatBoostClassifier(
            iterations=512, depth=depth, learning_rate=.05, l2_leaf_reg=20,
            verbose=False, random_seed=seed, thread_count=1
        ).fit(x[s["train"]],y[s["train"]])
        row={
            "depth":depth,
            "selection":float(log_loss(y[s["selection"]],m.predict_proba(x[s["selection"]]))),
            "ranking":float(log_loss(y[s["ranking"]],m.predict_proba(x[s["ranking"]]))),
            "seconds":time.time()-start,
        }
        cb_rows.append(row);cb_models[depth]=m
    cb_chosen=min(cb_rows,key=lambda z:z["selection"])
    cb=cb_models[cb_chosen["depth"]]
    cb_audit=float(log_loss(y[s["audit"]],cb.predict_proba(x[s["audit"]])))

    return {
        "dataset":"sklearn Covertype; classes 1 vs 2",
        "seed":seed,
        "rows":{"train":80000,"control":10000,"selection":10000,"ranking":10000,"audit":20000},
        "torchboost_candidates":candidates,
        "torchboost_chosen":chosen,
        "torchboost_audit":tb_audit,
        "catboost_candidates":cb_rows,
        "catboost_chosen":cb_chosen,
        "catboost_audit":cb_audit,
        "relative_audit_gain_torchboost_vs_catboost":float(1-tb_audit/cb_audit),
    }


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=59);p.add_argument("--out")
    a=p.parse_args();r=run(a.seed);text=json.dumps(r,indent=2,sort_keys=True)
    if a.out:
        from pathlib import Path
        Path(a.out).write_text(text)
    print(text)
