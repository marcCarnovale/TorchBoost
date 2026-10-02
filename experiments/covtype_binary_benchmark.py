"""Larger public tabular benchmark: Covertype classes 1 versus 2.

TorchBoost uses the fixed strong single-power-tree configuration from the
synthetic ratchet; its hyperparameters are not tuned on this dataset. CatBoost
chooses depth on the selection split only. Audit remains untouched.
"""
from __future__ import annotations

import argparse
import json
import time
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier

from experiments.deep_physics_power_tree import config
from torchboost.adaptive.unified_progressive import UnifiedProgressiveClassifier


def run(seed=53, ntrain=80000, ncontrol=10000, nselection=10000, naudit=20000):
    x, y = fetch_covtype(return_X_y=True)
    mask = np.isin(y, (1, 2))
    x = np.asarray(x[mask], dtype="float32")
    y = (np.asarray(y[mask]) == 2).astype(int)
    needed = ntrain + ncontrol + nselection + naudit
    if len(x) < needed:
        raise ValueError(f"need {needed} binary rows, found {len(x)}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))[:needed]
    x, y = x[idx], y[idx]
    a = ntrain
    b = a + ncontrol
    c = b + nselection
    tr = slice(0, a)
    control = slice(a, b)
    selection = slice(b, c)
    audit = slice(c, needed)

    t = time.time()
    tb = UnifiedProgressiveClassifier(config("none", seed, 512)).fit(
        x[tr],
        y[tr],
        control_set=(x[control], y[control]),
        eval_set=(x[selection], y[selection]),
    )
    tb_selection = log_loss(y[selection], tb.predict_proba(x[selection]))
    tb_audit = log_loss(y[audit], tb.predict_proba(x[audit]))
    tb_seconds = time.time() - t

    candidates = []
    for depth in (6, 8, 10):
        start = time.time()
        model = CatBoostClassifier(
            iterations=512,
            depth=depth,
            learning_rate=0.05,
            l2_leaf_reg=20,
            verbose=False,
            random_seed=seed,
            thread_count=1,
        ).fit(x[tr], y[tr])
        candidates.append(
            {
                "depth": depth,
                "selection": log_loss(y[selection], model.predict_proba(x[selection])),
                "model": model,
                "seconds": time.time() - start,
            }
        )
    chosen = min(candidates, key=lambda row: row["selection"])
    cat_audit = log_loss(y[audit], chosen["model"].predict_proba(x[audit]))
    return {
        "dataset": "sklearn Covertype; cover types 1 vs 2",
        "seed": seed,
        "rows": {
            "train": ntrain,
            "control": ncontrol,
            "selection": nselection,
            "audit": naudit,
        },
        "features": int(x.shape[1]),
        "torchboost": {
            "selection": float(tb_selection),
            "audit": float(tb_audit),
            "best_step": int(tb.trainer_.best_epoch),
            "seconds": tb_seconds,
        },
        "catboost": {
            "chosen_depth": int(chosen["depth"]),
            "selection": float(chosen["selection"]),
            "audit": float(cat_audit),
            "seconds": float(chosen["seconds"]),
            "selection_candidates": [
                {"depth": int(row["depth"]), "selection": float(row["selection"])}
                for row in candidates
            ],
        },
        "relative_audit_gain_torchboost_vs_catboost": float(1 - tb_audit / cat_audit),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--train", type=int, default=80000)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = run(seed=args.seed, ntrain=args.train)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(text)
    print(text)
