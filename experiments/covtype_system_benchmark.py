"""Covertype system-level benchmark with separated selection/ranking/audit.

The earlier larger-data probe deliberately reused the synthetic-task single-tree
configuration and exposed a large gap to CatBoost. This follow-up asks whether
TorchBoost's ensemble families recover that gap without touching audit.

Selection is used inside each candidate (checkpoint/prefix or CatBoost early
stopping). A separate ranking split chooses the final TorchBoost variant and
CatBoost depth. Audit is evaluated once per chosen family winner.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from catboost import CatBoostClassifier
from sklearn.datasets import fetch_covtype
from sklearn.metrics import log_loss

from experiments.deep_physics_power_tree import config as deep_config
from torchboost.adaptive.progressive import ProgressiveConfig, ProgressiveTreeClassifier
from torchboost.adaptive.unified_progressive import UnifiedProgressiveClassifier


def data(seed, ntrain):
    x, y = fetch_covtype(return_X_y=True)
    mask = np.isin(y, (1, 2))
    x = np.asarray(x[mask], dtype="float32")
    y = (np.asarray(y[mask]) == 2).astype(int)
    counts = {
        "train": ntrain,
        "control": 10000,
        "selection": 10000,
        "ranking": 10000,
        "audit": 20000,
    }
    needed = sum(counts.values())
    if len(x) < needed:
        raise ValueError(f"need {needed} binary rows, found {len(x)}")
    idx = np.random.default_rng(seed).permutation(len(x))[:needed]
    x, y = x[idx], y[idx]
    out = {}
    start = 0
    for name, count in counts.items():
        out[name] = (x[start:start + count], y[start:start + count])
        start += count
    return out


def progressive_config(seed, trees, depth, stage_updates):
    return ProgressiveConfig(
        n_trees=trees,
        depth=depth,
        stage_updates=stage_updates,
        batch_size=768,
        learning_rate=0.010,
        new_tree_shrinkage=0.30,
        old_tree_lr_decay=0.65,
        weight_decay=1e-5,
        cart_strength=7.0,
        leaf_l2=1e-6,
        depth_shrinkage=2e-4,
        readout="residual",
        learn_tree_rates=True,
        tree_rate_l2=1e-5,
        row_subsample=0.85,
        feature_subsample=0.85,
        cart_value_updates=8,
        patience_stages=8,
        min_improvement=1e-5,
        random_state=seed,
    )


def score(model, split):
    return float(log_loss(split[1], model.predict_proba(split[0])))


def run(seed=53, ntrain=80000):
    d = data(seed, ntrain)
    candidates = []

    start = time.time()
    single_cfg = deep_config("none", seed, 512)
    single = UnifiedProgressiveClassifier(single_cfg).fit(
        d["train"][0],
        d["train"][1],
        control_set=d["control"],
        eval_set=d["selection"],
    )
    candidates.append({
        "name": "single_power_tree",
        "model": single,
        "selection": score(single, d["selection"]),
        "ranking": score(single, d["ranking"]),
        "complexity": {"trees": 1, "best_step": int(single.trainer_.best_epoch)},
        "seconds": time.time() - start,
    })

    for name, trees, depth, updates in (
        ("progressive_24x5", 24, 5, 32),
        ("progressive_48x4", 48, 4, 24),
    ):
        start = time.time()
        model = ProgressiveTreeClassifier(
            progressive_config(seed + trees + depth, trees, depth, updates)
        ).fit(
            d["train"][0],
            d["train"][1],
            eval_set=d["selection"],
        )
        candidates.append({
            "name": name,
            "model": model,
            "selection": score(model, d["selection"]),
            "ranking": score(model, d["ranking"]),
            "complexity": {
                "trees": int(model.n_estimators_),
                "configured_trees": trees,
                "depth": depth,
                "stage_updates": updates,
            },
            "seconds": time.time() - start,
        })

    tb_winner = min(candidates, key=lambda row: row["ranking"])
    tb_audit = score(tb_winner["model"], d["audit"])

    cat_candidates = []
    for depth in (6, 8, 10):
        start = time.time()
        model = CatBoostClassifier(
            iterations=1024,
            depth=depth,
            learning_rate=0.05,
            l2_leaf_reg=20,
            loss_function="Logloss",
            eval_metric="Logloss",
            verbose=False,
            random_seed=seed,
            thread_count=1,
            od_type="Iter",
            od_wait=80,
            use_best_model=True,
        ).fit(
            d["train"][0],
            d["train"][1],
            eval_set=d["selection"],
        )
        cat_candidates.append({
            "depth": depth,
            "model": model,
            "selection": score(model, d["selection"]),
            "ranking": score(model, d["ranking"]),
            "trees": int(model.tree_count_),
            "seconds": time.time() - start,
        })
    cat_winner = min(cat_candidates, key=lambda row: row["ranking"])
    cat_audit = score(cat_winner["model"], d["audit"])

    def strip(rows):
        return [{k: v for k, v in row.items() if k != "model"} for row in rows]

    return {
        "dataset": "sklearn Covertype; cover types 1 vs 2",
        "seed": seed,
        "rows": {
            "train": ntrain,
            "control": 10000,
            "selection": 10000,
            "ranking": 10000,
            "audit": 20000,
        },
        "torchboost_candidates": strip(candidates),
        "torchboost_winner": {
            **{k: v for k, v in tb_winner.items() if k != "model"},
            "audit": tb_audit,
        },
        "catboost_candidates": strip(cat_candidates),
        "catboost_winner": {
            **{k: v for k, v in cat_winner.items() if k != "model"},
            "audit": cat_audit,
        },
        "relative_audit_gain_torchboost_vs_catboost": float(1 - tb_audit / cat_audit),
        "protocol": (
            "selection controls within-candidate checkpoints; ranking chooses "
            "the family candidate; audit is touched once after ranking."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--train", type=int, default=80000)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = run(args.seed, args.train)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(text)
    print(text)
