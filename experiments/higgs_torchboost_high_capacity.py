"""High-capacity TorchBoost scaling on canonical low-level HIGGS.

Unlike the constant-exposure correction, this arm lets both structural capacity
and differentiable optimization grow with the dataset. Each new tree begins as
a supervised residual CART proposal, then soft/oblique routing is optimized;
periodically, a recent tree window is jointly refined while the older prefix is
cached. The schedule is fixed from n before ranking/audit are opened.

CatBoost and MLP comparators are frozen from canonical run 36261511922.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from experiments.higgs_canonical_scaling import (
    LOW_FEATURES,
    arrays,
    fixed_splits,
    materialize,
    metrics,
)
from torchboost.adaptive.progressive import (
    RollingBoostClassifier,
    RollingBoostConfig,
    _contribution_count_stats,
    _realized_tree_contributions,
)

BATCH_SIZE = 2048
TARGET_PRIMARY_PASSES = 8.0
JOINT_EVERY = 4
ACTIVE_WINDOW = 6

SCALE_CAPACITY = {
    500_000: {"n_trees": 64, "depth": 6, "cart_sample_size": 250_000},
    1_000_000: {"n_trees": 96, "depth": 7, "cart_sample_size": 350_000},
    3_000_000: {"n_trees": 160, "depth": 8, "cart_sample_size": 500_000},
}

FROZEN_CANONICAL = {
    500_000: {
        "catboost": {"nll": 0.5787195076509121, "auc": 0.7618200137388893},
        "mlp": {"nll": 0.5720548068178406, "auc": 0.7675706244919547},
        "torchboost": {"nll": 0.6068553517788886, "auc": 0.7278099880572689},
    },
    1_000_000: {
        "catboost": {"nll": 0.5728503370526202, "auc": 0.7685197931951162},
        "mlp": {"nll": 0.545414291615545, "auc": 0.7941763122055557},
        "torchboost": {"nll": 0.6029146397141, "auc": 0.732247384290339},
    },
    3_000_000: {
        "catboost": {"nll": 0.5687870785277621, "auc": 0.7727770302112822},
        "mlp": {"nll": 0.5042313598634272, "auc": 0.8298690077191215},
        "torchboost": {"nll": 0.6028318476953671, "auc": 0.7317371353391786},
    },
}


def high_capacity_schedule(ntrain: int) -> dict:
    if ntrain not in SCALE_CAPACITY:
        raise ValueError("high-capacity schedule is fixed at 500k, 1M, and 3M")
    capacity = SCALE_CAPACITY[ntrain]
    trees = capacity["n_trees"]
    stage_updates = math.ceil(
        TARGET_PRIMARY_PASSES * ntrain / (trees * BATCH_SIZE)
    )
    joint_updates = math.ceil(stage_updates / 4)
    joint_events = trees // JOINT_EVERY
    primary = trees * stage_updates * BATCH_SIZE
    joint = joint_events * joint_updates * BATCH_SIZE
    return {
        **capacity,
        "stage_updates": stage_updates,
        "batch_size": BATCH_SIZE,
        "cart_value_updates": max(4, stage_updates // 4),
        "active_window": ACTIVE_WINDOW,
        "joint_every": JOINT_EVERY,
        "joint_updates": joint_updates,
        "target_primary_passes": TARGET_PRIMARY_PASSES,
        "planned_primary_presentations": primary,
        "planned_joint_presentations": joint,
        "planned_primary_passes": primary / ntrain,
        "planned_total_optimizer_passes": (primary + joint) / ntrain,
        "capacity_rule": "64xd6 @500k; 96xd7 @1M; 160xd8 @3M",
        "cart_rule": "250k/350k/500k sampled proposal rows per tree",
    }


def _effective_count(model, x):
    tx = model.preprocessor_.transform_x(x)
    with torch.no_grad():
        contribution = _realized_tree_contributions(model.model_, tx)
        effective, entropy = _contribution_count_stats(contribution)
    return {"participation": float(effective), "entropy": float(entropy)}


def fit_high_capacity(splits, seed: int, ntrain: int) -> dict:
    schedule = high_capacity_schedule(ntrain)
    cfg = RollingBoostConfig(
        n_trees=schedule["n_trees"],
        depth=schedule["depth"],
        stage_updates=schedule["stage_updates"],
        batch_size=schedule["batch_size"],
        learning_rate=.01,
        new_tree_shrinkage=.30,
        old_tree_lr_decay=.80,
        weight_decay=1e-5,
        cart_strength=7.,
        leaf_l2=1e-6,
        depth_shrinkage=2e-4,
        readout="residual",
        row_subsample=.85,
        feature_subsample=.90,
        cart_value_updates=schedule["cart_value_updates"],
        patience_stages=32,
        min_improvement=1e-6,
        active_window=schedule["active_window"],
        joint_updates=schedule["joint_updates"],
        joint_every=schedule["joint_every"],
        cart_sample_size=schedule["cart_sample_size"],
        verbose=True,
        random_state=seed + 46,
    )
    started = time.perf_counter()
    model = RollingBoostClassifier(cfg).fit(
        splits["train"][0], splits["train"][1], eval_set=splits["selection"]
    )
    ranking = metrics(
        splits["ranking"][1], model.predict_proba(splits["ranking"][0])[:, 1]
    )
    audit = metrics(
        splits["audit"][1], model.predict_proba(splits["audit"][0])[:, 1]
    )
    retained = int(model.n_estimators_)
    parameters = sum(p.numel() for p in model.model_.parameters())
    primary = retained * schedule["stage_updates"] * BATCH_SIZE
    joint_events = retained // schedule["joint_every"]
    joint = joint_events * schedule["joint_updates"] * BATCH_SIZE
    return {
        "ranking": ranking,
        "audit": audit,
        "retained_trees": retained,
        "parameters": int(parameters),
        "effective_tree_count": _effective_count(model, splits["ranking"][0]),
        "best_selection_nll": float(model.best_score_),
        "history_tail": model.history_[-8:],
        "schedule": schedule,
        "actual_optimizer_presentations": primary + joint,
        "actual_optimizer_passes": (primary + joint) / ntrain,
        "seconds": time.perf_counter() - started,
    }


def _comparison(ntrain: int, result: dict) -> dict:
    out = {}
    for family, reference in FROZEN_CANONICAL[ntrain].items():
        out[family] = {
            "reference": reference,
            "delta_nll": result["audit"]["nll"] - reference["nll"],
            "delta_auc": result["audit"]["auc"] - reference["auc"],
        }
    return out


def run(csv_gz, cache, ntrain: int, seed=509, out=None):
    torch.set_num_threads(4)
    schedule = high_capacity_schedule(ntrain)
    result = {
        "status": "running",
        "seed": seed,
        "ntrain": ntrain,
        "features": f"first {LOW_FEATURES} low-level detector features only",
        "canonical_split_contract": "identical to higgs_canonical_scaling.py",
        "comparison_contract": (
            "high-capacity TorchBoost-only rerun; schedule fixed from n before "
            "ranking/audit; frozen comparators from canonical run 36261511922"
        ),
        "schedule": schedule,
        "frozen_canonical_run_id": 36261511922,
        "frozen_canonical": FROZEN_CANONICAL[ntrain],
        "source": None,
        "torchboost": None,
        "comparison": None,
    }
    path = Path(out) if out else None
    if path:
        path.write_text(json.dumps(result, indent=2, sort_keys=True))
    x_path, y_path, source = materialize(Path(csv_gz), Path(cache))
    result["source"] = source
    x, y = arrays(x_path, y_path)
    splits = fixed_splits(x, y, ntrain)
    family_seed = seed + ntrain % 10007
    result["torchboost"] = fit_high_capacity(splits, family_seed, ntrain)
    result["comparison"] = _comparison(ntrain, result["torchboost"])
    result["status"] = "completed"
    if path:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        tmp.replace(path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-gz", required=True)
    parser.add_argument("--cache", default="/tmp/higgs-cache")
    parser.add_argument(
        "--ntrain",
        type=int,
        required=True,
        choices=(500_000, 1_000_000, 3_000_000),
    )
    parser.add_argument("--seed", type=int, default=509)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    answer = run(args.csv_gz, args.cache, args.ntrain, args.seed, args.out)
    print(json.dumps(answer, indent=2, sort_keys=True, allow_nan=False))
