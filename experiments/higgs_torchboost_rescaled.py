"""TorchBoost-only correction to the canonical HIGGS scaling study.

This rerun keeps the canonical 21-feature splits and seed construction from
higgs_canonical_scaling.py, but fixes a training-budget confound in the original
TorchBoost arm. The old arm used 40 depth-6 trees and 24 minibatch updates per
stage at every n, so differentiable sample exposure collapsed as n increased.

The corrected schedule is fixed before results are opened:
- approximately two differentiable sample presentations per training row;
- batch size 2048;
- tree count scales as 40*sqrt(n/500k), rounded to a multiple of 8 and capped
  at 64 for the GitHub-hosted CPU runner;
- depth is 6 through 1M and 7 at 3M;
- CART-only value warmup remains approximately one third of stage updates.

CatBoost and MLP are intentionally not rerun. Their canonical run-2 results are
the frozen comparators. This script opens the same ranking/audit slices only
after the corrected TorchBoost configuration is determined from n alone.
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
from torchboost.adaptive.progressive import ProgressiveConfig, ProgressiveTreeClassifier

REFERENCE_ROWS = 500_000
REFERENCE_TREES = 40
MAX_TREES = 64
TARGET_SAMPLE_PASSES = 2.0
BATCH_SIZE = 2048


def corrected_schedule(ntrain: int) -> dict:
    if ntrain not in (500_000, 1_000_000, 3_000_000):
        raise ValueError("corrected canonical rerun is prespecified for 500k, 1M, and 3M")
    raw_trees = REFERENCE_TREES * math.sqrt(ntrain / REFERENCE_ROWS)
    trees = min(MAX_TREES, max(REFERENCE_TREES, 8 * round(raw_trees / 8)))
    depth = 7 if ntrain >= 3_000_000 else 6
    stage_updates = max(1, math.ceil(TARGET_SAMPLE_PASSES * ntrain / (trees * BATCH_SIZE)))
    cart_value_updates = max(1, stage_updates // 3)
    presentations = trees * stage_updates * BATCH_SIZE
    return {
        "n_trees": trees,
        "depth": depth,
        "stage_updates": stage_updates,
        "batch_size": BATCH_SIZE,
        "cart_value_updates": cart_value_updates,
        "target_sample_passes": TARGET_SAMPLE_PASSES,
        "planned_sample_presentations": presentations,
        "planned_presentations_per_row": presentations / ntrain,
        "capacity_rule": "min(64, round_to_8(40*sqrt(n/500000)))",
        "depth_rule": "6 for <=1M; 7 for 3M",
    }


def fit_corrected_torchboost(splits, seed: int, ntrain: int) -> dict:
    schedule = corrected_schedule(ntrain)
    started = time.perf_counter()
    cfg = ProgressiveConfig(
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
        learn_tree_rates=True,
        tree_rate_l2=1e-5,
        tree_count_pressure=0.,
        row_subsample=.85,
        feature_subsample=.90,
        cart_value_updates=schedule["cart_value_updates"],
        patience_stages=10,
        min_improvement=1e-6,
        random_state=seed + 46,
    )
    model = ProgressiveTreeClassifier(cfg).fit(
        splits["train"][0], splits["train"][1], eval_set=splits["selection"]
    )
    rank = metrics(splits["ranking"][1], model.predict_proba(splits["ranking"][0])[:, 1])
    audit = metrics(splits["audit"][1], model.predict_proba(splits["audit"][0])[:, 1])
    parameters = sum(p.numel() for p in model.model_.parameters())
    return {
        "ranking": rank,
        "audit": audit,
        "retained_trees": int(model.n_estimators_),
        "parameters": int(parameters),
        "schedule": schedule,
        "seconds": time.perf_counter() - started,
    }


def run(csv_gz, cache, ntrain: int, seed=509, out=None):
    torch.set_num_threads(4)
    x_path, y_path, source = materialize(Path(csv_gz), Path(cache))
    x, y = arrays(x_path, y_path)
    splits = fixed_splits(x, y, ntrain)
    family_seed = seed + ntrain % 10007
    result = {
        "status": "running",
        "seed": seed,
        "family_seed": family_seed,
        "ntrain": ntrain,
        "source": source,
        "features": f"first {LOW_FEATURES} low-level detector features only",
        "canonical_split_contract": "identical to higgs_canonical_scaling.py",
        "comparison_contract": (
            "TorchBoost-only rerun; corrected schedule fixed from n before ranking/audit; "
            "compare to frozen CatBoost/MLP metrics from canonical run 36261511922"
        ),
        "torchboost": None,
    }
    path = Path(out) if out else None
    if path:
        path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    result["torchboost"] = fit_corrected_torchboost(splits, family_seed, ntrain)
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
    parser.add_argument("--ntrain", type=int, required=True, choices=(500_000, 1_000_000, 3_000_000))
    parser.add_argument("--seed", type=int, default=509)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(args.csv_gz, args.cache, args.ntrain, args.seed, args.out)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
