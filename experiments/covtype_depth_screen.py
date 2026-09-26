"""Development-only depth/count screen against a strengthened CatBoost search.

The full run uses canonical Covertype cover classes 1/2 with the existing five
partitions. No audit prediction is made. Counts, depths, budgets, time, and
versions are retained, including losses. This is not an equal-compute claim.
The unchanged locked CatBoost ratchet runs separately in CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import catboost
from catboost import CatBoostClassifier
import numpy as np
from sklearn.datasets import make_classification
import torch

from experiments.covtype_system_benchmark import data, progressive_config, score
from torchboost.adaptive.progressive import ProgressiveTreeClassifier


def run(seed=59, ntrain=80000, out="covtype-depth-screen.json", smoke=False):
    torch.set_num_threads(1)
    if smoke:
        x, y = make_classification(n_samples=2400, n_features=12, random_state=seed)
        x = np.asarray(x, dtype="float32")
        splits = {
            "train": (x[:1200], y[:1200]), "control": (x[1200:1500], y[1200:1500]),
            "selection": (x[1500:1800], y[1500:1800]), "ranking": (x[1800:2100], y[1800:2100]),
            "audit": (x[2100:], y[2100:]),
        }
        grid = ((2, 2), (2, 3))
        cat_depths, cat_iterations, stage_updates = (3, 4), 16, 4
    else:
        splits = data(seed, ntrain)
        grid = tuple((trees, depth) for trees in (12, 24, 32) for depth in (5, 6, 7))
        cat_depths, cat_iterations, stage_updates = (6, 8, 10, 12), 2048, 32
    result = {
        "status": "running", "development_only": True, "audit_evaluated": False,
        "dataset": "synthetic_smoke" if smoke else "canonical sklearn Covertype; classes 1 vs 2",
        "seed": seed, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "catboost": catboost.__version__},
        "rows": {name: len(split[0]) for name, split in splits.items()},
        "torchboost_candidates": [], "catboost_candidates": [],
        "protocol": "selection chooses checkpoint/prefix; ranking compares candidates; audit is not scored",
        "budget_note": "fixed updates per tree, not equal total updates or equal wall-clock compute",
    }
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        temporary.replace(output)

    save()
    try:
        for trees, depth in grid:
            started = time.perf_counter()
            cfg = progressive_config(seed, trees, depth, stage_updates)
            model = ProgressiveTreeClassifier(cfg).fit(*splits["train"], eval_set=splits["selection"])
            row = {
                "trees": trees, "depth": depth, "stage_updates": stage_updates,
                "configured_total_updates": trees * stage_updates,
                "retained_trees": int(model.n_estimators_),
                "selection_nll": score(model, splits["selection"]),
                "ranking_nll": score(model, splits["ranking"]),
                "seconds": time.perf_counter() - started,
            }
            result["torchboost_candidates"].append(row)
            save()
            print(json.dumps({"family": "torchboost", **row}), flush=True)
        for depth in cat_depths:
            started = time.perf_counter()
            model = CatBoostClassifier(
                iterations=cat_iterations, depth=depth, learning_rate=0.05,
                l2_leaf_reg=20, loss_function="Logloss", eval_metric="Logloss",
                verbose=False, random_seed=seed, thread_count=1,
                od_type="Iter", od_wait=80, use_best_model=True,
                allow_writing_files=False,
            ).fit(*splits["train"], eval_set=splits["selection"])
            row = {
                "depth": depth, "max_iterations": cat_iterations, "retained_trees": int(model.tree_count_),
                "selection_nll": score(model, splits["selection"]),
                "ranking_nll": score(model, splits["ranking"]),
                "seconds": time.perf_counter() - started,
            }
            result["catboost_candidates"].append(row)
            save()
            print(json.dumps({"family": "catboost", **row}), flush=True)
        result["torchboost_ranking_choice"] = min(result["torchboost_candidates"], key=lambda r: r["ranking_nll"])
        result["catboost_ranking_choice"] = min(result["catboost_candidates"], key=lambda r: r["ranking_nll"])
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        save()
        raise
    save()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--train", type=int, default=80000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.seed, args.train, args.out, args.smoke)
