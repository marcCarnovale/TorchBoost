"""Run a fixed binary smoke benchmark with four disjoint data roles.

Example: python -m benchmarks.run_binary --seeds 0 1 2 --output results.json
These small experiments are an engineering smoke test, not a tuned leaderboard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
import xgboost
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from torchboost import StagewiseBinaryClassifier


def config_for_json(config):
    """Encode NaN-valued configuration sentinels explicitly, never metric failures."""
    if isinstance(config, dict):
        return {key: config_for_json(value) for key, value in config.items()}
    if isinstance(config, (list, tuple)):
        return [config_for_json(value) for value in config]
    if isinstance(config, float) and not np.isfinite(config):
        return str(config)
    return config


def calibration_error(y, p, bins=10):
    """Equal-width binary calibration error; binning is part of the definition."""
    indices = np.minimum((p * bins).astype(int), bins-1)
    return float(sum(np.mean(indices == b) * abs(np.mean(p[indices == b])-np.mean(y[indices == b]))
                     for b in range(bins) if np.any(indices == b)))


def split_roles(y, seed):
    """50/15/15/20 percent train/controller/selection/test, approximately."""
    train, rest = train_test_split(np.arange(len(y)), train_size=0.5, stratify=y, random_state=seed)
    control, rest = train_test_split(rest, train_size=0.3, stratify=y[rest], random_state=seed+1000)
    selection, test = train_test_split(rest, train_size=3/7, stratify=y[rest], random_state=seed+2000)
    return {"train": train, "control": control, "selection": selection, "test": test}


def summarize(records):
    summary = []
    for dataset, model in sorted({(r["dataset"], r["model"]) for r in records}):
        rows = [r for r in records if (r["dataset"], r["model"]) == (dataset, model)]
        result = {"dataset": dataset, "model": model, "n_seeds": len(rows)}
        for metric in ("auc", "log_loss", "accuracy", "balanced_accuracy", "brier", "ece", "fit_seconds", "predict_seconds"):
            values = [r[metric] for r in rows]
            result[metric] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else None}
        summary.append(result)
    return summary


def run(seeds, quick=False):
    torch.set_num_threads(1)
    datasets = {
        "breast_cancer": load_breast_cancer(return_X_y=True),
        "synthetic_imbalanced": make_classification(n_samples=240 if quick else 1200,
            n_features=12, n_informative=8, n_redundant=2, class_sep=1.0,
            weights=[0.65, 0.35], flip_y=0.03, random_state=781),
    }
    records, splits = [], []
    for dataset, (x, y) in datasets.items():
        fingerprint = hashlib.sha256(np.ascontiguousarray(x).tobytes()+np.ascontiguousarray(y).tobytes()).hexdigest()
        for seed in seeds:
            roles = split_roles(y, seed)
            splits.append({"dataset": dataset, "seed": seed, "sha256_data": fingerprint,
                           "roles": {k: {"count": len(v), "sha256_indices": hashlib.sha256(v.astype("<i8").tobytes()).hexdigest()} for k, v in roles.items()}})
            train, control, selection, test = (roles[k] for k in ("train", "control", "selection", "test"))
            common = {
                "n_estimators": 6 if quick else 32, "max_depth": 3,
                "epochs_per_stage": 3 if quick else 12, "lr": 0.15, "optimizer_lr": 0.02,
                "batch_size": 512, "patience": 10, "random_state": seed,
            }
            models = {
                "torchboost_newton_cart": StagewiseBinaryClassifier(**common),
                "torchboost_newton_random": StagewiseBinaryClassifier(**{**common, "init": "random"}),
                "torchboost_first_order_cart": StagewiseBinaryClassifier(**{**common, "curvature": "first_order"}),
                "torchboost_capacitor_cart": StagewiseBinaryClassifier(**{**common, "controller": {}}),
                "xgboost_hist": XGBClassifier(n_estimators=common["n_estimators"], max_depth=3,
                    learning_rate=0.15, n_jobs=1, random_state=seed, tree_method="hist",
                    objective="binary:logistic", eval_metric="logloss", early_stopping_rounds=10),
            }
            for name, model in models.items():
                start = time.perf_counter()
                if isinstance(model, StagewiseBinaryClassifier):
                    model.fit(x[train], y[train], eval_set=(x[selection], y[selection]),
                              control_set=(x[control], y[control]) if model.controller is not None else None)
                else:
                    model.fit(x[train], y[train], eval_set=[(x[selection], y[selection])], verbose=False)
                elapsed = time.perf_counter()-start
                start = time.perf_counter()
                p = model.predict_proba(x[test])[:, 1]
                predict_elapsed = time.perf_counter()-start
                row = {"dataset": dataset, "seed": seed, "model": name,
                       "auc": float(roc_auc_score(y[test], p)),
                       "log_loss": float(log_loss(y[test], p, labels=[0, 1])),
                       "accuracy": float(accuracy_score(y[test], p >= 0.5)),
                       "balanced_accuracy": float(balanced_accuracy_score(y[test], p >= 0.5)),
                       "brier": float(brier_score_loss(y[test], p)),
                       "ece": calibration_error(y[test], p),
                       "fit_seconds": elapsed, "predict_seconds": predict_elapsed,
                       "peak_memory_bytes": None, "config": config_for_json(model.get_params())}
                if isinstance(model, StagewiseBinaryClassifier):
                    row.update(selected_stages=model.n_estimators_,
                               parameters=sum(v.numel() for v in model.model_.parameters()),
                               tensor_state_bytes=sum(v.numel()*v.element_size() for v in model.model_.state_dict().values()),
                               soft_hard_mean_absolute_probability_gap=float(np.abs(p-model.predict_proba(x[test], hard=True)[:, 1]).mean()),
                               total_corrective_heat=sum(v["heating"] for v in model.control_history_))
                else:
                    row["selected_stages"] = int(model.best_iteration)+1
                records.append(row)
                print(f'{dataset} seed={seed} {name}: AUC={row["auc"]:.4f} NLL={row["log_loss"]:.4f} fit={elapsed:.2f}s', flush=True)
    return {"schema": "torchboost.binary-smoke.v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "numpy": np.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                            "platform": platform.platform(), "device": "cpu", "torch_threads": 1},
            "quick": quick, "seeds": list(seeds), "splits": splits, "records": records,
            "summary": summarize(records), "limitations": [
                "Two small datasets; no hyperparameter search or matched wall-clock tuning budget.",
                "CART is an explicit hybrid warm start; random initialization is separately reported.",
                "Only the capacitor variant uses the controller split; additional label access is disclosed.",
                "Peak memory is not measured. Tensor state bytes are not peak training memory.",
                "Seed standard deviations are descriptive; this is not a general superiority claim.",
                "CPU timing includes metrics collection for TorchBoost, not a GPU throughput benchmark.",
                "Test results must not be used to retune this recorded experiment."]}


def deduplicate_configs(result):
    """Preserve full settings with per-record seed, without repeating templates."""
    templates = {}
    for row in result["records"]:
        config = row.pop("config")
        config.pop("random_state", None)
        if row["model"] in templates and templates[row["model"]] != config:
            raise ValueError("configuration changed between recorded runs")
        templates[row["model"]] = config
        row["config_ref"] = row["model"]
    result["configuration_templates"] = templates
    result["configuration_resolution"] = "Resolve config_ref in configuration_templates and set random_state to the record seed."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/binary_smoke.json"))
    args = parser.parse_args()
    result = deduplicate_configs(run(args.seeds, args.quick))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
