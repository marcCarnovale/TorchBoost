"""HIGGS representation-learning benchmark for TorchBoost.

Development target: OpenML HIGGS 100k proxy (data_id=23512). The canonical
11M-row UCI confirmation is a separate future run; this script never describes
100k results as canonical HIGGS performance.

The historically interesting condition uses only the first 21 low-level
kinematic features. Deep networks were introduced on this benchmark partly to
learn nonlinear structure that physicists otherwise encoded in the final seven
high-level features. We compare a conventional MLP, CatBoost, and a
TorchBoost progressive differentiable forest under train/selection/ranking/audit
separation. Audit is evaluated only after each family candidate is fixed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.datasets import fetch_openml, make_classification
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from torchboost.adaptive.progressive import ProgressiveConfig, ProgressiveTreeClassifier


@dataclass(frozen=True)
class Split:
    x: np.ndarray
    y: np.ndarray


def _finite_impute(train: np.ndarray, *others: np.ndarray):
    median = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=0)
    median = np.where(np.isfinite(median), median, 0.0).astype("float32")
    out = []
    for x in (train,) + others:
        z = np.asarray(x, dtype="float32").copy()
        bad = ~np.isfinite(z)
        if bad.any():
            z[bad] = np.take(median, np.nonzero(bad)[1])
        out.append(z)
    return out


def load_proxy(seed: int, smoke: bool = False):
    if smoke:
        x, y = make_classification(
            n_samples=5000, n_features=28, n_informative=18, n_redundant=6,
            class_sep=0.8, random_state=seed,
        )
        name = "synthetic-smoke"
    else:
        bunch = fetch_openml(data_id=23512, as_frame=False, parser="auto")
        x = np.asarray(bunch.data, dtype="float32")
        raw = np.asarray(bunch.target)
        classes = np.unique(raw)
        if len(classes) != 2:
            raise ValueError(f"HIGGS proxy must be binary, got {classes!r}")
        y = (raw == classes[-1]).astype("int64")
        name = "OpenML HIGGS proxy data_id=23512 (98,050 rows), not canonical 11M confirmation"
    if x.ndim != 2 or x.shape[1] < 28:
        raise ValueError(f"expected at least 28 HIGGS features, got {x.shape}")
    order = np.random.default_rng(seed + 1701).permutation(len(x))
    x, y = x[order], y[order]
    if smoke:
        counts = dict(train=3000, selection=600, ranking=600, audit=800)
    else:
        counts = dict(train=68050, selection=10000, ranking=10000, audit=10000)
    if sum(counts.values()) > len(x):
        raise ValueError(f"need {sum(counts.values())} rows, found {len(x)}")
    cuts, start = {}, 0
    arrays = []
    for role, n in counts.items():
        cuts[role] = (start, start + n)
        arrays.append(x[start:start+n])
        start += n
    arrays = _finite_impute(*arrays)
    splits = {}
    for (role, (a, b)), xx in zip(cuts.items(), arrays):
        splits[role] = Split(xx, y[a:b])
    digest = {
        role: hashlib.sha256(split.x.tobytes() + split.y.tobytes()).hexdigest()
        for role, split in splits.items()
    }
    return name, splits, digest


def subset(splits, feature_set: str):
    width = {"low": 21, "all": 28}[feature_set]
    return {k: Split(v.x[:, :width], v.y) for k, v in splits.items()}


def metrics_from_probability(y, p):
    p = np.asarray(p, dtype=float).reshape(-1)
    return {
        "nll": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)),
    }


def tb_candidates(seed, smoke=False):
    if smoke:
        return [(4, 3, 4), (6, 3, 4)]
    return [(24, 5, 24), (40, 6, 24), (56, 6, 20)]


def fit_torchboost(splits, seed, smoke=False):
    rows = []
    models = []
    for trees, depth, updates in tb_candidates(seed, smoke):
        started = time.perf_counter()
        cfg = ProgressiveConfig(
            n_trees=trees, depth=depth, stage_updates=updates,
            batch_size=512 if not smoke else 128,
            learning_rate=0.010,
            new_tree_shrinkage=0.30,
            old_tree_lr_decay=0.80,
            weight_decay=1e-5,
            cart_strength=7.0,
            leaf_l2=1e-6,
            depth_shrinkage=2e-4,
            readout="residual",
            learn_tree_rates=True,
            tree_rate_l2=1e-5,
            tree_count_pressure=0.0,
            row_subsample=0.85,
            feature_subsample=0.90,
            cart_value_updates=8,
            patience_stages=10,
            min_improvement=1e-6,
            random_state=seed + trees + depth,
        )
        model = ProgressiveTreeClassifier(cfg).fit(
            splits["train"].x, splits["train"].y,
            eval_set=(splits["selection"].x, splits["selection"].y),
        )
        p = model.predict_proba(splits["ranking"].x)[:, 1]
        row = {
            "trees": trees, "depth": depth, "stage_updates": updates,
            "retained_trees": int(model.n_estimators_),
            "ranking": metrics_from_probability(splits["ranking"].y, p),
            "seconds": time.perf_counter() - started,
        }
        rows.append(row)
        models.append(model)
    idx = min(range(len(rows)), key=lambda i: rows[i]["ranking"]["nll"])
    return rows, models[idx], idx


def fit_catboost(splits, seed, smoke=False):
    rows, models = [], []
    depths = (4, 6) if smoke else (6, 8, 10)
    iterations = 32 if smoke else 1536
    for depth in depths:
        started = time.perf_counter()
        model = CatBoostClassifier(
            iterations=iterations, depth=depth, learning_rate=0.05,
            l2_leaf_reg=20, loss_function="Logloss", eval_metric="Logloss",
            verbose=False, random_seed=seed, thread_count=4,
            od_type="Iter", od_wait=80, use_best_model=True,
            allow_writing_files=False,
        ).fit(
            splits["train"].x, splits["train"].y,
            eval_set=(splits["selection"].x, splits["selection"].y),
        )
        p = model.predict_proba(splits["ranking"].x)[:, 1]
        row = {
            "depth": depth, "retained_trees": int(model.tree_count_),
            "ranking": metrics_from_probability(splits["ranking"].y, p),
            "seconds": time.perf_counter() - started,
        }
        rows.append(row)
        models.append(model)
    idx = min(range(len(rows)), key=lambda i: rows[i]["ranking"]["nll"])
    return rows, models[idx], idx


class MLP(torch.nn.Module):
    def __init__(self, d, width, depth, dropout):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.extend([
                torch.nn.Linear(d if i == 0 else width, width),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ])
        layers.append(torch.nn.Linear(width, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def _mlp_probability(model, x, batch=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            z = torch.from_numpy(x[start:start+batch])
            out.append(torch.sigmoid(model(z)).cpu().numpy())
    return np.concatenate(out)


def fit_mlp(splits, seed, smoke=False):
    scaler = StandardScaler().fit(splits["train"].x)
    z = {k: Split(scaler.transform(v.x).astype("float32"), v.y) for k, v in splits.items()}
    grid = ((32, 2, 0.0),) if smoke else ((256, 4, 0.1), (300, 5, 0.1))
    rows, models, scalers = [], [], []
    epochs = 2 if smoke else 20
    batch = 256 if smoke else 2048
    for width, depth, dropout in grid:
        torch.manual_seed(seed + width + depth)
        model = MLP(z["train"].x.shape[1], width, depth, dropout)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(seed + 9001)
        best_sel, best_state, best_epoch = float("inf"), None, None
        started = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(z["train"].x), generator=generator)
            for start in range(0, len(order), batch):
                idx = order[start:start+batch].numpy()
                xb = torch.from_numpy(z["train"].x[idx])
                yb = torch.from_numpy(z["train"].y[idx].astype("float32"))
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
            sel = metrics_from_probability(
                z["selection"].y, _mlp_probability(model, z["selection"].x)
            )["nll"]
            if sel < best_sel:
                best_sel, best_epoch = sel, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        p = _mlp_probability(model, z["ranking"].x)
        rows.append({
            "width": width, "depth": depth, "dropout": dropout,
            "best_epoch": best_epoch,
            "ranking": metrics_from_probability(z["ranking"].y, p),
            "seconds": time.perf_counter() - started,
        })
        models.append(model)
        scalers.append(scaler)
    idx = min(range(len(rows)), key=lambda i: rows[i]["ranking"]["nll"])
    return rows, (models[idx], scalers[idx]), idx


def run(seed=313, smoke=False):
    torch.set_num_threads(4 if not smoke else 1)
    name, base_splits, hashes = load_proxy(seed, smoke)
    result = {
        "status": "running", "dataset": name, "seed": seed,
        "split_hashes": hashes, "audit_role": "unopened until family ranking completes",
        "conditions": {},
        "interpretation_contract": (
            "The 98,050-row OpenML proxy is architecture development only. The low-level condition "
            "tests representation learning; canonical claims require the 11M UCI dataset with "
            "its final 500k held out. Physics is not tested in this benchmark."
        ),
    }
    for feature_set in ("low", "all"):
        splits = subset(base_splits, feature_set)
        tb_rows, tb, tb_i = fit_torchboost(splits, seed, smoke)
        cat_rows, cat, cat_i = fit_catboost(splits, seed, smoke)
        mlp_rows, (mlp, scaler), mlp_i = fit_mlp(splits, seed, smoke)
        audit = {
            "torchboost": metrics_from_probability(
                splits["audit"].y, tb.predict_proba(splits["audit"].x)[:, 1]
            ),
            "catboost": metrics_from_probability(
                splits["audit"].y, cat.predict_proba(splits["audit"].x)[:, 1]
            ),
            "mlp": metrics_from_probability(
                splits["audit"].y,
                _mlp_probability(
                    mlp, scaler.transform(splits["audit"].x).astype("float32")
                ),
            ),
        }
        result["conditions"][feature_set] = {
            "features": 21 if feature_set == "low" else 28,
            "torchboost_candidates": tb_rows, "torchboost_choice": tb_i,
            "catboost_candidates": cat_rows, "catboost_choice": cat_i,
            "mlp_candidates": mlp_rows, "mlp_choice": mlp_i,
            "audit": audit,
        }
    result["status"] = "completed"
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=313)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = run(args.seed, args.smoke)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    Path(args.out).write_text(text)
    print(text)
