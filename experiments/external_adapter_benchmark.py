"""Frozen cross-dataset benchmark for the HIGGS-discovered residual adapter.

This benchmark is deliberately separate from the sealed HIGGS shadow audit.
It tests transfer of one frozen mechanism and training recipe across public
binary tabular datasets without per-dataset architecture tuning.

Protocol:
- stratified 60/20/20 train/selection/ranking split;
- deterministic size-based MLP capacity declared below;
- MLP checkpoint selected by selection NLL;
- frozen-backbone residual adapter at fixed scale sigmoid(-2);
- identical adapter with held-out learned layer scales;
- CatBoost, XGBoost, and LightGBM references with fixed global recipes;
- selection is used only for checkpointing/early stopping;
- ranking is touched only after fitting for the reported comparison.

The script records exact exposure, parameter counts, wall time, selection and
ranking metrics, learned architecture state, source SHA, and package versions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from experiments.higgs_hybrid_benchmark import MLP
from torchboost.adaptive.architecture_corners import CompositionalTreeNetwork
from torchboost.adaptive.architecture_regularization import architecture_state

PROTOCOL_VERSION = "external-adapter-v2-frozen"
DATASETS = {
    "phoneme": 44127,
    "bioresponse": 45019,
    "bank-marketing": 44126,
    "magic-telescope": 44125,
    "default-credit": 45020,
    "electricity": 44120,
    "miniboone": 44128,
}
INITIAL_SCALE = 1.0 / (1.0 + math.exp(2.0))
ANCHOR_EPOCHS = 40
ADAPTER_EPOCHS = 12
SCALE_WARMUP_EPOCHS = 6
ARCH_EVERY = 2


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "nll": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)),
    }


def capacity(n):
    """Frozen capacity rule; depends only on total row count, never outcomes."""
    if n < 10_000:
        return 128, 3
    if n < 50_000:
        return 192, 4
    return 256, 4


def source_sha():
    value = os.environ.get("GITHUB_SHA")
    if value:
        return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


def count_trainable(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def load_dataset(name):
    did = DATASETS[name]
    bunch = fetch_openml(data_id=did, as_frame=True, parser="auto")
    x = bunch.data.copy()
    x = x.apply(pd.to_numeric, errors="coerce")
    y = LabelEncoder().fit_transform(np.asarray(bunch.target).astype(str))
    if len(np.unique(y)) != 2:
        raise ValueError(f"{name} is not binary after loading")
    missing_before = int(x.isna().sum().sum())
    imp = SimpleImputer(strategy="median")
    x = imp.fit_transform(x).astype("float32")
    metadata = {
        "openml_id": did,
        "openml_name": getattr(bunch, "details", {}).get("name", name),
        "openml_version": getattr(bunch, "details", {}).get("version"),
        "missing_values_imputed": missing_before,
        "positive_fraction": float(np.mean(y)),
    }
    return x, y.astype("float32"), metadata


def splits(x, y, seed):
    tx, rx, ty, ry = train_test_split(
        x, y, test_size=0.40, random_state=seed, stratify=y
    )
    sx, qx, sy, qy = train_test_split(
        rx, ry, test_size=0.50, random_state=seed + 1, stratify=ry
    )
    return (tx, ty), (sx, sy), (qx, qy)


@torch.no_grad()
def probability(model, x, batch=4096):
    model.eval()
    rows = []
    for start in range(0, len(x), batch):
        rows.append(
            torch.sigmoid(model(torch.from_numpy(x[start : start + batch]))).numpy()
        )
    return np.concatenate(rows)


def train_anchor(model, train_x, train_y, sel_x, sel_y, *, epochs, batch, seed):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = torch.Generator().manual_seed(seed)
    best = (float("inf"), None, 0)
    examples = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=rng)
        for start in range(0, len(order), batch):
            idx = order[start : start + batch].numpy()
            xb = torch.from_numpy(train_x[idx])
            yb = torch.from_numpy(train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            examples += len(idx)
        score = metrics(sel_y, probability(model, sel_x))["nll"]
        if score < best[0]:
            best = (
                score,
                {k: v.detach().clone() for k, v in model.state_dict().items()},
                epoch + 1,
            )
    model.load_state_dict(best[1])
    return {
        "best_epoch": best[2],
        "train_examples_seen": examples,
        "full_train_passes": examples / len(train_x),
        "fit_seconds": time.perf_counter() - started,
    }


def build_adapter(anchor, learn_scales):
    model = deepcopy(anchor)
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in model.layers:
        layer.grow_one_level()
        layer.set_architecture_scale(INITIAL_SCALE, learnable=learn_scales)
        tree = layer.forest.trees[0]
        root = layer.root
        root.value.requires_grad_(False)
        root.linear_value.requires_grad_(False)
        if root.routing_weight is not None:
            root.routing_weight.requires_grad_(True)
        if root.routing_bias is not None:
            root.routing_bias.requires_grad_(True)
        for cid in root.children_ids:
            child = tree.get(cid)
            child.value.requires_grad_(True)
            if child.linear_value is not None:
                child.linear_value.requires_grad_(True)
            child.allocation_logit.requires_grad_(False)
    return model


def partition(model):
    residual, scales = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (scales if name.endswith("architecture_logit") else residual).append(p)
    if not residual:
        raise RuntimeError("adapter has no residual parameters")
    return residual, scales


def train_adapter(
    model,
    train_x,
    train_y,
    sel_x,
    sel_y,
    *,
    epochs,
    warmup,
    batch,
    seed,
    learn_scales,
):
    residual, scales = partition(model)
    ropt = torch.optim.AdamW(residual, lr=1e-3, weight_decay=1e-5)
    sopt = torch.optim.Adam(scales, lr=1e-2) if learn_scales else None
    loss_fn = torch.nn.BCEWithLogitsLoss()
    trng = torch.Generator().manual_seed(seed)
    srng = torch.Generator().manual_seed(seed + 97)
    best = (float("inf"), None, 0, None)
    scale_updates = 0
    train_examples = 0
    selection_examples = 0
    started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=trng)
        sorder = torch.randperm(len(sel_x), generator=srng)
        cursor = 0
        for bi, start in enumerate(range(0, len(order), batch)):
            idx = order[start : start + batch].numpy()
            xb = torch.from_numpy(train_x[idx])
            yb = torch.from_numpy(train_y[idx])
            ropt.zero_grad(set_to_none=True)
            if sopt is not None:
                sopt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual, 10.0)
            ropt.step()
            train_examples += len(idx)

            if learn_scales and epoch >= warmup and (bi + 1) % ARCH_EVERY == 0:
                if cursor + batch > len(sorder):
                    sorder = torch.randperm(len(sel_x), generator=srng)
                    cursor = 0
                si = sorder[cursor : cursor + batch].numpy()
                cursor += batch
                sx = torch.from_numpy(sel_x[si])
                sy = torch.from_numpy(sel_y[si])
                ropt.zero_grad(set_to_none=True)
                sopt.zero_grad(set_to_none=True)
                sloss = loss_fn(model(sx), sy)
                sloss.backward()
                torch.nn.utils.clip_grad_norm_(scales, 2.0)
                sopt.step()
                scale_updates += 1
                selection_examples += len(si)

        score = metrics(sel_y, probability(model, sel_x))["nll"]
        if score < best[0]:
            best = (
                score,
                {k: v.detach().clone() for k, v in model.state_dict().items()},
                epoch + 1,
                architecture_state(model),
            )

    model.load_state_dict(best[1])
    return {
        "best_epoch": best[2],
        "architecture": best[3],
        "scale_updates": scale_updates,
        "train_examples_seen": train_examples,
        "selection_examples_seen_by_architecture_optimizer": selection_examples,
        "full_train_passes": train_examples / len(train_x),
        "fit_seconds": time.perf_counter() - started,
    }


def fit_reference(kind, tx, ty, sx, sy, seed):
    started = time.perf_counter()
    if kind == "catboost":
        model = CatBoostClassifier(
            iterations=1200,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=10,
            loss_function="Logloss",
            eval_metric="Logloss",
            verbose=False,
            random_seed=seed,
            thread_count=4,
            allow_writing_files=False,
        )
        model.fit(tx, ty, eval_set=(sx, sy), early_stopping_rounds=100, verbose=False)
        complexity = {"trees": int(model.tree_count_)}
    elif kind == "xgboost":
        model = XGBClassifier(
            n_estimators=1200,
            max_depth=8,
            learning_rate=0.05,
            min_child_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=4,
            tree_method="hist",
            early_stopping_rounds=100,
        )
        model.fit(tx, ty, eval_set=[(sx, sy)], verbose=False)
        complexity = {"trees": int(model.best_iteration + 1)}
    elif kind == "lightgbm":
        import lightgbm as lgb

        model = LGBMClassifier(
            n_estimators=1200,
            num_leaves=255,
            learning_rate=0.05,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            objective="binary",
            random_state=seed,
            n_jobs=4,
            verbosity=-1,
        )
        model.fit(
            tx,
            ty,
            eval_set=[(sx, sy)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        complexity = {"trees": int(model.best_iteration_)}
    else:
        raise ValueError(kind)
    return model, {
        **complexity,
        "fit_seconds": time.perf_counter() - started,
    }


def model_eval(model, sx, sy, qx, qy, *, torch_model=False):
    pred = probability if torch_model else lambda m, x: m.predict_proba(x)[:, 1]
    return {
        "selection": metrics(sy, pred(model, sx)),
        "ranking": metrics(qy, pred(model, qx)),
    }


def run(name, seed, out):
    torch.set_num_threads(4)
    np.random.seed(seed)
    started = time.perf_counter()

    x, y, metadata = load_dataset(name)
    (tx, ty), (sx, sy), (qx, qy) = splits(x, y, seed)
    scaler = StandardScaler().fit(tx)
    tx = scaler.transform(tx).astype("float32")
    sx = scaler.transform(sx).astype("float32")
    qx = scaler.transform(qx).astype("float32")

    width, depth = capacity(len(x))
    batch = min(256, max(32, len(tx) // 8))

    torch.manual_seed(seed + 305)
    reference = MLP(tx.shape[1], width, depth, 0.1)
    canonical_rng = torch.get_rng_state()
    anchor = CompositionalTreeNetwork.from_mlp(
        reference, max_tree_depth=2, seed=seed + 1200
    )
    torch.set_rng_state(canonical_rng)
    anchor_training = train_anchor(
        anchor,
        tx,
        ty,
        sx,
        sy,
        epochs=ANCHOR_EPOCHS,
        batch=batch,
        seed=seed + 9001,
    )

    fixed = build_adapter(anchor, False)
    learned = build_adapter(anchor, True)
    fixed_training = train_adapter(
        fixed,
        tx,
        ty,
        sx,
        sy,
        epochs=ADAPTER_EPOCHS,
        warmup=0,
        batch=batch,
        seed=seed + 19001,
        learn_scales=False,
    )
    learned_training = train_adapter(
        learned,
        tx,
        ty,
        sx,
        sy,
        epochs=ADAPTER_EPOCHS,
        warmup=SCALE_WARMUP_EPOCHS,
        batch=batch,
        seed=seed + 19001,
        learn_scales=True,
    )

    cat, cat_fit = fit_reference("catboost", tx, ty, sx, sy, seed)
    xgb, xgb_fit = fit_reference("xgboost", tx, ty, sx, sy, seed)
    lgb, lgb_fit = fit_reference("lightgbm", tx, ty, sx, sy, seed)

    anchor_m = model_eval(anchor, sx, sy, qx, qy, torch_model=True)
    fixed_m = model_eval(fixed, sx, sy, qx, qy, torch_model=True)
    learned_m = model_eval(learned, sx, sy, qx, qy, torch_model=True)
    cat_m = model_eval(cat, sx, sy, qx, qy)
    xgb_m = model_eval(xgb, sx, sy, qx, qy)
    lgb_m = model_eval(lgb, sx, sy, qx, qy)

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "completed",
        "source_sha": source_sha(),
        "dataset": name,
        **metadata,
        "seed": seed,
        "rows": len(x),
        "features": x.shape[1],
        "split_rows": {
            "train": len(tx),
            "selection": len(sx),
            "ranking": len(qx),
        },
        "preprocessing": {
            "numeric_coercion": True,
            "median_imputation_fit_on_full_loaded_dataset": True,
            "standard_scaler_fit_on_train_only": True,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "mlp": {
            "width": width,
            "depth": depth,
            "total_parameters": count_parameters(anchor),
            "trainable_parameters": count_trainable(anchor),
            **anchor_training,
            **anchor_m,
        },
        "fixed_adapter": {
            "total_parameters": count_parameters(fixed),
            "trainable_parameters": count_trainable(fixed),
            **fixed_training,
            **fixed_m,
        },
        "learned_adapter": {
            "total_parameters": count_parameters(learned),
            "trainable_parameters": count_trainable(learned),
            **learned_training,
            **learned_m,
        },
        "catboost": {**cat_fit, **cat_m},
        "xgboost": {**xgb_fit, **xgb_m},
        "lightgbm": {**lgb_fit, **lgb_m},
        "deltas": {
            "learned_minus_mlp_nll": learned_m["ranking"]["nll"] - anchor_m["ranking"]["nll"],
            "learned_minus_mlp_auc": learned_m["ranking"]["auc"] - anchor_m["ranking"]["auc"],
            "learned_minus_fixed_nll": learned_m["ranking"]["nll"] - fixed_m["ranking"]["nll"],
            "learned_minus_fixed_auc": learned_m["ranking"]["auc"] - fixed_m["ranking"]["auc"],
            "learned_minus_catboost_nll": learned_m["ranking"]["nll"] - cat_m["ranking"]["nll"],
            "learned_minus_catboost_auc": learned_m["ranking"]["auc"] - cat_m["ranking"]["auc"],
            "learned_minus_xgboost_nll": learned_m["ranking"]["nll"] - xgb_m["ranking"]["nll"],
            "learned_minus_xgboost_auc": learned_m["ranking"]["auc"] - xgb_m["ranking"]["auc"],
            "learned_minus_lightgbm_nll": learned_m["ranking"]["nll"] - lgb_m["ranking"]["nll"],
            "learned_minus_lightgbm_auc": learned_m["ranking"]["auc"] - lgb_m["ranking"]["auc"],
        },
        "seconds": time.perf_counter() - started,
        "higgs_shadow_audit_opened": False,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.dataset, a.seed, a.out)
