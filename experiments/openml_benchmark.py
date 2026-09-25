"""Benchmark a harder public numerical tabular task against selection-tuned CatBoost."""
from __future__ import annotations

import time

import numpy as np
from catboost import CatBoostClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from torchboost.adaptive.config import StructureConfig
from torchboost.adaptive.progressive_regularizers import Regularizers
from torchboost.adaptive.unified_progressive import (
    UnifiedConfig,
    UnifiedProgressiveClassifier,
    default_native,
)


def load_binary(data_id: int):
    x, y = fetch_openml(data_id=data_id, as_frame=False, parser="auto", return_X_y=True)
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        finite = np.where(np.isfinite(x), x, np.nan)
        medians = np.nanmedian(finite, axis=0)
        bad = ~np.isfinite(x)
        x[bad] = medians[np.where(bad)[1]]
    y = LabelEncoder().fit_transform(np.asarray(y))
    if len(np.unique(y)) != 2:
        raise ValueError("binary classification dataset required")
    return x, y


def split_indices(n: int, seed: int):
    p = np.random.default_rng(seed + 1009).permutation(n)
    a, b, c = int(.60 * n), int(.70 * n), int(.85 * n)
    return p[:a], p[a:b], p[b:c], p[c:]


def torchboost_config(seed: int, updates: int):
    native = default_native()
    native.learning_rate = .01
    native.batch_size = 512
    native.structure = StructureConfig(
        dynamic=False,
        initial_depth=0,
        max_depth=6,
        max_nodes=511,
    )
    return UnifiedConfig(
        n_trees=1,
        updates_per_stage=updates,
        depth=5,
        bins=12,
        min_samples_leaf=24,
        linear_values=True,
        linear_l2=8.,
        proposal_mode="linear_model_tree",
        cart_strength=7.,
        warm_value_updates=8,
        gate_release="oblique",
        checkpoint_every=32,
        auto_complexity=True,
        proposal_candidates=1,
        regularizers=Regularizers(
            leaf_l2=1e-5,
            hierarchy=3e-4,
            linear_value_l2=2e-5,
        ),
        native=native,
        random_state=seed,
    )


def run(data_id: int = 44128, seed: int = 41, updates: int = 384):
    x, y = load_binary(data_id)
    train, control, selection, audit = split_indices(len(x), seed)
    start = time.time()
    model = UnifiedProgressiveClassifier(torchboost_config(seed, updates)).fit(
        x[train],
        y[train],
        control_set=(x[control], y[control]),
        eval_set=(x[selection], y[selection]),
    )
    torchboost_audit = float(log_loss(y[audit], model.predict_proba(x[audit])))

    candidates = []
    for depth in (6, 8, 10):
        cat = CatBoostClassifier(
            iterations=512,
            depth=depth,
            learning_rate=.05,
            l2_leaf_reg=20.,
            loss_function="Logloss",
            verbose=False,
            random_seed=seed,
            thread_count=1,
        ).fit(x[train], y[train])
        selection_loss = float(log_loss(y[selection], cat.predict_proba(x[selection])))
        candidates.append((selection_loss, depth, cat))
    cat_selection, cat_depth, cat = min(candidates, key=lambda item: item[0])
    catboost_audit = float(log_loss(y[audit], cat.predict_proba(x[audit])))
    return {
        "data_id": data_id,
        "rows": int(len(x)),
        "features": int(x.shape[1]),
        "seed": seed,
        "updates": updates,
        "torchboost_selection": float(model.best_score_),
        "torchboost_audit": torchboost_audit,
        "torchboost_best_step": int(model.trainer_.best_epoch),
        "catboost_selected_depth": int(cat_depth),
        "catboost_selection": cat_selection,
        "catboost_audit": catboost_audit,
        "relative_audit_gain": 1. - torchboost_audit / catboost_audit,
        "seconds": time.time() - start,
    }
