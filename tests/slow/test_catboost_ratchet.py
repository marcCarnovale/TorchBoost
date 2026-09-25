"""Slow performance ratchet: the power tree must retain its CatBoost advantage.

This is intentionally not part of the fast unit suite. CI runs it on every push/PR
with CatBoost pinned. Thresholds live in benchmarks/catboost_ratchet.json and may
only move in the stronger direction after a new locked benchmark result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss

from torchboost.adaptive.config import StructureConfig
from torchboost.adaptive.progressive_regularizers import Regularizers
from torchboost.adaptive.unified_progressive import (
    UnifiedConfig,
    UnifiedProgressiveClassifier,
    default_native,
)

ROOT = Path(__file__).resolve().parents[2]
RATCHET = json.loads((ROOT / "benchmarks" / "catboost_ratchet.json").read_text())


def _make(n: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype("float32")
    bits = (x[:, :4] > 0).astype(int)
    context = sum(bits[:, j] * (1 << j) for j in range(4))
    coefficient = rng.normal(size=(16, d))
    coefficient[:, :4] = 0
    raw = np.array([coefficient[c] @ row for c, row in zip(context, x)])
    raw = raw / np.std(raw) * 1.15
    probability = 1 / (1 + np.exp(-raw))
    y = rng.binomial(1, probability).astype(int)
    return x, y


def _run(seed: int):
    nfit = RATCHET["fit_rows"]
    nselection = RATCHET["selection_rows"]
    naudit = RATCHET["audit_rows"]
    x, y = _make(nfit + nselection + naudit, 16, seed)
    rng = np.random.default_rng(seed + 9)
    order = rng.permutation(len(x))
    fit = order[:nfit]
    selection = order[nfit : nfit + nselection]
    audit = order[nfit + nselection :]

    native = default_native()
    native.learning_rate = 0.01
    native.batch_size = 256
    native.structure = StructureConfig(
        dynamic=False,
        initial_depth=0,
        max_depth=6,
        max_nodes=511,
    )
    config = UnifiedConfig(
        n_trees=1,
        updates_per_stage=512,
        depth=4,
        bins=8,
        min_samples_leaf=12,
        linear_values=True,
        linear_l2=8.0,
        proposal_mode="linear_model_tree",
        cart_strength=7.0,
        warm_value_updates=8,
        gate_release="oblique",
        checkpoint_every=32,
        regularizers=Regularizers(
            leaf_l2=1e-5,
            hierarchy=3e-4,
            linear_value_l2=2e-5,
        ),
        native=native,
        random_state=seed,
        auto_complexity=True,
        proposal_candidates=1,
    )
    torchboost = UnifiedProgressiveClassifier(config).fit(
        x[fit], y[fit], eval_set=(x[selection], y[selection])
    )
    torchboost_nll = log_loss(y[audit], torchboost.predict_proba(x[audit]))

    catboost = CatBoostClassifier(
        iterations=256,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=20,
        verbose=False,
        random_seed=seed,
        thread_count=1,
    ).fit(x[fit], y[fit])
    catboost_nll = log_loss(y[audit], catboost.predict_proba(x[audit]))
    return torchboost_nll, catboost_nll


def test_single_power_tree_catboost_ratchet():
    rows = [_run(seed) for seed in RATCHET["seeds"]]
    torchboost = np.array([row[0] for row in rows])
    catboost = np.array([row[1] for row in rows])
    mean_torchboost = float(torchboost.mean())
    mean_catboost = float(catboost.mean())
    advantage = 1 - mean_torchboost / mean_catboost

    assert float(torchboost.max()) <= RATCHET["max_seed_nll"], rows
    assert mean_torchboost <= RATCHET["max_torchboost_mean_nll"], rows
    assert advantage >= RATCHET["min_relative_advantage"], {
        "rows": rows,
        "mean_torchboost": mean_torchboost,
        "mean_catboost": mean_catboost,
        "relative_advantage": advantage,
    }
