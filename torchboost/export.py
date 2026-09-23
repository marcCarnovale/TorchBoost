"""NumPy-only evaluation of the versioned hard-tree export."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def predict_exported_proba(path: str | Path, X) -> np.ndarray:
    """Evaluate hard-binary.v1 JSON without constructing a PyTorch model.

    The schema contains training-only imputation/scaling, affine node tests,
    left-on-zero tie handling, leaf scores, and stage coefficients. This matches
    hard routing up to floating point roundoff; soft/hard agreement is a separate
    measured property. Loading a data-only JSON file never executes model code.
    """
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != "torchboost.hard-binary.v1":
        raise ValueError("unsupported hard-tree schema")
    x = np.asarray(X, dtype=np.float64)
    center, scale = np.asarray(payload["center"]), np.asarray(payload["scale"])
    if x.ndim != 2 or x.shape[1] != len(center) or np.isinf(x).any():
        raise ValueError("X has invalid dimensions or infinities")
    x = ((np.where(np.isnan(x), center, x)-center)/scale).astype(np.float32)
    score = np.full(len(x), payload["base_score"], dtype=np.float64)
    for rate, tree in zip(payload["rates"], payload["trees"]):
        weights = np.asarray(tree["weights"], dtype=np.float32)
        biases = np.asarray(tree["biases"], dtype=np.float32)
        leaves = np.asarray(tree["leaf_values"], dtype=np.float64)
        node = np.zeros(len(x), dtype=np.int64)
        for _ in range(tree["depth"]):
            affine = np.einsum("bf,bf->b", x, weights[node])+biases[node]
            node = 2*node+1+(affine < 0).astype(np.int64)
        score += rate*leaves[node-(2**tree["depth"]-1)]
    # Stable logistic transform, including very large scores.
    p = np.exp(-np.logaddexp(0, -score))
    return np.column_stack((1-p, p))
