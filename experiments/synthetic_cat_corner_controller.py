"""Synthetic v3 trust-region evolution from the CatBoost corner.

This study intentionally does NOT touch any external benchmark dataset or the
locked HIGGS shadow audit.

v3 changes the control objective:
- CatBoost is an initialization/local chart, not an absorbing endpoint.
- The final model must remain a distinct TorchBoost hybrid: exact zero residual
  capacity is not admissible.
- A small trust radius is always retained ("harmless jitter").
- Cross-fitted evidence chooses a direction away from CatBoost and controls
  expansion/contraction of the radius.
- Weak or adverse evidence contracts to the minimum radius rather than
  collapsing to CatBoost.

The local departure basis contains three TorchBoost residual directions with
different capacities. The basis is deliberately simple for synthetic
calibration; later real-model work can replace these with semantic mechanism
directions (hard/oblique/affine/rate/neural).

No permanent validation split is consumed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from experiments.higgs_hybrid_benchmark import MLP
from torchboost.adaptive.architecture_corners import CompositionalTreeNetwork


DIRECTIONS = {
    "local_shallow": {"width": 64, "depth": 2, "grow": 1},
    "local_medium": {"width": 96, "depth": 3, "grow": 1},
    "local_deep": {"width": 128, "depth": 4, "grow": 2},
}
MIN_RADIUS = 0.01


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "nll": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)),
    }


def latent_problem(regime, seed, p=24):
    rng = np.random.default_rng(seed)
    problem = {"regime": regime, "p": int(p)}
    if regime == "oblique_dense":
        w = rng.normal(size=p)
        w /= np.linalg.norm(w)
        v = rng.normal(size=p)
        v /= np.linalg.norm(v)
        problem["w"] = w.astype("float32")
        problem["v"] = v.astype("float32")
    elif regime == "mixed":
        w = rng.normal(size=p)
        w /= np.linalg.norm(w)
        problem["w"] = w.astype("float32")
    elif regime != "axis_sparse":
        raise ValueError(regime)
    return problem


def sample_problem(problem, n, seed):
    rng = np.random.default_rng(seed)
    p = problem["p"]
    x = rng.normal(size=(n, p)).astype("float32")
    regime = problem["regime"]
    if regime == "axis_sparse":
        z = (
            1.8 * (x[:, 0] > 0.25)
            - 1.5 * (x[:, 1] < -0.35)
            + 1.3 * ((x[:, 2] > 0) & (x[:, 3] > 0.1))
            + 0.7 * np.tanh(2 * x[:, 4])
            - 0.25
        )
    elif regime == "oblique_dense":
        w = problem["w"]
        v = problem["v"]
        z = (
            2.2 * (x @ w)
            + 1.15 * np.sin(1.4 * (x @ v))
            + 0.45 * (x[:, 0] * x[:, 1])
            - 0.15
        )
    elif regime == "mixed":
        w = problem["w"]
        z = (
            1.25 * (x[:, 0] > 0.2)
            - 1.05 * (x[:, 1] < -0.4)
            + 1.45 * (x @ w)
            + 0.65 * np.tanh(x[:, 2] * x[:, 3])
            - 0.2
        )
    else:
        raise ValueError(regime)
    z = z + rng.normal(scale=0.7, size=n)
    prob = _sigmoid(z)
    y = rng.binomial(1, prob).astype("float32")
    return x, y


def regime_stats(x, y):
    n, p = x.shape
    yc = y - y.mean()
    corrs = []
    for j in range(p):
        xc = x[:, j] - x[:, j].mean()
        den = np.sqrt(np.sum(xc * xc) * np.sum(yc * yc))
        corrs.append(0.0 if den == 0 else abs(float(np.sum(xc * yc) / den)))
    corrs = np.sort(np.asarray(corrs))[::-1]
    total = float(corrs.sum() + 1e-12)
    return {
        "n": int(n),
        "p": int(p),
        "log10_n": float(np.log10(max(n, 1))),
        "top4_marginal_signal_share": float(corrs[: min(4, p)].sum() / total),
        "value_sparsity": float(np.mean(np.abs(x) < 1e-8)),
        "class_imbalance": float(abs(y.mean() - 0.5) * 2),
    }


def residual_prior(stats):
    n = stats["n"]
    size = (math.log10(max(n, 300)) - math.log10(1200)) / (
        math.log10(50000) - math.log10(1200)
    )
    size = float(np.clip(size, 0, 1))
    tree_evidence = (
        0.55 * stats["top4_marginal_signal_share"]
        + 0.25 * stats["value_sparsity"]
        + 0.20 * stats["class_imbalance"]
    )
    mean = 0.08 + 0.48 * size - 0.28 * tree_evidence
    return float(np.clip(mean, MIN_RADIUS, 0.50))


def radius_grid(prior):
    vals = [
        MIN_RADIUS,
        max(MIN_RADIUS, 0.5 * prior),
        max(MIN_RADIUS, prior),
        min(0.50, max(MIN_RADIUS, 2.0 * prior)),
    ]
    return sorted({float(np.clip(v, MIN_RADIUS, 0.50)) for v in vals})


def build_residual(p, direction, seed):
    cfg = DIRECTIONS[direction]
    torch.manual_seed(seed)
    base = MLP(p, cfg["width"], cfg["depth"], 0.05)
    model = CompositionalTreeNetwork.from_mlp(base, max_tree_depth=3, seed=seed + 17)
    for layer in model.layers:
        for _ in range(cfg["grow"]):
            layer.grow_one_level()
    return model


@torch.no_grad()
def residual_values(model, x, batch=2048):
    model.eval()
    out = []
    for start in range(0, len(x), batch):
        out.append(model(torch.from_numpy(x[start : start + batch])).cpu().numpy())
    return np.concatenate(out)


def cat_model(seed):
    return CatBoostClassifier(
        iterations=400,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=8,
        loss_function="Logloss",
        verbose=False,
        random_seed=seed,
        thread_count=4,
    )


def oof_cat_logits(x, y, seed, folds=3):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 101)
    out = np.zeros(len(y), dtype="float32")
    retained = []
    for fold, (tr, va) in enumerate(skf.split(x, y)):
        model = cat_model(seed + 1000 + fold)
        model.fit(
            x[tr],
            y[tr],
            eval_set=(x[va], y[va]),
            early_stopping_rounds=50,
            verbose=False,
        )
        out[va] = _logit(model.predict_proba(x[va])[:, 1]).astype("float32")
        retained.append(int(model.tree_count_))
    return out, retained


def train_residual(x, y, base_logits, radius, direction, seed, epochs=18, batch=256):
    model = build_residual(x.shape[1], direction, seed)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=8e-4, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = torch.Generator().manual_seed(seed + 7001)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    bt = torch.from_numpy(base_logits.astype("float32"))
    best = (float("inf"), None, 0)
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=rng)
        model.train()
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            opt.zero_grad(set_to_none=True)
            logits = bt[idx] + float(radius) * model(xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
        with torch.no_grad():
            p = torch.sigmoid(bt + float(radius) * model(xt)).numpy()
            score = metrics(y, p)["nll"]
        if score < best[0]:
            best = (
                score,
                {k: v.detach().clone() for k, v in model.state_dict().items()},
                epoch + 1,
            )
    model.load_state_dict(best[1])
    return model, best[2], best[0]


def architecture_evidence(x, y, prior, seed):
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 501)
    radii = radius_grid(prior)
    fold_records = []

    for fold, (tr, va) in enumerate(outer.split(x, y)):
        train_x, train_y = x[tr], y[tr]
        val_x, val_y = x[va], y[va]

        inner_logits, inner_trees = oof_cat_logits(
            train_x, train_y, seed + 10000 + fold * 100
        )
        anchor = cat_model(seed + 30000 + fold)
        anchor.fit(train_x, train_y, verbose=False)
        anchor_p = anchor.predict_proba(val_x)[:, 1]
        anchor_logits = _logit(anchor_p)
        anchor_nll = metrics(val_y, anchor_p)["nll"]

        directions = {}
        for di, direction in enumerate(DIRECTIONS):
            exploratory_radius = max(MIN_RADIUS, min(0.20, prior))
            residual, best_epoch, inner_nll = train_residual(
                train_x,
                train_y,
                inner_logits,
                exploratory_radius,
                direction,
                seed + 20000 + fold * 1000 + di * 100,
            )
            rv = residual_values(residual, val_x)
            scored = {}
            for radius in radii:
                p = _sigmoid(anchor_logits + radius * rv)
                m = metrics(val_y, p)
                scored[str(radius)] = {
                    "nll": m["nll"],
                    "auc": m["auc"],
                    "delta_nll_vs_anchor": m["nll"] - anchor_nll,
                }
            directions[direction] = {
                "exploratory_radius": exploratory_radius,
                "residual_best_epoch": best_epoch,
                "inner_oof_training_nll": inner_nll,
                "candidates": scored,
            }

        fold_records.append(
            {
                "fold": fold,
                "rows_fit": int(len(tr)),
                "rows_evidence": int(len(va)),
                "inner_oof_catboost_trees": inner_trees,
                "anchor_trees": int(anchor.tree_count_),
                "anchor_evidence_nll": anchor_nll,
                "directions": directions,
            }
        )

    summary = {}
    for direction in DIRECTIONS:
        summary[direction] = {}
        for radius in radii:
            key = str(radius)
            deltas = np.asarray(
                [
                    record["directions"][direction]["candidates"][key][
                        "delta_nll_vs_anchor"
                    ]
                    for record in fold_records
                ],
                dtype=float,
            )
            summary[direction][key] = {
                "mean_delta_nll": float(deltas.mean()),
                "median_delta_nll": float(np.median(deltas)),
                "std_delta_nll": float(deltas.std(ddof=0)),
                "improving_folds": int(np.sum(deltas < 0)),
                "fold_deltas": [float(v) for v in deltas],
            }

    # Expansion rule: expand beyond minimum jitter only with consistent evidence.
    # If no expanded candidate qualifies, choose the direction with the smallest
    # mean degradation/improvement at MIN_RADIUS. Thus the model always remains
    # a nonzero TorchBoost hybrid.
    eligible = []
    for direction in DIRECTIONS:
        for radius in radii:
            if radius <= MIN_RADIUS + 1e-12:
                continue
            s = summary[direction][str(radius)]
            stderr = s["std_delta_nll"] / math.sqrt(3.0)
            # Require 2/3 folds improving and mean gain exceeding a modest
            # noise margin. This is stricter than v2's sign-only rule.
            if s["improving_folds"] >= 2 and s["mean_delta_nll"] < -0.5 * stderr:
                eligible.append((s["mean_delta_nll"], direction, radius))

    if eligible:
        _, selected_direction, selected_radius = min(eligible)
        mode = "expanded"
    else:
        jitter = []
        for direction in DIRECTIONS:
            s = summary[direction][str(MIN_RADIUS)]
            jitter.append((s["mean_delta_nll"], direction))
        _, selected_direction = min(jitter)
        selected_radius = MIN_RADIUS
        mode = "minimum_jitter"

    return selected_direction, selected_radius, mode, radii, fold_records, summary


def run(regime, n, seed, out):
    torch.set_num_threads(4)
    started = time.perf_counter()

    problem = latent_problem(regime, seed + 41)
    x, y = sample_problem(problem, n, seed + 1001)
    qx, qy = sample_problem(problem, max(12000, n), seed + 500001)

    scaler = StandardScaler().fit(x)
    x = scaler.transform(x).astype("float32")
    qx = scaler.transform(qx).astype("float32")

    stats = regime_stats(x, y)
    prior = residual_prior(stats)
    (
        selected_direction,
        selected_radius,
        mode,
        radii,
        fold_records,
        evidence,
    ) = architecture_evidence(x, y, prior, seed)

    full_oof_logits, full_oof_trees = oof_cat_logits(x, y, seed + 60000)
    direction_index = list(DIRECTIONS).index(selected_direction)
    residual, final_best_epoch, full_oof_nll = train_residual(
        x,
        y,
        full_oof_logits,
        selected_radius,
        selected_direction,
        seed + 70000 + direction_index * 100,
        epochs=24,
    )

    cat = cat_model(seed + 90000)
    cat.fit(x, y, verbose=False)
    base_p = cat.predict_proba(qx)[:, 1]
    base_logits = _logit(base_p)
    hybrid_p = _sigmoid(
        base_logits + selected_radius * residual_values(residual, qx)
    )

    base_m = metrics(qy, base_p)
    hybrid_m = metrics(qy, hybrid_p)
    result = {
        "study": "synthetic_cat_corner_trust_region_v3",
        "regime": regime,
        "seed": seed,
        "train_rows": int(n),
        "ranking_rows": int(len(qy)),
        "features": int(x.shape[1]),
        "same_latent_problem_train_and_ranking": True,
        "train_only_stats": stats,
        "controller": {
            "prior_residual_radius": prior,
            "minimum_radius": MIN_RADIUS,
            "candidate_radii": radii,
            "directions": DIRECTIONS,
            "selected_direction": selected_direction,
            "selected_radius": selected_radius,
            "selection_mode": mode,
            "exact_catboost_fallback_allowed": False,
            "expansion_rule": ">=2/3 folds improve and mean_delta_nll < -0.5*stderr",
            "outer_fold_evidence": fold_records,
            "candidate_summary": evidence,
            "full_oof_catboost_retained_trees": full_oof_trees,
            "final_residual_best_epoch": final_best_epoch,
            "full_oof_training_nll": full_oof_nll,
        },
        "catboost": {
            "retained_trees": int(cat.tree_count_),
            "ranking": base_m,
        },
        "hybrid": {
            "ranking": hybrid_m,
        },
        "deltas": {
            "hybrid_minus_catboost_nll": hybrid_m["nll"] - base_m["nll"],
            "hybrid_minus_catboost_auc": hybrid_m["auc"] - base_m["auc"],
        },
        "seconds": time.perf_counter() - started,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regime",
        choices=["axis_sparse", "oblique_dense", "mixed"],
        required=True,
    )
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.regime, args.n, args.seed, args.out)
