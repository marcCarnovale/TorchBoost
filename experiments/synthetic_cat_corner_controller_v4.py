"""Prediction-space trust-region calibration around the CatBoost corner.

Synthetic-only v4. It imports the v3 data/problem helpers but replaces the
coefficient-space controller. A radius now means RMS logit displacement after
normalizing and bounding the residual direction, so a trainable residual cannot
evade a small trust region by increasing its raw output scale.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import experiments.synthetic_cat_corner_controller as v3

MIN_RMS_RADIUS = 0.005
MAX_RMS_RADIUS = 0.15
SHAPE_TRAIN_COEF = 0.05
EPS = 1e-8


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


def rms_radius_prior(stats):
    size = (math.log10(max(stats["n"], 300)) - math.log10(1200)) / (
        math.log10(50000) - math.log10(1200)
    )
    size = float(np.clip(size, 0, 1))
    tree_evidence = (
        0.55 * stats["top4_marginal_signal_share"]
        + 0.25 * stats["value_sparsity"]
        + 0.20 * stats["class_imbalance"]
    )
    return float(
        np.clip(0.015 + 0.070 * size - 0.015 * tree_evidence, 0.01, 0.10)
    )


def radius_grid(prior):
    values = [
        MIN_RMS_RADIUS,
        max(MIN_RMS_RADIUS, 0.5 * prior),
        max(MIN_RMS_RADIUS, prior),
        min(MAX_RMS_RADIUS, 2.0 * prior),
    ]
    return sorted(
        {float(np.clip(x, MIN_RMS_RADIUS, MAX_RMS_RADIUS)) for x in values}
    )


def fit_normalizer(raw):
    raw_rms = max(rms(raw), EPS)
    squashed = np.tanh(np.asarray(raw, dtype=float) / raw_rms)
    squashed_rms = max(rms(squashed), EPS)
    unit = squashed / squashed_rms
    return {
        "raw_rms": raw_rms,
        "squashed_rms": squashed_rms,
        "fit_unit_rms": rms(unit),
        "fit_unit_abs_max": float(np.max(np.abs(unit))),
    }


def unit_direction(raw, normalizer):
    z = np.tanh(
        np.asarray(raw, dtype=float) / max(normalizer["raw_rms"], EPS)
    )
    return z / max(normalizer["squashed_rms"], EPS)


def train_shape(x, y, base_logits, direction, seed, epochs=18, batch=256):
    model = v3.build_residual(x.shape[1], direction, seed)
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
            loss = loss_fn(
                bt[idx] + SHAPE_TRAIN_COEF * model(xt[idx]), yt[idx]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()

        raw = v3.residual_values(model, x)
        normalizer = fit_normalizer(raw)
        unit = unit_direction(raw, normalizer)
        score = v3.metrics(
            y, v3._sigmoid(base_logits + 0.05 * unit)
        )["nll"]
        if score < best[0]:
            best = (
                score,
                {
                    k: value.detach().clone()
                    for k, value in model.state_dict().items()
                },
                epoch + 1,
            )

    model.load_state_dict(best[1])
    normalizer = fit_normalizer(v3.residual_values(model, x))
    return model, normalizer, best[2], best[0]


def architecture_evidence(x, y, prior, seed):
    outer = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=seed + 501
    )
    radii = radius_grid(prior)
    records = []

    for fold, (tr, va) in enumerate(outer.split(x, y)):
        tx, ty, vx, vy = x[tr], y[tr], x[va], y[va]
        inner_logits, inner_trees = v3.oof_cat_logits(
            tx, ty, seed + 10000 + fold * 100
        )
        anchor = v3.cat_model(seed + 30000 + fold)
        anchor.fit(tx, ty, verbose=False)
        anchor_p = anchor.predict_proba(vx)[:, 1]
        anchor_logits = v3._logit(anchor_p)
        anchor_nll = v3.metrics(vy, anchor_p)["nll"]

        directions = {}
        for di, direction in enumerate(v3.DIRECTIONS):
            model, normalizer, epoch, inner_nll = train_shape(
                tx,
                ty,
                inner_logits,
                direction,
                seed + 20000 + fold * 1000 + di * 100,
            )
            unit = unit_direction(
                v3.residual_values(model, vx), normalizer
            )
            cells = {}
            for radius in radii:
                perturb = radius * unit
                m = v3.metrics(
                    vy, v3._sigmoid(anchor_logits + perturb)
                )
                cells[str(radius)] = {
                    "nll": m["nll"],
                    "auc": m["auc"],
                    "delta_nll_vs_anchor": m["nll"] - anchor_nll,
                    "realized_rms_logit_delta": rms(perturb),
                    "realized_abs_max_logit_delta": float(
                        np.max(np.abs(perturb))
                    ),
                }
            directions[direction] = {
                "normalizer": normalizer,
                "residual_best_epoch": epoch,
                "inner_oof_training_nll": inner_nll,
                "outer_unit_rms": rms(unit),
                "outer_unit_abs_max": float(np.max(np.abs(unit))),
                "candidates": cells,
            }

        records.append(
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
    for direction in v3.DIRECTIONS:
        summary[direction] = {}
        for radius in radii:
            key = str(radius)
            cells = [
                record["directions"][direction]["candidates"][key]
                for record in records
            ]
            deltas = np.asarray(
                [cell["delta_nll_vs_anchor"] for cell in cells],
                dtype=float,
            )
            realized = np.asarray(
                [cell["realized_rms_logit_delta"] for cell in cells],
                dtype=float,
            )
            summary[direction][key] = {
                "mean_delta_nll": float(deltas.mean()),
                "median_delta_nll": float(np.median(deltas)),
                "std_delta_nll": float(deltas.std(ddof=0)),
                "improving_folds": int(np.sum(deltas < 0)),
                "fold_deltas": [float(value) for value in deltas],
                "mean_realized_rms_logit_delta": float(realized.mean()),
            }

    eligible = []
    for direction in v3.DIRECTIONS:
        for radius in radii:
            if radius <= MIN_RMS_RADIUS + 1e-12:
                continue
            summary_cell = summary[direction][str(radius)]
            stderr = summary_cell["std_delta_nll"] / math.sqrt(3.0)
            if (
                summary_cell["improving_folds"] >= 2
                and summary_cell["mean_delta_nll"] < -0.5 * stderr
            ):
                eligible.append(
                    (
                        summary_cell["mean_delta_nll"],
                        direction,
                        radius,
                    )
                )

    if eligible:
        _, direction, radius = min(eligible)
        mode = "expanded"
    else:
        direction = min(
            v3.DIRECTIONS,
            key=lambda name: summary[name][str(MIN_RMS_RADIUS)][
                "mean_delta_nll"
            ],
        )
        radius = MIN_RMS_RADIUS
        mode = "minimum_prediction_jitter"
    return direction, radius, mode, radii, records, summary


def run(regime, n, seed, out):
    torch.set_num_threads(4)
    started = time.perf_counter()
    problem = v3.latent_problem(regime, seed + 41)
    x, y = v3.sample_problem(problem, n, seed + 1001)
    qx, qy = v3.sample_problem(
        problem, max(12000, n), seed + 500001
    )
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x).astype("float32")
    qx = scaler.transform(qx).astype("float32")

    stats = v3.regime_stats(x, y)
    prior = rms_radius_prior(stats)
    direction, radius, mode, radii, records, summary = (
        architecture_evidence(x, y, prior, seed)
    )

    full_oof_logits, full_oof_trees = v3.oof_cat_logits(
        x, y, seed + 60000
    )
    direction_index = list(v3.DIRECTIONS).index(direction)
    residual, normalizer, epoch, full_oof_nll = train_shape(
        x,
        y,
        full_oof_logits,
        direction,
        seed + 70000 + direction_index * 100,
        epochs=24,
    )

    cat = v3.cat_model(seed + 90000)
    cat.fit(x, y, verbose=False)
    base_p = cat.predict_proba(qx)[:, 1]
    base_logits = v3._logit(base_p)
    ranking_unit = unit_direction(
        v3.residual_values(residual, qx), normalizer
    )
    perturb = radius * ranking_unit
    hybrid_p = v3._sigmoid(base_logits + perturb)
    base_m = v3.metrics(qy, base_p)
    hybrid_m = v3.metrics(qy, hybrid_p)

    result = {
        "study": "synthetic_cat_corner_prediction_trust_region_v4",
        "regime": regime,
        "seed": seed,
        "train_rows": int(n),
        "ranking_rows": int(len(qy)),
        "features": int(x.shape[1]),
        "same_latent_problem_train_and_ranking": True,
        "train_only_stats": stats,
        "controller": {
            "prior_rms_logit_radius": prior,
            "minimum_rms_logit_radius": MIN_RMS_RADIUS,
            "candidate_radii": radii,
            "directions": v3.DIRECTIONS,
            "selected_direction": direction,
            "selected_rms_logit_radius": radius,
            "selection_mode": mode,
            "exact_catboost_fallback_allowed": False,
            "distance_definition": (
                "RMS normalized bounded residual logit displacement"
            ),
            "expansion_rule": (
                ">=2/3 folds improve and "
                "mean_delta_nll < -0.5*stderr"
            ),
            "outer_fold_evidence": records,
            "candidate_summary": summary,
            "full_oof_catboost_retained_trees": full_oof_trees,
            "final_residual_best_epoch": epoch,
            "final_normalizer": normalizer,
            "full_oof_training_nll": full_oof_nll,
            "ranking_unit_rms": rms(ranking_unit),
            "ranking_unit_abs_max": float(
                np.max(np.abs(ranking_unit))
            ),
            "ranking_realized_rms_logit_delta": rms(perturb),
            "ranking_realized_abs_max_logit_delta": float(
                np.max(np.abs(perturb))
            ),
        },
        "catboost": {
            "retained_trees": int(cat.tree_count_),
            "ranking": base_m,
        },
        "hybrid": {"ranking": hybrid_m},
        "deltas": {
            "hybrid_minus_catboost_nll": (
                hybrid_m["nll"] - base_m["nll"]
            ),
            "hybrid_minus_catboost_auc": (
                hybrid_m["auc"] - base_m["auc"]
            ),
        },
        "seconds": time.perf_counter() - started,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    )
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
