"""TorchBoost-native overcomplete-forest post-interpolation study.

Actual differentiable power trees are deliberately overprovided. Row/feature
subsampling and a joint tree-dropout phase create redundant solution paths.
From one shared interpolating checkpoint, generic regularizers compete without
any task-specific prior. Audit is touched only after a winner is chosen on a
separate ranking split.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import json
import numpy as np
import torch

from torchboost.adaptive.progressive import (
    ProgressiveConfig,
    ProgressiveTreeClassifier,
    _contribution_count_stats,
    _ensemble_regularization,
    _realized_tree_contributions,
)


def make_data(n, seed, train_noise=0.12):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 12)).astype("float32")
    score = np.where(
        x[:, 0] > 0,
        1.8 * x[:, 1] - 1.2 * x[:, 2] + 0.5 * x[:, 5],
        -1.5 * x[:, 3] + 1.1 * x[:, 4] - 0.4 * x[:, 6],
    )
    score += 0.7 * np.sin(1.5 * x[:, 7]) + 0.35 * x[:, 8] * x[:, 9]
    clean = (score > 0).astype(int)
    y = clean.copy()
    flip = rng.random(5000) < train_noise
    y[:5000] = np.where(flip, 1 - y[:5000], y[:5000])
    return x, y, clean


def dropout_forward(model, x, probability, generator):
    out = model.bias.expand(len(x), -1)
    if probability:
        keep = (torch.rand(len(model.trees), generator=generator) > probability).to(model.rates)
        keep = keep / (1 - probability)
    else:
        keep = model.rates.new_ones(len(model.trees))
    for mask, rate, tree in zip(keep, model.rates, model.trees):
        out = out + mask * rate * tree(x)
    return out


def metrics(model, split, objective):
    with torch.no_grad():
        raw = model(split.x)
        loss = float(objective.weighted_loss(raw, split.y, split.weight))
        probability = objective.response(raw)
        prediction = probability.argmax(1)
        contribution = _realized_tree_contributions(model, split.x)
        effective, entropy = _contribution_count_stats(contribution)
    return {
        "nll": loss,
        "accuracy": float((prediction == split.y).float().mean()),
        "effective_tree_count": float(effective),
        "entropy_tree_count": float(entropy),
        "tree_contribution_rms": contribution.tolist(),
        "rates": model.rates.detach().tolist(),
    }


def build_base(seed):
    x, y, clean = make_data(10500, seed)
    rows = {
        "train": np.arange(0, 5000),
        "control": np.arange(5000, 6250),
        "selection": np.arange(6250, 7500),
        "ranking": np.arange(7500, 8750),
        "audit": np.arange(8750, 10500),
    }
    cfg = ProgressiveConfig(
        n_trees=24,
        depth=5,
        stage_updates=24,
        batch_size=256,
        learning_rate=0.012,
        new_tree_shrinkage=0.45,
        old_tree_lr_decay=0.75,
        weight_decay=1e-5,
        cart_strength=7.0,
        leaf_l2=1e-6,
        learn_tree_rates=True,
        tree_rate_l2=1e-6,
        row_subsample=0.8,
        feature_subsample=0.8,
        cart_value_updates=8,
        patience_stages=100,
        min_improvement=-1e6,
        random_state=seed,
    )
    estimator = ProgressiveTreeClassifier(cfg).fit(
        x[rows["train"]],
        y[rows["train"]],
        eval_set=(x[rows["selection"]], clean[rows["selection"]]),
    )
    splits = {
        name: estimator.preprocessor_.split(
            x[idx], y[idx] if name == "train" else clean[idx]
        )
        for name, idx in rows.items()
    }

    # Before branching the comparison, jointly adapt the entire overcomplete
    # forest under stochastic tree deletion. This encourages multiple usable
    # representations rather than a single brittle decomposition.
    model = estimator.model_
    objective = estimator.objective_
    generator = torch.Generator().manual_seed(seed + 7001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    train = splits["train"]
    for _ in range(160):
        idx = torch.randint(len(train.x), (384,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        raw = dropout_forward(model, train.x[idx], 0.20, generator)
        loss = objective.weighted_loss(raw, train.y[idx], train.weight[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
    return estimator, splits


def variant_config(base, variant):
    cfg = deepcopy(base)
    cfg.leaf_l2 = 0.0
    cfg.depth_shrinkage = 0.0
    cfg.tree_l2 = 0.0
    cfg.tree_rate_l2 = 0.0
    cfg.tree_count_pressure = 0.0
    cfg.tree_count_start_stage = 0
    if variant == "l2":
        cfg.tree_rate_l2 = 0.08
    elif variant == "count":
        cfg.tree_rate_l2 = 0.01
        cfg.tree_count_pressure = 0.08
    elif variant == "hierarchy":
        cfg.tree_rate_l2 = 0.01
        cfg.tree_count_pressure = 0.08
        cfg.leaf_l2 = 2e-4
        cfg.depth_shrinkage = 2e-3
    elif variant != "none":
        raise ValueError(variant)
    return cfg


def continue_variant(estimator, splits, variant, seed, steps=320):
    model = deepcopy(estimator.model_)
    objective = estimator.objective_
    cfg = variant_config(estimator.config_, variant)
    train, control = splits["train"], splits["control"]
    generator = torch.Generator().manual_seed(seed + 11003)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-5)
    wanted = {0, 20, 50, 100, 160, 240, steps}
    checkpoints = {}
    for step in range(steps + 1):
        if step in wanted:
            checkpoints[str(step)] = {
                name: metrics(model, split, objective)
                for name, split in splits.items()
                if name != "audit"
            }
        if step == steps:
            break
        idx = torch.randint(len(train.x), (384,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = objective.weighted_loss(
            model(train.x[idx]), train.y[idx], train.weight[idx]
        )
        loss = loss + _ensemble_regularization(
            model, cfg, control.x, stage=len(model.trees)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
    return model, checkpoints


def run(seed=41):
    torch.set_num_threads(1)
    estimator, splits = build_base(seed)
    interpolation = {
        name: metrics(estimator.model_, split, estimator.objective_)
        for name, split in splits.items()
        if name != "audit"
    }
    variants = {}
    models = {}
    for variant in ("none", "l2", "count", "hierarchy"):
        model, checkpoints = continue_variant(estimator, splits, variant, seed)
        variants[variant] = {"checkpoints": checkpoints}
        models[variant] = model

    winner = min(
        variants,
        key=lambda name: variants[name]["checkpoints"]["320"]["ranking"]["nll"],
    )
    winner_audit = metrics(models[winner], splits["audit"], estimator.objective_)

    # Strict descriptive criterion: the pre-interpolation selection curve must
    # first attain a lower value, then be worse at the interpolating checkpoint,
    # and the selected post-interpolation trajectory must improve again.
    pre_curve = [row["validation_loss"] for row in estimator.history_]
    pre_min = min(pre_curve)
    interp_sel = interpolation["selection"]["nll"]
    post_sel = variants[winner]["checkpoints"]["320"]["selection"]["nll"]
    second_descent = bool(
        interpolation["train"]["accuracy"] >= 0.999
        and interp_sel > 1.02 * pre_min
        and post_sel < 0.98 * interp_sel
    )

    return {
        "seed": seed,
        "trees": len(estimator.model_.trees),
        "pre_interpolation_selection_curve": pre_curve,
        "interpolation_checkpoint": interpolation,
        "variants": variants,
        "ranking_winner": winner,
        "winner_audit": winner_audit,
        "selection_curve_second_descent_pattern": second_descent,
        "claim_deep_double_descent": second_descent,
        "note": (
            "No domain-specific penalty is used. Deep double descent is reported "
            "only if an interpolating checkpoint follows a worse-than-earlier "
            "selection region and generic post-interpolation optimization then "
            "improves selection again."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = run(args.seed)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(text)
    print(text)
