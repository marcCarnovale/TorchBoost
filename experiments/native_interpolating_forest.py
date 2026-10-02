"""Native overcomplete power-forest interpolation/compression experiment.

This is a mechanism study, not a production default.  The forest is deliberately
larger than necessary, diversified with row/feature subsampling and temporary
whole-tree dropout, then driven to a true interpolation checkpoint before
generic post-interpolation pressures are compared.  Audit is evaluated only for
the variant chosen on a disjoint ranking split.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import json
import numpy as np
import torch

from experiments.native_overcomplete_progressive import make_data, dropout_forward, metrics
from torchboost.adaptive.progressive import (
    ProgressiveConfig,
    ProgressiveTreeClassifier,
    _ensemble_regularization,
)


def build(seed):
    x, y, clean = make_data(7500, seed, train_noise=0.16)
    rows = {
        "train": np.arange(0, 2500),
        "control": np.arange(2500, 3500),
        "selection": np.arange(3500, 4500),
        "ranking": np.arange(4500, 5500),
        "audit": np.arange(5500, 7500),
    }
    cfg = ProgressiveConfig(
        n_trees=40,
        depth=6,
        stage_updates=24,
        batch_size=256,
        learning_rate=0.012,
        new_tree_shrinkage=0.50,
        old_tree_lr_decay=0.80,
        weight_decay=1e-5,
        cart_strength=7.0,
        leaf_l2=1e-6,
        learn_tree_rates=True,
        tree_rate_l2=1e-6,
        row_subsample=0.80,
        feature_subsample=0.80,
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
    return estimator, splits


def drive_to_interpolation(estimator, splits, seed, max_steps=1200):
    model = estimator.model_
    objective = estimator.objective_
    train = splits["train"]
    generator = torch.Generator().manual_seed(seed + 17001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0018, weight_decay=1e-6)
    trace = []
    reached = False
    reached_step = None
    for step in range(max_steps + 1):
        if step % 20 == 0:
            train_m = metrics(model, train, objective)
            selection_m = metrics(model, splits["selection"], objective)
            trace.append({
                "step": step,
                "train_nll": train_m["nll"],
                "train_accuracy": train_m["accuracy"],
                "selection_nll": selection_m["nll"],
                "effective_tree_count": selection_m["effective_tree_count"],
            })
            if train_m["accuracy"] >= 0.9999 and train_m["nll"] <= 0.04:
                reached = True
                reached_step = step
                break
        if step == max_steps:
            break
        idx = torch.randint(len(train.x), (384,), generator=generator)
        probability = 0.20 * max(0.0, 1.0 - step / 240.0)
        optimizer.zero_grad(set_to_none=True)
        raw = dropout_forward(model, train.x[idx], probability, generator)
        loss = objective.weighted_loss(raw, train.y[idx], train.weight[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
    return reached, reached_step, trace


def regularized_config(base, variant):
    cfg = deepcopy(base)
    cfg.leaf_l2 = 0.0
    cfg.depth_shrinkage = 0.0
    cfg.tree_l2 = 0.0
    cfg.tree_rate_l2 = 0.0
    cfg.tree_count_pressure = 0.0
    cfg.tree_count_start_stage = 0
    if variant == "l2":
        cfg.tree_rate_l2 = 0.05
    elif variant in ("count", "hierarchy", "hierarchy_dropout"):
        cfg.tree_rate_l2 = 0.005
        cfg.tree_count_pressure = 0.06
        if variant in ("hierarchy", "hierarchy_dropout"):
            cfg.leaf_l2 = 2e-4
            cfg.depth_shrinkage = 2e-3
    elif variant != "none":
        raise ValueError(variant)
    return cfg


def continue_variant(estimator, splits, variant, seed, steps=400):
    model = deepcopy(estimator.model_)
    objective = estimator.objective_
    cfg = regularized_config(estimator.config_, variant)
    train = splits["train"]
    control = splits["control"]
    generator = torch.Generator().manual_seed(seed + 23003)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=1e-6)
    wanted = {0, 50, 100, 200, 300, steps}
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
        if variant == "hierarchy_dropout":
            raw = dropout_forward(model, train.x[idx], 0.10, generator)
        else:
            raw = model(train.x[idx])
        loss = objective.weighted_loss(raw, train.y[idx], train.weight[idx])
        loss = loss + _ensemble_regularization(model, cfg, control.x, stage=len(model.trees))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
    return model, checkpoints


def run(seed=61):
    torch.set_num_threads(1)
    estimator, splits = build(seed)
    progressive_curve = [row["validation_loss"] for row in estimator.history_]
    reached, reached_step, interpolation_trace = drive_to_interpolation(estimator, splits, seed)
    branch_checkpoint = {
        name: metrics(estimator.model_, split, estimator.objective_)
        for name, split in splits.items()
        if name != "audit"
    }

    variants = {}
    models = {}
    for variant in ("none", "l2", "count", "hierarchy", "hierarchy_dropout"):
        model, checkpoints = continue_variant(estimator, splits, variant, seed)
        variants[variant] = {"checkpoints": checkpoints}
        models[variant] = model

    winner = min(
        variants,
        key=lambda name: variants[name]["checkpoints"]["400"]["ranking"]["nll"],
    )
    winner_audit = metrics(models[winner], splits["audit"], estimator.objective_)

    pre_values = progressive_curve + [row["selection_nll"] for row in interpolation_trace[:-1]]
    pre_min = min(pre_values) if pre_values else branch_checkpoint["selection"]["nll"]
    branch_sel = branch_checkpoint["selection"]["nll"]
    post_selection = [
        row["selection"]["nll"]
        for row in variants[winner]["checkpoints"].values()
    ]
    post_min = min(post_selection)
    second_descent = bool(
        reached
        and branch_checkpoint["train"]["accuracy"] >= 0.9999
        and branch_sel > 1.02 * pre_min
        and post_min < 0.98 * branch_sel
    )

    return {
        "seed": seed,
        "trees": len(estimator.model_.trees),
        "reached_interpolation": reached,
        "interpolation_step": reached_step,
        "progressive_selection_curve": progressive_curve,
        "interpolation_trace": interpolation_trace,
        "branch_checkpoint": branch_checkpoint,
        "variants": variants,
        "ranking_winner": winner,
        "winner_audit": winner_audit,
        "pre_interpolation_selection_min": pre_min,
        "post_interpolation_selection_min": post_min,
        "selection_curve_second_descent_pattern": second_descent,
        "claim_deep_double_descent": second_descent,
        "note": (
            "No task-specific regularizer is used. The claim flag requires an "
            "actual interpolation checkpoint, a worse selection region there "
            "than earlier in training, and a subsequent generic-reg second descent."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = run(args.seed)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(text)
    print(text)
