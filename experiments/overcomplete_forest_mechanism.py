"""Mechanism-isolation study for post-interpolation forest compression.

This deliberately does NOT claim to be the final TorchBoost forest.  It isolates
one hypothesis behind the planned overcomplete forest: redundant tree solutions
make interpolation easy; after interpolation, generic structural/parameter
regularization can move the predictor toward a lower-complexity decomposition
without any domain-specific prior.

The task is a generic noisy piecewise tabular rule.  The pool contains shallow
and fully-grown CART trees. Learned scalar front coefficients are optimized with
tree dropout before interpolation.  Post-interpolation variants compare:

  none      : predictive loss + tiny ordinary weight decay
  l2        : coefficient ridge only
  hierarchy : coefficient ridge + contribution shrinkage + generic tree
              structural complexity (log leaf count)

No Fourier, astronomy, target-rule, or problem-specific regularizer is used.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier


def make_rule(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 10)).astype("float32")
    score = np.where(
        x[:, 0] > 0,
        np.where(x[:, 1] > 0, 2.2, -1.8),
        np.where(x[:, 2] > 0.3, 1.7, -2.0),
    )
    score = score + 0.35 * x[:, 3]
    p = 1 / (1 + np.exp(-score))
    y = rng.binomial(1, p).astype("float32")
    return x, y


def tree_pool(x, y, train, valid, audit, seed):
    depths = [2] * 16 + [3] * 16 + [4] * 16 + [6] * 16 + [None] * 64
    matrices = [[], [], []]
    leaves = []
    for j, depth in enumerate(depths):
        tree = DecisionTreeClassifier(
            max_depth=depth,
            max_features=0.8,
            min_samples_leaf=1,
            random_state=seed * 100 + j,
        ).fit(x[train], y[train])
        leaves.append(tree.tree_.n_leaves)
        for bucket, rows in zip(matrices, (train, valid, audit)):
            p = np.clip(tree.predict_proba(x[rows])[:, 1], 0.01, 0.99)
            bucket.append(np.log(p / (1 - p)))
    return (
        [torch.tensor(np.stack(z, 1), dtype=torch.float32) for z in matrices],
        torch.tensor(leaves, dtype=torch.float32),
    )


def run(seed: int = 77, variant: str = "hierarchy", interpolation_steps: int = 500,
        total_steps: int = 1200):
    if variant not in ("none", "l2", "hierarchy"):
        raise ValueError("variant must be none, l2, or hierarchy")
    torch.set_num_threads(1)
    x, y = make_rule(9000, seed)
    train = np.arange(5000)
    valid = np.arange(5000, 7000)
    audit = np.arange(7000, 9000)
    (z_train, z_valid, z_audit), leaves = tree_pool(
        x, y, train, valid, audit, seed
    )
    y_train = torch.tensor(y[train], dtype=torch.float32)
    rms = z_train.square().mean(0).sqrt().clamp_min(1e-6)
    structural_cost = torch.log1p(leaves)
    structural_cost = structural_cost / structural_cost.mean()
    m = z_train.shape[1]

    torch.manual_seed(seed + 999)
    alpha = torch.nn.Parameter(torch.zeros(m))
    bias = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([alpha, bias], lr=0.03)

    checkpoints = {}
    wanted = {0, 1, 25, interpolation_steps, 600, 700, 900, total_steps}

    def snapshot(step):
        with torch.no_grad():
            contribution = alpha.abs() * rms
            total = contribution.sum()
            neff = (
                float(total.square() / contribution.square().sum().clamp_min(1e-12))
                if float(total) > 1e-12 else 0.0
            )

            def score(z, target):
                logits = bias + z @ alpha
                target_t = torch.tensor(target, dtype=torch.float32)
                return {
                    "nll": float(torch.nn.functional.binary_cross_entropy_with_logits(logits, target_t)),
                    "accuracy": float(((logits > 0).numpy() == target.astype(bool)).mean()),
                }

            return {
                "step": step,
                "train": score(z_train, y[train]),
                "selection": score(z_valid, y[valid]),
                "audit": score(z_audit, y[audit]),
                "effective_tree_count": neff,
                "shallow_contribution_mass": float(
                    contribution[:48].sum() / total.clamp_min(1e-12)
                ),
                "coefficient_l2": float(alpha.square().mean()),
            }

    for step in range(total_steps + 1):
        if step in wanted:
            checkpoints[step] = snapshot(step)
        if step == total_steps:
            break
        if step == interpolation_steps:
            optimizer = torch.optim.Adam([alpha, bias], lr=0.002)

        optimizer.zero_grad()
        dropout = 0.25 if step < interpolation_steps else 0.0
        if dropout:
            keep = (torch.rand(m) > dropout).float() / (1 - dropout)
        else:
            keep = torch.ones(m)
        logits = bias + z_train @ (alpha * keep)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        loss = loss + 1e-4 * alpha.square().mean()

        if step >= interpolation_steps:
            contribution = torch.sqrt((alpha * rms).square() + 1e-10)
            if variant in ("l2", "hierarchy"):
                loss = loss + 0.2 * alpha.square().mean()
            if variant == "hierarchy":
                # Generic post-interpolation pressure: contribution shrinkage
                # plus structural cost.  No target/domain semantics appear here.
                loss = loss + 0.01 * contribution.mean()
                loss = loss + 10.0 * (contribution * structural_cost).mean()

        loss.backward()
        optimizer.step()

    return {
        "seed": seed,
        "variant": variant,
        "interpolation_steps": interpolation_steps,
        "total_steps": total_steps,
        "checkpoints": checkpoints,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--variant", choices=["none", "l2", "hierarchy"], default="hierarchy")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = run(args.seed, args.variant)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(text)
    print(text)
