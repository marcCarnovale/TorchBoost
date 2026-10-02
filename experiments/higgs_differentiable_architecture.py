"""Differentiable architecture discovery from the exact canonical MLP anchor.

This is the primary superiority experiment.  It is not an outer-loop choice
between hand-enumerated architecture arms.

Protocol:
1. train the exact canonical 500k MLP-equivalent TorchBoost endpoint;
2. grow one zero-output residual tree level in every hidden layer;
3. preserve the anchor function exactly at birth;
4. train predictive weights on TRAIN;
5. train continuous architecture gates on SELECTION with a complexity
   Lagrangian;
6. evaluate the learned architecture once on RANKING;
7. never open the fresh shadow audit.

An unchanged MLP continuation receives the same number of full-data epochs as a
control.  Thus improvements distinguish differentiable architecture discovery
from merely training the MLP longer.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from experiments.higgs_canonical_scaling import (
    LOW_FEATURES,
    arrays,
    fixed_splits,
    materialize,
    metrics,
)
from experiments.higgs_hybrid_benchmark import MLP
from torchboost.adaptive.architecture_corners import CompositionalTreeNetwork
from torchboost.adaptive.architecture_regularization import (
    ArchitectureRegularization,
    architecture_state,
    differentiable_architecture_penalty,
)

NTRAIN = 500_000
SEED = 509
ANCHOR_EPOCHS = 20
DISCOVERY_EPOCHS = 5
WARMUP_EPOCHS = 2
BATCH = 4096
ARCH_EVERY = 2


@torch.no_grad()
def probability(model, x, batch=8192):
    model.eval()
    chunks = []
    for start in range(0, len(x), batch):
        chunks.append(
            torch.sigmoid(model(torch.from_numpy(x[start:start + batch]))).numpy()
        )
    return np.concatenate(chunks)


def _state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def train_anchor(model, train_x, train_y, selection_x, selection_y, seed):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed + 9001)

    best = (float("inf"), None, 0)
    history = []
    started = time.perf_counter()
    for epoch in range(ANCHOR_EPOCHS):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH].numpy()
            xb = torch.from_numpy(train_x[idx])
            yb = torch.from_numpy(train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()

        sel = metrics(selection_y, probability(model, selection_x))
        history.append({"epoch": epoch + 1, **sel})
        print(json.dumps({"phase": "anchor", **history[-1]}), flush=True)
        if sel["nll"] < best[0]:
            best = (sel["nll"], _state(model), epoch + 1)

    model.load_state_dict(best[1])
    return {
        "best_epoch": best[2],
        "history": history,
        "seconds": time.perf_counter() - started,
    }


def continue_mlp(model, train_x, train_y, selection_x, selection_y, *, epochs, seed):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)

    best = (float("inf"), None, 0)
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH].numpy()
            xb = torch.from_numpy(train_x[idx])
            yb = torch.from_numpy(train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
        sel = metrics(selection_y, probability(model, selection_x))
        history.append({"epoch": epoch + 1, **sel})
        if sel["nll"] < best[0]:
            best = (sel["nll"], _state(model), epoch + 1)
    model.load_state_dict(best[1])
    return {
        "best_epoch": best[2],
        "history": history,
        "seconds": time.perf_counter() - started,
        "examples_seen": epochs * len(train_x),
    }


def build_supernet(anchor):
    model = deepcopy(anchor)
    for layer in model.layers:
        layer.grow_one_level()
    return model


def parameter_partition(model):
    architecture = []
    weights = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("architecture_logit"):
            architecture.append(p)
        else:
            weights.append(p)
    if not architecture or not weights:
        raise RuntimeError("differentiable supernet needs weight and architecture parameters")
    return weights, architecture


def discover(
    model,
    train_x,
    train_y,
    selection_x,
    selection_y,
    *,
    epochs,
    warmup_epochs,
    seed,
    regularization,
):
    weight_params, architecture_params = parameter_partition(model)
    weight_opt = torch.optim.AdamW(weight_params, lr=1e-3, weight_decay=1e-5)
    architecture_opt = torch.optim.Adam(
        architecture_params, lr=1e-2, weight_decay=0.0
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train_rng = torch.Generator().manual_seed(seed)
    selection_rng = torch.Generator().manual_seed(seed + 97)

    best = (float("inf"), None, 0, None)
    history = []
    started = time.perf_counter()
    train_examples = 0
    selection_examples = 0
    architecture_updates = 0

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=train_rng)
        selection_order = torch.randperm(len(selection_x), generator=selection_rng)
        selection_cursor = 0

        for batch_index, start in enumerate(range(0, len(order), BATCH)):
            idx = order[start:start + BATCH].numpy()
            xb = torch.from_numpy(train_x[idx])
            yb = torch.from_numpy(train_y[idx])

            weight_opt.zero_grad(set_to_none=True)
            architecture_opt.zero_grad(set_to_none=True)
            prediction = model(xb)
            pred_loss = loss_fn(prediction, yb)
            # Predictive parameters must first get a fair chance to make the
            # zero-at-birth residual specialists useful. Complexity is an
            # architecture-selection charge, not a shrinkage term on every
            # training-weight step.
            pred_loss.backward()
            torch.nn.utils.clip_grad_norm_(weight_params, 10.0)
            weight_opt.step()
            train_examples += len(idx)

            if epoch >= warmup_epochs and (batch_index + 1) % ARCH_EVERY == 0:
                if selection_cursor + BATCH > len(selection_order):
                    selection_order = torch.randperm(
                        len(selection_x), generator=selection_rng
                    )
                    selection_cursor = 0
                sidx = selection_order[selection_cursor:selection_cursor + BATCH].numpy()
                selection_cursor += BATCH
                sx = torch.from_numpy(selection_x[sidx])
                sy = torch.from_numpy(np.asarray(selection_y[sidx], dtype="float32"))

                weight_opt.zero_grad(set_to_none=True)
                architecture_opt.zero_grad(set_to_none=True)
                selection_pred = model(sx)
                selection_loss = loss_fn(selection_pred, sy)
                architecture_penalty, _ = differentiable_architecture_penalty(
                    model, regularization
                )
                # No sparsity pressure during warm-up.  After warm-up, ramp
                # the full complexity Lagrangian in gradually so a useful
                # residual can establish a held-out signal before being taxed.
                progress = (epoch - warmup_epochs + 1) / max(
                    1, epochs - warmup_epochs
                )
                penalty_scale = min(1.0, max(0.0, progress))
                architecture_loss = selection_loss + penalty_scale * architecture_penalty
                architecture_loss.backward()
                torch.nn.utils.clip_grad_norm_(architecture_params, 2.0)
                architecture_opt.step()
                selection_examples += len(sidx)
                architecture_updates += 1

        sel = metrics(selection_y, probability(model, selection_x))
        penalty, components = differentiable_architecture_penalty(model, regularization)
        row = {
            "epoch": epoch + 1,
            "selection": sel,
            "architecture": architecture_state(model),
            "penalty": float(penalty.detach()),
            "penalty_scale": (
                0.0 if epoch < warmup_epochs else min(
                    1.0,
                    max(
                        0.0,
                        (epoch - warmup_epochs + 1)
                        / max(1, epochs - warmup_epochs),
                    ),
                )
            ),
            "penalty_components": {
                k: float(v.detach()) for k, v in components.items()
            },
        }
        history.append(row)
        print(json.dumps({"phase": "discovery", **row}), flush=True)
        if sel["nll"] < best[0]:
            best = (
                sel["nll"],
                _state(model),
                epoch + 1,
                architecture_state(model),
            )

    model.load_state_dict(best[1])
    return {
        "best_epoch": best[2],
        "best_architecture": best[3],
        "history": history,
        "seconds": time.perf_counter() - started,
        "train_examples_seen": train_examples,
        "selection_examples_seen_by_architecture": selection_examples,
        "architecture_updates": architecture_updates,
    }


def run(csv_gz, cache, out, checkpoint_dir, seed=SEED):
    torch.set_num_threads(4)
    x_path, y_path, source = materialize(Path(csv_gz), Path(cache))
    x, y = arrays(x_path, y_path)
    splits = fixed_splits(x, y, NTRAIN)
    family_seed = seed + NTRAIN % 10007

    scaler = StandardScaler().fit(splits["train"][0])
    train_x = scaler.transform(splits["train"][0]).astype("float32")
    selection_x = scaler.transform(splits["selection"][0]).astype("float32")
    ranking_x = scaler.transform(splits["ranking"][0]).astype("float32")
    train_y = np.asarray(splits["train"][1], dtype="float32")

    torch.manual_seed(family_seed + 305)
    reference = MLP(LOW_FEATURES, 300, 5, .1)
    canonical_training_rng = torch.get_rng_state()
    anchor = CompositionalTreeNetwork.from_mlp(
        reference, max_tree_depth=3, seed=family_seed + 1200
    )
    torch.set_rng_state(canonical_training_rng)

    anchor_training = train_anchor(
        anchor,
        train_x,
        train_y,
        selection_x,
        splits["selection"][1],
        family_seed,
    )
    anchor_selection = metrics(splits["selection"][1], probability(anchor, selection_x))
    anchor_ranking = metrics(splits["ranking"][1], probability(anchor, ranking_x))

    control = deepcopy(anchor)
    supernet = build_supernet(anchor)

    # Growth is exact at birth even though gates are already differentiable:
    # every residual child is initialized to the zero function.
    probe = selection_x[:8192]
    birth_diff = float(
        np.max(np.abs(probability(anchor, probe) - probability(supernet, probe)))
    )
    if birth_diff > 3e-6:
        raise RuntimeError(f"supernet growth changed anchor function: {birth_diff}")

    discovery_seed = family_seed + 19001
    control_training = continue_mlp(
        control,
        train_x,
        train_y,
        selection_x,
        splits["selection"][1],
        epochs=DISCOVERY_EPOCHS,
        seed=discovery_seed,
    )

    regularization = ArchitectureRegularization(
        # Residual scale is itself the architecture variable we want held-out
        # loss to discover. Do not directly tax gate openness; charge only for
        # the complexity of the residual/routing machinery it actually uses.
        gate_l1=0.0,
        gate_entropy=0.0,
        residual_l2=2e-7,
        routing_l1=5e-7,
    )
    discovery = discover(
        supernet,
        train_x,
        train_y,
        selection_x,
        splits["selection"][1],
        epochs=DISCOVERY_EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        seed=discovery_seed,
        regularization=regularization,
    )

    control_selection = metrics(
        splits["selection"][1], probability(control, selection_x)
    )
    control_ranking = metrics(
        splits["ranking"][1], probability(control, ranking_x)
    )
    learned_selection = metrics(
        splits["selection"][1], probability(supernet, selection_x)
    )
    learned_ranking = metrics(
        splits["ranking"][1], probability(supernet, ranking_x)
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": anchor.state_dict()}, checkpoint_dir / "anchor.pt")
    torch.save({"state": control.state_dict()}, checkpoint_dir / "control.pt")
    torch.save(
        {
            "state": supernet.state_dict(),
            "architecture": architecture_state(supernet),
        },
        checkpoint_dir / "differentiable-supernet.pt",
    )

    result = {
        "status": "completed",
        "source": source,
        "seed": seed,
        "family_seed": family_seed,
        "ntrain": NTRAIN,
        "audit_opened": False,
        "shadow_audit_opened": False,
        "protocol": "experiments/higgs_shadow_protocol.json",
        "birth_max_probability_diff": birth_diff,
        "anchor": {
            "selection": anchor_selection,
            "ranking": anchor_ranking,
            "best_epoch": anchor_training["best_epoch"],
            "seconds": anchor_training["seconds"],
        },
        "control": {
            "selection": control_selection,
            "ranking": control_ranking,
            **control_training,
        },
        "differentiable_discovery": {
            "selection": learned_selection,
            "ranking": learned_ranking,
            "architecture": architecture_state(supernet),
            "regularization": vars(regularization),
            **discovery,
        },
        "deltas": {
            "selection_nll_vs_control":
                learned_selection["nll"] - control_selection["nll"],
            "ranking_nll_vs_control":
                learned_ranking["nll"] - control_ranking["nll"],
            "ranking_auc_vs_control":
                learned_ranking["auc"] - control_ranking["auc"],
            "ranking_nll_vs_anchor":
                learned_ranking["nll"] - anchor_ranking["nll"],
            "ranking_auc_vs_anchor":
                learned_ranking["auc"] - anchor_ranking["auc"],
        },
    }
    Path(out).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv-gz", required=True)
    p.add_argument("--cache", default="/tmp/higgs-cache")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--checkpoint-dir",
        default="research-results/differentiable-architecture-checkpoints",
    )
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    answer = run(a.csv_gz, a.cache, a.out, a.checkpoint_dir, a.seed)
    print(json.dumps(answer, indent=2, sort_keys=True, allow_nan=False))
