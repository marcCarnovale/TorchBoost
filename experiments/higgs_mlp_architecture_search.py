"""First evidence-driven architecture search round from the exact MLP anchor.

The fresh shadow audit is never opened.  The experiment trains the canonical
500k MLP-equivalent CompositionalTreeNetwork, restores the best selection
checkpoint, then gives matched pilot budgets to:

- exact MLP continuation control;
- last hidden layer -> zero residual tree specialization, newborn/routing only;
- last hidden layer -> zero residual tree specialization, full end-to-end;
- all hidden layers -> zero residual tree specialization, residual/routing only.

Every structural mutation is function preserving at birth.  Architecture
admission uses selection + ranking evidence against the matched continuation
control.  This is the first local step in a multidimensional search, not a
CatBoost<->MLP interpolation scalar.
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
from torchboost.adaptive.architecture_search import (
    ArchitectureArchive,
    ArchitectureCandidate,
    ArchitectureCoordinates,
    ArchitectureEvidence,
    TrainingBudget,
)

NTRAIN = 500_000
SEED = 509
ANCHOR_EPOCHS = 20
PILOT_EPOCHS = 2
BATCH = 4096


@torch.no_grad()
def probability(model, x, batch=8192):
    model.eval()
    chunks = []
    for start in range(0, len(x), batch):
        xb = torch.from_numpy(x[start:start + batch])
        chunks.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(chunks)


def _state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _train(
    model,
    train_x,
    train_y,
    selection_x,
    selection_y,
    *,
    epochs,
    order_seed,
    reset_dropout_seed=None,
):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("candidate has no trainable parameters")
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(order_seed)
    if reset_dropout_seed is not None:
        torch.manual_seed(reset_dropout_seed)

    best = (float("inf"), None, 0)
    history = []
    started = time.perf_counter()
    examples_seen = 0
    updates = 0
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
            examples_seen += len(idx)
            updates += 1
        p = probability(model, selection_x)
        row = {"epoch": epoch + 1, **metrics(selection_y, p)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["nll"] < best[0]:
            best = (row["nll"], _state(model), epoch + 1)
    model.load_state_dict(best[1])
    return {
        "history": history,
        "best_epoch": best[2],
        "seconds": time.perf_counter() - started,
        "examples_seen": examples_seen,
        "optimizer_updates": updates,
    }


def _freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def _enable_residual_tree(layer):
    tree = layer.forest.trees[0]
    root = layer.root
    # Preserve the inherited affine packet; specialize only through routing and
    # zero-at-birth residual children.
    root.value.requires_grad_(False)
    root.linear_value.requires_grad_(False)
    if root.routing_weight is not None:
        root.routing_weight.requires_grad_(True)
    if root.routing_bias is not None:
        root.routing_bias.requires_grad_(True)
    for child_id in root.children_ids:
        child = tree.get(child_id)
        child.value.requires_grad_(True)
        if child.linear_value is not None:
            child.linear_value.requires_grad_(True)
        child.allocation_logit.requires_grad_(False)


def mutate(anchor, variant):
    model = deepcopy(anchor)
    if variant == "control":
        return model

    if variant == "last_residual":
        layer = model.layers[-1]
        layer.grow_one_level()
        _freeze_all(model)
        _enable_residual_tree(layer)
        return model

    if variant == "last_full":
        model.layers[-1].grow_one_level()
        # grow_one_level already leaves the represented tree packet trainable,
        # the old dense copy frozen, earlier dense layers trainable, and the
        # output head trainable.
        return model

    if variant == "all_residual":
        for layer in model.layers:
            layer.grow_one_level()
        _freeze_all(model)
        for layer in model.layers:
            _enable_residual_tree(layer)
        return model

    raise ValueError(variant)


def coordinates(variant):
    base = ArchitectureCoordinates(
        composition="latent_sequential",
        tree_depth=0,
        tree_count=0,
        layer_count=5,
        hidden_width=300,
        routing_hardness="none",
        routing_geometry="none",
        packet_type="affine",
        construction="gradient",
        aggregation="sequential",
        optimization_scope="full",
        proposal_subsample=1.0,
        refinement_subsample=1.0,
        global_polish=True,
    )
    if variant == "control":
        return base
    if variant == "last_residual":
        return base.mutate(
            tree_depth=1,
            tree_count=1,
            routing_hardness="soft",
            routing_geometry="oblique",
            optimization_scope="newborn_residual",
            global_polish=False,
        )
    if variant == "last_full":
        return base.mutate(
            tree_depth=1,
            tree_count=1,
            routing_hardness="soft",
            routing_geometry="oblique",
            optimization_scope="full",
            global_polish=True,
        )
    if variant == "all_residual":
        return base.mutate(
            tree_depth=1,
            tree_count=5,
            routing_hardness="soft",
            routing_geometry="oblique",
            optimization_scope="newborn_residual",
            global_polish=False,
        )
    raise ValueError(variant)


def run(csv_gz, cache, out, checkpoint_dir, seed=SEED, pilot_epochs=PILOT_EPOCHS):
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

    # Match the frozen canonical MLP initialization AND its dropout RNG state.
    torch.manual_seed(family_seed + 305)
    reference = MLP(LOW_FEATURES, 300, 5, .1)
    canonical_training_rng = torch.get_rng_state()
    anchor = CompositionalTreeNetwork.from_mlp(
        reference, max_tree_depth=3, seed=family_seed + 1200
    )
    torch.set_rng_state(canonical_training_rng)

    initial_probe = torch.from_numpy(train_x[:4096])
    reference.eval(); anchor.eval()
    with torch.no_grad():
        initial_diff = float((reference(initial_probe) - anchor(initial_probe)).abs().max())
    if initial_diff > 2e-6:
        raise RuntimeError(f"neural-anchor initial mismatch: {initial_diff}")

    anchor_train = _train(
        anchor,
        train_x,
        train_y,
        selection_x,
        splits["selection"][1],
        epochs=ANCHOR_EPOCHS,
        order_seed=family_seed + 9001,
        reset_dropout_seed=None,
    )
    anchor_selection = metrics(
        splits["selection"][1], probability(anchor, selection_x)
    )
    anchor_ranking = metrics(
        splits["ranking"][1], probability(anchor, ranking_x)
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": anchor.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "family_seed": family_seed,
            "selection": anchor_selection,
            "ranking": anchor_ranking,
        },
        checkpoint_dir / "neural-anchor.pt",
    )

    archive = ArchitectureArchive(
        selection_tolerance=0.0,
        ranking_nll_tolerance=0.0,
        ranking_auc_tolerance=0.0,
    )
    anchor_budget = TrainingBudget(
        optimizer_updates=anchor_train["optimizer_updates"],
        examples_seen=anchor_train["examples_seen"],
        full_data_passes=anchor_train["examples_seen"] / len(train_x),
        wall_seconds=anchor_train["seconds"],
    )
    archive.add(ArchitectureCandidate(
        "anchor",
        coordinates("control"),
        evidence=ArchitectureEvidence(
            selection_nll=anchor_selection["nll"],
            ranking_nll=anchor_ranking["nll"],
            ranking_auc=anchor_ranking["auc"],
            trainable_parameters=sum(p.numel() for p in anchor.parameters() if p.requires_grad),
            budget=anchor_budget,
        ),
        admitted=True,
    ))

    variants = ("control", "last_residual", "last_full", "all_residual")
    rows = {}
    pilot_order_seed = family_seed + 19001
    pilot_dropout_seed = family_seed + 19017

    # Give all children of the anchor the same order/dropout seed and exposure.
    for variant in variants:
        model = mutate(anchor, variant)
        before = probability(model, selection_x[:8192])
        parent_before = probability(anchor, selection_x[:8192])
        birth_diff = float(np.max(np.abs(before - parent_before)))
        if birth_diff > 3e-6:
            raise RuntimeError(f"{variant} is not function preserving at birth: {birth_diff}")

        trained = _train(
            model,
            train_x,
            train_y,
            selection_x,
            splits["selection"][1],
            epochs=pilot_epochs,
            order_seed=pilot_order_seed,
            reset_dropout_seed=pilot_dropout_seed,
        )
        sel = metrics(splits["selection"][1], probability(model, selection_x))
        rank = metrics(splits["ranking"][1], probability(model, ranking_x))
        budget = TrainingBudget(
            optimizer_updates=trained["optimizer_updates"],
            examples_seen=trained["examples_seen"],
            full_data_passes=trained["examples_seen"] / len(train_x),
            wall_seconds=trained["seconds"],
        )
        candidate = ArchitectureCandidate(
            variant,
            coordinates(variant),
            parent_id="anchor",
            mutation=variant,
            evidence=ArchitectureEvidence(
                selection_nll=sel["nll"],
                ranking_nll=rank["nll"],
                ranking_auc=rank["auc"],
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                budget=budget,
            ),
            metadata={"birth_max_probability_diff": birth_diff},
        )
        archive.add(candidate)
        torch.save(
            {"model_state": model.state_dict(), "selection": sel, "ranking": rank},
            checkpoint_dir / f"{variant}.pt",
        )
        rows[variant] = {
            "selection": sel,
            "ranking": rank,
            "trainable_parameters": candidate.evidence.trainable_parameters,
            "budget": {
                "optimizer_updates": budget.optimizer_updates,
                "examples_seen": budget.examples_seen,
                "full_data_passes": budget.full_data_passes,
                "seconds": budget.wall_seconds,
            },
            "birth_max_probability_diff": birth_diff,
            "history": trained["history"],
        }

    decisions = {}
    admitted = []
    for variant in variants:
        if variant == "control":
            continue
        decision = archive.compare_to_control(variant, "control")
        decisions[variant] = {
            "admitted": decision.admitted,
            "selection_delta_vs_control": decision.selection_delta,
            "ranking_nll_delta_vs_control": decision.ranking_nll_delta,
            "ranking_auc_delta_vs_control": decision.ranking_auc_delta,
            "reason": decision.reason,
        }
        if decision.admitted:
            admitted.append(variant)

    best = None
    if admitted:
        best = min(
            admitted,
            key=lambda v: (
                rows[v]["selection"]["nll"],
                rows[v]["ranking"]["nll"],
                -rows[v]["ranking"]["auc"],
            ),
        )

    result = {
        "status": "completed",
        "seed": seed,
        "family_seed": family_seed,
        "ntrain": NTRAIN,
        "source": source,
        "audit_opened": False,
        "shadow_audit_opened": False,
        "protocol": "experiments/higgs_shadow_protocol.json",
        "initial_max_logit_diff": initial_diff,
        "anchor": {
            "selection": anchor_selection,
            "ranking": anchor_ranking,
            "best_epoch": anchor_train["best_epoch"],
            "trainable_parameters": sum(p.numel() for p in anchor.parameters() if p.requires_grad),
            "seconds": anchor_train["seconds"],
        },
        "pilot_epochs": pilot_epochs,
        "variants": rows,
        "decisions": decisions,
        "admitted": admitted,
        "best_admitted": best,
        "pareto_archive": archive.non_dominated(),
    }
    Path(out).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def smoke(seed=7):
    # Controller-level smoke only; HIGGS data is intentionally not synthesized
    # into a fake research result.
    rng = np.random.default_rng(seed)
    a = rng.normal(size=4)
    return {
        "status": "completed",
        "seed": seed,
        "audit_opened": False,
        "shadow_audit_opened": False,
        "finite": bool(np.isfinite(a).all()),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv-gz")
    p.add_argument("--cache", default="/tmp/higgs-cache")
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint-dir", default="research-results/architecture-search-checkpoints")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--pilot-epochs", type=int, default=PILOT_EPOCHS)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        answer = smoke(a.seed)
        Path(a.out).write_text(json.dumps(answer, indent=2, sort_keys=True))
    else:
        if not a.csv_gz:
            p.error("--csv-gz is required unless --smoke")
        answer = run(
            a.csv_gz,
            a.cache,
            a.out,
            a.checkpoint_dir,
            a.seed,
            a.pilot_epochs,
        )
    print(json.dumps(answer, indent=2, sort_keys=True, allow_nan=False))
