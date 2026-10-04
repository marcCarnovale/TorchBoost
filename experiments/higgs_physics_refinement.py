"""Physics refinement of a boosted/differentiable HIGGS-style backbone.

The warm model is a ProgressiveTreeClassifier with learned front coefficients.
It is then materialized exactly into native nodes and every continuation arm
starts from the same snapshot. All arms share the same generic structural,
parameter, tree-contribution, dropout, and coefficient regularization. Adaptive
arms differ only in plastic/physical control.

The OpenML 98,050-row proxy is development-only. Canonical claims require the
11M UCI HIGGS data and its final 500k test rows.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.datasets import fetch_openml, make_classification
from sklearn.metrics import log_loss, roc_auc_score

from experiments.higgs_hybrid_benchmark import _finite_impute
from experiments.normalized_energy_controller import EnergySource, NormalizedEnergyController
from torchboost.adaptive.config import PhysicsConfig, PlasticityConfig
from torchboost.adaptive.progressive import ProgressiveConfig, ProgressiveTreeClassifier
from torchboost.adaptive.progressive_regularizers import Regularizers
from torchboost.adaptive.rated_forest import materialize_progressive_sum
from torchboost.adaptive.rated_training import RatedJointTrainer, RatedRegularizers
from torchboost.adaptive.training import model_snapshot, restore_model


ARMS = ("none", "plastic", "direct", "capacitor", "rlc")


def load_data(seed, smoke=False):
    if smoke:
        x, y = make_classification(
            n_samples=3200, n_features=28, n_informative=20, n_redundant=5,
            class_sep=.8, random_state=seed,
        )
        x = x.astype("float32")
        counts = dict(train=1800, control=350, selection=350, ranking=350, audit=350)
        name = "synthetic-smoke"
    else:
        bunch = fetch_openml(data_id=23512, as_frame=False, parser="auto")
        x = np.asarray(bunch.data, dtype="float32")
        raw = np.asarray(bunch.target)
        labels = np.unique(raw)
        if len(labels) != 2:
            raise ValueError("HIGGS proxy must be binary")
        y = (raw == labels[-1]).astype("int64")
        counts = dict(train=60050, control=8000, selection=10000, ranking=10000, audit=10000)
        name = "OpenML HIGGS proxy 98,050 rows; low-level 21-feature condition"
    order = np.random.default_rng(seed + 771).permutation(len(x))
    x, y = x[order, :21], np.asarray(y)[order]
    if sum(counts.values()) != len(x):
        raise ValueError("physics protocol must account for every row exactly")
    pieces, start = {}, 0
    xs = []
    roles = list(counts)
    for role in roles:
        n = counts[role]
        pieces[role] = [x[start:start+n], y[start:start+n]]
        xs.append(pieces[role][0])
        start += n
    fixed = _finite_impute(*xs)
    for role, xx in zip(roles, fixed):
        pieces[role][0] = xx
    hashes = {
        role: hashlib.sha256(a[0].tobytes() + a[1].tobytes()).hexdigest()
        for role, a in pieces.items()
    }
    return name, pieces, hashes


def warm_config(seed, smoke=False):
    return ProgressiveConfig(
        n_trees=4 if smoke else 24,
        depth=3 if smoke else 5,
        stage_updates=4 if smoke else 24,
        batch_size=128 if smoke else 512,
        learning_rate=.01,
        new_tree_shrinkage=.30,
        old_tree_lr_decay=.80,
        weight_decay=1e-5,
        cart_strength=7.,
        leaf_l2=1e-6,
        depth_shrinkage=2e-4,
        learn_tree_rates=True,
        tree_rate_l2=1e-5,
        row_subsample=.85,
        feature_subsample=.90,
        readout="residual",
        cart_value_updates=2 if smoke else 8,
        patience_stages=4 if smoke else 10,
        random_state=seed,
    )


def fork_config(base, kind, epochs, smoke=False):
    cfg = deepcopy(base)
    cfg.epochs = epochs
    cfg.batch_size = 128 if smoke else 512
    cfg.learning_rate = .0025
    cfg.weight_decay = 1e-5
    cfg.collect_metrics = True
    cfg.compact_history = True
    cfg.observation_every = 1
    cfg.control_sample_size = 256 if smoke else 2048
    cfg.feature_dropout = .01
    cfg.tree_dropout = .03
    cfg.structure.dynamic = False
    cfg.structure.structural_gate = False
    cfg.structure.complexity = 0.
    cfg.structure.allocation_regularization = 0.
    mode = {"none": "none", "plastic": "none", "direct": "cooling",
            "capacitor": "capacitor", "rlc": "rlc"}[kind]
    cfg.physics = PhysicsConfig(
        mode=mode,
        topology_normalization=True,
        allocation="gradient",
        capacitance=1.,
        discharge_time=2.,
        inductive_time=1.,
        cooling_time=8.,
        total_heat_capacity=.12,
        dt=.2,
        initial_temperature=1.,
        ambient_temperature=1.,
        thaw_temperature=1.08,
        max_temperature=1.5,
        max_charge=2.,
        lr_coupling=0.,
    )
    cfg.plasticity = PlasticityConfig()
    if kind != "none":
        cfg.plasticity = PlasticityConfig(
            mode="full",
            stiffness=.02,
            yield_threshold=.001,
            mobility=.008,
            work_hardening=.08,
            consolidation_rate=.02,
            thermal_softening=(12.5 if kind in ("direct", "capacitor", "rlc") else 0.),
            release_policy="persistent_harm",
            release_patience=3,
        )
    cfg.__post_init__()
    return cfg


def generic_regularizers():
    return RatedRegularizers(
        structural=Regularizers(
            leaf_l2=1e-6,
            hierarchy=2e-5,
            depth_slope=.5,
            tree_l2=1e-6,
            feature_l1=1e-7,
        ),
        rate_l2=1e-6,
        count_pressure=2e-4,
    )


def metric(y, p):
    return {"nll": float(log_loss(y, p, labels=[0, 1])),
            "auc": float(roc_auc_score(y, p))}


def run(seed=401, epochs=4, smoke=False, arms=ARMS):
    if not set(arms) <= set(ARMS) or "none" not in arms:
        raise ValueError("predeclared known arms and no-control reference required")
    torch.set_num_threads(1 if smoke else 4)
    name, raw, hashes = load_data(seed, smoke)
    warm = ProgressiveTreeClassifier(warm_config(seed, smoke)).fit(
        raw["train"][0], raw["train"][1],
        eval_set=(raw["selection"][0], raw["selection"][1]),
    )
    native = materialize_progressive_sum(warm.model_, learn_rates=True)
    warm_snapshot = model_snapshot(native)
    splits = {
        role: warm.preprocessor_.split(x, y)
        for role, (x, y) in raw.items()
    }
    with torch.no_grad():
        exact = torch.max(torch.abs(
            warm.model_(splits["ranking"].x)
            - native(splits["ranking"].x)
        )).item()
    if exact > 5e-6:
        raise RuntimeError(f"packed/native warm-start mismatch: {exact}")
    source = EnergySource(rate=.5, threshold=.5, clip=4., smoothing=.9, warmup=2 if smoke else 4)
    reg = generic_regularizers()
    result = {
        "status": "running",
        "dataset": name,
        "seed": seed,
        "epochs": epochs,
        "arms_predeclared": list(arms),
        "split_hashes": hashes,
        "warm": {
            "trees": warm.n_estimators_,
            "selection_nll": float(warm.best_score_),
            "native_identity_max_abs": exact,
            "rates": warm.model_.rates.detach().tolist(),
            "effective_count": native.effective_tree_counts(splits["selection"].x),
        },
        "generic_regularizers": {
            "structural": asdict(reg.structural),
            "rate_l2": reg.rate_l2,
            "count_pressure": reg.count_pressure,
            "feature_dropout": .01,
            "tree_dropout": .03,
        },
        "source": asdict(source),
        "arms": {},
        "audit": None,
        "contract": (
            "shared low-level HIGGS warm start; identical generic objective and dropout; "
            "control drives adaptive policies only; selection checkpoints; ranking compares arms; "
            "all five arms are prespecified before untouched audit"
        ),
    }
    trained = {}
    for kind in arms:
        started = time.perf_counter()
        cfg = fork_config(native.config, kind, epochs, smoke)
        model = restore_model(warm_snapshot, native.input_dim, native.output_dim, cfg)
        trainer = RatedJointTrainer(
            model,
            warm.objective_,
            cfg,
            torch.Generator().manual_seed(seed + 100 + ARMS.index(kind)),
            rated_regularizers=reg,
        )
        if kind in ("direct", "capacitor", "rlc"):
            trainer.physical = NormalizedEnergyController(
                cfg.physics, source=source, seed=seed + 301
            )
            trainer.physical.synchronize(
                {key: node.tree_id for key, node in model.node_map().items()}
            )
        trainer.fit(splits["train"], splits["control"], splits["selection"])
        selected = restore_model(
            trainer.best_snapshot, native.input_dim, native.output_dim, cfg
        )
        selected.eval()
        with torch.no_grad():
            p = warm.objective_.response(selected(splits["ranking"].x))[:, 1].numpy()
        result["arms"][kind] = {
            "ranking": metric(raw["ranking"][1], p),
            "selection_best": float(trainer.best_score),
            "best_epoch": trainer.best_epoch,
            "effective_count": selected.effective_tree_counts(splits["ranking"].x),
            "source_energy": float(getattr(trainer.physical, "source_energy_total", 0.)),
            "events": {
                "thermal_thaw": sum(e.get("event") == "thermal_thaw" for e in trainer.events),
                "terminal_lock": sum(e.get("event") == "terminal_lock" for e in trainer.events),
            },
            "seconds": time.perf_counter() - started,
        }
        trained[kind] = selected
        trainer.close()
    result["ranking_winner"] = min(
        result["arms"], key=lambda k: result["arms"][k]["ranking"]["nll"]
    )
    # All five comparisons are prespecified; audit reports all arms only after
    # every ranking score above is fixed. Nothing below may tune/retrain.
    result["audit"] = {}
    for kind in arms:
        with torch.no_grad():
            p = warm.objective_.response(
                trained[kind](splits["audit"].x)
            )[:, 1].numpy()
        result["audit"][kind] = {
            **metric(raw["audit"][1], p),
            "effective_count": trained[kind].effective_tree_counts(splits["audit"].x),
        }
    result["status"] = "completed"
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=401)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = run(args.seed, args.epochs, args.smoke)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    Path(args.out).write_text(text)
    print(text)
