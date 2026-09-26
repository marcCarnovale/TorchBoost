"""Stationary and localized-change power-tree controller development study.

One shared trained initialization, same native optimizer update path and generic
regularizers in every arm; no new trees or oracle change times are passed to
controllers. Pretraining and continuation have a common optimizer-reset fork.
Development never scores audit. Confirmation freezes source/plastic settings
before ranking and audits only its winner and the prespecified no-control arm.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import sklearn
import torch

from experiments.deep_physics_power_tree import config as base_config
from experiments.normalized_energy_controller import EnergySource, NormalizedEnergyController
from experiments.thermal_exposure import thermal_exposure
from torchboost.adaptive.config import PhysicsConfig, PlasticityConfig
from torchboost.adaptive.training import model_snapshot, restore_model
from torchboost.adaptive.unified_progressive import UnifiedProgressiveClassifier, UnifiedTrainer


ARMS = ("none", "plastic", "direct", "capacitor", "rlc")


def dataset(seed, ntrain, depth):
    """Random hierarchical local-affine teacher; localized sign change in 1/4.

Ground-truth region labels serve reporting only, never source allocation or loss
regularization. Different roles receive independent rows from distinct streams.
"""
    dimensions = max(16, depth + 8)
    rng = np.random.default_rng(seed)
    levels = sorted(set((0, 2, depth)))
    weights = {}
    for level in levels:
        w = rng.normal(size=(2**level, dimensions))
        w[:, :depth] = 0
        w /= np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)
        weights[level] = w
    sizes = {"train": ntrain, "control": 768, "selection": 1200, "ranking": 1200, "audit": 2400}
    out = {}
    for role, (name, n) in enumerate(sizes.items()):
        gen = np.random.default_rng(np.random.SeedSequence([seed, 7919, role]))
        x = gen.normal(size=(n, dimensions)).astype("float32")
        raw = np.zeros(n)
        for level in levels:
            context = sum((x[:, j] > 0).astype(int) * (1 << j) for j in range(level)) if level else np.zeros(n, int)
            raw += np.einsum("ij,ij->i", weights[level][context], x) / np.sqrt(len(levels))
        raw *= 1.8
        changed = (x[:, 0] > 0) & (x[:, 1] > 0)
        uniform = gen.random(n)
        y_a = (uniform < 1 / (1 + np.exp(-raw))).astype(int)
        y_b = (uniform < 1 / (1 + np.exp(-np.where(changed, -raw, raw)))).astype(int)
        out[name] = (x, y_a, y_b, changed)
    return out


def fingerprint(data):
    return {name: hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()
            for name, arrays in data.items()}


def study_config(seed, depth, warm_updates, phase_updates, kind, stiffness, yield_strain, release_policy="persistent_harm"):
    cfg = base_config("none", seed, warm_updates)
    cfg.depth = depth
    cfg.native.structure.max_depth = depth
    cfg.native.structure.max_nodes = max(511, 2**(depth + 1) - 1)
    cfg.native.structure.dynamic = False
    cfg.native.structure.structural_gate = False
    cfg.native.structure.complexity = 0.0
    cfg.native.structure.allocation_regularization = 0.0
    cfg.native.learning_rate = 0.005
    cfg.native.collect_metrics = True
    cfg.native.observation_every = 8
    cfg.native.control_sample_size = 256
    cfg.age_decay = 1.0
    cfg.active_window = 1
    cfg.anchor_min_passes = 0.0
    cfg.anchor_min_updates = 2
    cfg.anchor_require_utility = True
    cfg.checkpoint_every = 64
    cfg.native.physics = PhysicsConfig(
        mode={"none": "none", "plastic": "none", "direct": "cooling", "capacitor": "capacitor", "rlc": "rlc"}[kind],
        topology_normalization=True, allocation="uniform", capacitance=1.0,
        discharge_time=1.0, inductive_time=0.5, cooling_time=4.0,
        total_heat_capacity=0.08, dt=0.2, initial_temperature=1.0,
        ambient_temperature=1.0, thaw_temperature=1.08, max_temperature=1.4,
        max_charge=1.0, lr_coupling=0.0,
    )
    if kind != "none":
        # Threshold is specified as an interpretable elastic RMS strain.
        # Thermal softening uses thaw units without changing the production law.
        span = cfg.native.physics.thaw_temperature - cfg.native.physics.ambient_temperature
        cfg.native.plasticity = PlasticityConfig(
            mode="full", stiffness=stiffness, yield_threshold=stiffness * yield_strain,
            mobility=0.04, work_hardening=0.12, consolidation_rate=0.02,
            thermal_softening=1.0 / span if kind not in ("none", "plastic") else 0.0,
            release_policy=release_policy, release_patience=3,
        )
    cfg.__post_init__()
    return cfg


def diagnostics(trainer):
    events = trainer.plastic.events
    ratios = [e["stress"] / max(e["effective_yield"], 1e-30) for e in events]
    return {"admitted_anchors": len(trainer.admitted), "plastic_observations": len(events),
            "max_yield_ratio": max(ratios, default=0.0),
            "yield_events": sum(e["flow_fraction"] > 0 for e in events),
            "blocked_release_events": sum(e["blocked_release"] for e in events),
            "damage_events": sum(e["damage"] > 0 for e in events),
            "consolidated_observations": sum(e["consolidated"] for e in events),
            "source_energy": getattr(trainer.physical, "source_energy_total", 0.0),
            "thermal": thermal_exposure(trainer.physical.history, trainer.config.physics)}


def run(seed, depth, ntrain, warm_updates, phase_updates, rate, stiffness, yield_strain, regime, out,
        confirmation=False, arms=ARMS, protocol=None, release_policy="persistent_harm"):
    if min(depth, ntrain, warm_updates, phase_updates) <= 0 or depth > 10:
        raise ValueError("positive budgets and depth <= 10 required")
    if regime not in ("stationary", "recurring") or stiffness <= 0 or yield_strain <= 0:
        raise ValueError("valid regime and positive plastic settings required")
    if not set(arms) <= set(ARMS) or "none" not in arms:
        raise ValueError("known arms and no-control reference required")
    torch.set_num_threads(1)
    data = dataset(seed, ntrain, depth)
    source = EnergySource(rate=rate, threshold=0.5, warmup=8)
    config = study_config(seed, depth, warm_updates, phase_updates, "none", stiffness, yield_strain, release_policy)
    result = {"status": "running", "development_only": not confirmation, "audit": None,
              "seed": seed, "depth": depth, "ntrain": ntrain, "warm_updates": warm_updates,
              "phase_updates": phase_updates, "regime": regime, "source": asdict(source),
              "stiffness": stiffness, "yield_strain": yield_strain, "release_policy": release_policy, "splits": fingerprint(data),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "sklearn": sklearn.__version__},
              "source_files": {}, "arms": {},
              "protocol": "shared warm model; reset optimizer at common fork only; native fixed-structure continuation; ranking trajectory chooses arm; audit opens last"}
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "torchboost").rglob("*.py")) + [Path(__file__).resolve(), root / "experiments/normalized_energy_controller.py", root / "experiments/deep_physics_power_tree.py", root / "experiments/thermal_exposure.py"]:
        result["source_files"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if confirmation:
        if protocol is None:
            raise ValueError("confirmation requires a completed development protocol")
        protocol_bytes = Path(protocol).read_bytes()
        development = json.loads(protocol_bytes)
        if development.get("status") != "completed" or not development.get("development_only"):
            raise ValueError("completed development result required")
        for field in ("depth", "ntrain", "warm_updates", "phase_updates", "regime", "source",
                      "stiffness", "yield_strain", "release_policy", "source_files"):
            if development[field] != result[field]:
                raise ValueError(f"confirmation protocol mismatch: {field}")
        if development["seed"] == seed or set(development["arms"]) != set(arms):
            raise ValueError("confirmation needs a fresh seed and the same frozen arms")
        result["protocol_sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
        result["development_seed"] = development["seed"]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        tmp.replace(out)

    save()
    try:
        base = UnifiedProgressiveClassifier(config).fit(*data["train"][:2],
                control_set=data["control"][:2], eval_set=data["selection"][:2])
        warm = model_snapshot(base.trainer_.model)
        result["warm_model"] = {"nodes": len(base.trainer_.model.node_map()),
                              "actual_max_depth": max(n.depth for n in base.trainer_.model.iter_nodes()),
                              "trees": len(base.trainer_.model.trees)}
        splits = {phase: {name: base.preprocessor_.split(a[0], a[1 if phase == "A" else 2])
                          for name, a in data.items() if name != "audit"} for phase in ("A", "B")}
        sequence = ("A", "B", "A") if regime == "recurring" else ("A", "A", "A")
        selected_models = {}
        audit_snapshots = {}
        for kind in arms:
            started = time.perf_counter()
            cfg = study_config(seed, depth, warm_updates, phase_updates, kind, stiffness, yield_strain, release_policy)
            model = restore_model(warm, base.n_features_in_, base.objective_.output_dim, cfg.native)
            trainer = UnifiedTrainer(model, base.objective_, cfg)
            trainer.stage = len(model.trees)
            trainer.ntrain = ntrain
            trainer.config.epochs = 3 * phase_updates
            if kind in ("direct", "capacitor", "rlc"):
                trainer.physical = NormalizedEnergyController(cfg.native.physics, source=source, seed=seed + 101)
                trainer.physical.synchronize({key: n.tree_id for key, n in model.node_map().items()})
            rows = []
            audit_snapshots[kind] = []
            try:
                for phase_index, phase in enumerate(sequence):
                    train, control, selection, ranking = (splits[phase][name] for name in ("train", "control", "selection", "ranking"))
                    # Checkpoint selection resets for each changed objective;
                    # optimizer, source, tracker, anchors, and model do NOT.
                    trainer.best_score = float("inf")
                    trainer.evaluate(train, selection, "phase_start")
                    pool = torch.arange(ntrain)
                    for within in range(phase_updates):
                        values = trainer.schedule.apply(model, trainer.tick)
                        trainer._set_active()
                        trainer._set_rates(values)
                        trainer.fixed_cache = None
                        trainer._update(train, pool, values, 0, warm_updates + within)
                        if trainer.tick % cfg.native.observation_every == 0:
                            trainer._observe_and_control(control, trainer.tick)
                        trainer.tick += 1
                        trainer.epoch = trainer.tick
                        if (within + 1) % 64 == 0 or within + 1 == phase_updates:
                            trainer.evaluate(train, selection, phase)
                            if confirmation:
                                audit_snapshots[kind].append((phase, model_snapshot(model)))
                            rows.append({"step": trainer.tick, "phase": phase_index, "domain": phase,
                                         "train_nll": trainer.loss(train), "control_nll": trainer.loss(control),
                                         "selection_nll": trainer.loss(selection), "ranking_nll": trainer.loss(ranking)})
                selected = restore_model(trainer.best_snapshot, base.n_features_in_, base.objective_.output_dim, trainer.config)
                selected_models[kind] = selected
                row = {"ranking_trajectory_nll": float(np.mean([r["ranking_nll"] for r in rows])),
                       "trace": rows, "config": asdict(cfg), "diagnostics": diagnostics(trainer),
                       "seconds": time.perf_counter() - started}
                result["arms"][kind] = row
                save()
                print(json.dumps({"kind": kind, "seed": seed, "ranking_trajectory_nll": row["ranking_trajectory_nll"],
                                  "diagnostics": row["diagnostics"], "seconds": row["seconds"]}), flush=True)
            finally:
                trainer.close()
        winner = min(result["arms"], key=lambda k: result["arms"][k]["ranking_trajectory_nll"])
        result["ranking_winner"] = winner
        if confirmation:
            audit_splits = {phase: base.preprocessor_.split(data["audit"][0], data["audit"][1 if phase == "A" else 2])
                            for phase in ("A", "B")}
            audit = audit_splits["A"]
            result["audit"] = {}
            for kind in sorted({"none", winner}):
                selected_models[kind].eval()
                with torch.no_grad():
                    score = float(base.objective_.weighted_loss(selected_models[kind](audit.x), audit.y, audit.weight))
                trajectory = []
                with torch.no_grad():
                    for phase, snapshot in audit_snapshots[kind]:
                        candidate = restore_model(snapshot, base.n_features_in_, base.objective_.output_dim,
                                                  selected_models[kind].config)
                        candidate.eval()
                        split = audit_splits[phase]
                        trajectory.append(float(base.objective_.weighted_loss(candidate(split.x), split.y, split.weight)))
                result["audit"][kind] = {"final_phase_selected_nll": score,
                                        "trajectory_nll": float(np.mean(trajectory)), "trajectory": trajectory}
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        save()
        raise
    save()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=211)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--train", type=int, default=16000)
    p.add_argument("--warm-updates", type=int, default=256)
    p.add_argument("--phase-updates", type=int, default=512)
    p.add_argument("--rate", type=float, default=0.1)
    p.add_argument("--stiffness", type=float, default=0.1)
    p.add_argument("--yield-strain", type=float, default=0.02)
    p.add_argument("--regime", choices=("stationary", "recurring"), default="stationary")
    p.add_argument("--confirmation", action="store_true")
    p.add_argument("--protocol")
    p.add_argument("--release-policy", choices=("persistent_harm", "stress"), default="persistent_harm")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run(args.seed, args.depth, args.train, args.warm_updates, args.phase_updates,
        args.rate, args.stiffness, args.yield_strain, args.regime, args.out, args.confirmation, protocol=args.protocol, release_policy=args.release_policy)
