"""Development-only deep-tree smoke for the normalized controller source.

The adaptive-energy source standardizes generic control-loss surprise online and
expresses source input as a fraction of live ambient-to-thaw thermal energy.
Audit is not scored. The legacy capacitor remains a separate reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import log_loss

import experiments.deep_physics_power_tree as deep
from experiments.direct_feedback_control import DirectFeedbackController
import torchboost.adaptive.training as training
from torchboost.adaptive.unified_progressive import UnifiedProgressiveClassifier


KINDS = ("none", "cap_legacy", "cap_energy", "direct_energy")


def config(kind: str, seed: int, updates: int):
    if kind == "none":
        return deep.config("none", seed, updates)
    cfg = deep.config("cap", seed, updates)
    if kind == "cap_legacy":
        return cfg
    if kind not in ("cap_energy", "direct_energy"):
        raise ValueError(kind)
    if kind == "direct_energy":
        cfg.native.physics.mode = "cooling"
    cfg.native.physics.source_normalization = "adaptive_energy"
    cfg.native.physics.charge_gain = 0.025
    cfg.native.physics.max_injection = 0.05
    cfg.native.physics.max_charge = 10.0
    cfg.native.physics.__post_init__()
    cfg.native.__post_init__()
    return cfg


def run(kind: str, seed: int = 107, updates: int = 256):
    x, y = deep.make_dataset(seed)
    order = np.random.default_rng(seed + 9).permutation(len(x))
    train = order[:12000]
    control = order[12000:13800]
    selection = order[13800:14800]
    ranking = order[14800:15800]
    audit = order[15800:]
    split_hashes = {}
    for name, idx in (("train", train), ("control", control), ("selection", selection),
                      ("ranking", ranking), ("audit", audit)):
        h = hashlib.sha256()
        h.update(idx.tobytes())
        h.update(x[idx].tobytes())
        h.update(y[idx].tobytes())
        split_hashes[name] = {"rows": len(idx), "sha256": h.hexdigest()}

    cfg = config(kind, seed, updates)
    old_controller = training.PhysicalController
    if kind == "direct_energy":
        training.PhysicalController = DirectFeedbackController
    try:
        started = time.perf_counter()
        model = UnifiedProgressiveClassifier(cfg).fit(
            x[train], y[train],
            control_set=(x[control], y[control]),
            eval_set=(x[selection], y[selection]),
        )
        history = (model.trainer_.physical.history
                   if getattr(model.trainer_, "physical", None) is not None else [])
        row = {
            "kind": kind,
            "seed": seed,
            "updates": updates,
            "selection_nll": float(model.best_score_),
            "ranking_nll": float(log_loss(y[ranking], model.predict_proba(x[ranking]))),
            "last_ranking_nll": float(log_loss(y[ranking], model.predict_proba(x[ranking], last=True))),
            "best_step": int(model.trainer_.best_epoch),
            "audit_evaluated": False,
            "splits": split_hashes,
            "seconds": time.perf_counter() - started,
        }
        if history:
            row["controller"] = {
                "source_work": float(sum(h.get("source_work", 0.0) for h in history)),
                "external_heat": float(sum(h.get("external_heat", 0.0) for h in history)),
                "resistor_heat": float(sum(h.get("resistor_heat", 0.0) for h in history)),
                "max_temperature": float(max(
                    max(n["temperature"] for n in h["nodes"].values()) for h in history
                )),
                "mean_source_signal": float(np.mean([
                    h.get("source_signal", 0.0) for h in history
                ])),
            }
        return row
    finally:
        training.PhysicalController = old_controller


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--seed", type=int, default=107)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(args.kind, args.seed, args.updates)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps(result, sort_keys=True))
