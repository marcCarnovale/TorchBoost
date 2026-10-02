"""Long-horizon direct-feedback study on the CatBoost-ratchet task."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

import experiments.deep_physics_power_tree as deep
import torchboost.adaptive.training as training
from experiments.direct_feedback_control import DirectFeedbackController
from torchboost.adaptive.config import PhysicsConfig
from torchboost.adaptive.unified_progressive import UnifiedProgressiveClassifier


def direct_config(seed, updates):
    cfg = deep.config("plastic", seed, updates)
    cfg.native.plasticity.thermal_softening = 0.18
    cfg.native.physics = PhysicsConfig(
        mode="cooling",
        topology_normalization=True,
        capacitance=1.0,
        discharge_time=5.0,
        inductive_time=2.0,
        cooling_time=24.0,
        total_heat_capacity=0.06,
        dt=0.2,
        initial_temperature=1.0,
        ambient_temperature=1.0,
        max_temperature=3.0,
        thaw_temperature=1.08,
        charge_gain=1.0,
        max_injection=0.03,
        smoothing=0.8,
        lr_coupling=0.04,
    )
    cfg.native.__post_init__()
    return cfg


def run(seed, updates):
    x, y = deep.make_dataset(seed)
    p = np.random.default_rng(seed + 9).permutation(len(x))
    tr = p[:12000]
    control = p[12000:13800]
    selection = p[13800:15800]
    audit = p[15800:]
    old_controller = training.PhysicalController
    training.PhysicalController = DirectFeedbackController
    try:
        started = time.time()
        model = UnifiedProgressiveClassifier(direct_config(seed, updates)).fit(
            x[tr], y[tr],
            control_set=(x[control], y[control]),
            eval_set=(x[selection], y[selection]),
        )
        h = model.trainer_.history[-1]
        return {
            "kind": "direct",
            "seed": seed,
            "updates": updates,
            "selection": model.best_score_,
            "audit": log_loss(y[audit], model.predict_proba(x[audit])),
            "last_audit": log_loss(y[audit], model.predict_proba(x[audit], last=True)),
            "best_step": model.trainer_.best_epoch,
            "anchors": h.get("admitted_anchors", 0),
            "events": h.get("event_counts", {}),
            "direct_heat_total": float(getattr(model.trainer_.physical, "direct_heat_total", 0.0)),
            "max_temperature": h.get("max_temperature_seen", 1.0),
            "seconds": time.time() - started,
        }
    finally:
        training.PhysicalController = old_controller


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--updates", type=int, default=2048)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(args.seed, args.updates)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
