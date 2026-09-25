"""Direct generic feedback versus electrical and oracle-thaw controls.

This is an experiment-only controller. It uses exactly the same generic
controller loss deterioration signal and the same normalized thermal state as
the circuit controllers, but converts positive loss surprise directly into
heat.  It has no capacitor/inductor memory.  This tests whether electrical
state variables add value beyond a simple feedback law.

The fixed pulse remains an oracle-timed positive control because it knows the
stage boundary. Production defaults are not changed.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from sklearn.metrics import log_loss

import experiments.long_regimes as lr
import torchboost.adaptive.training as training
from torchboost.adaptive.physics import PhysicalController
from torchboost.adaptive.config import PhysicsConfig
from torchboost.adaptive.unified_progressive import UnifiedProgressiveClassifier


class DirectFeedbackController(PhysicalController):
    """Loss-surprise -> normalized heat, with no electrical reservoir."""

    def __init__(self, config, *, seed=0):
        super().__init__(config, seed=seed)
        self.direct_heat_total = 0.0

    def advance(self, loss, observations, step):
        cfg = self.config
        direct = 0.0
        if self.reference is not None:
            direct = min(cfg.max_injection, cfg.charge_gain * max(0.0, loss - self.reference))
        if direct > 0 and self.nodes:
            keys = sorted(self.nodes)
            resistance = self._resistances(keys, observations)
            conductance = 1.0 / resistance
            weights = conductance / conductance.sum()
            capacities = np.asarray([self.nodes[k]["capacity"] for k in keys], dtype=float)
            for j, key in enumerate(keys):
                self.nodes[key]["temperature"] = float(
                    self.nodes[key]["temperature"] + direct * weights[j] / capacities[j]
                )
            self.direct_heat_total += float(direct)
        out = super().advance(loss, observations, step)
        out["direct_heat_injection"] = float(direct)
        out["direct_heat_total"] = float(self.direct_heat_total)
        return out


BASE_CFG = lr.cfg


def direct_cfg(seed, stages, updates, gain, max_heat):
    cfg = BASE_CFG("plastic", seed, stages, updates)
    cfg.native.plasticity.thermal_softening = 0.2
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
        charge_gain=gain,          # experiment-only interpretation: heat gain
        max_injection=max_heat,    # experiment-only interpretation: heat cap
        smoothing=0.8,
        lr_coupling=0.04,
    )
    cfg.native.__post_init__()
    return cfg


def run_direct(seed, sequence, updates, gain, max_heat):
    old_controller = training.PhysicalController
    training.PhysicalController = DirectFeedbackController
    try:
        stages = len(sequence)
        m = UnifiedProgressiveClassifier(direct_cfg(seed, stages, updates, gain, max_heat))
        trajectory = []
        for stage, name in enumerate(sequence):
            x, y = lr.domain(1600, name, seed * 1000 + stage * 31 + 1)
            xc, yc = lr.domain(650, name, seed * 1000 + stage * 31 + 2)
            xs, ys = lr.domain(650, name, seed * 1000 + stage * 31 + 3)
            if stage == 0:
                m.fit(x, y, control_set=(xc, yc), eval_set=(xs, ys), stop_stages=1)
            else:
                m.continue_fit(
                    x, y,
                    control_set=(xc, yc),
                    eval_set=(xs, ys),
                    stop_stages=stage + 1,
                    allow_domain_shift=True,
                )
            trajectory.append({
                "stage": stage,
                "domain": name,
                "A": log_loss(lr.AUD["A"][1], m.predict_proba(lr.AUD["A"][0], last=True)),
                "B": log_loss(lr.AUD["B"][1], m.predict_proba(lr.AUD["B"][0], last=True)),
            })
        h = m.trainer_.history[-1]
        return {
            "gain": gain,
            "max_heat": max_heat,
            "current_regret_proxy": float(sum(r[r["domain"]] for r in trajectory)),
            "final_A": float(trajectory[-1]["A"]),
            "trajectory": trajectory,
            "direct_heat_total": float(getattr(m.trainer_.physical, "direct_heat_total", 0.0)),
            "max_temperature": float(h.get("max_temperature_seen", 1.0)),
            "events": h.get("event_counts", {}),
            "anchors": h.get("admitted_anchors", 0),
        }
    finally:
        training.PhysicalController = old_controller


def study(seed=11, updates=16):
    sequence = ["A", "B", "A", "B", "A"]
    # Development-only screen. Evaluation seeds remain 17 and 29.
    candidates = [
        (0.10, 0.015),
        (0.25, 0.020),
        (0.50, 0.025),
        (1.00, 0.030),
    ]
    baseline = {kind: lr.run(kind, seed, sequence, updates) for kind in ("none", "cap", "pulse")}
    direct = [run_direct(seed, sequence, updates, gain, cap) for gain, cap in candidates]
    none = baseline["none"]["current_regret_proxy"]
    none_a = baseline["none"]["trajectory"][-1]["A"]
    for row in direct:
        row["regret_gain_vs_none"] = 1 - row["current_regret_proxy"] / none
        row["final_A_gain_vs_none"] = 1 - row["final_A"] / none_a
    return {
        "seed": seed,
        "updates": updates,
        "baseline": {
            k: {
                "current_regret_proxy": v["current_regret_proxy"],
                "final_A": v["trajectory"][-1]["A"],
                "regret_gain_vs_none": 0.0 if k == "none" else 1 - v["current_regret_proxy"] / none,
                "final_A_gain_vs_none": 0.0 if k == "none" else 1 - v["trajectory"][-1]["A"] / none_a,
                "max_temperature": v["max_temperature"],
            }
            for k, v in baseline.items()
        },
        "direct": direct,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--updates", type=int, default=16)
    args = ap.parse_args()
    print(json.dumps(study(args.seed, args.updates), indent=2, sort_keys=True))
