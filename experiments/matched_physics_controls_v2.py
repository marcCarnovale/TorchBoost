"""Matched recurring-regime physics study.

The fixed-pulse arm uses the same generic thermal-softening and temperature-to-LR
couplings as the circuit arms; only the source of heat differs. Production
defaults are unchanged.
"""
from __future__ import annotations

import argparse
import json

import experiments.long_regimes as lr

BASE_CFG = lr.cfg


def matched_cfg(kind, seed, stages, updates):
    cfg = BASE_CFG(kind, seed, stages, updates)
    if kind == "pulse":
        cfg.native.plasticity.thermal_softening = 0.2
        cfg.native.physics.lr_coupling = 0.04
        cfg.native.__post_init__()
    return cfg


def study(seed=17, updates=16):
    lr.cfg = matched_cfg
    sequence = ["A", "B", "A", "B", "A"]
    kinds = ("none", "plastic", "pulse", "cap", "rlc")
    results = {kind: lr.run(kind, seed, sequence, updates) for kind in kinds}
    base = results["none"]
    rows = {}
    for kind, result in results.items():
        rows[kind] = {
            "current_regret_proxy": result["current_regret_proxy"],
            "final_A": result["trajectory"][-1]["A"],
            "regret_gain_vs_none": (
                0.0
                if kind == "none"
                else 1 - result["current_regret_proxy"] / base["current_regret_proxy"]
            ),
            "final_A_gain_vs_none": (
                0.0
                if kind == "none"
                else 1 - result["trajectory"][-1]["A"] / base["trajectory"][-1]["A"]
            ),
            "max_temperature": result["max_temperature"],
            "injection": result["injection"],
            "anchors": result["anchors"],
            "events": result["events"],
        }
    return {"seed": seed, "updates": updates, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(study(args.seed, args.updates), indent=2, sort_keys=True))
