"""Temperature-matched direct-feedback evaluation.

The primary direct controller is selected by development regret on seed 11.
A second, prespecified fairness control chooses the direct candidate whose peak
temperature on the development seed is closest to the matched oracle pulse.
Neither choice uses evaluation seeds 17 or 29.
"""
from __future__ import annotations

import argparse
import json
from statistics import mean

import experiments.long_regimes as lr
from experiments.direct_feedback_control import run_direct, study as direct_development
from experiments.matched_physics_controls_v2 import matched_cfg, study as matched_development

SEQUENCE = ["A", "B", "A", "B", "A"]
EVAL_SEEDS = (17, 29)


def _baseline(seed, updates):
    old = lr.cfg
    lr.cfg = matched_cfg
    try:
        return {
            kind: lr.run(kind, seed, SEQUENCE, updates)
            for kind in ("none", "plastic", "pulse", "cap", "rlc")
        }
    finally:
        lr.cfg = old


def _gain(row, none):
    return {
        "current_regret_proxy": float(row["current_regret_proxy"]),
        "final_A": float(row["trajectory"][-1]["A"]),
        "regret_gain_vs_none": 1 - row["current_regret_proxy"] / none["current_regret_proxy"],
        "final_A_gain_vs_none": 1 - row["trajectory"][-1]["A"] / none["trajectory"][-1]["A"],
        "max_temperature": float(row["max_temperature"]),
    }


def run(updates=16):
    dev_direct = direct_development(seed=11, updates=updates)
    dev_matched = matched_development(seed=11, updates=updates)
    best = min(dev_direct["direct"], key=lambda row: row["current_regret_proxy"])
    target_temp = dev_matched["rows"]["pulse"]["max_temperature"]
    matched = min(
        dev_direct["direct"],
        key=lambda row: abs(row["max_temperature"] - target_temp),
    )
    choices = {
        "direct_best": {"gain": float(best["gain"]), "max_heat": float(best["max_heat"])},
        "direct_temp_matched": {
            "gain": float(matched["gain"]),
            "max_heat": float(matched["max_heat"]),
        },
    }

    evaluations = []
    for seed in EVAL_SEEDS:
        raw = _baseline(seed, updates)
        none = raw["none"]
        row = {
            "seed": seed,
            "baseline": {kind: _gain(value, none) for kind, value in raw.items()},
        }
        for name, pars in choices.items():
            direct = run_direct(seed, SEQUENCE, updates, pars["gain"], pars["max_heat"])
            row[name] = {
                **direct,
                "regret_gain_vs_none": 1 - direct["current_regret_proxy"] / none["current_regret_proxy"],
                "final_A_gain_vs_none": 1 - direct["final_A"] / none["trajectory"][-1]["A"],
            }
        evaluations.append(row)

    methods = ("plastic", "pulse", "cap", "rlc", "direct_best", "direct_temp_matched")
    summary = {}
    for method in methods:
        rows = [
            (r[method] if method.startswith("direct_") else r["baseline"][method])
            for r in evaluations
        ]
        summary[method] = {
            "regret_gain_vs_none": mean(x["regret_gain_vs_none"] for x in rows),
            "final_A_gain_vs_none": mean(x["final_A_gain_vs_none"] for x in rows),
            "max_temperature": mean(x["max_temperature"] for x in rows),
        }

    return {
        "development_seed": 11,
        "evaluation_seeds": list(EVAL_SEEDS),
        "updates": updates,
        "pulse_temperature_target": float(target_temp),
        "choices": choices,
        "development_direct": dev_direct,
        "development_matched": dev_matched,
        "evaluation": evaluations,
        "mean_evaluation": summary,
        "selection_contract": (
            "direct_best minimizes development regret; direct_temp_matched minimizes "
            "development peak-temperature mismatch to the matched oracle pulse. "
            "Evaluation seeds never choose hyperparameters."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = run(args.updates)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(text)
    print(text)
