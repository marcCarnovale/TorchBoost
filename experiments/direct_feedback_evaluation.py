"""Held-out evaluation of the direct generic feedback controller.

Hyperparameters are selected on development seed 11 only. Seeds 17 and 29 are
then evaluated without retuning. The fixed-pulse comparator uses the corrected
matched thermal couplings; it remains an oracle-timed positive control because
it knows regime boundaries.
"""
from __future__ import annotations
import argparse
import json
from statistics import mean

import experiments.long_regimes as lr
from experiments.direct_feedback_control import run_direct, study as development_study
from experiments.matched_physics_controls_v2 import matched_cfg

SEQUENCE = ["A", "B", "A", "B", "A"]
EVAL_SEEDS = (17, 29)

def summarize_baseline(result, none):
    return {
        "current_regret_proxy": float(result["current_regret_proxy"]),
        "final_A": float(result["trajectory"][-1]["A"]),
        "regret_gain_vs_none": 0.0 if result["kind"] == "none" else 1 - result["current_regret_proxy"] / none["current_regret_proxy"],
        "final_A_gain_vs_none": 0.0 if result["kind"] == "none" else 1 - result["trajectory"][-1]["A"] / none["trajectory"][-1]["A"],
        "max_temperature": float(result["max_temperature"]),
    }

def run(updates=16):
    dev = development_study(seed=11, updates=updates)
    chosen = min(dev["direct"], key=lambda row: row["current_regret_proxy"])
    gain, max_heat = float(chosen["gain"]), float(chosen["max_heat"])
    old_cfg = lr.cfg
    lr.cfg = matched_cfg
    try:
        evaluations = []
        for seed in EVAL_SEEDS:
            baseline_raw = {kind: lr.run(kind, seed, SEQUENCE, updates) for kind in ("none", "plastic", "pulse", "cap", "rlc")}
            none = baseline_raw["none"]
            baseline = {kind: summarize_baseline(result, none) for kind, result in baseline_raw.items()}
            direct = run_direct(seed, SEQUENCE, updates, gain, max_heat)
            direct["regret_gain_vs_none"] = 1 - direct["current_regret_proxy"] / none["current_regret_proxy"]
            direct["final_A_gain_vs_none"] = 1 - direct["final_A"] / none["trajectory"][-1]["A"]
            evaluations.append({"seed": seed, "baseline": baseline, "direct": direct})
    finally:
        lr.cfg = old_cfg
    means = {}
    for method in ("plastic", "pulse", "cap", "rlc", "direct"):
        rows = [(row["direct"] if method == "direct" else row["baseline"][method]) for row in evaluations]
        means[method] = {
            "regret_gain_vs_none": mean(r["regret_gain_vs_none"] for r in rows),
            "final_A_gain_vs_none": mean(r["final_A_gain_vs_none"] for r in rows),
            "max_temperature": mean(r["max_temperature"] for r in rows),
        }
    return {
        "development_seed": 11,
        "evaluation_seeds": list(EVAL_SEEDS),
        "updates": updates,
        "selection_rule": "minimum current_regret_proxy on development seed only",
        "chosen_direct": {"gain": gain, "max_heat": max_heat},
        "development": dev,
        "evaluation": evaluations,
        "mean_evaluation": means,
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
