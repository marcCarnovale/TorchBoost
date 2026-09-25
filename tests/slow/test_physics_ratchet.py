"""Slow mechanism ratchet for the topology-normalized capacitor controller."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.long_regimes import run

ROOT = Path(__file__).resolve().parents[2]
RATCHET = json.loads((ROOT / "benchmarks" / "physics_ratchet.json").read_text())


def test_topology_normalized_capacitor_retains_recurring_regime_gain():
    sequence = ["A", "B", "A", "B", "A"]
    seed = RATCHET["seed"]
    updates = RATCHET["updates_per_stage"]
    none = run("none", seed, sequence, updates)
    cap = run("cap", seed, sequence, updates)

    regret_gain = 1 - cap["current_regret_proxy"] / none["current_regret_proxy"]
    final_gain = 1 - cap["trajectory"][-1]["A"] / none["trajectory"][-1]["A"]

    assert regret_gain >= RATCHET["min_regret_improvement"], (none, cap, regret_gain)
    assert final_gain >= RATCHET["min_final_A_improvement"], (none, cap, final_gain)


def test_epicycle_regularization_mechanism_report():
    """Falsifiable warm-start representation-competition mechanism study."""
    import math

    from experiments.epicycle_regularization import run as run_epicycle

    result = run_epicycle(seed=17)
    for variant in result["variants"]:
        for phase in ("initial", "final"):
            metrics = variant[phase]
            assert math.isfinite(metrics["train_mse"])
            assert math.isfinite(metrics["holdout_mse"])
            assert math.isfinite(metrics["ood_mse"])
            assert math.isfinite(metrics["effective_order"])
            assert math.isfinite(metrics["alpha"])
    raise AssertionError("EPICYCLE_REPORT=" + json.dumps(result, sort_keys=True))
