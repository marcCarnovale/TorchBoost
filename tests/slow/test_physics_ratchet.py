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


def test_central_force_learning_curve_report():
    """Probe whether more capacity/data moves the learned field toward inverse-square structure."""
    import math

    from experiments.scientific_law_discovery import run as run_science

    rows = [
        run_science(seed=31, nfit=2000, depth=2, updates=128, noise=.05),
        run_science(seed=31, nfit=8000, depth=4, updates=512, noise=.05),
    ]
    for row in rows:
        for learner in ("torchboost", "torchboost_ood", "catboost", "catboost_ood"):
            metrics = row[learner]
            assert math.isfinite(metrics["rmse"])
            assert math.isfinite(metrics["radial_alignment"])
            assert math.isfinite(metrics["inverse_power_exponent"])
    raise AssertionError("CENTRAL_FORCE_REPORT=" + json.dumps(rows, sort_keys=True))
