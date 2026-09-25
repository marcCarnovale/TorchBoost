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


def test_miniboone_public_benchmark_report():
    """Run one harder public numerical benchmark without making it a ratchet."""
    import math

    from experiments.openml_benchmark import run as run_openml

    result = run_openml(data_id=44128, seed=41, updates=384)
    assert result["rows"] > 50000
    assert result["features"] >= 40
    assert math.isfinite(result["torchboost_audit"])
    assert math.isfinite(result["catboost_audit"])
    raise AssertionError("MINIBOONE_REPORT=" + json.dumps(result, sort_keys=True))
