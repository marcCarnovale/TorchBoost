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


def test_deep_power_tree_2048_mechanism_report():
    """Report five-way stationary-task behavior without turning it into a ratchet."""
    import math

    from experiments.deep_physics_power_tree import run as run_deep

    rows = [run_deep(kind, 71, 2048) for kind in ("none", "plastic", "cap", "rlc", "full")]
    assert all(math.isfinite(row["audit"]) and math.isfinite(row["last_audit"]) for row in rows)
    by_kind = {row["kind"]: row for row in rows}
    assert by_kind["none"]["injection"] == 0.
    assert by_kind["plastic"]["anchors"] > 0
    assert by_kind["cap"]["injection"] > 0
    assert by_kind["rlc"]["injection"] > 0
    raise AssertionError("DEEP2048_REPORT=" + json.dumps(rows, sort_keys=True))
