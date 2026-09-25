"""Slow mechanism ratchet for the topology-normalized capacitor controller."""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

from experiments.deep_physics_power_tree import run as run_deep
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


def test_deep_power_tree_control_study_reports_incremental_gain():
    """Study, not a promotion gate: preserve negative results in CI logs."""
    none = run_deep("none", 71, 512)
    full = run_deep("full", 71, 512)
    gain = 1 - full["audit"] / none["audit"]
    assert math.isfinite(gain)
    assert full["events"], full
    warnings.warn(json.dumps({
        "study": "deep-power-tree-controls",
        "none_audit": none["audit"],
        "full_audit": full["audit"],
        "relative_gain": gain,
        "none_last_audit": none["last_audit"],
        "full_last_audit": full["last_audit"],
        "full_events": full["events"],
        "full_injection": full["injection"],
        "full_max_temperature": full["max_temperature"],
    }, sort_keys=True), RuntimeWarning)
