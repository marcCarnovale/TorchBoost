"""Bounded development tuning followed by source-bound fresh-seed confirmations.

Rate and release-policy are the only screened knobs. Every screen includes all
five arms. Confirmations use the configuration selected by development ranking;
no confirmation metric retunes its source. Partial runs are always preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.deep_local_physics import run


def campaign(out, regime, depth=6, ntrain=16000, warm_updates=256, phase_updates=512):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    candidates = []
    for policy in ("persistent_harm", "stress"):
        for rate in (0.1, 0.5):
            path = out / f"development-{policy}-{rate:g}.json"
            result = run(211, depth, ntrain, warm_updates, phase_updates, rate, 0.1, 0.02, regime,
                         path, release_policy=policy)
            candidates.append({"path": str(path), "rate": rate, "release_policy": policy,
                               "best_thermal_ranking": min(result["arms"][name]["ranking_trajectory_nll"]
                                                           for name in ("direct", "capacitor", "rlc"))})
    selected = min(candidates, key=lambda row: row["best_thermal_ranking"])
    manifest = {"status": "selected_before_confirmation", "candidates": candidates,
                "selected": selected, "confirmation_seeds": [223, 227],
                "development_sha256": hashlib.sha256(Path(selected["path"]).read_bytes()).hexdigest(),
                "selection_note": "development thermal-family tuning; confirmations include no control and plasticity at identical settings"}
    (out / "frozen-selection.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    for seed in manifest["confirmation_seeds"]:
        run(seed, depth, ntrain, warm_updates, phase_updates, selected["rate"], 0.1, 0.02, regime,
            out / f"confirmation-{seed}.json", confirmation=True,
            protocol=selected["path"], release_policy=selected["release_policy"])
    manifest["status"] = "completed"
    (out / "frozen-selection.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--regime", choices=("stationary", "recurring"), required=True)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--train", type=int, default=16000)
    parser.add_argument("--warm-updates", type=int, default=256)
    parser.add_argument("--phase-updates", type=int, default=512)
    args = parser.parse_args()
    campaign(args.out, args.regime, args.depth, args.train, args.warm_updates, args.phase_updates)
