"""Bounded, development-only refinement of a coarse generic heat-dose screen.

Predictive scores never choose the refinement gain. The original ranking gain
is held fixed. A failed coarse match and every refinement are retained. This
calibrates a controller, not a domain/scientific-law regularizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback

import torch

from experiments.deep_controller_protocol import (
    MATCH_TOLERANCE, fit_candidate, partition, source_fingerprint, write_result,
)
from experiments.thermal_exposure import exposure_distance, exposure_matched


def refinement_gain(gain, candidate, target):
    fields = ("peak_thaw_units", "mean_thaw_exposure")
    values = [float(row[k]) for row in (candidate, target) for k in fields]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("positive finite peak and exposure are required")
    correction = math.exp(sum(math.log(target[k] / candidate[k]) for k in fields) / 2.0)
    return max(1e-6, min(10.0, float(gain) * correction))


def run(development_path, out, max_refinements=3):
    if not 0 <= max_refinements <= 3:
        raise ValueError("refinement budget must be between zero and three")
    torch.set_num_threads(1)
    input_path = Path(development_path)
    result = json.loads(input_path.read_text())
    if result.get("phase") != "development" or result.get("status") != "completed":
        raise ValueError("a completed development screen is required")
    if result["source_fingerprint"] != source_fingerprint() or result["audit"] is not None:
        raise ValueError("source must match and development audit must remain closed")
    out = Path(out)
    if out.resolve() == input_path.resolve():
        raise ValueError("preserve the original screen; use a new output path")
    out.parent.mkdir(parents=True, exist_ok=True)
    seed, updates = result["seed"], result["updates"]
    x, y, rows, manifest = partition(seed)
    if manifest != result["splits"]:
        raise ValueError("development data fingerprint changed")
    result["calibration"] = {
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "max_refinements": max_refinements, "refinements": [],
        "original_protocol": dict(result["frozen_protocol"]),
        "choice_uses": ["development peak excursion", "development integrated exposure"],
    }
    result["status"] = "running"
    write_result(out, result)
    target = result["candidates"]["cap"]["thermal"]
    try:
        for attempt in range(max_refinements + 1):
            direct = {k: v for k, v in result["candidates"].items() if v["kind"] == "direct"}
            name = min(direct, key=lambda k: exposure_distance(direct[k]["thermal"], target))
            best = direct[name]
            matched = exposure_matched(best["thermal"], target, MATCH_TOLERANCE)
            if matched or attempt == max_refinements:
                result["frozen_protocol"]["direct_dose_gain"] = best["gain"]
                result["frozen_protocol"]["development_dose_matched"] = matched
                break
            gain = refinement_gain(best["gain"], best["thermal"], target)
            name = f"direct_{gain:g}"
            if name in result["candidates"]:
                raise ValueError("refinement did not create a new gain")
            _, row = fit_candidate("direct", gain, seed, updates, x, y, rows)
            result["candidates"][name] = row
            result["calibration"]["refinements"].append(name)
            write_result(out, result)
            print(json.dumps({"refinement": name, "gain": gain, "thermal": row["thermal"]}), flush=True)
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_result(out, result)
        raise
    write_result(out, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-refinements", type=int, default=3)
    args = parser.parse_args()
    run(args.development, args.out, args.max_refinements)
