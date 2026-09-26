"""Deep-tree controller comparison with separated development and audit data.

All active thermal arms share the capacitor arm's ordinary cooling/softening/LR
settings. Only the heat source changes. Direct feedback is calibrated on a
DEVELOPMENT seed to peak excursion AND integrated exposure, never to audit loss.
Confirmation uses new generated data, its own five disjoint partitions, and
opens audit only for the ranking-selected model and the prespecified no-control
reference. The CatBoost ratchet and production defaults are not modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import sklearn
from sklearn.metrics import log_loss
import torch

import experiments.deep_physics_power_tree as deep
from experiments.direct_feedback_control import DirectFeedbackController
from experiments.thermal_exposure import exposure_distance, exposure_matched, thermal_exposure
import torchboost.adaptive.training as training
from torchboost.adaptive.unified_progressive import UnifiedProgressiveClassifier

ROOT = Path(__file__).resolve().parents[1]
BASE_ARMS = ("none", "plastic", "cooling", "cap", "rlc")
DIRECT_GAINS = (0.10, 0.25, 0.50, 1.00)
DIRECT_HEAT_CAP = 0.03
MATCH_TOLERANCE = 0.20


def source_fingerprint():
    paths = sorted((ROOT / "torchboost").rglob("*.py")) + [
        Path(__file__).resolve(), ROOT / "experiments/deep_physics_power_tree.py",
        ROOT / "experiments/direct_feedback_control.py", ROOT / "experiments/thermal_exposure.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def partition(seed):
    x, y = deep.make_dataset(seed)
    order = np.random.default_rng(seed + 9).permutation(len(x))
    boundaries = (0, 12000, 13800, 14800, 15800, len(x))
    names = ("train", "control", "selection", "ranking", "audit")
    rows = {name: order[a:b] for name, a, b in zip(names, boundaries[:-1], boundaries[1:])}
    manifest = {}
    for name, idx in rows.items():
        digest = hashlib.sha256()
        digest.update(idx.tobytes())
        digest.update(x[idx].tobytes())
        digest.update(y[idx].tobytes())
        manifest[name] = {"rows": len(idx), "sha256": digest.hexdigest()}
    return x, y, rows, manifest


def candidate_config(kind, seed, updates, gain=None):
    if kind not in BASE_ARMS + ("direct",):
        raise ValueError(f"unknown controller {kind}")
    base = kind if kind in ("none", "plastic", "rlc") else "cap"
    cfg = deep.config(base, seed, updates)
    if kind in ("cooling", "direct"):
        cfg.native.physics.mode = "cooling"
    if kind == "direct":
        if gain is None or not np.isfinite(gain) or gain <= 0:
            raise ValueError("positive finite direct heat gain required")
        cfg.native.physics.charge_gain = float(gain)
        cfg.native.physics.max_injection = DIRECT_HEAT_CAP
    cfg.native.physics.__post_init__()
    cfg.native.__post_init__()
    return cfg


@contextmanager
def controller_type(direct):
    # The existing trainer constructs its controller from this module symbol.
    # Scope the experiment patch and always restore it; jobs are process-isolated.
    original = training.PhysicalController
    if direct:
        training.PhysicalController = DirectFeedbackController
    try:
        yield
    finally:
        training.PhysicalController = original


def fit_candidate(kind, gain, seed, updates, x, y, rows):
    cfg = candidate_config(kind, seed, updates, gain)
    started = time.perf_counter()
    with controller_type(kind == "direct"):
        model = UnifiedProgressiveClassifier(cfg).fit(
            x[rows["train"]], y[rows["train"]],
            control_set=(x[rows["control"]], y[rows["control"]]),
            eval_set=(x[rows["selection"]], y[rows["selection"]]),
        )
    ranking = rows["ranking"]
    selected_ranking = log_loss(y[ranking], model.predict_proba(x[ranking]))
    last_ranking = log_loss(y[ranking], model.predict_proba(x[ranking], last=True))
    row = {
        "kind": kind, "gain": gain, "config": asdict(cfg),
        "selection_nll": float(model.best_score_),
        "ranking_nll": float(selected_ranking), "last_ranking_nll": float(last_ranking),
        "best_step": model.trainer_.best_epoch,
        "thermal": thermal_exposure(model.trainer_.physical.history, cfg.native.physics),
        "training_trace": [
            {k: v for k, v in record.items() if k in (
                "step", "epoch", "train_loss", "control_loss", "selection_loss",
                "admitted_anchors", "max_temperature_seen", "event_counts",
            )} for record in model.trainer_.history
        ],
        "seconds": time.perf_counter() - started,
    }
    return model, row


def write_result(path, result):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def run(phase, seed, updates, out, protocol_path=None):
    torch.set_num_threads(1)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint()
    protocol = None
    if phase == "confirmation":
        if protocol_path is None:
            raise ValueError("confirmation requires a completed development protocol")
        development = json.loads(Path(protocol_path).read_text())
        if development.get("phase") != "development" or development.get("status") != "completed":
            raise ValueError("development protocol must be complete")
        protocol = development["frozen_protocol"]
        if protocol["development_seed"] == seed:
            raise ValueError("confirmation must use a new data seed")
        if protocol["updates"] != updates or protocol["source_fingerprint"] != fingerprint:
            raise ValueError("confirmation must use identical source and update budget")
        gains = sorted(set((protocol["direct_dose_gain"], protocol["direct_rank_gain"])))
    elif phase == "development":
        gains = list(DIRECT_GAINS)
    else:
        raise ValueError("unknown study phase")
    x, y, rows, manifest = partition(seed)
    result = {
        "status": "running", "phase": phase, "seed": seed, "updates": updates,
        "source_fingerprint": fingerprint, "splits": manifest,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "sklearn": sklearn.__version__},
        "matching_tolerance": MATCH_TOLERANCE, "candidates": {}, "audit": None,
        "contract": "train fits; control drives physics; selection chooses checkpoint; ranking chooses candidate; audit opens last",
    }
    write_result(out, result)
    models = {}
    candidates = [(kind, None) for kind in BASE_ARMS] + [("direct", gain) for gain in gains]
    try:
        for kind, gain in candidates:
            name = kind if gain is None else f"direct_{gain:g}"
            model, row = fit_candidate(kind, gain, seed, updates, x, y, rows)
            models[name] = model
            result["candidates"][name] = row
            write_result(out, result)
            print(json.dumps({"candidate": name, "seed": seed, "ranking": row["ranking_nll"],
                              "last_ranking": row["last_ranking_nll"], "thermal": row["thermal"]}), flush=True)
        direct = {name: row for name, row in result["candidates"].items() if row["kind"] == "direct"}
        target = result["candidates"]["cap"]["thermal"]
        if phase == "development":
            dose_name = min(direct, key=lambda name: exposure_distance(direct[name]["thermal"], target))
            rank_name = min(direct, key=lambda name: direct[name]["ranking_nll"])
            protocol = {
                "development_seed": seed, "updates": updates, "source_fingerprint": fingerprint,
                "direct_dose_gain": direct[dose_name]["gain"],
                "direct_rank_gain": direct[rank_name]["gain"],
                "development_dose_matched": exposure_matched(direct[dose_name]["thermal"], target, MATCH_TOLERANCE),
                "dose_target": "capacitor peak excursion and integrated capacity-weighted thaw exposure",
            }
            result["frozen_protocol"] = protocol
        else:
            result["frozen_protocol"] = protocol
            dose_name = f"direct_{protocol['direct_dose_gain']:g}"
            result["confirmation_dose_matched"] = exposure_matched(direct[dose_name]["thermal"], target, MATCH_TOLERANCE)
            winner = min(result["candidates"], key=lambda name: result["candidates"][name]["ranking_nll"])
            result["ranking_winner"] = winner
            # No score or choice above this point has accessed these observations.
            audit_rows = rows["audit"]
            result["audit"] = {
                name: {"selected_nll": float(log_loss(y[audit_rows], models[name].predict_proba(x[audit_rows]))),
                       "last_nll": float(log_loss(y[audit_rows], models[name].predict_proba(x[audit_rows], last=True)))}
                for name in sorted({"none", winner})
            }
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
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=2048)
    parser.add_argument("--protocol")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("updates must be positive")
    run(args.phase, args.seed, args.updates, args.out, args.protocol)
