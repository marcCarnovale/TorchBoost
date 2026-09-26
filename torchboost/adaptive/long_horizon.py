"""Diagnostic contracts for long training; never a double-descent proof detector."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class LongHorizonPlan:
    epochs: int = 8192
    checkpoints: tuple[int, ...] = (32, 128, 512, 2048, 8192)
    interpolation_patience: int = 5
    regression_mse_threshold: float = 1e-6
    post_interpolation_multiple: float = 10.

    def __post_init__(self):
        if not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if (not self.checkpoints or tuple(sorted(set(self.checkpoints))) != self.checkpoints
                or self.checkpoints[0] < 1 or self.checkpoints[-1] != self.epochs):
            raise ValueError("checkpoints must be increasing unique positive epochs ending at the horizon")
        if self.interpolation_patience < 1:
            raise ValueError("interpolation_patience must be positive")
        for value in (self.regression_mse_threshold, self.post_interpolation_multiple):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("diagnostic thresholds must be finite and positive")

    def validate_config(self, config):
        if config.epochs != self.epochs:
            raise ValueError("declare the full horizon before training; do not restart schedules at each checkpoint")
        if not config.record_diagnostics:
            raise ValueError("record_diagnostics is required")
        if config.plasticity.terminal_lock or config.structure.rolling_freeze or config.freeze_windows:
            raise ValueError("unlocked long-horizon arm must not terminate parameter learning through freeze/lock policies")
        schedule = config.schedules.get("learning_rate")
        if schedule is not None and any(schedule.value(float(t)) <= 0 for t in np.linspace(0, 1, 1001)):
            raise ValueError("the diagnostic arm requires a strictly positive learning rate throughout")


@torch.no_grad()
def training_diagnostics(model, prediction, target, weight, task):
    """Use post-transition deterministic TRAIN predictions, never a test split.

    Classification interpolation means no errors on positive-weight training
    examples, not cross entropy equal to zero. Regression records weighted MSE
    in the trainer's target coordinates (standardized by the public estimator).
    Trainable counts are syntactic parameter counts, not effective complexity.
    """
    positive = weight > 0
    if not positive.any():
        raise ValueError("at least one positive-weight training example is required")
    prediction, target, weight = prediction[positive], target[positive], weight[positive]
    nodes = list(model.iter_nodes())
    temperature = [float(node.temperature) for node in nodes]
    result = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_nodes": sum(node.frozen for node in nodes),
        "locked_nodes": sum(node.locked for node in nodes),
        "temperature_min": min(temperature), "temperature_max": max(temperature),
        "temperature_mean": float(np.mean(temperature)),
        "parameter_l2": math.sqrt(sum(float(p.detach().square().sum()) for p in model.parameters())),
        "topology_version": model.topology_version,
    }
    if task in ("binary", "multiclass"):
        labels = target.long().reshape(-1)
        if task == "binary":
            correct = (prediction[:, 0] >= 0).long() == labels
            margins = prediction[:, 0] * (2 * labels - 1)
        else:
            correct = prediction.argmax(1) == labels
            other = prediction.clone()
            true_score = prediction.gather(1, labels[:, None])[:, 0]
            other.scatter_(1, labels[:, None], -torch.inf)
            margins = true_score - other.max(1).values
        result.update({"train_errors": int((~correct).sum()),
                       "train_error_rate": float((~correct).float().mean()),
                       "train_weighted_error_rate": float(((~correct).float() * weight).sum() / weight.sum()),
                       "train_margin_min": float(margins.min()),
                       "train_margin_p05": float(torch.quantile(margins, .05))})
    else:
        errors = (prediction - target.reshape_as(prediction)).square().mean(1)
        result["train_mse"] = float((errors * weight).sum() / weight.sum())
    return result


def summarize_trajectory(history, task, plan):
    """Return observed coverage. Shape screening is explicitly exploratory.

    A validation-curve peak is not sufficient evidence of epoch-wise double
    descent. Topology changes, schedules, adaptation and random fluctuations
    can all produce peaks. Nothing here examines held-out test losses.
    """
    if not history:
        raise ValueError("history is empty")
    checks = [(row["train_errors"] == 0 if task in ("binary", "multiclass")
               else row["train_mse"] <= plan.regression_mse_threshold) for row in history]
    run = 0
    first = None
    for i, interpolated in enumerate(checks):
        run = run + 1 if interpolated else 0
        if run >= plan.interpolation_patience:
            first = i - plan.interpolation_patience + 1
            break
    result = {
        "epochs_completed": history[-1]["epoch"] + 1,
        "budget_complete": history[-1]["epoch"] + 1 == plan.epochs,
        "optimizer_steps": history[-1]["optimizer_steps"],
        "examples_seen": history[-1]["examples_seen"],
        "counter_origin_epoch": history[-1].get("counter_origin_epoch", 0),
        "first_sustained_interpolation_epoch": None if first is None else history[first]["epoch"] + 1,
        "final_interpolates": bool(checks[-1]),
        "late_interpolation_fraction": float(np.mean(checks[-max(1, len(checks)//5):])),
        "diagnostic_only": True,
        "double_descent_established": False,
    }
    if first is not None:
        before = history[first]["optimizer_steps"]
        after = history[-1]["optimizer_steps"] - before
        result.update({"updates_after_first_interpolation": after,
                       "post_to_pre_interpolation_update_ratio": after / max(1, before),
                       "target_post_interpolation_coverage_met": after >= plan.post_interpolation_multiple * before})
    else:
        result.update({"updates_after_first_interpolation": None,
                       "post_to_pre_interpolation_update_ratio": None,
                       "target_post_interpolation_coverage_met": False})
    # Fixed windows, specified independently of any resulting curve: first,
    # middle and final thirds. Screening only, not optimized peak selection.
    if len(history) >= 30:
        windows = np.array_split(np.array([h['selection_score'] for h in history]), 3)
        medians = [float(np.median(window)) for window in windows]
        result['selection_third_medians'] = medians
        result['nonmonotone_selection_screen'] = bool(medians[1] > medians[0] and medians[2] < medians[0])
    return result
