"""Generic regularization trainer for rated native forests."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .progressive_regularizers import Regularizers, penalties
from .training import JointTrainer


@dataclass
class RatedRegularizers:
    structural: Regularizers
    rate_l2: float = 0.0
    count_pressure: float = 0.0

    def __post_init__(self):
        if isinstance(self.structural, dict):
            self.structural = Regularizers(**self.structural)
        for name in ("rate_l2", "count_pressure"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


class RatedJointTrainer(JointTrainer):
    """JointTrainer plus domain-agnostic rated-ensemble penalties."""

    def __init__(self, *args, rated_regularizers: RatedRegularizers, **kwargs):
        self.rated_regularizers = rated_regularizers
        super().__init__(*args, **kwargs)
        if not hasattr(self.model, "rates"):
            raise TypeError("RatedJointTrainer requires a model with front coefficients")

    def _needs_trace(self, values):
        return True

    def _regularization(self, x, prediction, trace, weights, values):
        base = super()._regularization(x, prediction, trace, weights, values)
        cfg = self.rated_regularizers
        shared = penalties(self.model, weights, trace, cfg.structural)
        result = base + sum(shared.values())
        if cfg.rate_l2:
            result = result + cfg.rate_l2 * self.model.rates.square().mean()
        if cfg.count_pressure and len(self.model.trees) > 1:
            realized = trace.tree_outputs * trace.coefficients
            contribution = realized.square().mean((0, 2)).clamp_min(1e-12).sqrt()
            total = contribution.sum()
            effective = total.square() / contribution.square().sum().clamp_min(1e-12)
            normalized = (effective - 1).clamp_min(0) / (len(contribution) - 1)
            result = result + cfg.count_pressure * normalized
        return result
