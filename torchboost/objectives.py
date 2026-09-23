"""Score-space objectives. Derivatives are per example, before sample weighting."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


class BinaryLogisticObjective:
    """Binary negative log-likelihood with analytic, non-placeholder curvature.

    For score F and label y in {0, 1}, l=softplus(F)-y*F,
    g=sigmoid(F)-y, h=sigmoid(F)*(1-sigmoid(F)). Hessians returned here
    are not clipped; any numerical curvature floor belongs to the solver.
    """

    @staticmethod
    def loss(scores: Tensor, targets: Tensor) -> Tensor:
        if scores.shape != targets.shape:
            raise ValueError("scores and targets must have identical shapes")
        return F.binary_cross_entropy_with_logits(scores, targets, reduction="none")

    @staticmethod
    def derivatives(scores: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        if scores.shape != targets.shape:
            raise ValueError("scores and targets must have identical shapes")
        p = torch.sigmoid(scores)
        # This symmetric form avoids cancellation in 1-sigmoid(large_score).
        return p - targets, torch.sigmoid(scores) * torch.sigmoid(-scores)

    @staticmethod
    def initial_score(targets: Tensor, weights: Tensor) -> Tensor:
        rate = (targets * weights).sum() / weights.sum()
        if not 0 < float(rate) < 1:
            raise ValueError("both classes must have positive training weight")
        return torch.logit(rate)
