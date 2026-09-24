"""Explicit task contracts; losses are always per example before weighting."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class Objective:
    task: str
    output_dim: int
    custom_loss: Callable[[Tensor, Tensor], Tensor] | None = None

    def loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.ndim != 2 or prediction.shape[1] != self.output_dim:
            raise ValueError("prediction violates output shape contract")
        if self.custom_loss is not None:
            result = self.custom_loss(prediction, target)
        elif self.task == "binary":
            result = F.binary_cross_entropy_with_logits(prediction[:, 0], target.float().reshape(-1), reduction="none")
        elif self.task == "multiclass":
            result = F.cross_entropy(prediction, target.long().reshape(-1), reduction="none")
        elif self.task == "regression":
            result = (prediction - target.reshape(-1, self.output_dim)).square().mean(1)
        else:
            raise ValueError(f"unknown task {self.task}")
        if result.shape != (len(prediction),):
            raise ValueError("custom/default objective must return one scalar loss per example")
        return result

    def weighted_loss(self, prediction: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        return (self.loss(prediction, target) * weight).sum() / weight.sum().clamp_min(torch.finfo(prediction.dtype).tiny)

    def response(self, prediction: Tensor) -> Tensor:
        if self.task == "binary":
            p = prediction[:, :1].sigmoid()
            return torch.cat((1 - p, p), 1)
        if self.task == "multiclass":
            return prediction.softmax(1)
        return prediction

    def information(self, branches: Tensor, reach_weight: Tensor, target: Tensor) -> float:
        total = reach_weight.sum()
        if total <= 1e-12:
            return 0.
        weights = reach_weight / total
        mass = (branches * weights[:, None]).sum(0)
        if self.task in ("binary", "multiclass"):
            classes = 2 if self.task == "binary" else self.output_dim
            labels = F.one_hot(target.long().reshape(-1), classes).to(branches.dtype)
            joint = branches.T @ (weights[:, None] * labels)
            independent = mass[:, None] * joint.sum(0)[None, :]
            terms = torch.where(joint > 0, joint * (joint.clamp_min(1e-12) / independent.clamp_min(1e-12)).log(), 0.)
            return max(0., float(terms.sum()))
        labels = target.reshape(-1, self.output_dim)
        mean = (weights[:, None] * labels).sum(0)
        variance = (weights[:, None] * (labels - mean).square()).sum(0)
        conditional = branches.T @ (weights[:, None] * labels) / mass[:, None].clamp_min(1e-12)
        explained = (mass[:, None] * (conditional - mean).square()).sum(0)
        return float((explained / variance.clamp_min(1e-12)).clamp(0., 1.).mean())
