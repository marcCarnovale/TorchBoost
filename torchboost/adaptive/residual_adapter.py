"""Reusable frozen-backbone residual-tree adapter.

This module contains the mechanism that earned the replicated HIGGS result.
Experiment scripts own data splits and optimization schedules; model surgery is
centralized here so HIGGS and transfer studies cannot silently diverge.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

from torch import nn

SIGMOID_MINUS_TWO = 1.0 / (1.0 + math.exp(2.0))


@dataclass(frozen=True)
class ResidualAdapterConfig:
    """Frozen structural configuration for one-level residual adapters."""

    initial_scale: float = SIGMOID_MINUS_TWO

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_scale) or self.initial_scale <= 0:
            raise ValueError("initial_scale must be finite and positive")


def grow_frozen_backbone_adapter(
    anchor: nn.Module,
    *,
    learn_scales: bool,
    config: ResidualAdapterConfig | None = None,
) -> nn.Module:
    """Grow one zero-at-birth residual refinement in every compositional layer.

    The inherited affine backbone and output head are frozen. New routing and
    residual child packets are trainable. Layerwise architecture scales are
    either fixed or trainable, depending on learn_scales.

    Growth is function-preserving because newborn child packets are zero.
    """
    cfg = config or ResidualAdapterConfig()
    model = deepcopy(anchor)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError("anchor must expose compositional .layers")

    for layer in layers:
        grown = layer.grow_one_level()
        if grown < 1:
            raise RuntimeError("adapter growth produced no residual children")
        layer.set_architecture_scale(cfg.initial_scale, learnable=learn_scales)

        tree = layer.forest.trees[0]
        root = layer.root
        root.value.requires_grad_(False)
        root.linear_value.requires_grad_(False)
        if root.routing_weight is not None:
            root.routing_weight.requires_grad_(True)
        if root.routing_bias is not None:
            root.routing_bias.requires_grad_(True)

        if not root.children_ids:
            raise RuntimeError("adapter root has no residual children after growth")
        for child_id in root.children_ids:
            child = tree.get(child_id)
            child.value.requires_grad_(True)
            if child.linear_value is not None:
                child.linear_value.requires_grad_(True)
            child.allocation_logit.requires_grad_(False)

    return model


def partition_adapter_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return residual parameters and architecture-scale parameters."""
    residual: list[nn.Parameter] = []
    scales: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("architecture_logit"):
            scales.append(parameter)
        else:
            residual.append(parameter)
    if not residual:
        raise RuntimeError("adapter has no trainable residual parameters")
    return residual, scales


def adapter_signature(model: nn.Module) -> dict:
    """Machine-readable structural signature for experiment artifacts."""
    residual, scales = partition_adapter_parameters(model)
    layers = getattr(model, "layers", ())
    return {
        "layers": len(layers),
        "residual_parameters": int(sum(p.numel() for p in residual)),
        "scale_parameters": int(sum(p.numel() for p in scales)),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
    }
