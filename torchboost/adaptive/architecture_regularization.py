"""Differentiable regularization for architecture variables.

The architecture variables are continuous gates on function-preserving residual
refinements.  Prediction weights and architecture gates can therefore be
optimized on different data splits (train vs selection) while retaining one
end-to-end differentiable model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class ArchitectureRegularization:
    gate_l1: float = 1e-4
    gate_entropy: float = 0.0
    residual_l2: float = 1e-6
    routing_l1: float = 1e-6

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def differentiable_architecture_penalty(model, cfg: ArchitectureRegularization):
    """Return a scalar penalty and interpretable component diagnostics."""
    anchor = next(model.parameters())
    zero = anchor.new_zeros(())
    gate_cost = zero
    entropy_cost = zero
    residual_cost = zero
    routing_cost = zero

    for layer in getattr(model, "layers", ()):
        if getattr(layer, "_dense_endpoint", True):
            continue
        gate = layer.architecture_gate
        gate_cost = gate_cost + gate

        if cfg.gate_entropy:
            p = gate.clamp(1e-8, 1 - 1e-8)
            entropy_cost = entropy_cost + (
                -p * p.log() - (1 - p) * (1 - p).log()
            )

        tree = layer.forest.trees[0]
        for node in tree.nodes.values():
            if node.node_id == tree.root_id:
                continue
            local = node.value.square().mean()
            if node.linear_value is not None:
                local = local + node.linear_value.square().mean()
            residual_cost = residual_cost + gate * local

        for node in tree.nodes.values():
            if node.routing_weight is not None:
                routing_cost = routing_cost + gate * node.routing_weight.abs().mean()

    total = (
        cfg.gate_l1 * gate_cost
        + cfg.gate_entropy * entropy_cost
        + cfg.residual_l2 * residual_cost
        + cfg.routing_l1 * routing_cost
    )
    return total, {
        "gate_l1_raw": gate_cost,
        "gate_entropy_raw": entropy_cost,
        "residual_l2_raw": residual_cost,
        "routing_l1_raw": routing_cost,
    }


def architecture_state(model) -> dict:
    """Serializable continuous architecture coordinates learned by gradient."""
    rows = []
    for i, layer in enumerate(getattr(model, "layers", ())):
        row = {
            "layer": i,
            "released": not getattr(layer, "_dense_endpoint", True),
            "gate": (
                float(layer.architecture_gate.detach())
                if not getattr(layer, "_dense_endpoint", True)
                else 0.0
            ),
        }
        if row["released"]:
            tree = layer.forest.trees[0]
            row["nodes"] = len(tree.nodes)
            row["depth"] = max(n.depth for n in tree.nodes.values())
            routing = [
                n.routing_weight.detach().abs().mean()
                for n in tree.nodes.values()
                if n.routing_weight is not None
            ]
            row["mean_abs_routing"] = (
                float(torch.stack(routing).mean()) if routing else 0.0
            )
            residual = []
            for n in tree.nodes.values():
                if n.node_id == tree.root_id:
                    continue
                residual.append(n.value.detach().square().mean())
                if n.linear_value is not None:
                    residual.append(n.linear_value.detach().square().mean())
            row["residual_rms"] = (
                float(torch.stack(residual).mean().sqrt()) if residual else 0.0
            )
        rows.append(row)
    return {"layers": rows}
