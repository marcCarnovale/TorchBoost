"""Native adaptive forest with explicit learned per-tree front coefficients.

This is the bridge between progressively warm-started differentiable trees and
the node-owned adaptive engine.  It keeps tree identity, routing/value
parameters, and front coefficients explicit so physics/plasticity can act on
native nodes without discarding the boosted initialization.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from .forest import AdaptiveForest


def _breadth(tree):
    nodes = [tree.get(tree.root_id)]
    for node in nodes:
        nodes.extend(tree.get(key) for key in node.children_ids)
    return nodes


class RatedAdaptiveForest(AdaptiveForest):
    def __init__(
        self,
        input_dim,
        output_dim,
        config,
        *,
        rates=None,
        learn_rates=True,
        generator=None,
        schema=None,
    ):
        payload = schema
        if schema is not None:
            rates = schema["rates"]
            learn_rates = bool(schema.get("learn_rates", True))
            payload = {"version_offset": schema.get("version_offset", 0), "trees": schema["trees"]}
        super().__init__(
            input_dim, output_dim, config, generator=generator, schema=payload
        )
        value = self.bias.new_tensor(
            [1.0] * len(self.trees) if rates is None else rates
        )
        if value.ndim != 1 or len(value) != len(self.trees):
            raise ValueError("one scalar rate is required per live tree")
        self.learn_rates = bool(learn_rates)
        self.rates = nn.Parameter(value.clone(), requires_grad=self.learn_rates)

    def coefficient_transform(self, x, coefficients):
        del x
        return coefficients * self.rates[None, :, None]

    def schema(self):
        return {
            **super().schema(),
            "family": "rated-native-v1",
            "rates": self.rates.detach().cpu().tolist(),
            "learn_rates": self.learn_rates,
        }

    def remove_tree(self, tree_id):
        keep = torch.tensor(
            [i for i, t in enumerate(self.trees) if t.tree_id != tree_id],
            device=self.rates.device,
        )
        removed, migrations = super().remove_tree(tree_id)
        if not removed:
            return removed, migrations
        old = self.rates
        new = nn.Parameter(
            old.detach().index_select(0, keep).clone(),
            requires_grad=old.requires_grad,
        )
        self.rates = new
        migrations.append((old, new, keep))
        return removed, migrations

    @torch.no_grad()
    def realized_contributions(self, x):
        outputs = torch.stack([tree(x)[0] for tree in self.trees], dim=1)
        return (outputs * self.rates[None, :, None]).square().mean((0, 2)).sqrt()

    @torch.no_grad()
    def effective_tree_counts(self, x, eps=1e-12):
        c = self.realized_contributions(x).clamp_min(0)
        total = c.sum()
        if float(total) <= eps:
            return {"participation": 0.0, "entropy": 0.0}
        p = c / total
        participation = total.square() / c.square().sum().clamp_min(eps)
        entropy = torch.exp(-(p * p.clamp_min(eps).log()).sum())
        return {
            "participation": float(participation),
            "entropy": float(entropy),
        }


@torch.no_grad()
def materialize_progressive_sum(progressive, *, learn_rates=True):
    """Exactly materialize a ProgressiveSum of packed trees into native nodes."""
    if not getattr(progressive, "trees", None):
        raise ValueError("progressive ensemble must contain at least one tree")
    packed = list(progressive.trees)
    depths = {tree.depth for tree in packed}
    arities = {tree.arity for tree in packed}
    if len(depths) != 1 or len(arities) != 1:
        raise ValueError("materialization currently requires common depth and arity")

    sources = [tree.to_native() for tree in packed]
    cfg = deepcopy(sources[0].config)
    cfg.n_trees = len(sources)
    cfg.aggregation = "additive"
    cfg.residual_weights = False
    cfg.shrinkage = 1.0
    cfg.structure.dynamic = False
    cfg.structure.initial_depth = 0
    cfg.structure.max_depth = next(iter(depths))
    cfg.structure.arity = next(iter(arities))
    cfg.structure.max_nodes = max(
        cfg.structure.max_nodes, max(len(s.trees[0].nodes) for s in sources)
    )
    cfg.structure.max_parameters = max(
        cfg.structure.max_parameters,
        2 * sum(p.numel() for source in sources for p in source.parameters()),
    )
    cfg.physics.mode = "none"
    cfg.plasticity.mode = "none"
    cfg.online.enabled = False
    cfg.collect_metrics = False
    cfg.__post_init__()

    model = RatedAdaptiveForest(
        sources[0].input_dim,
        sources[0].output_dim,
        cfg,
        rates=progressive.rates.detach(),
        learn_rates=learn_rates,
    )
    model.to(device=progressive.bias.device, dtype=progressive.bias.dtype)
    model.bias.copy_(progressive.bias)

    for target_tree, source_model, packed_tree in zip(model.trees, sources, packed):
        target_nodes = _breadth(target_tree)
        source_nodes = _breadth(source_model.trees[0])
        if len(target_nodes) != len(source_nodes):
            raise RuntimeError("native materialization topology mismatch")
        for target, source in zip(target_nodes, source_nodes):
            target.value.copy_(source.value)
            target.temperature.copy_(source.temperature)
            if target.routing_weight is not None:
                target.routing_weight.copy_(source.routing_weight)
                target.routing_bias.copy_(source.routing_bias)
        # Packed trees retain their own scalar bias. Absorb it into the native
        # root, which contributes on every path.
        target_nodes[0].value.add_(packed_tree.bias)
    return model.eval()
