"""Whole-forest depth/arity batching over only currently allocated nodes.

No parameter or optimizer ownership changes. No topology cache can become stale.
This trades per-tree operator launch overhead for a batch*live_nodes*outputs
activation tensor. Large-output workloads should benchmark the older backend.
"""
from __future__ import annotations

import torch

from .forest import NodeTrace


def forest_packed_forward(forest, x, *, trace=False, disabled_nodes=frozenset(),
                          disabled_refinements=frozenset(), disabled_tree_ids=frozenset()):
    trees = list(forest.trees)
    tree_slots = {tree.tree_id: index for index, tree in enumerate(trees)}
    feature_masks = {tree.tree_id: tree.feature_mask for tree in trees}
    levels = {}
    for tree in trees:
        if tree.tree_id not in disabled_tree_ids:
            for node in tree.nodes.values():
                levels.setdefault(node.depth, []).append(node)
    reaches = {tree.root_id: x.new_ones(len(x)) for tree in trees
               if tree.tree_id not in disabled_tree_ids}
    outputs = x.new_zeros((len(x), len(trees), forest.output_dim))
    reached, probabilities = [], {}
    for depth in sorted(levels):
        nodes = [node for node in levels[depth] if node.node_id in reaches
                 and node.active and node.node_id not in disabled_nodes]
        if not nodes:
            continue
        reached.extend(nodes)
        mass = torch.stack([reaches[node.node_id] for node in nodes], dim=1)
        values = torch.stack([node.value for node in nodes])
        slots = torch.tensor([tree_slots[node.tree_id] for node in nodes], device=x.device)
        outputs = outputs.index_add(1, slots, mass[..., None] * values[None])
        groups = {}
        for node in nodes:
            if not node.is_leaf and node.node_id not in disabled_refinements:
                groups.setdefault(len(node.children_ids), []).append(node)
        for arity, group in groups.items():
            weights = torch.stack([node.routing_weight * feature_masks[node.tree_id][None]
                                   for node in group])
            bias = torch.stack([node.routing_bias for node in group])
            temperatures = torch.stack([node.temperature for node in group])
            scores = (torch.einsum('bd,kad->bka', x, weights) + bias) / temperatures[None, :, None]
            routing = scores.softmax(dim=2)
            parent = torch.stack([reaches[node.node_id] for node in group], dim=1)
            gates = torch.stack([node.gate() for node in group])
            child_mass = routing * (parent * gates)[..., None]
            for k, node in enumerate(group):
                if trace:
                    probabilities[node.node_id] = routing[:, k]
                for j, child in enumerate(node.children_ids):
                    reaches[child] = child_mass[:, k, j]
    if not trace:
        return [(outputs[:, index], {}) for index in range(len(trees))]
    traces = {}
    zero = x.new_zeros((len(x), forest.output_dim))
    for node in reversed(reached):
        routing = probabilities.get(node.node_id)
        refinement = zero
        if routing is not None:
            children = torch.stack([traces[c].output if c in traces else zero
                                    for c in node.children_ids], dim=1)
            refinement = node.gate() * (routing[..., None] * children).sum(dim=1)
        local = node.value.expand(len(x), -1) + refinement
        traces[node.node_id] = NodeTrace(node.node_id, node.tree_id, reaches[node.node_id],
                                        routing, local, refinement)
    return [(traces[tree.root_id].output if tree.root_id in traces else zero,
             {key: observation for key, observation in traces.items()
              if observation.tree_id == tree.tree_id}) for tree in trees]
