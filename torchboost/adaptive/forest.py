"""Joint differentiable forests with true, ragged, dynamically allocated topology.

A node represents a residual function
    f_n(x) = v_n + s_n * sum_c p_nc(x) f_c(x),   0 <= s_n <= 1.
Removing a zero refinement is an exact no-op. A new leaf-to-split expansion
initializes every child residual to zero and therefore preserves the function.
No eventual full tree or dead branch tensor is preallocated in dynamic mode.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator

import torch
from torch import Tensor, nn

from .config import ForestConfig


@dataclass
class NodeTrace:
    node_id: str
    tree_id: int
    reach: Tensor
    probabilities: Tensor | None
    output: Tensor
    refinement: Tensor


@dataclass
class ForestTrace:
    nodes: dict[str, NodeTrace]
    coefficients: Tensor                   # [batch, tree, output]
    tree_outputs: Tensor                  # [batch, tree, output]
    tree_slots: dict[int, int]             # stable tree ID -> current packed slot


class ResidualNode(nn.Module):
    def __init__(self, node_id: str, tree_id: int, depth: int, input_dim: int,
                 output_dim: int, temperature: float, linear_values: bool = False):
        super().__init__()
        self.node_id, self.tree_id, self.depth = node_id, tree_id, depth
        self.input_dim, self.output_dim = input_dim, output_dim
        self.children_ids: list[str] = []
        self.parent_id: str | None = None
        self.active = True
        self.frozen = False
        self.locked = False
        self.value = nn.Parameter(torch.zeros(output_dim))
        self.linear_value = nn.Parameter(torch.zeros(input_dim, output_dim), requires_grad=linear_values) if linear_values else None
        self.register_parameter("routing_weight", None)
        self.register_parameter("routing_bias", None)
        self.register_parameter("structural", None)
        self.allocation_logit = nn.Parameter(torch.zeros(()))
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    @property
    def is_leaf(self) -> bool:
        return not self.children_ids

    def gate(self, hard: bool = False) -> Tensor:
        if self.structural is None:
            return self.value.new_tensor(1.)
        return (self.structural >= .5).to(self.value.dtype) if hard else self.structural.clamp(0., 1.)

    def parameters_for_plasticity(self) -> dict[str, nn.Parameter]:
        return {name: p for name, p in self.named_parameters(recurse=False)
                if name != "allocation_logit"}

    def no_op_reference(self, name: str, parameter: Tensor) -> Tensor:
        # The zero packet is a genuine zero residual: v=0 and s=0, irrespective
        # of the shape of the oblique routing matrix. Not a rectangular identity.
        return torch.zeros_like(parameter.detach())

    def set_frozen(self, frozen: bool, *, lock: bool = False) -> None:
        if self.locked and not frozen:
            return
        self.frozen = bool(frozen)
        self.locked = self.locked or bool(lock)
        for p in self.parameters():
            p.requires_grad_(not self.frozen)
            if self.frozen:
                p.grad = None

    @torch.no_grad()
    def set_temperature(self, value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("temperature must be finite and positive")
        self.temperature.fill_(value)


class RaggedTree(nn.Module):
    def __init__(self, tree_id: int, input_dim: int, output_dim: int, config: ForestConfig,
                 generator: torch.Generator, schema: dict | None = None):
        super().__init__()
        self.tree_id, self.input_dim, self.output_dim = tree_id, input_dim, output_dim
        self.config = config
        self.nodes = nn.ModuleDict()
        self.next_id = 0
        self.topology_version = 0
        self.root_id = ""
        self.depth_logits = nn.Parameter(torch.zeros(config.structure.max_depth + 1))
        mask = torch.ones(input_dim)
        if config.interaction_groups:
            mask.zero_()
            group = config.interaction_groups[tree_id % len(config.interaction_groups)]
            if not group or min(group) < 0 or max(group) >= input_dim:
                raise ValueError("interaction group contains no features or an invalid feature")
            mask[list(group)] = 1.
        self.register_buffer("feature_mask", mask)
        if schema is not None:
            self._load_schema(schema, generator)
        else:
            root = self._new_node(0)
            self.root_id = root.node_id
            initial = config.structure.initial_depth if config.structure.dynamic else config.structure.max_depth
            for _ in range(initial):
                for node in list(self.nodes.values()):
                    if node.is_leaf:
                        self.grow(node.node_id, generator=generator)

    def _key(self, node_id: str) -> str:
        return node_id.replace(":", "_")

    def get(self, node_id: str) -> ResidualNode:
        return self.nodes[self._key(node_id)]

    def _new_node(self, depth: int, node_id: str | None = None) -> ResidualNode:
        if node_id is None:
            node_id = f"{self.tree_id}:{self.next_id}"
            self.next_id += 1
        node = ResidualNode(node_id, self.tree_id, depth, self.input_dim, self.output_dim,
                            self.config.physics.initial_temperature, self.config.node_linear_values)
        node.to(device=self.depth_logits.device, dtype=self.depth_logits.dtype)
        self.nodes[self._key(node_id)] = node
        return node

    def grow(self, node_id: str, *, generator: torch.Generator, arity: int | None = None) -> list[str]:
        node = self.get(node_id)
        arity = self.config.structure.arity if arity is None else arity
        if not isinstance(arity, int) or arity < 2:
            raise ValueError("arity must be an integer >= 2")
        if not node.is_leaf or node.depth >= self.config.structure.max_depth or node.locked or node.frozen:
            return []
        if len(self.nodes) + arity > self.config.structure.max_nodes:
            return []
        # Existing value retains its identity and optimizer state.
        weight = torch.randn(arity, self.input_dim, generator=generator) / math.sqrt(self.input_dim)
        node.routing_weight = nn.Parameter(weight.to(node.value))
        node.routing_bias = nn.Parameter(torch.zeros(arity, device=node.value.device, dtype=node.value.dtype))
        if self.config.structure.structural_gate:
            node.structural = nn.Parameter(node.value.new_tensor(1.))
        result = []
        for _ in range(arity):
            child = self._new_node(node.depth + 1)
            child.parent_id = node_id
            result.append(child.node_id)
        node.children_ids = result
        if node.frozen:
            node.set_frozen(True)
        self.topology_version += 1
        return result

    def descendants(self, node_id: str) -> list[str]:
        result: list[str] = []
        for child_id in self.get(node_id).children_ids:
            result.append(child_id)
            result.extend(self.descendants(child_id))
        return result

    def prune(self, node_id: str) -> list[str]:
        """Collapse only the refinement; retain the node's bypass value.

        The caller must accept any nonzero functional change before committing.
        Optimizer/controller/tracker reconciliation is the trainer's transaction.
        """
        node = self.get(node_id)
        if node.is_leaf or node.locked:
            return []
        removed = self.descendants(node_id)
        if any(self.get(key).locked for key in removed):
            return []
        for child_id in reversed(removed):
            del self.nodes[self._key(child_id)]
        node.children_ids = []
        node.routing_weight = None
        node.routing_bias = None
        node.structural = None
        self.topology_version += 1
        return removed

    def set_frozen(self, node_id: str, frozen: bool, *, recursive: bool = False, lock: bool = False) -> None:
        for key in [node_id] + (self.descendants(node_id) if recursive else []):
            self.get(key).set_frozen(frozen, lock=lock)

    def deactivate(self, node_id: str, active: bool = False) -> None:
        self.get(node_id).active = bool(active)
        self.topology_version += 1

    @torch.no_grad()
    def reinitialize(self, node_id: str, generator: torch.Generator) -> None:
        node = self.get(node_id)
        if node.locked:
            raise ValueError("a locked node cannot be reinitialized")
        node.value.zero_()
        if node.linear_value is not None: node.linear_value.zero_()
        if node.routing_weight is not None:
            new = torch.randn(node.routing_weight.shape, generator=generator) / math.sqrt(self.input_dim)
            node.routing_weight.copy_(new.to(node.value))
            node.routing_bias.zero_()
            if node.structural is not None:
                node.structural.fill_(1.)
        self.topology_version += 1

    def forward(self, x: Tensor, *, hard: bool = False, trace: bool = False,
                disabled_nodes: frozenset[str] = frozenset(),
                disabled_refinements: frozenset[str] = frozenset()) -> tuple[Tensor, dict[str, NodeTrace]]:
        x = x * self.feature_mask
        if self.config.execution in ("packed", "forest_packed"):
            if hard and not trace and not self.training:
                return self._hard_path_forward(x, disabled_nodes, disabled_refinements), {}
            return self._packed_forward(x, hard, trace, disabled_nodes, disabled_refinements)
        observations: dict[str, NodeTrace] = {}

        def visit(key: str, reach: Tensor) -> Tensor:
            node = self.get(key)
            if not node.active or key in disabled_nodes:
                return x.new_zeros((len(x), self.output_dim))
            value = node.value.expand(len(x), -1)
            if node.linear_value is not None: value = value + x @ node.linear_value
            refinement = torch.zeros_like(value)
            probabilities = None
            if not node.is_leaf and key not in disabled_refinements:
                score = (x @ node.routing_weight.T + node.routing_bias) / node.temperature
                probabilities = (torch.nn.functional.one_hot(score.argmax(1), len(node.children_ids)).to(x.dtype)
                                 if hard else score.softmax(dim=1))
                gate = node.gate(hard)
                children = torch.stack([visit(child, reach * gate * probabilities[:, j])
                                        for j, child in enumerate(node.children_ids)], dim=1)
                refinement = gate * (probabilities.unsqueeze(-1) * children).sum(1)
            output = value + refinement
            if trace:
                observations[key] = NodeTrace(key, self.tree_id, reach, probabilities, output, refinement)
            return output

        return visit(self.root_id, x.new_ones(len(x))), observations

    def _hard_path_forward(self, x: Tensor, disabled_nodes: frozenset[str],
                           disabled_refinements: frozenset[str]) -> Tensor:
        """Route only visited examples: no dense all-node hard inference pass.

        Hard ties select the first child, identically to the reference engine.
        The explicit stack also avoids a Python recursion-depth limit.
        """
        result = x.new_zeros((len(x), self.output_dim))
        pending = [(self.root_id, torch.arange(len(x), device=x.device))]
        while pending:
            key, indices = pending.pop()
            node = self.get(key)
            if not len(indices) or not node.active or key in disabled_nodes:
                continue
            local = node.value.expand(len(indices), -1)
            if node.linear_value is not None: local = local + x[indices] @ node.linear_value
            result = result.index_add(0, indices, local)
            if node.is_leaf or key in disabled_refinements or float(node.gate(hard=True)) == 0.:
                continue
            score = x[indices] @ node.routing_weight.T + node.routing_bias
            selected = score.argmax(1)
            for j, child in enumerate(node.children_ids):
                subset = indices[selected == j]
                if len(subset):
                    pending.append((child, subset))
        return result

    def _packed_forward(self, x: Tensor, hard: bool, trace: bool,
                        disabled_nodes: frozenset[str], disabled_refinements: frozenset[str]):
        """Batch nodes by depth/arity without allocating absent topology.

        Soft execution evaluates every live branch. Its temporary routing cost
        is O(batch * live nodes), not the capacity of a hypothetical full tree.
        The reference path and this path share parameters and gate semantics.
        """
        levels: dict[int, list[ResidualNode]] = {}
        for node in self.nodes.values():
            levels.setdefault(node.depth, []).append(node)
        reaches = {self.root_id: x.new_ones(len(x))}
        probabilities = {}
        output = x.new_zeros((len(x), self.output_dim))
        reached = []
        for depth in sorted(levels):
            nodes = [n for n in levels[depth] if n.node_id in reaches
                     and n.active and n.node_id not in disabled_nodes]
            if not nodes:
                continue
            reached.extend(nodes)
            mass = torch.stack([reaches[n.node_id] for n in nodes], 1)
            values = torch.stack([n.value for n in nodes])
            if nodes[0].linear_value is None:
                output = output + mass @ values
            else:
                linear = torch.stack([n.linear_value for n in nodes])
                local = values[None,:,:] + torch.einsum("nd,kdo->nko", x, linear)
                output = output + (mass[...,None] * local).sum(1)
            groups: dict[int, list[ResidualNode]] = {}
            for node in nodes:
                if not node.is_leaf and node.node_id not in disabled_refinements:
                    groups.setdefault(len(node.children_ids), []).append(node)
            for arity, group in groups.items():
                weights = torch.stack([n.routing_weight for n in group])
                bias = torch.stack([n.routing_bias for n in group])
                temperature = torch.stack([n.temperature for n in group])
                scores = (torch.einsum("nd,kad->nka", x, weights) + bias) / temperature[None, :, None]
                p = (torch.nn.functional.one_hot(scores.argmax(2), arity).to(x.dtype)
                     if hard else scores.softmax(2))
                parent_mass = torch.stack([reaches[n.node_id] for n in group], 1)
                gates = torch.stack([n.gate(hard) for n in group])
                child_mass = p * (parent_mass * gates)[..., None]
                for k, node in enumerate(group):
                    if trace:
                        probabilities[node.node_id] = p[:, k]
                    for j, child in enumerate(node.children_ids):
                        reaches[child] = child_mass[:, k, j]
        if not trace:
            return output, {}
        # Explicit reverse topological evaluation supplies local counterfactual
        # functions. It does not recursively traverse a potentially deep tree.
        traces: dict[str, NodeTrace] = {}
        zero = x.new_zeros((len(x), self.output_dim))
        for node in reversed(reached):
            p = probabilities.get(node.node_id)
            refinement = zero
            if p is not None:
                children = torch.stack([traces[c].output if c in traces else zero for c in node.children_ids], 1)
                refinement = node.gate(hard) * (p[..., None] * children).sum(1)
            local = node.value.expand(len(x), -1) + refinement
            traces[node.node_id] = NodeTrace(node.node_id, self.tree_id, reaches[node.node_id], p, local, refinement)
        return traces[self.root_id].output if self.root_id in traces else zero, traces

    def schema(self) -> dict:
        return {"tree_id": self.tree_id, "root_id": self.root_id, "next_id": self.next_id,
                "topology_version": self.topology_version,
                "nodes": [{"node_id": n.node_id, "depth": n.depth, "children_ids": n.children_ids,
                           "parent_id": n.parent_id, "active": n.active, "frozen": n.frozen,
                           "locked": n.locked, "has_structural": n.structural is not None}
                          for n in self.nodes.values()]}

    def _load_schema(self, schema: dict, generator: torch.Generator) -> None:
        for description in schema["nodes"]:
            node = self._new_node(description["depth"], description["node_id"])
            node.parent_id = description["parent_id"]
            node.children_ids = list(description["children_ids"])
            if node.children_ids:
                node.routing_weight = nn.Parameter(torch.zeros(len(node.children_ids), self.input_dim))
                node.routing_bias = nn.Parameter(torch.zeros(len(node.children_ids)))
                if description["has_structural"]:
                    node.structural = nn.Parameter(torch.ones(()))
            node.active = description["active"]
            node.set_frozen(description["frozen"], lock=description["locked"])
        self.root_id = schema["root_id"]
        self.next_id = schema["next_id"]
        self.topology_version = schema["topology_version"]


class AdaptiveForest(nn.Module):
    """Jointly trained shared- or specialized-head attention forest."""
    def __init__(self, input_dim: int, output_dim: int, config: ForestConfig | None = None,
                 *, generator: torch.Generator | None = None, schema: dict | list[dict] | None = None):
        super().__init__()
        if min(input_dim, output_dim) < 1:
            raise ValueError("input and output dimensions must be positive")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.config = config or ForestConfig()
        generator = generator or torch.Generator().manual_seed(self.config.random_state)
        self.head_count = output_dim if self.config.head_mode == "specialized" else 1
        self.version_offset = int(schema.get("version_offset", 0)) if isinstance(schema, dict) else 0
        descriptions = schema["trees"] if isinstance(schema, dict) else schema
        tree_ids = [d["tree_id"] for d in descriptions] if descriptions is not None else list(range(self.config.n_trees))
        if not tree_ids or len(set(tree_ids)) != len(tree_ids):
            raise ValueError("forest schema needs unique live tree identities")
        self.trees = nn.ModuleList([RaggedTree(tree_id, input_dim, output_dim, self.config, generator,
                                             descriptions[j] if descriptions is not None else None)
                                   for j, tree_id in enumerate(tree_ids)])
        count = len(self.trees)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.attention_weight = nn.Parameter(.02 * torch.randn(count, self.head_count,
                                                              input_dim, generator=generator))
        self.attention_bias = nn.Parameter(torch.zeros(count, self.head_count))
        self.residual_logits = nn.Parameter(torch.full((count, 1), 2.))
        self.attention_weight.requires_grad_(self.config.aggregation == "attention")
        self.attention_bias.requires_grad_(self.config.aggregation == "attention")
        self.residual_logits.requires_grad_(self.config.residual_weights)
        self.feature_dropout = self.config.feature_dropout
        self.tree_dropout = self.config.tree_dropout
        self.to(self.config.device)
        if sum(p.numel() for p in self.parameters()) > self.config.structure.max_parameters:
            raise ValueError("initial forest exceeds max_parameters")

    @property
    def topology_version(self) -> int:
        return self.version_offset + sum(t.topology_version for t in self.trees)

    def iter_nodes(self) -> Iterator[ResidualNode]:
        for tree in self.trees:
            yield from tree.nodes.values()

    def node_map(self) -> dict[str, ResidualNode]:
        return {n.node_id: n for n in self.iter_nodes()}

    def forward(self, x: Tensor, *, hard: bool = False, trace: bool = False,
                generator: torch.Generator | None = None,
                disabled_nodes: frozenset[str] = frozenset(),
                disabled_refinements: frozenset[str] = frozenset(),
                disabled_tree_ids: frozenset[int] = frozenset()) -> Tensor | tuple[Tensor, ForestTrace]:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("x has the wrong feature shape")
        if self.training and self.feature_dropout:
            mask = torch.rand(x.shape, generator=generator, device="cpu").to(x.device) >= self.feature_dropout
            x = x * mask / (1. - self.feature_dropout)
        if self.config.execution == "forest_packed" and not self.config.node_linear_values and not hard and not any(getattr(t, "force_hard", False) for t in self.trees):
            from .forest_packed import forest_packed_forward
            results = forest_packed_forward(self, x, trace=trace, disabled_nodes=disabled_nodes,
                                            disabled_refinements=disabled_refinements,
                                            disabled_tree_ids=disabled_tree_ids)
        else:
            results = [(x.new_zeros((len(x), self.output_dim)), {}) if t.tree_id in disabled_tree_ids
                       else t(x, hard=hard or getattr(t, "force_hard", False), trace=trace, disabled_nodes=disabled_nodes,
                              disabled_refinements=disabled_refinements) for t in self.trees]
        outputs = torch.stack([r[0] for r in results], dim=1)
        active = torch.tensor([t.get(t.root_id).active and t.tree_id not in disabled_tree_ids
                               for t in self.trees], device=x.device)
        if not active.any():
            raise RuntimeError("all trees have been deactivated")
        if self.training and self.tree_dropout:
            keep = torch.rand(len(self.trees), generator=generator, device="cpu").to(x.device) >= self.tree_dropout
            keep &= active
            if not keep.any():
                keep[torch.where(active)[0][0]] = True
            active = keep
        if self.config.aggregation == "attention":
            scores = torch.einsum("nd,thd->nth", x, self.attention_weight) + self.attention_bias
            weights = scores.masked_fill(~active[None, :, None], -torch.inf).softmax(1)
        else:
            weights = active.to(x.dtype)[None, :, None].expand(len(x), -1, self.head_count)
            if self.config.aggregation == "mean":
                weights = weights / active.sum()
        if self.head_count == 1:
            weights = weights.expand(-1, -1, self.output_dim)
        residual = self.residual_logits.sigmoid() if self.config.residual_weights else torch.ones_like(self.residual_logits)
        coefficients = self.coefficient_transform(x, weights * residual * self.config.shrinkage)
        result = self.bias + (coefficients * outputs).sum(1)
        if trace:
            return result, ForestTrace({key: val for _, traces in results for key, val in traces.items()},
                                       coefficients, outputs, {t.tree_id: i for i, t in enumerate(self.trees)})
        return result

    def coefficient_transform(self, x, coefficients):
        return coefficients

    @torch.no_grad()
    def project(self) -> None:
        for node in self.iter_nodes():
            if node.structural is not None:
                node.structural.clamp_(0., 1.)

    def get_tree(self, tree_id: int) -> RaggedTree:
        for tree in self.trees:
            if tree.tree_id == tree_id:
                return tree
        raise KeyError(tree_id)

    def remove_tree(self, tree_id: int):
        """Physically remove a tree and its attention/residual rows.

        The trainer must first approve the resulting attention renormalization
        and must migrate optimizer row state using the returned transactions.
        Surviving tree/node identities never change when packed slots shift.
        """
        tree = self.get_tree(tree_id)
        if len(self.trees) <= 1 or any(n.locked for n in tree.nodes.values()):
            return [], []
        remaining = [t for t in self.trees if t.tree_id != tree_id]
        if not any(t.get(t.root_id).active for t in remaining):
            return [], []
        keep = torch.tensor([i for i, t in enumerate(self.trees) if t.tree_id != tree_id], device=self.bias.device)
        migrations = []
        for name in ("attention_weight", "attention_bias", "residual_logits"):
            old = getattr(self, name)
            new = nn.Parameter(old.detach().index_select(0, keep).clone(), requires_grad=old.requires_grad)
            setattr(self, name, new)
            migrations.append((old, new, keep))
        self.trees = nn.ModuleList(remaining)
        self.version_offset += tree.topology_version + 1
        return [n.node_id for n in tree.nodes.values()], migrations

    def schema(self) -> dict:
        return {"version_offset": self.version_offset, "trees": [tree.schema() for tree in self.trees]}

    def tensor_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in list(self.parameters()) + list(self.buffers()))