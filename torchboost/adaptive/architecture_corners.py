"""Exact baseline corners for TorchBoost's architecture space.

ObliviousSoftForest exactly embeds numerical CatBoost symmetric trees in hard
mode, then permits those splits to be softened/rotated and refined.
CompositionalTreeNetwork exactly embeds an ordinary ReLU MLP when every layer is
one depth-zero affine tree. Tree refinements can then be grown from that
function-preserving initialization.

These are representation endpoints. Matching a baseline training algorithm is
a separate experiment and must not be inferred from exact embedding.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .config import ForestConfig, StructureConfig
from .forest import AdaptiveForest


class ObliviousSoftTree(nn.Module):
    """Differentiable symmetric tree with CatBoost-compatible hard semantics."""

    def __init__(self, input_dim: int, depth: int, output_dim: int = 1, *, temperature: float = 1.0):
        super().__init__()
        if input_dim < 1 or output_dim < 1 or depth < 0:
            raise ValueError("invalid oblivious-tree dimensions")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive")
        self.input_dim, self.depth, self.output_dim = input_dim, depth, output_dim
        self.routing_weight = nn.Parameter(torch.zeros(depth, input_dim))
        self.routing_bias = nn.Parameter(torch.zeros(depth))
        self.leaf_values = nn.Parameter(torch.zeros(1 << depth, output_dim))
        self.temperature = nn.Parameter(torch.tensor(float(temperature)), requires_grad=False)
        self.force_hard = True

    def forward(self, x: torch.Tensor, *, hard: bool | None = None) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("x has the wrong feature shape")
        hard = self.force_hard if hard is None else bool(hard)
        if self.depth == 0:
            return self.leaf_values[0].expand(len(x), -1)
        score = x @ self.routing_weight.T + self.routing_bias
        if hard:
            bits = (score > 0).to(torch.long)
            powers = (1 << torch.arange(self.depth, device=x.device, dtype=torch.long))
            index = (bits * powers).sum(1)
            return self.leaf_values[index]
        q = torch.sigmoid(score / self.temperature.clamp_min(1e-4))
        mass = x.new_ones((len(x), 1))
        for j in range(self.depth):
            p = q[:, j:j + 1]
            mass = torch.cat((mass * (1 - p), mass * p), dim=1)
        return mass @ self.leaf_values

    def release(self, *, learn_temperature: bool = False) -> None:
        self.force_hard = False
        self.temperature.requires_grad_(learn_temperature)


class ObliviousSoftForest(nn.Module):
    """Additive symmetric-tree forest that can exactly import numerical CatBoost."""

    def __init__(self, trees: list[ObliviousSoftTree], *, scale: float = 1.0, bias: float = 0.0):
        super().__init__()
        if not trees:
            raise ValueError("at least one tree is required")
        dims = {(t.input_dim, t.output_dim) for t in trees}
        if len(dims) != 1:
            raise ValueError("all trees must share input/output dimensions")
        self.trees = nn.ModuleList(trees)
        self.input_dim, self.output_dim = next(iter(dims))
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.bias = nn.Parameter(torch.full((self.output_dim,), float(bias)))

    def forward(self, x: torch.Tensor, *, hard: bool | None = None) -> torch.Tensor:
        total = torch.stack([tree(x, hard=hard) for tree in self.trees]).sum(0)
        return self.bias + self.scale * total

    def release(self, *, learn_temperature: bool = False) -> None:
        for tree in self.trees:
            tree.release(learn_temperature=learn_temperature)

    @classmethod
    def from_catboost_json(cls, payload: dict) -> "ObliviousSoftForest":
        """Import a numerical scalar-output CatBoost JSON model exactly."""
        floats = payload.get("features_info", {}).get("float_features", [])
        if not floats:
            raise ValueError("CatBoost JSON contains no numerical features")
        feature_map = {
            int(row.get("feature_index", i)): int(
                row.get("flat_feature_index", row.get("feature_index", i))
            )
            for i, row in enumerate(floats)
        }
        input_dim = max(feature_map.values()) + 1
        trees = []
        for source in payload.get("oblivious_trees", []):
            splits = source.get("splits", [])
            depth = len(splits)
            leaf = source.get("leaf_values", [])
            leaves = 1 << depth
            if len(leaf) % leaves:
                raise ValueError("CatBoost leaf-value shape is inconsistent with depth")
            output_dim = len(leaf) // leaves
            tree = ObliviousSoftTree(input_dim, depth, output_dim)
            with torch.no_grad():
                tree.leaf_values.copy_(
                    torch.tensor(leaf, dtype=tree.leaf_values.dtype).reshape(leaves, output_dim)
                )
                for j, split in enumerate(splits):
                    if split.get("split_type") != "FloatFeature":
                        raise ValueError(
                            "exact CatBoost import currently supports FloatFeature splits only"
                        )
                    feature = feature_map[int(split["float_feature_index"])]
                    tree.routing_weight[j, feature] = 1.0
                    tree.routing_bias[j] = -float(split["border"])
            trees.append(tree)
        if not trees:
            raise ValueError("CatBoost JSON contains no oblivious trees")
        scale_and_bias = payload.get("scale_and_bias", [1.0, [0.0]])
        raw_bias = scale_and_bias[1]
        if len(raw_bias) != trees[0].output_dim:
            raise ValueError("CatBoost bias dimension does not match tree output")
        if trees[0].output_dim != 1:
            raise ValueError("exact endpoint currently targets scalar-output CatBoost models")
        return cls(trees, scale=float(scale_and_bias[0]), bias=float(raw_bias[0]))


class ExpandableAffineTreeLayer(nn.Module):
    """One affine tree layer; depth zero is exactly a dense affine transform."""

    def __init__(self, input_dim: int, output_dim: int, *, max_tree_depth: int = 3, seed: int = 0):
        super().__init__()
        structure = StructureConfig(
            dynamic=True,
            initial_depth=0,
            max_depth=max_tree_depth,
            max_nodes=(1 << (max_tree_depth + 1)) - 1,
            max_parameters=max(2_000_000, 20 * input_dim * output_dim),
            structural_gate=False,
            complexity=0.0,
            allocation_regularization=0.0,
            depth_allocation="uniform",
        )
        cfg = ForestConfig(
            n_trees=1,
            aggregation="additive",
            shrinkage=1.0,
            residual_weights=False,
            node_linear_values=True,
            collect_metrics=True,
            structure=structure,
            random_state=seed,
        )
        self.forest = AdaptiveForest(input_dim, output_dim, cfg)
        self.input_dim, self.output_dim = input_dim, output_dim
        with torch.no_grad():
            self.forest.bias.zero_()
        self.forest.bias.requires_grad_(False)
        tree = self.forest.trees[0]
        tree.depth_logits.requires_grad_(False)
        for node in tree.nodes.values():
            node.allocation_logit.requires_grad_(False)

    @property
    def root(self):
        tree = self.forest.trees[0]
        return tree.get(tree.root_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forest(x)

    @torch.no_grad()
    def grow_one_level(self) -> int:
        """Function-preserving expansion: new residual children start at zero."""
        tree = self.forest.trees[0]
        leaves = [
            node.node_id for node in tree.nodes.values()
            if node.is_leaf and node.depth < tree.config.structure.max_depth
        ]
        grown = 0
        generator = torch.Generator().manual_seed(
            self.forest.config.random_state + tree.topology_version + 7001
        )
        for key in leaves:
            children = tree.grow(key, generator=generator)
            if children:
                grown += 1
                for child in children:
                    tree.get(child).allocation_logit.requires_grad_(False)
        return grown


class CompositionalTreeNetwork(nn.Module):
    """Sequential tree layers. Depth-zero layers are exactly an ordinary MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_widths: tuple[int, ...],
        *,
        dropout: float = 0.1,
        max_tree_depth: int = 3,
        seed: int = 0,
    ):
        super().__init__()
        if not hidden_widths or min(hidden_widths) < 1:
            raise ValueError("hidden_widths must be nonempty and positive")
        dims = (input_dim,) + tuple(hidden_widths)
        self.layers = nn.ModuleList([
            ExpandableAffineTreeLayer(
                dims[i], dims[i + 1], max_tree_depth=max_tree_depth, seed=seed + i
            )
            for i in range(len(hidden_widths))
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_widths[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.dropout(torch.relu(layer(x)))
        return self.head(x).squeeze(1)

    @classmethod
    def from_mlp(
        cls, mlp: nn.Module, *, max_tree_depth: int = 3, seed: int = 0
    ) -> "CompositionalTreeNetwork":
        sequence = getattr(mlp, "net", None)
        if sequence is None:
            raise ValueError("expected an MLP with a .net Sequential module")
        linears = [m for m in sequence if isinstance(m, nn.Linear)]
        if len(linears) < 2:
            raise ValueError("MLP must contain hidden and output linear layers")
        relus = sum(isinstance(m, nn.ReLU) for m in sequence)
        if relus != len(linears) - 1:
            raise ValueError("exact endpoint currently supports ReLU hidden layers")
        drops = [m.p for m in sequence if isinstance(m, nn.Dropout)]
        if drops and (len(drops) != len(linears) - 1 or max(drops) != min(drops)):
            raise ValueError(
                "exact endpoint requires one common dropout probability per hidden layer"
            )
        model = cls(
            linears[0].in_features,
            tuple(layer.out_features for layer in linears[:-1]),
            dropout=drops[0] if drops else 0.0,
            max_tree_depth=max_tree_depth,
            seed=seed,
        )
        with torch.no_grad():
            for target, source in zip(model.layers, linears[:-1]):
                target.root.linear_value.copy_(source.weight.T)
                target.root.value.copy_(source.bias)
            model.head.weight.copy_(linears[-1].weight)
            model.head.bias.copy_(linears[-1].bias)
        return model

    @torch.no_grad()
    def grow_one_level(self) -> int:
        return sum(layer.grow_one_level() for layer in self.layers)


def canonical_catboost_corner() -> dict:
    """Structural/training target matching the frozen HIGGS CatBoost baseline."""
    return {
        "iterations": 1536,
        "depth": 10,
        "learning_rate": 0.05,
        "l2_leaf_reg": 20.0,
        "grow_policy": "SymmetricTree",
        "boosting_type": "Plain",
        "border_count": 254,
        "routing": "hard_axis_aligned_shared_by_depth",
        "aggregation": "stagewise_additive",
        "differentiable_refinement": False,
    }


def canonical_mlp_corner() -> dict:
    """Exact architectural target matching the frozen HIGGS MLP baseline."""
    return {
        "hidden_widths": [300] * 5,
        "tree_depth": 0,
        "trees_per_layer": 1,
        "node_packet": "affine",
        "activation": "relu",
        "dropout": 0.1,
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "epochs": 20,
        "batch_size": 4096,
        "training": "full_end_to_end",
    }
