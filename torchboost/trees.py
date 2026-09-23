"""Binary differentiable routing with one explicit owner of gate temperature."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RoutingTrace:
    """Ephemeral routing tensors, not retained on the model or across batches."""
    reach: Tensor       # [examples, internal nodes], probability of reaching node
    left: Tensor        # [examples, internal nodes], conditional left probability
    leaves: Tensor      # [examples, leaves], unconditional leaf probabilities


class BinarySoftTree(nn.Module):
    """Complete binary tree with oblique splits and real-valued leaf scores.

    Internal nodes use breadth-first heap indices; the left child is 2*j+1.
    A positive affine score routes left. Temperature is a positive buffer, not
    simultaneously a learned parameter and a scheduled value. Forward passes
    are pure: no controller, metrics history, or topology is mutated.

    This reference implementation allocates the complete tree. It deliberately
    does not claim dynamic sparsity; maximum depth is bounded to avoid accidental
    exponential allocations. Inputs must already be finite and preprocessed.
    """

    def __init__(self, input_dim: int, depth: int = 3, temperature: float = 1.0,
                 seed: int = 0):
        super().__init__()
        if not isinstance(input_dim, int) or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        if not isinstance(depth, int) or not 1 <= depth <= 10:
            raise ValueError("depth must be an integer in [1, 10] for dense trees")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.input_dim, self.depth = input_dim, depth
        self.num_nodes, self.num_leaves = 2**depth - 1, 2**depth
        generator = torch.Generator().manual_seed(seed)
        self.weights = nn.Parameter(torch.randn(self.num_nodes, input_dim,
                                               generator=generator) / math.sqrt(input_dim))
        self.biases = nn.Parameter(torch.zeros(self.num_nodes))
        self.leaf_values = nn.Parameter(0.05 * torch.randn(self.num_leaves,
                                                         generator=generator))
        self.register_buffer("temperature", torch.full((self.num_nodes,), float(temperature)))

    @torch.no_grad()
    def set_temperature(self, temperature: float | Tensor) -> None:
        """Update gate temperatures in place, preserving buffer identity."""
        value = torch.as_tensor(temperature, device=self.temperature.device,
                                dtype=self.temperature.dtype)
        if value.numel() not in (1, self.num_nodes) or not torch.isfinite(value).all():
            raise ValueError("temperature must be a finite scalar or node vector")
        if (value <= 0).any():
            raise ValueError("temperature must be positive")
        self.temperature.copy_(value.reshape(-1).expand_as(self.temperature))

    def routing(self, x: Tensor, *, hard: bool = False) -> RoutingTrace:
        """Return mass-conserving routes; hard ties deterministically go left."""
        logits = (x @ self.weights.T + self.biases) / self.temperature
        left = (logits >= 0).to(x.dtype) if hard else logits.sigmoid()
        path = x.new_ones((len(x), 1))
        reaches = []
        for depth in range(self.depth):
            reaches.append(path)
            start, stop = 2**depth - 1, 2**(depth + 1) - 1
            p = left[:, start:stop]
            path = torch.stack((path * p, path * (1 - p)), dim=-1).flatten(1)
        return RoutingTrace(torch.cat(reaches, dim=1), left, path)

    def forward(self, x: Tensor, *, hard: bool = False) -> Tensor:
        return self.routing(x, hard=hard).leaves @ self.leaf_values

    @torch.no_grad()
    def initialize_cart(self, x: Tensor, target: Tensor, weight: Tensor, *,
                        min_samples_leaf: int, seed: int, gate_scale: float = 4.0) -> None:
        """Warm-start with CART; subsequent optimization is still differentiable.

        The initialization is a declared hybrid, not a differentiable topology
        search. Short CART leaves are broadcast through complete descendant
        subtrees, so the initialized hard function equals CART's function.
        """
        from sklearn.tree import DecisionTreeRegressor

        cart = DecisionTreeRegressor(max_depth=self.depth,
                                     min_samples_leaf=min_samples_leaf, random_state=seed)
        cart.fit(x.detach().cpu().numpy(), target.detach().cpu().numpy(),
                 sample_weight=weight.detach().cpu().numpy())
        self.weights.zero_()
        self.biases.zero_()

        def fill(source: int, node: int, depth: int) -> None:
            if cart.tree_.children_left[source] == -1 or depth == self.depth:
                first = (node - (2**depth - 1)) * 2**(self.depth - depth)
                count = 2**(self.depth - depth)
                self.leaf_values[first:first+count] = float(cart.tree_.value[source, 0, 0])
                return
            feature = int(cart.tree_.feature[source])
            self.weights[node, feature] = -gate_scale
            self.biases[node] = gate_scale * float(cart.tree_.threshold[source])
            fill(int(cart.tree_.children_left[source]), 2*node+1, depth+1)
            fill(int(cart.tree_.children_right[source]), 2*node+2, depth+1)

        fill(0, 0, 0)

    @torch.no_grad()
    def solve_leaves(self, x: Tensor, gradient: Tensor, hessian: Tensor,
                     weights: Tensor, l2: float, batch_size: int = 1024) -> None:
        """Solve the *coupled* soft-leaf Newton ridge problem in float64.

        A = R.T diag(w*h) R / sum(w) + l2*I; b = -R.T(w*g)/sum(w).
        Soft leaves overlap, so independent hard-leaf formulas are incorrect.
        Only a batch of routes is materialized. Dense A costs O(leaves**2).
        """
        if l2 <= 0:
            raise ValueError("l2 must be positive for a unique soft-leaf solve")
        a = torch.eye(self.num_leaves, dtype=torch.float64, device=self.weights.device) * l2
        b = torch.zeros(self.num_leaves, dtype=torch.float64, device=self.weights.device)
        total_weight = weights.double().sum().to(self.weights.device)
        for start in range(0, len(x), batch_size):
            sl = slice(start, start + batch_size)
            routes = self.routing(x[sl].to(self.weights.device)).leaves.double()
            wh = (weights[sl].double() * hessian[sl].double()).to(a.device) / total_weight
            wg = (weights[sl].double() * gradient[sl].double()).to(a.device) / total_weight
            a.add_(routes.T @ (wh[:, None] * routes))
            b.sub_(routes.T @ wg)
        solution = torch.linalg.solve(a, b)
        if not torch.isfinite(solution).all():
            raise FloatingPointError("nonfinite soft-leaf Newton solution")
        self.leaf_values.copy_(solution.to(self.leaf_values.dtype))

    def to_dict(self) -> dict:
        """Export hard routing (not a claim that soft and hard predictions agree)."""
        return {"depth": self.depth, "weights": self.weights.detach().cpu().tolist(),
                "biases": self.biases.detach().cpu().tolist(),
                "leaf_values": self.leaf_values.detach().cpu().tolist(),
                "left_when": "affine_score >= 0"}
