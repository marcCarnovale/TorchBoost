"""Observation only: training -> collector -> tracker; no plasticity dependency."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from .trees import RoutingTrace


@dataclass(frozen=True)
class SplitObservation:
    """One node's epoch metrics. Node identity is never averaged away.

    information_gain is empirical I(branch; label | reached node), in nats.
    routing_entropy is conditional gate uncertainty, *not* information gain.
    Gradient magnitude and path length describe dynamics, not evidence that a
    deformation caused improvement. Causal/delayed intervention scoring is future
    work; this class deliberately carries no invented success label.
    """
    node_id: str
    epoch: int
    reach_mass: float
    effective_samples: float
    routing_entropy: float
    information_gain: float
    gradient_norm: float
    update_path_length: float


class SplitMetricsCollector:
    """Collect sufficient statistics from explicit training-loop calls.

    Does not inspect plasticity, own a scheduler, change model parameters, or
    retain autograd graphs. Memory is O(number of observed internal nodes), not
    O(number of samples times training steps). Instantiate per candidate tree;
    call reset_epoch between reporting windows.
    """

    def __init__(self, node_count: int, stage_id: int, device: str | torch.device = "cpu"):
        self.node_count, self.stage_id, self.device = node_count, stage_id, device
        self.reset_epoch()

    def reset_epoch(self) -> None:
        n = self.node_count
        self.mass = torch.zeros(n, device=self.device, dtype=torch.float64)
        self.squared_mass = torch.zeros_like(self.mass)
        self.entropy_sum = torch.zeros_like(self.mass)
        self.joint = torch.zeros(n, 2, 2, device=self.device, dtype=torch.float64)
        self.gradient_sum = torch.zeros_like(self.mass)
        self.path = torch.zeros_like(self.mass)
        self.update_count = 0

    @torch.no_grad()
    def observe_batch(self, trace: RoutingTrace, labels: Tensor, weights: Tensor) -> None:
        reach = trace.reach.detach().double() * weights.detach().double()[:, None]
        left = trace.left.detach().double()
        probabilities = torch.stack((left, 1-left), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
        self.mass.add_(reach.sum(0))
        self.squared_mass.add_(reach.square().sum(0))
        self.entropy_sum.add_((reach * entropy).sum(0))
        onehot = torch.stack((1-labels, labels), dim=-1).detach().double()
        self.joint.add_(torch.einsum("bn,bnk,bc->nkc", reach, probabilities, onehot))

    @torch.no_grad()
    def observe_update(self, before_w: Tensor, before_b: Tensor, after_w: Tensor,
                       after_b: Tensor, grad_w: Tensor, grad_b: Tensor) -> None:
        """Count every observed update's magnitude, including cancelling motion."""
        delta = (after_w.detach()-before_w).double().square().sum(-1)
        delta += (after_b.detach()-before_b).double().square()
        self.path.add_(delta.sqrt())
        norm = grad_w.detach().double().square().sum(-1) + grad_b.detach().double().square()
        self.gradient_sum.add_(norm.sqrt())
        self.update_count += 1

    def finish_epoch(self, epoch: int) -> list[SplitObservation]:
        """Emit CPU values; zero-traffic nodes have zero empirical information."""
        mass = self.mass.clamp_min(1e-30)
        joint = self.joint / mass[:, None, None]
        product = joint.sum(-1, keepdim=True) * joint.sum(-2, keepdim=True)
        mi = (joint * (joint.clamp_min(1e-30).log()
                       - product.clamp_min(1e-30).log())).sum((-1, -2)).clamp_min(0)
        values = torch.stack((self.mass,
                              self.mass.square()/self.squared_mass.clamp_min(1e-30),
                              self.entropy_sum/mass, mi,
                              self.gradient_sum/max(1, self.update_count), self.path), dim=1)
        return [SplitObservation(f"stage:{self.stage_id}/node:{i}", epoch, *row)
                for i, row in enumerate(values.cpu().tolist())]


class PerformanceTracker:
    """Bounded split-specific history, reusable by any future policy.

    Owned by a training run, not a process-global singleton. `state_dict` emits
    primitive values and can be serialized without accessing live model tensors.
    A future OnlineScheduler consumes snapshots and sets controller parameters;
    neither the tracker nor the collector owns that scheduler.
    """

    def __init__(self, history_length: int = 8):
        if history_length < 1:
            raise ValueError("history_length must be positive")
        self.history_length = history_length
        self.history: dict[str, deque] = {}

    def record(self, observations: list[SplitObservation]) -> None:
        for observation in observations:
            history = self.history.setdefault(observation.node_id, deque(maxlen=self.history_length))
            if history and observation.epoch <= history[-1]["epoch"]:
                raise ValueError("epochs must increase for each node")
            history.append(asdict(observation))

    def snapshot(self) -> dict[str, dict]:
        return {node: dict(history[-1]) for node, history in self.history.items() if history}

    def state_dict(self) -> dict:
        return {"history_length": self.history_length,
                "history": {key: list(value) for key, value in self.history.items()}}

    def load_state_dict(self, state: dict) -> None:
        self.history_length = int(state["history_length"])
        self.history = {key: deque(value, maxlen=self.history_length)
                        for key, value in state["history"].items()}
