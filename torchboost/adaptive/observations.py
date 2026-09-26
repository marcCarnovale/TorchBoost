"""Observation and delayed evidence, independent of plasticity and policy.

Local utility is an exact control-set ablation of a node's residual contribution
under the current input-dependent attention. It is not a causal intervention
estimate over training trajectories. Retained delayed gains are observational.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict
import math

import torch
from torch import Tensor

from .contracts import Observation, Proposal, TrialOutcome
from .forest import AdaptiveForest, ForestTrace
from .objectives import Objective
from .leaf_evidence import leaf_evidence


def parameter_vector(node) -> Tensor:
    pieces = [p.detach().reshape(-1) for p in node.parameters_for_plasticity().values()]
    return torch.cat(pieces) if pieces else torch.empty(0)


class SplitMetricsCollector:
    def __init__(self):
        self.state: dict[str, dict] = {}

    def synchronize(self, forest: AdaptiveForest) -> None:
        keys = forest.node_map()
        self.state = {k: v for k, v in self.state.items() if k in keys}
        for node in keys.values():
            self.state.setdefault(node.node_id, {"path": 0., "motion": 0., "count": 0,
                                                 "gradient": 0., "structural_gradient": 0.,
                                                 "direction": 1., "last_update": None, "last_utility": None})

    @torch.no_grad()
    def before_step(self, forest: AdaptiveForest) -> dict[str, Tensor]:
        self.synchronize(forest)
        return {n.node_id: parameter_vector(n).clone() for n in forest.iter_nodes()}

    @torch.no_grad()
    def after_step(self, forest: AdaptiveForest, before: dict[str, Tensor]) -> None:
        for node in forest.iter_nodes():
            key = node.node_id
            delta = parameter_vector(node) - before[key]
            norm = float(torch.linalg.vector_norm(delta))
            entry = self.state[key]
            previous = entry["last_update"]
            if previous is not None and previous.shape == delta.shape and norm > 1e-12:
                denominator = float(previous.norm()) * norm
                entry["direction"] = float(torch.dot(previous.to(delta), delta) / denominator) if denominator > 1e-12 else 1.
            entry["last_update"] = delta.cpu().clone()
            entry["path"] += norm
            entry["motion"] += norm
            entry["count"] += int(norm > 1e-12)
            grads = [p.grad.detach().square().sum() for p in node.parameters_for_plasticity().values() if p.grad is not None]
            entry["gradient"] = math.sqrt(float(sum(grads))) if grads else 0.
            entry["structural_gradient"] = (float(node.structural.grad.abs())
                                              if node.structural is not None and node.structural.grad is not None else 0.)

    @torch.no_grad()
    def collect(self, forest: AdaptiveForest, x: Tensor, logits: Tensor, trace: ForestTrace,
                target: Tensor, weights: Tensor, objective: Objective, step: int, *,
                physical_context: dict | None = None, plastic_context: dict | None = None,
                phase_context: dict | None = None, progress: float = 0.) -> list[Observation]:
        self.synchronize(forest)
        physical_context, plastic_context = physical_context or {}, plastic_context or {}
        phase_context = phase_context or {}
        base_loss = float(objective.weighted_loss(logits, target, weights))
        result = []
        for node in forest.iter_nodes():
            key, history = node.node_id, self.state[node.node_id]
            raw = trace.nodes.get(key)
            occupancy, entropy, information, utility, refinement_utility, uncertainty = (0.,) * 6
            if raw is not None:
                rw = weights * raw.reach
                occupancy = float(rw.sum() / weights.sum())
                if objective.task in ("binary", "multiclass"):
                    response = objective.response(logits)
                    local_uncertainty = -(response * response.clamp_min(1e-12).log()).sum(1)
                else:
                    residual = logits - target
                    mean = (rw[:, None] * residual).sum(0) / rw.sum().clamp_min(1e-12)
                    local_uncertainty = (residual - mean).square().mean(1)
                uncertainty = float((local_uncertainty * rw).sum() / rw.sum().clamp_min(1e-12))
                coefficient = trace.coefficients[:, trace.tree_slots[node.tree_id]]
                removed = coefficient * raw.reach[:, None] * raw.output
                utility = float(objective.weighted_loss(logits - removed, target, weights)) - base_loss
                refinement_removed = coefficient * raw.reach[:, None] * raw.refinement
                refinement_utility = float(objective.weighted_loss(logits - refinement_removed, target, weights)) - base_loss
                if raw.probabilities is not None:
                    p = raw.probabilities
                    local_entropy = -(p * p.clamp_min(1e-12).log()).sum(1)
                    entropy = float((local_entropy * rw).sum() / rw.sum().clamp_min(1e-12))
                    information = objective.information(p, rw, target)
            evidence = (leaf_evidence(x, logits, target, weights, raw.reach, objective.task)
                        if raw is not None and node.is_leaf else None)
            result.append(Observation(key, node.tree_id, node.depth, step, forest.topology_version,
                                      occupancy, entropy, information, utility, refinement_utility,
                                      history["gradient"], history["structural_gradient"], history["motion"],
                                      history["path"], history["direction"], float(parameter_vector(node).norm()),
                                      float(node.temperature), node.frozen, not node.is_leaf,
                                      float(node.gate()), history["count"],
                                      float(physical_context.get("charge", 0.)),
                                      float(physical_context.get("nodes", {}).get(key, {}).get("heat", 0.)),
                                      float(physical_context.get("nodes", {}).get(key, {}).get("inductive_energy", 0.)),
                                      float(plastic_context.get(key, {}).get("hardness", 0.)),
                                      float(plastic_context.get(key, {}).get("integrity", 1.)),
                                      float(plastic_context.get(key, {}).get("reference_path", 0.)),
                                      ("grow", "evaluate", "consolidate", "reopen").index(phase_context.get(node.tree_id, "grow")),
                                      progress, uncertainty,
                                      0. if history.get("last_utility") is None else utility - history["last_utility"],
                                      0. if evidence is None else evidence.effective_n,
                                      0. if evidence is None else evidence.residual_variance,
                                      0. if evidence is None else evidence.reducible_loss,
                                      0. if evidence is None else evidence.explainable_fraction,
                                      1. if evidence is None else evidence.noise_fraction,
                                      0. if evidence is None else evidence.confident_error_mass,
                                      0. if evidence is None else evidence.exploration_score,
                                      0. if evidence is None else evidence.budget_score))
            history["motion"] = 0.
            history["last_utility"] = utility
        return result

    def state_dict(self) -> dict:
        return self.state

    def load_state_dict(self, state: dict) -> None:
        self.state = state


class PerformanceTracker:
    def __init__(self, history_size: int = 64):
        self.history_size = history_size
        self.histories: dict[str, deque[Observation]] = {}
        self.trials: dict[int, dict] = {}
        self.cancelled: list[dict] = []
        self.events: list[dict] = []

    def update(self, observations: list[Observation]) -> None:
        for observation in observations:
            history = self.histories.setdefault(observation.node_id, deque(maxlen=self.history_size))
            if history and observation.step <= history[-1].step:
                if observation == history[-1]:
                    continue
                raise ValueError("observations must have strictly increasing step IDs")
            history.append(observation)

    def latest(self) -> dict[str, Observation]:
        return {key: values[-1] for key, values in self.histories.items() if values}

    def synchronize(self, alive: set[str]) -> None:
        self.histories = {k: v for k, v in self.histories.items() if k in alive}
        for trial_id, trial in list(self.trials.items()):
            if trial["proposal"].node_id not in alive:
                self.cancelled.append({"trial_id": trial_id, "reason": "node_deleted"})
                del self.trials[trial_id]

    def begin_trial(self, proposal: Proposal, window: int, *, deformation_source: str = "parameters") -> bool:
        latest = self.latest().get(proposal.node_id)
        if latest is None or proposal.trial_id in self.trials:
            return False
        if any(t["proposal"].node_id == proposal.node_id for t in self.trials.values()):
            return False
        baseline_window = [o for o in self.histories[proposal.node_id]
                           if o.topology_version == latest.topology_version][-3:]
        baseline = sum(o.utility for o in baseline_window) / len(baseline_window)
        self.trials[proposal.trial_id] = {"proposal": proposal, "baseline": baseline,
                                         "start_path": latest.path_length, "window": window,
                                         "start_reference_path": latest.reference_path_length,
                                         "deformation_source": deformation_source}
        return True

    def mature(self, step: int, *, minimum_movement: float, minimum_gain: float) -> list[TrialOutcome]:
        outcomes = []
        for trial_id, trial in list(self.trials.items()):
            p = trial["proposal"]
            observations = [o for o in self.histories.get(p.node_id, ()) if o.step > p.step]
            if len(observations) < trial["window"]:
                continue
            observations = observations[:trial["window"]]
            tail = observations[-min(2, len(observations)):]
            retained = min(o.utility for o in tail) - trial["baseline"]
            movement = (observations[-1].reference_path_length - trial["start_reference_path"]
                        if trial.get("deformation_source") == "reference"
                        else observations[-1].path_length - trial["start_path"])
            # Both actual motion and retained improvement are necessary for a
            # positive deformation outcome; entropy/gradient size is no reward.
            accepted = movement >= minimum_movement and retained > minimum_gain
            reward = retained if movement >= minimum_movement else min(0., retained)
            reason = "retained_gain" if accepted else ("no_deformation" if movement < minimum_movement else "no_retained_gain")
            outcomes.append(TrialOutcome(trial_id, p.node_id, p.action, p.context, reward,
                                         retained, movement, accepted, reason))
            del self.trials[trial_id]
        return outcomes

    def state_dict(self) -> dict:
        return {"history_size": self.history_size,
                "histories": {k: [asdict(o) for o in h] for k, h in self.histories.items()},
                "trials": {k: {**v, "proposal": asdict(v["proposal"])} for k, v in self.trials.items()},
                "cancelled": self.cancelled, "events": self.events}

    def load_state_dict(self, state: dict) -> None:
        self.history_size = state["history_size"]
        self.histories = {k: deque((Observation(**o) for o in values), maxlen=self.history_size)
                          for k, values in state["histories"].items()}
        self.trials = {int(k): {**v, "proposal": Proposal(**v["proposal"])} for k, v in state["trials"].items()}
        self.cancelled, self.events = state["cancelled"], state["events"]
