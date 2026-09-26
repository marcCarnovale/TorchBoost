"""The sole owner of training-time state transitions and structural transactions."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, replace
import math
import time
import uuid
from typing import Callable

import torch
from torch import Tensor

from .config import ForestConfig
from .contracts import Proposal, StructuralAction
from .data import DataSplit
from .forest import AdaptiveForest
from .objectives import Objective
from .observations import PerformanceTracker, SplitMetricsCollector
from .online import ACTIONS, OnlineScheduler
from .optim import DynamicOptimizer
from .physics import PhysicalController
from .plasticity import PlasticityModule
from .structure import GrowthPruningPolicy, ScheduleManager, StructuralRelaxation, structural_penalty


def model_snapshot(model: AdaptiveForest) -> dict:
    return {"schema": deepcopy(model.schema()),
            "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}


def restore_model(snapshot: dict, input_dim: int, output_dim: int, config: ForestConfig) -> AdaptiveForest:
    family = snapshot["schema"].get("family")
    if family == "unified-progressive-v1":
        from .unified_progressive import ProgressiveForest
        model = ProgressiveForest(input_dim, output_dim, config, schema=snapshot["schema"])
    elif family == "rated-native-v1":
        from .rated_forest import RatedAdaptiveForest
        model = RatedAdaptiveForest(input_dim, output_dim, config, schema=snapshot["schema"])
    else:
        model = AdaptiveForest(input_dim, output_dim, config, schema=snapshot["schema"])
    model.load_state_dict(snapshot["state"])
    return model


def packet_map(model: AdaptiveForest) -> dict:
    return {node.node_id: node.parameters_for_plasticity() for node in model.iter_nodes()}


class JointTrainer:
    def __init__(self, model: AdaptiveForest, objective: Objective, config: ForestConfig,
                 generator: torch.Generator, *, selection_metric: Callable | None = None,
                 greater_is_better: bool = False):
        self.model, self.objective, self.config, self.generator = model, objective, config, generator
        self.selection_metric, self.greater_is_better = selection_metric, greater_is_better
        self.optimizer = DynamicOptimizer(model, config)
        self.collector = SplitMetricsCollector()
        self.tracker = PerformanceTracker()
        self.physical = PhysicalController(config.physics, seed=config.random_state + 101)
        self.plastic = PlasticityModule(config.plasticity, seed=config.random_state + 211)
        self.scheduler = OnlineScheduler(config.online, str(uuid.uuid4()), seed=config.random_state + 307)
        self.structure = GrowthPruningPolicy(config.structure, seed=config.random_state + 401)
        self.schedule = ScheduleManager(config)
        self.relaxation = StructuralRelaxation(config.structure)
        self.epoch = 0
        self.optimizer_steps = 0
        self.examples_seen = 0
        self.counter_origin_epoch = 0
        self.last_gradient_norm = 0.
        self.history: list[dict] = []
        self.events: list[dict] = []
        self.best_score = math.inf
        self.best_epoch = -1
        self.best_snapshot = model_snapshot(model)
        self.fingerprints: dict[str, str] = {}
        self.control_indices: Tensor | None = None
        self.max_model_bytes = model.tensor_bytes()
        self.max_optimizer_bytes = 0
        self._synchronize()

    def _synchronize(self, changed_nodes: set[str] | None = None) -> None:
        changed_nodes = changed_nodes or set()
        nodes = self.model.node_map()
        alive = set(nodes)
        for trial_id, trial in list(self.tracker.trials.items()):
            if (trial["proposal"].node_id in changed_nodes or trial["proposal"].node_id not in alive
                    or trial["proposal"].topology_version != self.model.topology_version):
                self.plastic.finish_trial(trial_id, retain=False)
                self.tracker.cancelled.append({"trial_id": trial_id, "reason": "topology_changed"})
                del self.tracker.trials[trial_id]
        self.optimizer.synchronize(self.model)
        if self.config.collect_metrics:
            self.collector.synchronize(self.model)
        for key in changed_nodes & set(self.collector.state):
            self.collector.state[key]["last_update"] = None
            self.collector.state[key]["direction"] = 1.
            self.collector.state[key]["last_utility"] = None
        self.tracker.synchronize(alive)
        if self.config.physics.mode != "none":
            self.physical.synchronize({key: node.tree_id for key, node in nodes.items()})
        self.plastic.synchronize(packet_map(self.model))
        self.scheduler.synchronize(alive)
        self.relaxation.synchronize(alive, changed_nodes)
        self.max_model_bytes = max(self.max_model_bytes, self.model.tensor_bytes())

    @torch.no_grad()
    def logits(self, data: DataSplit, *, hard: bool = False) -> Tensor:
        self.model.eval()
        return torch.cat([self.model(data.x[start:start + self.config.batch_size].to(self.config.device), hard=hard).cpu()
                          for start in range(0, len(data.x), self.config.batch_size)])

    @torch.no_grad()
    def loss(self, data: DataSplit) -> float:
        return float(self.objective.weighted_loss(self.logits(data), data.y, data.weight))

    def _regularization(self, x: Tensor, prediction: Tensor, trace, weights: Tensor, values: dict) -> Tensor:
        cfg = self.config
        result = structural_penalty(self.model,
                                    complexity=values.get("complexity", cfg.structure.complexity),
                                    bimodality=values.get("bimodality", cfg.structure.gate_bimodality),
                                    local_multipliers=self.relaxation.multipliers)
        result = result + self.plastic.penalty(packet_map(self.model))
        if cfg.feature_penalties:
            if len(cfg.feature_penalties) != self.model.input_dim:
                raise ValueError("feature_penalties must have one weight per feature")
            penalties = x.new_tensor(cfg.feature_penalties)
            terms = [(node.routing_weight.abs() * penalties).mean()
                     for node in self.model.iter_nodes() if node.routing_weight is not None]
            if terms:
                result = result + torch.stack(terms).mean()
        diversity = values.get("diversity", cfg.diversity)
        if diversity and len(self.model.trees) > 1:
            outputs = trace.tree_outputs
            centered = outputs - outputs.mean(0, keepdim=True)
            vectors = centered.permute(1, 0, 2).flatten(1)
            vectors = vectors / vectors.norm(dim=1, keepdim=True).clamp_min(1e-6)
            correlations = vectors @ vectors.T
            mask = ~torch.eye(len(vectors), dtype=torch.bool, device=x.device)
            result = result + diversity * correlations[mask].square().mean()
        if cfg.monotonicity and cfg.monotonicity_penalty:
            response = self.objective.response(prediction)
            for output, feature, sign in cfg.monotonicity:
                if output >= response.shape[1] or feature >= x.shape[1]:
                    raise ValueError("monotonicity constraint index is out of bounds")
                gradient = torch.autograd.grad(response[:, output].sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
                gradient = x.new_zeros(len(x)) if gradient is None else gradient[:, feature]
                result = result + cfg.monotonicity_penalty * (torch.relu(-sign * gradient).square() * weights).sum() / weights.sum().clamp_min(1e-12)
        return result

    def _needs_trace(self, values: dict) -> bool:
        return bool(values.get("diversity", self.config.diversity))

    def _train_epoch(self, train: DataSplit, values: dict) -> None:
        self.model.train()
        cfg = self.config
        order = torch.randperm(len(train.x), generator=self.generator)
        chunk_size = cfg.batch_size * cfg.accumulation_steps
        for start in range(0, len(order), chunk_size):
            effective = order[start:start + chunk_size]
            total_weight = float(train.weight[effective].sum())
            if total_weight <= 0:
                continue
            self.optimizer.zero_grad()
            for offset in range(0, len(effective), cfg.batch_size):
                indices = effective[offset:offset + cfg.batch_size]
                weights = train.weight[indices].to(cfg.device)
                if weights.sum() <= 0:
                    continue
                x, target = train.x[indices].to(cfg.device), train.y[indices].to(cfg.device)
                if cfg.monotonicity and cfg.monotonicity_penalty:
                    x = x.detach().requires_grad_(True)
                needs_trace = self._needs_trace(values)
                forward = self.model(x, trace=needs_trace, generator=self.generator)
                prediction, trace = forward if needs_trace else (forward, None)
                primary = (self.objective.loss(prediction, target) * weights).sum() / total_weight
                regularizer = self._regularization(x, prediction, trace, weights, values)
                loss = primary + regularizer * (weights.sum() / total_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip, error_if_nonfinite=True)
            self.last_gradient_norm = float(gradient_norm)
            before = self.collector.before_step(self.model) if cfg.collect_metrics else None
            self.optimizer.step()
            self.optimizer_steps += 1
            self.examples_seen += len(effective)
            self.model.project()
            if cfg.collect_metrics:
                self.collector.after_step(self.model, before)
            self.max_optimizer_bytes = max(self.max_optimizer_bytes, self.optimizer.tensor_bytes())

    def _apply_online(self, step: int) -> None:
        for proposal in self.scheduler.collect(wait=True):
            node = self.model.node_map().get(proposal.node_id)
            if node is None or node.frozen or not self.scheduler.valid(
                proposal, topology_version=self.model.topology_version, versions=self.plastic.versions, step=step):
                continue
            if self.tracker.begin_trial(proposal, self.config.online.window,
                                         deformation_source=self.config.online.deformation_source):
                self.plastic.apply_trial(proposal.trial_id, proposal.node_id, ACTIONS[proposal.action])
                self.scheduler.accepted(proposal, step)

    @torch.no_grad()
    def _apply_structure(self, actions: list[StructuralAction], control: DataSplit, step: int) -> None:
        initial_version = self.model.topology_version
        self.model.eval()
        for action in actions:
            if action.topology_version != initial_version:
                self.events.append({"step": step, "event": "stale_structure", **asdict(action)})
                continue
            node = self.model.node_map().get(action.node_id)
            if node is None or node.locked:
                continue
            tree = self.model.get_tree(node.tree_id)
            before_bytes = self.model.tensor_bytes() + self.optimizer.tensor_bytes()
            changed = set()
            if action.kind == "grow":
                arity = action.arity or self.config.structure.arity
                # Exact anticipated parameter cost: W,b,s, then arity*(v,beta).
                added = arity * self.model.input_dim + arity + int(self.config.structure.structural_gate)
                added += arity * (self.model.output_dim + 1)
                if sum(p.numel() for p in self.model.parameters()) + added > self.config.structure.max_parameters:
                    self.events.append({"step": step, "event": "growth_budget_rejected", "node_id": node.node_id})
                    continue
                new_ids = tree.grow(node.node_id, generator=self.generator, arity=arity)
                if not new_ids:
                    continue
                changed = {node.node_id}
                detail = {"added": new_ids}
            elif action.kind == "remove_tree":
                if len(self.model.trees) <= 1:
                    continue
                x, y, w = control.x.to(self.config.device), control.y.to(self.config.device), control.weight.to(self.config.device)
                current = self.model(x)
                candidate = self.model(x, disabled_tree_ids=frozenset({node.tree_id}))
                delta = float(self.objective.weighted_loss(candidate, y, w) - self.objective.weighted_loss(current, y, w))
                if delta > self.config.structure.prune_tolerance:
                    self.events.append({"step": step, "event": "tree_prune_rejected", "node_id": node.node_id,
                                        "control_loss_delta": delta})
                    continue
                removed, migrations = self.model.remove_tree(node.tree_id)
                if not removed:
                    continue
                self.optimizer.migrate_rows(migrations)
                changed = set(removed)
                detail = {"removed": removed, "control_loss_delta": delta}
            elif action.kind in ("freeze", "thaw"):
                tree.set_frozen(node.node_id, action.kind == "freeze", recursive=action.recursive)
                detail = {"recursive": action.recursive}
            elif action.kind in ("deactivate", "activate"):
                if action.kind == "deactivate" and node.node_id == tree.root_id:
                    remaining = [t for t in self.model.trees if t.tree_id != tree.tree_id and t.get(t.root_id).active]
                    if not remaining:
                        raise ValueError("cannot deactivate the final active tree")
                tree.deactivate(node.node_id, active=action.kind == "activate")
                changed = {node.node_id}
                detail = {}
            elif action.kind == "reinitialize":
                # Reinitialization is an explicit reset; ordinary thaw never
                # resets a parameter, anchor, optimizer moment, or history.
                tree.reinitialize(node.node_id, self.generator)
                for parameter in node.parameters():
                    self.optimizer.optimizer.state.pop(parameter, None)
                self.collector.state.pop(node.node_id, None)
                self.tracker.histories.pop(node.node_id, None)
                self.plastic.states.pop(node.node_id, None)
                self.plastic.settings.pop(node.node_id, None)
                self.plastic.versions.pop(node.node_id, None)
                if node.node_id in self.physical.nodes:
                    self.physical.synchronize({key: n.tree_id for key, n in self.model.node_map().items() if key != node.node_id})
                changed = {node.node_id}
                detail = {"reset_local_state": True}
            elif action.kind == "prune":
                x, y, w = control.x.to(self.config.device), control.y.to(self.config.device), control.weight.to(self.config.device)
                current = self.model(x)
                candidate = self.model(x, disabled_refinements=frozenset({node.node_id}))
                delta = float(self.objective.weighted_loss(candidate, y, w) - self.objective.weighted_loss(current, y, w))
                if delta > self.config.structure.prune_tolerance:
                    self.events.append({"step": step, "event": "prune_rejected", "node_id": node.node_id, "control_loss_delta": delta})
                    continue
                removed = tree.prune(node.node_id)
                if not removed:
                    continue
                changed = {node.node_id, *removed}
                detail = {"removed": removed, "control_loss_delta": delta}
            else:
                raise ValueError(f"unknown structural action {action.kind}")
            self._synchronize(changed)
            after_bytes = self.model.tensor_bytes() + self.optimizer.tensor_bytes()
            event = {"step": step, "event": action.kind, "node_id": node.node_id,
                     "topology_version": self.model.topology_version, "live_bytes_before": before_bytes,
                     "live_bytes_after": after_bytes, "byte_scope": "model_and_optimizer_tensors", **detail}
            self.events.append(event)
            self.structure.events.append(event)

    @torch.no_grad()
    def _observe_and_control(self, control: DataSplit, step: int) -> None:
        cfg = self.config
        self.model.eval()
        x, y, w = control.x.to(cfg.device), control.y.to(cfg.device), control.weight.to(cfg.device)
        forward = self.model(x, trace=cfg.collect_metrics)
        prediction, trace = forward if cfg.collect_metrics else (forward, None)
        control_loss = float(self.objective.weighted_loss(prediction, y, w))
        observations = (self.collector.collect(self.model, x, prediction, trace, y, w, self.objective, step,
                        physical_context={"charge": self.physical.charge, "nodes": self.physical.nodes},
                        plastic_context=self.plastic.snapshot(), phase_context=self.schedule.phases(step),
                        progress=(step + 1) / cfg.epochs) if cfg.collect_metrics else [])
        self.tracker.update(observations)
        latest = self.tracker.latest()
        self.relaxation.update(self.model, latest, step, self.schedule.phases(step))
        outcomes = self.tracker.mature(step, minimum_movement=cfg.online.minimum_movement,
                                      minimum_gain=cfg.online.minimum_gain)
        self.scheduler.learn(outcomes, step)
        for outcome in outcomes:
            self.plastic.finish_trial(outcome.trial_id, retain=outcome.accepted)
        if cfg.physics.mode != "none":
            state = self.physical.advance(control_loss, latest, step)
            protected = math.ceil(cfg.n_trees * cfg.structure.protect_tree_fraction)
            for key, physical in state["nodes"].items():
                node = self.model.node_map()[key]
                if node.frozen and not node.locked and physical["temperature"] >= cfg.physics.thaw_temperature:
                    node.set_frozen(False)
                    self.events.append({"step": step, "event": "thermal_thaw", "node_id": key})
                if not node.frozen:
                    node.set_temperature(physical["temperature"])
        temperatures = {n.node_id: float(n.temperature) for n in self.model.iter_nodes()}
        plastic_result = self.plastic.advance(packet_map(self.model), latest, temperatures, step,
                                              progress=(step + 1) / cfg.epochs,
                                              ambient_temperature=cfg.physics.ambient_temperature)
        protected = math.ceil(cfg.n_trees * cfg.structure.protect_tree_fraction)
        for key in plastic_result["locks"]:
            node = self.model.node_map()[key]
            if node.tree_id < protected:
                self.plastic.states[key]["locked"] = False
            else:
                node.set_frozen(True, lock=True)
                self.events.append({"step": step, "event": "terminal_lock", "node_id": key})
        actions = self.structure.propose(self.model, latest, step,
                                         self.schedule.phases(step), (step + 1) / cfg.epochs)
        waiting = bool(self.config.online.defer_structure_for_trials and actions and self.tracker.trials)
        if waiting:
            self.events.append({"step": step, "event": "structure_waits_for_trials",
                                "pending_trials": len(self.tracker.trials)})
        else:
            self._apply_structure(actions, control, step)
        occupied = {trial["proposal"].node_id for trial in self.tracker.trials.values()}
        # Stop admitting fresh trials on a structural event; existing ones can
        # mature. Structural proposals are recomputed rather than replayed stale.
        if not (self.config.online.defer_structure_for_trials and actions):
            self.scheduler.request(self.tracker.latest(), self.plastic.versions, step,
                                   self.model.topology_version, occupied)

    @torch.no_grad()
    def _selection_score(self, selection: DataSplit) -> float:
        prediction = self.logits(selection)
        score = (float(self.objective.weighted_loss(prediction, selection.y, selection.weight))
                 if self.selection_metric is None else float(self.selection_metric(selection.y, prediction)))
        if not math.isfinite(score):
            raise FloatingPointError("nonfinite selection metric")
        return -score if self.greater_is_better else score

    def fit(self, train: DataSplit, control: DataSplit, selection: DataSplit, *, stop_epoch: int | None = None,
            verify_fingerprints: bool = True) -> None:
        cfg = self.config
        fingerprints = {"train": train.fingerprint(), "control": control.fingerprint(), "selection": selection.fingerprint()}
        if self.fingerprints and verify_fingerprints and fingerprints != self.fingerprints:
            raise ValueError("resume data differs from the checkpoint; use an explicit new adaptation phase")
        self.fingerprints = fingerprints
        if self.control_indices is None:
            self.control_indices = torch.randperm(len(control.x), generator=self.generator)[:cfg.control_sample_size]
        indices = self.control_indices
        observed_control = DataSplit(control.x[indices], control.y[indices], control.weight[indices])
        if self.epoch == 0:
            self.best_score = self._selection_score(selection)
            self.best_snapshot = model_snapshot(self.model)
        end = cfg.epochs if stop_epoch is None else min(cfg.epochs, stop_epoch)
        if end < self.epoch:
            raise ValueError("stop_epoch precedes completed training")
        for epoch in range(self.epoch, end):
            started = time.perf_counter()
            values = self.schedule.apply(self.model, epoch)
            self._apply_online(epoch)
            self.optimizer.set_controls(values.get("learning_rate", cfg.learning_rate), self.physical.nodes)
            self.plastic.stiffness_multiplier = values.get("plastic_stiffness", 1.)
            self._train_epoch(train, values)
            if epoch % cfg.observation_every == 0:
                self._observe_and_control(observed_control, epoch)
            # Select AFTER every prediction-affecting change. The saved topology,
            # parameters, and temperatures are exactly the evaluated model.
            train_prediction = self.logits(train)
            train_loss = float(self.objective.weighted_loss(train_prediction, train.y, train.weight))
            control_loss = self.loss(control)
            score = self._selection_score(selection)
            if score < self.best_score:
                self.best_score, self.best_epoch = score, epoch
                self.best_snapshot = model_snapshot(self.model)
            self.epoch = epoch + 1
            record = {"epoch": epoch, "train_loss": train_loss, "control_loss": control_loss,
                      "selection_score": score, "best_epoch": self.best_epoch,
                      "nodes": len(self.model.node_map()), "model_bytes": self.model.tensor_bytes(),
                      "optimizer_bytes": self.optimizer.tensor_bytes(),
                      "optimizer_steps": self.optimizer_steps, "examples_seen": self.examples_seen,
                      "counter_origin_epoch": self.counter_origin_epoch}
            if cfg.record_diagnostics:
                from .long_horizon import training_diagnostics
                record.update(training_diagnostics(self.model, train_prediction, train.y,
                                                    train.weight, self.objective.task))
                record["last_preclip_gradient_norm"] = self.last_gradient_norm
            if not cfg.compact_history:
                record.update({
                    "temperatures": {n.node_id: float(n.temperature) for n in self.model.iter_nodes()},
                    "momentum": {g["owner"]: g.get("last_beta", None) for g in self.optimizer.optimizer.param_groups},
                    "learning_rates": {g["owner"]: g["lr"] for g in self.optimizer.optimizer.param_groups}})
            else:
                rates = [g["lr"] for g in self.optimizer.optimizer.param_groups]
                record.update({"learning_rate_min": min(rates), "learning_rate_max": max(rates)})
            record["seconds"] = time.perf_counter() - started
            self.history.append(record)
        self.model.eval()

    def state_dict(self) -> dict:
        return {"model": model_snapshot(self.model), "optimizer": self.optimizer.state_dict(),
                "collector": deepcopy(self.collector.state_dict()), "tracker": self.tracker.state_dict(),
                "physical": self.physical.state_dict(), "plastic": self.plastic.state_dict(),
                "scheduler": self.scheduler.state_dict(), "structure": self.structure.state_dict(),
                "schedule": self.schedule.state_dict(), "relaxation": self.relaxation.state_dict(), "generator": self.generator.get_state(),
                "epoch": self.epoch, "history": deepcopy(self.history), "events": deepcopy(self.events),
                "optimizer_steps": self.optimizer_steps, "examples_seen": self.examples_seen,
                "counter_origin_epoch": self.counter_origin_epoch,
                "last_gradient_norm": self.last_gradient_norm,
                "best_score": self.best_score, "best_epoch": self.best_epoch,
                "best_snapshot": deepcopy(self.best_snapshot), "fingerprints": self.fingerprints.copy(),
                "control_indices": self.control_indices, "max_model_bytes": self.max_model_bytes,
                "max_optimizer_bytes": self.max_optimizer_bytes}

    def load_state_dict(self, state: dict) -> None:
        self.model = restore_model(state["model"], self.model.input_dim, self.model.output_dim, self.config)
        self.optimizer = DynamicOptimizer(self.model, self.config)
        self.optimizer.load_state_dict(self.model, state["optimizer"])
        for name in ("collector", "tracker", "physical", "plastic", "scheduler", "structure", "schedule", "relaxation"):
            getattr(self, name).load_state_dict(deepcopy(state[name]))
        self.generator.set_state(state["generator"])
        self.epoch, self.history, self.events = state["epoch"], deepcopy(state["history"]), deepcopy(state["events"])
        self.optimizer_steps = state.get("optimizer_steps", 0)
        self.examples_seen = state.get("examples_seen", 0)
        self.counter_origin_epoch = state.get("counter_origin_epoch", state["epoch"])
        self.last_gradient_norm = state.get("last_gradient_norm", 0.)
        self.best_score, self.best_epoch, self.best_snapshot = state["best_score"], state["best_epoch"], deepcopy(state["best_snapshot"])
        self.fingerprints, self.control_indices = state["fingerprints"].copy(), state["control_indices"]
        self.max_model_bytes, self.max_optimizer_bytes = state["max_model_bytes"], state["max_optimizer_bytes"]
        self.optimizer.assert_ownership(self.model)

    def close(self) -> None:
        self.scheduler.close()
