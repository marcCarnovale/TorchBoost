"""Composable typed structural policies and normalized complexity budgets."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import replace
import math
import numpy as np
import torch
from torch import Tensor

from .config import ForestConfig, StructureConfig
from .contracts import Observation, StructuralAction
from .forest import AdaptiveForest, RaggedTree


def depth_node_allocation(tree: RaggedTree, mode: str) -> dict[str, Tensor]:
    """A per-tree unit budget: softmax(alpha_d*d), then softmax(-beta_n).

    Absent levels receive no budget. The exp(-d) alternative normalizes over
    present levels. Learned allocation coefficients have a separate penalty.
    """
    nodes = [n for n in tree.nodes.values() if not n.is_leaf]
    if not nodes:
        return {}
    depths = sorted({n.depth for n in nodes})
    depth_tensor = tree.depth_logits.new_tensor(depths)
    if mode == "learned":
        logits = tree.depth_logits[depths] * depth_tensor
    elif mode == "exponential":
        logits = -depth_tensor
    elif mode == "uniform":
        logits = torch.zeros_like(depth_tensor)
    else:
        raise ValueError(f"unknown allocation mode {mode}")
    budgets = logits.softmax(0)
    result = {}
    for depth, budget in zip(depths, budgets):
        group = [n for n in nodes if n.depth == depth]
        within = torch.stack([-n.allocation_logit for n in group]).softmax(0)
        for node, fraction in zip(group, within):
            result[node.node_id] = budget * fraction
    return result


def structural_penalty(forest: AdaptiveForest, *, complexity: float, bimodality: float,
                       local_multipliers: dict[str, float] | None = None) -> Tensor:
    result = forest.bias.new_zeros(())
    cfg = forest.config.structure
    if complexity == 0 and bimodality == 0 and cfg.allocation_regularization == 0:
        return result
    for tree in forest.trees:
        allocation = depth_node_allocation(tree, cfg.depth_allocation)
        for key, weight in allocation.items():
            node = tree.get(key)
            gate = node.gate()
            child_values = torch.stack([tree.get(c).value for c in node.children_ids])
            magnitude = child_values.square().mean()
            coupling = 1 + cfg.structural_temperature_coupling * max(0., float(node.temperature) - forest.config.physics.ambient_temperature)
            result = result + weight * (complexity * (local_multipliers or {}).get(key, 1.) / coupling * gate.square() * (1 + magnitude)
                                        + bimodality * gate * (1 - gate))
        # The allocation optimizer cannot evade complexity without paying its
        # own explicit coefficient penalty. Empty levels are still bounded.
        if cfg.depth_allocation == "learned":
            result = result + cfg.allocation_regularization * tree.depth_logits.square().mean()
        if allocation:
            result = result + cfg.allocation_regularization * torch.stack(
                [tree.get(k).allocation_logit.square() for k in allocation]).mean()
    return result / len(forest.trees)


class PruningStrategy(ABC):
    @abstractmethod
    def proposals(self, forest: AdaptiveForest, observations: dict[str, Observation]) -> list[StructuralAction]:
        raise NotImplementedError


class CustomPruning(PruningStrategy):
    """One configurable strategy expresses node, depth, tree, and hybrid modes.

    Scores rank candidates only. Training independently re-evaluates every
    proposed collapse on the control split before committing its removal.
    """
    def __init__(self, mode: str = "node", *, tolerance: float = 1e-5, max_actions: int = 4):
        if mode not in ("node", "depth", "tree", "hybrid"):
            raise ValueError("invalid pruning mode")
        self.mode, self.tolerance, self.max_actions = mode, tolerance, max_actions

    def proposals(self, forest: AdaptiveForest, observations: dict[str, Observation]) -> list[StructuralAction]:
        candidates = []
        for node in forest.iter_nodes():
            observation = observations.get(node.node_id)
            if (node.is_leaf and self.mode != "tree") or node.locked or observation is None:
                continue
            if self.mode == "tree" and node.depth != 0:
                continue
            if self.mode == "tree" and len(forest.trees) <= 1:
                continue
            utility = observation.utility if self.mode == "tree" else observation.refinement_utility
            if utility > self.tolerance and float(node.gate().detach()) > 1e-6:
                continue
            priority = -utility
            if self.mode in ("depth", "hybrid"):
                priority += 1e-6 * node.depth
            candidates.append(StructuralAction("remove_tree" if self.mode == "tree" else "prune", node.node_id, forest.topology_version,
                                                priority, f"{self.mode}: control utility proxy={utility:g}"))
        candidates.sort(key=lambda action: (-action.priority, action.node_id))
        return candidates[:self.max_actions]


class NodePruning(CustomPruning):
    def __init__(self, **kwargs):
        super().__init__("node", **kwargs)


class DepthPruning(CustomPruning):
    def __init__(self, **kwargs):
        super().__init__("depth", **kwargs)


class TreePruning(CustomPruning):
    def __init__(self, **kwargs):
        super().__init__("tree", **kwargs)


class CompositePruning(PruningStrategy):
    def __init__(self, strategies: list[tuple[float, PruningStrategy]]):
        if not strategies or any(not math.isfinite(weight) or weight < 0 for weight, _ in strategies):
            raise ValueError("nonnegative finite strategy weights are required")
        self.strategies = strategies

    def proposals(self, forest: AdaptiveForest, observations: dict[str, Observation]) -> list[StructuralAction]:
        combined = {}
        for weight, strategy in self.strategies:
            for action in strategy.proposals(forest, observations):
                key = (action.kind, action.node_id)
                if key in combined:
                    previous = combined[key]
                    combined[key] = replace(previous, priority=previous.priority + weight * action.priority)
                else:
                    combined[key] = replace(action, priority=weight * action.priority)
        # Typed operations are deduplicated, never numerically averaged as sets.
        return sorted(combined.values(), key=lambda a: (-a.priority, a.node_id))


class GrowthPruningPolicy:
    def __init__(self, config: StructureConfig, *, seed: int = 0,
                 pruning: PruningStrategy | None = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.pruning = pruning or CustomPruning(config.pruning_policy, tolerance=config.prune_tolerance)
        self.last_step = -1
        self.events: list[dict] = []

    def propose(self, forest: AdaptiveForest, observations: dict[str, Observation], step: int,
                phases: dict[int, str], progress: float) -> list[StructuralAction]:
        cfg = self.config
        if not cfg.dynamic or step <= self.last_step:
            return []
        self.last_step = step
        result = []
        if progress >= cfg.initial_dormant_fraction and step % cfg.grow_every == 0:
            for tree in forest.trees:
                if phases[tree.tree_id] not in ("grow", "reopen"):
                    continue
                eligible = []
                for node in tree.nodes.values():
                    observation = observations.get(node.node_id)
                    if not node.is_leaf or node.depth >= cfg.max_depth or node.locked or node.frozen:
                        continue
                    if observation is None or observation.occupancy < cfg.min_occupancy:
                        continue
                    if cfg.growth_policy == "preprune" and node.parent_id is not None:
                        parent = observations.get(node.parent_id)
                        if parent is None or parent.information < cfg.preprune_min_information:
                            continue
                    score = observation.occupancy * (observation.gradient_norm + abs(observation.utility) + .001)
                    if cfg.growth_policy == "level":
                        score = -node.depth + .001 * score
                    elif cfg.growth_policy == "random":
                        score = float(self.rng.random())
                    elif cfg.growth_policy == "uncertainty":
                        score = observation.occupancy * observation.uncertainty
                    elif cfg.growth_policy in ("information", "hybrid"):
                        parent = observations.get(node.parent_id)
                        information = parent.information if parent is not None else 0.
                        score = observation.occupancy * information
                        if cfg.growth_policy == "hybrid":
                            score += observation.occupancy * observation.gradient_norm * (1 + observation.uncertainty)
                    eligible.append((score, node.node_id))
                eligible.sort(key=lambda x: (-x[0], x[1]))
                for score, key in eligible[:cfg.grow_per_event]:
                    result.append(StructuralAction("grow", key, forest.topology_version, score,
                                                    f"{cfg.growth_policy} growth", cfg.arity))
        if step % cfg.prune_every == 0:
            for action in self.pruning.proposals(forest, observations):
                node = forest.node_map()[action.node_id]
                if phases[node.tree_id] == "consolidate":
                    result.append(action)
        return result

    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "last_step": self.last_step, "events": self.events}

    def load_state_dict(self, value: dict) -> None:
        self.rng.bit_generator.state = value["rng"]
        self.last_step, self.events = value["last_step"], value["events"]


class ScheduleManager:
    """One explicit, idempotent epoch clock with independently phased copses."""
    def __init__(self, config: ForestConfig):
        self.config = config
        self.last_epoch = -1
        self.events: list[dict] = []
        self.owned_frozen: set[str] = set()

    def phases(self, epoch: int) -> dict[int, str]:
        cfg = self.config.structure
        copses = math.ceil(self.config.n_trees / cfg.copse_size)
        phases = ("grow", "evaluate", "consolidate", "reopen")
        return {tree: phases[min(3, int(4 * ((epoch + (tree // cfg.copse_size) * cfg.cycle_epochs / copses)
                                             % cfg.cycle_epochs) / cfg.cycle_epochs))]
                for tree in range(self.config.n_trees)}

    def apply(self, forest: AdaptiveForest, epoch: int) -> dict[str, float]:
        cfg = self.config
        progress = epoch / max(1, cfg.epochs - 1)
        values = {name: schedule.value(progress) for name, schedule in cfg.schedules.items()}
        if epoch <= self.last_epoch:
            if epoch == self.last_epoch:
                return values
            raise ValueError("schedule epoch cannot move backwards")
        if "temperature" in values:
            for node in forest.iter_nodes():
                if not node.frozen:
                    node.set_temperature(values["temperature"])
        forest.feature_dropout = values.get("feature_dropout", cfg.feature_dropout)
        forest.tree_dropout = values.get("tree_dropout", cfg.tree_dropout)
        if not 0 <= forest.feature_dropout < 1 or not 0 <= forest.tree_dropout < 1:
            raise ValueError("scheduled dropout must remain in [0,1)")
        if cfg.structure.rolling_freeze or cfg.freeze_windows:
            protected = math.ceil(cfg.n_trees * cfg.structure.protect_tree_fraction)
            phases = self.phases(epoch)
            alive = set(forest.node_map())
            self.owned_frozen.intersection_update(alive)
            for node in forest.iter_nodes():
                if node.locked:
                    continue
                tree = forest.get_tree(node.tree_id)
                target, lock = False, False
                if node.tree_id >= protected and cfg.structure.rolling_freeze and phases[node.tree_id] == "consolidate":
                    scope = cfg.structure.rolling_scope
                    if scope == "tree":
                        target = True
                    elif scope == "depth":
                        first = (epoch // cfg.structure.cycle_epochs) % (cfg.structure.max_depth + 1)
                        target = first <= node.depth < first + cfg.structure.rolling_depth_band
                    else:
                        children = tree.get(tree.root_id).children_ids
                        if children:
                            key = children[(epoch // cfg.structure.cycle_epochs) % len(children)]
                            target = node.node_id == key or node.node_id in tree.descendants(key)
                if node.tree_id >= protected:
                    for window in cfg.freeze_windows:
                        if window.tree_id != node.tree_id or not window.start <= progress <= window.stop:
                            continue
                        if progress == window.stop and window.stop < 1.:
                            continue
                        if node.depth < window.min_depth or (window.max_depth is not None and node.depth > window.max_depth):
                            continue
                        if window.subtree_id is not None:
                            if window.subtree_id not in tree.nodes and tree._key(window.subtree_id) not in tree.nodes:
                                continue
                            if node.node_id != window.subtree_id and node.node_id not in tree.descendants(window.subtree_id):
                                continue
                        target, lock = True, lock or window.lock
                if target:
                    self.owned_frozen.add(node.node_id)
                    if not node.frozen or lock:
                        node.set_frozen(True, lock=lock)
                        self.events.append({"epoch": epoch, "node_id": node.node_id,
                                            "copse": node.tree_id // cfg.structure.copse_size,
                                            "event": "lock" if lock else "freeze"})
                elif node.node_id in self.owned_frozen:
                    node.set_frozen(False)
                    self.owned_frozen.remove(node.node_id)
                    self.events.append({"epoch": epoch, "node_id": node.node_id,
                                        "copse": node.tree_id // cfg.structure.copse_size, "event": "thaw"})
        self.last_epoch = epoch
        return values

    def state_dict(self) -> dict:
        return {"last_epoch": self.last_epoch, "events": self.events, "owned_frozen": sorted(self.owned_frozen)}

    def load_state_dict(self, state: dict) -> None:
        self.last_epoch, self.events = state["last_epoch"], state["events"]
        self.owned_frozen = set(state.get("owned_frozen", []))


class StructuralRelaxation:
    """Bounded local relaxation trials with delayed retain/restore decisions.

    Each trial halves (by default) one split's structural complexity pressure,
    without changing temperature or topology. It retains the lower pressure
    only after actual parameter motion and retained refinement-utility gain.
    This is an observational local experiment, not an unbiased causal estimate.
    """
    def __init__(self, config: StructureConfig):
        self.config = config
        self.multipliers: dict[str, float] = {}
        self.trials: dict[str, dict] = {}
        self.cooldowns: dict[str, int] = {}
        self.next_id = 0
        self.history: list[dict] = []
        self.last_step = -1

    def synchronize(self, alive: set[str], changed: set[str]) -> None:
        for key in list(self.trials):
            if key not in alive or key in changed:
                trial = self.trials.pop(key)
                if key in alive:
                    self.multipliers[key] = trial['baseline_multiplier']
                self.history.append({'event': 'relaxation_cancelled', 'node_id': key,
                                     'trial_id': trial['trial_id'], 'reason': 'topology_changed'})
        for mapping in (self.multipliers, self.cooldowns):
            for key in set(mapping) - alive:
                del mapping[key]

    def update(self, forest: AdaptiveForest, observations: dict[str, Observation],
               step: int, phases: dict[int, str]) -> None:
        cfg = self.config
        if not cfg.trial_relaxation or step <= self.last_step:
            return
        self.last_step = step
        for key in list(self.trials):
            trial = self.trials[key]
            observation = observations.get(key)
            if observation is None or observation.step <= trial['started']:
                continue
            trial['outcomes'].append(observation.refinement_utility)
            if len(trial['outcomes']) < cfg.relaxation_window:
                continue
            gain = min(trial['outcomes'][-min(2, cfg.relaxation_window):]) - trial['baseline_utility']
            movement = observation.path_length - trial['baseline_path']
            accepted = gain > cfg.relaxation_min_gain and movement > 1e-10
            if not accepted:
                self.multipliers[key] = trial['baseline_multiplier']
            self.cooldowns[key] = step + cfg.cycle_epochs
            self.history.append({'event': 'relaxation_outcome', 'node_id': key,
                                 'trial_id': trial['trial_id'], 'step': step,
                                 'retained_gain': gain, 'movement': movement, 'accepted': accepted})
            del self.trials[key]
        occupied_trees = {forest.node_map()[key].tree_id for key in self.trials if key in forest.node_map()}
        for tree in forest.trees:
            if tree.tree_id in occupied_trees or phases[tree.tree_id] != 'reopen':
                continue
            eligible = [o for o in observations.values() if o.tree_id == tree.tree_id
                        and o.topology_version == forest.topology_version and o.has_children and not o.frozen
                        and o.occupancy >= cfg.min_occupancy and o.node_id not in self.trials
                        and step >= self.cooldowns.get(o.node_id, -1)]
            if not eligible:
                continue
            observation = max(eligible, key=lambda o: (o.gradient_norm, -o.refinement_utility, o.node_id))
            key = observation.node_id
            baseline = self.multipliers.get(key, 1.)
            relaxed = max(.1, baseline * cfg.relaxation_factor)
            if relaxed >= baseline:
                continue
            trial_id = self.next_id; self.next_id += 1
            self.trials[key] = {'trial_id': trial_id, 'started': step,
                                'baseline_multiplier': baseline, 'baseline_utility': observation.refinement_utility,
                                'baseline_path': observation.path_length, 'outcomes': []}
            self.multipliers[key] = relaxed
            self.history.append({'event': 'relaxation_started', 'trial_id': trial_id,
                                 'node_id': key, 'step': step, 'baseline': baseline, 'relaxed': relaxed})

    def state_dict(self) -> dict:
        from copy import deepcopy
        return deepcopy({'multipliers': self.multipliers, 'trials': self.trials,
                         'cooldowns': self.cooldowns, 'next_id': self.next_id,
                         'history': self.history, 'last_step': self.last_step})

    def load_state_dict(self, state: dict) -> None:
        from copy import deepcopy
        for name, value in state.items():
            setattr(self, name, deepcopy(value))
