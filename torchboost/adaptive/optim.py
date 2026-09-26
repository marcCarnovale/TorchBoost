"""Optimizer lifecycle and explicit local momentum, separate from gate heat."""
from __future__ import annotations
from copy import deepcopy
import math
import torch
from torch import Tensor

from .config import ForestConfig
from .forest import AdaptiveForest


class LocalMomentum(torch.optim.Optimizer):
    """One EMA buffer per parameter, with independently controlled node groups.

    Energy feedback uses the previous buffer, so there is no circular update.
    The coefficient is bounded below one. Negative directional alignment damps
    stale state. This optimizer does not silently stack momentum on Adam.
    """
    def __init__(self, groups: list[dict], config: ForestConfig):
        defaults = {"lr": config.learning_rate, "weight_decay": config.weight_decay,
                    "beta": config.momentum, "beta_max": config.momentum_max,
                    "gain": config.momentum_energy_gain, "reversal_decay": config.reversal_decay,
                    "mode": config.optimizer, "inductive_energy": 0., "last_beta": 0.}
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            mode = group["mode"]
            energy = 0.
            if mode == "energy_momentum":
                buffers = [self.state[p]["momentum_buffer"] for p in group["params"]
                           if "momentum_buffer" in self.state[p]]
                elements = sum(buffer.numel() for buffer in buffers)
                energy = (.5 * sum(float(buffer.square().sum()) for buffer in buffers)
                          / max(1, elements))
            elif mode == "circuit_momentum":
                energy = max(0., group["inductive_energy"])
            beta = 0. if mode == "sgd" else group["beta"]
            if mode in ("energy_momentum", "circuit_momentum"):
                beta += (group["beta_max"] - beta) * (-math.expm1(-min(700., group["gain"] * energy)))
            group["last_beta"] = beta
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("nonfinite gradient")
                state = self.state[parameter]
                buffer = state.setdefault("momentum_buffer", torch.zeros_like(parameter))
                if torch.sum(buffer * gradient) < 0:
                    buffer.mul_(group["reversal_decay"])
                buffer.mul_(beta).add_(gradient, alpha=1. - beta)
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(buffer, alpha=-group["lr"])
        return loss


class DynamicOptimizer:
    def __init__(self, forest: AdaptiveForest, config: ForestConfig):
        self.config = config
        self.base_lr = config.learning_rate
        groups = self._groups(forest)
        self.optimizer = (torch.optim.AdamW(groups, lr=config.learning_rate, weight_decay=config.weight_decay)
                          if config.optimizer == "adamw" else LocalMomentum(groups, config))
        self.events: list[dict] = []

    @staticmethod
    def _groups(forest: AdaptiveForest) -> list[dict]:
        owners = {id(p): n.node_id for n in forest.iter_nodes() for p in n.parameters()}
        groups: dict[str, dict] = {}
        for name, parameter in forest.named_parameters():
            owner = owners.get(id(parameter), "global")
            group = groups.setdefault(owner, {"owner": owner, "params": [], "names": []})
            group["params"].append(parameter)
            group["names"].append(name)
        return list(groups.values())

    def synchronize(self, forest: AdaptiveForest) -> None:
        groups = self._groups(forest)
        alive = {p for g in groups for p in g["params"]}
        removed = [p for p in self.optimizer.state if p not in alive]
        freed_bytes = sum(v.numel() * v.element_size() for p in removed
                          for v in self.optimizer.state[p].values() if isinstance(v, Tensor))
        for p in removed:
            del self.optimizer.state[p]
        old = {g["owner"]: g for g in self.optimizer.param_groups}
        updated = []
        new_groups = []
        for group in groups:
            if group["owner"] in old:
                existing = old[group["owner"]]
                existing["params"], existing["names"] = group["params"], group["names"]
                updated.append(existing)
            else:
                new_groups.append(group)
        self.optimizer.param_groups[:] = updated
        for group in new_groups:
            self.optimizer.add_param_group(group)
        if removed or new_groups:
            self.events.append({"removed_parameter_states": len(removed), "freed_state_bytes": freed_bytes,
                                "new_groups": len(new_groups), "topology_version": forest.topology_version})
        self.assert_ownership(forest)

    def migrate_rows(self, migrations: list[tuple]) -> None:
        """Retain surviving Adam/EMA rows when packed ensemble heads shrink."""
        for old, new, keep in migrations:
            if old not in self.optimizer.state:
                continue
            values = self.optimizer.state.pop(old)
            self.optimizer.state[new] = {
                name: (value.index_select(0, keep.to(value.device)).clone()
                       if isinstance(value, Tensor) and value.shape == old.shape
                       else value.clone() if isinstance(value, Tensor) else deepcopy(value))
                for name, value in values.items()}

    def assert_ownership(self, forest: AdaptiveForest) -> None:
        parameters = [p for g in self.optimizer.param_groups for p in g["params"]]
        assert len(parameters) == len({id(p) for p in parameters}), "duplicate optimizer parameter"
        assert {id(p) for p in parameters} == {id(p) for p in forest.parameters()}, "optimizer/forest ownership diverged"
        assert {id(p) for p in self.optimizer.state} <= {id(p) for p in parameters}, "retired state remains owned"

    def set_controls(self, learning_rate: float, physical_nodes: dict[str, dict]) -> None:
        self.base_lr = learning_rate
        cfg = self.config.physics
        node_count = max(1, len(physical_nodes))
        thermal_scale = max(abs(cfg.thaw_temperature - cfg.ambient_temperature), 1e-8)
        for group in self.optimizer.param_groups:
            state = physical_nodes.get(group["owner"])
            factor = 1.
            if state is not None and cfg.lr_coupling:
                excursion = (float(state["temperature"]) - cfg.ambient_temperature) / thermal_scale
                factor = min(4., max(.25, 1 + cfg.lr_coupling * excursion))
            group["lr"] = learning_rate * factor
            if state is not None and self.config.optimizer == "circuit_momentum":
                group["inductive_energy"] = max(0., float(state["inductive_energy"])) * node_count

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        self.optimizer.step()

    def tensor_bytes(self) -> int:
        return sum(v.numel() * v.element_size() for values in self.optimizer.state.values()
                   for v in values.values() if isinstance(v, Tensor))

    def state_dict(self) -> dict:
        groups = []
        states = {}
        for group in self.optimizer.param_groups:
            groups.append({k: deepcopy(v) for k, v in group.items() if k != "params"})
            for name, parameter in zip(group["names"], group["params"]):
                if parameter in self.optimizer.state:
                    states[name] = deepcopy(self.optimizer.state[parameter])
        return {"groups": groups, "states": states, "events": deepcopy(self.events), "base_lr": self.base_lr}

    def load_state_dict(self, forest: AdaptiveForest, state: dict) -> None:
        parameters = dict(forest.named_parameters())
        self.optimizer.param_groups.clear()
        self.optimizer.state.clear()
        for description in state["groups"]:
            self.optimizer.add_param_group({**deepcopy(description), "params": [parameters[n] for n in description["names"]]})
        for name, values in state["states"].items():
            parameter = parameters[name]
            self.optimizer.state[parameter] = {k: (v.to(parameter.device) if isinstance(v, Tensor) else deepcopy(v)) for k, v in values.items()}
        self.events, self.base_lr = deepcopy(state["events"]), state["base_lr"]
        self.assert_ownership(forest)
