"""Detached elasto-plastic references; no metric collector or optimizer ownership.

Penalty: U = .5 * K * integrity * mean((theta-anchor)^2).
Yield uses RMS elastic stress K*integrity*RMS(theta-anchor). The radial flow
fraction is dt*mobility*positive(stress/Y_eff-1)^exponent, bounded below one.
Work hardening multiplies the base yield threshold by (1+hardness).
Consolidation evidence and release/overstress are deliberately separate signals.
"""
from __future__ import annotations
from copy import deepcopy
import math
import numpy as np
import torch
from torch import Tensor, nn

from .config import PlasticityConfig
from .contracts import Observation


class PlasticityModule:
    def __init__(self, config: PlasticityConfig, *, seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.states: dict[str, dict] = {}
        self.settings: dict[str, dict[str, float]] = {}
        self.versions: dict[str, int] = {}
        self.trial_backups: dict[int, dict] = {}
        self.events: list[dict] = []
        self.last_step = -1
        self.stiffness_multiplier = 1.

    def synchronize(self, packets: dict[str, dict[str, nn.Parameter]]) -> None:
        if self.config.mode == "none":
            self.states.clear(); self.settings.clear(); self.versions.clear()
            return
        alive = set(packets)
        for mapping in (self.states, self.settings, self.versions):
            for key in set(mapping) - alive:
                del mapping[key]
        for trial_id, backup in list(self.trial_backups.items()):
            if backup["node_id"] not in alive:
                del self.trial_backups[trial_id]
        for key, params in packets.items():
            if key not in self.states:
                self.states[key] = {"anchors": {}, "candidate": {}, "evidence": 0., "hardness": 0.,
                                    "integrity": 1., "last_strain": self.last_step, "last_step": self.last_step,
                                    "consolidated": False, "locked": False, "plastic_strain": 0.,
                                    "candidate_utility": -math.inf, "observed_movement": 0., "reference_path": 0.}
                factor = math.exp(float(self.rng.normal(0., self.config.randomized)))
                self.settings[key] = {"stiffness": factor, "yield_threshold": 1., "mobility": 1.,
                                      "consolidation_rate": 1.}
                self.versions[key] = 0
            state = self.states[key]
            for field in ("anchors", "candidate"):
                state[field] = {name: v for name, v in state[field].items() if name in params}
            for name, parameter in params.items():
                if name not in state["anchors"] or state["anchors"][name].shape != parameter.shape:
                    state["anchors"][name] = torch.zeros_like(parameter.detach())
                    state["candidate"][name] = torch.zeros_like(parameter.detach())
                else:
                    state["anchors"][name] = state["anchors"][name].to(parameter).detach()
                    state["candidate"][name] = state["candidate"][name].to(parameter).detach()

    def penalty(self, packets: dict[str, dict[str, nn.Parameter]]) -> Tensor:
        example = next(iter(next(iter(packets.values())).values()))
        result = example.new_zeros(())
        if self.config.mode == "none":
            return result
        for key, params in packets.items():
            state = self.states[key]
            if not state.get("admitted", True):
                continue
            stiffness = self.config.stiffness * self.settings[key]["stiffness"] * self.stiffness_multiplier
            squared = sum((p - state["anchors"][name]).square().sum() for name, p in params.items())
            size = sum(p.numel() for p in params.values())
            result = result + .5 * stiffness * state["integrity"] * squared / max(1, size)
        return result / max(1, len(packets))

    @torch.no_grad()
    def advance(self, packets: dict[str, dict[str, nn.Parameter]], observations: dict[str, Observation],
                temperatures: dict[str, float], step: int, *, progress: float,
                ambient_temperature: float = .2) -> dict:
        cfg = self.config
        if step <= self.last_step:
            if step == self.last_step:
                return {"locks": [], "events": []}
            raise ValueError("plasticity step must not move backwards")
        self.synchronize(packets)
        locks, events = [], []
        if cfg.mode == "none":
            self.last_step = step
            return {"locks": locks, "events": events}
        for key, params in packets.items():
            observation = observations.get(key)
            state, settings = self.states[key], self.settings[key]
            dt = max(1, step - state["last_step"])
            prior_step = state["last_step"]
            state["last_step"] = step
            if not state.get("admitted", True) or state["locked"] or observation is None or observation.frozen:
                continue
            total_size = sum(p.numel() for p in params.values())
            distance_squared = sum(float((p - state["anchors"][name]).square().sum()) for name, p in params.items()) / max(1, total_size)
            distance = math.sqrt(distance_squared)
            stiffness = cfg.stiffness * settings["stiffness"] * self.stiffness_multiplier
            stress = stiffness * state["integrity"] * distance
            temperature = temperatures.get(key, ambient_temperature)
            softening = math.exp(-min(50., cfg.thermal_softening * max(0., temperature - ambient_temperature)))
            yield_threshold = cfg.yield_threshold * settings["yield_threshold"] * (1 + state["hardness"]) * softening
            yield_ratio = stress / max(yield_threshold, 1e-12)
            flow_fraction, strain, consolidation_motion = 0., 0., 0.
            energy_before = .5 * stiffness * state["integrity"] * distance_squared
            # The observation is supplied by the independent collector. This law
            # does not collect metrics, evaluate success, or own an optimizer.
            # Negative ablation utility is evidence, not a causal retraining claim.
            harmful = (observation.utility < -cfg.release_min_utility
                       and observation.occupancy >= cfg.minimum_occupancy
                       and observation.update_norm > 0)
            state["harm_streak"] = state.get("harm_streak", 0) + 1 if harmful else 0
            release_allowed = (cfg.release_policy == "stress"
                               or state["harm_streak"] >= cfg.release_patience)
            if cfg.mode in ("plastic", "full") and yield_ratio > 1 and release_allowed:
                flow_fraction = min(cfg.max_flow_fraction,
                                    dt * cfg.mobility * settings["mobility"] * min(1e6, yield_ratio - 1.)**cfg.exponent)
                for name, parameter in params.items():
                    anchor = state["anchors"][name]
                    anchor.add_(parameter.detach() - anchor, alpha=flow_fraction)
                strain = distance * flow_fraction
                state["plastic_strain"] += strain
                state["hardness"] += cfg.work_hardening * strain
                state["last_strain"] = step
            elif cfg.recovery_rate:
                recovered_interval = max(0, step - max(prior_step, state["last_strain"] + cfg.recovery_delay))
                state["hardness"] *= math.exp(-cfg.recovery_rate * recovered_interval)
            damage = 0.
            if cfg.mode == "full" and yield_ratio > cfg.damage_threshold and release_allowed:
                instability = max(0., -observation.direction) * observation.update_norm / cfg.stability_scale
                adverse_utility = max(0., -observation.utility) / max(cfg.minimum_utility, 1e-4)
                damage = dt * cfg.damage_rate * (yield_ratio - cfg.damage_threshold) * (1 + min(4., instability + adverse_utility))
                state["integrity"] = max(0., state["integrity"] - damage)
                if state["integrity"] <= cfg.break_threshold:
                    state["integrity"] = 0.
            state["observed_movement"] += observation.update_norm
            stable = math.exp(-observation.update_norm / (cfg.stability_scale * (1 + observation.parameter_norm)))
            # Oscillating updates lose evidence even if their endpoint cancels.
            stable *= max(0., .5 + .5 * observation.direction)
            useful = (observation.utility > cfg.minimum_utility
                      and observation.occupancy >= cfg.minimum_occupancy
                      and observation.update_count > 0 and state["observed_movement"] > 1e-10)
            if cfg.mode in ("anchor", "plastic", "full"):
                state["evidence"] = cfg.evidence_decay**dt * state["evidence"] + (stable * dt if useful else 0.)
                if useful and observation.utility >= state["candidate_utility"]:
                    state["candidate"] = {name: p.detach().clone() for name, p in params.items()}
                    state["candidate_utility"] = observation.utility
                if state["evidence"] >= cfg.evidence_threshold:
                    state["consolidated"] = True
                elif state["evidence"] < max(0., cfg.evidence_threshold - cfg.hysteresis):
                    state["consolidated"] = False
                if cfg.sticky:
                    z = float(state["consolidated"])
                else:
                    logistic = lambda z: 1. / (1. + math.exp(-max(-60., min(60., z))))
                    baseline = logistic(-cfg.evidence_sharpness * cfg.evidence_threshold)
                    z = max(0., (logistic(cfg.evidence_sharpness * (state["evidence"] - cfg.evidence_threshold)) - baseline) / (1 - baseline))
                if useful and z > 0:
                    rate = min(1., cfg.consolidation_rate * settings["consolidation_rate"] * z * dt)
                    consolidation_motion = rate * math.sqrt(sum(float((state["candidate"][name] - state["anchors"][name]).square().sum()) for name in params) / max(1, total_size))
                    for name in params:
                        state["anchors"][name].lerp_(state["candidate"][name], rate)
                    if cfg.healing_rate and state["integrity"] < 1:
                        state["integrity"] = min(1., state["integrity"] + cfg.healing_rate * z * dt)
                if cfg.terminal_lock and useful and progress >= cfg.lock_fraction and state["evidence"] >= cfg.lock_evidence:
                    state["locked"] = True
                    locks.append(key)
            state["reference_path"] += strain + consolidation_motion
            energy_after = .5 * stiffness * state["integrity"] * sum(
                float((p - state["anchors"][name]).square().sum()) for name, p in params.items()) / max(1, total_size)
            event = {"step": step, "node_id": key, "stress": stress, "effective_yield": yield_threshold,
                     "flow_fraction": flow_fraction, "plastic_strain": strain,
                     "anchor_motion": strain + consolidation_motion, "reference_path": state["reference_path"],
                     "hardness": state["hardness"], "integrity": state["integrity"],
                     "damage": damage, "evidence": state["evidence"], "consolidated": state["consolidated"],
                     "locked": state["locked"], "reference_energy_change": energy_after - energy_before,
                     "harm_streak": state["harm_streak"], "release_allowed": release_allowed,
                     "blocked_release": bool(yield_ratio > 1 and not release_allowed)}
            events.append(event)
        self.events.extend(events)
        self.last_step = step
        return {"locks": locks, "events": events}

    def apply_trial(self, trial_id: int, node_id: str, factors: dict[str, float]) -> None:
        if node_id not in self.states:
            raise KeyError(node_id)
        if trial_id in self.trial_backups or any(v["node_id"] == node_id for v in self.trial_backups.values()):
            raise ValueError("duplicate or overlapping local trial")
        for name, factor in factors.items():
            if name not in self.settings[node_id] or not math.isfinite(factor) or factor <= 0:
                raise ValueError("unknown setting or nonpositive trial factor")
        self.trial_backups[trial_id] = {"node_id": node_id, "settings": deepcopy(self.settings[node_id])}
        for name, factor in factors.items():
            self.settings[node_id][name] = float(np.clip(self.settings[node_id][name] * factor, .1, 10.))
        self.versions[node_id] += 1

    def finish_trial(self, trial_id: int, *, retain: bool) -> None:
        backup = self.trial_backups.pop(trial_id, None)
        if backup is None:
            return
        key = backup["node_id"]
        if key in self.settings:
            if not retain:
                self.settings[key] = backup["settings"]
            self.versions[key] += 1

    @torch.no_grad()
    def reset_reference(self, node_id: str, parameters: dict[str, nn.Parameter], *, to_current: bool = False) -> None:
        state = self.states[node_id]
        if state["locked"]:
            raise ValueError("locked reference cannot be reset")
        state["anchors"] = {name: (p.detach().clone() if to_current else torch.zeros_like(p)) for name, p in parameters.items()}
        state["candidate"] = {name: p.detach().clone() for name, p in state["anchors"].items()}
        state["evidence"], state["candidate_utility"], state["consolidated"] = 0., -math.inf, False
        self.versions[node_id] += 1

    def snapshot(self) -> dict:
        return {key: {name: deepcopy(value) for name, value in state.items() if name not in ("anchors", "candidate")}
                for key, state in self.states.items()}

    def state_dict(self) -> dict:
        return {"states": deepcopy(self.states), "settings": deepcopy(self.settings), "versions": self.versions.copy(),
                "trial_backups": deepcopy(self.trial_backups), "events": deepcopy(self.events),
                "last_step": self.last_step, "stiffness_multiplier": self.stiffness_multiplier,
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, value: dict) -> None:
        self.states, self.settings = deepcopy(value["states"]), deepcopy(value["settings"])
        self.versions, self.trial_backups = value["versions"].copy(), deepcopy(value["trial_backups"])
        self.events, self.last_step = deepcopy(value["events"]), value["last_step"]
        self.stiffness_multiplier = value["stiffness_multiplier"]
        self.rng.bit_generator.state = value["rng"]
