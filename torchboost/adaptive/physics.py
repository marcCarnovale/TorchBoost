"""Passive capacitor/RL networks and explicitly separate thermal dynamics.

Capacitor-only discharge is analytic. Parallel RL branches use implicit midpoint;
for fixed R,L,C its discrete energy identity is exact up to round-off:
  E_electrical(new) - E_electrical(old) = -dt * sum_j R_j * i_mid_j**2.
An ideal inductor stores energy; only resistance generates Joule heat.
The thermal update is operator-split, in dimensionless model units. It is not
claimed to be an exact solution of the fully coupled electrothermal ODE.
"""
from __future__ import annotations
from copy import deepcopy
import math
from typing import Callable
import numpy as np

from .config import PhysicsConfig
from .contracts import Observation


class PhysicalController:
    def __init__(self, config: PhysicsConfig, *, seed: int = 0,
                 resistance_policy: Callable[[Observation], float] | None = None):
        self.config = config
        self.resistance_policy = resistance_policy
        self.rng = np.random.default_rng(seed)
        self.nodes: dict[str, dict[str, float | int]] = {}
        self.charge = 0.
        self.reference: float | None = None
        self.last_step = -1
        self.last_loss: float | None = None
        self.history: list[dict] = []
        self.retired_energy = 0.
        self.birth_energy = 0.

    def _normalized_components(self, count: int) -> tuple[float, float, float, float]:
        cfg = self.config
        resistance = count * cfg.discharge_time / cfg.capacitance
        inductance = resistance * cfg.inductive_time
        capacity = cfg.total_heat_capacity / count
        cooling = capacity / cfg.cooling_time
        return resistance, inductance, capacity, cooling

    def _recalibrate_topology(self, thermal_target: float | None = None, magnetic_target: float | None = None) -> None:
        """Reparameterize normalized components without creating stored energy.

        Thermal deviations retain their relative pattern up to one global scale;
        magnetic currents retain signs/relative magnitudes up to one global scale.
        This avoids artificial local temperature spikes when the node count changes.
        """
        cfg = self.config
        if not cfg.topology_normalization or not self.nodes:
            return
        states=list(self.nodes.values())
        thermal_before=(sum(float(st["capacity"])*(float(st["temperature"])-cfg.ambient_temperature) for st in states)
                        if thermal_target is None else float(thermal_target))
        magnetic_before=(sum(.5*float(st.get("inductance",cfg.inductance))*float(st["current"])**2 for st in states)
                         if magnetic_target is None else float(magnetic_target))
        resistance, inductance, capacity, cooling = self._normalized_components(len(states))
        deltas=np.asarray([float(st["temperature"])-cfg.ambient_temperature for st in states],dtype=float)
        thermal_raw=capacity*float(deltas.sum())
        thermal_scale=(thermal_before/thermal_raw if abs(thermal_raw)>1e-15 else 0.)
        currents=np.asarray([float(st["current"]) for st in states],dtype=float)
        magnetic_raw=.5*inductance*float((currents**2).sum())
        current_scale=math.sqrt(magnetic_before/magnetic_raw) if magnetic_raw>1e-30 else 0.
        for i,state in enumerate(states):
            state["capacity"]=capacity;state["cooling"]=cooling
            state["resistance"]=resistance;state["inductance"]=inductance
            state["temperature"]=cfg.ambient_temperature+deltas[i]*thermal_scale
            state["current"]=float(currents[i]*current_scale)

    def synchronize(self, identities: dict[str, int]) -> None:
        cfg = self.config
        for key in set(self.nodes) - set(identities):
            state = self.nodes.pop(key)
            inductance = float(state.get("inductance", cfg.inductance))
            self.retired_energy += (state["capacity"] * (state["temperature"] - cfg.ambient_temperature)
                                    + .5 * inductance * state["current"] ** 2)
        thermal_target=sum(float(st["capacity"])*(float(st["temperature"])-cfg.ambient_temperature) for st in self.nodes.values())
        magnetic_target=sum(.5*float(st.get("inductance",cfg.inductance))*float(st["current"])**2 for st in self.nodes.values())
        count=max(1,len(identities))
        existing_mean=(sum(float(st["temperature"]) for st in self.nodes.values())/len(self.nodes)
                       if self.nodes else cfg.initial_temperature)
        nr,nl,nc,nk=self._normalized_components(count) if cfg.topology_normalization else (cfg.resistance,cfg.inductance,cfg.heat_capacity,cfg.cooling)
        for key, tree in identities.items():
            if key not in self.nodes:
                capacity = nc * math.exp(float(self.rng.normal(0., cfg.heterogeneity)))
                cooling = nk * math.exp(float(self.rng.normal(0., cfg.heterogeneity)))
                self.nodes[key] = {"tree": tree, "capacity": capacity, "cooling": cooling,
                                   "temperature": existing_mean if cfg.topology_normalization else cfg.initial_temperature, "current": 0.,
                                   "resistance": nr, "inductance": nl, "heat": 0., "power": 0.,
                                   "cooling_energy": 0., "inductive_energy": 0.}
                self.birth_energy += capacity * (cfg.initial_temperature - cfg.ambient_temperature)
        self._recalibrate_topology(thermal_target, magnetic_target)

    def electrical_energy(self) -> float:
        return self.charge ** 2 / (2. * self.config.capacitance) + sum(
            .5 * float(s.get("inductance", self.config.inductance)) * s["current"] ** 2 for s in self.nodes.values())

    def thermal_energy(self) -> float:
        return sum(s["capacity"] * (s["temperature"] - self.config.ambient_temperature) for s in self.nodes.values())

    def _resistances(self, keys: list[str], observations: dict[str, Observation]) -> np.ndarray:
        cfg = self.config
        result = []
        scale = max(1e-4, float(np.median([abs(o.utility) for o in observations.values()])) if observations else 1e-4)
        for key in keys:
            observation = observations.get(key)
            if self.resistance_policy is not None and observation is not None:
                resistance = float(self.resistance_policy(observation))
                if not math.isfinite(resistance) or resistance <= 0:
                    raise ValueError("resistance policy must return a finite positive number")
            else:
                log_multiplier = 0.
                if observation is not None:
                    usefulness = max(0., observation.utility) / scale
                    if cfg.allocation == "protective":
                        log_multiplier = min(4., usefulness) - min(2., observation.entropy) - min(2., observation.gradient_norm)
                    elif cfg.allocation == "uncertainty":
                        log_multiplier = -min(4., observation.entropy + max(0., -observation.utility / scale))
                    elif cfg.allocation == "gradient":
                        log_multiplier = -min(4., observation.gradient_norm + observation.structural_gradient)
                resistance = float(self.nodes[key].get("resistance", cfg.resistance)) * math.exp(log_multiplier)
            result.append(np.clip(resistance, cfg.resistance_min, cfg.resistance_max))
        result = np.asarray(result, dtype=np.float64)
        trees = np.asarray([self.nodes[k]["tree"] for k in keys])
        base_conductance = sum(1. / float(self.nodes[k].get("resistance", cfg.resistance)) for k in keys)
        if cfg.granularity == "global":
            result[:] = float(np.mean(result))
        elif cfg.granularity == "tree":
            for tree in np.unique(trees):
                mask = trees == tree
                result[mask] = float(np.mean(result[mask]))
        if cfg.hierarchical:
            conductance = 1. / result
            totals = []
            tree_ids = np.unique(trees)
            for tree in tree_ids:
                vals = conductance[trees == tree]
                totals.append(float(vals.sum() if cfg.bottom_up else vals.mean()))
            total = sum(totals)
            for tree, tree_total in zip(tree_ids, totals):
                mask = trees == tree
                # Fixed whole-network conductance budget 1 / base R.
                conductance[mask] *= (tree_total / total * base_conductance) / conductance[mask].sum()
            result = 1. / conductance
        if cfg.topology_normalization:
            conductance = 1. / result
            total_conductance = float(conductance.sum())
            if total_conductance > 0 and base_conductance > 0:
                conductance *= base_conductance / total_conductance
                result = 1. / conductance
        return result

    def _cool(self, temperature: np.ndarray, capacities: np.ndarray, cooling: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.cooling_law == "linear":
            return cfg.ambient_temperature + (temperature - cfg.ambient_temperature) * np.exp(-cooling * cfg.dt / capacities)
        low = np.full_like(temperature, cfg.ambient_temperature)
        high = temperature.copy()
        for _ in range(48):
            middle = (low + high) / 2
            residual = middle - temperature + cfg.dt * cooling / capacities * (middle**4 - cfg.ambient_temperature**4)
            low = np.where(residual < 0., middle, low)
            high = np.where(residual >= 0., middle, high)
        return (low + high) / 2

    def advance(self, loss: float, observations: dict[str, Observation], step: int) -> dict:
        cfg = self.config
        if not math.isfinite(loss):
            raise ValueError("controller loss must be finite")
        if step <= self.last_step:
            if step == self.last_step and loss == self.last_loss:
                return deepcopy(self.history[-1])
            raise ValueError("controller steps must increase; a repeated step must have identical input")
        if not self.nodes:
            raise ValueError("synchronize live nodes before advancing the controller")
        keys = sorted(self.nodes)
        before_electrical = self.electrical_energy()
        before_thermal = self.thermal_energy()
        injection = 0.
        if self.reference is not None and cfg.mode in ("capacitor", "rlc"):
            injection = min(cfg.max_injection, cfg.charge_gain * max(0., loss - self.reference))
            injection = min(injection, max(0., cfg.max_charge - self.charge))
        old_charge = self.charge
        self.charge += injection
        source_work = (self.charge**2 - old_charge**2) / (2 * cfg.capacitance)
        self.reference = loss if self.reference is None else cfg.smoothing * self.reference + (1 - cfg.smoothing) * loss
        resistance = self._resistances(keys, observations)
        current0 = np.asarray([self.nodes[k]["current"] for k in keys], dtype=np.float64)
        heat = np.zeros(len(keys))
        power = np.zeros(len(keys))
        current1 = np.zeros(len(keys))
        voltage0 = self.charge / cfg.capacitance
        if cfg.mode == "capacitor":
            conductance = 1. / resistance
            charge1 = self.charge * math.exp(-conductance.sum() * cfg.dt / cfg.capacitance)
            released = (self.charge**2 - charge1**2) / (2 * cfg.capacitance)
            heat = released * conductance / conductance.sum()
            instantaneous_current = voltage0 / resistance
            power = voltage0 * instantaneous_current
            self.charge = charge1
        elif cfg.mode == "rlc":
            h = cfg.dt
            inductance = np.asarray([self.nodes[k].get("inductance", cfg.inductance) for k in keys], dtype=np.float64)
            denominator = 1 + h * resistance / (2 * inductance)
            a = (1 - h * resistance / (2 * inductance)) / denominator
            b = h / (2 * inductance) / denominator
            z = h / (2 * cfg.capacitance)
            voltage1 = (voltage0 * (1 - z * b.sum()) - z * ((1 + a) * current0).sum()) / (1 + z * b.sum())
            current1 = a * current0 + b * (voltage0 + voltage1)
            midpoint = (current0 + current1) / 2
            heat = h * resistance * midpoint**2
            power = resistance * current0**2
            instantaneous_current = current0
            self.charge = cfg.capacitance * voltage1
        else:
            instantaneous_current = np.zeros(len(keys))
            # With no prior electrical state, cooling-only has exactly zero
            # injected/discharged heat, not a different cooling schedule.
            if abs(self.charge) > 1e-12 or np.any(current0):
                raise ValueError("cannot discard charged electrical state by changing controller mode")
        capacities = np.asarray([self.nodes[k]["capacity"] for k in keys])
        temperature0 = np.asarray([self.nodes[k]["temperature"] for k in keys])
        cooling = np.asarray([self.nodes[k]["cooling"] for k in keys])
        sparks = np.zeros(len(keys))
        if cfg.mode not in ("none", "cooling") and cfg.spark_probability > 0:
            selected = self.rng.random(len(keys)) < cfg.spark_probability
            if selected.any():
                sparks[selected] = cfg.spark_energy / selected.sum()
        temperature_heated = temperature0 + (heat + sparks) / capacities
        transfer = 0.
        if cfg.transfer_fraction and len(keys) > 1 and cfg.mode != "none":
            donor = int(np.argmax(capacities * (temperature_heated - cfg.ambient_temperature)))
            candidates = [i for i in range(len(keys)) if self.nodes[keys[i]]["tree"] != self.nodes[keys[donor]]["tree"]]
            receiver = min(candidates, key=lambda i: observations[keys[i]].utility if keys[i] in observations else 0.) if candidates else donor
            if donor != receiver:
                transfer = cfg.transfer_fraction * capacities[donor] * max(0., temperature_heated[donor] - cfg.ambient_temperature)
                temperature_heated[donor] -= transfer / capacities[donor]
                temperature_heated[receiver] += transfer / capacities[receiver]
        temperature_cooled = temperature_heated if cfg.mode == "none" else self._cool(temperature_heated, capacities, cooling)
        cooled_energy = capacities * (temperature_heated - temperature_cooled)
        temperature1 = np.clip(temperature_cooled, cfg.ambient_temperature, cfg.max_temperature)
        vented = float(np.sum(capacities * (temperature_cooled - temperature1)))
        if cfg.granularity in ("global", "tree"):
            groups = {0: np.ones(len(keys), bool)} if cfg.granularity == "global" else {
                tree: np.asarray([self.nodes[k]["tree"] == tree for k in keys])
                for tree in set(self.nodes[k]["tree"] for k in keys)}
            for mask in groups.values():
                temperature1[mask] = np.sum(temperature1[mask] * capacities[mask]) / capacities[mask].sum()
        node_log = {}
        for index, key in enumerate(keys):
            state = self.nodes[key]
            state.update(temperature=float(temperature1[index]), current=float(current1[index]),
                         resistance=float(resistance[index]), heat=float(heat[index]), power=float(power[index]),
                         cooling_energy=float(cooled_energy[index]),
                         inductive_energy=float(.5 * state.get("inductance", cfg.inductance) * current1[index]**2))
            node_log[key] = {**state, "instantaneous_current": float(instantaneous_current[index]),
                             "spark_energy": float(sparks[index])}
        after_electrical, after_thermal = self.electrical_energy(), self.thermal_energy()
        error = (after_electrical + after_thermal - before_electrical - before_thermal
                 - source_work - float(sparks.sum()) + float(cooled_energy.sum()) + vented)
        result = {"step": step, "loss": loss, "reference": self.reference, "charge": self.charge,
                  "injected_charge": injection, "source_work": source_work,
                  "electrical_before": before_electrical, "electrical_after": after_electrical,
                  "thermal_before": before_thermal, "thermal_after": after_thermal,
                  "resistor_heat": float(heat.sum()), "cooling_energy": float(cooled_energy.sum()),
                  "spark_energy": float(sparks.sum()), "transferred_energy": transfer,
                  "vented_energy": vented, "energy_error": error, "nodes": node_log}
        if abs(error) > 1e-8 * (1 + abs(before_electrical) + abs(before_thermal) + abs(source_work)):
            raise FloatingPointError(f"electrothermal energy-account error {error}")
        self.history.append(result)
        self.last_step, self.last_loss = step, loss
        return deepcopy(result)

    def snapshot(self) -> dict:
        return {"charge": self.charge, "nodes": deepcopy(self.nodes), "last_step": self.last_step,
                "retired_energy": self.retired_energy, "birth_energy": self.birth_energy}

    def state_dict(self) -> dict:
        return {**self.snapshot(), "reference": self.reference, "last_loss": self.last_loss,
                "rng": self.rng.bit_generator.state, "history": deepcopy(self.history)}

    def load_state_dict(self, value: dict) -> None:
        self.charge, self.nodes = value["charge"], deepcopy(value["nodes"])
        self.last_step, self.last_loss, self.reference = value["last_step"], value["last_loss"], value["reference"]
        self.retired_energy, self.birth_energy = value["retired_energy"], value["birth_energy"]
        self.rng.bit_generator.state = value["rng"]
        self.history = deepcopy(value["history"])