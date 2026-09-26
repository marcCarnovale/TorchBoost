"""Experiment-only dimensionless loss signal and energy-budgeted controllers.

No production solver/default is changed. All source arms consume the same
causal loss-innovation signal. One source unit per controller-time unit supplies
H_total*(T_thaw-T_ambient) energy. Direct heat and electrical stored energy are
separate ledger entries. Equal source energy is NOT equal thermal exposure.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from torchboost.adaptive.physics import PhysicalController


@dataclass(frozen=True)
class EnergySource:
    rate: float = 0.1
    threshold: float = 1.0
    clip: float = 4.0
    smoothing: float = 0.9
    warmup: int = 8

    def __post_init__(self):
        for name in ("rate", "threshold", "clip", "smoothing"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.rate < 0 or self.threshold < 0 or self.clip <= 0:
            raise ValueError("invalid source strength or surprise limits")
        if not 0 < self.smoothing < 1 or not isinstance(self.warmup, int) or self.warmup < 2:
            raise ValueError("invalid signal memory or warmup")


class InnovationSignal:
    """Past-only EW innovation scale, affine-invariant for positive loss scaling.

Warmup observes without sourcing. Zero past variation uses |innovation| as an
explicit fallback, giving a unit shock rather than infinite surprise. This is
not a significance test: overlapping control observations are dependent.
"""

    def __init__(self, config: EnergySource):
        self.config = config
        self.count = 0
        self.mean = None
        self.variance = 0.0

    def observe(self, loss):
        if not math.isfinite(loss):
            raise ValueError("finite loss required")
        delta = 0.0 if self.mean is None else loss - self.mean
        scale = math.sqrt(self.variance) if self.variance > 0 else abs(delta)
        z = delta / scale if scale > 0 else 0.0
        active = self.count >= self.config.warmup
        drive = min(self.config.clip, max(0.0, z - self.config.threshold)) if active else 0.0
        rho = self.config.smoothing
        self.mean = loss if self.mean is None else rho * self.mean + (1 - rho) * loss
        self.variance = rho * self.variance + (1 - rho) * delta**2
        self.count += 1
        return {"innovation": delta, "past_scale": scale, "normalized_surprise": z,
                "source_drive": drive, "signal_count": self.count}

    def state_dict(self):
        return {"count": self.count, "mean": self.mean, "variance": self.variance}

    def load_state_dict(self, state):
        self.count, self.mean, self.variance = state["count"], state["mean"], state["variance"]


def add_capacitor_energy(charge, capacitance, energy, maximum):
    """Increase |q| while preserving sign, respecting |q| <= maximum.

For RLC this must handle negative charge. Source WORK is nonnegative even when
injected signed charge is negative. Excess requested energy is not hidden.
"""
    if not all(math.isfinite(v) for v in (charge, capacitance, energy, maximum)):
        raise ValueError("finite capacitor source values required")
    if capacitance <= 0 or energy < 0 or maximum <= 0:
        raise ValueError("invalid capacitor source values")
    if abs(charge) > maximum:
        # A passive RLC oscillation may move previously stored magnetic energy
        # into the capacitor above its SOURCE limit; never erase that energy.
        return charge, 0.0
    target = min(maximum, math.sqrt(charge**2 + 2 * capacitance * energy))
    updated = math.copysign(target, charge if charge else 1.0)
    actual = (updated**2 - charge**2) / (2 * capacitance)
    return updated, max(0.0, actual)


class NormalizedEnergyController(PhysicalController):
    """Energy-input comparator using the existing passive electrical solver.

Modes cooling/capacitor/rlc mean direct heat/capacitor/RLC respectively. The
original loss->charge source is disabled in a PRIVATE copied configuration.
Checkpoint configuration identity is checked before source-state restoration.
"""

    def __init__(self, config, *, source=None, seed=0, resistance_policy=None):
        if config.mode not in ("cooling", "capacitor", "rlc"):
            raise ValueError("normalized source requires cooling, capacitor, or rlc")
        if not config.topology_normalization or config.thaw_temperature <= config.ambient_temperature:
            raise ValueError("normalized topology and positive thaw interval required")
        self.source = source or EnergySource()
        super().__init__(replace(config, charge_gain=0.0, max_injection=0.0),
                         seed=seed, resistance_policy=resistance_policy)
        self.signal = InnovationSignal(self.source)
        self.source_energy_total = 0.0

    def advance(self, loss, observations, step):
        if not math.isfinite(loss) or step <= self.last_step or not self.nodes:
            return super().advance(loss, observations, step)
        keys = sorted(self.nodes)
        # Validate spatial allocation before mutating signal/physical state.
        resistance = self._resistances(keys, observations)
        next_signal = deepcopy(self.signal)
        signal = next_signal.observe(loss)
        cfg = self.config
        unit = cfg.total_heat_capacity * (cfg.thaw_temperature - cfg.ambient_temperature)
        requested = self.source.rate * cfg.dt * unit * signal["source_drive"]
        before_e, before_t, old_q = self.electrical_energy(), self.thermal_energy(), self.charge
        direct = electrical = 0.0
        additions = np.zeros(len(keys))
        if cfg.mode == "cooling":
            direct = requested
            conductance = 1.0 / resistance
            additions = direct * conductance / conductance.sum()
            for key, addition in zip(keys, additions):
                self.nodes[key]["temperature"] += float(addition) / self.nodes[key]["capacity"]
        else:
            self.charge, electrical = add_capacitor_energy(old_q, cfg.capacitance, requested, cfg.max_charge)
        sourced_q = self.charge
        result = super().advance(loss, observations, step)
        self.signal = next_signal
        self.source_energy_total += direct + electrical
        result.update(signal)
        result.update(electrical_before=before_e, thermal_before=before_t,
                      injected_charge=sourced_q - old_q,
                      source_work=electrical, external_heat=direct,
                      source_energy_requested=requested, source_energy_rejected=max(0., requested - direct - electrical),
                      source_energy_total=self.source_energy_total)
        for key, addition in zip(keys, additions):
            result["nodes"][key]["direct_heat"] = float(addition)
        result["energy_error"] = (
            result["electrical_after"] + result["thermal_after"] - before_e - before_t
            - electrical - direct - result["spark_energy"] + result["cooling_energy"] + result["vented_energy"])
        if abs(result["energy_error"]) > 1e-8 * (1 + abs(before_e) + abs(before_t) + requested):
            raise FloatingPointError("normalized source ledger failed")
        self.history[-1] = deepcopy(result)
        return deepcopy(result)

    def state_dict(self):
        return {**super().state_dict(), "energy_source": asdict(self.source),
                "source_physics": asdict(self.config), "innovation_signal": self.signal.state_dict(),
                "source_energy_total": self.source_energy_total}

    def load_state_dict(self, state):
        if state.get("energy_source") != asdict(self.source) or state.get("source_physics") != asdict(self.config):
            raise ValueError("normalized source checkpoint configuration mismatch")
        super().load_state_dict(state)
        self.signal.load_state_dict(state["innovation_signal"])
        self.source_energy_total = state["source_energy_total"]
