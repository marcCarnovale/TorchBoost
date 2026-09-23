"""Energy-accounted capacitor control. Heating and cooling are distinct operations."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class CapacitorController(nn.Module):
    """A scalar capacitor feeding parallel resistive heaters for active nodes.

    Q is charge; E=Q**2/(2*C) is electrical energy, not heat. After a positive
    validation-loss regression, charge is injected once. The first observation
    only establishes a reference. Existing charge discharges every advance,
    including after improvement. With G=sum(1/R), exact discharge is
    Q_next=Q*exp(-G*dt/C). Integrated resistor heat is the decrease in capacitor
    energy, allocated in proportion to conductance. This prevents creating energy
    by independently assigning a full discharge to every node.

    Temperatures are mapped to gate softness, never directly to optimizer LR.
    Linear cooling means Newton cooling, NOT exact Stefan-Boltzmann radiation.
    `cooling_law='radiative'` instead uses an implicit T**4-T_ambient**4 cooling
    step. Units are dimensionless model units. Exponential discharge, heat
    deposition, then cooling form a stable operator-splitting discretization;
    dt must be kept consistent when comparing experiments.

    All evolving state is registered as buffers and checkpointed. `advance` is
    called by training after observations; inference never advances this system.
    Frozen accepted boosting stages are never reheated by the candidate's controller.
    """

    def __init__(self, node_count: int, *, capacitance: float = 10.0,
                 injection_gain: float = 5.0, max_charge: float = 10.0,
                 ambient: float = 0.2, initial_temperature: float = 1.0,
                 heat_capacity: float = 1.0, cooling_rate: float = 0.02,
                 reference_decay: float = 0.8, regression_tolerance: float = 0.0,
                 cooling_law: str = "linear"):
        super().__init__()
        if node_count < 1:
            raise ValueError("node_count must be positive")
        numbers = [capacitance, injection_gain, max_charge, ambient,
                   initial_temperature, heat_capacity, cooling_rate,
                   reference_decay, regression_tolerance]
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("controller parameters must be finite")
        if min(capacitance, max_charge, ambient, heat_capacity) <= 0:
            raise ValueError("C, max_charge, ambient, and heat capacity must be positive")
        if min(injection_gain, cooling_rate, regression_tolerance) < 0:
            raise ValueError("gains and rates must be nonnegative")
        if not 0 <= reference_decay < 1 or initial_temperature < ambient:
            raise ValueError("invalid decay or initial temperature")
        if cooling_law not in ("linear", "radiative"):
            raise ValueError("cooling_law must be linear or radiative")
        self.cooling_law = cooling_law
        for key, value in dict(capacitance=capacitance, injection_gain=injection_gain,
                               max_charge=max_charge, ambient=ambient,
                               cooling_rate=cooling_rate, reference_decay=reference_decay,
                               regression_tolerance=regression_tolerance).items():
            self.register_buffer(key, torch.tensor(value, dtype=torch.float64))
        self.register_buffer("heat_capacity", torch.full((node_count,), heat_capacity, dtype=torch.float64))
        self.register_buffer("temperature", torch.full((node_count,), initial_temperature, dtype=torch.float64))
        self.register_buffer("charge", torch.zeros((), dtype=torch.float64))
        self.register_buffer("reference", torch.zeros((), dtype=torch.float64))
        self.register_buffer("has_reference", torch.tensor(False))
        self.register_buffer("last_injected_charge", torch.zeros((), dtype=torch.float64))
        self.register_buffer("last_rejected_charge", torch.zeros((), dtype=torch.float64))
        self.register_buffer("last_source_energy", torch.zeros((), dtype=torch.float64))
        self.register_buffer("last_heat", torch.zeros(node_count, dtype=torch.float64))
        self.register_buffer("last_cooling", torch.zeros(node_count, dtype=torch.float64))

    @torch.no_grad()
    def observe_validation(self, loss: float) -> None:
        """Use a controller-only development set; never pass final test losses."""
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("loss must be finite and nonnegative")
        self.last_injected_charge.zero_()
        self.last_rejected_charge.zero_()
        self.last_source_energy.zero_()
        if self.has_reference:
            delta = max(0.0, loss-float(self.reference)-float(self.regression_tolerance))
            requested = self.injection_gain * delta
            old_energy = self.charge.square()/(2*self.capacitance)
            new_charge = torch.minimum(self.charge+requested, self.max_charge)
            self.last_injected_charge.copy_(new_charge-self.charge)
            self.last_rejected_charge.copy_(requested-self.last_injected_charge)
            self.charge.copy_(new_charge)
            self.last_source_energy.copy_(new_charge.square()/(2*self.capacitance)-old_energy)
            self.reference.mul_(self.reference_decay).add_((1-self.reference_decay)*loss)
        else:
            self.reference.fill_(loss)
            self.has_reference.fill_(True)

    @torch.no_grad()
    def advance(self, resistance: Tensor, dt: float = 1.0) -> dict:
        """Discharge, deposit resistor heat, then cool. Return observable accounting.

        `current` and `power` are instantaneous at the beginning of this step;
        `heat` is integrated over the step and generally is not `power*dt`.
        Returns copies so a later call cannot mutate an earlier log record.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        r = resistance.detach().to(self.temperature)
        if r.shape != self.temperature.shape or not torch.isfinite(r).all() or (r <= 0).any():
            raise ValueError("resistance must be a finite positive node vector")
        conductance = r.reciprocal()
        total_g = conductance.sum()
        voltage = self.charge/self.capacitance
        current = voltage/r
        power = voltage*current
        before = self.charge.square()/(2*self.capacitance)
        self.charge.mul_(torch.exp(-total_g*dt/self.capacitance))
        released = before-self.charge.square()/(2*self.capacitance)
        self.last_heat.copy_(released*conductance/total_g)
        hot = self.temperature+self.last_heat/self.heat_capacity
        if self.cooling_law == "linear":
            cooled = self.ambient+(hot-self.ambient)*torch.exp(-self.cooling_rate*dt/self.heat_capacity)
        else:
            # Solve implicit passive radiation in [ambient, hot]. This never
            # overshoots below ambient, including for large controller timesteps.
            low, high = torch.full_like(hot, float(self.ambient)), hot.clone()
            for _ in range(48):
                mid = (low+high)/2
                residual = mid-hot+(self.cooling_rate*dt/self.heat_capacity)*(mid**4-self.ambient**4)
                high = torch.where(residual > 0, mid, high)
                low = torch.where(residual > 0, low, mid)
            cooled = (low+high)/2
        self.last_cooling.copy_(self.heat_capacity*(hot-cooled))
        self.temperature.copy_(cooled)
        return {"voltage": voltage.clone(), "current": current.clone(),
                "power": power.clone(), "heat": self.last_heat.clone(),
                "cooling": self.last_cooling.clone(), "charge": self.charge.clone(),
                "temperature": self.temperature.clone(), "released_energy": released.clone()}
