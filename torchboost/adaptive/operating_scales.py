"""Topology-calibrated dimensionless operating scales for the native controller.

For N parallel, identical branches choose R_j=N*tau_Q/C, L_j=R_j*tau_L,
c_j=H/N and k_j=c_j/tau_T. Thus uniform charge, total current and temperature
trajectories are independent of N. The RLC midpoint discretization retains its
energy identity. This is an initialization law, not permission to change
inductance or heat capacity of a charged live circuit without accounting work.
"""
from dataclasses import dataclass
import math
from .config import PhysicsConfig


@dataclass(frozen=True)
class OperatingScales:
    discharge_time: float = 5.
    inductive_time: float = 2.
    cooling_time: float = 64.
    total_heat_capacity: float = .1
    capacitance: float = 1.
    dt: float = 1.
    initial_temperature: float = 2.
    ambient_temperature: float = 1.
    max_temperature: float = 5.
    charge_gain: float = 8.
    max_injection: float = .25
    max_charge: float = 2.

    def __post_init__(self):
        for name in ('discharge_time','inductive_time','cooling_time','total_heat_capacity','capacitance','dt'):
            v=getattr(self,name)
            if not math.isfinite(v) or v <= 0: raise ValueError(f'{name} must be positive')

    def physics(self, nodes: int, *, mode: str = 'capacitor', allocation: str = 'uniform') -> PhysicsConfig:
        if not isinstance(nodes,int) or nodes < 1: raise ValueError('nodes must be a positive integer')
        r=nodes*self.discharge_time/self.capacitance
        capacity=self.total_heat_capacity/nodes
        return PhysicsConfig(mode=mode, allocation=allocation, resistance=r,
            resistance_min=r/1000.,resistance_max=r*1000.,
            capacitance=self.capacitance,inductance=r*self.inductive_time,
            heat_capacity=capacity,cooling=capacity/self.cooling_time,
            initial_temperature=self.initial_temperature,ambient_temperature=self.ambient_temperature,
            max_temperature=self.max_temperature,dt=self.dt,charge_gain=self.charge_gain,
            max_injection=self.max_injection,max_charge=self.max_charge,thaw_temperature=2.5)
