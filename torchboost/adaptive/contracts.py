"""Immutable messages crossing observation, policy, and training boundaries."""
from __future__ import annotations
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Observation:
    node_id: str
    tree_id: int
    depth: int
    step: int
    topology_version: int
    occupancy: float
    entropy: float
    information: float
    utility: float
    refinement_utility: float
    gradient_norm: float
    structural_gradient: float
    update_norm: float
    path_length: float
    direction: float
    parameter_norm: float
    temperature: float
    frozen: bool
    has_children: bool
    structural_gate: float
    update_count: int
    charge: float = 0.
    heat: float = 0.
    inductive_energy: float = 0.
    hardness: float = 0.
    integrity: float = 1.
    reference_path_length: float = 0.
    phase: int = 0
    training_progress: float = 0.
    uncertainty: float = 0.
    utility_change: float = 0.

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StructuralAction:
    kind: str
    node_id: str
    topology_version: int
    priority: float = 0.
    reason: str = ""
    arity: int | None = None
    recursive: bool = False


@dataclass(frozen=True)
class Proposal:
    run_id: str
    trial_id: int
    node_id: str
    topology_version: int
    settings_version: int
    step: int
    action: int
    propensity: float
    context: tuple[float, ...]


@dataclass(frozen=True)
class TrialOutcome:
    trial_id: int
    node_id: str
    action: int
    context: tuple[float, ...]
    reward: float
    retained_gain: float
    movement: float
    accepted: bool
    reason: str
