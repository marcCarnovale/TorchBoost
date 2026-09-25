"""Validated, serializable experiment controls for the adaptive joint forest.

No field is silently accepted: unknown keywords fail through dataclass construction.
Research mechanisms are disabled in the conservative baseline configuration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


def positive(name: str, value: float, *, zero: bool = False) -> None:
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def choice(name: str, value: str, values: tuple[str, ...]) -> None:
    if value not in values:
        raise ValueError(f"{name} must be one of {values}, got {value!r}")


def probability(name: str, value: float, *, inclusive_one: bool = True) -> None:
    if not math.isfinite(value) or value < 0 or (value > 1 if inclusive_one else value >= 1):
        raise ValueError(f"{name} must be in [0, {'1]' if inclusive_one else '1)'}")


@dataclass
class ScheduleConfig:
    """Frequency is cycles per full training horizon; phase is radians."""
    kind: str = "constant"
    low: float = 1.0
    high: float = 1.0
    cycles: float = 1.0
    phase: float = 0.0

    def __post_init__(self) -> None:
        choice("schedule kind", self.kind, ("constant", "linear", "cosine", "geometric", "oscillatory"))
        for name in ("low", "high", "cycles"):
            positive(name, getattr(self, name), zero=True)
        if not math.isfinite(self.phase):
            raise ValueError("phase must be finite")
        if self.kind == "geometric" and min(self.low, self.high) <= 0:
            raise ValueError("geometric endpoints must be positive")

    def value(self, progress: float) -> float:
        t = min(1., max(0., float(progress)))
        if self.kind == "constant":
            return self.low
        if self.kind == "linear":
            z = t
        elif self.kind == "cosine":
            z = .5 - .5 * math.cos(math.pi * t)
        elif self.kind == "geometric":
            return self.low * (self.high / self.low) ** t
        else:
            z = .5 - .5 * math.cos(2 * math.pi * self.cycles * t + self.phase)
        return self.low + (self.high - self.low) * z


@dataclass
class StructureConfig:
    dynamic: bool = False
    initial_depth: int = 1
    max_depth: int = 3
    arity: int = 2
    max_nodes: int = 127                    # Per tree, real allocated-node budget.
    max_parameters: int = 1_000_000         # Whole forest budget, checked on growth.
    grow_every: int = 6
    prune_every: int = 6
    growth_policy: str = "best_first"
    pruning_policy: str = "node"
    grow_per_event: int = 1
    min_occupancy: float = .01
    prune_tolerance: float = 1e-5
    initial_dormant_fraction: float = .10
    protect_tree_fraction: float = 0.
    rolling_freeze: bool = False
    rolling_scope: str = "tree"
    rolling_depth_band: int = 1
    copse_size: int = 2
    cycle_epochs: int = 12
    structural_gate: bool = True
    gate_bimodality: float = 1e-4
    complexity: float = 1e-4
    depth_allocation: str = "learned"
    allocation_regularization: float = 1e-4
    structural_temperature_coupling: float = 0.
    trial_relaxation: bool = False
    relaxation_factor: float = .5
    relaxation_window: int = 3
    relaxation_min_gain: float = 1e-5
    preprune_min_information: float = 0.

    def __post_init__(self) -> None:
        for name in ("initial_depth", "max_depth"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.initial_depth > self.max_depth:
            raise ValueError("initial_depth exceeds max_depth")
        if not isinstance(self.arity, int) or self.arity < 2:
            raise ValueError("arity must be an integer >= 2")
        for name in ("max_nodes", "max_parameters", "grow_every", "prune_every", "grow_per_event", "copse_size", "cycle_epochs", "relaxation_window", "rolling_depth_band"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.relaxation_factor < 1:
            raise ValueError("relaxation_factor must lie strictly between zero and one")
        positive("relaxation_min_gain", self.relaxation_min_gain, zero=True)
        choice("rolling_scope", self.rolling_scope, ("tree", "depth", "subtree"))
        choice("growth_policy", self.growth_policy, ("best_first", "level", "random", "preprune", "uncertainty", "information", "hybrid"))
        choice("pruning_policy", self.pruning_policy, ("node", "depth", "tree", "hybrid"))
        choice("depth_allocation", self.depth_allocation, ("learned", "exponential", "uniform"))
        for name in ("min_occupancy", "initial_dormant_fraction", "protect_tree_fraction"):
            probability(name, getattr(self, name))
        for name in ("prune_tolerance", "gate_bimodality", "complexity", "allocation_regularization", "structural_temperature_coupling", "preprune_min_information"):
            positive(name, getattr(self, name), zero=True)
        depth = self.initial_depth if self.dynamic else self.max_depth
        needed = (self.arity ** (depth + 1) - 1) // (self.arity - 1)
        if needed > self.max_nodes:
            raise ValueError(f"initial tree needs {needed} nodes but max_nodes={self.max_nodes}")


@dataclass
class PhysicsConfig:
    mode: str = "none"                     # none / cooling / capacitor / rlc
    allocation: str = "uniform"            # uniform / protective / uncertainty / gradient
    granularity: str = "node"              # global / tree / node
    hierarchical: bool = False
    bottom_up: bool = False
    capacitance: float = 1.
    inductance: float = 1.
    charge_gain: float = 3.
    max_injection: float = .5
    max_charge: float = 5.
    smoothing: float = .8
    dt: float = .2
    resistance: float = 10.
    resistance_min: float = .1
    resistance_max: float = 1000.
    initial_temperature: float = 1.
    ambient_temperature: float = 1.
    max_temperature: float = 5.
    heat_capacity: float = 1.
    cooling: float = .2
    cooling_law: str = "linear"
    heterogeneity: float = 0.
    thaw_temperature: float = 1.2
    lr_coupling: float = 0.
    spark_probability: float = 0.
    spark_energy: float = .02
    transfer_fraction: float = 0.
    # Optional system-level calibration. When enabled, component values are
    # derived from whole-controller time constants and rescaled as topology
    # changes while preserving stored thermal/inductive energy.
    topology_normalization: bool = False
    discharge_time: float = 5.
    inductive_time: float = 2.
    cooling_time: float = 64.
    total_heat_capacity: float = .1

    def __post_init__(self) -> None:
        choice("physics mode", self.mode, ("none", "cooling", "capacitor", "rlc"))
        choice("allocation", self.allocation, ("uniform", "protective", "uncertainty", "gradient"))
        choice("granularity", self.granularity, ("global", "tree", "node"))
        choice("cooling_law", self.cooling_law, ("linear", "radiative"))
        for name in ("capacitance", "inductance", "dt", "resistance", "resistance_min", "resistance_max", "initial_temperature", "ambient_temperature", "max_temperature", "heat_capacity", "thaw_temperature", "max_charge", "discharge_time", "inductive_time", "cooling_time", "total_heat_capacity"):
            positive(name, getattr(self, name))
        for name in ("charge_gain", "max_injection", "cooling", "heterogeneity", "lr_coupling", "spark_energy"):
            positive(name, getattr(self, name), zero=True)
        for name in ("spark_probability", "transfer_fraction"):
            probability(name, getattr(self, name))
        probability("smoothing", self.smoothing, inclusive_one=False)
        if not self.ambient_temperature <= self.initial_temperature <= self.max_temperature:
            raise ValueError("require ambient <= initial <= maximum temperature")
        if self.resistance_min > self.resistance_max:
            raise ValueError("resistance bounds are reversed")


@dataclass
class PlasticityConfig:
    mode: str = "none"                     # none / anchor / elastic / plastic / full
    stiffness: float = 1e-3
    yield_threshold: float = .1
    mobility: float = .1
    exponent: float = 1.
    max_flow_fraction: float = .25
    work_hardening: float = .1
    thermal_softening: float = 0.
    recovery_rate: float = 0.
    recovery_delay: int = 3
    damage_rate: float = .05
    damage_threshold: float = 2.
    break_threshold: float = .05
    healing_rate: float = 0.
    evidence_decay: float = .9
    evidence_threshold: float = 2.
    evidence_sharpness: float = 3.
    consolidation_rate: float = .05
    stability_scale: float = .1
    minimum_utility: float = 1e-5
    minimum_occupancy: float = .01
    hysteresis: float = .5
    sticky: bool = True
    terminal_lock: bool = False
    lock_fraction: float = 1.0
    lock_evidence: float = 8.
    randomized: float = 0.
    release_policy: str = "stress"
    release_patience: int = 3
    release_min_utility: float = 1e-4

    def __post_init__(self) -> None:
        choice("plasticity mode", self.mode, ("none", "anchor", "elastic", "plastic", "full"))
        choice("release_policy", self.release_policy, ("stress", "persistent_harm"))
        if not isinstance(self.release_patience, int) or self.release_patience < 1:
            raise ValueError("release_patience must be positive")
        positive("release_min_utility", self.release_min_utility, zero=True)
        for name in ("stiffness", "mobility", "work_hardening", "thermal_softening", "recovery_rate", "damage_rate", "healing_rate", "consolidation_rate", "minimum_utility", "hysteresis", "randomized"):
            positive(name, getattr(self, name), zero=True)
        for name in ("yield_threshold", "exponent", "evidence_threshold", "evidence_sharpness", "stability_scale", "damage_threshold", "lock_evidence"):
            positive(name, getattr(self, name))
        for name in ("max_flow_fraction", "break_threshold", "minimum_occupancy", "lock_fraction"):
            probability(name, getattr(self, name))
        probability("evidence_decay", self.evidence_decay, inclusive_one=False)
        if not isinstance(self.recovery_delay, int) or self.recovery_delay < 0:
            raise ValueError("recovery_delay must be nonnegative")


@dataclass
class OnlineConfig:
    defer_structure_for_trials: bool = False
    deformation_source: str = "reference"
    enabled: bool = False
    concurrent: bool = False
    interval: int = 4
    window: int = 3
    cooldown: int = 3
    max_trials: int = 4
    exploration: float = .2
    ridge: float = 1.
    ucb: float = .2
    forgetting: float = .995
    minimum_movement: float = 1e-5
    minimum_gain: float = 1e-5

    def __post_init__(self) -> None:
        choice("deformation_source", self.deformation_source, ("reference", "parameters"))
        for name in ("interval", "window", "cooldown", "max_trials"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        probability("exploration", self.exploration)
        probability("forgetting", self.forgetting)
        positive("ridge", self.ridge)
        for name in ("ucb", "minimum_movement", "minimum_gain"):
            positive(name, getattr(self, name), zero=True)


@dataclass(frozen=True)
class FreezeWindow:
    """Automatic freeze interval on a stable subtree and/or inclusive depth band.

    Fractions use the original configured training horizon. ``stop=1`` includes
    its final epoch. ``lock=True`` makes the freeze terminal once the window is
    entered; stopping the window does not thaw a terminally locked node.
    """
    tree_id: int
    start: float
    stop: float
    min_depth: int = 0
    max_depth: int | None = None
    subtree_id: str | None = None
    lock: bool = False

    def __post_init__(self):
        if not isinstance(self.tree_id, int) or self.tree_id < 0:
            raise ValueError("freeze tree_id must be nonnegative")
        probability("freeze start", self.start); probability("freeze stop", self.stop)
        if self.stop < self.start:
            raise ValueError("freeze window endpoints are reversed")
        if self.min_depth < 0 or (self.max_depth is not None and self.max_depth < self.min_depth):
            raise ValueError("invalid freeze depth range")
        if self.subtree_id is not None and not self.subtree_id.startswith(f"{self.tree_id}:"):
            raise ValueError("freeze subtree must belong to its declared tree")


@dataclass
class ForestConfig:
    n_trees: int = 6
    execution: str = "packed"              # vectorized soft / sparse-path hard execution
    head_mode: str = "shared"
    aggregation: str = "attention"
    shrinkage: float = 1.
    residual_weights: bool = True
    epochs: int = 40
    batch_size: int = 128
    accumulation_steps: int = 1
    learning_rate: float = .03
    optimizer: str = "adamw"
    weight_decay: float = 1e-5
    momentum: float = .8
    momentum_max: float = .95
    momentum_energy_gain: float = 1.
    reversal_decay: float = .2
    feature_dropout: float = 0.
    tree_dropout: float = 0.
    node_linear_values: bool = False
    diversity: float = 0.
    feature_penalties: tuple[float, ...] = ()
    monotonicity: tuple[tuple[int, int, int], ...] = ()  # (output, feature, sign)
    monotonicity_penalty: float = 0.
    interaction_groups: tuple[tuple[int, ...], ...] = ()
    # Semantic feature groups constrain oblique routing and affine node models.
    # Unlike interaction_groups (tree-level masks), these may overlap and are
    # chosen per node by proposal/optimization logic.
    feature_groups: tuple[tuple[int, ...], ...] = ()
    record_diagnostics: bool = False
    compact_history: bool = False
    collect_metrics: bool = True
    observation_every: int = 1
    control_sample_size: int = 256
    gradient_clip: float = 10.
    random_state: int = 0
    device: str = "cpu"
    structure: StructureConfig = field(default_factory=StructureConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    schedules: dict[str, ScheduleConfig] = field(default_factory=dict)
    freeze_windows: tuple[FreezeWindow, ...] = ()

    def __post_init__(self) -> None:
        for name, cls in (("structure", StructureConfig), ("physics", PhysicsConfig), ("plasticity", PlasticityConfig), ("online", OnlineConfig)):
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, cls(**value))
        self.freeze_windows = tuple(FreezeWindow(**w) if isinstance(w, dict) else w for w in self.freeze_windows)
        if any(w.tree_id >= self.n_trees for w in self.freeze_windows):
            raise ValueError("freeze window names a tree outside the original ensemble")
        self.schedules = {k: ScheduleConfig(**v) if isinstance(v, dict) else v for k, v in self.schedules.items()}
        allowed = {"learning_rate", "temperature", "complexity", "bimodality", "diversity", "plastic_stiffness", "feature_dropout", "tree_dropout"}
        if set(self.schedules) - allowed:
            raise ValueError(f"unknown scheduled controls: {set(self.schedules) - allowed}")
        if "temperature" in self.schedules and self.physics.mode != "none":
            raise ValueError("temperature has one owner: choose schedule OR physics")
        choice("execution", self.execution, ("reference", "packed", "forest_packed"))
        choice("head_mode", self.head_mode, ("shared", "specialized"))
        choice("aggregation", self.aggregation, ("attention", "mean", "additive"))
        choice("optimizer", self.optimizer, ("adamw", "sgd", "momentum", "energy_momentum", "circuit_momentum"))
        for name in ("n_trees", "epochs", "batch_size", "accumulation_steps", "observation_every", "control_sample_size"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("shrinkage", "learning_rate", "gradient_clip"):
            positive(name, getattr(self, name))
        for name in ("weight_decay", "momentum_energy_gain", "diversity", "monotonicity_penalty"):
            positive(name, getattr(self, name), zero=True)
        for name in ("feature_dropout", "tree_dropout", "momentum", "momentum_max"):
            probability(name, getattr(self, name), inclusive_one=False)
        probability("reversal_decay", self.reversal_decay)
        if self.momentum_max < self.momentum:
            raise ValueError("momentum_max must be >= momentum")
        if not self.collect_metrics and (self.structure.dynamic or self.structure.trial_relaxation
                or self.plasticity.mode != "none" or self.online.enabled or self.physics.allocation != "uniform"):
            raise ValueError("data-guided policies require split metric collection")
        if self.online.enabled and self.plasticity.mode == "none":
            raise ValueError("online plasticity experiments require a plasticity mode")
        if self.optimizer == "circuit_momentum" and self.physics.mode != "rlc":
            raise ValueError("circuit_momentum requires the RLC controller")
        for group in self.feature_groups:
            if not group or min(group) < 0 or len(set(group)) != len(group):
                raise ValueError("feature groups must contain unique nonnegative feature indices")
        if self.interaction_groups and self.aggregation != "additive":
            raise ValueError("hard interaction groups require additive aggregation; attention renormalization would reintroduce interactions")
        for value in self.feature_penalties:
            positive("feature penalty", value, zero=True)
        for output, feature, sign in self.monotonicity:
            if output < 0 or feature < 0 or sign not in (-1, 1):
                raise ValueError("monotonicity entries must be (nonnegative output, feature, +/-1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ForestConfig":
        return cls(**value)