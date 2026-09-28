"""Evidence-driven navigation of TorchBoost's architecture space.

This module deliberately does not encode a one-dimensional interpolation
between a tree model and an MLP.  It provides small, testable primitives for
tracking multi-coordinate architecture states, matched controls, and
successive resource allocation during training.

No audit metrics belong here.  Search decisions are made from selection and
ranking evidence supplied by an experiment harness.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable
import math


@dataclass(frozen=True)
class ArchitectureCoordinates:
    """A compact description of one region of TorchBoost architecture space."""

    composition: str = "additive"
    tree_depth: int = 0
    tree_count: int = 0
    layer_count: int = 0
    hidden_width: int = 0
    routing_hardness: str = "hard"
    routing_geometry: str = "axis"
    packet_type: str = "scalar"
    construction: str = "gradient"
    aggregation: str = "additive"
    optimization_scope: str = "full"
    proposal_subsample: float = 1.0
    refinement_subsample: float = 1.0
    global_polish: bool = False

    def __post_init__(self) -> None:
        if self.tree_depth < 0 or self.tree_count < 0 or self.layer_count < 0:
            raise ValueError("architecture sizes must be nonnegative")
        if self.hidden_width < 0:
            raise ValueError("hidden_width must be nonnegative")
        for name in ("proposal_subsample", "refinement_subsample"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")

    def mutate(self, **changes: Any) -> "ArchitectureCoordinates":
        """Return a new immutable state with named coordinate changes."""
        return replace(self, **changes)


@dataclass(frozen=True)
class TrainingBudget:
    """Matched compute/exposure budget for one architecture arm."""

    optimizer_updates: int
    examples_seen: int
    full_data_passes: float
    wall_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.optimizer_updates < 0 or self.examples_seen < 0:
            raise ValueError("budget counts must be nonnegative")
        if not math.isfinite(self.full_data_passes) or self.full_data_passes < 0:
            raise ValueError("full_data_passes must be nonnegative")
        if self.wall_seconds is not None and (
            not math.isfinite(self.wall_seconds) or self.wall_seconds < 0
        ):
            raise ValueError("wall_seconds must be nonnegative")


@dataclass(frozen=True)
class ArchitectureEvidence:
    """Held-out evidence for an architecture checkpoint.

    Selection NLL is the admission signal. Ranking NLL/AUC are independent
    architecture-comparison signals. Training loss is intentionally absent.
    """

    selection_nll: float
    ranking_nll: float
    ranking_auc: float
    trainable_parameters: int
    budget: TrainingBudget
    peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("selection_nll", "ranking_nll", "ranking_auc"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.trainable_parameters < 0:
            raise ValueError("trainable_parameters must be nonnegative")
        if self.peak_memory_bytes is not None and self.peak_memory_bytes < 0:
            raise ValueError("peak_memory_bytes must be nonnegative")


@dataclass
class ArchitectureCandidate:
    """One checkpoint in the adaptive search graph."""

    candidate_id: str
    coordinates: ArchitectureCoordinates
    parent_id: str | None = None
    mutation: str = "anchor"
    evidence: ArchitectureEvidence | None = None
    admitted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdmissionDecision:
    candidate_id: str
    parent_id: str
    admitted: bool
    selection_delta: float
    ranking_nll_delta: float
    ranking_auc_delta: float
    reason: str


class ArchitectureArchive:
    """Archive and admission logic for evidence-driven architecture training.

    Every mutation is compared with a matched continuation of its parent under
    the same budget.  A mutation cannot be admitted merely because training
    loss improved.
    """

    def __init__(
        self,
        *,
        selection_tolerance: float = 0.0,
        ranking_nll_tolerance: float = 0.0,
        ranking_auc_tolerance: float = 0.0,
    ) -> None:
        self.selection_tolerance = float(selection_tolerance)
        self.ranking_nll_tolerance = float(ranking_nll_tolerance)
        self.ranking_auc_tolerance = float(ranking_auc_tolerance)
        self.candidates: dict[str, ArchitectureCandidate] = {}
        self.decisions: list[AdmissionDecision] = []

    def add(self, candidate: ArchitectureCandidate) -> None:
        if candidate.candidate_id in self.candidates:
            raise ValueError(f"duplicate candidate_id {candidate.candidate_id!r}")
        if candidate.parent_id is not None and candidate.parent_id not in self.candidates:
            raise ValueError("parent must be added before child")
        self.candidates[candidate.candidate_id] = candidate

    @staticmethod
    def _same_budget(a: TrainingBudget, b: TrainingBudget) -> bool:
        return (
            a.optimizer_updates == b.optimizer_updates
            and a.examples_seen == b.examples_seen
            and math.isclose(a.full_data_passes, b.full_data_passes, rel_tol=0, abs_tol=1e-12)
        )

    def compare_to_control(
        self,
        candidate_id: str,
        control_id: str,
    ) -> AdmissionDecision:
        cand = self.candidates[candidate_id]
        control = self.candidates[control_id]
        if cand.parent_id is None:
            raise ValueError("anchor candidates do not require admission")
        if cand.parent_id != control.parent_id:
            raise ValueError("candidate and control must share a parent")
        if cand.evidence is None or control.evidence is None:
            raise ValueError("both candidate and control need evidence")
        if not self._same_budget(cand.evidence.budget, control.evidence.budget):
            raise ValueError("candidate and control budgets are not matched")

        selection_delta = cand.evidence.selection_nll - control.evidence.selection_nll
        ranking_nll_delta = cand.evidence.ranking_nll - control.evidence.ranking_nll
        ranking_auc_delta = cand.evidence.ranking_auc - control.evidence.ranking_auc

        selection_ok = selection_delta <= self.selection_tolerance
        ranking_nll_ok = ranking_nll_delta <= self.ranking_nll_tolerance
        ranking_auc_ok = ranking_auc_delta >= -self.ranking_auc_tolerance

        # Require non-regression on all held-out signals and a strict gain on
        # at least one. This prevents "admission by noise-equivalent tie".
        strict_gain = (
            selection_delta < -self.selection_tolerance
            or ranking_nll_delta < -self.ranking_nll_tolerance
            or ranking_auc_delta > self.ranking_auc_tolerance
        )
        admitted = selection_ok and ranking_nll_ok and ranking_auc_ok and strict_gain
        reason = (
            "held-out improvement under matched budget"
            if admitted
            else "no matched held-out improvement"
        )
        cand.admitted = admitted
        decision = AdmissionDecision(
            candidate_id=candidate_id,
            parent_id=cand.parent_id,
            admitted=admitted,
            selection_delta=selection_delta,
            ranking_nll_delta=ranking_nll_delta,
            ranking_auc_delta=ranking_auc_delta,
            reason=reason,
        )
        self.decisions.append(decision)
        return decision

    def non_dominated(self, candidate_ids: Iterable[str] | None = None) -> list[str]:
        """Return the predictive/efficiency Pareto archive.

        Lower ranking NLL, higher ranking AUC, fewer parameters and fewer
        examples seen are preferred. Selection NLL remains an admission gate,
        not an extra objective here.
        """
        ids = list(self.candidates if candidate_ids is None else candidate_ids)
        ids = [i for i in ids if self.candidates[i].evidence is not None]
        result: list[str] = []
        for i in ids:
            a = self.candidates[i].evidence
            assert a is not None
            dominated = False
            for j in ids:
                if i == j:
                    continue
                b = self.candidates[j].evidence
                assert b is not None
                weak = (
                    b.ranking_nll <= a.ranking_nll
                    and b.ranking_auc >= a.ranking_auc
                    and b.trainable_parameters <= a.trainable_parameters
                    and b.budget.examples_seen <= a.budget.examples_seen
                )
                strict = (
                    b.ranking_nll < a.ranking_nll
                    or b.ranking_auc > a.ranking_auc
                    or b.trainable_parameters < a.trainable_parameters
                    or b.budget.examples_seen < a.budget.examples_seen
                )
                if weak and strict:
                    dominated = True
                    break
            if not dominated:
                result.append(i)
        return result


class SuccessiveHalvingAllocator:
    """Allocate later training rounds to candidates that earn evidence.

    This object is intentionally agnostic to the model implementation. An
    experiment harness trains each active candidate for the returned budget,
    attaches ArchitectureEvidence, and asks the allocator which candidates
    survive to the next round.
    """

    def __init__(self, *, keep_fraction: float = 0.5, min_survivors: int = 1):
        if not 0 < keep_fraction <= 1:
            raise ValueError("keep_fraction must lie in (0, 1]")
        if min_survivors < 1:
            raise ValueError("min_survivors must be positive")
        self.keep_fraction = float(keep_fraction)
        self.min_survivors = int(min_survivors)

    def survivors(
        self,
        candidates: Iterable[ArchitectureCandidate],
    ) -> list[str]:
        ready = [c for c in candidates if c.evidence is not None]
        if not ready:
            return []
        # Selection is primary; ranking NLL/AUC break close/tied selection.
        ready.sort(
            key=lambda c: (
                c.evidence.selection_nll,
                c.evidence.ranking_nll,
                -c.evidence.ranking_auc,
                c.evidence.trainable_parameters,
            )
        )
        keep = max(self.min_survivors, math.ceil(len(ready) * self.keep_fraction))
        return [c.candidate_id for c in ready[:keep]]
