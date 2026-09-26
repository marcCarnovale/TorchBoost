"""Generic, diagnostic-only thermal exposure and development dose matching.

Integrals are right-endpoint sums over accepted controller advances, with dt in
controller-time units. They are not an exact continuous-time ODE reconstruction.
Matching uses no predictive losses, labels, selection scores, or audit outcomes.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any


def thermal_exposure(history: Iterable[Mapping[str, Any]], config: Any) -> dict[str, float]:
    dt = float(config.dt)
    span = float(config.thaw_temperature - config.ambient_temperature)
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(span) or span <= 0:
        raise ValueError("positive finite dt and ambient-to-thaw span required")
    result = {
        "steps": 0, "duration": 0.0, "peak_temperature": float(config.ambient_temperature),
        "peak_thaw_units": 0.0, "mean_thaw_exposure": 0.0, "energy_exposure": 0.0,
        "above_thaw_exposure": 0.0, "external_heat": 0.0, "resistor_heat": 0.0,
        "source_work": 0.0, "cooling_energy": 0.0, "vented_energy": 0.0,
        "max_energy_error": 0.0,
    }
    previous = -math.inf
    for row in history:
        if row["step"] <= previous:
            raise ValueError("thermal history must contain strictly increasing accepted steps")
        previous = row["step"]
        nodes = list(row["nodes"].values())
        if not nodes:
            raise ValueError("thermal observation must have live nodes")
        capacity = sum(float(n["capacity"]) for n in nodes)
        if not math.isfinite(capacity) or capacity <= 0:
            raise ValueError("positive finite live heat capacity required")
        excursions = [(float(n["temperature"]) - config.ambient_temperature) / span for n in nodes]
        if not all(math.isfinite(v) for v in excursions):
            raise ValueError("finite temperatures required")
        mean = sum(float(n["capacity"]) * max(0.0, e) for n, e in zip(nodes, excursions)) / capacity
        above = sum(float(n["capacity"]) * max(0.0, e - 1.0) for n, e in zip(nodes, excursions)) / capacity
        result["steps"] += 1
        result["duration"] += dt
        result["peak_temperature"] = max(result["peak_temperature"], max(float(n["temperature"]) for n in nodes))
        result["peak_thaw_units"] = max(result["peak_thaw_units"], max(excursions))
        result["mean_thaw_exposure"] += dt * mean
        result["above_thaw_exposure"] += dt * above
        result["energy_exposure"] += dt * float(row["thermal_after"])
        for field in ("external_heat", "resistor_heat", "source_work", "cooling_energy", "vented_energy"):
            result[field] += float(row.get(field, 0.0))
        result["max_energy_error"] = max(result["max_energy_error"], abs(float(row["energy_error"])))
    return result


def exposure_distance(candidate: Mapping[str, float], target: Mapping[str, float]) -> float:
    """Squared log-ratio distance for peak excursion AND integrated exposure."""
    fields = ("peak_thaw_units", "mean_thaw_exposure")
    values = [(float(candidate[k]), float(target[k])) for k in fields]
    if any(not math.isfinite(x) or x <= 0 for pair in values for x in pair):
        return math.inf
    return sum(math.log(value / reference) ** 2 for value, reference in values)


def exposure_matched(candidate: Mapping[str, float], target: Mapping[str, float], tolerance: float = 0.20) -> bool:
    if not 0.0 < tolerance < 1.0:
        raise ValueError("relative matching tolerance must lie in (0, 1)")
    fields = ("peak_thaw_units", "mean_thaw_exposure")
    return all(
        math.isfinite(float(candidate[k])) and math.isfinite(float(target[k]))
        and target[k] > 0 and abs(candidate[k] / target[k] - 1.0) <= tolerance
        for k in fields
    )
