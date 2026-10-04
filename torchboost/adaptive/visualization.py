"""Read-only visualizations; no controller or training-clock transitions."""
from __future__ import annotations
from copy import deepcopy


def visualize_physical_state(physical_history, *, plastic_history=(), training_history=(),
                             structural_events=(), variables=None):
    """Return independent matplotlib figures, one metric per figure.

    Inputs are history snapshots, not live model/controller objects. Ragged node
    lifetimes remain gaps in their own series. The caller owns display/saving.
    No explicit colors or plotting style is imposed.
    """
    import matplotlib.pyplot as plt
    rows = deepcopy(list(physical_history))
    plastic = deepcopy(list(plastic_history))
    training = deepcopy(list(training_history))
    events = deepcopy(list(structural_events))
    wanted = set(variables or ("charge", "current", "power", "heat", "temperature",
                              "hardness", "integrity", "evidence", "reference_path", "momentum", "nodes"))
    figures = {}
    if rows and "charge" in wanted:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r["step"] for r in rows], [r["charge"] for r in rows])
        ax.set(xlabel="Control step", ylabel="Charge", title="Stored capacitor charge")
        fig.tight_layout(); figures["charge"] = fig
    for variable in ("current", "power", "heat", "temperature"):
        if variable not in wanted or not rows:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        keys = sorted({k for r in rows for k in r["nodes"]})
        for key in keys:
            subset = [r for r in rows if key in r["nodes"]]
            ax.plot([r["step"] for r in subset], [r["nodes"][key][variable] for r in subset], label=key)
        ax.set(xlabel="Control step", ylabel=variable.replace('_', ' ').capitalize(),
               title=f"Per-node {variable}")
        if len(keys) <= 12: ax.legend(title="Stable node ID", fontsize="small")
        fig.tight_layout(); figures[variable] = fig
    for variable in ("hardness", "integrity", "evidence", "reference_path"):
        if variable not in wanted or not plastic:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        keys = sorted({r["node_id"] for r in plastic})
        for key in keys:
            subset = [r for r in plastic if r["node_id"] == key]
            ax.plot([r["step"] for r in subset], [r[variable] for r in subset], label=key)
        ax.set(xlabel="Control step", ylabel=variable.replace('_', ' ').capitalize(),
               title=f"Per-node {variable.replace('_', ' ')}")
        if len(keys) <= 12: ax.legend(title="Stable node ID", fontsize="small")
        fig.tight_layout(); figures[variable] = fig
    if training and "nodes" in wanted:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r["epoch"] for r in training], [r["nodes"] for r in training])
        for event in events:
            if event["event"] in ("grow", "prune", "remove_tree"):
                ax.axvline(event["step"], alpha=.15)
        ax.set(xlabel="Epoch", ylabel="Live nodes", title="Real structural allocation")
        fig.tight_layout(); figures["nodes"] = fig
    if training and "momentum" in wanted:
        fig, ax = plt.subplots(figsize=(7, 4))
        owners = sorted({k for r in training for k, v in r.get("momentum", {}).items() if v is not None})
        for owner in owners:
            subset = [r for r in training if r.get("momentum", {}).get(owner) is not None]
            ax.plot([r["epoch"] for r in subset], [r["momentum"][owner] for r in subset], label=owner)
        ax.set(xlabel="Epoch", ylabel="EMA coefficient", title="Local optimizer persistence")
        if 0 < len(owners) <= 12: ax.legend(fontsize="small")
        fig.tight_layout(); figures["momentum"] = fig
    return figures
