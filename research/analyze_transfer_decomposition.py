"""Secondary transfer decomposition for completed external benchmark artifacts.

This is intentionally separate from the frozen primary benchmark aggregator.
It decomposes transfer into:
1. fixed residual-tree adapter minus MLP anchor;
2. learned-scale adapter minus fixed adapter.

Dataset means are the cross-task inferential units.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_records(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.rglob("*.json")):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            row.get("protocol_version") == "external-adapter-v2-frozen"
            and row.get("status") == "completed"
        ):
            records.append(row)
    if not records:
        raise SystemExit("no completed external-adapter-v2-frozen records found")
    return records


def clustered(records: list[dict], numerator: str, denominator: str, metric: str) -> dict:
    grouped: dict[str, list[float]] = {}
    for row in records:
        value = (
            row[numerator]["ranking"][metric]
            - row[denominator]["ranking"][metric]
        )
        grouped.setdefault(row["dataset"], []).append(float(value))

    dataset_means = {
        dataset: float(np.mean(values))
        for dataset, values in sorted(grouped.items())
    }
    values = np.asarray(list(dataset_means.values()), dtype=float)
    favorable = values < 0 if metric == "nll" else values > 0
    return {
        "per_dataset_mean": dataset_means,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "wins": int(np.sum(favorable)),
        "losses": int(np.sum(~favorable)),
        "datasets": len(values),
    }


def summarize(records: list[dict]) -> dict:
    return {
        "status": "secondary_post_hoc_decomposition",
        "records": len(records),
        "datasets": sorted({row["dataset"] for row in records}),
        "fixed_adapter_minus_mlp": {
            "nll": clustered(records, "fixed_adapter", "mlp", "nll"),
            "auc": clustered(records, "fixed_adapter", "mlp", "auc"),
        },
        "learned_adapter_minus_fixed": {
            "nll": clustered(records, "learned_adapter", "fixed_adapter", "nll"),
            "auc": clustered(records, "learned_adapter", "fixed_adapter", "auc"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = summarize(load_records(Path(args.root)))
    Path(args.out).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
