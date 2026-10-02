"""Aggregate frozen external-adapter benchmark records.

Consumes per-dataset/per-seed JSON records and emits:
- machine-readable summary.json;
- human-readable SUMMARY.md;
- paired mean deltas, standard errors, deterministic bootstrap 95% intervals;
- win/tie/loss counts against every baseline.

No model selection is performed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODELS = ["mlp", "fixed_adapter", "learned_adapter", "catboost", "xgboost", "lightgbm"]
COMPARATORS = ["mlp", "fixed_adapter", "catboost", "xgboost", "lightgbm"]


def load_records(root: Path):
    rows = []
    for path in sorted(root.rglob("*.json")):
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        if obj.get("protocol_version") == "external-adapter-v2-frozen" and obj.get("status") == "completed":
            obj["_path"] = str(path)
            rows.append(obj)
    if not rows:
        raise SystemExit("no completed external-adapter-v2-frozen records found")
    return rows


def ci(values, seed=20261002, draws=20000):
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    if len(x) == 1:
        return {"mean": mean, "se": None, "ci95": [mean, mean]}
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(draws, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": mean, "se": se, "ci95": [float(lo), float(hi)]}


def summarize(rows):
    datasets = sorted({r["dataset"] for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    expected = {(d, s) for d in datasets for s in seeds}
    observed = {(r["dataset"], int(r["seed"])) for r in rows}
    missing = sorted(expected - observed)

    absolute = {}
    for model in MODELS:
        absolute[model] = {
            "ranking_nll": ci([r[model]["ranking"]["nll"] for r in rows]),
            "ranking_auc": ci([r[model]["ranking"]["auc"] for r in rows]),
        }

    paired = {}
    for base in COMPARATORS:
        dnll = [
            r["learned_adapter"]["ranking"]["nll"] - r[base]["ranking"]["nll"]
            for r in rows
        ]
        dauc = [
            r["learned_adapter"]["ranking"]["auc"] - r[base]["ranking"]["auc"]
            for r in rows
        ]
        paired[base] = {
            "nll_delta": ci(dnll),
            "auc_delta": ci(dauc),
            "nll_wins_ties_losses": {
                "wins": int(sum(v < -1e-12 for v in dnll)),
                "ties": int(sum(abs(v) <= 1e-12 for v in dnll)),
                "losses": int(sum(v > 1e-12 for v in dnll)),
            },
            "auc_wins_ties_losses": {
                "wins": int(sum(v > 1e-12 for v in dauc)),
                "ties": int(sum(abs(v) <= 1e-12 for v in dauc)),
                "losses": int(sum(v < -1e-12 for v in dauc)),
            },
        }

    per_dataset = {}
    for dataset in datasets:
        subset = [r for r in rows if r["dataset"] == dataset]
        per_dataset[dataset] = {}
        for base in COMPARATORS:
            dnll = [
                r["learned_adapter"]["ranking"]["nll"] - r[base]["ranking"]["nll"]
                for r in subset
            ]
            dauc = [
                r["learned_adapter"]["ranking"]["auc"] - r[base]["ranking"]["auc"]
                for r in subset
            ]
            per_dataset[dataset][base] = {
                "nll_delta": ci(dnll, seed=20261002 + len(dataset)),
                "auc_delta": ci(dauc, seed=20262002 + len(dataset)),
            }

    return {
        "protocol_version": "external-adapter-v2-frozen",
        "records": len(rows),
        "datasets": datasets,
        "seeds": seeds,
        "missing_dataset_seed_pairs": missing,
        "absolute": absolute,
        "learned_adapter_paired_comparisons": paired,
        "per_dataset": per_dataset,
        "source_shas": sorted({r.get("source_sha") for r in rows if r.get("source_sha")}),
        "higgs_shadow_audit_opened": False,
    }


def fmt(x):
    return "NA" if x is None else f"{x:.6f}"


def markdown(summary):
    lines = [
        "# External residual-adapter benchmark summary",
        "",
        f"Protocol: `{summary['protocol_version']}`.",
        f"Completed records: **{summary['records']}** across **{len(summary['datasets'])} datasets** and seeds {summary['seeds']}.",
        "",
        "The HIGGS shadow audit remains unopened. This benchmark uses only external public datasets.",
        "",
        "## Paired comparison of learned adapter",
        "",
        "| Comparator | mean ΔNLL | 95% bootstrap CI | NLL W/T/L | mean ΔAUC | 95% bootstrap CI | AUC W/T/L |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for base, row in summary["learned_adapter_paired_comparisons"].items():
        n = row["nll_delta"]
        a = row["auc_delta"]
        nw = row["nll_wins_ties_losses"]
        aw = row["auc_wins_ties_losses"]
        lines.append(
            f"| {base} | {fmt(n['mean'])} | [{fmt(n['ci95'][0])}, {fmt(n['ci95'][1])}] | "
            f"{nw['wins']}/{nw['ties']}/{nw['losses']} | {fmt(a['mean'])} | "
            f"[{fmt(a['ci95'][0])}, {fmt(a['ci95'][1])}] | {aw['wins']}/{aw['ties']}/{aw['losses']} |"
        )
    lines += [
        "",
        "Negative ΔNLL and positive ΔAUC favor the learned adapter.",
        "",
        "## Absolute ranking metrics",
        "",
        "| Model | mean NLL | mean AUC |",
        "|---|---:|---:|",
    ]
    for model, row in summary["absolute"].items():
        lines.append(
            f"| {model} | {fmt(row['ranking_nll']['mean'])} | {fmt(row['ranking_auc']['mean'])} |"
        )
    if summary["missing_dataset_seed_pairs"]:
        lines += [
            "",
            "## Incomplete cells",
            "",
            "The following dataset/seed cells are missing and must not be silently ignored:",
            "",
            "```json",
            json.dumps(summary["missing_dataset_seed_pairs"], indent=2),
            "```",
        ]
    lines += [
        "",
        "## Interpretation contract",
        "",
        "- These are repeated holdout comparisons, not a hyperparameter leaderboard.",
        "- Architecture and optimizer rules are frozen globally rather than tuned per dataset.",
        "- The learned adapter is compared pairwise against the same split/seed baselines.",
        "- Bootstrap intervals summarize dataset-seed variability; they are not claims of population-level statistical significance.",
        "- A strong HIGGS result plus mixed external results should be reported as mixed external transfer, not broad superiority.",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--json-out", required=True)
    p.add_argument("--md-out", required=True)
    a = p.parse_args()
    summary = summarize(load_records(Path(a.root)))
    Path(a.json_out).write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    Path(a.md_out).write_text(markdown(summary))
    print(markdown(summary))


if __name__ == "__main__":
    main()
