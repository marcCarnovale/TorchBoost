"""Aggregate frozen external-adapter benchmark records.

The dataset is the primary inferential unit. Repeated seeds quantify
within-dataset variability and are never treated as independent benchmark tasks.
Headline intervals bootstrap dataset-level means.
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
        if (
            obj.get("protocol_version") == "external-adapter-v2-frozen"
            and obj.get("status") == "completed"
        ):
            obj["_path"] = str(path)
            rows.append(obj)
    if not rows:
        raise SystemExit("no completed external-adapter-v2-frozen records found")
    return rows


def interval(values, seed=20261002, draws=20000):
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    median = float(np.median(x))
    if len(x) == 1:
        return {"mean": mean, "median": median, "se": None, "ci95": [mean, mean], "n": 1}
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(draws, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "mean": mean,
        "median": median,
        "se": se,
        "ci95": [float(lo), float(hi)],
        "n": int(len(x)),
    }


def grouped(rows):
    out = {}
    for row in rows:
        out.setdefault(row["dataset"], []).append(row)
    return out


def dataset_means(rows, value_fn):
    return {
        dataset: float(np.mean([value_fn(row) for row in subset]))
        for dataset, subset in grouped(rows).items()
    }


def wins_ties_losses(values, favorable):
    eps = 1e-12
    if favorable == "negative":
        return {
            "wins": int(sum(v < -eps for v in values)),
            "ties": int(sum(abs(v) <= eps for v in values)),
            "losses": int(sum(v > eps for v in values)),
        }
    return {
        "wins": int(sum(v > eps for v in values)),
        "ties": int(sum(abs(v) <= eps for v in values)),
        "losses": int(sum(v < -eps for v in values)),
    }


def summarize(rows):
    datasets = sorted({r["dataset"] for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    expected = {(d, s) for d in datasets for s in seeds}
    observed = {(r["dataset"], int(r["seed"])) for r in rows}
    missing = sorted(expected - observed)

    absolute = {}
    for model in MODELS:
        nll_by_dataset = dataset_means(rows, lambda r, m=model: r[m]["ranking"]["nll"])
        auc_by_dataset = dataset_means(rows, lambda r, m=model: r[m]["ranking"]["auc"])
        absolute[model] = {
            "ranking_nll_across_datasets": interval(list(nll_by_dataset.values())),
            "ranking_auc_across_datasets": interval(list(auc_by_dataset.values())),
            "per_dataset_mean_nll": nll_by_dataset,
            "per_dataset_mean_auc": auc_by_dataset,
        }

    paired = {}
    for base in COMPARATORS:
        dnll_by_dataset = dataset_means(
            rows,
            lambda r, b=base: (
                r["learned_adapter"]["ranking"]["nll"] - r[b]["ranking"]["nll"]
            ),
        )
        dauc_by_dataset = dataset_means(
            rows,
            lambda r, b=base: (
                r["learned_adapter"]["ranking"]["auc"] - r[b]["ranking"]["auc"]
            ),
        )
        dnll_cells = [
            r["learned_adapter"]["ranking"]["nll"] - r[base]["ranking"]["nll"]
            for r in rows
        ]
        dauc_cells = [
            r["learned_adapter"]["ranking"]["auc"] - r[base]["ranking"]["auc"]
            for r in rows
        ]
        paired[base] = {
            "dataset_clustered_nll_delta": interval(list(dnll_by_dataset.values())),
            "dataset_clustered_auc_delta": interval(list(dauc_by_dataset.values())),
            "dataset_nll_wins_ties_losses": wins_ties_losses(
                list(dnll_by_dataset.values()), "negative"
            ),
            "dataset_auc_wins_ties_losses": wins_ties_losses(
                list(dauc_by_dataset.values()), "positive"
            ),
            "cell_level_seed_variation": {
                "nll_delta": interval(dnll_cells),
                "auc_delta": interval(dauc_cells),
                "nll_wins_ties_losses": wins_ties_losses(dnll_cells, "negative"),
                "auc_wins_ties_losses": wins_ties_losses(dauc_cells, "positive"),
            },
        }

    per_dataset = {}
    for dataset, subset in grouped(rows).items():
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
                "seed_level_nll_delta": interval(dnll, seed=20261002 + len(dataset)),
                "seed_level_auc_delta": interval(dauc, seed=20262002 + len(dataset)),
            }

    return {
        "protocol_version": "external-adapter-v2-frozen",
        "inferential_unit": "dataset",
        "records": len(rows),
        "datasets": datasets,
        "seeds": seeds,
        "missing_dataset_seed_pairs": missing,
        "absolute": absolute,
        "learned_adapter_paired_comparisons": paired,
        "per_dataset": per_dataset,
        "source_shas": sorted(
            {r.get("source_sha") for r in rows if r.get("source_sha")}
        ),
        "higgs_shadow_audit_opened": False,
    }


def fmt(x):
    return "NA" if x is None else f"{x:.6f}"


def markdown(summary):
    lines = [
        "# External residual-adapter benchmark summary",
        "",
        f"Protocol: `{summary['protocol_version']}`.",
        f"Completed records: **{summary['records']}** across "
        f"**{len(summary['datasets'])} datasets** and seeds {summary['seeds']}.",
        "",
        "**Primary inferential unit: dataset.** Seed replicates are averaged within "
        "each dataset before cross-dataset uncertainty is computed.",
        "",
        "The HIGGS shadow audit remains unopened.",
        "",
        "## Dataset-clustered paired comparison of learned adapter",
        "",
        "| Comparator | mean dataset ΔNLL | median ΔNLL | 95% dataset-bootstrap CI | datasets W/T/L | mean dataset ΔAUC | median ΔAUC | 95% dataset-bootstrap CI | datasets W/T/L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for base, row in summary["learned_adapter_paired_comparisons"].items():
        n = row["dataset_clustered_nll_delta"]
        a = row["dataset_clustered_auc_delta"]
        nw = row["dataset_nll_wins_ties_losses"]
        aw = row["dataset_auc_wins_ties_losses"]
        lines.append(
            f"| {base} | {fmt(n['mean'])} | {fmt(n['median'])} | "
            f"[{fmt(n['ci95'][0])}, {fmt(n['ci95'][1])}] | "
            f"{nw['wins']}/{nw['ties']}/{nw['losses']} | "
            f"{fmt(a['mean'])} | {fmt(a['median'])} | "
            f"[{fmt(a['ci95'][0])}, {fmt(a['ci95'][1])}] | "
            f"{aw['wins']}/{aw['ties']}/{aw['losses']} |"
        )

    lines += [
        "",
        "Negative ΔNLL and positive ΔAUC favor the learned adapter.",
        "",
        "## Absolute metrics",
        "",
        "Absolute NLL/AUC are shown only as descriptive averages of per-dataset "
        "means; heterogeneous tasks should not be interpreted as one pooled test set.",
        "",
        "| Model | mean of dataset NLL means | mean of dataset AUC means |",
        "|---|---:|---:|",
    ]
    for model, row in summary["absolute"].items():
        lines.append(
            f"| {model} | "
            f"{fmt(row['ranking_nll_across_datasets']['mean'])} | "
            f"{fmt(row['ranking_auc_across_datasets']['mean'])} |"
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
        "- The dataset, not a seed replicate, is the primary cross-task sampling unit.",
        "- Seed-level intervals describe optimization/split variability within each dataset.",
        "- Architecture and optimizer rules are frozen globally rather than tuned per dataset.",
        "- Fixed global tree recipes are reference baselines, not claims of optimally tuned CatBoost/XGBoost/LightGBM.",
        "- A favorable grand mean with concentrated dataset losses must be reported as mixed transfer rather than broad superiority.",
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
    Path(a.json_out).write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    )
    Path(a.md_out).write_text(markdown(summary))
    print(markdown(summary))


if __name__ == "__main__":
    main()
