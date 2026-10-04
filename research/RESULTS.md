# Research evidence ledger

This file separates established evidence from active experiments. It is intended
to be auditable: every quantitative claim should point to a source SHA, Actions
run, job, and artifact where available.

## HIGGS differentiable residual adapter

### Development seed

- Source SHA: `265a8e040a060080d198bb5fce5578a1ccc1825f`
- GitHub Actions run: `36525742176`
- Job: `109268282950`
- Artifact: `11014458749` (`higgs-differentiable-adapter`)
- Status: completed successfully
- Ranking fixed-scale adapter: NLL `0.57106812`, AUC `0.76946808`
- Ranking learned-scale adapter: NLL `0.57049915`, AUC `0.76991939`
- Learned minus fixed: ΔNLL `-0.00056897`, ΔAUC `+0.00045131`
- Learned versus canonical MLP anchor: approximately ΔNLL `-0.00304`,
  ΔAUC `+0.00445`
- Fresh shadow audit opened: **no**

This establishes that held-out differentiable learning of the five residual
scales improved over an otherwise matched fixed-scale adapter on the
development protocol. It is not by itself a final unseen-test claim.

### Predeclared replication

- Source SHA: `f18ffe936825ff4dea65d75805dc90cfecf6c5bf`
- GitHub Actions run: `36576510711`
- Status: all three predeclared jobs completed successfully
- Fresh shadow audit opened: **no**

| Seed | Job | Artifact | learned − fixed ranking ΔNLL | learned − fixed ranking ΔAUC |
|---:|---:|---:|---:|---:|
| 733 | `109433405368` | `11039690046` | `-0.0001071258` | `+0.0001569420` |
| 2027 | `109433405526` | `11038735390` | `-0.0002529960` | `+0.0003390280` |
| 4099 | `109433405076` | `11038164840` | `-0.0003309902` | `+0.0003832027` |

The predeclared replication criterion in
`research/higgs_adapter_replication_protocol.md` was satisfied: all three
independent seeds improved ranking NLL, so the mean direction is favorable and
the 2-of-3 requirement is exceeded. Across the three replications, mean learned
minus fixed ranking delta is approximately ΔNLL `-0.000230371` and ΔAUC
`+0.000293058`. The larger per-seed improvements versus the MLP anchor are a
different comparison and are not reported in this table.

## HIGGS scale-source causal control

### Invalid first attempt — retained for provenance, not evidence

- Source SHA: `bba7028547e46615f0efb0429a58ddd28251f901`
- GitHub Actions run: `37073647196`
- Job: `111058769477`
- Artifact: `11256446010`
- Status: completed computationally, **invalid for the primary causal comparison**
- Fresh shadow audit opened: **no**

The train-scale arm executed scheduled scale updates, but checkpoint selection
restored epoch 1, which occurred before the warmup permitted any scale update.
Its retained scale coefficients therefore remained at their initialization.
The apparent heldout-vs-train-scale ranking difference from this run must not be
cited as evidence that held-out architecture learning beats TRAIN-updated
scales.

The corrected control requires all trainable-scale checkpoints to occur after
at least one scale update and splits the development SELECTION block into
disjoint architecture-update and checkpoint-selection halves. See
`research/higgs_scale_source_control_protocol.md`.

## Frozen mechanism

The implementation used by current HIGGS and transfer studies is centralized
in `torchboost/adaptive/residual_adapter.py`. The inherited MLP backbone/head
is frozen, each hidden layer receives one function-preserving zero-at-birth
residual-tree refinement, TRAIN updates residual routing/packets, and SELECTION
may update only the positive layerwise residual scales.

Refactoring this mechanism into a shared library primitive does not constitute
new evidence. Any post-refactor experiment must record its own SHA and cannot
inherit metrics from the historical runs above.

## External transfer benchmark

Protocol: `research/external_adapter_benchmark_protocol.md`.

The v2 benchmark is frozen before reading its results. It evaluates seven
public OpenML binary datasets across five seeds, with train-only preprocessing,
paired train/selection/ranking splits, and fixed global comparator recipes for
CatBoost, XGBoost, and LightGBM. The workflow emits per-cell records and an
aggregate paired report.

Until that matrix completes, there is **no recorded broad cross-dataset
superiority claim**.

## Final HIGGS shadow audit

Locked range: UCI HIGGS rows `[9,600,000, 10,100,000)`.

Status: **unopened**.

The shadow audit must not be evaluated while architecture or optimization
choices remain subject to change. When a configuration is frozen for final
evaluation, the opening must be a one-way workflow with the exact source SHA
and checkpoint/configuration recorded here before metrics are added.
