# Research evidence ledger

This file separates established evidence from active experiments. Every headline quantitative claim should point to a source SHA, GitHub Actions run, job, and artifact where available.

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
- Learned versus canonical MLP anchor: approximately ΔNLL `-0.00304`, ΔAUC `+0.00445`
- Fresh shadow audit opened: **no**

This establishes that held-out differentiable learning of the five residual scales improved over an otherwise matched fixed-scale adapter on the development protocol. It is not by itself a final unseen-test claim.

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

The predeclared replication criterion in `research/higgs_adapter_replication_protocol.md` was satisfied: all three independent seeds improved ranking NLL. Across these three replications, mean learned-minus-fixed ranking delta is approximately ΔNLL `-0.000230371` and ΔAUC `+0.000293058`.

## HIGGS scale-source causal control

### Invalid first attempt — provenance only

- Source SHA: `bba7028547e46615f0efb0429a58ddd28251f901`
- GitHub Actions run: `37073647196`
- Job: `111058769477`
- Artifact: `11256446010`
- Status: completed computationally, **invalid for the primary causal comparison**
- Fresh shadow audit opened: **no**

The train-scale arm restored an epoch-1 checkpoint from before scale updates were permitted. Its retained scale coefficients therefore remained at initialization. The apparent heldout-vs-train result from this run must not be cited as evidence.

### Corrected matched control

- Source SHA: `70e21908c6fa63405dbc718c7d87a095ce4fbde3`
- GitHub Actions run: `37220316023`
- Job: `111489148610`
- Artifact: `11310199732` (`higgs-differentiable-adapter`)
- Status: completed successfully
- Fresh shadow audit opened: **no**
- Scale updates: `122` for TRAIN-scale and heldout-scale
- Scale-update examples: `499,712` for each trainable-scale arm
- Scale-source pools: fixed `100,000`-row TRAIN and `100,000`-row ARCHITECTURE-SELECTION pools
- CHECKPOINT-SELECTION: disjoint `100,000` rows
- RANKING: evaluation only

| Arm | Best epoch | CHECKPOINT-SELECTION NLL / AUC | RANKING NLL / AUC | Runtime |
|---|---:|---|---|---:|
| fixed | 2 | `0.570676625 / 0.769398323` | `0.571336907 / 0.768764188` | `189.54 s` |
| TRAIN-scale | 4 | `0.571093633 / 0.769217958` | `0.571734560 / 0.768675883` | `238.26 s` |
| heldout-scale | 4 | **`0.570032980 / 0.769936675`** | **`0.570802453 / 0.769270809`** | `235.95 s` |

Primary corrected causal contrast, heldout-scale minus TRAIN-scale:

- ranking ΔNLL: **`-0.000932107`**
- ranking ΔAUC: **`+0.000594926`**

Heldout-scale minus fixed:

- ranking ΔNLL: `-0.000534454`
- ranking ΔAUC: `+0.000506621`

Heldout-scale minus MLP anchor:

- ranking ΔNLL: `-0.002811093`
- ranking ΔAUC: `+0.003140992`

The retained TRAIN-scale vector was approximately `[0.21360, 0.15081, 0.18085, 0.13854, 0.15064]`; the retained heldout-scale vector was approximately `[0.13153, 0.09866, 0.09160, 0.04002, 0.03950]`, versus common initialization `0.1192029` per scale. This canonical control favors held-out architecture allocation, but it is one corrected causal-control seed and is not yet a replicated causal claim.

## Frozen mechanism

The implementation used by current HIGGS and transfer studies is centralized in `torchboost/adaptive/residual_adapter.py`. The inherited MLP backbone/head is frozen, each hidden layer receives one function-preserving zero-at-birth residual-tree refinement, TRAIN updates residual routing/packets, and SELECTION may update only the positive layerwise residual scales.

Refactoring this mechanism into a shared library primitive does not constitute new evidence. Any post-refactor experiment must record its own SHA and cannot inherit metrics from historical runs.

## External transfer benchmark

Protocol: `research/external_adapter_benchmark_protocol.md`.

- Source SHA: `b3c867c77ba21b033bc4d68916f82ee4f82fa099`
- GitHub Actions run: `37220258044`
- Aggregate job: `111492040882`
- Summary artifact: `11310109413` (`external-adapter-summary`)
- Status: completed successfully
- Coverage: **35/35** predeclared cells = 7 datasets × 5 seeds
- Fresh HIGGS shadow audit opened: **no**

Datasets: phoneme, bioresponse, bank-marketing, magic-telescope, default-credit, electricity, and miniboone. Seeds: `509, 733, 2027, 4099, 8191`.

### Learned adapter versus inherited MLP

Dataset-level wins: **7/7** in both NLL and AUC.

- mean dataset ΔNLL: **`-0.004773`**
- 95% dataset-bootstrap CI: `[-0.007932, -0.002308]`
- mean dataset ΔAUC: **`+0.003448`**
- 95% dataset-bootstrap CI: `[+0.001291, +0.006955]`

This is the strongest current broad transfer result: function-preserving residual-tree expansion improves the inherited neural predictor across all seven external datasets in the frozen benchmark.

### Learned adapter versus fixed-scale residual adapter

- mean dataset ΔNLL: `-0.000391`
- NLL wins: `5/7`
- 95% CI: `[-0.001099, +0.000309]`
- mean dataset ΔAUC: `+0.000042`
- AUC wins: `4/7`
- 95% CI: `[-0.000350, +0.000538]`

The cross-dataset evidence therefore supports the residual adapter more strongly than it supports held-out scale learning over fixed residual scales.

### Learned adapter versus strong tree references

The learned adapter loses all seven datasets to CatBoost, XGBoost, and LightGBM in both NLL and AUC under the frozen global-reference recipes.

Mean dataset gaps (learned adapter minus reference):

| Reference | ΔNLL | ΔAUC |
|---|---:|---:|
| CatBoost | `+0.04768` | `-0.02264` |
| XGBoost | `+0.04683` | `-0.02221` |
| LightGBM | `+0.04576` | `-0.02149` |

These are reference-baseline results, not evidence of optimized benchmark superiority. The repository makes no tabular-SOTA claim.

## Final HIGGS shadow audit

Locked range: UCI HIGGS rows `[9,600,000, 10,100,000)`.

Status: **unopened**.

The shadow audit must not be evaluated while architecture or optimization choices remain subject to change. When a final configuration is frozen, the opening must be a one-way workflow with the exact source SHA and checkpoint/configuration recorded here before metrics are added. Shadow metrics may not select a seed, checkpoint, architecture, hyperparameter, or follow-up configuration.
