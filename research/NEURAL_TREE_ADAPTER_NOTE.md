# Function-Preserving Neural–Tree Expansion for Tabular Learning

## Abstract

TorchBoost studies a narrow architecture question: can a trained tabular neural network be expanded into a richer tree-structured model without damaging the inherited predictor, and can held-out evidence govern how much newborn structural capacity is used?

Each dense hidden layer of a ReLU MLP is represented exactly as a depth-zero affine tree layer. A hidden layer can then grow a zero-at-birth residual-tree refinement, preserving the represented function at the instant of expansion. The inherited MLP backbone and output head remain frozen while only newborn routing and residual packets are optimized.

On HIGGS, held-out learning of five positive residual architecture scales beats a matched fixed-scale adapter, with the direction reproduced in all three predeclared independent seeds. A corrected matched scale-source control also favors held-out scale updates over TRAIN-sourced updates on the canonical seed. On a frozen seven-dataset, five-seed external benchmark, the learned residual adapter improves its inherited MLP on all seven datasets in both NLL and AUC. It does not beat the strong tree-reference models. The fresh 500k-row HIGGS shadow audit remains unopened.

## 1. Architecture

Let a dense layer be

```text
h(x) = W x + b.
```

TorchBoost embeds this exactly as a depth-zero affine tree packet. Structural growth introduces a residual refinement `r(x)` with zero initial output:

```text
h_new(x) = h(x) + alpha * r(x),     r(x) = 0 at birth.
```

Therefore `h_new == h` at the growth event for every positive architecture scale `alpha`. Expansion is function-preserving independently of the later routing geometry or residual-packet optimization.

For a multilayer network, this operation is applied independently at each hidden layer. In the current HIGGS adapter, the inherited affine packets and output head are frozen after expansion. TRAIN updates only newborn routing and residual packets. Five positive layerwise scales determine how much residual structure each hidden layer contributes.

## 2. Why the scale-source control matters

A learned-scale adapter has two advantages over a fixed-scale adapter:

1. five additional trainable coefficients;
2. the ability to allocate structural capacity using held-out predictive evidence.

A learned-vs-fixed comparison cannot isolate those explanations.

The corrected scale-source control therefore contains three matched arms:

- **fixed** — scales remain at `sigmoid(-2)`;
- **TRAIN-scale** — the five scales are optimized from TRAIN;
- **heldout-scale** — the same five scales are optimized from ARCHITECTURE-SELECTION.

TRAIN-scale and heldout-scale use the same scale optimizer, learning rate, warmup, clipping, update cadence, and total scale-update examples. Trainable-scale checkpoints are ineligible until at least one scale update has occurred.

The original development SELECTION block is split into disjoint 100k ARCHITECTURE-SELECTION and 100k CHECKPOINT-SELECTION pools. RANKING remains evaluation-only.

## 3. HIGGS evidence

### Development and predeclared replication

Initial development run: SHA `265a8e040a060080d198bb5fce5578a1ccc1825f`, Actions run `36525742176`, job `109268282950`, artifact `11014458749`.

| Arm | Ranking NLL | Ranking AUC |
|---|---:|---:|
| fixed-scale adapter | 0.57106812 | 0.76946808 |
| held-out learned-scale adapter | 0.57049915 | 0.76991939 |

Heldout minus fixed: **−0.00056897 NLL / +0.00045131 AUC**.

The effect direction then reproduced on all three predeclared independent seeds:

| Seed | learned − fixed ranking ΔNLL | learned − fixed ranking ΔAUC |
|---:|---:|---:|
| 733 | −0.0001071258 | +0.0001569420 |
| 2027 | −0.0002529960 | +0.0003390280 |
| 4099 | −0.0003309902 | +0.0003832027 |

Mean replicated effect: approximately **−0.000230371 NLL / +0.000293058 AUC**.

### Corrected matched causal control

SHA `70e21908c6fa63405dbc718c7d87a095ce4fbde3`, Actions run `37220316023`, job `111489148610`, artifact `11310199732`.

| Arm | Ranking NLL | Ranking AUC |
|---|---:|---:|
| fixed | 0.571336907 | 0.768764188 |
| TRAIN-scale | 0.571734560 | 0.768675883 |
| heldout-scale | **0.570802453** | **0.769270809** |

Heldout-scale minus TRAIN-scale: **−0.000932107 NLL / +0.000594926 AUC**.

The retained TRAIN-scale vector increases all five coefficients above their common initialization, while the heldout-scale vector strongly suppresses deeper residual packets. This is evidence consistent with held-out architecture allocation rather than merely granting more residual amplitude. It remains a one-seed corrected causal control.

The earlier Actions run `37073647196` is retained only as an invalid control: its TRAIN-scale arm restored a pre-warmup checkpoint and must not be cited as causal evidence.

## 4. External transfer

Frozen protocol: `research/external_adapter_benchmark_protocol.md`.

SHA `b3c867c77ba21b033bc4d68916f82ee4f82fa099`, Actions run `37220258044`, aggregate job `111492040882`, summary artifact `11310109413`.

The study completed all **35/35** planned cells: seven public OpenML binary datasets × five predeclared seeds. Preprocessing is fit on TRAIN only. The dataset is the primary inferential unit; seed replicates estimate within-dataset variability and headline intervals bootstrap datasets as clusters.

Learned residual adapter versus inherited MLP:

- **7/7 dataset wins** in NLL;
- **7/7 dataset wins** in AUC;
- mean dataset ΔNLL **−0.004773**, 95% CI `[-0.007932, -0.002308]`;
- mean dataset ΔAUC **+0.003448**, 95% CI `[+0.001291, +0.006955]`.

Learned versus fixed residual-scale adapter is substantially weaker: mean ΔNLL `−0.000391` with 5/7 wins and a CI crossing zero; mean ΔAUC `+0.000042` with 4/7 wins and a CI crossing zero.

The learned adapter loses all seven datasets to CatBoost, XGBoost, and LightGBM under the frozen global reference recipes. The result therefore supports **function-preserving residual-tree expansion of an MLP**, not a claim of tree-ensemble or tabular SOTA superiority.

## 5. Audit discipline

The fresh HIGGS shadow range is UCI HIGGS rows `[9,600,000, 10,100,000)`.

It remains **unopened**.

The legacy final-500k audit is already opened and is retained only for frozen historical comparisons. The fresh shadow audit may be opened only after a final candidate/configuration is frozen without shadow information. Once opened, shadow metrics may not select a seed, checkpoint, architecture, hyperparameter, or follow-up configuration.

## 6. Current claim boundary

The evidence supports:

> A tabular MLP can be embedded exactly in a tree-expandable architecture; zero-at-birth residual-tree structure can be introduced without perturbing the inherited predictor; training that residual structure improves the inherited MLP across all seven datasets in the frozen external transfer benchmark; and, on HIGGS, held-out residual architecture weights reproducibly beat a matched fixed-scale adapter, with a corrected canonical control also favoring held-out over TRAIN-sourced scale updates.

It does **not** establish:

- broad tabular state of the art;
- superiority over properly tuned CatBoost/XGBoost/LightGBM;
- a final unseen HIGGS improvement;
- replicated causal superiority of held-out over TRAIN-sourced scale learning;
- successful differentiable search over the full TorchBoost architecture space.

## 7. Remaining decision gate

The main methodological gate before a final unseen HIGGS claim is to freeze one candidate/configuration without shadow information and perform the one-way shadow audit. Any further causal replication or architecture study must remain shadow-blind.

Exact evidence provenance is maintained in `research/RESULTS.md`.
