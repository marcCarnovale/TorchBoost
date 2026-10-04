# Function-Preserving Neural–Tree Expansion for Tabular Learning

## Abstract

This project studies a narrow architecture question: can a trained tabular neural
network be expanded into a richer tree-structured model without first damaging
the inherited predictor, and can held-out evidence determine how much of that new
structural capacity should be used?

TorchBoost represents each dense hidden layer of a ReLU MLP as a depth-zero
affine tree layer. The representation is exact. A hidden layer can then grow
zero-at-birth residual-tree children, preserving the represented function at the
instant of expansion. The inherited MLP backbone and output head can remain
frozen while only newborn routing and residual packets are optimized.

On the 500k-row HIGGS development protocol, a five-layer residual-tree adapter
with five held-out learned positive architecture scales improved over an
otherwise matched fixed-scale adapter. The direction reproduced on all three
predeclared independent seeds. The fresh 500k-row HIGGS shadow audit remains
unopened.

The current causal-control study goes one step further: the same five scale
parameters are trained either from TRAIN or from a disjoint
ARCHITECTURE-SELECTION subset, with matched update cadence and a separate
CHECKPOINT-SELECTION subset. This tests whether held-out architectural
allocation adds value beyond merely introducing five extra supervised
parameters.

## 1. Architecture

Let a dense layer be

[
h(x)=Wx+b.
]

TorchBoost embeds this exactly as a depth-zero affine tree packet. Structural
growth introduces a residual refinement (r(x)) with zero initial output:

[
	ilde h(x)=h(x)+alpha r(x), qquad r(x)=0 	ext{ at birth}.
]

Therefore (	ilde h=h) at the growth event for every positive architecture
scale (alpha). Expansion is function-preserving independently of the later
routing geometry or residual-packet optimization.

For a multilayer network, the operation is applied independently at each hidden
layer. In the current HIGGS adapter, the inherited affine packets and output
head are frozen after expansion. TRAIN updates only newborn routing and residual
packets. Five positive layerwise scales determine how much residual structure
each hidden layer contributes.

## 2. Why the control matters

A learned-scale adapter has two advantages over a fixed-scale adapter:

1. five additional trainable coefficients;
2. the ability to allocate structural capacity using held-out predictive
   evidence.

A learned-vs-fixed comparison cannot isolate those explanations.

The corrected scale-source control therefore contains three matched arms:

- **fixed** — scales remain at `sigmoid(-2)`;
- **train-scale** — the five scales are optimized from TRAIN;
- **heldout-scale** — the same five scales are optimized from
  ARCHITECTURE-SELECTION.

Train-scale and heldout-scale use the same scale optimizer, learning rate,
warmup, clipping, and update cadence. Trainable-scale checkpoints are ineligible
until at least one scale update has occurred.

The original 200k development SELECTION block is split deterministically into
100k ARCHITECTURE-SELECTION rows and 100k CHECKPOINT-SELECTION rows. Thus the
heldout-scale optimizer does not consume the examples used to choose the final
adapter checkpoint.

RANKING remains evaluation-only.

## 3. Existing HIGGS evidence

The development run at source SHA
`265a8e040a060080d198bb5fce5578a1ccc1825f`, Actions run
`36525742176`, job `109268282950`, artifact `11014458749` produced:

| Arm | Ranking NLL | Ranking AUC |
|---|---:|---:|
| fixed-scale adapter | 0.57106812 | 0.76946808 |
| held-out learned-scale adapter | 0.57049915 | 0.76991939 |

Heldout minus fixed:

- ΔNLL: **−0.00056897**
- ΔAUC: **+0.00045131**

The held-out scale vector was approximately

[
[0.1329,;0.0923,;0.0644,;0.0423,;0.0385].
]

The effect direction then reproduced on three predeclared independent seeds:

| Seed | learned − fixed ranking ΔNLL | learned − fixed ranking ΔAUC |
|---:|---:|---:|
| 733 | −0.0001071258 | +0.0001569420 |
| 2027 | −0.0002529960 | +0.0003390280 |
| 4099 | −0.0003309902 | +0.0003832027 |

Mean replicated effect:

- ΔNLL: approximately **−0.000230371**
- ΔAUC: approximately **+0.000293058**

These are development/ranking results, not final unseen-test results.

## 4. Invalid causal-control attempt retained for provenance

Actions run `37073647196` produced an apparent heldout-vs-train-scale
difference, but its train-scale arm restored a pre-warmup epoch-1 checkpoint.
The retained scales were therefore still at initialization even though later
scale updates had occurred.

That comparison is explicitly invalid and must not be cited as evidence. The
corrected protocol requires post-update checkpoint eligibility.

## 5. External-transfer study

The transfer study uses seven public OpenML binary tabular datasets and five
predeclared seeds per dataset. It compares:

- MLP anchor;
- fixed-scale residual adapter;
- held-out learned-scale residual adapter;
- CatBoost;
- XGBoost;
- LightGBM.

Preprocessing is fit on TRAIN only. Numeric features use median imputation and
standard scaling. Categorical features use most-frequent imputation and one-hot
encoding, preserving categorical information rather than coercing categories
to missing values.

The dataset is the primary inferential unit. Seed replicates estimate
within-dataset variability; headline uncertainty averages effects within
dataset and bootstraps datasets as clusters. Fixed global tree recipes are
reference baselines, not claims of optimally tuned competitors.

## 6. Audit discipline

The fresh HIGGS shadow range is UCI HIGGS rows
`[9,600,000, 10,100,000)`.

It remains unopened.

The earlier shadow-freeze manifest has been explicitly marked superseded because
the causal-control protocol changed after that freeze. A new final freeze may be
created only after the corrected causal-control and transfer studies are
interpreted and the final configuration is fixed without shadow information.

Once opened, shadow metrics may not select a seed, checkpoint, architecture,
hyperparameter, or follow-up configuration.

## 7. Current claim boundary

The evidence currently supports:

> A tabular MLP can be embedded exactly in a tree-expandable architecture;
> zero-at-birth residual-tree structure can be introduced without perturbing
> the inherited predictor; and held-out learning of five residual architecture
> weights reproducibly improves the hybrid over a matched fixed-scale adapter
> on the HIGGS development protocol.

It does not yet establish:

- broad tabular state of the art;
- superiority over optimally tuned CatBoost/XGBoost/LightGBM;
- a final unseen HIGGS improvement;
- successful differentiable search over the full TorchBoost architecture space;
- a causal advantage of held-out scale learning over TRAIN-scale learning
  until the corrected control completes.

## 8. Decision gates

A stronger paper/resume claim requires, in order:

1. corrected heldout-scale vs train-scale causal-control result;
2. completed seven-dataset transfer matrix with dataset-clustered analysis;
3. one frozen final candidate/configuration;
4. one-way shadow audit with no post-open retuning.

Exact evidence provenance is maintained in `research/RESULTS.md`.
