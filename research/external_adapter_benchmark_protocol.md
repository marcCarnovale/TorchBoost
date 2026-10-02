# External residual-adapter benchmark protocol

Status: frozen before reading v2 benchmark results.

## Purpose

Test whether the residual-tree adapter discovered on HIGGS transfers to unrelated public binary tabular datasets without per-dataset architecture tuning. This study does not touch the locked HIGGS shadow audit.

## Frozen datasets

OpenML IDs:

- phoneme: 44127
- bioresponse: 45019
- bank-marketing: 44126
- magic-telescope: 44125
- default-credit: 45020
- electricity: 44120
- miniboone: 44128

No automotive-insurance or proprietary-employment dataset is included.

## Frozen splits

For each dataset and each seed, stratified 60/20/20 train/selection/ranking splits are created. Seeds are:

- 509
- 733
- 2027
- 4099
- 8191

Imputation and standardization are fit on TRAIN only. SELECTION may be used for checkpointing, early stopping, and the adapter's architecture-scale updates. RANKING is evaluation only.

## Frozen neural family

The MLP capacity rule depends only on total row count:

- fewer than 10k rows: width 128, depth 3;
- 10k to fewer than 50k: width 192, depth 4;
- 50k or more: width 256, depth 4.

The anchor receives 40 AdamW epochs. The residual adapter receives 12 epochs. The learned-scale arm begins architecture updates after 6 epochs and updates scales every 2 residual minibatches. The fixed and learned arms share the same residual-training seed and initial residual scale sigmoid(-2).

The inherited MLP backbone and output head remain frozen after growth. One zero-at-birth residual-tree refinement is grown in every hidden layer. Only newborn residual packets/routing are trained from TRAIN; only layerwise architecture scales are trained from SELECTION.

## Frozen tree references

CatBoost, XGBoost, and LightGBM use one global recipe each across all datasets. Their iteration budgets are upper bounds and SELECTION is used for early stopping. No dataset-specific hyperparameter search is permitted.

## Recorded quantities

Every cell records:

- repository source SHA;
- OpenML ID/name/version;
- rows, columns, class balance, and missing-value count;
- exact split sizes;
- model parameter counts where available;
- examples seen and equivalent full TRAIN passes for neural models;
- fit wall time;
- selected epoch/tree count;
- SELECTION NLL/AUC;
- RANKING NLL/AUC;
- learned architecture state;
- paired RANKING deltas versus MLP, fixed adapter, CatBoost, XGBoost, and LightGBM.

The aggregate report uses paired dataset-seed differences, win/tie/loss counts, standard errors, and deterministic bootstrap 95% intervals. These intervals summarize variation across the benchmark cells; they are not a claim that the seven datasets are a random sample from a formal population.

## Interpretation

A positive transfer result requires more than a favorable grand mean. The report must expose per-dataset behavior and paired win/loss counts. If gains are HIGGS-specific or concentrated in one dataset, the repository must say so.

The HIGGS shadow audit at rows [9,600,000, 10,100,000) remains unopened throughout this study.
