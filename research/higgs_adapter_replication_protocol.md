# HIGGS differentiable-adapter replication protocol

Declared before inspecting any results from these replication seeds.

## Frozen experiment

Source experiment: `experiments/higgs_differentiable_adapter.py`.

The scientific comparison is unchanged from run `36525742176`:

- 500,000 HIGGS training rows and the existing canonical split protocol;
- canonical five-layer width-300 MLP anchor;
- inherited MLP backbone and output head frozen;
- one zero-at-birth residual tree refinement in every hidden layer;
- residual/routing parameters trained on TRAIN;
- residual scales initialized at `sigmoid(-2) = 0.1192029`;
- fixed-scale control receives identical residual/routing training;
- learned-scale arm updates only its five scales from SELECTION;
- four adapter epochs, two scale warm-up epochs, architecture update every two
  residual minibatches;
- RANKING is evaluation only;
- the fresh shadow audit at rows [9,600,000, 10,100,000) remains unopened.

No architecture, optimizer, regularization, checkpoint-selection, or exposure
change is permitted between replication seeds.

## Predeclared independent seeds

- 733
- 2027
- 4099

Seed 509 is the development seed and is not counted as an independent
replication.

## Success criterion

The primary replication quantity is learned-scale minus fixed-scale ranking
NLL.  A negative value is favorable. Ranking AUC and selection NLL are
secondary directional checks.

Treat the result as robust enough to freeze for the one-way shadow audit if:

1. learned-scale ranking NLL improves over fixed-scale in at least 2 of 3
   independent seeds;
2. mean learned-minus-fixed ranking NLL across the three seeds is negative;
3. there is no large opposing selection-NLL regression suggesting the scale
   optimizer is exploiting ranking noise.

These rules are frozen before the runs. Do not change them after observing the
replication results.
