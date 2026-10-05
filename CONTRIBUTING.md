# Contributing

TorchBoost is an active research repository. Contributions are welcome when they
preserve the distinction between **implemented mechanism**, **measured behavior**,
and **supported research claim**.

## Before changing research code

Read:

- [README.md](README.md) for the current public claim boundary;
- [research/RESULTS.md](research/RESULTS.md) for evidence provenance;
- [AGENTS.md](AGENTS.md) for engineering and research rules;
- the relevant frozen protocol under `research/` or `experiments/`.

Do not use the fresh HIGGS shadow audit at rows
`[9,600,000, 10,100,000)` for development, architecture selection,
checkpoint selection, seed selection, or hyperparameter tuning.

## Pull requests

A research-facing pull request should, where applicable:

1. identify the mechanism or hypothesis being changed;
2. keep TRAIN / control / architecture-selection / checkpoint-selection /
   RANKING / audit roles explicit;
3. add or update tests for the changed contract;
4. record reproducibility information for quantitative experiments;
5. retain negative or invalid results rather than silently deleting them;
6. avoid state-of-the-art, novelty, or causal claims unsupported by the
   recorded evidence.

Run at minimum:

```bash
python -m pip install -e '.[dev,benchmark]'
pytest -q
ruff check .
```

The CI workflow additionally runs the evidence-bearing architecture tests,
CatBoost performance ratchet, and the experimental mechanism monitor.

## Research provenance

Headline quantitative results belong in `research/RESULTS.md` with exact
source SHA, GitHub Actions run, job, and artifact identifiers where available.
A refactor does not inherit the evidentiary status of an older result unless the
relevant behavior is reproduced at the new source state.
