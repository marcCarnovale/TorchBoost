# Unified progressive TorchBoost: historical execution record

> **Historical / superseded research path.**
>
> This document records an earlier progressive-forest study and is retained for
> provenance. It is **not** the current headline TorchBoost result. The active
> research program is the function-preserving neural→tree residual adapter
> documented in [README.md](README.md), [research/RESULTS.md](research/RESULTS.md),
> and [research/NEURAL_TREE_ADAPTER_NOTE.md](research/NEURAL_TREE_ADAPTER_NOTE.md).
> Do not use this report to infer the current claim boundary.

## Original study status

**Status at the time of execution: development build; requested full study not verified complete.**

### Verification

Test return code: 0; passed: 254; failed: None. Full output was recorded in
`results/current/final_tests.log` in the original development workspace.

| Block | Registered | Successful | Failed | Missing |
|---|---:|---:|---:|---:|
| tuning | 36 | 35 | 0 | 1 |
| components | 224 | 224 | 0 | 0 |

Reference results: `{'success': 96}`.

Unfinished workspace processes were stopped before packaging. Launching a job
was never counted as completing it.

## Implementation scope

The study centered on `torchboost.adaptive.unified_progressive`. It reused
native observations, physical control, plasticity, structural transactions and
optimizer-state management in a progressive score-sum model. Its proposal
builder used binary/multiclass/regression curvature and explicit split
constraints. Stage row pools and permanent feature masks replaced earlier
ineffective controls. Custom penalties acted on canonical leaf scores, residual
node values, feature use, routing support and rate-scaled tree predictions.

These components remain available as secondary experimental machinery, but the
current evidence-bearing path is the residual neural→tree adapter.

## Original reproduction notes

The original study used:

```bash
python -m pip install -e ".[experiment]"
python -m pytest -q
```

Historical experiment entry points include
`experiments/unified_study.py`, `experiments/unified_baselines.py`, and
`experiments/finish_unified.py`. Existing failed records should be inspected,
not silently discarded. Checkpoints use Python-backed serialization and must be
trusted before loading.

## Evidence limits

The original context problems were synthetic and diamonds was a reused public
dataset. Search and compute budgets differed from reference systems. There was
no independent frontier-performance claim. Soft monotonicity was not a global
constraint proof. Online rewards were observational, not isolated causal
estimates.

The integer-clock `before_selected` convenience count could include an
intervention tied in clock value but later in event order than a selected
proposal. It should not be used alone for attribution; phase/event ordering
must be inspected.

## Current status

The active repository has since advanced beyond this study. For current
quantitative claims, exact SHAs, Actions run/job/artifact IDs, negative results,
invalidated controls, and the sealed HIGGS shadow-audit policy, use
[research/RESULTS.md](research/RESULTS.md).
