# Unified progressive TorchBoost: execution record

**Status: development build; requested full study not verified complete.**

## Verification

Test return code: 0; passed: 254; failed: None. Full output is in `results/current/final_tests.log`.

| Block | Registered | Successful | Failed | Missing |
|---|---:|---:|---:|---:|
| tuning | 36 | 35 | 0 | 1 |
| components | 224 | 224 | 0 | 0 |

Reference results: {'success': 96}.
Unfinished workspace processes were stopped before packaging. Launching a job is never counted as completing it.

## Implementation scope

The new module is `torchboost.adaptive.unified_progressive`. It reuses native observations, physical control, plasticity, structural transactions and optimizer-state management in a progressive score-sum model. Its proposal builder uses full binary/multiclass/regression curvature and explicit split constraints. Stage row pools and permanent feature masks replace the earlier ineffective controls. Custom penalties act on actual canonical leaf scores, residual node values, feature use, routing support and rate-scaled tree predictions. Existing packed progressive APIs are historical references, not silently recertified implementations.

## Reproduction

Install using `python -m pip install -e ".[experiment]"`; run `python -m pytest -q`; inspect `examples/unified_progressive.py`. Experiment entry points are `experiments/unified_study.py`, `experiments/unified_baselines.py`, and `experiments/finish_unified.py`. Existing failed records must be inspected, not silently discarded. Checkpoints use Python-backed serialization and must be trusted.

## Evidence limits

The two context problems are synthetic and diamonds is a reused public dataset. Search and compute budgets differ from reference systems. There is no independent frontier-performance claim. Soft monotonicity is not a global constraint proof. Online rewards are observational, not isolated causal estimates. Additional historical exploratory circuits, accelerator kernels, PID/Kalman alternatives and structural RL are not declared complete.

The integer-clock `before_selected` convenience count can include an intervention tied in clock value but later in event order than a selected proposal. Do not use that count alone for attribution; inspect phase/event ordering. This limitation is retained explicitly.

No remote GitHub changes were made.
