# Deep-controller cross-environment reproduction — 2026-09-25

## What reproduced exactly

GitHub Actions run `36145015457` reran the deep-controller development, bounded thermal-only calibration, and fresh confirmations from source commit `05c4ec6fe7385cbde0e94d409935c0f6f187c320`. The experiment source fingerprint is `a30a241ef35d2108ed4b8bb2559b0029d605d64ff27839ebfb1e226f27ce5c6c`. Train/control/selection/ranking/audit hashes for seeds 97, 101, and 103 exactly match the earlier local run, so the data and split identities are not the cause of the numerical difference.

The earlier local execution used Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, and scikit-learn 1.8.0. GitHub Actions used Python 3.12.14, PyTorch 2.14.0+cpu, NumPy 2.5.3, and scikit-learn 1.9.1. The full Actions `pip freeze` is retained in the workflow artifact.

## Fresh Actions confirmation

Lower NLL is better.

| Controller | Seed 101 ranking | Seed 103 ranking |
|---|---:|---:|
| no adaptive controls | 0.604129136 | 0.628655612 |
| plasticity only | **0.604066789** | **0.628030837** |
| plasticity + cooling only | **0.604066789** | **0.628030837** |
| capacitor | 0.604271889 | 0.629197180 |
| RLC | 0.604288101 | 0.629194319 |
| direct, development-dose calibrated | 0.604108155 | 0.628883183 |
| direct, development-ranking selected | 0.604942918 | 0.632403374 |

Plasticity-only was the ranking winner on both fresh Actions seeds. By protocol, audit was then opened for that winner and the prespecified no-control reference only:

| Seed | no-control selected audit | plasticity selected audit | no-control last audit | plasticity last audit |
|---|---:|---:|---:|---:|
| 101 | **0.606572092** | 0.606705189 | 0.625104070 | **0.625017166** |
| 103 | **0.630484998** | 0.630713284 | 0.641266346 | **0.638955295** |

Thus the ranking advantage did **not** transfer to selected-checkpoint audit on either confirmation seed. Plasticity slightly improved the final iterate but the selected checkpoint remained better than either last iterate. There is no new deep-double-descent claim here.

## Thermal matching

The bounded development calibration selected direct energy-gain proxy `0.043183767887204905`; the separately ranking-selected direct gain was `0.25`. Development achieved the declared joint peak/integrated-exposure tolerance, but that dose match failed on both confirmation seeds. The failed confirmations remain failed; they are not post-hoc recalibrated.

## Reproducibility consequence

The local and Actions candidate orderings differ even though source and data hashes agree. Effects are presently of the same order as cross-software numerical variation. Consequently:

- do not promote an adaptive controller from these small differences;
- do not pool local and Actions ranking orderings as interchangeable replicates;
- pin the research environment before interpreting 1e-4-scale NLL differences;
- retain environment metadata and split/source hashes with every result;
- prefer conclusions that survive fresh audit and software-environment replication.

Blocking CI is now constrained to the direct dependency versions used by this Actions reproduction, with PyTorch 2.14.0 installed from the CPU wheel index. The CatBoost ratchet thresholds and assertions are unchanged.
