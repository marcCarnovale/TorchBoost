# Deep-controller results — 2026-09-25

## Scope and provenance

Completed locally: ten development fits (nine coarse candidates plus one thermal-only refinement), then seven frozen candidate fits on each of fresh data seeds 101 and 103, all at 2,048 updates. Source code is commit `05c4ec6fe7385cbde0e94d409935c0f6f187c320`; the same implementation with style-only test corrections passed blocking remote CI at `95da54e3f26e1f80d2facefb6c8ac0cf88e915da`.

The experiment source fingerprint is `a30a241ef35d2108ed4b8bb2559b0029d605d64ff27839ebfb1e226f27ce5c6c`. The machine-readable record in `results/controller-protocol-2026-09-25.json.gz` retains all scalar scores, configurations by checksum, thermal diagnostics, environments, split fingerprints, and full raw-result checksums. The full raw JSON files include configurations and training traces; the script and GitHub workflow reproduce them.

Every data seed uses 12,000 train / 1,800 control / 1,000 selection / 1,000 ranking / 1,200 audit rows. Development never scores audit. Confirmation freezes the development gains and only audits its ranking winner plus the prespecified no-control reference. Lower NLL is better.

## Fresh confirmation ranking

| Controller | Seed 101 ranking NLL | Seed 103 ranking NLL |
|---|---:|---:|
| No adaptive controls | 0.603877060 | 0.627785814 |
| Plasticity only | 0.603924711 | 0.628039992 |
| Plasticity + cooling, no heat source | 0.603924711 | 0.628039992 |
| Capacitor | 0.604225401 | 0.628287461 |
| RLC | 0.604243671 | 0.628281243 |
| Direct, development-dose calibrated | 0.604088383 | 0.628237808 |
| Direct, development-ranking selected | 0.604296320 | 0.628437021 |

**No adaptive controls won ranking on both fresh seeds.** The differences are small; two seeds do not establish universal controller inferiority. Losing controllers were not subsequently audited, so these are ranking comparisons, not undisclosed pairwise audit estimates.

| Data seed | Audited ranking winner | Selected audit NLL | Last-iterate audit NLL |
|---|---|---:|---:|
| 101 | none | 0.606483268 | 0.621921569 |
| 103 | none | 0.630501061 | 0.636784865 |

The control-free selected checkpoint is better than the last iterate in both audits. This study does not supply a new double-descent result. No controller or production default is promoted.

## Thermal matching: development success, confirmation failure

The frozen direct-dose gain is `0.045810275458604145`; the separately frozen development-ranking gain is `0.1`. The latter was not relabeled dose-matched after inspecting confirmation. Both keep maximum per-step direct heat at 0.03.

The coarse closest gain 0.1 failed matching. One bounded thermal-only refinement succeeded on development seed 97. Matching requires peak excursion AND integrated capacity-weighted excursion, in ambient-to-thaw units, each within 20% of capacitor. Neither predictive loss nor audit chooses the refinement.

| Phase / seed | Direct-to-capacitor peak-excursion ratio | Direct-to-capacitor integrated-exposure ratio | Joint match |
|---|---:|---:|---|
| development / 97 | 0.949780 | 1.056278 | True |
| confirmation / 101 | 0.545488 | 0.573365 | False |
| confirmation / 103 | 0.639595 | 0.714718 | False |

The development ranking gain also failed to beat no control on either fresh ranking split. The observations do not establish an energy/dose-controlled advantage for electrical or direct feedback. Equal peak alone would not have been sufficient, and a nearest candidate is not automatically a matched candidate.

## Remaining source normalization

The uncapped capacitor source maps loss surprise to charge linearly, so stored energy and resistor heat scale quadratically in loss units. Direct feedback maps surprise to heat linearly. The new source-scaling characterization gives 100x capacitor heat versus 10x direct heat when all losses are multiplied by ten, while constant loss offsets leave both unchanged. This is a comparator/loss-unit confound, not a failure of electrical energy conservation. It can break fixed-gain dose transfer and remains to be normalized before a stronger causal controller claim.

## Test coverage and queued work

Expanded local contract suite: 87 tests passed, including three new source-scaling characterizations. The unchanged CatBoost ratchet passed locally and in blocking remote CI on the implementation. The final record/test-only commit reruns the stronger CI matrix; it does not rerun or alter the ongoing research source.

GitHub research run `36145015457` independently executes the same development/refinement and seeds 101/103, then the development-only 80k Covertype depth/count screen against CatBoost depths up to 12 and 2,048 iterations, then native-forest seeds 67/73. Launching the workflow is not a completed benchmark result.

The native forest replication retains its existing generic machinery and learned tree coefficients. Its existing `none` continuation disables additional explicit penalties but still shares AdamW weight decay 1e-6; it is not the strict zero-regularization epicycle arm. Any switched-objective second-descent pattern must be described as such.

The five-arm epicycle experiment is not yet executed. Its primary must call the same generic regularizer used for ordinary unknown problems; the no-regularization arm must also set optimizer weight decay to zero. Fourier/order/radial/inverse-power quantities are diagnostics only in primary. The separate Fourier-informed secondary positive control remains in the plan.
