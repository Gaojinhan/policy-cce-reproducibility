# CNC outcome-method provenance

`policy_cce_repro/recompute_outcomes.py` is a portable reporting refactor. It reads the supplied offline dataset, reconstructs metrics from the saved 500-rollout records, and repeats the historical bootstrap. It does not call the simulator, refit q, import historical manuscript builders, or copy final numbers into computed results. It validates its output directory through `Dataset.validate_output_dir`, including when called directly without the CLI.

## Frozen metric definitions

The following files are byte-identical copies covered by `metadata/frozen-source-manifest.json` and `metadata/source-copy-manifest.json`:

| Frozen source | Imported functionality | SHA-256 |
| --- | --- | --- |
| `cmfg_cce/evaluation/independent_audit.py` | `distribution_hash` | `afdeb3de0bc7f4c555787e0a9fd493fb6f31e8baca6704d1cbcabfa69283ea1e` |
| `cmfg_cce/evaluation/revision_statistics.py` | `weighted_outcome_replications`, `aggregate_outcome_replications`, `_reconstruct_outcome_metric_samples` | `b30da8d053b0a92523149f4e0e391571688c92e066c76ef700ac8b2be406034a` |

The underscore-prefixed reconstruction function is used deliberately: it is the frozen common definition for rates, supplier-availability measures, HHI, submitted-markup/lead-time ratios and manufacturer profit. A separately rewritten approximation is not substituted for it.

## Ported reporting logic

Historical files are identified below for lineage, not as runtime dependencies. Their original machine paths and cloud-access code are not used by this package.

| Historical reporting file | Logic preserved in the portable module | SHA-256 of reviewed historical file |
| --- | --- | --- |
| `cnc_paired_mc.py` | Raw/vector verification, three-stratum resampling, case bootstrap metrics, absolute mechanism/factor contrasts, percentile intervals | `e784198ca237c291263a78332fc0ed45116821e999cfd2b9523718f0e8db9940` |
| `build_percent_effects.py` | Relative change in equal-case means; recomputed reference denominator in every paired draw | `67e18f8d5c773296671fe7876f778d1d05aa4604f448268978243606070312da` |
| `restore_cnc_figure_style.py` | Added T/M3/F4 factor metrics; mechanism levels and sample SD; grouped policy shares | `d09192635a8d2c0268d02bc2fa267155007a7c14581b5dbadead5b2bb0d7dc4f` |
| `build_cnc_restoration.py` | Policy-position versus winning-order shares; normal/combined-pressure endpoints; family-mismatch percentage-point display | `a56eef3965ba8d9e08bddd230c86bb2d47d098983d7ca9b0356272dd8963f063` |

These are file-byte hashes, not hashes of an extracted function's source text. The portable module has a different file hash because file access, orchestration, validation and output handling were refactored. Figures and manuscript layout code were not ported into this recomputation command.

## Preserved statistical contract

1. **Fixed distributions and raw records.** Verify each original q, the ordered outcome profiles, all 500-replication payloads, the complete raw-vector hash and the q-weighted raw-vector hash. Weight raw counts/returns by q using the original function and probability order.
2. **Shared paired resampling.** `SeedSequence(2026090902).spawn(3)` creates one generator for each fixed stratum 0, 1 and 2. Each generator draws 5,000 multinomial count vectors with total count 500 and equal probabilities. All mechanisms and operating conditions within that stratum use the same count vectors. Strata are not resampled.
3. **Nonlinear metric reconstruction.** For each bootstrap draw, multiply the count vector by the q-weighted raw records, then reconstruct the case's ratios and HHI from those totals. Averaging episode-level ratios or already computed HHI values would be a different method.
4. **Mechanism comparisons.** Absolute effects are the equal mean of 24 paired case differences. Relative effects are `100 * (mean_target / mean_reference - 1)`, recalculating both equal-case means in every draw. They are neither the mean of case-level percentage changes nor percentage-point changes.
5. **Operating-factor comparisons.** Form 48 matched pairs with mechanism, stratum and the other factors fixed. Average reconstructed case differences equally. F4 availability and family mismatch retain their within-family denominators. The capacity table's adverse-direction sign convention is a presentation transformation, not a change to the stored high-minus-low effects.
6. **Intervals.** Report the 2.5th and 97.5th percentiles of paired bootstrap draws and their sample SD (`ddof=1`). These are pointwise conditional simulation intervals, not simultaneous intervals or uncertainty over fitted q or an industry population. No p-values are calculated.
7. **Lead time.** Relative comparison uses the submitted lead-time multiplier and applies only to AUC4 versus AUC3. The near-zero plotted difference from requested due time is not used as the percentage-effect denominator.
8. **Descriptive summaries.** Mechanism bars use equal-case means and between-case sample SD, not bootstrap CI. Policy composition averages q-weighted manufacturer-position shares; winning-policy shares use q-weighted winning counts divided by assigned counts within each case. Capacity endpoints average case-level ratios, not pooled records.

The portable loader visits jobs in sorted ID order. Historical plotting code visited manifest order. Pair construction explicitly fixes the same stratum/condition/mechanism ordering for inferential contrasts. Descriptive floating-point reductions may differ in their final bits; the predeclared comparison tolerance is `atol=1e-9`, `rtol=1e-10`. No tolerance is adjusted to obtain a pass.

## Numerical reference files

References are used only after recomputation to detect mismatches. Their immutable archive manifest entries are checked by `Dataset.reference`.

| Reference | SHA-256 |
| --- | --- |
| `cnc-paired-mc-summary.json` | `4c4ec4bb0ca5f3a1b140bae62778b9737c9fa6834f2f59360608baf2ce6b45e0` |
| `percent-effects.json` | `9d4bf8685ef646f0a05e484f06d1f9e6baa16f9a21b147b709febd95798ec8a3` |
| `cnc-original-style-evidence.json` | `e4051367e34c3f397569daf520bcdff9b8efd752a1fbb480a2b4197ad2ce6fdb` |
| `cnc-restoration-evidence.json` | `e1b9a0bc8c8db90e9084ce5e014b864d0469a6de7104635a5ad4ac5906d9271c` |

The first full offline run on the pinned NumPy 2.1.3 environment validated all 96 raw and 96 weighted-vector hashes and passed all **7,697** numerical comparisons. It reported zero mismatches and a maximum absolute difference of **1.1368683772161603e-13**. That run produced `outcomes-recomputed.json`; no raw records or reference values were changed.

The host BLAS emitted floating-point status warnings for matrix multiplication during that run, as it did for a tiny finite-input unit test. The output arrays remained finite and all reference comparisons passed. These warnings were retained, not suppressed; the validation result establishes numerical agreement for this run, not a general diagnosis of the host BLAS.

