# Saved-data statistical reconstruction

## What is held fixed

Every result retains its original job ID, seeds, finite policy library and solved distribution `q`. The inputs are archived simulation evaluations, not training data for a new fit. Reanalysis uses the same scientific functions or explicitly documented reporting calculations as the original analysis.

The frozen campaign identity is `cc8e7f81c117c512cdb18f9993bc4b67960d7ac8a85d3329969ad550ff4b2ff9`, matrix identity is `8f77eaa2704659d9c4b6ef2f4f95ee1fa21f57695c8e63e1e6954afb9efa111a`, and source identity is `eb0143548676863326355ff2e94e81e6081735a0be952e25ab8e2a190dada900`. File-level hashes and original completion records are verified before use.

## Independent equilibrium checks

The original solve used 200 simulations for each evaluated joint policy profile. The independent evaluation then held `q` fixed and used 2,000 new simulations, with common random numbers across the profile and its policy replacements. Reanalysis loads these saved samples; it does not generate another evaluation stream.

For benchmark and sparse-scalability jobs, the saved arrays contain returns under `q` and paired policy-replacement gains, grouped by distribution hash. For CNC jobs, returns are reconstructed from the archived support-deviation-closure chunks, then combined using the fixed distribution. For transplant jobs, the saved arrays retain the same-mechanism control, transferred distribution, and re-solved target distribution as three paired arms.

The original scientific functions recompute nominal gaps, payoff and gain standard errors, the recorded upper-bound diagnostics, and transplant effects using 5,000 bootstrap draws. The stable seed suffixes are `formal-bootstrap`, `formal-audit-shared-bootstrap` and `three-arm-joint-bootstrap`, with the original campaign label and job ID. All original diagnostic fields are retained for verification, regardless of whether the manuscript displays them.

## CNC operating outcomes

Operating outcomes use the separate 500 saved evaluations for each fixed case and distribution. The analysis compares mechanisms within matched cases and compares changes in platform load, route mix and outside capacity pressure. Original seed strata remain fixed, with independent bootstrap streams between the three strata and shared replication weights within each stratum.

The bootstrap seed is `2026090902`, with 5,000 draws. Each draw reconstructs a case's rates and concentration measures from resampled raw totals. Case-level paired differences are then averaged equally. Mechanism comparisons contain 24 matched cases; each operating-factor comparison contains 48.

Percentage effects are `100 * (target mean / reference mean - 1)`, where each mean gives equal weight to its cases. The reference mean is recalculated within each bootstrap draw. These are not averages of case percentages, and the intervals are not formed by subtracting two separately calculated intervals.

The confidence intervals describe simulation uncertainty conditional on the specified cases and fitted distributions. They do not resample the fitted distributions or estimate variation from retraining.

## Comparison and failure policy

All numeric checks use the tolerance fixed before reanalysis: absolute `1e-9`, relative `1e-10`. Discrete identities and hashes must match exactly. Recomputed values and disagreements are saved separately from immutable inputs. No mismatch triggers automatic data repair, retuning, exclusion or tolerance relaxation.

The default audit route covers 72 benchmark, 12 sparse-scalability, 96 CNC and 96 transplant jobs. Archived Runtime and parameter-robustness results are included in data-integrity verification but are not rerun by these statistical commands.
