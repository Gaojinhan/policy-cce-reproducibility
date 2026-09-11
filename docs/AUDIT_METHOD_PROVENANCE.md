# Saved-sample audit reanalysis

`policy_cce_repro.recompute_audits.recompute_audits` recalculates statistics from
the saved independent-evaluation records. It uses the copied campaign functions
listed below. It performs no simulation, optimization, distribution selection,
or scientific repair. The manuscript inputs and canonical results are read-only.

## Scope and inputs

The paper-selected replay covers 276 formal jobs:

| Family | Jobs | Saved inputs |
| --- | ---: | --- |
| Solver benchmark | 72 | `formal_audit_samples.npz` |
| Sparse scalability | 12 | `formal_audit_samples.npz` |
| CNC main cases | 96 | Complete `stages/formal_audit/chunks/` plans and paired JSON/NPZ chunks |
| Policy transplant | 96 | `transplant_audit_samples.npz` |

N7 runtime-only measurements have no independent gap audit in this replay.
Runtime records, parameter-robustness outcomes, and calibration records are
handled elsewhere in the package. A replay subset selects whole original jobs;
it does not reduce their sample or bootstrap counts.

Each audit uses the original 2,000 saved replications, 5,000 bootstrap draws,
and alpha of 0.05. No new random simulation streams are created. The random
draws here are the original bootstrap resampling indices.

## Original functions and seed recipes

All scientific functions below are imported directly from `cmfg_cce` in this
package. Their source bytes are part of the frozen source manifest.

| Calculation | Original function |
| --- | --- |
| Frozen distribution validation | `experiments.run_revision_cnc._distribution_from_payload` |
| Distribution closure ordering | `experiments.revision_pipeline.frozen_closure` |
| Per-replication policy-replacement gains | `evaluation.independent_audit.build_joint_audit_samples` |
| Gap, payoff and replacement-gain standard errors, gap bounds | `evaluation.independent_audit.summarize_audit_samples` |
| Three transplant arms and paired gap effects | `evaluation.revision_statistics.summarize_three_arm_transplant` |
| Bootstrap seed derivation | `experiments.revision_pipeline.stable_solver_seed` |
| CNC reconstructed-sample digest | `experiments.run_revision_cnc._audit_sample_hash` |
| CNC raw-vector digest | `experiments.run_revision_cnc._profile_vector_hash` |
| Transplant sample serialization | `orchestration.chunks.deterministic_npz` |

The exact seed calls are:

```python
# Solver benchmark and sparse scalability:
stable_solver_seed("revision-full-v1", original_job_id, "formal-bootstrap")

# CNC:
stable_solver_seed("revision-full-v1", original_job_id,
                   "formal-audit-shared-bootstrap")

# Three-arm transplant:
stable_solver_seed("revision-full-v1", original_job_id,
                   "three-arm-joint-bootstrap")
```

The original call sites are `_run_formal_audit` in
`experiments/run_revision_full_v1.py`, `_run_formal_audit` in
`experiments/run_revision_cnc.py`, and `run_policy_transplant_pipeline` in
`experiments/run_revision_transplant.py`.

For benchmark and sparse jobs, solvers sharing the same full distribution hash
share one reanalysis of their stored sample group. Every solver's saved summary
is compared separately. CNC samples are reconstructed from the complete frozen
support-deviation closure in its original profile order. Transplant arms share
the original replacement-label ordering and replication index.

## What the reported errors measure

For each manufacturer, the original code first forms the distribution-weighted
payoff in every replication. It computes that manufacturer's payoff standard
error as its sample standard deviation divided by the square root of 2,000.
The reported maximum payoff standard error is the largest across manufacturers.
The replacement-gain standard error is calculated from paired, per-replication
replacement gains; its reported maximum is across the library's replacement
checks. Neither quantity is the standard error of the maximum gap itself.

The original bootstrap gap-bound diagnostics are reproduced for traceability.
Reproducing those fields does not introduce a new threshold or alter the
manuscript's reporting scope. The transplant replay also reproduces the paired
cross-minus-control and cross-minus-recomputed gap effects and their original
intervals.

## Validation and comparisons

The offline reader checks manifest membership, byte count and SHA-256 for every
read. The adapter additionally checks the sample artifact's result-declared
hash, frozen distribution hashes, exact array shapes and dtypes, label order,
finite samples, and CNC chunk/closure completeness. Derived CNC and transplant
digests are compared with the saved result digests.

Numerical tolerances are fixed before replay: absolute tolerance `1e-9` and
relative tolerance `1e-10`, using Python `math.isclose`. Strings, hashes,
integers, and booleans match exactly. The replay compares every statistic
returned by the original functions. Additional saved execution metadata and
historical threshold-classification flags are not newly estimated; frozen
distribution and sample-group identities are checked explicitly.

No mismatch changes a tolerance, distribution, seed, or input. Each job retains
its recalculated fields and every difference. A missing or corrupt input is
recorded as a job error, while independent jobs may finish. Unexpected
nonfinite calculated values are explicit tagged values in valid JSON and fail
the comparison. Reports use `allow_nan=False`.

`audit-jobs/<original-job-id>.json` contains per-job evidence;
`audit-summary.json` contains the full replay status and counts. Existing
reports are not overwritten. Both the CLI and the direct Python API require an
output directory outside the input dataset.

Bootstrap calculations use the pinned NumPy/SciPy environment. Platform-level
floating-point warnings, if any, must be retained in the execution log; a pass
requires finite statistics and agreement within the fixed tolerances. No
warning suppression or alternate formula is applied by the adapter.

## Adapter tests

Deterministic synthetic fixtures cover benchmark q deduplication, CNC closure
reconstruction, three-arm effects and intervals, malformed labels, missing
chunks, corrupt payloads, numerical mismatch retention, exact discrete fields,
nonfinite output evidence, job selection, partial failures, output isolation,
and refusal to overwrite reports. They retain the 2,000/5,000 settings. The
small synthetic games test the adapter; complete paper-data replay is a
separate validation step.
