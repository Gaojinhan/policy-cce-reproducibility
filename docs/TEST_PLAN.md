# Offline reproduction test plan

## Objective and boundaries

Verify that the publication reader accepts only declared, hash-checked local inputs and that reporting keeps the original scientific identities and statistical recipes. A successful byte check is not a reproduced statistic; a reproduced statistic is not a rerun of the simulator or solver.

Reader safety tests use small synthetic JSON/NPZ files in pytest temporary directories. They require no real campaign data, cloud account, credentials or network. Their job IDs and hashes are clearly artificial. A separate source-integrity test reads the 105 frozen scientific files and their packaged manifests without importing or executing those files. Real-data tests use a separate user-supplied data root and write only to a new output directory outside it.

## Risk-based test map

| Risk | Deterministic automated check | Level |
| --- | --- | --- |
| Undeclared or missing input silently substituted | Reject unlisted paths and missing listed files; no network/process fallback | Unit / filesystem integration |
| Corrupt bytes accepted because a filename matches | Reject changed size or SHA; verify every listed input | Unit / filesystem integration |
| Input root escaped | Reject URI, absolute path, parent traversal, file symlink and symlinked directory | Unit / filesystem integration |
| Malformed manifest broadens authority | Reject wrong schema and duplicate declared paths; object/reference maps cannot bypass path checks | Unit |
| A checksum-valid but malformed scientific record leaks an unhandled exception | Rehashed invalid JSON, non-object JSON, missing result/provenance/marker fields and malformed artifacts all raise `DataError` | Unit |
| Packaging silently edits frozen scientific code/configuration | Verify all 105 copied file hashes/sizes and membership against the frozen source manifest | Source regression |
| Data from another job/campaign counted | Validate result identity, complete/non-smoke status, campaign/matrix and original marker logical node/slot | Unit |
| Unsafe NPZ payload interpreted as Python objects | Numeric arrays round-trip; object-dtype/pickled members fail with `allow_pickle=False` | Unit |
| Reporting overwrites canonical inputs | Reject output equal to, below or above the data root, including symlink aliases; allow a distinct sibling without creating it | Unit / filesystem integration |
| Correct imports depend on author's cloud setup | Safety suite blocks network and subprocess seams; later CLI smoke runs outside the research directory with no cloud credentials | Integration |
| Statistical result drifts during refactoring | Recompute from fixed samples with original bootstrap/aggregation rules; compare unrounded values to high-precision references | Real-data regression |
| Runtime mixed with another fitted distribution | Verify original job/result links and q hashes before combining runtime with audit metrics | Scientific integration |

No test should weaken a frozen scientific assertion merely to make a different numerical result pass. Diagnose a mismatch and preserve both the original evidence and the new diagnostic.

## Automated safety suite

Run after installing this package and its test dependencies:

```sh
python -m pytest tests/test_offline_safety.py -q
```

The suite tests public `Dataset` behavior rather than private implementation details. Network/subprocess calls are blocked in the synthetic fixture; core hashing, path resolution, JSON parsing and NPZ loading are real. Errors are checked as `DataError`, with descriptive test cases, rather than matching incidental message formatting.

## Offline workflow check

1. Install into a clean environment and use an absolute data-root path.
2. Change to a temporary directory outside both the source repository and data root. Clear author-specific `PYTHONPATH` and cloud credential variables for the check.
3. Run the documented verify/report CLI command, with a fresh output path that does not overlap input storage. Network and account access are not prerequisites; a missing local file must fail explicitly.
4. Confirm the command resolves packaged modules/configuration rather than the original research tree.
5. Compare generated unrounded statistics with the reference values. Inspect rendered tables/figures separately for units, signs, precision, captions and layout.
6. Rehash the original data manifests/files and confirm they are unchanged.

Record the command, software versions, data-manifest digest, output digest and pass/fail report. An end-to-end success is claimed only after these commands actually run. The default safety suite does not run the full formal campaign, cloud orchestration or a fresh solver.

## Scientific regression boundaries

- CNC operational outcomes use saved 500-rollout records. Reconstruct ratios and allocation concentration inside each paired bootstrap draw; preserve the original fixed-q strata and pairing.
- Independent equilibrium auditing uses distinct 2,000-rollout return arrays. Recompute the specified gap, gain/payoff error and diagnostic bounds without fitting or repairing q.
- Transplant checks retain all three arms and matching source/target distributions; do not substitute the outcome-evaluation records for audit returns.
- Runtime and audit inputs must refer to the same frozen q. Hardware-dependent runtime need not reproduce bit-for-bit, and an offline statistical calculation does not constitute a fresh runtime measurement.
- Parameter tables retain all required settings and both positive and negative comparisons. Rounded manuscript values are not statistical inputs.
- Original logical assignment and actual executor provenance remain separate. A null executor record stays unknown; do not infer the physical machine from `node_id`.

## Frozen code and optional cloud facilities

The frozen scientific tree also contains historical orchestration modules. Their presence is not a promise that the entire codebase has no cloud-capable functions. The publication reader/reporting path must not instantiate them or require their optional dependencies.

Source inspection found a lightweight `cmfg_cce` initializer; orchestration initialization imports chunk/manifest types, not a cloud client. `cmfg_cce/orchestration/storage.py` imports the Google storage library only when its Python GCS backend is explicitly constructed, while a different backend invokes `gcloud` when its methods are used. These are preserved historical facilities, not default publication entry points. The sampled statistical modules import NumPy/SciPy and scientific helpers, not an authentication provider. Import and CLI tests still need to verify the installed publication path in practice.

Do not delete or rewrite frozen cloud-capable source to make a security statement true. Keep public defaults offline, document optional historical modules, and verify the actual supported entry points.

## Exit criteria and remaining limits

- All synthetic safety tests pass, with no network/process fallback and no changes to real inputs.
- Installed-package CLI works from an unrelated working directory using only declared local inputs.
- Real-data verification covers exact scope, hashes, job identity, stage and q links; each claimed statistical reproduction has a stored comparison report.
- No current inference relies only on rounded tables. Preserved numerical shims and any observed activation remain explicit provenance.
- Visual checks, rights/licenses, redaction and public release approval are separate gates.

The scenarios above are acceptance targets; only the checks explicitly recorded below are claimed as completed here. Concurrent adversarial file replacement, resource-exhaustion archives, exhaustive platform coverage and a full codebase security review are outside the initial safety-suite guarantee and must not be claimed as validated.

## Recorded checks — 2026-09-11

The assigned safety/source-integrity suite was run after the loader fixes: **60 tests passed**. It first exposed two error-boundary defects: a JSON-array manifest raised `AttributeError`, and a plain NPY file presented as NPZ raised `TypeError`. The loader was hardened; the tests retain both regression cases rather than weakening their assertions. Ten further tests cover deliberately rehashed malformed result/provenance/marker records, so they reach content validation rather than failing only at a checksum. The source-integrity test confirms all 105 scientific code/configuration files retain their frozen bytes.

Environment: Python 3.12.14, NumPy 2.1.3, SciPy 1.15.3 and pytest 8.3.4 in this package's virtual environment.

The installed `policy-cce-repro verify` entry point was independently run from `/tmp`, outside the source and data roots, with `PYTHONPATH`, `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_CLOUD_PROJECT` removed for that process. It passed for **6,161 files / 588 formal jobs / 838,732,517 declared bytes**, with input-manifest digest `a0b5b46789c1a879a55ad82d57e4ccb07dc2a5c66b73438ab2d27127b4406ff7`. This command did not run the simulator, solver or bootstrap and did not write input files.

That smoke test used the currently installed local package. It establishes working-directory independence for the tested entry point, **not** a separately built wheel's completeness or a comprehensive OS-level network denial test. Network/process fallback was directly blocked in the synthetic unit suite. Wheel installation and visual inspection have separate execution records and are not implied by these safety checks.

The completed real-data reports were also inspected, without rerunning those calculations in this safety audit:

- `reproducibility-prep/offline-validation-v1/audits/audit-summary.json` records 276 selected/completed/matched audit jobs, zero failed jobs and zero mismatched fields. Its scope is saved-sample reanalysis with no simulator, solver, distribution selection or repair.
- `reproducibility-prep/offline-validation-v1/outcomes/outcomes-recomputed.json` records 96 CNC cases, 7,697 comparisons, zero mismatches, and 96 verified raw-vector plus 96 weighted-vector hashes. The maximum absolute difference is approximately `1.14e-13`, within the declared `atol=1e-9`, `rtol=1e-10`.

These paths identify the local preparation run's reports, relative to the research workspace; they are not runtime dependencies of this code package. They support the stated statistical-reanalysis scope, not a claim that fresh experiments or all historical campaign jobs were rerun.

## Result-display checks: version 0.2.0

The complete suite now passes **167 tests**, including a real-data benchmark join check. The paper-specific tests cover Runtime/q identity, preserved numerical evidence, percentage versus percentage-point units, signed confidence-interval endpoints, table precision, labels/captions and exact figure-file scope. They reject a missing expected PDF/PNG or an added structural diagram even when the generator's returned dictionary has the expected labels.

The full `paper` workflow separately recalculates the declared audit and outcome data and checks all 13 table displays against the versioned manuscript contract. All four result figures and all 13 tables were rendered in a standalone 17-page proof and visually inspected. Source-data integrity, numerical parity and layout are distinct checks; successful visual inspection alone does not establish scientific parity.
