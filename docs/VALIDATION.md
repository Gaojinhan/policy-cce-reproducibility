# Validation history

Validation date: 2026-09-11. This is a saved-data and packaging check, not a rerun of the formal experiments.

## GitHub release validation: version 0.3.0

The bounded simulator demo and complete saved-data reconstruction have passed on macOS ARM64 and in a Linux/AMD64 Docker container on GitHub Actions. Container computation runs with networking disabled and data mounted read-only. The data-v1 release archive was downloaded, checksum-verified and unpacked by the Linux runner. The versioned code release includes `release-validation.json` with the exact tested commit, Actions run, test counts and output hashes.

The sections below retain the earlier local checks and their original artifact identities. Statements about work pending at those stages are historical. Public visibility, reuse licensing and a full formal-campaign rerun are outside the completed private release. GitHub is the only publication destination; no Zenodo archive or DOI is planned.

## Scientific parity

- Independent audit reanalysis: **276 / 276 jobs matched**, no failed jobs and no mismatching fields. This includes 72 solver benchmarks, 12 sparse-scalability cases, 96 CNC cases and 96 three-arm transplant audits.
- CNC outcomes: all **96 cases** passed raw and weighted-vector hash checks. **7,697 compared fields** matched the archived references and per-case results; the largest absolute numerical difference was approximately `1.14e-13`.
- Tolerances were fixed before the run: absolute `1e-9`, relative `1e-10`; discrete fields and hashes match exactly. The 5,000-draw bootstrap settings and original seeds were retained.
- The selected input manifest contains **6,161 payload files**, **838,732,517 bytes**, and **588 formal jobs** plus the separate calibration records. Its SHA-256 is `a0b5b46789c1a879a55ad82d57e4ccb07dc2a5c66b73438ab2d27127b4406ff7`.

The local research preparation directory keeps the detailed reports under `offline-validation-v1/`, including every per-job audit, the CNC reconstruction, command logs and the final validation manifest. This document is a readable summary; those reports retain the full precision and method settings.

## Safety checks

In the first statistical stage, all **85 tests passed**: 60 reader/source-integrity checks, 14 audit-adapter checks and 11 CNC outcome checks. Warnings are retained in the test log. This historical count describes version 0.1.0, before the table/figure workflow was added.

The tests use synthetic inputs to check missing files, hash mismatches, original job and marker identity, closed job scope, unsafe paths, symlinks, corrupt arrays and overlapping output directories. They also exercise paired resampling, nonlinear outcome reconstruction, fixed distribution checks, preserved mismatches and partial job failures. See [TEST_PLAN.md](TEST_PLAN.md).

The scientific source is checked against the file-level frozen manifest. The local adapters and tests are separate from `cmfg_cce/`; they do not patch original solver or simulator code.

## Independent wheel installation

The wheel `policy_cce_reproducibility-0.1.0-py3-none-any.whl` was built and installed non-editably in a second temporary virtual environment, from a local wheel directory using `--no-index`. Validation then ran from an unrelated working directory with Python `-I` and cloud credential environment variables removed. Python socket connection/DNS operations were blocked during the saved-data run; no attempt was recorded. This is a process-level socket guard, not a claim of an OS-wide network sandbox.

Both packages resolved inside the new environment. All 105 installed scientific files, including the 23 YAML resources and the two formal configuration paths, matched their frozen hashes. The full 276-job audit and 96-case outcome reconstruction passed again using that installed copy. All 6,161 input payloads were verified before and after and remained unchanged. The cloud-storage client was not imported.

Tested wheel size: 404,153 bytes. SHA-256: `a7557b57cea51619ef38b39ff2ee7a92b52f16338619f21e4ed848503465f08c`. This identifies the local test artifact, not a public release or a platform-independent validation claim. The package itself is pure Python; the numerical dependency wheels used here are macOS ARM64 builds.

## Environment and warnings

The tested environment is macOS ARM64 with Python 3.12.14 and the package versions in `metadata/validated-environment.json`. The commands set the BLAS thread environment to one per statistical worker; the full-data validation used two workers.

The local NumPy build emitted divide/overflow/invalid warnings at some matrix multiplications, including tests with small finite inputs. All checked results remained finite and passed the fixed-tolerance comparisons. These warnings and dependency deprecation warnings remain in the logs. Their cause was not established, and they were not suppressed or treated as a reason to alter the calculations. Linux/AMD64 validation was still pending at this local stage; it is covered by the later release checks above.

## Result-display stage: version 0.2.0

The `paper` command recomputed the complete 276-job audit and 96-case CNC outcome set, then generated **13 numerical tables and four result figures**. All displayed table values, captions, labels and checked units matched the display contract for manuscript commit `45d3a588672b54799a3c6393938b275a67dddc3b`. The 216 retained parameter-result records also passed their 2,428 comparison checks. Figures use the underlying numerical evidence rather than copied figure images. Structural/architecture diagrams are excluded from this scope.

All **167 tests passed**, including the optional full-data benchmark join test; 50 warnings are retained. Additional tests check table precision and units, CI signs, result identity, caption and label drift, missing and extra figure files, and exclusive creation of output files. The 17-page standalone proof compiled with no overfull-box or undefined-reference warnings, and every rendered page was visually inspected. This proof is a layout check, not a revised manuscript.

The full-precision evidence and generated-file hashes are in `reproducibility-prep/paper-render-v1/final/paper-report.json`; the complete test log and XML are in the same stage directory. Version 0.2.0 uses `metadata/paper-layout-v2.json`. The earlier display contract and first statistical-stage validation records remain unchanged. An installed-wheel run has its own evidence and is not implied by the checkout tests or the historical version 0.1.0 installation above.

PDF byte equality across environments is not required: PDF timestamps, installed fonts and backend metadata can differ. Numerical inputs, plotted values, table output and original scientific identities are checked separately. This local-stage validation platform was macOS ARM64.

## What this does not verify

The saved-data validation does not rerun payoff simulation, LP solves, training or original Runtime measurements. Runtime and parameter-robustness results are verified as archived records. The separate bounded demo performs a small synthetic simulation and LP solve; it does not reproduce a formal paper experiment. Full formal-job execution, public-data review and additional reuse licensing remain outside this private release.

Saved statistical parity does not fill missing actual-executor provenance or establish that a historical scientific result should be interpreted more broadly than the fixed case and policy library.
