# Policy-space CCE reproducibility

Frozen scientific code, saved experimental data and instructions for the policy-space CCE manuscript. This package reconstructs **13 numerical tables and four result figures** and provides a small simulator example. Structural/architecture diagrams are excluded.

Repository: [Gaojinhan/policy-cce-reproducibility](https://github.com/Gaojinhan/policy-cce-reproducibility). This is an author-controlled **private** repository. Code and documentation are in Git; data are attached to the **data-v1** GitHub Release. No Zenodo archive or DOI is used. See [NOTICE.md](NOTICE.md) before sharing.

## Install

Use Python 3.12. These shell commands work on macOS, Linux and Windows through WSL2. GitHub CLI access is needed to clone/download while private; the scientific commands need no cloud credentials.

```sh
gh repo clone Gaojinhan/policy-cce-reproducibility
cd policy-cce-reproducibility
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install .
```

## Try the small example

```sh
.venv/bin/policy-cce-repro demo --output outputs/demo-001
```

This runs a small synthetic game, computes distributions with the full LP and DSS-CCE, and checks their policy-replacement gaps with separate evaluation samples. Settings, seeds, actual runtime and results are saved. It is a functional example, not a paper experiment. See [DEMO.md](docs/DEMO.md).

## Reproduce the paper tables and figures

Download and extract the data once:

```sh
gh release download data-v1 --repo Gaojinhan/policy-cce-reproducibility \
  --pattern policy-cce-data-v1.tar.gz --dir downloads
.venv/bin/policy-cce-repro unpack \
  --archive downloads/policy-cce-data-v1.tar.gz --output data
```

The archive is about 547 MB, expanding to about 846 MB including its manifest. `unpack` checks the pinned release SHA-256, safe paths and every original data file. It refuses existing output directories. If `data/offline-inputs-v1` is already present and verified, skip the download/unpack steps.

```sh
.venv/bin/policy-cce-repro paper \
  --data data/offline-inputs-v1 --output outputs/paper-001 --workers 2
```

This recomputes all 276 saved-sample gap audits and 96 CNC outcome cases before building the displays. It keeps the solved distributions and original statistical settings fixed; it does not rerun training or replace any formal result.

| Output | Contents |
| --- | --- |
| `tables/` | 13 LaTeX tables, preserving labels and displayed precision |
| `figures/` | Four result figures, each as PDF and PNG |
| `statistics/` | Full-precision recalculations and comparisons |
| `paper-report.json` | Input/output hashes, numerical evidence, fonts and display checks |
| `paper-preview.tex` | Optional standalone proof using the paper's table/figure numbers |

Every run needs a **new output directory**. Missing input or a numerical mismatch produces an error, without GCS fallback or silent repair.

## Docker

The image defaults to help, not a worker, and uses no private image registry. Docker must already be running.

```sh
docker build -t policy-cce-repro:0.3.0 .
mkdir -p outputs
docker run --rm --network none --read-only --tmpfs /tmp \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$(pwd)/outputs,target=/out" \
  policy-cce-repro:0.3.0 demo --output /out/demo-docker-001
```

After downloading and unpacking the data above:

```sh
docker run --rm --network none --read-only --tmpfs /tmp \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$(pwd)/data/offline-inputs-v1,target=/data,readonly" \
  --mount "type=bind,source=$(pwd)/outputs,target=/out" \
  policy-cce-repro:0.3.0 paper --data /data --output /out/paper-docker-001 --workers 2
```

Inputs are read-only, networking is disabled during computation, and no host credential directory is mounted. These commands do not start an old experiment container. The official Python base is pinned by digest. Reporting dependencies are pinned in `requirements.lock`; no proprietary host font is bundled.

## Other commands

```sh
.venv/bin/policy-cce-repro verify --data data/offline-inputs-v1
.venv/bin/policy-cce-repro audits --data data/offline-inputs-v1 \
  --output outputs/audits-001 --workers 2
.venv/bin/policy-cce-repro outcomes --data data/offline-inputs-v1 \
  --output outputs/outcomes-001 --workers 2
```

`audits` accepts repeated `--job-id ORIGINAL_JOB_ID` options for a subset. Numerical comparison uses absolute tolerance `1e-9` plus relative tolerance `1e-10`; hashes and discrete identities match exactly. The original 5,000 bootstrap draws and seeds are retained. See [STATISTICAL_METHODS.md](docs/STATISTICAL_METHODS.md).

The optional preview needs an external TeX installation. Inside the output folder run `pdflatex -no-shell-escape paper-preview.tex`. Required packages: `newtxtext`, `newtxmath`, `booktabs`, `array`, `graphicx`, `xcolor`, `caption`, `amsmath` and `geometry`. TeX is not required to generate the individual result PDFs. Fonts/backend metadata can change PDF bytes; numerical parity and layout are checked separately.

## Source and data scope

The data contain **588 formal result jobs and 24 calibration jobs**, with 6,161 payload files (838,732,517 bytes). Raw saved evaluations needed for statistical reconstruction are included. Data selection is bound to manuscript commit `ab181cb144a64e61dc6ee2c8b4fa0321c0747dc6`; display formatting is separately bound to `45d3a588672b54799a3c6393938b275a67dddc3b` in `metadata/paper-layout-v2.json`. Later manuscript edits do not silently change these records.

`cmfg_cce/` contains 105 byte-exact scientific files: 82 Python modules and 23 YAML configurations. New readers, display builders and the demo live in `policy_cce_repro/`. Tests check the frozen copy against `metadata/source-copy-manifest.json`.

The selected displays exclude N=12 and withdrawn analyses. Historical configs retain the original larger campaign; they are not default tasks. The six non-pure examples are a benchmark subset. Definition-only tables and architecture diagrams are not generated; the mixed definition/result parameter-settings table is included.

Logical node IDs and actual executor identities are separate. The 96 selected Runtime records lack actual-executor provenance, which remains unknown. Historical numerical compatibility code is kept as inert provenance, not a global startup patch. See [DATA.md](docs/DATA.md) and [EXPERIMENTS.md](docs/EXPERIMENTS.md) for exact scope, the output map and advanced formal-rerun requirements. The full campaign is not rerun as a release test.

## Tests and validation

```sh
.venv/bin/python -m pip install '.[test]'
.venv/bin/python -m pytest
```

Default tests use synthetic inputs. The optional full-data benchmark test requires `POLICY_CCE_TEST_DATA` and `POLICY_CCE_TEST_AUDIT_REPORT`. GitHub Actions runs the tests, small example and offline Docker demo. Its manual `full_data` option also downloads the private Release data and rebuilds every selected display in the container.

Actual outcomes are recorded in [Actions](https://github.com/Gaojinhan/policy-cce-reproducibility/actions) and versioned release evidence. [VALIDATION.md](docs/VALIDATION.md) preserves earlier local-stage checks. [THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md) records dependency notices and the private-use boundary.
