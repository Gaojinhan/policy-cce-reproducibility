# Experiments and reproduction routes

This repository separates three tasks: rebuilding the paper from saved data,
running a small demonstration, and rerunning the original experiments.
They have different costs and validation status.

| Route | What it computes | What a successful run establishes |
| --- | --- | --- |
| Saved-data reproduction | Statistics and displays from the archived distributions and samples | The selected paper results can be reconstructed from those inputs |
| Bounded demonstration | A small synthetic game, new simulation, and new solver outputs | The demonstrated simulator/solver path runs at the stated small settings |
| Formal rerun | Original job configurations, new training and independent evaluations | Requires separate job-level validation; the full formal campaign has not been rerun as a release test |

The saved-data route has been tested both in the source environment and from
an installed wheel. See [VALIDATION.md](VALIDATION.md) for the specific tested
stage. A demo is not a replacement for a formal result or a paper Runtime
measurement. Its own settings and validation record must accompany it.

## Current paper map

The selected archive contains 588 formal jobs and 24 MWU calibration jobs.
Its selection is recorded in `manifest.json` and
`metadata/paper-selected-jobs.json` inside the separate data directory.
The current display contract is `metadata/paper-layout-v2.json` in this code
package. It targets manuscript commit
`45d3a588672b54799a3c6393938b275a67dddc3b`; the data-selection record retains
its earlier manuscript commit. These are separate versioned records.

| Archived family | Formal jobs | Current paper output |
| --- | ---: | --- |
| `solver_benchmark` | 72 | Tables 4–7 and Figure 5: gaps, uncertainty, support, and computational comparisons |
| `solver_benchmark_runtime` | 72 | Runtime/profile measurements used with the benchmark displays |
| `scalability_exact_runtime` | 12 | Table 8: the N=7, J=6 timing extension |
| `scalability_sparse` | 12 | Figure 6: N=8 and N=10, J=8 independent gaps |
| `scalability_sparse_runtime` | 12 | Figure 6: corresponding archived Runtime measurements |
| `cnc_main` | 96 | Table 11, Tables 13–16, and Figures 7–8: gaps, mechanism effects, operating conditions, and policy composition |
| `policy_transplant` | 96 | Table 12: same-mechanism control, cross-mechanism transfer, and recomputation |
| `parameter_robustness` | 216 | Tables 17–18: parameter definitions, all matched profit comparisons, and combined-pressure examples |

The six non-pure cases in Table 7 belong to the 72 benchmark jobs; they are
not six additional mixed-challenge experiments. The selected displays do
not include N=12, separate mixed-challenge jobs, or the removed library and
selection-sensitivity displays. Historical source/configuration files retain
those families for provenance. Their presence is not an instruction to run
or add them to the current paper.

The numerical command produces 13 tables and four result figures. Figures
1–4 are structural diagrams and are excluded. Definition-only Tables 1–3
and 9–10 are also outside that command. Table 17 includes both definitions
and computed comparisons, so it is retained.

## 1. Rebuild from saved data

After installation, use a new output directory outside the data directory:

```sh
policy-cce-repro verify --data /path/to/offline-inputs-v1
policy-cce-repro paper \
  --data /path/to/offline-inputs-v1 \
  --output /path/to/new-paper-output --workers 2
```

`paper` repeats the 276 saved-sample gap audits and the 96-case CNC outcome
analysis before producing the displays. Parameter tables read all 216
validated archived result records. They recompute displayed means and sign
counts from the saved metric estimates; they do not repeat parameter-game
simulation or resampling. Runtime values are original measurements, not
the time spent rebuilding tables.

The solved distributions remain fixed. Reanalysis uses the stored 2,000
independent gap-evaluation replications and the original 5,000 bootstrap
draws. CNC operating-outcome comparisons use 500 saved replications.
Read [AUDIT_METHOD_PROVENANCE.md](AUDIT_METHOD_PROVENANCE.md),
[OUTCOME_METHOD_PROVENANCE.md](OUTCOME_METHOD_PROVENANCE.md), and
[STATISTICAL_METHODS.md](STATISTICAL_METHODS.md) for formulas, matching,
aggregation, and comparison tolerances.

This route needs no GCP credentials and has no cloud download fallback.
Missing inputs or numerical differences are reported, not repaired.

## 2. A bounded demonstration

Use [DEMO.md](DEMO.md) and its recorded validation status for the dedicated
demo. A demo must declare its smaller game, simulation
counts, random streams, output directory, and solver checks explicitly.
It must keep its outputs separate from the formal data.

The dedicated command `policy-cce-repro demo --output /path/to/new-demo-output`
has been tested on its fixed three-manufacturer, three-policy M1 example.
It uses separate empty training caches for FullTensor-CCE-LP and DSS-CCE,
then evaluates their fixed distributions with new simulations. The settings
and small-workload limits are listed in [DEMO.md](DEMO.md); they are not the
formal benchmark settings.

The historical `cmfg_cce.experiments.run_toy_correctness` CLI is not a
single-case bounded demo. Its stock `cmfg_cce/configs/toy.yaml` contains
multiple sizes, seeds, and mechanisms. Its final Parquet export also
requires `pyarrow`, which is not among this package's pinned runtime
dependencies. Do not use that old command as the release smoke test.

## 3. Formal rerun: configuration and execution contract

This section documents the frozen entry points and required inputs.
It is not a claim that all formal jobs have been rerun successfully in the
packaged environment. Full reruns can take substantial compute time.

### Original identities

| Record | SHA-256 |
| --- | --- |
| Formal campaign identity | `cc8e7f81c117c512cdb18f9993bc4b67960d7ac8a85d3329969ad550ff4b2ff9` |
| Frozen experiment matrix | `8f77eaa2704659d9c4b6ef2f4f95ee1fa21f57695c8e63e1e6954afb9efa111a` |
| Frozen scientific source identity | `eb0143548676863326355ff2e94e81e6081735a0be952e25ab8e2a190dada900` |
| Original `campaign-formal.json` file | `2d2ff5730202f3f3fa8b4051f33b629271bfa84a2248978c91326e31317dcb97` |
| Original `mwu_selection.json` file | `50838fd60da47a324f53ba8c0b2755670f72d1c3ee4802e11215710abf73629c` |

The full historical campaign contains 1,744 jobs. It is larger than the
paper-selected data. Loading it for identity checks does not authorize or
require running every listed job.

The preserved MWU selection is already included at
`data/offline-inputs-v1/metadata/mwu_selection.json` after the data archive is
unpacked at the repository's documented location. It freezes
`eta-1__constant__explore-0__burn-0`; its internal selection digest is
`d5630cd38581937146a10035b19054abb65adbd2296559008b0e4e8d776bb913`.
The file is listed in the data manifest even though it is not in the
reader's named `references` mapping. Use it directly; do not select a new
configuration from the paper results.

The matrix comes from `RevisionFullV1Matrix()` in
`cmfg_cce.experiments.revision_full_v1_spec`. Its `matrix_hash` hashes its
canonical definition. `CampaignManifest.load(path)` checks the full
campaign's internal digest and dependencies. The loader
`load_global_mwu_selection(path, matrix=matrix, smoke=False)` checks the
selection digest, matrix identity, and membership in the frozen tuning
grid. These read-only checks were exercised during packaging. The hashes
are integrity records, not a separate public-key signature service.

### Before running a formal job

1. Verify the data manifest and the 105-file source copy. Obtain the exact
   full `campaign-formal.json` specified above from the `data-v1`
   GitHub Release into `downloads/campaign-formal.json`. The code package's
   `metadata/formal-rerun-inputs.json` records this extra byte-exact input.
   It is not one of the four contextual metadata files in the current
   `offline-inputs-v1` archive.
2. Choose an exact original job ID from the declared paper selection and
   look it up in the full campaign. Preserve its logical `node_id`, slot,
   seed namespaces, configuration hash, and dependencies.
3. Use the pinned environment and inspect the chosen family's dependency
   requirements. The saved-data archive is not a full historical training
   cache or worker-state backup. A dependency directory must have the
   structure expected by the family runner, not just an arbitrary folder
   containing result JSON files.
4. Reserve a new rerun directory, a compute budget, and an actual executor.
   Record the host/architecture, OS, Python/dependencies, worker count,
   source hash, and time separately from the original job identity.
5. Review any job-specific numerical-repair provenance. The historical v9
   file in `metadata/compatibility/` is inert in this package. It must not
   be silently enabled as `sitecustomize`. Saved-data agreement does not
   establish that a fresh solve will reproduce a repair-assisted execution.

### Read-only setup check

Run from the repository root, after installing the package and downloading
the setup/data assets. This loads the full historical manifest but
checks only that it agrees with the selected archive. It starts no worker.

```sh
python - <<'PY'
import hashlib
from pathlib import Path
from cmfg_cce.orchestration.manifest import CampaignManifest, SourceManifest
from cmfg_cce.experiments.revision_full_v1_spec import RevisionFullV1Matrix
from cmfg_cce.experiments.run_revision_full_v1 import load_global_mwu_selection
from policy_cce_repro.offline import Dataset
import json

data = Dataset('data/offline-inputs-v1')
manifest_path = Path('downloads/campaign-formal.json')
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == \
    '2d2ff5730202f3f3fa8b4051f33b629271bfa84a2248978c91326e31317dcb97'
campaign = CampaignManifest.load(manifest_path)
matrix = RevisionFullV1Matrix()
assert campaign.matrix_sha256 == matrix.matrix_hash == data.manifest['matrix_sha256']
assert campaign.campaign_sha256 == data.manifest['campaign_sha256']
source = SourceManifest.from_payload(json.loads(Path('metadata/frozen-source-manifest.json').read_text()))
copied = json.loads(Path('metadata/source-copy-manifest.json').read_text())
assert len(copied['files']) == 105
for item in copied['files']:
    path = Path(item['path'])
    assert path.stat().st_size == item['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], str(path)
assert campaign.source_sha256 == source.source_sha256 == copied['source_sha256'] == data.manifest['source_sha256']
data.read_json('metadata/mwu_selection.json')  # Verify the archived bytes first.
selection, _ = load_global_mwu_selection(
    data.root / 'metadata/mwu_selection.json', matrix=matrix, smoke=False)
jobs = {job.job_id: job for job in campaign.jobs}
assert set(data.jobs) <= set(jobs)
print('Setup identities verified; no experiment started.', selection.config_id)
PY
```

The historical source manifest has 109 entries. This package preserves the
105 scientific Python/YAML files byte-for-byte. The other four original
paths were `AGENTS.md`, `pyproject.toml`, the infrastructure dependency lock,
and the instance scale-to-zero shell script. The historical lock is retained
at `metadata/historical-requirements.lock`; the new package has its own
packaging instructions and no cloud self-stop deployment. Therefore the
check above validates the original manifest's identity and the 105 copied
files, not an unchanged 109-file historical worker directory. Calling the
old worker's whole-tree validator on this package is not a supported check.

### One-job interface

The frozen direct entry point accepts exactly one original ID:

```sh
CMFG_CAMPAIGN_SHA256=cc8e7f81c117c512cdb18f9993bc4b67960d7ac8a85d3329969ad550ff4b2ff9 \
python -m cmfg_cce.experiments.run_revision_campaign \
  --job-id ORIGINAL_JOB_ID \
  --matrix-hash 8f77eaa2704659d9c4b6ef2f4f95ee1fa21f57695c8e63e1e6954afb9efa111a \
  --state-dir reruns/new-run/state \
  --output-dir reruns/new-run/result \
  --mwu-selection data/offline-inputs-v1/metadata/mwu_selection.json \
  --dependency-dir reruns/prepared-dependencies \
  --workers 1
```

This is an advanced invocation template, not a completed rerun recipe.
Resolve `ORIGINAL_JOB_ID` against the full campaign first. The direct
entry point dispatches general, CNC, tuning, and transplant jobs; it does
not itself load the external campaign manifest or create a full
orchestration completion record. Supplying no `--smoke` requests the formal
family settings. `--smoke` reduces those settings and marks the result as
a smoke, so it cannot reproduce a formal result.

Do not substitute the historical generic worker for this one-job interface.
That worker filters families rather than exact IDs and has no complete
physical-executor provenance override. The repository's setup
assets are not permission to restart an old cloud deployment.

### What a rerun must preserve and report

Keep training, independent gap evaluation, and operating-outcome evaluation
separate, with their original streams and common-random-number matching.
Preserve every failure and reversal. Never use independent audit outcomes
to tune or repair a distribution. Runtime attempts must start from empty
payoff caches and finish as whole attempts; resumed fragments cannot be
combined into a solver wall-clock measurement. Verification time remains
separate from solver runtime.

New files are rerun evidence linked to original jobs, not replacements for
the archived canonical results. Do not write them into the supplied data
directory or the original campaign's cloud paths. Historical logical
assignments do not identify the actual machine: actual executor provenance
is unknown for the 96 selected Runtime records and remains unknown. A new
rerun must record its own actual executor honestly.
