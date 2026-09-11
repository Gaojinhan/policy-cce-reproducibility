# Run a small synthetic example

The demo runs the simulator, solves a small empirical policy game, and checks
the resulting distributions with new simulations. It needs no paper dataset,
cloud account, or credentials.

```sh
policy-cce-repro demo --output /path/to/new-demo-output
```

Choose a directory that does not exist. The command refuses to overwrite an
earlier run. Keep demo output outside source directories and paper datasets.

## What runs

The scenario has three manufacturers, three policies (A1–A3), and eight order
arrival steps under the M1 first-price mechanism. All 27 joint profiles fit in
a tiny payoff tensor. Each evaluated training profile uses four simulations.
The workflow uses the original simulator and solvers with a separate, fixed
demo configuration; the scientific source files are unchanged.

1. FullTensor-CCE-LP evaluates all training profiles and solves the full LP.
2. DSS-CCE starts with its own empty payoff cache and performs a bounded sparse
   search. Both solvers use the same training random streams.
3. Each output distribution is fixed. Its training gap is checked using both
   the complete payoff tensor and its support-deviation closure.
4. New random streams generate eight simulations for each joint profile.
   These samples check the same fixed distributions. The full-tensor,
   closure-only, and paired-sample gap calculations must agree.

The manufacturer population stays fixed between training and evaluation.
Within either stage, common replication indices use common random streams
across profiles. Training and evaluation streams differ. All demo seeds are
derived from `policy-cce-repro-synthetic-demo-v1`; no campaign configuration,
job record, or saved experiment result is loaded.

## Read the output

- `demo-config.json`: the complete settings and seed integers.
- `demo-FullTensor-CCE-LP-solver.json` and `demo-DSS-CCE-solver.json`: fixed
  distributions, solver status, evaluated-profile counts, and actual timings.
- `demo-samples.npz`: the complete training mean-payoff tensor, its confidence
  radii and selection objectives, DSS's evaluated profile indices, and fresh
  per-profile return samples.
- `demo-result.json`: both gap checks, independent sample statistics, matching
  before/after distribution hashes, sample-file checksum, and timings.

`status: "pass"` means that the software checks succeeded. It is not a claim
that this small sample gives a precise estimate of the true equilibrium gap.
Positive independent gaps are retained. The original audit routine also
produces standard errors and confidence-bound diagnostics, using only 100
bootstrap draws here. These demo statistics are not paper results.

`solver_wall_seconds` includes each solver's own training simulations and
computation from an empty cache. The later training verification and fresh
evaluation have separate timings. Small runs are affected by startup costs,
so these timings should not be used to compare solver performance.

The workload is fixed at one simulation worker and at most 432 episodes of
eight steps each. It has no flags for enlarging the experiment. Running the
same demo again in a new directory should reproduce its numerical outputs
under the pinned environment; elapsed times vary. If execution fails, the
new directory is kept with `demo-error.json` and any completed outputs.

For Python use, call `policy_cce_repro.demo.run_demo(output_dir)`. It returns
the same dictionary saved in `demo-result.json`.
