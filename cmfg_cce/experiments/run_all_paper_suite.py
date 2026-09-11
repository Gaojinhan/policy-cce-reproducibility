from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    commands = [
        ["cmfg_cce.experiments.run_toy_correctness", "--config", "cmfg_cce/configs/toy.yaml"],
        ["cmfg_cce.experiments.run_core_mechanisms", "--config", "cmfg_cce/configs/core.yaml"],
        ["cmfg_cce.experiments.run_scalability", "--config", "cmfg_cce/configs/scalability.yaml"],
        ["cmfg_cce.experiments.run_full_robustness", "--config", "cmfg_cce/configs/full_robustness.yaml"],
        ["cmfg_cce.experiments.run_ablations", "--config", "cmfg_cce/configs/ablations.yaml"],
    ]
    for module, *extra in commands:
        subprocess.run(
            [sys.executable, "-u", "-m", module, *extra, "--output-dir", args.output_dir],
            check=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    print(f"All paper-suite experiments wrote outputs under {args.output_dir}.")


if __name__ == "__main__":
    main()
