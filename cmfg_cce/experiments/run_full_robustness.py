from __future__ import annotations

import argparse

from cmfg_cce.experiments.common import run_sparse_matrix, write_experiment_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cmfg_cce/configs/full_robustness.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    records, payoff_table = run_sparse_matrix(args.config, "full_robustness")
    write_experiment_outputs(records, payoff_table, args.output_dir, "full_robustness", "full_robustness.csv")
    print(f"Full robustness experiment wrote {len(records)} result records.")


if __name__ == "__main__":
    main()

