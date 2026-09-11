from __future__ import annotations

import argparse

from cmfg_cce.experiments.common import run_sparse_matrix, write_experiment_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cmfg_cce/configs/pair_repair.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    records, payoff_table, paired_delta_table = run_sparse_matrix(
        args.config,
        "pair_repair",
        return_paired_delta=True,
    )
    write_experiment_outputs(
        records,
        payoff_table,
        args.output_dir,
        "pair_repair",
        "pair_repair.csv",
        paired_delta_table=paired_delta_table,
    )
    print(f"PAIR/REPAIR experiment wrote {len(records)} result records.")


if __name__ == "__main__":
    main()
