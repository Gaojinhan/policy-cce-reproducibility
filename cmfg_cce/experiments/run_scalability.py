from __future__ import annotations

import argparse

from cmfg_cce.experiments.common import run_sparse_matrix, save_scalability_figures, write_experiment_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cmfg_cce/configs/scalability.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    records, payoff_table = run_sparse_matrix(args.config, "scalability")
    flat = write_experiment_outputs(records, payoff_table, args.output_dir, "scalability", "scalability.csv")
    save_scalability_figures(flat, args.output_dir)
    print(f"Scalability experiment wrote {len(records)} result records.")


if __name__ == "__main__":
    main()

