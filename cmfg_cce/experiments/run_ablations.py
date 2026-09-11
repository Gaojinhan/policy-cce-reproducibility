from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from cmfg_cce.experiments.common import load_yaml, run_sparse_matrix, write_experiment_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cmfg_cce/configs/ablations.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    base = load_yaml(args.config)
    variants = base.pop("variants")
    all_records = []
    payoff_tables = []
    with TemporaryDirectory() as tmpdir:
        for name, override in variants.items():
            cfg = deepcopy(base)
            cfg["experiment"] = f"ablation_{name}"
            cfg["solver"].update(override.get("solver", {}))
            cfg["ablations"].update(override.get("ablations", {}))
            cfg_path = Path(tmpdir) / f"{name}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            records, payoff_table = run_sparse_matrix(cfg_path, "ablations")
            for record in records:
                record["ablation_variant"] = name
            all_records.extend(records)
            payoff_tables.append(payoff_table)
    payoff_table = payoff_tables[0].__class__.concat(payoff_tables, ignore_index=True) if False else None
    import pandas as pd

    payoff_table = pd.concat(payoff_tables, ignore_index=True)
    write_experiment_outputs(all_records, payoff_table, args.output_dir, "ablations", "ablation.csv")
    print(f"Ablation experiment wrote {len(all_records)} result records.")


if __name__ == "__main__":
    main()

