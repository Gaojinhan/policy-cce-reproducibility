from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from cmfg_cce.evaluation.empirical_game import build_empirical_game
from cmfg_cce.evaluation.full_audit import audit_sparse_result_on_full_game
from cmfg_cce.evaluation.rollout import estimates_to_records
from cmfg_cce.experiments.common import LEAD_TIME_GRID, MARKUP_GRID, run_solver
from cmfg_cce.experiments.run_oracle_baselines import (
    evaluate_full_tensor,
    load_yaml,
    make_solver_cache,
    solver_seed,
    sparse_solution_record,
)
from cmfg_cce.envs.toy import config_from_mapping
from cmfg_cce.policies.policy_library import build_policy_library_for_mechanism
from cmfg_cce.solvers.cce_lp import solve_full_tensor_min_epsilon


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def run_config(config_path: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    raw = load_yaml(config_path)
    env_config = config_from_mapping(raw)
    matrix = raw["no_regret_round_ablation"]
    base_solver_cfg = raw.get("solver", {})
    workers = int(base_solver_cfg.get("workers", raw.get("workers", 1)))
    markup_grid = tuple(raw.get("policies", {}).get("markup_grid", MARKUP_GRID))
    lead_time_grid = tuple(raw.get("policies", {}).get("lead_time_grid", LEAD_TIME_GRID))
    solver_payoff_mode = str(matrix.get("solver_payoff_mode", "independent_resimulate"))
    run_budget_label = str(raw.get("run_budget_label", "no_regret_round_ablation"))
    solvers = tuple(matrix.get("solvers", ["MWU-PolicyTrace", "RegretMatching-PolicyTrace"]))
    rounds_list = [int(rounds) for rounds in matrix["rounds"]]
    records: list[dict[str, Any]] = []
    payoff_rows: list[dict[str, Any]] = []

    for n_agents in matrix["N_values"]:
        for k in matrix["policies_per_agent"]:
            for seed in matrix["seeds"]:
                seeds = solver_seed(int(seed))
                for mechanism in raw["mechanisms"]:
                    policies = build_policy_library_for_mechanism(mechanism, int(k))
                    policy_ids = tuple(policies.keys())
                    oracle_cache, estimates, full_eval_time = evaluate_full_tensor(
                        mechanism=mechanism,
                        policies=policies,
                        config=env_config,
                        n_agents=int(n_agents),
                        seeds=seeds,
                        n_rollouts=int(matrix["rollouts_per_profile"]),
                        markup_grid=markup_grid,
                        lead_time_grid=lead_time_grid,
                        workers=workers,
                    )
                    payoff_rows.extend(estimates_to_records(list(estimates), mechanism, int(n_agents)))
                    game = build_empirical_game(list(estimates), policy_ids)
                    full_gap_oracle = solve_full_tensor_min_epsilon(game, ucb=False)
                    full_ucb_oracle = solve_full_tensor_min_epsilon(game, ucb=True)

                    for rounds in rounds_list:
                        for solver_name in solvers:
                            solver_cfg = dict(base_solver_cfg)
                            solver_cfg["mwu_rounds"] = rounds
                            solver_cfg["regret_matching_rounds"] = rounds
                            solver_cache = make_solver_cache(
                                mechanism=mechanism,
                                policies=policies,
                                config=env_config,
                                n_agents=int(n_agents),
                                seeds=seeds,
                                n_rollouts=int(matrix["rollouts_per_profile"]),
                                markup_grid=markup_grid,
                                lead_time_grid=lead_time_grid,
                                workers=workers,
                                oracle_estimates=estimates,
                                payoff_mode=solver_payoff_mode,
                            )
                            solver_cache.reset_access_log()
                            start = time.perf_counter()
                            result = run_solver(
                                solver_name,
                                solver_cache,
                                policy_ids,
                                solver_cfg,
                                seed=int(seed) + 1009 * rounds + 7919 * len(records),
                                metadata={
                                    "experiment": raw["experiment"],
                                    "env_version": raw.get("env_version", ""),
                                    "N": int(n_agents),
                                    "K": int(k),
                                    "seed": int(seed),
                                    "run_budget_label": run_budget_label,
                                },
                            )
                            solver_runtime = time.perf_counter() - start
                            solver_stats = solver_cache.access_stats()
                            solver_payoff_eval_time = float(solver_cache.eval_time_seconds)
                            solver_stats["solver_payoff_eval_time"] = solver_payoff_eval_time
                            solver_stats["solver_compute_time"] = max(0.0, solver_runtime - solver_payoff_eval_time)
                            audit_start = time.perf_counter()
                            full_audit = audit_sparse_result_on_full_game(game, result)
                            full_audit_compute_time = time.perf_counter() - audit_start
                            record = sparse_solution_record(
                                raw["experiment"],
                                mechanism,
                                result,
                                full_audit,
                                full_gap_oracle,
                                full_ucb_oracle,
                                solver_cache,
                                game,
                                int(n_agents),
                                int(k),
                                int(seed),
                                env_config.horizon,
                                solver_runtime,
                                solver_stats,
                                None,
                                full_eval_time,
                                full_audit_compute_time,
                                run_budget_label,
                            )
                            record["rounds"] = int(rounds)
                            record["experiment_family"] = "no_regret_round_ablation"
                            records.append(record)
                            print(
                                f"{raw['experiment']} N={n_agents} K={k} seed={seed} "
                                f"{mechanism} {solver_name} rounds={rounds}: "
                                f"full_audit_gap={full_audit.cce_gap_nominal:.3g} "
                                f"full_audit_ucb={full_audit.cce_gap_ucb:.3g} "
                                f"profiles={solver_cache.access_stats()['solver_required_profile_count']}/{game.n_profiles}",
                                flush=True,
                            )
    return records, pd.DataFrame(payoff_rows)


def write_outputs(records: list[dict[str, Any]], payoff_table: pd.DataFrame, output_dir: Path) -> None:
    raw_dir = output_dir / "raw"
    table_dir = output_dir / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    write_json(raw_dir / "cce_results_no_regret_round_ablation.json", records)
    flat = pd.DataFrame(
        [
            {
                **{key: value for key, value in record.items() if key not in {"support", "mechanism_metrics"}},
                **record["mechanism_metrics"],
            }
            for record in records
        ]
    )
    flat.to_csv(table_dir / "no_regret_round_ablation.csv", index=False)
    payoff_table.to_csv(raw_dir / "payoff_cache_no_regret_round_ablation.csv", index=False)
    summary_cols = [
        "cce_gap_full_audit",
        "cce_gap_ucb_full_audit",
        "solver_required_profile_count",
        "profile_saving_ratio",
        "support_size",
        "solver_time",
        "runtime_seconds",
    ]
    flat.groupby(["N", "K", "rounds", "solver"], as_index=False)[summary_cols].mean().to_csv(
        table_dir / "no_regret_round_ablation_summary.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cmfg_cce/configs/no_regret_round_ablation.yaml")
    parser.add_argument("--output-dir", default="outputs/no_regret_round_ablation")
    args = parser.parse_args()
    records, payoff_table = run_config(Path(args.config))
    write_outputs(records, payoff_table, Path(args.output_dir))


if __name__ == "__main__":
    main()
