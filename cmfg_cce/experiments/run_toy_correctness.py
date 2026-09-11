from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from cmfg_cce.envs.toy import config_from_mapping
from cmfg_cce.evaluation.empirical_game import EmpiricalGame, build_empirical_game, enumerate_profiles
from cmfg_cce.evaluation.rollout import RolloutSeeds, estimate_profile, estimates_to_records
from cmfg_cce.policies.policy_library import build_price_policy_library
from cmfg_cce.solvers.cce_lp import CceSolution, compute_cce_gap_from_deviation_closure
from cmfg_cce.solvers.cg_cce import solve_cg_cce_exhaustive
from cmfg_cce.solvers.full_tensor_cce import solve_full_tensor_cce
from cmfg_cce.solvers.sad_cce import SadConfig, solve_sad_cce


MARKUP_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


def weighted_metrics(game: EmpiricalGame, q) -> dict[str, float]:
    keys = set().union(*(metrics.keys() for metrics in game.metrics))
    out: dict[str, float] = {}
    for key in keys:
        out[key] = float(sum(prob * game.metrics[idx].get(key, 0.0) for idx, prob in enumerate(q)))
    return out


def solution_record(
    solution: CceSolution,
    game: EmpiricalGame,
    mechanism_id: str,
    n_agents: int,
    k: int,
    horizon: int,
    evaluated_profile_count: int,
    rollouts_per_profile: int,
    runtime_seconds: float,
    payoff_eval_time: float,
    solver_time: float,
    run_budget_label: str,
    seed: int,
) -> dict:
    full_tensor_size = len(game.profiles)
    support = [
        {"profile": list(profile), "prob": prob}
        for profile, prob in zip(solution.support_profiles, solution.support_probabilities, strict=True)
    ]
    metrics = weighted_metrics(game, solution.q)
    max_dev = dict(solution.max_deviation or {})
    if "gain_nominal" in max_dev and "gain_ucb" not in max_dev:
        max_dev["gain_ucb"] = max_dev["gain_nominal"]
    return {
        "experiment": "toy_correctness",
        "mechanism": mechanism_id,
        "solver": solution.solver,
        "pricing_mode": solution.pricing_mode,
        "full_optimality_certified": solution.full_optimality_certified,
        "N": n_agents,
        "K": k,
        "seed": seed,
        "horizon": horizon,
        "support_size": solution.support_size,
        "evaluated_profile_count": evaluated_profile_count,
        "new_profiles_evaluated": evaluated_profile_count,
        "cache_hit_count": 0,
        "full_tensor_size": full_tensor_size,
        "profile_saving_ratio": evaluated_profile_count / max(1, full_tensor_size),
        "cce_gap_nominal": solution.cce_gap_nominal,
        "cce_gap_ucb": getattr(solution, "cce_gap_ucb", solution.cce_gap_nominal),
        "objective_value": solution.objective_value,
        "runtime_seconds": runtime_seconds,
        "rollout_episode_total": evaluated_profile_count * rollouts_per_profile,
        "rollout_steps_total": evaluated_profile_count * rollouts_per_profile * horizon,
        "rollouts_initial": rollouts_per_profile,
        "rollouts_min": rollouts_per_profile,
        "rollouts_mean": float(rollouts_per_profile),
        "rollouts_max": rollouts_per_profile,
        "lp_solve_time": solver_time,
        "pricing_time": 0.0 if solution.pricing_mode is None else solver_time,
        "payoff_eval_time": payoff_eval_time,
        "run_budget_label": run_budget_label,
        "support_expansion_rounds": solution.pricing_iterations or 0,
        "active_sampling_rounds": 0,
        "policy_expansion_rounds": 0,
        "audit_mode": "full_tensor_deviation_closure",
        "pricing_iterations": solution.pricing_iterations,
        "max_reduced_cost": solution.max_reduced_cost,
        "max_deviation_agent": max_dev.get("agent"),
        "max_deviation_policy": max_dev.get("policy"),
        "max_deviation_gain": max_dev.get("gain_nominal"),
        "max_deviation_gain_ucb": max_dev.get("gain_ucb"),
        "support": support,
        "max_deviation": max_dev,
        "mechanism_metrics": metrics,
    }


def run_config(config_path: Path) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    env_config = config_from_mapping(raw_config)
    toy = raw_config["toy"]
    solver_cfg = raw_config.get("solver", {})
    run_budget_label = str(raw_config.get("run_budget_label", "paper_run"))
    output_records: list[dict] = []
    solver_rows: list[dict] = []
    payoff_rows: list[dict] = []

    for n_agents in toy["N_values"]:
        for k in toy["policies_per_agent"]:
            policies = build_price_policy_library(int(k))
            policy_ids = tuple(policies.keys())
            profiles = enumerate_profiles(policy_ids, int(n_agents))
            for seed in toy["seeds"]:
                seeds = RolloutSeeds(
                    type_seed=int(seed),
                    order_seed=1000 + int(seed),
                    tie_break_seed=2000 + int(seed),
                    rollout_replication_seed=3000 + int(seed),
                )
                for mechanism_id in raw_config["mechanisms"]:
                    estimates = []
                    eval_start = time.perf_counter()
                    for profile in profiles:
                        estimates.append(
                            estimate_profile(
                                profile=profile,
                                mechanism_id=mechanism_id,
                                policies=policies,
                                config=env_config,
                                n_agents=int(n_agents),
                                seeds=seeds,
                                n_rollouts=int(toy["rollouts_per_profile"]),
                                markup_grid=MARKUP_GRID,
                            )
                        )
                    eval_seconds = time.perf_counter() - eval_start
                    payoff_rows.extend(estimates_to_records(estimates, mechanism_id, int(n_agents)))
                    game = build_empirical_game(estimates, policy_ids)

                    full_start = time.perf_counter()
                    full_solution = solve_full_tensor_cce(game)
                    full_runtime = time.perf_counter() - full_start

                    cg_start = time.perf_counter()
                    cg_solution = solve_cg_cce_exhaustive(game)
                    cg_runtime = time.perf_counter() - cg_start

                    sad_config = SadConfig(
                        initial_support_size=int(solver_cfg.get("sad_initial_support_size", 8)),
                        max_support_size=int(solver_cfg.get("sad_max_support_size", 48)),
                        support_add_batch_size=int(solver_cfg.get("sad_support_add_batch_size", 8)),
                        max_rounds=int(solver_cfg.get("sad_max_rounds", 8)),
                        target_gap=float(solver_cfg.get("target_gap", 1e-8)),
                    )
                    sad_start = time.perf_counter()
                    sad_solution = solve_sad_cce(game, sad_config)
                    sad_runtime = time.perf_counter() - sad_start

                    evaluated_profile_count = len(estimates)
                    output_records.extend(
                        [
                            solution_record(
                                full_solution,
                                game,
                                mechanism_id,
                                int(n_agents),
                                int(k),
                                env_config.horizon,
                                evaluated_profile_count,
                                int(toy["rollouts_per_profile"]),
                                eval_seconds + full_runtime,
                                eval_seconds,
                                full_runtime,
                                run_budget_label,
                                int(seed),
                            ),
                            solution_record(
                                cg_solution,
                                game,
                                mechanism_id,
                                int(n_agents),
                                int(k),
                                env_config.horizon,
                                evaluated_profile_count,
                                int(toy["rollouts_per_profile"]),
                                eval_seconds + cg_runtime,
                                eval_seconds,
                                cg_runtime,
                                run_budget_label,
                                int(seed),
                            ),
                            solution_record(
                                sad_solution,
                                game,
                                mechanism_id,
                                int(n_agents),
                                int(k),
                                env_config.horizon,
                                evaluated_profile_count,
                                int(toy["rollouts_per_profile"]),
                                eval_seconds + sad_runtime,
                                eval_seconds,
                                sad_runtime,
                                run_budget_label,
                                int(seed),
                            ),
                        ]
                    )
                    closure_gap = compute_cce_gap_from_deviation_closure(
                        game,
                        cg_solution.support_profiles,
                        cg_solution.support_probabilities,
                    )
                    solver_rows.append(
                        {
                            "N": int(n_agents),
                            "K": int(k),
                            "mechanism": mechanism_id,
                            "full_LP_objective": full_solution.objective_value,
                            "CG_objective": cg_solution.objective_value,
                            "objective_gap": abs(full_solution.objective_value - cg_solution.objective_value),
                            "full_LP_eps": full_solution.cce_gap_nominal,
                            "CG_eps": cg_solution.cce_gap_nominal,
                            "CG_closure_eps": closure_gap,
                            "profiles_evaluated": evaluated_profile_count,
                            "runtime": eval_seconds + full_runtime + cg_runtime,
                        }
                    )
                    print(
                        f"{mechanism_id} N={n_agents} K={k}: "
                        f"full_gap={full_solution.cce_gap_nominal:.3g}, "
                        f"cg_gap={cg_solution.cce_gap_nominal:.3g}, "
                        f"sad_gap={sad_solution.cce_gap_nominal:.3g}, "
                        f"profiles={evaluated_profile_count}"
                    )

    return output_records, pd.DataFrame(solver_rows), pd.DataFrame(payoff_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cmfg_cce/configs/toy.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    table_dir = output_dir / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    records, solver_table, payoff_table = run_config(Path(args.config))
    result_path = raw_dir / "cce_results_toy.json"
    table_path = table_dir / "solver_correctness.csv"
    payoff_path = raw_dir / "payoff_cache_toy.parquet"

    result_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    solver_table.to_csv(table_path, index=False)
    for column in payoff_table.columns:
        if payoff_table[column].dtype == "object":
            payoff_table[column] = payoff_table[column].fillna("").astype(str)
    payoff_table.to_parquet(payoff_path, index=False, engine="pyarrow")
    pd.read_parquet(payoff_path, engine="pyarrow")

    print(f"Wrote {result_path}")
    print(f"Wrote {table_path}")
    print(f"Wrote {payoff_path}")


if __name__ == "__main__":
    main()
