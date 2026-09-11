from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cmfg_cce.envs.toy import ToyEnvConfig, config_from_mapping
from cmfg_cce.evaluation.empirical_game import EmpiricalGame, build_empirical_game
from cmfg_cce.evaluation.full_audit import audit_distribution_on_full_game, audit_sparse_result_on_full_game
from cmfg_cce.evaluation.payoff_cache import PayoffCache, full_tensor_size, profile_space
from cmfg_cce.evaluation.rollout import RolloutEstimate, RolloutSeeds, estimates_to_records
from cmfg_cce.experiments.common import LEAD_TIME_GRID, MARKUP_GRID, run_solver
from cmfg_cce.policies.policy_library import build_policy_library_for_mechanism
from cmfg_cce.solvers.cce_lp import CceSolution, solve_full_cce_lp
from cmfg_cce.solvers.cg_cce import solve_cg_cce_exhaustive
from cmfg_cce.solvers.sparse_cce import SparseCceResult


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def solver_seed(seed: int) -> RolloutSeeds:
    return RolloutSeeds(
        type_seed=seed,
        order_seed=1000 + seed,
        tie_break_seed=2000 + seed,
        rollout_replication_seed=3000 + seed,
        outside_seed=4000 + seed,
        availability_seed=5000 + seed,
    )


def weighted_metrics(game: EmpiricalGame, q) -> dict[str, float]:
    keys = set().union(*(metrics.keys() for metrics in game.metrics))
    return {
        key: float(sum(prob * game.metrics[idx].get(key, 0.0) for idx, prob in enumerate(q)))
        for key in keys
    }


def evaluate_full_tensor(
    mechanism: str,
    policies: dict,
    config: ToyEnvConfig,
    n_agents: int,
    seeds: RolloutSeeds,
    n_rollouts: int,
    markup_grid: tuple[float, ...],
    lead_time_grid: tuple[float, ...],
    workers: int,
) -> tuple[PayoffCache, tuple[RolloutEstimate, ...], float]:
    policy_ids = tuple(policies.keys())
    profiles = tuple(profile_space(policy_ids, n_agents))
    cache = PayoffCache(
        mechanism_id=mechanism,
        policies=policies,
        config=config,
        n_agents=n_agents,
        seeds=seeds,
        n_rollouts=n_rollouts,
        markup_grid=markup_grid,
        lead_time_grid=lead_time_grid,
        workers=workers,
    )
    start = time.perf_counter()
    cache.ensure(profiles)
    elapsed = time.perf_counter() - start
    estimates = tuple(cache.estimates[profile] for profile in profiles)
    return cache, estimates, elapsed


def make_solver_cache(
    mechanism: str,
    policies: dict,
    config: ToyEnvConfig,
    n_agents: int,
    seeds: RolloutSeeds,
    n_rollouts: int,
    markup_grid: tuple[float, ...],
    lead_time_grid: tuple[float, ...],
    workers: int,
    oracle_estimates: tuple[RolloutEstimate, ...] | None = None,
    payoff_mode: str = "independent_resimulate",
) -> PayoffCache:
    if payoff_mode == "oracle_lookup":
        raise ValueError(
            "oracle_lookup is forbidden for solver timing; solvers must start from an empty payoff cache."
        )
    if payoff_mode not in {"independent_resimulate", "resimulate"}:
        raise ValueError(f"Unsupported solver_payoff_mode: {payoff_mode}")
    cache = PayoffCache(
        mechanism_id=mechanism,
        policies=policies,
        config=config,
        n_agents=n_agents,
        seeds=seeds,
        n_rollouts=n_rollouts,
        markup_grid=markup_grid,
        lead_time_grid=lead_time_grid,
        workers=workers,
    )
    return cache


def oracle_solution_record(
    experiment: str,
    mechanism: str,
    solution: CceSolution,
    game: EmpiricalGame,
    n_agents: int,
    k: int,
    seed: int,
    horizon: int,
    full_eval_time: float,
    lp_time: float,
    rollouts_per_profile: int,
    run_budget_label: str,
    full_audit_eval_time: float | None = None,
    full_audit_compute_time: float = 0.0,
    certificate_status: str = "full_tensor_ground_truth",
    gap_to_full_tensor_optimum: float = 0.0,
    ucb_gap_to_full_tensor_optimum: float = 0.0,
) -> dict[str, Any]:
    metrics = weighted_metrics(game, solution.q)
    max_dev = dict(solution.max_deviation or {})
    tensor_size = full_tensor_size(n_agents, k)
    return {
        "experiment": experiment,
        "mechanism": mechanism,
        "solver": solution.solver,
        "pricing_mode": solution.pricing_mode,
        "certificate_status": certificate_status,
        "full_optimality_certified": solution.full_optimality_certified,
        "N": n_agents,
        "K": k,
        "seed": seed,
        "horizon": horizon,
        "support_size": solution.support_size,
        "evaluated_profile_count": tensor_size,
        "solver_required_profile_count": tensor_size,
        "full_tensor_size": tensor_size,
        "profile_saving_ratio": 1.0,
        "cce_gap_nominal": solution.cce_gap_nominal,
        "cce_gap_ucb": solution.cce_gap_ucb,
        "cce_gap_full_audit": solution.cce_gap_nominal,
        "cce_gap_ucb_full_audit": solution.cce_gap_ucb,
        "gap_to_full_tensor_optimum": gap_to_full_tensor_optimum,
        "ucb_gap_to_full_tensor_optimum": ucb_gap_to_full_tensor_optimum,
        "objective_value": solution.objective_value,
        "runtime_seconds": full_eval_time + lp_time,
        "full_tensor_eval_time": full_eval_time,
        "full_audit_eval_time": full_eval_time if full_audit_eval_time is None else full_audit_eval_time,
        "full_audit_compute_time": full_audit_compute_time,
        "full_audit_runtime_seconds": (full_eval_time if full_audit_eval_time is None else full_audit_eval_time)
        + full_audit_compute_time,
        "solver_time": lp_time,
        "solver_compute_time": lp_time,
        "solver_payoff_eval_time": full_eval_time,
        "rollout_episode_total": tensor_size * int(rollouts_per_profile),
        "rollout_steps_total": tensor_size * int(rollouts_per_profile) * horizon,
        "run_budget_label": run_budget_label,
        "audit_mode": "full_tensor",
        "max_deviation_agent": max_dev.get("agent"),
        "max_deviation_policy": max_dev.get("policy"),
        "max_deviation_gain": max_dev.get("gain_nominal"),
        "max_deviation_gain_ucb": max_dev.get("gain_ucb"),
        "support": [
            {"profile": list(profile), "prob": prob}
            for profile, prob in zip(solution.support_profiles, solution.support_probabilities, strict=True)
        ],
        "mechanism_metrics": metrics,
    }


def sparse_solution_record(
    experiment: str,
    mechanism: str,
    result: SparseCceResult,
    full_audit: CceSolution,
    full_gap_oracle: CceSolution,
    full_ucb_oracle: CceSolution,
    cache: PayoffCache,
    game: EmpiricalGame,
    n_agents: int,
    k: int,
    seed: int,
    horizon: int,
    solver_runtime: float,
    solver_stats: dict[str, float],
    solver_full_tensor_eval_time: float | None,
    full_audit_eval_time: float,
    full_audit_compute_time: float,
    run_budget_label: str,
) -> dict[str, Any]:
    metrics = weighted_metrics(game, full_audit.q)
    max_dev = dict(full_audit.max_deviation or {})
    tensor_size = full_tensor_size(n_agents, k)
    solver_profile_count = int(solver_stats["solver_required_profile_count"])
    solver_rollout_episodes = int(solver_stats["solver_required_rollout_episode_total"])
    solver_compute_time = float(solver_stats.get("solver_compute_time", solver_runtime))
    solver_payoff_eval_time = float(solver_stats.get("solver_payoff_eval_time", 0.0))
    return {
        "experiment": experiment,
        "mechanism": mechanism,
        "solver": result.solver,
        "pricing_mode": result.pricing_mode,
        "certificate_status": "full_tensor_audited_output_distribution",
        "full_optimality_certified": result.full_optimality_certified,
        "N": n_agents,
        "K": k,
        "seed": seed,
        "horizon": horizon,
        "support_size": result.support_size,
        "evaluated_profile_count": solver_profile_count,
        "solver_required_profile_count": solver_profile_count,
        "global_cache_profile_count": cache.evaluated_profile_count,
        "full_tensor_size": tensor_size,
        "profile_saving_ratio": solver_profile_count / max(1, tensor_size),
        "cce_gap_nominal": result.cce_gap_nominal,
        "cce_gap_ucb": result.cce_gap_ucb,
        "cce_gap_full_audit": full_audit.cce_gap_nominal,
        "cce_gap_ucb_full_audit": full_audit.cce_gap_ucb,
        "gap_to_full_tensor_optimum": full_audit.cce_gap_nominal - full_gap_oracle.cce_gap_nominal,
        "ucb_gap_to_full_tensor_optimum": full_audit.cce_gap_ucb - full_ucb_oracle.cce_gap_ucb,
        "objective_value": full_audit.objective_value,
        "support_local_objective_value": result.objective_value,
        "runtime_seconds": solver_runtime,
        "full_tensor_eval_time": solver_full_tensor_eval_time,
        "full_audit_eval_time": full_audit_eval_time,
        "full_audit_compute_time": full_audit_compute_time,
        "full_audit_runtime_seconds": full_audit_eval_time + full_audit_compute_time,
        "solver_time": solver_compute_time,
        "solver_compute_time": solver_compute_time,
        "solver_payoff_eval_time": solver_payoff_eval_time,
        "rollout_episode_total": solver_rollout_episodes,
        "rollout_steps_total": int(solver_stats["solver_required_rollout_steps_total"]),
        "paired_delta_pair_count": int(solver_stats.get("paired_delta_pair_count", 0)),
        "paired_delta_sample_count": int(solver_stats.get("paired_delta_sample_count", 0)),
        "paired_delta_rollout_episode_total": int(solver_stats.get("paired_delta_rollout_episode_total", 0)),
        "run_budget_label": run_budget_label,
        "audit_mode": "support_search_then_full_tensor_audit",
        "max_deviation_agent": max_dev.get("agent"),
        "max_deviation_policy": max_dev.get("policy"),
        "max_deviation_gain": max_dev.get("gain_nominal"),
        "max_deviation_gain_ucb": max_dev.get("gain_ucb"),
        "support": [
            {"profile": list(profile), "prob": prob}
            for profile, prob in zip(result.support_profiles, result.support_probabilities, strict=True)
        ],
        "mechanism_metrics": metrics,
    }


def run_config(config_path: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    raw = load_yaml(config_path)
    env_config = config_from_mapping(raw)
    matrix = raw["oracle_baselines"]
    solver_cfg = raw.get("solver", {})
    run_budget_label = str(raw.get("run_budget_label", "oracle_small"))
    workers = int(solver_cfg.get("workers", raw.get("workers", 1)))
    markup_grid = tuple(raw.get("policies", {}).get("markup_grid", MARKUP_GRID))
    lead_time_grid = tuple(raw.get("policies", {}).get("lead_time_grid", LEAD_TIME_GRID))
    solver_payoff_mode = str(matrix.get("solver_payoff_mode", "independent_resimulate"))
    if solver_payoff_mode not in {"independent_resimulate", "resimulate"}:
        raise ValueError(
            f"Unsupported solver_payoff_mode for independent runtime accounting: {solver_payoff_mode}"
        )
    records: list[dict[str, Any]] = []
    payoff_rows: list[dict[str, Any]] = []

    for n_agents in matrix["N_values"]:
        for k in matrix["policies_per_agent"]:
            for seed in matrix["seeds"]:
                seeds = solver_seed(int(seed))
                for mechanism in raw["mechanisms"]:
                    policies = build_policy_library_for_mechanism(mechanism, int(k))
                    policy_ids = tuple(policies.keys())
                    audit_cache, audit_estimates, audit_eval_time = evaluate_full_tensor(
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
                    payoff_rows.extend(
                        estimates_to_records(list(audit_estimates), mechanism, int(n_agents))
                    )
                    audit_game = build_empirical_game(list(audit_estimates), policy_ids)

                    full_objective_start = time.perf_counter()
                    full_objective_oracle = solve_full_cce_lp(audit_game)
                    full_objective_time = time.perf_counter() - full_objective_start
                    records.append(
                        oracle_solution_record(
                            raw["experiment"],
                            mechanism,
                            full_objective_oracle,
                            audit_game,
                            int(n_agents),
                            int(k),
                            int(seed),
                            env_config.horizon,
                            audit_eval_time,
                            full_objective_time,
                            int(matrix["rollouts_per_profile"]),
                            run_budget_label,
                            certificate_status="full_tensor_objective_cce_lp",
                        )
                    )

                    for solver_name in matrix["solvers"]:
                        if solver_name in {
                            "FullTensor-CCE-LP",
                            "FullTensor-MinGap-CCE-LP",
                            "FullTensor-MinUCB-CCE-LP",
                        }:
                            continue
                        if solver_name == "ExhaustiveCG-CCE":
                            cg_cache, cg_estimates, cg_eval_time = evaluate_full_tensor(
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
                            cg_game = build_empirical_game(list(cg_estimates), policy_ids)
                            cg_start = time.perf_counter()
                            cg_solution = solve_cg_cce_exhaustive(cg_game)
                            cg_solver_time = time.perf_counter() - cg_start
                            solver_runtime = cg_eval_time + cg_solver_time
                            audit_start = time.perf_counter()
                            full_audit = audit_distribution_on_full_game(
                                audit_game,
                                cg_solution.support_profiles,
                                cg_solution.support_probabilities,
                                solver_name="ExhaustiveCG-CCE-FullTensorAudit",
                            )
                            full_audit_compute_time = time.perf_counter() - audit_start
                            result_like = SparseCceResult(
                                solver="ExhaustiveCG-CCE",
                                support_profiles=cg_solution.support_profiles,
                                support_probabilities=cg_solution.support_probabilities,
                                objective_value=cg_solution.objective_value,
                                cce_gap_nominal=cg_solution.cce_gap_nominal,
                                cce_gap_ucb=cg_solution.cce_gap_ucb,
                                max_deviation=cg_solution.max_deviation or {},
                                status=cg_solution.status,
                                certificate_status="exhaustive_column_generation",
                                pricing_mode=cg_solution.pricing_mode,
                                full_optimality_certified=True,
                                support_expansion_rounds=cg_solution.pricing_iterations or 0,
                            )
                            solver_stats = {
                                "solver_required_profile_count": cg_game.n_profiles,
                                "solver_required_rollout_episode_total": cg_game.n_profiles * int(matrix["rollouts_per_profile"]),
                                "solver_required_rollout_steps_total": cg_game.n_profiles
                                * int(matrix["rollouts_per_profile"])
                                * env_config.horizon,
                                "paired_delta_pair_count": 0,
                                "paired_delta_sample_count": 0,
                                "paired_delta_rollout_episode_total": 0,
                                "solver_compute_time": cg_solver_time,
                                "solver_payoff_eval_time": cg_eval_time,
                            }
                            record_cache = cg_cache
                            solver_full_tensor_eval_time: float | None = cg_eval_time
                        else:
                            solver_cache = PayoffCache(
                                mechanism_id=mechanism,
                                policies=policies,
                                config=env_config,
                                n_agents=int(n_agents),
                                seeds=seeds,
                                n_rollouts=int(matrix["rollouts_per_profile"]),
                                markup_grid=markup_grid,
                                lead_time_grid=lead_time_grid,
                                workers=workers,
                            )
                            solver_cache.reset_access_log()
                            start = time.perf_counter()
                            result_like = run_solver(
                                solver_name,
                                solver_cache,
                                policy_ids,
                                solver_cfg,
                                seed=int(seed) + 7919 * len(records),
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
                            solver_stats["solver_compute_time"] = max(
                                0.0,
                                solver_runtime - solver_payoff_eval_time,
                            )
                            audit_start = time.perf_counter()
                            full_audit = audit_sparse_result_on_full_game(audit_game, result_like)
                            full_audit_compute_time = time.perf_counter() - audit_start
                            record_cache = solver_cache
                            solver_full_tensor_eval_time = None
                        records.append(
                            sparse_solution_record(
                                raw["experiment"],
                                mechanism,
                                result_like,
                                full_audit,
                                full_objective_oracle,
                                full_objective_oracle,
                                record_cache,
                                audit_game,
                                int(n_agents),
                                int(k),
                                int(seed),
                                env_config.horizon,
                                solver_runtime,
                                solver_stats,
                                solver_full_tensor_eval_time,
                                audit_eval_time,
                                full_audit_compute_time,
                                run_budget_label,
                            )
                        )
                        print(
                            f"{raw['experiment']} N={n_agents} K={k} seed={seed} {mechanism} {solver_name}: "
                            f"full_audit_gap={full_audit.cce_gap_nominal:.3g} "
                            f"full_audit_ucb={full_audit.cce_gap_ucb:.3g} "
                            f"profiles={int(solver_stats['solver_required_profile_count'])}/{audit_game.n_profiles} "
                            f"runtime={solver_runtime:.2f}s audit_compute={full_audit_compute_time:.2f}s"
                        )
    return records, pd.DataFrame(payoff_rows)


def flatten_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {key: value for key, value in record.items() if key not in {"support", "mechanism_metrics"}}
        for key, value in record["mechanism_metrics"].items():
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(records: list[dict[str, Any]], payoff_table: pd.DataFrame, output_dir: Path) -> None:
    raw_dir = output_dir / "raw"
    table_dir = output_dir / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    write_json(raw_dir / "cce_results_oracle_baselines.json", records)
    flat = flatten_records(records)
    flat.to_csv(table_dir / "oracle_baselines.csv", index=False)
    payoff_table.to_csv(raw_dir / "payoff_cache_oracle_baselines.csv", index=False)
    solver_cols = [
        "cce_gap_full_audit",
        "cce_gap_ucb_full_audit",
        "gap_to_full_tensor_optimum",
        "ucb_gap_to_full_tensor_optimum",
        "solver_required_profile_count",
        "profile_saving_ratio",
        "runtime_seconds",
        "solver_payoff_eval_time",
        "solver_compute_time",
        "full_audit_eval_time",
        "full_audit_compute_time",
        "full_audit_runtime_seconds",
        "support_size",
    ]
    available = [col for col in solver_cols if col in flat.columns]
    flat.groupby(["N", "K", "mechanism", "solver"], as_index=False)[available].mean().sort_values(
        ["N", "K", "mechanism", "cce_gap_ucb_full_audit"]
    ).to_csv(table_dir / "paper_oracle_solver_comparison.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cmfg_cce/configs/oracle_baselines.yaml")
    parser.add_argument("--output-dir", default="outputs/oracle_baselines")
    args = parser.parse_args()
    records, payoff_table = run_config(Path(args.config))
    write_outputs(records, payoff_table, Path(args.output_dir))


if __name__ == "__main__":
    main()
