from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from cmfg_cce.baselines import (
    mwu_policy_trace,
    random_support_cce,
    regret_matching_policy_trace,
    uniform_mixture,
    vanilla_sampled_cce,
    welfare_best_pure,
)
from cmfg_cce.envs.toy import ToyEnvConfig, config_from_mapping
from cmfg_cce.evaluation.payoff_cache import PayoffCache, full_tensor_size
from cmfg_cce.evaluation.rollout import RolloutSeeds
from cmfg_cce.policies.policy_library import build_policy_library_for_mechanism
from cmfg_cce.solvers.certified_kkt_cg import solve_certified_kkt_cg_sparse, solve_heuristic_kkt_cg_sparse
from cmfg_cce.solvers.sparse_cce import (
    SparseCceResult,
    solve_cg_beam_sparse,
    solve_pair_sparse,
    solve_repair_pair_sparse,
    solve_repair_sparse,
    solve_sad_sparse,
)
from cmfg_cce.solvers.t_carm import solve_t_carm_sparse


MARKUP_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
LEAD_TIME_GRID = (0.55, 0.70, 0.85, 1.00)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, records: list[dict]) -> None:
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


def result_metrics(cache: PayoffCache, result: SparseCceResult) -> dict[str, float]:
    keys: set[str] = set()
    for profile in result.support_profiles:
        keys.update(cache.get(profile).mean_metrics.keys())
    out: dict[str, float] = {}
    for key in keys:
        out[key] = float(
            sum(
                prob * cache.get(profile).mean_metrics.get(key, 0.0)
                for profile, prob in zip(result.support_profiles, result.support_probabilities, strict=True)
            )
        )
    return out


def result_record(
    experiment: str,
    mechanism: str,
    result: SparseCceResult,
    cache: PayoffCache,
    n: int,
    k: int,
    horizon: int,
    runtime_seconds: float,
    payoff_eval_time: float,
    solver_time: float,
    run_budget_label: str,
    seed: int,
    solver_stats: dict[str, float],
    global_cache_profile_count: int,
    new_profiles_evaluated: int,
    new_rollout_episode_evaluations: int,
) -> dict:
    tensor_size = full_tensor_size(n, k)
    solver_profile_count = int(solver_stats["solver_required_profile_count"])
    solver_rollout_episodes = int(solver_stats["solver_required_rollout_episode_total"])
    mechanism_metrics = result_metrics(cache, result)
    manufacturer_profits = [
        abs(float(value))
        for key, value in mechanism_metrics.items()
        if key.startswith("profit_manufacturer_") and value is not None
    ]
    if manufacturer_profits:
        mean_abs_episode_profit = float(sum(manufacturer_profits) / len(manufacturer_profits))
    else:
        mean_abs_episode_profit = abs(float(mechanism_metrics.get("manufacturer_total_profit", 0.0))) / max(1, n)
    payoff_values = []
    for estimate in cache.estimates.values():
        payoff_values.extend(float(value) for value in estimate.mean_returns)
    payoff_range = max(payoff_values) - min(payoff_values) if payoff_values else 0.0
    return {
        "experiment": experiment,
        "mechanism": mechanism,
        "solver": result.solver,
        "pricing_mode": result.pricing_mode,
        "certificate_status": result.certificate_status,
        "full_optimality_certified": result.full_optimality_certified,
        "N": n,
        "K": k,
        "seed": seed,
        "horizon": horizon,
        "support_size": result.support_size,
        "evaluated_profile_count": solver_profile_count,
        "solver_required_profile_count": solver_profile_count,
        "global_cache_profile_count": global_cache_profile_count,
        "new_profiles_evaluated": new_profiles_evaluated,
        "cache_hit_count": max(0, solver_profile_count - new_profiles_evaluated),
        "full_tensor_size": tensor_size,
        "profile_saving_ratio": solver_profile_count / max(1, tensor_size),
        "cce_gap_nominal": result.cce_gap_nominal,
        "cce_gap_ci": result.cce_gap_ci,
        "cce_gap_ucb": result.cce_gap_ucb,
        "gap_per_order": result.cce_gap_ucb / max(1, horizon),
        "gap_rel_profit": result.cce_gap_ucb / mean_abs_episode_profit if mean_abs_episode_profit > 0 else None,
        "gap_rel_range": result.cce_gap_ucb / payoff_range if payoff_range > 0 else None,
        "mean_abs_episode_profit": mean_abs_episode_profit,
        "payoff_range": payoff_range,
        "objective_value": result.objective_value,
        "runtime_seconds": runtime_seconds,
        "rollout_episode_total": solver_rollout_episodes,
        "rollout_steps_total": int(solver_stats["solver_required_rollout_steps_total"]),
        "new_rollout_episode_evaluations": new_rollout_episode_evaluations,
        "rollouts_initial": cache.n_rollouts,
        "rollouts_min": int(solver_stats["solver_required_rollouts_min"]),
        "rollouts_mean": float(solver_stats["solver_required_rollouts_mean"]),
        "rollouts_max": int(solver_stats["solver_required_rollouts_max"]),
        "lp_solve_time": solver_time,
        "pricing_time": 0.0 if result.pricing_mode is None else solver_time,
        "payoff_eval_time": payoff_eval_time,
        "run_budget_label": run_budget_label,
        "support_expansion_rounds": result.support_expansion_rounds,
        "active_sampling_rounds": result.active_sampling_rounds,
        "policy_expansion_rounds": result.policy_expansion_rounds,
        "repair_rounds": result.repair_rounds,
        "repair_profiles_added": result.repair_profiles_added,
        "paired_delta_pair_count": result.paired_delta_pair_count or int(solver_stats.get("paired_delta_pair_count", 0)),
        "paired_delta_sample_count": result.paired_delta_sample_count or int(solver_stats.get("paired_delta_sample_count", 0)),
        "paired_delta_rollout_episode_total": int(solver_stats.get("paired_delta_rollout_episode_total", 0)),
        "paired_estimator_mode": result.paired_estimator_mode,
        "audit_mode": result.audit_mode,
        "max_deviation_agent": result.max_deviation.get("agent"),
        "max_deviation_policy": result.max_deviation.get("policy"),
        "max_deviation_gain": result.max_deviation.get("gain_nominal"),
        "max_deviation_gain_ucb": result.max_deviation.get("gain_ucb"),
        "worst_mean_agent": result.max_deviation.get("agent"),
        "worst_mean_policy": result.max_deviation.get("policy"),
        "worst_ucb_agent": result.max_deviation.get("ucb_agent"),
        "worst_ucb_policy": result.max_deviation.get("ucb_policy"),
        "worst_ci_agent": result.max_deviation.get("ci_agent", result.max_deviation.get("ucb_agent")),
        "worst_ci_policy": result.max_deviation.get("ci_policy", result.max_deviation.get("ucb_policy")),
        "worst_ci_gain": result.max_deviation.get("gain_ci", result.max_deviation.get("gap_ci_at_ucb")),
        "gap_mean_at_ucb": result.diagnostics.get("gap_mean_at_ucb"),
        "support": [
            {"profile": list(profile), "prob": prob}
            for profile, prob in zip(result.support_profiles, result.support_probabilities, strict=True)
        ],
        "mechanism_metrics": mechanism_metrics,
    }


def run_solver(
    solver_name: str,
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    solver_cfg: dict[str, Any],
    seed: int,
    metadata: dict[str, Any] | None = None,
) -> SparseCceResult:
    if solver_name == "SAD-CCE":
        return solve_sad_sparse(
            cache,
            policy_ids,
            initial_support_size=int(solver_cfg.get("initial_support_size", 12)),
            max_support_size=int(solver_cfg.get("max_support_size", 40)),
            support_add_batch_size=int(solver_cfg.get("support_add_batch_size", 8)),
            max_rounds=int(solver_cfg.get("max_rounds", 4)),
            target_gap=float(solver_cfg.get("target_gap", 1.0)),
            seed=seed,
            rollouts_max=int(solver_cfg.get("rollouts_max", cache.n_rollouts)),
            active_sampling=bool(solver_cfg.get("active_sampling", True)),
        )
    if solver_name == "PAIR-SAD-CCE":
        pair_cfg = solver_cfg.get("pair", {})
        return solve_pair_sparse(
            cache,
            policy_ids,
            initial_support_size=int(solver_cfg.get("initial_support_size", 12)),
            max_support_size=int(solver_cfg.get("max_support_size", 40)),
            support_add_batch_size=int(solver_cfg.get("support_add_batch_size", 8)),
            max_rounds=int(solver_cfg.get("max_rounds", 4)),
            target_gap=float(solver_cfg.get("target_gap", 1.0)),
            seed=seed,
            rollouts_max=int(solver_cfg.get("rollouts_max", cache.n_rollouts)),
            active_sampling=bool(solver_cfg.get("active_sampling", True)),
            beta=float(pair_cfg.get("beta", 1.96)),
            n_min_pair_samples=int(pair_cfg.get("n_min_pair_samples", 4)),
            pair_active_rounds=int(pair_cfg.get("active_rounds", 2)),
            top_constraints=int(pair_cfg.get("top_constraints", 5)),
            top_pairs_per_constraint=int(pair_cfg.get("top_pairs_per_constraint", 10)),
            batch_samples_per_pair=int(pair_cfg.get("batch_samples_per_pair", 2)),
            q_min=float(pair_cfg.get("q_min", 1.0e-4)),
            metadata=metadata,
        )
    if solver_name == "REPAIR-SAD-CCE":
        repair_cfg = solver_cfg.get("repair", {})
        return solve_repair_sparse(
            cache,
            policy_ids,
            initial_support_size=int(solver_cfg.get("initial_support_size", 12)),
            max_support_size=int(solver_cfg.get("max_support_size", 40)),
            support_add_batch_size=int(solver_cfg.get("support_add_batch_size", 8)),
            max_rounds=int(solver_cfg.get("max_rounds", 4)),
            target_gap=float(solver_cfg.get("target_gap", 1.0)),
            seed=seed,
            rollouts_max=int(solver_cfg.get("rollouts_max", cache.n_rollouts)),
            active_sampling=bool(solver_cfg.get("active_sampling", True)),
            repair_rounds=int(repair_cfg.get("repair_rounds", 2)),
            top_constraints=int(repair_cfg.get("top_constraints", 3)),
            top_profiles_per_constraint=int(repair_cfg.get("top_profiles_per_constraint", 10)),
            q_min=float(repair_cfg.get("q_min", 1.0e-4)),
            mean_threshold=float(repair_cfg.get("mean_threshold", 1.0)),
            contribution_min=float(repair_cfg.get("contribution_min", 0.0)),
            enable_multi_agent_repair=bool(repair_cfg.get("enable_multi_agent_repair", True)),
            max_agents_repaired_per_profile=int(repair_cfg.get("max_agents_repaired_per_profile", 2)),
            beam_width=int(repair_cfg.get("beam_width", 10)),
            profile_budget_multiplier=float(repair_cfg.get("profile_budget_multiplier", 2.0)),
            targeted_deviation_policies=repair_cfg.get("targeted_deviation_policies", {}),
            selector=str(solver_cfg.get("selector", "platform_operating_score")),
            epsilon_tolerance=float(solver_cfg.get("epsilon_tolerance", 1.0e-9)),
        )
    if solver_name == "REPAIR-PAIR-SAD-CCE":
        pair_cfg = solver_cfg.get("pair", {})
        repair_cfg = solver_cfg.get("repair", {})
        return solve_repair_pair_sparse(
            cache,
            policy_ids,
            initial_support_size=int(solver_cfg.get("initial_support_size", 12)),
            max_support_size=int(solver_cfg.get("max_support_size", 40)),
            support_add_batch_size=int(solver_cfg.get("support_add_batch_size", 8)),
            max_rounds=int(solver_cfg.get("max_rounds", 4)),
            target_gap=float(solver_cfg.get("target_gap", 1.0)),
            seed=seed,
            rollouts_max=int(solver_cfg.get("rollouts_max", cache.n_rollouts)),
            active_sampling=bool(solver_cfg.get("active_sampling", True)),
            beta=float(pair_cfg.get("beta", 1.96)),
            n_min_pair_samples=int(pair_cfg.get("n_min_pair_samples", 4)),
            pair_active_rounds=int(pair_cfg.get("active_rounds", 2)),
            top_constraints=int(pair_cfg.get("top_constraints", 5)),
            top_pairs_per_constraint=int(pair_cfg.get("top_pairs_per_constraint", 10)),
            batch_samples_per_pair=int(pair_cfg.get("batch_samples_per_pair", 2)),
            q_min=float(pair_cfg.get("q_min", 1.0e-4)),
            metadata=metadata,
            repair_rounds=int(repair_cfg.get("repair_rounds", 2)),
            repair_top_constraints=int(repair_cfg.get("top_constraints", 3)),
            top_profiles_per_constraint=int(repair_cfg.get("top_profiles_per_constraint", 10)),
            mean_threshold=float(repair_cfg.get("mean_threshold", 1.0)),
            contribution_min=float(repair_cfg.get("contribution_min", 0.0)),
            enable_multi_agent_repair=bool(repair_cfg.get("enable_multi_agent_repair", True)),
            max_agents_repaired_per_profile=int(repair_cfg.get("max_agents_repaired_per_profile", 2)),
            beam_width=int(repair_cfg.get("beam_width", 10)),
            profile_budget_multiplier=float(repair_cfg.get("profile_budget_multiplier", 2.0)),
            targeted_deviation_policies=repair_cfg.get("targeted_deviation_policies", {}),
        )
    if solver_name == "CG-CCE":
        return solve_cg_beam_sparse(
            cache,
            policy_ids,
            initial_support_size=int(solver_cfg.get("initial_support_size", 12)),
            beam_width=int(solver_cfg.get("beam_width", 24)),
            max_rounds=int(solver_cfg.get("max_rounds", 4)),
            seed=seed,
        )
    if solver_name == "T-CARM-CCE":
        return solve_t_carm_sparse(
            cache,
            policy_ids,
            iterations=int(solver_cfg.get("t_carm_iterations", solver_cfg.get("mwu_rounds", 200))),
            seed=seed,
            solver_name="T-CARM-CCE",
            oracle_mode=str(solver_cfg.get("t_carm_oracle_mode", "exact")),
            audit_every=solver_cfg.get("t_carm_audit_every"),
            target_gap=float(solver_cfg.get("target_gap", 1.0e-8)),
            nested_inner_iterations=int(solver_cfg.get("t_carm_nested_inner_iterations", 40)),
            nested_restarts=int(solver_cfg.get("t_carm_nested_restarts", 6)),
        )
    if solver_name == "T-CARM-Nested-CCE":
        return solve_t_carm_sparse(
            cache,
            policy_ids,
            iterations=int(solver_cfg.get("t_carm_nested_iterations", solver_cfg.get("t_carm_iterations", 200))),
            seed=seed,
            solver_name="T-CARM-Nested-CCE",
            oracle_mode="nested",
            audit_every=solver_cfg.get("t_carm_audit_every"),
            target_gap=float(solver_cfg.get("target_gap", 1.0e-8)),
            nested_inner_iterations=int(solver_cfg.get("t_carm_nested_inner_iterations", 24)),
            nested_restarts=int(solver_cfg.get("t_carm_nested_restarts", 4)),
        )
    if solver_name == "CertifiedKKT-CG-CCE":
        return solve_certified_kkt_cg_sparse(
            cache,
            policy_ids,
            initial_support_size=int(solver_cfg.get("initial_support_size", 1)),
            tolerance=float(solver_cfg.get("target_gap", 1.0e-8)),
            max_iterations=solver_cfg.get("certified_cg_max_iterations"),
            seed=seed,
        )
    if solver_name == "HeuristicKKT-CG-CCE":
        heuristic_cfg = solver_cfg.get("heuristic_cg", {})
        return solve_heuristic_kkt_cg_sparse(
            cache,
            policy_ids,
            initial_support_size=int(heuristic_cfg.get("initial_support_size", solver_cfg.get("initial_support_size", 12))),
            max_support_size=int(heuristic_cfg.get("max_support_size", solver_cfg.get("max_support_size", 80))),
            tolerance=float(solver_cfg.get("target_gap", 1.0e-8)),
            max_iterations=int(heuristic_cfg.get("max_iterations", solver_cfg.get("certified_cg_max_iterations", 8))),
            pricing_sample_size=int(heuristic_cfg.get("pricing_sample_size", 96)),
            pricing_restarts=int(heuristic_cfg.get("pricing_restarts", 8)),
            pricing_local_steps=int(heuristic_cfg.get("pricing_local_steps", 4)),
            seed=seed,
        )
    if solver_name == "VanillaSampled-CCE":
        return vanilla_sampled_cce(cache, policy_ids, int(solver_cfg.get("vanilla_support_size", 24)), seed)
    if solver_name == "RandomSupport-CCE":
        return random_support_cce(cache, policy_ids, int(solver_cfg.get("vanilla_support_size", 24)), seed)
    if solver_name == "UniformMixture":
        return uniform_mixture(cache, policy_ids, int(solver_cfg.get("vanilla_support_size", 24)), seed)
    if solver_name == "WelfareBestPure":
        return welfare_best_pure(cache, policy_ids, int(solver_cfg.get("candidate_count", 48)), seed)
    if solver_name == "MWU-PolicyTrace":
        mwu_cfg = solver_cfg.get("mwu", {})
        return mwu_policy_trace(
            cache,
            policy_ids,
            int(solver_cfg.get("mwu_rounds", mwu_cfg.get("rounds", 24))),
            seed,
            eta=float(mwu_cfg.get("eta", solver_cfg.get("mwu_eta", 0.15))),
            schedule=str(mwu_cfg.get("schedule", solver_cfg.get("mwu_schedule", "constant"))),
            exploration_floor=float(
                mwu_cfg.get("exploration_floor", solver_cfg.get("mwu_exploration_floor", 0.0))
            ),
            burn_in_rounds=int(
                mwu_cfg.get("burn_in_rounds", solver_cfg.get("mwu_burn_in_rounds", 0))
            ),
        )
    if solver_name == "RegretMatching-PolicyTrace":
        rounds = int(solver_cfg.get("regret_matching_rounds", solver_cfg.get("mwu_rounds", 24)))
        return regret_matching_policy_trace(cache, policy_ids, rounds, seed)
    raise ValueError(f"Unsupported solver or diagnostic: {solver_name}")


def run_sparse_matrix(
    config_path: str | Path,
    experiment_key: str,
    return_paired_delta: bool = False,
) -> tuple[list[dict], pd.DataFrame] | tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    raw = load_yaml(config_path)
    env_config: ToyEnvConfig = config_from_mapping(raw)
    matrix = raw[experiment_key]
    solver_cfg = raw.get("solver", {})
    records: list[dict] = []
    payoff_records: list[dict] = []
    paired_delta_records: list[dict] = []
    for n in matrix["N_values"]:
        for k in matrix["policies_per_agent"]:
            for seed in matrix["seeds"]:
                for mechanism in raw["mechanisms"]:
                    policies = build_policy_library_for_mechanism(mechanism, int(k))
                    policy_ids = tuple(policies.keys())
                    cache = PayoffCache(
                        mechanism_id=mechanism,
                        policies=policies,
                        config=env_config,
                        n_agents=int(n),
                        seeds=solver_seed(int(seed)),
                        n_rollouts=int(matrix.get("rollouts_initial", solver_cfg.get("rollouts_initial", 3))),
                        markup_grid=tuple(raw.get("policies", {}).get("markup_grid", MARKUP_GRID)),
                        lead_time_grid=tuple(raw.get("policies", {}).get("lead_time_grid", LEAD_TIME_GRID)),
                        workers=int(solver_cfg.get("workers", raw.get("workers", 1))),
                    )
                    for solver_name in matrix["solvers"]:
                        cache.reset_access_log()
                        profiles_before = cache.evaluated_profile_count
                        eval_time_before = cache.eval_time_seconds
                        eval_rollout_episodes_before = cache.eval_rollout_episode_count
                        start = time.perf_counter()
                        solver_metadata = {
                            "experiment": raw["experiment"],
                            "env_version": raw.get("env_version", ""),
                            "N": int(n),
                            "K": int(k),
                            "seed": int(seed),
                            "run_budget_label": str(raw.get("run_budget_label", "paper_candidate")),
                        }
                        result = run_solver(
                            solver_name,
                            cache,
                            policy_ids,
                            solver_cfg,
                            seed=int(seed) + 17 * len(records),
                            metadata=solver_metadata,
                        )
                        runtime = time.perf_counter() - start
                        solver_stats = cache.access_stats()
                        profiles_after = cache.evaluated_profile_count
                        payoff_delta = cache.eval_time_seconds - eval_time_before
                        eval_rollout_episodes_after = cache.eval_rollout_episode_count
                        records.append(
                            result_record(
                                raw["experiment"],
                                mechanism,
                                result,
                                cache,
                                int(n),
                                int(k),
                                env_config.horizon,
                                runtime,
                                payoff_delta,
                                max(0.0, runtime - payoff_delta),
                                str(raw.get("run_budget_label", "paper_candidate")),
                                int(seed),
                                solver_stats,
                                profiles_after,
                                profiles_after - profiles_before,
                                eval_rollout_episodes_after - eval_rollout_episodes_before,
                            )
                        )
                        print(
                            f"{raw['experiment']} N={n} K={k} seed={seed} {mechanism} {solver_name}: "
                            f"gap={result.cce_gap_nominal:.3g} ucb={result.cce_gap_ucb:.3g} "
                            f"support={result.support_size} eval={cache.evaluated_profile_count}"
                        )
                    payoff_records.extend(
                        cache.records(
                            {
                                "experiment": raw["experiment"],
                                "solver": "shared_payoff_cache",
                                "N": int(n),
                                "K": int(k),
                                "seed": int(seed),
                                "run_budget_label": str(raw.get("run_budget_label", "paper_candidate")),
                            }
                        )
                    )
                    if return_paired_delta:
                        paired_delta_records.extend(cache.paired_delta_rows)
    if return_paired_delta:
        return records, pd.DataFrame(payoff_records), pd.DataFrame(paired_delta_records)
    return records, pd.DataFrame(payoff_records)


def flatten_records(records: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for record in records:
        row = {key: value for key, value in record.items() if key not in {"support", "mechanism_metrics"}}
        for key, value in record["mechanism_metrics"].items():
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def write_experiment_outputs(
    records: list[dict],
    payoff_table: pd.DataFrame,
    output_dir: str | Path,
    raw_name: str,
    table_name: str,
    paired_delta_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    table_dir = output_dir / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    write_json(raw_dir / f"cce_results_{raw_name}.json", records)
    flat = flatten_records(records)
    flat.to_csv(table_dir / table_name, index=False)
    payoff_path = raw_dir / f"payoff_cache_{raw_name}.parquet"
    csv_cache_path = raw_dir / f"payoff_cache_{raw_name}.csv"
    payoff_table = payoff_table.copy()
    for column in payoff_table.columns:
        if payoff_table[column].dtype == "object":
            payoff_table[column] = payoff_table[column].fillna("").astype(str)
    payoff_table.to_parquet(payoff_path, index=False, engine="pyarrow")
    pd.read_parquet(payoff_path, engine="pyarrow")
    payoff_table.to_csv(csv_cache_path, index=False)
    if paired_delta_table is not None:
        paired_delta_path = raw_dir / f"paired_delta_cache_{raw_name}.parquet"
        paired_delta_csv_path = raw_dir / f"paired_delta_cache_{raw_name}.csv"
        paired_delta_table = paired_delta_table.copy()
        for column in paired_delta_table.columns:
            if paired_delta_table[column].dtype == "object":
                paired_delta_table[column] = paired_delta_table[column].fillna("").astype(str)
        paired_delta_table.to_parquet(paired_delta_path, index=False, engine="pyarrow")
        pd.read_parquet(paired_delta_path, engine="pyarrow")
        paired_delta_table.to_csv(paired_delta_csv_path, index=False)
        if not paired_delta_table.empty:
            diagnostics = paired_delta_table.copy()
            diagnostics["pair_key"] = (
                diagnostics["support_profile"].astype(str)
                + "||"
                + diagnostics["agent_id"].astype(str)
                + "||"
                + diagnostics["dev_policy"].astype(str)
            )
            diagnostics.groupby(["experiment", "mechanism", "solver"], as_index=False).agg(
                paired_delta_pair_count=("pair_key", "nunique"),
                paired_delta_sample_count=("delta_return", "size"),
                delta_return_mean=("delta_return", "mean"),
                delta_return_std=("delta_return", "std"),
            ).to_csv(table_dir / f"paired_delta_diagnostics_{raw_name}.csv", index=False)
        else:
            paired_delta_table.to_csv(table_dir / f"paired_delta_diagnostics_{raw_name}.csv", index=False)
    write_paper_tables(flat, table_dir, table_name)
    return flat


def write_paper_tables(flat: pd.DataFrame, table_dir: Path, table_name: str) -> None:
    if table_name == "core_mechanism_comparison.csv":
        sad = flat[flat["solver"] == "SAD-CCE"]
        cols = [
            "cce_gap_ucb",
            "platform_operating_score",
            "platform_total_payment",
            "manufacturer_total_profit",
            "participation_rate",
            "fulfillment_rate",
            "late_completion_rate",
            "overload_rate",
            "average_platform_utilization",
            "average_markup",
            "average_lead_time_multiplier",
            "winner_concentration_hhi",
        ]
        sad.groupby("mechanism", as_index=False)[cols].mean().to_csv(table_dir / "paper_core_mechanism_comparison.csv", index=False)
    elif table_name == "scalability.csv":
        flat.groupby(["N", "K", "mechanism", "solver"], as_index=False)[
            [
                "full_tensor_size",
                "evaluated_profile_count",
                "new_profiles_evaluated",
                "profile_saving_ratio",
                "rollout_steps_total",
                "runtime_seconds",
                "cce_gap_ucb",
                "support_size",
            ]
        ].mean().to_csv(table_dir / "paper_scalability.csv", index=False)
    elif table_name == "ablation.csv":
        group_col = "ablation_variant" if "ablation_variant" in flat.columns else "experiment"
        flat.groupby(group_col, as_index=False)[
            ["cce_gap_ucb", "evaluated_profile_count", "rollout_steps_total", "runtime_seconds", "support_size", "objective_value"]
        ].mean().to_csv(table_dir / "paper_ablation.csv", index=False)
    elif table_name == "full_robustness.csv":
        flat.groupby(["N", "K", "mechanism", "solver"], as_index=False)[
            ["cce_gap_ucb", "runtime_seconds", "support_size", "evaluated_profile_count", "rollout_steps_total", "platform_operating_score"]
        ].mean().to_csv(table_dir / "paper_full_robustness.csv", index=False)
    elif table_name == "pair_repair.csv":
        comparison_cols = [
            "cce_gap_nominal",
            "cce_gap_ci",
            "cce_gap_ucb",
            "gap_per_order",
            "gap_rel_profit",
            "gap_rel_range",
            "support_size",
            "evaluated_profile_count",
            "rollout_steps_total",
            "runtime_seconds",
            "platform_operating_score",
            "platform_total_payment",
            "manufacturer_total_profit",
            "participation_rate",
            "fulfillment_rate",
            "late_completion_rate",
            "overload_rate",
            "average_platform_utilization",
            "average_markup",
            "average_lead_time_multiplier",
            "winner_concentration_hhi",
        ]
        available_cols = [col for col in comparison_cols if col in flat.columns]
        flat.groupby(["mechanism", "solver"], as_index=False)[available_cols].mean().sort_values(
            ["mechanism", "cce_gap_ucb"]
        ).to_csv(table_dir / "paper_pair_repair_by_mechanism.csv", index=False)
        flat.groupby("solver", as_index=False)[available_cols].mean().sort_values("cce_gap_ucb").to_csv(
            table_dir / "paper_pair_repair_solver_comparison.csv",
            index=False,
        )
        if "max_deviation_policy" in flat.columns:
            flat.groupby(["mechanism", "solver", "max_deviation_policy"], as_index=False).size().rename(
                columns={"size": "count"}
            ).sort_values(["mechanism", "solver", "count"], ascending=[True, True, False]).to_csv(
                table_dir / "pair_repair_worst_deviation_policy_counts.csv",
                index=False,
            )


def save_core_mechanism_figure(df: pd.DataFrame, output_dir: str | Path) -> None:
    Path(output_dir, "figures").mkdir(parents=True, exist_ok=True)
    sad = df[df["solver"] == "SAD-CCE"]
    grouped = sad.groupby("mechanism", as_index=False)[
        ["platform_operating_score", "late_completion_rate", "overload_rate", "participation_rate"]
    ].mean()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(axes.flat, grouped.columns[1:], strict=True):
        ax.bar(grouped["mechanism"], grouped[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "figures" / "mechanism_outcomes.png", dpi=180)
    plt.close(fig)

    delivery = sad[sad["mechanism"].str.contains("delivery")]
    if delivery.empty:
        delivery = sad
    quantile_rows = []
    for _, row in delivery.iterrows():
        for label in ["low_alpha", "high_alpha"]:
            quantile_rows.append(
                {
                    "mechanism": row["mechanism"],
                    "alpha_quantile": label,
                    "win_share": row.get(f"{label}_win_share", 0.0),
                    "avg_markup": row.get(f"{label}_avg_markup", 0.0),
                    "avg_lead_time_multiplier": row.get(f"{label}_avg_lead_time_multiplier", 0.0),
                    "profit": row.get(f"{label}_profit", 0.0),
                    "utilization": row.get(f"{label}_utilization", 0.0),
                }
            )
    quantile_df = pd.DataFrame(quantile_rows)
    quantile_df.groupby(["mechanism", "alpha_quantile"], as_index=False).mean().to_csv(
        Path(output_dir) / "tables" / "paper_alpha_quantile_behavior.csv",
        index=False,
    )
    grouped_delivery = quantile_df.groupby("alpha_quantile", as_index=False)[
        ["win_share", "avg_markup", "avg_lead_time_multiplier", "profit"]
    ].mean()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(axes.flat, grouped_delivery.columns[1:], strict=True):
        ax.bar(grouped_delivery["alpha_quantile"], grouped_delivery[metric])
        ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "figures" / "alpha_quantile_behavior.png", dpi=180)
    plt.close(fig)


def save_scalability_figures(df: pd.DataFrame, output_dir: str | Path) -> None:
    Path(output_dir, "figures").mkdir(parents=True, exist_ok=True)
    sad = df[df["solver"] == "SAD-CCE"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sad["full_tensor_size"], sad["evaluated_profile_count"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("full tensor size")
    ax.set_ylabel("evaluated profiles")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "figures" / "scalability_profiles_vs_full_tensor.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for solver, group in df.groupby("solver"):
        ax.plot(group.groupby("N")["runtime_seconds"].mean().index, group.groupby("N")["runtime_seconds"].mean(), marker="o", label=solver)
    ax.set_xlabel("N")
    ax.set_ylabel("runtime seconds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "figures" / "runtime_vs_n.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for solver, group in df.groupby("solver"):
        ax.plot(group.groupby("rollout_steps_total")["cce_gap_ucb"].mean().index, group.groupby("rollout_steps_total")["cce_gap_ucb"].mean(), marker="o", label=solver)
    ax.set_xlabel("rollout steps total")
    ax.set_ylabel("CCE gap UCB")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "figures" / "cce_gap_vs_budget.png", dpi=180)
    plt.close(fig)
