from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cmfg_cce.baselines.simple import regret_matching_policy_trace
from cmfg_cce.envs.toy import ToyEnvConfig
from cmfg_cce.evaluation.empirical_game import EmpiricalGame, build_empirical_game
from cmfg_cce.evaluation.full_audit import audit_sparse_result_on_full_game
from cmfg_cce.evaluation.payoff_cache import PayoffCache, profile_space
from cmfg_cce.evaluation.rollout import Profile, RolloutEstimate
from cmfg_cce.experiments.common import LEAD_TIME_GRID, MARKUP_GRID, solver_seed
from cmfg_cce.policies.policy_library import build_policy_library_for_mechanism
from cmfg_cce.solvers.cce_lp import CceSolution, compute_cce_gap, solve_full_cce_lp
from cmfg_cce.solvers.sparse_cce import SparseCceResult, solve_repair_sparse


SYNTHETIC_RM_ROUNDS = (20, 100, 500)
SYNTHETIC_SEEDS = tuple(range(10))
CMFG_MECHANISMS = (
    "M1_price_first",
    "M2_price_critical",
    "M3_delivery_first",
    "M4_delivery_critical",
)


@dataclass(frozen=True)
class SyntheticSpec:
    name: str
    description: str
    game: EmpiricalGame


@dataclass
class _InMemoryConfig:
    horizon: int = 1


class InMemoryEmpiricalGameCache:
    """PayoffCache-compatible adapter for deterministic empirical games."""

    def __init__(self, game: EmpiricalGame, mechanism_id: str = "synthetic") -> None:
        self.game = game
        self.mechanism_id = mechanism_id
        self.n_agents = game.n_agents
        self.n_rollouts = 1
        self.config = _InMemoryConfig()
        self.eval_time_seconds = 0.0
        self.accessed_profiles: set[Profile] = set()
        self.requested_rollouts_by_profile: dict[Profile, int] = {}
        self.estimates: dict[Profile, RolloutEstimate] = {
            profile: RolloutEstimate(
                profile=profile,
                n_rollouts=1,
                mean_returns=np.array(game.payoffs[idx], dtype=float),
                var_returns=np.zeros(game.n_agents, dtype=float),
                ci_radius=np.array(game.ci_radius[idx], dtype=float),
                mean_metrics=dict(game.metrics[idx]),
                var_metrics={},
            )
            for idx, profile in enumerate(game.profiles)
        }

    def reset_access_log(self) -> None:
        self.accessed_profiles.clear()
        self.requested_rollouts_by_profile.clear()

    def record_access(self, profiles: list[Profile] | set[Profile] | tuple[Profile, ...]) -> None:
        for profile in profiles:
            profile_t = tuple(profile)
            self.accessed_profiles.add(profile_t)
            self.requested_rollouts_by_profile[profile_t] = 1

    def ensure(self, profiles) -> None:
        self.record_access(list(profiles))

    def resample_many(self, profiles, target_rollouts: int) -> None:
        self.record_access(list(profiles))

    def get(self, profile: Profile) -> RolloutEstimate:
        profile_t = tuple(profile)
        self.record_access([profile_t])
        return self.estimates[profile_t]

    def payoff(self, profile: Profile, agent: int) -> float:
        return float(self.get(profile).mean_returns[int(agent)])

    def payoff_ucb(self, profile: Profile, agent: int) -> float:
        estimate = self.get(profile)
        return float(estimate.mean_returns[int(agent)] + estimate.ci_radius[int(agent)])

    def payoff_lcb(self, profile: Profile, agent: int) -> float:
        estimate = self.get(profile)
        return float(estimate.mean_returns[int(agent)] - estimate.ci_radius[int(agent)])

    def support_objective(self, profile: Profile, objective_key: str = "platform_operating_score") -> float:
        return float(self.get(profile).mean_metrics.get(objective_key, 0.0))

    @property
    def evaluated_profile_count(self) -> int:
        return len(self.accessed_profiles)

    def access_stats(self) -> dict[str, float]:
        requests = list(self.requested_rollouts_by_profile.values())
        rollout_episodes = int(sum(requests))
        return {
            "solver_required_profile_count": int(len(self.accessed_profiles)),
            "solver_required_rollout_episode_total": rollout_episodes,
            "solver_required_rollout_steps_total": rollout_episodes,
            "solver_required_rollouts_min": int(min(requests)) if requests else 0,
            "solver_required_rollouts_mean": float(np.mean(requests)) if requests else 0.0,
            "solver_required_rollouts_max": int(max(requests)) if requests else 0,
            "paired_delta_pair_count": 0,
            "paired_delta_sample_count": 0,
            "paired_delta_rollout_episode_total": 0,
        }


def _game_from_payoff_fn(
    name: str,
    policy_ids: tuple[str, ...],
    n_agents: int,
    payoff_fn: Callable[[Profile], np.ndarray],
) -> SyntheticSpec:
    profiles = tuple(tuple(profile) for profile in product(policy_ids, repeat=n_agents))
    payoffs = np.vstack([payoff_fn(profile) for profile in profiles])
    objectives = np.sum(payoffs, axis=1)
    metrics = tuple(
        {
            "platform_operating_score": float(objectives[idx]),
            "manufacturer_total_profit": float(objectives[idx]),
        }
        for idx in range(len(profiles))
    )
    return SyntheticSpec(
        name=name,
        description="",
        game=EmpiricalGame(
            profiles=profiles,
            policy_ids=policy_ids,
            payoffs=payoffs,
            ci_radius=np.zeros_like(payoffs),
            objectives=objectives,
            metrics=metrics,
        ),
    )


def build_rps_cycle() -> SyntheticSpec:
    policy_ids = ("R", "P", "S")
    wins = {("R", "S"), ("S", "P"), ("P", "R")}

    def payoff(profile: Profile) -> np.ndarray:
        first, second = profile
        if first == second:
            return np.array([0.0, 0.0])
        return np.array([1.0, -1.0]) if (first, second) in wins else np.array([-1.0, 1.0])

    spec = _game_from_payoff_fn("RPS-Cycle", policy_ids, 2, payoff)
    return SyntheticSpec(
        name=spec.name,
        description="Two-player rock-paper-scissors; the zero-gap solution requires mixing.",
        game=spec.game,
    )


def build_successor_ring(k: int = 6) -> SyntheticSpec:
    policy_ids = tuple(f"A{i}" for i in range(k))

    def payoff(profile: Profile) -> np.ndarray:
        first = policy_ids.index(profile[0])
        second = policy_ids.index(profile[1])
        return np.array(
            [
                1.0 if first == (second + 1) % k else 0.0,
                1.0 if second == (first + 1) % k else 0.0,
            ],
            dtype=float,
        )

    spec = _game_from_payoff_fn("Successor-Ring", policy_ids, 2, payoff)
    return SyntheticSpec(
        name=spec.name,
        description="Two-player cyclic best responses on a six-policy ring.",
        game=spec.game,
    )


def build_coordination_lock(n_agents: int = 3, k: int = 6) -> SyntheticSpec:
    policy_ids = tuple(f"A{i}" for i in range(k))

    def payoff(profile: Profile) -> np.ndarray:
        classes = [policy_ids.index(policy) % 3 for policy in profile]
        returns = []
        for agent, policy_class in enumerate(classes):
            predecessor = classes[(agent - 1) % n_agents]
            returns.append(1.0 if policy_class == (predecessor + 1) % 3 else 0.0)
        return np.array(returns, dtype=float)

    spec = _game_from_payoff_fn("Coordination-Lock", policy_ids, n_agents, payoff)
    return SyntheticSpec(
        name=spec.name,
        description="Three-player cyclic coordination with many policies but few stable classes.",
        game=spec.game,
    )


def build_synthetic_specs() -> list[SyntheticSpec]:
    return [build_rps_cycle(), build_successor_ring(), build_coordination_lock()]


def expected_payoff_scale(game: EmpiricalGame, q: np.ndarray) -> float:
    expected = q @ game.payoffs
    return max(1.0, float(np.mean(np.abs(expected))))


def relative_gap_pct(game: EmpiricalGame, q: np.ndarray, gap: float) -> float:
    return 100.0 * max(0.0, float(gap)) / expected_payoff_scale(game, q)


def manual_cce_gap(game: EmpiricalGame, q: np.ndarray) -> float:
    profile_to_index = game.profile_to_index
    best = 0.0
    for agent in range(game.n_agents):
        for policy_id in game.policy_ids:
            gain = 0.0
            for idx, profile in enumerate(game.profiles):
                dev_profile = list(profile)
                dev_profile[agent] = policy_id
                dev_idx = profile_to_index[tuple(dev_profile)]
                gain += float(q[idx]) * float(game.payoffs[dev_idx, agent] - game.payoffs[idx, agent])
            best = max(best, gain)
    return max(0.0, best)


def has_zero_gap_pure_profile(game: EmpiricalGame, tol: float = 1.0e-8) -> bool:
    for idx, _profile in enumerate(game.profiles):
        q = np.zeros(game.n_profiles, dtype=float)
        q[idx] = 1.0
        if compute_cce_gap(game, q) <= tol:
            return True
    return False


def _full_solution_record(
    scenario_family: str,
    scenario: str,
    game: EmpiricalGame,
    solution: CceSolution,
    runtime_seconds: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "scenario_family": scenario_family,
        "scenario": scenario,
        "solver": "FullTensor-CCE-LP",
        "N": game.n_agents,
        "K": len(game.policy_ids),
        "seed": -1,
        "rounds": 0,
        "cce_gap_full_verification": float(solution.cce_gap_nominal),
        "relative_gap_pct": relative_gap_pct(game, solution.q, solution.cce_gap_nominal),
        "runtime_seconds": float(runtime_seconds),
        "full_verification_runtime_seconds": 0.0,
        "support_size": int(solution.support_size),
        "solver_required_profile_count": int(game.n_profiles),
        "full_tensor_size": int(game.n_profiles),
        "profile_coverage_pct": 100.0,
        "objective_value": float(solution.objective_value),
        "status": str(solution.status),
    }
    if extra:
        record.update(extra)
    return record


def _sparse_solution_record(
    scenario_family: str,
    scenario: str,
    game: EmpiricalGame,
    result: SparseCceResult,
    verified: CceSolution,
    runtime_seconds: float,
    verification_runtime_seconds: float,
    access_stats: dict[str, float],
    seed: int,
    rounds: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_count = int(access_stats.get("solver_required_profile_count", 0))
    record = {
        "scenario_family": scenario_family,
        "scenario": scenario,
        "solver": str(result.solver),
        "N": game.n_agents,
        "K": len(game.policy_ids),
        "seed": int(seed),
        "rounds": int(rounds),
        "cce_gap_full_verification": float(verified.cce_gap_nominal),
        "relative_gap_pct": relative_gap_pct(game, verified.q, verified.cce_gap_nominal),
        "runtime_seconds": float(runtime_seconds),
        "full_verification_runtime_seconds": float(verification_runtime_seconds),
        "support_size": int(verified.support_size),
        "solver_required_profile_count": profile_count,
        "full_tensor_size": int(game.n_profiles),
        "profile_coverage_pct": 100.0 * profile_count / max(1, int(game.n_profiles)),
        "objective_value": float(verified.objective_value),
        "max_deviation_agent": verified.max_deviation.get("agent") if verified.max_deviation else None,
        "max_deviation_policy": verified.max_deviation.get("policy") if verified.max_deviation else None,
        "status": str(result.status),
        "support": [
            {"profile": list(profile), "prob": float(prob)}
            for profile, prob in zip(verified.support_profiles, verified.support_probabilities, strict=True)
        ],
    }
    if extra:
        record.update(extra)
    return record


def run_dss_on_game(game: EmpiricalGame, seed: int) -> tuple[SparseCceResult, float, dict[str, float]]:
    cache = InMemoryEmpiricalGameCache(game, mechanism_id="synthetic")
    cache.reset_access_log()
    start = time.perf_counter()
    result = solve_repair_sparse(
        cache,
        game.policy_ids,
        initial_support_size=min(6, game.n_profiles),
        max_support_size=min(24, game.n_profiles),
        support_add_batch_size=1,
        max_rounds=4,
        target_gap=1.0e-9,
        seed=seed,
        rollouts_max=None,
        active_sampling=False,
        repair_rounds=2,
        top_constraints=4,
        top_profiles_per_constraint=8,
        mean_threshold=0.0,
        contribution_min=0.0,
        enable_multi_agent_repair=True,
        max_agents_repaired_per_profile=2,
        beam_width=10,
        profile_budget_multiplier=2.0,
    )
    runtime = time.perf_counter() - start
    return result, runtime, cache.access_stats()


def run_rm_on_game(game: EmpiricalGame, rounds: int, seed: int) -> tuple[SparseCceResult, float, dict[str, float]]:
    cache = InMemoryEmpiricalGameCache(game, mechanism_id="synthetic")
    cache.reset_access_log()
    start = time.perf_counter()
    result = regret_matching_policy_trace(cache, game.policy_ids, rounds=int(rounds), seed=seed)
    runtime = time.perf_counter() - start
    return result, runtime, cache.access_stats()


def run_synthetic_stress() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    definitions: list[dict[str, Any]] = []
    for spec in build_synthetic_specs():
        game = spec.game
        start = time.perf_counter()
        full = solve_full_cce_lp(game)
        full_runtime = time.perf_counter() - start
        definitions.append(
            {
                "name": spec.name,
                "description": spec.description,
                "N": game.n_agents,
                "K": len(game.policy_ids),
                "policy_ids": list(game.policy_ids),
                "full_tensor_size": int(game.n_profiles),
                "has_zero_gap_pure_profile": has_zero_gap_pure_profile(game),
                "full_lp_gap": float(full.cce_gap_nominal),
                "full_lp_support": [
                    {"profile": list(profile), "prob": float(prob)}
                    for profile, prob in zip(full.support_profiles, full.support_probabilities, strict=True)
                ],
                "payoffs": [
                    {"profile": list(profile), "payoff": game.payoffs[idx].tolist()}
                    for idx, profile in enumerate(game.profiles)
                ],
            }
        )
        records.append(
            _full_solution_record(
                "synthetic",
                spec.name,
                game,
                full,
                full_runtime,
                {"scenario_description": spec.description},
            )
        )
        for seed in SYNTHETIC_SEEDS:
            dss, dss_runtime, dss_stats = run_dss_on_game(game, seed)
            start = time.perf_counter()
            verified = audit_sparse_result_on_full_game(game, dss)
            verify_runtime = time.perf_counter() - start
            records.append(
                _sparse_solution_record(
                    "synthetic",
                    spec.name,
                    game,
                    dss,
                    verified,
                    dss_runtime,
                    verify_runtime,
                    dss_stats,
                    seed,
                    0,
                    {"scenario_description": spec.description},
                )
            )
            for rounds in SYNTHETIC_RM_ROUNDS:
                rm, rm_runtime, rm_stats = run_rm_on_game(game, rounds, seed)
                start = time.perf_counter()
                verified_rm = audit_sparse_result_on_full_game(game, rm)
                verify_runtime = time.perf_counter() - start
                records.append(
                    _sparse_solution_record(
                        "synthetic",
                        spec.name,
                        game,
                        rm,
                        verified_rm,
                        rm_runtime,
                        verify_runtime,
                        rm_stats,
                        seed,
                        rounds,
                        {"scenario_description": spec.description},
                    )
                )
    return records, definitions


def cmfg_candidate_configs() -> dict[str, ToyEnvConfig]:
    return {
        "tight_capacity": ToyEnvConfig(
            layer="core",
            horizon=12,
            workload_range=(25, 45),
            due_date_range=(5, 12),
            capacity_range=(55.0, 75.0),
            c_base_range=(1.5, 4.0),
            c_setup_range=(6.0, 14.0),
            p_out_range=(0.10, 0.25),
            alpha_range=(0.3, 1.8),
            pressure_decay=0.95,
            shock_size_range=(0.10, 0.25),
            max_pressure=0.70,
        ),
        "tight_due_dates": ToyEnvConfig(
            layer="core",
            horizon=12,
            workload_range=(20, 45),
            due_date_range=(3, 8),
            capacity_range=(70.0, 100.0),
            c_base_range=(1.0, 4.5),
            c_setup_range=(5.0, 16.0),
            p_out_range=(0.05, 0.20),
            alpha_range=(0.5, 2.5),
            pressure_decay=0.92,
            shock_size_range=(0.05, 0.20),
            max_pressure=0.60,
        ),
        "high_core_pressure": ToyEnvConfig(
            layer="core",
            horizon=12,
            workload_range=(15, 40),
            due_date_range=(4, 12),
            capacity_range=(70.0, 110.0),
            c_base_range=(1.0, 4.0),
            c_setup_range=(5.0, 15.0),
            p_out_range=(0.55, 0.80),
            alpha_range=(0.2, 1.8),
            pressure_decay=0.98,
            shock_size_range=(0.15, 0.35),
            max_pressure=0.85,
        ),
        "high_alpha_heterogeneity": ToyEnvConfig(
            layer="core",
            horizon=12,
            workload_range=(15, 45),
            due_date_range=(3, 10),
            capacity_range=(65.0, 120.0),
            c_base_range=(1.0, 4.0),
            c_setup_range=(5.0, 14.0),
            p_out_range=(0.10, 0.30),
            alpha_range=(0.05, 3.00),
            pressure_decay=0.94,
            shock_size_range=(0.05, 0.25),
            max_pressure=0.70,
        ),
        "narrow_capacity": ToyEnvConfig(
            layer="core",
            horizon=12,
            workload_range=(20, 40),
            due_date_range=(4, 12),
            capacity_range=(80.0, 85.0),
            c_base_range=(1.5, 2.0),
            c_setup_range=(7.0, 9.0),
            p_out_range=(0.15, 0.35),
            alpha_range=(0.3, 2.0),
            pressure_decay=0.95,
            shock_size_range=(0.10, 0.25),
            max_pressure=0.75,
        ),
    }


def build_full_cmfg_game(
    config: ToyEnvConfig,
    mechanism: str,
    n_agents: int,
    k: int,
    seed: int,
    rollouts: int,
    workers: int,
) -> tuple[EmpiricalGame, float]:
    policies = build_policy_library_for_mechanism(mechanism, k)
    policy_ids = tuple(policies.keys())
    cache = PayoffCache(
        mechanism_id=mechanism,
        policies=policies,
        config=config,
        n_agents=n_agents,
        seeds=solver_seed(seed),
        n_rollouts=rollouts,
        markup_grid=MARKUP_GRID,
        lead_time_grid=LEAD_TIME_GRID,
        workers=workers,
    )
    profiles = list(profile_space(policy_ids, n_agents))
    start = time.perf_counter()
    cache.ensure(profiles)
    full_eval_time = time.perf_counter() - start
    game = build_empirical_game([cache.estimates[profile] for profile in profiles], policy_ids)
    return game, full_eval_time


def run_cmfg_dss(
    config: ToyEnvConfig,
    mechanism: str,
    n_agents: int,
    k: int,
    seed: int,
    rollouts: int,
    workers: int,
) -> tuple[SparseCceResult, float, dict[str, float]]:
    policies = build_policy_library_for_mechanism(mechanism, k)
    cache = PayoffCache(
        mechanism_id=mechanism,
        policies=policies,
        config=config,
        n_agents=n_agents,
        seeds=solver_seed(seed),
        n_rollouts=rollouts,
        markup_grid=MARKUP_GRID,
        lead_time_grid=LEAD_TIME_GRID,
        workers=workers,
    )
    policy_ids = tuple(policies.keys())
    cache.reset_access_log()
    start = time.perf_counter()
    result = solve_repair_sparse(
        cache,
        policy_ids,
        initial_support_size=8,
        max_support_size=32,
        support_add_batch_size=6,
        max_rounds=4,
        target_gap=1.0e-8,
        seed=seed,
        rollouts_max=None,
        active_sampling=False,
        repair_rounds=2,
        top_constraints=4,
        top_profiles_per_constraint=8,
        q_min=1.0e-4,
        mean_threshold=0.0,
        contribution_min=0.0,
        enable_multi_agent_repair=True,
        max_agents_repaired_per_profile=2,
        beam_width=10,
        profile_budget_multiplier=2.0,
    )
    runtime = time.perf_counter() - start
    return result, runtime, cache.access_stats()


def run_cmfg_rm(
    config: ToyEnvConfig,
    mechanism: str,
    n_agents: int,
    k: int,
    seed: int,
    rollouts: int,
    workers: int,
    rounds: int,
) -> tuple[SparseCceResult, float, dict[str, float]]:
    policies = build_policy_library_for_mechanism(mechanism, k)
    cache = PayoffCache(
        mechanism_id=mechanism,
        policies=policies,
        config=config,
        n_agents=n_agents,
        seeds=solver_seed(seed),
        n_rollouts=rollouts,
        markup_grid=MARKUP_GRID,
        lead_time_grid=LEAD_TIME_GRID,
        workers=workers,
    )
    policy_ids = tuple(policies.keys())
    cache.reset_access_log()
    start = time.perf_counter()
    result = regret_matching_policy_trace(cache, policy_ids, rounds=rounds, seed=seed)
    runtime = time.perf_counter() - start
    return result, runtime, cache.access_stats()


def run_cmfg_screening(
    limit: int | None,
    rollouts: int,
    workers: int,
    rm_rounds: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    scenario_records: list[dict[str, Any]] = []
    configs = cmfg_candidate_configs()
    scenario_iter = []
    for config_name in configs:
        for n_agents in (3, 4):
            for mechanism in CMFG_MECHANISMS:
                for seed in (0, 1, 2):
                    scenario_iter.append((config_name, configs[config_name], n_agents, mechanism, seed))
    if limit is not None:
        scenario_iter = scenario_iter[: max(0, int(limit))]

    for config_name, config, n_agents, mechanism, seed in scenario_iter:
        k = 6
        scenario = f"{config_name}_N{n_agents}_K{k}_{mechanism}_seed{seed}"
        try:
            game, full_eval_time = build_full_cmfg_game(config, mechanism, n_agents, k, seed, rollouts, workers)
            start = time.perf_counter()
            full = solve_full_cce_lp(game)
            full_lp_time = time.perf_counter() - start
            records.append(
                _full_solution_record(
                    "cmfg_screen",
                    scenario,
                    game,
                    full,
                    full_eval_time + full_lp_time,
                    {
                        "config_name": config_name,
                        "mechanism": mechanism,
                        "seed": seed,
                        "rollouts_per_profile": rollouts,
                        "full_tensor_eval_time": full_eval_time,
                    },
                )
            )
            dss, dss_runtime, dss_stats = run_cmfg_dss(config, mechanism, n_agents, k, seed, rollouts, workers)
            start = time.perf_counter()
            dss_verified = audit_sparse_result_on_full_game(game, dss)
            dss_verify_runtime = time.perf_counter() - start
            dss_record = _sparse_solution_record(
                "cmfg_screen",
                scenario,
                game,
                dss,
                dss_verified,
                dss_runtime,
                dss_verify_runtime,
                dss_stats,
                seed,
                0,
                {
                    "config_name": config_name,
                    "mechanism": mechanism,
                    "rollouts_per_profile": rollouts,
                    "full_tensor_eval_time": full_eval_time,
                },
            )
            records.append(dss_record)
            rm, rm_runtime, rm_stats = run_cmfg_rm(config, mechanism, n_agents, k, seed, rollouts, workers, rm_rounds)
            start = time.perf_counter()
            rm_verified = audit_sparse_result_on_full_game(game, rm)
            rm_verify_runtime = time.perf_counter() - start
            rm_record = _sparse_solution_record(
                "cmfg_screen",
                scenario,
                game,
                rm,
                rm_verified,
                rm_runtime,
                rm_verify_runtime,
                rm_stats,
                seed,
                rm_rounds,
                {
                    "config_name": config_name,
                    "mechanism": mechanism,
                    "rollouts_per_profile": rollouts,
                    "full_tensor_eval_time": full_eval_time,
                },
            )
            records.append(rm_record)
            scenario_records.append(
                {
                    "scenario": scenario,
                    "config_name": config_name,
                    "N": n_agents,
                    "K": k,
                    "mechanism": mechanism,
                    "seed": seed,
                    "dss_relative_gap_pct": dss_record["relative_gap_pct"],
                    "rm_relative_gap_pct": rm_record["relative_gap_pct"],
                    "gap_difference_pct": rm_record["relative_gap_pct"] - dss_record["relative_gap_pct"],
                    "dss_runtime_seconds": dss_runtime,
                    "rm_runtime_seconds": rm_runtime,
                    "selected": False,
                }
            )
        except Exception as exc:  # Keep screening robust and explicit.
            scenario_records.append(
                {
                    "scenario": scenario,
                    "config_name": config_name,
                    "N": n_agents,
                    "K": 6,
                    "mechanism": mechanism,
                    "seed": seed,
                    "error": f"{type(exc).__name__}: {exc}",
                    "selected": False,
                }
            )

    screen = pd.DataFrame(scenario_records)
    if not screen.empty and {"dss_relative_gap_pct", "rm_relative_gap_pct"}.issubset(screen.columns):
        eligible = screen[
            (screen["dss_relative_gap_pct"] <= 1.0)
            & (screen["rm_relative_gap_pct"] >= 5.0)
        ].copy()
        eligible = eligible.sort_values("gap_difference_pct", ascending=False).head(2)
        if not eligible.empty:
            selected = set(eligible["scenario"])
            screen.loc[screen["scenario"].isin(selected), "selected"] = True
    return records, screen


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["scenario_family", "scenario", "solver", "rounds"]
    metrics = [
        "cce_gap_full_verification",
        "relative_gap_pct",
        "runtime_seconds",
        "full_verification_runtime_seconds",
        "support_size",
        "profile_coverage_pct",
        "objective_value",
    ]
    present = [metric for metric in metrics if metric in rows.columns]
    summary = rows.groupby(group_cols, dropna=False)[present].agg(["mean", "std", "min", "max", "count"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    return summary.reset_index()


def write_outputs(
    records: list[dict[str, Any]],
    definitions: list[dict[str, Any]],
    cmfg_screen: pd.DataFrame,
    output_dir: Path,
) -> None:
    raw_dir = output_dir / "raw"
    table_dir = output_dir / "tables"
    fig_dir = output_dir / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "synthetic_game_definitions.json").write_text(json.dumps(definitions, indent=2), encoding="utf-8")
    (raw_dir / "regret_matching_stress_results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    rows = pd.DataFrame(records)
    rows.to_csv(table_dir / "regret_matching_stress_solver_rows.csv", index=False)
    summarize_rows(rows).to_csv(table_dir / "regret_matching_stress_summary.csv", index=False)
    cmfg_screen.to_csv(table_dir / "regret_matching_stress_cmfg_screening.csv", index=False)
    make_stress_figure(rows, cmfg_screen, fig_dir)


def make_stress_figure(rows: pd.DataFrame, cmfg_screen: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), constrained_layout=True)

    synthetic = rows[
        (rows["scenario_family"] == "synthetic")
        & (rows["solver"] == "RegretMatching-PolicyTrace")
    ].copy()
    if not synthetic.empty:
        for scenario, group in synthetic.groupby("scenario"):
            means = group.groupby("rounds")["relative_gap_pct"].mean().sort_index()
            stds = group.groupby("rounds")["relative_gap_pct"].std().reindex(means.index).fillna(0.0)
            axes[0].errorbar(means.index, means.values, yerr=stds.values, marker="o", capsize=3, label=scenario)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Regret Matching rounds")
    axes[0].set_ylabel("Relative CCE gap (%)")
    axes[0].legend(frameon=False, fontsize=8)

    paired = rows[rows["scenario_family"] == "synthetic"].copy()
    rm500 = paired[(paired["solver"] == "RegretMatching-PolicyTrace") & (paired["rounds"] == 500)]
    dss = paired[paired["solver"] == "REPAIR-SAD-CCE"]
    if not rm500.empty and not dss.empty:
        dss_mean = dss.groupby("scenario")["relative_gap_pct"].mean()
        rm_mean = rm500.groupby("scenario")["relative_gap_pct"].mean()
        common = sorted(set(dss_mean.index).intersection(rm_mean.index))
        axes[1].scatter(
            [dss_mean[item] for item in common],
            [rm_mean[item] for item in common],
            s=55,
            color="#D07A2D",
        )
        for item in common:
            axes[1].annotate(item, (dss_mean[item], rm_mean[item]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_xlim(-0.02, 0.02)
    axes[1].set_xticks([0.0])
    axes[1].set_xticklabels(["$\\approx 0$"])
    axes[1].set_xlabel("DSS-CCE relative gap (%)")
    axes[1].set_ylabel("RM relative gap at 500 rounds (%)")

    if not cmfg_screen.empty and {"dss_relative_gap_pct", "rm_relative_gap_pct"}.issubset(cmfg_screen.columns):
        clean = cmfg_screen.dropna(subset=["dss_relative_gap_pct", "rm_relative_gap_pct"])
        selected = clean[clean.get("selected", False).astype(bool)] if "selected" in clean.columns else clean.iloc[0:0]
        axes[2].scatter(clean["dss_relative_gap_pct"], clean["rm_relative_gap_pct"], s=18, color="0.65", alpha=0.7)
        if not selected.empty:
            axes[2].scatter(selected["dss_relative_gap_pct"], selected["rm_relative_gap_pct"], s=55, color="#D07A2D")
        axes[2].axvline(1.0, color="0.35", linestyle="--", linewidth=1.0)
        axes[2].axhline(5.0, color="0.35", linestyle="--", linewidth=1.0)
        axes[2].set_xlabel("DSS-CCE relative gap (%)")
        axes[2].set_ylabel("RM relative gap (%)")
    else:
        axes[2].text(0.5, 0.5, "CMfg screening not run", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_axis_off()

    for idx, ax in enumerate(axes):
        ax.text(-0.08, 1.04, chr(ord("A") + idx), transform=ax.transAxes, fontsize=11, fontweight="bold")
        ax.grid(axis="both", color="0.90", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(fig_dir / "regret_matching_stress.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "regret_matching_stress.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_stress(
    output_dir: Path,
    skip_cmfg: bool,
    cmfg_limit: int | None,
    cmfg_rollouts: int,
    cmfg_workers: int,
    cmfg_rm_rounds: int,
) -> Path:
    records, definitions = run_synthetic_stress()
    if skip_cmfg:
        cmfg_records: list[dict[str, Any]] = []
        cmfg_screen = pd.DataFrame(
            [{"scenario": "cmfg_screening_skipped", "selected": False, "note": "CMfg screening skipped by CLI flag."}]
        )
    else:
        cmfg_records, cmfg_screen = run_cmfg_screening(
            limit=cmfg_limit,
            rollouts=cmfg_rollouts,
            workers=cmfg_workers,
            rm_rounds=cmfg_rm_rounds,
        )
    write_outputs(records + cmfg_records, definitions, cmfg_screen, output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Regret Matching failure stress tests for DSS-CCE.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-cmfg", action="store_true")
    parser.add_argument("--cmfg-limit", type=int, default=None)
    parser.add_argument("--cmfg-rollouts", type=int, default=3)
    parser.add_argument("--cmfg-workers", type=int, default=4)
    parser.add_argument("--cmfg-rm-rounds", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"regret_matching_stress_{stamp}"
    out = run_stress(
        output_dir=output_dir,
        skip_cmfg=bool(args.skip_cmfg),
        cmfg_limit=args.cmfg_limit,
        cmfg_rollouts=max(1, int(args.cmfg_rollouts)),
        cmfg_workers=max(1, int(args.cmfg_workers)),
        cmfg_rm_rounds=max(1, int(args.cmfg_rm_rounds)),
    )
    print(f"Wrote Regret Matching stress-test outputs to {out}")


if __name__ == "__main__":
    main()
