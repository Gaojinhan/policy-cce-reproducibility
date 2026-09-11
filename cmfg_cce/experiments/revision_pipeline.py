from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cmfg_cce.baselines.simple import mwu_policy_trace, mwu_policy_trace_time_budget
from cmfg_cce.evaluation.backend_cache import BackendPayoffCache
from cmfg_cce.evaluation.empirical_game import EmpiricalGame
from cmfg_cce.evaluation.independent_audit import (
    FrozenDistribution,
    freeze_distribution,
)
from cmfg_cce.evaluation.payoff_cache import support_deviation_closure
from cmfg_cce.evaluation.profile_backend import ProfileEvaluationBackend
from cmfg_cce.evaluation.revision_statistics import (
    pure_nash_diagnostics,
    support_diversity,
)
from cmfg_cce.evaluation.rollout import Profile
from cmfg_cce.experiments.common import run_solver
from cmfg_cce.solvers.cce_lp import CceSolution, solve_full_cce_lp
from cmfg_cce.solvers.cg_cce import solve_cg_cce_exhaustive
from cmfg_cce.solvers.sparse_cce import SparseCceResult


DEFAULT_SELECTOR = "platform_operating_score"
DEFAULT_EPSILON_TOLERANCE = 1.0e-9

# These raw fields are sufficient to reconstruct every reported CNC ratio,
# concentration index, family diagnostic, and machine-group capacity mean in
# each bootstrap draw.  Precomputed per-episode ratios are intentionally not
# used for formal uncertainty calculations.
CNC_RAW_OUTCOME_FIELDS = (
    "orders_offered_count",
    "assignment_count",
    "invitation_count",
    "valid_bid_count",
    "platform_total_payment",
    "manufacturer_discounted_profit_sum",
    "submitted_markup_sum",
    "submitted_markup_count",
    "submitted_lead_time_sum",
    "submitted_lead_time_count",
    "submitted_commitment_periods_sum",
    "capability_feasible_manufacturer_opportunities",
    "scalar_capacity_feasible_manufacturer_opportunities",
    "route_capacity_feasible_manufacturer_opportunities",
    "scalar_route_false_positive_opportunities",
    "orders_with_scalar_route_mismatch_count",
    *(f"wins_manufacturer_{index}_count" for index in range(4)),
    *(f"winner_policy_A{index}_count" for index in range(1, 9)),
    *(f"orders_offered_F{index}" for index in range(1, 5)),
    *(f"orders_assigned_F{index}" for index in range(1, 5)),
    *(
        f"capability_feasible_manufacturer_opportunities_F{index}"
        for index in range(1, 5)
    ),
    *(
        f"scalar_capacity_feasible_manufacturer_opportunities_F{index}"
        for index in range(1, 5)
    ),
    *(
        f"route_capacity_feasible_manufacturer_opportunities_F{index}"
        for index in range(1, 5)
    ),
    *(f"scalar_route_false_positive_opportunities_F{index}" for index in range(1, 5)),
    *(
        f"orders_with_at_least_three_route_capacity_feasible_manufacturers_F{index}"
        for index in range(1, 5)
    ),
    *(f"machine_group_remaining_capacity_sum_{group}" for group in ("T", "M3", "M5", "G", "EDM")),
    *(f"machine_group_snapshot_count_{group}" for group in ("T", "M3", "M5", "G", "EDM")),
)


def empirical_game_from_replications(
    profiles: Sequence[Profile],
    policy_ids: Sequence[str],
    profile_returns: Mapping[Profile, np.ndarray],
    platform_objectives: Mapping[Profile, Sequence[float]],
) -> EmpiricalGame:
    ordered = tuple(tuple(profile) for profile in profiles)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("The empirical game requires unique profiles in frozen order.")
    counts = {
        np.asarray(profile_returns[profile], dtype=float).shape[0] for profile in ordered
    }
    if len(counts) != 1 or next(iter(counts)) < 2:
        raise ValueError("Every training profile must have the same R >= 2.")
    rollouts = counts.pop()
    n_agents = len(ordered[0])
    returns = np.stack([np.asarray(profile_returns[profile], dtype=float) for profile in ordered])
    if returns.shape != (len(ordered), rollouts, n_agents) or not np.all(np.isfinite(returns)):
        raise ValueError("Training payoff replication arrays have invalid shape or values.")
    objectives = np.stack(
        [np.asarray(platform_objectives[profile], dtype=float) for profile in ordered]
    )
    if objectives.shape != (len(ordered), rollouts) or not np.all(np.isfinite(objectives)):
        raise ValueError("Training platform-objective arrays have invalid shape or values.")
    mean_returns = np.mean(returns, axis=1)
    variance = np.var(returns, axis=1, ddof=1)
    mean_objectives = np.mean(objectives, axis=1)
    return EmpiricalGame(
        profiles=ordered,
        policy_ids=tuple(str(value) for value in policy_ids),
        payoffs=mean_returns,
        ci_radius=1.96 * np.sqrt(variance / rollouts),
        objectives=mean_objectives,
        metrics=tuple(
            {"platform_operating_score": float(value)} for value in mean_objectives
        ),
    )


def freeze_solution(
    solution: CceSolution | SparseCceResult,
    *,
    n_agents: int,
    policy_ids: Sequence[str],
) -> FrozenDistribution:
    return freeze_distribution(
        solution.solver,
        solution.support_profiles,
        solution.support_probabilities,
        n_agents,
        policy_ids,
    )


def complete_solver_bundle(
    game: EmpiricalGame,
    *,
    selector: str = DEFAULT_SELECTOR,
    epsilon_tolerance: float = DEFAULT_EPSILON_TOLERANCE,
    require_shared_q: bool = True,
) -> tuple[CceSolution, CceSolution]:
    full = solve_full_cce_lp(
        game,
        selector=selector,
        epsilon_tolerance=epsilon_tolerance,
    )
    cg = solve_cg_cce_exhaustive(
        game,
        selector=selector,
        epsilon_tolerance=epsilon_tolerance,
    )
    if require_shared_q and not np.allclose(full.q, cg.q, atol=1.0e-8, rtol=0.0):
        raise RuntimeError(
            "FullTensor and ExhaustiveCG returned different q vectors; a shared fresh "
            "audit is not valid for this game."
        )
    return full, cg


def default_dss_config(*, train_rollouts: int, workers: int) -> dict[str, Any]:
    return {
        "workers": int(workers),
        "target_gap": 1.0e-8,
        "active_sampling": False,
        "rollouts_max": int(train_rollouts),
        "initial_support_size": 24,
        "max_support_size": 180,
        "support_add_batch_size": 16,
        "max_rounds": 5,
        "selector": DEFAULT_SELECTOR,
        "epsilon_tolerance": DEFAULT_EPSILON_TOLERANCE,
        "repair": {
            "repair_rounds": 3,
            "top_constraints": 5,
            "top_profiles_per_constraint": 24,
            "q_min": 1.0e-4,
            "mean_threshold": 1.0,
            "contribution_min": 0.0,
            "enable_multi_agent_repair": True,
            "max_agents_repaired_per_profile": 2,
            "beam_width": 24,
            "profile_budget_multiplier": 2.0,
            "targeted_deviation_policies": {
                "M1_price_first": ["A6", "A3"],
                "M2_price_critical": ["A6", "A3"],
                "M3_delivery_first": ["A5"],
                "M4_delivery_critical": ["A1", "A5"],
            },
        },
    }


def run_dss_empty_cache(
    backend: ProfileEvaluationBackend,
    *,
    train_rollouts: int,
    workers: int,
    solver_seed: int,
    selector: str = DEFAULT_SELECTOR,
) -> tuple[SparseCceResult, BackendPayoffCache, dict[str, float]]:
    cache = BackendPayoffCache(backend=backend, n_rollouts=int(train_rollouts), workers=int(workers))
    cache.reset_access_log()
    config = default_dss_config(train_rollouts=train_rollouts, workers=workers)
    config["selector"] = selector
    started = time.perf_counter()
    result = run_solver(
        "REPAIR-SAD-CCE",
        cache,
        tuple(backend.policies),  # type: ignore[attr-defined]
        config,
        seed=int(solver_seed),
        metadata={"revision_campaign": "revision-full-v1"},
    )
    elapsed = time.perf_counter() - started
    stats = cache.access_stats()
    stats["runtime_seconds"] = float(elapsed)
    stats["payoff_evaluation_seconds"] = float(cache.eval_time_seconds)
    stats["solver_compute_seconds"] = max(0.0, elapsed - cache.eval_time_seconds)
    return result, cache, stats


def run_mwu_empty_cache(
    backend: ProfileEvaluationBackend,
    *,
    train_rollouts: int,
    workers: int,
    solver_seed: int,
    budget_seconds: float,
    eta: float,
    schedule: str,
    exploration_floor: float,
    burn_in_rounds: int,
) -> tuple[SparseCceResult, BackendPayoffCache, dict[str, float]]:
    cache = BackendPayoffCache(backend=backend, n_rollouts=int(train_rollouts), workers=int(workers))
    cache.reset_access_log()
    started = time.perf_counter()
    result = mwu_policy_trace_time_budget(
        cache,
        tuple(backend.policies),  # type: ignore[attr-defined]
        budget_seconds=float(budget_seconds),
        seed=int(solver_seed),
        eta=float(eta),
        schedule=str(schedule),
        exploration_floor=float(exploration_floor),
        burn_in_rounds=int(burn_in_rounds),
    )
    elapsed = time.perf_counter() - started
    stats = cache.access_stats()
    stats["runtime_seconds"] = float(elapsed)
    stats["payoff_evaluation_seconds"] = float(cache.eval_time_seconds)
    stats["solver_compute_seconds"] = max(0.0, elapsed - cache.eval_time_seconds)
    return result, cache, stats


def run_mwu_rounds_empty_cache(
    backend: ProfileEvaluationBackend,
    *,
    train_rollouts: int,
    workers: int,
    solver_seed: int,
    rounds: int,
    eta: float,
    schedule: str,
    exploration_floor: float,
    burn_in_rounds: int,
) -> tuple[SparseCceResult, BackendPayoffCache, dict[str, float]]:
    """Run the formal MWU trace with a host-independent round budget."""

    if int(rounds) <= int(burn_in_rounds):
        raise ValueError("Formal MWU rounds must exceed burn-in rounds.")
    cache = BackendPayoffCache(
        backend=backend,
        n_rollouts=int(train_rollouts),
        workers=int(workers),
    )
    cache.reset_access_log()
    started = time.perf_counter()
    result = mwu_policy_trace(
        cache,
        tuple(backend.policies),  # type: ignore[attr-defined]
        rounds=int(rounds),
        seed=int(solver_seed),
        eta=float(eta),
        schedule=str(schedule),
        exploration_floor=float(exploration_floor),
        burn_in_rounds=int(burn_in_rounds),
    )
    elapsed = time.perf_counter() - started
    stats = cache.access_stats()
    stats["runtime_seconds"] = float(elapsed)
    stats["payoff_evaluation_seconds"] = float(cache.eval_time_seconds)
    stats["solver_compute_seconds"] = max(0.0, elapsed - cache.eval_time_seconds)
    stats["formal_rounds"] = int(rounds)
    return result, cache, stats


def frozen_closure(
    distributions: Sequence[FrozenDistribution],
    policy_ids: Sequence[str],
) -> tuple[Profile, ...]:
    if not distributions:
        raise ValueError("At least one frozen q is required.")
    profiles: set[Profile] = set()
    for distribution in distributions:
        profiles.update(support_deviation_closure(distribution.support, tuple(policy_ids)))
    return tuple(sorted(profiles))


def solution_diagnostics(
    distribution: FrozenDistribution,
    *,
    training_game: EmpiricalGame | None = None,
) -> dict[str, Any]:
    diversity = support_diversity(distribution.probabilities)
    payload: dict[str, Any] = {
        "solver": distribution.solver,
        "q_hash": distribution.q_hash,
        "support": [
            {"profile": list(profile), "probability": probability}
            for profile, probability in zip(
                distribution.support, distribution.probabilities, strict=True
            )
        ],
        "support_diversity": asdict(diversity),
    }
    if training_game is not None:
        pure = pure_nash_diagnostics(training_game)
        payload["pure_nash_count"] = len(pure.pure_nash_profiles)
        payload["pure_nash_profiles"] = [list(profile) for profile in pure.pure_nash_profiles]
        payload["weakly_dominated_policies"] = {
            str(agent): list(policies)
            for agent, policies in pure.weakly_dominated_policies.items()
        }
    return payload


def stable_solver_seed(*parts: object) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (
        2**31 - 1
    )
