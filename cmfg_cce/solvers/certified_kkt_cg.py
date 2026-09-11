from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linprog

from cmfg_cce.evaluation.empirical_game import EmpiricalGame
from cmfg_cce.evaluation.payoff_cache import PayoffCache, profile_space, random_profiles, support_deviation_closure
from cmfg_cce.evaluation.rollout import Profile
from cmfg_cce.solvers.cce_lp import (
    CceSolution,
    compute_cce_gap,
    compute_cce_gap_ucb,
    deviation_rows,
    distribution_support,
    max_deviation,
    max_deviation_ucb,
)
from cmfg_cce.solvers.sparse_cce import SparseCceResult, audit_sparse_distribution, deviation_matrix


@dataclass
class RestrictedMasterResult:
    support: list[Profile]
    probabilities: np.ndarray
    t_value: float
    row_gains: np.ndarray
    row_weights: np.ndarray
    result: Any
    dual_weight_sum: float
    dual_mode: str


@dataclass
class CertifiedCgTrace:
    support: list[Profile]
    probabilities: np.ndarray
    ub: float
    lb: float
    certificate_gap: float
    certified: bool
    iterations: int
    pricing_mode: str
    log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SparseMasterResult:
    support: list[Profile]
    probabilities: np.ndarray
    t_value: float
    row_gains: np.ndarray
    row_weights: np.ndarray
    labels: list[tuple[int, str]]
    objective_value: float
    result: Any
    dual_weight_sum: float
    dual_mode: str


def _unique_profiles(profiles: list[Profile]) -> list[Profile]:
    return list(dict.fromkeys(tuple(profile) for profile in profiles))


def _solve_restricted_master(game: EmpiricalGame, support: list[Profile]) -> RestrictedMasterResult:
    if not support:
        raise ValueError("Restricted master requires a nonempty support.")
    a = deviation_rows(game, support)
    n_support = len(support)
    c = np.zeros(n_support + 1, dtype=float)
    c[-1] = 1.0
    a_ub = np.hstack([a, -np.ones((a.shape[0], 1), dtype=float)])
    a_eq = np.zeros((1, n_support + 1), dtype=float)
    a_eq[0, :n_support] = 1.0
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=np.zeros(a.shape[0], dtype=float),
        A_eq=a_eq,
        b_eq=np.array([1.0], dtype=float),
        bounds=[(0.0, None)] * n_support + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Certified CG restricted master failed: {result.message}")
    probabilities = np.maximum(result.x[:n_support], 0.0)
    probabilities = probabilities / np.sum(probabilities)
    row_gains = a @ probabilities
    t_value = float(np.max(row_gains))

    raw_duals = np.maximum(0.0, -np.array(result.ineqlin.marginals, dtype=float))
    dual_sum = float(np.sum(raw_duals))
    dual_mode = "lp_dual"
    if dual_sum > 1.0e-12:
        row_weights = raw_duals / dual_sum
    else:
        active = np.flatnonzero(row_gains >= np.max(row_gains) - 1.0e-8)
        row_weights = np.zeros_like(raw_duals)
        row_weights[active] = 1.0 / max(1, len(active))
        dual_mode = "active_row_fallback"
    return RestrictedMasterResult(
        support=list(support),
        probabilities=probabilities,
        t_value=t_value,
        row_gains=row_gains,
        row_weights=row_weights,
        result=result,
        dual_weight_sum=dual_sum,
        dual_mode=dual_mode,
    )


def _pricing_values(game: EmpiricalGame, row_weights: np.ndarray, candidates: list[Profile]) -> np.ndarray:
    if float(np.sum(np.abs(row_weights))) <= 1.0e-15:
        return np.zeros(len(candidates), dtype=float)
    columns = deviation_rows(game, candidates)
    return columns.T @ row_weights


def _exact_pricing(game: EmpiricalGame, row_weights: np.ndarray) -> tuple[float, Profile]:
    candidates = list(game.profiles)
    values = _pricing_values(game, row_weights, candidates)
    best_idx = int(np.argmin(values))
    return float(values[best_idx]), candidates[best_idx]


def _full_q(game: EmpiricalGame, support: list[Profile], probabilities: np.ndarray) -> np.ndarray:
    q = np.zeros(game.n_profiles, dtype=float)
    profile_to_index = game.profile_to_index
    for profile, prob in zip(support, probabilities, strict=True):
        q[profile_to_index[profile]] += float(prob)
    total = float(np.sum(q))
    if total <= 0.0:
        raise RuntimeError("Certified CG produced a zero-mass distribution.")
    return q / total


def _default_initial_support(game: EmpiricalGame, seed: int, initial_support_size: int) -> list[Profile]:
    rng = np.random.default_rng(seed)
    size = min(max(1, int(initial_support_size)), game.n_profiles)
    if size == game.n_profiles:
        return list(game.profiles)
    indices = list(rng.choice(game.n_profiles, size=size, replace=False))
    return [game.profiles[int(idx)] for idx in indices]


def _solve_sparse_restricted_master(
    cache: PayoffCache,
    support: list[Profile],
    policy_ids: tuple[str, ...],
) -> SparseMasterResult:
    support = _unique_profiles(support)
    if not support:
        raise ValueError("Sparse restricted master requires a nonempty support.")
    cache.ensure(support_deviation_closure(support, policy_ids))
    a, labels = deviation_matrix(cache, support, policy_ids, ucb=False)
    n_support = len(support)
    c = np.zeros(n_support + 1, dtype=float)
    c[-1] = 1.0
    a_ub = np.hstack([a, -np.ones((a.shape[0], 1), dtype=float)])
    a_eq = np.zeros((1, n_support + 1), dtype=float)
    a_eq[0, :n_support] = 1.0
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=np.zeros(a.shape[0], dtype=float),
        A_eq=a_eq,
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * n_support + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Heuristic KKT-CG restricted master failed: {result.message}")
    probabilities = np.maximum(result.x[:n_support], 0.0)
    probabilities = probabilities / np.sum(probabilities)
    row_gains = a @ probabilities
    raw_duals = np.maximum(0.0, -np.array(result.ineqlin.marginals, dtype=float))
    dual_sum = float(np.sum(raw_duals))
    dual_mode = "lp_dual"
    if dual_sum > 1.0e-12:
        row_weights = raw_duals / dual_sum
    else:
        active = np.flatnonzero(row_gains >= np.max(row_gains) - 1.0e-8)
        row_weights = np.zeros_like(raw_duals)
        row_weights[active] = 1.0 / max(1, len(active))
        dual_mode = "active_row_fallback"
    objective = float(
        sum(prob * cache.support_objective(profile) for profile, prob in zip(support, probabilities, strict=True))
    )
    return SparseMasterResult(
        support=support,
        probabilities=probabilities,
        t_value=float(np.max(row_gains)),
        row_gains=row_gains,
        row_weights=row_weights,
        labels=labels,
        objective_value=objective,
        result=result,
        dual_weight_sum=dual_sum,
        dual_mode=dual_mode,
    )


def _weighted_deviation_objective_cache(
    cache: PayoffCache,
    profile: Profile,
    row_weights: np.ndarray,
    labels: list[tuple[int, str]],
) -> float:
    base_by_agent: dict[int, float] = {}
    total = 0.0
    for weight, (agent, dev_policy) in zip(row_weights, labels, strict=True):
        weight_f = float(weight)
        if abs(weight_f) <= 1.0e-15 or profile[agent] == dev_policy:
            continue
        if agent not in base_by_agent:
            base_by_agent[agent] = cache.payoff(profile, agent)
        dev_profile = list(profile)
        dev_profile[agent] = dev_policy
        gain = cache.payoff(tuple(dev_profile), agent) - base_by_agent[agent]
        total += weight_f * gain
    return float(total)


def _best_coordinate_descent_profile(
    cache: PayoffCache,
    start: Profile,
    row_weights: np.ndarray,
    labels: list[tuple[int, str]],
    policy_ids: tuple[str, ...],
    local_steps: int,
) -> tuple[Profile, float, int]:
    profile = tuple(start)
    value = _weighted_deviation_objective_cache(cache, profile, row_weights, labels)
    objective_calls = 1
    for _ in range(max(1, int(local_steps))):
        improved = False
        for agent in range(cache.n_agents):
            best_profile = profile
            best_value = value
            for policy_id in policy_ids:
                candidate = list(profile)
                candidate[agent] = policy_id
                candidate_t = tuple(candidate)
                candidate_value = _weighted_deviation_objective_cache(cache, candidate_t, row_weights, labels)
                objective_calls += 1
                if candidate_value < best_value - 1.0e-12:
                    best_profile = candidate_t
                    best_value = candidate_value
            if best_profile != profile:
                profile = best_profile
                value = best_value
                improved = True
        if not improved:
            break
    return profile, float(value), int(objective_calls)


def _heuristic_pricing(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    row_weights: np.ndarray,
    labels: list[tuple[int, str]],
    support: list[Profile],
    probabilities: np.ndarray,
    rng: np.random.Generator,
    sample_size: int,
    restarts: int,
    local_steps: int,
) -> tuple[float, Profile, dict[str, Any]]:
    candidates: list[Profile] = []
    candidates.extend(support)
    ranked_support = [
        profile
        for profile, _ in sorted(zip(support, probabilities, strict=True), key=lambda item: item[1], reverse=True)
    ]
    candidates.extend(ranked_support[: max(1, int(restarts))])
    candidates.extend(random_profiles(policy_ids, cache.n_agents, max(1, int(sample_size)), rng))
    candidates = _unique_profiles(candidates)

    best_profile: Profile | None = None
    best_value = float("inf")
    objective_calls = 0
    local_start_count = 0
    for candidate in candidates:
        value = _weighted_deviation_objective_cache(cache, candidate, row_weights, labels)
        objective_calls += 1
        if value < best_value:
            best_value = value
            best_profile = candidate
    start_pool = _unique_profiles(ranked_support[: max(1, int(restarts))] + candidates[: max(1, int(restarts))])
    for start in start_pool[: max(1, int(restarts))]:
        local_start_count += 1
        local_profile, local_value, calls = _best_coordinate_descent_profile(
            cache,
            start,
            row_weights,
            labels,
            policy_ids,
            local_steps=local_steps,
        )
        objective_calls += calls
        if local_value < best_value:
            best_value = local_value
            best_profile = local_profile
    if best_profile is None:
        raise RuntimeError("Heuristic pricing did not produce a candidate profile.")
    return best_value, best_profile, {
        "pricing_certified": False,
        "pricing_method": "sampled_coordinate_local_search",
        "candidate_count": len(candidates),
        "local_start_count": int(local_start_count),
        "objective_calls": int(objective_calls),
    }


def run_certified_kkt_cg_trace(
    game: EmpiricalGame,
    initial_support: list[Profile] | None = None,
    initial_support_size: int = 1,
    tolerance: float = 1.0e-8,
    max_iterations: int | None = None,
    seed: int = 0,
) -> CertifiedCgTrace:
    if game.n_profiles == 0:
        raise ValueError("Cannot solve an empty empirical game.")
    support = _unique_profiles(
        list(initial_support) if initial_support is not None else _default_initial_support(game, seed, initial_support_size)
    )
    max_iterations = max_iterations or (game.n_profiles + 5)
    best_master: RestrictedMasterResult | None = None
    best_ub = float("inf")
    best_lb = float("-inf")
    log: list[dict[str, Any]] = []
    certified = False

    for iteration in range(int(max_iterations)):
        master = _solve_restricted_master(game, support)
        ub = float(np.max(master.row_gains))
        if ub < best_ub:
            best_ub = ub
            best_master = master

        pricing_value, pricing_profile = _exact_pricing(game, master.row_weights)
        best_lb = max(best_lb, float(pricing_value))
        certificate_gap = max(0.0, float(ub - best_lb))
        log.append(
            {
                "iteration": int(iteration),
                "support_size": len(support),
                "restricted_value": float(master.t_value),
                "primal_UB": float(ub),
                "dual_LB": float(best_lb),
                "certificate_gap": certificate_gap,
                "pricing_value": float(pricing_value),
                "pricing_profile": list(pricing_profile),
                "pricing_certified": True,
                "dual_mode": master.dual_mode,
                "dual_weight_sum": float(master.dual_weight_sum),
            }
        )
        if pricing_value >= master.t_value - tolerance or certificate_gap <= tolerance:
            certified = True
            break
        if pricing_profile not in support:
            support.append(pricing_profile)
            continue
        raise RuntimeError(
            "Exact pricing returned an already-supported violating profile; "
            "check restricted-master dual extraction."
        )

    if best_master is None:
        raise RuntimeError("Certified CG did not solve any restricted master.")
    final_support = list(best_master.support)
    final_probabilities = np.array(best_master.probabilities, dtype=float)
    final_gap = max(0.0, float(best_ub - best_lb))
    return CertifiedCgTrace(
        support=final_support,
        probabilities=final_probabilities,
        ub=float(best_ub),
        lb=float(best_lb),
        certificate_gap=final_gap,
        certified=bool(certified and final_gap <= max(tolerance, 1.0e-10)),
        iterations=len(log),
        pricing_mode="generic_lp_exact_pricing",
        log=log,
    )


def solve_certified_kkt_cg_cce(
    game: EmpiricalGame,
    initial_support: list[Profile] | None = None,
    initial_support_size: int = 1,
    tolerance: float = 1.0e-8,
    max_iterations: int | None = None,
    seed: int = 0,
) -> CceSolution:
    trace = run_certified_kkt_cg_trace(
        game,
        initial_support=initial_support,
        initial_support_size=initial_support_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
        seed=seed,
    )
    q = _full_q(game, trace.support, trace.probabilities)
    support_profiles, support_probabilities = distribution_support(game, q)
    ucb_dev = max_deviation_ucb(game, q)
    nominal_dev = max_deviation(game, q)
    nominal_dev["gain_ucb"] = ucb_dev["gain_ucb"]
    nominal_dev["ucb_agent"] = ucb_dev["agent"]
    nominal_dev["ucb_policy"] = ucb_dev["policy"]
    gap = compute_cce_gap(game, q)
    ucb_gap = compute_cce_gap_ucb(game, q)
    cert_gap = trace.certificate_gap
    return CceSolution(
        solver="CertifiedKKT-CG-CCE",
        q=q,
        objective_value=float(np.dot(game.objectives, q)),
        cce_gap_nominal=gap,
        cce_gap_ucb=ucb_gap,
        support_profiles=support_profiles,
        support_probabilities=support_probabilities,
        status="certified" if trace.certified else "iteration_limit_or_uncertified",
        full_optimality_certified=bool(trace.certified and cert_gap <= tolerance),
        pricing_mode=trace.pricing_mode,
        max_reduced_cost=trace.certificate_gap,
        pricing_iterations=trace.iterations,
        max_deviation=nominal_dev,
        lower_bound=float(trace.lb),
        certificate_gap=cert_gap,
        diagnostics={"column_generation_log": trace.log},
    )


def empirical_game_from_cache(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    profiles: list[Profile] | None = None,
    objective_key: str = "platform_operating_score",
) -> EmpiricalGame:
    profile_list = profiles if profiles is not None else list(profile_space(policy_ids, cache.n_agents))
    cache.ensure(profile_list)
    payoffs = np.vstack([cache.get(profile).mean_returns for profile in profile_list])
    ci_radius = np.vstack([cache.get(profile).ci_radius for profile in profile_list])
    objectives = np.array(
        [cache.get(profile).mean_metrics.get(objective_key, 0.0) for profile in profile_list],
        dtype=float,
    )
    metrics = tuple(cache.get(profile).mean_metrics for profile in profile_list)
    return EmpiricalGame(
        profiles=tuple(profile_list),
        policy_ids=policy_ids,
        payoffs=payoffs,
        ci_radius=ci_radius,
        objectives=objectives,
        metrics=metrics,
    )


def solve_certified_kkt_cg_sparse(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    initial_support_size: int = 1,
    tolerance: float = 1.0e-8,
    max_iterations: int | None = None,
    seed: int = 0,
) -> SparseCceResult:
    profiles = list(profile_space(policy_ids, cache.n_agents))
    game = empirical_game_from_cache(cache, policy_ids, profiles=profiles)
    solution = solve_certified_kkt_cg_cce(
        game,
        initial_support_size=initial_support_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
        seed=seed,
    )
    return SparseCceResult(
        solver="CertifiedKKT-CG-CCE",
        support_profiles=solution.support_profiles,
        support_probabilities=solution.support_probabilities,
        objective_value=solution.objective_value,
        cce_gap_nominal=solution.cce_gap_nominal,
        cce_gap_ucb=solution.cce_gap_ucb,
        max_deviation=solution.max_deviation or {},
        status=solution.status,
        certificate_status="certified_kkt_column_generation"
        if solution.full_optimality_certified
        else "certified_kkt_column_generation_uncertified",
        pricing_mode=solution.pricing_mode,
        full_optimality_certified=solution.full_optimality_certified,
        support_expansion_rounds=solution.pricing_iterations or 0,
        diagnostics={
            **solution.diagnostics,
            "dual_lower_bound": solution.lower_bound,
            "certificate_gap": solution.certificate_gap,
        },
    )


def solve_heuristic_kkt_cg_sparse(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    initial_support_size: int = 12,
    max_support_size: int = 80,
    tolerance: float = 1.0e-8,
    max_iterations: int = 8,
    pricing_sample_size: int = 96,
    pricing_restarts: int = 8,
    pricing_local_steps: int = 4,
    seed: int = 0,
) -> SparseCceResult:
    rng = np.random.default_rng(seed)
    support = random_profiles(policy_ids, cache.n_agents, max(1, int(initial_support_size)), rng)
    best_master: SparseMasterResult | None = None
    best_clipped_gap = float("inf")
    log: list[dict[str, Any]] = []
    stopped_reason = "iteration_limit"

    for iteration in range(max(1, int(max_iterations))):
        master = _solve_sparse_restricted_master(cache, support, policy_ids)
        clipped_gap = max(0.0, float(np.max(master.row_gains)))
        if clipped_gap < best_clipped_gap:
            best_clipped_gap = clipped_gap
            best_master = master
        pricing_value, pricing_profile, pricing_info = _heuristic_pricing(
            cache,
            policy_ids,
            master.row_weights,
            master.labels,
            master.support,
            master.probabilities,
            rng,
            sample_size=pricing_sample_size,
            restarts=pricing_restarts,
            local_steps=pricing_local_steps,
        )
        improvement = float(master.t_value - pricing_value)
        log.append(
            {
                "iteration": int(iteration),
                "support_size": len(support),
                "restricted_value": float(master.t_value),
                "support_clipped_gap": clipped_gap,
                "heuristic_pricing_value": float(pricing_value),
                "heuristic_improvement": improvement,
                "pricing_profile": list(pricing_profile),
                "dual_mode": master.dual_mode,
                "dual_weight_sum": float(master.dual_weight_sum),
                **pricing_info,
            }
        )
        if clipped_gap <= tolerance:
            stopped_reason = "support_solution_reaches_target_gap"
            break
        if pricing_profile in support or improvement <= tolerance:
            stopped_reason = "no_heuristic_improving_column"
            break
        if len(support) >= max(1, int(max_support_size)):
            stopped_reason = "max_support_size"
            break
        support.append(pricing_profile)
        support = _unique_profiles(support)[: max(1, int(max_support_size))]

    if best_master is None:
        raise RuntimeError("Heuristic KKT-CG did not solve any restricted master.")
    audited = audit_sparse_distribution(
        cache,
        best_master.support,
        [float(prob) for prob in best_master.probabilities],
        policy_ids,
        solver_name="HeuristicKKT-CG-CCE",
        status=stopped_reason,
    )
    audited.pricing_mode = "sampled_coordinate_local_pricing"
    audited.certificate_status = "heuristic_pricing_full_tensor_audit_required"
    audited.full_optimality_certified = False
    audited.support_expansion_rounds = len(log)
    audited.diagnostics = {
        "column_generation_log": log,
        "pricing_certified": False,
        "stopped_reason": stopped_reason,
        "target_gap": float(tolerance),
        "max_support_size": int(max_support_size),
        "pricing_sample_size": int(pricing_sample_size),
        "pricing_restarts": int(pricing_restarts),
        "pricing_local_steps": int(pricing_local_steps),
    }
    return audited
