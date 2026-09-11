from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Iterable

import numpy as np

from cmfg_cce.evaluation.empirical_game import EmpiricalGame
from cmfg_cce.evaluation.payoff_cache import PayoffCache, profile_space
from cmfg_cce.evaluation.rollout import Profile
from cmfg_cce.solvers.cce_lp import (
    CceSolution,
    compute_cce_gap,
    compute_cce_gap_ucb,
    distribution_support,
    max_deviation,
    max_deviation_ucb,
)
from cmfg_cce.solvers.sparse_cce import SparseCceResult, audit_sparse_distribution


PayoffFn = Callable[[Profile, int], float]
ObjectiveFn = Callable[[Profile], float]
DeviationRow = tuple[int, str]


@dataclass
class TCarmTrace:
    support_profiles: list[Profile]
    support_probabilities: list[float]
    ub: float
    lb: float
    certificate_gap: float
    lower_bound_certified: bool
    generated_profiles: list[Profile]
    deviation_rows: list[DeviationRow]
    max_deviation: dict[str, Any]
    objective_value: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def deviation_row_labels(n_agents: int, policy_ids: tuple[str, ...]) -> list[DeviationRow]:
    return [(agent, policy_id) for agent in range(n_agents) for policy_id in policy_ids]


def deviated_profile(profile: Profile, agent: int, dev_policy: str) -> Profile:
    if profile[agent] == dev_policy:
        return profile
    updated = list(profile)
    updated[agent] = dev_policy
    return tuple(updated)


def deviation_gain(row: DeviationRow, profile: Profile, payoff: PayoffFn) -> float:
    agent, dev_policy = row
    if profile[agent] == dev_policy:
        return 0.0
    return float(payoff(deviated_profile(profile, agent, dev_policy), agent) - payoff(profile, agent))


def deviation_gain_vector(rows: list[DeviationRow], profile: Profile, payoff: PayoffFn) -> np.ndarray:
    return np.array([deviation_gain(row, profile, payoff) for row in rows], dtype=float)


def weighted_deviation_objective(
    profile: Profile,
    weights: np.ndarray,
    rows: list[DeviationRow],
    payoff: PayoffFn,
) -> float:
    total = 0.0
    for weight, row in zip(weights, rows, strict=True):
        if abs(float(weight)) > 0.0:
            total += float(weight) * deviation_gain(row, profile, payoff)
    return float(total)


def merge_profiles_to_distribution(profiles: Iterable[Profile]) -> tuple[list[Profile], list[float]]:
    counts = Counter(tuple(profile) for profile in profiles)
    total = sum(counts.values())
    if total <= 0:
        return [], []
    support = list(counts.keys())
    probabilities = [float(count / total) for count in counts.values()]
    return support, probabilities


def exact_anti_deviation_oracle(
    weights: np.ndarray,
    rows: list[DeviationRow],
    candidate_profiles: Iterable[Profile],
    payoff: PayoffFn,
) -> tuple[Profile, dict[str, Any]]:
    best_profile: Profile | None = None
    best_value = float("inf")
    for profile in candidate_profiles:
        profile_t = tuple(profile)
        value = weighted_deviation_objective(profile_t, weights, rows, payoff)
        if value < best_value - 1.0e-12:
            best_value = value
            best_profile = profile_t
    if best_profile is None:
        raise ValueError("Exact anti-deviation oracle received no candidate profiles.")
    return best_profile, {
        "objective": float(best_value),
        "delta": 0.0,
        "method": "exact_enumeration",
        "certified": True,
    }


def nested_rm_anti_deviation_oracle(
    weights: np.ndarray,
    rows: list[DeviationRow],
    policy_ids: tuple[str, ...],
    n_agents: int,
    payoff: PayoffFn,
    rng: np.random.Generator,
    inner_iterations: int = 40,
    restarts: int = 6,
    epsilon: float = 0.05,
) -> tuple[Profile, dict[str, Any]]:
    if not policy_ids:
        raise ValueError("Nested anti-deviation oracle requires at least one policy.")
    best_profile: Profile | None = None
    best_value = float("inf")
    objective_calls = 0
    policy_count = len(policy_ids)

    for restart in range(max(1, int(restarts))):
        if restart == 0:
            profile = tuple(policy_ids[0] for _ in range(n_agents))
        else:
            profile = tuple(policy_ids[int(rng.integers(0, policy_count))] for _ in range(n_agents))
        anti_regrets = np.zeros((n_agents, policy_count), dtype=float)
        current_value = weighted_deviation_objective(profile, weights, rows, payoff)
        objective_calls += 1
        if current_value < best_value:
            best_value = current_value
            best_profile = profile

        for _ in range(max(1, int(inner_iterations))):
            changed = False
            for agent in range(n_agents):
                candidate_values = np.empty(policy_count, dtype=float)
                for policy_idx, policy_id in enumerate(policy_ids):
                    candidate = deviated_profile(profile, agent, policy_id)
                    candidate_values[policy_idx] = weighted_deviation_objective(candidate, weights, rows, payoff)
                    objective_calls += 1
                anti_regrets[agent] += current_value - candidate_values
                positive = np.maximum(anti_regrets[agent], 0.0)
                if float(np.sum(positive)) > 0.0:
                    probs = positive / np.sum(positive)
                    probs = (1.0 - epsilon) * probs + epsilon / policy_count
                    selected_idx = int(rng.choice(policy_count, p=probs))
                else:
                    min_value = float(np.min(candidate_values))
                    best_candidates = np.flatnonzero(candidate_values <= min_value + 1.0e-12)
                    selected_idx = int(rng.choice(best_candidates))
                selected_policy = policy_ids[selected_idx]
                next_profile = deviated_profile(profile, agent, selected_policy)
                next_value = float(candidate_values[selected_idx])
                if next_profile != profile:
                    changed = True
                profile = next_profile
                current_value = next_value
                if current_value < best_value:
                    best_value = current_value
                    best_profile = profile
            if not changed:
                break

    if best_profile is None:
        raise RuntimeError("Nested anti-deviation oracle did not produce a profile.")
    return best_profile, {
        "objective": float(best_value),
        "delta": None,
        "method": "nested_regret_matching_coordinate_search",
        "certified": False,
        "objective_calls": int(objective_calls),
    }


def _upper_bound(
    support: list[Profile],
    probabilities: list[float],
    rows: list[DeviationRow],
    payoff: PayoffFn,
) -> tuple[float, dict[str, Any]]:
    row_values = []
    for row in rows:
        value = 0.0
        for profile, prob in zip(support, probabilities, strict=True):
            value += float(prob) * deviation_gain(row, profile, payoff)
        row_values.append(value)
    values = np.array(row_values, dtype=float)
    max_idx = int(np.argmax(values))
    agent, policy = rows[max_idx]
    raw_gain = float(values[max_idx])
    gain = max(0.0, raw_gain)
    return raw_gain, {
        "agent": int(agent),
        "policy": policy,
        "gain_nominal": gain,
        "raw_gain_nominal": raw_gain,
    }


def solve_t_carm_from_payoff(
    policy_ids: tuple[str, ...],
    n_agents: int,
    payoff: PayoffFn,
    objective: ObjectiveFn | None = None,
    iterations: int = 500,
    seed: int = 0,
    oracle_mode: str = "exact",
    exact_profiles: Iterable[Profile] | None = None,
    audit_every: int | None = None,
    nested_inner_iterations: int = 40,
    nested_restarts: int = 6,
    nested_epsilon: float = 0.05,
) -> TCarmTrace:
    if iterations <= 0:
        raise ValueError("T-CARM requires a positive number of iterations.")
    rng = np.random.default_rng(seed)
    rows = deviation_row_labels(n_agents, policy_ids)
    if not rows:
        raise ValueError("T-CARM requires at least one deviation row.")
    exact_profile_list = list(exact_profiles) if exact_profiles is not None else None
    if oracle_mode == "exact" and exact_profile_list is None:
        exact_profile_list = [tuple(profile) for profile in product(policy_ids, repeat=n_agents)]

    row_regrets = np.zeros(len(rows), dtype=float)
    generated_profiles: list[Profile] = []
    row_distributions: list[np.ndarray] = []
    history: list[dict[str, Any]] = []
    oracle_methods: list[str] = []
    oracle_certified = True

    for iteration in range(1, int(iterations) + 1):
        positive = np.maximum(row_regrets, 0.0)
        if float(np.sum(positive)) > 0.0:
            row_dist = positive / np.sum(positive)
        else:
            row_dist = np.full(len(rows), 1.0 / len(rows), dtype=float)

        if oracle_mode == "exact":
            assert exact_profile_list is not None
            profile, oracle_info = exact_anti_deviation_oracle(row_dist, rows, exact_profile_list, payoff)
        elif oracle_mode in {"nested", "nested_rm"}:
            profile, oracle_info = nested_rm_anti_deviation_oracle(
                row_dist,
                rows,
                policy_ids,
                n_agents,
                payoff,
                rng,
                inner_iterations=nested_inner_iterations,
                restarts=nested_restarts,
                epsilon=nested_epsilon,
            )
        else:
            raise ValueError(f"Unsupported T-CARM oracle_mode: {oracle_mode}")

        generated_profiles.append(profile)
        row_distributions.append(row_dist.copy())
        oracle_methods.append(str(oracle_info.get("method", oracle_mode)))
        oracle_certified = oracle_certified and bool(oracle_info.get("certified", False))
        gains = deviation_gain_vector(rows, profile, payoff)
        baseline = float(np.dot(row_dist, gains))
        row_regrets += gains - baseline

        if audit_every is not None and (iteration % int(audit_every) == 0 or iteration == int(iterations)):
            support_now, probabilities_now = merge_profiles_to_distribution(generated_profiles)
            ub_now, _ = _upper_bound(support_now, probabilities_now, rows, payoff)
            r_bar_now = np.mean(np.vstack(row_distributions), axis=0)
            if oracle_mode == "exact":
                assert exact_profile_list is not None
                _, lb_info = exact_anti_deviation_oracle(r_bar_now, rows, exact_profile_list, payoff)
                lb_now = float(lb_info["objective"])
            else:
                _, lb_info = nested_rm_anti_deviation_oracle(
                    r_bar_now,
                    rows,
                    policy_ids,
                    n_agents,
                    payoff,
                    rng,
                    inner_iterations=nested_inner_iterations,
                    restarts=nested_restarts,
                    epsilon=nested_epsilon,
                )
                lb_now = float(lb_info["objective"])
            history.append(
                {
                    "iteration": int(iteration),
                    "UB": float(ub_now),
                    "LB": float(lb_now),
                    "certificate_gap": float(max(0.0, ub_now - lb_now)) if oracle_mode == "exact" else None,
                    "heuristic_gap_estimate": float(max(0.0, ub_now - lb_now))
                    if oracle_mode != "exact"
                    else None,
                    "oracle_value": baseline,
                }
            )

    support, probabilities = merge_profiles_to_distribution(generated_profiles)
    ub, max_dev = _upper_bound(support, probabilities, rows, payoff)
    r_bar = np.mean(np.vstack(row_distributions), axis=0)
    if oracle_mode == "exact":
        assert exact_profile_list is not None
        lb_profile, lb_info = exact_anti_deviation_oracle(r_bar, rows, exact_profile_list, payoff)
        lower_bound_certified = True
    else:
        lb_profile, lb_info = nested_rm_anti_deviation_oracle(
            r_bar,
            rows,
            policy_ids,
            n_agents,
            payoff,
            rng,
            inner_iterations=nested_inner_iterations,
            restarts=nested_restarts,
            epsilon=nested_epsilon,
        )
        lower_bound_certified = False
    lb = float(lb_info["objective"])
    objective_value = 0.0
    if objective is not None:
        objective_value = float(
            sum(prob * objective(profile) for profile, prob in zip(support, probabilities, strict=True))
        )
    diagnostics = {
        "iterations": int(iterations),
        "oracle_mode": oracle_mode,
        "oracle_methods": sorted(set(oracle_methods)),
        "lower_bound_profile": list(lb_profile),
        "lower_bound_certified": lower_bound_certified,
        "row_regret_max": float(np.max(row_regrets)),
        "row_regret_l1_positive": float(np.sum(np.maximum(row_regrets, 0.0))),
        "history": history,
    }
    return TCarmTrace(
        support_profiles=support,
        support_probabilities=probabilities,
        ub=float(ub),
        lb=float(lb),
        certificate_gap=float(max(0.0, ub - lb)),
        lower_bound_certified=lower_bound_certified and oracle_certified,
        generated_profiles=generated_profiles,
        deviation_rows=rows,
        max_deviation=max_dev,
        objective_value=objective_value,
        diagnostics=diagnostics,
    )


def solve_t_carm_empirical(
    game: EmpiricalGame,
    iterations: int = 500,
    seed: int = 0,
    oracle_mode: str = "exact",
    audit_every: int | None = None,
    tolerance: float = 1.0e-8,
    nested_inner_iterations: int = 40,
    nested_restarts: int = 6,
) -> CceSolution:
    profile_to_index = game.profile_to_index

    def payoff(profile: Profile, agent: int) -> float:
        return float(game.payoffs[profile_to_index[profile], agent])

    def objective(profile: Profile) -> float:
        return float(game.objectives[profile_to_index[profile]])

    trace = solve_t_carm_from_payoff(
        policy_ids=game.policy_ids,
        n_agents=game.n_agents,
        payoff=payoff,
        objective=objective,
        iterations=iterations,
        seed=seed,
        oracle_mode=oracle_mode,
        exact_profiles=game.profiles if oracle_mode == "exact" else None,
        audit_every=audit_every,
        nested_inner_iterations=nested_inner_iterations,
        nested_restarts=nested_restarts,
    )
    q = np.zeros(game.n_profiles, dtype=float)
    for profile, prob in zip(trace.support_profiles, trace.support_probabilities, strict=True):
        q[profile_to_index[profile]] += float(prob)
    q = q / np.sum(q)
    support_profiles, support_probabilities = distribution_support(game, q)
    ucb_dev = max_deviation_ucb(game, q)
    nominal_dev = max_deviation(game, q)
    nominal_dev["gain_ucb"] = ucb_dev["gain_ucb"]
    nominal_dev["ucb_agent"] = ucb_dev["agent"]
    nominal_dev["ucb_policy"] = ucb_dev["policy"]
    raw_cert_gap = max(0.0, float(trace.ub - trace.lb))
    certified_lower_bound = bool(trace.lower_bound_certified)
    diagnostics = dict(trace.diagnostics)
    diagnostics["raw_upper_bound"] = float(trace.ub)
    if certified_lower_bound:
        diagnostics["dual_lower_bound"] = float(trace.lb)
        diagnostics["certificate_gap"] = raw_cert_gap
    else:
        diagnostics["empirical_lower_estimate"] = float(trace.lb)
        diagnostics["heuristic_gap_estimate"] = raw_cert_gap
    return CceSolution(
        solver="T-CARM-CCE",
        q=q,
        objective_value=float(np.dot(game.objectives, q)),
        cce_gap_nominal=compute_cce_gap(game, q),
        cce_gap_ucb=compute_cce_gap_ucb(game, q),
        support_profiles=support_profiles,
        support_probabilities=support_probabilities,
        status="completed",
        full_optimality_certified=bool(certified_lower_bound and raw_cert_gap <= tolerance),
        pricing_mode="exact_anti_deviation" if oracle_mode == "exact" else "nested_rm_anti_deviation",
        pricing_iterations=int(iterations),
        max_deviation=nominal_dev,
        lower_bound=float(trace.lb),
        certificate_gap=raw_cert_gap if certified_lower_bound else None,
        diagnostics=diagnostics,
    )


def solve_t_carm_sparse(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    iterations: int,
    seed: int,
    solver_name: str = "T-CARM-CCE",
    oracle_mode: str = "exact",
    audit_every: int | None = None,
    target_gap: float = 1.0e-8,
    nested_inner_iterations: int = 40,
    nested_restarts: int = 6,
) -> SparseCceResult:
    exact_profiles = None
    if oracle_mode == "exact":
        exact_profiles = list(profile_space(policy_ids, cache.n_agents))
        cache.ensure(exact_profiles)

    def payoff(profile: Profile, agent: int) -> float:
        return cache.payoff(profile, agent)

    trace = solve_t_carm_from_payoff(
        policy_ids=policy_ids,
        n_agents=cache.n_agents,
        payoff=payoff,
        objective=cache.support_objective,
        iterations=iterations,
        seed=seed,
        oracle_mode=oracle_mode,
        exact_profiles=exact_profiles,
        audit_every=audit_every,
        nested_inner_iterations=nested_inner_iterations,
        nested_restarts=nested_restarts,
    )
    audited = audit_sparse_distribution(
        cache,
        trace.support_profiles,
        trace.support_probabilities,
        policy_ids,
        solver_name=solver_name,
        status="completed",
    )
    raw_cert_gap = max(0.0, float(trace.ub - trace.lb))
    certified_lower_bound = bool(trace.lower_bound_certified)
    audited.pricing_mode = "exact_anti_deviation" if oracle_mode == "exact" else "nested_rm_anti_deviation"
    audited.full_optimality_certified = bool(certified_lower_bound and raw_cert_gap <= target_gap)
    audited.support_expansion_rounds = int(iterations)
    audited.certificate_status = (
        "t_carm_exact_certificate"
        if audited.full_optimality_certified
        else ("t_carm_exact_unconverged" if oracle_mode == "exact" else "t_carm_nested_empirical_estimate")
    )
    diagnostics = {
        **trace.diagnostics,
        "raw_upper_bound": float(trace.ub),
        "generated_profile_count": len(trace.generated_profiles),
        "unique_generated_profile_count": len(trace.support_profiles),
    }
    if certified_lower_bound:
        diagnostics["dual_lower_bound"] = float(trace.lb)
        diagnostics["certificate_gap"] = raw_cert_gap
    else:
        diagnostics["empirical_lower_estimate"] = float(trace.lb)
        diagnostics["heuristic_gap_estimate"] = raw_cert_gap
    audited.diagnostics = diagnostics
    return audited
