from __future__ import annotations

import time
from typing import Literal

import numpy as np

from cmfg_cce.evaluation.payoff_cache import PayoffCache, random_profiles, support_deviation_closure
from cmfg_cce.solvers.sparse_cce import SparseCceResult, audit_sparse_distribution, solve_sparse_support


MwuSchedule = Literal["constant", "inverse_sqrt"]


def _canonical_mwu_schedule(schedule: str) -> MwuSchedule:
    normalized = str(schedule).strip().lower().replace(" ", "")
    if normalized == "constant":
        return "constant"
    if normalized in {"inverse_sqrt", "inverse-sqrt", "1/sqrt(t)", "1/sqrt_t"}:
        return "inverse_sqrt"
    raise ValueError(
        "MWU schedule must be 'constant' or 'inverse_sqrt' "
        f"(also accepts '1/sqrt(t)'); got {schedule!r}."
    )


def _validate_mwu_parameters(
    policy_ids: tuple[str, ...],
    eta: float,
    exploration_floor: float,
    burn_in_rounds: int,
) -> tuple[float, float, int]:
    if not policy_ids:
        raise ValueError("MWU requires at least one policy.")
    eta = float(eta)
    exploration_floor = float(exploration_floor)
    burn_in_rounds = int(burn_in_rounds)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError(f"MWU eta must be finite and positive; got {eta!r}.")
    if not np.isfinite(exploration_floor) or not 0.0 <= exploration_floor < 1.0:
        raise ValueError(
            "MWU exploration_floor is the mass mixed with the uniform policy "
            f"distribution and must be in [0, 1); got {exploration_floor!r}."
        )
    if burn_in_rounds < 0:
        raise ValueError(f"MWU burn_in_rounds must be nonnegative; got {burn_in_rounds}.")
    return eta, exploration_floor, burn_in_rounds


def _mwu_learning_rate(eta: float, schedule: MwuSchedule, round_index: int) -> float:
    if round_index < 0:
        raise ValueError("MWU round_index must be nonnegative.")
    if schedule == "constant":
        return float(eta)
    return float(eta) / float(np.sqrt(round_index + 1.0))


def _mwu_sampling_probabilities(weights: np.ndarray, exploration_floor: float) -> np.ndarray:
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise FloatingPointError("MWU weights must have a finite positive sum.")
    probabilities = np.asarray(weights, dtype=float) / weight_sum
    if exploration_floor > 0.0:
        probabilities = (
            (1.0 - exploration_floor) * probabilities
            + exploration_floor / float(len(probabilities))
        )
    return probabilities / float(np.sum(probabilities))


def _mwu_round(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    rng: np.random.Generator,
    weights: list[np.ndarray],
    learning_rate: float,
    exploration_floor: float,
) -> tuple[str, ...]:
    profile = tuple(
        policy_ids[
            int(
                rng.choice(
                    np.arange(len(policy_ids)),
                    p=_mwu_sampling_probabilities(weight, exploration_floor),
                )
            )
        ]
        for weight in weights
    )
    cache.ensure(support_deviation_closure([profile], policy_ids))
    for agent in range(cache.n_agents):
        action_payoffs = []
        for dev_policy in policy_ids:
            dev_profile = list(profile)
            dev_profile[agent] = dev_policy
            action_payoffs.append(cache.payoff(tuple(dev_profile), agent))
        payoff_vector = np.asarray(action_payoffs, dtype=float)
        scale = max(1.0, float(np.max(np.abs(payoff_vector))))
        weights[agent] *= np.exp(float(learning_rate) * payoff_vector / scale)
        weights[agent] = np.clip(weights[agent], 1.0e-12, 1.0e12)
    return profile


def uniform_mixture(cache: PayoffCache, policy_ids: tuple[str, ...], support_size: int, seed: int) -> SparseCceResult:
    rng = np.random.default_rng(seed)
    support = random_profiles(policy_ids, cache.n_agents, support_size, rng)
    prob = 1.0 / len(support)
    return audit_sparse_distribution(
        cache,
        support,
        [prob] * len(support),
        policy_ids,
        "UniformMixture",
        "audited uniform support",
    )


def welfare_best_pure(cache: PayoffCache, policy_ids: tuple[str, ...], candidate_count: int, seed: int) -> SparseCceResult:
    rng = np.random.default_rng(seed)
    candidates = random_profiles(policy_ids, cache.n_agents, candidate_count, rng)
    cache.ensure(candidates)
    best_profile = max(candidates, key=lambda profile: cache.support_objective(profile))
    cache.ensure(support_deviation_closure([best_profile], policy_ids))
    return audit_sparse_distribution(
        cache,
        [best_profile],
        [1.0],
        policy_ids,
        "WelfareBestPure",
        "audited best sampled pure profile",
    )


def random_support_cce(cache: PayoffCache, policy_ids: tuple[str, ...], support_size: int, seed: int) -> SparseCceResult:
    rng = np.random.default_rng(seed)
    support = random_profiles(policy_ids, cache.n_agents, support_size, rng)
    return solve_sparse_support(cache, support, policy_ids, "RandomSupport-CCE")


def vanilla_sampled_cce(cache: PayoffCache, policy_ids: tuple[str, ...], support_size: int, seed: int) -> SparseCceResult:
    result = random_support_cce(cache, policy_ids, support_size, seed)
    result.solver = "VanillaSampled-CCE"
    return result


def mwu_policy_trace(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    rounds: int,
    seed: int,
    *,
    eta: float = 0.15,
    schedule: str = "constant",
    exploration_floor: float = 0.0,
    burn_in_rounds: int = 0,
) -> SparseCceResult:
    eta, exploration_floor, burn_in_rounds = _validate_mwu_parameters(
        policy_ids,
        eta,
        exploration_floor,
        burn_in_rounds,
    )
    schedule_name = _canonical_mwu_schedule(schedule)
    rng = np.random.default_rng(seed)
    rounds = max(1, int(rounds))
    if burn_in_rounds >= rounds:
        raise ValueError(
            "MWU burn_in_rounds must be smaller than rounds so that the "
            "reported empirical trace is nonempty."
        )
    weights = [np.ones(len(policy_ids), dtype=float) for _ in range(cache.n_agents)]
    support_counts: dict[tuple[str, ...], int] = {}
    for round_index in range(rounds):
        profile = _mwu_round(
            cache,
            policy_ids,
            rng,
            weights,
            _mwu_learning_rate(eta, schedule_name, round_index),
            exploration_floor,
        )
        if round_index >= burn_in_rounds:
            support_counts[profile] = support_counts.get(profile, 0) + 1
    support = list(support_counts.keys())
    total = sum(support_counts.values())
    probs = [support_counts[profile] / total for profile in support]
    result = audit_sparse_distribution(
        cache,
        support,
        probs,
        policy_ids,
        "MWU-PolicyTrace",
        "audited MWU empirical trace",
    )
    result.pricing_mode = "multiplicative_weights_trace"
    result.diagnostics = {
        "no_regret_algorithm": "multiplicative_weights",
        "rounds": int(rounds),
        "trace_rounds": int(rounds - burn_in_rounds),
        "eta": eta,
        "learning_rate_schedule": schedule_name,
        "exploration_floor": exploration_floor,
        "exploration_interpretation": "total_uniform_mixture_mass",
        "burn_in_rounds": burn_in_rounds,
    }
    return result


def _trace_only_result(
    cache: PayoffCache,
    support_counts: dict[tuple[str, ...], int],
    solver_name: str,
    pricing_mode: str,
    status: str,
) -> SparseCceResult:
    support = list(support_counts.keys())
    if not support:
        raise RuntimeError(f"{solver_name} did not generate any policy profiles.")
    total = sum(support_counts.values())
    probs = [support_counts[profile] / total for profile in support]
    objective = float(
        sum(prob * cache.support_objective(profile) for profile, prob in zip(support, probs, strict=True))
    )
    result = SparseCceResult(
        solver=solver_name,
        support_profiles=support,
        support_probabilities=[float(prob) for prob in probs],
        objective_value=objective,
        cce_gap_nominal=0.0,
        cce_gap_ucb=0.0,
        max_deviation={},
        status=status,
        certificate_status="time_budgeted_trace_full_tensor_verification_required",
        pricing_mode=pricing_mode,
        full_optimality_certified=False,
        audit_mode="trace_only_full_tensor_verification",
    )
    result.diagnostics["support_local_gap_computed"] = False
    return result


def mwu_policy_trace_time_budget(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    budget_seconds: float,
    seed: int,
    min_rounds: int = 1,
    *,
    eta: float = 0.15,
    schedule: str = "constant",
    exploration_floor: float = 0.0,
    burn_in_rounds: int = 0,
) -> SparseCceResult:
    eta, exploration_floor, burn_in_rounds = _validate_mwu_parameters(
        policy_ids,
        eta,
        exploration_floor,
        burn_in_rounds,
    )
    schedule_name = _canonical_mwu_schedule(schedule)
    rng = np.random.default_rng(seed)
    budget_seconds = max(0.0, float(budget_seconds))
    min_rounds = max(1, int(min_rounds))
    minimum_total_rounds = burn_in_rounds + min_rounds
    weights = [np.ones(len(policy_ids), dtype=float) for _ in range(cache.n_agents)]
    support_counts: dict[tuple[str, ...], int] = {}
    start = time.perf_counter()
    rounds = 0
    while rounds < minimum_total_rounds or time.perf_counter() - start < budget_seconds:
        profile = _mwu_round(
            cache,
            policy_ids,
            rng,
            weights,
            _mwu_learning_rate(eta, schedule_name, rounds),
            exploration_floor,
        )
        if rounds >= burn_in_rounds:
            support_counts[profile] = support_counts.get(profile, 0) + 1
        rounds += 1
    elapsed = time.perf_counter() - start
    result = _trace_only_result(
        cache,
        support_counts,
        "MWU-PolicyTrace",
        "multiplicative_weights_time_budget_trace",
        "time-budgeted MWU empirical trace; full tensor verification required",
    )
    result.diagnostics.update(
        {
            "no_regret_algorithm": "multiplicative_weights",
            "rounds": int(rounds),
            "trace_rounds": int(rounds - burn_in_rounds),
            "eta": eta,
            "learning_rate_schedule": schedule_name,
            "exploration_floor": exploration_floor,
            "exploration_interpretation": "total_uniform_mixture_mass",
            "burn_in_rounds": burn_in_rounds,
            "time_budget_seconds": budget_seconds,
            "trace_runtime_seconds": elapsed,
            "budget_stop_rule": "checked_after_each_round",
        }
    )
    return result


def regret_matching_policy_trace(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    rounds: int,
    seed: int,
) -> SparseCceResult:
    rng = np.random.default_rng(seed)
    rounds = max(1, int(rounds))
    k = len(policy_ids)
    action_indices = [int(rng.integers(0, k)) for _ in range(cache.n_agents)]
    regrets = [np.zeros((k, k), dtype=float) for _ in range(cache.n_agents)]
    support_counts: dict[tuple[str, ...], int] = {}
    transition_scale = 2.0 * max(1, k) * rounds
    for _ in range(rounds):
        profile = tuple(policy_ids[action_idx] for action_idx in action_indices)
        support_counts[profile] = support_counts.get(profile, 0) + 1
        cache.ensure(support_deviation_closure([profile], policy_ids))
        next_actions: list[int] = []
        for agent in range(cache.n_agents):
            current = action_indices[agent]
            action_payoffs = []
            for alternative in range(k):
                dev_profile = list(profile)
                dev_profile[agent] = policy_ids[alternative]
                action_payoffs.append(cache.payoff(tuple(dev_profile), agent))
            payoff_vector = np.array(action_payoffs, dtype=float)
            scale = max(1.0, float(np.max(np.abs(payoff_vector))))
            normalized_payoffs = payoff_vector / scale
            current_payoff = float(normalized_payoffs[current])
            for alternative in range(k):
                if alternative == current:
                    continue
                regrets[agent][current, alternative] += float(normalized_payoffs[alternative]) - current_payoff
            positive = np.maximum(regrets[agent][current], 0.0)
            positive[current] = 0.0
            probabilities = np.zeros(k, dtype=float)
            probabilities += positive / transition_scale
            probabilities[current] = max(0.0, 1.0 - float(np.sum(probabilities)))
            probabilities = probabilities / np.sum(probabilities)
            next_actions.append(int(rng.choice(np.arange(k), p=probabilities)))
        action_indices = next_actions
    support = list(support_counts.keys())
    total = sum(support_counts.values())
    probs = [support_counts[profile] / total for profile in support]
    result = audit_sparse_distribution(
        cache,
        support,
        probs,
        policy_ids,
        "RegretMatching-PolicyTrace",
        "audited regret-matching empirical trace",
    )
    result.pricing_mode = "regret_matching_trace"
    result.diagnostics = {
        "no_regret_algorithm": "regret_matching",
        "rounds": int(rounds),
        "regret_type": "internal_pairwise_policy_regret",
        "transition_scale": transition_scale,
    }
    return result


def regret_matching_policy_trace_time_budget(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    budget_seconds: float,
    seed: int,
    min_rounds: int = 1,
) -> SparseCceResult:
    rng = np.random.default_rng(seed)
    budget_seconds = max(0.0, float(budget_seconds))
    min_rounds = max(1, int(min_rounds))
    k = len(policy_ids)
    action_indices = [int(rng.integers(0, k)) for _ in range(cache.n_agents)]
    regrets = [np.zeros((k, k), dtype=float) for _ in range(cache.n_agents)]
    support_counts: dict[tuple[str, ...], int] = {}
    start = time.perf_counter()
    rounds = 0
    while rounds < min_rounds or time.perf_counter() - start < budget_seconds:
        profile = tuple(policy_ids[action_idx] for action_idx in action_indices)
        support_counts[profile] = support_counts.get(profile, 0) + 1
        cache.ensure(support_deviation_closure([profile], policy_ids))
        next_actions: list[int] = []
        for agent in range(cache.n_agents):
            current = action_indices[agent]
            action_payoffs = []
            for alternative in range(k):
                dev_profile = list(profile)
                dev_profile[agent] = policy_ids[alternative]
                action_payoffs.append(cache.payoff(tuple(dev_profile), agent))
            payoff_vector = np.array(action_payoffs, dtype=float)
            scale = max(1.0, float(np.max(np.abs(payoff_vector))))
            normalized_payoffs = payoff_vector / scale
            current_payoff = float(normalized_payoffs[current])
            for alternative in range(k):
                if alternative == current:
                    continue
                regrets[agent][current, alternative] += float(normalized_payoffs[alternative]) - current_payoff
            positive = np.maximum(regrets[agent][current], 0.0)
            positive[current] = 0.0
            probabilities = np.zeros(k, dtype=float)
            total_positive = float(np.sum(positive))
            if total_positive > 0.0:
                probabilities = positive / total_positive
            else:
                probabilities[current] = 1.0
            next_actions.append(int(rng.choice(np.arange(k), p=probabilities)))
        action_indices = next_actions
        rounds += 1
    elapsed = time.perf_counter() - start
    result = _trace_only_result(
        cache,
        support_counts,
        "RegretMatching-PolicyTrace",
        "regret_matching_time_budget_trace",
        "time-budgeted regret-matching empirical trace; full tensor verification required",
    )
    result.diagnostics.update(
        {
            "no_regret_algorithm": "regret_matching",
            "rounds": int(rounds),
            "regret_type": "internal_pairwise_policy_regret",
            "time_budget_seconds": budget_seconds,
            "trace_runtime_seconds": elapsed,
            "budget_stop_rule": "checked_after_each_round",
        }
    )
    return result
