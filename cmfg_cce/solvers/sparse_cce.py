from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from cmfg_cce.evaluation.payoff_cache import PayoffCache, random_profiles, support_deviation_closure
from cmfg_cce.evaluation.rollout import Profile
from cmfg_cce.solvers.cce_lp import (
    DEFAULT_LEXICOGRAPHIC_EPSILON_TOLERANCE,
    DEFAULT_SELECTOR,
    LexicographicSelection,
    canonical_selector,
    solve_lexicographic_distribution,
)


@dataclass
class SparseCceResult:
    solver: str
    support_profiles: list[Profile]
    support_probabilities: list[float]
    objective_value: float
    cce_gap_nominal: float
    cce_gap_ucb: float
    max_deviation: dict
    status: str
    cce_gap_ci: float = 0.0
    certificate_status: str = "audited_support_deviation_closure"
    pricing_mode: str | None = None
    full_optimality_certified: bool = False
    support_expansion_rounds: int = 0
    active_sampling_rounds: int = 0
    policy_expansion_rounds: int = 0
    repair_rounds: int = 0
    repair_profiles_added: int = 0
    paired_delta_pair_count: int = 0
    paired_delta_sample_count: int = 0
    paired_estimator_mode: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    audit_mode: str = "full_deviation_closure"

    @property
    def support_size(self) -> int:
        return len(self.support_profiles)


def deviation_matrix(
    cache: PayoffCache,
    support: list[Profile],
    policy_ids: tuple[str, ...],
    ucb: bool = False,
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    rows: list[list[float]] = []
    labels: list[tuple[int, str]] = []
    for agent in range(cache.n_agents):
        for dev_policy in policy_ids:
            row: list[float] = []
            for profile in support:
                dev_profile = list(profile)
                dev_profile[agent] = dev_policy
                dev_profile_t = tuple(dev_profile)
                if ucb:
                    gain = cache.payoff_ucb(dev_profile_t, agent) - cache.payoff_lcb(profile, agent)
                else:
                    gain = cache.payoff(dev_profile_t, agent) - cache.payoff(profile, agent)
                row.append(gain)
            rows.append(row)
            labels.append((agent, dev_policy))
    return np.array(rows, dtype=float), labels


def selector_values_from_cache(
    cache: PayoffCache,
    support: list[Profile],
    selector: str,
) -> np.ndarray:
    """Build the common stage-2 objective on a sparse profile set."""

    canonical = canonical_selector(selector)
    if canonical == "platform_operating_score":
        values = [cache.support_objective(profile) for profile in support]
    elif canonical == "total_manufacturer_return":
        values = [
            sum(cache.payoff(profile, agent) for agent in range(cache.n_agents))
            for profile in support
        ]
    else:
        values = [0.0] * len(support)
    result = np.asarray(values, dtype=float)
    if result.shape != (len(support),) or not np.all(np.isfinite(result)):
        raise ValueError(f"Selector {canonical!r} produced invalid sparse objective values.")
    return result


def _distribution_entropy(probs: np.ndarray) -> float:
    positive = probs[probs > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def _sparse_selection_diagnostics(
    cache: PayoffCache,
    support: list[Profile],
    selection: LexicographicSelection,
) -> dict[str, Any]:
    probs = selection.q
    platform_values = selector_values_from_cache(cache, support, "platform_operating_score")
    manufacturer_values = selector_values_from_cache(cache, support, "total_manufacturer_return")
    return {
        "selector": selection.selector,
        "stage1_epsilon": selection.epsilon_star,
        "stage2_epsilon_bound": selection.epsilon_bound,
        "epsilon_tolerance": selection.epsilon_bound - selection.epsilon_star,
        "secondary_objective_value": selection.objective_value,
        "platform_operating_score": float(np.dot(platform_values, probs)),
        "total_manufacturer_return": float(np.dot(manufacturer_values, probs)),
        "distribution_entropy": _distribution_entropy(probs),
        "stage1_status": selection.stage1_status,
        "stage2_status": selection.stage2_status,
    }


def solve_sparse_support(
    cache: PayoffCache,
    support: list[Profile],
    policy_ids: tuple[str, ...],
    solver_name: str,
    selector: str = DEFAULT_SELECTOR,
    epsilon_tolerance: float = DEFAULT_LEXICOGRAPHIC_EPSILON_TOLERANCE,
) -> SparseCceResult:
    closure = support_deviation_closure(support, policy_ids)
    cache.ensure(closure)
    a_nominal, labels = deviation_matrix(cache, support, policy_ids, ucb=False)
    a_ucb, _ = deviation_matrix(cache, support, policy_ids, ucb=True)
    selection = solve_lexicographic_distribution(
        a_nominal,
        selector_values_from_cache(cache, support, selector),
        selector=selector,
        epsilon_tolerance=epsilon_tolerance,
    )
    probs = selection.q
    nominal_gains = a_nominal @ probs
    ucb_gains = a_ucb @ probs
    max_idx = int(np.argmax(nominal_gains))
    max_ucb_idx = int(np.argmax(ucb_gains))
    gap_ci = max(0.0, float(ucb_gains[max_ucb_idx] - nominal_gains[max_ucb_idx]))
    objective = selection.objective_value
    filtered_support: list[Profile] = []
    filtered_probs: list[float] = []
    for profile, prob in zip(support, probs, strict=True):
        if prob > 0.0:
            filtered_support.append(profile)
            filtered_probs.append(float(prob))
    agent, policy = labels[max_idx]
    ucb_agent, ucb_policy = labels[max_ucb_idx]
    return SparseCceResult(
        solver=solver_name,
        support_profiles=filtered_support,
        support_probabilities=filtered_probs,
        objective_value=objective,
        cce_gap_nominal=max(0.0, float(nominal_gains[max_idx])),
        cce_gap_ucb=max(0.0, float(ucb_gains[max_ucb_idx])),
        cce_gap_ci=gap_ci,
        max_deviation={
            "agent": int(agent),
            "policy": policy,
            "gain_nominal": max(0.0, float(nominal_gains[max_idx])),
            "ucb_agent": int(ucb_agent),
            "ucb_policy": ucb_policy,
            "gain_ucb": max(0.0, float(ucb_gains[max_ucb_idx])),
            "gap_ci_at_ucb": gap_ci,
        },
        status=f"stage 1: {selection.stage1_status}; stage 2: {selection.stage2_status}",
        diagnostics=_sparse_selection_diagnostics(cache, support, selection),
    )


def audit_sparse_distribution(
    cache: PayoffCache,
    support: list[Profile],
    probabilities: list[float],
    policy_ids: tuple[str, ...],
    solver_name: str,
    status: str,
) -> SparseCceResult:
    closure = support_deviation_closure(support, policy_ids)
    cache.ensure(closure)
    probs = np.array(probabilities, dtype=float)
    probs = probs / np.sum(probs)
    a_nominal, labels = deviation_matrix(cache, support, policy_ids, ucb=False)
    a_ucb, _ = deviation_matrix(cache, support, policy_ids, ucb=True)
    nominal_gains = a_nominal @ probs
    ucb_gains = a_ucb @ probs
    max_idx = int(np.argmax(nominal_gains))
    max_ucb_idx = int(np.argmax(ucb_gains))
    gap_ci = max(0.0, float(ucb_gains[max_ucb_idx] - nominal_gains[max_ucb_idx]))
    objective = float(sum(prob * cache.support_objective(profile) for profile, prob in zip(support, probs, strict=True)))
    agent, policy = labels[max_idx]
    ucb_agent, ucb_policy = labels[max_ucb_idx]
    return SparseCceResult(
        solver=solver_name,
        support_profiles=support,
        support_probabilities=[float(prob) for prob in probs],
        objective_value=objective,
        cce_gap_nominal=max(0.0, float(nominal_gains[max_idx])),
        cce_gap_ucb=max(0.0, float(ucb_gains[max_ucb_idx])),
        cce_gap_ci=gap_ci,
        max_deviation={
            "agent": int(agent),
            "policy": policy,
            "gain_nominal": max(0.0, float(nominal_gains[max_idx])),
            "ucb_agent": int(ucb_agent),
            "ucb_policy": ucb_policy,
            "gain_ucb": max(0.0, float(ucb_gains[max_ucb_idx])),
            "gap_ci_at_ucb": gap_ci,
        },
        status=status,
    )


def paired_deviation_arrays(
    cache: PayoffCache,
    support: list[Profile],
    policy_ids: tuple[str, ...],
    target_samples: int,
    solver_name: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, str]]]:
    means: list[list[float]] = []
    variances: list[list[float]] = []
    counts: list[list[int]] = []
    labels: list[tuple[int, str]] = []
    for agent in range(cache.n_agents):
        for dev_policy in policy_ids:
            mean_row: list[float] = []
            var_row: list[float] = []
            count_row: list[int] = []
            for profile in support:
                if profile[agent] == dev_policy:
                    mean_row.append(0.0)
                    var_row.append(0.0)
                    count_row.append(max(1, int(target_samples)))
                    continue
                samples = cache.ensure_paired_delta_samples(
                    profile,
                    agent,
                    dev_policy,
                    target_samples,
                    solver_name=solver_name,
                    metadata=metadata,
                )
                arr = np.array(samples, dtype=float)
                mean_row.append(float(np.mean(arr)) if arr.size else 0.0)
                var_row.append(float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0)
                count_row.append(max(1, int(arr.size)))
            means.append(mean_row)
            variances.append(var_row)
            counts.append(count_row)
            labels.append((agent, dev_policy))
    return (
        np.array(means, dtype=float),
        np.array(variances, dtype=float),
        np.array(counts, dtype=float),
        labels,
    )


def solve_delta_support_lp(
    cache: PayoffCache,
    support: list[Profile],
    policy_ids: tuple[str, ...],
    delta_mean: np.ndarray,
    delta_var: np.ndarray,
    delta_n: np.ndarray,
    labels: list[tuple[int, str]],
    solver_name: str,
    beta: float,
    pricing_mode: str,
    audit_mode: str,
    certificate_status: str,
    selector: str = DEFAULT_SELECTOR,
    epsilon_tolerance: float = DEFAULT_LEXICOGRAPHIC_EPSILON_TOLERANCE,
) -> tuple[SparseCceResult, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selection = solve_lexicographic_distribution(
        delta_mean,
        selector_values_from_cache(cache, support, selector),
        selector=selector,
        epsilon_tolerance=epsilon_tolerance,
    )
    probs = selection.q
    nominal_gains = delta_mean @ probs
    se_gains = np.sqrt(np.sum((probs[None, :] ** 2) * delta_var / np.maximum(delta_n, 1.0), axis=1))
    ci_gains = float(beta) * se_gains
    ucb_gains = nominal_gains + ci_gains
    max_idx = int(np.argmax(nominal_gains))
    max_ucb_idx = int(np.argmax(ucb_gains))
    max_ci_idx = int(np.argmax(ci_gains))
    objective = selection.objective_value
    filtered_support: list[Profile] = []
    filtered_probs: list[float] = []
    for profile, prob in zip(support, probs, strict=True):
        if prob > 0.0:
            filtered_support.append(profile)
            filtered_probs.append(float(prob))
    agent, policy = labels[max_idx]
    ucb_agent, ucb_policy = labels[max_ucb_idx]
    ci_agent, ci_policy = labels[max_ci_idx]
    sparse_result = SparseCceResult(
        solver=solver_name,
        support_profiles=filtered_support,
        support_probabilities=filtered_probs,
        objective_value=objective,
        cce_gap_nominal=max(0.0, float(nominal_gains[max_idx])),
        cce_gap_ucb=max(0.0, float(ucb_gains[max_ucb_idx])),
        cce_gap_ci=max(0.0, float(ci_gains[max_ucb_idx])),
        max_deviation={
            "agent": int(agent),
            "policy": policy,
            "gain_nominal": max(0.0, float(nominal_gains[max_idx])),
            "ucb_agent": int(ucb_agent),
            "ucb_policy": ucb_policy,
            "gain_ucb": max(0.0, float(ucb_gains[max_ucb_idx])),
            "ci_agent": int(ci_agent),
            "ci_policy": ci_policy,
            "gain_ci": max(0.0, float(ci_gains[max_ci_idx])),
            "gap_ci_at_ucb": max(0.0, float(ci_gains[max_ucb_idx])),
        },
        status=f"stage 1: {selection.stage1_status}; stage 2: {selection.stage2_status}",
        certificate_status=certificate_status,
        pricing_mode=pricing_mode,
        paired_estimator_mode="paired_crn",
        audit_mode=audit_mode,
    )
    sparse_result.diagnostics = {
        **_sparse_selection_diagnostics(cache, support, selection),
        "worst_mean_constraint": labels[max_idx],
        "worst_ucb_constraint": labels[max_ucb_idx],
        "worst_ci_constraint": labels[max_ci_idx],
        "gap_mean_at_ucb": float(nominal_gains[max_ucb_idx]),
        "beta": float(beta),
    }
    return sparse_result, probs, nominal_gains, ci_gains, ucb_gains


def audit_delta_distribution(
    cache: PayoffCache,
    support: list[Profile],
    probabilities: list[float],
    delta_mean: np.ndarray,
    delta_var: np.ndarray,
    delta_n: np.ndarray,
    labels: list[tuple[int, str]],
    solver_name: str,
    beta: float,
    pricing_mode: str,
    audit_mode: str,
    certificate_status: str,
) -> SparseCceResult:
    probs = np.array(probabilities, dtype=float)
    probs = probs / np.sum(probs)
    nominal_gains = delta_mean @ probs
    se_gains = np.sqrt(np.sum((probs[None, :] ** 2) * delta_var / np.maximum(delta_n, 1.0), axis=1))
    ci_gains = float(beta) * se_gains
    ucb_gains = nominal_gains + ci_gains
    max_idx = int(np.argmax(nominal_gains))
    max_ucb_idx = int(np.argmax(ucb_gains))
    max_ci_idx = int(np.argmax(ci_gains))
    objective = float(sum(prob * cache.support_objective(profile) for profile, prob in zip(support, probs, strict=True)))
    agent, policy = labels[max_idx]
    ucb_agent, ucb_policy = labels[max_ucb_idx]
    ci_agent, ci_policy = labels[max_ci_idx]
    sparse_result = SparseCceResult(
        solver=solver_name,
        support_profiles=list(support),
        support_probabilities=[float(prob) for prob in probs],
        objective_value=objective,
        cce_gap_nominal=max(0.0, float(nominal_gains[max_idx])),
        cce_gap_ucb=max(0.0, float(ucb_gains[max_ucb_idx])),
        cce_gap_ci=max(0.0, float(ci_gains[max_ucb_idx])),
        max_deviation={
            "agent": int(agent),
            "policy": policy,
            "gain_nominal": max(0.0, float(nominal_gains[max_idx])),
            "ucb_agent": int(ucb_agent),
            "ucb_policy": ucb_policy,
            "gain_ucb": max(0.0, float(ucb_gains[max_ucb_idx])),
            "ci_agent": int(ci_agent),
            "ci_policy": ci_policy,
            "gain_ci": max(0.0, float(ci_gains[max_ci_idx])),
            "gap_ci_at_ucb": max(0.0, float(ci_gains[max_ucb_idx])),
        },
        status="audited frozen distribution",
        certificate_status=certificate_status,
        pricing_mode=pricing_mode,
        paired_estimator_mode="paired_crn",
        audit_mode=audit_mode,
    )
    sparse_result.diagnostics = {
        "worst_mean_constraint": labels[max_idx],
        "worst_ucb_constraint": labels[max_ucb_idx],
        "worst_ci_constraint": labels[max_ci_idx],
        "gap_mean_at_ucb": float(nominal_gains[max_ucb_idx]),
        "beta": float(beta),
    }
    return sparse_result


def expand_by_deviation(
    support: list[Profile],
    result: SparseCceResult,
    policy_ids: tuple[str, ...],
    batch_size: int,
    rng: np.random.Generator,
) -> list[Profile]:
    expanded = list(dict.fromkeys(support))
    agent = int(result.max_deviation["agent"])
    policy = str(result.max_deviation["policy"])
    ranked = sorted(
        zip(result.support_profiles, result.support_probabilities, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    for profile, _ in ranked:
        dev_profile = list(profile)
        dev_profile[agent] = policy
        dev_profile_t = tuple(dev_profile)
        if dev_profile_t not in expanded:
            expanded.append(dev_profile_t)
        if len(expanded) - len(support) >= batch_size:
            break
    while len(expanded) - len(support) < batch_size:
        candidate = tuple(str(policy_ids[int(rng.integers(0, len(policy_ids)))]) for _ in range(len(support[0])))
        if candidate not in expanded:
            expanded.append(candidate)
    return expanded


def active_resample(
    cache: PayoffCache,
    result: SparseCceResult,
    policy_ids: tuple[str, ...],
    target_rollouts: int,
    max_profiles: int,
) -> int:
    profiles: list[Profile] = []
    agent = int(result.max_deviation["ucb_agent"])
    policy = str(result.max_deviation["ucb_policy"])
    ranked = sorted(
        zip(result.support_profiles, result.support_probabilities, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    for profile, _ in ranked:
        profiles.append(profile)
        dev_profile = list(profile)
        dev_profile[agent] = policy
        profiles.append(tuple(dev_profile))
        if len(profiles) >= max_profiles:
            break
    # Include a small deterministic sweep of other deviations to avoid
    # over-focusing on one noisy active row.
    for profile, _ in ranked:
        for other_agent in range(cache.n_agents):
            for dev_policy in policy_ids[: min(3, len(policy_ids))]:
                dev_profile = list(profile)
                dev_profile[other_agent] = dev_policy
                profiles.append(tuple(dev_profile))
                if len(profiles) >= max_profiles:
                    break
            if len(profiles) >= max_profiles:
                break
        if len(profiles) >= max_profiles:
            break
    before = {profile: cache.get(profile).n_rollouts for profile in set(profiles)}
    cache.resample_many(set(profiles), target_rollouts)
    return sum(1 for profile, n_before in before.items() if cache.get(profile).n_rollouts > n_before)


def solve_sad_sparse(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    initial_support_size: int,
    max_support_size: int,
    support_add_batch_size: int,
    max_rounds: int,
    target_gap: float,
    seed: int,
    rollouts_max: int | None = None,
    active_sampling: bool = True,
    selector: str = DEFAULT_SELECTOR,
    epsilon_tolerance: float = DEFAULT_LEXICOGRAPHIC_EPSILON_TOLERANCE,
) -> SparseCceResult:
    rng = np.random.default_rng(seed)
    support = random_profiles(policy_ids, cache.n_agents, initial_support_size, rng)
    best: SparseCceResult | None = None
    active_rounds = 0
    for round_idx in range(max_rounds):
        result = solve_sparse_support(
            cache,
            support,
            policy_ids,
            "SAD-CCE",
            selector=selector,
            epsilon_tolerance=epsilon_tolerance,
        )
        result.support_expansion_rounds = round_idx
        result.active_sampling_rounds = active_rounds
        if active_sampling and rollouts_max and result.cce_gap_ucb > target_gap:
            changed = active_resample(
                cache,
                result,
                policy_ids,
                target_rollouts=rollouts_max,
                max_profiles=max(4, support_add_batch_size * 2),
            )
            if changed:
                active_rounds += 1
                result = solve_sparse_support(
                    cache,
                    support,
                    policy_ids,
                    "SAD-CCE",
                    selector=selector,
                    epsilon_tolerance=epsilon_tolerance,
                )
                result.support_expansion_rounds = round_idx
                result.active_sampling_rounds = active_rounds
        best = result
        if result.cce_gap_ucb <= target_gap or len(support) >= max_support_size:
            break
        support = expand_by_deviation(support, result, policy_ids, support_add_batch_size, rng)
        support = support[:max_support_size]
    if best is None:
        raise RuntimeError("SAD-CCE sparse solver did not produce a result.")
    return best


def solve_pair_sparse(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    initial_support_size: int,
    max_support_size: int,
    support_add_batch_size: int,
    max_rounds: int,
    target_gap: float,
    seed: int,
    rollouts_max: int | None = None,
    active_sampling: bool = True,
    beta: float = 1.96,
    n_min_pair_samples: int = 4,
    pair_active_rounds: int = 2,
    top_constraints: int = 5,
    top_pairs_per_constraint: int = 10,
    batch_samples_per_pair: int = 2,
    q_min: float = 1.0e-4,
    metadata: dict[str, Any] | None = None,
    initial_result: SparseCceResult | None = None,
    solver_name: str = "PAIR-SAD-CCE",
    selector: str = DEFAULT_SELECTOR,
    epsilon_tolerance: float = DEFAULT_LEXICOGRAPHIC_EPSILON_TOLERANCE,
) -> SparseCceResult:
    if initial_result is None:
        initial_result = solve_sad_sparse(
            cache,
            policy_ids,
            initial_support_size=initial_support_size,
            max_support_size=max_support_size,
            support_add_batch_size=support_add_batch_size,
            max_rounds=max_rounds,
            target_gap=target_gap,
            seed=seed,
            rollouts_max=rollouts_max,
            active_sampling=active_sampling,
            selector=selector,
            epsilon_tolerance=epsilon_tolerance,
        )
    support = list(initial_result.support_profiles)
    if not support:
        raise RuntimeError("PAIR-SAD-CCE requires a nonempty initial support.")
    best: SparseCceResult | None = None
    active_rounds = 0
    for round_idx in range(max(1, int(pair_active_rounds) + 1)):
        delta_mean, delta_var, delta_n, labels = paired_deviation_arrays(
            cache,
            support,
            policy_ids,
            max(1, int(n_min_pair_samples)),
            solver_name=solver_name,
            metadata=metadata,
        )
        used_pair_count = 0
        used_sample_count = 0
        for row_idx, (agent, dev_policy) in enumerate(labels):
            for col_idx, profile in enumerate(support):
                if profile[agent] == dev_policy:
                    continue
                used_pair_count += 1
                used_sample_count += int(delta_n[row_idx, col_idx])
        result, probs, nominal_gains, ci_gains, ucb_gains = solve_delta_support_lp(
            cache,
            support,
            policy_ids,
            delta_mean,
            delta_var,
            delta_n,
            labels,
            solver_name=solver_name,
            beta=beta,
            pricing_mode="paired_active_resampling",
            audit_mode="paired_support_deviation_closure",
            certificate_status="paired_crn_audit",
            selector=selector,
            epsilon_tolerance=epsilon_tolerance,
        )
        result.support_expansion_rounds = initial_result.support_expansion_rounds
        result.active_sampling_rounds = initial_result.active_sampling_rounds + active_rounds
        result.policy_expansion_rounds = initial_result.policy_expansion_rounds
        result.repair_profiles_added = initial_result.repair_profiles_added
        result.repair_rounds = initial_result.repair_rounds
        result.paired_delta_pair_count = int(used_pair_count)
        result.paired_delta_sample_count = int(used_sample_count)
        best = result
        if result.cce_gap_ucb <= target_gap or round_idx >= int(pair_active_rounds):
            break
        constraint_order = np.argsort(-(np.maximum(0.0, ucb_gains - target_gap) * np.maximum(ci_gains, 1e-9)))
        selected_pairs = 0
        for row_idx in constraint_order[: max(1, int(top_constraints))]:
            agent, dev_policy = labels[int(row_idx)]
            pair_scores = []
            for col_idx, profile in enumerate(support):
                prob = float(probs[col_idx])
                if prob < q_min or profile[agent] == dev_policy:
                    continue
                uncertainty = (prob**2) * float(delta_var[row_idx, col_idx]) / max(1.0, float(delta_n[row_idx, col_idx]))
                positive_mean = 0.1 * prob * max(0.0, float(delta_mean[row_idx, col_idx]))
                pair_scores.append((uncertainty + positive_mean, col_idx, profile))
            pair_scores.sort(reverse=True, key=lambda item: item[0])
            for _, col_idx, profile in pair_scores[: max(1, int(top_pairs_per_constraint))]:
                current_n = int(delta_n[row_idx, col_idx])
                cache.ensure_paired_delta_samples(
                    profile,
                    agent,
                    dev_policy,
                    current_n + max(1, int(batch_samples_per_pair)),
                    solver_name=solver_name,
                    metadata=metadata,
                )
                selected_pairs += 1
        if selected_pairs == 0:
            break
        active_rounds += 1
        support = list(result.support_profiles)
        if not support:
            break
    if best is None:
        raise RuntimeError("PAIR-SAD-CCE sparse solver did not produce a result.")
    return best


def _targeted_policy_ids(mechanism_id: str, policy_ids: tuple[str, ...], targeted_deviation_policies: dict[str, list[str]] | None) -> set[str]:
    if not targeted_deviation_policies:
        return set()
    aliases = {
        "M1_price_first": "M1",
        "M2_price_critical": "M2",
        "M3_delivery_first": "M3",
        "M4_delivery_critical": "M4",
    }
    candidates = set(targeted_deviation_policies.get(mechanism_id, []))
    short = aliases.get(mechanism_id)
    if short:
        candidates.update(targeted_deviation_policies.get(short, []))
    return {policy for policy in candidates if policy in policy_ids}


def solve_repair_sparse(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    initial_support_size: int,
    max_support_size: int,
    support_add_batch_size: int,
    max_rounds: int,
    target_gap: float,
    seed: int,
    rollouts_max: int | None = None,
    active_sampling: bool = True,
    repair_rounds: int = 2,
    top_constraints: int = 3,
    top_profiles_per_constraint: int = 10,
    q_min: float = 1.0e-4,
    mean_threshold: float = 1.0,
    contribution_min: float = 0.0,
    enable_multi_agent_repair: bool = True,
    max_agents_repaired_per_profile: int = 2,
    beam_width: int = 10,
    profile_budget_multiplier: float = 2.0,
    targeted_deviation_policies: dict[str, list[str]] | None = None,
    solver_name: str = "REPAIR-SAD-CCE",
    selector: str = DEFAULT_SELECTOR,
    epsilon_tolerance: float = DEFAULT_LEXICOGRAPHIC_EPSILON_TOLERANCE,
) -> SparseCceResult:
    base = solve_sad_sparse(
        cache,
        policy_ids,
        initial_support_size=initial_support_size,
        max_support_size=max_support_size,
        support_add_batch_size=support_add_batch_size,
        max_rounds=max_rounds,
        target_gap=target_gap,
        seed=seed,
        rollouts_max=rollouts_max,
        active_sampling=active_sampling,
        selector=selector,
        epsilon_tolerance=epsilon_tolerance,
    )
    support = list(dict.fromkeys(base.support_profiles))
    budget = max(len(support), int(max_support_size * max(1.0, float(profile_budget_multiplier))))
    repair_added = 0
    completed_rounds = 0
    targeted = _targeted_policy_ids(cache.mechanism_id, policy_ids, targeted_deviation_policies)
    for round_idx in range(max(0, int(repair_rounds))):
        current = solve_sparse_support(
            cache,
            support,
            policy_ids,
            solver_name,
            selector=selector,
            epsilon_tolerance=epsilon_tolerance,
        )
        support = list(dict.fromkeys(current.support_profiles))
        probs = np.array(current.support_probabilities, dtype=float)
        if not support or probs.size == 0:
            break
        a_nominal, labels = deviation_matrix(cache, support, policy_ids, ucb=False)
        gains = a_nominal @ probs
        selected_rows = [
            int(idx)
            for idx in np.argsort(-gains)
            if float(gains[int(idx)]) > float(mean_threshold)
        ][: max(1, int(top_constraints))]
        for policy in sorted(targeted):
            policy_rows = [idx for idx, (_, row_policy) in enumerate(labels) if row_policy == policy]
            if not policy_rows:
                continue
            best_policy_row = max(policy_rows, key=lambda idx: float(gains[idx]))
            if best_policy_row not in selected_rows:
                selected_rows.append(int(best_policy_row))
        new_profiles: list[Profile] = []
        repair_candidates_by_profile: dict[Profile, list[tuple[float, int, str]]] = {}
        for row_idx in selected_rows:
            agent, dev_policy = labels[int(row_idx)]
            contributions = probs * np.maximum(0.0, a_nominal[int(row_idx)])
            ranked_profiles = [
                (float(contrib), profile)
                for contrib, profile, prob in zip(contributions, support, probs, strict=True)
                if float(prob) >= q_min and float(contrib) > contribution_min and profile[agent] != dev_policy
            ]
            ranked_profiles.sort(reverse=True, key=lambda item: item[0])
            for contrib, profile in ranked_profiles[: max(1, int(top_profiles_per_constraint))]:
                repaired = list(profile)
                repaired[agent] = dev_policy
                repaired_t = tuple(repaired)
                if repaired_t not in support and repaired_t not in new_profiles:
                    new_profiles.append(repaired_t)
                repair_candidates_by_profile.setdefault(profile, []).append((contrib, agent, dev_policy))
        if enable_multi_agent_repair and max_agents_repaired_per_profile > 1:
            high_q_profiles = sorted(zip(support, probs, strict=True), key=lambda item: item[1], reverse=True)[: max(1, int(beam_width))]
            for profile, _ in high_q_profiles:
                candidates = sorted(repair_candidates_by_profile.get(profile, []), reverse=True)[: max_agents_repaired_per_profile + 1]
                for combo_size in range(2, min(max_agents_repaired_per_profile, len(candidates)) + 1):
                    for combo in combinations(candidates, combo_size):
                        agents = [agent for _, agent, _ in combo]
                        if len(set(agents)) != len(agents):
                            continue
                        repaired = list(profile)
                        for _, agent, dev_policy in combo:
                            repaired[agent] = dev_policy
                        repaired_t = tuple(repaired)
                        if repaired_t not in support and repaired_t not in new_profiles:
                            new_profiles.append(repaired_t)
                        if len(new_profiles) >= int(beam_width):
                            break
                    if len(new_profiles) >= int(beam_width):
                        break
        if not new_profiles:
            break
        slots = max(0, budget - len(support))
        if slots <= 0:
            break
        added_now = list(dict.fromkeys(new_profiles))[:slots]
        support = list(dict.fromkeys(support + added_now))
        repair_added += len(added_now)
        completed_rounds = round_idx + 1
        if len(support) >= budget:
            break
    final = solve_sparse_support(
        cache,
        support,
        policy_ids,
        solver_name,
        selector=selector,
        epsilon_tolerance=epsilon_tolerance,
    )
    final.pricing_mode = "constraint_repair"
    final.certificate_status = "audited_support_deviation_closure"
    final.support_expansion_rounds = base.support_expansion_rounds
    final.active_sampling_rounds = base.active_sampling_rounds
    final.policy_expansion_rounds = base.policy_expansion_rounds
    final.repair_rounds = completed_rounds
    final.repair_profiles_added = repair_added
    final.diagnostics.update({
        "targeted_deviation_policies": sorted(targeted),
        "profile_budget": int(budget),
    })
    return final


def solve_repair_pair_sparse(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    initial_support_size: int,
    max_support_size: int,
    support_add_batch_size: int,
    max_rounds: int,
    target_gap: float,
    seed: int,
    rollouts_max: int | None = None,
    active_sampling: bool = True,
    beta: float = 1.96,
    n_min_pair_samples: int = 4,
    pair_active_rounds: int = 2,
    top_constraints: int = 5,
    top_pairs_per_constraint: int = 10,
    batch_samples_per_pair: int = 2,
    q_min: float = 1.0e-4,
    metadata: dict[str, Any] | None = None,
    repair_rounds: int = 2,
    repair_top_constraints: int = 3,
    top_profiles_per_constraint: int = 10,
    mean_threshold: float = 1.0,
    contribution_min: float = 0.0,
    enable_multi_agent_repair: bool = True,
    max_agents_repaired_per_profile: int = 2,
    beam_width: int = 10,
    profile_budget_multiplier: float = 2.0,
    targeted_deviation_policies: dict[str, list[str]] | None = None,
    selector: str = DEFAULT_SELECTOR,
    epsilon_tolerance: float = DEFAULT_LEXICOGRAPHIC_EPSILON_TOLERANCE,
) -> SparseCceResult:
    repaired = solve_repair_sparse(
        cache,
        policy_ids,
        initial_support_size=initial_support_size,
        max_support_size=max_support_size,
        support_add_batch_size=support_add_batch_size,
        max_rounds=max_rounds,
        target_gap=target_gap,
        seed=seed,
        rollouts_max=rollouts_max,
        active_sampling=active_sampling,
        repair_rounds=repair_rounds,
        top_constraints=repair_top_constraints,
        top_profiles_per_constraint=top_profiles_per_constraint,
        q_min=q_min,
        mean_threshold=mean_threshold,
        contribution_min=contribution_min,
        enable_multi_agent_repair=enable_multi_agent_repair,
        max_agents_repaired_per_profile=max_agents_repaired_per_profile,
        beam_width=beam_width,
        profile_budget_multiplier=profile_budget_multiplier,
        targeted_deviation_policies=targeted_deviation_policies,
        solver_name="REPAIR-SAD-CCE",
        selector=selector,
        epsilon_tolerance=epsilon_tolerance,
    )
    paired = solve_pair_sparse(
        cache,
        policy_ids,
        initial_support_size=initial_support_size,
        max_support_size=max_support_size,
        support_add_batch_size=support_add_batch_size,
        max_rounds=max_rounds,
        target_gap=target_gap,
        seed=seed,
        rollouts_max=rollouts_max,
        active_sampling=active_sampling,
        beta=beta,
        n_min_pair_samples=n_min_pair_samples,
        pair_active_rounds=pair_active_rounds,
        top_constraints=top_constraints,
        top_pairs_per_constraint=top_pairs_per_constraint,
        batch_samples_per_pair=batch_samples_per_pair,
        q_min=q_min,
        metadata=metadata,
        initial_result=repaired,
        solver_name="REPAIR-PAIR-SAD-CCE",
        selector=selector,
        epsilon_tolerance=epsilon_tolerance,
    )
    paired.repair_rounds = repaired.repair_rounds
    paired.repair_profiles_added = repaired.repair_profiles_added
    return paired


def solve_cg_beam_sparse(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    initial_support_size: int,
    beam_width: int,
    max_rounds: int,
    seed: int,
    selector: str = DEFAULT_SELECTOR,
    epsilon_tolerance: float = DEFAULT_LEXICOGRAPHIC_EPSILON_TOLERANCE,
) -> SparseCceResult:
    rng = np.random.default_rng(seed)
    support = random_profiles(policy_ids, cache.n_agents, initial_support_size, rng)
    best: SparseCceResult | None = None
    for round_idx in range(max_rounds):
        result = solve_sparse_support(
            cache,
            support,
            policy_ids,
            "CG-CCE",
            selector=selector,
            epsilon_tolerance=epsilon_tolerance,
        )
        result.pricing_mode = "beam_search"
        result.full_optimality_certified = False
        best = result
        candidates = random_profiles(policy_ids, cache.n_agents, beam_width, rng)
        cache.ensure(candidates)
        ranked = sorted(candidates, key=lambda profile: cache.support_objective(profile), reverse=True)
        for profile in ranked[: max(1, beam_width // 4)]:
            if profile not in support:
                support.append(profile)
        support = expand_by_deviation(support, result, policy_ids, max(1, beam_width // 4), rng)
        support = list(dict.fromkeys(support))
        if round_idx >= max_rounds - 1:
            break
    if best is None:
        raise RuntimeError("CG beam sparse solver did not produce a result.")
    return best
