from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import exp, log, sqrt
import re
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from cmfg_cce.evaluation.empirical_game import EmpiricalGame
from cmfg_cce.evaluation.independent_audit import (
    AuditSampleMatrix,
    summarize_audit_samples,
)
from cmfg_cce.evaluation.rollout import Profile


@dataclass(frozen=True)
class SupportDiversity:
    support_size: int
    shannon_entropy: float
    normalized_entropy: float
    effective_support_size: float
    largest_probability: float


@dataclass(frozen=True)
class PureNashDiagnostics:
    pure_nash_profiles: tuple[Profile, ...]
    profile_maximum_gains: Mapping[Profile, float]
    profile_raw_maximum_gains: Mapping[Profile, float]
    profile_best_replacements: Mapping[Profile, tuple[int, str] | None]
    weak_dominance_pairs: tuple[tuple[int, str, str], ...]
    weak_dominance_margins: Mapping[tuple[int, str, str], tuple[float, float, float]]
    weakly_dominated_policies: Mapping[int, tuple[str, ...]]

    @property
    def has_pure_nash(self) -> bool:
        return bool(self.pure_nash_profiles)


@dataclass(frozen=True)
class RouteFeasibilityFunnel:
    order_count: int
    manufacturer_count: int
    mean_capability_feasible_manufacturers: float
    mean_scalar_capacity_feasible_manufacturers: float
    mean_route_capacity_feasible_manufacturers: float
    capability_to_scalar_loss_per_order: float
    scalar_to_route_loss_per_order: float
    orders_with_scalar_route_mismatch_rate: float
    orders_with_no_route_feasible_manufacturer_rate: float
    orders_with_at_least_three_route_feasible_manufacturers_rate: float
    manufacturer_order_capability_rate: float
    manufacturer_order_scalar_capacity_rate: float
    manufacturer_order_route_capacity_rate: float


@dataclass(frozen=True)
class PairedSeedInference:
    seed_count: int
    cell_count: int
    mean_effect: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    sign_flip_p_value: float
    standardized_paired_effect: float
    seed_mean_effects: tuple[float, ...]


@dataclass(frozen=True)
class OutcomeBootstrapInterval:
    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float


def support_diversity(probabilities: Sequence[float]) -> SupportDiversity:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("Support probabilities must be a nonempty one-dimensional vector.")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("Support probabilities must be finite and nonnegative.")
    positive = probs[probs > 0.0]
    if positive.size == 0:
        raise ValueError("Support probabilities must have positive total mass.")
    positive = positive / np.sum(positive)
    entropy = float(-np.sum(positive * np.log(positive)))
    normalized = 0.0 if positive.size == 1 else entropy / log(positive.size)
    return SupportDiversity(
        support_size=int(positive.size),
        shannon_entropy=entropy,
        normalized_entropy=float(normalized),
        effective_support_size=float(exp(entropy)),
        largest_probability=float(np.max(positive)),
    )


def pure_nash_diagnostics(game: EmpiricalGame, tolerance: float = 1.0e-9) -> PureNashDiagnostics:
    """Enumerate pure Nash profiles and pairwise policy dominance in a complete game."""

    if float(tolerance) < 0.0:
        raise ValueError("tolerance must be nonnegative.")
    profile_index = game.profile_to_index
    expected_profiles = len(game.policy_ids) ** game.n_agents
    if game.n_profiles != expected_profiles or len(profile_index) != expected_profiles:
        raise ValueError("Pure-Nash diagnostics require a complete, duplicate-free policy table.")

    maximum_gains: dict[Profile, float] = {}
    raw_maximum_gains: dict[Profile, float] = {}
    best_replacements: dict[Profile, tuple[int, str] | None] = {}
    pure: list[Profile] = []
    for row, profile in enumerate(game.profiles):
        maximum = -np.inf
        best: tuple[int, str] | None = None
        for agent in range(game.n_agents):
            for policy in game.policy_ids:
                if policy == profile[agent]:
                    continue
                deviated = list(profile)
                deviated[agent] = policy
                gain = float(
                    game.payoffs[profile_index[tuple(deviated)], agent]
                    - game.payoffs[row, agent]
                )
                if gain > maximum:
                    maximum = gain
                    best = (agent, policy)
        raw_maximum_gains[profile] = float(maximum)
        maximum = max(0.0, float(maximum))
        maximum_gains[profile] = maximum
        best_replacements[profile] = best
        if maximum <= float(tolerance):
            pure.append(profile)

    weak_pairs: list[tuple[int, str, str]] = []
    weak_margins: dict[tuple[int, str, str], tuple[float, float, float]] = {}
    dominated: dict[int, set[str]] = {agent: set() for agent in range(game.n_agents)}
    for agent in range(game.n_agents):
        other_agents = [idx for idx in range(game.n_agents) if idx != agent]
        opponents = product(game.policy_ids, repeat=len(other_agents))
        opponent_profiles = tuple(opponents)
        for dominating, candidate in combinations(game.policy_ids, 2):
            for first, second in ((dominating, candidate), (candidate, dominating)):
                differences: list[float] = []
                for opponents_profile in opponent_profiles:
                    profile = [""] * game.n_agents
                    profile[agent] = first
                    for idx, other_agent in enumerate(other_agents):
                        profile[other_agent] = opponents_profile[idx]
                    first_payoff = game.payoffs[profile_index[tuple(profile)], agent]
                    profile[agent] = second
                    second_payoff = game.payoffs[profile_index[tuple(profile)], agent]
                    differences.append(float(first_payoff - second_payoff))
                if all(value >= -float(tolerance) for value in differences) and any(
                    value > float(tolerance) for value in differences
                ):
                    key = (agent, first, second)
                    weak_pairs.append(key)
                    weak_margins[key] = (
                        float(np.min(differences)),
                        float(np.mean(differences)),
                        float(np.max(differences)),
                    )
                    dominated[agent].add(second)
    return PureNashDiagnostics(
        pure_nash_profiles=tuple(sorted(pure)),
        profile_maximum_gains=maximum_gains,
        profile_raw_maximum_gains=raw_maximum_gains,
        profile_best_replacements=best_replacements,
        weak_dominance_pairs=tuple(sorted(set(weak_pairs))),
        weak_dominance_margins=dict(sorted(weak_margins.items())),
        weakly_dominated_policies={
            agent: tuple(sorted(values)) for agent, values in dominated.items()
        },
    )


def route_feasibility_funnel(
    capability_feasible: np.ndarray,
    scalar_capacity_feasible: np.ndarray,
    route_capacity_feasible: np.ndarray,
) -> RouteFeasibilityFunnel:
    capability = np.asarray(capability_feasible, dtype=bool)
    scalar = np.asarray(scalar_capacity_feasible, dtype=bool)
    route = np.asarray(route_capacity_feasible, dtype=bool)
    if capability.ndim != 2 or capability.size == 0:
        raise ValueError("Feasibility inputs must be nonempty order-by-manufacturer matrices.")
    if scalar.shape != capability.shape or route.shape != capability.shape:
        raise ValueError("All feasibility matrices must have identical shapes.")
    if np.any(scalar & ~capability):
        raise ValueError("Scalar-capacity feasibility must imply capability feasibility.")
    if np.any(route & ~scalar):
        raise ValueError("Route-capacity feasibility must imply scalar-capacity feasibility.")
    capability_counts = np.sum(capability, axis=1)
    scalar_counts = np.sum(scalar, axis=1)
    route_counts = np.sum(route, axis=1)
    order_count, manufacturer_count = capability.shape
    return RouteFeasibilityFunnel(
        order_count=int(order_count),
        manufacturer_count=int(manufacturer_count),
        mean_capability_feasible_manufacturers=float(np.mean(capability_counts)),
        mean_scalar_capacity_feasible_manufacturers=float(np.mean(scalar_counts)),
        mean_route_capacity_feasible_manufacturers=float(np.mean(route_counts)),
        capability_to_scalar_loss_per_order=float(np.mean(capability_counts - scalar_counts)),
        scalar_to_route_loss_per_order=float(np.mean(scalar_counts - route_counts)),
        orders_with_scalar_route_mismatch_rate=float(np.mean(np.any(scalar & ~route, axis=1))),
        orders_with_no_route_feasible_manufacturer_rate=float(np.mean(route_counts == 0)),
        orders_with_at_least_three_route_feasible_manufacturers_rate=float(
            np.mean(route_counts >= 3)
        ),
        manufacturer_order_capability_rate=float(np.mean(capability)),
        manufacturer_order_scalar_capacity_rate=float(np.mean(scalar)),
        manufacturer_order_route_capacity_rate=float(np.mean(route)),
    )


def _relative_gap(samples: AuditSampleMatrix, indices: np.ndarray | None = None) -> float:
    gains = np.asarray(samples.gain_samples, dtype=float)
    returns = np.asarray(samples.q_return_samples, dtype=float)
    if indices is not None:
        gains = gains[indices]
        returns = returns[indices]
    numerator = max(0.0, float(np.max(np.mean(gains, axis=0))))
    denominator = max(1.0, float(np.mean(np.abs(np.mean(returns, axis=0)))))
    return 100.0 * numerator / denominator


def summarize_three_arm_transplant(
    same_mechanism: AuditSampleMatrix,
    cross_mechanism: AuditSampleMatrix,
    target_recomputed: AuditSampleMatrix,
    *,
    alpha: float = 0.05,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Jointly audit same, cross, and target arms using common bootstrap draws."""

    arms = {
        "same_control": same_mechanism,
        "cross_transplant": cross_mechanism,
        "target_recomputed": target_recomputed,
    }
    counts = {np.asarray(value.gain_samples).shape[0] for value in arms.values()}
    labels = {value.labels for value in arms.values()}
    if len(counts) != 1 or len(labels) != 1:
        raise ValueError("Transplant arms must align on replication count and replacement labels.")
    replications = counts.pop()
    if replications < 2 or int(bootstrap_samples) < 100:
        raise ValueError("Joint transplant inference requires R >= 2 and at least 100 bootstraps.")
    if not 0.0 < float(alpha) < 0.5:
        raise ValueError("alpha must lie in (0, 0.5).")
    rng = np.random.default_rng(int(bootstrap_seed))
    draws = rng.integers(
        0,
        replications,
        size=(int(bootstrap_samples), replications),
        dtype=np.int32,
    )
    count_dtype = (
        np.uint16 if replications <= np.iinfo(np.uint16).max else np.uint32
    )
    bootstrap_counts = np.stack(
        [np.bincount(draw, minlength=replications) for draw in draws]
    ).astype(count_dtype, copy=False)
    del draws
    point = {name: _relative_gap(samples) for name, samples in arms.items()}
    effects = {
        "cross_minus_same_control_percent": point["cross_transplant"] - point["same_control"],
        "cross_minus_target_recomputed_percent": point["cross_transplant"]
        - point["target_recomputed"],
    }
    boot_relative: dict[str, np.ndarray] = {}
    for name, samples in arms.items():
        gains = np.asarray(samples.gain_samples, dtype=float)
        returns = np.asarray(samples.q_return_samples, dtype=float)
        gain_means = bootstrap_counts @ gains / float(replications)
        return_means = bootstrap_counts @ returns / float(replications)
        numerators = np.maximum(0.0, np.max(gain_means, axis=1))
        denominators = np.maximum(1.0, np.mean(np.abs(return_means), axis=1))
        boot_relative[name] = 100.0 * numerators / denominators
    boot_effects = {
        "cross_minus_same_control_percent": (
            boot_relative["cross_transplant"] - boot_relative["same_control"]
        ),
        "cross_minus_target_recomputed_percent": (
            boot_relative["cross_transplant"]
            - boot_relative["target_recomputed"]
        ),
    }
    intervals = {
        name: {
            "lower": float(np.quantile(values, float(alpha) / 2.0)),
            "upper": float(np.quantile(values, 1.0 - float(alpha) / 2.0)),
            "standard_error": float(np.std(values, ddof=1)),
        }
        for name, values in boot_effects.items()
    }
    arm_summaries = {
        name: summarize_audit_samples(
            samples,
            alpha=alpha,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for name, samples in arms.items()
    }
    return {
        "replications": int(replications),
        "arm_relative_nominal_gap_percent": point,
        "effects": effects,
        "effect_intervals": intervals,
        "arm_audits": arm_summaries,
    }


def familywise_audit_sensitivity(
    games: Mapping[str, AuditSampleMatrix],
    *,
    alpha: float = 0.05,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Report ordinary and across-game Bonferroni-sensitive audit bounds."""

    if not games:
        raise ValueError("At least one game is required for family-wise sensitivity.")
    family_alpha = float(alpha) / len(games)
    rows: dict[str, object] = {}
    for offset, (game_id, samples) in enumerate(sorted(games.items())):
        rows[game_id] = {
            "per_game": summarize_audit_samples(
                samples,
                alpha=alpha,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=int(bootstrap_seed) + offset,
            ),
            "familywise_bonferroni": summarize_audit_samples(
                samples,
                alpha=family_alpha,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=int(bootstrap_seed) + offset,
            ),
        }
    return {
        "game_count": len(games),
        "nominal_alpha": float(alpha),
        "familywise_alpha_per_game": family_alpha,
        "games": rows,
    }


def analytic_familywise_relative_gap_bounds(
    games: Mapping[str, AuditSampleMatrix],
    *,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Bonferroni-t relative-gap UCBs across a predeclared game family.

    This deliberately avoids estimating extreme bootstrap quantiles after an
    across-game multiplicity correction.  The family error rate is split over
    lower/upper replacement-gain tails and lower/upper payoff-denominator
    tails.  Bounds on ``|E[U_i]|`` are obtained from simultaneous t intervals
    before averaging across agents.
    """

    if not games:
        raise ValueError("At least one game is required for family-wise bounds.")
    if not 0.0 < float(alpha) < 0.5:
        raise ValueError("alpha must lie in (0, 0.5).")
    normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    total_constraints = 0
    total_payoff_means = 0
    for game_id, samples in sorted(games.items()):
        gains = np.asarray(samples.gain_samples, dtype=float)
        returns = np.asarray(samples.q_return_samples, dtype=float)
        if (
            gains.ndim != 2
            or returns.ndim != 2
            or gains.shape[0] != returns.shape[0]
            or gains.shape[0] < 2
            or gains.shape[1] != len(samples.labels)
            or not np.all(np.isfinite(gains))
            or not np.all(np.isfinite(returns))
        ):
            raise ValueError(f"Invalid family-wise audit matrix for {game_id!r}.")
        normalized[str(game_id)] = (gains, returns)
        total_constraints += int(gains.shape[1])
        total_payoff_means += int(returns.shape[1])
    if total_constraints <= 0 or total_payoff_means <= 0:
        raise ValueError("Family-wise audit matrices must contain gains and payoffs.")

    # Four simultaneous tails are needed for a genuine interval: lower and
    # upper bounds for the maximum replacement gain, and lower and upper
    # bounds for the payoff denominator.  Keeping these allocations explicit
    # lets downstream transplant reducers subtract two certified intervals
    # without silently reusing a one-sided 95% bound as a two-sided one.
    numerator_alpha = float(alpha) / 4.0
    denominator_alpha = float(alpha) / 4.0
    rows: dict[str, object] = {}
    for game_id, (gains, returns) in normalized.items():
        df = int(gains.shape[0] - 1)
        gain_means = np.mean(gains, axis=0)
        gain_se = np.std(gains, axis=0, ddof=1) / np.sqrt(gains.shape[0])
        payoff_means = np.mean(returns, axis=0)
        payoff_se = np.std(returns, axis=0, ddof=1) / np.sqrt(returns.shape[0])
        gain_critical = float(
            student_t.ppf(1.0 - numerator_alpha / total_constraints, df=df)
        )
        payoff_critical = float(
            student_t.ppf(1.0 - denominator_alpha / total_payoff_means, df=df)
        )
        numerator_ucb = max(
            0.0, float(np.max(gain_means + gain_critical * gain_se))
        )
        numerator_lcb = max(
            0.0, float(np.max(gain_means - gain_critical * gain_se))
        )
        denominator_lcb = max(
            1.0,
            float(
                np.mean(
                    np.maximum(
                        0.0,
                        np.abs(payoff_means) - payoff_critical * payoff_se,
                    )
                )
            ),
        )
        denominator_ucb = max(
            1.0,
            float(np.mean(np.abs(payoff_means) + payoff_critical * payoff_se)),
        )
        rows[game_id] = {
            "audit_rollouts": int(gains.shape[0]),
            "constraint_count": int(gains.shape[1]),
            "payoff_component_count": int(returns.shape[1]),
            "gain_critical": gain_critical,
            "payoff_critical": payoff_critical,
            "absolute_gap_lcb": numerator_lcb,
            "absolute_gap_ucb": numerator_ucb,
            "payoff_denominator_lcb": denominator_lcb,
            "payoff_denominator_ucb": denominator_ucb,
            "relative_gap_lcb_percent": 100.0 * numerator_lcb / denominator_ucb,
            "relative_gap_ucb_percent": 100.0 * numerator_ucb / denominator_lcb,
        }
    return {
        "method": "analytic_Bonferroni_t_across_games_constraints_and_denominators",
        "game_count": len(normalized),
        "family_alpha": float(alpha),
        "alpha_per_gain_tail": numerator_alpha,
        "alpha_per_payoff_tail": denominator_alpha,
        "total_replacement_constraints": total_constraints,
        "total_payoff_components": total_payoff_means,
        "games": rows,
    }


def paired_seed_inference(
    effects: Mapping[int, Sequence[float]],
    *,
    alpha: float = 0.05,
) -> PairedSeedInference:
    """Average fixed operating cells within seed, then infer across seed clusters."""

    if len(effects) < 2:
        raise ValueError("At least two independent seeds are required.")
    lengths = {len(values) for values in effects.values()}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("Every seed must contain the same nonzero number of matched cells.")
    ordered = sorted((int(seed), np.asarray(values, dtype=float)) for seed, values in effects.items())
    if any(not np.all(np.isfinite(values)) for _, values in ordered):
        raise ValueError("Paired effects must be finite.")
    seed_means = np.array([float(np.mean(values)) for _, values in ordered], dtype=float)
    n = len(seed_means)
    mean = float(np.mean(seed_means))
    standard_deviation = float(np.std(seed_means, ddof=1))
    standard_error = standard_deviation / sqrt(n)
    critical = float(student_t.ppf(1.0 - float(alpha) / 2.0, df=n - 1))
    observed = abs(mean)
    signs = np.array(tuple(product((-1.0, 1.0), repeat=n)), dtype=float)
    permutation_means = np.abs(np.mean(signs * seed_means[None, :], axis=1))
    p_value = float(np.mean(permutation_means >= observed - 1.0e-15))
    standardized = 0.0 if standard_deviation <= 1.0e-15 else mean / standard_deviation
    return PairedSeedInference(
        seed_count=n,
        cell_count=int(lengths.pop()),
        mean_effect=mean,
        standard_error=standard_error,
        ci_lower=mean - critical * standard_error,
        ci_upper=mean + critical * standard_error,
        sign_flip_p_value=p_value,
        standardized_paired_effect=float(standardized),
        seed_mean_effects=tuple(float(value) for value in seed_means),
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        raise ValueError("At least one p-value is required.")
    if any(not np.isfinite(value) or not 0.0 <= float(value) <= 1.0 for value in p_values.values()):
        raise ValueError("p-values must be finite values in [0, 1].")
    ordered = sorted((float(value), key) for key, value in p_values.items())
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (value, key) in enumerate(ordered):
        running = max(running, (count - rank) * value)
        adjusted[key] = min(1.0, running)
    return adjusted


def weighted_outcome_replications(
    profile_records: Mapping[Profile, Mapping[str, Sequence[float]]],
    support: Sequence[Profile],
    probabilities: Sequence[float],
) -> dict[str, np.ndarray]:
    """Form q-weighted per-replication raw outcomes while preserving CRN indices."""

    if len(support) != len(probabilities) or not support:
        raise ValueError("Support and probability vectors must be nonempty and aligned.")
    probs = np.asarray(probabilities, dtype=float)
    if np.any(probs < 0.0) or not np.all(np.isfinite(probs)) or np.sum(probs) <= 0.0:
        raise ValueError("Outcome weights must be finite and nonnegative with positive mass.")
    probs = probs / np.sum(probs)
    missing = [profile for profile in support if profile not in profile_records]
    if missing:
        raise KeyError(f"Missing outcome records for support profiles: {missing[:3]}")
    field_sets = [set(profile_records[profile]) for profile in support]
    if any(fields != field_sets[0] for fields in field_sets[1:]):
        raise ValueError("Every support profile must expose the same raw outcome fields.")
    result: dict[str, np.ndarray] = {}
    replication_count: int | None = None
    for field_name in sorted(field_sets[0]):
        arrays = [np.asarray(profile_records[profile][field_name], dtype=float) for profile in support]
        if any(array.ndim != 1 or not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError(f"Outcome field {field_name!r} must contain finite vectors.")
        lengths = {len(array) for array in arrays}
        if len(lengths) != 1:
            raise ValueError(f"Outcome field {field_name!r} has inconsistent replication counts.")
        current_count = lengths.pop()
        if replication_count is None:
            replication_count = current_count
        elif current_count != replication_count:
            raise ValueError("All raw outcome fields must share one replication count.")
        result[field_name] = np.sum(
            np.vstack([probability * array for probability, array in zip(probs, arrays, strict=True)]),
            axis=0,
        )
    if replication_count is None or replication_count < 2:
        raise ValueError("At least two outcome replications are required.")
    return result


def _require_raw_outcome_fields(
    totals: Mapping[str, np.ndarray], fields: Sequence[str]
) -> None:
    missing = [field for field in fields if field not in totals]
    if missing:
        raise KeyError(f"Missing required raw outcome fields: {', '.join(missing)}")


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return result


def _reconstruct_outcome_metric_samples(
    totals: Mapping[str, np.ndarray],
    *,
    n_manufacturers: int,
    sample_sizes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reconstruct every nonlinear outcome from resampled raw totals.

    Each value in ``totals`` is a vector with one element per bootstrap draw (or
    one element for the point estimate).  Keeping the reconstruction here means
    the reported point estimates and bootstrap draws cannot silently use
    different ratio, HHI, or family-availability definitions.
    """

    required = (
        "orders_offered_count",
        "assignment_count",
        "invitation_count",
        "valid_bid_count",
        "platform_total_payment",
        "manufacturer_discounted_profit_sum",
        *(f"wins_manufacturer_{idx}_count" for idx in range(n_manufacturers)),
    )
    _require_raw_outcome_fields(totals, required)
    sample_sizes = np.asarray(sample_sizes, dtype=float)
    if sample_sizes.ndim != 1 or np.any(sample_sizes <= 0.0):
        raise ValueError("Outcome sample sizes must be positive one-dimensional values.")
    shapes = {np.asarray(values).shape for values in totals.values()}
    if shapes != {(len(sample_sizes),)}:
        raise ValueError("Every raw total must align with the outcome sample-size vector.")

    assignments = np.asarray(totals["assignment_count"], dtype=float)
    orders = np.asarray(totals["orders_offered_count"], dtype=float)
    invitations = np.asarray(totals["invitation_count"], dtype=float)
    valid_bids = np.asarray(totals["valid_bid_count"], dtype=float)
    total_payment = np.asarray(totals["platform_total_payment"], dtype=float)
    discounted_profit = np.asarray(
        totals["manufacturer_discounted_profit_sum"], dtype=float
    )
    wins = np.column_stack(
        [totals[f"wins_manufacturer_{idx}_count"] for idx in range(n_manufacturers)]
    )
    total_wins = np.sum(wins, axis=1)
    win_shares = _ratio(wins, total_wins[:, None])
    hhi = np.sum(win_shares**2, axis=1)
    normalized_hhi = (hhi - 1.0 / n_manufacturers) / (
        1.0 - 1.0 / n_manufacturers
    )

    output = {
        "payment_per_assignment": _ratio(total_payment, assignments),
        "conditional_bid_rate": _ratio(valid_bids, invitations),
        "assignment_rate": _ratio(assignments, orders),
        "winner_concentration_hhi": hhi,
        "normalized_winner_hhi": normalized_hhi,
        "platform_total_payment": total_payment / sample_sizes,
        "manufacturer_discounted_profit": discounted_profit / sample_sizes,
        # Retain the submitted-paper public name, but derive it exclusively
        # from the formal discounted-profit raw field above.
        "manufacturer_total_profit": discounted_profit / sample_sizes,
    }

    submitted_metrics = (
        ("average_markup", "submitted_markup_sum", "submitted_markup_count"),
        (
            "submitted_lead_time_multiplier",
            "submitted_lead_time_sum",
            "submitted_lead_time_count",
        ),
        (
            "submitted_commitment_periods",
            "submitted_commitment_periods_sum",
            "submitted_lead_time_count",
        ),
    )
    for public_name, numerator, denominator in submitted_metrics:
        if numerator in totals or denominator in totals:
            _require_raw_outcome_fields(totals, (numerator, denominator))
            output[public_name] = _ratio(totals[numerator], totals[denominator])

    funnel_fields = (
        "capability_feasible_manufacturer_opportunities",
        "scalar_capacity_feasible_manufacturer_opportunities",
        "route_capacity_feasible_manufacturer_opportunities",
    )
    if any(field in totals for field in funnel_fields):
        _require_raw_outcome_fields(totals, funnel_fields)
        capability = np.asarray(totals[funnel_fields[0]], dtype=float)
        scalar = np.asarray(totals[funnel_fields[1]], dtype=float)
        route = np.asarray(totals[funnel_fields[2]], dtype=float)
        tolerance = 1.0e-8
        if np.any(scalar > capability + tolerance) or np.any(route > scalar + tolerance):
            raise ValueError(
                "Raw outcome funnel must satisfy capability >= scalar-capacity >= "
                "route-capacity feasibility."
            )
        output.update(
            {
                "capability_feasible_manufacturers_per_order": _ratio(
                    capability, orders
                ),
                "scalar_capacity_feasible_manufacturers_per_order": _ratio(
                    scalar, orders
                ),
                "route_capacity_feasible_manufacturers_per_order": _ratio(
                    route, orders
                ),
                "capability_to_scalar_loss_per_order": _ratio(
                    capability - scalar, orders
                ),
                "scalar_to_route_loss_per_order": _ratio(scalar - route, orders),
                "capability_feasible_manufacturer_opportunity_rate": _ratio(
                    capability, orders * n_manufacturers
                ),
                "scalar_capacity_feasible_manufacturer_opportunity_rate": _ratio(
                    scalar, orders * n_manufacturers
                ),
                "route_capacity_feasible_manufacturer_opportunity_rate": _ratio(
                    route, orders * n_manufacturers
                ),
                "route_capacity_feasible_share_of_capability": _ratio(
                    route, capability
                ),
            }
        )
        false_positive = "scalar_route_false_positive_opportunities"
        if false_positive in totals:
            output["conditional_route_false_positive_rate"] = _ratio(
                totals[false_positive], scalar
            )

    mismatch_orders = "orders_with_scalar_route_mismatch_count"
    if mismatch_orders in totals:
        output["orders_with_scalar_route_mismatch_rate"] = _ratio(
            totals[mismatch_orders], orders
        )

    family_codes = sorted(
        {
            match.group(1)
            for field in totals
            if (match := re.fullmatch(r"orders_offered_(F\d+)", field))
        },
        key=lambda code: int(code[1:]),
    )
    family_availability_totals: list[np.ndarray] = []
    for family in family_codes:
        offered_field = f"orders_offered_{family}"
        assigned_field = f"orders_assigned_{family}"
        capability_field = (
            f"capability_feasible_manufacturer_opportunities_{family}"
        )
        scalar_field = (
            f"scalar_capacity_feasible_manufacturer_opportunities_{family}"
        )
        route_field = f"route_capacity_feasible_manufacturer_opportunities_{family}"
        false_positive_field = f"scalar_route_false_positive_opportunities_{family}"
        availability_field = (
            "orders_with_at_least_three_route_capacity_feasible_manufacturers_"
            f"{family}"
        )
        legacy_availability_field = (
            "orders_with_at_least_three_route_capacity_feasible_providers_"
            f"{family}"
        )
        if availability_field not in totals and legacy_availability_field in totals:
            # Compatibility for archived submission-era raw records.  The
            # revision-full-v1 contract uses the manufacturer-named field.
            availability_field = legacy_availability_field
        _require_raw_outcome_fields(
            totals,
            (
                offered_field,
                assigned_field,
                capability_field,
                scalar_field,
                route_field,
                false_positive_field,
                availability_field,
            ),
        )
        family_availability_totals.append(
            np.asarray(totals[availability_field], dtype=float)
        )
        offered = np.asarray(totals[offered_field], dtype=float)
        capability = np.asarray(totals[capability_field], dtype=float)
        scalar = np.asarray(totals[scalar_field], dtype=float)
        route = np.asarray(totals[route_field], dtype=float)
        tolerance = 1.0e-8
        if np.any(scalar > capability + tolerance) or np.any(route > scalar + tolerance):
            raise ValueError(
                f"Family {family} raw outcome funnel is logically inconsistent."
            )
        output.update(
            {
                f"assignment_rate_{family}": _ratio(
                    totals[assigned_field], offered
                ),
                f"capability_feasible_manufacturers_per_order_{family}": _ratio(
                    capability, offered
                ),
                f"scalar_capacity_feasible_manufacturers_per_order_{family}": _ratio(
                    scalar, offered
                ),
                f"route_capacity_feasible_manufacturers_per_order_{family}": _ratio(
                    route, offered
                ),
                f"conditional_route_false_positive_rate_{family}": _ratio(
                    totals[false_positive_field], scalar
                ),
                f"route_capacity_feasible_share_of_capability_{family}": _ratio(
                    route, capability
                ),
                (
                    "at_least_three_route_capacity_feasible_manufacturers_rate_"
                    f"{family}"
                ): _ratio(totals[availability_field], offered),
            }
        )
    if family_availability_totals:
        output[
            "at_least_three_route_capacity_feasible_manufacturers_rate"
        ] = _ratio(np.sum(family_availability_totals, axis=0), orders)

    policy_pattern = re.compile(r"winner_policy_(.+)_count")
    policy_ids = sorted(
        match.group(1)
        for field in totals
        if (match := policy_pattern.fullmatch(field))
    )
    for policy_id in policy_ids:
        output[f"winner_policy_{policy_id}_share"] = _ratio(
            totals[f"winner_policy_{policy_id}_count"], assignments
        )

    capacity_prefix = "machine_group_remaining_capacity_sum_"
    group_codes = sorted(
        field[len(capacity_prefix) :]
        for field in totals
        if field.startswith(capacity_prefix)
    )
    for code in group_codes:
        count_field = f"machine_group_snapshot_count_{code}"
        _require_raw_outcome_fields(totals, (count_field,))
        output[f"machine_group_remaining_capacity_{code}"] = _ratio(
            totals[f"{capacity_prefix}{code}"], totals[count_field]
        )
    return output


def aggregate_outcome_replications(
    records: Mapping[str, Sequence[float]],
    *,
    n_manufacturers: int,
    indices: Sequence[int] | None = None,
) -> dict[str, float]:
    """Recompute nonlinear outcome metrics from raw counts, never from episode ratios."""

    arrays = {name: np.asarray(values, dtype=float) for name, values in records.items()}
    if not arrays or int(n_manufacturers) < 2:
        raise ValueError("Raw outcomes and at least two manufacturers are required.")
    lengths = {len(array) for array in arrays.values() if array.ndim == 1}
    if any(array.ndim != 1 for array in arrays.values()) or len(lengths) != 1:
        raise ValueError("Raw outcome fields must be aligned one-dimensional vectors.")
    replication_count = lengths.pop()
    if replication_count < 2 or any(not np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError("Raw outcome fields need at least two finite replications.")
    draw = (
        np.arange(replication_count, dtype=int)
        if indices is None
        else np.asarray(indices, dtype=int)
    )
    if draw.ndim != 1 or draw.size == 0 or np.any(draw < 0) or np.any(draw >= replication_count):
        raise ValueError("Outcome resampling indices are invalid.")

    totals = {
        name: np.array([float(np.sum(values[draw]))], dtype=float)
        for name, values in arrays.items()
    }
    reconstructed = _reconstruct_outcome_metric_samples(
        totals,
        n_manufacturers=n_manufacturers,
        sample_sizes=np.array([len(draw)], dtype=float),
    )
    return {name: float(values[0]) for name, values in reconstructed.items()}


def bootstrap_outcome_replications(
    records: Mapping[str, Sequence[float]],
    *,
    n_manufacturers: int,
    alpha: float = 0.05,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 0,
) -> dict[str, OutcomeBootstrapInterval]:
    """Bootstrap aligned replications and reconstruct ratios/HHI inside every draw."""

    arrays = {name: np.asarray(values, dtype=float) for name, values in records.items()}
    if not arrays:
        raise ValueError("Raw outcome records cannot be empty.")
    replication_counts = {len(array) for array in arrays.values()}
    if len(replication_counts) != 1:
        raise ValueError("Raw outcome fields must share one replication count.")
    replications = replication_counts.pop()
    if replications < 2 or int(bootstrap_samples) < 100:
        raise ValueError("Outcome bootstrap requires R >= 2 and at least 100 draws.")
    if not 0.0 < float(alpha) < 0.5:
        raise ValueError("alpha must lie in (0, 0.5).")
    point = aggregate_outcome_replications(arrays, n_manufacturers=n_manufacturers)
    rng = np.random.default_rng(int(bootstrap_seed))
    # A multinomial count row is exactly a nonparametric bootstrap resample of
    # the aligned replication indices.  Matrix multiplication reconstructs all
    # raw totals for all draws at once, avoiding 5000 Python aggregation loops.
    bootstrap_weights = rng.multinomial(
        replications,
        np.full(replications, 1.0 / replications),
        size=int(bootstrap_samples),
    )
    field_names = tuple(arrays)
    raw_matrix = np.column_stack([arrays[name] for name in field_names])
    resampled_totals_matrix = bootstrap_weights @ raw_matrix
    resampled_totals = {
        name: resampled_totals_matrix[:, index]
        for index, name in enumerate(field_names)
    }
    samples = _reconstruct_outcome_metric_samples(
        resampled_totals,
        n_manufacturers=n_manufacturers,
        sample_sizes=np.full(int(bootstrap_samples), replications, dtype=float),
    )
    result: dict[str, OutcomeBootstrapInterval] = {}
    for name, estimate in point.items():
        values = samples[name]
        finite = values[np.isfinite(values)]
        if not np.isfinite(estimate) or finite.size < max(2, int(0.9 * len(values))):
            result[name] = OutcomeBootstrapInterval(
                estimate=float(estimate),
                standard_error=float("nan"),
                ci_lower=float("nan"),
                ci_upper=float("nan"),
            )
            continue
        result[name] = OutcomeBootstrapInterval(
            estimate=float(estimate),
            standard_error=float(np.std(finite, ddof=1)),
            ci_lower=float(np.quantile(finite, float(alpha) / 2.0)),
            ci_upper=float(np.quantile(finite, 1.0 - float(alpha) / 2.0)),
        )
    return result
