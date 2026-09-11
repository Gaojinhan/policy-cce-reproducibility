from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from cmfg_cce.evaluation.rollout import Profile


@dataclass(frozen=True)
class FrozenDistribution:
    solver: str
    support: tuple[Profile, ...]
    probabilities: tuple[float, ...]
    q_hash: str


@dataclass(frozen=True)
class AuditSampleMatrix:
    labels: tuple[tuple[int, str], ...]
    gain_samples: np.ndarray
    q_return_samples: np.ndarray


def distribution_hash(support: Sequence[Profile], probabilities: Sequence[float]) -> str:
    pairs = sorted(
        (
            list(profile),
            float(probability).hex(),
        )
        for profile, probability in zip(support, probabilities, strict=True)
    )
    payload = json.dumps(pairs, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def freeze_distribution(
    solver: str,
    support: Sequence[Profile],
    probabilities: Sequence[float],
    n_agents: int,
    policy_ids: Sequence[str],
) -> FrozenDistribution:
    if not support:
        raise ValueError("A frozen audit distribution must have nonempty support.")
    if len(support) != len(probabilities):
        raise ValueError("Support and probability lengths differ.")
    if len(set(support)) != len(support):
        raise ValueError("A frozen audit distribution contains duplicate support profiles.")
    allowed = set(policy_ids)
    normalized_support: list[Profile] = []
    for profile in support:
        profile_t = tuple(str(value) for value in profile)
        if len(profile_t) != int(n_agents):
            raise ValueError(f"Profile {profile_t} does not have N={n_agents} entries.")
        unknown = set(profile_t).difference(allowed)
        if unknown:
            raise ValueError(f"Profile {profile_t} contains unknown policies: {sorted(unknown)}")
        normalized_support.append(profile_t)
    probs = np.asarray(probabilities, dtype=float)
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("Distribution probabilities must be finite and nonnegative.")
    total = float(np.sum(probs))
    if total <= 0.0:
        raise ValueError("Distribution probabilities must have positive total mass.")
    # A distribution that has already been frozen must survive a JSON
    # round-trip without acquiring a different hash.  Unconditionally
    # dividing by a total such as ``1.0000000000000002`` changes the last
    # bits of otherwise valid probabilities and makes a second freeze
    # non-idempotent.  Normalize genuinely unnormalised input, but preserve
    # an already unit-sum vector bit for bit.
    if not np.isclose(total, 1.0, rtol=0.0, atol=1.0e-12):
        probs = probs / total
    if np.any(probs <= 0.0):
        raise ValueError("Frozen support must contain only positive-probability profiles.")
    support_t = tuple(normalized_support)
    probs_t = tuple(float(value) for value in probs)
    return FrozenDistribution(
        solver=str(solver),
        support=support_t,
        probabilities=probs_t,
        q_hash=distribution_hash(support_t, probs_t),
    )


def build_joint_audit_samples(
    profile_returns: Mapping[Profile, np.ndarray],
    distribution: FrozenDistribution,
    policy_ids: Sequence[str],
    n_agents: int,
    sample_count: int | None = None,
) -> AuditSampleMatrix:
    """Build per-replication gains, preserving covariance across support profiles.

    Every returns array must have shape ``(replications, n_agents)`` and use the
    same replication index for the same common-random-number realization.
    """

    if distribution.q_hash != distribution_hash(distribution.support, distribution.probabilities):
        raise ValueError("Frozen distribution hash no longer matches its support and probabilities.")
    arrays: dict[Profile, np.ndarray] = {}
    available_samples: int | None = None
    for profile, values in profile_returns.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != int(n_agents):
            raise ValueError(f"Returns for {profile} must have shape (R, {n_agents}).")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Returns for {profile} contain non-finite values.")
        if available_samples is None:
            available_samples = int(array.shape[0])
        elif int(array.shape[0]) != available_samples:
            raise ValueError("All profile-return arrays must use the same replication count.")
        arrays[tuple(profile)] = array
    if available_samples is None or available_samples < 2:
        raise ValueError("At least two audit replications are required.")
    use_samples = available_samples if sample_count is None else int(sample_count)
    if use_samples < 2 or use_samples > available_samples:
        raise ValueError(
            f"Requested sample_count={use_samples} is outside [2, {available_samples}]."
        )

    missing = [profile for profile in distribution.support if profile not in arrays]
    if missing:
        raise KeyError(f"Missing support return arrays: {missing[:3]}")

    probs = np.asarray(distribution.probabilities, dtype=float)
    q_returns = np.zeros((use_samples, int(n_agents)), dtype=float)
    for profile, probability in zip(distribution.support, probs, strict=True):
        q_returns += probability * arrays[profile][:use_samples]

    labels: list[tuple[int, str]] = []
    gains: list[np.ndarray] = []
    for agent in range(int(n_agents)):
        for dev_policy in policy_ids:
            row = np.zeros(use_samples, dtype=float)
            for profile, probability in zip(distribution.support, probs, strict=True):
                if profile[agent] == str(dev_policy):
                    continue
                dev_profile = list(profile)
                dev_profile[agent] = str(dev_policy)
                dev_profile_t = tuple(dev_profile)
                if dev_profile_t not in arrays:
                    raise KeyError(
                        f"Missing deviation returns for profile={profile}, agent={agent}, "
                        f"policy={dev_policy}: {dev_profile_t}"
                    )
                row += probability * (
                    arrays[dev_profile_t][:use_samples, agent]
                    - arrays[profile][:use_samples, agent]
                )
            labels.append((agent, str(dev_policy)))
            gains.append(row)
    return AuditSampleMatrix(
        labels=tuple(labels),
        gain_samples=np.column_stack(gains),
        q_return_samples=q_returns,
    )


def _standard_errors(samples: np.ndarray) -> np.ndarray:
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("Standard errors require a two-dimensional array with R >= 2.")
    return np.std(samples, axis=0, ddof=1) / np.sqrt(samples.shape[0])


def _bootstrap_counts(bootstrap_indices: np.ndarray, replications: int) -> np.ndarray:
    """Compress joint index draws into a reusable bootstrap weight matrix."""

    indices = np.asarray(bootstrap_indices)
    if indices.ndim != 2 or indices.shape[1] != int(replications):
        raise ValueError("Bootstrap indices must have shape (B, R).")
    if np.any(indices < 0) or np.any(indices >= int(replications)):
        raise ValueError("Bootstrap indices fall outside the replication range.")
    # uint16 is sufficient for the frozen R=2000 design.  The matrix is cast
    # only by BLAS as needed and is about half the size of the original int64
    # index array.
    dtype = np.uint16 if int(replications) <= np.iinfo(np.uint16).max else np.uint32
    return np.stack(
        [np.bincount(row, minlength=int(replications)) for row in indices]
    ).astype(dtype, copy=False)


def _bootstrap_means(samples: np.ndarray, bootstrap_counts: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    counts = np.asarray(bootstrap_counts)
    if values.ndim != 2 or counts.ndim != 2 or counts.shape[1] != values.shape[0]:
        raise ValueError("Bootstrap counts and samples are not aligned.")
    return np.asarray(counts @ values / float(values.shape[0]), dtype=float)


def _max_t_critical_from_means(
    gain_samples: np.ndarray,
    centered_bootstrap_means: np.ndarray,
    alpha: float,
    tail: str = "upper",
) -> float:
    if tail not in {"upper", "lower"}:
        raise ValueError("tail must be 'upper' or 'lower'.")
    means = np.mean(gain_samples, axis=0)
    se = _standard_errors(gain_samples)
    valid = se > 1.0e-12
    if not np.any(valid):
        return 0.0
    boot_means = np.asarray(centered_bootstrap_means, dtype=float)
    if boot_means.ndim != 2 or boot_means.shape[1] != gain_samples.shape[1]:
        raise ValueError("Centered bootstrap means do not match gain constraints.")
    standardized = boot_means[:, valid] / se[valid]
    if tail == "lower":
        standardized = -standardized
    statistics = np.max(standardized, axis=1)
    return max(0.0, float(np.quantile(statistics, 1.0 - float(alpha))))


def _denominator(q_return_samples: np.ndarray) -> float:
    means = np.mean(q_return_samples, axis=0)
    return max(1.0, float(np.mean(np.abs(means))))


def _denominator_bounds_from_means(
    q_return_samples: np.ndarray,
    bootstrap_return_means: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    estimate = _denominator(q_return_samples)
    means = np.asarray(bootstrap_return_means, dtype=float)
    if means.ndim != 2 or means.shape[1] != q_return_samples.shape[1]:
        raise ValueError("Bootstrap return means do not match q-return samples.")
    boot = np.maximum(
        1.0,
        np.mean(np.abs(means), axis=1),
    )
    lower_error = max(0.0, float(np.quantile(boot - estimate, 1.0 - float(alpha))))
    upper_error = max(0.0, float(np.quantile(estimate - boot, 1.0 - float(alpha))))
    return max(1.0, estimate - lower_error), max(1.0, estimate + upper_error)


def summarize_audit_samples(
    samples: AuditSampleMatrix,
    alpha: float = 0.05,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, float | int | str]:
    gains = np.asarray(samples.gain_samples, dtype=float)
    q_returns = np.asarray(samples.q_return_samples, dtype=float)
    if gains.ndim != 2 or q_returns.ndim != 2 or gains.shape[0] != q_returns.shape[0]:
        raise ValueError("Gain and q-return samples must be aligned two-dimensional arrays.")
    if gains.shape[1] != len(samples.labels):
        raise ValueError("Gain columns do not match replacement labels.")
    if gains.shape[0] < 2:
        raise ValueError("At least two audit replications are required.")
    if not 0.0 < float(alpha) < 0.5:
        raise ValueError("alpha must lie in (0, 0.5).")
    if int(bootstrap_samples) < 100:
        raise ValueError("At least 100 bootstrap samples are required.")

    replications, constraint_count = gains.shape
    means = np.mean(gains, axis=0)
    se = _standard_errors(gains)
    nominal_idx = int(np.argmax(means))
    nominal_gap = max(0.0, float(means[nominal_idx]))
    denominator = _denominator(q_returns)

    df = replications - 1
    point_beta = float(student_t.ppf(1.0 - float(alpha), df=df))
    bonf_beta = float(student_t.ppf(1.0 - float(alpha) / max(1, constraint_count), df=df))
    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(
        0,
        replications,
        size=(int(bootstrap_samples), replications),
        dtype=np.int32,
    )
    counts = _bootstrap_counts(indices, replications)
    del indices
    centered_bootstrap_means = _bootstrap_means(gains - means, counts)
    bootstrap_return_means = _bootstrap_means(q_returns, counts)
    max_t_beta = _max_t_critical_from_means(
        gains, centered_bootstrap_means, alpha=float(alpha)
    )
    max_t_beta_lower = _max_t_critical_from_means(
        gains,
        centered_bootstrap_means,
        alpha=float(alpha),
        tail="lower",
    )

    point_ucbs = means + point_beta * se
    bonf_ucbs = means + bonf_beta * se
    max_t_ucbs = means + max_t_beta * se
    point_lcbs = means - point_beta * se
    bonf_lcbs = means - bonf_beta * se
    max_t_lcbs = means - max_t_beta_lower * se
    point_idx = int(np.argmax(point_ucbs))
    bonf_idx = int(np.argmax(bonf_ucbs))
    max_t_idx = int(np.argmax(max_t_ucbs))

    # Split alpha across the simultaneous numerator and the denominator lower
    # bound so the relative-gap upper bound propagates uncertainty in both.
    split_alpha = float(alpha) / 2.0
    point_beta_rel = float(student_t.ppf(1.0 - split_alpha, df=df))
    bonf_beta_rel = float(
        student_t.ppf(1.0 - split_alpha / max(1, constraint_count), df=df)
    )
    max_t_beta_rel = _max_t_critical_from_means(
        gains, centered_bootstrap_means, alpha=split_alpha
    )
    max_t_beta_rel_lower = _max_t_critical_from_means(
        gains,
        centered_bootstrap_means,
        alpha=split_alpha,
        tail="lower",
    )
    denominator_lcb, denominator_ucb = _denominator_bounds_from_means(
        q_returns, bootstrap_return_means, alpha=split_alpha
    )
    point_gap_rel = max(0.0, float(np.max(means + point_beta_rel * se)))
    bonf_gap_rel = max(0.0, float(np.max(means + bonf_beta_rel * se)))
    max_t_gap_rel = max(0.0, float(np.max(means + max_t_beta_rel * se)))
    point_gap_rel_lower = max(0.0, float(np.max(means - point_beta_rel * se)))
    bonf_gap_rel_lower = max(0.0, float(np.max(means - bonf_beta_rel * se)))
    max_t_gap_rel_lower = max(
        0.0, float(np.max(means - max_t_beta_rel_lower * se))
    )

    worst_agent, worst_policy = samples.labels[nominal_idx]
    point_agent, point_policy = samples.labels[point_idx]
    bonf_agent, bonf_policy = samples.labels[bonf_idx]
    max_t_agent, max_t_policy = samples.labels[max_t_idx]
    return {
        "audit_rollouts": int(replications),
        "constraint_count": int(constraint_count),
        "nominal_gap": nominal_gap,
        "relative_nominal_gap_percent": 100.0 * nominal_gap / denominator,
        "payoff_denominator": denominator,
        "payoff_denominator_lcb": denominator_lcb,
        "payoff_denominator_ucb": denominator_ucb,
        "max_payoff_standard_error": float(np.max(_standard_errors(q_returns))),
        "max_replacement_gain_standard_error": float(np.max(se)),
        "pointwise_gap_ucb95": max(0.0, float(point_ucbs[point_idx])),
        "bonferroni_gap_ucb95": max(0.0, float(bonf_ucbs[bonf_idx])),
        "max_t_gap_ucb95": max(0.0, float(max_t_ucbs[max_t_idx])),
        "pointwise_gap_lcb95": max(0.0, float(np.max(point_lcbs))),
        "bonferroni_gap_lcb95": max(0.0, float(np.max(bonf_lcbs))),
        "max_t_gap_lcb95": max(0.0, float(np.max(max_t_lcbs))),
        "pointwise_relative_gap_ucb95_percent": 100.0 * point_gap_rel / denominator_lcb,
        "bonferroni_relative_gap_ucb95_percent": 100.0 * bonf_gap_rel / denominator_lcb,
        "max_t_relative_gap_ucb95_percent": 100.0 * max_t_gap_rel / denominator_lcb,
        "pointwise_relative_gap_lcb95_percent": 100.0
        * point_gap_rel_lower
        / denominator_ucb,
        "bonferroni_relative_gap_lcb95_percent": 100.0
        * bonf_gap_rel_lower
        / denominator_ucb,
        "max_t_relative_gap_lcb95_percent": 100.0
        * max_t_gap_rel_lower
        / denominator_ucb,
        "pointwise_beta": point_beta,
        "bonferroni_beta": bonf_beta,
        "max_t_beta": max_t_beta,
        "max_t_beta_lower": max_t_beta_lower,
        "max_t_beta_relative": max_t_beta_rel,
        "max_t_beta_relative_lower": max_t_beta_rel_lower,
        "worst_mean_agent": int(worst_agent),
        "worst_mean_policy": str(worst_policy),
        "worst_pointwise_agent": int(point_agent),
        "worst_pointwise_policy": str(point_policy),
        "worst_bonferroni_agent": int(bonf_agent),
        "worst_bonferroni_policy": str(bonf_policy),
        "worst_max_t_agent": int(max_t_agent),
        "worst_max_t_policy": str(max_t_policy),
    }
