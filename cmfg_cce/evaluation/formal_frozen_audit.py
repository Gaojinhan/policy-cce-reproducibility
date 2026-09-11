from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from cmfg_cce.evaluation.independent_audit import (
    AuditSampleMatrix,
    FrozenDistribution,
    build_joint_audit_samples,
    distribution_hash,
    summarize_audit_samples,
)
from cmfg_cce.evaluation.rollout import Profile
from cmfg_cce.evaluation.revision_statistics import familywise_audit_sensitivity


FORMAL_AUDIT_ROLLOUTS = 2000
FORMAL_ALPHA = 0.05


@dataclass(frozen=True)
class FormalFrozenAuditResult:
    distribution: FrozenDistribution
    samples: AuditSampleMatrix
    summary: Mapping[str, float | int | str]
    profile_count: int
    audit_rollouts: int


def _validate_profile_returns(
    profile_returns: Mapping[Profile, np.ndarray],
    n_agents: int,
    audit_rollouts: int,
) -> None:
    if not profile_returns:
        raise ValueError("A formal audit requires profile-level replication vectors.")
    for profile, values in profile_returns.items():
        array = np.asarray(values, dtype=float)
        if array.shape != (int(audit_rollouts), int(n_agents)):
            raise ValueError(
                f"Formal audit returns for {profile} have shape {array.shape}; expected "
                f"({audit_rollouts}, {n_agents})."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Formal audit returns for {profile} are not finite.")


def audit_frozen_distribution(
    profile_returns: Mapping[Profile, np.ndarray],
    distribution: FrozenDistribution,
    policy_ids: Sequence[str],
    n_agents: int,
    *,
    audit_rollouts: int = FORMAL_AUDIT_ROLLOUTS,
    alpha: float = FORMAL_ALPHA,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 0,
) -> FormalFrozenAuditResult:
    """Compute the publication audit without allowing any post-selection q update."""

    if int(audit_rollouts) != FORMAL_AUDIT_ROLLOUTS:
        raise ValueError("revision-full-v1 freezes every formal equilibrium audit at R=2000.")
    before_hash = distribution.q_hash
    if before_hash != distribution_hash(distribution.support, distribution.probabilities):
        raise ValueError("Frozen q hash is invalid before formal audit.")
    _validate_profile_returns(profile_returns, n_agents, audit_rollouts)
    samples = build_joint_audit_samples(
        profile_returns,
        distribution,
        policy_ids,
        n_agents=n_agents,
        sample_count=audit_rollouts,
    )
    summary = summarize_audit_samples(
        samples,
        alpha=alpha,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    after_hash = distribution_hash(distribution.support, distribution.probabilities)
    if before_hash != after_hash:
        raise RuntimeError("Frozen q changed during formal audit.")
    return FormalFrozenAuditResult(
        distribution=distribution,
        samples=samples,
        summary=summary,
        profile_count=len(profile_returns),
        audit_rollouts=int(audit_rollouts),
    )


def audit_frozen_family(
    games: Mapping[
        str,
        tuple[
            Mapping[Profile, np.ndarray],
            FrozenDistribution,
            Sequence[str],
            int,
        ],
    ],
    *,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 0,
) -> tuple[dict[str, FormalFrozenAuditResult], dict[str, object]]:
    """Audit a predeclared game family and add across-game Bonferroni sensitivity."""

    if not games:
        raise ValueError("A formal audit family cannot be empty.")
    results: dict[str, FormalFrozenAuditResult] = {}
    for offset, (game_id, (returns, distribution, policy_ids, n_agents)) in enumerate(
        sorted(games.items())
    ):
        results[game_id] = audit_frozen_distribution(
            returns,
            distribution,
            policy_ids,
            n_agents,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=int(bootstrap_seed) + offset,
        )
    sensitivity = familywise_audit_sensitivity(
        {game_id: result.samples for game_id, result in results.items()},
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return results, sensitivity
