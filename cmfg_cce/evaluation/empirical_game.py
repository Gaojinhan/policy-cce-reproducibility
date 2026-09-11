from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from cmfg_cce.evaluation.rollout import Profile, RolloutEstimate


@dataclass(frozen=True)
class EmpiricalGame:
    profiles: tuple[Profile, ...]
    policy_ids: tuple[str, ...]
    payoffs: np.ndarray
    ci_radius: np.ndarray
    objectives: np.ndarray
    metrics: tuple[dict[str, float], ...]

    @property
    def n_profiles(self) -> int:
        return len(self.profiles)

    @property
    def n_agents(self) -> int:
        return int(self.payoffs.shape[1])

    @property
    def profile_to_index(self) -> dict[Profile, int]:
        return {profile: idx for idx, profile in enumerate(self.profiles)}


def enumerate_profiles(policy_ids: tuple[str, ...], n_agents: int) -> tuple[Profile, ...]:
    return tuple(tuple(profile) for profile in product(policy_ids, repeat=n_agents))


def build_empirical_game(
    estimates: list[RolloutEstimate],
    policy_ids: tuple[str, ...],
    objective_key: str = "platform_operating_score",
) -> EmpiricalGame:
    estimates_by_profile = {estimate.profile: estimate for estimate in estimates}
    profiles = tuple(estimates_by_profile.keys())
    payoffs = np.vstack([estimates_by_profile[profile].mean_returns for profile in profiles])
    ci_radius = np.vstack([estimates_by_profile[profile].ci_radius for profile in profiles])
    objectives = np.array(
        [estimates_by_profile[profile].mean_metrics.get(objective_key, 0.0) for profile in profiles],
        dtype=float,
    )
    metrics = tuple(estimates_by_profile[profile].mean_metrics for profile in profiles)
    return EmpiricalGame(
        profiles=profiles,
        policy_ids=policy_ids,
        payoffs=payoffs,
        ci_radius=ci_radius,
        objectives=objectives,
        metrics=metrics,
    )
