from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from cmfg_cce.evaluation.rollout import Profile, RolloutEstimate


@runtime_checkable
class ProfileEvaluationBackend(Protocol):
    """Environment-independent profile evaluator used by policy-game caches.

    A backend owns all environment-specific objects (population, order generator,
    mechanisms, policies, and random-stream conventions).  The CCE solvers only
    need payoff estimates and therefore interact with it through this small
    interface.
    """

    mechanism_id: str
    n_agents: int
    horizon: int
    env_version: str

    @property
    def cache_identity(self) -> dict[str, Any]:
        """Stable metadata distinguishing incompatible simulation environments."""

    def estimate(self, profile: Profile, n_rollouts: int) -> RolloutEstimate:
        """Estimate one joint-policy profile using the backend's CRN streams."""

    def run_episode(
        self,
        profile: Profile,
        replication: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Run one replication, used for paired and holdout deviation audits."""
