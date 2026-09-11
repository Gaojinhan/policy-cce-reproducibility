from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cmfg_cce.envs.toy import ToyEnvConfig
from cmfg_cce.evaluation.rollout import (
    Profile,
    RolloutEstimate,
    RolloutSeeds,
    estimate_profile,
    mechanism_from_id,
    run_episode,
)
from cmfg_cce.policies.base import BiddingPolicy


@dataclass(frozen=True)
class LegacyAggregateBackend:
    """Adapter exposing the archived aggregate-capacity simulator as a backend.

    The adapter delegates to the existing rollout functions without changing
    their seeds, mechanisms, policy interpretation, or metric definitions.  It
    lets new solver/cache orchestration share one protocol while the original
    ``PayoffCache`` and published benchmark artifacts remain reproducible.
    """

    mechanism_id: str
    policies: dict[str, BiddingPolicy]
    config: ToyEnvConfig
    n_agents: int
    seeds: RolloutSeeds
    markup_grid: tuple[float, ...]
    lead_time_grid: tuple[float, ...] = (0.55, 0.70, 0.85, 1.00)
    environment_label: str = "legacy_aggregate_capacity_v1"

    @property
    def horizon(self) -> int:
        return int(self.config.horizon)

    @property
    def env_version(self) -> str:
        return f"{self.environment_label}:{self.config.layer}"

    @property
    def cache_identity(self) -> dict[str, Any]:
        return {
            "env_version": self.env_version,
            "fleet_version": "legacy_generated_population_v1",
            "order_generator_version": "legacy_scalar_orders_v1",
            "pressure_cell": str(self.config.layer),
            "mechanism": self.mechanism_id,
            "seed": {
                "type": int(self.seeds.type_seed),
                "order": int(self.seeds.order_seed),
                "tie_break": int(self.seeds.tie_break_seed),
                "rollout": int(self.seeds.rollout_replication_seed),
                "outside": int(self.seeds.outside_seed),
                "availability": int(self.seeds.availability_seed),
            },
        }

    def run_episode(
        self,
        profile: Profile,
        replication: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        mechanism = mechanism_from_id(self.mechanism_id, self.config)
        return run_episode(
            profile,
            mechanism,
            self.policies,
            self.config,
            self.n_agents,
            self.seeds,
            int(replication),
            self.markup_grid,
            self.lead_time_grid,
        )

    def estimate(self, profile: Profile, n_rollouts: int) -> RolloutEstimate:
        return estimate_profile(
            profile,
            self.mechanism_id,
            self.policies,
            self.config,
            self.n_agents,
            self.seeds,
            int(n_rollouts),
            self.markup_grid,
            self.lead_time_grid,
        )
