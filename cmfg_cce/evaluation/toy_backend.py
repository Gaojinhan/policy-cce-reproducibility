from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
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


class ToyPolicyGameBackend:
    """Profile backend for the aggregate CMfg benchmark.

    The legacy :class:`PayoffCache` remains available for archived experiments.
    This adapter gives the revision runner the same backend interface used by
    the route-capacity case, which is needed for common chunking, fresh-stream
    audits, and cross-machine deterministic replay.
    """

    def __init__(
        self,
        *,
        mechanism_id: str,
        policies: dict[str, BiddingPolicy],
        config: ToyEnvConfig,
        n_agents: int,
        seeds: RolloutSeeds,
        markup_grid: tuple[float, ...],
        lead_time_grid: tuple[float, ...],
        stream_label: str,
    ) -> None:
        if n_agents <= 0:
            raise ValueError("n_agents must be positive.")
        if not policies:
            raise ValueError("The policy library cannot be empty.")
        self.mechanism_id = str(mechanism_id)
        self.policies = dict(policies)
        self.config = config
        self.n_agents = int(n_agents)
        self.seeds = seeds
        self.markup_grid = tuple(float(value) for value in markup_grid)
        self.lead_time_grid = tuple(float(value) for value in lead_time_grid)
        self.stream_label = str(stream_label)
        # Validate the mechanism/config pair before a long campaign starts.
        mechanism_from_id(self.mechanism_id, self.config)

    @property
    def horizon(self) -> int:
        return int(self.config.horizon)

    @property
    def env_version(self) -> str:
        return str(self.config.layer)

    @property
    def cache_identity(self) -> dict[str, Any]:
        payload = {
            "backend": "toy_policy_game_v1",
            "mechanism_id": self.mechanism_id,
            "n_agents": self.n_agents,
            "policy_ids": list(self.policies),
            "config": asdict(self.config),
            "seeds": asdict(self.seeds),
            "markup_grid": self.markup_grid,
            "lead_time_grid": self.lead_time_grid,
            "stream_label": self.stream_label,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {
            "backend": payload["backend"],
            "mechanism_id": self.mechanism_id,
            "n_agents": self.n_agents,
            "env_version": self.env_version,
            "stream_label": self.stream_label,
            "identity_sha256": hashlib.sha256(encoded).hexdigest(),
        }

    def run_episode(
        self,
        profile: Profile,
        replication: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        mechanism = mechanism_from_id(self.mechanism_id, self.config)
        return run_episode(
            profile=tuple(profile),
            mechanism=mechanism,
            policies=self.policies,
            config=self.config,
            n_agents=self.n_agents,
            seeds=self.seeds,
            replication=int(replication),
            markup_grid=self.markup_grid,
            lead_time_grid=self.lead_time_grid,
        )

    def estimate(self, profile: Profile, n_rollouts: int) -> RolloutEstimate:
        return estimate_profile(
            profile=tuple(profile),
            mechanism_id=self.mechanism_id,
            policies=self.policies,
            config=self.config,
            n_agents=self.n_agents,
            seeds=self.seeds,
            n_rollouts=int(n_rollouts),
            markup_grid=self.markup_grid,
            lead_time_grid=self.lead_time_grid,
        )
