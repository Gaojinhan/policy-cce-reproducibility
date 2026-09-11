from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import product
import multiprocessing as mp
import os
import time
from typing import Any, Iterable

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


def full_tensor_size(n_agents: int, k: int) -> int:
    return int(k**n_agents)


def profile_space(policy_ids: tuple[str, ...], n_agents: int) -> Iterable[Profile]:
    return (tuple(profile) for profile in product(policy_ids, repeat=n_agents))


def random_profiles(
    policy_ids: tuple[str, ...],
    n_agents: int,
    count: int,
    rng: np.random.Generator,
) -> list[Profile]:
    profiles: list[Profile] = []
    seen: set[Profile] = set()
    max_unique = full_tensor_size(n_agents, len(policy_ids))
    count = min(count, max_unique)
    while len(profiles) < count:
        profile = tuple(str(policy_ids[int(rng.integers(0, len(policy_ids)))]) for _ in range(n_agents))
        if profile not in seen:
            seen.add(profile)
            profiles.append(profile)
    return profiles


def support_deviation_closure(support: Iterable[Profile], policy_ids: tuple[str, ...]) -> set[Profile]:
    closure = set(support)
    for profile in list(closure):
        for agent in range(len(profile)):
            for policy_id in policy_ids:
                dev_profile = list(profile)
                dev_profile[agent] = policy_id
                closure.add(tuple(dev_profile))
    return closure


def _estimate_profile_worker(args: tuple) -> tuple[Profile, RolloutEstimate]:
    (
        profile,
        mechanism_id,
        policies,
        config,
        n_agents,
        seeds,
        n_rollouts,
        markup_grid,
        lead_time_grid,
    ) = args
    return profile, estimate_profile(
        profile=profile,
        mechanism_id=mechanism_id,
        policies=policies,
        config=config,
        n_agents=n_agents,
        seeds=seeds,
        n_rollouts=n_rollouts,
        markup_grid=markup_grid,
        lead_time_grid=lead_time_grid,
    )


@dataclass
class PayoffCache:
    mechanism_id: str
    policies: dict[str, BiddingPolicy]
    config: ToyEnvConfig
    n_agents: int
    seeds: RolloutSeeds
    n_rollouts: int
    markup_grid: tuple[float, ...]
    lead_time_grid: tuple[float, ...]
    workers: int = 1
    estimates: dict[Profile, RolloutEstimate] = field(default_factory=dict)
    eval_time_seconds: float = 0.0
    eval_rollout_episode_count: int = 0
    accessed_profiles: set[Profile] = field(default_factory=set)
    requested_rollouts_by_profile: dict[Profile, int] = field(default_factory=dict)
    paired_delta_samples: dict[tuple[Profile, int, str], list[float]] = field(default_factory=dict)
    paired_delta_rows: list[dict[str, Any]] = field(default_factory=list)
    paired_episode_cache: dict[tuple[Profile, int], tuple[np.ndarray, dict[str, float]]] = field(default_factory=dict)
    accessed_paired_delta_keys: set[tuple[Profile, int, str]] = field(default_factory=set)
    paired_delta_episode_count: int = 0
    paired_delta_eval_time_seconds: float = 0.0
    _paired_delta_episode_count_at_reset: int = 0
    _paired_delta_sample_count_at_reset: int = 0

    def _worker_count(self, task_count: int) -> int:
        configured = max(1, int(self.workers))
        available = max(1, os.cpu_count() or 1)
        return min(configured, available, max(1, task_count))

    def reset_access_log(self) -> None:
        self.accessed_profiles.clear()
        self.requested_rollouts_by_profile.clear()
        self.accessed_paired_delta_keys.clear()
        self._paired_delta_episode_count_at_reset = self.paired_delta_episode_count
        self._paired_delta_sample_count_at_reset = len(self.paired_delta_rows)

    def record_access(self, profiles: Iterable[Profile], requested_rollouts: int) -> None:
        for profile in profiles:
            self.accessed_profiles.add(profile)
            self.requested_rollouts_by_profile[profile] = max(
                int(requested_rollouts),
                self.requested_rollouts_by_profile.get(profile, 0),
            )

    def access_stats(self) -> dict[str, float]:
        requests = list(self.requested_rollouts_by_profile.values())
        paired_episodes = max(0, self.paired_delta_episode_count - self._paired_delta_episode_count_at_reset)
        paired_samples = max(0, len(self.paired_delta_rows) - self._paired_delta_sample_count_at_reset)
        rollout_episodes = int(sum(requests) + paired_episodes)
        return {
            "solver_required_profile_count": int(len(self.accessed_profiles)),
            "solver_required_rollout_episode_total": rollout_episodes,
            "solver_required_rollout_steps_total": int(rollout_episodes * self.config.horizon),
            "solver_required_rollouts_min": int(min(requests)) if requests else 0,
            "solver_required_rollouts_mean": float(np.mean(requests)) if requests else 0.0,
            "solver_required_rollouts_max": int(max(requests)) if requests else 0,
            "paired_delta_pair_count": int(len(self.accessed_paired_delta_keys)),
            "paired_delta_sample_count": int(paired_samples),
            "paired_delta_rollout_episode_total": int(paired_episodes),
        }

    def _estimate_profiles_parallel(self, profiles: list[Profile], n_rollouts: int) -> None:
        if not profiles:
            return
        worker_count = self._worker_count(len(profiles))
        start = time.perf_counter()
        if worker_count <= 1 or len(profiles) < worker_count * 2:
            for profile in profiles:
                self.estimates[profile] = estimate_profile(
                    profile=profile,
                    mechanism_id=self.mechanism_id,
                    policies=self.policies,
                    config=self.config,
                    n_agents=self.n_agents,
                    seeds=self.seeds,
                    n_rollouts=n_rollouts,
                    markup_grid=self.markup_grid,
                    lead_time_grid=self.lead_time_grid,
                )
        else:
            args = [
                (
                    profile,
                    self.mechanism_id,
                    self.policies,
                    self.config,
                    self.n_agents,
                    self.seeds,
                    n_rollouts,
                    self.markup_grid,
                    self.lead_time_grid,
                )
                for profile in profiles
            ]
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context("fork")) as executor:
                for profile, estimate in executor.map(_estimate_profile_worker, args, chunksize=1):
                    self.estimates[profile] = estimate
        self.eval_time_seconds += time.perf_counter() - start
        self.eval_rollout_episode_count += len(profiles) * int(n_rollouts)

    def get(self, profile: Profile) -> RolloutEstimate:
        self.record_access([profile], self.n_rollouts)
        if profile not in self.estimates:
            start = time.perf_counter()
            self.estimates[profile] = estimate_profile(
                profile=profile,
                mechanism_id=self.mechanism_id,
                policies=self.policies,
                config=self.config,
                n_agents=self.n_agents,
                seeds=self.seeds,
                n_rollouts=self.n_rollouts,
                markup_grid=self.markup_grid,
                lead_time_grid=self.lead_time_grid,
            )
            self.eval_time_seconds += time.perf_counter() - start
            self.eval_rollout_episode_count += int(self.n_rollouts)
        return self.estimates[profile]

    def resample(self, profile: Profile, target_rollouts: int) -> RolloutEstimate:
        self.record_access([profile], target_rollouts)
        current = self.estimates.get(profile)
        if current is not None and current.n_rollouts >= target_rollouts:
            return current
        start = time.perf_counter()
        self.estimates[profile] = estimate_profile(
            profile=profile,
            mechanism_id=self.mechanism_id,
            policies=self.policies,
            config=self.config,
            n_agents=self.n_agents,
            seeds=self.seeds,
            n_rollouts=target_rollouts,
            markup_grid=self.markup_grid,
            lead_time_grid=self.lead_time_grid,
        )
        self.eval_time_seconds += time.perf_counter() - start
        self.eval_rollout_episode_count += int(target_rollouts)
        return self.estimates[profile]

    def ensure(self, profiles: Iterable[Profile]) -> None:
        unique_profiles = list(dict.fromkeys(profiles))
        self.record_access(unique_profiles, self.n_rollouts)
        missing = [profile for profile in unique_profiles if profile not in self.estimates]
        self._estimate_profiles_parallel(missing, self.n_rollouts)

    def resample_many(self, profiles: Iterable[Profile], target_rollouts: int) -> None:
        unique_profiles = list(dict.fromkeys(profiles))
        self.record_access(unique_profiles, target_rollouts)
        needs_resample = [
            profile
            for profile in unique_profiles
            if self.estimates.get(profile) is None or self.estimates[profile].n_rollouts < target_rollouts
        ]
        self._estimate_profiles_parallel(needs_resample, target_rollouts)

    def _paired_episode(self, profile: Profile, replication: int) -> tuple[np.ndarray, dict[str, float]]:
        key = (profile, int(replication))
        self.record_access([profile], self.n_rollouts)
        if key not in self.paired_episode_cache:
            start = time.perf_counter()
            returns, metrics = run_episode(
                profile=profile,
                mechanism=mechanism_from_id(self.mechanism_id, self.config),
                policies=self.policies,
                config=self.config,
                n_agents=self.n_agents,
                seeds=self.seeds,
                replication=int(replication),
                markup_grid=self.markup_grid,
                lead_time_grid=self.lead_time_grid,
            )
            elapsed = time.perf_counter() - start
            self.eval_time_seconds += elapsed
            self.paired_delta_eval_time_seconds += elapsed
            self.eval_rollout_episode_count += 1
            self.paired_delta_episode_count += 1
            self.paired_episode_cache[key] = (returns, metrics)
        return self.paired_episode_cache[key]

    def ensure_paired_delta_samples(
        self,
        profile: Profile,
        agent: int,
        dev_policy: str,
        target_samples: int,
        solver_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[float]:
        key = (profile, int(agent), str(dev_policy))
        self.accessed_paired_delta_keys.add(key)
        samples = self.paired_delta_samples.setdefault(key, [])
        target_samples = max(0, int(target_samples))
        if profile[int(agent)] == str(dev_policy):
            return samples
        dev_profile = list(profile)
        dev_profile[int(agent)] = str(dev_policy)
        dev_profile_t = tuple(dev_profile)
        while len(samples) < target_samples:
            replication = len(samples)
            baseline_returns, _ = self._paired_episode(profile, replication)
            deviation_returns, _ = self._paired_episode(dev_profile_t, replication)
            baseline_return = float(baseline_returns[int(agent)])
            deviation_return = float(deviation_returns[int(agent)])
            delta = deviation_return - baseline_return
            samples.append(delta)
            meta = metadata or {}
            self.paired_delta_rows.append(
                {
                    "experiment": str(meta.get("experiment", "")),
                    "mechanism": self.mechanism_id,
                    "env_version": str(meta.get("env_version", "")),
                    "cache_version": "paired_delta_v1",
                    "N": int(self.n_agents),
                    "K": int(meta.get("K", len(self.policies))),
                    "horizon": int(self.config.horizon),
                    "seed": int(meta.get("seed", self.seeds.type_seed)),
                    "solver": solver_name,
                    "support_profile": "|".join(profile),
                    "agent_id": int(agent),
                    "dev_policy": str(dev_policy),
                    "baseline_profile": "|".join(profile),
                    "deviation_profile": "|".join(dev_profile_t),
                    "pair_sample_index": int(replication),
                    "delta_return": delta,
                    "baseline_return": baseline_return,
                    "deviation_return": deviation_return,
                    "type_seed": int(self.seeds.type_seed),
                    "order_seed": int(self.seeds.order_seed),
                    "outside_seed": int(self.seeds.outside_seed),
                    "availability_seed": int(self.seeds.availability_seed),
                    "tie_break_seed": int(self.seeds.tie_break_seed),
                    "rollout_replication_seed": int(self.seeds.rollout_replication_seed),
                    "replication": int(replication),
                    "paired_estimator_mode": "paired_crn",
                }
            )
        return samples

    def support_objective(self, profile: Profile, objective_key: str = "platform_operating_score") -> float:
        return float(self.get(profile).mean_metrics.get(objective_key, 0.0))

    def payoff(self, profile: Profile, agent: int) -> float:
        return float(self.get(profile).mean_returns[agent])

    def payoff_ucb(self, profile: Profile, agent: int) -> float:
        estimate = self.get(profile)
        return float(estimate.mean_returns[agent] + estimate.ci_radius[agent])

    def payoff_lcb(self, profile: Profile, agent: int) -> float:
        estimate = self.get(profile)
        return float(estimate.mean_returns[agent] - estimate.ci_radius[agent])

    @property
    def evaluated_profile_count(self) -> int:
        return len(self.estimates)

    @property
    def rollout_episode_total(self) -> int:
        return int(sum(estimate.n_rollouts for estimate in self.estimates.values()))

    @property
    def rollout_steps_total(self) -> int:
        return int(self.rollout_episode_total * self.config.horizon)

    @property
    def rollout_min(self) -> int:
        if not self.estimates:
            return 0
        return int(min(estimate.n_rollouts for estimate in self.estimates.values()))

    @property
    def rollout_max(self) -> int:
        if not self.estimates:
            return 0
        return int(max(estimate.n_rollouts for estimate in self.estimates.values()))

    @property
    def rollout_mean(self) -> float:
        if not self.estimates:
            return 0.0
        return float(np.mean([estimate.n_rollouts for estimate in self.estimates.values()]))

    def records(self, metadata: dict | None = None) -> list[dict]:
        records: list[dict] = []
        metadata = metadata or {}
        for estimate in self.estimates.values():
            record = {
                **metadata,
                "mechanism": self.mechanism_id,
                "profile": "|".join(estimate.profile),
                "n_rollouts": int(estimate.n_rollouts),
                "env_layer": self.config.layer,
                "env_horizon": int(self.config.horizon),
                "type_seed": int(self.seeds.type_seed),
                "order_seed": int(self.seeds.order_seed),
                "outside_seed": int(self.seeds.outside_seed),
                "availability_seed": int(self.seeds.availability_seed),
                "tie_break_seed": int(self.seeds.tie_break_seed),
                "rollout_replication_seed": int(self.seeds.rollout_replication_seed),
                "policy_library_version": ",".join(self.policies.keys()),
            }
            for i in range(self.n_agents):
                record[f"return_agent_{i}"] = float(estimate.mean_returns[i])
                record[f"var_agent_{i}"] = float(estimate.var_returns[i])
                record[f"ci_agent_{i}"] = float(estimate.ci_radius[i])
            for key, value in estimate.mean_metrics.items():
                record[key] = float(value)
            records.append(record)
        return records
