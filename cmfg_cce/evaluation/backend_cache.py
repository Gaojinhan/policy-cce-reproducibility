from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import multiprocessing as mp
import os
import time
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

from cmfg_cce.evaluation.profile_backend import ProfileEvaluationBackend
from cmfg_cce.evaluation.rollout import Profile, RolloutEstimate


def _estimate_backend_worker(
    args: tuple[Profile, ProfileEvaluationBackend, int],
) -> tuple[Profile, RolloutEstimate]:
    profile, backend, n_rollouts = args
    return profile, backend.estimate(profile, n_rollouts)


@dataclass
class BackendPayoffCache:
    """Payoff-cache implementation for pluggable simulation backends.

    It intentionally mirrors the public surface of the legacy ``PayoffCache``
    so that sparse CCE solvers can use either environment without branching on
    manufacturing-model details.  The legacy cache remains unchanged to keep
    archived aggregate-capacity experiments reproducible.
    """

    backend: ProfileEvaluationBackend
    n_rollouts: int
    workers: int = 1
    estimates: dict[Profile, RolloutEstimate] = field(default_factory=dict)
    eval_time_seconds: float = 0.0
    eval_rollout_episode_count: int = 0
    accessed_profiles: set[Profile] = field(default_factory=set)
    requested_rollouts_by_profile: dict[Profile, int] = field(default_factory=dict)
    paired_delta_samples: dict[tuple[Profile, int, str], list[float]] = field(default_factory=dict)
    paired_delta_rows: list[dict[str, Any]] = field(default_factory=list)
    paired_episode_cache: dict[tuple[Profile, int], tuple[np.ndarray, dict[str, float]]] = field(
        default_factory=dict
    )
    accessed_paired_delta_keys: set[tuple[Profile, int, str]] = field(default_factory=set)
    paired_delta_episode_count: int = 0
    paired_delta_eval_time_seconds: float = 0.0
    _paired_delta_episode_count_at_reset: int = 0
    _paired_delta_sample_count_at_reset: int = 0

    @property
    def mechanism_id(self) -> str:
        return str(self.backend.mechanism_id)

    @property
    def n_agents(self) -> int:
        return int(self.backend.n_agents)

    @property
    def config(self) -> SimpleNamespace:
        """Compatibility view used by existing accounting helpers."""

        return SimpleNamespace(
            horizon=int(self.backend.horizon),
            layer=str(self.backend.env_version),
        )

    def _worker_count(self, task_count: int) -> int:
        configured = max(1, int(self.workers))
        available = max(1, os.cpu_count() or 1)
        return min(configured, available, max(1, task_count))

    @staticmethod
    def _process_context() -> mp.context.BaseContext:
        """Use one multiprocessing contract on Linux, macOS, and Windows.

        Revision workers run inside both Linux containers and Windows Docker
        Desktop hosts.  ``fork`` is not available on Windows and can also hide
        pickling errors that would appear after a Spot VM restart.  ``spawn``
        gives every worker the same clean-process semantics.
        """

        return mp.get_context("spawn")

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
        paired_episodes = max(
            0,
            self.paired_delta_episode_count - self._paired_delta_episode_count_at_reset,
        )
        paired_samples = max(
            0,
            len(self.paired_delta_rows) - self._paired_delta_sample_count_at_reset,
        )
        rollout_episodes = int(sum(requests) + paired_episodes)
        return {
            "solver_required_profile_count": int(len(self.accessed_profiles)),
            "solver_required_rollout_episode_total": rollout_episodes,
            "solver_required_rollout_steps_total": int(rollout_episodes * self.backend.horizon),
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
                self.estimates[profile] = self.backend.estimate(profile, n_rollouts)
        else:
            args = [(profile, self.backend, int(n_rollouts)) for profile in profiles]
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=self._process_context(),
            ) as executor:
                for profile, estimate in executor.map(
                    _estimate_backend_worker,
                    args,
                    chunksize=1,
                ):
                    self.estimates[profile] = estimate
        self.eval_time_seconds += time.perf_counter() - start
        self.eval_rollout_episode_count += len(profiles) * int(n_rollouts)

    def get(self, profile: Profile) -> RolloutEstimate:
        self.record_access([profile], self.n_rollouts)
        if profile not in self.estimates:
            self._estimate_profiles_parallel([profile], self.n_rollouts)
        return self.estimates[profile]

    def ensure(self, profiles: Iterable[Profile]) -> None:
        unique_profiles = list(dict.fromkeys(profiles))
        self.record_access(unique_profiles, self.n_rollouts)
        missing = [profile for profile in unique_profiles if profile not in self.estimates]
        self._estimate_profiles_parallel(missing, self.n_rollouts)

    def resample(self, profile: Profile, target_rollouts: int) -> RolloutEstimate:
        self.record_access([profile], target_rollouts)
        current = self.estimates.get(profile)
        if current is None or current.n_rollouts < int(target_rollouts):
            self._estimate_profiles_parallel([profile], int(target_rollouts))
        return self.estimates[profile]

    def resample_many(self, profiles: Iterable[Profile], target_rollouts: int) -> None:
        unique_profiles = list(dict.fromkeys(profiles))
        self.record_access(unique_profiles, target_rollouts)
        needs_resample = [
            profile
            for profile in unique_profiles
            if self.estimates.get(profile) is None
            or self.estimates[profile].n_rollouts < int(target_rollouts)
        ]
        self._estimate_profiles_parallel(needs_resample, int(target_rollouts))

    def _paired_episode(
        self,
        profile: Profile,
        replication: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        key = (profile, int(replication))
        self.record_access([profile], self.n_rollouts)
        if key not in self.paired_episode_cache:
            start = time.perf_counter()
            returns, metrics = self.backend.run_episode(profile, int(replication))
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
        if profile[int(agent)] == str(dev_policy):
            return samples
        deviation = list(profile)
        deviation[int(agent)] = str(dev_policy)
        deviation_profile = tuple(deviation)
        meta = metadata or {}
        while len(samples) < int(target_samples):
            replication = len(samples)
            baseline_returns, _ = self._paired_episode(profile, replication)
            deviation_returns, _ = self._paired_episode(deviation_profile, replication)
            baseline_return = float(baseline_returns[int(agent)])
            deviation_return = float(deviation_returns[int(agent)])
            delta = deviation_return - baseline_return
            samples.append(delta)
            self.paired_delta_rows.append(
                {
                    **self.backend.cache_identity,
                    "experiment": str(meta.get("experiment", "")),
                    "mechanism": self.mechanism_id,
                    "N": self.n_agents,
                    "K": int(meta.get("K", 0)),
                    "horizon": int(self.backend.horizon),
                    "seed": int(meta.get("seed", 0)),
                    "solver": str(solver_name),
                    "support_profile": "|".join(profile),
                    "agent_id": int(agent),
                    "dev_policy": str(dev_policy),
                    "baseline_profile": "|".join(profile),
                    "deviation_profile": "|".join(deviation_profile),
                    "pair_sample_index": int(replication),
                    "delta_return": delta,
                    "baseline_return": baseline_return,
                    "deviation_return": deviation_return,
                    "paired_estimator_mode": "paired_crn",
                }
            )
        return samples

    def support_objective(
        self,
        profile: Profile,
        objective_key: str = "platform_operating_score",
    ) -> float:
        return float(self.get(profile).mean_metrics.get(objective_key, 0.0))

    def payoff(self, profile: Profile, agent: int) -> float:
        return float(self.get(profile).mean_returns[int(agent)])

    def payoff_ucb(self, profile: Profile, agent: int) -> float:
        estimate = self.get(profile)
        return float(estimate.mean_returns[int(agent)] + estimate.ci_radius[int(agent)])

    def payoff_lcb(self, profile: Profile, agent: int) -> float:
        estimate = self.get(profile)
        return float(estimate.mean_returns[int(agent)] - estimate.ci_radius[int(agent)])

    @property
    def evaluated_profile_count(self) -> int:
        return len(self.estimates)

    @property
    def rollout_episode_total(self) -> int:
        return int(sum(estimate.n_rollouts for estimate in self.estimates.values()))

    @property
    def rollout_steps_total(self) -> int:
        return int(self.rollout_episode_total * self.backend.horizon)

    @property
    def rollout_min(self) -> int:
        return int(min((e.n_rollouts for e in self.estimates.values()), default=0))

    @property
    def rollout_max(self) -> int:
        return int(max((e.n_rollouts for e in self.estimates.values()), default=0))

    @property
    def rollout_mean(self) -> float:
        if not self.estimates:
            return 0.0
        return float(np.mean([e.n_rollouts for e in self.estimates.values()]))

    def records(self, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        prefix = {**self.backend.cache_identity, **(metadata or {})}
        rows: list[dict[str, Any]] = []
        for estimate in self.estimates.values():
            row: dict[str, Any] = {
                **prefix,
                "mechanism": self.mechanism_id,
                "profile": "|".join(estimate.profile),
                "n_rollouts": int(estimate.n_rollouts),
                "env_horizon": int(self.backend.horizon),
            }
            for agent in range(self.n_agents):
                row[f"return_agent_{agent}"] = float(estimate.mean_returns[agent])
                row[f"var_agent_{agent}"] = float(estimate.var_returns[agent])
                row[f"ci_agent_{agent}"] = float(estimate.ci_radius[agent])
            row.update({key: float(value) for key, value in estimate.mean_metrics.items()})
            rows.append(row)
        return rows
