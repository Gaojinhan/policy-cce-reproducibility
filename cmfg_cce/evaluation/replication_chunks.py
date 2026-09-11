from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from cmfg_cce.evaluation.profile_backend import ProfileEvaluationBackend
from cmfg_cce.evaluation.rollout import Profile
from cmfg_cce.orchestration.chunks import ChunkPlan, ChunkStatus, ImmutableChunkStore


def _evaluate_profile_replications(
    args: tuple[
        int,
        Profile,
        ProfileEvaluationBackend,
        int,
        tuple[str, ...],
    ],
) -> tuple[int, np.ndarray, dict[str, np.ndarray]]:
    index, profile, backend, rollouts, metric_names = args
    returns = np.empty((int(rollouts), int(backend.n_agents)), dtype=np.float64)
    metric_values = {
        name: np.empty(int(rollouts), dtype=np.float64) for name in metric_names
    }
    for replication in range(int(rollouts)):
        episode_returns, metrics = backend.run_episode(profile, replication)
        values = np.asarray(episode_returns, dtype=np.float64)
        if values.shape != (int(backend.n_agents),) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Profile {profile!r}, replication {replication} returned invalid payoffs."
            )
        returns[replication] = values
        for name in metric_names:
            if name not in metrics or not np.isfinite(float(metrics[name])):
                raise ValueError(
                    f"Profile {profile!r}, replication {replication} has no finite "
                    f"raw metric {name!r}."
                )
            metric_values[name][replication] = float(metrics[name])
    return int(index), returns, metric_values


@dataclass(frozen=True)
class ReplicationChunkIdentity:
    campaign_sha256: str
    matrix_sha256: str
    job_id: str
    stage_id: str
    kind: str
    chunk_size: int

    @property
    def store_job_id(self) -> str:
        return f"{self.job_id}::{self.stage_id}"


class ProfileReplicationChunks:
    """Crash-safe replication vectors for one frozen profile list.

    Training tables use 128-profile chunks.  Fresh equilibrium audits and
    independent outcome evaluation use 8-profile chunks.  Profiles and their
    ordering are frozen in ``chunk_plan.json`` before the first simulation.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        identity: ReplicationChunkIdentity,
        profiles: Sequence[Profile],
    ) -> None:
        self.root = Path(root)
        self.identity = identity
        self.profiles = tuple(tuple(str(value) for value in profile) for profile in profiles)
        if not self.profiles or len(set(self.profiles)) != len(self.profiles):
            raise ValueError("Replication chunks require nonempty unique profiles.")
        expected = 128 if identity.kind == "training" else 8 if identity.kind == "audit" else None
        if expected is None or int(identity.chunk_size) != expected:
            raise ValueError(f"{identity.kind!r} stages require chunk size {expected}.")
        self.root.mkdir(parents=True, exist_ok=True)
        item_ids = tuple("|".join(profile) for profile in self.profiles)
        self.plan = ChunkPlan.create(
            campaign_sha256=identity.campaign_sha256,
            matrix_sha256=identity.matrix_sha256,
            job_id=identity.store_job_id,
            kind=identity.kind,
            chunk_size=identity.chunk_size,
            item_ids=item_ids,
        )
        plan_path = self.root / "chunk_plan.json"
        self.plan.write_immutable(plan_path)
        self.store = ImmutableChunkStore(
            self.root,
            campaign_sha256=identity.campaign_sha256,
            matrix_sha256=identity.matrix_sha256,
            job_id=identity.store_job_id,
            kind=identity.kind,
            item_count=len(self.profiles),
            chunk_size=identity.chunk_size,
        )

    @staticmethod
    def _process_context() -> mp.context.BaseContext:
        # Spawn has the same semantics on Linux, macOS, and Windows workers.
        return mp.get_context("spawn")

    def evaluate_pending(
        self,
        backend: ProfileEvaluationBackend,
        *,
        rollouts: int,
        workers: int,
        metric_names: Sequence[str] = (),
    ) -> None:
        if int(rollouts) < 2:
            raise ValueError("Replication stages require at least two rollouts.")
        if len(self.profiles[0]) != int(backend.n_agents):
            raise ValueError("Profile size and backend manufacturer count differ.")
        metrics = tuple(str(value) for value in metric_names)
        if len(set(metrics)) != len(metrics):
            raise ValueError("metric_names must be unique.")
        for start, stop in self.store.pending_bounds():
            chunk_profiles = self.profiles[start:stop]
            tasks = [
                (start + offset, profile, backend, int(rollouts), metrics)
                for offset, profile in enumerate(chunk_profiles)
            ]
            worker_count = min(
                max(1, int(workers)),
                max(1, os.cpu_count() or 1),
                len(tasks),
            )
            if worker_count == 1:
                evaluated = [_evaluate_profile_replications(task) for task in tasks]
            else:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=self._process_context(),
                ) as executor:
                    evaluated = list(executor.map(_evaluate_profile_replications, tasks, chunksize=1))
            evaluated.sort(key=lambda item: item[0])
            expected_indices = list(range(start, stop))
            if [item[0] for item in evaluated] != expected_indices:
                raise RuntimeError("Parallel profile evaluation changed the frozen item order.")
            arrays: dict[str, np.ndarray] = {
                "returns": np.stack([item[1] for item in evaluated]),
                "profile_indices": np.arange(start, stop, dtype=np.int64),
                "rollout_count": np.asarray([int(rollouts)], dtype=np.int64),
            }
            for name in metrics:
                arrays[f"metric__{name}"] = np.stack([item[2][name] for item in evaluated])
            self.store.commit(start, stop, arrays)

    def validate_complete(self, *, rollouts: int, metric_names: Sequence[str] = ()) -> None:
        metrics = tuple(str(value) for value in metric_names)
        for record in self.store.validate_complete():
            arrays = self.store.read(record.start, record.stop)
            count = record.stop - record.start
            expected_return_shape = (count, int(rollouts), len(self.profiles[0]))
            if arrays.get("returns", np.empty(0)).shape != expected_return_shape:
                raise ValueError(
                    f"Chunk {record.chunk_id} has an unexpected returns shape."
                )
            np.testing.assert_array_equal(
                arrays.get("profile_indices"),
                np.arange(record.start, record.stop, dtype=np.int64),
            )
            if arrays.get("rollout_count", np.empty(0)).tolist() != [int(rollouts)]:
                raise ValueError(f"Chunk {record.chunk_id} has a different rollout count.")
            for name in metrics:
                values = arrays.get(f"metric__{name}")
                if values is None or values.shape != (count, int(rollouts)):
                    raise ValueError(
                        f"Chunk {record.chunk_id} has no valid metric {name!r}."
                    )

    def return_vectors(self, *, rollouts: int) -> dict[Profile, np.ndarray]:
        self.validate_complete(rollouts=rollouts)
        result: dict[Profile, np.ndarray] = {}
        for start, stop in self.store.bounds():
            values = self.store.read(start, stop)["returns"]
            for offset, profile in enumerate(self.profiles[start:stop]):
                result[profile] = np.asarray(values[offset], dtype=float)
        return result

    def metric_vectors(
        self,
        *,
        rollouts: int,
        metric_names: Sequence[str],
    ) -> dict[Profile, dict[str, np.ndarray]]:
        metrics = tuple(str(value) for value in metric_names)
        self.validate_complete(rollouts=rollouts, metric_names=metrics)
        result: dict[Profile, dict[str, np.ndarray]] = {}
        for start, stop in self.store.bounds():
            values = self.store.read(start, stop)
            for offset, profile in enumerate(self.profiles[start:stop]):
                result[profile] = {
                    name: np.asarray(values[f"metric__{name}"][offset], dtype=float)
                    for name in metrics
                }
        return result

    def invalid_bounds(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            bounds
            for bounds in self.store.bounds()
            if self.store.status(*bounds) is not ChunkStatus.VALID
        )
