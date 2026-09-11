from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cmfg_cce.evaluation.backend_cache import BackendPayoffCache
from cmfg_cce.evaluation.profile_backend import ProfileEvaluationBackend
from cmfg_cce.evaluation.rollout import Profile, RolloutEstimate
from cmfg_cce.orchestration.chunks import ChunkPlan, ChunkStatus, ImmutableChunkStore
from cmfg_cce.orchestration.manifest import atomic_json, canonical_json, sha256_bytes


TRAINING_CHUNK_SIZE = 128
ASSIGNMENT_SCHEMA_VERSION = "revision_full_v1_adaptive_training_assignments_v1"


def _evaluate_training_profile(
    args: tuple[int, Profile, ProfileEvaluationBackend, int],
) -> tuple[int, Profile, np.ndarray, np.ndarray]:
    index, profile, backend, rollouts = args
    returns = np.empty((int(rollouts), int(backend.n_agents)), dtype=np.float64)
    objectives = np.empty(int(rollouts), dtype=np.float64)
    for replication in range(int(rollouts)):
        episode_returns, metrics = backend.run_episode(profile, replication)
        values = np.asarray(episode_returns, dtype=np.float64)
        objective = float(metrics.get("platform_operating_score", float("nan")))
        if values.shape != (int(backend.n_agents),) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Training profile {profile!r}, replication {replication} returned "
                "invalid manufacturer payoffs."
            )
        if not np.isfinite(objective):
            raise ValueError(
                f"Training profile {profile!r}, replication {replication} has no "
                "finite platform_operating_score."
            )
        returns[replication] = values
        objectives[replication] = objective
    return int(index), tuple(profile), returns, objectives


def _estimate_from_vectors(
    profile: Profile,
    returns: np.ndarray,
    objectives: np.ndarray,
) -> RolloutEstimate:
    values = np.asarray(returns, dtype=float)
    objective_values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Training return vectors must have shape (R, N) with R >= 2.")
    if objective_values.shape != (values.shape[0],):
        raise ValueError("Training objective vector does not align with returns.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(objective_values)):
        raise ValueError("Training replication vectors must be finite.")
    variance = np.var(values, axis=0, ddof=1)
    return RolloutEstimate(
        profile=tuple(profile),
        n_rollouts=int(values.shape[0]),
        mean_returns=np.mean(values, axis=0),
        var_returns=variance,
        ci_radius=1.96 * np.sqrt(variance / values.shape[0]),
        mean_metrics={"platform_operating_score": float(np.mean(objective_values))},
        var_metrics={"platform_operating_score": float(np.var(objective_values, ddof=1))},
    )


def _mapping_sha256(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.ascontiguousarray(np.asarray(values[name]))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class AdaptiveTrainingChunks:
    """Crash-safe raw training vectors for an adaptive sparse solver.

    The immutable plan contains one slot for every possible joint profile, but
    simulation is performed only for profiles requested by the deterministic
    sparse search.  The append-only assignment checkpoint maps those slots to
    profiles before simulation starts.  Complete 128-profile blocks are then
    immutable.  On restart, valid blocks are loaded, while missing or corrupt
    assigned blocks are recomputed from the frozen assignment list.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        campaign_sha256: str,
        matrix_sha256: str,
        job_id: str,
        policy_ids: Sequence[str],
        n_agents: int,
        rollouts: int,
        backend_identity: Mapping[str, Any],
    ) -> None:
        self.root = Path(root)
        self.campaign_sha256 = str(campaign_sha256)
        self.matrix_sha256 = str(matrix_sha256)
        self.job_id = str(job_id)
        self.policy_ids = tuple(str(value) for value in policy_ids)
        self.n_agents = int(n_agents)
        self.rollouts = int(rollouts)
        if self.n_agents <= 0 or self.rollouts < 2 or not self.policy_ids:
            raise ValueError("Adaptive training dimensions are invalid.")
        self.profile_capacity = len(self.policy_ids) ** self.n_agents
        self.policy_index = {policy: index for index, policy in enumerate(self.policy_ids)}
        self.backend_identity = dict(backend_identity)
        self.backend_identity_sha256 = sha256_bytes(canonical_json(self.backend_identity))
        self.root.mkdir(parents=True, exist_ok=True)

        store_job_id = f"{self.job_id}::training"
        self.plan = ChunkPlan.create(
            campaign_sha256=self.campaign_sha256,
            matrix_sha256=self.matrix_sha256,
            job_id=store_job_id,
            kind="training",
            chunk_size=TRAINING_CHUNK_SIZE,
            item_ids=tuple(
                f"adaptive-training-slot-{index:08d}"
                for index in range(self.profile_capacity)
            ),
        )
        self.plan.write_immutable(self.root / "chunk_plan.json")
        self.store = ImmutableChunkStore(
            self.root,
            campaign_sha256=self.campaign_sha256,
            matrix_sha256=self.matrix_sha256,
            job_id=store_job_id,
            kind="training",
            item_count=self.profile_capacity,
            chunk_size=TRAINING_CHUNK_SIZE,
        )
        self.assignment_path = self.root / "assignment_checkpoint.json"
        self.assignments, self.finalized = self._load_or_initialize_assignments()
        self._profile_to_slot = {
            profile: index
            for index, profile in enumerate(self.assignments)
            if profile is not None
        }
        if len(self._profile_to_slot) != sum(
            profile is not None for profile in self.assignments
        ):
            raise ValueError("Adaptive training assignment checkpoint repeats a profile.")
        self._vectors: dict[Profile, tuple[np.ndarray, np.ndarray]] = {}

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": ASSIGNMENT_SCHEMA_VERSION,
            "campaign_sha256": self.campaign_sha256,
            "matrix_sha256": self.matrix_sha256,
            "job_id": self.job_id,
            "item_order_sha256": self.plan.item_order_sha256,
            "backend_identity_sha256": self.backend_identity_sha256,
            "policy_ids": list(self.policy_ids),
            "n_agents": self.n_agents,
            "rollouts": self.rollouts,
            "profile_capacity": self.profile_capacity,
        }

    def _load_or_initialize_assignments(self) -> tuple[list[Profile | None], bool]:
        if not self.assignment_path.exists():
            payload = {
                **self._identity_payload(),
                "assignments": [],
                "finalized": False,
            }
            atomic_json(self.assignment_path, payload)
            return [], False
        payload = json.loads(self.assignment_path.read_text(encoding="utf-8"))
        expected = self._identity_payload()
        observed = {name: payload.get(name) for name in expected}
        if observed != expected:
            raise ValueError("Adaptive training assignment identity mismatch.")
        raw_assignments = list(payload.get("assignments", ()))
        if len(raw_assignments) > self.profile_capacity:
            raise ValueError("Adaptive training assignment list exceeds the policy space.")
        assignments: list[Profile | None] = []
        padding_started = False
        allowed = set(self.policy_ids)
        for raw in raw_assignments:
            if raw is None:
                padding_started = True
                assignments.append(None)
                continue
            if padding_started:
                raise ValueError("Adaptive training assignments contain a profile after padding.")
            profile = tuple(str(value) for value in raw)
            if len(profile) != self.n_agents or set(profile).difference(allowed):
                raise ValueError("Adaptive training assignment contains an invalid profile.")
            assignments.append(profile)
        finalized = bool(payload.get("finalized", False))
        if finalized and len(assignments) != self.profile_capacity:
            raise ValueError("A finalized adaptive training plan must fill every slot.")
        if not finalized and any(profile is None for profile in assignments):
            raise ValueError("An unfinished adaptive training plan cannot contain padding.")
        return assignments, finalized

    def _write_assignments(self) -> None:
        atomic_json(
            self.assignment_path,
            {
                **self._identity_payload(),
                "assignments": [list(profile) if profile is not None else None for profile in self.assignments],
                "finalized": bool(self.finalized),
                "assignment_sha256": sha256_bytes(
                    canonical_json(
                        [list(profile) if profile is not None else None for profile in self.assignments]
                    )
                ),
            },
        )

    @staticmethod
    def _worker_count(configured: int, tasks: int) -> int:
        return min(max(1, int(configured)), max(1, os.cpu_count() or 1), max(1, tasks))

    def _evaluate(
        self,
        profiles: Sequence[Profile],
        backend: ProfileEvaluationBackend,
        workers: int,
    ) -> dict[Profile, tuple[np.ndarray, np.ndarray]]:
        ordered = tuple(tuple(profile) for profile in profiles)
        if not ordered:
            return {}
        tasks = [
            (self._profile_to_slot[profile], profile, backend, self.rollouts)
            for profile in ordered
        ]
        worker_count = self._worker_count(workers, len(tasks))
        if worker_count == 1:
            rows = [_evaluate_training_profile(task) for task in tasks]
        else:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context("spawn"),
            ) as executor:
                rows = list(executor.map(_evaluate_training_profile, tasks, chunksize=1))
        rows.sort(key=lambda row: row[0])
        return {
            profile: (np.asarray(returns, dtype=float), np.asarray(objectives, dtype=float))
            for _index, profile, returns, objectives in rows
        }

    def _chunk_arrays(self, start: int, stop: int) -> dict[str, np.ndarray] | None:
        assigned = self.assignments[start:stop]
        if len(assigned) != stop - start:
            return None
        if not self.finalized and stop > len(self.assignments):
            return None
        actual = [profile for profile in assigned if profile is not None]
        if any(profile not in self._vectors for profile in actual):
            return None
        width = stop - start
        profile_indices = np.full((width, self.n_agents), -1, dtype=np.int16)
        returns = np.full(
            (width, self.rollouts, self.n_agents), np.nan, dtype=np.float64
        )
        objectives = np.full((width, self.rollouts), np.nan, dtype=np.float64)
        evaluated = np.zeros(width, dtype=np.uint8)
        for offset, profile in enumerate(assigned):
            if profile is None:
                continue
            evaluated[offset] = 1
            profile_indices[offset] = [self.policy_index[policy] for policy in profile]
            returns[offset], objectives[offset] = self._vectors[profile]
        return {
            "slot_indices": np.arange(start, stop, dtype=np.int64),
            "profile_policy_indices": profile_indices,
            "returns": returns,
            "platform_operating_score": objectives,
            "is_evaluated": evaluated,
            "rollout_count": np.asarray([self.rollouts], dtype=np.int32),
            "backend_identity_sha256": np.frombuffer(
                bytes.fromhex(self.backend_identity_sha256), dtype=np.uint8
            ),
        }

    def _validate_chunk_arrays(
        self, start: int, stop: int, arrays: Mapping[str, np.ndarray]
    ) -> dict[Profile, tuple[np.ndarray, np.ndarray]]:
        width = stop - start
        expected = {
            "slot_indices": (width,),
            "profile_policy_indices": (width, self.n_agents),
            "returns": (width, self.rollouts, self.n_agents),
            "platform_operating_score": (width, self.rollouts),
            "is_evaluated": (width,),
            "rollout_count": (1,),
            "backend_identity_sha256": (32,),
        }
        if {name: np.asarray(arrays.get(name, np.empty(0))).shape for name in expected} != expected:
            raise ValueError(f"Adaptive training chunk [{start},{stop}) has invalid shapes.")
        np.testing.assert_array_equal(
            arrays["slot_indices"], np.arange(start, stop, dtype=np.int64)
        )
        if arrays["rollout_count"].tolist() != [self.rollouts]:
            raise ValueError("Adaptive training chunk changed its rollout count.")
        if bytes(np.asarray(arrays["backend_identity_sha256"], dtype=np.uint8)) != bytes.fromhex(
            self.backend_identity_sha256
        ):
            raise ValueError("Adaptive training chunk backend identity mismatch.")
        restored: dict[Profile, tuple[np.ndarray, np.ndarray]] = {}
        assigned = self.assignments[start:stop]
        for offset, profile in enumerate(assigned):
            flag = int(arrays["is_evaluated"][offset])
            if profile is None:
                if flag != 0 or np.any(arrays["profile_policy_indices"][offset] != -1):
                    raise ValueError("Adaptive training padding slot contains profile data.")
                continue
            if flag != 1:
                raise ValueError("Adaptive training assigned profile is missing from its chunk.")
            expected_indices = [self.policy_index[policy] for policy in profile]
            np.testing.assert_array_equal(
                arrays["profile_policy_indices"][offset], expected_indices
            )
            returns = np.asarray(arrays["returns"][offset], dtype=float)
            objectives = np.asarray(arrays["platform_operating_score"][offset], dtype=float)
            if not np.all(np.isfinite(returns)) or not np.all(np.isfinite(objectives)):
                raise ValueError("Adaptive training chunk contains non-finite profile vectors.")
            restored[profile] = (returns, objectives)
        return restored

    def restore_or_repair(
        self,
        backend: ProfileEvaluationBackend,
        *,
        workers: int,
    ) -> tuple[dict[Profile, tuple[np.ndarray, np.ndarray]], int]:
        if dict(backend.cache_identity) != self.backend_identity:
            raise ValueError("Training backend does not match the checkpoint identity.")
        simulated = 0
        for start, stop in self.store.bounds():
            assigned_count = min(max(0, len(self.assignments) - start), stop - start)
            if assigned_count <= 0:
                break
            assigned = self.assignments[start : start + assigned_count]
            is_complete_assignment = assigned_count == stop - start
            if self.store.status(start, stop) is ChunkStatus.VALID:
                if not is_complete_assignment:
                    raise ValueError("A valid training chunk extends beyond frozen assignments.")
                arrays = self.store.read(start, stop)
                self._vectors.update(self._validate_chunk_arrays(start, stop, arrays))
                continue
            actual = [profile for profile in assigned if profile is not None]
            if actual:
                evaluated = self._evaluate(actual, backend, workers)
                self._vectors.update(evaluated)
                simulated += len(actual)
            if is_complete_assignment:
                arrays = self._chunk_arrays(start, stop)
                if arrays is None:
                    raise RuntimeError("Assigned training chunk could not be reconstructed.")
                self.store.commit(start, stop, arrays)
        return dict(self._vectors), simulated

    def assign(self, profiles: Sequence[Profile]) -> tuple[Profile, ...]:
        normalized = tuple(sorted(set(tuple(profile) for profile in profiles)))
        unknown = [
            profile
            for profile in normalized
            if len(profile) != self.n_agents or set(profile).difference(self.policy_index)
        ]
        if unknown:
            raise ValueError(f"Adaptive training requested invalid profiles: {unknown[:3]}")
        missing = tuple(profile for profile in normalized if profile not in self._profile_to_slot)
        if missing and self.finalized:
            raise RuntimeError("Finalized adaptive training plan received a new profile request.")
        if len(self._profile_to_slot) + len(missing) > self.profile_capacity:
            raise RuntimeError("Adaptive training search exceeded the joint-policy space.")
        for profile in missing:
            slot = len(self.assignments)
            self.assignments.append(profile)
            self._profile_to_slot[profile] = slot
        if missing:
            self._write_assignments()
        return missing

    def record(
        self,
        values: Mapping[Profile, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        for profile, (returns, objectives) in values.items():
            if profile not in self._profile_to_slot:
                raise KeyError(f"Training vectors were not assigned a slot: {profile}")
            _estimate_from_vectors(profile, returns, objectives)
            self._vectors[profile] = (
                np.asarray(returns, dtype=float),
                np.asarray(objectives, dtype=float),
            )
        for start, stop in self.store.bounds():
            if stop > len(self.assignments) or self.store.status(start, stop) is ChunkStatus.VALID:
                continue
            arrays = self._chunk_arrays(start, stop)
            if arrays is not None:
                self.store.commit(start, stop, arrays)

    def finalize(self) -> None:
        if not self.finalized:
            self.assignments.extend([None] * (self.profile_capacity - len(self.assignments)))
            self.finalized = True
            self._write_assignments()
        for start, stop in self.store.bounds():
            if self.store.status(start, stop) is ChunkStatus.VALID:
                self._validate_chunk_arrays(start, stop, self.store.read(start, stop))
                continue
            arrays = self._chunk_arrays(start, stop)
            if arrays is None:
                raise RuntimeError("Cannot finalize an incomplete adaptive training chunk.")
            self.store.commit(start, stop, arrays)
        self.store.validate_complete()

    @property
    def evaluated_profiles(self) -> tuple[Profile, ...]:
        return tuple(profile for profile in self.assignments if profile is not None)

    def vector_hash(self) -> str:
        ordered = self.evaluated_profiles
        if any(profile not in self._vectors for profile in ordered):
            raise RuntimeError("Training vector hash requested before all assigned profiles exist.")
        profile_indices = np.asarray(
            [[self.policy_index[policy] for policy in profile] for profile in ordered],
            dtype=np.int16,
        )
        return _mapping_sha256(
            {
                "profile_indices": profile_indices,
                "returns": np.stack([self._vectors[profile][0] for profile in ordered]),
                "platform_operating_score": np.stack(
                    [self._vectors[profile][1] for profile in ordered]
                ),
            }
        )

    def checkpoint_payload(self) -> dict[str, Any]:
        assignment_bytes = self.assignment_path.read_bytes()
        return {
            "chunk_size": TRAINING_CHUNK_SIZE,
            "profile_capacity": self.profile_capacity,
            "evaluated_profile_count": len(self.evaluated_profiles),
            "item_order_sha256": self.plan.item_order_sha256,
            "assignment_checkpoint_sha256": sha256_bytes(assignment_bytes),
            "backend_identity_sha256": self.backend_identity_sha256,
            "chunks": [record.to_payload() for record in self.store.validate_complete()],
        }


class ResumableTrainingPayoffCache(BackendPayoffCache):
    """Backend payoff cache whose raw R-vectors are persisted in 128-profile blocks."""

    def __init__(
        self,
        *,
        backend: ProfileEvaluationBackend,
        n_rollouts: int,
        workers: int,
        chunks: AdaptiveTrainingChunks,
    ) -> None:
        super().__init__(backend=backend, n_rollouts=int(n_rollouts), workers=int(workers))
        if int(n_rollouts) != chunks.rollouts:
            raise ValueError("Training cache and chunk checkpoint rollout counts differ.")
        self.chunks = chunks
        started = time.perf_counter()
        restored, simulated = chunks.restore_or_repair(backend, workers=workers)
        for profile, (returns, objectives) in restored.items():
            self.estimates[profile] = _estimate_from_vectors(profile, returns, objectives)
        elapsed = time.perf_counter() - started
        self.eval_time_seconds += elapsed if simulated else 0.0
        self.eval_rollout_episode_count += int(simulated) * int(n_rollouts)

    def _estimate_profiles_parallel(self, profiles: list[Profile], n_rollouts: int) -> None:
        unique = tuple(sorted(set(tuple(profile) for profile in profiles)))
        missing = tuple(profile for profile in unique if profile not in self.estimates)
        if not missing:
            return
        if int(n_rollouts) != self.n_rollouts:
            raise ValueError("revision-full-v1 sparse training uses one frozen R_train value.")
        self.chunks.assign(missing)
        started = time.perf_counter()
        evaluated = self.chunks._evaluate(missing, self.backend, self.workers)
        elapsed = time.perf_counter() - started
        self.chunks.record(evaluated)
        for profile, (returns, objectives) in evaluated.items():
            self.estimates[profile] = _estimate_from_vectors(profile, returns, objectives)
        self.eval_time_seconds += elapsed
        self.eval_rollout_episode_count += len(evaluated) * self.n_rollouts

    def finalize_training(self) -> None:
        self.chunks.finalize()

    @property
    def raw_training_vectors_sha256(self) -> str:
        return self.chunks.vector_hash()
