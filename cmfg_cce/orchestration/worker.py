from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence
import uuid

from cmfg_cce.orchestration.chunks import (
    ChunkPlan,
    ChunkRecord,
    ChunkStatus,
    ImmutableChunkStore,
)
from cmfg_cce.orchestration.manifest import (
    CampaignManifest,
    JobSpec,
    ManifestError,
    SourceManifest,
    atomic_json,
    canonical_json,
    sha256_bytes,
    sha256_file,
)
from cmfg_cce.orchestration.storage import (
    GCloudObjectStore,
    GoogleCloudStorageStore,
    LocalMirrorStore,
    RemoteStore,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ResourceGate:
    """Allow concurrent bulk jobs while giving runtime measurements the whole node."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active_bulk = 0
        self._exclusive_active = False
        self._exclusive_waiting = 0

    @contextmanager
    def acquire(self, mode: str):
        if mode not in {"bulk", "exclusive_runtime"}:
            raise ValueError(f"Unknown resource mode: {mode!r}")
        if mode == "bulk":
            with self._condition:
                self._condition.wait_for(
                    lambda: not self._exclusive_active and self._exclusive_waiting == 0
                )
                self._active_bulk += 1
            try:
                yield
            finally:
                with self._condition:
                    self._active_bulk -= 1
                    self._condition.notify_all()
            return

        with self._condition:
            self._exclusive_waiting += 1
            try:
                self._condition.wait_for(
                    lambda: not self._exclusive_active and self._active_bulk == 0
                )
                self._exclusive_active = True
            finally:
                self._exclusive_waiting -= 1
        try:
            yield
        finally:
            with self._condition:
                self._exclusive_active = False
                self._condition.notify_all()


@dataclass
class RuntimeLedger:
    state_path: Path
    remote: RemoteStore | None
    remote_key: str
    session_id: str
    soft_limit_hours: float

    def __post_init__(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if self.remote is not None:
            try:
                self.remote.download(self.remote_key, self.state_path)
            except Exception:
                if not self.state_path.exists():
                    raise
        accumulated = 0.0
        if self.state_path.exists():
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            accumulated = float(payload.get("accumulated_seconds", 0.0))
            if accumulated < 0:
                raise ValueError("Spot runtime ledger has negative accumulated time.")
        self.base_seconds = accumulated
        self.started = time.monotonic()

    @property
    def accumulated_seconds(self) -> float:
        return self.base_seconds + max(0.0, time.monotonic() - self.started)

    @property
    def exhausted(self) -> bool:
        return self.accumulated_seconds >= 3600.0 * float(self.soft_limit_hours)

    def checkpoint(self) -> dict[str, Any]:
        payload = {
            "schema_version": "revision_full_v1_spot_runtime_v1",
            "session_id": self.session_id,
            "accumulated_seconds": self.accumulated_seconds,
            "soft_limit_hours": float(self.soft_limit_hours),
            "updated_utc": utc_now(),
            "exhausted": self.exhausted,
        }
        atomic_json(self.state_path, payload)
        if self.remote is not None:
            self.remote.upload_mutable(self.state_path, self.remote_key)
        return payload


class CampaignWorker:
    def __init__(
        self,
        manifest: CampaignManifest,
        *,
        source_manifest: SourceManifest,
        source_root: Path,
        node_id: str,
        state_dir: Path,
        slots: int,
        workers_per_job: int,
        remote: RemoteStore | None = None,
        heartbeat_seconds: float = 60.0,
        spot_accounting: bool = False,
        scale_to_zero_command: Sequence[str] = (),
        session_id: str | None = None,
        dependency_wait_seconds: float = 86_400.0,
        dependency_poll_seconds: float = 30.0,
        include_families: Sequence[str] = (),
        exclude_families: Sequence[str] = (),
        prune_completed_local: bool = False,
    ) -> None:
        manifest.validate()
        source_manifest.validate_tree(source_root)
        if source_manifest.source_sha256 != manifest.source_sha256:
            raise ManifestError("Campaign and source-manifest hashes differ.")
        if slots <= 0 or workers_per_job <= 0:
            raise ValueError("slots and workers_per_job must be positive.")
        self.manifest = manifest
        self.source_manifest = source_manifest
        self.source_root = source_root
        self.node_id = str(node_id)
        self.state_dir = Path(state_dir)
        self.slots = int(slots)
        self.workers_per_job = int(workers_per_job)
        self.remote = remote
        self.prune_completed_local = bool(prune_completed_local)
        if self.prune_completed_local and self.remote is None:
            raise ValueError("Completed local state may only be pruned when a remote store is configured.")
        self.heartbeat_seconds = max(1.0, float(heartbeat_seconds))
        self.scale_to_zero_command = tuple(scale_to_zero_command)
        self.session_id = session_id or os.environ.get("CMFG_INSTANCE_ID") or uuid.uuid4().hex
        self.dependency_wait_seconds = max(0.0, float(dependency_wait_seconds))
        self.dependency_poll_seconds = max(0.1, float(dependency_poll_seconds))
        included = {str(value) for value in include_families if str(value)}
        excluded = {str(value) for value in exclude_families if str(value)}
        if included.intersection(excluded):
            raise ValueError("A family cannot be both included and excluded.")
        self.include_families = tuple(sorted(included))
        self.exclude_families = tuple(sorted(excluded))
        self.node_jobs = tuple(
            job
            for job in self.manifest.jobs_for(self.node_id)
            if (not included or job.family in included) and job.family not in excluded
        )
        if not self.node_jobs:
            raise ManifestError(f"Campaign has no jobs assigned to node {self.node_id!r}.")
        if any(job.slot >= self.slots for job in self.node_jobs):
            raise ManifestError(f"Node {self.node_id} has a job assigned outside {self.slots} slots.")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir = self.state_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        self.active: dict[int, str] = {}
        self.active_lock = threading.Lock()
        # Heartbeats may run while a runner atomically commits a new chunk or
        # sidecar.  Serialize remote synchronization and remember what has
        # already been uploaded so a 30-second heartbeat never re-hashes every
        # historical NPZ in a large profile table.
        self.runner_sync_lock = threading.RLock()
        self._uploaded_runner_plans: dict[tuple[str, str], str] = {}
        self._uploaded_runner_chunks: dict[tuple[str, str, str], tuple[str, str]] = {}
        self._uploaded_runner_sidecars: dict[tuple[str, str], tuple[int, int, str]] = {}
        self.resource_gate = ResourceGate()
        self.completed: set[str] = set()
        self.failures: dict[str, str] = {}
        self.ledger = (
            RuntimeLedger(
                self.state_dir / "control" / "spot_runtime.json",
                self.remote,
                f"control/{self.node_id}/spot_runtime.json",
                self.session_id,
                self.manifest.spot_vm_hour_soft_limit,
            )
            if spot_accounting
            else None
        )

    def _remote_key(self, job: JobSpec, relative: str) -> str:
        return f"jobs/{job.job_id}/{relative.replace(os.sep, '/')}"

    def _completion_path(self, job: JobSpec) -> Path:
        return self.jobs_dir / job.job_id / "job_complete.json"

    def _validate_completion(self, job: JobSpec, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "campaign_sha256": self.manifest.campaign_sha256,
            "source_sha256": self.manifest.source_sha256,
            "matrix_sha256": self.manifest.matrix_sha256,
            "config_sha256": job.config_sha256,
            "job_id": job.job_id,
        }
        observed = {key: payload.get(key) for key in expected}
        if observed != expected or payload.get("status") != "complete":
            raise ManifestError(
                f"Completion marker identity mismatch for {job.job_id}: {observed!r}"
            )
        return payload

    def _restore_completion(self, job: JobSpec) -> bool:
        path = self._completion_path(job)
        if path.exists():
            self._validate_completion(job, path)
            if self.prune_completed_local:
                self._prune_completed_job(job)
            return True
        if self.remote is None:
            return False
        if self.remote.download(self._remote_key(job, "job_complete.json"), path):
            self._validate_completion(job, path)
            if self.prune_completed_local:
                self._prune_completed_job(job)
            return True
        return False

    def _prune_completed_job(self, job: JobSpec) -> None:
        """Remove a completed job only after its immutable remote marker exists."""

        if not self.prune_completed_local:
            return
        if self.remote is None or not self.remote.exists(
            self._remote_key(job, "job_complete.json")
        ):
            raise RuntimeError(
                f"Refusing to prune {job.job_id}: its remote completion marker is missing."
            )
        job_dir = self.jobs_dir / job.job_id
        with self.runner_sync_lock:
            shutil.rmtree(job_dir, ignore_errors=False)
            self._uploaded_runner_plans = {
                key: value
                for key, value in self._uploaded_runner_plans.items()
                if key[0] != job.job_id
            }
            self._uploaded_runner_chunks = {
                key: value
                for key, value in self._uploaded_runner_chunks.items()
                if key[0] != job.job_id
            }
            self._uploaded_runner_sidecars = {
                key: value
                for key, value in self._uploaded_runner_sidecars.items()
                if key[0] != job.job_id
            }

    def _dependency_root(self, job: JobSpec) -> Path:
        return self.jobs_dir / job.job_id / "dependencies"

    def _restore_dependency(self, job: JobSpec, dependency_id: str) -> Path:
        dependency = next(
            item for item in self.manifest.jobs if item.job_id == dependency_id
        )
        target_root = self._dependency_root(job) / dependency_id
        marker = target_root / "job_complete.json"
        deadline = time.monotonic() + self.dependency_wait_seconds
        while True:
            if marker.exists():
                break
            local_marker = self._completion_path(dependency)
            if local_marker.is_file():
                target_root.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local_marker, marker)
                break
            if self.remote is not None and self.remote.download(
                self._remote_key(dependency, "job_complete.json"), marker
            ):
                break
            if self.stop_event.is_set():
                raise InterruptedError(
                    f"Worker stopped while waiting for dependency {dependency_id}."
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Dependency {dependency_id} for {job.job_id} did not complete within "
                    f"{self.dependency_wait_seconds:g} seconds."
                )
            self.stop_event.wait(
                min(self.dependency_poll_seconds, max(0.0, deadline - time.monotonic()))
            )

        completion = self._validate_completion(dependency, marker)
        artifacts_root = target_root / "artifacts"
        for record in completion.get("artifacts", []):
            relative = str(record.get("path", ""))
            expected = str(record.get("sha256", ""))
            if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ManifestError(
                    f"Dependency {dependency_id} declares an unsafe artifact {relative!r}."
                )
            target = artifacts_root / relative
            if target.is_file() and sha256_file(target) == expected:
                continue
            local_source = self.jobs_dir / dependency_id / "output" / relative
            if local_source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local_source, target)
            elif self.remote is None or not self.remote.download(
                self._remote_key(dependency, f"artifacts/{relative}"),
                target,
                expected_sha256=expected,
            ):
                raise RuntimeError(
                    f"Completed dependency {dependency_id} is missing artifact {relative}."
                )
            if sha256_file(target) != expected:
                raise RuntimeError(
                    f"Dependency artifact checksum mismatch: {dependency_id}/{relative}."
                )
        return target_root

    def _prepare_dependencies(self, job: JobSpec) -> None:
        for dependency_id in job.depends_on:
            self._restore_dependency(job, dependency_id)

    def _validate_frozen_mwu_selection(self, job: JobSpec) -> None:
        frozen = job.metadata.get("mwu_selection")
        if frozen is None:
            return
        if not isinstance(frozen, Mapping):
            raise ManifestError(f"Job {job.job_id} has invalid MWU-selection metadata.")
        configured = os.environ.get("CMFG_MWU_SELECTION_PATH", "")
        if not configured:
            raise ManifestError(
                f"Formal job {job.job_id} requires CMFG_MWU_SELECTION_PATH."
            )
        path = Path(configured)
        if not path.is_file():
            raise ManifestError(f"Frozen MWU-selection file is missing: {path}.")
        expected_file_hash = str(frozen.get("selection_file_sha256", ""))
        if sha256_file(path) != expected_file_hash:
            raise ManifestError("MWU-selection file hash differs from the formal manifest.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "revision_full_v1_global_mwu_selection_v2":
            raise ManifestError("Formal worker requires the hyperparameter-only MWU selection schema.")
        unsigned = dict(payload)
        observed_hash = str(unsigned.pop("selection_sha256", ""))
        if observed_hash != sha256_bytes(canonical_json(unsigned)):
            raise ManifestError("MWU-selection signed payload hash is invalid.")
        selected = dict(payload.get("selected_config", {}))
        if "formal_rounds" in selected or "formal_rounds_rule" in selected:
            raise ManifestError("Formal MWU selection must not contain a global round budget.")
        expected = {
            "selection_sha256": observed_hash,
            "selection_file_sha256": expected_file_hash,
            "mwu_config_id": str(selected.get("mwu_config_id", "")),
        }
        observed = {name: frozen.get(name) for name in expected}
        if observed != expected or payload.get("matrix_sha256") != self.manifest.matrix_sha256:
            raise ManifestError(
                f"MWU selection does not match the frozen formal job {job.job_id}."
            )

    def _chunk_store(
        self,
        job: JobSpec,
        *,
        item_count: int | None = None,
        root_prefix: str = "chunks",
        kind: str | None = None,
        chunk_size: int | None = None,
        store_job_id: str | None = None,
    ) -> ImmutableChunkStore:
        effective_kind = job.chunk_kind if kind is None else kind
        effective_size = job.chunk_size if chunk_size is None else int(chunk_size)
        if effective_kind is None or effective_size is None:
            raise ValueError(f"Job {job.job_id} is not chunked.")
        return ImmutableChunkStore(
            self.jobs_dir / job.job_id / root_prefix,
            campaign_sha256=self.manifest.campaign_sha256,
            matrix_sha256=self.manifest.matrix_sha256,
            job_id=job.job_id if store_job_id is None else store_job_id,
            kind=effective_kind,
            item_count=job.item_count if item_count is None else int(item_count),
            chunk_size=effective_size,
        )

    def _restore_chunk(
        self,
        job: JobSpec,
        store: ImmutableChunkStore,
        start: int,
        stop: int,
        *,
        remote_chunk_prefix: str = "chunks",
    ) -> None:
        if store.status(start, stop) is ChunkStatus.VALID or self.remote is None:
            return
        payload_path, metadata_path = store.paths(start, stop)
        prefix = f"{remote_chunk_prefix}/{metadata_path.stem}"
        downloaded_metadata = self.remote.download(
            self._remote_key(job, prefix + ".json"), metadata_path
        )
        if not downloaded_metadata:
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            digest = str(metadata["payload_sha256"])
        except (KeyError, ValueError, json.JSONDecodeError):
            return
        downloaded_payload = self.remote.download(
            self._remote_key(job, prefix + ".npz"),
            payload_path,
            expected_sha256=digest,
        )
        if downloaded_payload and store.status(start, stop) is ChunkStatus.VALID:
            self._uploaded_runner_chunks[
                (job.job_id, remote_chunk_prefix, metadata_path.stem)
            ] = (digest, sha256_file(metadata_path))

    def _upload_chunk(
        self,
        job: JobSpec,
        store: ImmutableChunkStore,
        start: int,
        stop: int,
        *,
        remote_chunk_prefix: str = "chunks",
    ) -> None:
        if self.remote is None:
            return
        payload_path, metadata_path = store.paths(start, stop)
        if store.status(start, stop) is not ChunkStatus.VALID:
            raise RuntimeError(f"Refusing to upload invalid chunk for {job.job_id} [{start},{stop}).")
        prefix = f"{remote_chunk_prefix}/{metadata_path.stem}"
        self.remote.upload_immutable(payload_path, self._remote_key(job, prefix + ".npz"))
        self.remote.upload_immutable(metadata_path, self._remote_key(job, prefix + ".json"))

    def _runner_stage_specs(
        self, job: JobSpec
    ) -> tuple[tuple[str | None, str, int, str], ...]:
        raw_stages = tuple(str(value) for value in job.metadata.get("stages", ()))
        if not raw_stages:
            if job.chunk_kind is None or job.chunk_size is None:
                return ()
            return ((None, job.chunk_kind, int(job.chunk_size), "chunks"),)
        stage_kinds = {
            "training": ("training", 128),
            "formal_audit": ("audit", 8),
            "outcome_evaluation": ("audit", 8),
            "mwu_tuning_audit": ("audit", 8),
            "transplant_audit": ("audit", 8),
        }
        specs: list[tuple[str | None, str, int, str]] = []
        for stage in raw_stages:
            if stage == "runtime_measurement":
                # Runtime attempts are deliberately atomic and restart from an
                # empty cache; they must not be stitched from profile chunks.
                continue
            if stage not in stage_kinds or "/" in stage or "\\" in stage or stage == "..":
                raise ManifestError(f"Job {job.job_id} has an unsupported stage {stage!r}.")
            kind, size = stage_kinds[stage]
            specs.append((stage, kind, size, f"stages/{stage}/chunks"))
        return tuple(specs)

    def _runner_plan_path(self, job: JobSpec, root_prefix: str = "chunks") -> Path:
        return self.jobs_dir / job.job_id / root_prefix / "chunk_plan.json"

    def _runner_sidecar_relatives(self, job: JobSpec) -> tuple[str, ...]:
        values = tuple(str(value) for value in job.metadata.get("runner_sidecars", ()))
        normalized: list[str] = []
        for value in values:
            path = Path(value)
            if (
                not value
                or path.is_absolute()
                or ".." in path.parts
                or value.endswith("/")
                or value.endswith("\\")
            ):
                raise ManifestError(
                    f"Job {job.job_id} declares an unsafe runner sidecar {value!r}."
                )
            relative = path.as_posix()
            if relative not in normalized:
                normalized.append(relative)
        return tuple(normalized)

    def _sidecar_pointer_path(self, job: JobSpec, relative: str) -> Path:
        return (
            self.state_dir
            / "control"
            / "sidecars"
            / job.job_id
            / f"{relative}.pointer.json"
        )

    def _sidecar_pointer_key(self, job: JobSpec, relative: str) -> str:
        return self._remote_key(job, f"sidecars/pointers/{relative}.pointer.json")

    def _validate_sidecar_payload(self, job: JobSpec, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestError(
                f"Runner sidecar for {job.job_id} is not valid JSON: {path}."
            ) from exc
        expected = {
            "campaign_sha256": self.manifest.campaign_sha256,
            "matrix_sha256": self.manifest.matrix_sha256,
            "job_id": job.job_id,
        }
        observed = {name: payload.get(name) for name in expected}
        if observed != expected:
            raise ManifestError(
                f"Runner sidecar identity mismatch for {job.job_id}: {observed!r}."
            )
        return payload

    def _validate_sidecar_pointer(
        self,
        job: JobSpec,
        relative: str,
        path: Path,
    ) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestError(
                f"Runner sidecar pointer for {job.job_id} is invalid: {relative}."
            ) from exc
        expected = {
            "schema_version": "revision_full_v1_sidecar_pointer_v1",
            "campaign_sha256": self.manifest.campaign_sha256,
            "source_sha256": self.manifest.source_sha256,
            "matrix_sha256": self.manifest.matrix_sha256,
            "config_sha256": job.config_sha256,
            "job_id": job.job_id,
            "relative_path": relative,
        }
        observed = {name: payload.get(name) for name in expected}
        digest = str(payload.get("sha256", ""))
        size = payload.get("size")
        if (
            observed != expected
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ManifestError(
                f"Runner sidecar pointer identity mismatch for {job.job_id}: {relative}."
            )
        return payload

    def _restore_runner_sidecars(self, job: JobSpec) -> None:
        """Restore mutable runner checkpoints via hash-addressed immutable objects.

        The pointer is mutable, but the referenced object is immutable and its
        SHA256 is checked on download.  A persistent local checkpoint is kept
        when valid because it can be newer than the last heartbeat uploaded
        immediately before a host shutdown.
        """

        if self.remote is None:
            return
        job_root = self.jobs_dir / job.job_id
        for relative in self._runner_sidecar_relatives(job):
            local = job_root / relative
            if local.is_file():
                try:
                    self._validate_sidecar_payload(job, local)
                    continue
                except ManifestError:
                    # An invalid local sidecar may be the result of storage
                    # damage.  Prefer the last checksum-verified remote copy.
                    pass
            pointer_path = self._sidecar_pointer_path(job, relative)
            if not self.remote.download(
                self._sidecar_pointer_key(job, relative), pointer_path
            ):
                if local.exists():
                    raise ManifestError(
                        f"Invalid local sidecar has no remote recovery copy: {relative}."
                    )
                continue
            pointer = self._validate_sidecar_pointer(job, relative, pointer_path)
            digest = str(pointer["sha256"])
            object_key = self._remote_key(job, f"sidecars/objects/{digest}.json")
            if not self.remote.download(
                object_key,
                local,
                expected_sha256=digest,
            ):
                raise RuntimeError(
                    f"Runner sidecar pointer references a missing object: {job.job_id}/{relative}."
                )
            if int(local.stat().st_size) != int(pointer["size"]):
                raise ManifestError(
                    f"Restored runner sidecar size mismatch: {job.job_id}/{relative}."
                )
            self._validate_sidecar_payload(job, local)
            stat = local.stat()
            self._uploaded_runner_sidecars[(job.job_id, relative)] = (
                int(stat.st_mtime_ns),
                int(stat.st_size),
                digest,
            )

    def _sync_runner_sidecars(self, job: JobSpec, *, full_verify: bool) -> None:
        if self.remote is None:
            return
        job_root = self.jobs_dir / job.job_id
        for relative in self._runner_sidecar_relatives(job):
            local = job_root / relative
            if not local.is_file():
                continue
            stat = local.stat()
            cache_key = (job.job_id, relative)
            cached = self._uploaded_runner_sidecars.get(cache_key)
            signature = (int(stat.st_mtime_ns), int(stat.st_size))
            if not full_verify and cached is not None and cached[:2] == signature:
                continue
            self._validate_sidecar_payload(job, local)
            digest = sha256_file(local)
            if cached == (*signature, digest):
                continue
            self.remote.upload_immutable(
                local,
                self._remote_key(job, f"sidecars/objects/{digest}.json"),
            )
            pointer_path = self._sidecar_pointer_path(job, relative)
            atomic_json(
                pointer_path,
                {
                    "schema_version": "revision_full_v1_sidecar_pointer_v1",
                    "campaign_sha256": self.manifest.campaign_sha256,
                    "source_sha256": self.manifest.source_sha256,
                    "matrix_sha256": self.manifest.matrix_sha256,
                    "config_sha256": job.config_sha256,
                    "job_id": job.job_id,
                    "relative_path": relative,
                    "sha256": digest,
                    "size": int(stat.st_size),
                    "updated_utc": utc_now(),
                },
            )
            self.remote.upload_mutable(
                pointer_path,
                self._sidecar_pointer_key(job, relative),
            )
            self._uploaded_runner_sidecars[cache_key] = (*signature, digest)

    def _restore_one_runner_plan_and_chunks(
        self,
        job: JobSpec,
        *,
        kind: str,
        chunk_size: int,
        root_prefix: str,
        stage: str | None,
    ) -> tuple[ChunkPlan, ImmutableChunkStore] | None:
        plan_path = self._runner_plan_path(job, root_prefix)
        downloaded_plan = False
        if not plan_path.exists() and self.remote is not None:
            downloaded_plan = self.remote.download(
                self._remote_key(job, f"{root_prefix}/chunk_plan.json"), plan_path
            )
        if not plan_path.exists():
            return None
        plan = ChunkPlan.load(plan_path)
        expected_plan_job_id = job.job_id if stage is None else f"{job.job_id}::{stage}"
        if (
            plan.campaign_sha256 != self.manifest.campaign_sha256
            or plan.matrix_sha256 != self.manifest.matrix_sha256
            or plan.job_id != expected_plan_job_id
            or plan.kind != kind
            or plan.chunk_size != chunk_size
        ):
            raise ManifestError(f"Runner-managed chunk plan identity mismatch for {job.job_id}.")
        if downloaded_plan:
            self._uploaded_runner_plans[(job.job_id, root_prefix)] = sha256_file(plan_path)
        store = self._chunk_store(
            job,
            item_count=plan.item_count,
            root_prefix=root_prefix,
            kind=kind,
            chunk_size=chunk_size,
            store_job_id=expected_plan_job_id,
        )
        for start, stop in store.bounds():
            self._restore_chunk(
                job,
                store,
                start,
                stop,
                remote_chunk_prefix=root_prefix,
            )
        return plan, store

    def _restore_runner_plans_and_chunks(
        self, job: JobSpec
    ) -> dict[str | None, tuple[ChunkPlan, ImmutableChunkStore]]:
        restored: dict[str | None, tuple[ChunkPlan, ImmutableChunkStore]] = {}
        for stage, kind, size, root_prefix in self._runner_stage_specs(job):
            value = self._restore_one_runner_plan_and_chunks(
                job,
                kind=kind,
                chunk_size=size,
                root_prefix=root_prefix,
                stage=stage,
            )
            if value is not None:
                restored[stage] = value
        return restored

    def _sync_runner_managed_job(
        self,
        job: JobSpec,
        *,
        full_verify: bool = False,
    ) -> None:
        if not job.runner_managed_chunks or self.remote is None:
            return
        with self.runner_sync_lock:
            for stage, kind, size, root_prefix in self._runner_stage_specs(job):
                plan_path = self._runner_plan_path(job, root_prefix)
                if not plan_path.is_file():
                    continue
                plan = ChunkPlan.load(plan_path)
                expected_plan_job_id = (
                    job.job_id if stage is None else f"{job.job_id}::{stage}"
                )
                if (
                    plan.campaign_sha256 != self.manifest.campaign_sha256
                    or plan.matrix_sha256 != self.manifest.matrix_sha256
                    or plan.job_id != expected_plan_job_id
                    or plan.kind != kind
                    or plan.chunk_size != size
                ):
                    raise ManifestError(
                        f"Runner-managed chunk plan identity mismatch for {job.job_id}."
                    )
                plan_digest = sha256_file(plan_path)
                plan_key = (job.job_id, root_prefix)
                if self._uploaded_runner_plans.get(plan_key) != plan_digest:
                    self.remote.upload_immutable(
                        plan_path,
                        self._remote_key(job, f"{root_prefix}/chunk_plan.json"),
                    )
                    self._uploaded_runner_plans[plan_key] = plan_digest

                store = self._chunk_store(
                    job,
                    item_count=plan.item_count,
                    root_prefix=root_prefix,
                    kind=kind,
                    chunk_size=size,
                    store_job_id=expected_plan_job_id,
                )
                if full_verify:
                    records = store.validate_complete()
                else:
                    records = []
                    # Chunk metadata is atomically committed after the NPZ.
                    # Scan those small files and fully validate only a chunk
                    # that has not already been synchronized.
                    for metadata_path in sorted(store.root.glob(f"{kind}-*.json")):
                        try:
                            record = ChunkRecord.from_payload(
                                json.loads(metadata_path.read_text(encoding="utf-8"))
                            )
                        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                            continue
                        record_key = (job.job_id, root_prefix, record.chunk_id)
                        metadata_digest = sha256_file(metadata_path)
                        signature = (record.payload_sha256, metadata_digest)
                        if self._uploaded_runner_chunks.get(record_key) == signature:
                            continue
                        if (
                            record.campaign_sha256 != self.manifest.campaign_sha256
                            or record.matrix_sha256 != self.manifest.matrix_sha256
                            or record.job_id != expected_plan_job_id
                            or record.kind != kind
                            or (record.start, record.stop) not in store.bounds()
                            or store.status(record.start, record.stop) is not ChunkStatus.VALID
                        ):
                            continue
                        records.append(record)
                for record in records:
                    record_key = (job.job_id, root_prefix, record.chunk_id)
                    metadata_path = store.paths(record.start, record.stop)[1]
                    signature = (record.payload_sha256, sha256_file(metadata_path))
                    if self._uploaded_runner_chunks.get(record_key) != signature:
                        self._upload_chunk(
                            job,
                            store,
                            record.start,
                            record.stop,
                            remote_chunk_prefix=root_prefix,
                        )
                        self._uploaded_runner_chunks[record_key] = signature

            # Upload chunks first.  A sidecar may point at the newest adaptive
            # assignment or frozen q and is published only after its supporting
            # chunks are durable in the remote store.
            self._sync_runner_sidecars(job, full_verify=full_verify)

    def _run_command(
        self,
        job: JobSpec,
        *,
        chunk_start: int = 0,
        chunk_stop: int | None = None,
    ) -> None:
        job_dir = self.jobs_dir / job.job_id
        output_dir = job_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        argv = job.formatted_argv(
            job_dir,
            output_dir,
            chunk_start=chunk_start,
            chunk_stop=chunk_stop,
        )
        resource_mode = str(job.metadata.get("resource_mode", "bulk"))
        if resource_mode not in {"bulk", "exclusive_runtime"}:
            raise ManifestError(f"Job {job.job_id} has invalid resource mode {resource_mode!r}.")
        effective_workers = (
            self.slots * self.workers_per_job
            if resource_mode == "exclusive_runtime"
            else self.workers_per_job
        )
        env = os.environ.copy()
        env.update(
            {
                "CMFG_CAMPAIGN_ID": self.manifest.campaign_id,
                "CMFG_CAMPAIGN_SHA256": self.manifest.campaign_sha256,
                "CMFG_SOURCE_SHA256": self.manifest.source_sha256,
                "CMFG_MATRIX_SHA256": self.manifest.matrix_sha256,
                "CMFG_JOB_ID": job.job_id,
                "CMFG_NODE_ID": self.node_id,
                "CMFG_WORKERS": str(effective_workers),
                "CMFG_RESOURCE_MODE": resource_mode,
                "CMFG_RESTART_FROM_EMPTY_CACHE": (
                    "1" if resource_mode == "exclusive_runtime" else "0"
                ),
                "CMFG_STATE_DIR": str(job_dir),
                "CMFG_OUTPUT_DIR": str(output_dir),
                "CMFG_CHUNK_START": str(int(chunk_start)),
                "CMFG_CHUNK_STOP": str(job.item_count if chunk_stop is None else int(chunk_stop)),
                "CMFG_ITEM_COUNT": str(int(job.item_count)),
                "CMFG_CHUNK_KIND": str(job.chunk_kind or ""),
                "CMFG_CHUNK_SIZE": str(int(job.chunk_size or 0)),
                "CMFG_JOB_METADATA_JSON": json.dumps(job.metadata, sort_keys=True),
                "CMFG_TRAINING_SEEDS_JSON": json.dumps(job.training_seeds, sort_keys=True),
                "CMFG_AUDIT_SEEDS_JSON": json.dumps(job.audit_seeds, sort_keys=True),
                "CMFG_DEPENDENCY_DIR": str(self._dependency_root(job)),
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        log_path = job_dir / "worker.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{utc_now()}] argv={json.dumps(argv)}\n")
            log.flush()
            completed = subprocess.run(
                argv,
                cwd=self.source_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"Job {job.job_id} command failed with exit code {completed.returncode}.")

    def _validate_artifacts(self, job: JobSpec) -> list[dict[str, Any]]:
        output_dir = self.jobs_dir / job.job_id / "output"
        records: list[dict[str, Any]] = []
        for relative in job.expected_artifacts:
            path = output_dir / relative
            if not path.is_file():
                raise RuntimeError(f"Job {job.job_id} did not produce expected artifact {relative}.")
            record = {
                "path": relative,
                "size": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            if self.remote is not None:
                self.remote.upload_immutable(
                    path,
                    self._remote_key(job, f"artifacts/{relative}"),
                )
            records.append(record)
        return records

    def _run_job(self, job: JobSpec) -> None:
        self._validate_frozen_mwu_selection(job)
        if self._restore_completion(job):
            self.completed.add(job.job_id)
            return
        self._prepare_dependencies(job)
        started = time.monotonic()
        chunk_records: list[dict[str, Any]] = []
        if job.chunk_kind is not None:
            if job.runner_managed_chunks:
                self._restore_runner_sidecars(job)
                self._restore_runner_plans_and_chunks(job)
                self._run_command(job)
                restored = self._restore_runner_plans_and_chunks(job)
                expected_stages = self._runner_stage_specs(job)
                missing_stages = [stage for stage, _kind, _size, _root in expected_stages if stage not in restored]
                if missing_stages:
                    raise RuntimeError(
                        f"Runner-managed job {job.job_id} did not freeze plans for stages "
                        f"{missing_stages}."
                    )
                self._sync_runner_managed_job(job, full_verify=True)
                for stage, _kind, _size, _root in expected_stages:
                    _plan, store = restored[stage]
                    for record in store.validate_complete():
                        payload = record.to_payload()
                        if stage is not None:
                            payload["stage"] = stage
                        chunk_records.append(payload)
            else:
                store = self._chunk_store(job)
                for start, stop in store.bounds():
                    self._restore_chunk(job, store, start, stop)
                for start, stop in store.pending_bounds():
                    if self.stop_event.is_set():
                        raise InterruptedError(f"Worker stopped before chunk [{start},{stop}) of {job.job_id}.")
                    self._run_command(job, chunk_start=start, chunk_stop=stop)
                    if store.status(start, stop) is not ChunkStatus.VALID:
                        raise RuntimeError(
                            f"Chunk command for {job.job_id} did not atomically commit [{start},{stop})."
                        )
                    self._upload_chunk(job, store, start, stop)
                chunk_records = [record.to_payload() for record in store.validate_complete()]
        else:
            self._run_command(job)
        artifacts = self._validate_artifacts(job)
        completion = {
            "schema_version": "revision_full_v1_job_complete_v1",
            "status": "complete",
            "campaign_id": self.manifest.campaign_id,
            "campaign_sha256": self.manifest.campaign_sha256,
            "source_sha256": self.manifest.source_sha256,
            "matrix_sha256": self.manifest.matrix_sha256,
            "config_sha256": job.config_sha256,
            "job_id": job.job_id,
            "phase": job.phase,
            "family": job.family,
            "node_id": self.node_id,
            "slot": int(job.slot),
            "session_id": self.session_id,
            "runtime_seconds": time.monotonic() - started,
            "completed_utc": utc_now(),
            "chunks": chunk_records,
            "artifacts": artifacts,
        }
        completion_path = self._completion_path(job)
        atomic_json(completion_path, completion)
        self._validate_completion(job, completion_path)
        if self.remote is not None:
            # This marker is intentionally the last object written for the job.
            self.remote.upload_immutable(
                completion_path,
                self._remote_key(job, "job_complete.json"),
            )
            for artifact in artifacts:
                if not self.remote.exists(
                    self._remote_key(job, f"artifacts/{artifact['path']}")
                ):
                    raise RuntimeError(
                        f"Refusing to finalize {job.job_id}: remote artifact "
                        f"{artifact['path']} is missing."
                    )
        self.completed.add(job.job_id)
        self._prune_completed_job(job)

    def _heartbeat_payload(self) -> dict[str, Any]:
        with self.active_lock:
            active = dict(self.active)
        return {
            "schema_version": "revision_full_v1_heartbeat_v1",
            "campaign_id": self.manifest.campaign_id,
            "campaign_sha256": self.manifest.campaign_sha256,
            "node_id": self.node_id,
            "session_id": self.session_id,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "active_jobs": active,
            "completed_jobs": sorted(self.completed),
            "failed_jobs": dict(sorted(self.failures.items())),
            "spot_hours": self.ledger.accumulated_seconds / 3600.0 if self.ledger else None,
            "spot_hour_limit": self.manifest.spot_vm_hour_soft_limit if self.ledger else None,
            "include_families": list(self.include_families),
            "exclude_families": list(self.exclude_families),
            "updated_utc": utc_now(),
        }

    def _write_heartbeat(self) -> None:
        with self.active_lock:
            active_job_ids = tuple(self.active.values())
        jobs_by_id = {job.job_id: job for job in self.node_jobs}
        for job_id in active_job_ids:
            self._sync_runner_managed_job(jobs_by_id[job_id])
        path = self.state_dir / "control" / "heartbeat.json"
        atomic_json(path, self._heartbeat_payload())
        if self.remote is not None:
            self.remote.upload_mutable(path, f"control/{self.node_id}/heartbeat.json")
        if self.ledger is not None:
            self.ledger.checkpoint()
            if self.ledger.exhausted:
                self.stop_event.set()

    def _heartbeat_loop(self) -> None:
        while not self.stop_event.wait(self.heartbeat_seconds):
            try:
                self._write_heartbeat()
            except Exception as exc:
                error_path = self.state_dir / "control" / "heartbeat_error.txt"
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(f"{utc_now()} {type(exc).__name__}: {exc}\n", encoding="utf-8")

    def _run_slot(self, slot: int) -> None:
        jobs = [job for job in self.node_jobs if job.slot == slot]
        for job in jobs:
            if self.stop_event.is_set():
                return
            with self.active_lock:
                self.active[slot] = job.job_id
            try:
                resource_mode = str(job.metadata.get("resource_mode", "bulk"))
                with self.resource_gate.acquire(resource_mode):
                    self._run_job(job)
            except InterruptedError:
                return
            except Exception as exc:
                self.failures[job.job_id] = f"{type(exc).__name__}: {exc}"
                self.stop_event.set()
                return
            finally:
                with self.active_lock:
                    self.active.pop(slot, None)

    def _scale_to_zero(self) -> None:
        if not self.scale_to_zero_command:
            return
        completed = subprocess.run(self.scale_to_zero_command, text=True)
        if completed.returncode != 0:
            raise RuntimeError("Scale-to-zero hook failed.")

    def run(self) -> int:
        def stop_handler(_signum: int, _frame: Any) -> None:
            self.stop_event.set()

        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, stop_handler)
        heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat.start()
        try:
            with ThreadPoolExecutor(max_workers=self.slots) as executor:
                futures = [executor.submit(self._run_slot, slot) for slot in range(self.slots)]
                for future in as_completed(futures):
                    future.result()
            self._write_heartbeat()
            all_done = len(self.completed) == len(self.node_jobs)
            budget_exhausted = bool(self.ledger and self.ledger.exhausted)
            if self.failures:
                failed_path = self.state_dir / "control" / "node_failed.json"
                atomic_json(
                    failed_path,
                    {
                        "campaign_sha256": self.manifest.campaign_sha256,
                        "node_id": self.node_id,
                        "session_id": self.session_id,
                        "failures": dict(sorted(self.failures.items())),
                        "completed_jobs": sorted(self.completed),
                        "written_utc": utc_now(),
                    },
                )
                if self.remote is not None:
                    self.remote.upload_mutable(
                        failed_path, f"control/{self.node_id}/node_failed.json"
                    )
                # Only the GCP launcher supplies this hook. Scaling the MIG to
                # zero prevents Docker's restart policy from burning Spot time
                # on a deterministic terminal failure; desktop nodes remain
                # available for inspection and an explicit restart.
                self._scale_to_zero()
                return 1
            if (all_done or budget_exhausted) and not self.failures:
                done_path = self.state_dir / "control" / "node_complete.json"
                atomic_json(
                    done_path,
                    {
                        "campaign_sha256": self.manifest.campaign_sha256,
                        "node_id": self.node_id,
                        "all_jobs_complete": all_done,
                        "spot_budget_exhausted": budget_exhausted,
                        "completed_jobs": sorted(self.completed),
                        "written_utc": utc_now(),
                    },
                )
                if self.remote is not None:
                    self.remote.upload_mutable(done_path, f"control/{self.node_id}/node_complete.json")
                self._scale_to_zero()
                return 0
            return 1
        finally:
            self.stop_event.set()
            heartbeat.join(timeout=min(5.0, self.heartbeat_seconds))


def build_remote(args: argparse.Namespace) -> RemoteStore | None:
    if args.remote_root is None:
        return None
    if args.remote_root.startswith("gs://"):
        if not args.project:
            raise SystemExit("--project is required with a gs:// remote root.")
        if args.gcs_backend == "gcloud":
            return GCloudObjectStore(args.remote_root, project=args.project)
        return GoogleCloudStorageStore(args.remote_root, project=args.project)
    return LocalMirrorStore(args.remote_root)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one node of the revision-full-v1 campaign.")
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--workers-per-job", type=int, default=8)
    parser.add_argument("--remote-root")
    parser.add_argument("--project")
    parser.add_argument("--gcs-backend", choices=("python", "gcloud"), default="python")
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--spot-accounting", action="store_true")
    parser.add_argument("--scale-to-zero-command", nargs="*", default=())
    parser.add_argument("--dependency-wait-seconds", type=float, default=86_400.0)
    parser.add_argument("--dependency-poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--prune-completed-local",
        action="store_true",
        help=(
            "Remove a completed local job directory only after immutable artifacts "
            "and job_complete.json have been uploaded to the remote store."
        ),
    )
    parser.add_argument(
        "--include-families",
        default=os.environ.get("CMFG_INCLUDE_FAMILIES", ""),
        help="Comma-separated family allow-list for a campaign wave.",
    )
    parser.add_argument(
        "--exclude-families",
        default=os.environ.get("CMFG_EXCLUDE_FAMILIES", ""),
        help="Comma-separated family deny-list for a campaign wave.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = CampaignManifest.load(args.campaign)
    source_payload = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    source_manifest = SourceManifest.from_payload(source_payload)
    include_families = tuple(
        value.strip() for value in str(args.include_families).split(",") if value.strip()
    )
    exclude_families = tuple(
        value.strip() for value in str(args.exclude_families).split(",") if value.strip()
    )
    worker = CampaignWorker(
        manifest,
        source_manifest=source_manifest,
        source_root=args.source_root.resolve(),
        node_id=args.node_id,
        state_dir=args.state_dir.resolve(),
        slots=args.slots,
        workers_per_job=args.workers_per_job,
        remote=build_remote(args),
        heartbeat_seconds=args.heartbeat_seconds,
        spot_accounting=bool(args.spot_accounting),
        scale_to_zero_command=args.scale_to_zero_command,
        dependency_wait_seconds=args.dependency_wait_seconds,
        dependency_poll_seconds=args.dependency_poll_seconds,
        include_families=include_families,
        exclude_families=exclude_families,
        prune_completed_local=bool(args.prune_completed_local),
    )
    return worker.run()


if __name__ == "__main__":
    raise SystemExit(main())
