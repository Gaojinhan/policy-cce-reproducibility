from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cmfg_cce.baselines.mwu_tuning import (
    MwuTuningConfig,
    select_global_mwu_config,
)
from cmfg_cce.evaluation.backend_cache import BackendPayoffCache
from cmfg_cce.evaluation.independent_audit import (
    FrozenDistribution,
    build_joint_audit_samples,
    distribution_hash,
    freeze_distribution,
    summarize_audit_samples,
)
from cmfg_cce.evaluation.replication_chunks import (
    ProfileReplicationChunks,
    ReplicationChunkIdentity,
)
from cmfg_cce.evaluation.rollout import Profile
from cmfg_cce.experiments.revision_backends import build_benchmark_backend
from cmfg_cce.experiments.revision_full_v1_spec import (
    RevisionFullV1Matrix,
    RevisionJobKey,
)
from cmfg_cce.experiments.common import run_solver
from cmfg_cce.experiments.revision_pipeline import (
    frozen_closure,
    run_dss_empty_cache,
    run_mwu_empty_cache,
    stable_solver_seed,
)
from cmfg_cce.orchestration.chunks import (
    ChunkPlan,
    ChunkStatus,
    ImmutableChunkStore,
    deterministic_npz,
)
from cmfg_cce.orchestration.manifest import (
    atomic_json,
    canonical_json,
    sha256_bytes,
    sha256_file,
)


SCHEMA_VERSION = "revision_full_v1_mwu_tuning_pipeline_v1"
ATTEMPT_SCHEMA_VERSION = "revision_full_v1_atomic_runtime_attempt_v1"
TRAINING_CHECKPOINT_SCHEMA_VERSION = "revision_full_v1_mwu_training_checkpoint_v1"
TRAINING_CHUNK_SIZE = 128
AUDIT_CHUNK_SIZE = 8
PRODUCTION_TRAIN_ROLLOUTS = 200
PRODUCTION_AUDIT_ROLLOUTS = 500
SMOKE_TRAIN_ROLLOUTS = 2
SMOKE_AUDIT_ROLLOUTS = 4
SMOKE_BOOTSTRAP_SAMPLES = 100
ATTEMPT_PADDING_ITEMS = TRAINING_CHUNK_SIZE - 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_sha256(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{label} must be a lowercase SHA256 digest.")
    return normalized


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _game_key(job: RevisionJobKey) -> str:
    return (
        f"n{job.n_agents}j{job.policies_per_agent}__"
        f"{job.mechanism}__s{job.seed}"
    )


def _find_pipeline_job(matrix: RevisionFullV1Matrix, job_id: str) -> RevisionJobKey:
    matches = [job for job in matrix.mwu_tuning_pipeline_jobs() if job.job_id == str(job_id)]
    if len(matches) != 1:
        raise ValueError(
            f"job-id must identify exactly one of the 24 frozen MWU tuning pipelines; "
            f"found {len(matches)} matches for {job_id!r}."
        )
    return matches[0]


def _campaign_hash(matrix_hash: str, job_id: str, *, smoke: bool) -> str:
    value = os.environ.get("CMFG_CAMPAIGN_SHA256", "")
    if value:
        return _require_sha256(value, "CMFG_CAMPAIGN_SHA256")
    if not smoke:
        raise ValueError("CMFG_CAMPAIGN_SHA256 is required outside --smoke mode.")
    return sha256_bytes(f"smoke:{matrix_hash}:{job_id}".encode("utf-8"))


def _backend_for_stage(
    job: RevisionJobKey,
    namespace: str,
    *,
    smoke: bool,
):
    backend = build_benchmark_backend(
        mechanism=job.mechanism,
        n_agents=2 if smoke else job.n_agents,
        policies_per_agent=2 if smoke else job.policies_per_agent,
        seed=job.seed,
        namespace=namespace,
        variant="base",
        stream_label=f"revision-full-v1:mwu-tuning:{namespace}",
    )
    if smoke:
        backend.config = replace(backend.config, horizon=2)
    return backend


def _training_tensor(cache: BackendPayoffCache, policy_ids: Sequence[str]) -> bytes:
    profiles = tuple(sorted(cache.estimates))
    if not profiles:
        raise RuntimeError("A completed runtime attempt has an empty payoff cache.")
    policy_index = {str(policy): index for index, policy in enumerate(policy_ids)}
    profile_indices = np.asarray(
        [[policy_index[str(policy)] for policy in profile] for profile in profiles],
        dtype=np.int16,
    )
    estimates = [cache.estimates[profile] for profile in profiles]
    arrays = {
        "profile_indices": profile_indices,
        "mean_returns": np.stack([estimate.mean_returns for estimate in estimates]),
        "var_returns": np.stack([estimate.var_returns for estimate in estimates]),
        "ci_radius": np.stack([estimate.ci_radius for estimate in estimates]),
        "rollout_counts": np.asarray(
            [estimate.n_rollouts for estimate in estimates], dtype=np.int32
        ),
        "platform_operating_score": np.asarray(
            [
                float(estimate.mean_metrics.get("platform_operating_score", 0.0))
                for estimate in estimates
            ],
            dtype=np.float64,
        ),
    }
    return deterministic_npz(arrays)


def _frozen_payload(distribution: FrozenDistribution) -> dict[str, Any]:
    return {
        "solver": distribution.solver,
        "q_hash": distribution.q_hash,
        "support": [list(profile) for profile in distribution.support],
        "probabilities": list(distribution.probabilities),
    }


def _distribution_from_payload(
    value: Mapping[str, Any],
    *,
    n_agents: int,
    policy_ids: Sequence[str],
) -> FrozenDistribution:
    distribution = freeze_distribution(
        str(value["solver"]),
        [tuple(str(item) for item in profile) for profile in value["support"]],
        [float(probability) for probability in value["probabilities"]],
        n_agents,
        policy_ids,
    )
    if distribution.q_hash != str(value.get("q_hash", "")):
        raise ValueError("Checkpoint q hash does not match its support and probabilities.")
    return distribution


def _attempt_payload(
    *,
    job: RevisionJobKey,
    matrix_hash: str,
    campaign_hash: str,
    attempt_key: str,
    solver_kind: str,
    solver_config: Mapping[str, Any],
    distribution: FrozenDistribution,
    cache: BackendPayoffCache,
    stats: Mapping[str, Any],
    train_rollouts: int,
    dss_reference_runtime_seconds: float | None,
    time_budget_seconds: float | None,
    training_tensor_sha256: str,
    smoke: bool,
) -> dict[str, Any]:
    return {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "status": "complete",
        "job_id": job.job_id,
        "matrix_sha256": matrix_hash,
        "campaign_sha256": campaign_hash,
        "game_key": _game_key(job),
        "attempt_key": attempt_key,
        "solver_kind": solver_kind,
        "solver_config": _jsonable(solver_config),
        "training_namespace": "training",
        "training_backend_identity": _jsonable(cache.backend.cache_identity),
        "training_seeds": _jsonable(asdict(cache.backend.seeds)),
        "train_rollouts": int(train_rollouts),
        "empty_cache_confirmed": True,
        "attempt_stitched": False,
        "dss_reference_runtime_seconds": dss_reference_runtime_seconds,
        "time_budget_seconds": time_budget_seconds,
        "runtime_seconds": float(stats["runtime_seconds"]),
        "payoff_evaluation_seconds": float(stats["payoff_evaluation_seconds"]),
        "solver_compute_seconds": float(stats["solver_compute_seconds"]),
        "training_profile_count": int(cache.evaluated_profile_count),
        "training_rollout_episode_count": int(cache.rollout_episode_total),
        "access_stats": _jsonable(stats),
        "solver_diagnostics": _jsonable(
            getattr(stats, "diagnostics", {}) if not isinstance(stats, Mapping) else {}
        ),
        "distribution": _frozen_payload(distribution),
        "training_tensor_sha256": training_tensor_sha256,
        "smoke": bool(smoke),
    }


class RuntimeAttemptCheckpoints:
    """One immutable training chunk per completed empty-cache runtime attempt.

    The 128-item chunk grid is retained for campaign compatibility.  A tuning
    runtime cannot itself resume from profile chunks without invalidating its
    wall-clock measurement, so each chunk contains one whole, completed
    attempt plus 127 declared padding items.  Interrupted attempts remain only
    as incomplete local directories and are never decoded or combined.
    """

    def __init__(
        self,
        root: Path,
        *,
        job_id: str,
        matrix_hash: str,
        campaign_hash: str,
        attempt_keys: Sequence[str],
    ) -> None:
        self.root = root
        self.job_id = str(job_id)
        self.matrix_hash = str(matrix_hash)
        self.campaign_hash = str(campaign_hash)
        self.attempt_keys = tuple(str(value) for value in attempt_keys)
        if not self.attempt_keys or len(set(self.attempt_keys)) != len(self.attempt_keys):
            raise ValueError("Runtime attempt keys must be nonempty and unique.")
        item_ids: list[str] = []
        for key in self.attempt_keys:
            item_ids.append(f"{key}::completed-attempt")
            item_ids.extend(
                f"{key}::reserved-{index:03d}"
                for index in range(ATTEMPT_PADDING_ITEMS)
            )
        store_job_id = f"{self.job_id}::training"
        self.plan = ChunkPlan.create(
            campaign_sha256=self.campaign_hash,
            matrix_sha256=self.matrix_hash,
            job_id=store_job_id,
            kind="training",
            chunk_size=TRAINING_CHUNK_SIZE,
            item_ids=tuple(item_ids),
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.plan.write_immutable(self.root / "chunk_plan.json")
        self.store = ImmutableChunkStore(
            self.root,
            campaign_sha256=self.campaign_hash,
            matrix_sha256=self.matrix_hash,
            job_id=store_job_id,
            kind="training",
            item_count=len(item_ids),
            chunk_size=TRAINING_CHUNK_SIZE,
        )

    def bounds(self, attempt_key: str) -> tuple[int, int]:
        index = self.attempt_keys.index(str(attempt_key))
        start = index * TRAINING_CHUNK_SIZE
        return start, start + TRAINING_CHUNK_SIZE

    def load(self, attempt_key: str) -> tuple[dict[str, Any], bytes] | None:
        start, stop = self.bounds(attempt_key)
        status = self.store.status(start, stop)
        if status is not ChunkStatus.VALID:
            return None
        arrays = self.store.read(start, stop)
        if arrays.get("attempt_index", np.empty(0)).tolist() != [self.attempt_keys.index(attempt_key)]:
            raise ValueError(f"Training checkpoint index mismatch for {attempt_key}.")
        payload_bytes = bytes(np.asarray(arrays["attempt_payload"], dtype=np.uint8))
        tensor_bytes = bytes(np.asarray(arrays["training_tensor_payload"], dtype=np.uint8))
        payload = json.loads(payload_bytes.decode("utf-8"))
        self._validate_payload(attempt_key, payload, tensor_bytes)
        return payload, tensor_bytes

    def _validate_payload(
        self,
        attempt_key: str,
        payload: Mapping[str, Any],
        tensor_bytes: bytes,
    ) -> None:
        expected = {
            "schema_version": ATTEMPT_SCHEMA_VERSION,
            "status": "complete",
            "job_id": self.job_id,
            "matrix_sha256": self.matrix_hash,
            "campaign_sha256": self.campaign_hash,
            "attempt_key": str(attempt_key),
            "empty_cache_confirmed": True,
            "attempt_stitched": False,
        }
        observed = {key: payload.get(key) for key in expected}
        if observed != expected:
            raise ValueError(
                f"Runtime attempt checkpoint identity mismatch for {attempt_key}: {observed!r}."
            )
        if payload.get("training_tensor_sha256") != sha256_bytes(tensor_bytes):
            raise ValueError(f"Training tensor hash mismatch for {attempt_key}.")

    def commit(
        self,
        attempt_key: str,
        payload: Mapping[str, Any],
        tensor_bytes: bytes,
    ) -> None:
        self._validate_payload(attempt_key, payload, tensor_bytes)
        start, stop = self.bounds(attempt_key)
        self.store.commit(
            start,
            stop,
            {
                "attempt_index": np.asarray(
                    [self.attempt_keys.index(attempt_key)], dtype=np.int16
                ),
                "attempt_payload": np.frombuffer(canonical_json(payload), dtype=np.uint8),
                "training_tensor_payload": np.frombuffer(tensor_bytes, dtype=np.uint8),
                "declared_logical_item_count": np.asarray(
                    [TRAINING_CHUNK_SIZE], dtype=np.int16
                ),
            },
        )

    def records(self) -> list[dict[str, Any]]:
        return [record.to_payload() for record in self.store.validate_complete()]


def _attempt_directories(root: Path, attempt_key: str) -> tuple[Path, ...]:
    directory = root / attempt_key
    if not directory.exists():
        return ()
    return tuple(sorted(path for path in directory.iterdir() if path.is_dir()))


def _load_local_complete_attempt(
    root: Path,
    checkpoints: RuntimeAttemptCheckpoints,
    attempt_key: str,
) -> tuple[dict[str, Any], bytes] | None:
    complete: list[tuple[dict[str, Any], bytes]] = []
    for directory in _attempt_directories(root, attempt_key):
        marker = directory / "attempt_complete.json"
        tensor = directory / "training_tensor.npz"
        if not marker.exists() or not tensor.exists():
            continue
        payload = json.loads(marker.read_text(encoding="utf-8"))
        tensor_bytes = tensor.read_bytes()
        checkpoints._validate_payload(attempt_key, payload, tensor_bytes)
        complete.append((payload, tensor_bytes))
    if len(complete) > 1:
        raise ValueError(f"Multiple completed runtime attempts exist for {attempt_key}.")
    return complete[0] if complete else None


def _next_attempt_directory(root: Path, attempt_key: str) -> Path:
    parent = root / attempt_key
    parent.mkdir(parents=True, exist_ok=True)
    existing = _attempt_directories(root, attempt_key)
    directory = parent / f"attempt-{len(existing) + 1:04d}"
    directory.mkdir(exist_ok=False)
    return directory


def _run_atomic_attempt(
    *,
    attempts_root: Path,
    checkpoints: RuntimeAttemptCheckpoints,
    attempt_key: str,
    identity: Mapping[str, Any],
    run: Callable[[], tuple[dict[str, Any], bytes]],
) -> dict[str, Any]:
    restored = checkpoints.load(attempt_key)
    if restored is not None:
        return restored[0]
    local = _load_local_complete_attempt(attempts_root, checkpoints, attempt_key)
    if local is not None:
        checkpoints.commit(attempt_key, local[0], local[1])
        return local[0]

    directory = _next_attempt_directory(attempts_root, attempt_key)
    atomic_json(
        directory / "attempt_started.json",
        {
            **dict(identity),
            "attempt_key": attempt_key,
            "status": "started_from_empty_cache",
            "started_utc": _utc_now(),
        },
    )
    try:
        payload, tensor_bytes = run()
        _atomic_bytes(directory / "training_tensor.npz", tensor_bytes)
        atomic_json(directory / "attempt_complete.json", payload)
        checkpoints.commit(attempt_key, payload, tensor_bytes)
        return payload
    except BaseException as exc:
        atomic_json(
            directory / "attempt_failed.json",
            {
                **dict(identity),
                "attempt_key": attempt_key,
                "status": "incomplete_not_reusable",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_utc": _utc_now(),
            },
        )
        raise


def _attempt_trace(attempts_root: Path, attempt_keys: Sequence[str]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for key in attempt_keys:
        for directory in _attempt_directories(attempts_root, key):
            complete = (directory / "attempt_complete.json").exists()
            failed = (directory / "attempt_failed.json").exists()
            trace.append(
                {
                    "attempt_key": key,
                    "directory": directory.relative_to(attempts_root.parent.parent.parent).as_posix(),
                    "complete": bool(complete),
                    "failed_or_interrupted": bool(failed or not complete),
                    "reusable": bool(complete),
                }
            )
    return trace


def _smoke_or_production_counts(smoke: bool) -> tuple[int, int, int]:
    if smoke:
        return SMOKE_TRAIN_ROLLOUTS, SMOKE_AUDIT_ROLLOUTS, SMOKE_BOOTSTRAP_SAMPLES
    return PRODUCTION_TRAIN_ROLLOUTS, PRODUCTION_AUDIT_ROLLOUTS, 5000


def _mwu_budget(dss_runtime_seconds: float, *, smoke: bool) -> float:
    # A zero-second smoke budget makes every configuration run one required,
    # deterministic round.  Production always uses the exact measured DSS
    # wall time without rounding or scenario-specific adjustment.
    return 0.0 if smoke else float(dss_runtime_seconds)


def _validate_runtime_environment(*, workers: int, smoke: bool) -> dict[str, Any]:
    resource_mode = str(os.environ.get("CMFG_RESOURCE_MODE", ""))
    restart_flag = str(os.environ.get("CMFG_RESTART_FROM_EMPTY_CACHE", ""))
    node_id = str(os.environ.get("CMFG_NODE_ID", ""))
    if not smoke:
        if resource_mode != "exclusive_runtime" or restart_flag != "1":
            raise ValueError(
                "Production MWU tuning must run as an exclusive_runtime job with "
                "CMFG_RESTART_FROM_EMPTY_CACHE=1."
            )
        if int(workers) != 32:
            raise ValueError(
                "Production MWU/DSS runtime matching is frozen to the exclusive "
                "32-worker GCP execution mode."
            )
    return {
        "resource_mode": "smoke" if smoke else resource_mode,
        "restart_from_empty_cache": True if smoke else restart_flag == "1",
        "node_id": node_id or ("local-smoke" if smoke else ""),
        "workers": int(workers),
        "runtime_attempts_are_atomic": True,
    }


def _audit_one_distribution(
    profile_returns: Mapping[Profile, np.ndarray],
    distribution: FrozenDistribution,
    policy_ids: Sequence[str],
    *,
    n_agents: int,
    audit_rollouts: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], str]:
    before = distribution.q_hash
    samples = build_joint_audit_samples(
        profile_returns,
        distribution,
        policy_ids,
        n_agents=n_agents,
        sample_count=audit_rollouts,
    )
    summary = summarize_audit_samples(
        samples,
        alpha=0.05,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    after = distribution_hash(distribution.support, distribution.probabilities)
    if before != after:
        raise RuntimeError("Frozen q changed during MWU tuning audit.")
    sample_hash = sha256_bytes(
        deterministic_npz(
            {
                "gain_samples": samples.gain_samples,
                "q_return_samples": samples.q_return_samples,
            }
        )
    )
    return dict(summary), sample_hash


def run_mwu_tuning_pipeline(
    *,
    job_id: str,
    state_dir: Path,
    output_dir: Path,
    matrix_hash: str,
    workers: int,
    smoke: bool = False,
) -> dict[str, Any]:
    matrix = RevisionFullV1Matrix()
    expected_matrix_hash = matrix.matrix_hash
    matrix_hash = _require_sha256(matrix_hash, "matrix-hash")
    if matrix_hash != expected_matrix_hash:
        raise ValueError(
            f"Matrix hash mismatch: CLI={matrix_hash}, current={expected_matrix_hash}."
        )
    job = _find_pipeline_job(matrix, job_id)
    campaign_hash = _campaign_hash(matrix_hash, job.job_id, smoke=smoke)
    workers = max(1, int(workers))
    runtime_environment = _validate_runtime_environment(workers=workers, smoke=smoke)
    state_dir = Path(state_dir)
    output_dir = Path(output_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rollouts, audit_rollouts, bootstrap_samples = _smoke_or_production_counts(smoke)

    identity = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job.job_id,
        "matrix_sha256": matrix_hash,
        "campaign_sha256": campaign_hash,
        "game_key": _game_key(job),
        "smoke": bool(smoke),
    }
    identity_path = state_dir / "runner_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("State directory belongs to a different MWU tuning pipeline.")
    else:
        atomic_json(identity_path, identity)

    grid_by_id = {config.config_id: config for config in matrix.mwu_tuning_configs}
    config_ids = tuple(job.tuning_config_ids)
    if tuple(grid_by_id) != config_ids:
        raise ValueError("Physical tuning job does not contain the frozen 16-point grid.")
    attempt_keys = ("dss-reference",) + tuple(f"mwu--{value}" for value in config_ids)
    training_root = state_dir / "stages" / "training" / "chunks"
    attempts_root = state_dir / "stages" / "runtime_measurement" / "attempts"
    checkpoints = RuntimeAttemptCheckpoints(
        training_root,
        job_id=job.job_id,
        matrix_hash=matrix_hash,
        campaign_hash=campaign_hash,
        attempt_keys=attempt_keys,
    )

    training_backend = _backend_for_stage(job, "training", smoke=smoke)
    policy_ids = tuple(training_backend.policies)
    n_agents = int(training_backend.n_agents)
    dss_seed = stable_solver_seed("revision-full-v1", _game_key(job), "dss")

    def run_dss() -> tuple[dict[str, Any], bytes]:
        if smoke:
            cache = BackendPayoffCache(
                backend=training_backend,
                n_rollouts=train_rollouts,
                workers=workers,
            )
            cache.reset_access_log()
            started = time.perf_counter()
            result = run_solver(
                "REPAIR-SAD-CCE",
                cache,
                policy_ids,
                {
                    "target_gap": 1.0e-8,
                    "active_sampling": False,
                    "rollouts_max": train_rollouts,
                    "initial_support_size": 1,
                    "max_support_size": 2,
                    "support_add_batch_size": 1,
                    "max_rounds": 1,
                    "selector": "platform_operating_score",
                    "repair": {
                        "repair_rounds": 0,
                        "profile_budget_multiplier": 1.0,
                    },
                },
                seed=dss_seed,
                metadata={"revision_campaign": "revision-full-v1-smoke"},
            )
            elapsed = time.perf_counter() - started
            stats = cache.access_stats()
            stats.update(
                {
                    "runtime_seconds": float(elapsed),
                    "payoff_evaluation_seconds": float(cache.eval_time_seconds),
                    "solver_compute_seconds": max(
                        0.0, float(elapsed) - float(cache.eval_time_seconds)
                    ),
                }
            )
        else:
            result, cache, stats = run_dss_empty_cache(
                training_backend,
                train_rollouts=train_rollouts,
                workers=workers,
                solver_seed=dss_seed,
            )
        distribution = freeze_distribution(
            result.solver,
            result.support_profiles,
            result.support_probabilities,
            n_agents,
            policy_ids,
        )
        tensor_bytes = _training_tensor(cache, policy_ids)
        payload = _attempt_payload(
            job=job,
            matrix_hash=matrix_hash,
            campaign_hash=campaign_hash,
            attempt_key="dss-reference",
            solver_kind="REPAIR-SAD-CCE",
            solver_config={"solver_seed": dss_seed},
            distribution=distribution,
            cache=cache,
            stats=stats,
            train_rollouts=train_rollouts,
            dss_reference_runtime_seconds=None,
            time_budget_seconds=None,
            training_tensor_sha256=sha256_bytes(tensor_bytes),
            smoke=smoke,
        )
        payload["solver_diagnostics"] = _jsonable(result.diagnostics)
        return payload, tensor_bytes

    dss_payload = _run_atomic_attempt(
        attempts_root=attempts_root,
        checkpoints=checkpoints,
        attempt_key="dss-reference",
        identity=identity,
        run=run_dss,
    )
    dss_runtime = float(dss_payload["runtime_seconds"])
    budget_seconds = _mwu_budget(dss_runtime, smoke=smoke)
    mwu_seed = stable_solver_seed("revision-full-v1", _game_key(job), "mwu-shared")

    mwu_payloads: list[dict[str, Any]] = []
    for config_id in config_ids:
        config = grid_by_id[config_id]
        attempt_key = f"mwu--{config_id}"

        def run_mwu(config: MwuTuningConfig = config, attempt_key: str = attempt_key):
            result, cache, stats = run_mwu_empty_cache(
                training_backend,
                train_rollouts=train_rollouts,
                workers=workers,
                solver_seed=mwu_seed,
                budget_seconds=budget_seconds,
                eta=config.eta,
                schedule=config.schedule,
                exploration_floor=config.exploration_floor,
                burn_in_rounds=config.burn_in_rounds,
            )
            distribution = freeze_distribution(
                result.solver,
                result.support_profiles,
                result.support_probabilities,
                n_agents,
                policy_ids,
            )
            tensor_bytes = _training_tensor(cache, policy_ids)
            payload = _attempt_payload(
                job=job,
                matrix_hash=matrix_hash,
                campaign_hash=campaign_hash,
                attempt_key=attempt_key,
                solver_kind="MWU-PolicyTrace",
                solver_config={
                    **config.to_solver_mapping(),
                    "mwu_config_id": config.config_id,
                    "solver_seed": mwu_seed,
                },
                distribution=distribution,
                cache=cache,
                stats=stats,
                train_rollouts=train_rollouts,
                dss_reference_runtime_seconds=dss_runtime,
                time_budget_seconds=budget_seconds,
                training_tensor_sha256=sha256_bytes(tensor_bytes),
                smoke=smoke,
            )
            payload["solver_diagnostics"] = _jsonable(result.diagnostics)
            return payload, tensor_bytes

        mwu_payloads.append(
            _run_atomic_attempt(
                attempts_root=attempts_root,
                checkpoints=checkpoints,
                attempt_key=attempt_key,
                identity=identity,
                run=run_mwu,
            )
        )

    distributions: dict[str, FrozenDistribution] = {
        "dss-reference": _distribution_from_payload(
            dss_payload["distribution"], n_agents=n_agents, policy_ids=policy_ids
        )
    }
    for payload in mwu_payloads:
        config_id = str(payload["solver_config"]["mwu_config_id"])
        distributions[config_id] = _distribution_from_payload(
            payload["distribution"], n_agents=n_agents, policy_ids=policy_ids
        )
    closure = frozen_closure(tuple(distributions.values()), policy_ids)
    audit_backend = _backend_for_stage(job, "mwu_tuning_audit", smoke=smoke)
    if tuple(audit_backend.policies) != policy_ids or audit_backend.n_agents != n_agents:
        raise RuntimeError("Training and audit policy games are incompatible.")
    if audit_backend.seeds.type_seed != training_backend.seeds.type_seed:
        raise RuntimeError("Fresh audit changed the frozen manufacturer population.")
    training_dynamic = {
        training_backend.seeds.order_seed,
        training_backend.seeds.outside_seed,
        training_backend.seeds.availability_seed,
        training_backend.seeds.tie_break_seed,
        training_backend.seeds.rollout_replication_seed,
    }
    audit_dynamic = {
        audit_backend.seeds.order_seed,
        audit_backend.seeds.outside_seed,
        audit_backend.seeds.availability_seed,
        audit_backend.seeds.tie_break_seed,
        audit_backend.seeds.rollout_replication_seed,
    }
    if training_dynamic.intersection(audit_dynamic):
        raise RuntimeError("MWU tuning training and +15,000,000 audit streams overlap.")

    audit_chunks = ProfileReplicationChunks(
        state_dir / "stages" / "mwu_tuning_audit" / "chunks",
        identity=ReplicationChunkIdentity(
            campaign_sha256=campaign_hash,
            matrix_sha256=matrix_hash,
            job_id=job.job_id,
            stage_id="mwu_tuning_audit",
            kind="audit",
            chunk_size=AUDIT_CHUNK_SIZE,
        ),
        profiles=closure,
    )
    audit_chunks.evaluate_pending(
        audit_backend,
        rollouts=audit_rollouts,
        workers=workers,
    )
    profile_returns = audit_chunks.return_vectors(rollouts=audit_rollouts)
    bootstrap_seed = stable_solver_seed(
        "revision-full-v1", _game_key(job), "mwu-tuning-shared-bootstrap"
    )
    audit_results: dict[str, dict[str, Any]] = {}
    for label, distribution in distributions.items():
        summary, samples_hash = _audit_one_distribution(
            profile_returns,
            distribution,
            policy_ids,
            n_agents=n_agents,
            audit_rollouts=audit_rollouts,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        audit_results[label] = {
            **summary,
            "q_hash_before": distribution.q_hash,
            "q_hash_after": distribution_hash(
                distribution.support, distribution.probabilities
            ),
            "joint_replication_samples_sha256": samples_hash,
        }

    payload_by_config = {
        str(payload["solver_config"]["mwu_config_id"]): payload
        for payload in mwu_payloads
    }
    selection_rows: list[dict[str, Any]] = []
    for config_id in config_ids:
        attempt = payload_by_config[config_id]
        audit = audit_results[config_id]
        distribution = distributions[config_id]
        selection_rows.append(
            {
                "status": "complete",
                "game_key": _game_key(job),
                "mwu_config_id": config_id,
                **dict(attempt["solver_config"]),
                "fresh_nominal_relative_gap_percent": float(
                    audit["relative_nominal_gap_percent"]
                ),
                "fresh_max_t_relative_gap_ucb_percent": float(
                    audit["max_t_relative_gap_ucb95_percent"]
                ),
                "fresh_max_t_relative_gap_lcb_percent": float(
                    audit["max_t_relative_gap_lcb95_percent"]
                ),
                "fresh_nominal_gap": float(audit["nominal_gap"]),
                "fresh_max_t_gap_ucb95": float(audit["max_t_gap_ucb95"]),
                "audit_rollouts": int(audit_rollouts),
                "audit_profile_count": len(closure),
                "q_hash": distribution.q_hash,
                "support_size": len(distribution.support),
                "support": [
                    {"profile": list(profile), "probability": float(probability)}
                    for profile, probability in zip(
                        distribution.support,
                        distribution.probabilities,
                        strict=True,
                    )
                ],
                "runtime_seconds": float(attempt["runtime_seconds"]),
                "dss_reference_runtime_seconds": dss_runtime,
                "time_budget_seconds": budget_seconds,
                "training_profile_count": int(attempt["training_profile_count"]),
                "training_rollout_episode_count": int(
                    attempt["training_rollout_episode_count"]
                ),
                "trace_rounds": int(
                    attempt.get("solver_diagnostics", {}).get("trace_rounds", 0)
                ),
            }
        )

    result = {
        **identity,
        "status": "complete",
        "family": "mwu_tuning_pipeline",
        "N": n_agents,
        "J": len(policy_ids),
        "reported_N": job.n_agents,
        "reported_J": job.policies_per_agent,
        "mechanism": job.mechanism,
        "seed": job.seed,
        "policy_ids": list(policy_ids),
        "train_rollouts": train_rollouts,
        "audit_rollouts": audit_rollouts,
        "bootstrap_samples": bootstrap_samples,
        "runtime_environment": runtime_environment,
        "dss_reference": dss_payload,
        "mwu_attempts": mwu_payloads,
        "selection_rows": selection_rows,
        "audit_results": audit_results,
        "training_backend_identity": training_backend.cache_identity,
        "audit_backend_identity": audit_backend.cache_identity,
        "training_seeds": asdict(training_backend.seeds),
        "audit_seeds": asdict(audit_backend.seeds),
        "seed_isolation": {
            "manufacturer_population_held_fixed": True,
            "dynamic_streams_disjoint": True,
            "audit_namespace_offset": 15_000_000,
        },
        "training_checkpoint": {
            "schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
            "chunk_size": TRAINING_CHUNK_SIZE,
            "attempt_count": len(attempt_keys),
            "item_order_sha256": checkpoints.plan.item_order_sha256,
            "chunks": checkpoints.records(),
            "attempt_trace": _attempt_trace(attempts_root, attempt_keys),
        },
        "audit_checkpoint": {
            "chunk_size": AUDIT_CHUNK_SIZE,
            "profile_count": len(closure),
            "item_order_sha256": audit_chunks.plan.item_order_sha256,
            "chunks": [
                record.to_payload() for record in audit_chunks.store.validate_complete()
            ],
            "raw_replication_vectors": {
                "location": "stages/mwu_tuning_audit/chunks",
                "array": "returns",
                "shape_per_profile": [audit_rollouts, n_agents],
                "common_replication_index_across_profiles": True,
            },
        },
    }
    atomic_json(output_dir / "result.json", result)
    return result


def reduce_mwu_tuning_results(
    result_paths: Sequence[Path],
    *,
    matrix: RevisionFullV1Matrix | None = None,
) -> dict[str, Any]:
    """Require all 24 pipelines, then freeze one globally selected MWU config."""

    matrix = matrix or RevisionFullV1Matrix()
    expected_jobs = matrix.mwu_tuning_pipeline_jobs()
    expected_games = tuple(_game_key(job) for job in expected_jobs)
    expected_by_game = {_game_key(job): job for job in expected_jobs}
    observed: dict[str, tuple[Path, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for path in result_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        game_key = str(payload.get("game_key", ""))
        if game_key in observed:
            raise ValueError(f"Duplicate MWU tuning pipeline result for {game_key}.")
        if game_key not in expected_by_game:
            raise ValueError(f"Unexpected MWU tuning game {game_key!r}.")
        if payload.get("status") != "complete" or payload.get("matrix_sha256") != matrix.matrix_hash:
            raise ValueError(f"Incomplete or incompatible MWU tuning result: {path}.")
        game_rows = list(payload.get("selection_rows", ()))
        if len(game_rows) != len(matrix.mwu_tuning_configs):
            raise ValueError(f"MWU tuning result {path} does not contain all 16 configurations.")
        observed[game_key] = (Path(path), payload)
        rows.extend(game_rows)
    missing = sorted(set(expected_games).difference(observed))
    if missing:
        raise ValueError(f"Cannot select MWU configuration before all 24 games complete: {missing}.")
    selected, summaries = select_global_mwu_config(
        rows,
        expected_game_keys=expected_games,
        expected_grid=matrix.mwu_tuning_configs,
    )
    selected_rows = [
        row for row in rows if str(row.get("mwu_config_id")) == selected.config_id
    ]
    trace_rounds = [int(row.get("trace_rounds", 0)) for row in selected_rows]
    if len(trace_rounds) != len(expected_games) or min(trace_rounds, default=0) <= 0:
        raise ValueError("Selected MWU configuration has no positive trace-round count.")
    source_results = [
        {
            "game_key": game_key,
            "path": str(observed[game_key][0]),
            "sha256": sha256_file(observed[game_key][0]),
        }
        for game_key in sorted(observed)
    ]
    payload = {
        "schema_version": "revision_full_v1_global_mwu_selection_v2",
        "status": "complete",
        "matrix_sha256": matrix.matrix_hash,
        "calibration_game_count": len(expected_games),
        "configuration_count": len(matrix.mwu_tuning_configs),
        "selected_config": {
            "mwu_config_id": selected.config_id,
            **selected.to_solver_mapping(),
        },
        "selection_scope": "global_hyperparameters_only",
        "formal_q_contract": (
            "each benchmark/scalability runtime job measures DSS from an empty "
            "cache and gives MWU exactly that wall-time budget"
        ),
        "summaries": summaries,
        "source_results": source_results,
    }
    payload["selection_sha256"] = sha256_bytes(canonical_json(payload))
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one revision-full-v1 MWU global-tuning physical pipeline."
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix-hash", required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("CMFG_WORKERS", "1")),
    )
    # Accepted for compatibility with the generic campaign worker.  This
    # runner owns its dynamic profile plans and therefore ignores outer bounds.
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--chunk-stop", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_mwu_tuning_pipeline(
        job_id=args.job_id,
        state_dir=args.state_dir,
        output_dir=args.output_dir,
        matrix_hash=args.matrix_hash,
        workers=args.workers,
        smoke=bool(args.smoke),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
