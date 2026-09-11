from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from cmfg_cce.evaluation.independent_audit import (
    FrozenDistribution,
    build_joint_audit_samples,
    distribution_hash,
    freeze_distribution,
)
from cmfg_cce.evaluation.revision_statistics import summarize_three_arm_transplant
from cmfg_cce.evaluation.rollout import Profile
from cmfg_cce.experiments.revision_backends import build_cnc_backend
from cmfg_cce.experiments.revision_full_v1_spec import (
    RevisionFullV1Matrix,
    RevisionJobKey,
)
from cmfg_cce.experiments.revision_pipeline import frozen_closure, stable_solver_seed
from cmfg_cce.orchestration.chunks import ChunkPlan, ImmutableChunkStore, deterministic_npz
from cmfg_cce.orchestration.manifest import atomic_json, sha256_bytes


SCHEMA_VERSION = "revision_full_v1_policy_transplant_v1"
AUDIT_CHUNK_SIZE = 8
PRODUCTION_AUDIT_ROLLOUTS = 2000
PRODUCTION_BOOTSTRAP_SAMPLES = 5000
SMOKE_AUDIT_ROLLOUTS = 4
SMOKE_BOOTSTRAP_SAMPLES = 100


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Commit a binary result only after the full deterministic payload exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _audit_sample_artifact(
    samples_by_arm: Mapping[str, object],
    policy_ids: Sequence[str],
) -> bytes:
    """Serialize the three paired arms for campaign-level simultaneous inference."""

    policy_index = {str(policy): index for index, policy in enumerate(policy_ids)}
    arrays: dict[str, np.ndarray] = {}
    common_labels: tuple[tuple[int, str], ...] | None = None
    for arm, raw_samples in sorted(samples_by_arm.items()):
        labels = tuple((int(agent), str(policy)) for agent, policy in raw_samples.labels)
        if common_labels is None:
            common_labels = labels
        elif labels != common_labels:
            raise ValueError("Transplant audit arms changed the replacement-label order.")
        arrays[f"{arm}__gain_samples"] = np.asarray(
            raw_samples.gain_samples, dtype=np.float64
        )
        arrays[f"{arm}__q_return_samples"] = np.asarray(
            raw_samples.q_return_samples, dtype=np.float64
        )
    if common_labels is None:
        raise ValueError("A transplant audit artifact requires at least one arm.")
    arrays["label_agent"] = np.asarray(
        [agent for agent, _policy in common_labels], dtype=np.int16
    )
    arrays["label_policy_index"] = np.asarray(
        [policy_index[policy] for _agent, policy in common_labels], dtype=np.int16
    )
    return deterministic_npz(arrays)


def _require_sha256(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{label} must be a lowercase SHA256 digest.")
    return normalized


def _find_job(matrix: RevisionFullV1Matrix, job_id: str) -> RevisionJobKey:
    matches = [job for job in matrix.transplant_jobs() if job.job_id == str(job_id)]
    if len(matches) != 1:
        raise ValueError(
            "job-id must identify exactly one frozen policy-transplant direction; "
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


def _distribution_payload(distribution: FrozenDistribution) -> dict[str, Any]:
    return {
        "solver": distribution.solver,
        "q_hash": distribution.q_hash,
        "support": [list(profile) for profile in distribution.support],
        "probabilities": list(distribution.probabilities),
    }


def _distribution_from_result(
    payload: Mapping[str, Any],
    *,
    expected_mechanism: str,
    expected_condition: str,
    expected_seed: int,
    matrix_hash: str,
) -> FrozenDistribution:
    expected = {
        "status": "complete",
        "mechanism": expected_mechanism,
        "condition": expected_condition,
        "seed": int(expected_seed),
        "matrix_sha256": matrix_hash,
    }
    observed = {name: payload.get(name) for name in expected}
    if observed != expected:
        raise ValueError(
            f"CNC dependency identity mismatch: expected {expected!r}, observed {observed!r}."
        )
    value = payload.get("distribution")
    if not isinstance(value, Mapping):
        raise ValueError("CNC dependency has no frozen top-level distribution.")
    support = [tuple(str(item) for item in profile) for profile in value.get("support", ())]
    policy_ids = tuple(f"A{index}" for index in range(1, 7))
    distribution = freeze_distribution(
        str(value.get("solver", "REPAIR-SAD-CCE")),
        support,
        [float(item) for item in value.get("probabilities", ())],
        4,
        policy_ids,
    )
    if distribution.q_hash != str(value.get("q_hash", "")):
        raise ValueError("CNC dependency q hash is inconsistent.")
    return distribution


def _load_dependency_payloads(
    dependency_dir: Path,
) -> tuple[dict[str, Any], ...]:
    paths = sorted(Path(dependency_dir).glob("*/artifacts/result.json"))
    if len(paths) != 2:
        raise ValueError(
            "A policy transplant requires exactly two restored CNC result dependencies; "
            f"found {len(paths)} in {dependency_dir}."
        )
    return tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths)


def _evaluate_entry(
    args: tuple[int, str, Profile, object, int],
) -> tuple[int, str, Profile, np.ndarray]:
    index, mechanism, profile, backend, rollouts = args
    returns = np.empty((int(rollouts), int(backend.n_agents)), dtype=np.float64)
    for replication in range(int(rollouts)):
        values, _metrics = backend.run_episode(profile, replication)
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (int(backend.n_agents),) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Transplant profile {mechanism}/{profile} returned invalid payoffs."
            )
        returns[replication] = values
    return int(index), str(mechanism), tuple(profile), returns


class TaggedMechanismAuditChunks:
    """One immutable audit plan spanning source- and target-mechanism profiles."""

    def __init__(
        self,
        root: Path,
        *,
        campaign_hash: str,
        matrix_hash: str,
        job_id: str,
        entries: Sequence[tuple[str, Profile]],
        policy_ids: Sequence[str],
    ) -> None:
        self.root = Path(root)
        self.campaign_hash = str(campaign_hash)
        self.matrix_hash = str(matrix_hash)
        self.job_id = str(job_id)
        self.entries = tuple((str(mechanism), tuple(profile)) for mechanism, profile in entries)
        self.policy_ids = tuple(str(value) for value in policy_ids)
        if not self.entries or len(set(self.entries)) != len(self.entries):
            raise ValueError("Tagged audit entries must be nonempty and unique.")
        allowed = set(self.policy_ids)
        n_agents = len(self.entries[0][1])
        if any(
            len(profile) != n_agents or set(profile).difference(allowed)
            for _mechanism, profile in self.entries
        ):
            raise ValueError("Tagged audit profiles do not match the frozen policy game.")
        self.root.mkdir(parents=True, exist_ok=True)
        store_job_id = f"{self.job_id}::transplant_audit"
        item_ids = tuple(
            f"{mechanism}::{('|'.join(profile))}" for mechanism, profile in self.entries
        )
        self.plan = ChunkPlan.create(
            campaign_sha256=self.campaign_hash,
            matrix_sha256=self.matrix_hash,
            job_id=store_job_id,
            kind="audit",
            chunk_size=AUDIT_CHUNK_SIZE,
            item_ids=item_ids,
        )
        self.plan.write_immutable(self.root / "chunk_plan.json")
        self.store = ImmutableChunkStore(
            self.root,
            campaign_sha256=self.campaign_hash,
            matrix_sha256=self.matrix_hash,
            job_id=store_job_id,
            kind="audit",
            item_count=len(self.entries),
            chunk_size=AUDIT_CHUNK_SIZE,
        )

    def evaluate_pending(
        self,
        backends: Mapping[str, object],
        *,
        rollouts: int,
        workers: int,
    ) -> None:
        policy_index = {policy: index for index, policy in enumerate(self.policy_ids)}
        mechanism_ids = {mechanism: index for index, mechanism in enumerate(sorted(backends))}
        for start, stop in self.store.pending_bounds():
            tasks = [
                (index, mechanism, profile, backends[mechanism], int(rollouts))
                for index, (mechanism, profile) in enumerate(
                    self.entries[start:stop], start=start
                )
            ]
            worker_count = min(
                max(1, int(workers)),
                max(1, os.cpu_count() or 1),
                len(tasks),
            )
            if worker_count == 1:
                rows = [_evaluate_entry(task) for task in tasks]
            else:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=mp.get_context("spawn"),
                ) as executor:
                    rows = list(executor.map(_evaluate_entry, tasks, chunksize=1))
            rows.sort(key=lambda row: row[0])
            if [row[0] for row in rows] != list(range(start, stop)):
                raise RuntimeError("Tagged transplant evaluation changed frozen item order.")
            self.store.commit(
                start,
                stop,
                {
                    "returns": np.stack([row[3] for row in rows]),
                    "profile_indices": np.asarray(
                        [
                            [policy_index[policy] for policy in row[2]]
                            for row in rows
                        ],
                        dtype=np.int16,
                    ),
                    "mechanism_indices": np.asarray(
                        [mechanism_ids[row[1]] for row in rows], dtype=np.int8
                    ),
                    "item_indices": np.arange(start, stop, dtype=np.int64),
                    "rollout_count": np.asarray([int(rollouts)], dtype=np.int64),
                },
            )

    def return_vectors(
        self, *, rollouts: int, n_agents: int
    ) -> dict[str, dict[Profile, np.ndarray]]:
        result: dict[str, dict[Profile, np.ndarray]] = {}
        for start, stop in self.store.bounds():
            arrays = self.store.read(start, stop)
            expected = (stop - start, int(rollouts), int(n_agents))
            if arrays.get("returns", np.empty(0)).shape != expected:
                raise ValueError(f"Transplant chunk [{start},{stop}) has invalid returns shape.")
            if arrays.get("rollout_count", np.empty(0)).tolist() != [int(rollouts)]:
                raise ValueError(f"Transplant chunk [{start},{stop}) changed rollout count.")
            for offset, (mechanism, profile) in enumerate(self.entries[start:stop]):
                result.setdefault(mechanism, {})[profile] = np.asarray(
                    arrays["returns"][offset], dtype=float
                )
        return result


def run_policy_transplant_pipeline(
    *,
    job_id: str,
    state_dir: Path,
    output_dir: Path,
    matrix_hash: str,
    workers: int,
    dependency_dir: Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    matrix = RevisionFullV1Matrix()
    matrix_hash = _require_sha256(matrix_hash, "matrix-hash")
    if matrix_hash != matrix.matrix_hash:
        raise ValueError("Policy-transplant runner matrix hash mismatch.")
    job = _find_job(matrix, job_id)
    campaign_hash = _campaign_hash(matrix_hash, job.job_id, smoke=smoke)
    state_dir = Path(state_dir)
    output_dir = Path(output_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    dependency_root = Path(
        dependency_dir
        or os.environ.get("CMFG_DEPENDENCY_DIR", state_dir / "dependencies")
    )
    payloads = _load_dependency_payloads(dependency_root)
    by_mechanism = {str(payload.get("mechanism")): payload for payload in payloads}
    if set(by_mechanism) != {job.source_mechanism, job.target_mechanism}:
        raise ValueError("Restored CNC dependencies do not match transplant direction.")
    source_q = _distribution_from_result(
        by_mechanism[str(job.source_mechanism)],
        expected_mechanism=str(job.source_mechanism),
        expected_condition=job.variant,
        expected_seed=job.seed,
        matrix_hash=matrix_hash,
    )
    target_q = _distribution_from_result(
        by_mechanism[str(job.target_mechanism)],
        expected_mechanism=str(job.target_mechanism),
        expected_condition=job.variant,
        expected_seed=job.seed,
        matrix_hash=matrix_hash,
    )
    policy_ids = tuple(f"A{index}" for index in range(1, 7))
    source_closure = frozen_closure((source_q,), policy_ids)
    target_closure = frozen_closure((source_q, target_q), policy_ids)
    entries = tuple(
        sorted(
            [(str(job.source_mechanism), profile) for profile in source_closure]
            + [(str(job.target_mechanism), profile) for profile in target_closure]
        )
    )
    source_backend = build_cnc_backend(
        mechanism=str(job.source_mechanism),
        seed=job.seed,
        condition=job.variant,
        namespace="transplant_audit",
        stream_label="revision-full-v1:transplant-audit",
    )
    target_backend = build_cnc_backend(
        mechanism=str(job.target_mechanism),
        seed=job.seed,
        condition=job.variant,
        namespace="transplant_audit",
        stream_label="revision-full-v1:transplant-audit",
    )
    if tuple(source_backend.policies) != policy_ids or tuple(target_backend.policies) != policy_ids:
        raise RuntimeError("Policy-transplant libraries do not share the frozen A1--A6 IDs.")
    if asdict(source_backend.seeds) != asdict(target_backend.seeds):
        raise RuntimeError("Policy-transplant arms do not share common random numbers.")
    training_backend = build_cnc_backend(
        mechanism=str(job.source_mechanism),
        seed=job.seed,
        condition=job.variant,
        namespace="training",
    )
    dynamic_fields = (
        "order_seed",
        "outside_seed",
        "availability_seed",
        "tie_break_seed",
        "rollout_replication_seed",
    )
    if any(
        getattr(source_backend.seeds, field) == getattr(training_backend.seeds, field)
        for field in dynamic_fields
    ):
        raise RuntimeError("Transplant audit streams overlap training streams.")
    if source_backend.seeds.type_seed != training_backend.seeds.type_seed:
        raise RuntimeError("Transplant audit changed the manufacturer population.")

    rollouts = SMOKE_AUDIT_ROLLOUTS if smoke else PRODUCTION_AUDIT_ROLLOUTS
    bootstrap_samples = (
        SMOKE_BOOTSTRAP_SAMPLES if smoke else PRODUCTION_BOOTSTRAP_SAMPLES
    )
    chunks = TaggedMechanismAuditChunks(
        state_dir / "stages" / "transplant_audit" / "chunks",
        campaign_hash=campaign_hash,
        matrix_hash=matrix_hash,
        job_id=job.job_id,
        entries=entries,
        policy_ids=policy_ids,
    )
    chunks.evaluate_pending(
        {
            str(job.source_mechanism): source_backend,
            str(job.target_mechanism): target_backend,
        },
        rollouts=rollouts,
        workers=max(1, int(workers)),
    )
    returns = chunks.return_vectors(rollouts=rollouts, n_agents=4)
    same_samples = build_joint_audit_samples(
        returns[str(job.source_mechanism)], source_q, policy_ids, n_agents=4
    )
    cross_samples = build_joint_audit_samples(
        returns[str(job.target_mechanism)], source_q, policy_ids, n_agents=4
    )
    target_samples = build_joint_audit_samples(
        returns[str(job.target_mechanism)], target_q, policy_ids, n_agents=4
    )
    bootstrap_seed = stable_solver_seed(
        "revision-full-v1",
        job.job_id,
        "three-arm-joint-bootstrap",
    )
    summary = summarize_three_arm_transplant(
        same_samples,
        cross_samples,
        target_samples,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    samples_by_arm = {
        "same_mechanism_control": same_samples,
        "cross_mechanism": cross_samples,
        "target_recomputed": target_samples,
    }
    sample_artifact_bytes = _audit_sample_artifact(samples_by_arm, policy_ids)
    sample_artifact_path = output_dir / "transplant_audit_samples.npz"
    _atomic_bytes(sample_artifact_path, sample_artifact_bytes)
    sample_hashes = {
        name: sha256_bytes(
            deterministic_npz(
                {
                    "gain_samples": samples.gain_samples,
                    "q_return_samples": samples.q_return_samples,
                }
            )
        )
        for name, samples in samples_by_arm.items()
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "job_id": job.job_id,
        "matrix_sha256": matrix_hash,
        "campaign_sha256": campaign_hash,
        "family": "policy_transplant",
        "condition": job.variant,
        "seed": int(job.seed),
        "source_mechanism": job.source_mechanism,
        "target_mechanism": job.target_mechanism,
        "source_distribution": _distribution_payload(source_q),
        "target_distribution": _distribution_payload(target_q),
        "audit_rollouts": int(rollouts),
        "bootstrap_samples": int(bootstrap_samples),
        "three_arm_summary": summary,
        "joint_sample_hashes": sample_hashes,
        "joint_sample_artifact": {
            "path": sample_artifact_path.name,
            "sha256": sha256_bytes(sample_artifact_bytes),
            "arms": sorted(samples_by_arm),
            "common_replication_index_across_arms": True,
        },
        "source_backend_identity": source_backend.cache_identity,
        "target_backend_identity": target_backend.cache_identity,
        "transplant_seeds": asdict(source_backend.seeds),
        "seed_isolation": {
            "manufacturer_population_held_fixed": True,
            "dynamic_streams_disjoint_from_training": True,
            "paired_common_random_numbers_across_arms": True,
            "namespace_offset": 40_000_000,
        },
        "checkpoint": {
            "chunk_size": AUDIT_CHUNK_SIZE,
            "profile_mechanism_pair_count": len(entries),
            "item_order_sha256": chunks.plan.item_order_sha256,
            "chunks": [record.to_payload() for record in chunks.store.validate_complete()],
            "raw_replication_vectors": {
                "location": "stages/transplant_audit/chunks",
                "array": "returns",
                "common_replication_index_across_mechanisms": True,
            },
        },
        "smoke": bool(smoke),
    }
    if (
        source_q.q_hash
        != distribution_hash(source_q.support, source_q.probabilities)
        or target_q.q_hash
        != distribution_hash(target_q.support, target_q.probabilities)
    ):
        raise RuntimeError("A frozen transplant distribution changed during audit.")
    atomic_json(output_dir / "result.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one revision-full-v1 three-arm policy-transplant audit."
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix-hash", required=True)
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("CMFG_WORKERS", "1"))
    )
    parser.add_argument("--dependency-dir", type=Path)
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--chunk-stop", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_policy_transplant_pipeline(
        job_id=args.job_id,
        state_dir=args.state_dir,
        output_dir=args.output_dir,
        matrix_hash=args.matrix_hash,
        workers=args.workers,
        dependency_dir=args.dependency_dir,
        smoke=bool(args.smoke),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
