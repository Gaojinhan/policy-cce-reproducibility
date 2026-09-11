from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cmfg_cce.evaluation.independent_audit import (
    AuditSampleMatrix,
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
from cmfg_cce.evaluation.resumable_training_chunks import (
    AdaptiveTrainingChunks,
    ResumableTrainingPayoffCache,
    TRAINING_CHUNK_SIZE,
)
from cmfg_cce.evaluation.revision_statistics import (
    bootstrap_outcome_replications,
    weighted_outcome_replications,
)
from cmfg_cce.evaluation.rollout import Profile
from cmfg_cce.experiments.common import run_solver
from cmfg_cce.experiments.revision_backends import build_cnc_backend
from cmfg_cce.experiments.revision_full_v1_spec import (
    RevisionFullV1Matrix,
    RevisionJobKey,
    RobustnessSetting,
)
from cmfg_cce.experiments.revision_pipeline import (
    CNC_RAW_OUTCOME_FIELDS,
    default_dss_config,
    empirical_game_from_replications,
    frozen_closure,
    stable_solver_seed,
)
from cmfg_cce.experiments.revision_variants import (
    LibraryVariant,
    revision_library_variants,
)
from cmfg_cce.orchestration.manifest import atomic_json, canonical_json, sha256_bytes
from cmfg_cce.solvers.cce_lp import solve_full_cce_lp


SCHEMA_VERSION = "revision_full_v1_cnc_pipeline_v1"
TRAINING_RESULT_SCHEMA_VERSION = "revision_full_v1_cnc_training_result_v1"
AUDIT_CHUNK_SIZE = 8
PRODUCTION_TRAIN_ROLLOUTS = 200
PRODUCTION_AUDIT_ROLLOUTS = 2000
PRODUCTION_OUTCOME_ROLLOUTS = 500
PRODUCTION_BOOTSTRAP_SAMPLES = 5000
SMOKE_TRAIN_ROLLOUTS = 2
SMOKE_AUDIT_ROLLOUTS = 4
SMOKE_OUTCOME_ROLLOUTS = 4
SMOKE_BOOTSTRAP_SAMPLES = 100
LOW_GAP_CERTIFICATION_PERCENT = 2.0
_DYNAMIC_SEED_FIELDS = (
    "order_seed",
    "outside_seed",
    "availability_seed",
    "tie_break_seed",
    "rollout_replication_seed",
)


def _require_sha256(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{label} must be a lowercase SHA256 digest.")
    return normalized


def _campaign_hash(matrix_hash: str, job_id: str, *, smoke: bool) -> str:
    configured = os.environ.get("CMFG_CAMPAIGN_SHA256", "")
    if configured:
        return _require_sha256(configured, "CMFG_CAMPAIGN_SHA256")
    if not smoke:
        raise ValueError("CMFG_CAMPAIGN_SHA256 is required outside --smoke mode.")
    return sha256_bytes(f"smoke:{matrix_hash}:{job_id}".encode("utf-8"))


def _physical_cnc_jobs(matrix: RevisionFullV1Matrix) -> tuple[RevisionJobKey, ...]:
    return (
        matrix.cnc_main_jobs()
        + matrix.selection_sensitivity_jobs()
        + matrix.policy_library_sensitivity_jobs("training")
        + matrix.parameter_robustness_jobs("training")
    )


def _find_job(matrix: RevisionFullV1Matrix, job_id: str) -> RevisionJobKey:
    matches = [job for job in _physical_cnc_jobs(matrix) if job.job_id == str(job_id)]
    if len(matches) != 1:
        raise ValueError(
            "job-id must identify exactly one frozen CNC, selection, library, or "
            f"parameter pipeline; found {len(matches)} matches for {job_id!r}."
        )
    return matches[0]


def _split_condition_variant(job: RevisionJobKey) -> tuple[str, str | None]:
    if job.family in {"policy_library_sensitivity", "parameter_robustness"}:
        condition, variant = job.variant.rsplit("__", maxsplit=1)
        return condition, variant
    return job.variant, None


def _matched_cnc_dss_seed(job: RevisionJobKey) -> int:
    """Hold search randomization fixed across matched CNC variants.

    Every CNC comparison is paired by reporting seed.  The mechanism,
    operating condition, policy-library variant, and parameter setting must
    therefore not change the DSS initialization or search ordering.  Keeping
    the search seed fixed removes avoidable solver-randomization noise from
    both mechanism and operating-factor contrasts.
    """

    return stable_solver_seed(
        "revision-full-v1",
        "cnc-matched-dss-training",
        job.seed,
    )


def _job_variants(
    job: RevisionJobKey,
) -> tuple[str, LibraryVariant | None, RobustnessSetting | None, str | None]:
    condition, variant_id = _split_condition_variant(job)
    if job.family == "policy_library_sensitivity":
        by_id = {variant.variant_id: variant for variant in revision_library_variants()}
        if variant_id not in by_id:
            raise ValueError(f"Unknown policy-library variant in job: {variant_id!r}")
        return condition, by_id[str(variant_id)], None, str(variant_id)
    if job.family == "parameter_robustness":
        by_id = {
            setting.setting_id: setting
            for setting in RevisionFullV1Matrix().robustness_settings
        }
        if variant_id not in by_id:
            raise ValueError(f"Unknown parameter setting in job: {variant_id!r}")
        return condition, None, by_id[str(variant_id)], str(variant_id)
    return condition, None, None, None


def _counts(smoke: bool) -> tuple[int, int, int, int]:
    if smoke:
        return (
            SMOKE_TRAIN_ROLLOUTS,
            SMOKE_AUDIT_ROLLOUTS,
            SMOKE_OUTCOME_ROLLOUTS,
            SMOKE_BOOTSTRAP_SAMPLES,
        )
    return (
        PRODUCTION_TRAIN_ROLLOUTS,
        PRODUCTION_AUDIT_ROLLOUTS,
        PRODUCTION_OUTCOME_ROLLOUTS,
        PRODUCTION_BOOTSTRAP_SAMPLES,
    )


def _backend_for_stage(
    job: RevisionJobKey,
    *,
    condition: str,
    namespace: str,
    library_variant: LibraryVariant | None,
    robustness: RobustnessSetting | None,
    smoke: bool,
):
    backend = build_cnc_backend(
        mechanism=job.mechanism,
        seed=job.seed,
        condition=condition,
        namespace=namespace,
        library_variant=library_variant,
        robustness=robustness,
        stream_label=f"revision-full-v1:{job.family}:{namespace}",
    )
    if not smoke:
        return backend
    all_policy_ids = tuple(backend.policies)
    selected_ids = (
        all_policy_ids[:2] + all_policy_ids[-2:]
        if library_variant is not None
        and library_variant.variant_id == "expanded_a1_a8"
        else all_policy_ids[:2]
    )
    return replace(
        backend,
        config=replace(backend.config, horizon=2, outside_burn_in=1),
        policies={policy_id: backend.policies[policy_id] for policy_id in selected_ids},
    )


def _validate_stage_backends(training, audit, outcome) -> None:
    policy_ids = tuple(training.policies)
    if tuple(audit.policies) != policy_ids or tuple(outcome.policies) != policy_ids:
        raise RuntimeError("CNC training, audit, and outcome stages changed the policy library.")
    if training.n_agents != audit.n_agents or training.n_agents != outcome.n_agents:
        raise RuntimeError("CNC stage backends changed the manufacturer population size.")
    if not (
        training.seeds.type_seed == audit.seeds.type_seed == outcome.seeds.type_seed
    ):
        raise RuntimeError("Fresh CNC stages changed the frozen manufacturer population.")
    streams = {
        "training": {getattr(training.seeds, name) for name in _DYNAMIC_SEED_FIELDS},
        "formal_audit": {getattr(audit.seeds, name) for name in _DYNAMIC_SEED_FIELDS},
        "outcome_evaluation": {getattr(outcome.seeds, name) for name in _DYNAMIC_SEED_FIELDS},
    }
    for first, second in (("training", "formal_audit"), ("training", "outcome_evaluation"), ("formal_audit", "outcome_evaluation")):
        if streams[first].intersection(streams[second]):
            raise RuntimeError(f"CNC seed namespaces overlap: {first} and {second}.")


def _distribution_payload(distribution: FrozenDistribution) -> dict[str, Any]:
    return {
        "solver": distribution.solver,
        "q_hash": distribution.q_hash,
        "support": [list(profile) for profile in distribution.support],
        "probabilities": list(distribution.probabilities),
    }


def _distribution_from_payload(
    payload: Mapping[str, Any],
    *,
    n_agents: int,
    policy_ids: Sequence[str],
) -> FrozenDistribution:
    distribution = freeze_distribution(
        str(payload.get("solver", "")),
        [tuple(str(value) for value in profile) for profile in payload.get("support", ())],
        [float(value) for value in payload.get("probabilities", ())],
        n_agents,
        policy_ids,
    )
    if distribution.q_hash != str(payload.get("q_hash", "")):
        raise ValueError("Stored CNC distribution q hash is inconsistent.")
    return distribution


def _array_mapping_sha256(values: Mapping[str, np.ndarray]) -> str:
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


def _profile_vector_hash(
    profiles: Sequence[Profile],
    policy_ids: Sequence[str],
    returns: Mapping[Profile, np.ndarray],
    metrics: Mapping[Profile, Mapping[str, np.ndarray]] | None = None,
) -> str:
    policy_index = {str(policy): index for index, policy in enumerate(policy_ids)}
    arrays: dict[str, np.ndarray] = {
        "profile_indices": np.asarray(
            [[policy_index[policy] for policy in profile] for profile in profiles],
            dtype=np.int16,
        ),
        "returns": np.stack([np.asarray(returns[profile], dtype=float) for profile in profiles]),
    }
    if metrics is not None:
        metric_names = tuple(sorted(next(iter(metrics.values()))))
        for metric_name in metric_names:
            arrays[f"metric__{metric_name}"] = np.stack(
                [np.asarray(metrics[profile][metric_name], dtype=float) for profile in profiles]
            )
    return _array_mapping_sha256(arrays)


def _audit_sample_hash(samples: AuditSampleMatrix) -> str:
    return _array_mapping_sha256(
        {
            "gain_samples": samples.gain_samples,
            "q_return_samples": samples.q_return_samples,
        }
    )


def _json_number(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_number(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_number(item) for item in value]
    return value


def _training_checkpoint_payload(chunks: ProfileReplicationChunks) -> dict[str, Any]:
    return {
        "chunk_size": TRAINING_CHUNK_SIZE,
        "profile_count": len(chunks.profiles),
        "item_order_sha256": chunks.plan.item_order_sha256,
        "chunks": [record.to_payload() for record in chunks.store.validate_complete()],
    }


def _smoke_dss_config(train_rollouts: int, workers: int) -> dict[str, Any]:
    return {
        "workers": int(workers),
        "target_gap": 1.0e-8,
        "active_sampling": False,
        "rollouts_max": int(train_rollouts),
        "initial_support_size": 4,
        "max_support_size": 8,
        "support_add_batch_size": 2,
        "max_rounds": 1,
        "selector": "platform_operating_score",
        "epsilon_tolerance": 1.0e-9,
        "repair": {
            "repair_rounds": 0,
            "profile_budget_multiplier": 1.0,
        },
    }


def _run_or_load_training(
    *,
    job: RevisionJobKey,
    state_dir: Path,
    matrix_hash: str,
    campaign_hash: str,
    backend,
    train_rollouts: int,
    workers: int,
    smoke: bool,
) -> tuple[dict[str, FrozenDistribution], dict[str, Any]]:
    stage_dir = state_dir / "stages" / "training"
    stage_dir.mkdir(parents=True, exist_ok=True)
    result_path = stage_dir / "training_result.json"
    policy_ids = tuple(backend.policies)
    n_agents = int(backend.n_agents)
    selectors = (
        tuple(job.selectors)
        if job.family == "selection_sensitivity"
        else ("platform_operating_score",)
    )
    expected_distribution_labels = selectors

    if job.family == "selection_sensitivity":
        profiles = tuple(product(policy_ids, repeat=n_agents))
        chunks = ProfileReplicationChunks(
            stage_dir / "chunks",
            identity=ReplicationChunkIdentity(
                campaign_sha256=campaign_hash,
                matrix_sha256=matrix_hash,
                job_id=job.job_id,
                stage_id="training",
                kind="training",
                chunk_size=TRAINING_CHUNK_SIZE,
            ),
            profiles=profiles,
        )
        chunks.evaluate_pending(
            backend,
            rollouts=train_rollouts,
            workers=workers,
            metric_names=("platform_operating_score",),
        )
        profile_returns = chunks.return_vectors(rollouts=train_rollouts)
        metric_vectors = chunks.metric_vectors(
            rollouts=train_rollouts,
            metric_names=("platform_operating_score",),
        )
        game = empirical_game_from_replications(
            profiles,
            policy_ids,
            profile_returns,
            {
                profile: metric_vectors[profile]["platform_operating_score"]
                for profile in profiles
            },
        )
        distributions: dict[str, FrozenDistribution] = {}
        solver_payloads: dict[str, Any] = {}
        started = time.perf_counter()
        for selector in selectors:
            solution = solve_full_cce_lp(game, selector=selector)
            distribution = freeze_distribution(
                solution.solver,
                solution.support_profiles,
                solution.support_probabilities,
                n_agents,
                policy_ids,
            )
            distributions[selector] = distribution
            solver_payloads[selector] = {
                **solution.to_jsonable(),
                "q_hash": distribution.q_hash,
            }
        training_payload = {
            "schema_version": TRAINING_RESULT_SCHEMA_VERSION,
            "status": "complete",
            "job_id": job.job_id,
            "matrix_sha256": matrix_hash,
            "campaign_sha256": campaign_hash,
            "mode": "shared_complete_table_three_selectors",
            "train_rollouts": int(train_rollouts),
            "policy_ids": list(policy_ids),
            "distributions": {
                label: _distribution_payload(distribution)
                for label, distribution in distributions.items()
            },
            "solver_outputs": _json_number(solver_payloads),
            "solver_seconds": float(time.perf_counter() - started),
            "raw_training_vectors_sha256": _profile_vector_hash(
                profiles, policy_ids, profile_returns, metric_vectors
            ),
            "checkpoint": _training_checkpoint_payload(chunks),
            "backend_identity": backend.cache_identity,
            "seeds": asdict(backend.seeds),
        }
    else:
        adaptive_chunks = AdaptiveTrainingChunks(
            stage_dir / "chunks",
            campaign_sha256=campaign_hash,
            matrix_sha256=matrix_hash,
            job_id=job.job_id,
            policy_ids=policy_ids,
            n_agents=n_agents,
            rollouts=train_rollouts,
            backend_identity=backend.cache_identity,
        )
        cache = ResumableTrainingPayoffCache(
            backend=backend,
            n_rollouts=train_rollouts,
            workers=workers,
            chunks=adaptive_chunks,
        )
        if result_path.exists():
            stored = json.loads(result_path.read_text(encoding="utf-8"))
            expected = {
                "schema_version": TRAINING_RESULT_SCHEMA_VERSION,
                "status": "complete",
                "job_id": job.job_id,
                "matrix_sha256": matrix_hash,
                "campaign_sha256": campaign_hash,
                "train_rollouts": int(train_rollouts),
                "policy_ids": list(policy_ids),
            }
            if {key: stored.get(key) for key in expected} != expected:
                raise ValueError("Stored CNC training result identity mismatch.")
            cache.finalize_training()
            if stored.get("raw_training_vectors_sha256") != cache.raw_training_vectors_sha256:
                raise ValueError("Stored CNC training raw-vector hash changed.")
            distributions = {
                label: _distribution_from_payload(
                    value, n_agents=n_agents, policy_ids=policy_ids
                )
                for label, value in dict(stored.get("distributions", {})).items()
            }
            if set(distributions) != set(expected_distribution_labels):
                raise ValueError("Stored CNC training selectors differ from the job contract.")
            return distributions, stored

        cache.reset_access_log()
        config = (
            _smoke_dss_config(train_rollouts, workers)
            if smoke
            else default_dss_config(train_rollouts=train_rollouts, workers=workers)
        )
        solver_seed = _matched_cnc_dss_seed(job)
        started = time.perf_counter()
        solution = run_solver(
            "REPAIR-SAD-CCE",
            cache,
            policy_ids,
            config,
            seed=solver_seed,
            metadata={"revision_campaign": "revision-full-v1", "job_id": job.job_id},
        )
        primary_solver_seconds = time.perf_counter() - started
        distribution = freeze_distribution(
            solution.solver,
            solution.support_profiles,
            solution.support_probabilities,
            n_agents,
            policy_ids,
        )
        distributions = {"platform_operating_score": distribution}
        solver_payloads: dict[str, Any] = {
            "platform_operating_score": {
                "solver": solution.solver,
                "objective_value": float(solution.objective_value),
                "training_gap": float(solution.cce_gap_nominal),
                "training_gap_ucb": float(solution.cce_gap_ucb),
                "diagnostics": _json_number(solution.diagnostics),
                "solver_seed": int(solver_seed),
                "solver_seed_match_group": {
                    "condition": _split_condition_variant(job)[0],
                    "mechanism": job.mechanism,
                    "reporting_seed": int(job.seed),
                    "variant_excluded": True,
                },
                "solver_config": config,
                "solver_seconds": float(primary_solver_seconds),
                "access_stats": cache.access_stats(),
            }
        }
        cache.finalize_training()
        training_payload = {
            "schema_version": TRAINING_RESULT_SCHEMA_VERSION,
            "status": "complete",
            "job_id": job.job_id,
            "matrix_sha256": matrix_hash,
            "campaign_sha256": campaign_hash,
            "mode": "adaptive_dss_cce",
            "train_rollouts": int(train_rollouts),
            "policy_ids": list(policy_ids),
            "distributions": {
                "platform_operating_score": _distribution_payload(distribution)
            },
            "solver_outputs": solver_payloads,
            "solver_seconds": float(
                sum(float(row["solver_seconds"]) for row in solver_payloads.values())
            ),
            "raw_training_vectors_sha256": cache.raw_training_vectors_sha256,
            "checkpoint": adaptive_chunks.checkpoint_payload(),
            "backend_identity": backend.cache_identity,
            "seeds": asdict(backend.seeds),
        }

    atomic_json(result_path, training_payload)
    return distributions, training_payload


def _dependency_candidates(root: Path) -> tuple[Path, ...]:
    candidates = {
        path.resolve()
        for pattern in ("result.json", "artifacts/result.json", "*/artifacts/result.json", "**/result.json")
        for path in root.glob(pattern)
        if path.is_file()
    }
    return tuple(sorted(candidates))


def _load_core_dependency(
    dependency_dir: Path,
    *,
    job: RevisionJobKey,
    condition: str,
    matrix_hash: str,
    expanded_policy_ids: Sequence[str],
) -> tuple[FrozenDistribution, dict[str, Any]]:
    matched: list[tuple[FrozenDistribution, dict[str, Any], Path]] = []
    for path in _dependency_candidates(Path(dependency_dir)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = (
            payload.get("status") == "complete"
            and payload.get("family") == "policy_library_sensitivity"
            and payload.get("library_variant") == "base_a1_a6"
            and payload.get("mechanism") == job.mechanism
            and payload.get("condition") == condition
            and payload.get("seed") == job.seed
            and payload.get("matrix_sha256") == matrix_hash
        )
        if not identity:
            continue
        value = payload.get("distribution")
        if not isinstance(value, Mapping):
            raise ValueError("Core-library dependency has no top-level frozen distribution.")
        distribution = _distribution_from_payload(
            value,
            n_agents=4,
            policy_ids=expanded_policy_ids,
        )
        if set(policy for profile in distribution.support for policy in profile).difference(
            {f"A{index}" for index in range(1, 7)}
        ):
            raise ValueError("Core-library dependency contains A7/A8 policies.")
        matched.append((distribution, payload, path))
    if len(matched) != 1:
        raise ValueError(
            "Expanded A1--A8 sensitivity requires exactly one matching base A1--A6 "
            f"CNC result dependency; found {len(matched)} in {dependency_dir}."
        )
    distribution, payload, path = matched[0]
    return distribution, {
        "path": str(path),
        "job_id": payload.get("job_id"),
        "q_hash": distribution.q_hash,
    }


def _new_policy_gain_summary(
    samples: AuditSampleMatrix,
    audit_summary: Mapping[str, Any],
    *,
    policy_ids: Sequence[str] = ("A7", "A8"),
) -> dict[str, Any]:
    gains = np.asarray(samples.gain_samples, dtype=float)
    means = np.mean(gains, axis=0)
    standard_errors = np.std(gains, axis=0, ddof=1) / math.sqrt(gains.shape[0])
    beta = float(audit_summary["max_t_beta_relative"])
    denominator_lcb = float(audit_summary["payoff_denominator_lcb"])
    rows: list[dict[str, Any]] = []
    for index, (agent, policy) in enumerate(samples.labels):
        if policy not in set(policy_ids):
            continue
        absolute_ucb = max(0.0, float(means[index] + beta * standard_errors[index]))
        rows.append(
            {
                "agent": int(agent),
                "replacement_policy": str(policy),
                "mean_gain": float(means[index]),
                "standard_error": float(standard_errors[index]),
                "simultaneous_gain_ucb95": absolute_ucb,
                "simultaneous_relative_gain_ucb95_percent": (
                    100.0 * absolute_ucb / denominator_lcb
                ),
            }
        )
    maximum = max(
        (row["simultaneous_relative_gain_ucb95_percent"] for row in rows),
        default=float("nan"),
    )
    return {
        "policies": list(policy_ids),
        "rows": rows,
        "maximum_new_policy_relative_gain_ucb95_percent": maximum,
        "passes_two_percent_gain_stop": bool(maximum <= LOW_GAP_CERTIFICATION_PERCENT),
    }


def _run_formal_audit(
    *,
    job: RevisionJobKey,
    state_dir: Path,
    matrix_hash: str,
    campaign_hash: str,
    backend,
    distributions: Mapping[str, FrozenDistribution],
    rollouts: int,
    bootstrap_samples: int,
    workers: int,
    smoke: bool,
) -> tuple[dict[str, Any], Mapping[Profile, np.ndarray]]:
    policy_ids = tuple(backend.policies)
    profiles = frozen_closure(tuple(distributions.values()), policy_ids)
    union_hash = sha256_bytes(
        canonical_json(
            {
                label: distribution.q_hash
                for label, distribution in sorted(distributions.items())
            }
        )
    )
    chunks = ProfileReplicationChunks(
        state_dir / "stages" / "formal_audit" / "chunks",
        identity=ReplicationChunkIdentity(
            campaign_sha256=campaign_hash,
            matrix_sha256=matrix_hash,
            job_id=job.job_id,
            # The campaign worker restores stages by this stable identity.
            # Any change in the frozen q union already changes the plan's
            # item ordering and is rejected by ChunkPlan.write_immutable().
            stage_id="formal_audit",
            kind="audit",
            chunk_size=AUDIT_CHUNK_SIZE,
        ),
        profiles=profiles,
    )
    chunks.evaluate_pending(backend, rollouts=rollouts, workers=workers)
    profile_returns = chunks.return_vectors(rollouts=rollouts)
    summaries: dict[str, Any] = {}
    for label, distribution in distributions.items():
        before_hash = distribution.q_hash
        samples = build_joint_audit_samples(
            profile_returns,
            distribution,
            policy_ids,
            n_agents=backend.n_agents,
            sample_count=rollouts,
        )
        bootstrap_seed = stable_solver_seed(
            "revision-full-v1", job.job_id, "formal-audit-shared-bootstrap"
        )
        summary = dict(
            summarize_audit_samples(
                samples,
                alpha=0.05,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
        )
        after_hash = distribution_hash(distribution.support, distribution.probabilities)
        if before_hash != after_hash:
            raise RuntimeError("Frozen CNC q changed during formal audit.")
        summaries[label] = {
            **summary,
            "q_hash_before": before_hash,
            "q_hash_after": after_hash,
            "joint_replication_samples_sha256": _audit_sample_hash(samples),
            "certified_low_gap_approximate_cce": bool(
                float(summary["max_t_relative_gap_ucb95_percent"])
                <= LOW_GAP_CERTIFICATION_PERCENT
            ),
        }
        if label == "core_q_under_expanded_library":
            summaries[label]["new_policy_replacement_gains"] = _new_policy_gain_summary(
                samples, summary
            )
    payload = {
        "rollouts": int(rollouts),
        "bootstrap_samples": int(bootstrap_samples),
        "profile_count": len(profiles),
        "q_union_sha256": union_hash,
        "raw_profile_return_vectors_sha256": _profile_vector_hash(
            profiles, policy_ids, profile_returns
        ),
        "distributions": summaries,
        "checkpoint": {
            "chunk_size": AUDIT_CHUNK_SIZE,
            "item_order_sha256": chunks.plan.item_order_sha256,
            "chunks": [record.to_payload() for record in chunks.store.validate_complete()],
            "common_replication_index_across_profiles": True,
        },
        "backend_identity": backend.cache_identity,
        "seeds": asdict(backend.seeds),
        "smoke": bool(smoke),
    }
    return payload, profile_returns


def _run_outcomes(
    *,
    job: RevisionJobKey,
    state_dir: Path,
    matrix_hash: str,
    campaign_hash: str,
    backend,
    distributions: Mapping[str, FrozenDistribution],
    rollouts: int,
    bootstrap_samples: int,
    workers: int,
) -> dict[str, Any]:
    policy_ids = tuple(backend.policies)
    profiles = tuple(
        sorted({profile for distribution in distributions.values() for profile in distribution.support})
    )
    union_hash = sha256_bytes(
        canonical_json(
            {
                label: distribution.q_hash
                for label, distribution in sorted(distributions.items())
            }
        )
    )
    chunks = ProfileReplicationChunks(
        state_dir / "stages" / "outcome_evaluation" / "chunks",
        identity=ReplicationChunkIdentity(
            campaign_sha256=campaign_hash,
            matrix_sha256=matrix_hash,
            job_id=job.job_id,
            stage_id="outcome_evaluation",
            kind="audit",
            chunk_size=AUDIT_CHUNK_SIZE,
        ),
        profiles=profiles,
    )
    chunks.evaluate_pending(
        backend,
        rollouts=rollouts,
        workers=workers,
        metric_names=CNC_RAW_OUTCOME_FIELDS,
    )
    profile_returns = chunks.return_vectors(rollouts=rollouts)
    profile_records = chunks.metric_vectors(
        rollouts=rollouts,
        metric_names=CNC_RAW_OUTCOME_FIELDS,
    )
    summaries: dict[str, Any] = {}
    for label, distribution in distributions.items():
        weighted = weighted_outcome_replications(
            profile_records,
            distribution.support,
            distribution.probabilities,
        )
        intervals = bootstrap_outcome_replications(
            weighted,
            n_manufacturers=backend.n_agents,
            alpha=0.05,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=stable_solver_seed(
                "revision-full-v1", job.job_id, "outcome-shared-bootstrap"
            ),
        )
        summaries[label] = {
            "q_hash": distribution.q_hash,
            "weighted_raw_vectors_sha256": _array_mapping_sha256(weighted),
            "metrics": {
                name: _json_number(asdict(interval))
                for name, interval in intervals.items()
            },
        }
    return {
        "rollouts": int(rollouts),
        "bootstrap_samples": int(bootstrap_samples),
        "profile_count": len(profiles),
        "q_union_sha256": union_hash,
        "raw_profile_vectors_sha256": _profile_vector_hash(
            profiles, policy_ids, profile_returns, profile_records
        ),
        "distributions": summaries,
        "checkpoint": {
            "chunk_size": AUDIT_CHUNK_SIZE,
            "item_order_sha256": chunks.plan.item_order_sha256,
            "chunks": [record.to_payload() for record in chunks.store.validate_complete()],
            "common_replication_index_across_profiles": True,
            "ratios_and_hhi_reconstructed_inside_each_bootstrap_draw": True,
        },
        "backend_identity": backend.cache_identity,
        "seeds": asdict(backend.seeds),
    }


def run_cnc_revision_pipeline(
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
        raise ValueError("CNC revision runner matrix hash mismatch.")
    job = _find_job(matrix, job_id)
    campaign_hash = _campaign_hash(matrix_hash, job.job_id, smoke=smoke)
    state_dir = Path(state_dir)
    output_dir = Path(output_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(workers))
    train_rollouts, audit_rollouts, outcome_rollouts, bootstrap_samples = _counts(smoke)
    condition, library_variant, robustness, _setting_id = _job_variants(job)

    identity = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job.job_id,
        "matrix_sha256": matrix_hash,
        "campaign_sha256": campaign_hash,
        "family": job.family,
        "mechanism": job.mechanism,
        "condition": condition,
        "seed": int(job.seed),
        "library_variant": library_variant.variant_id if library_variant else None,
        "robustness_setting": robustness.setting_id if robustness else None,
        "smoke": bool(smoke),
    }
    identity_path = state_dir / "runner_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("State directory belongs to a different CNC revision pipeline.")
    else:
        atomic_json(identity_path, identity)

    training_backend = _backend_for_stage(
        job,
        condition=condition,
        namespace="training",
        library_variant=library_variant,
        robustness=robustness,
        smoke=smoke,
    )
    audit_backend = _backend_for_stage(
        job,
        condition=condition,
        namespace="formal_audit",
        library_variant=library_variant,
        robustness=robustness,
        smoke=smoke,
    )
    outcome_backend = _backend_for_stage(
        job,
        condition=condition,
        namespace="outcome_evaluation",
        library_variant=library_variant,
        robustness=robustness,
        smoke=smoke,
    )
    _validate_stage_backends(training_backend, audit_backend, outcome_backend)
    distributions, training_payload = _run_or_load_training(
        job=job,
        state_dir=state_dir,
        matrix_hash=matrix_hash,
        campaign_hash=campaign_hash,
        backend=training_backend,
        train_rollouts=train_rollouts,
        workers=workers,
        smoke=smoke,
    )
    primary_label = "platform_operating_score"
    if primary_label not in distributions:
        raise RuntimeError("CNC pipeline did not return the platform-objective distribution.")
    primary = distributions[primary_label]

    audit_distributions = dict(distributions)
    core_dependency: dict[str, Any] | None = None
    if library_variant is not None and library_variant.variant_id == "expanded_a1_a8":
        dependency_root = Path(
            dependency_dir
            or os.environ.get("CMFG_DEPENDENCY_DIR", state_dir / "dependencies")
        )
        core_q, core_dependency = _load_core_dependency(
            dependency_root,
            job=job,
            condition=condition,
            matrix_hash=matrix_hash,
            expanded_policy_ids=tuple(audit_backend.policies),
        )
        audit_distributions["core_q_under_expanded_library"] = core_q

    audit_payload, _profile_returns = _run_formal_audit(
        job=job,
        state_dir=state_dir,
        matrix_hash=matrix_hash,
        campaign_hash=campaign_hash,
        backend=audit_backend,
        distributions=audit_distributions,
        rollouts=audit_rollouts,
        bootstrap_samples=bootstrap_samples,
        workers=workers,
        smoke=smoke,
    )
    outcome_payload = _run_outcomes(
        job=job,
        state_dir=state_dir,
        matrix_hash=matrix_hash,
        campaign_hash=campaign_hash,
        backend=outcome_backend,
        # The expanded-library job also evaluates the frozen core q on the
        # same +30M outcome stream, giving a paired behavioral/outcome
        # comparison in addition to the A7/A8 replacement-gain audit.
        distributions=audit_distributions,
        rollouts=outcome_rollouts,
        bootstrap_samples=bootstrap_samples,
        workers=workers,
    )

    result = {
        **identity,
        "status": "complete",
        "N": int(training_backend.n_agents),
        "J": len(training_backend.policies),
        "reported_N": int(job.n_agents),
        "reported_J": int(job.policies_per_agent),
        "variant": job.variant,
        "policy_ids": list(training_backend.policies),
        "train_rollouts": int(train_rollouts),
        "audit_rollouts": int(audit_rollouts),
        "outcome_rollouts": int(outcome_rollouts),
        "bootstrap_samples": int(bootstrap_samples),
        # Stable dependency contract used by the policy-transplant runner.
        "distribution": _distribution_payload(primary),
        "distributions": {
            label: _distribution_payload(distribution)
            for label, distribution in distributions.items()
        },
        "training": training_payload,
        "formal_audit": audit_payload,
        "outcome_evaluation": outcome_payload,
        "seed_isolation": {
            "manufacturer_population_held_fixed": True,
            "dynamic_streams_pairwise_disjoint": True,
            "formal_audit_offset": 20_000_000,
            "outcome_evaluation_offset": 30_000_000,
            "paired_common_random_numbers_across_profiles": True,
        },
        "worker_stage_contract": {
            "runner_managed_chunks": True,
            "stage_order": ["training", "formal_audit", "outcome_evaluation"],
            "training_chunk_size": TRAINING_CHUNK_SIZE,
            "audit_chunk_size": AUDIT_CHUNK_SIZE,
            "outcome_chunk_size": AUDIT_CHUNK_SIZE,
            "one_immutable_chunk_plan_per_stage": True,
        },
    }
    if core_dependency is not None:
        core_audit = audit_payload["distributions"]["core_q_under_expanded_library"]
        enrichment_steps = [
            {
                "from": "A1--A6",
                "to": "A1--A8",
                "gain_audit": core_audit["new_policy_replacement_gains"],
            },
        ]
        result["core_q_under_expanded_library_audit"] = {
            "dependency": core_dependency,
            "distribution": _distribution_payload(
                audit_distributions["core_q_under_expanded_library"]
            ),
            "audit": core_audit,
            "outcome_evaluation": outcome_payload["distributions"][
                "core_q_under_expanded_library"
            ],
            "expanded_q_outcome_evaluation": outcome_payload["distributions"][
                "platform_operating_score"
            ],
            "policy_richness_stop": {
                "gain_threshold_percent": LOW_GAP_CERTIFICATION_PERCENT,
                "enrichment_steps": enrichment_steps,
                "all_new_policy_gain_steps_pass": all(
                    bool(step["gain_audit"]["passes_two_percent_gain_stop"])
                    for step in enrichment_steps
                ),
                "mechanism_contrast_stability": (
                    "computed across matched jobs by the campaign reducer"
                ),
            },
        }
    if primary.q_hash != distribution_hash(primary.support, primary.probabilities):
        raise RuntimeError("Primary CNC distribution changed before publication output.")
    atomic_json(output_dir / "result.json", _json_number(result))
    return _json_number(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one revision-full-v1 CNC main, CCE-selection, policy-library, "
            "or parameter-robustness physical pipeline."
        )
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix-hash", required=True)
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("CMFG_WORKERS", "1"))
    )
    parser.add_argument("--dependency-dir", type=Path)
    # The generic worker passes these arguments.  This runner owns one plan per
    # dynamic stage, so outer profile bounds are intentionally ignored.
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--chunk-stop", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_cnc_revision_pipeline(
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
