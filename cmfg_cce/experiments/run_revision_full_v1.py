from __future__ import annotations

"""Physical runner for the aggregate-game part of ``revision-full-v1``.

The campaign manifest gives this module one immutable physical job at a time.
This file owns the benchmark, mixed-support challenge, and exact/sparse
scalability pipelines.  CNC, transplant, and sensitivity pipelines are
dispatched elsewhere; MWU calibration has its own runner.

Two execution modes are deliberately different:

* bulk jobs freeze replication-level training and audit vectors in resumable
  profile chunks;
* runtime jobs start every solver attempt from an empty payoff cache and only
  commit an attempt after it has finished.  An interrupted attempt is never
  stitched into a later wall-clock measurement.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cmfg_cce.baselines.mwu_tuning import MwuTuningConfig
from cmfg_cce.baselines.simple import mwu_policy_trace
from cmfg_cce.evaluation.backend_cache import BackendPayoffCache
from cmfg_cce.evaluation.empirical_game import (
    EmpiricalGame,
    build_empirical_game,
    enumerate_profiles,
)
from cmfg_cce.evaluation.full_audit import audit_sparse_result_on_full_game
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
from cmfg_cce.evaluation.revision_statistics import (
    pure_nash_diagnostics,
    support_diversity,
)
from cmfg_cce.evaluation.rollout import Profile
from cmfg_cce.experiments.common import run_solver
from cmfg_cce.experiments.revision_backends import build_benchmark_backend
from cmfg_cce.experiments.revision_full_v1_spec import (
    FULL_SOLVER_BUNDLE,
    SPARSE_SOLVER_BUNDLE,
    RevisionFullV1Matrix,
    RevisionJobKey,
)
from cmfg_cce.experiments.revision_pipeline import (
    DEFAULT_EPSILON_TOLERANCE,
    DEFAULT_SELECTOR,
    default_dss_config,
    frozen_closure,
    run_dss_empty_cache,
    run_mwu_empty_cache,
    run_mwu_rounds_empty_cache,
    stable_solver_seed,
)
from cmfg_cce.experiments.regret_matching_stress import (
    InMemoryEmpiricalGameCache,
    build_rps_cycle,
    build_successor_ring,
    has_zero_gap_pure_profile,
    run_dss_on_game,
)
from cmfg_cce.solvers.cce_lp import CceSolution, solve_full_cce_lp
from cmfg_cce.solvers.cg_cce import solve_cg_cce_exhaustive
from cmfg_cce.solvers.sparse_cce import SparseCceResult
from cmfg_cce.orchestration.chunks import deterministic_npz
from cmfg_cce.orchestration.manifest import (
    atomic_json,
    canonical_json,
    sha256_bytes,
    sha256_file,
)


SCHEMA_VERSION = "revision_full_v1_general_physical_runner_v1"
RUNTIME_SCHEMA_VERSION = "revision_full_v1_atomic_runtime_result_v1"
TRAINING_CHUNK_SIZE = 128
AUDIT_CHUNK_SIZE = 8
PRODUCTION_TRAIN_ROLLOUTS = 200
PRODUCTION_AUDIT_ROLLOUTS = 2000
PRODUCTION_BOOTSTRAP_SAMPLES = 5000
SMOKE_TRAIN_ROLLOUTS = 2
SMOKE_AUDIT_ROLLOUTS = 4
SMOKE_BOOTSTRAP_SAMPLES = 100
LOW_GAP_CERTIFICATE_PERCENT = 2.0

GENERAL_BULK_FAMILIES = {
    "solver_benchmark",
    "mixed_challenge",
    "scalability_exact",
    "scalability_sparse",
}
GENERAL_RUNTIME_FAMILIES = {
    "solver_benchmark_runtime",
    "scalability_exact_runtime",
    "scalability_sparse_runtime",
}
GENERAL_FAMILIES = GENERAL_BULK_FAMILIES | GENERAL_RUNTIME_FAMILIES


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _require_sha256(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{label} must be a lowercase SHA256 digest.")
    return normalized


def _campaign_hash(matrix_hash: str, job_id: str, *, smoke: bool) -> str:
    value = os.environ.get("CMFG_CAMPAIGN_SHA256", "")
    if value:
        return _require_sha256(value, "CMFG_CAMPAIGN_SHA256")
    if not smoke:
        raise ValueError("CMFG_CAMPAIGN_SHA256 is required outside --smoke mode.")
    return sha256_bytes(f"smoke:{matrix_hash}:{job_id}".encode("utf-8"))


def _general_jobs(matrix: RevisionFullV1Matrix) -> tuple[RevisionJobKey, ...]:
    return tuple(job for job in matrix.physical_jobs() if job.family in GENERAL_FAMILIES)


def find_general_physical_job(
    matrix: RevisionFullV1Matrix,
    job_id: str,
) -> RevisionJobKey:
    """Resolve one exact manifest job and reject conceptual/non-general IDs."""

    matches = [job for job in _general_jobs(matrix) if job.job_id == str(job_id)]
    if len(matches) != 1:
        raise ValueError(
            "job-id must identify exactly one frozen benchmark, mixed-challenge, "
            f"or scalability physical job; found {len(matches)} matches for {job_id!r}."
        )
    return matches[0]


def _selected_config_from_payload(
    payload: Mapping[str, Any],
    *,
    matrix: RevisionFullV1Matrix,
) -> MwuTuningConfig:
    if payload.get("schema_version") != "revision_full_v1_global_mwu_selection_v2":
        raise ValueError("Unsupported MWU-selection schema.")
    if payload.get("status") != "complete" or payload.get("matrix_sha256") != matrix.matrix_hash:
        raise ValueError("MWU selection is incomplete or belongs to a different matrix.")
    observed_hash = str(payload.get("selection_sha256", ""))
    unsigned = dict(payload)
    unsigned.pop("selection_sha256", None)
    if observed_hash != sha256_bytes(canonical_json(unsigned)):
        raise ValueError("MWU-selection hash mismatch.")
    selected = dict(payload.get("selected_config", {}))
    try:
        candidate = MwuTuningConfig(
            eta=float(selected["eta"]),
            schedule=str(selected["schedule"]),
            exploration_floor=float(selected["exploration_floor"]),
            burn_in_rounds=int(selected["burn_in_rounds"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("MWU selection has no complete selected_config.") from exc
    if selected.get("mwu_config_id") != candidate.config_id:
        raise ValueError("MWU selected config ID does not match its parameter values.")
    frozen = {config.config_id: config for config in matrix.mwu_tuning_configs}
    if candidate.config_id not in frozen or frozen[candidate.config_id] != candidate:
        raise ValueError("MWU selection is not one of the frozen sixteen configurations.")
    return candidate


def load_global_mwu_selection(
    path: str | Path | None,
    *,
    matrix: RevisionFullV1Matrix,
    smoke: bool,
) -> tuple[MwuTuningConfig, dict[str, Any]]:
    """Load the one globally tuned configuration, never a per-game choice."""

    selected_path = path or os.environ.get("CMFG_MWU_SELECTION_PATH")
    if selected_path is None:
        if not smoke:
            raise ValueError(
                "--mwu-selection or CMFG_MWU_SELECTION_PATH is required for formal jobs."
            )
        config = matrix.mwu_tuning_configs[0]
        return config, {
            "source": "smoke_default_only",
            "mwu_config_id": config.config_id,
            **config.to_solver_mapping(),
        }
    source = Path(selected_path)
    if not source.is_file():
        raise FileNotFoundError(f"MWU-selection JSON does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    config = _selected_config_from_payload(payload, matrix=matrix)
    selected = dict(payload.get("selected_config", {}))
    if "formal_rounds" in selected or "formal_rounds_rule" in selected:
        raise ValueError("MWU selection must freeze hyperparameters only.")
    return config, {
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "selection_sha256": str(payload["selection_sha256"]),
        "mwu_config_id": config.config_id,
        **config.to_solver_mapping(),
    }


def _counts(smoke: bool) -> tuple[int, int, int]:
    if smoke:
        return SMOKE_TRAIN_ROLLOUTS, SMOKE_AUDIT_ROLLOUTS, SMOKE_BOOTSTRAP_SAMPLES
    return (
        PRODUCTION_TRAIN_ROLLOUTS,
        PRODUCTION_AUDIT_ROLLOUTS,
        PRODUCTION_BOOTSTRAP_SAMPLES,
    )


def _variant_for_backend(job: RevisionJobKey) -> str:
    return job.variant if job.family == "mixed_challenge" else "base"


def _general_solver_seed(job: RevisionJobKey, solver: str) -> int:
    """Use the same solver randomization in evidence and runtime jobs."""

    if "scalability_exact" in job.family:
        experiment = "scalability_exact"
    elif "scalability_sparse" in job.family:
        experiment = "scalability_sparse"
    elif "solver_benchmark" in job.family:
        experiment = "solver_benchmark"
    else:
        experiment = job.family
    return stable_solver_seed(
        "revision-full-v1",
        experiment,
        job.n_agents,
        job.policies_per_agent,
        job.mechanism,
        job.seed,
        solver,
    )


def _backend_for_stage(job: RevisionJobKey, namespace: str, *, smoke: bool):
    backend = build_benchmark_backend(
        mechanism=job.mechanism,
        n_agents=2 if smoke else job.n_agents,
        policies_per_agent=2 if smoke else job.policies_per_agent,
        seed=job.seed,
        namespace=namespace,
        variant=_variant_for_backend(job),
        stream_label=f"revision-full-v1:{job.family}:{namespace}",
    )
    if smoke:
        backend.config = replace(backend.config, horizon=2)
    return backend


def _dynamic_seed_values(backend: Any) -> set[int]:
    return {
        int(backend.seeds.order_seed),
        int(backend.seeds.outside_seed),
        int(backend.seeds.availability_seed),
        int(backend.seeds.tie_break_seed),
        int(backend.seeds.rollout_replication_seed),
    }


def _validate_training_audit_backends(training: Any, audit: Any) -> None:
    if tuple(training.policies) != tuple(audit.policies) or training.n_agents != audit.n_agents:
        raise RuntimeError("Training and formal-audit policy games are incompatible.")
    if int(training.seeds.type_seed) != int(audit.seeds.type_seed):
        raise RuntimeError("Formal audit changed the frozen manufacturer population.")
    if _dynamic_seed_values(training).intersection(_dynamic_seed_values(audit)):
        raise RuntimeError("Training and +20,000,000 formal-audit streams overlap.")


def _smoke_dss_config(*, train_rollouts: int, workers: int) -> dict[str, Any]:
    config = default_dss_config(train_rollouts=train_rollouts, workers=workers)
    config.update(
        {
            "initial_support_size": 1,
            "max_support_size": 4,
            "support_add_batch_size": 1,
            "max_rounds": 1,
        }
    )
    config["repair"] = {
        **dict(config.get("repair", {})),
        "repair_rounds": 0,
        "profile_budget_multiplier": 1.0,
    }
    return config


def _freeze_result(
    result: CceSolution | SparseCceResult,
    *,
    n_agents: int,
    policy_ids: Sequence[str],
) -> FrozenDistribution:
    return freeze_distribution(
        result.solver,
        result.support_profiles,
        result.support_probabilities,
        n_agents,
        policy_ids,
    )


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
    value = freeze_distribution(
        str(payload["solver"]),
        [tuple(str(item) for item in profile) for profile in payload["support"]],
        [float(value) for value in payload["probabilities"]],
        n_agents,
        policy_ids,
    )
    if value.q_hash != payload.get("q_hash"):
        raise ValueError("Stored frozen distribution has an invalid q hash.")
    return value


def _solver_payload(
    result: CceSolution | SparseCceResult,
    distribution: FrozenDistribution,
    *,
    runtime_seconds: float,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    diversity = support_diversity(distribution.probabilities)
    return {
        "solver": result.solver,
        "distribution": _distribution_payload(distribution),
        "training_cce_gap_nominal": float(result.cce_gap_nominal),
        "training_cce_gap_ucb": float(result.cce_gap_ucb),
        "training_objective_value": float(result.objective_value),
        "runtime_seconds": float(runtime_seconds),
        "certificate_status": str(
            getattr(
                result,
                "certificate_status",
                "full_tensor_two_stage_lp"
                if result.full_optimality_certified
                else "fresh_formal_audit_required",
            )
        ),
        "pricing_mode": (
            None if result.pricing_mode is None else str(result.pricing_mode)
        ),
        "full_optimality_certified": bool(result.full_optimality_certified),
        "support_diversity": asdict(diversity),
        "max_deviation": _jsonable(result.max_deviation or {}),
        "diagnostics": _jsonable(result.diagnostics),
        "training_access": _jsonable(stats),
    }


def _checkpoint_summary(chunks: ProfileReplicationChunks, *, rollouts: int) -> dict[str, Any]:
    records = [record.to_payload() for record in chunks.store.validate_complete()]
    return {
        "chunk_size": int(chunks.identity.chunk_size),
        "profile_count": len(chunks.profiles),
        "rollouts": int(rollouts),
        "item_order_sha256": chunks.plan.item_order_sha256,
        "chunks": records,
    }


def _complete_training_stage(
    *,
    job: RevisionJobKey,
    state_dir: Path,
    campaign_hash: str,
    matrix_hash: str,
    backend: Any,
    profiles: Sequence[Profile],
    train_rollouts: int,
    workers: int,
) -> tuple[
    ProfileReplicationChunks,
    dict[Profile, np.ndarray],
    dict[Profile, np.ndarray],
]:
    chunks = ProfileReplicationChunks(
        state_dir / "stages" / "training" / "chunks",
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
    returns = chunks.return_vectors(rollouts=train_rollouts)
    metrics = chunks.metric_vectors(
        rollouts=train_rollouts,
        metric_names=("platform_operating_score",),
    )
    objectives = {
        profile: values["platform_operating_score"] for profile, values in metrics.items()
    }
    return chunks, returns, objectives


def _solve_exact_training(
    game: EmpiricalGame,
    backend: Any,
    *,
    train_rollouts: int,
    workers: int,
    job: RevisionJobKey,
    mwu_config: MwuTuningConfig,
    smoke: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, FrozenDistribution]]:
    records: dict[str, dict[str, Any]] = {}
    distributions: dict[str, FrozenDistribution] = {}
    policy_ids = tuple(game.policy_ids)

    started = time.perf_counter()
    full = solve_full_cce_lp(
        game,
        selector=DEFAULT_SELECTOR,
        epsilon_tolerance=DEFAULT_EPSILON_TOLERANCE,
    )
    full_runtime = time.perf_counter() - started
    full_distribution = _freeze_result(
        full, n_agents=game.n_agents, policy_ids=policy_ids
    )
    records[full.solver] = _solver_payload(
        full,
        full_distribution,
        runtime_seconds=full_runtime,
        stats={"shared_complete_training_table": True, "profile_count": game.n_profiles},
    )
    distributions[full.solver] = full_distribution

    started = time.perf_counter()
    cg = solve_cg_cce_exhaustive(
        game,
        selector=DEFAULT_SELECTOR,
        epsilon_tolerance=DEFAULT_EPSILON_TOLERANCE,
    )
    cg_runtime = time.perf_counter() - started
    # Usually both exact methods return the same selected q, in which case
    # the downstream audit is automatically deduplicated by q hash.  Exact
    # objective equivalence does not mathematically guarantee a unique q,
    # however, so retain and independently audit CG's own distribution when
    # the common two-stage face is degenerate.
    cg_distribution = _freeze_result(
        cg, n_agents=game.n_agents, policy_ids=policy_ids
    )
    exact_q_shared = bool(cg_distribution.q_hash == full_distribution.q_hash)
    records[cg.solver] = _solver_payload(
        cg,
        cg_distribution,
        runtime_seconds=cg_runtime,
        stats={
            "shared_complete_training_table": True,
            "profile_count": game.n_profiles,
            "same_selected_q_as_full_tensor": exact_q_shared,
        },
    )
    distributions[cg.solver] = cg_distribution

    dss_seed = _general_solver_seed(job, "REPAIR-SAD-CCE")
    if smoke:
        dss_cache = BackendPayoffCache(
            backend=backend, n_rollouts=train_rollouts, workers=workers
        )
        dss_cache.reset_access_log()
        started = time.perf_counter()
        dss = run_solver(
            "REPAIR-SAD-CCE",
            dss_cache,
            policy_ids,
            _smoke_dss_config(train_rollouts=train_rollouts, workers=workers),
            seed=dss_seed,
            metadata={"revision_campaign": "revision-full-v1-smoke"},
        )
        dss_runtime = time.perf_counter() - started
        dss_stats = dss_cache.access_stats()
        dss_stats.update(
            {
                "runtime_seconds": dss_runtime,
                "payoff_evaluation_seconds": dss_cache.eval_time_seconds,
                "solver_compute_seconds": max(
                    0.0, dss_runtime - dss_cache.eval_time_seconds
                ),
            }
        )
    else:
        dss, dss_cache, dss_stats = run_dss_empty_cache(
            backend,
            train_rollouts=train_rollouts,
            workers=workers,
            solver_seed=dss_seed,
        )
        dss_runtime = float(dss_stats["runtime_seconds"])
    dss_distribution = _freeze_result(
        dss, n_agents=game.n_agents, policy_ids=policy_ids
    )
    dss_stats["same_replication_stream_as_complete_table"] = True
    records[dss.solver] = _solver_payload(
        dss,
        dss_distribution,
        runtime_seconds=dss_runtime,
        stats=dss_stats,
    )
    distributions[dss.solver] = dss_distribution

    mwu_seed = _general_solver_seed(job, "MWU-PolicyTrace")
    mixed_rounds = 100 if smoke else 2000
    mwu, mwu_cache, mwu_stats = run_mwu_rounds_empty_cache(
        backend,
        train_rollouts=train_rollouts,
        workers=workers,
        solver_seed=mwu_seed,
        rounds=mixed_rounds,
        eta=mwu_config.eta,
        schedule=mwu_config.schedule,
        exploration_floor=mwu_config.exploration_floor,
        burn_in_rounds=mwu_config.burn_in_rounds,
    )
    mwu_runtime = float(mwu_stats["runtime_seconds"])
    mwu_distribution = _freeze_result(
        mwu, n_agents=game.n_agents, policy_ids=policy_ids
    )
    mwu_stats.update(
        {
            "same_replication_stream_as_complete_table": True,
            "fixed_rounds_for_mixed_support_analysis": True,
            "formal_rounds": mixed_rounds,
            "completed_trace_rounds": int(
                mwu.diagnostics.get("trace_rounds", 0)
            ),
            "formal_rounds_are_reproduction_provenance": True,
        }
    )
    records[mwu.solver] = _solver_payload(
        mwu,
        mwu_distribution,
        runtime_seconds=mwu_runtime,
        stats=mwu_stats,
    )
    distributions[mwu.solver] = mwu_distribution
    return records, distributions


def _pure_training_summary(game: EmpiricalGame) -> dict[str, Any]:
    diagnostics = pure_nash_diagnostics(game)
    gains = np.asarray(tuple(diagnostics.profile_maximum_gains.values()), dtype=float)
    policy_counts = {
        agent: {policy: 0 for policy in game.policy_ids}
        for agent in range(game.n_agents)
    }
    for profile in diagnostics.pure_nash_profiles:
        for agent, policy in enumerate(profile):
            policy_counts[agent][policy] += 1
    pure_count = len(diagnostics.pure_nash_profiles)
    return {
        "complete_table": True,
        "pure_nash_count": pure_count,
        "pure_nash_profiles": [list(profile) for profile in diagnostics.pure_nash_profiles],
        "pure_profile_rows": [
            {
                "profile": list(profile),
                "raw_maximum_replacement_gain": float(
                    diagnostics.profile_raw_maximum_gains[profile]
                ),
                "stability_margin": max(
                    0.0, -float(diagnostics.profile_raw_maximum_gains[profile])
                ),
                "best_replacement": (
                    None
                    if diagnostics.profile_best_replacements[profile] is None
                    else {
                        "agent": int(diagnostics.profile_best_replacements[profile][0]),
                        "policy": str(diagnostics.profile_best_replacements[profile][1]),
                    }
                ),
            }
            for profile in diagnostics.pure_nash_profiles
        ],
        "pure_policy_composition": {
            str(agent): {
                policy: {
                    "count": int(count),
                    "share": (0.0 if pure_count == 0 else float(count / pure_count)),
                }
                for policy, count in counts.items()
            }
            for agent, counts in policy_counts.items()
        },
        "weak_dominance_pairs": [list(value) for value in diagnostics.weak_dominance_pairs],
        "weak_dominance_margin_rows": [
            {
                "agent": int(agent),
                "dominating_policy": dominating,
                "dominated_policy": dominated,
                "minimum_payoff_advantage": margins[0],
                "mean_payoff_advantage": margins[1],
                "maximum_payoff_advantage": margins[2],
            }
            for (agent, dominating, dominated), margins in diagnostics.weak_dominance_margins.items()
        ],
        "weakly_dominated_policies": {
            str(agent): list(values)
            for agent, values in diagnostics.weakly_dominated_policies.items()
        },
        "profile_maximum_gain_summary": {
            "minimum": float(np.min(gains)),
            "median": float(np.median(gains)),
            "maximum": float(np.max(gains)),
        },
    }


def deterministic_mixed_support_checks(
    mwu_config: MwuTuningConfig,
    *,
    mwu_rounds: int = 2000,
) -> dict[str, Any]:
    """Run the two predeclared, noise-free cyclic-game sanity checks.

    These checks are algorithm diagnostics, not extra CMfg cases.  They prove
    that the common two-stage selector can return genuinely mixed CCEs when a
    pure equilibrium does not exist, and that both sparse output formats can
    be subjected to an exact unilateral-replacement audit.
    """

    rows: list[dict[str, Any]] = []
    for offset, spec in enumerate((build_rps_cycle(), build_successor_ring())):
        game = spec.game
        full = solve_full_cce_lp(
            game,
            selector=DEFAULT_SELECTOR,
            epsilon_tolerance=DEFAULT_EPSILON_TOLERANCE,
        )
        dss, _runtime, _stats = run_dss_on_game(game, seed=7300 + offset)
        dss_audit = audit_sparse_result_on_full_game(game, dss)
        mwu_cache = InMemoryEmpiricalGameCache(game, mechanism_id="synthetic")
        mwu = mwu_policy_trace(
            mwu_cache,
            game.policy_ids,
            rounds=int(mwu_rounds),
            seed=8300 + offset,
            eta=mwu_config.eta,
            schedule=mwu_config.schedule,
            exploration_floor=mwu_config.exploration_floor,
            burn_in_rounds=mwu_config.burn_in_rounds,
        )
        mwu_audit = audit_sparse_result_on_full_game(game, mwu)
        rows.append(
            {
                "game": spec.name,
                "description": spec.description,
                "has_zero_gap_pure_profile": has_zero_gap_pure_profile(game),
                "two_stage_full_lp": {
                    "selector": DEFAULT_SELECTOR,
                    "support_size": int(full.support_size),
                    "gap": float(full.cce_gap_nominal),
                    "mixed_support": bool(full.support_size > 1),
                },
                "dss_exact_full_game_audit": {
                    "support_size": int(dss_audit.support_size),
                    "gap": float(dss_audit.cce_gap_nominal),
                    "audited": True,
                },
                "mwu_exact_full_game_audit": {
                    "rounds": int(mwu_rounds),
                    "support_size": int(mwu_audit.support_size),
                    "gap": float(mwu_audit.cce_gap_nominal),
                    "audited": True,
                },
            }
        )
    if any(row["has_zero_gap_pure_profile"] for row in rows) or any(
        not row["two_stage_full_lp"]["mixed_support"] for row in rows
    ):
        raise RuntimeError("A deterministic cyclic-game mixed-support invariant failed.")
    return {
        "role": "deterministic_algorithm_sanity_checks_not_CMfg_case_selection",
        "all_predeclared_games_reported": True,
        "games": rows,
    }


def _audit_sample_arrays(
    samples_by_hash: Mapping[str, AuditSampleMatrix],
    policy_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    policy_index = {str(policy): index for index, policy in enumerate(policy_ids)}
    arrays: dict[str, np.ndarray] = {}
    for q_hash, samples in sorted(samples_by_hash.items()):
        prefix = f"q_{q_hash[:16]}"
        arrays[f"{prefix}_gain_samples"] = np.asarray(samples.gain_samples, dtype=float)
        arrays[f"{prefix}_q_return_samples"] = np.asarray(
            samples.q_return_samples, dtype=float
        )
        arrays[f"{prefix}_label_agent"] = np.asarray(
            [agent for agent, _policy in samples.labels], dtype=np.int16
        )
        arrays[f"{prefix}_label_policy_index"] = np.asarray(
            [policy_index[policy] for _agent, policy in samples.labels], dtype=np.int16
        )
    return arrays


def _run_formal_audit(
    *,
    job: RevisionJobKey,
    state_dir: Path,
    output_dir: Path,
    campaign_hash: str,
    matrix_hash: str,
    training_backend: Any,
    distributions: Mapping[str, FrozenDistribution],
    audit_rollouts: int,
    bootstrap_samples: int,
    workers: int,
    smoke: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    audit_backend = _backend_for_stage(job, "formal_audit", smoke=smoke)
    _validate_training_audit_backends(training_backend, audit_backend)
    policy_ids = tuple(audit_backend.policies)
    closure = frozen_closure(tuple(distributions.values()), policy_ids)
    chunks = ProfileReplicationChunks(
        state_dir / "stages" / "formal_audit" / "chunks",
        identity=ReplicationChunkIdentity(
            campaign_sha256=campaign_hash,
            matrix_sha256=matrix_hash,
            job_id=job.job_id,
            stage_id="formal_audit",
            kind="audit",
            chunk_size=AUDIT_CHUNK_SIZE,
        ),
        profiles=closure,
    )
    chunks.evaluate_pending(
        audit_backend,
        rollouts=audit_rollouts,
        workers=workers,
    )
    returns = chunks.return_vectors(rollouts=audit_rollouts)

    samples_by_hash: dict[str, AuditSampleMatrix] = {}
    summary_by_hash: dict[str, dict[str, Any]] = {}
    for distribution in distributions.values():
        if distribution.q_hash in samples_by_hash:
            continue
        before = distribution.q_hash
        samples = build_joint_audit_samples(
            returns,
            distribution,
            policy_ids,
            n_agents=audit_backend.n_agents,
            sample_count=audit_rollouts,
        )
        summary = dict(
            summarize_audit_samples(
                samples,
                alpha=0.05,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=stable_solver_seed(
                    "revision-full-v1", job.job_id, "formal-bootstrap"
                ),
            )
        )
        after = distribution_hash(distribution.support, distribution.probabilities)
        if before != after:
            raise RuntimeError("Frozen q changed during independent formal audit.")
        summary.update(
            {
                "q_hash_before": before,
                "q_hash_after": after,
                "certified_low_gap_approximate_cce": bool(
                    summary["max_t_relative_gap_ucb95_percent"]
                    <= LOW_GAP_CERTIFICATE_PERCENT
                ),
                "certificate_threshold_percent": LOW_GAP_CERTIFICATE_PERCENT,
            }
        )
        samples_by_hash[distribution.q_hash] = samples
        summary_by_hash[distribution.q_hash] = summary

    arrays = _audit_sample_arrays(samples_by_hash, policy_ids)
    artifact_bytes = deterministic_npz(arrays)
    artifact_path = output_dir / "formal_audit_samples.npz"
    _atomic_bytes(artifact_path, artifact_bytes)
    artifact = {
        "path": artifact_path.name,
        "sha256": sha256_bytes(artifact_bytes),
        "q_count": len(samples_by_hash),
        "common_replication_index_across_profiles_and_q": True,
    }
    results: dict[str, dict[str, Any]] = {}
    for solver, distribution in distributions.items():
        summary = dict(summary_by_hash[distribution.q_hash])
        summary["audit_sample_group"] = f"q_{distribution.q_hash[:16]}"
        results[solver] = summary
    return results, {
        **_checkpoint_summary(chunks, rollouts=audit_rollouts),
        "backend_identity": audit_backend.cache_identity,
        "seeds": asdict(audit_backend.seeds),
        "namespace_offset": 20_000_000,
        "raw_replication_vectors": {
            "location": "stages/formal_audit/chunks",
            "array": "returns",
            "shape_per_profile": [audit_rollouts, audit_backend.n_agents],
            "common_replication_index_across_profiles": True,
        },
        "sample_artifact": artifact,
    }


def _write_identity(
    state_dir: Path,
    *,
    job: RevisionJobKey,
    matrix_hash: str,
    campaign_hash: str,
    smoke: bool,
) -> dict[str, Any]:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job.job_id,
        "matrix_sha256": matrix_hash,
        "campaign_sha256": campaign_hash,
        "family": job.family,
        "smoke": bool(smoke),
    }
    path = state_dir / "runner_identity.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError("State directory belongs to a different physical job.")
    else:
        atomic_json(path, identity)
    return identity


def _paired_runtime_job(
    matrix: RevisionFullV1Matrix,
    evidence_job: RevisionJobKey,
) -> RevisionJobKey:
    family_map = {
        "solver_benchmark": "solver_benchmark_runtime",
        "scalability_exact": "scalability_exact_runtime",
        "scalability_sparse": "scalability_sparse_runtime",
    }
    runtime_family = family_map.get(evidence_job.family)
    if runtime_family is None:
        raise ValueError(f"{evidence_job.family} has no paired formal runtime-q job.")
    matches = [
        candidate
        for candidate in matrix.physical_jobs()
        if candidate.family == runtime_family
        and candidate.n_agents == evidence_job.n_agents
        and candidate.policies_per_agent == evidence_job.policies_per_agent
        and candidate.mechanism == evidence_job.mechanism
        and candidate.seed == evidence_job.seed
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one paired runtime-q job for {evidence_job.job_id}; found {len(matches)}."
        )
    return matches[0]


def _load_runtime_q_source(
    *,
    matrix: RevisionFullV1Matrix,
    job: RevisionJobKey,
    dependency_dir: Path,
    campaign_hash: str,
    policy_ids: Sequence[str],
    n_agents: int,
    mwu_selection: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, FrozenDistribution],
    dict[str, Any],
]:
    runtime_job = _paired_runtime_job(matrix, job)
    result_path = (
        Path(dependency_dir)
        / runtime_job.job_id
        / "artifacts"
        / "result.json"
    )
    if not result_path.is_file():
        raise FileNotFoundError(
            f"Paired runtime-q artifact is missing for {job.job_id}: {result_path}."
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    expected_identity = {
        "status": "complete",
        "job_id": runtime_job.job_id,
        "matrix_sha256": matrix.matrix_hash,
        "campaign_sha256": campaign_hash,
        "family": runtime_job.family,
        "reported_N": job.n_agents,
        "reported_J": job.policies_per_agent,
        "mechanism": job.mechanism,
        "seed": job.seed,
    }
    observed_identity = {name: payload.get(name) for name in expected_identity}
    if observed_identity != expected_identity:
        raise ValueError(
            f"Paired runtime-q identity mismatch for {job.job_id}: {observed_identity!r}."
        )
    runtime_selection = dict(payload.get("mwu_selection", {}))
    if (
        runtime_selection.get("selection_sha256")
        != mwu_selection.get("selection_sha256")
        or runtime_selection.get("mwu_config_id")
        != mwu_selection.get("mwu_config_id")
    ):
        raise ValueError("Runtime q used a different frozen MWU hyperparameter selection.")
    contract = dict(payload.get("runtime_contract", {}))
    required_contract = {
        "empty_cache_per_solver": True,
        "interrupted_attempts_discarded": True,
        "attempts_stitched": False,
        "verification_runtime_included": False,
        "runtime_job_is_formal_q_source": True,
        "evidence_jobs_must_restore_q_without_resolving": True,
        "mwu_rounds_recorded_only_as_q_reproduction_provenance": True,
    }
    if {name: contract.get(name) for name in required_contract} != required_contract:
        raise ValueError("Paired runtime-q artifact violates the frozen runtime contract.")

    raw_results = payload.get("solver_results")
    if not isinstance(raw_results, Mapping) or set(raw_results) != set(job.solvers):
        raise ValueError("Paired runtime-q solver bundle does not match the evidence job.")
    distributions: dict[str, FrozenDistribution] = {}
    records: dict[str, dict[str, Any]] = {}
    for solver in job.solvers:
        raw = dict(raw_results[solver])
        if raw.get("q_generated_by_runtime_job") is not True:
            raise ValueError(f"Runtime artifact does not identify {solver} q as runtime-generated.")
        distribution = _distribution_from_payload(
            dict(raw.get("distribution", {})),
            n_agents=n_agents,
            policy_ids=policy_ids,
        )
        if distribution.solver != solver:
            raise ValueError(f"Runtime distribution solver mismatch for {solver}.")
        diversity = support_diversity(distribution.probabilities)
        access = dict(raw.get("access_stats", {}))
        if not access:
            access = {
                "solver_required_profile_count": int(raw["training_profile_count"]),
                "rollout_episode_total": int(raw["training_rollout_episode_count"]),
            }
        elif "solver_required_profile_count" not in access:
            access["solver_required_profile_count"] = int(raw["training_profile_count"])
        records[solver] = {
            **raw,
            "solver": solver,
            "support_diversity": asdict(diversity),
            "training_access": access,
            "runtime_q_source_job_id": runtime_job.job_id,
            "runtime_q_source_sha256": sha256_file(result_path),
            "q_restored_without_resolving": True,
        }
        distributions[solver] = distribution

    paired = dict(payload.get("paired_dss_mwu", {}))
    dss_runtime = float(paired.get("dss_runtime_seconds", float("nan")))
    mwu_budget = float(paired.get("mwu_budget_seconds", float("nan")))
    mwu_runtime = float(paired.get("mwu_runtime_seconds", float("nan")))
    ratio = float(paired.get("mwu_to_dss_runtime_ratio", float("nan")))
    completed_rounds = int(paired.get("mwu_completed_trace_rounds", 0))
    if not (
        np.isfinite(dss_runtime)
        and dss_runtime >= 0.0
        and np.isfinite(mwu_budget)
        and mwu_budget == dss_runtime
        and np.isfinite(mwu_runtime)
        and mwu_runtime >= 0.0
        and np.isfinite(ratio)
        and completed_rounds > 0
        and int(records["MWU-PolicyTrace"].get("formal_rounds", 0))
        == completed_rounds
    ):
        raise ValueError("Paired DSS/MWU runtime provenance is incomplete or inconsistent.")
    provenance = {
        "runtime_job_id": runtime_job.job_id,
        "runtime_result_sha256": sha256_file(result_path),
        "q_hashes": {
            solver: distribution.q_hash for solver, distribution in distributions.items()
        },
        "dss_runtime_seconds": dss_runtime,
        "mwu_budget_seconds": mwu_budget,
        "mwu_runtime_seconds": mwu_runtime,
        "mwu_to_dss_runtime_ratio": ratio,
        "mwu_completed_trace_rounds": completed_rounds,
        "formal_rounds_are_reproduction_provenance_only": True,
    }
    return records, distributions, provenance


def run_general_bulk_pipeline(
    *,
    job: RevisionJobKey,
    state_dir: Path,
    output_dir: Path,
    matrix: RevisionFullV1Matrix,
    matrix_hash: str,
    campaign_hash: str,
    workers: int,
    mwu_config: MwuTuningConfig,
    mwu_selection: Mapping[str, Any],
    dependency_dir: Path | None,
    smoke: bool,
) -> dict[str, Any]:
    if job.family not in GENERAL_BULK_FAMILIES:
        raise ValueError(f"{job.family} is not a general bulk family.")
    train_rollouts, audit_rollouts, bootstrap_samples = _counts(smoke)
    training_backend = _backend_for_stage(job, "training", smoke=smoke)
    policy_ids = tuple(training_backend.policies)
    complete_table = job.family != "scalability_sparse"
    paired_runtime_family = job.family in {
        "solver_benchmark",
        "scalability_exact",
        "scalability_sparse",
    }
    if paired_runtime_family and dependency_dir is None:
        raise ValueError(
            f"Evidence job {job.job_id} requires its paired runtime-q dependency."
        )
    use_runtime_q = paired_runtime_family and dependency_dir is not None
    runtime_q_provenance: dict[str, Any] | None = None
    synthetic_checks = (
        deterministic_mixed_support_checks(
            mwu_config,
            mwu_rounds=100 if smoke else 2000,
        )
        if job.family == "mixed_challenge"
        else None
    )

    if complete_table:
        profiles = enumerate_profiles(policy_ids, training_backend.n_agents)
        training_chunks, returns, objectives = _complete_training_stage(
            job=job,
            state_dir=state_dir,
            campaign_hash=campaign_hash,
            matrix_hash=matrix_hash,
            backend=training_backend,
            profiles=profiles,
            train_rollouts=train_rollouts,
            workers=workers,
        )
        game = EmpiricalGame(
            profiles=tuple(profiles),
            policy_ids=policy_ids,
            payoffs=np.stack([np.mean(returns[p], axis=0) for p in profiles]),
            ci_radius=np.stack(
                [
                    1.96
                    * np.std(returns[p], axis=0, ddof=1)
                    / np.sqrt(train_rollouts)
                    for p in profiles
                ]
            ),
            objectives=np.asarray([np.mean(objectives[p]) for p in profiles], dtype=float),
            metrics=tuple(
                {"platform_operating_score": float(np.mean(objectives[p]))}
                for p in profiles
            ),
        )
        if use_runtime_q:
            solver_results, distributions, runtime_q_provenance = _load_runtime_q_source(
                matrix=matrix,
                job=job,
                dependency_dir=Path(dependency_dir),
                campaign_hash=campaign_hash,
                policy_ids=policy_ids,
                n_agents=training_backend.n_agents,
                mwu_selection=mwu_selection,
            )
        else:
            solver_results, distributions = _solve_exact_training(
                game,
                training_backend,
                train_rollouts=train_rollouts,
                workers=workers,
                job=job,
                mwu_config=mwu_config,
                smoke=smoke,
            )
        expected_solvers = tuple(FULL_SOLVER_BUNDLE)
        pure_summary: dict[str, Any] | None = _pure_training_summary(game)
    else:
        solver_results, distributions, runtime_q_provenance = _load_runtime_q_source(
            matrix=matrix,
            job=job,
            dependency_dir=Path(dependency_dir),
            campaign_hash=campaign_hash,
            policy_ids=policy_ids,
            n_agents=training_backend.n_agents,
            mwu_selection=mwu_selection,
        )
        profiles = frozen_closure(tuple(distributions.values()), policy_ids)
        training_chunks, _returns, _objectives = _complete_training_stage(
            job=job,
            state_dir=state_dir,
            campaign_hash=campaign_hash,
            matrix_hash=matrix_hash,
            backend=training_backend,
            profiles=profiles,
            train_rollouts=train_rollouts,
            workers=workers,
        )
        expected_solvers = tuple(SPARSE_SOLVER_BUNDLE)
        pure_summary = None

    if set(solver_results) != set(expected_solvers) or set(distributions) != set(expected_solvers):
        raise RuntimeError(
            f"{job.family} returned {sorted(solver_results)} instead of {sorted(expected_solvers)}."
        )
    formal_results, formal_checkpoint = _run_formal_audit(
        job=job,
        state_dir=state_dir,
        output_dir=output_dir,
        campaign_hash=campaign_hash,
        matrix_hash=matrix_hash,
        training_backend=training_backend,
        distributions=distributions,
        audit_rollouts=audit_rollouts,
        bootstrap_samples=bootstrap_samples,
        workers=workers,
        smoke=smoke,
    )
    for solver, distribution in distributions.items():
        solver_results[solver]["formal_audit"] = formal_results[solver]
        solver_results[solver]["approximate_pure_policy_space_nash"] = bool(
            len(distribution.support) == 1
            and formal_results[solver]["certified_low_gap_approximate_cce"]
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "job_id": job.job_id,
        "matrix_sha256": matrix_hash,
        "campaign_sha256": campaign_hash,
        "family": job.family,
        "reported_N": job.n_agents,
        "reported_J": job.policies_per_agent,
        "N": training_backend.n_agents,
        "J": len(policy_ids),
        "mechanism": job.mechanism,
        "seed": job.seed,
        "variant": job.variant,
        "stages": list(job.stages),
        "smoke": bool(smoke),
        "train_rollouts": train_rollouts,
        "audit_rollouts": audit_rollouts,
        "bootstrap_samples": bootstrap_samples,
        "policy_ids": list(policy_ids),
        "complete_training_table": complete_table,
        "mwu_selection": dict(mwu_selection),
        "runtime_q_provenance": runtime_q_provenance,
        "training_backend_identity": training_backend.cache_identity,
        "training_seeds": asdict(training_backend.seeds),
        "training_checkpoint": _checkpoint_summary(
            training_chunks, rollouts=train_rollouts
        ),
        "formal_audit_checkpoint": formal_checkpoint,
        "seed_isolation": {
            "manufacturer_population_held_fixed": True,
            "dynamic_streams_disjoint": True,
            "formal_audit_namespace_offset": 20_000_000,
        },
        "solver_results": solver_results,
        "pure_policy_game_diagnostics": pure_summary,
        "deterministic_mixed_support_checks": synthetic_checks,
    }
    atomic_json(output_dir / "result.json", result)
    return result


def _validate_runtime_environment(*, workers: int, smoke: bool) -> dict[str, Any]:
    observed = {
        "resource_mode": os.environ.get("CMFG_RESOURCE_MODE", ""),
        "restart_from_empty_cache": os.environ.get(
            "CMFG_RESTART_FROM_EMPTY_CACHE", ""
        ),
        "workers": int(workers),
    }
    if smoke:
        return observed
    if observed["resource_mode"] != "exclusive_runtime":
        raise ValueError("Formal runtime jobs require exclusive_runtime resource mode.")
    if observed["restart_from_empty_cache"] != "1":
        raise ValueError("Formal runtime jobs must restart every attempt from an empty cache.")
    if int(workers) != 32:
        raise ValueError("Formal GCP runtime jobs require the exclusive 32-worker node.")
    return observed


def _runtime_attempt_directories(root: Path, solver: str) -> tuple[Path, ...]:
    parent = root / solver.replace("/", "_")
    if not parent.exists():
        return ()
    return tuple(sorted(path for path in parent.iterdir() if path.is_dir()))


def _load_completed_runtime_attempt(
    root: Path,
    *,
    solver: str,
    identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    completed: list[dict[str, Any]] = []
    for directory in _runtime_attempt_directories(root, solver):
        marker = directory / "attempt_complete.json"
        if not marker.exists():
            continue
        payload = json.loads(marker.read_text(encoding="utf-8"))
        expected = {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "status": "complete",
            "job_id": identity["job_id"],
            "matrix_sha256": identity["matrix_sha256"],
            "campaign_sha256": identity["campaign_sha256"],
            "solver": solver,
            "empty_cache_confirmed": True,
            "attempt_stitched": False,
        }
        if {key: payload.get(key) for key in expected} != expected:
            raise ValueError(f"Completed runtime attempt identity mismatch for {solver}.")
        completed.append(payload)
    if len(completed) > 1:
        raise ValueError(f"Multiple completed runtime attempts exist for {solver}.")
    return completed[0] if completed else None


def _atomic_runtime_attempt(
    root: Path,
    *,
    solver: str,
    identity: Mapping[str, Any],
    run: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    restored = _load_completed_runtime_attempt(
        root, solver=solver, identity=identity
    )
    if restored is not None:
        return restored
    parent = root / solver.replace("/", "_")
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / f"attempt-{len(_runtime_attempt_directories(root, solver)) + 1:04d}"
    directory.mkdir(exist_ok=False)
    atomic_json(
        directory / "attempt_started.json",
        {
            **dict(identity),
            "solver": solver,
            "status": "started_from_empty_cache",
            "started_utc": _utc_now(),
        },
    )
    try:
        payload = {
            **dict(identity),
            **run(),
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "status": "complete",
            "solver": solver,
            "empty_cache_confirmed": True,
            "attempt_stitched": False,
            "completed_utc": _utc_now(),
        }
        atomic_json(directory / "attempt_complete.json", payload)
        return payload
    except BaseException as exc:
        atomic_json(
            directory / "attempt_failed.json",
            {
                **dict(identity),
                "solver": solver,
                "status": "incomplete_not_reusable",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_utc": _utc_now(),
            },
        )
        raise


def _runtime_exact_attempt(
    *,
    job: RevisionJobKey,
    solver: str,
    train_rollouts: int,
    workers: int,
    smoke: bool,
) -> dict[str, Any]:
    backend = _backend_for_stage(job, "training", smoke=smoke)
    policy_ids = tuple(backend.policies)
    profiles = enumerate_profiles(policy_ids, backend.n_agents)
    cache = BackendPayoffCache(backend=backend, n_rollouts=train_rollouts, workers=workers)
    if cache.estimates:
        raise RuntimeError("Runtime measurement did not start from an empty cache.")
    started = time.perf_counter()
    # Do not materialize hundreds of thousands of process-pool task payloads
    # at once for N=7,J=6.  These are in-memory batches inside one atomic
    # attempt, not reusable checkpoints: an interruption still discards the
    # whole measurement and starts again from an empty cache.
    for start in range(0, len(profiles), 1024):
        cache.ensure(profiles[start : start + 1024])
    game = build_empirical_game(list(cache.estimates.values()), policy_ids)
    if solver == "FullTensor-CCE-LP":
        solution = solve_full_cce_lp(
            game,
            selector=DEFAULT_SELECTOR,
            epsilon_tolerance=DEFAULT_EPSILON_TOLERANCE,
        )
    elif solver == "ExhaustiveCG-CCE":
        solution = solve_cg_cce_exhaustive(
            game,
            selector=DEFAULT_SELECTOR,
            epsilon_tolerance=DEFAULT_EPSILON_TOLERANCE,
        )
    else:
        raise ValueError(f"Unsupported exact runtime solver: {solver}")
    elapsed = time.perf_counter() - started
    distribution = _freeze_result(
        solution, n_agents=backend.n_agents, policy_ids=policy_ids
    )
    return {
        "runtime_seconds": float(elapsed),
        "payoff_evaluation_seconds": float(cache.eval_time_seconds),
        "solver_compute_seconds": max(0.0, elapsed - cache.eval_time_seconds),
        "train_rollouts": int(train_rollouts),
        "training_profile_count": len(profiles),
        "training_rollout_episode_count": int(cache.rollout_episode_total),
        "distribution": _distribution_payload(distribution),
        "q_generated_by_runtime_job": True,
        "training_cce_gap_nominal": float(solution.cce_gap_nominal),
        "training_cce_gap_ucb": float(solution.cce_gap_ucb),
        "backend_identity": backend.cache_identity,
        "seeds": asdict(backend.seeds),
    }


def _runtime_sparse_attempt(
    *,
    job: RevisionJobKey,
    solver: str,
    train_rollouts: int,
    workers: int,
    mwu_config: MwuTuningConfig,
    mwu_budget_seconds: float | None,
    smoke: bool,
) -> dict[str, Any]:
    backend = _backend_for_stage(job, "training", smoke=smoke)
    seed = _general_solver_seed(job, solver)
    if solver == "REPAIR-SAD-CCE":
        if smoke:
            cache = BackendPayoffCache(
                backend=backend, n_rollouts=train_rollouts, workers=workers
            )
            started = time.perf_counter()
            solution = run_solver(
                solver,
                cache,
                tuple(backend.policies),
                _smoke_dss_config(train_rollouts=train_rollouts, workers=workers),
                seed=seed,
                metadata={"revision_campaign": "revision-full-v1-smoke-runtime"},
            )
            elapsed = time.perf_counter() - started
            stats = cache.access_stats()
            stats.update(
                {
                    "runtime_seconds": elapsed,
                    "payoff_evaluation_seconds": cache.eval_time_seconds,
                    "solver_compute_seconds": max(0.0, elapsed - cache.eval_time_seconds),
                }
            )
        else:
            solution, cache, stats = run_dss_empty_cache(
                backend,
                train_rollouts=train_rollouts,
                workers=workers,
                solver_seed=seed,
            )
    elif solver == "MWU-PolicyTrace":
        if mwu_budget_seconds is None or float(mwu_budget_seconds) < 0.0:
            raise ValueError("MWU runtime q requires the paired measured DSS wall-time budget.")
        solution, cache, stats = run_mwu_empty_cache(
            backend,
            train_rollouts=train_rollouts,
            workers=workers,
            solver_seed=seed,
            budget_seconds=float(mwu_budget_seconds),
            eta=mwu_config.eta,
            schedule=mwu_config.schedule,
            exploration_floor=mwu_config.exploration_floor,
            burn_in_rounds=mwu_config.burn_in_rounds,
        )
    else:
        raise ValueError(f"Unsupported sparse runtime solver: {solver}")
    distribution = _freeze_result(
        solution, n_agents=backend.n_agents, policy_ids=tuple(backend.policies)
    )
    completed_rounds = (
        int(solution.diagnostics.get("trace_rounds", 0))
        if solver == "MWU-PolicyTrace"
        else None
    )
    if solver == "MWU-PolicyTrace" and (completed_rounds is None or completed_rounds <= 0):
        raise RuntimeError("Time-budgeted MWU returned no completed trace rounds.")
    return {
        "runtime_seconds": float(stats["runtime_seconds"]),
        "payoff_evaluation_seconds": float(stats["payoff_evaluation_seconds"]),
        "solver_compute_seconds": float(stats["solver_compute_seconds"]),
        "train_rollouts": int(train_rollouts),
        "training_profile_count": int(cache.evaluated_profile_count),
        "training_rollout_episode_count": int(cache.rollout_episode_total),
        "formal_rounds": completed_rounds,
        "formal_rounds_are_reproduction_provenance": solver == "MWU-PolicyTrace",
        "time_budget_seconds": (
            float(mwu_budget_seconds) if solver == "MWU-PolicyTrace" else None
        ),
        "q_generated_by_runtime_job": True,
        "distribution": _distribution_payload(distribution),
        "training_cce_gap_nominal": float(solution.cce_gap_nominal),
        "training_cce_gap_ucb": float(solution.cce_gap_ucb),
        "backend_identity": backend.cache_identity,
        "seeds": asdict(backend.seeds),
        "access_stats": _jsonable(stats),
    }


def _runtime_dss_mwu_pair_attempt(
    *,
    job: RevisionJobKey,
    train_rollouts: int,
    workers: int,
    mwu_config: MwuTuningConfig,
    smoke: bool,
) -> dict[str, Any]:
    """Generate the formal DSS and MWU q values in one atomic paired attempt."""

    dss = _runtime_sparse_attempt(
        job=job,
        solver="REPAIR-SAD-CCE",
        train_rollouts=train_rollouts,
        workers=workers,
        mwu_config=mwu_config,
        mwu_budget_seconds=None,
        smoke=smoke,
    )
    dss_runtime = float(dss["runtime_seconds"])
    if not np.isfinite(dss_runtime) or dss_runtime < 0.0:
        raise RuntimeError("DSS produced an invalid matched-runtime budget.")
    mwu = _runtime_sparse_attempt(
        job=job,
        solver="MWU-PolicyTrace",
        train_rollouts=train_rollouts,
        workers=workers,
        mwu_config=mwu_config,
        mwu_budget_seconds=dss_runtime,
        smoke=smoke,
    )
    mwu_runtime = float(mwu["runtime_seconds"])
    ratio = float(mwu_runtime / dss_runtime) if dss_runtime > 0.0 else float("nan")
    if not np.isfinite(ratio):
        # Production DSS jobs always have positive wall time.  Smoke execution
        # can be too short for a useful ratio, so retain a finite sentinel.
        if not smoke:
            raise RuntimeError("DSS/MWU runtime ratio is not finite.")
        ratio = 1.0
    return {
        "pair_contract": {
            "dss_first": True,
            "both_started_from_empty_payoff_caches": True,
            "mwu_budget_equals_measured_dss_wall_time": True,
            "interruption_invalidates_entire_pair": True,
            "machine_class": "c4-highcpu-32",
        },
        "dss_runtime_seconds": dss_runtime,
        "mwu_budget_seconds": dss_runtime,
        "mwu_runtime_seconds": mwu_runtime,
        "mwu_to_dss_runtime_ratio": ratio,
        "mwu_completed_trace_rounds": int(mwu["formal_rounds"]),
        "solver_results": {
            "REPAIR-SAD-CCE": dss,
            "MWU-PolicyTrace": mwu,
        },
    }


def run_general_runtime_pipeline(
    *,
    job: RevisionJobKey,
    state_dir: Path,
    output_dir: Path,
    matrix_hash: str,
    campaign_hash: str,
    workers: int,
    mwu_config: MwuTuningConfig,
    mwu_selection: Mapping[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    if job.family not in GENERAL_RUNTIME_FAMILIES:
        raise ValueError(f"{job.family} is not a general runtime family.")
    runtime_environment = _validate_runtime_environment(workers=workers, smoke=smoke)
    train_rollouts = SMOKE_TRAIN_ROLLOUTS if smoke else PRODUCTION_TRAIN_ROLLOUTS
    identity = {
        "job_id": job.job_id,
        "matrix_sha256": matrix_hash,
        "campaign_sha256": campaign_hash,
        "family": job.family,
        "smoke": bool(smoke),
    }
    attempts_root = state_dir / "stages" / "runtime_measurement" / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    if "REPAIR-SAD-CCE" not in job.solvers or "MWU-PolicyTrace" not in job.solvers:
        raise RuntimeError("Every formal runtime job must contain the paired DSS and MWU solvers.")
    paired = _atomic_runtime_attempt(
        attempts_root,
        solver="DSS-MWU-pair",
        identity=identity,
        run=lambda: _runtime_dss_mwu_pair_attempt(
            job=job,
            train_rollouts=train_rollouts,
            workers=workers,
            mwu_config=mwu_config,
            smoke=smoke,
        ),
    )
    results: dict[str, dict[str, Any]] = {
        solver: dict(payload)
        for solver, payload in dict(paired["solver_results"]).items()
    }
    for solver in job.solvers:
        if solver in {"REPAIR-SAD-CCE", "MWU-PolicyTrace"}:
            continue

        def run(solver: str = solver) -> dict[str, Any]:
            return _runtime_exact_attempt(
                job=job,
                solver=solver,
                train_rollouts=train_rollouts,
                workers=workers,
                smoke=smoke,
            )

        payload = _atomic_runtime_attempt(
            attempts_root,
            solver=solver,
            identity=identity,
            run=run,
        )
        results[solver] = payload

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        **identity,
        "reported_N": job.n_agents,
        "reported_J": job.policies_per_agent,
        "mechanism": job.mechanism,
        "seed": job.seed,
        "variant": job.variant,
        "stages": list(job.stages),
        "train_rollouts": train_rollouts,
        "mwu_selection": dict(mwu_selection),
        "paired_dss_mwu": {
            key: value
            for key, value in paired.items()
            if key not in {"solver_results"}
        },
        "runtime_environment": runtime_environment,
        "runtime_contract": {
            "empty_cache_per_solver": True,
            "interrupted_attempts_discarded": True,
            "attempts_stitched": False,
            "verification_runtime_included": False,
            "runtime_job_is_formal_q_source": True,
            "evidence_jobs_must_restore_q_without_resolving": True,
            "mwu_rounds_recorded_only_as_q_reproduction_provenance": True,
        },
        "solver_results": results,
    }
    atomic_json(output_dir / "result.json", result)
    return result


def run_general_physical_job(
    *,
    job_id: str,
    state_dir: Path,
    output_dir: Path,
    matrix_hash: str,
    workers: int,
    mwu_selection_path: str | Path | None = None,
    dependency_dir: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    matrix = RevisionFullV1Matrix()
    observed_matrix_hash = _require_sha256(matrix_hash, "matrix-hash")
    if observed_matrix_hash != matrix.matrix_hash:
        raise ValueError(
            f"Matrix hash mismatch: CLI={observed_matrix_hash}, current={matrix.matrix_hash}."
        )
    job = find_general_physical_job(matrix, job_id)
    campaign_hash = _campaign_hash(observed_matrix_hash, job.job_id, smoke=smoke)
    state_dir = Path(state_dir)
    output_dir = Path(output_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(workers))
    _write_identity(
        state_dir,
        job=job,
        matrix_hash=observed_matrix_hash,
        campaign_hash=campaign_hash,
        smoke=smoke,
    )
    mwu_config, selection = load_global_mwu_selection(
        mwu_selection_path,
        matrix=matrix,
        smoke=smoke,
    )
    if job.family in GENERAL_RUNTIME_FAMILIES:
        return run_general_runtime_pipeline(
            job=job,
            state_dir=state_dir,
            output_dir=output_dir,
            matrix_hash=observed_matrix_hash,
            campaign_hash=campaign_hash,
            workers=workers,
            mwu_config=mwu_config,
            mwu_selection=selection,
            smoke=smoke,
        )
    return run_general_bulk_pipeline(
        job=job,
        state_dir=state_dir,
        output_dir=output_dir,
        matrix=matrix,
        matrix_hash=observed_matrix_hash,
        campaign_hash=campaign_hash,
        workers=workers,
        mwu_config=mwu_config,
        mwu_selection=selection,
        dependency_dir=(Path(dependency_dir) if dependency_dir is not None else None),
        smoke=smoke,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one revision-full-v1 physical experiment pipeline."
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix-hash", required=True)
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("CMFG_WORKERS", "1"))
    )
    parser.add_argument(
        "--mwu-selection",
        type=Path,
        default=(
            Path(os.environ["CMFG_MWU_SELECTION_PATH"])
            if os.environ.get("CMFG_MWU_SELECTION_PATH")
            else None
        ),
    )
    parser.add_argument(
        "--dependency-dir",
        type=Path,
        default=(
            Path(os.environ["CMFG_DEPENDENCY_DIR"])
            if os.environ.get("CMFG_DEPENDENCY_DIR")
            else None
        ),
    )
    # The generic worker supplies outer bounds.  Physical runners own the
    # immutable profile lists and therefore deliberately ignore these values.
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--chunk-stop", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_general_physical_job(
        job_id=args.job_id,
        state_dir=args.state_dir,
        output_dir=args.output_dir,
        matrix_hash=args.matrix_hash,
        workers=args.workers,
        mwu_selection_path=args.mwu_selection,
        dependency_dir=args.dependency_dir,
        smoke=bool(args.smoke),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
