from __future__ import annotations

"""Campaign-level statistical reducers for ``revision-full-v1``.

The physical runners deliberately write one immutable result per game.  This
module is the only layer that combines those results into claims spanning
games, reporting seeds, mechanisms, policy libraries, or robustness settings.
Every public reducer first checks that its entire predeclared matrix is
present.  It therefore cannot silently summarize a convenient subset of the
formal campaign.
"""

from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cmfg_cce.evaluation.independent_audit import AuditSampleMatrix, freeze_distribution
from cmfg_cce.evaluation.revision_statistics import (
    analytic_familywise_relative_gap_bounds,
    holm_adjust,
    paired_seed_inference,
    support_diversity,
)
from cmfg_cce.experiments.revision_full_v1_spec import (
    FULL_SOLVER_BUNDLE,
    MECHANISMS,
    SELECTION_SENSITIVITY_SELECTORS,
    RevisionFullV1Matrix,
    RevisionJobKey,
)
from cmfg_cce.orchestration.chunks import deterministic_npz
from cmfg_cce.orchestration.manifest import (
    CampaignManifest,
    sha256_bytes,
    sha256_file,
)


BENCHMARK_REDUCER_SCHEMA = "revision_full_v1_benchmark_reducer_v1"
CNC_REDUCER_SCHEMA = "revision_full_v1_cnc_mechanism_reducer_v1"
TRANSPLANT_REDUCER_SCHEMA = "revision_full_v1_transplant_reducer_v1"
LIBRARY_REDUCER_SCHEMA = "revision_full_v1_library_reducer_v1"
PARAMETER_REDUCER_SCHEMA = "revision_full_v1_parameter_reducer_v1"
SELECTION_REDUCER_SCHEMA = "revision_full_v1_selection_reducer_v1"
PURE_MIXED_REDUCER_SCHEMA = "revision_full_v1_pure_mixed_reducer_v1"
SCALABILITY_REDUCER_SCHEMA = "revision_full_v1_scalability_reducer_v1"

# All effects are right minus left.  These four comparisons isolate the two
# design choices without asking readers to decode the internal M1--M4 labels.
MECHANISM_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    (
        "threshold_vs_own_price__price_only",
        "M1_price_first",
        "M2_price_critical",
    ),
    (
        "threshold_vs_own_price__price_plus_delivery",
        "M3_delivery_first",
        "M4_delivery_critical",
    ),
    (
        "delivery_vs_price_only__own_price",
        "M1_price_first",
        "M3_delivery_first",
    ),
    (
        "delivery_vs_price_only__threshold",
        "M2_price_critical",
        "M4_delivery_critical",
    ),
)

# These metrics are defined for every formal CNC mechanism and every frozen
# library/parameter variant.  Policy-specific winner shares are intentionally
# excluded because leave-one-out libraries do not share those fields.
DEFAULT_CNC_REDUCER_METRICS: tuple[str, ...] = (
    "payment_per_assignment",
    "manufacturer_discounted_profit",
    "conditional_bid_rate",
    "assignment_rate",
    "normalized_winner_hhi",
    "submitted_lead_time_multiplier",
    "capability_feasible_manufacturers_per_order",
    "scalar_capacity_feasible_manufacturers_per_order",
    "route_capacity_feasible_manufacturers_per_order",
    "scalar_to_route_loss_per_order",
    "at_least_three_route_capacity_feasible_manufacturers_rate",
    "orders_with_scalar_route_mismatch_rate",
    "conditional_route_false_positive_rate",
    "machine_group_remaining_capacity_T",
    "machine_group_remaining_capacity_M3",
    "machine_group_remaining_capacity_M5",
    "machine_group_remaining_capacity_G",
    "machine_group_remaining_capacity_EDM",
    *(
        metric
        for family in ("F1", "F2", "F3", "F4")
        for metric in (
            f"capability_feasible_manufacturers_per_order_{family}",
            f"scalar_capacity_feasible_manufacturers_per_order_{family}",
            f"route_capacity_feasible_manufacturers_per_order_{family}",
            f"scalar_to_route_loss_per_order_{family}",
            f"conditional_route_false_positive_rate_{family}",
            f"at_least_three_route_capacity_feasible_manufacturers_rate_{family}",
        )
    ),
)
DEFAULT_CNC_SENSITIVITY_METRICS: tuple[str, ...] = tuple(
    metric
    for metric in DEFAULT_CNC_REDUCER_METRICS
    if metric != "submitted_lead_time_multiplier"
)

_PROPORTION_OR_INDEX_METRICS = {
    "conditional_bid_rate",
    "assignment_rate",
    "normalized_winner_hhi",
    "at_least_three_route_capacity_feasible_manufacturers_rate",
    "conditional_route_false_positive_rate",
    "machine_group_remaining_capacity_T",
    "machine_group_remaining_capacity_M3",
    "machine_group_remaining_capacity_M5",
    "machine_group_remaining_capacity_G",
    "machine_group_remaining_capacity_EDM",
}

CORE_EXPANDED_STABILITY_METRICS: tuple[str, ...] = (
    "payment_per_assignment",
    "manufacturer_discounted_profit",
    "conditional_bid_rate",
    "assignment_rate",
    "normalized_winner_hhi",
)

PRIMARY_CNC_MECHANISM_METRICS: tuple[str, ...] = (
    "payment_per_assignment",
    "manufacturer_discounted_profit",
    "conditional_bid_rate",
    "assignment_rate",
    "normalized_winner_hhi",
)

OPERATING_FACTOR_PRIMARY_METRICS: Mapping[str, tuple[str, ...]] = {
    "higher_platform_workload": (
        "route_capacity_feasible_manufacturers_per_order",
        "at_least_three_route_capacity_feasible_manufacturers_rate",
        "machine_group_remaining_capacity_M5",
        "machine_group_remaining_capacity_EDM",
    ),
    "m5_edm_intensive_order_mix": (
        "route_capacity_feasible_manufacturers_per_order",
        "at_least_three_route_capacity_feasible_manufacturers_rate",
        "machine_group_remaining_capacity_M5",
        "machine_group_remaining_capacity_EDM",
    ),
    "higher_outside_workload_m5_g_edm": (
        "route_capacity_feasible_manufacturers_per_order",
        "at_least_three_route_capacity_feasible_manufacturers_rate",
        "machine_group_remaining_capacity_M5",
        "machine_group_remaining_capacity_G",
        "machine_group_remaining_capacity_EDM",
    ),
}


@dataclass(frozen=True)
class _LoadedResult:
    path: Path
    payload: Mapping[str, Any]
    job: RevisionJobKey


def _result_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = (path / "result.json", path / "artifacts" / "result.json")
        existing = [candidate for candidate in candidates if candidate.is_file()]
        if len(existing) == 1:
            return existing[0]
        if len(existing) > 1:
            raise ValueError(f"Ambiguous result directory: {path}")
    raise FileNotFoundError(f"Result JSON does not exist: {path}")


def _load_complete_results(
    result_paths: Sequence[str | Path],
    *,
    expected_jobs: Sequence[RevisionJobKey],
    matrix: RevisionFullV1Matrix,
) -> tuple[dict[str, _LoadedResult], str]:
    expected = {job.job_id: job for job in expected_jobs}
    if len(expected) != len(expected_jobs):
        raise RuntimeError("The frozen reducer matrix contains duplicate job IDs.")
    loaded: dict[str, _LoadedResult] = {}
    campaign_hashes: set[str] = set()
    for raw_path in result_paths:
        path = _result_path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        job_id = str(payload.get("job_id", ""))
        if job_id not in expected:
            raise ValueError(f"Unexpected result job_id {job_id!r}: {path}")
        if job_id in loaded:
            raise ValueError(f"Duplicate result for frozen job {job_id}.")
        job = expected[job_id]
        required_identity = {
            "status": "complete",
            "matrix_sha256": matrix.matrix_hash,
            "family": job.family,
            "mechanism": job.mechanism,
            "seed": int(job.seed),
            "variant": job.variant,
        }
        observed_identity = {key: payload.get(key) for key in required_identity}
        # Transplant results expose the frozen operating-condition variant as
        # ``condition`` because source/target mechanism is already explicit.
        if observed_identity["variant"] is None and "condition" in payload:
            observed_identity["variant"] = payload.get("condition")
        if observed_identity != required_identity:
            raise ValueError(
                f"Frozen result identity mismatch for {job_id}: expected "
                f"{required_identity!r}, observed {observed_identity!r}."
            )
        if bool(payload.get("smoke", False)):
            raise ValueError("Smoke outputs cannot enter a formal campaign reducer.")
        if int(payload.get("reported_N", job.n_agents)) != job.n_agents or int(
            payload.get("reported_J", job.policies_per_agent)
        ) != job.policies_per_agent:
            raise ValueError(f"Reported N/J changed for {job_id}.")
        campaign_hash = str(payload.get("campaign_sha256", ""))
        if len(campaign_hash) != 64:
            raise ValueError(f"Result {job_id} has no valid campaign hash.")
        campaign_hashes.add(campaign_hash)
        loaded[job_id] = _LoadedResult(path=path, payload=payload, job=job)
    missing = sorted(set(expected).difference(loaded))
    if missing:
        raise ValueError(
            f"Formal reducer is missing {len(missing)} of {len(expected)} frozen jobs; "
            f"first missing job: {missing[0]}."
        )
    if len(loaded) != len(expected):
        raise ValueError("Formal reducer result count does not match its frozen matrix.")
    if len(campaign_hashes) != 1:
        raise ValueError("Formal results do not share one immutable campaign hash.")
    return loaded, campaign_hashes.pop()


def _flat_artifact_path(result_path: Path, name: object) -> Path:
    value = str(name or "")
    if not value or Path(value).name != value:
        raise ValueError(f"Unsafe or missing flat artifact name in {result_path}.")
    return result_path.parent / value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Referenced audit artifact is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _audit_matrix_from_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    prefix: str,
    policy_ids: Sequence[str],
    n_agents: int,
    replications: int,
) -> AuditSampleMatrix:
    names = {
        "gains": f"{prefix}_gain_samples",
        "returns": f"{prefix}_q_return_samples",
        "agents": f"{prefix}_label_agent",
        "policies": f"{prefix}_label_policy_index",
    }
    missing = [name for name in names.values() if name not in arrays]
    if missing:
        raise ValueError(f"Audit sample group {prefix!r} is incomplete: {missing}.")
    gains = np.asarray(arrays[names["gains"]], dtype=float)
    returns = np.asarray(arrays[names["returns"]], dtype=float)
    agents = np.asarray(arrays[names["agents"]], dtype=int)
    policy_indices = np.asarray(arrays[names["policies"]], dtype=int)
    expected_labels = tuple(
        (agent, str(policy))
        for agent in range(int(n_agents))
        for policy in policy_ids
    )
    if gains.shape != (int(replications), len(expected_labels)):
        raise ValueError(f"Audit gain matrix {prefix!r} has shape {gains.shape}.")
    if returns.shape != (int(replications), int(n_agents)):
        raise ValueError(f"Audit payoff matrix {prefix!r} has shape {returns.shape}.")
    if agents.shape != (len(expected_labels),) or policy_indices.shape != (
        len(expected_labels),
    ):
        raise ValueError(f"Audit labels {prefix!r} have invalid shape.")
    if np.any(policy_indices < 0) or np.any(policy_indices >= len(policy_ids)):
        raise ValueError(f"Audit labels {prefix!r} contain an unknown policy index.")
    labels = tuple(
        (int(agent), str(policy_ids[int(policy_index)]))
        for agent, policy_index in zip(agents, policy_indices, strict=True)
    )
    if labels != expected_labels:
        raise ValueError(f"Audit replacement-label order changed for {prefix!r}.")
    if not np.all(np.isfinite(gains)) or not np.all(np.isfinite(returns)):
        raise ValueError(f"Audit samples {prefix!r} contain non-finite values.")
    return AuditSampleMatrix(
        labels=labels,
        gain_samples=gains,
        q_return_samples=returns,
    )


def _relative_gap_percent(samples: AuditSampleMatrix) -> float:
    gains = np.asarray(samples.gain_samples, dtype=float)
    returns = np.asarray(samples.q_return_samples, dtype=float)
    numerator = max(0.0, float(np.max(np.mean(gains, axis=0))))
    denominator = max(1.0, float(np.mean(np.abs(np.mean(returns, axis=0)))))
    return 100.0 * numerator / denominator


def _formal_completion_identity(item: _LoadedResult) -> dict[str, str]:
    """Validate the worker completion marker protecting a formal result.

    Benchmark timing claims join two separately scheduled physical jobs.  The
    adjacent completion markers are therefore the authoritative source/config
    identity, while the result payload supplies the runtime dependency hash.
    """

    if item.path.parent.name == "artifacts":
        marker_path = item.path.parent.parent / "job_complete.json"
    else:
        marker_path = item.path.parent / "job_complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"Formal benchmark completion marker is missing: {marker_path}"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable completion marker: {marker_path}") from exc
    expected = {
        "schema_version": "revision_full_v1_job_complete_v1",
        "status": "complete",
        "campaign_sha256": item.payload.get("campaign_sha256"),
        "matrix_sha256": item.payload.get("matrix_sha256"),
        "job_id": item.job.job_id,
        "family": item.job.family,
    }
    if {name: marker.get(name) for name in expected} != expected:
        raise ValueError(
            f"Benchmark completion identity mismatch for {item.job.job_id}."
        )
    identity: dict[str, str] = {}
    for name in ("campaign_sha256", "matrix_sha256", "source_sha256", "config_sha256"):
        value = str(marker.get(name, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(
                f"Benchmark completion marker has an invalid {name}: {item.job.job_id}."
            )
        identity[name] = value
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(
            f"Benchmark completion marker has no artifact manifest: {item.job.job_id}."
        )
    result_records = [
        record
        for record in artifacts
        if isinstance(record, Mapping) and record.get("path") == item.path.name
    ]
    if len(result_records) != 1:
        raise ValueError(
            f"Benchmark completion marker does not identify result.json exactly once: "
            f"{item.job.job_id}."
        )
    record = result_records[0]
    if (
        int(record.get("size", -1)) != item.path.stat().st_size
        or str(record.get("sha256", "")) != sha256_file(item.path)
    ):
        raise ValueError(
            f"Benchmark result differs from its completion manifest: {item.job.job_id}."
        )
    return identity


def _benchmark_cell(item: _LoadedResult) -> tuple[int, int, str, int]:
    return (
        int(item.job.n_agents),
        int(item.job.policies_per_agent),
        str(item.job.mechanism),
        int(item.job.seed),
    )


def reduce_solver_benchmark_audits(
    result_paths: Sequence[str | Path],
    runtime_result_paths: Sequence[str | Path],
    *,
    campaign_manifest: str | Path | CampaignManifest,
    matrix: RevisionFullV1Matrix | None = None,
    alpha: float = 0.05,
    certificate_threshold_percent: float = 2.0,
) -> dict[str, Any]:
    """Join all 72 fresh audits to their formal empty-cache runtime jobs."""

    matrix = matrix or RevisionFullV1Matrix()
    loaded, campaign_hash = _load_complete_results(
        result_paths,
        expected_jobs=matrix.solver_jobs(),
        matrix=matrix,
    )
    runtime_loaded, runtime_campaign_hash = _load_complete_results(
        runtime_result_paths,
        expected_jobs=matrix.solver_runtime_jobs(),
        matrix=matrix,
    )
    if runtime_campaign_hash != campaign_hash:
        raise ValueError("Benchmark bulk/runtime results span different campaigns.")
    manifest = (
        campaign_manifest
        if isinstance(campaign_manifest, CampaignManifest)
        else CampaignManifest.load(campaign_manifest)
    )
    if (
        manifest.campaign_sha256 != campaign_hash
        or manifest.matrix_sha256 != matrix.matrix_hash
    ):
        raise ValueError(
            "Benchmark results do not belong to the supplied formal campaign manifest."
        )
    manifest_jobs = {job.job_id: job for job in manifest.jobs}

    completion_identities = {
        item.job.job_id: _formal_completion_identity(item)
        for item in tuple(loaded.values()) + tuple(runtime_loaded.values())
    }
    source_hashes = {
        identity["source_sha256"] for identity in completion_identities.values()
    }
    config_hashes = {
        identity["config_sha256"] for identity in completion_identities.values()
    }
    if len(source_hashes) != 1:
        raise ValueError("Benchmark bulk/runtime jobs do not share one source hash.")
    if len(config_hashes) != 1:
        raise ValueError("Benchmark bulk/runtime jobs do not share one config hash.")
    for item in tuple(loaded.values()) + tuple(runtime_loaded.values()):
        manifest_job = manifest_jobs.get(item.job.job_id)
        identity = completion_identities[item.job.job_id]
        if (
            manifest_job is None
            or manifest_job.family != item.job.family
            or manifest_job.matrix_sha256 != matrix.matrix_hash
            or manifest_job.config_sha256 != identity["config_sha256"]
            or identity["source_sha256"] != manifest.source_sha256
        ):
            raise ValueError(
                f"Benchmark manifest/source/config mismatch for {item.job.job_id}."
            )

    runtime_by_cell: dict[tuple[int, int, str, int], _LoadedResult] = {}
    required_runtime_contract = {
        "empty_cache_per_solver": True,
        "interrupted_attempts_discarded": True,
        "attempts_stitched": False,
        "verification_runtime_included": False,
        "runtime_job_is_formal_q_source": True,
        "evidence_jobs_must_restore_q_without_resolving": True,
        "mwu_rounds_recorded_only_as_q_reproduction_provenance": True,
    }
    required_pair_contract = {
        "dss_first": True,
        "both_started_from_empty_payoff_caches": True,
        "mwu_budget_equals_measured_dss_wall_time": True,
        "interruption_invalidates_entire_pair": True,
        "machine_class": "c4-highcpu-32",
    }
    for item in runtime_loaded.values():
        cell = _benchmark_cell(item)
        if cell in runtime_by_cell:
            raise ValueError(f"Duplicate benchmark runtime cell: {cell}.")
        contract = item.payload.get("runtime_contract")
        if not isinstance(contract, Mapping) or {
            name: contract.get(name) for name in required_runtime_contract
        } != required_runtime_contract:
            raise ValueError(
                f"Benchmark runtime violates the frozen timing contract: "
                f"{item.job.job_id}."
            )
        environment = item.payload.get("runtime_environment")
        if not isinstance(environment, Mapping) or {
            "resource_mode": environment.get("resource_mode"),
            "restart_from_empty_cache": environment.get("restart_from_empty_cache"),
            "workers": environment.get("workers"),
        } != {
            "resource_mode": "exclusive_runtime",
            "restart_from_empty_cache": "1",
            "workers": 32,
        }:
            raise ValueError(
                f"Benchmark runtime was not measured on the exclusive 32-worker "
                f"empty-cache node: {item.job.job_id}."
            )
        runtime_by_cell[cell] = item

    games_by_solver: dict[str, dict[str, AuditSampleMatrix]] = {
        solver: {} for solver in FULL_SOLVER_BUNDLE
    }
    main_max_t_by_solver: dict[str, dict[str, dict[str, float]]] = {
        solver: {} for solver in FULL_SOLVER_BUNDLE
    }
    joined_rows: list[dict[str, Any]] = []
    matched_runtime_rows: list[dict[str, Any]] = []
    for item in loaded.values():
        payload = item.payload
        cell = _benchmark_cell(item)
        runtime_item = runtime_by_cell.get(cell)
        if runtime_item is None:
            raise ValueError(f"Missing benchmark runtime result for cell {cell}.")
        bulk_completion = completion_identities[item.job.job_id]
        runtime_completion = completion_identities[runtime_item.job.job_id]
        if (
            bulk_completion["source_sha256"] != runtime_completion["source_sha256"]
            or bulk_completion["config_sha256"]
            != runtime_completion["config_sha256"]
        ):
            raise ValueError(
                f"Benchmark bulk/runtime source or config mismatch for cell {cell}."
            )
        manifest_bulk_job = manifest_jobs[item.job.job_id]
        if manifest_bulk_job.depends_on != (runtime_item.job.job_id,):
            raise ValueError(
                f"Benchmark manifest dependency does not point to the matched runtime "
                f"job for cell {cell}."
            )
        if int(payload.get("train_rollouts", -1)) != matrix.train_rollouts or int(
            payload.get("audit_rollouts", -1)
        ) != matrix.audit_rollouts:
            raise ValueError(f"Benchmark rollout contract changed for {item.job.job_id}.")
        if int(runtime_item.payload.get("train_rollouts", -1)) != matrix.train_rollouts:
            raise ValueError(
                f"Benchmark runtime training-rollout contract changed for "
                f"{runtime_item.job.job_id}."
            )
        if payload.get("mwu_selection") != runtime_item.payload.get("mwu_selection"):
            raise ValueError(
                f"Benchmark bulk/runtime used different frozen MWU tuning for cell {cell}."
            )
        policy_ids = tuple(str(value) for value in payload.get("policy_ids", ()))
        if len(policy_ids) != item.job.policies_per_agent or len(set(policy_ids)) != len(
            policy_ids
        ):
            raise ValueError(f"Benchmark policy library changed for {item.job.job_id}.")
        solver_results = payload.get("solver_results")
        runtime_solver_results = runtime_item.payload.get("solver_results")
        if not isinstance(solver_results, Mapping) or set(solver_results) != set(
            FULL_SOLVER_BUNDLE
        ):
            raise ValueError(f"Benchmark solver bundle changed for {item.job.job_id}.")
        if not isinstance(runtime_solver_results, Mapping) or set(
            runtime_solver_results
        ) != set(FULL_SOLVER_BUNDLE):
            raise ValueError(
                f"Benchmark runtime solver bundle changed for {runtime_item.job.job_id}."
            )
        provenance = payload.get("runtime_q_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(
                f"Benchmark result has no runtime-q dependency provenance: "
                f"{item.job.job_id}."
            )
        runtime_result_sha256 = sha256_file(runtime_item.path)
        if provenance.get("runtime_job_id") != runtime_item.job.job_id:
            raise ValueError(
                f"Benchmark runtime dependency job mismatch for cell {cell}."
            )
        provenance_q_hashes = provenance.get("q_hashes")
        if not isinstance(provenance_q_hashes, Mapping) or set(
            provenance_q_hashes
        ) != set(FULL_SOLVER_BUNDLE):
            raise ValueError(f"Benchmark runtime dependency q map changed for cell {cell}.")
        artifact_spec = (
            payload.get("formal_audit_checkpoint", {}).get("sample_artifact", {})
        )
        artifact_path = _flat_artifact_path(item.path, artifact_spec.get("path"))
        if sha256_file(artifact_path) != str(artifact_spec.get("sha256", "")):
            raise ValueError(f"Benchmark audit artifact hash mismatch: {artifact_path}")
        arrays = _load_npz(artifact_path)
        expected_array_names: set[str] = set()
        samples_by_group: dict[str, AuditSampleMatrix] = {}
        cell_rows: list[dict[str, Any]] = []
        for solver in FULL_SOLVER_BUNDLE:
            bulk_solver = solver_results[solver]
            runtime_solver = runtime_solver_results[solver]
            if not isinstance(bulk_solver, Mapping) or not isinstance(
                runtime_solver, Mapping
            ):
                raise ValueError(f"Malformed benchmark solver result for {cell}/{solver}.")
            formal = bulk_solver.get("formal_audit", {})
            group = str(formal.get("audit_sample_group", ""))
            q_hash = str(formal.get("q_hash_before", ""))
            if group != f"q_{q_hash[:16]}" or formal.get("q_hash_after") != q_hash:
                raise ValueError(
                    f"Frozen q/audit group mismatch for {item.job.job_id}/{solver}."
                )
            bulk_q_hash = str(
                bulk_solver.get("distribution", {}).get("q_hash", "")
            )
            runtime_q_hash = str(
                runtime_solver.get("distribution", {}).get("q_hash", "")
            )
            try:
                bulk_frozen = freeze_distribution(
                    solver,
                    tuple(
                        tuple(profile)
                        for profile in bulk_solver.get("distribution", {}).get(
                            "support", ()
                        )
                    ),
                    bulk_solver.get("distribution", {}).get("probabilities", ()),
                    item.job.n_agents,
                    policy_ids,
                )
                runtime_frozen = freeze_distribution(
                    solver,
                    tuple(
                        tuple(profile)
                        for profile in runtime_solver.get("distribution", {}).get(
                            "support", ()
                        )
                    ),
                    runtime_solver.get("distribution", {}).get("probabilities", ()),
                    item.job.n_agents,
                    policy_ids,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Benchmark frozen distribution is malformed for {cell}/{solver}."
                ) from exc
            if (
                len(q_hash) != 64
                or q_hash != bulk_q_hash
                or q_hash != runtime_q_hash
                or q_hash != bulk_frozen.q_hash
                or q_hash != runtime_frozen.q_hash
                or q_hash != str(provenance_q_hashes.get(solver, ""))
            ):
                raise ValueError(
                    f"Benchmark frozen q differs between runtime, evidence, and audit "
                    f"for {cell}/{solver}."
                )
            if (
                bulk_solver.get("runtime_q_source_job_id")
                != runtime_item.job.job_id
                or bulk_solver.get("runtime_q_source_sha256")
                != runtime_result_sha256
                or bulk_solver.get("q_restored_without_resolving") is not True
            ):
                raise ValueError(
                    f"Benchmark solver dependency provenance changed for {cell}/{solver}."
                )
            if (
                bulk_solver.get("backend_identity")
                != runtime_solver.get("backend_identity")
                or bulk_solver.get("seeds") != runtime_solver.get("seeds")
            ):
                raise ValueError(
                    f"Benchmark bulk/runtime backend or seed identity differs for "
                    f"{cell}/{solver}."
                )
            if (
                runtime_solver.get("q_generated_by_runtime_job") is not True
                or runtime_solver.get("empty_cache_confirmed") is not True
                or runtime_solver.get("attempt_stitched") is not False
                or int(runtime_solver.get("train_rollouts", -1))
                != matrix.train_rollouts
            ):
                raise ValueError(
                    f"Benchmark solver timing attempt violates the empty-cache contract "
                    f"for {cell}/{solver}."
                )
            if group not in samples_by_group:
                samples_by_group[group] = _audit_matrix_from_arrays(
                    arrays,
                    prefix=group,
                    policy_ids=policy_ids,
                    n_agents=item.job.n_agents,
                    replications=matrix.audit_rollouts,
                )
                expected_array_names.update(
                    {
                        f"{group}_gain_samples",
                        f"{group}_q_return_samples",
                        f"{group}_label_agent",
                        f"{group}_label_policy_index",
                    }
                )
            game_key = (
                f"N{item.job.n_agents}_J{item.job.policies_per_agent}::"
                f"{item.job.mechanism}::seed_{item.job.seed}"
            )
            games_by_solver[solver][game_key] = samples_by_group[group]
            try:
                audit_rollouts = int(formal["audit_rollouts"])
                nominal = float(formal["relative_nominal_gap_percent"])
                max_t_ucb = float(formal["max_t_relative_gap_ucb95_percent"])
                max_t_lcb = float(formal["max_t_relative_gap_lcb95_percent"])
                max_payoff_se = float(formal["max_payoff_standard_error"])
                max_gain_se = float(formal["max_replacement_gain_standard_error"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Benchmark result lacks its per-game max-t audit for "
                    f"{item.job.job_id}/{solver}."
                ) from exc
            if (
                audit_rollouts != matrix.audit_rollouts
                or not np.all(
                    np.isfinite(
                        (nominal, max_t_ucb, max_t_lcb, max_payoff_se, max_gain_se)
                    )
                )
                or max_payoff_se < 0.0
                or max_gain_se < 0.0
                or max_t_lcb > nominal + 1.0e-10
                or nominal > max_t_ucb + 1.0e-10
                or not np.isclose(
                    nominal,
                    _relative_gap_percent(samples_by_group[group]),
                    atol=1.0e-8,
                    rtol=0.0,
                )
            ):
                raise ValueError(
                    f"Stored per-game max-t audit is inconsistent for "
                    f"{item.job.job_id}/{solver}."
                )
            main_max_t_by_solver[solver][game_key] = {
                "relative_nominal_gap_percent": nominal,
                "max_t_relative_gap_lcb95_percent": max_t_lcb,
                "max_t_relative_gap_ucb95_percent": max_t_ucb,
                "max_payoff_standard_error": max_payoff_se,
                "max_replacement_gain_standard_error": max_gain_se,
            }
            full_profile_count = int(
                item.job.policies_per_agent ** item.job.n_agents
            )
            training_access = bulk_solver.get("training_access")
            if not isinstance(training_access, Mapping):
                raise ValueError(
                    f"Benchmark solver lacks evaluated-profile accounting: "
                    f"{cell}/{solver}."
                )
            evaluated_profiles = int(
                training_access.get(
                    "solver_required_profile_count",
                    training_access.get("profile_count", -1),
                )
            )
            runtime_profile_count = int(
                runtime_solver.get("training_profile_count", -1)
            )
            if (
                evaluated_profiles != runtime_profile_count
                or not 0 < evaluated_profiles <= full_profile_count
            ):
                raise ValueError(
                    f"Benchmark evaluated-profile accounting differs between bulk and "
                    f"runtime for {cell}/{solver}."
                )
            runtime_seconds = float(runtime_solver.get("runtime_seconds", float("nan")))
            payoff_seconds = float(
                runtime_solver.get("payoff_evaluation_seconds", float("nan"))
            )
            solver_seconds = float(
                runtime_solver.get("solver_compute_seconds", float("nan"))
            )
            if (
                not np.all(np.isfinite((runtime_seconds, payoff_seconds, solver_seconds)))
                or runtime_seconds <= 0.0
                or payoff_seconds < 0.0
                or solver_seconds < 0.0
                or not np.isclose(
                    runtime_seconds,
                    payoff_seconds + solver_seconds,
                    atol=max(1.0e-9, runtime_seconds * 1.0e-9),
                    rtol=0.0,
                )
            ):
                raise ValueError(
                    f"Benchmark solver runtime decomposition is invalid for "
                    f"{cell}/{solver}."
                )
            row = {
                "game_key": game_key,
                "N": int(item.job.n_agents),
                "J": int(item.job.policies_per_agent),
                "mechanism": item.job.mechanism,
                "seed": int(item.job.seed),
                "solver": solver,
                "q_hash": q_hash,
                "full_joint_policy_space_size": full_profile_count,
                "evaluated_profile_count": evaluated_profiles,
                "profile_saving_ratio": 1.0
                - evaluated_profiles / float(full_profile_count),
                "end_to_end_runtime_seconds": runtime_seconds,
                "payoff_evaluation_seconds": payoff_seconds,
                "solver_compute_seconds": solver_seconds,
                "fresh_relative_nominal_gap_percent": nominal,
                "fresh_max_t_relative_gap_lcb95_percent": max_t_lcb,
                "fresh_max_t_relative_gap_ucb95_percent": max_t_ucb,
                "max_payoff_standard_error": max_payoff_se,
                "max_replacement_gain_standard_error": max_gain_se,
                "certified_low_gap_approximate_cce": bool(
                    max_t_ucb <= float(certificate_threshold_percent)
                ),
            }
            cell_rows.append(row)
        if set(arrays) != expected_array_names:
            raise ValueError(
                f"Benchmark audit artifact contains missing or unreferenced groups: "
                f"{artifact_path}."
            )
        if int(artifact_spec.get("q_count", -1)) != len(samples_by_group):
            raise ValueError(f"Benchmark artifact q_count changed: {artifact_path}")
        if provenance.get("runtime_result_sha256") != runtime_result_sha256:
            raise ValueError(
                f"Benchmark runtime dependency hash mismatch for cell {cell}."
            )
        oracle = next(
            row for row in cell_rows if row["solver"] == "FullTensor-CCE-LP"
        )
        for row in cell_rows:
            row["gap_to_oracle_nominal_percent"] = float(
                row["fresh_relative_nominal_gap_percent"]
                - oracle["fresh_relative_nominal_gap_percent"]
            )
            row["gap_to_oracle_ucb95_percent"] = float(
                row["fresh_max_t_relative_gap_ucb95_percent"]
                - oracle["fresh_max_t_relative_gap_ucb95_percent"]
            )
            joined_rows.append(row)

        paired = runtime_item.payload.get("paired_dss_mwu")
        if not isinstance(paired, Mapping) or {
            name: paired.get("pair_contract", {}).get(name)
            for name in required_pair_contract
        } != required_pair_contract:
            raise ValueError(f"Benchmark DSS/MWU pair contract changed for cell {cell}.")
        dss_runtime = float(
            runtime_solver_results["REPAIR-SAD-CCE"]["runtime_seconds"]
        )
        mwu_runtime = float(runtime_solver_results["MWU-PolicyTrace"]["runtime_seconds"])
        mwu_budget = float(paired.get("mwu_budget_seconds", float("nan")))
        stored_ratio = float(
            paired.get("mwu_to_dss_runtime_ratio", float("nan"))
        )
        observed_ratio = mwu_runtime / dss_runtime
        if (
            not np.isclose(float(paired.get("dss_runtime_seconds", float("nan"))), dss_runtime)
            or not np.isclose(float(paired.get("mwu_runtime_seconds", float("nan"))), mwu_runtime)
            or not np.isclose(mwu_budget, dss_runtime)
            or not np.isclose(stored_ratio, observed_ratio)
        ):
            raise ValueError(
                f"Benchmark matched DSS/MWU timing provenance is inconsistent for "
                f"cell {cell}."
            )
        matched_runtime_rows.append(
            {
                "game_key": (
                    f"N{item.job.n_agents}_J{item.job.policies_per_agent}::"
                    f"{item.job.mechanism}::seed_{item.job.seed}"
                ),
                "N": int(item.job.n_agents),
                "J": int(item.job.policies_per_agent),
                "mechanism": item.job.mechanism,
                "seed": int(item.job.seed),
                "dss_runtime_seconds": dss_runtime,
                "mwu_runtime_seconds": mwu_runtime,
                "mwu_budget_seconds": mwu_budget,
                "mwu_to_dss_runtime_ratio": observed_ratio,
                "dss_to_mwu_runtime_ratio": 1.0 / observed_ratio,
            }
        )

    solver_summaries: dict[str, Any] = {}
    for solver, games in games_by_solver.items():
        if len(games) != len(matrix.solver_jobs()):
            raise RuntimeError(f"Solver {solver} does not have all 72 benchmark games.")
        bounds = analytic_familywise_relative_gap_bounds(games, alpha=alpha)
        rows = bounds["games"]
        main_rows = main_max_t_by_solver[solver]
        solver_summaries[solver] = {
            "main_per_game_max_t_audits": main_rows,
            "analytic_familywise_bounds": bounds,
            "certification_threshold_percent": float(certificate_threshold_percent),
            "certified_game_count": sum(
                float(row["max_t_relative_gap_ucb95_percent"])
                <= float(certificate_threshold_percent)
                for row in main_rows.values()
            ),
            "maximum_per_game_max_t_relative_gap_ucb_percent": max(
                float(row["max_t_relative_gap_ucb95_percent"])
                for row in main_rows.values()
            ),
            "familywise_sensitivity_certified_game_count": sum(
                float(row["relative_gap_ucb_percent"])
                <= float(certificate_threshold_percent)
                for row in rows.values()
            ),
            "maximum_familywise_relative_gap_ucb_percent": max(
                float(row["relative_gap_ucb_percent"]) for row in rows.values()
            ),
        }

    for row in joined_rows:
        family = solver_summaries[row["solver"]]["analytic_familywise_bounds"][
            "games"
        ][row["game_key"]]
        row["familywise_relative_gap_lcb_percent"] = float(
            family["relative_gap_lcb_percent"]
        )
        row["familywise_relative_gap_ucb_percent"] = float(
            family["relative_gap_ucb_percent"]
        )

    size_solver_summaries: list[dict[str, Any]] = []
    matched_runtime_size_summaries: list[dict[str, Any]] = []
    for n_agents, policies_per_agent in matrix.solver_sizes:
        for solver in FULL_SOLVER_BUNDLE:
            rows = [
                row
                for row in joined_rows
                if row["N"] == int(n_agents)
                and row["J"] == int(policies_per_agent)
                and row["solver"] == solver
            ]
            if len(rows) != len(matrix.mechanisms) * len(matrix.solver_seeds):
                raise RuntimeError(
                    f"Benchmark size summary is incomplete for "
                    f"N={n_agents}, J={policies_per_agent}, solver={solver}."
                )
            size_solver_summaries.append(
                {
                    "N": int(n_agents),
                    "J": int(policies_per_agent),
                    "solver": solver,
                    "game_count": len(rows),
                    "end_to_end_runtime_seconds": _descriptive_summary(
                        [row["end_to_end_runtime_seconds"] for row in rows]
                    ),
                    "evaluated_profile_count": _descriptive_summary(
                        [row["evaluated_profile_count"] for row in rows]
                    ),
                    "profile_saving_ratio": _descriptive_summary(
                        [row["profile_saving_ratio"] for row in rows]
                    ),
                    "fresh_relative_nominal_gap_percent": _descriptive_summary(
                        [row["fresh_relative_nominal_gap_percent"] for row in rows]
                    ),
                    "fresh_max_t_relative_gap_ucb95_percent": _descriptive_summary(
                        [row["fresh_max_t_relative_gap_ucb95_percent"] for row in rows]
                    ),
                    "gap_to_oracle_nominal_percent": _descriptive_summary(
                        [row["gap_to_oracle_nominal_percent"] for row in rows]
                    ),
                    "gap_to_oracle_ucb95_percent": _descriptive_summary(
                        [row["gap_to_oracle_ucb95_percent"] for row in rows]
                    ),
                }
            )
        pair_rows = [
            row
            for row in matched_runtime_rows
            if row["N"] == int(n_agents) and row["J"] == int(policies_per_agent)
        ]
        if len(pair_rows) != len(matrix.mechanisms) * len(matrix.solver_seeds):
            raise RuntimeError(
                f"Benchmark matched-runtime summary is incomplete for "
                f"N={n_agents}, J={policies_per_agent}."
            )
        matched_runtime_size_summaries.append(
            {
                "N": int(n_agents),
                "J": int(policies_per_agent),
                "game_count": len(pair_rows),
                "mwu_to_dss_runtime_ratio": _descriptive_summary(
                    [row["mwu_to_dss_runtime_ratio"] for row in pair_rows]
                ),
                "dss_to_mwu_runtime_ratio": _descriptive_summary(
                    [row["dss_to_mwu_runtime_ratio"] for row in pair_rows]
                ),
            }
        )
    return {
        "schema_version": BENCHMARK_REDUCER_SCHEMA,
        "matrix_sha256": matrix.matrix_hash,
        "campaign_sha256": campaign_hash,
        "game_count": len(loaded),
        "exclusive_runtime_job_count": len(runtime_loaded),
        "solver_count": len(FULL_SOLVER_BUNDLE),
        "bounds_are_separate_by_solver": True,
        "source_sha256": source_hashes.pop(),
        "config_sha256": config_hashes.pop(),
        "dependency_lock_sha256": manifest.dependency_lock_sha256,
        "runtime_dependency_result_hash_verified_count": len(loaded),
        "runtime_contract": (
            "exclusive c4-highcpu-32 empty-cache end-to-end wall time; interrupted "
            "attempts discarded; verification runtime excluded"
        ),
        "joined_solver_rows": joined_rows,
        "size_solver_summaries": size_solver_summaries,
        "matched_dss_mwu_runtime_rows": matched_runtime_rows,
        "matched_dss_mwu_runtime_size_summaries": matched_runtime_size_summaries,
        "solvers": solver_summaries,
    }


def _primary_outcome_metrics(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        metrics = payload["outcome_evaluation"]["distributions"][
            "platform_operating_score"
        ]["metrics"]
    except (KeyError, TypeError) as exc:
        raise ValueError("CNC result has no primary independent-outcome metrics.") from exc
    if not isinstance(metrics, Mapping):
        raise ValueError("CNC independent-outcome metrics are malformed.")
    return metrics


def _metric_estimates(
    payload: Mapping[str, Any], metrics: Sequence[str]
) -> dict[str, float]:
    source = _primary_outcome_metrics(payload)
    result: dict[str, float] = {}
    for metric in metrics:
        try:
            if metric.startswith("scalar_to_route_loss_per_order_F") and metric not in source:
                family = metric.rsplit("_", maxsplit=1)[-1]
                value = float(
                    source[f"scalar_capacity_feasible_manufacturers_per_order_{family}"][
                        "estimate"
                    ]
                ) - float(
                    source[f"route_capacity_feasible_manufacturers_per_order_{family}"][
                        "estimate"
                    ]
                )
            else:
                value = float(source[metric]["estimate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"CNC result is missing metric {metric!r}.") from exc
        if not np.isfinite(value):
            raise ValueError(f"CNC metric {metric!r} is non-finite.")
        result[str(metric)] = value
    return result


def _validate_cnc_rollouts(payload: Mapping[str, Any], matrix: RevisionFullV1Matrix) -> None:
    expected = {
        "train_rollouts": matrix.train_rollouts,
        "audit_rollouts": matrix.audit_rollouts,
        "outcome_rollouts": matrix.outcome_rollouts,
    }
    observed = {name: payload.get(name) for name in expected}
    if observed != expected:
        raise ValueError(
            f"CNC rollout contract changed: expected {expected!r}, observed {observed!r}."
        )
    reconstructed = (
        payload.get("outcome_evaluation", {})
        .get("checkpoint", {})
        .get("ratios_and_hhi_reconstructed_inside_each_bootstrap_draw")
    )
    if reconstructed is not True:
        raise ValueError(
            "CNC outcomes do not certify raw-count reconstruction of ratios and HHI."
        )


def _mechanism_cube(
    loaded: Mapping[str, _LoadedResult],
    *,
    metrics: Sequence[str],
) -> dict[tuple[str, int, str], dict[str, float]]:
    cube: dict[tuple[str, int, str], dict[str, float]] = {}
    for item in loaded.values():
        condition = str(item.payload.get("condition", ""))
        key = (condition, int(item.job.seed), item.job.mechanism)
        if key in cube:
            raise ValueError(f"Duplicate CNC operating cell: {key}.")
        cube[key] = _metric_estimates(item.payload, metrics)
    return cube


def _inference_payload(value: object) -> dict[str, Any]:
    return {key: item for key, item in asdict(value).items()}


def reduce_cnc_mechanism_results(
    result_paths: Sequence[str | Path],
    *,
    matrix: RevisionFullV1Matrix | None = None,
    metrics: Sequence[str] = DEFAULT_CNC_REDUCER_METRICS,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Reduce 320 CNC games using reporting seeds as independent clusters."""

    matrix = matrix or RevisionFullV1Matrix()
    metrics = tuple(str(value) for value in metrics)
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("CNC reducer metrics must be nonempty and unique.")
    loaded, campaign_hash = _load_complete_results(
        result_paths,
        expected_jobs=matrix.cnc_main_jobs(),
        matrix=matrix,
    )
    for item in loaded.values():
        _validate_cnc_rollouts(item.payload, matrix)
        if item.payload.get("condition") != item.job.variant:
            raise ValueError(f"CNC condition changed for {item.job.job_id}.")
    cube = _mechanism_cube(loaded, metrics=metrics)
    matched_cells: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    raw_p_values: dict[str, float] = {}
    primary_metrics = tuple(
        metric for metric in PRIMARY_CNC_MECHANISM_METRICS if metric in set(metrics)
    )
    applicable_contrasts = {
        metric: (
            tuple(
                contrast
                for contrast in MECHANISM_CONTRASTS
                if contrast[0] == "threshold_vs_own_price__price_plus_delivery"
            )
            if metric == "submitted_lead_time_multiplier"
            else MECHANISM_CONTRASTS
        )
        for metric in metrics
    }
    for metric in metrics:
        for contrast_id, left, right in applicable_contrasts[metric]:
            clustered: dict[int, list[float]] = {
                int(seed): [] for seed in matrix.cnc_seeds
            }
            for seed in matrix.cnc_seeds:
                for condition in matrix.cnc_conditions:
                    try:
                        left_value = cube[(condition, int(seed), left)][metric]
                        right_value = cube[(condition, int(seed), right)][metric]
                    except KeyError as exc:
                        raise ValueError(
                            f"Missing matched CNC cell for {contrast_id}/{metric}."
                        ) from exc
                    effect = float(right_value - left_value)
                    clustered[int(seed)].append(effect)
                    matched_cells.append(
                        {
                            "contrast": contrast_id,
                            "left_mechanism": left,
                            "right_mechanism": right,
                            "metric": metric,
                            "condition": condition,
                            "seed": int(seed),
                            "right_minus_left": effect,
                        }
                    )
            hypothesis = f"{contrast_id}::{metric}"
            if metric in set(primary_metrics):
                inference = paired_seed_inference(clustered, alpha=alpha)
                if inference.seed_count != 10 or inference.cell_count != 8:
                    raise RuntimeError(
                        "CNC inference did not use 10 seeds times 8 conditions."
                    )
                raw_p_values[hypothesis] = inference.sign_flip_p_value
                inference_rows.append(
                    {
                        "hypothesis": hypothesis,
                        "contrast": contrast_id,
                        "left_mechanism": left,
                        "right_mechanism": right,
                        "effect_definition": "right_minus_left",
                        "metric": metric,
                        "inference_role": "predeclared_primary_holm_family",
                        **_inference_payload(inference),
                    }
                )
            else:
                seed_means = tuple(
                    float(np.mean(clustered[int(seed)]))
                    for seed in matrix.cnc_seeds
                )
                diagnostic_rows.append(
                    {
                        "contrast": contrast_id,
                        "left_mechanism": left,
                        "right_mechanism": right,
                        "effect_definition": "right_minus_left",
                        "metric": metric,
                        "inference_role": "descriptive_diagnostic_no_hypothesis_test",
                        "seed_count": len(seed_means),
                        "cell_count": len(matrix.cnc_conditions),
                        "seed_mean_effects": seed_means,
                        **_descriptive_summary(seed_means),
                    }
                )
    adjusted = holm_adjust(raw_p_values) if raw_p_values else {}
    for row in inference_rows:
        row["holm_adjusted_p_value"] = adjusted[row["hypothesis"]]

    condition_levels: dict[str, tuple[str, str, str]] = {}
    for condition in matrix.cnc_conditions:
        levels = tuple(str(condition).split("__"))
        if len(levels) != 3:
            raise ValueError(f"Malformed frozen CNC operating condition: {condition}")
        condition_levels[str(condition)] = levels  # type: ignore[assignment]
    factor_specs = (
        ("higher_platform_workload", 0, "nominal", "high"),
        ("m5_edm_intensive_order_mix", 1, "balanced", "m5_edm_intensive"),
        (
            "higher_outside_workload_m5_g_edm",
            2,
            "normal",
            "high_outside_m5_g_edm",
        ),
    )
    factor_cells: list[dict[str, Any]] = []
    factor_primary_rows: list[dict[str, Any]] = []
    factor_diagnostic_rows: list[dict[str, Any]] = []
    factor_p_values: dict[str, float] = {}
    for factor, level_index, low_level, high_level in factor_specs:
        primary_for_factor = set(OPERATING_FACTOR_PRIMARY_METRICS[factor])
        for metric in metrics:
            clustered: dict[int, list[float]] = {
                int(seed): [] for seed in matrix.cnc_seeds
            }
            mechanism_effects: dict[str, list[float]] = {
                mechanism: [] for mechanism in MECHANISMS
            }
            for seed in matrix.cnc_seeds:
                for mechanism in MECHANISMS:
                    low_values = [
                        cube[(condition, int(seed), mechanism)][metric]
                        for condition, levels in condition_levels.items()
                        if levels[level_index] == low_level
                    ]
                    high_values = [
                        cube[(condition, int(seed), mechanism)][metric]
                        for condition, levels in condition_levels.items()
                        if levels[level_index] == high_level
                    ]
                    if len(low_values) != 4 or len(high_values) != 4:
                        raise RuntimeError(
                            f"Operating-factor balance changed for {factor}/{metric}."
                        )
                    effect = float(np.mean(high_values) - np.mean(low_values))
                    clustered[int(seed)].append(effect)
                    mechanism_effects[mechanism].append(effect)
                    factor_cells.append(
                        {
                            "factor": factor,
                            "low_level": low_level,
                            "high_level": high_level,
                            "metric": metric,
                            "mechanism": mechanism,
                            "seed": int(seed),
                            "high_minus_low": effect,
                            "other_two_factors_averaged": True,
                        }
                    )
            hypothesis = f"{factor}::{metric}"
            if metric in primary_for_factor:
                inference = paired_seed_inference(clustered, alpha=alpha)
                if inference.seed_count != 10 or inference.cell_count != 4:
                    raise RuntimeError(
                        "Operating-factor inference did not average four mechanisms "
                        "inside each of ten reporting seeds."
                    )
                factor_p_values[hypothesis] = inference.sign_flip_p_value
                factor_primary_rows.append(
                    {
                        "hypothesis": hypothesis,
                        "factor": factor,
                        "low_level": low_level,
                        "high_level": high_level,
                        "metric": metric,
                        "effect_definition": "high_minus_low",
                        "mechanisms_averaged_within_seed": True,
                        **_inference_payload(inference),
                    }
                )
            else:
                seed_means = tuple(
                    float(np.mean(clustered[int(seed)]))
                    for seed in matrix.cnc_seeds
                )
                factor_diagnostic_rows.append(
                    {
                        "factor": factor,
                        "low_level": low_level,
                        "high_level": high_level,
                        "metric": metric,
                        "effect_definition": "high_minus_low",
                        "inference_role": "descriptive_diagnostic_no_hypothesis_test",
                        "mechanisms_averaged_within_seed": True,
                        "seed_mean_effects": seed_means,
                        "mechanism_stratified": {
                            mechanism: _descriptive_summary(values)
                            for mechanism, values in mechanism_effects.items()
                        },
                        **_descriptive_summary(seed_means),
                    }
                )
    factor_adjusted = holm_adjust(factor_p_values) if factor_p_values else {}
    for row in factor_primary_rows:
        row["holm_adjusted_p_value"] = factor_adjusted[row["hypothesis"]]
    return {
        "schema_version": CNC_REDUCER_SCHEMA,
        "matrix_sha256": matrix.matrix_hash,
        "campaign_sha256": campaign_hash,
        "game_count": len(loaded),
        "metrics": list(metrics),
        "matched_comparison_count": len(matched_cells),
        "matched_cells": matched_cells,
        "inference_rows": inference_rows,
        "diagnostic_rows": diagnostic_rows,
        "inference_unit": "reporting seed after averaging eight operating conditions",
        "confidence_interval": "two-sided Student-t interval across n=10 seed means",
        "randomization_test": "exact two-sided sign-flip test across n=10 seed means",
        "multiple_testing": (
            "Holm adjustment only across the predeclared five primary metrics and "
            "their applicable mechanism contrasts; manufacturing and lead-time "
            "diagnostics are descriptive"
        ),
        "primary_holm_family": [
            f"{contrast_id}::{metric}"
            for metric in primary_metrics
            for contrast_id, _left, _right in applicable_contrasts[metric]
        ],
        "primary_metrics": list(primary_metrics),
        "descriptive_diagnostic_metrics": [
            metric for metric in metrics if metric not in set(primary_metrics)
        ],
        "submitted_lead_time_contract": {
            "M1_price_first": "structural_NA",
            "M2_price_critical": "structural_NA",
            "reported_mechanisms": ["M3_delivery_first", "M4_delivery_critical"],
            "allowed_contrast": "M4_delivery_critical minus M3_delivery_first",
            "internal_price_only_default_excluded": True,
        },
        "operating_factor_analysis": {
            "matched_cells": factor_cells,
            "primary_inference_rows": factor_primary_rows,
            "diagnostic_rows": factor_diagnostic_rows,
            "primary_holm_family": [
                f"{factor}::{metric}"
                for factor, _index, _low, _high in factor_specs
                for metric in OPERATING_FACTOR_PRIMARY_METRICS[factor]
                if metric in set(metrics)
            ],
            "inference_unit": (
                "reporting seed after averaging the other two operating factors "
                "and the four auction mechanisms"
            ),
            "m5_edm_evidence_contract": (
                "The M5/EDM-intensive-mix claim is tested with M5/EDM remaining "
                "capacity and manufacturer-redundancy metrics. Route mismatch is "
                "reported only as a descriptive funnel diagnostic."
            ),
            "route_feasibility_funnel_metrics": [
                metric
                for metric in metrics
                if (
                    "feasible_manufacturers_per_order" in metric
                    or "scalar_to_route_loss" in metric
                    or "mismatch" in metric
                    or "false_positive" in metric
                )
            ],
        },
    }


def reduce_cce_selection_sensitivity(
    result_paths: Sequence[str | Path],
    *,
    matrix: RevisionFullV1Matrix | None = None,
    metrics: Sequence[str] = DEFAULT_CNC_SENSITIVITY_METRICS,
) -> dict[str, Any]:
    """Describe the three CCE selectors on all 40 shared payoff tables."""

    matrix = matrix or RevisionFullV1Matrix()
    metrics = tuple(str(value) for value in metrics)
    if "submitted_lead_time_multiplier" in metrics:
        raise ValueError("CCE-selection sensitivity excludes structural price-only lead time.")
    loaded, campaign_hash = _load_complete_results(
        result_paths,
        expected_jobs=matrix.selection_sensitivity_jobs(),
        matrix=matrix,
    )
    selector_cube: dict[tuple[str, str, int, str], dict[str, float]] = {}
    selector_rows: list[dict[str, Any]] = []
    for item in loaded.values():
        _validate_cnc_rollouts(item.payload, matrix)
        if item.payload.get("condition") != item.job.variant:
            raise ValueError(f"Selection condition changed for {item.job.job_id}.")
        distributions = item.payload.get("distributions")
        audit_distributions = item.payload.get("formal_audit", {}).get("distributions")
        outcome_distributions = item.payload.get("outcome_evaluation", {}).get(
            "distributions"
        )
        if not (
            isinstance(distributions, Mapping)
            and isinstance(audit_distributions, Mapping)
            and isinstance(outcome_distributions, Mapping)
            and set(distributions) == set(SELECTION_SENSITIVITY_SELECTORS)
            and set(audit_distributions) == set(SELECTION_SENSITIVITY_SELECTORS)
            and set(outcome_distributions) == set(SELECTION_SENSITIVITY_SELECTORS)
        ):
            raise ValueError("Selection result does not contain all three selector outputs.")
        for selector in SELECTION_SENSITIVITY_SELECTORS:
            distribution = distributions[selector]
            support = tuple(tuple(profile) for profile in distribution.get("support", ()))
            probabilities = tuple(float(value) for value in distribution.get("probabilities", ()))
            if not support or len(support) != len(probabilities):
                raise ValueError("Selection distribution support/probabilities are malformed.")
            diversity = support_diversity(probabilities)
            audit = audit_distributions[selector]
            try:
                max_t_ucb = float(audit["max_t_relative_gap_ucb95_percent"])
                nominal_gap = float(audit["relative_nominal_gap_percent"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Selection result lacks a fresh max-t audit.") from exc
            source_metrics = outcome_distributions[selector].get("metrics")
            if not isinstance(source_metrics, Mapping):
                raise ValueError("Selection result lacks independent outcome metrics.")
            estimates: dict[str, float] = {}
            for metric in metrics:
                try:
                    value = float(source_metrics[metric]["estimate"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Selection result is missing metric {metric!r}."
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError("Selection outcome metric is non-finite.")
                estimates[metric] = value
            key = (item.job.variant, item.job.mechanism, int(item.job.seed), selector)
            if key in selector_cube:
                raise ValueError(f"Duplicate selector output: {key}.")
            selector_cube[key] = estimates
            selector_rows.append(
                {
                    "condition": item.job.variant,
                    "mechanism": item.job.mechanism,
                    "seed": int(item.job.seed),
                    "selector": selector,
                    "q_hash": distribution.get("q_hash"),
                    **asdict(diversity),
                    "relative_nominal_gap_percent": nominal_gap,
                    "max_t_relative_gap_ucb95_percent": max_t_ucb,
                    "certified_low_gap_approximate_cce": bool(
                        audit.get("certified_low_gap_approximate_cce", False)
                    ),
                }
            )
    matched_cells: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for left, right in combinations(SELECTION_SENSITIVITY_SELECTORS, 2):
        for metric in metrics:
            effects: list[float] = []
            strata: list[dict[str, Any]] = []
            for condition in matrix.representative_cnc_conditions:
                for mechanism in MECHANISMS:
                    stratum_values: list[float] = []
                    for seed in matrix.sensitivity_seeds:
                        value = (
                            selector_cube[(condition, mechanism, int(seed), right)][metric]
                            - selector_cube[(condition, mechanism, int(seed), left)][metric]
                        )
                        effects.append(value)
                        stratum_values.append(value)
                        matched_cells.append(
                            {
                                "left_selector": left,
                                "right_selector": right,
                                "metric": metric,
                                "condition": condition,
                                "mechanism": mechanism,
                                "seed": int(seed),
                                "right_minus_left": value,
                            }
                        )
                    strata.append(
                        {
                            "condition": condition,
                            "mechanism": mechanism,
                            **_descriptive_summary(stratum_values),
                        }
                    )
            summaries.append(
                {
                    "left_selector": left,
                    "right_selector": right,
                    "metric": metric,
                    "effect_definition": "right_minus_left_on_the_same_payoff_table",
                    "stratified_five_seed_summaries": strata,
                    **_descriptive_summary(effects),
                }
            )
    return {
        "schema_version": SELECTION_REDUCER_SCHEMA,
        "matrix_sha256": matrix.matrix_hash,
        "campaign_sha256": campaign_hash,
        "shared_payoff_table_count": len(loaded),
        "selector_output_count": len(selector_rows),
        "metrics": list(metrics),
        "selector_rows": selector_rows,
        "matched_cells": matched_cells,
        "paired_difference_summaries": summaries,
        "inference_policy": _descriptive_inference_policy(
            len(matrix.sensitivity_seeds)
        ),
        "shared_table_pairing": True,
    }


def reduce_pure_mixed_analysis(
    benchmark_paths: Sequence[str | Path],
    mixed_challenge_paths: Sequence[str | Path],
    *,
    matrix: RevisionFullV1Matrix | None = None,
) -> dict[str, Any]:
    """Aggregate all 72 main and all 100 predeclared mixed-challenge games."""

    matrix = matrix or RevisionFullV1Matrix()
    benchmark, benchmark_campaign = _load_complete_results(
        benchmark_paths,
        expected_jobs=matrix.solver_jobs(),
        matrix=matrix,
    )
    mixed, mixed_campaign = _load_complete_results(
        mixed_challenge_paths,
        expected_jobs=matrix.mixed_challenge_jobs(),
        matrix=matrix,
    )
    if benchmark_campaign != mixed_campaign:
        raise ValueError("Pure/mixed result families do not share one campaign hash.")
    game_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    for analysis_family, loaded in (("main_72", benchmark), ("challenge_100", mixed)):
        for item in loaded.values():
            if item.payload.get("complete_training_table") is not True:
                raise ValueError("Pure-Nash enumeration requires a complete training table.")
            pure = item.payload.get("pure_policy_game_diagnostics")
            if not isinstance(pure, Mapping) or pure.get("complete_table") is not True:
                raise ValueError("Complete game lacks pure-policy Nash diagnostics.")
            try:
                pure_count = int(pure["pure_nash_count"])
                pure_profiles = pure["pure_nash_profiles"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Pure-policy Nash diagnostics are malformed.") from exc
            if pure_count < 0 or not isinstance(pure_profiles, list) or len(
                pure_profiles
            ) != pure_count:
                raise ValueError("Pure-policy Nash count/profile list is inconsistent.")
            game_rows.append(
                {
                    "analysis_family": analysis_family,
                    "job_id": item.job.job_id,
                    "N": item.job.n_agents,
                    "J": item.job.policies_per_agent,
                    "mechanism": item.job.mechanism,
                    "seed": int(item.job.seed),
                    "variant": item.job.variant,
                    "estimated_pure_policy_nash_count": pure_count,
                    "has_estimated_pure_policy_nash": bool(pure_count > 0),
                    "weak_dominance_pair_count": len(
                        pure.get("weak_dominance_pairs", ())
                    ),
                }
            )
            solver_results = item.payload.get("solver_results")
            if not isinstance(solver_results, Mapping) or set(solver_results) != set(
                FULL_SOLVER_BUNDLE
            ):
                raise ValueError("Pure/mixed game changed the four-solver bundle.")
            for solver in FULL_SOLVER_BUNDLE:
                result = solver_results[solver]
                distribution = result.get("distribution", {})
                probabilities = tuple(
                    float(value) for value in distribution.get("probabilities", ())
                )
                support = tuple(distribution.get("support", ()))
                if not support or len(support) != len(probabilities):
                    raise ValueError("Pure/mixed solver distribution is malformed.")
                diversity = support_diversity(probabilities)
                formal = result.get("formal_audit", {})
                try:
                    ucb = float(formal["max_t_relative_gap_ucb95_percent"])
                    nominal = float(formal["relative_nominal_gap_percent"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("Pure/mixed solver lacks a fresh max-t audit.") from exc
                certified = bool(formal.get("certified_low_gap_approximate_cce", False))
                approximate_pure = bool(result.get("approximate_pure_policy_space_nash"))
                if approximate_pure != bool(diversity.support_size == 1 and certified):
                    raise ValueError(
                        "Approximate pure-policy Nash label is inconsistent with support/UCB."
                    )
                solver_rows.append(
                    {
                        "analysis_family": analysis_family,
                        "job_id": item.job.job_id,
                        "N": item.job.n_agents,
                        "J": item.job.policies_per_agent,
                        "mechanism": item.job.mechanism,
                        "seed": int(item.job.seed),
                        "variant": item.job.variant,
                        "solver": solver,
                        **asdict(diversity),
                        "relative_nominal_gap_percent": nominal,
                        "max_t_relative_gap_ucb95_percent": ucb,
                        "certified_low_gap_approximate_cce": certified,
                        "approximate_pure_policy_space_nash": approximate_pure,
                    }
                )
    family_summaries: list[dict[str, Any]] = []
    for analysis_family, expected_count in (("main_72", 72), ("challenge_100", 100)):
        games = [row for row in game_rows if row["analysis_family"] == analysis_family]
        if len(games) != expected_count:
            raise RuntimeError("Pure/mixed reducer did not retain its full frozen family.")
        for solver in FULL_SOLVER_BUNDLE:
            rows = [
                row
                for row in solver_rows
                if row["analysis_family"] == analysis_family and row["solver"] == solver
            ]
            family_summaries.append(
                {
                    "analysis_family": analysis_family,
                    "solver": solver,
                    "game_count": len(rows),
                    "single_point_support_count": sum(
                        int(row["support_size"] == 1) for row in rows
                    ),
                    "fresh_certified_approximate_pure_count": sum(
                        bool(row["approximate_pure_policy_space_nash"])
                        for row in rows
                    ),
                    "mixed_support_count": sum(
                        int(row["support_size"] > 1) for row in rows
                    ),
                    "mean_normalized_support_entropy": float(
                        np.mean([row["normalized_entropy"] for row in rows])
                    ),
                }
            )
    return {
        "schema_version": PURE_MIXED_REDUCER_SCHEMA,
        "matrix_sha256": matrix.matrix_hash,
        "campaign_sha256": benchmark_campaign,
        "main_game_count": len(benchmark),
        "mixed_challenge_game_count": len(mixed),
        "all_predeclared_games_retained": True,
        "no_case_selected_by_observed_support": True,
        "game_rows": game_rows,
        "solver_rows": solver_rows,
        "family_solver_summaries": family_summaries,
        "interpretation_contract": (
            "Training-table pure profiles are estimated diagnostics; only a one-point "
            "returned q whose fresh max-t relative-gap UCB passes the threshold is "
            "labelled an approximate pure policy-space Nash equilibrium."
        ),
    }


def reduce_scalability_results(
    exact_bulk_paths: Sequence[str | Path],
    sparse_bulk_paths: Sequence[str | Path],
    exact_runtime_paths: Sequence[str | Path],
    sparse_runtime_paths: Sequence[str | Path],
    *,
    matrix: RevisionFullV1Matrix | None = None,
) -> dict[str, Any]:
    """Join formal output audits to their exclusive-GCP empty-cache runtimes."""

    matrix = matrix or RevisionFullV1Matrix()
    exact_bulk, campaign_1 = _load_complete_results(
        exact_bulk_paths,
        expected_jobs=matrix.exact_scalability_jobs(),
        matrix=matrix,
    )
    sparse_bulk, campaign_2 = _load_complete_results(
        sparse_bulk_paths,
        expected_jobs=matrix.sparse_scalability_jobs(),
        matrix=matrix,
    )
    exact_runtime, campaign_3 = _load_complete_results(
        exact_runtime_paths,
        expected_jobs=matrix.exact_scalability_runtime_jobs(),
        matrix=matrix,
    )
    sparse_runtime, campaign_4 = _load_complete_results(
        sparse_runtime_paths,
        expected_jobs=matrix.sparse_scalability_runtime_jobs(),
        matrix=matrix,
    )
    if len({campaign_1, campaign_2, campaign_3, campaign_4}) != 1:
        raise ValueError("Scalability bulk/runtime results span multiple campaigns.")

    def cell_key(item: _LoadedResult) -> tuple[int, int, str, int]:
        return (
            item.job.n_agents,
            item.job.policies_per_agent,
            item.job.mechanism,
            int(item.job.seed),
        )

    runtime_by_cell: dict[tuple[int, int, str, int], _LoadedResult] = {}
    for item in tuple(exact_runtime.values()) + tuple(sparse_runtime.values()):
        key = cell_key(item)
        if key in runtime_by_cell:
            raise ValueError(f"Duplicate scalability runtime cell: {key}.")
        contract = item.payload.get("runtime_contract", {})
        if not (
            contract.get("empty_cache_per_solver") is True
            and contract.get("interrupted_attempts_discarded") is True
            and contract.get("attempts_stitched") is False
            and contract.get("verification_runtime_included") is False
        ):
            raise ValueError("Scalability runtime violates the frozen timing contract.")
        runtime_by_cell[key] = item

    rows: list[dict[str, Any]] = []
    for scale_family, loaded in (("exact_N7_J6", exact_bulk), ("sparse_large", sparse_bulk)):
        for item in loaded.values():
            key = cell_key(item)
            if key not in runtime_by_cell:
                raise ValueError(f"Missing matched scalability runtime: {key}.")
            runtime = runtime_by_cell[key]
            bulk_solvers = item.payload.get("solver_results")
            runtime_solvers = runtime.payload.get("solver_results")
            if not isinstance(bulk_solvers, Mapping) or not isinstance(
                runtime_solvers, Mapping
            ) or set(bulk_solvers) != set(item.job.solvers) or set(runtime_solvers) != set(
                item.job.solvers
            ):
                raise ValueError("Scalability solver bundle changed between bulk/runtime.")
            oracle_nominal = None
            if "FullTensor-CCE-LP" in bulk_solvers:
                oracle_nominal = float(
                    bulk_solvers["FullTensor-CCE-LP"]["formal_audit"][
                        "relative_nominal_gap_percent"
                    ]
                )
            full_space_size = int(item.job.policies_per_agent ** item.job.n_agents)
            for solver in item.job.solvers:
                bulk = bulk_solvers[solver]
                timed = runtime_solvers[solver]
                bulk_q = str(bulk.get("distribution", {}).get("q_hash", ""))
                runtime_q = str(timed.get("distribution", {}).get("q_hash", ""))
                if not bulk_q or bulk_q != runtime_q:
                    raise ValueError(
                        f"Scalability runtime q differs from formal evidence for {key}/{solver}."
                    )
                formal = bulk.get("formal_audit", {})
                training_access = bulk.get("training_access", {})
                profile_count = training_access.get(
                    "solver_required_profile_count", training_access.get("profile_count")
                )
                if profile_count is None:
                    raise ValueError("Scalability solver lacks profile-evaluation accounting.")
                profile_count = int(profile_count)
                if not 0 < profile_count <= full_space_size:
                    raise ValueError("Scalability evaluated-profile count is invalid.")
                runtime_seconds = float(timed.get("runtime_seconds", float("nan")))
                if not np.isfinite(runtime_seconds) or runtime_seconds < 0.0:
                    raise ValueError("Scalability runtime is non-finite or negative.")
                probabilities = tuple(
                    float(value)
                    for value in bulk.get("distribution", {}).get("probabilities", ())
                )
                diversity = support_diversity(probabilities)
                nominal = float(formal["relative_nominal_gap_percent"])
                ucb = float(formal["max_t_relative_gap_ucb95_percent"])
                rows.append(
                    {
                        "scale_family": scale_family,
                        "N": item.job.n_agents,
                        "J": item.job.policies_per_agent,
                        "mechanism": item.job.mechanism,
                        "seed": int(item.job.seed),
                        "solver": solver,
                        "q_hash": bulk_q,
                        "full_joint_policy_space_size": full_space_size,
                        "evaluated_profile_count": profile_count,
                        "profile_saving_ratio": 1.0 - profile_count / full_space_size,
                        **asdict(diversity),
                        "relative_nominal_gap_percent": nominal,
                        "max_t_relative_gap_ucb95_percent": ucb,
                        "certified_low_gap_approximate_cce": bool(
                            formal.get("certified_low_gap_approximate_cce", False)
                        ),
                        "gap_to_full_tensor_nominal_percent": (
                            None if oracle_nominal is None else nominal - oracle_nominal
                        ),
                        "empty_cache_runtime_seconds": runtime_seconds,
                        "payoff_evaluation_seconds": float(
                            timed.get("payoff_evaluation_seconds", float("nan"))
                        ),
                        "solver_compute_seconds": float(
                            timed.get("solver_compute_seconds", float("nan"))
                        ),
                    }
                )
    summaries: list[dict[str, Any]] = []
    for scale_family in ("exact_N7_J6", "sparse_large"):
        family_rows = [row for row in rows if row["scale_family"] == scale_family]
        for n_agents in sorted({int(row["N"]) for row in family_rows}):
            for solver in sorted({str(row["solver"]) for row in family_rows if row["N"] == n_agents}):
                selected = [
                    row
                    for row in family_rows
                    if row["N"] == n_agents and row["solver"] == solver
                ]
                summaries.append(
                    {
                        "scale_family": scale_family,
                        "N": n_agents,
                        "J": int(selected[0]["J"]),
                        "solver": solver,
                        "game_count": len(selected),
                        "runtime_seconds": _descriptive_summary(
                            [row["empty_cache_runtime_seconds"] for row in selected]
                        ),
                        "profile_saving_ratio": _descriptive_summary(
                            [row["profile_saving_ratio"] for row in selected]
                        ),
                        "max_t_relative_gap_ucb_percent": _descriptive_summary(
                            [row["max_t_relative_gap_ucb95_percent"] for row in selected]
                        ),
                    }
                )
    return {
        "schema_version": SCALABILITY_REDUCER_SCHEMA,
        "matrix_sha256": matrix.matrix_hash,
        "campaign_sha256": campaign_1,
        "exact_game_count": len(exact_bulk),
        "sparse_game_count": len(sparse_bulk),
        "exclusive_runtime_job_count": len(runtime_by_cell),
        "rows": rows,
        "summaries": summaries,
        "runtime_contract": (
            "exclusive 32-worker GCP empty-cache wall time for the same frozen q; "
            "verification time excluded and interrupted attempts discarded"
        ),
        "sparse_claim_boundary": (
            "large sparse games certify the returned q against all unilateral policy "
            "replacements but do not claim full-space oracle optimality"
        ),
    }


def _transplant_samples_from_artifact(
    item: _LoadedResult,
    *,
    matrix: RevisionFullV1Matrix,
) -> dict[str, AuditSampleMatrix]:
    spec = item.payload.get("joint_sample_artifact")
    if not isinstance(spec, Mapping):
        raise ValueError("Transplant result has no joint sample artifact specification.")
    path = _flat_artifact_path(item.path, spec.get("path"))
    if sha256_file(path) != str(spec.get("sha256", "")):
        raise ValueError(f"Transplant sample artifact hash mismatch: {path}")
    arrays = _load_npz(path)
    policy_ids = tuple(f"A{index}" for index in range(1, 7))
    expected_names = {"label_agent", "label_policy_index"}
    agents = np.asarray(arrays.get("label_agent", np.empty(0)), dtype=int)
    policy_indices = np.asarray(
        arrays.get("label_policy_index", np.empty(0)), dtype=int
    )
    expected_labels = tuple(
        (agent, policy) for agent in range(4) for policy in policy_ids
    )
    labels = tuple(
        (int(agent), policy_ids[int(policy_index)])
        for agent, policy_index in zip(agents, policy_indices, strict=True)
    ) if (
        agents.shape == (len(expected_labels),)
        and policy_indices.shape == (len(expected_labels),)
        and np.all((policy_indices >= 0) & (policy_indices < len(policy_ids)))
    ) else ()
    if labels != expected_labels:
        raise ValueError("Transplant replacement-label order changed.")
    samples: dict[str, AuditSampleMatrix] = {}
    for arm in ("same_mechanism_control", "cross_mechanism", "target_recomputed"):
        gain_name = f"{arm}__gain_samples"
        return_name = f"{arm}__q_return_samples"
        expected_names.update((gain_name, return_name))
        gains = np.asarray(arrays.get(gain_name, np.empty(0)), dtype=float)
        returns = np.asarray(arrays.get(return_name, np.empty(0)), dtype=float)
        if gains.shape != (matrix.audit_rollouts, len(expected_labels)) or returns.shape != (
            matrix.audit_rollouts,
            4,
        ):
            raise ValueError(f"Transplant arm {arm!r} has invalid sample shape.")
        if not np.all(np.isfinite(gains)) or not np.all(np.isfinite(returns)):
            raise ValueError(f"Transplant arm {arm!r} contains non-finite samples.")
        samples[arm] = AuditSampleMatrix(labels, gains, returns)
        expected_hash = item.payload.get("joint_sample_hashes", {}).get(arm)
        if expected_hash is not None:
            observed_hash = sha256_bytes(
                deterministic_npz(
                    {"gain_samples": gains, "q_return_samples": returns}
                )
            )
            if observed_hash != str(expected_hash):
                raise ValueError(f"Transplant arm hash mismatch for {arm!r}.")
    if set(arrays) != expected_names:
        raise ValueError("Transplant artifact contains missing or unreferenced arrays.")
    return samples


def simultaneous_transplant_effect_bounds(
    cells: Mapping[str, Mapping[str, AuditSampleMatrix]],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Simultaneously bound cross-minus-same relative gaps over all cells.

    The Bonferroni-t family contains both arms of every supplied cell.  On the
    event that all arm intervals cover, subtracting the same-arm upper bound
    from the cross-arm lower bound (and vice versa) gives a conservative joint
    interval for every transplant effect.
    """

    if not cells:
        raise ValueError("At least one transplant cell is required.")
    flattened: dict[str, AuditSampleMatrix] = {}
    for cell_id, arms in sorted(cells.items()):
        if set(arms) != {
            "same_mechanism_control",
            "cross_mechanism",
            "target_recomputed",
        }:
            raise ValueError(
                "Each simultaneous transplant cell needs same, cross, and target arms."
            )
        flattened[f"{cell_id}::same"] = arms["same_mechanism_control"]
        flattened[f"{cell_id}::cross"] = arms["cross_mechanism"]
        flattened[f"{cell_id}::target"] = arms["target_recomputed"]
    arm_bounds = analytic_familywise_relative_gap_bounds(flattened, alpha=alpha)
    rows: dict[str, Any] = {}
    for cell_id in sorted(cells):
        same = arm_bounds["games"][f"{cell_id}::same"]
        cross = arm_bounds["games"][f"{cell_id}::cross"]
        target = arm_bounds["games"][f"{cell_id}::target"]
        rows[cell_id] = {
            "cross_minus_same_control": {
                "lower": float(cross["relative_gap_lcb_percent"])
                - float(same["relative_gap_ucb_percent"]),
                "upper": float(cross["relative_gap_ucb_percent"])
                - float(same["relative_gap_lcb_percent"]),
            },
            "cross_minus_target_recomputed": {
                "lower": float(cross["relative_gap_lcb_percent"])
                - float(target["relative_gap_ucb_percent"]),
                "upper": float(cross["relative_gap_ucb_percent"])
                - float(target["relative_gap_lcb_percent"]),
            },
            "cross_relative_gap_interval": [
                float(cross["relative_gap_lcb_percent"]),
                float(cross["relative_gap_ucb_percent"]),
            ],
            "same_control_relative_gap_interval": [
                float(same["relative_gap_lcb_percent"]),
                float(same["relative_gap_ucb_percent"]),
            ],
            "target_recomputed_relative_gap_interval": [
                float(target["relative_gap_lcb_percent"]),
                float(target["relative_gap_ucb_percent"]),
            ],
        }
    return {
        "method": "across-cell-and-arm analytic Bonferroni-t interval subtraction",
        "family_alpha": float(alpha),
        "cell_count": len(cells),
        "arm_family": arm_bounds,
        "cells": rows,
    }


def joint_bootstrap_transplant_direction(
    cells: Mapping[tuple[int, str], Mapping[str, AuditSampleMatrix]],
    *,
    seeds: Sequence[int],
    conditions: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
    alpha: float,
) -> dict[str, Any]:
    """Hierarchically bootstrap paired arms and reporting seeds.

    One replication-index draw is shared by all three arms and all eight
    operating conditions belonging to a reporting seed.  This preserves both
    the transplant-arm pairing and the condition-level common random numbers.
    After averaging the fixed condition set within each seed, each bootstrap
    draw resamples the reporting seeds with replacement.  The percentile
    interval therefore includes both Monte Carlo error and across-seed
    variation.  Seed-cluster t intervals and exact sign-flip tests remain
    separate checks in the campaign reducer.
    """

    if int(bootstrap_samples) < 100:
        raise ValueError("Joint transplant bootstrap needs at least 100 draws.")
    if not 0.0 < float(alpha) < 0.5:
        raise ValueError("alpha must lie in (0, 0.5).")
    arm_names = (
        "same_mechanism_control",
        "cross_mechanism",
        "target_recomputed",
    )
    primary_seed_samples: list[np.ndarray] = []
    secondary_seed_samples: list[np.ndarray] = []
    primary_seed_points: list[float] = []
    secondary_seed_points: list[float] = []
    rng = np.random.default_rng(int(bootstrap_seed))
    for seed in seeds:
        ordered_cells: list[Mapping[str, AuditSampleMatrix]] = []
        for condition in conditions:
            key = (int(seed), str(condition))
            if key not in cells:
                raise ValueError(f"Joint transplant bootstrap is missing cell {key}.")
            arms = cells[key]
            if set(arms) != set(arm_names):
                raise ValueError(f"Joint transplant cell {key} does not have three arms.")
            ordered_cells.append(arms)
        replication_counts = {
            np.asarray(arms[arm].gain_samples).shape[0]
            for arms in ordered_cells
            for arm in arm_names
        }
        if len(replication_counts) != 1:
            raise ValueError("Joint transplant cells do not share one replication count.")
        replications = replication_counts.pop()
        if replications < 2:
            raise ValueError("Joint transplant bootstrap requires R >= 2.")
        # Multinomial counts are equivalent to resampling replication indices,
        # while allowing one BLAS multiplication for all cell/arm constraints.
        counts = rng.multinomial(
            replications,
            np.full(replications, 1.0 / replications),
            size=int(bootstrap_samples),
        ).astype(
            np.uint16 if replications <= np.iinfo(np.uint16).max else np.uint32,
            copy=False,
        )
        gain_blocks: list[np.ndarray] = []
        return_blocks: list[np.ndarray] = []
        gain_slices: dict[tuple[int, str], slice] = {}
        return_slices: dict[tuple[int, str], slice] = {}
        gain_offset = 0
        return_offset = 0
        for condition_index, arms in enumerate(ordered_cells):
            common_labels: tuple[tuple[int, str], ...] | None = None
            for arm in arm_names:
                samples = arms[arm]
                gains = np.asarray(samples.gain_samples, dtype=float)
                returns = np.asarray(samples.q_return_samples, dtype=float)
                if (
                    gains.ndim != 2
                    or returns.ndim != 2
                    or gains.shape[0] != replications
                    or returns.shape[0] != replications
                    or gains.shape[1] != len(samples.labels)
                    or not np.all(np.isfinite(gains))
                    or not np.all(np.isfinite(returns))
                ):
                    raise ValueError("Joint transplant audit samples are malformed.")
                if common_labels is None:
                    common_labels = samples.labels
                elif samples.labels != common_labels:
                    raise ValueError("Joint transplant arms changed replacement labels.")
                gain_blocks.append(gains)
                return_blocks.append(returns)
                gain_slices[(condition_index, arm)] = slice(
                    gain_offset, gain_offset + gains.shape[1]
                )
                return_slices[(condition_index, arm)] = slice(
                    return_offset, return_offset + returns.shape[1]
                )
                gain_offset += gains.shape[1]
                return_offset += returns.shape[1]
        gain_means = counts @ np.column_stack(gain_blocks) / float(replications)
        return_means = counts @ np.column_stack(return_blocks) / float(replications)
        del counts
        primary_conditions: list[np.ndarray] = []
        secondary_conditions: list[np.ndarray] = []
        primary_points: list[float] = []
        secondary_points: list[float] = []
        for condition_index, arms in enumerate(ordered_cells):
            relative_samples: dict[str, np.ndarray] = {}
            relative_points: dict[str, float] = {}
            for arm in arm_names:
                boot_gains = gain_means[:, gain_slices[(condition_index, arm)]]
                boot_returns = return_means[:, return_slices[(condition_index, arm)]]
                numerator = np.maximum(0.0, np.max(boot_gains, axis=1))
                denominator = np.maximum(
                    1.0, np.mean(np.abs(boot_returns), axis=1)
                )
                relative_samples[arm] = 100.0 * numerator / denominator
                relative_points[arm] = _relative_gap_percent(arms[arm])
            primary_conditions.append(
                relative_samples["cross_mechanism"]
                - relative_samples["same_mechanism_control"]
            )
            secondary_conditions.append(
                relative_samples["cross_mechanism"]
                - relative_samples["target_recomputed"]
            )
            primary_points.append(
                relative_points["cross_mechanism"]
                - relative_points["same_mechanism_control"]
            )
            secondary_points.append(
                relative_points["cross_mechanism"]
                - relative_points["target_recomputed"]
            )
        primary_seed_samples.append(np.mean(primary_conditions, axis=0))
        secondary_seed_samples.append(np.mean(secondary_conditions, axis=0))
        primary_seed_points.append(float(np.mean(primary_points)))
        secondary_seed_points.append(float(np.mean(secondary_points)))
    if len(primary_seed_samples) != len(seeds) or not primary_seed_samples:
        raise RuntimeError("Joint transplant bootstrap did not construct every seed.")
    primary_by_seed = np.stack(primary_seed_samples, axis=0)
    secondary_by_seed = np.stack(secondary_seed_samples, axis=0)
    seed_draws = rng.integers(
        0,
        len(seeds),
        size=(int(bootstrap_samples), len(seeds)),
    )
    bootstrap_index = np.arange(int(bootstrap_samples), dtype=int)[:, None]
    primary = np.mean(primary_by_seed.T[bootstrap_index, seed_draws], axis=1)
    secondary = np.mean(secondary_by_seed.T[bootstrap_index, seed_draws], axis=1)

    def interval(values: np.ndarray, estimate: float) -> dict[str, float]:
        return {
            "estimate": float(estimate),
            "bootstrap_standard_error": float(np.std(values, ddof=1)),
            "ci_lower": float(np.quantile(values, float(alpha) / 2.0)),
            "ci_upper": float(np.quantile(values, 1.0 - float(alpha) / 2.0)),
        }

    return {
        "bootstrap_samples": int(bootstrap_samples),
        "replication_index_draw_shared_across_three_arms": True,
        "replication_index_draw_shared_across_conditions_within_seed": True,
        "reporting_seeds_resampled": True,
        "reporting_seed_resample_size": len(seeds),
        "aggregation_order": (
            "recompute each cell arm gap, subtract paired arms, average eight "
            "fixed conditions within seed, resample reporting seeds with replacement, "
            "then average the resampled seed effects"
        ),
        "cross_minus_same_control_percent": interval(
            primary, float(np.mean(primary_seed_points))
        ),
        "cross_minus_target_recomputed_percent": interval(
            secondary, float(np.mean(secondary_seed_points))
        ),
    }


def reduce_policy_transplant_results(
    result_paths: Sequence[str | Path],
    *,
    matrix: RevisionFullV1Matrix | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Reduce all 320 three-arm transplant audits at the seed-cluster level."""

    matrix = matrix or RevisionFullV1Matrix()
    loaded, campaign_hash = _load_complete_results(
        result_paths,
        expected_jobs=matrix.transplant_jobs(),
        matrix=matrix,
    )
    primary_cell_effects: dict[tuple[str, str, int], float] = {}
    secondary_cell_effects: dict[tuple[str, str, int], float] = {}
    artifact_presence: list[bool] = []
    audit_cells: dict[str, dict[str, AuditSampleMatrix]] = {}
    cell_metadata: dict[str, tuple[str, int, str]] = {}
    for item in loaded.values():
        payload = item.payload
        expected_direction = (item.job.source_mechanism, item.job.target_mechanism)
        if (
            payload.get("source_mechanism"),
            payload.get("target_mechanism"),
        ) != expected_direction or payload.get("condition") != item.job.variant:
            raise ValueError(f"Transplant direction/condition changed for {item.job.job_id}.")
        if int(payload.get("audit_rollouts", -1)) != matrix.audit_rollouts:
            raise ValueError(f"Transplant rollout contract changed for {item.job.job_id}.")
        direction = f"{expected_direction[0]}__to__{expected_direction[1]}"
        try:
            primary_effect = float(
                payload["three_arm_summary"]["effects"][
                    "cross_minus_same_control_percent"
                ]
            )
            secondary_effect = float(
                payload["three_arm_summary"]["effects"][
                    "cross_minus_target_recomputed_percent"
                ]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Transplant result has no complete three-arm effects.") from exc
        if not np.all(np.isfinite((primary_effect, secondary_effect))):
            raise ValueError("Transplant effect is non-finite.")
        key = (direction, item.job.variant, int(item.job.seed))
        if key in primary_cell_effects:
            raise ValueError(f"Duplicate transplant cell: {key}.")
        primary_cell_effects[key] = primary_effect
        secondary_cell_effects[key] = secondary_effect
        has_artifact = isinstance(payload.get("joint_sample_artifact"), Mapping)
        artifact_presence.append(has_artifact)
        if has_artifact:
            samples = _transplant_samples_from_artifact(item, matrix=matrix)
            observed_primary = _relative_gap_percent(samples["cross_mechanism"]) - (
                _relative_gap_percent(samples["same_mechanism_control"])
            )
            observed_secondary = _relative_gap_percent(samples["cross_mechanism"]) - (
                _relative_gap_percent(samples["target_recomputed"])
            )
            if not (
                np.isclose(observed_primary, primary_effect, atol=1.0e-10, rtol=0.0)
                and np.isclose(
                    observed_secondary, secondary_effect, atol=1.0e-10, rtol=0.0
                )
            ):
                raise ValueError("Stored transplant effects do not match their raw samples.")
            cell_id = item.job.job_id
            audit_cells[cell_id] = {
                "same_mechanism_control": samples["same_mechanism_control"],
                "cross_mechanism": samples["cross_mechanism"],
                "target_recomputed": samples["target_recomputed"],
            }
            cell_metadata[cell_id] = (direction, int(item.job.seed), item.job.variant)
    if any(artifact_presence) and not all(artifact_presence):
        raise ValueError(
            "Transplant replication artifacts are only partially present; selective "
            "simultaneous inference is forbidden."
        )

    matched_cells: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    secondary_rows: list[dict[str, Any]] = []
    primary_p_values: dict[str, float] = {}
    secondary_p_values: dict[str, float] = {}
    for source, target in matrix.transplant_directions:
        direction = f"{source}__to__{target}"
        primary_clustered: dict[int, list[float]] = {
            int(seed): [] for seed in matrix.cnc_seeds
        }
        secondary_clustered: dict[int, list[float]] = {
            int(seed): [] for seed in matrix.cnc_seeds
        }
        for seed in matrix.cnc_seeds:
            for condition in matrix.cnc_conditions:
                key = (direction, condition, int(seed))
                if key not in primary_cell_effects or key not in secondary_cell_effects:
                    raise ValueError(f"Missing transplant cell: {key}.")
                primary = primary_cell_effects[key]
                secondary = secondary_cell_effects[key]
                primary_clustered[int(seed)].append(primary)
                secondary_clustered[int(seed)].append(secondary)
                matched_cells.append(
                    {
                        "direction": direction,
                        "source_mechanism": source,
                        "target_mechanism": target,
                        "condition": condition,
                        "seed": int(seed),
                        "cross_minus_same_control_relative_gap_percent": primary,
                        "cross_minus_target_recomputed_relative_gap_percent": secondary,
                    }
                )
        primary_inference = paired_seed_inference(primary_clustered, alpha=alpha)
        secondary_inference = paired_seed_inference(secondary_clustered, alpha=alpha)
        if (
            primary_inference.seed_count != 10
            or primary_inference.cell_count != 8
            or secondary_inference.seed_count != 10
            or secondary_inference.cell_count != 8
        ):
            raise RuntimeError("Transplant inference did not use 10 seeds times 8 conditions.")
        primary_p_values[direction] = primary_inference.sign_flip_p_value
        secondary_p_values[direction] = secondary_inference.sign_flip_p_value
        primary_rows.append(
            {
                "direction": direction,
                "source_mechanism": source,
                "target_mechanism": target,
                **_inference_payload(primary_inference),
            }
        )
        secondary_rows.append(
            {
                "direction": direction,
                "source_mechanism": source,
                "target_mechanism": target,
                **_inference_payload(secondary_inference),
            }
        )
    primary_adjusted = holm_adjust(primary_p_values)
    secondary_adjusted = holm_adjust(secondary_p_values)
    for row in primary_rows:
        row["holm_adjusted_p_value"] = primary_adjusted[row["direction"]]
    for row in secondary_rows:
        row["holm_adjusted_p_value"] = secondary_adjusted[row["direction"]]

    simultaneous_payload: dict[str, Any]
    if all(artifact_presence):
        raw_bounds = simultaneous_transplant_effect_bounds(audit_cells, alpha=alpha)
        direction_intervals: dict[str, Any] = {}
        joint_bootstrap_rows: dict[str, Any] = {}
        for source, target in matrix.transplant_directions:
            direction = f"{source}__to__{target}"
            primary_seed_rows: list[dict[str, Any]] = []
            secondary_seed_rows: list[dict[str, Any]] = []
            for seed in matrix.cnc_seeds:
                cell_ids = [
                    cell_id
                    for cell_id, metadata in cell_metadata.items()
                    if metadata[0] == direction and metadata[1] == int(seed)
                ]
                if len(cell_ids) != 8:
                    raise RuntimeError("Simultaneous transplant seed cluster is incomplete.")
                primary_seed_rows.append(
                    {
                        "seed": int(seed),
                        "lower": float(
                            np.mean(
                                [
                                    raw_bounds["cells"][cell][
                                        "cross_minus_same_control"
                                    ]["lower"]
                                    for cell in cell_ids
                                ]
                            )
                        ),
                        "upper": float(
                            np.mean(
                                [
                                    raw_bounds["cells"][cell][
                                        "cross_minus_same_control"
                                    ]["upper"]
                                    for cell in cell_ids
                                ]
                            )
                        ),
                    }
                )
                secondary_seed_rows.append(
                    {
                        "seed": int(seed),
                        "lower": float(
                            np.mean(
                                [
                                    raw_bounds["cells"][cell][
                                        "cross_minus_target_recomputed"
                                    ]["lower"]
                                    for cell in cell_ids
                                ]
                            )
                        ),
                        "upper": float(
                            np.mean(
                                [
                                    raw_bounds["cells"][cell][
                                        "cross_minus_target_recomputed"
                                    ]["upper"]
                                    for cell in cell_ids
                                ]
                            )
                        ),
                    }
                )
            direction_intervals[direction] = {
                "cross_minus_same_control": {
                    "lower": float(
                        np.mean([row["lower"] for row in primary_seed_rows])
                    ),
                    "upper": float(
                        np.mean([row["upper"] for row in primary_seed_rows])
                    ),
                    "seed_cluster_intervals": primary_seed_rows,
                },
                "cross_minus_target_recomputed": {
                    "lower": float(
                        np.mean([row["lower"] for row in secondary_seed_rows])
                    ),
                    "upper": float(
                        np.mean([row["upper"] for row in secondary_seed_rows])
                    ),
                    "seed_cluster_intervals": secondary_seed_rows,
                },
            }
            direction_cells = {
                (int(seed), condition): audit_cells[cell_id]
                for cell_id, metadata in cell_metadata.items()
                for seed, condition in [(metadata[1], metadata[2])]
                if metadata[0] == direction
            }
            joint_bootstrap_rows[direction] = joint_bootstrap_transplant_direction(
                direction_cells,
                seeds=matrix.cnc_seeds,
                conditions=matrix.cnc_conditions,
                bootstrap_samples=matrix.bootstrap_samples,
                bootstrap_seed=int.from_bytes(
                    hashlib.sha256(
                        b"revision-full-v1:transplant-campaign-joint-bootstrap"
                    ).digest()[:8],
                    "big",
                ),
                alpha=alpha,
            )
        simultaneous_payload = {
            "available": True,
            "coverage_scope": (
                "simultaneous over all three arms in every one of the 320 "
                "predeclared transplant cells"
            ),
            "method": raw_bounds["method"],
            "family_alpha": float(alpha),
            "directions": direction_intervals,
            "arm_family": raw_bounds["arm_family"],
        }
        joint_bootstrap_payload: dict[str, Any] = {
            "available": True,
            "interval": "two-sided 95% percentile interval",
            "directions": joint_bootstrap_rows,
        }
    else:
        simultaneous_payload = {
            "available": False,
            "reason": "No per-replication transplant artifacts were supplied for any cell.",
        }
        joint_bootstrap_payload = {
            "available": False,
            "reason": "No per-replication transplant artifacts were supplied for any cell.",
        }
    return {
        "schema_version": TRANSPLANT_REDUCER_SCHEMA,
        "matrix_sha256": matrix.matrix_hash,
        "campaign_sha256": campaign_hash,
        "game_count": len(loaded),
        "matched_cells": matched_cells,
        "direction_rows": primary_rows,
        "secondary_direction_rows": secondary_rows,
        "effect_definition": (
            "cross-mechanism relative gap minus same-mechanism-control relative gap"
        ),
        "inference_unit": "reporting seed after averaging eight operating conditions",
        "confidence_interval": "two-sided Student-t interval across n=10 seed means",
        "randomization_test": "exact two-sided sign-flip test across n=10 seed means",
        "multiple_testing": (
            "separate Holm families across the four frozen directions for the "
            "primary cross-minus-control and secondary cross-minus-target effects"
        ),
        "simultaneous_effect_intervals": simultaneous_payload,
        "joint_bootstrap_effect_intervals": joint_bootstrap_payload,
    }


def _descriptive_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Descriptive sensitivity values must be finite and nonempty.")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "sample_sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "positive_count": int(np.sum(array > 1.0e-12)),
        "negative_count": int(np.sum(array < -1.0e-12)),
        "near_zero_count": int(np.sum(np.abs(array) <= 1.0e-12)),
    }


def _sign(value: float, tolerance: float = 1.0e-12) -> int:
    return int(value > tolerance) - int(value < -tolerance)


def _setting_summaries(
    cube: Mapping[tuple[str, str, str, int], Mapping[str, float]],
    *,
    conditions: Sequence[str],
    settings: Sequence[str],
    seeds: Sequence[int],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for setting in settings:
            for mechanism in MECHANISMS:
                for metric in metrics:
                    values = [
                        cube[(condition, setting, mechanism, int(seed))][metric]
                        for seed in seeds
                    ]
                    rows.append(
                        {
                            "condition": condition,
                            "setting": setting,
                            "mechanism": mechanism,
                            "metric": metric,
                            **_descriptive_summary(values),
                        }
                    )
    return rows


def _paired_setting_changes(
    cube: Mapping[tuple[str, str, str, int], Mapping[str, float]],
    *,
    conditions: Sequence[str],
    settings: Sequence[str],
    base_setting: str,
    seeds: Sequence[int],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for setting in settings:
            if setting == base_setting:
                continue
            for mechanism in MECHANISMS:
                for metric in metrics:
                    changes = [
                        cube[(condition, setting, mechanism, int(seed))][metric]
                        - cube[(condition, base_setting, mechanism, int(seed))][metric]
                        for seed in seeds
                    ]
                    rows.append(
                        {
                            "condition": condition,
                            "setting": setting,
                            "base_setting": base_setting,
                            "mechanism": mechanism,
                            "metric": metric,
                            "effect_definition": "setting_minus_base_within_seed",
                            **_descriptive_summary(changes),
                        }
                    )
    return rows


def _ranking_reversals(
    cube: Mapping[tuple[str, str, str, int], Mapping[str, float]],
    *,
    conditions: Sequence[str],
    settings: Sequence[str],
    base_setting: str,
    seeds: Sequence[int],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for setting in settings:
            if setting == base_setting:
                continue
            for metric in metrics:
                means = {
                    candidate: {
                        mechanism: float(
                            np.mean(
                                [
                                    cube[(condition, candidate, mechanism, int(seed))][
                                        metric
                                    ]
                                    for seed in seeds
                                ]
                            )
                        )
                        for mechanism in MECHANISMS
                    }
                    for candidate in (base_setting, setting)
                }
                for first, second in combinations(MECHANISMS, 2):
                    base_difference = (
                        means[base_setting][second] - means[base_setting][first]
                    )
                    setting_difference = means[setting][second] - means[setting][first]
                    if _sign(base_difference) * _sign(setting_difference) < 0:
                        rows.append(
                            {
                                "condition": condition,
                                "setting": setting,
                                "metric": metric,
                                "first_mechanism": first,
                                "second_mechanism": second,
                                "base_second_minus_first": base_difference,
                                "setting_second_minus_first": setting_difference,
                                "kind": "strict_pairwise_ranking_reversal",
                            }
                        )
    return rows


def _sensitivity_cube(
    loaded: Mapping[str, _LoadedResult],
    *,
    setting_field: str,
    metrics: Sequence[str],
) -> dict[tuple[str, str, str, int], dict[str, float]]:
    cube: dict[tuple[str, str, str, int], dict[str, float]] = {}
    for item in loaded.values():
        condition = str(item.payload.get("condition", ""))
        setting = str(item.payload.get(setting_field, ""))
        key = (condition, setting, item.job.mechanism, int(item.job.seed))
        if not condition or not setting or key in cube:
            raise ValueError(f"Malformed or duplicate sensitivity cell: {key}.")
        cube[key] = _metric_estimates(item.payload, metrics)
    return cube


def _descriptive_inference_policy(seed_count: int) -> dict[str, Any]:
    return {
        "seed_count": int(seed_count),
        "inferential_claims_made": False,
        "confidence_intervals_or_significance_tests_reported": False,
        "reason": (
            "These five-seed analyses are descriptive robustness checks; the "
            "minimum attainable two-sided exact sign-flip p-value is 0.0625."
        ),
    }


def reduce_policy_library_sensitivity(
    result_paths: Sequence[str | Path],
    *,
    matrix: RevisionFullV1Matrix | None = None,
    metrics: Sequence[str] = DEFAULT_CNC_SENSITIVITY_METRICS,
) -> dict[str, Any]:
    """Describe all ten frozen library settings without five-seed p-values."""

    matrix = matrix or RevisionFullV1Matrix()
    metrics = tuple(str(value) for value in metrics)
    if "submitted_lead_time_multiplier" in metrics:
        raise ValueError(
            "Policy-library sensitivity cannot treat the price-only internal lead-time "
            "default as a public outcome; omit submitted_lead_time_multiplier."
        )
    loaded, campaign_hash = _load_complete_results(
        result_paths,
        expected_jobs=matrix.policy_library_sensitivity_jobs("training"),
        matrix=matrix,
    )
    for item in loaded.values():
        _validate_cnc_rollouts(item.payload, matrix)
    cube = _sensitivity_cube(
        loaded, setting_field="library_variant", metrics=metrics
    )
    conditions = tuple(matrix.representative_cnc_conditions)
    settings = tuple(matrix.library_variants)
    seeds = tuple(matrix.sensitivity_seeds)
    setting_rows = _setting_summaries(
        cube,
        conditions=conditions,
        settings=settings,
        seeds=seeds,
        metrics=metrics,
    )
    change_rows = _paired_setting_changes(
        cube,
        conditions=conditions,
        settings=settings,
        base_setting="base_a1_a6",
        seeds=seeds,
        metrics=metrics,
    )
    reversals = _ranking_reversals(
        cube,
        conditions=conditions,
        settings=settings,
        base_setting="base_a1_a6",
        seeds=seeds,
        metrics=metrics,
    )

    new_policy_rows: list[dict[str, Any]] = []
    for item in loaded.values():
        if item.payload.get("library_variant") != "expanded_a1_a8":
            continue
        try:
            gain = item.payload["formal_audit"]["distributions"][
                "core_q_under_expanded_library"
            ]["new_policy_replacement_gains"]
            maximum = float(gain["maximum_new_policy_relative_gain_ucb95_percent"])
            passed = bool(gain["passes_two_percent_gain_stop"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Expanded-library result lacks the frozen-core A7/A8 gain audit."
            ) from exc
        new_policy_rows.append(
            {
                "condition": item.payload["condition"],
                "mechanism": item.job.mechanism,
                "seed": int(item.job.seed),
                "maximum_new_policy_relative_gain_ucb95_percent": maximum,
                "passes_two_percent_gain_stop": passed,
            }
        )
    expected_expanded = (
        len(conditions) * len(MECHANISMS) * len(seeds)
    )
    if len(new_policy_rows) != expected_expanded:
        raise RuntimeError("Expanded-library gain-audit matrix is incomplete.")

    stability_metrics = tuple(
        metric for metric in CORE_EXPANDED_STABILITY_METRICS if metric in set(metrics)
    )
    if not stability_metrics:
        raise ValueError(
            "Library reduction needs at least one primary mechanism-stability metric."
        )
    stability_rows: list[dict[str, Any]] = []
    for contrast_id, left, right in MECHANISM_CONTRASTS:
        for metric in stability_metrics:
            base_effects: list[float] = []
            expanded_effects: list[float] = []
            for condition in conditions:
                for seed in seeds:
                    base_effects.append(
                        cube[(condition, "base_a1_a6", right, int(seed))][metric]
                        - cube[(condition, "base_a1_a6", left, int(seed))][metric]
                    )
                    expanded_effects.append(
                        cube[(condition, "expanded_a1_a8", right, int(seed))][metric]
                        - cube[(condition, "expanded_a1_a8", left, int(seed))][metric]
                    )
            base_mean = float(np.mean(base_effects))
            expanded_mean = float(np.mean(expanded_effects))
            absolute_change = abs(expanded_mean - base_mean)
            direction_reversal = _sign(base_mean) * _sign(expanded_mean) < 0
            if metric in _PROPORTION_OR_INDEX_METRICS:
                reporting_threshold = 0.005
                relative_change = None
                material_change = absolute_change > reporting_threshold + 1.0e-12
                threshold_rule = "absolute change greater than 0.5 percentage points"
            else:
                relative_change = (
                    None
                    if abs(base_mean) <= 1.0e-12
                    else absolute_change / abs(base_mean)
                )
                reporting_threshold = 0.10
                material_change = (
                    absolute_change > 1.0e-12
                    if relative_change is None
                    else relative_change > reporting_threshold + 1.0e-12
                )
                threshold_rule = "relative change greater than 10% of the core contrast"
            stability_rows.append(
                {
                    "contrast": contrast_id,
                    "left_mechanism": left,
                    "right_mechanism": right,
                    "metric": metric,
                    "base_mean_effect": base_mean,
                    "expanded_mean_effect": expanded_mean,
                    "direction_reversal": direction_reversal,
                    "absolute_magnitude_change": absolute_change,
                    "relative_magnitude_change": relative_change,
                    "reporting_threshold": reporting_threshold,
                    "reporting_threshold_rule": threshold_rule,
                    "material_change_flag": bool(material_change),
                    "flagged_for_discussion": bool(direction_reversal or material_change),
                    "included_in_policy_richness_stopping_gate": True,
                }
            )
    gain_pass = all(row["passes_two_percent_gain_stop"] for row in new_policy_rows)
    contrast_pass = not any(
        bool(row["flagged_for_discussion"]) for row in stability_rows
    )
    return {
        "schema_version": LIBRARY_REDUCER_SCHEMA,
        "matrix_sha256": matrix.matrix_hash,
        "campaign_sha256": campaign_hash,
        "combination_count": len(loaded),
        "metrics": list(metrics),
        "setting_summaries": setting_rows,
        "paired_changes_from_core": change_rows,
        "ranking_reversals": reversals,
        "core_to_expanded_stability": {
            "comparison": "A1--A6 to A1--A8",
            "single_predeclared_enrichment_comparison": True,
            "new_policy_gain_rows": new_policy_rows,
            "all_new_policy_gain_ucbs_at_most_two_percent": gain_pass,
            "mechanism_contrast_rows": stability_rows,
            "outcome_reporting_rule": (
                "Report every sign reversal; also flag changes above 0.5 percentage "
                "points for rates/indices or above 10% for payment/profit. These "
                "flags fail the mechanism-contrast component of the policy-richness "
                "stopping gate."
            ),
            "flagged_mechanism_contrast_count": sum(
                bool(row["flagged_for_discussion"]) for row in stability_rows
            ),
            "all_primary_mechanism_contrasts_pass": contrast_pass,
            "policy_richness_stop_passes": bool(gain_pass and contrast_pass),
            "stopping_gate": (
                "both conditions are required: all simultaneous A7/A8 replacement-"
                "gain UCBs are at most 2%, and no predeclared primary mechanism "
                "contrast changes direction or exceeds its 10%/0.5-percentage-point "
                "magnitude threshold"
            ),
            "stopping_components": {
                "new_policy_gain_condition_passes": gain_pass,
                "primary_mechanism_contrast_condition_passes": contrast_pass,
                "failed_primary_contrasts": [
                    {
                        "contrast": row["contrast"],
                        "metric": row["metric"],
                        "direction_reversal": row["direction_reversal"],
                        "material_change_flag": row["material_change_flag"],
                    }
                    for row in stability_rows
                    if row["flagged_for_discussion"]
                ],
            },
        },
        "inference_policy": _descriptive_inference_policy(len(seeds)),
    }


def reduce_parameter_robustness(
    result_paths: Sequence[str | Path],
    *,
    matrix: RevisionFullV1Matrix | None = None,
    metrics: Sequence[str] = DEFAULT_CNC_SENSITIVITY_METRICS,
) -> dict[str, Any]:
    """Describe the base plus eight Resolution-IV parameter settings."""

    matrix = matrix or RevisionFullV1Matrix()
    metrics = tuple(str(value) for value in metrics)
    if "submitted_lead_time_multiplier" in metrics:
        raise ValueError(
            "Parameter robustness cannot treat the price-only internal lead-time "
            "default as a public outcome; omit submitted_lead_time_multiplier."
        )
    loaded, campaign_hash = _load_complete_results(
        result_paths,
        expected_jobs=matrix.parameter_robustness_jobs("training"),
        matrix=matrix,
    )
    for item in loaded.values():
        _validate_cnc_rollouts(item.payload, matrix)
    cube = _sensitivity_cube(
        loaded, setting_field="robustness_setting", metrics=metrics
    )
    conditions = tuple(matrix.representative_cnc_conditions)
    settings = tuple(setting.setting_id for setting in matrix.robustness_settings)
    seeds = tuple(matrix.robustness_seeds)
    setting_rows = _setting_summaries(
        cube,
        conditions=conditions,
        settings=settings,
        seeds=seeds,
        metrics=metrics,
    )
    changes = _paired_setting_changes(
        cube,
        conditions=conditions,
        settings=settings,
        base_setting="base",
        seeds=seeds,
        metrics=metrics,
    )
    reversals = _ranking_reversals(
        cube,
        conditions=conditions,
        settings=settings,
        base_setting="base",
        seeds=seeds,
        metrics=metrics,
    )

    factor_attributes = {
        "cost_dispersion": "cost_dispersion_scale",
        "alpha": "alpha_scale",
        "effective_rate": "effective_rate_scale",
        "policy_state_coefficient": "policy_coefficient_scale",
    }
    setting_by_id = {
        setting.setting_id: setting for setting in matrix.robustness_settings
        if setting.setting_id != "base"
    }
    factor_rows: list[dict[str, Any]] = []
    for factor, attribute in factor_attributes.items():
        observed_levels = sorted(
            {float(getattr(setting, attribute)) for setting in setting_by_id.values()}
        )
        if len(observed_levels) != 2:
            raise RuntimeError(f"Resolution-IV factor {factor} is not two-level.")
        low, high = observed_levels
        low_settings = tuple(
            setting_id for setting_id, setting in setting_by_id.items()
            if float(getattr(setting, attribute)) == low
        )
        high_settings = tuple(
            setting_id for setting_id, setting in setting_by_id.items()
            if float(getattr(setting, attribute)) == high
        )
        if len(low_settings) != 4 or len(high_settings) != 4:
            raise RuntimeError("Resolution-IV factor levels are unbalanced.")
        for condition in conditions:
            for mechanism in MECHANISMS:
                for metric in metrics:
                    seed_effects = []
                    for seed in seeds:
                        high_mean = float(
                            np.mean(
                                [
                                    cube[(condition, setting, mechanism, int(seed))][metric]
                                    for setting in high_settings
                                ]
                            )
                        )
                        low_mean = float(
                            np.mean(
                                [
                                    cube[(condition, setting, mechanism, int(seed))][metric]
                                    for setting in low_settings
                                ]
                            )
                        )
                        seed_effects.append(high_mean - low_mean)
                    factor_rows.append(
                        {
                            "factor": factor,
                            "low_level": low,
                            "high_level": high,
                            "condition": condition,
                            "mechanism": mechanism,
                            "metric": metric,
                            "effect_definition": (
                                "mean_high_minus_mean_low_within_seed_over_the_"
                                "eight_run_resolution_IV_fraction"
                            ),
                            **_descriptive_summary(seed_effects),
                        }
                    )
    return {
        "schema_version": PARAMETER_REDUCER_SCHEMA,
        "matrix_sha256": matrix.matrix_hash,
        "campaign_sha256": campaign_hash,
        "combination_count": len(loaded),
        "metrics": list(metrics),
        "setting_summaries": setting_rows,
        "paired_changes_from_base": changes,
        "ranking_reversals": reversals,
        "resolution_iv_factor_contrasts": factor_rows,
        "resolution_iv_interpretation": (
            "descriptive main-effect contrasts; each main effect is aliased with a "
            "three-factor interaction in the D=ABC half fraction"
        ),
        "inference_policy": _descriptive_inference_policy(len(seeds)),
    }
