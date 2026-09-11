"""Replay frozen audit statistics from saved samples, without running experiments.

The scientific calculations and bootstrap seeds come directly from the copied
campaign source. This adapter only loads verified records, checks their schema,
calls those functions, and records comparisons. It never evaluates a simulator,
solves an LP, changes a distribution, or writes into the input dataset.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import time
import traceback
from typing import Any, Mapping

import numpy as np

from cmfg_cce.evaluation.independent_audit import (
    AuditSampleMatrix,
    build_joint_audit_samples,
    summarize_audit_samples,
)
from cmfg_cce.evaluation.revision_statistics import summarize_three_arm_transplant
from cmfg_cce.experiments.revision_pipeline import frozen_closure, stable_solver_seed
from cmfg_cce.experiments.run_revision_cnc import (
    _audit_sample_hash,
    _distribution_from_payload,
    _profile_vector_hash,
)
from cmfg_cce.orchestration.chunks import ChunkPlan, deterministic_npz
from cmfg_cce.orchestration.manifest import canonical_json


AUDIT_FAMILIES = frozenset(
    {"solver_benchmark", "scalability_sparse", "cnc_main", "policy_transplant"}
)
AUDIT_ROLLOUTS = 2000
BOOTSTRAP_SAMPLES = 5000
ALPHA = 0.05
# Fixed before running the replay; a mismatch never relaxes these tolerances.
ABSOLUTE_TOLERANCE = 1.0e-9
RELATIVE_TOLERANCE = 1.0e-10


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _differences(recomputed: Any, saved: Any, path: str = "") -> list[dict[str, Any]]:
    """Compare generated fields; preserve every difference and its two values.

    Saved audit dictionaries include additional execution metadata. Such fields
    are checked explicitly by the adapters, not treated as extra statistics.
    Discrete fields match exactly; floating fields use the fixed tolerances.
    """
    differences: list[dict[str, Any]] = []
    if isinstance(recomputed, Mapping):
        if not isinstance(saved, Mapping):
            return [{"field": path, "reason": "type", "saved": saved, "recomputed": recomputed}]
        for name, value in recomputed.items():
            child = f"{path}.{name}" if path else str(name)
            if name not in saved:
                differences.append({"field": child, "reason": "missing_saved_field", "recomputed": value})
            else:
                differences.extend(_differences(value, saved[name], child))
        return differences
    if isinstance(recomputed, (tuple, list)):
        if not isinstance(saved, (tuple, list)) or len(recomputed) != len(saved):
            return [{"field": path, "reason": "sequence", "saved": saved, "recomputed": recomputed}]
        for index, (actual, expected) in enumerate(zip(recomputed, saved, strict=True)):
            differences.extend(_differences(actual, expected, f"{path}[{index}]"))
        return differences
    if isinstance(recomputed, float):
        valid = (
            isinstance(saved, (int, float))
            and not isinstance(saved, bool)
            and math.isfinite(recomputed)
            and math.isfinite(float(saved))
        )
        if valid and math.isclose(
            recomputed, float(saved), rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE
        ):
            return []
        mismatch = {"field": path, "reason": "numeric", "saved": saved, "recomputed": recomputed}
        if valid:
            mismatch["absolute_difference"] = abs(recomputed - float(saved))
        return [mismatch]
    if type(recomputed) is not type(saved) or recomputed != saved:
        return [{"field": path, "reason": "exact", "saved": saved, "recomputed": recomputed}]
    return []


def _read_arrays(dataset: Any, key: str, expected_sha256: str) -> dict[str, np.ndarray]:
    payload = dataset.object_bytes(key)
    _require(sha256(payload).hexdigest() == expected_sha256, f"Payload SHA256 mismatch: {key}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        _require(len(archive.files) == len(set(archive.files)), f"Duplicate arrays: {key}")
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _samples(
    arrays: Mapping[str, np.ndarray],
    *,
    gain_key: str,
    return_key: str,
    agent_key: str,
    policy_key: str,
    n_agents: int,
    policy_ids: tuple[str, ...],
) -> AuditSampleMatrix:
    gains = np.asarray(arrays[gain_key])
    returns = np.asarray(arrays[return_key])
    agents = np.asarray(arrays[agent_key])
    policies = np.asarray(arrays[policy_key])
    count = n_agents * len(policy_ids)
    _require(gains.shape == (AUDIT_ROLLOUTS, count), f"Wrong gain shape: {gains.shape}")
    _require(returns.shape == (AUDIT_ROLLOUTS, n_agents), f"Wrong return shape: {returns.shape}")
    _require(gains.dtype == np.dtype("float64") and returns.dtype == np.dtype("float64"), "Expected float64 samples")
    _require(np.all(np.isfinite(gains)) and np.all(np.isfinite(returns)), "Non-finite saved samples")
    _require(agents.shape == (count,) and policies.shape == (count,), "Wrong replacement-label shape")
    _require(agents.dtype == np.dtype("int16") and policies.dtype == np.dtype("int16"), "Expected int16 replacement labels")
    _require(np.array_equal(agents, np.repeat(np.arange(n_agents), len(policy_ids))), "Agent label order differs from frozen order")
    _require(np.array_equal(policies, np.tile(np.arange(len(policy_ids)), n_agents)), "Policy label order differs from frozen order")
    labels = tuple((int(agent), policy_ids[int(policy)]) for agent, policy in zip(agents, policies, strict=True))
    return AuditSampleMatrix(labels=labels, gain_samples=gains, q_return_samples=returns)


def _summarize(samples: AuditSampleMatrix, job_id: str, suffix: str) -> dict[str, Any]:
    return dict(summarize_audit_samples(
        samples,
        alpha=ALPHA,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=stable_solver_seed("revision-full-v1", job_id, suffix),
    ))


def _benchmark(dataset: Any, job_id: str, result: dict[str, Any], report: dict[str, Any]) -> None:
    policy_ids = tuple(result["policy_ids"])
    n_agents = int(result["N"])
    artifact = result["formal_audit_checkpoint"]["sample_artifact"]
    _require(artifact["path"] == "formal_audit_samples.npz", "Unexpected formal-audit artifact path")
    key = f"jobs/{job_id}/artifacts/{artifact['path']}"
    arrays = _read_arrays(dataset, key, artifact["sha256"])
    report["sample_objects"] = [key]
    by_q: dict[str, dict[str, Any]] = {}
    groups: dict[str, str] = {}
    expected_array_names: set[str] = set()
    generated: dict[str, Any] = {}
    report["recomputed"] = {"solver_audits": generated}
    for solver_name, solver in result["solver_results"].items():
        distribution = _distribution_from_payload(solver["distribution"], n_agents=n_agents, policy_ids=policy_ids)
        stored = solver["formal_audit"]
        q_hash = distribution.q_hash
        group = f"q_{q_hash[:16]}"
        _require(stored["q_hash_before"] == q_hash == stored["q_hash_after"], f"Frozen q identity mismatch for {solver_name}")
        _require(stored["audit_sample_group"] == group, f"Sample group mismatch for {solver_name}")
        _require(group not in groups or groups[group] == q_hash, "Truncated q-hash collision")
        groups[group] = q_hash
        names = {kind: f"{group}_{kind}" for kind in ("gain_samples", "q_return_samples", "label_agent", "label_policy_index")}
        expected_array_names.update(names.values())
        if q_hash not in by_q:
            samples = _samples(
                arrays, gain_key=names["gain_samples"], return_key=names["q_return_samples"],
                agent_key=names["label_agent"], policy_key=names["label_policy_index"],
                n_agents=n_agents, policy_ids=policy_ids,
            )
            by_q[q_hash] = _summarize(samples, job_id, "formal-bootstrap")
        generated[solver_name] = {"q_hash": q_hash, "statistics": by_q[q_hash]}
        report["mismatches"].extend(_differences(by_q[q_hash], stored, f"solver_results.{solver_name}.formal_audit"))
    _require(len(by_q) == artifact["q_count"] and bool(by_q), "Unique q count differs from artifact declaration")
    _require(set(arrays) == expected_array_names, "Formal-audit NPZ arrays differ from declared q groups")
    report["unique_distribution_count"] = len(by_q)


def _cnc(dataset: Any, job_id: str, result: dict[str, Any], report: dict[str, Any]) -> None:
    policy_ids = tuple(result["policy_ids"])
    n_agents = int(result["N"])
    audit = result["formal_audit"]
    _require(audit["rollouts"] == AUDIT_ROLLOUTS and audit["bootstrap_samples"] == BOOTSTRAP_SAMPLES, "Unexpected CNC audit settings")
    distributions = {
        label: _distribution_from_payload(payload, n_agents=n_agents, policy_ids=policy_ids)
        for label, payload in result["distributions"].items()
    }
    _require(bool(distributions) and set(distributions) == set(audit["distributions"]), "CNC distribution labels mismatch")
    profiles = frozen_closure(tuple(distributions.values()), policy_ids)
    prefix = f"jobs/{job_id}/stages/formal_audit/chunks"
    plan_key = f"{prefix}/chunk_plan.json"
    plan = dataset.object_json(plan_key)
    expected_plan = ChunkPlan.create(
        campaign_sha256=result["campaign_sha256"], matrix_sha256=result["matrix_sha256"],
        job_id=f"{job_id}::formal_audit", kind="audit", chunk_size=8,
        item_ids=["|".join(profile) for profile in profiles],
    ).to_payload()
    _require(plan == expected_plan, "CNC audit plan differs from frozen distribution closure")
    _require(audit["checkpoint"]["item_order_sha256"] == plan["item_order_sha256"], "CNC item-order digest mismatch")
    _require(audit["profile_count"] == len(profiles), "CNC profile count mismatch")
    returns: dict[tuple[str, ...], np.ndarray] = {}
    report["sample_objects"] = [plan_key]
    next_index = 0
    for descriptor in audit["checkpoint"]["chunks"]:
        start, stop = descriptor["start"], descriptor["stop"]
        _require(start == next_index and stop == min(start + 8, len(profiles)), "Missing, duplicated, or reordered CNC audit chunk")
        chunk_id = f"audit-{start:09d}-{stop:09d}"
        _require(descriptor["chunk_id"] == chunk_id, "CNC chunk identity mismatch")
        descriptor_key = f"{prefix}/{chunk_id}.json"
        _require(dataset.object_json(descriptor_key) == descriptor, "CNC descriptor differs from result checkpoint")
        for identity_key in ("campaign_sha256", "matrix_sha256", "job_id", "kind"):
            _require(descriptor[identity_key] == expected_plan[identity_key], f"CNC chunk {identity_key} mismatch")
        key = f"{prefix}/{chunk_id}.npz"
        arrays = _read_arrays(dataset, key, descriptor["payload_sha256"])
        _require(set(arrays) == {"returns", "profile_indices", "rollout_count"}, "Unexpected CNC chunk arrays")
        for name, array in arrays.items():
            _require({"dtype": str(array.dtype), "shape": list(array.shape)} == descriptor["arrays"][name], f"CNC chunk array schema mismatch: {name}")
        _require(arrays["returns"].shape == (stop - start, AUDIT_ROLLOUTS, n_agents), "CNC returns shape mismatch")
        _require(arrays["returns"].dtype == np.dtype("float64") and np.all(np.isfinite(arrays["returns"])), "Invalid CNC return samples")
        _require(np.array_equal(arrays["profile_indices"], np.arange(start, stop, dtype=np.int64)), "CNC profile indices mismatch")
        _require(np.array_equal(arrays["rollout_count"], np.array([AUDIT_ROLLOUTS], dtype=np.int64)), "CNC rollout count mismatch")
        for offset, profile in enumerate(profiles[start:stop]):
            returns[profile] = arrays["returns"][offset]
        next_index = stop
        report["sample_objects"].extend([descriptor_key, key])
    _require(next_index == len(profiles), "Incomplete CNC audit closure")
    derived = {
        "raw_profile_return_vectors_sha256": _profile_vector_hash(profiles, policy_ids, returns),
        "q_union_sha256": sha256(canonical_json({label: q.q_hash for label, q in sorted(distributions.items())})).hexdigest(),
    }
    report["mismatches"].extend(_differences(derived, audit, "formal_audit"))
    generated: dict[str, Any] = {}
    report["recomputed"] = {**derived, "distribution_audits": generated}
    for label, distribution in distributions.items():
        stored = audit["distributions"][label]
        _require(stored["q_hash_before"] == distribution.q_hash == stored["q_hash_after"], f"Frozen CNC q identity mismatch: {label}")
        samples = build_joint_audit_samples(returns, distribution, policy_ids, n_agents=n_agents, sample_count=AUDIT_ROLLOUTS)
        summary = _summarize(samples, job_id, "formal-audit-shared-bootstrap")
        summary["joint_replication_samples_sha256"] = _audit_sample_hash(samples)
        generated[label] = {"q_hash": distribution.q_hash, "statistics": summary}
        report["mismatches"].extend(_differences(summary, stored, f"formal_audit.distributions.{label}"))


def _transplant(dataset: Any, job_id: str, result: dict[str, Any], report: dict[str, Any]) -> None:
    # These are the explicit frozen dimensions in run_revision_transplant.py.
    metadata = dataset.jobs[job_id]["metadata"]
    _require(metadata["n_agents"] == 4 and metadata["policies_per_agent"] == 6, "Unexpected transplant dimensions")
    policy_ids = tuple(f"A{index}" for index in range(1, 7))
    for name in ("source_distribution", "target_distribution"):
        _distribution_from_payload(result[name], n_agents=4, policy_ids=policy_ids)
    artifact = result["joint_sample_artifact"]
    _require(artifact["path"] == "transplant_audit_samples.npz", "Unexpected transplant artifact path")
    key = f"jobs/{job_id}/artifacts/{artifact['path']}"
    arrays = _read_arrays(dataset, key, artifact["sha256"])
    arms = ("same_mechanism_control", "cross_mechanism", "target_recomputed")
    _require(artifact["arms"] == sorted(arms) and artifact["common_replication_index_across_arms"] is True, "Transplant arm declaration mismatch")
    expected_names = {"label_agent", "label_policy_index"} | {f"{arm}__{kind}" for arm in arms for kind in ("gain_samples", "q_return_samples")}
    _require(set(arrays) == expected_names, "Unexpected transplant NPZ arrays")
    samples_by_arm: dict[str, AuditSampleMatrix] = {}
    sample_hashes: dict[str, str] = {}
    for arm in arms:
        samples = _samples(
            arrays, gain_key=f"{arm}__gain_samples", return_key=f"{arm}__q_return_samples",
            agent_key="label_agent", policy_key="label_policy_index", n_agents=4, policy_ids=policy_ids,
        )
        samples_by_arm[arm] = samples
        sample_hashes[arm] = sha256(deterministic_npz({"gain_samples": samples.gain_samples, "q_return_samples": samples.q_return_samples})).hexdigest()
    report["sample_objects"] = [key]
    report["mismatches"].extend(_differences(sample_hashes, result["joint_sample_hashes"], "joint_sample_hashes"))
    summary = summarize_three_arm_transplant(
        samples_by_arm[arms[0]], samples_by_arm[arms[1]], samples_by_arm[arms[2]],
        alpha=ALPHA, bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=stable_solver_seed("revision-full-v1", job_id, "three-arm-joint-bootstrap"),
    )
    report["recomputed"] = {"joint_sample_hashes": sample_hashes, "three_arm_summary": summary}
    report["mismatches"].extend(_differences(summary, result["three_arm_summary"], "three_arm_summary"))


def _recompute_job(dataset: Any, job_id: str) -> dict[str, Any]:
    started = time.monotonic()
    family = dataset.jobs[job_id]["family"]
    report: dict[str, Any] = {"job_id": job_id, "family": family, "mismatches": [], "recomputed": {}}
    try:
        result = dataset.result(job_id)
        _require(result["job_id"] == job_id and result["family"] == family and result["status"] == "complete", "Result identity/status mismatch")
        _require(result["audit_rollouts"] == AUDIT_ROLLOUTS and result["bootstrap_samples"] == BOOTSTRAP_SAMPLES, "Result does not use frozen 2,000/5,000 audit settings")
        if family in {"solver_benchmark", "scalability_sparse"}:
            _benchmark(dataset, job_id, result, report)
        elif family == "cnc_main":
            _cnc(dataset, job_id, result, report)
        elif family == "policy_transplant":
            _transplant(dataset, job_id, result, report)
        else:
            raise ValueError(f"Unsupported audit family: {family}")
        report["status"] = "mismatch" if report["mismatches"] else "pass"
    except Exception as error:
        # Keep completed partial calculations and other jobs. No retry, repair,
        # change to the saved records, or tolerance adjustment happens here.
        report["status"] = "error"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    report["elapsed_seconds"] = time.monotonic() - started
    return _json_safe(report)


def _json_safe(value: Any) -> Any:
    """Retain any unexpected nonfinite result as explicit evidence in valid JSON.

    Such values already fail the numerical comparison. Tagged strings preserve
    them without allowing NaN/Infinity tokens or losing other completed jobs.
    """
    if isinstance(value, Mapping):
        return {key: _json_safe(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    return value


def _write_new_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def recompute_audits(dataset: Any, output_dir: str | Path, *, workers: int = 2, job_ids: Any = None) -> dict[str, Any]:
    """Recompute saved audit statistics, writing only new replay reports.

    The caller supplies a verified offline Dataset and an output location outside
    its immutable inputs. Selection defaults to the 276 paper-selected audit
    jobs. Explicit job IDs must be unique, present, and from these four families.
    Report-file collisions fail before any work; original outputs are never
    overwritten. Per-job data failures and numeric mismatches remain in reports.
    """
    _require(isinstance(workers, int) and not isinstance(workers, bool) and workers >= 1, "workers must be a positive integer")
    eligible = {job_id for job_id, job in dataset.jobs.items() if job["family"] in AUDIT_FAMILIES}
    selected = sorted(eligible) if job_ids is None else list(job_ids)
    _require(bool(selected), "No audit jobs selected")
    _require(all(isinstance(job_id, str) for job_id in selected), "Job IDs must be strings")
    _require(len(selected) == len(set(selected)), "Duplicate requested audit job IDs")
    _require(set(selected).issubset(eligible), f"Unknown or ineligible audit job IDs: {sorted(set(selected) - eligible)}")
    _require(all(Path(job_id).name == job_id and job_id not in {".", ".."} for job_id in selected), "Unsafe job ID")
    selected.sort()
    output_dir = dataset.validate_output_dir(Path(output_dir))
    job_dir = output_dir / "audit-jobs"
    summary_path = output_dir / "audit-summary.json"
    _require(not job_dir.exists() and not summary_path.exists(), "Audit output already exists; choose a new output directory")
    job_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    reports: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_recompute_job, dataset, job_id): job_id for job_id in selected}
        for future in as_completed(futures):
            report = future.result()
            reports[report["job_id"]] = report
            _write_new_json(job_dir / f"{report['job_id']}.json", report)
    ordered = [reports[job_id] for job_id in selected]
    errors = [report for report in ordered if report["status"] == "error"]
    mismatch_jobs = [report for report in ordered if report["status"] == "mismatch"]
    matched = [report for report in ordered if report["status"] == "pass"]
    summary = {
        "schema": "policy-cce-recomputed-audits-v1",
        "status": "pass" if not errors and not mismatch_jobs else "fail",
        "selected_job_count": len(selected),
        "completed_job_count": len(selected) - len(errors),
        "matched_job_count": len(matched),
        "mismatch_job_count": len(mismatch_jobs),
        "failed_job_count": len(errors),
        "mismatch_field_count": sum(len(report["mismatches"]) for report in ordered),
        "tolerances": {"absolute": ABSOLUTE_TOLERANCE, "relative": RELATIVE_TOLERANCE, "discrete_fields": "exact"},
        "settings": {"audit_rollouts": AUDIT_ROLLOUTS, "bootstrap_samples": BOOTSTRAP_SAMPLES, "alpha": ALPHA, "workers": workers, "seed_recipe": "original stable_solver_seed with original per-family suffix"},
        "scope": "saved-sample reanalysis only; no simulator, solver, distribution selection, or repair",
        "elapsed_seconds": time.monotonic() - started,
        "jobs": ordered,
        "failures": [{"job_id": report["job_id"], **report["error"]} for report in errors],
    }
    _write_new_json(summary_path, summary)
    return summary
