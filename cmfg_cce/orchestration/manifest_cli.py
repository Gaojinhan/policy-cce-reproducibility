from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable, Sequence

from cmfg_cce.experiments.common import solver_seed
from cmfg_cce.experiments.revision_backends import cnc_seed
from cmfg_cce.experiments.revision_full_v1_spec import (
    RevisionJobKey,
    default_revision_matrix,
)
from cmfg_cce.orchestration.manifest import (
    CampaignManifest,
    JobSpec,
    SourceManifest,
    atomic_json,
    canonical_json,
    sha256_bytes,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "cmfg_cce/configs/revision_full_v1.yaml"
DEFAULT_LOCK = ROOT / "infra/revision-full-v1/requirements.lock"
PRODUCTION_RUNNER_MODULE = "cmfg_cce.experiments.run_revision_campaign"
DEFAULT_SOURCE_PATHS = (
    "cmfg_cce",
    "scripts/revision_full_v1",
    "infra/revision-full-v1/requirements.lock",
    "pyproject.toml",
    "AGENTS.md",
)

# These small JSON checkpoints are intentionally separate from the immutable
# numerical chunks.  The worker publishes them through hash-addressed objects
# plus a mutable pointer, allowing a recreated Spot VM to resume an adaptive
# search without changing the frozen q or profile-to-slot assignment.
RUNNER_SIDECARS = (
    "runner_identity.json",
    "stages/training/chunks/assignment_checkpoint.json",
    "stages/training/training_result.json",
    "stages/training/frozen_solver_outputs.json",
)
CNC_SEED_FAMILIES = {
    "cnc_main",
    "selection_sensitivity",
    "policy_library_sensitivity",
    "parameter_robustness",
    "policy_transplant",
}
FORMAL_SAMPLE_ARTIFACT_FAMILIES = {
    "solver_benchmark",
    "mixed_challenge",
    "scalability_exact",
    "scalability_sparse",
}


def _expected_artifacts(key: RevisionJobKey) -> tuple[str, ...]:
    if key.family in FORMAL_SAMPLE_ARTIFACT_FAMILIES:
        return ("result.json", "formal_audit_samples.npz")
    if key.family == "policy_transplant":
        return ("result.json", "transplant_audit_samples.npz")
    return ("result.json",)


def _all_matrix_jobs(matrix: object) -> tuple[RevisionJobKey, ...]:
    physical = getattr(matrix, "physical_jobs", None)
    if callable(physical):
        items = tuple(physical())
        if not items or any(not isinstance(item, RevisionJobKey) for item in items):
            raise ValueError("physical_jobs() must return nonempty RevisionJobKey values.")
        jobs = {item.job_id: item for item in items}
        if len(jobs) != len(items):
            raise ValueError("physical_jobs() emits duplicate job IDs.")
        return items

    # Compatibility fallback for older matrix objects.
    jobs: dict[str, RevisionJobKey] = {}
    for name in sorted(dir(matrix)):
        if not name.endswith("_jobs"):
            continue
        method = getattr(matrix, name)
        if not callable(method):
            continue
        for item in method():
            if not isinstance(item, RevisionJobKey):
                continue
            if item.job_id in jobs and jobs[item.job_id] != item:
                raise ValueError(f"Matrix emits conflicting duplicate job {item.job_id}.")
            jobs[item.job_id] = item
    return tuple(jobs[key] for key in sorted(jobs))


def _audit_seed_namespace(key: RevisionJobKey) -> str:
    if key.family == "policy_transplant" or "transplant_audit" in key.stages:
        return "transplant_audit"
    if key.family == "mwu_tuning_pipeline" or "mwu_tuning_audit" in key.stages:
        return "mwu_tuning_audit"
    if key.phase == "evaluation" or (
        key.stages == ("outcome_evaluation",)
    ):
        return "outcome_evaluation"
    return "formal_audit"


def _stage_seed_payloads(
    key: RevisionJobKey,
    offsets: dict[str, int],
) -> dict[str, dict[str, int]]:
    namespace_by_stage = {
        "training": "training",
        "formal_audit": "formal_audit",
        "outcome_evaluation": "outcome_evaluation",
        "mwu_tuning_audit": "mwu_tuning_audit",
        "transplant_audit": "transplant_audit",
        "runtime_measurement": "training",
    }
    return {
        stage: _seed_payload_for_key(key, offsets[namespace_by_stage[stage]])
        for stage in key.stages
    }


def _seed_payload(seed: int, offset: int) -> dict[str, int]:
    base = solver_seed(int(seed))
    return {
        "type_seed": int(base.type_seed),
        "order_seed": int(base.order_seed) + int(offset),
        "outside_seed": int(base.outside_seed) + int(offset),
        "availability_seed": int(base.availability_seed) + int(offset),
        "tie_break_seed": int(base.tie_break_seed) + int(offset),
        "rollout_replication_seed": int(base.rollout_replication_seed) + int(offset),
    }


def _seed_payload_for_key(key: RevisionJobKey, offset: int) -> dict[str, int]:
    base = cnc_seed(int(key.seed)) if key.family in CNC_SEED_FAMILIES else solver_seed(int(key.seed))
    return {
        "type_seed": int(base.type_seed),
        "order_seed": int(base.order_seed) + int(offset),
        "outside_seed": int(base.outside_seed) + int(offset),
        "availability_seed": int(base.availability_seed) + int(offset),
        "tie_break_seed": int(base.tie_break_seed) + int(offset),
        "rollout_replication_seed": int(base.rollout_replication_seed) + int(offset),
    }


def _formal_selection_metadata(
    path: Path,
    *,
    matrix: object,
) -> dict[str, object]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "revision_full_v1_global_mwu_selection_v2"
        or payload.get("status") != "complete"
        or payload.get("matrix_sha256") != getattr(matrix, "matrix_hash")
    ):
        raise ValueError("Formal campaign requires a complete MWU selection for this matrix.")
    unsigned = dict(payload)
    observed_selection_hash = str(unsigned.pop("selection_sha256", ""))
    if observed_selection_hash != sha256_bytes(canonical_json(unsigned)):
        raise ValueError("MWU selection hash mismatch while freezing the formal manifest.")
    selected = dict(payload.get("selected_config", {}))
    config_id = str(selected.get("mwu_config_id", ""))
    frozen_ids = {
        str(config.config_id): config for config in getattr(matrix, "mwu_tuning_configs")
    }
    if config_id not in frozen_ids:
        raise ValueError("Formal MWU selection is not in the frozen tuning grid.")
    expected = frozen_ids[config_id]
    observed_values = (
        float(selected.get("eta", float("nan"))),
        str(selected.get("schedule", "")),
        float(selected.get("exploration_floor", float("nan"))),
        int(selected.get("burn_in_rounds", -1)),
    )
    expected_values = (
        float(expected.eta),
        str(expected.schedule),
        float(expected.exploration_floor),
        int(expected.burn_in_rounds),
    )
    if observed_values != expected_values:
        raise ValueError("Formal MWU hyperparameter selection is invalid.")
    if "formal_rounds" in selected or "formal_rounds_rule" in selected:
        raise ValueError("The global MWU selection must not freeze a formal round budget.")
    return {
        "selection_sha256": observed_selection_hash,
        "selection_file_sha256": sha256_file(source),
        "mwu_config_id": config_id,
    }


def _jobs_for_campaign_phase(
    keys: Sequence[RevisionJobKey],
    phase: str,
) -> tuple[RevisionJobKey, ...]:
    if phase == "all":
        return tuple(keys)
    if phase == "calibration":
        selected = tuple(key for key in keys if key.family == "mwu_tuning_pipeline")
    elif phase == "formal":
        selected = tuple(key for key in keys if key.family != "mwu_tuning_pipeline")
    else:
        raise ValueError("campaign phase must be all, calibration, or formal")
    if not selected:
        raise ValueError(f"Campaign phase {phase!r} contains no jobs.")
    return selected


def _estimate_vcpu_hours(key: RevisionJobKey, *, train_rollouts: int, audit_rollouts: int) -> float:
    # Calibrated from the completed N=5,J=8,R=200 tensor on the M4: 7--8 vCPU-hours.
    episode_vcpu_hours = 1.15e-6
    n, j = int(key.n_agents), int(key.policies_per_agent)
    full_profiles = j**n
    sparse_training_profiles = min(full_profiles, max(512, 32 * n * j))
    closure_profiles = min(full_profiles, max(128, 24 * n * j))
    stages = set(key.stages)

    if key.family in {"solver_benchmark", "mixed_challenge", "scalability_exact"}:
        training_profiles = full_profiles
    elif "training" in stages:
        training_profiles = sparse_training_profiles
    else:
        training_profiles = 0

    episodes = training_profiles * int(train_rollouts)
    if "formal_audit" in stages:
        episodes += closure_profiles * int(audit_rollouts)
    if "outcome_evaluation" in stages:
        episodes += min(full_profiles, max(32, 4 * n * j)) * 500
    if "transplant_audit" in stages:
        episodes += min(full_profiles, 3 * closure_profiles) * int(audit_rollouts)
    if "mwu_tuning_audit" in stages:
        episodes += min(full_profiles, 17 * closure_profiles) * 500

    if key.family in {"solver_benchmark_runtime", "scalability_exact_runtime"}:
        # FullTensor and ExhaustiveCG each start from an empty payoff cache;
        # DSS and MWU add two sparse attempts.
        episodes = (2 * full_profiles + 2 * sparse_training_profiles) * int(train_rollouts)
    elif key.family == "scalability_sparse_runtime":
        episodes = 2 * sparse_training_profiles * int(train_rollouts)
    elif key.family == "mwu_tuning_pipeline":
        # One DSS reference plus sixteen matched-time MWU configurations.
        episodes += 17 * sparse_training_profiles * int(train_rollouts)
    return max(0.01, float(episodes) * episode_vcpu_hours)


def _assign_lpt(
    keys: Iterable[RevisionJobKey],
    costs: dict[str, float],
) -> dict[str, tuple[str, int]]:
    gcp_runtime_slot = ("gcp-spot", 0)
    x86_slots = [
        ("gcp-spot", 0),
        ("gcp-spot", 1),
        ("gcp-spot", 2),
        ("gcp-spot", 3),
        ("win1", 0),
        ("win2", 0),
    ]
    all_slots = [*x86_slots, ("mac", 0)]
    loads = {slot: 0.0 for slot in all_slots}
    assignments: dict[str, tuple[str, int]] = {}
    for key in sorted(keys, key=lambda item: (-costs[item.job_id], item.job_id)):
        if key.resource_mode == "exclusive_runtime":
            eligible = [gcp_runtime_slot]
        else:
            # Formal audits are bound to their physical training pipeline, so
            # excluding the Mac here would leave that machine idle.  The
            # deterministic spawn backend, source/image hashes, canonical
            # solver seeds, and cross-platform smoke checks make every bulk
            # pipeline portable across the seven available slots.
            eligible = all_slots
        slot = min(eligible, key=lambda value: (loads[value], value))
        assignments[key.job_id] = slot
        loads[slot] += costs[key.job_id]
    return assignments


def build_campaign(
    *,
    source_manifest: SourceManifest,
    dependency_lock: Path,
    config_path: Path,
    runner_module: str = PRODUCTION_RUNNER_MODULE,
    image_amd64: str,
    image_arm64: str,
    phase: str = "all",
    mwu_selection: Path | None = None,
) -> CampaignManifest:
    if runner_module != PRODUCTION_RUNNER_MODULE:
        raise ValueError(
            "revision-full-v1 manifests must use the unified campaign dispatcher "
            f"{PRODUCTION_RUNNER_MODULE!r}."
        )
    matrix = default_revision_matrix()
    keys = _jobs_for_campaign_phase(_all_matrix_jobs(matrix), phase)
    selection_metadata: dict[str, object] | None = None
    if phase == "formal":
        if mwu_selection is None:
            raise ValueError("--mwu-selection is required when building the formal campaign.")
        selection_metadata = _formal_selection_metadata(mwu_selection, matrix=matrix)
    elif mwu_selection is not None:
        raise ValueError("An MWU selection may only be frozen into the formal campaign.")
    costs = {
        key.job_id: _estimate_vcpu_hours(
            key,
            train_rollouts=matrix.train_rollouts,
            audit_rollouts=matrix.audit_rollouts,
        )
        for key in keys
    }
    assignments = _assign_lpt(keys, costs)
    config_hash = sha256_file(config_path)
    offsets = matrix.seed_namespaces.offsets()
    cnc_dependencies = {
        (key.mechanism, key.seed, key.variant): key.job_id
        for key in matrix.cnc_main_jobs()
    }
    library_base_dependencies = {
        (key.mechanism, key.seed, key.variant.rsplit("__", maxsplit=1)[0]): key.job_id
        for key in keys
        if key.family == "policy_library_sensitivity"
        and key.variant.endswith("__base_a1_a6")
    }
    runtime_jobs = {
        (
            key.family.removesuffix("_runtime"),
            key.n_agents,
            key.policies_per_agent,
            key.mechanism,
            key.seed,
        ): key.job_id
        for key in keys
        if key.family
        in {
            "solver_benchmark_runtime",
            "scalability_exact_runtime",
            "scalability_sparse_runtime",
        }
    }
    jobs: list[JobSpec] = []
    for key in keys:
        node_id, slot = assignments[key.job_id]
        kind = "training" if key.phase == "training" else "audit"
        chunk_size = 128 if kind == "training" else 8
        dependencies: tuple[str, ...] = ()
        if key.family in {
            "solver_benchmark",
            "scalability_exact",
            "scalability_sparse",
        }:
            runtime_key = (
                key.family,
                key.n_agents,
                key.policies_per_agent,
                key.mechanism,
                key.seed,
            )
            try:
                dependencies = (runtime_jobs[runtime_key],)
            except KeyError as exc:
                raise ValueError(
                    f"Evidence job {key.job_id} has no paired GCP runtime-q job."
                ) from exc
        elif key.family == "policy_transplant":
            source_key = (
                str(key.source_mechanism),
                int(key.seed),
                str(key.variant),
            )
            target_key = (
                str(key.target_mechanism),
                int(key.seed),
                str(key.variant),
            )
            try:
                dependencies = (
                    cnc_dependencies[source_key],
                    cnc_dependencies[target_key],
                )
            except KeyError as exc:
                raise ValueError(
                    f"Policy transplant {key.job_id} has no matching CNC source/target job."
                ) from exc
        elif (
            key.family == "policy_library_sensitivity"
            and key.variant.endswith("__expanded_a1_a8")
        ):
            condition = key.variant.rsplit("__", maxsplit=1)[0]
            dependency_key = (key.mechanism, key.seed, condition)
            try:
                dependencies = (library_base_dependencies[dependency_key],)
            except KeyError as exc:
                raise ValueError(
                    f"Expanded-library job {key.job_id} has no matched A1-A6 dependency."
                ) from exc
        metadata = {
            **asdict(key),
            "campaign_phase": phase,
            "stage_seed_namespaces": _stage_seed_payloads(key, offsets),
            "runner_sidecars": list(RUNNER_SIDECARS),
        }
        if key.family in {
            "solver_benchmark",
            "scalability_exact",
            "scalability_sparse",
        }:
            metadata["runtime_q_source_job_id"] = dependencies[0]
        if selection_metadata is not None:
            metadata["mwu_selection"] = dict(selection_metadata)
        jobs.append(
            JobSpec(
                job_id=key.job_id,
                phase=key.phase,
                family=key.family,
                node_id=node_id,
                slot=slot,
                argv=(
                    "python",
                    "-m",
                    runner_module,
                    "--job-id",
                    "{job_id}",
                    "--matrix-hash",
                    "{matrix_hash}",
                    "--state-dir",
                    "{state_dir}",
                    "--output-dir",
                    "{output_dir}",
                    "--chunk-start",
                    "{chunk_start}",
                    "--chunk-stop",
                    "{chunk_stop}",
                ),
                config_sha256=config_hash,
                matrix_sha256=matrix.matrix_hash,
                estimated_vcpu_hours=costs[key.job_id],
                item_count=0,
                chunk_kind=kind,
                chunk_size=chunk_size,
                runner_managed_chunks=True,
                expected_artifacts=_expected_artifacts(key),
                depends_on=dependencies,
                training_seeds=_seed_payload_for_key(key, offsets["training"]),
                audit_seeds=_seed_payload_for_key(
                    key,
                    offsets[_audit_seed_namespace(key)],
                ),
                metadata=metadata,
            )
        )
    campaign = CampaignManifest(
        campaign_id=(matrix.campaign if phase == "all" else f"{matrix.campaign}-{phase}"),
        source_sha256=source_manifest.source_sha256,
        dependency_lock_sha256=sha256_file(dependency_lock),
        matrix_sha256=matrix.matrix_hash,
        image_by_platform={
            "linux/amd64": image_amd64,
            "linux/arm64": image_arm64,
        },
        jobs=tuple(jobs),
        spot_vm_hour_soft_limit=300.0,
    )
    campaign.validate()
    return campaign


def command_freeze(args: argparse.Namespace) -> None:
    paths = tuple(args.paths) if args.paths else DEFAULT_SOURCE_PATHS
    manifest = SourceManifest.build(args.root, paths)
    atomic_json(args.output, manifest.to_payload())
    print(f"source_sha256={manifest.source_sha256} files={len(manifest.files)}")


def command_build(args: argparse.Namespace) -> None:
    source = SourceManifest.from_payload(json.loads(args.source_manifest.read_text(encoding="utf-8")))
    source.validate_tree(args.root)
    campaign = build_campaign(
        source_manifest=source,
        dependency_lock=args.dependency_lock,
        config_path=args.config,
        runner_module=args.runner_module,
        image_amd64=args.image_amd64,
        image_arm64=args.image_arm64,
        phase=args.phase,
        mwu_selection=args.mwu_selection,
    )
    campaign.write(args.output)
    assignment_rows = [
        {
            "job_id": job.job_id,
            "phase": job.phase,
            "family": job.family,
            "node_id": job.node_id,
            "slot": job.slot,
            "estimated_vcpu_hours": job.estimated_vcpu_hours,
        }
        for job in campaign.jobs
    ]
    atomic_json(args.output.with_name("job_assignments.json"), {"jobs": assignment_rows})
    print(f"campaign_sha256={campaign.campaign_sha256} jobs={len(campaign.jobs)}")


def command_validate(args: argparse.Namespace) -> None:
    source = SourceManifest.from_payload(json.loads(args.source_manifest.read_text(encoding="utf-8")))
    source.validate_tree(args.root)
    campaign = CampaignManifest.load(args.campaign)
    if campaign.source_sha256 != source.source_sha256:
        raise SystemExit("Campaign/source hash mismatch.")
    if campaign.dependency_lock_sha256 != sha256_file(args.dependency_lock):
        raise SystemExit("Campaign/dependency-lock hash mismatch.")
    matrix = default_revision_matrix()
    if campaign.matrix_sha256 != matrix.matrix_hash:
        raise SystemExit("Campaign/current matrix hash mismatch.")
    expected_argv = ("python", "-m", PRODUCTION_RUNNER_MODULE)
    if any(tuple(job.argv[:3]) != expected_argv for job in campaign.jobs):
        raise SystemExit("Campaign contains a job that bypasses the unified dispatcher.")
    print(
        f"valid campaign={campaign.campaign_id} sha256={campaign.campaign_sha256} "
        f"jobs={len(campaign.jobs)}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze and validate revision-full-v1 manifests.")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-source")
    freeze.add_argument("--root", type=Path, default=ROOT)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("paths", nargs="*")
    freeze.set_defaults(handler=command_freeze)
    build = sub.add_parser("build-campaign")
    build.add_argument("--root", type=Path, default=ROOT)
    build.add_argument("--source-manifest", type=Path, required=True)
    build.add_argument("--dependency-lock", type=Path, default=DEFAULT_LOCK)
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build.add_argument("--runner-module", default=PRODUCTION_RUNNER_MODULE)
    build.add_argument("--image-amd64", required=True)
    build.add_argument("--image-arm64", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--phase",
        choices=("all", "calibration", "formal"),
        default="all",
    )
    build.add_argument("--mwu-selection", type=Path)
    build.set_defaults(handler=command_build)
    validate = sub.add_parser("validate")
    validate.add_argument("--root", type=Path, default=ROOT)
    validate.add_argument("--source-manifest", type=Path, required=True)
    validate.add_argument("--dependency-lock", type=Path, default=DEFAULT_LOCK)
    validate.add_argument("--campaign", type=Path, required=True)
    validate.set_defaults(handler=command_validate)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
