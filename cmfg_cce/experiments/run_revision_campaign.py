from __future__ import annotations

"""Single worker entrypoint for every ``revision-full-v1`` physical job.

The immutable campaign manifest intentionally names one runner module.  The
individual experiment families retain separate implementations, while this
module resolves the frozen job ID and delegates to exactly one implementation.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

from cmfg_cce.experiments.revision_full_v1_spec import (
    RevisionFullV1Matrix,
    RevisionJobKey,
)
from cmfg_cce.experiments.run_revision_cnc import run_cnc_revision_pipeline
from cmfg_cce.experiments.run_revision_full_v1 import (
    GENERAL_FAMILIES,
    run_general_physical_job,
)
from cmfg_cce.experiments.run_revision_mwu_tuning import run_mwu_tuning_pipeline
from cmfg_cce.experiments.run_revision_transplant import (
    run_policy_transplant_pipeline,
)


CNC_FAMILIES = {
    "cnc_main",
    "selection_sensitivity",
    "policy_library_sensitivity",
    "parameter_robustness",
}
TUNING_FAMILIES = {"mwu_tuning_pipeline"}
TRANSPLANT_FAMILIES = {"policy_transplant"}
DISPATCHED_FAMILIES = (
    GENERAL_FAMILIES | CNC_FAMILIES | TUNING_FAMILIES | TRANSPLANT_FAMILIES
)


def find_campaign_job(matrix: RevisionFullV1Matrix, job_id: str) -> RevisionJobKey:
    matches = [job for job in matrix.physical_jobs() if job.job_id == str(job_id)]
    if len(matches) != 1:
        raise ValueError(
            "job-id must identify exactly one frozen revision-full-v1 physical job; "
            f"found {len(matches)} matches for {job_id!r}."
        )
    job = matches[0]
    if job.family not in DISPATCHED_FAMILIES:
        raise ValueError(f"No revision-full-v1 runner is registered for {job.family!r}.")
    return job


def run_revision_campaign_job(
    *,
    job_id: str,
    state_dir: Path,
    output_dir: Path,
    matrix_hash: str,
    workers: int,
    dependency_dir: Path | None = None,
    mwu_selection_path: Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    matrix = RevisionFullV1Matrix()
    job = find_campaign_job(matrix, job_id)
    common = {
        "job_id": job.job_id,
        "state_dir": Path(state_dir),
        "output_dir": Path(output_dir),
        "matrix_hash": str(matrix_hash),
        "workers": max(1, int(workers)),
        "smoke": bool(smoke),
    }
    if job.family in GENERAL_FAMILIES:
        return run_general_physical_job(
            **common,
            mwu_selection_path=mwu_selection_path,
            dependency_dir=dependency_dir,
        )
    if job.family in CNC_FAMILIES:
        return run_cnc_revision_pipeline(
            **common,
            dependency_dir=dependency_dir,
        )
    if job.family in TUNING_FAMILIES:
        return run_mwu_tuning_pipeline(**common)
    if job.family in TRANSPLANT_FAMILIES:
        return run_policy_transplant_pipeline(
            **common,
            dependency_dir=dependency_dir,
        )
    raise AssertionError(f"Unhandled revision family: {job.family}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one physical job from the revision-full-v1 campaign."
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix-hash", required=True)
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("CMFG_WORKERS", "1"))
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
    parser.add_argument(
        "--mwu-selection",
        type=Path,
        default=(
            Path(os.environ["CMFG_MWU_SELECTION_PATH"])
            if os.environ.get("CMFG_MWU_SELECTION_PATH")
            else None
        ),
    )
    # The family runners own their immutable profile plans.  These generic
    # worker arguments are accepted only to keep one manifest command shape.
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--chunk-stop", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_revision_campaign_job(
        job_id=args.job_id,
        state_dir=args.state_dir,
        output_dir=args.output_dir,
        matrix_hash=args.matrix_hash,
        workers=args.workers,
        dependency_dir=args.dependency_dir,
        mwu_selection_path=args.mwu_selection,
        smoke=bool(args.smoke),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
