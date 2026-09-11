from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from cmfg_cce.orchestration.chunks import ImmutableChunkStore
from cmfg_cce.orchestration.manifest import (
    CampaignManifest,
    JobSpec,
    SourceManifest,
    atomic_json,
    sha256_bytes,
    sha256_file,
)


MATRIX_SHA256 = sha256_bytes(b"revision-full-v1-recovery-smoke-matrix-v1")
CONFIG_SHA256 = sha256_bytes(b"revision-full-v1-recovery-smoke-config-v1")
ITEM_COUNT = 512
CHUNK_SIZE = 128


def _result_payload(mode: str) -> dict[str, object]:
    if mode == "resumable":
        values = np.arange(ITEM_COUNT, dtype=np.int64)
        transformed = values * values + 17 * values + 23
        return {
            "schema_version": "revision_full_v1_recovery_smoke_result_v1",
            "mode": mode,
            "item_count": ITEM_COUNT,
            "values_sha256": sha256_bytes(transformed.tobytes(order="C")),
            "resumed_from_immutable_chunks": True,
        }
    if mode == "exclusive_runtime":
        return {
            "schema_version": "revision_full_v1_recovery_smoke_result_v1",
            "mode": mode,
            "restart_from_empty_cache_contract": True,
            "deterministic_checksum": sha256_bytes(
                np.linspace(0.0, 1.0, 4096, dtype=np.float64).tobytes(order="C")
            ),
        }
    raise ValueError(f"Unknown recovery-smoke mode: {mode}")


def _write_reference(mode: str, output: Path) -> None:
    atomic_json(output, _result_payload(mode))


def _run_resumable(output_dir: Path, delay_seconds: float) -> None:
    root = Path(os.environ["CMFG_STATE_DIR"])
    start = int(os.environ["CMFG_CHUNK_START"])
    stop = int(os.environ["CMFG_CHUNK_STOP"])
    store = ImmutableChunkStore(
        root / "chunks",
        campaign_sha256=os.environ["CMFG_CAMPAIGN_SHA256"],
        matrix_sha256=os.environ["CMFG_MATRIX_SHA256"],
        job_id=os.environ["CMFG_JOB_ID"],
        kind="training",
        item_count=ITEM_COUNT,
        chunk_size=CHUNK_SIZE,
    )
    if delay_seconds:
        time.sleep(delay_seconds)
    indices = np.arange(start, stop, dtype=np.int64)
    store.commit(start, stop, {"values": indices * indices + 17 * indices + 23})
    if stop == ITEM_COUNT:
        # Completion is possible only if the worker restored every preceding
        # immutable chunk after recreation.
        combined = np.concatenate(
            [store.read(chunk_start, chunk_stop)["values"] for chunk_start, chunk_stop in store.bounds()]
        )
        expected = _result_payload("resumable")
        if sha256_bytes(combined.tobytes(order="C")) != expected["values_sha256"]:
            raise RuntimeError("Recovered resumable chunks differ from the deterministic reference.")
        atomic_json(output_dir / "result.json", expected)


def _run_exclusive(output_dir: Path, delay_seconds: float) -> None:
    if os.environ.get("CMFG_RESOURCE_MODE") != "exclusive_runtime":
        raise RuntimeError("The recovery smoke's runtime attempt was not scheduled exclusively.")
    if os.environ.get("CMFG_RESTART_FROM_EMPTY_CACHE") != "1":
        raise RuntimeError("Exclusive recovery smoke did not receive the empty-cache contract.")
    job_root = Path(os.environ["CMFG_STATE_DIR"])
    residue = job_root / "exclusive_attempt_residue.json"
    if residue.exists():
        # Exclusive runtime evidence is an atomic empty-cache attempt. A host
        # restart may retain a local volume, but it must never stitch the old
        # partial attempt into the new wall-clock measurement.
        residue.unlink()
    atomic_json(
        residue,
        {
            "session": os.environ.get("CMFG_INSTANCE_ID", "unknown"),
            "started": True,
        },
    )
    if delay_seconds:
        time.sleep(delay_seconds)
    atomic_json(output_dir / "result.json", _result_payload("exclusive_runtime"))


def run_job(mode: str, output_dir: Path, delay_seconds: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "resumable":
        _run_resumable(output_dir, delay_seconds)
    elif mode == "exclusive_runtime":
        _run_exclusive(output_dir, delay_seconds)
    else:
        raise ValueError(f"Unknown recovery-smoke mode: {mode}")


def build_campaign(
    *,
    source_manifest_path: Path,
    dependency_lock: Path,
    image_amd64: str,
    image_arm64: str,
    output: Path,
    run_id: str,
) -> CampaignManifest:
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in run_id):
        raise ValueError("run-id must contain only lowercase letters, digits, and hyphens.")
    source = SourceManifest.from_payload(
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
    )
    suffix = sha256_bytes(f"{source.source_sha256}:{run_id}".encode("utf-8"))[:12]
    resumable_id = f"recovery_smoke_resumable_{suffix}"
    exclusive_id = f"recovery_smoke_exclusive_{suffix}"
    common = {
        "phase": "smoke",
        "node_id": "gcp-spot",
        "slot": 0,
        "config_sha256": CONFIG_SHA256,
        "matrix_sha256": MATRIX_SHA256,
        "estimated_vcpu_hours": 0.01,
        "expected_artifacts": ("result.json",),
    }
    resumable = JobSpec(
        job_id=resumable_id,
        family="recovery_smoke_resumable",
        argv=(
            "python",
            "-m",
            "cmfg_cce.orchestration.recovery_smoke",
            "run",
            "--mode",
            "resumable",
            "--output-dir",
            "{output_dir}",
        ),
        item_count=ITEM_COUNT,
        chunk_kind="training",
        chunk_size=CHUNK_SIZE,
        metadata={"resource_mode": "bulk", "recovery_smoke": True},
        **common,
    )
    exclusive = JobSpec(
        job_id=exclusive_id,
        family="recovery_smoke_exclusive",
        argv=(
            "python",
            "-m",
            "cmfg_cce.orchestration.recovery_smoke",
            "run",
            "--mode",
            "exclusive_runtime",
            "--output-dir",
            "{output_dir}",
        ),
        depends_on=(resumable_id,),
        metadata={"resource_mode": "exclusive_runtime", "recovery_smoke": True},
        **common,
    )
    campaign = CampaignManifest(
        campaign_id="revision-full-v1-calibration",
        source_sha256=source.source_sha256,
        dependency_lock_sha256=sha256_file(dependency_lock),
        matrix_sha256=MATRIX_SHA256,
        image_by_platform={
            "linux/amd64": image_amd64,
            "linux/arm64": image_arm64,
        },
        jobs=(resumable, exclusive),
        spot_vm_hour_soft_limit=1.0,
    )
    campaign.write(output)
    return campaign


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dedicated Spot recreation recovery smoke.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-campaign")
    build.add_argument("--source-manifest", type=Path, required=True)
    build.add_argument("--dependency-lock", type=Path, required=True)
    build.add_argument("--image-amd64", required=True)
    build.add_argument("--image-arm64", required=True)
    build.add_argument("--run-id", required=True)
    build.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--mode", choices=("resumable", "exclusive_runtime"), required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--delay-seconds",
        type=float,
        default=float(os.environ.get("CMFG_RECOVERY_SMOKE_DELAY_SECONDS", "90")),
    )
    reference = subparsers.add_parser("reference")
    reference.add_argument("--mode", choices=("resumable", "exclusive_runtime"), required=True)
    reference.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build-campaign":
        build_campaign(
            source_manifest_path=args.source_manifest,
            dependency_lock=args.dependency_lock,
            image_amd64=args.image_amd64,
            image_arm64=args.image_arm64,
            output=args.output,
            run_id=args.run_id,
        )
    elif args.command == "run":
        run_job(args.mode, args.output_dir, max(0.0, float(args.delay_seconds)))
    else:
        _write_reference(args.mode, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
