from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cmfg_cce.orchestration.manifest import atomic_json, canonical_json, sha256_bytes, sha256_file


VOLATILE_KEYS = {
    "runtime_seconds",
    "solver_seconds",
    "payoff_evaluation_seconds",
    "solver_compute_seconds",
    "eval_time_seconds",
    "created_utc",
    "completed_utc",
    "updated_utc",
    "hostname",
    "platform",
    "python",
}


def _stable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable(item)
            for key, item in sorted(value.items())
            if str(key) not in VOLATILE_KEYS
            and not str(key).endswith("_runtime_seconds")
        }
    if isinstance(value, list):
        return [_stable(item) for item in value]
    if isinstance(value, float):
        # The raw training/audit chunks and the replication-level NPZ are
        # compared byte for byte elsewhere in this gate.  Recomputing a
        # summary from those identical arrays can nevertheless differ in the
        # final floating-point bit across CPU architectures (for example,
        # 345.84420677506023 versus 345.8442067750603).  Canonicalize only the
        # derived JSON summaries to 12 significant digits so that such
        # sub-picounit reduction noise does not fail the release gate.  A
        # scientifically meaningful change remains visible, while the raw
        # evidence is still required to be exactly identical.
        return float(f"{value:.12g}")
    return value


def _evidence(payload: Mapping[str, Any], artifact: Path) -> dict[str, Any]:
    solver_results = dict(payload["solver_results"])
    return {
        "q_hashes": {
            solver: row["distribution"]["q_hash"]
            for solver, row in sorted(solver_results.items())
        },
        "training_raw_chunk_sha256": [
            row["payload_sha256"] for row in payload["training_checkpoint"]["chunks"]
        ],
        "formal_audit_raw_chunk_sha256": [
            row["payload_sha256"]
            for row in payload["formal_audit_checkpoint"]["chunks"]
        ],
        "formal_audit_sample_declared_sha256": payload["formal_audit_checkpoint"][
            "sample_artifact"
        ]["sha256"],
        "formal_audit_sample_file_sha256": sha256_file(artifact),
        "stable_result_sha256": sha256_bytes(canonical_json(_stable(payload))),
    }


def compare(
    amd64_result: Path,
    amd64_artifact: Path,
    arm64_result: Path,
    arm64_artifact: Path,
) -> dict[str, Any]:
    amd64 = _evidence(
        json.loads(amd64_result.read_text(encoding="utf-8")), amd64_artifact
    )
    arm64 = _evidence(
        json.loads(arm64_result.read_text(encoding="utf-8")), arm64_artifact
    )
    differing = sorted(key for key in amd64 if amd64[key] != arm64[key])
    if differing:
        raise RuntimeError(
            "Cross-architecture smoke differs for: " + ", ".join(differing)
        )
    return {
        "schema_version": "revision_full_v1_cross_arch_smoke_v1",
        "status": "passed",
        "matched_fields": sorted(amd64),
        "evidence": amd64,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AMD64 and ARM64 revision-smoke evidence.")
    parser.add_argument("--amd64-result", type=Path, required=True)
    parser.add_argument("--amd64-artifact", type=Path, required=True)
    parser.add_argument("--arm64-result", type=Path, required=True)
    parser.add_argument("--arm64-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = compare(
        args.amd64_result,
        args.amd64_artifact,
        args.arm64_result,
        args.arm64_artifact,
    )
    atomic_json(args.output, report)
    print(f"PASS cross-architecture identity: {report['evidence']['stable_result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
