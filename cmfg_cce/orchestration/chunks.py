from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import io
import json
import os
from pathlib import Path
import uuid
from typing import Any, Mapping
import zipfile

import numpy as np

from cmfg_cce.orchestration.manifest import atomic_json, sha256_bytes, sha256_file


CHUNK_SCHEMA_VERSION = "revision_full_v1_chunk_v1"
CHUNK_PLAN_SCHEMA_VERSION = "revision_full_v1_chunk_plan_v1"
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


class ChunkConflictError(RuntimeError):
    """Raised when an immutable chunk name is reused for different content."""


class ChunkStatus(str, Enum):
    MISSING = "missing"
    VALID = "valid"
    CORRUPT = "corrupt"


def deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Serialize arrays without wall-clock timestamps or mapping-order differences."""

    normalized: dict[str, np.ndarray] = {}
    for raw_name, value in arrays.items():
        name = str(raw_name)
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(f"Unsafe NPZ array name: {name!r}")
        if name in normalized:
            raise ValueError(f"Duplicate NPZ array name after normalization: {name!r}")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"Chunk array {name!r} may not use an object dtype.")
        normalized[name] = array

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(normalized):
            array_bytes = io.BytesIO()
            np.lib.format.write_array(array_bytes, normalized[name], allow_pickle=False)
            info = zipfile.ZipInfo(filename=f"{name}.npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_bytes.getvalue())
    return payload.getvalue()


@dataclass(frozen=True)
class ChunkRecord:
    campaign_sha256: str
    matrix_sha256: str
    job_id: str
    kind: str
    start: int
    stop: int
    payload_sha256: str
    payload_size: int
    arrays: Mapping[str, Mapping[str, Any]]
    created_utc: str
    schema_version: str = CHUNK_SCHEMA_VERSION

    @property
    def chunk_id(self) -> str:
        return f"{self.kind}-{self.start:09d}-{self.stop:09d}"

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_sha256": self.campaign_sha256,
            "matrix_sha256": self.matrix_sha256,
            "job_id": self.job_id,
            "kind": self.kind,
            "start": int(self.start),
            "stop": int(self.stop),
            "chunk_id": self.chunk_id,
            "payload_sha256": self.payload_sha256,
            "payload_size": int(self.payload_size),
            "arrays": {key: dict(value) for key, value in sorted(self.arrays.items())},
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ChunkRecord":
        if payload.get("schema_version") != CHUNK_SCHEMA_VERSION:
            raise ValueError("Unsupported chunk schema.")
        normalized = dict(payload)
        observed_id = normalized.pop("chunk_id", None)
        record = cls(**normalized)
        if observed_id != record.chunk_id:
            raise ValueError("Chunk ID does not match its bounds.")
        return record


@dataclass(frozen=True)
class ChunkPlan:
    campaign_sha256: str
    matrix_sha256: str
    job_id: str
    kind: str
    chunk_size: int
    item_ids: tuple[str, ...]
    item_order_sha256: str
    schema_version: str = CHUNK_PLAN_SCHEMA_VERSION

    @property
    def item_count(self) -> int:
        return len(self.item_ids)

    @classmethod
    def create(
        cls,
        *,
        campaign_sha256: str,
        matrix_sha256: str,
        job_id: str,
        kind: str,
        chunk_size: int,
        item_ids: list[str] | tuple[str, ...],
    ) -> "ChunkPlan":
        ordered = tuple(str(value) for value in item_ids)
        if not ordered or len(set(ordered)) != len(ordered):
            raise ValueError("A chunk plan requires nonempty unique item IDs in frozen order.")
        digest = sha256_bytes(json.dumps(ordered, separators=(",", ":")).encode("utf-8"))
        plan = cls(
            campaign_sha256=str(campaign_sha256),
            matrix_sha256=str(matrix_sha256),
            job_id=str(job_id),
            kind=str(kind),
            chunk_size=int(chunk_size),
            item_ids=ordered,
            item_order_sha256=digest,
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        expected_size = 128 if self.kind == "training" else 8 if self.kind == "audit" else None
        if expected_size is None or self.chunk_size != expected_size:
            raise ValueError("Chunk-plan kind or frozen chunk size is invalid.")
        if not self.item_ids or len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("Chunk-plan item IDs must be nonempty and unique.")
        digest = sha256_bytes(json.dumps(self.item_ids, separators=(",", ":")).encode("utf-8"))
        if digest != self.item_order_sha256:
            raise ValueError("Chunk-plan item ordering hash mismatch.")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_sha256": self.campaign_sha256,
            "matrix_sha256": self.matrix_sha256,
            "job_id": self.job_id,
            "kind": self.kind,
            "chunk_size": self.chunk_size,
            "item_count": self.item_count,
            "item_ids": list(self.item_ids),
            "item_order_sha256": self.item_order_sha256,
        }

    def write_immutable(self, path: str | Path) -> Path:
        self.validate()
        target = Path(path)
        if target.exists():
            observed = ChunkPlan.load(target)
            if observed != self:
                raise ChunkConflictError("An immutable chunk plan already exists with different content.")
            return target
        atomic_json(target, self.to_payload())
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ChunkPlan":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != CHUNK_PLAN_SCHEMA_VERSION:
            raise ValueError("Unsupported chunk-plan schema.")
        normalized = dict(payload)
        observed_count = int(normalized.pop("item_count", -1))
        normalized["item_ids"] = tuple(normalized.get("item_ids", ()))
        plan = cls(**normalized)
        plan.validate()
        if observed_count != plan.item_count:
            raise ValueError("Chunk-plan item count mismatch.")
        return plan


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ImmutableChunkStore:
    """Atomic, content-verified local storage for restartable profile chunks."""

    def __init__(
        self,
        root: str | Path,
        *,
        campaign_sha256: str,
        matrix_sha256: str,
        job_id: str,
        kind: str,
        item_count: int,
        chunk_size: int,
    ) -> None:
        self.root = Path(root)
        self.campaign_sha256 = str(campaign_sha256)
        self.matrix_sha256 = str(matrix_sha256)
        self.job_id = str(job_id)
        self.kind = str(kind)
        self.item_count = int(item_count)
        self.chunk_size = int(chunk_size)
        expected = 128 if self.kind == "training" else 8 if self.kind == "audit" else None
        if expected is None or self.chunk_size != expected:
            raise ValueError(f"{self.kind!r} chunks must use the frozen size {expected}.")
        if self.item_count <= 0:
            raise ValueError("item_count must be positive.")
        self.root.mkdir(parents=True, exist_ok=True)

    def bounds(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (start, min(self.item_count, start + self.chunk_size))
            for start in range(0, self.item_count, self.chunk_size)
        )

    def _stem(self, start: int, stop: int) -> str:
        self._validate_bounds(start, stop)
        return f"{self.kind}-{int(start):09d}-{int(stop):09d}"

    def paths(self, start: int, stop: int) -> tuple[Path, Path]:
        stem = self._stem(start, stop)
        return self.root / f"{stem}.npz", self.root / f"{stem}.json"

    def _validate_bounds(self, start: int, stop: int) -> None:
        start, stop = int(start), int(stop)
        if start < 0 or start >= stop or stop > self.item_count:
            raise ValueError(f"Invalid chunk bounds [{start}, {stop}).")
        if start % self.chunk_size != 0 or stop != min(self.item_count, start + self.chunk_size):
            raise ValueError(f"Chunk [{start}, {stop}) is not on the frozen {self.chunk_size}-item grid.")

    def _record_for(self, start: int, stop: int, payload: bytes, arrays: Mapping[str, np.ndarray]) -> ChunkRecord:
        array_meta = {
            str(name): {
                "shape": [int(value) for value in np.asarray(array).shape],
                "dtype": str(np.asarray(array).dtype),
            }
            for name, array in arrays.items()
        }
        return ChunkRecord(
            campaign_sha256=self.campaign_sha256,
            matrix_sha256=self.matrix_sha256,
            job_id=self.job_id,
            kind=self.kind,
            start=int(start),
            stop=int(stop),
            payload_sha256=sha256_bytes(payload),
            payload_size=len(payload),
            arrays=array_meta,
            created_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )

    def status(self, start: int, stop: int) -> ChunkStatus:
        payload_path, record_path = self.paths(start, stop)
        if not payload_path.exists() and not record_path.exists():
            return ChunkStatus.MISSING
        if not payload_path.is_file() or not record_path.is_file():
            return ChunkStatus.CORRUPT
        try:
            record = ChunkRecord.from_payload(json.loads(record_path.read_text(encoding="utf-8")))
            if (
                record.campaign_sha256 != self.campaign_sha256
                or record.matrix_sha256 != self.matrix_sha256
                or record.job_id != self.job_id
                or record.kind != self.kind
                or record.start != int(start)
                or record.stop != int(stop)
                or record.payload_size != payload_path.stat().st_size
                or record.payload_sha256 != sha256_file(payload_path)
            ):
                return ChunkStatus.CORRUPT
            with np.load(payload_path, allow_pickle=False) as values:
                if set(values.files) != set(record.arrays):
                    return ChunkStatus.CORRUPT
                for name, expected in record.arrays.items():
                    array = values[name]
                    if list(array.shape) != expected["shape"] or str(array.dtype) != expected["dtype"]:
                        return ChunkStatus.CORRUPT
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return ChunkStatus.CORRUPT
        return ChunkStatus.VALID

    def read(self, start: int, stop: int) -> dict[str, np.ndarray]:
        status = self.status(start, stop)
        if status is not ChunkStatus.VALID:
            raise ValueError(f"Cannot read {status.value} chunk [{start}, {stop}).")
        payload_path, _ = self.paths(start, stop)
        with np.load(payload_path, allow_pickle=False) as values:
            return {name: np.array(values[name], copy=True) for name in values.files}

    def _quarantine(self, start: int, stop: int) -> None:
        payload_path, record_path = self.paths(start, stop)
        suffix = f".corrupt-{uuid.uuid4().hex}"
        for path in (payload_path, record_path):
            if path.exists():
                path.replace(path.with_name(path.name + suffix))

    def commit(
        self,
        start: int,
        stop: int,
        arrays: Mapping[str, np.ndarray],
        *,
        repair_corrupt: bool = True,
    ) -> ChunkRecord:
        self._validate_bounds(start, stop)
        if not arrays:
            raise ValueError("A chunk must contain at least one named array.")
        normalized = {str(name): np.asarray(value) for name, value in arrays.items()}
        payload = deterministic_npz(normalized)
        record = self._record_for(start, stop, payload, normalized)
        current = self.status(start, stop)
        if current is ChunkStatus.VALID:
            _, record_path = self.paths(start, stop)
            existing = ChunkRecord.from_payload(json.loads(record_path.read_text(encoding="utf-8")))
            if existing.payload_sha256 != record.payload_sha256 or existing.arrays != record.arrays:
                raise ChunkConflictError(f"Immutable chunk {record.chunk_id} already has different content.")
            return existing
        if current is ChunkStatus.CORRUPT:
            if not repair_corrupt:
                raise ChunkConflictError(f"Chunk {record.chunk_id} is corrupt and repair is disabled.")
            self._quarantine(start, stop)
        payload_path, record_path = self.paths(start, stop)
        payload_tmp = payload_path.with_name(f".{payload_path.name}.{uuid.uuid4().hex}.tmp")
        record_tmp = record_path.with_name(f".{record_path.name}.{uuid.uuid4().hex}.tmp")
        with payload_tmp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        payload_tmp.replace(payload_path)
        _fsync_directory(self.root)
        atomic_json(record_tmp, record.to_payload())
        record_tmp.replace(record_path)
        _fsync_directory(self.root)
        if self.status(start, stop) is not ChunkStatus.VALID:
            raise IOError(f"Chunk {record.chunk_id} failed its post-commit validation.")
        return record

    def pending_bounds(self) -> tuple[tuple[int, int], ...]:
        return tuple(bounds for bounds in self.bounds() if self.status(*bounds) is not ChunkStatus.VALID)

    def validate_complete(self) -> tuple[ChunkRecord, ...]:
        records: list[ChunkRecord] = []
        bad: list[str] = []
        for start, stop in self.bounds():
            if self.status(start, stop) is not ChunkStatus.VALID:
                bad.append(f"[{start},{stop})")
                continue
            _, record_path = self.paths(start, stop)
            records.append(ChunkRecord.from_payload(json.loads(record_path.read_text(encoding="utf-8"))))
        if bad:
            raise ValueError(f"Job {self.job_id} has missing or corrupt chunks: {', '.join(bad)}")
        return tuple(records)
