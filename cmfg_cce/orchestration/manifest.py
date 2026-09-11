from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "revision_full_v1_campaign_v1"
SOURCE_SCHEMA_VERSION = "revision_full_v1_source_v1"
DYNAMIC_SEED_FIELDS = (
    "order_seed",
    "outside_seed",
    "availability_seed",
    "tie_break_seed",
    "rollout_replication_seed",
)


class ManifestError(ValueError):
    """Raised when a frozen campaign identity is missing or inconsistent."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
    temporary.replace(target)
    return target


@dataclass(frozen=True)
class SourceFile:
    path: str
    size: int
    sha256: str

    @classmethod
    def from_path(cls, root: Path, path: Path) -> "SourceFile":
        relative = path.relative_to(root).as_posix()
        return cls(path=relative, size=int(path.stat().st_size), sha256=sha256_file(path))


@dataclass(frozen=True)
class SourceManifest:
    files: tuple[SourceFile, ...]
    schema_version: str = SOURCE_SCHEMA_VERSION

    @property
    def source_sha256(self) -> str:
        return sha256_bytes(canonical_json(self.to_payload(include_hash=False)))

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "files": [asdict(item) for item in self.files],
        }
        if include_hash:
            payload["source_sha256"] = self.source_sha256
        return payload

    @classmethod
    def build(
        cls,
        root: str | Path,
        paths: Iterable[str | Path],
        *,
        excluded_parts: Sequence[str] = ("__pycache__", ".pytest_cache"),
    ) -> "SourceManifest":
        root_path = Path(root).resolve()
        files: dict[str, SourceFile] = {}
        for raw_path in paths:
            candidate = (root_path / raw_path).resolve()
            if not candidate.is_relative_to(root_path):
                raise ManifestError(f"Source path escapes the root: {raw_path}")
            candidates = [candidate] if candidate.is_file() else sorted(candidate.rglob("*"))
            for path in candidates:
                if not path.is_file() or any(part in excluded_parts for part in path.parts):
                    continue
                record = SourceFile.from_path(root_path, path)
                files[record.path] = record
        if not files:
            raise ManifestError("The source manifest must contain at least one file.")
        return cls(files=tuple(files[key] for key in sorted(files)))

    def validate_tree(self, root: str | Path) -> None:
        root_path = Path(root).resolve()
        for record in self.files:
            path = (root_path / record.path).resolve()
            if not path.is_relative_to(root_path) or not path.is_file():
                raise ManifestError(f"Frozen source file is missing: {record.path}")
            if int(path.stat().st_size) != record.size or sha256_file(path) != record.sha256:
                raise ManifestError(f"Frozen source file changed: {record.path}")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SourceManifest":
        if payload.get("schema_version") != SOURCE_SCHEMA_VERSION:
            raise ManifestError("Unsupported source-manifest schema.")
        manifest = cls(files=tuple(SourceFile(**dict(item)) for item in payload.get("files", [])))
        if not manifest.files:
            raise ManifestError("The source manifest is empty.")
        if payload.get("source_sha256") != manifest.source_sha256:
            raise ManifestError("Source-manifest hash mismatch.")
        return manifest


def _normalize_seed_mapping(value: Mapping[str, Any] | None) -> dict[str, int]:
    return {str(key): int(seed) for key, seed in dict(value or {}).items()}


def validate_seed_isolation(
    training: Mapping[str, Any] | None,
    audit: Mapping[str, Any] | None,
) -> None:
    train = _normalize_seed_mapping(training)
    holdout = _normalize_seed_mapping(audit)
    if not train and not holdout:
        return
    missing = [field for field in DYNAMIC_SEED_FIELDS if field not in train or field not in holdout]
    if missing:
        raise ManifestError(f"Training/audit seed maps omit dynamic fields: {missing}")
    train_values = {train[field] for field in DYNAMIC_SEED_FIELDS}
    audit_values = {holdout[field] for field in DYNAMIC_SEED_FIELDS}
    overlap = sorted(train_values.intersection(audit_values))
    if overlap:
        raise ManifestError(f"Training and audit dynamic seed streams overlap: {overlap}")
    if "type_seed" in train and "type_seed" in holdout and train["type_seed"] != holdout["type_seed"]:
        raise ManifestError("type_seed must remain fixed for a fresh audit of the same population.")


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    phase: str
    family: str
    node_id: str
    slot: int
    argv: tuple[str, ...]
    config_sha256: str
    matrix_sha256: str
    estimated_vcpu_hours: float
    item_count: int = 0
    chunk_kind: str | None = None
    chunk_size: int | None = None
    runner_managed_chunks: bool = False
    expected_artifacts: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    training_seeds: Mapping[str, int] = field(default_factory=dict)
    audit_seeds: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.job_id or any(value in self.job_id for value in ("/", "\\", "..")):
            raise ManifestError(f"Unsafe job_id: {self.job_id!r}")
        if not self.phase or not self.family or not self.node_id:
            raise ManifestError(f"Job {self.job_id} is missing phase, family, or node_id.")
        if self.slot < 0 or self.estimated_vcpu_hours < 0:
            raise ManifestError(f"Job {self.job_id} has invalid slot or estimated cost.")
        if not self.argv or any(not isinstance(value, str) or not value for value in self.argv):
            raise ManifestError(f"Job {self.job_id} must use a nonempty argv list.")
        for name, digest in (("config", self.config_sha256), ("matrix", self.matrix_sha256)):
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ManifestError(f"Job {self.job_id} has an invalid {name} SHA256.")
        if self.chunk_kind is not None:
            expected = 128 if self.chunk_kind == "training" else 8 if self.chunk_kind == "audit" else None
            if expected is None:
                raise ManifestError(f"Job {self.job_id} has unsupported chunk kind {self.chunk_kind!r}.")
            if self.chunk_size != expected:
                raise ManifestError(
                    f"Job {self.job_id} must use {expected}-profile {self.chunk_kind} chunks."
                )
            if self.item_count <= 0 and not self.runner_managed_chunks:
                raise ManifestError(f"Chunked job {self.job_id} must have a positive item_count.")
            if self.item_count != 0 and self.runner_managed_chunks:
                raise ManifestError(
                    f"Runner-managed job {self.job_id} must derive its item count from chunk_plan.json."
                )
        elif self.chunk_size is not None:
            raise ManifestError(f"Job {self.job_id} specifies chunk_size without chunk_kind.")
        elif self.runner_managed_chunks:
            raise ManifestError(f"Job {self.job_id} manages chunks without a chunk kind.")
        for artifact in self.expected_artifacts:
            path = Path(artifact)
            if path.is_absolute() or ".." in path.parts:
                raise ManifestError(f"Job {self.job_id} has an unsafe expected artifact: {artifact}")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ManifestError(f"Job {self.job_id} repeats a dependency.")
        for dependency in self.depends_on:
            if (
                not dependency
                or dependency == self.job_id
                or any(value in dependency for value in ("/", "\\", ".."))
            ):
                raise ManifestError(
                    f"Job {self.job_id} has an unsafe dependency {dependency!r}."
                )
        validate_seed_isolation(self.training_seeds, self.audit_seeds)

    def formatted_argv(
        self,
        state_dir: Path,
        output_dir: Path,
        *,
        chunk_start: int = 0,
        chunk_stop: int | None = None,
    ) -> tuple[str, ...]:
        effective_stop = self.item_count if chunk_stop is None else int(chunk_stop)
        replacements = {
            "job_id": self.job_id,
            "state_dir": str(state_dir),
            "output_dir": str(output_dir),
            "matrix_hash": self.matrix_sha256,
            "chunk_start": str(int(chunk_start)),
            "chunk_stop": str(effective_stop),
        }
        formatted: list[str] = []
        for value in self.argv:
            rendered = value
            for name, replacement in replacements.items():
                rendered = rendered.replace("{" + name + "}", replacement)
            formatted.append(rendered)
        return tuple(formatted)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "JobSpec":
        normalized = dict(payload)
        for key in ("argv", "expected_artifacts", "depends_on"):
            normalized[key] = tuple(normalized.get(key, ()))
        return cls(**normalized)


@dataclass(frozen=True)
class CampaignManifest:
    campaign_id: str
    source_sha256: str
    dependency_lock_sha256: str
    matrix_sha256: str
    image_by_platform: Mapping[str, str]
    jobs: tuple[JobSpec, ...]
    spot_vm_hour_soft_limit: float = 300.0
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    schema_version: str = SCHEMA_VERSION

    @property
    def campaign_sha256(self) -> str:
        return sha256_bytes(canonical_json(self.to_payload(include_hash=False)))

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "created_utc": self.created_utc,
            "source_sha256": self.source_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "matrix_sha256": self.matrix_sha256,
            "image_by_platform": dict(sorted(self.image_by_platform.items())),
            "spot_vm_hour_soft_limit": float(self.spot_vm_hour_soft_limit),
            "jobs": [asdict(job) for job in self.jobs],
        }
        if include_hash:
            payload["campaign_sha256"] = self.campaign_sha256
        return payload

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not self.campaign_id:
            raise ManifestError("Unsupported or unnamed campaign manifest.")
        for name, digest in (
            ("source", self.source_sha256),
            ("dependency lock", self.dependency_lock_sha256),
            ("matrix", self.matrix_sha256),
        ):
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ManifestError(f"Invalid {name} SHA256.")
        if not self.image_by_platform or self.spot_vm_hour_soft_limit <= 0:
            raise ManifestError("Campaign image map and positive Spot-hour limit are required.")
        seen: set[str] = set()
        assignments: set[tuple[str, int, str]] = set()
        for job in self.jobs:
            job.validate()
            if job.job_id in seen:
                raise ManifestError(f"Duplicate job_id: {job.job_id}")
            seen.add(job.job_id)
            assignment = (job.node_id, int(job.slot), job.job_id)
            if assignment in assignments:
                raise ManifestError(f"Duplicate job assignment: {assignment}")
            assignments.add(assignment)
            if job.matrix_sha256 != self.matrix_sha256:
                raise ManifestError(f"Job {job.job_id} has a different matrix hash.")
        jobs_by_id = {job.job_id: job for job in self.jobs}
        for job in self.jobs:
            missing = sorted(set(job.depends_on).difference(jobs_by_id))
            if missing:
                raise ManifestError(f"Job {job.job_id} has missing dependencies: {missing}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(job_id: str) -> None:
            if job_id in visited:
                return
            if job_id in visiting:
                raise ManifestError(f"Campaign dependency cycle includes {job_id}.")
            visiting.add(job_id)
            for dependency in jobs_by_id[job_id].depends_on:
                visit(dependency)
            visiting.remove(job_id)
            visited.add(job_id)

        for job_id in jobs_by_id:
            visit(job_id)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CampaignManifest":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ManifestError("Unsupported campaign-manifest schema.")
        normalized = dict(payload)
        observed_hash = normalized.pop("campaign_sha256", None)
        normalized["jobs"] = tuple(JobSpec.from_payload(item) for item in normalized.get("jobs", []))
        manifest = cls(**normalized)
        manifest.validate()
        if observed_hash != manifest.campaign_sha256:
            raise ManifestError("Campaign-manifest hash mismatch.")
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> "CampaignManifest":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))

    def write(self, path: str | Path) -> Path:
        self.validate()
        return atomic_json(path, self.to_payload())

    def jobs_for(self, node_id: str, slot: int | None = None) -> tuple[JobSpec, ...]:
        return tuple(
            job
            for job in self.jobs
            if job.node_id == node_id and (slot is None or job.slot == int(slot))
        )
