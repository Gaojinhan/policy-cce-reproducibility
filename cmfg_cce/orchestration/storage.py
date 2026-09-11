from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid
from typing import Callable, Protocol

from cmfg_cce.orchestration.manifest import sha256_file


class RemoteConflictError(RuntimeError):
    """Raised when an immutable remote key already has different content."""


class RemoteStore(Protocol):
    def upload_immutable(self, local_path: Path, object_name: str) -> str: ...

    def upload_mutable(self, local_path: Path, object_name: str) -> str: ...

    def download(self, object_name: str, local_path: Path, *, expected_sha256: str | None = None) -> bool: ...

    def exists(self, object_name: str) -> bool: ...


def _safe_object_name(value: str) -> str:
    normalized = value.replace("\\", "/").lstrip("/")
    parts = Path(normalized).parts
    if not normalized or ".." in parts:
        raise ValueError(f"Unsafe object name: {value!r}")
    return normalized


@dataclass
class LocalMirrorStore:
    """Filesystem-backed remote-store implementation used for local/offline nodes."""

    root: Path

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _target(self, object_name: str) -> Path:
        return self.root / _safe_object_name(object_name)

    def exists(self, object_name: str) -> bool:
        return self._target(object_name).is_file()

    def upload_immutable(self, local_path: Path, object_name: str) -> str:
        source = Path(local_path)
        digest = sha256_file(source)
        target = self._target(object_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(target) != digest:
                raise RemoteConflictError(f"Remote key {object_name} already has different content.")
            return digest
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise IOError(f"Copied object failed checksum validation: {object_name}")
        try:
            os.link(temporary, target)
            temporary.unlink()
        except FileExistsError:
            temporary.unlink(missing_ok=True)
            if sha256_file(target) != digest:
                raise RemoteConflictError(f"Concurrent writer changed {object_name}.")
        return digest

    def upload_mutable(self, local_path: Path, object_name: str) -> str:
        source = Path(local_path)
        digest = sha256_file(source)
        target = self._target(object_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        return digest

    def download(self, object_name: str, local_path: Path, *, expected_sha256: str | None = None) -> bool:
        source = self._target(object_name)
        if not source.is_file():
            return False
        digest = sha256_file(source)
        if expected_sha256 is not None and digest != expected_sha256:
            raise RemoteConflictError(f"Remote checksum mismatch for {object_name}.")
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise IOError(f"Downloaded object failed checksum validation: {object_name}")
        temporary.replace(target)
        return True


class GCloudObjectStore:
    """GCS object store using the installed Google Cloud CLI and immutable generations."""

    def __init__(
        self,
        root_uri: str,
        *,
        project: str,
        retries: int = 7,
        initial_backoff_seconds: float = 1.0,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not root_uri.startswith("gs://"):
            raise ValueError("GCS root must begin with gs://")
        self.root_uri = root_uri.rstrip("/")
        self.project = str(project)
        self.retries = max(1, int(retries))
        self.initial_backoff_seconds = max(0.0, float(initial_backoff_seconds))
        self._runner = runner
        self._sleeper = sleeper

    def _uri(self, object_name: str) -> str:
        return f"{self.root_uri}/{_safe_object_name(object_name)}"

    def _run(self, args: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
        return self._runner(
            args,
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _describe_sha256(self, object_name: str) -> str | None:
        result = self._run(
            [
                "gcloud",
                "storage",
                "objects",
                "describe",
                self._uri(object_name),
                f"--project={self.project}",
                "--format=json",
            ]
        )
        if result.returncode != 0:
            return None
        payload = json.loads(result.stdout or "{}")
        metadata = payload.get("metadata") or payload.get("customMetadata") or {}
        return metadata.get("sha256")

    def exists(self, object_name: str) -> bool:
        return self._describe_sha256(object_name) is not None

    def upload_immutable(self, local_path: Path, object_name: str) -> str:
        source = Path(local_path)
        digest = sha256_file(source)
        remote = self._uri(object_name)
        for attempt in range(self.retries):
            result = self._run(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    str(source),
                    remote,
                    f"--project={self.project}",
                    "--if-generation-match=0",
                    f"--custom-metadata=sha256={digest}",
                    "--quiet",
                ]
            )
            if result.returncode == 0:
                return digest
            observed = self._describe_sha256(object_name)
            if observed is not None:
                if observed == digest:
                    return digest
                raise RemoteConflictError(
                    f"GCS key {object_name} exists with SHA256 {observed}, expected {digest}."
                )
            if attempt + 1 < self.retries:
                self._sleeper(min(30.0, self.initial_backoff_seconds * (2**attempt)))
        raise RuntimeError(f"Failed to upload {object_name} after {self.retries} attempts: {result.stderr}")

    def upload_mutable(self, local_path: Path, object_name: str) -> str:
        source = Path(local_path)
        digest = sha256_file(source)
        for attempt in range(self.retries):
            result = self._run(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    str(source),
                    self._uri(object_name),
                    f"--project={self.project}",
                    f"--custom-metadata=sha256={digest}",
                    "--quiet",
                ]
            )
            if result.returncode == 0:
                return digest
            if attempt + 1 < self.retries:
                self._sleeper(min(30.0, self.initial_backoff_seconds * (2**attempt)))
        raise RuntimeError(f"Failed to update {object_name} after {self.retries} attempts: {result.stderr}")

    def download(self, object_name: str, local_path: Path, *, expected_sha256: str | None = None) -> bool:
        observed = self._describe_sha256(object_name)
        if observed is None:
            return False
        if expected_sha256 is not None and observed != expected_sha256:
            raise RemoteConflictError(f"GCS checksum metadata mismatch for {object_name}.")
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        for attempt in range(self.retries):
            result = self._run(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    self._uri(object_name),
                    str(temporary),
                    f"--project={self.project}",
                    "--quiet",
                ]
            )
            if result.returncode == 0:
                actual = sha256_file(temporary)
                expected = expected_sha256 or observed
                if expected is not None and actual != expected:
                    temporary.unlink(missing_ok=True)
                    raise RemoteConflictError(f"Downloaded GCS object failed SHA256: {object_name}")
                temporary.replace(target)
                return True
            if attempt + 1 < self.retries:
                self._sleeper(min(30.0, self.initial_backoff_seconds * (2**attempt)))
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {object_name} after {self.retries} attempts: {result.stderr}")


class GoogleCloudStorageStore:
    """GCS store backed by Application Default Credentials and generation guards."""

    def __init__(
        self,
        root_uri: str,
        *,
        project: str,
        retries: int = 7,
        initial_backoff_seconds: float = 1.0,
        sleeper: Callable[[float], None] = time.sleep,
        client: object | None = None,
    ) -> None:
        if not root_uri.startswith("gs://"):
            raise ValueError("GCS root must begin with gs://")
        bucket_and_prefix = root_uri[5:].split("/", 1)
        self.bucket_name = bucket_and_prefix[0]
        self.prefix = bucket_and_prefix[1].strip("/") if len(bucket_and_prefix) == 2 else ""
        self.project = str(project)
        self.retries = max(1, int(retries))
        self.initial_backoff_seconds = max(0.0, float(initial_backoff_seconds))
        self._sleeper = sleeper
        if client is None:
            try:
                from google.cloud import storage  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "google-cloud-storage is required for the Python GCS backend."
                ) from exc
            client = storage.Client(project=self.project)
        self.client = client
        self.bucket = self.client.bucket(self.bucket_name)

    def _key(self, object_name: str) -> str:
        name = _safe_object_name(object_name)
        return f"{self.prefix}/{name}" if self.prefix else name

    def _blob(self, object_name: str):
        return self.bucket.blob(self._key(object_name))

    def _remote_sha(self, object_name: str) -> str | None:
        blob = self._blob(object_name)
        try:
            blob.reload()
        except Exception as exc:  # provider exception types are optional locally
            if exc.__class__.__name__ in {"NotFound", "Forbidden"}:
                if exc.__class__.__name__ == "Forbidden":
                    raise
                return None
            raise
        return dict(blob.metadata or {}).get("sha256")

    def exists(self, object_name: str) -> bool:
        return self._remote_sha(object_name) is not None

    def upload_immutable(self, local_path: Path, object_name: str) -> str:
        source = Path(local_path)
        digest = sha256_file(source)
        for attempt in range(self.retries):
            blob = self._blob(object_name)
            blob.metadata = {"sha256": digest}
            try:
                blob.upload_from_filename(str(source), if_generation_match=0)
                return digest
            except Exception as exc:
                observed = self._remote_sha(object_name)
                if observed is not None:
                    if observed == digest:
                        return digest
                    raise RemoteConflictError(
                        f"GCS key {object_name} exists with SHA256 {observed}, expected {digest}."
                    ) from exc
                if attempt + 1 >= self.retries:
                    raise RuntimeError(
                        f"Failed to upload {object_name} after {self.retries} attempts."
                    ) from exc
                self._sleeper(min(30.0, self.initial_backoff_seconds * (2**attempt)))
        raise AssertionError("unreachable")

    def upload_mutable(self, local_path: Path, object_name: str) -> str:
        source = Path(local_path)
        digest = sha256_file(source)
        for attempt in range(self.retries):
            blob = self._blob(object_name)
            blob.metadata = {"sha256": digest}
            try:
                blob.upload_from_filename(str(source))
                return digest
            except Exception as exc:
                if attempt + 1 >= self.retries:
                    raise RuntimeError(
                        f"Failed to update {object_name} after {self.retries} attempts."
                    ) from exc
                self._sleeper(min(30.0, self.initial_backoff_seconds * (2**attempt)))
        raise AssertionError("unreachable")

    def download(self, object_name: str, local_path: Path, *, expected_sha256: str | None = None) -> bool:
        observed = self._remote_sha(object_name)
        if observed is None:
            return False
        if expected_sha256 is not None and observed != expected_sha256:
            raise RemoteConflictError(f"GCS checksum metadata mismatch for {object_name}.")
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        for attempt in range(self.retries):
            try:
                self._blob(object_name).download_to_filename(str(temporary))
                actual = sha256_file(temporary)
                expected = expected_sha256 or observed
                if expected is not None and actual != expected:
                    temporary.unlink(missing_ok=True)
                    raise RemoteConflictError(f"Downloaded GCS object failed SHA256: {object_name}")
                temporary.replace(target)
                return True
            except RemoteConflictError:
                raise
            except Exception as exc:
                if attempt + 1 >= self.retries:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Failed to download {object_name} after {self.retries} attempts."
                    ) from exc
                self._sleeper(min(30.0, self.initial_backoff_seconds * (2**attempt)))
        raise AssertionError("unreachable")
