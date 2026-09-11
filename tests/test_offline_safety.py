"""Synthetic tests: no real campaign inputs, accounts, or cloud access."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import socket
import subprocess
import urllib.request

import numpy as np
import pytest

from policy_cce_repro.offline import DataError, Dataset


CAMPAIGN = "a" * 64
MATRIX = "b" * 64
SOURCE = "c" * 64
JOB = "synthetic__offline_reader__job0"
RESULT = "data/result.json"
PROVENANCE = "data/read_provenance.json"
MARKER = "data/job_complete.json"
ARRAYS = "data/samples.npz"
REFERENCE = "references/example.json"
OBJECT_KEY = f"jobs/{JOB}/artifacts/samples.npz"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


@dataclass
class SyntheticInput:
    root: Path
    manifest: dict
    result: dict
    marker: dict

    def write(self, relative: str, payload: bytes) -> None:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        self.manifest["files"] = [
            item for item in self.manifest["files"] if item["path"] != relative
        ]
        self.manifest["files"].append(
            {"path": relative, "bytes": len(payload), "sha256": _sha(payload)}
        )

    def save(self) -> None:
        (self.root / "manifest.json").write_bytes(_json(self.manifest))

    def save_linked_records(self) -> None:
        """Rehash synthetic records so semantic negatives reach identity checks."""
        result_bytes = _json(self.result)
        self.marker["artifacts"] = [
            {"path": "result.json", "size": len(result_bytes), "sha256": _sha(result_bytes)}
        ]
        marker_bytes = _json(self.marker)
        provenance = {
            "job_id": JOB,
            "result_sha256": _sha(result_bytes),
            "result_size": len(result_bytes),
            "marker_sha256": _sha(marker_bytes),
            "marker_generation": "1",
            "result_generation": "1",
            "executor_provenance": None,
        }
        self.write(RESULT, result_bytes)
        self.write(MARKER, marker_bytes)
        self.write(PROVENANCE, _json(provenance))
        self.save()


@pytest.fixture(autouse=True)
def deny_external_access(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Offline safety tests must not use network or subprocesses")

    monkeypatch.setattr(urllib.request, "urlopen", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)


@pytest.fixture
def synthetic_input(tmp_path) -> SyntheticInput:
    root = tmp_path / "inputs"
    root.mkdir()
    spec = {
        "job_id": JOB,
        "family": "solver_benchmark",
        "phase": "training",
        "node_id": "logical-test-node",
        "slot": 0,
        "config_sha256": "d" * 64,
        "depends_on": [],
    }
    result = {
        "job_id": JOB,
        "family": spec["family"],
        "status": "complete",
        "smoke": False,
        "campaign_sha256": CAMPAIGN,
        "matrix_sha256": MATRIX,
        "source_sha256": SOURCE,
    }
    marker = {
        **spec,
        "status": "complete",
        "campaign_sha256": CAMPAIGN,
        "matrix_sha256": MATRIX,
        "source_sha256": SOURCE,
        "artifacts": [],
    }
    manifest = {
        "schema": "policy_cce_offline_data_v1",
        "campaign_sha256": CAMPAIGN,
        "matrix_sha256": MATRIX,
        "source_sha256": SOURCE,
        "files": [],
        "jobs": {JOB: spec},
        "results": {
            JOB: {"path": RESULT, "provenance_path": PROVENANCE, "marker_path": MARKER}
        },
        "objects": {OBJECT_KEY: ARRAYS},
        "references": {"example": REFERENCE},
    }
    fixture = SyntheticInput(root, manifest, result, marker)
    buffer = io.BytesIO()
    np.savez(buffer, returns=np.arange(12, dtype=float).reshape(4, 3))
    fixture.write(ARRAYS, buffer.getvalue())
    fixture.write(REFERENCE, _json({"mean": 5.5}))
    fixture.save_linked_records()
    return fixture


def test_declared_local_inputs_work_without_account_or_network(synthetic_input):
    dataset = Dataset(synthetic_input.root)
    assert dataset.jobs[JOB]["node_id"] == "logical-test-node"
    assert dataset.result(JOB)["job_id"] == JOB
    assert dataset.reference("example") == {"mean": 5.5}
    assert dataset.object_bytes(OBJECT_KEY) == (synthetic_input.root / ARRAYS).read_bytes()
    np.testing.assert_array_equal(
        dataset.object_arrays(OBJECT_KEY)["returns"], np.arange(12).reshape(4, 3)
    )
    dataset.verify_all()


@pytest.mark.parametrize("relative", [
    "gs://private-bucket/object", "https://example.invalid/data.json",
    "file:///tmp/input.json", "/tmp/input.json", "../outside.json",
    "data/../../outside.json", "data/../result.json", r"C:\private\input.json",
])
def test_unsafe_read_paths_are_rejected(synthetic_input, relative):
    with pytest.raises(DataError):
        Dataset(synthetic_input.root).read_bytes(relative)


def test_present_but_unlisted_file_is_not_an_input(synthetic_input):
    (synthetic_input.root / "unlisted.json").write_bytes(b"{}")
    with pytest.raises(DataError):
        Dataset(synthetic_input.root).read_json("unlisted.json")


def test_missing_listed_file_fails_without_fallback(synthetic_input):
    (synthetic_input.root / REFERENCE).unlink()
    with pytest.raises(DataError):
        Dataset(synthetic_input.root).reference("example")


@pytest.mark.parametrize("same_size", [True, False])
def test_corrupted_bytes_are_rejected(synthetic_input, same_size):
    target = synthetic_input.root / REFERENCE
    before = target.read_bytes()
    target.write_bytes(b"x" * len(before) if same_size else before + b"extra")
    with pytest.raises(DataError):
        Dataset(synthetic_input.root).reference("example")


def test_verify_all_rechecks_bytes_after_an_initial_read(synthetic_input):
    dataset = Dataset(synthetic_input.root)
    dataset.reference("example")
    (synthetic_input.root / REFERENCE).write_bytes(b"changed")
    with pytest.raises(DataError):
        dataset.verify_all()


@pytest.mark.parametrize("inside_root", [True, False])
def test_symlinked_file_is_rejected_even_if_its_bytes_match(synthetic_input, inside_root):
    file = synthetic_input.root / REFERENCE
    destination = (synthetic_input.root if inside_root else synthetic_input.root.parent) / "other.json"
    destination.write_bytes(file.read_bytes())
    file.unlink()
    file.symlink_to(destination)
    with pytest.raises(DataError):
        Dataset(synthetic_input.root).reference("example")


def test_symlinked_directory_cannot_bypass_read_root(synthetic_input):
    directory = synthetic_input.root / "references"
    outside = synthetic_input.root.parent / "moved-references"
    directory.rename(outside)
    directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(DataError):
        Dataset(synthetic_input.root).reference("example")


def test_wrong_schema_is_rejected(synthetic_input):
    synthetic_input.manifest["schema"] = "unrelated_schema"
    synthetic_input.save()
    with pytest.raises(DataError):
        Dataset(synthetic_input.root)


def test_non_object_manifest_is_a_data_error(synthetic_input):
    (synthetic_input.root / "manifest.json").write_bytes(b"[]")
    with pytest.raises(DataError):
        Dataset(synthetic_input.root)


def test_missing_dependency_does_not_silently_shrink_the_scope(synthetic_input):
    synthetic_input.manifest["jobs"][JOB]["depends_on"] = ["absent-original-job"]
    synthetic_input.save()
    with pytest.raises(DataError):
        Dataset(synthetic_input.root)


def test_result_scope_must_equal_declared_job_scope(synthetic_input):
    synthetic_input.manifest["results"] = {}
    synthetic_input.save()
    with pytest.raises(DataError):
        Dataset(synthetic_input.root)


def test_duplicate_manifest_path_is_rejected(synthetic_input):
    synthetic_input.manifest["files"].append(copy.deepcopy(synthetic_input.manifest["files"][0]))
    synthetic_input.save()
    with pytest.raises(DataError):
        Dataset(synthetic_input.root)


@pytest.mark.parametrize("mapping", ["objects", "references"])
def test_object_and_reference_maps_cannot_read_unlisted_paths(synthetic_input, mapping):
    key = OBJECT_KEY if mapping == "objects" else "example"
    synthetic_input.manifest[mapping][key] = "unlisted.json"
    (synthetic_input.root / "unlisted.json").write_bytes(b"{}")
    synthetic_input.save()
    with pytest.raises(DataError):
        dataset = Dataset(synthetic_input.root)
        dataset.object_json(key) if mapping == "objects" else dataset.reference(key)


@pytest.mark.parametrize("method,key", [
    ("result", "unknown-job"), ("object_json", "jobs/unknown/item.json"),
    ("reference", "unknown-reference"),
])
def test_unknown_lookup_keys_fail_closed(synthetic_input, method, key):
    with pytest.raises(DataError):
        getattr(Dataset(synthetic_input.root), method)(key)


@pytest.mark.parametrize("field,value", [
    ("job_id", "different-job"), ("family", "different-family"),
    ("status", "failed"), ("smoke", True),
    ("campaign_sha256", "e" * 64), ("matrix_sha256", "f" * 64),
])
def test_result_identity_and_status_are_validated_after_hashes(synthetic_input, field, value):
    synthetic_input.result[field] = value
    synthetic_input.save_linked_records()
    with pytest.raises(DataError):
        Dataset(synthetic_input.root).result(JOB)


@pytest.mark.parametrize("field,value", [
    ("node_id", "different-logical-node"), ("slot", 3),
    ("config_sha256", "e" * 64), ("source_sha256", "f" * 64),
    ("status", "failed"),
])
def test_marker_preserves_frozen_logical_and_source_identity(synthetic_input, field, value):
    synthetic_input.marker[field] = value
    synthetic_input.save_linked_records()
    with pytest.raises(DataError):
        Dataset(synthetic_input.root).result(JOB)


def _replace_linked_record(fixture: SyntheticInput, relative: str, payload: bytes) -> None:
    """Preserve synthetic byte/hash links while inserting a malformed record."""
    result_bytes = payload if relative == RESULT else (fixture.root / RESULT).read_bytes()
    marker = json.loads((fixture.root / MARKER).read_bytes())
    marker["artifacts"] = [
        {"path": "result.json", "sha256": _sha(result_bytes), "size": len(result_bytes)}
    ]
    marker_bytes = payload if relative == MARKER else _json(marker)
    provenance = json.loads((fixture.root / PROVENANCE).read_bytes())
    provenance.update(
        result_sha256=_sha(result_bytes), result_size=len(result_bytes),
        marker_sha256=_sha(marker_bytes),
    )
    fixture.write(RESULT, result_bytes)
    fixture.write(MARKER, marker_bytes)
    fixture.write(PROVENANCE, payload if relative == PROVENANCE else _json(provenance))
    fixture.save()


@pytest.mark.parametrize("relative", [RESULT, PROVENANCE, MARKER])
@pytest.mark.parametrize("payload", [b"{broken-json", b"[]"], ids=["invalid-json", "non-object"])
def test_rehashed_malformed_linked_records_raise_data_error(synthetic_input, relative, payload):
    _replace_linked_record(synthetic_input, relative, payload)
    dataset = Dataset(synthetic_input.root)
    # The file passes the archive checksum: rejection must come from its content.
    assert dataset.read_bytes(relative) == payload
    with pytest.raises(DataError):
        dataset.result(JOB)


@pytest.mark.parametrize("relative,missing", [
    (RESULT, "status"), (PROVENANCE, "result_sha256"), (MARKER, "slot"),
])
def test_rehashed_incomplete_linked_records_raise_data_error(synthetic_input, relative, missing):
    record = json.loads((synthetic_input.root / relative).read_bytes())
    del record[missing]
    payload = _json(record)
    _replace_linked_record(synthetic_input, relative, payload)
    dataset = Dataset(synthetic_input.root)
    assert dataset.read_bytes(relative) == payload
    with pytest.raises(DataError):
        dataset.result(JOB)


def test_rehashed_malformed_marker_artifacts_raise_data_error(synthetic_input):
    marker = json.loads((synthetic_input.root / MARKER).read_bytes())
    marker["artifacts"] = None
    payload = _json(marker)
    _replace_linked_record(synthetic_input, MARKER, payload)
    dataset = Dataset(synthetic_input.root)
    assert dataset.read_bytes(MARKER) == payload
    with pytest.raises(DataError):
        dataset.result(JOB)


def test_pickled_npz_members_are_never_loaded(synthetic_input):
    buffer = io.BytesIO()
    np.savez(buffer, unsafe=np.asarray([{"not": "numeric"}], dtype=object))
    synthetic_input.write(ARRAYS, buffer.getvalue())
    synthetic_input.save()
    with pytest.raises(DataError):
        Dataset(synthetic_input.root).object_arrays(OBJECT_KEY)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_array_values_are_rejected(synthetic_input, value):
    buffer = io.BytesIO()
    np.savez(buffer, returns=np.asarray([1.0, value]))
    synthetic_input.write(ARRAYS, buffer.getvalue())
    synthetic_input.save()
    with pytest.raises(DataError):
        Dataset(synthetic_input.root).object_arrays(OBJECT_KEY)


def test_plain_npy_file_is_not_accepted_as_an_npz_archive(synthetic_input):
    buffer = io.BytesIO()
    np.save(buffer, np.arange(3))
    synthetic_input.write(ARRAYS, buffer.getvalue())
    synthetic_input.save()
    with pytest.raises(DataError):
        Dataset(synthetic_input.root).object_arrays(OBJECT_KEY)


def test_verify_all_rejects_extra_physical_files(synthetic_input):
    (synthetic_input.root / "unlisted.json").write_bytes(b"{}")
    with pytest.raises(DataError):
        Dataset(synthetic_input.root).verify_all()


def test_output_sibling_is_allowed_without_creating_it(synthetic_input):
    output = synthetic_input.root.parent / "new-report"
    assert Dataset(synthetic_input.root).validate_output_dir(output) == output.resolve()
    assert not output.exists()


@pytest.mark.parametrize("target", ["same", "child", "ancestor"])
def test_output_must_not_overlap_input_root_in_either_direction(synthetic_input, target):
    path = {"same": synthetic_input.root, "child": synthetic_input.root / "reports",
            "ancestor": synthetic_input.root.parent}[target]
    with pytest.raises(DataError):
        Dataset(synthetic_input.root).validate_output_dir(path)


def test_output_symlink_alias_to_inputs_is_rejected(synthetic_input):
    alias = synthetic_input.root.parent / "input-alias"
    alias.symlink_to(synthetic_input.root, target_is_directory=True)
    with pytest.raises(DataError):
        Dataset(synthetic_input.root).validate_output_dir(alias / "new-report")


def test_all_105_frozen_scientific_files_remain_byte_exact():
    repository = Path(__file__).resolve().parents[1]
    copy_manifest = json.loads((repository / "metadata/source-copy-manifest.json").read_bytes())
    frozen_bytes = (repository / "metadata/frozen-source-manifest.json").read_bytes()
    frozen_manifest = json.loads(frozen_bytes)
    assert copy_manifest["schema"] == "offline_frozen_source_copy_v1"
    assert _sha(frozen_bytes) == copy_manifest["frozen_manifest_sha256"]
    assert copy_manifest["source_sha256"] == frozen_manifest["source_sha256"] == (
        "eb0143548676863326355ff2e94e81e6081735a0be952e25ab8e2a190dada900"
    )
    expected = {item["path"]: item for item in frozen_manifest["files"]
                if item["path"].startswith("cmfg_cce/")}
    entries = copy_manifest["files"]
    assert len(entries) == len({item["path"] for item in entries}) == len(expected) == 105
    assert {item["path"] for item in entries} == set(expected)
    for item in entries:
        relative = Path(item["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        target = repository / relative
        assert not target.is_symlink(), str(relative)
        payload = target.read_bytes()
        original = expected[item["path"]]
        assert item["copy_kind"] == "byte_exact"
        assert item["origin_path"] == item["path"]
        assert len(payload) == item["bytes"] == original["size"], str(relative)
        assert _sha(payload) == item["sha256"] == original["sha256"], str(relative)
