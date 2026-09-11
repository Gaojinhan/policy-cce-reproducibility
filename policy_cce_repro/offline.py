"""Fail-closed, local-only reader for immutable publication inputs.

No cloud client, credentials, network fallback or experiment runner is used.
Every read checks the supplied file's bytes against the declared manifest.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath


class DataError(ValueError):
    """Missing, corrupt, unsafe or inconsistent archived input."""


def require(condition, message):
    if not condition:
        raise DataError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value):
    require(isinstance(value, str) and bool(value), 'Input path must be a nonempty relative path')
    require('\\' not in value and ':' not in value, 'URLs and drive paths are not local input keys')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and bool(path.parts), 'Unsafe relative input path')
    require(path.as_posix() == value and value != '.', 'Noncanonical input path')
    return path


class Dataset:
    def __init__(self, root):
        given = Path(root).expanduser()
        require(not given.is_symlink(), 'Input root must not be a symlink')
        self.root = given.resolve()
        require(self.root.is_dir(), 'Input data directory does not exist')
        manifest_path = self.root / 'manifest.json'
        require(not manifest_path.is_symlink(), 'Manifest must not be a symlink')
        try:
            raw = manifest_path.read_bytes()
            self.manifest = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise DataError('Missing or invalid data manifest') from exc
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        require(isinstance(self.manifest, dict), 'Data manifest must be a JSON object')
        require(self.manifest.get('schema') == 'policy_cce_offline_data_v1', 'Unsupported data manifest schema')
        entries = self.manifest.get('files', [])
        require(isinstance(entries, list), 'Manifest files must be a list')
        self.files = {}
        for entry in entries:
            require(isinstance(entry, dict), 'Manifest file entry must be an object')
            relative = entry.get('path')
            safe_relative(relative)
            require(relative not in self.files, 'Duplicate input file in manifest: ' + relative)
            require(type(entry.get('bytes')) is int and entry['bytes'] >= 0, 'Invalid input byte count')
            h = entry.get('sha256', '')
            require(isinstance(h, str) and len(h) == 64 and all(c in '0123456789abcdef' for c in h), 'Invalid input SHA-256')
            self.files[relative] = entry
        self.jobs = self.manifest.get('jobs', {})
        self.results = self.manifest.get('results', {})
        require(isinstance(self.jobs, dict) and self.jobs and isinstance(self.results, dict)
                and set(self.jobs) == set(self.results), 'Job/result scope mismatch')
        for jid, spec in self.jobs.items():
            require(isinstance(spec, dict) and isinstance(self.results[jid], dict), 'Invalid job/result declaration')
            require(jid == spec.get('job_id'), 'Original job ID mismatch')
            dependencies = spec.get('depends_on', [])
            require(isinstance(dependencies, list) and all(isinstance(d, str) for d in dependencies), 'Invalid selected job dependencies')
            require(set(dependencies) <= set(self.jobs), 'Missing selected job dependency')
            for field in ('path', 'provenance_path', 'marker_path'):
                value = self.results[jid].get(field)
                require(isinstance(value, str) and value in self.files, 'Unlisted job input: ' + jid + '/' + field)
        for field in ('objects', 'references'):
            require(isinstance(self.manifest.get(field, {}), dict), 'Invalid manifest mapping: ' + field)
        for key, relative in self.manifest.get('objects', {}).items():
            safe_relative(key)
            require(key.startswith('jobs/'), 'Canonical data key must retain jobs/<id>')
            require(key.split('/')[1] in self.jobs, 'Canonical object outside selected jobs')
            require(isinstance(relative, str) and relative in self.files, 'Unlisted canonical input: ' + key)
        for relative in self.manifest.get('references', {}).values():
            require(isinstance(relative, str) and relative in self.files, 'Unlisted reference input')

    def _path(self, relative):
        path = safe_relative(relative)
        require(relative in self.files, 'Input is not in the archive manifest: ' + relative)
        target = self.root
        for part in path.parts:
            target = target / part
            require(not target.is_symlink(), 'Symlink input is forbidden: ' + relative)
        require(target.resolve().is_relative_to(self.root), 'Input path escapes data root')
        require(target.is_file(), 'Missing archived input (no network fallback): ' + relative)
        return target

    def read_bytes(self, relative):
        path = self._path(relative)
        entry = self.files[relative]
        data = path.read_bytes()
        require(len(data) == entry['bytes'], 'Input size mismatch: ' + relative)
        require(hashlib.sha256(data).hexdigest() == entry['sha256'], 'Input SHA-256 mismatch: ' + relative)
        return data

    def read_json(self, relative):
        try:
            return json.loads(self.read_bytes(relative))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DataError('Invalid archived JSON: ' + relative) from exc

    def read_arrays(self, relative):
        import numpy as np
        try:
            archive = np.load(io.BytesIO(self.read_bytes(relative)), allow_pickle=False)
            require(isinstance(archive, np.lib.npyio.NpzFile), 'Input must be a numeric NPZ archive')
            with archive:
                require(len(archive.files) == len(set(archive.files)), 'Duplicate NPZ array name')
                arrays = {name: archive[name] for name in archive.files}
                for name, array in arrays.items():
                    require(array.dtype.kind in 'biufc' and not array.dtype.hasobject, 'Non-numeric NPZ array: ' + name)
                    require(bool(np.all(np.isfinite(array))), 'Non-finite NPZ array: ' + name)
                return arrays
        except DataError:
            raise
        except (ValueError, OSError, KeyError, EOFError, TypeError, zipfile.BadZipFile) as exc:
            raise DataError('Invalid numeric NPZ archive: ' + relative) from exc

    def _object_path(self, key):
        safe_relative(key)
        path = self.manifest.get('objects', {}).get(key)
        require(path is not None, 'Canonical object not supplied (no network fallback): ' + key)
        return path

    def object_bytes(self, key):
        return self.read_bytes(self._object_path(key))

    def object_json(self, key):
        return self.read_json(self._object_path(key))

    def object_arrays(self, key):
        return self.read_arrays(self._object_path(key))

    def reference(self, name):
        safe_relative(name)
        path = self.manifest.get('references', {}).get(name)
        require(path is not None, 'Reference not in data manifest: ' + name)
        return self.read_json(path)

    def result(self, job_id):
        try:
            return self._checked_result(job_id)
        except DataError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError, UnicodeDecodeError) as exc:
            raise DataError('Malformed archived result/provenance/marker: ' + str(job_id)) from exc

    def _checked_result(self, job_id):
        require(job_id in self.jobs, 'Job is outside declared paper scope: ' + job_id)
        spec, paths = self.jobs[job_id], self.results[job_id]
        raw = self.read_bytes(paths['path'])
        result = json.loads(raw)
        provenance = self.read_json(paths['provenance_path'])
        marker = self.read_json(paths['marker_path'])
        require(result['job_id'] == provenance['job_id'] == marker['job_id'] == job_id, 'Result job identity mismatch')
        require(result['status'] == marker['status'] == 'complete' and result['smoke'] is False, 'Not a complete formal result')
        require(result['family'] == spec['family'], 'Result family mismatch')
        require(hashlib.sha256(raw).hexdigest() == provenance['result_sha256'] and len(raw) == provenance['result_size'],
                'Result disagrees with original provenance')
        marker_raw = self.read_bytes(paths['marker_path'])
        require(hashlib.sha256(marker_raw).hexdigest() == provenance['marker_sha256'], 'Completion marker provenance mismatch')
        for field in ('campaign_sha256', 'matrix_sha256'):
            require(result[field] == marker[field] == self.manifest[field], 'Campaign/matrix mismatch')
        require(marker['source_sha256'] == self.manifest['source_sha256'], 'Source identity mismatch')
        for field in ('node_id', 'slot', 'family', 'phase', 'config_sha256'):
            require(marker[field] == spec[field], 'Frozen logical/config identity mismatch: ' + field)
        declaration = [a for a in marker['artifacts'] if a['path'] == 'result.json']
        require(len(declaration) == 1 and declaration[0]['sha256'] == provenance['result_sha256'] and
                declaration[0]['size'] == len(raw), 'Completion marker/result mismatch')
        return result

    def validate_output_dir(self, output):
        out = Path(output).expanduser()
        require(not out.is_symlink(), 'Output path must not be a symlink')
        out = out.resolve()
        require(not out.is_relative_to(self.root) and not self.root.is_relative_to(out), 'Output and input directories must not overlap')
        return out

    def verify_all(self):
        actual = set()
        for path in self.root.rglob('*'):
            require(not path.is_symlink(), 'Symlink in archived data')
            if path.is_file():
                actual.add(path.relative_to(self.root).as_posix())
        require(actual == set(self.files) | {'manifest.json'}, 'Missing or undeclared data files')
        for relative in self.files:
            self.read_bytes(relative)
        for jid in self.jobs:
            self.result(jid)
        return {'status': 'pass', 'files': len(self.files), 'formal_jobs': len(self.jobs),
                'manifest_sha256': self.manifest_sha256, 'bytes': sum(e['bytes'] for e in self.files.values())}
