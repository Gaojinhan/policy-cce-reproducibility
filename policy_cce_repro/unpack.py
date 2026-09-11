"""Extract only the checksum-pinned release data into a new local directory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile

from .offline import DataError, Dataset, file_sha256, require, safe_relative


def release_descriptor():
    for path in (Path(__file__).resolve().parent.parent/'metadata/release-data-v1.json',
                 Path(sys.prefix)/'share/policy-cce-reproducibility/metadata/release-data-v1.json'):
        if path.is_file():
            descriptor=json.loads(path.read_text())
            require(descriptor.get('schema')=='policy_cce_github_data_release_v1', 'Unknown release data descriptor')
            return descriptor
    raise DataError('Release data descriptor is missing')


def unpack_data(archive, output_dir, *, descriptor=None):
    """No download, pickle, shell execution, overwrite, or tar links are allowed."""
    expected=release_descriptor() if descriptor is None else descriptor
    archive=Path(archive).expanduser()
    out=Path(output_dir).expanduser()
    require(archive.is_file() and not archive.is_symlink(), 'Archive must be a regular local file')
    require(not out.exists() and not out.is_symlink(), 'Unpack output must not exist; use a new directory')
    out=out.resolve()
    require(not archive.resolve().is_relative_to(out), 'Unpack output must not contain its archive')
    require(archive.stat().st_size==expected['archive_bytes'], 'Release archive size mismatch')
    require(file_sha256(archive)==expected['archive_sha256'], 'Release archive SHA-256 mismatch')
    root_name=expected['top_level_directory']
    require(root_name=='offline-inputs-v1', 'Unexpected data root')
    try:
        with tarfile.open(archive,mode='r:gz') as bundle:
            members=bundle.getmembers()
            require(len(members)==expected['member_count'], 'Archive member count mismatch')
            names=set()
            for member in members:
                relative=safe_relative(member.name)
                require(member.isfile() and not member.issym() and not member.islnk(), 'Only regular archive files are allowed')
                require(relative.parts[0]==root_name and len(relative.parts)>1, 'Archive path outside declared root')
                require(member.name not in names and member.size>=0, 'Duplicate or invalid archive member')
                names.add(member.name)
            require(sum(m.size for m in members)==expected['uncompressed_bytes'], 'Archive expanded size mismatch')
            manifest_name=root_name+'/manifest.json'
            require(manifest_name in names, 'Data manifest missing from archive')
            with bundle.extractfile(manifest_name) as stream:
                raw=stream.read()
            require(hashlib.sha256(raw).hexdigest()==expected['manifest_sha256'], 'Archived manifest SHA-256 mismatch')
            manifest=json.loads(raw)
            entries={entry['path']:entry for entry in manifest['files']}
            require(len(entries)==len(manifest['files']), 'Duplicate declared data file')
            require(names=={root_name+'/'+p for p in entries}|{manifest_name}, 'Archive differs from manifest whitelist')
            for member in members:
                if member.name!=manifest_name:
                    entry=entries[member.name[len(root_name)+1:]]
                    require(member.size==entry['bytes'], 'Archived payload size differs from manifest')
            out.mkdir(parents=True,exist_ok=False)
            for member in members:
                target=out/member.name
                target.parent.mkdir(parents=True,exist_ok=True)
                digest=hashlib.sha256()
                with bundle.extractfile(member) as source, target.open('xb') as dest:
                    for block in iter(lambda:source.read(1024*1024),b''):
                        dest.write(block)
                        digest.update(block)
                sha=expected['manifest_sha256'] if member.name==manifest_name else entries[member.name[len(root_name)+1:]]['sha256']
                require(digest.hexdigest()==sha, 'Extracted payload hash mismatch')
    except (tarfile.TarError,KeyError,TypeError,ValueError,OSError) as error:
        if isinstance(error,DataError):
            raise
        raise DataError('Invalid release archive; any partial output is preserved: '+str(error)) from error
    result=Dataset(out/root_name).verify_all()
    return result|{'data_directory':str(out/root_name),'archive_sha256':expected['archive_sha256']}
