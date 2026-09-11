"""Exercise archive safety using tiny, honestly synthetic scientific inputs."""
import hashlib
import io
import tarfile

import pytest

from policy_cce_repro.offline import DataError
from policy_cce_repro.unpack import unpack_data
from test_offline_safety import synthetic_input, deny_external_access


def bundle(fixture,tmp_path,mutate=None):
    rows=[]
    for path in sorted(fixture.root.rglob('*')):
        if path.is_file():
            raw=path.read_bytes()
            info=tarfile.TarInfo('offline-inputs-v1/'+path.relative_to(fixture.root).as_posix())
            info.size=len(raw)
            rows.append([info,raw])
    if mutate:mutate(rows)
    target=tmp_path/'bundle.tar.gz'
    with tarfile.open(target,'w:gz') as archive:
        for info,raw in rows:
            archive.addfile(info,io.BytesIO(raw))
    descriptor={'archive_bytes':target.stat().st_size,
                'archive_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'top_level_directory':'offline-inputs-v1','member_count':len(rows),
                'uncompressed_bytes':sum(i.size for i,_ in rows),
                'manifest_sha256':hashlib.sha256((fixture.root/'manifest.json').read_bytes()).hexdigest()}
    return target,descriptor


def test_complete_archive_extracts_and_rechecks_original_records(synthetic_input,tmp_path):
    archive,descriptor=bundle(synthetic_input,tmp_path)
    result=unpack_data(archive,tmp_path/'new',descriptor=descriptor)
    assert result['status']=='pass' and result['formal_jobs']==1
    for original in synthetic_input.root.rglob('*'):
        if original.is_file():
            assert original.read_bytes()==(tmp_path/'new/offline-inputs-v1'/original.relative_to(synthetic_input.root)).read_bytes()


@pytest.mark.parametrize('field,value',[('archive_sha256','0'*64),('archive_bytes',1),('manifest_sha256','0'*64)])
def test_wrong_digest_or_size_fails_before_output(synthetic_input,tmp_path,field,value):
    archive,descriptor=bundle(synthetic_input,tmp_path)
    descriptor[field]=value
    with pytest.raises(DataError):unpack_data(archive,tmp_path/'new',descriptor=descriptor)
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('name',['../escape','/tmp/escape','offline-inputs-v1/../escape','other/data.json','offline-inputs-v1/a\\b'])
def test_unsafe_archive_path_cannot_escape(synthetic_input,tmp_path,name):
    archive,descriptor=bundle(synthetic_input,tmp_path,lambda rows:setattr(rows[0][0],'name',name))
    with pytest.raises(DataError):unpack_data(archive,tmp_path/'new',descriptor=descriptor)
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('kind',[tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.DIRTYPE,tarfile.FIFOTYPE])
def test_nonregular_members_are_rejected(synthetic_input,tmp_path,kind):
    def mutate(rows):
        rows[0][0].type=kind
        rows[0][0].size=0
        rows[0][0].linkname='outside'
        rows[0][1]=b''
    archive,descriptor=bundle(synthetic_input,tmp_path,mutate)
    with pytest.raises(DataError):unpack_data(archive,tmp_path/'new',descriptor=descriptor)
    assert not (tmp_path/'new').exists()


def test_duplicate_member_is_rejected(synthetic_input,tmp_path):
    archive,descriptor=bundle(synthetic_input,tmp_path,lambda rows:rows.append(rows[0]))
    with pytest.raises(DataError):unpack_data(archive,tmp_path/'new',descriptor=descriptor)


def test_missing_declared_file_is_rejected(synthetic_input,tmp_path):
    archive,descriptor=bundle(synthetic_input,tmp_path,lambda rows:rows.pop(0))
    with pytest.raises(DataError):unpack_data(archive,tmp_path/'new',descriptor=descriptor)


def test_existing_output_is_preserved(synthetic_input,tmp_path):
    archive,descriptor=bundle(synthetic_input,tmp_path)
    out=tmp_path/'old'
    out.mkdir()
    (out/'keep').write_text('unchanged')
    with pytest.raises(DataError):unpack_data(archive,out,descriptor=descriptor)
    assert (out/'keep').read_text()=='unchanged'


def test_corrupt_payload_preserves_failure_and_original_inputs(synthetic_input,tmp_path):
    original={p:p.read_bytes() for p in synthetic_input.root.rglob('*') if p.is_file()}
    def mutate(rows):
        rows[0][1]=b'x'*len(rows[0][1])
    archive,descriptor=bundle(synthetic_input,tmp_path,mutate)
    with pytest.raises(DataError):unpack_data(archive,tmp_path/'new',descriptor=descriptor)
    assert (tmp_path/'new').exists()
    assert all(p.read_bytes()==raw for p,raw in original.items())
