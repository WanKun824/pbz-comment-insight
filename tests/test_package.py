"""权重发布回归：必需文件、模型隔离及下载/包内完整性校验。"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import package  # noqa: E402


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(package, 'ROOT', str(tmp_path))
    monkeypatch.setattr(package, 'DIST', str(tmp_path / 'dist'))
    for name in package.WEIGHTS_FILES:
        path = tmp_path / 'artifacts/ckpt/laya-pbz-v1' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f'synthetic fixture: {name}'.encode())
    for name in package.WEIGHTS_EXTRA.values():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Synthetic fixture, no real weights or comments.', encoding='utf-8')
    return tmp_path


def test_weights_bundle_is_complete_and_excludes_other_models(checkpoint):
    other = checkpoint / 'artifacts/ckpt/other-model/model.safetensors'
    other.parent.mkdir(parents=True)
    other.write_bytes(b'excluded other model')
    snapshot = checkpoint / 'artifacts/ckpt/laya-pbz-v1/checkpoint_latest/model.safetensors'
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b'excluded snapshot')
    output = package.make('weights', package.WEIGHTS_INCLUDE, level=0)
    manifest = package.verify_weights(output)
    assert set(manifest['files']) == set(package.WEIGHTS_FILES) | set(package.WEIGHTS_EXTRA)
    with zipfile.ZipFile(output) as archive:
        assert len(archive.namelist()) == len(manifest['files']) + 1
        assert not any('other-model' in name or 'checkpoint_latest' in name for name in archive.namelist())
    checksum = Path(output + '.sha256').read_text(encoding='utf-8').split()[0]
    assert len(checksum) == 64
    assert checksum == package.sha256_file(output)


def test_incomplete_checkpoint_stops_packaging(checkpoint):
    (checkpoint / 'artifacts/ckpt/laya-pbz-v1/tokenizer/tokenizer.json').unlink()
    with pytest.raises(FileNotFoundError, match='tokenizer/tokenizer.json'):
        package.make('weights', package.WEIGHTS_INCLUDE)
    assert not (checkpoint / 'dist/laya-pbz-v1-weights.zip').exists()


def test_changed_download_is_rejected(checkpoint):
    output = package.make('weights', package.WEIGHTS_INCLUDE, level=0)
    with open(output, 'ab') as stream:
        stream.write(b'corrupted download')
    with pytest.raises(ValueError, match='压缩包 SHA-256 不匹配'):
        package.verify_weights(output)


def test_changed_tensor_payload_is_rejected_even_with_new_archive_checksum(checkpoint):
    output = package.make('weights', package.WEIGHTS_INCLUDE, level=0)
    with zipfile.ZipFile(output) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    key = 'laya-pbz-v1/model.safetensors'
    files[key] = b'X' + files[key][1:]
    with zipfile.ZipFile(output, 'w') as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    Path(output + '.sha256').write_text(
        f'{package.sha256_file(output)}  {Path(output).name}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='文件 SHA-256 不匹配'):
        package.verify_weights(output)


def test_unexpected_archive_path_is_rejected(checkpoint):
    output = package.make('weights', package.WEIGHTS_INCLUDE, level=0)
    with zipfile.ZipFile(output, 'a') as archive:
        archive.writestr('laya-pbz-v1/../unexpected.csv', b'extra file')
    Path(output + '.sha256').write_text(
        f'{package.sha256_file(output)}  {Path(output).name}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='目录结构不匹配'):
        package.verify_weights(output)
