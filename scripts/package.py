"""打三个分发包：src / full / weights。

    python scripts/package.py              # 全打
    python scripts/package.py --only src
    python scripts/package.py --verify dist/laya-pbz-v1-weights.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from fnmatch import fnmatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from pbz import __version__  # noqa: E402

DIST = os.path.join(ROOT, 'dist')
NAME = 'pbz-comment-insight'

EXCLUDE = ['*/__pycache__/*', '*.pyc', '*.egg-info/*', '.git/*', 'dist/*',
           '*/.ipynb_checkpoints/*', '*.tmp']

SRC_INCLUDE = ['src/*', 'tests/*', 'scripts/*', 'examples/*', 'taxonomy/*', 'docs/*',
               'README.md', 'LICENSE', 'config.yaml', 'requirements.txt',
               'pyproject.toml', 'MANIFEST.in', '.gitignore']
FULL_EXTRA = ['data/*', 'artifacts/eval/*', 'artifacts/report/*',
              'artifacts/pred/*', 'logs/*',
              'artifacts/ckpt/*/rl_agent_config.json',
              'artifacts/ckpt/*/questions.json',
              'artifacts/ckpt/*/train_summary.json',
              'artifacts/ckpt/*/train_report.json']
WEIGHTS_FILES = (
    'model.safetensors', 'encoder/config.json', 'tokenizer/tokenizer.json',
    'tokenizer/tokenizer_config.json', 'rl_agent_config.json', 'questions.json',
    'train_summary.json', 'train_report.json',
)
WEIGHTS_INCLUDE = [f'artifacts/ckpt/laya-pbz-v1/{name}' for name in WEIGHTS_FILES]
WEIGHTS_EXTRA = {
    'LICENSE': 'LICENSE',
    'MODEL_CARD.md': 'docs/MODEL_CARD.md',
    '模型下载与调用.md': 'docs/模型下载与调用.md',
    '两级中文标签.md': 'docs/两级中文标签.md',
}


def sha256_stream(stream) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(chunk)
    return h.hexdigest()


def sha256_file(path: str) -> str:
    with open(path, 'rb') as stream:
        return sha256_stream(stream)


def verify_weights(path: str) -> dict:
    """校验完整压缩包、必需文件及每个文件的字节数和 SHA-256，无需模型依赖。"""
    with open(path + '.sha256', encoding='utf-8') as stream:
        fields = stream.read().split()
    if len(fields) != 2 or fields[1] != os.path.basename(path):
        raise ValueError('校验文件格式或压缩包文件名不匹配')
    actual = sha256_file(path)
    if fields[0] != actual:
        raise ValueError('压缩包 SHA-256 不匹配，请重新下载 zip 和 .sha256')
    prefix = 'laya-pbz-v1/'
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read(prefix + 'manifest.json'))
        if manifest.get('format_version') != 1 or manifest.get('checkpoint') != 'laya-pbz-v1':
            raise ValueError('权重包清单版本或检查点名称不匹配')
        files = manifest['files']
        if set(files) != set(WEIGHTS_FILES) | set(WEIGHTS_EXTRA):
            raise ValueError('权重包文件清单不完整或含未声明文件')
        expected = {prefix + name for name in files} | {prefix + 'manifest.json'}
        members = archive.namelist()
        if len(members) != len(expected) or set(members) != expected:
            raise ValueError('权重包目录结构不匹配或存在重复/额外文件')
        for name, info in files.items():
            member = prefix + name
            if archive.getinfo(member).file_size != info['size']:
                raise ValueError(f'文件字节数不匹配：{name}')
            with archive.open(member) as stream:
                if sha256_stream(stream) != info['sha256']:
                    raise ValueError(f'文件 SHA-256 不匹配：{name}')
    print(f'[verified] {os.path.basename(path)} · {len(files)} 个文件 · SHA-256 {actual}')
    return manifest


def walk(patterns: list[str]) -> list[str]:
    out = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', 'dist')]
        for f in files:
            full = os.path.join(base, f)
            rel = os.path.relpath(full, ROOT).replace(os.sep, '/')
            if any(fnmatch(rel, e) for e in EXCLUDE):
                continue
            if any(fnmatch(rel, p) or rel == p.rstrip('/*') for p in patterns):
                out.append(rel)
    return sorted(set(out))


def make(tag: str, patterns: list[str], level: int = 9) -> str | None:
    rels = walk(patterns)
    if tag == 'weights' and set(rels) != set(WEIGHTS_INCLUDE):
        missing = sorted(set(WEIGHTS_INCLUDE) - set(rels))
        raise FileNotFoundError(f'检查点不完整，停止打包：{missing}')
    if not rels:
        print(f'[skip] {tag}: 没有匹配文件')
        return None
    os.makedirs(DIST, exist_ok=True)
    stem = 'laya-pbz-v1-weights' if tag == 'weights' else f'{NAME}-{__version__}-{tag}'
    path = os.path.join(DIST, stem + '.zip')
    root_in_zip = f'{NAME}-{__version__}' if tag != 'weights' else 'laya-pbz-v1'
    payload = {}
    if tag == 'weights':
        payload = {name: f'artifacts/ckpt/laya-pbz-v1/{name}' for name in WEIGHTS_FILES}
        payload.update(WEIGHTS_EXTRA)
        # 记录解压后文件的校验值；只打固定检查点，排除快照及其他模型目录。
        manifest = {
            'format_version': 1,
            'checkpoint': 'laya-pbz-v1',
            'code_version': __version__,
            'files': {name: {'size': os.path.getsize(os.path.join(ROOT, rel)),
                             'sha256': sha256_file(os.path.join(ROOT, rel))}
                      for name, rel in payload.items()},
        }
        try:
            manifest['source_commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        except subprocess.CalledProcessError:
            manifest['source_commit'] = None
    print(f'[package] {os.path.basename(path)}，开始压缩…', flush=True)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=level) as z:
        if tag == 'weights':
            for name, rel in payload.items():
                z.write(os.path.join(ROOT, rel), f'{root_in_zip}/{name}')
            z.writestr(f'{root_in_zip}/manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        else:
            for rel in rels:
                z.write(os.path.join(ROOT, rel), f'{root_in_zip}/{rel}')
    h = sha256_file(path)
    with open(path + '.sha256', 'w', encoding='utf-8', newline='\n') as stream:
        stream.write(f'{h}  {os.path.basename(path)}\n')
    if tag == 'weights':
        verify_weights(path)
    print(f'[ok] {os.path.basename(path):<46} '
          f'{os.path.getsize(path) / 1e6:>8.1f} MB  sha256:{h}')
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['src', 'full', 'weights'], default=None)
    ap.add_argument('--verify', metavar='ZIP', help='校验已下载的权重 zip、同名 .sha256 和包内清单')
    a = ap.parse_args()
    if a.verify:
        verify_weights(os.path.abspath(a.verify))
        return 0
    jobs = {'src': SRC_INCLUDE, 'full': SRC_INCLUDE + FULL_EXTRA, 'weights': WEIGHTS_INCLUDE}
    for tag, pats in jobs.items():
        if a.only and tag != a.only:
            continue
        make(tag, pats, level=6 if tag == 'weights' else 9)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
