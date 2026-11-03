"""打三个分发包：src / full / weights。

    python scripts/package.py              # 全打
    python scripts/package.py --only src
"""
from __future__ import annotations

import argparse
import hashlib
import os
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

SRC_INCLUDE = ['src/*', 'tests/*', 'scripts/*', 'taxonomy/*', 'docs/*',
               'README.md', 'LICENSE', 'config.yaml', 'requirements.txt',
               'pyproject.toml', 'MANIFEST.in', '.gitignore']
FULL_EXTRA = ['data/*', 'artifacts/eval/*', 'artifacts/report/*',
              'artifacts/pred/*', 'logs/*',
              'artifacts/ckpt/*/rl_agent_config.json',
              'artifacts/ckpt/*/questions.json',
              'artifacts/ckpt/*/train_summary.json',
              'artifacts/ckpt/*/train_report.json']
WEIGHTS_INCLUDE = ['artifacts/ckpt/*']


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
    if not rels:
        print(f'[skip] {tag}: 没有匹配文件')
        return None
    os.makedirs(DIST, exist_ok=True)
    stem = 'laya-pbz-v1-weights' if tag == 'weights' else f'{NAME}-{__version__}-{tag}'
    path = os.path.join(DIST, stem + '.zip')
    root_in_zip = f'{NAME}-{__version__}' if tag != 'weights' else 'laya-pbz-v1'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=level) as z:
        for rel in rels:
            arc = rel
            if tag == 'weights':
                # artifacts/ckpt/laya-pbz-v1/model.safetensors -> model.safetensors
                arc = rel.split('artifacts/ckpt/', 1)[-1].split('/', 1)[-1]
            z.write(os.path.join(ROOT, rel), f'{root_in_zip}/{arc}')
    h = hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]
    print(f'[ok] {os.path.basename(path):<46} {len(rels):>5} 文件  '
          f'{os.path.getsize(path) / 1e6:>8.1f} MB  sha256:{h}')
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['src', 'full', 'weights'], default=None)
    a = ap.parse_args()
    jobs = {'src': SRC_INCLUDE, 'full': SRC_INCLUDE + FULL_EXTRA, 'weights': WEIGHTS_INCLUDE}
    for tag, pats in jobs.items():
        if a.only and tag != a.only:
            continue
        make(tag, pats, level=6 if tag == 'weights' else 9)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
