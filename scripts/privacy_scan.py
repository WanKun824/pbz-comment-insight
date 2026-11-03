"""隐私扫描：找出含评论原文 / 昵称 / UID 的文件，决定哪些不能进公开仓库。

    python scripts/privacy_scan.py
    python scripts/privacy_scan.py --tracked-only   # 发布前只复核 Git 跟踪文件
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {'.git', '__pycache__', 'dist', '.ruff_cache', '.pytest_cache', 'node_modules'}
TEXT_EXT = ('.md', '.csv', '.html', '.json', '.jsonl', '.txt', '.log', '.py', '.yml', '.yaml')
BINARY_PII = ('.parquet', '.xlsx', '.xls', '.pkl')


def load_probes(clean: str):
    """探针：昵称 / UID / 评论片段。

    视频标题与 UP 主名是**公开元数据**，聚合报告里本来就会出现，
    所以要从探针里剔除，否则会把 `视频维度.csv` 这类纯聚合文件误报成含隐私。
    """
    cl = pd.read_parquet(clean)
    public = ' '.join(cl['视频标题'].astype(str).unique()) + ' ' +              ' '.join(cl['UP主'].astype(str).unique() if 'UP主' in cl else [])
    names = {n for n in cl['昵称'].astype(str).unique()
             if len(n) > 3 and n not in public}
    uids = {u for u in cl['UID'].astype(str).unique() if len(u) > 6}
    texts = [t[:18] for t in cl['内容'].astype(str).unique()
             if len(t) > 18 and t[:18] not in public][:600]
    return names, uids, texts


def scan_text(path: str, names, uids, texts) -> list[str]:
    try:
        s = open(path, encoding='utf-8', errors='ignore').read()
    except OSError:
        return []
    hit = []
    if any(n in s for n in names):
        hit.append('昵称')
    if any(t in s for t in texts):
        hit.append('评论原文')
    if any(u in s for u in uids):
        hit.append('UID')
    return hit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracked-only', action='store_true', help='仅扫描 Git 跟踪文件，适合发布前检查')
    args = parser.parse_args()
    tracked = None
    if args.tracked_only:
        result = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, check=True, capture_output=True)
        tracked = set(result.stdout.decode('utf-8').rstrip('\0').split('\0'))
    clean = os.path.join(ROOT, 'data', 'clean', 'comments_clean.parquet')
    if not os.path.exists(clean):
        print('需要 data/clean/comments_clean.parquet（先跑 `pbz rebuild`）')
        return 1
    names, uids, texts = load_probes(clean)
    print(f'探针：{len(names):,} 个昵称 / {len(uids):,} 个 UID / {len(texts):,} 条评论片段', flush=True)

    pii, safe = [], []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            p = os.path.join(base, f)
            rel = os.path.relpath(p, ROOT).replace(os.sep, '/')
            if tracked is not None and rel not in tracked:
                continue
            size = os.path.getsize(p)
            if f.endswith(BINARY_PII):
                pii.append((rel, '二进制·含原文', size))
            elif f.endswith(TEXT_EXT) and size < 60e6:
                h = scan_text(p, names, uids, texts)
                (pii if h else safe).append((rel, ','.join(h) or '—', size))

    print(f'{"含个人数据的文件":<60}{"命中":<18}大小')
    print('-' * 94)
    for rel, h, s in sorted(pii, key=lambda x: -x[2]):
        print(f'{rel:<60}{h:<18}{s / 1024:>8.0f} KB')
    print(f'\n合计 {len(pii)} 个文件含个人数据，{sum(s for _, _, s in pii) / 1e6:.1f} MB')
    print(f'其余 {len(safe)} 个文本文件未命中。')
    return 1 if tracked is not None and pii else 0


if __name__ == '__main__':
    sys.exit(main())
