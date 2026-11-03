# -*- coding: utf-8 -*-
"""隐私扫描：找出含评论原文 / 昵称 / UID 的文件，决定哪些不能进公开仓库。

    python scripts/privacy_scan.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {'.git', '__pycache__', 'dist', '.ruff_cache', '.pytest_cache', 'node_modules'}
TEXT_EXT = ('.md', '.csv', '.html', '.json', '.jsonl', '.txt', '.log', '.py', '.yml', '.yaml')
BINARY_PII = ('.parquet', '.xlsx', '.xls', '.pkl')


def load_probes(clean: str):
    cl = pd.read_parquet(clean)
    names = {n for n in cl['昵称'].astype(str).head(500) if len(n) > 3}
    uids = {u for u in cl['UID'].astype(str).head(500) if len(u) > 6}
    texts = [t[:18] for t in cl['内容'].astype(str) if len(t) > 18][:400]
    return names, uids, texts


def scan_text(path: str, names, uids, texts) -> list[str]:
    try:
        s = open(path, encoding='utf-8', errors='ignore').read()
    except OSError:
        return []
    hit = []
    if sum(1 for n in names if n in s) >= 2:
        hit.append('昵称')
    if sum(1 for t in texts if t in s) >= 2:
        hit.append('评论原文')
    if sum(1 for u in uids if u in s) >= 3:
        hit.append('UID')
    return hit


def main() -> int:
    clean = os.path.join(ROOT, 'data', 'clean', 'comments_clean.parquet')
    if not os.path.exists(clean):
        print('需要 data/clean/comments_clean.parquet（先跑 `pbz rebuild`）')
        return 1
    names, uids, texts = load_probes(clean)

    pii, safe = [], []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            p = os.path.join(base, f)
            rel = os.path.relpath(p, ROOT).replace(os.sep, '/')
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
    return 0


if __name__ == '__main__':
    sys.exit(main())
