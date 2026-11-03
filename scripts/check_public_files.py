"""检查 Git 跟踪文件是否混入评论原文、昵称、UID 或私有产物路径。

    python scripts/check_public_files.py

与 privacy_scan.py 的内容探针互补；此检查不依赖本地原始评论，可直接在 CI 运行。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = [
    (r'"uname"\s*:\s*"[^<]', 'json 里的 uname 字段'),
    (r'"uid"\s*:\s*"\d{5,}"', 'json 里的 uid'),
    (r'^\s*\d{11,12}\s*,', 'csv 首列裸 rpid'),
]
PII_PATHS = [
    'data/raw/cache/', 'data/raw/markdown/', 'comments_clean.parquet',
    'comments_clean.csv', 'weak_labels.parquet', 'pred_full.parquet',
    'spotcheck.csv', 'golden_todo', '可行动清单', '舆情报告', '.jsonl', 'logs/',
]


def main() -> int:
    result = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, check=True, capture_output=True)
    tracked = result.stdout.decode('utf-8').rstrip('\0').split('\0')
    bad = []
    for name in tracked:
        for fragment in PII_PATHS:
            if fragment in name:
                bad.append(f'{name} ← 路径命中 {fragment}')
        path = ROOT / name
        if name == 'data/README.md' or path.suffix not in {'.json', '.csv', '.md', '.html', '.txt'}:
            continue
        text = path.read_text(encoding='utf-8', errors='ignore')
        for pattern, reason in FORBIDDEN:
            if re.search(pattern, text, re.M):
                bad.append(f'{name} ← 内容命中 {reason}')
    if bad:
        print('隐私门禁失败：\n' + '\n'.join('  ' + item for item in bad))
        return 1
    print(f'ok  {len(tracked)} 个跟踪文件，未命中评论原文 / 昵称 / UID 检查规则')
    return 0


if __name__ == '__main__':
    sys.exit(main())
