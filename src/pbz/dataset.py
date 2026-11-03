"""阶段 3b：弱标签 → Laya 训练 JSONL，并导出人工标注待办。

三个关键处理（少一个都会出问题）
--------------------------------
1. **按 (维度, 标签) 封顶** —— `is_toxic` 原始 6003:11，不封顶模型只会学先验。
2. **按 rpid 切分** —— 同一条评论的所有决策进同一个 split，杜绝泄漏。
3. **gold schema** —— `laya.train` 要的是 ``{dim: {"probabilities": {key: 1.0}}}``，
   不是裸标签。第一版写成裸标签，`items_from_rows` 直接返回 0 items / invalid_target。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .config import Config
from .schema import (
    DIMS_BIN,
    DIMS_MULTI,
    DIMS_UNTRAINED,
    invert_keymap,
    load_schema,
    pass1_questions,
    render_keys,
    render_opts,
)

DIMS = DIMS_MULTI + DIMS_BIN

#  稀有类定向补齐：这些词命中的样本在随机抽样里几乎抽不到
RARE_PATTERN = (r'外挂|着色器|编译|崩|闪退|退款|周边|典藏|手办|剧透|结局|'
                r'傻|智障|阴阳|反讽|建议|希望|应该')


def run(cfg: Config, cap: int | None = None) -> dict[str, int]:
    cap = cap or int(cfg.get('dataset.cap_per_label', 450))
    seed = int(cfg.get('dataset.seed', 0))
    dev_frac = float(cfg.get('dataset.dev_frac', 0.10))
    test_frac = float(cfg.get('dataset.test_frac', 0.12))
    golden_n = int(cfg.get('dataset.golden_n', 600))
    rng = np.random.RandomState(seed)

    sc = load_schema(cfg.path('paths.taxonomy'))
    cn, ks, pc = render_opts(sc)
    q1, km1 = render_keys(pass1_questions(sc), cn, ks, pc)
    inv = invert_keymap(km1)

    w = pd.read_parquet(cfg.path('paths.weak'))
    out_dir = cfg.path('paths.train_dir')
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- 封顶
    keep: dict[str, set] = {d: set() for d in DIMS}
    cap_log: list[tuple[str, str, int, int]] = []
    for d in DIMS:
        sub = w[w['w_' + d].notna()]
        for lab, g in sub.groupby('w_' + d):
            take = g if len(g) <= cap else g.sample(cap, random_state=seed)
            keep[d] |= set(take['rpid'])
            cap_log.append((d, lab, len(g), len(take)))

    # ---------------- 按 rpid 切分
    rpids = w['rpid'].tolist()
    rng.shuffle(rpids)
    n_test, n_dev = int(len(rpids) * test_frac), int(len(rpids) * dev_frac)
    split = {rp: ('test' if i < n_test else ('dev' if i < n_test + n_dev else 'train'))
             for i, rp in enumerate(rpids)}

    # ---------------- 生成行
    rows: dict[str, list] = {'train': [], 'dev': [], 'test': []}
    stat = {k: {d: {} for d in DIMS} for k in rows}
    for r in w.itertuples(index=False):
        gold, qs = {}, {}
        for d in DIMS:
            lab = getattr(r, 'w_' + d)
            if lab is None or (isinstance(lab, float) and np.isnan(lab)):
                continue
            if r.rpid not in keep[d]:
                continue
            key = inv.get(d, {}).get(lab, lab)        # 语义键 → O_n（二分维度仍是 A/B）
            gold[d] = {'probabilities': {key: 1.0}}
            qs[d] = q1[d]
            s = split[r.rpid]
            stat[s][d][lab] = stat[s][d].get(lab, 0) + 1
        if gold:
            rows[split[r.rpid]].append({'state': r.state, 'questions': qs, 'gold': gold})

    counts = {}
    for k, v in rows.items():
        p = os.path.join(out_dir, f'{k}.jsonl')
        with open(p, 'w', encoding='utf-8') as f:
            for row in v:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        counts[k] = sum(len(x['gold']) for x in v)
        print(f'[dataset] {k:<5} {len(v):>5} 条　{counts[k]:>6} 决策　→ {p}')

    # ---------------- 统计
    L = ['# 训练集统计\n',
         f'封顶 cap={cap}　seed={seed}　test={test_frac:.0%} dev={dev_frac:.0%}\n',
         '## 切分规模\n', '| split | 条数 | 决策数 |', '|---|---:|---:|']
    for k, v in rows.items():
        L.append(f'| {k} | {len(v)} | {counts[k]} |')
    L += ['', '## 封顶前后（全量）\n', '| 维度 | 标签 | 原始 | 保留 |', '|---|---|---:|---:|']
    for d, lab, a, b in cap_log:
        L.append(f'| `{d}` | {lab} | {a} | {b}{" ⬇" if b < a else ""} |')
    L += ['', '## 各 split 标签分布\n']
    for d in DIMS:
        L.append(f'\n**{d}**\n')
        labs = sorted({l for s in rows for l in stat[s][d]})
        L.append('| 标签 | ' + ' | '.join(rows) + ' |')
        L.append('|---|' + '---:|' * len(rows))
        for lab in labs:
            L.append(f'| {lab} | ' + ' | '.join(str(stat[s][d].get(lab, 0)) for s in rows) + ' |')
    open(os.path.join(out_dir, 'stats.md'), 'w', encoding='utf-8').write('\n'.join(L))

    _golden(w, split, out_dir, golden_n)
    return counts


def _golden(w: pd.DataFrame, split: dict, out_dir: str, n: int) -> None:
    """人工标注待办：分层 + 稀有类过采样。标完后替换 test 集，才是真指标。"""
    pool = [sub.sample(min(len(sub), 25), random_state=1)
            for _, sub in w.groupby(['len_bucket', 'like_bucket'], observed=True)]
    rare = w[w['内容'].astype(str).str.contains(RARE_PATTERN, regex=True, na=False)]
    if len(rare):
        pool.append(rare.sample(min(220, len(rare)), random_state=1))
    g = (pd.concat(pool).drop_duplicates('rpid')
         .sample(frac=1, random_state=1).head(n))
    g = g.assign(split_hint=g['rpid'].map(split))

    cols = ['rpid', 'split_hint', '视频标题', '层级', '字数', '点赞', '父评论', '内容']
    out = g[[c for c in cols if c in g.columns]].copy()
    for d in DIMS:
        out['弱_' + d] = g['w_' + d].values
        out['人工_' + d] = ''
    for d in DIMS_UNTRAINED + ['topic_l2']:
        out['人工_' + d] = ''

    p = os.path.join(out_dir, 'golden_todo.xlsx')
    try:
        out.to_excel(p, index=False)
    except Exception as e:                            # noqa: BLE001
        p = os.path.join(out_dir, 'golden_todo.csv')
        out.to_csv(p, index=False, encoding='utf-8-sig')
        print(f'[dataset] xlsx 不可用（{e}），落 csv')
    print(f'[dataset] 人工标注待办 {len(out)} 条 → {p}')
