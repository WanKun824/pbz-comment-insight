"""阶段 6：两级推理打标。

Pass1 一次前向出 14 个正交维度；`valid` 闸门拦下 fragment/spam；
Pass2 按 topic_l1 分支问对应的 topic_l2。
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from .config import Config
from .schema import (
    DIMS_BIN,
    DIMS_MULTI,
    DIMS_UNTRAINED,
    expand,
    load_agent,
    load_base_agent,
    load_schema,
    pass1_questions,
    pass2_questions,
    reduce_answers,
    render_keys,
    render_opts,
)

GATE = {'fragment', 'spam'}


def _agent(cfg: Config, ckpt: str | None, base: bool):
    ml, hl = int(cfg.get('model.max_len', 768)), int(cfg.get('model.head_max_len', 320))
    if base:
        return load_base_agent(cfg.get('model.base_id'), cfg.get('model.base_subfolder'),
                               cfg.device, ml, hl)
    path = ckpt or cfg.path('paths.ckpt')
    if not os.path.exists(path):
        raise FileNotFoundError(f'找不到检查点：{path}（先跑 `pbz train`，或加 --base 用零样本）')
    return load_agent(path, cfg.device, ml, hl)


def run(cfg: Config, *, out: str | None = None, limit: int = 0, ckpt: str | None = None,
        base: bool = False, rotate: str | None = None, batch_size: int | None = None,
        no_pass2: bool = False, states: str | None = None) -> pd.DataFrame:
    rotate = rotate or str(cfg.get('model.rotate', '1'))
    bs = batch_size or int(cfg.get('model.batch_size', 8))
    out = out or cfg.path('paths.pred')

    sc = load_schema(cfg.path('paths.taxonomy'))
    cn, ks, pc = render_opts(sc)
    q1_raw, q2_raw = pass1_questions(sc), pass2_questions(sc)

    src = states or cfg.path('paths.clean')
    df = pd.read_parquet(src) if src.endswith('.parquet') else pd.read_csv(src)
    if limit:
        df = df.sample(min(limit, len(df)), random_state=0)
    df = df.reset_index(drop=True)
    print(f'[predict] {len(df)} 条　state 字数中位 {df["state"].str.len().median():.0f}')

    agent = _agent(cfg, ckpt, base)
    print(f'[predict] device={agent.device} dtype={agent.dtype} '
          f'layout={agent.cfg.get("option_layout", "sequential")} '
          f'max_len={agent.cfg["max_len"]} head_max_len={agent.cfg["head_max_len"]}')

    states_list = df['state'].tolist()

    # ---------------- Pass1
    q1, km1 = render_keys(q1_raw, cn, ks, pc)
    lq1, meta1 = expand(q1, rotate)
    print(f'[predict] pass1 {len(q1_raw)} 维度 → {len(lq1)} 行/条（轮转 {rotate}、键 {ks}）'
          f'　总决策 {len(lq1) * len(states_list):,}')
    t0 = time.time()
    r1 = agent.predict_batch(states_list, lq1, batch_size=bs, sort_by_length=True)
    dt = time.time() - t0
    print(f'[predict] pass1 {dt:.1f}s　{dt / len(states_list) * 1000:.0f}ms/条　'
          f'{len(lq1) * len(states_list) / dt:,.0f} 决策/秒')
    res = [reduce_answers(r['answers'], meta1, km1) for r in r1]

    # ---------------- Pass2
    if not no_pass2:
        groups: dict[str, list[int]] = {}
        for i, o in enumerate(res):
            if o['valid']['label'] in GATE:
                continue
            l1 = o['topic_l1']['label']
            if l1 in q2_raw:
                groups.setdefault(l1, []).append(i)
        gated = len(states_list) - sum(len(v) for v in groups.values())
        print(f'[predict] pass2 {len(groups)} 分支，待问 {len(states_list) - gated} 条，'
              f'valid 闸门拦下 {gated} 条')
        t0 = time.time()
        for l1, idx in sorted(groups.items(), key=lambda x: -len(x[1])):
            q2, km2 = render_keys({'topic_l2': q2_raw[l1]}, cn, ks, pc)
            lq2, meta2 = expand(q2, rotate)
            r2 = agent.predict_batch([states_list[i] for i in idx], lq2,
                                     batch_size=bs, sort_by_length=True)
            for i, r in zip(idx, r2, strict=True):
                res[i].update(reduce_answers(r['answers'], meta2, km2))
            print(f'           {l1:<10} n={len(idx)}')
        print(f'[predict] pass2 {time.time() - t0:.1f}s')

    # ---------------- 落盘
    rec: dict[str, list] = {}
    for name in list(q1_raw) + ['topic_l2']:
        rec[f'p_{name}'] = [o.get(name, {}).get('label') for o in res]
        rec[f'c_{name}'] = [o.get(name, {}).get('p') for o in res]
    rec['m_topic_l1'] = [o.get('topic_l1', {}).get('margin') for o in res]
    rec['probs_json'] = [json.dumps({k: v.get('probs', {}) for k, v in o.items()},
                                    ensure_ascii=False) for o in res]
    final = pd.concat([df, pd.DataFrame(rec)], axis=1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    final.to_parquet(out, index=False)
    print(f'\n[predict] → {out}　{final.shape}')

    _preview(final)
    return final


def _preview(res: pd.DataFrame) -> None:
    print('\n=== 分布速览 ===')
    for name in DIMS_MULTI + ['topic_l2']:
        col = f'p_{name}'
        if col not in res:
            continue
        vc = res[col].value_counts(dropna=False)
        print(f'\n[{name}] 平均置信 {res[f"c_{name}"].mean():.3f}')
        for k, v in vc.head(10).items():
            print(f'   {str(k):<16} {v:>5}  {v / len(res) * 100:>5.1f}%')
    print('\n[二分维度 A 占比]')
    for b in DIMS_BIN + DIMS_UNTRAINED:
        c = f'p_{b}'
        if c in res:
            tag = '  ← 未训练，不可用' if b in DIMS_UNTRAINED else ''
            print(f'   {b:<22} A={np.mean(res[c] == "A") * 100:>5.1f}%  '
                  f'平均置信 {res[f"c_{b}"].mean():.3f}{tag}')
