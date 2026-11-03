"""阶段 2：零样本坍缩诊断 —— 位置偏置 vs 标签文本劫持。

这个脚本的产出直接改变了方案（详见 docs/训练实录.md）：

    topic_l1              collapse  entropy  flip_vs_A  行/条
    A 语义键·无轮转          0.712    0.508       —        14
    B 语义键·满轮转          0.730    0.499     17.2%      40   ← 轮转无改善，贵 11.5 倍
    C 不透明键·满轮转        0.460    0.776     45.2%      40
    D 不透明键·无轮转        0.355    0.792     52.5%      14   ← 主因在键的字面
    E 不透明键·中文前缀      0.320    0.871        —       14   ← 采用
    G 中文键·中文前缀        0.320    0.869        —       14   ← speech_act 坍缩 0.742，弃用

指标
  collapse  最大类占比（越低越健康）
  entropy   归一化熵（越高越健康）
  flip      与基准变体相比答案变化率
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from .config import Config
from .schema import expand, load_schema, pass1_questions, reduce_answers

DIMS = ['valid', 'topic_l1', 'sentiment', 'target', 'speech_act']


def _remap(questions: dict, cn_names: dict, key_style: str, prefix_cn: bool):
    """本模块自带渲染（不复用 schema.render_keys），以便枚举 schema 不支持的变体。"""
    out, keymap = {}, {}
    for name, q in questions.items():
        q2 = {'type': q['type'], 'instructions': q['instructions']}
        crit = q.get('criteria') or {}
        if len(crit) <= 2:
            q2['criteria'] = crit
            out[name] = q2
            continue
        cn = (cn_names or {}).get(name, {})
        newc, km = {}, {}
        for i, (k, v) in enumerate(crit.items()):
            nk = {'opaque': f'O{i + 1}', 'cn': cn.get(k, k)}.get(key_style, k)
            newc[nk] = f'{cn[k]}：{v}' if (prefix_cn and cn.get(k)) else v
            km[nk] = k
        q2['criteria'] = newc
        out[name] = q2
        keymap[name] = km
    return out, keymap


def _collapse(s: pd.Series) -> tuple[float, float]:
    vc = s.value_counts(normalize=True)
    if vc.empty:
        return 0.0, 0.0
    k = max(s.nunique(), 2)
    return round(float(vc.iloc[0]), 3), round(float(-(vc * np.log(vc)).sum() / np.log(k)), 3)


def run(cfg: Config, *, n: int = 400, full_rotation: bool = False,
        batch_size: int | None = None) -> pd.DataFrame:
    from .predict import _agent

    bs = batch_size or int(cfg.get('model.batch_size', 8))
    sc = load_schema(cfg.path('paths.taxonomy'))
    cn = sc.get('cn_names', {})
    q1 = pass1_questions(sc)

    df = pd.read_parquet(cfg.path('paths.clean'))
    df = pd.concat([g.sample(min(len(g), max(1, n // 5)), random_state=0)
                    for _, g in df.groupby('len_bucket', observed=True)]).reset_index(drop=True)
    states = df['state'].tolist()
    print(f'[diag] {len(states)} 条（按长度分层）')

    variants: dict[str, tuple[str, bool, str]] = {
        'A_语义键_无轮转':    ('semantic', False, '1'),
        'D_不透明键_原描述':  ('opaque', False, '1'),
        'E_不透明键_中文前缀': ('opaque', True, '1'),
        'G_中文键_中文前缀':   ('cn', True, '1'),
    }
    if full_rotation:
        variants['B_语义键_满轮转'] = ('semantic', False, 'full')
        variants['C_不透明键_满轮转'] = ('opaque', True, 'full')

    agent = _agent(cfg, None, base=True)
    res: dict[str, pd.DataFrame] = {}
    for vname, (ks, pc, rot) in variants.items():
        qs, km = _remap(q1, cn, ks, pc)
        lq, meta = expand(qs, rot)
        t0 = time.time()
        rr = agent.predict_batch(states, lq, batch_size=bs, sort_by_length=True)
        red = [reduce_answers(r['answers'], meta, km) for r in rr]
        rows = {}
        for d in DIMS:
            rows[d] = [o.get(d, {}).get('label') for o in red]
            rows[d + '_p'] = [o.get(d, {}).get('p') for o in red]
        res[vname] = pd.DataFrame(rows)
        print(f'[diag] {vname:<22} {len(lq)} 行/条  {time.time() - t0:.0f}s')

    base_v = next(iter(res))
    out_rows = []
    for d in DIMS:
        for vname, r in res.items():
            c, e = _collapse(r[d])
            out_rows.append({'维度': d, '变体': vname, 'collapse': c, 'entropy': e,
                             'flip_vs_基准': '' if vname == base_v
                             else f'{(r[d] != res[base_v][d]).mean() * 100:.1f}%',
                             '平均置信': round(float(r[d + "_p"].mean()), 3)})
    tab = pd.DataFrame(out_rows)

    out_dir = cfg.path('paths.eval_dir')
    os.makedirs(out_dir, exist_ok=True)
    L = ['# 零样本坍缩诊断\n', f'样本 {len(states)} 条（按长度分层），基座零样本\n',
         '指标：`collapse` = 最大类占比（越低越健康）；`entropy` = 归一化熵（越高越健康）\n',
         tab.to_markdown(index=False), '']
    for d in ['topic_l1', 'valid', 'sentiment']:
        cmp = pd.DataFrame({v: r[d].value_counts(normalize=True) for v, r in res.items()})
        L += [f'\n## {d} 各变体分布 (%)\n', (cmp * 100).round(1).fillna(0).to_markdown(), '']
    L += ['\n## 结论\n',
          '- 坍缩主因是模型按**字面**读英文语义键，不是位置偏置',
          '- `option_order` 满轮转 collapse 无改善，成本却是 11.5 倍 → 关闭',
          '- 采用「不透明键 O1..On + criteria 前置中文类名」',
          '- 所有变体的 `sentiment` 都把多数判为负面，而本批数据偏正面 → 零样本不可用，必须微调']
    open(os.path.join(out_dir, 'diag.md'), 'w', encoding='utf-8').write('\n'.join(L))
    print('\n' + tab.to_string(index=False))
    print(f'\n[diag] → {out_dir}/diag.md')
    return tab
