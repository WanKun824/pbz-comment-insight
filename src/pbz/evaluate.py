"""阶段 5：冻结 test 集评测。

必须同时看三样东西，少一样就会得出错误结论：

- **accuracy**：直觉指标，但会被多数类骗（`churn_signal` 的多数类就是 0.897）
- **macro-F1**：类不平衡下的真实水平（`is_toxic` accuracy 0.958 而 macro-F1 只有 0.489
  —— 它其实是全预测 B）
- **ECE + 覆盖率-准确率曲线**：决定能把多少比例自动化

⚠️ test 标签来自规则弱监督，不是人工。这里的分数衡量「模型是否学会了规则体现的那套
判别」，不等于真实人工一致率。`data/train/golden_todo.xlsx` 标完后用同一脚本重跑，
才是真指标。
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

from .config import Config
from .schema import expand, load_schema, pass1_questions, reduce_answers, render_keys, render_opts

COV_STEPS = (1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1)


def macro_f1(y: list, p: list, labels: list) -> float:
    out = []
    for l in labels:
        tp = sum(1 for a, b in zip(y, p, strict=True) if a == l and b == l)
        fp = sum(1 for a, b in zip(y, p, strict=True) if a != l and b == l)
        fn = sum(1 for a, b in zip(y, p, strict=True) if a == l and b != l)
        if tp + fn == 0:                              # 该类在 test 里没有真值 → 不计入
            continue
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn)
        out.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    return float(np.mean(out)) if out else 0.0


def ece(conf, correct, bins: int = 10) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if not len(conf):
        return 0.0
    e = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.sum() / len(conf) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def cov_acc(conf, correct, steps=COV_STEPS):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(-conf)
    out = []
    for frac in steps:
        idx = order[:max(1, int(len(conf) * frac))]
        out.append((frac, float(correct[idx].mean()), float(conf[idx].min())))
    return out


def run(cfg: Config, *, data: str | None = None, ckpt: str | None = None,
        no_base: bool = False, batch_size: int | None = None) -> pd.DataFrame:
    import torch

    from .predict import _agent

    data = data or os.path.join(cfg.path('paths.train_dir'), 'test.jsonl')
    bs = batch_size or int(cfg.get('model.batch_size', 8))
    out_dir = cfg.path('paths.eval_dir')
    os.makedirs(out_dir, exist_ok=True)

    sc = load_schema(cfg.path('paths.taxonomy'))
    cn, ks, pc = render_opts(sc)
    q1, km1 = render_keys(pass1_questions(sc), cn, ks, pc)
    inv_name = {d: m for d, m in km1.items()}          # 不透明键 → 语义键（显示用）

    rows = [json.loads(l) for l in open(data, encoding='utf-8')]
    states = [r['state'] for r in rows]
    gold = [{d: max(g['probabilities'], key=g['probabilities'].get) for d, g in r['gold'].items()}
            for r in rows]
    dims = sorted({d for g in gold for d in g})
    print(f'[eval] {len(rows)} 条　{sum(len(g) for g in gold)} 决策　{len(dims)} 维度')

    preds: dict[str, list] = {}
    todo = [] if no_base else [('base', True)]
    todo.append(('finetuned', False))
    for name, use_base in todo:
        try:
            agent = _agent(cfg, ckpt, use_base)
        except FileNotFoundError as e:
            print(f'[eval] 跳过 {name}: {e}')
            continue
        lq, meta = expand({d: q1[d] for d in dims}, '1')
        t0 = time.time()
        rr = agent.predict_batch(states, lq, batch_size=bs, sort_by_length=True)
        dt = time.time() - t0
        preds[name] = [reduce_answers(r['answers'], meta, None) for r in rr]
        print(f'[eval] {name:<10} {dt:.1f}s  {len(lq) * len(states) / dt:,.0f} 决策/秒  '
              f'layout={agent.cfg.get("option_layout", "sequential")}')
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------- 指标
    summary = []
    for d in dims:
        idxs = [i for i, g in enumerate(gold) if d in g]
        y = [gold[i][d] for i in idxs]
        labels = sorted(set(y))
        maj_key, maj_n = Counter(y).most_common(1)[0]
        row = {'维度': d, 'n': len(y), '类数': len(labels),
               '多数类': f'{inv_name.get(d, {}).get(maj_key, maj_key)} {maj_n / len(y):.3f}'}
        for name, pred in preds.items():
            p = [pred[i].get(d, {}).get('label') for i in idxs]
            c = [pred[i].get(d, {}).get('p', 0.0) for i in idxs]
            corr = [int(a == b) for a, b in zip(y, p, strict=True)]
            row[f'{name}_acc'] = round(float(np.mean(corr)), 4)
            row[f'{name}_mF1'] = round(macro_f1(y, p, labels), 4)
            row[f'{name}_ECE'] = round(ece(c, corr), 4)
        summary.append(row)
    sm = pd.DataFrame(summary)

    L = ['# 评测报告（冻结 test 集）\n',
         f'数据 `{os.path.relpath(data, cfg.root)}`　{len(rows)} 条　'
         f'{sum(len(g) for g in gold)} 决策\n',
         '> ⚠️ 标签来自规则弱监督，不是人工标注。这些分数衡量「模型是否学会了规则体现的'
         '那套判别」，不等于真实人工一致率。`data/train/golden_todo.xlsx` 标完后用同一'
         '脚本重跑即为真指标。\n',
         '## 总览\n', sm.to_markdown(index=False), '',
         '> 必须看 **macro-F1**：类不平衡时 accuracy 会被多数类骗。\n']

    if 'finetuned' in preds:
        pred = preds['finetuned']
        for d in dims:
            idxs = [i for i, g in enumerate(gold) if d in g]
            y = [gold[i][d] for i in idxs]
            p = [pred[i].get(d, {}).get('label') for i in idxs]
            c = [pred[i].get(d, {}).get('p', 0.0) for i in idxs]
            corr = [int(a == b) for a, b in zip(y, p, strict=True)]
            m = inv_name.get(d, {})
            cm = pd.crosstab(pd.Series([m.get(x, x) for x in y], name='真值'),
                             pd.Series([m.get(x, x) for x in p], name='预测'))
            L += [f'\n### {d}\n', '混淆矩阵（行=真值，列=预测）\n', cm.to_markdown(), '',
                  '覆盖率–准确率（按置信度降序保留 top X%）\n',
                  '| 覆盖率 | accuracy | 该点阈值 |', '|---:|---:|---:|']
            L += [f'| {f:.0%} | {a:.3f} | {th:.3f} |' for f, a, th in cov_acc(c, corr)]

    open(os.path.join(out_dir, 'report.md'), 'w', encoding='utf-8').write('\n'.join(L))
    sm.to_csv(os.path.join(out_dir, 'summary.csv'), index=False, encoding='utf-8-sig')
    print('\n' + sm.to_string(index=False))
    print(f'\n[eval] → {out_dir}/report.md, summary.csv')
    return sm
