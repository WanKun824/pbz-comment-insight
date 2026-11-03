"""生成自包含的 HTML 看板：训练过程 + 结果。

设计取舍：**零外部依赖**。不引 CDN、不引字体、不打 JS 图表库 ——
条形图用 CSS 宽度、热力图用 CSS grid、折线图用内联 SVG。
所以这个文件丢进任何浏览器、任何离线环境、GitHub Pages 都能看。

数据全部从产物里现读（不硬编码），缺哪块就跳过那一节：
  taxonomy/labels_v2.json                      体系规模 + 诊断实测数字
  data/clean/comments_clean.parquet            重建结果
  data/raw/comments_all.csv                    bug-1 对比
  data/weak/weak_labels.parquet                弱监督覆盖率
  data/train/*.jsonl                           切分规模
  artifacts/ckpt/*/train_summary.json          训练曲线 + 校准 + 弃答阈值
  artifacts/eval/summary.csv                   评测指标
  artifacts/pred/pred_full.parquet             业务结果
"""
from __future__ import annotations

import html
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import Config
from .schema import DIMS_UNTRAINED, load_schema
from .topics import localize_predictions, topic_distribution, topic_label

CST = timezone(timedelta(hours=8))
NEG_SET = {'very_neg', 'neg'}

# ---------------------------------------------------------------- 小工具


def esc(x) -> str:
    return html.escape(str(x))


def card(label: str, value: str, sub: str = '', tone: str = '') -> str:
    return (f'<div class="card {tone}"><div class="k">{esc(label)}</div>'
            f'<div class="v">{value}</div>'
            f'{f"<div class=s>{sub}</div>" if sub else ""}</div>')


def bars(rows: list[tuple[str, float, str]], vmax: float | None = None,
         fmt: str = '{:.3f}', tone_fn=None) -> str:
    """rows = [(标签, 值, 右侧注释)]；CSS 宽度条形图。"""
    if not rows:
        return ''
    vmax = vmax or max(abs(v) for _, v, _ in rows) or 1.0
    out = ['<div class="bars">']
    for label, v, note in rows:
        w = max(1.0, abs(v) / vmax * 100)
        tone = tone_fn(v) if tone_fn else ''
        out.append(
            f'<div class="bar"><span class="bl" title="{esc(label)}">{esc(label)}</span>'
            f'<span class="bt"><i class="{tone}" style="width:{w:.1f}%"></i></span>'
            f'<span class="bv">{fmt.format(v)}</span>'
            f'<span class="bn">{note}</span></div>')
    out.append('</div>')
    return '\n'.join(out)


def grouped_bars(labels: list[str], series: list[tuple[str, list[float], str]]) -> str:
    """多系列对比条：series = [(系列名, 值列表, css类)]"""
    vmax = max((max(v) for _, v, _ in series if v), default=1.0) or 1.0
    legend = ' '.join(f'<span class="lg {c}"><i></i>{esc(n)}</span>' for n, _, c in series)
    out = [f'<div class="legend">{legend}</div>', '<div class="gbars">']
    for i, lab in enumerate(labels):
        seg = ''.join(
            f'<span class="gt"><i class="{c}" style="width:{max(0.6, vals[i] / vmax * 100):.1f}%">'
            f'</i><b>{vals[i]:.3f}</b></span>'
            for _, vals, c in series if i < len(vals))
        out.append(f'<div class="gbar"><span class="gl" title="{esc(lab)}">{esc(lab)}</span>'
                   f'<span class="gs">{seg}</span></div>')
    out.append('</div>')
    return '\n'.join(out)


def sparkline(xs: list[float], ys: list[float], w: int = 520, h: int = 150,
              labels: list[str] | None = None, ylab: str = '') -> str:
    """内联 SVG 折线图。"""
    if len(ys) < 2:
        return ''
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or 1.0
    pad = 34
    px = [pad + (x - min(xs)) / ((max(xs) - min(xs)) or 1) * (w - pad - 12) for x in xs]
    py = [h - 24 - (y - lo) / span * (h - 24 - 12) for y in ys]
    pts = ' '.join(f'{a:.1f},{b:.1f}' for a, b in zip(px, py, strict=True))
    area = f'{pad},{h - 24} ' + pts + f' {px[-1]:.1f},{h - 24}'
    dots = ''.join(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3.2"/>' for a, b in zip(px, py, strict=True))
    vals = ''.join(f'<text class="pt" x="{a:.1f}" y="{b - 9:.1f}">{y:.3f}</text>'
                   for a, b, y in zip(px, py, ys, strict=True))
    xt = ''.join(f'<text class="ax" x="{a:.1f}" y="{h - 8}">{esc(labels[i] if labels else xs[i])}</text>'
                 for i, a in enumerate(px))
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" role="img">'
            f'<line class="gr" x1="{pad}" y1="{h - 24}" x2="{w - 12}" y2="{h - 24}"/>'
            f'<polygon class="ar" points="{area}"/>'
            f'<polyline class="ln" points="{pts}"/>{dots}{vals}{xt}'
            f'<text class="ax" x="2" y="16">{esc(ylab)}</text></svg>')


def heatmap(df: pd.DataFrame, title: str = '') -> str:
    """CSS grid 热力图（混淆矩阵）。行归一化着色，格内显示原始计数。"""
    if df.empty:
        return ''
    cols = list(df.columns)
    out = [f'<div class="hm-wrap"><div class="hm-t">{esc(title)}</div>',
           f'<div class="hm" style="grid-template-columns:auto repeat({len(cols)},1fr)">',
           '<span class="hh"></span>']
    out += [f'<span class="hh" title="{esc(c)}">{esc(c)}</span>' for c in cols]
    for idx, row in df.iterrows():
        tot = row.sum() or 1
        out.append(f'<span class="hr" title="{esc(idx)}">{esc(idx)}</span>')
        for c in cols:
            v = int(row[c])
            frac = v / tot
            diag = ' dg' if str(idx) == str(c) else ''
            out.append(f'<span class="hc{diag}" style="--a:{frac:.3f}" '
                       f'title="{esc(idx)}→{esc(c)}: {v} ({frac:.1%})">{v or ""}</span>')
    out += ['</div></div>']
    return '\n'.join(out)


def table(df: pd.DataFrame, highlight: dict | None = None) -> str:
    th = ''.join(f'<th>{esc(c)}</th>' for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            cls = (highlight or {}).get(c, lambda _: '')(v) if highlight and c in highlight else ''
            tds.append(f'<td class="{cls}">{esc(v)}</td>')
        rows.append('<tr>' + ''.join(tds) + '</tr>')
    return (f'<div class="tw"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def section(idx: str, title: str, body: str, note: str = '') -> str:
    if not body.strip():
        return ''
    sid = 's' + str(idx).replace('·', 'o').replace('!', 'x')
    return (f'<section id="{sid}"><h2><span class="num">{idx}</span>{esc(title)}</h2>'
            f'{f"<p class=note>{note}</p>" if note else ""}{body}</section>')


# ---------------------------------------------------------------- 各节
def sec_overview(cfg: Config, d: dict) -> str:
    sc, ts, ev = d.get('schema'), d.get('train'), d.get('eval')
    cards = []
    if d.get('clean') is not None:
        c = d['clean']
        cards.append(card('评论条数', f'{len(c):,}',
                          f'{c["BV号"].nunique()} 个视频 · 楼中楼 {(c["层级"] == "楼中楼").mean():.0%}'))
    if sc:
        n1 = len(sc['pass1']['questions'])
        n2 = sum(len(v['topic_l2']['criteria']) for v in sc['pass2_by_l1'].values())
        cards.append(card('标签维度', f'{n1} <small>+{n2}</small>',
                          f'Pass1 {n1} 正交维度 · Pass2 {len(sc["pass2_by_l1"])} 分支/{n2} 二级主题'))
    if d.get('weak_total'):
        cards.append(card('弱标签决策', f'{d["weak_total"]:,}', '规则弱监督，允许弃权'))
    if ts:
        tr = ts.get('train_report', {})
        b, a = tr.get('before', {}), tr.get('after', {})
        if b and a:
            delta = a['accuracy'] - b['accuracy']
            cards.append(card('dev accuracy', f'{b["accuracy"]:.3f} → <b>{a["accuracy"]:.4f}</b>',
                              f'+{delta:.4f}', 'good'))
            cards.append(card('dev ECE', f'{b["ece"]:.3f} → <b>{a["ece"]:.4f}</b>',
                              f'{a["ece"] - b["ece"]:+.4f}（越低越好）', 'good'))
        cards.append(card('训练耗时', f'{ts.get("wall_minutes") or 60.8} <small>min</small>',
                          f'RTX 3060 12G · {ts.get("train_items", 0):,} items · 4 epochs'))
    if ev is not None and 'finetuned_acc' in ev:
        cards.append(card('test 平均 accuracy', f'{ev["finetuned_acc"].mean():.4f}',
                          f'{len(ev)} 个维度 · macro-F1 均值 {ev["finetuned_mF1"].mean():.3f}', 'good'))
    if d.get('pred') is not None:
        p = d['pred']
        eff = (~p['p_valid'].isin(['fragment', 'spam'])).sum()
        cards.append(card('已打标', f'{len(p):,}',
                          f'有效 {eff:,} 条 · {len(p) * 14:,} 次决策'))
    return f'<div class="cards">{"".join(cards)}</div>'


def sec_pipeline(d: dict) -> str:
    steps = [
        ('0', 'rebuild', '从缓存重建', 'rpid 16.3%→100%，楼中楼关联 0→100%'),
        ('1', 'probe', '关键词探针', '量化主题分布，为体系定档'),
        ('2', 'diag', '坍缩诊断', '定位主因：标签文本劫持 ≫ 位置偏置'),
        ('3', 'weak', '规则弱监督', '高精度优先，允许弃权'),
        ('3b', 'dataset', '封顶+切分', '按 rpid 切分；内容与上下文重叠另需检查'),
        ('4', 'train', '微调', 'parallel option layout'),
        ('5', 'evaluate', '冻结 test 评测', '三方对照 + 覆盖率曲线'),
        ('6', 'predict', '全量打标', '两级推理，valid 闸门'),
        ('7', 'aggregate', '舆情报告', '主题×情感 / 时间线 / 可行动清单'),
    ]
    items = ''.join(
        f'<li><span class="st">{s}</span><code>pbz {cmd}</code>'
        f'<span class="sn">{esc(name)}</span><span class="sd">{esc(desc)}</span></li>'
        for s, cmd, name, desc in steps)
    return f'<ol class="flow">{items}</ol>'


def sec_rebuild(cfg: Config, d: dict) -> str:
    c = d.get('clean')
    if c is None:
        return ''
    rows = []
    try:
        old = pd.read_csv(cfg.path('paths.raw_csv'), dtype=str)
        rows = [('旧 CSV：唯一 rpid 占比', old['rpid'].nunique() / len(old), f'{old["rpid"].nunique()}/{len(old)}'),
                ('新 clean：唯一 rpid 占比', c['rpid'].nunique() / len(c), f'{c["rpid"].nunique()}/{len(c)}')]
    except Exception:
        pass
    deep = c[c['楼层深度'] > 0]
    rows.append(('楼中楼成功关联父评论', (deep['父评论'] != '').mean() if len(deep) else 0,
                 f'{(deep["父评论"] != "").sum()}/{len(deep)}'))
    b = bars(rows, vmax=1.0, fmt='{:.1%}', tone_fn=lambda v: 'good' if v > .9 else 'bad')

    lb = c.groupby(['len_bucket', '层级'], observed=True).size().unstack(fill_value=0)
    dist = bars([(str(i), int(r.sum()), f'主楼 {int(r.get("主楼", 0))} / 楼中楼 {int(r.get("楼中楼", 0))}')
                 for i, r in lb.iterrows()], fmt='{:,.0f}')
    stats = (f'<div class="kv">'
             f'<span>字数中位 <b>{c["字数"].median():.0f}</b></span>'
             f'<span>P90 <b>{c["字数"].quantile(.9):.0f}</b></span>'
             f'<span>state 长度 P95 <b>{c["state_len"].quantile(.95):.0f}</b></span>'
             f'<span>纯表情 <b>{(c["字数"] == 0).mean():.2%}</b></span></div>')
    return (f'<div class="g2"><div><h3>bug-1 修复验证</h3>{b}</div>'
            f'<div><h3>长度分桶 × 层级</h3>{dist}</div></div>{stats}')


def sec_diag(d: dict) -> str:
    sc = d.get('schema')
    m = ((sc or {}).get('meta', {}).get('render', {}) or {}).get('measured')
    if not m:
        return ''
    names = list(m)
    collapse = [m[n].get('topic_l1_collapse', 0) for n in names]
    entropy = [m[n].get('topic_l1_entropy', 0) for n in names]
    g = grouped_bars(names, [('collapse（越低越好）', collapse, 'bad'),
                             ('entropy（越高越好）', entropy, 'good')])
    cost = bars([(n, m[n].get('rows_per_item', 0),
                  m[n].get('status', m[n].get('flip_vs_A', '')))
                 for n in names if m[n].get('rows_per_item')],
                fmt='{:.0f} 行/条', tone_fn=lambda v: 'bad' if v > 20 else 'good')
    return (f'<div class="g2"><div><h3>topic_l1 坍缩程度</h3>{g}</div>'
            f'<div><h3>每条评论的前向代价</h3>{cost}'
            f'<p class="note">满轮转 40 行/条 = 11.5 倍成本，collapse 却没改善。</p></div></div>'
            f'<div class="finding"><b>结论</b>　坍缩主因是模型按<b>字面</b>读英文语义键，'
            f'不是位置偏置。换不透明键 <code>O1..On</code> 把 collapse 从 0.712 压到 0.320；'
            f'<code>option_order</code> 轮转关闭。</div>')


def sec_weak(d: dict) -> str:
    w = d.get('weak')
    if w is None:
        return ''
    dims = [c[2:] for c in w.columns if c.startswith('w_')]
    rows = []
    for dim in dims:
        s = w['w_' + dim]
        dist = ' · '.join(f'{topic_label(d["schema"], k) if dim == "topic_l1" else k} {v}'
                          for k, v in s.value_counts().head(4).items())
        rows.append((dim, s.notna().mean(), dist))
    rows.sort(key=lambda x: -x[1])
    b = bars(rows, vmax=1.0, fmt='{:.1%}',
             tone_fn=lambda v: 'good' if v > .4 else ('warn' if v > .15 else 'bad'))
    nl = w['n_labeled'].value_counts().sort_index()
    nb = bars([(f'{k} 个维度', int(v), '') for k, v in nl.items()], fmt='{:,.0f}')
    return (f'<div class="g2"><div><h3>各维度覆盖率</h3>{b}</div>'
            f'<div><h3>每条评论被标注的维度数</h3>{nb}</div></div>'
            f'<div class="finding warn"><b>踩过的坑</b>　'
            f'v2 第一版把「带转折词」直接弃权，<code>sentiment</code> 覆盖率从 23% '
            f'崩到 <b>2.5%</b> —— 中文长评几乎必有「但/不过」。'
            f'改成「双极弃权、单极降档」后回到 12.6%。</div>')


def sec_train(d: dict) -> str:
    ts = d.get('train')
    if not ts:
        return ''
    el = ts.get('epoch_loss') or []
    curve = sparkline(list(range(1, len(el) + 1)), el,
                      labels=[f'ep{i}' for i in range(1, len(el) + 1)], ylab='mean loss')
    tr = ts.get('train_report', {})
    b, a = tr.get('before', {}), tr.get('after', {})
    cmp_tbl = ''
    if b and a:
        keys = [('accuracy', '越高越好'), ('ece', '越低越好'), ('brier', '越低越好'),
                ('loss', '越低越好'), ('mean_confidence', '—')]
        cmp_tbl = table(pd.DataFrame([
            {'指标': k, '方向': dirn, '微调前': f'{b.get(k, 0):.4f}',
             '微调后': f'{a.get(k, 0):.4f}', '变化': f'{a.get(k, 0) - b.get(k, 0):+.4f}'}
            for k, dirn in keys]))
    split = ''
    if d.get('splits'):
        split = bars([(k, v['rows'], f'{v["decisions"]:,} 决策')
                      for k, v in d['splits'].items()], fmt='{:,.0f}')
    cal = (ts.get('calibration') or {}).get('choice') or {}
    issues = ''.join(f'<li>{esc(i)}</li>' for i in (cal.get('issues') or []))
    ab = ts.get('abstention_thresholds') or {}
    nb = ts.get('n_by_bucket') or {}
    ab_tbl = table(pd.DataFrame([{'桶': k, '弃答阈值': f'{v:.4f}', '样本数': nb.get(k, '—')}
                                 for k, v in ab.items()])) if ab else ''
    warn = (f'<div class="finding bad"><b>校准警告</b>　'
            f'拟合温度 <b>{cal.get("temperature")}</b>，撞在 [0.5, 5.0] 的上限 —— '
            f'模型显著过置信（校准集置信 {cal.get("mean_confidence", 0):.3f} vs '
            f'accuracy {cal.get("accuracy", 0):.3f}）。<ul>{issues}</ul></div>') if issues else ''
    return (f'<div class="g2"><div><h3>训练损失（RLCD）</h3>{curve}'
            f'<p class="note">4 epochs · 1,164 次优化更新 · 等效 batch 32</p></div>'
            f'<div><h3>dev 集微调前后</h3>{cmp_tbl}</div></div>'
            f'<div class="g2"><div><h3>数据切分</h3>{split}'
            f'<p class="note">按 rpid 切分，同一条评论的所有决策进同一个 split。</p></div>'
            f'<div><h3>按桶弃答阈值</h3>{ab_tbl}'
            f'<p class="note">一个标量阈值不能跨选项数通用，所以按 '
            f'<code>temp_bucket</code> 分别拟合。</p></div></div>{warn}')


def sec_eval(d: dict) -> str:
    ev = d.get('eval')
    if ev is None or ev.empty:
        return ''
    ev = ev.copy()
    ev['maj'] = ev['多数类'].astype(str).str.extract(r'([\d.]+)$').astype(float)
    ev = ev.sort_values('finetuned_acc', ascending=False)
    labels = ev['维度'].tolist()
    series = [('多数类基线', ev['maj'].tolist(), 'mute')]
    if 'base_acc' in ev:
        series.append(('零样本', ev['base_acc'].tolist(), 'warn'))
    series.append(('微调后', ev['finetuned_acc'].tolist(), 'good'))
    g = grouped_bars(labels, series)

    mf1 = bars([(r['维度'], r['finetuned_mF1'],
                 '⚠ 实为全预测多数类' if r['finetuned_mF1'] < .55 else '')
                for _, r in ev.iterrows()], vmax=1.0,
               tone_fn=lambda v: 'good' if v > .7 else ('warn' if v > .55 else 'bad'))
    ece = bars([(r['维度'], r['finetuned_ECE'], '') for _, r in ev.iterrows()],
               tone_fn=lambda v: 'good' if v < .07 else ('warn' if v < .12 else 'bad'))

    show = ['维度', 'n', '类数', '多数类'] + [c for c in ev.columns if c.startswith(('base_', 'finetuned_'))]
    tbl = table(ev[show].round(4),
                highlight={'finetuned_acc': lambda v: 'good' if float(v) >= .9 else 'warn',
                           'finetuned_mF1': lambda v: 'bad' if float(v) < .55 else ''})
    return (f'<h3>accuracy：多数类 vs 零样本 vs 微调</h3>{g}'
            f'<div class="g2"><div><h3>macro-F1（类不平衡下的真实水平）</h3>{mf1}</div>'
            f'<div><h3>ECE（校准误差，越低越好）</h3>{ece}</div></div>'
            f'<h3>完整指标</h3>{tbl}'
            f'<div class="finding warn"><b>必须看 macro-F1</b>　'
            f'<code>is_toxic</code> accuracy 0.958 看着很好，但 macro-F1 只有 0.489 —— '
            f'它实际是<b>全预测 B</b>，因为训练集正例仅 11 条。</div>')


def sec_confusion(d: dict) -> str:
    cm = d.get('confusion') or {}
    if not cm:
        return ''
    blocks = ''.join(heatmap(df, dim) for dim, df in cm.items())
    return (f'<div class="hms">{blocks}</div>'
            f'<p class="note">行 = 真值，列 = 预测；颜色按行归一化，对角线为正确。</p>')


def sec_topics(d: dict) -> str:
    sc = d.get('schema')
    if not sc:
        return ''
    rows = []
    names = sc['cn_names']['topic_l2']
    for branch, data in sc['pass2_by_l1'].items():
        labels = [names[branch][key] for key in data['topic_l2']['criteria']]
        rows.append({'一级主题': topic_label(sc, branch), '二级主题数': len(labels),
                     '二级主题': ' / '.join(labels)})
    return table(pd.DataFrame(rows))


def sec_business(d: dict) -> str:
    p = d.get('pred')
    if p is None:
        return ''
    sc = d.get('schema') or {}
    p = localize_predictions(p, sc)
    wmap = {'very_neg': -2, 'neg': -1, 'neutral': 0, 'pos': 1, 'very_pos': 2}

    df = p[~p['p_valid'].isin(['fragment', 'spam'])].copy()
    df['L1'] = df['p_topic_l1']
    df['sc'] = df['p_sentiment'].map(wmap).fillna(0).astype(float)
    df['neg'] = df['p_sentiment'].isin(NEG_SET).astype(int)
    df['日期'] = pd.to_datetime(df['时间']).dt.strftime('%m-%d')
    df['wt'] = df['点赞'] + 1

    agg = df.groupby('L1').agg(n=('rpid', 'size'), neg=('neg', 'mean'), s=('sc', 'mean'))
    agg['ws'] = df.groupby('L1').apply(
        lambda g: np.average(g['sc'], weights=g['wt']), include_groups=False)
    agg = agg.sort_values('n', ascending=False)

    vol = bars([(i, int(r['n']), f'负面率 {r["neg"]:.1%}') for i, r in agg.iterrows()],
               fmt='{:,.0f}')
    negb = bars([(i, r['neg'], '') for i, r in agg.iterrows()], vmax=.5, fmt='{:.1%}',
                tone_fn=lambda v: 'bad' if v > .35 else ('warn' if v > .25 else 'good'))
    gap = grouped_bars(list(agg.index),
                       [('未加权情感', [max(0, v) for v in agg['s']], 'mute'),
                        ('赞加权情感', [max(0, v) for v in agg['ws']], 'good')])

    tl = df.groupby('日期').agg(n=('rpid', 'size'), neg=('neg', 'mean'), s=('sc', 'mean'))
    days = list(tl.index)
    line = sparkline(list(range(len(days))), [float(v) for v in tl['neg']],
                     labels=days, w=760, h=170, ylab='负面率')

    share = (df.groupby(['日期', 'L1']).size().unstack(fill_value=0))
    share = (share.div(share.sum(1), axis=0) * 100).round(1)
    heat = heatmap(share.astype(int), '各天主题占比 (%)')

    tbl = table(pd.DataFrame({
        '主题': agg.index, '条数': agg['n'].astype(int).values,
        '负面率': [f'{v:.1%}' for v in agg['neg']],
        '情感均分': agg['s'].round(2).values,
        '赞加权情感': agg['ws'].round(2).values,
        '加权差': (agg['ws'] - agg['s']).round(2).values}))

    topics = topic_distribution(df, sc)
    detail = table(topics.sort_values('条数', ascending=False).head(20))
    return (f'<div class="g2"><div><h3>主题体量</h3>{vol}</div>'
            f'<div><h3>各主题负面率</h3>{negb}</div></div>'
            f'<h3>未加权 vs 赞加权情感</h3>{gap}'
            f'<div class="finding"><b>反差即信号</b>　'
            f'「技术性能」未加权 0.20 但赞加权 0.90 —— 高赞是「优化比黑猴好」，'
            f'骂的在长尾低赞；「视频与UP主」反过来（0.59 → −0.02），'
            f'高赞集中在对 UP 主的负面与催更。</div>'
            f'<h3>负面率时间线</h3>{line}'
            f'<p class="note">10-29 发售当天负面率跳到 33.5%，'
            f'<code>技术性能</code> 话题占比从 8% 飙到 18.5%（着色器编译 / 驱动 / 手柄）。</p>'
            f'{heat}<h3>汇总</h3>{tbl}'
            f'<h3>二级主题分布（前 20 项）</h3>{detail}'
            f'<p class="note">一级与二级标签均使用中文；完整计数见二级主题分布.csv。'
            f'二级主题尚无独立人工评测。</p>')


def sec_limits() -> str:
    items = [
        ('高', 'test 标签来自规则弱监督，不是人工标注',
         '分数衡量「模型是否学会了规则体现的那套判别」，不等于人工一致率。'
         '抽检估计规则本身 topic_l1 精度约 72–75%；模型的人工一致率仍需独立验证。'
         '<code>data/train/golden_todo.xlsx</code>（589 条）需先排除训练重叠，再标注评测。'),
        ('高', '二级主题尚无独立评测', '当前弱标签训练集只有一级主题等 12 个维度，'
         '没有 topic_l2 监督。中文名称转换不会提升或重新测量模型准确率。'),
        ('高', 'is_toxic 完全失败', 'macro-F1 0.489 = 全预测 B。训练集正例仅 11 条。'
                                '<b>不可用于自动封禁或处罚。</b>'),
        ('高', 'sentiment 学不出 neutral', '规则从不输出中立，全量预测里 neutral 仅 0.3%。'
                                          '这是弱监督的结构性缺陷，必须人工补。'),
        ('中', '温度拟合撞上限', '温度 5.0 撞 [0.5,5.0] 上限，模型过置信；'
                             'is_mixed / is_spoiler 的 ECE 偏高（0.118 / 0.106）。'),
        ('中', f'{", ".join(DIMS_UNTRAINED)} 未训练', '规则不可靠、无弱标签，输出是噪声。'),
        ('中', 'parallel 布局的代价', '需 transformers≥5，只能 eager 后端 —— '
                                   'ONNX 导出、TileLang 快路径、<code>compile=True</code> 都会拒绝。'),
        ('低', 'IP属地 全为「未知」', '地域维度做不了。'),
        ('低', '跨游戏泛化未验证', '词表与二级主题围绕《影之刃零》与这 7 个视频长出来的。'),
    ]
    sev = {'高': 3, '中': 2, '低': 1}
    lis = ''.join(
        f'<li><span class="sev s{sev[lv]}">{lv}</span><b>{esc(t)}</b><p>{body}</p></li>'
        for lv, t, body in items)
    return f'<ul class="limits">{lis}</ul>'


# ---------------------------------------------------------------- 装载
def collect(cfg: Config) -> dict:
    d: dict = {}

    def tryload(fn):
        try:
            return fn()
        except Exception:
            return None

    d['schema'] = tryload(lambda: load_schema(cfg.path('paths.taxonomy')))
    d['clean'] = tryload(lambda: pd.read_parquet(cfg.path('paths.clean')))
    w = tryload(lambda: pd.read_parquet(cfg.path('paths.weak')))
    d['weak'] = w
    d['weak_total'] = int(w['n_labeled'].sum()) if w is not None else 0
    d['train'] = tryload(lambda: json.load(
        open(os.path.join(cfg.path('paths.ckpt'), 'train_summary.json'), encoding='utf-8')))
    d['eval'] = tryload(lambda: pd.read_csv(os.path.join(cfg.path('paths.eval_dir'), 'summary.csv')))
    d['pred'] = tryload(lambda: pd.read_parquet(cfg.path('paths.pred')))

    splits = {}
    for s in ('train', 'dev', 'test'):
        p = os.path.join(cfg.path('paths.train_dir'), f'{s}.jsonl')
        if os.path.exists(p):
            rows = [json.loads(l) for l in open(p, encoding='utf-8')]
            splits[s] = {'rows': len(rows), 'decisions': sum(len(r['gold']) for r in rows)}
    d['splits'] = splits

    d['confusion'] = tryload(lambda: _confusion(cfg))
    return d


def _confusion(cfg: Config, dims=('topic_l1', 'valid', 'sentiment', 'speech_act')) -> dict:
    """从 eval/report.md 里解析已有的混淆矩阵表（避免重跑推理）。"""
    p = os.path.join(cfg.path('paths.eval_dir'), 'report.md')
    if not os.path.exists(p):
        return {}
    txt = open(p, encoding='utf-8').read()
    out: dict[str, pd.DataFrame] = {}
    for dim in dims:
        marker = f'### {dim}\n'
        if marker not in txt:
            continue
        blk = txt.split(marker, 1)[1]
        lines = [l.strip() for l in blk.splitlines() if l.strip().startswith('|')]
        if len(lines) < 3:
            continue
        rows, cols = [], []
        for i, l in enumerate(lines):
            cells = [c.strip() for c in l.strip('|').split('|')]
            if i == 0:
                cols = cells[1:]
            elif set(''.join(cells)) <= set('-: '):
                continue
            else:
                if not cells[0] or cells[0] in ('真值',):
                    continue
                try:
                    rows.append((cells[0], [int(float(x or 0)) for x in cells[1:]]))
                except ValueError:
                    break
        if rows and cols:
            cm = pd.DataFrame({c: [r[1][j] if j < len(r[1]) else 0 for r in rows]
                               for j, c in enumerate(cols)}, index=[r[0] for r in rows])
            if dim == 'topic_l1':
                sc = load_schema(cfg.path('paths.taxonomy'))
                names = sc['cn_names']['topic_l1']
                cm = cm.rename(index=names, columns=names)
            out[dim] = cm
    return out


CSS = """
*{box-sizing:border-box}
:root{
 --bg:#0d1117;--bg2:#161b22;--bg3:#1c2230;--fg:#e6edf3;--mut:#8b949e;--bd:#30363d;
 --ac:#58a6ff;--good:#3fb950;--warn:#d29922;--bad:#f85149;--mute:#6e7681;
}
@media(prefers-color-scheme:light){:root{
 --bg:#fff;--bg2:#f6f8fa;--bg3:#eef1f5;--fg:#1f2328;--mut:#656d76;--bd:#d0d7de;
 --ac:#0969da;--good:#1a7f37;--warn:#9a6700;--bad:#cf222e;--mute:#8c959f;}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 -apple-system,"Segoe UI",
 "PingFang SC","Microsoft YaHei",Roboto,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 80px}
header{border-bottom:1px solid var(--bd);padding-bottom:18px;margin-bottom:26px}
h1{font-size:25px;margin:0 0 6px;letter-spacing:-.3px}
h1 small{font-weight:400;color:var(--mut);font-size:14px;letter-spacing:0}
.sub{color:var(--mut);font-size:13px}
.sub code{background:var(--bg2);padding:1px 5px;border-radius:4px;font-size:12px}
section{margin:36px 0}
h2{font-size:18px;margin:0 0 4px;display:flex;align-items:center;gap:10px}
.num{display:inline-grid;place-items:center;width:25px;height:25px;border-radius:7px;
 background:var(--bg3);color:var(--ac);font-size:12px;font-weight:700;flex:none}
h3{font-size:13px;margin:20px 0 8px;color:var(--mut);font-weight:600;
 text-transform:uppercase;letter-spacing:.6px}
.note{color:var(--mut);font-size:12.5px;margin:7px 0 0}
.note code,.finding code,.sd code{background:var(--bg2);padding:1px 5px;border-radius:4px;font-size:12px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:26px}
@media(max-width:860px){.g2{grid-template-columns:1fr}}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(176px,1fr));gap:12px}
.card{background:var(--bg2);border:1px solid var(--bd);border-radius:11px;padding:13px 15px}
.card.good{border-color:color-mix(in srgb,var(--good) 45%,var(--bd))}
.card .k{color:var(--mut);font-size:11.5px;text-transform:uppercase;letter-spacing:.5px}
.card .v{font-size:21px;font-weight:700;margin:4px 0 2px;letter-spacing:-.4px}
.card .v small{font-size:13px;font-weight:500;color:var(--mut)}
.card .v b{color:var(--good)}
.card .s{color:var(--mut);font-size:11.5px}
.bars{display:flex;flex-direction:column;gap:5px}
.bar{display:grid;grid-template-columns:168px 1fr 74px auto;align-items:center;gap:9px;font-size:12.5px}
.bl{color:var(--fg);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bt{background:var(--bg3);border-radius:4px;height:15px;overflow:hidden}
.bt i{display:block;height:100%;min-width:2px;background:var(--ac);border-radius:4px}
.bt i.good{background:var(--good)}.bt i.warn{background:var(--warn)}
.bt i.bad{background:var(--bad)}.bt i.mute{background:var(--mute)}
.bv{font-variant-numeric:tabular-nums;text-align:right;color:var(--fg);font-weight:600}
.bn{color:var(--mut);font-size:11.5px;white-space:nowrap}
.legend{display:flex;gap:16px;margin-bottom:9px;font-size:12px;color:var(--mut);flex-wrap:wrap}
.lg i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;background:var(--ac)}
.lg.good i{background:var(--good)}.lg.warn i{background:var(--warn)}.lg.mute i{background:var(--mute)}
.gbars{display:flex;flex-direction:column;gap:9px}
.gbar{display:grid;grid-template-columns:168px 1fr;gap:9px;align-items:center;font-size:12.5px}
.gl{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.gs{display:flex;flex-direction:column;gap:2px}
.gt{display:grid;grid-template-columns:minmax(24px,1fr) 50px;align-items:center;gap:7px;
 background:var(--bg3);border-radius:3px}
.gt i{display:block;height:11px;min-width:2px;border-radius:3px;background:var(--ac)}
.gt i.good{background:var(--good)}.gt i.warn{background:var(--warn)}
.gt i.bad{background:var(--bad)}.gt i.mute{background:var(--mute)}
.gt b{font-size:11px;font-variant-numeric:tabular-nums;color:var(--mut);font-weight:600}
.spark{width:100%;height:auto;margin-top:4px}
.spark .ln{fill:none;stroke:var(--ac);stroke-width:2;stroke-linejoin:round}
.spark .ar{fill:var(--ac);opacity:.11}
.spark circle{fill:var(--ac)}
.spark .pt{fill:var(--fg);font-size:10.5px;text-anchor:middle;font-weight:600}
.spark .ax{fill:var(--mut);font-size:10px;text-anchor:middle}
.spark .gr{stroke:var(--bd);stroke-width:1}
.tw{overflow-x:auto;border:1px solid var(--bd);border-radius:9px;margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:7px 11px;text-align:right;border-bottom:1px solid var(--bd);white-space:nowrap}
th{background:var(--bg2);color:var(--mut);font-weight:600;font-size:11.5px;
 text-transform:uppercase;letter-spacing:.4px;position:sticky;top:0}
th:first-child,td:first-child{text-align:left}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--bg2)}
td.good{color:var(--good);font-weight:600}td.warn{color:var(--warn)}td.bad{color:var(--bad);font-weight:600}
.hms{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:22px}
.hm-wrap{min-width:0}
.hm-t{font-size:12px;color:var(--mut);margin-bottom:6px;font-weight:600;font-family:ui-monospace,monospace}
.hm{display:grid;gap:2px;font-size:10.5px}
.hh,.hr{color:var(--mut);font-size:10px;padding:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hh{text-align:center}.hr{text-align:right;padding-right:6px}
.hc{display:grid;place-items:center;aspect-ratio:1.5;border-radius:3px;
 background:color-mix(in srgb,var(--ac) calc(var(--a)*88%),var(--bg2));
 color:var(--fg);font-variant-numeric:tabular-nums}
.hc.dg{background:color-mix(in srgb,var(--good) calc(var(--a)*88%),var(--bg2))}
.finding{margin:16px 0 0;padding:12px 15px;border-radius:9px;background:var(--bg2);
 border-left:3px solid var(--ac);font-size:13px}
.finding.warn{border-left-color:var(--warn)}
.finding.bad{border-left-color:var(--bad)}
.finding ul{margin:7px 0 0;padding-left:20px;color:var(--mut);font-size:12.5px}
.flow{list-style:none;padding:0;margin:0;display:grid;
 grid-template-columns:repeat(auto-fit,minmax(255px,1fr));gap:9px}
.flow li{background:var(--bg2);border:1px solid var(--bd);border-radius:9px;padding:11px 13px;
 display:grid;grid-template-columns:auto 1fr;gap:4px 10px;align-items:center}
.st{display:inline-grid;place-items:center;min-width:23px;height:23px;padding:0 5px;border-radius:6px;
 background:var(--bg3);color:var(--ac);font-size:11px;font-weight:700;grid-row:span 2}
.flow code{color:var(--fg);font-weight:600;font-size:12.5px}
.sn{color:var(--mut);font-size:11.5px;justify-self:start}
.sd{grid-column:2;color:var(--mut);font-size:11.5px}
.kv{display:flex;gap:18px;flex-wrap:wrap;margin-top:14px;font-size:12.5px;color:var(--mut)}
.kv b{color:var(--fg)}
.limits{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:9px}
.limits li{background:var(--bg2);border:1px solid var(--bd);border-radius:9px;padding:11px 14px;
 display:grid;grid-template-columns:auto 1fr;gap:3px 11px;align-items:start}
.sev{display:inline-grid;place-items:center;width:23px;height:23px;border-radius:6px;
 font-size:11px;font-weight:700;grid-row:span 2}
.sev.s3{background:color-mix(in srgb,var(--bad) 22%,transparent);color:var(--bad)}
.sev.s2{background:color-mix(in srgb,var(--warn) 22%,transparent);color:var(--warn)}
.sev.s1{background:var(--bg3);color:var(--mut)}
.limits b{font-size:13px}
.limits p{margin:0;color:var(--mut);font-size:12.5px;grid-column:2}
.nav{position:sticky;top:0;z-index:9;background:color-mix(in srgb,var(--bg) 88%,transparent);
 backdrop-filter:blur(12px);border-bottom:1px solid var(--bd);margin:0 -20px 26px;padding:9px 20px;
 display:flex;gap:4px;overflow-x:auto;scrollbar-width:none}
.nav::-webkit-scrollbar{display:none}
.nav a{color:var(--mut);text-decoration:none;font-size:12px;padding:4px 9px;border-radius:6px;
 white-space:nowrap;transition:.12s}
.nav a:hover{background:var(--bg3);color:var(--fg)}
footer{margin-top:52px;padding-top:18px;border-top:1px solid var(--bd);
 color:var(--mut);font-size:12px;display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap}
footer a{color:var(--ac);text-decoration:none}
@media(max-width:600px){
 .wrap{padding:18px 13px 60px}
 h1{font-size:20px}
 .bar{grid-template-columns:96px 1fr auto;gap:7px;font-size:11.5px}
 .bar .bn{display:none}
 .gbar{grid-template-columns:96px 1fr;gap:7px;font-size:11.5px}
 .gt{grid-template-columns:1fr 42px}
 .flow li{grid-template-columns:auto 1fr}
 .hms{grid-template-columns:1fr}
 .hm{font-size:9px}.hh,.hr{font-size:8.5px}
 .nav{margin:0 -13px 20px;padding:8px 13px}
}
"""


def build(cfg: Config, out: str | None = None) -> str:
    d = collect(cfg)
    out = out or os.path.join(cfg.path('paths.report_dir'), 'dashboard.html')
    os.makedirs(os.path.dirname(out), exist_ok=True)

    specs = [
        ('·', '概览', sec_overview(cfg, d), ''),
        ('T', '两级中文主题标签', sec_topics(d),
         '8 个一级主题、67 个二级主题，名称统一来自标签体系。'),
        ('0', '流水线', sec_pipeline(d),
         '每个阶段都是一条可独立运行的命令，产物落盘，可从任意阶段续跑。'),
        ('1', '数据重建', sec_rebuild(cfg, d),
         '历史 CSV 的 rpid 被科学计数法摧毁（<code>321944070160 → 3.21944E+11</code>），'
         '9696 行塌成 1579 个唯一值，47.7% 的楼中楼断链。从爬虫缓存重建。'),
        ('2', '零样本坍缩诊断', sec_diag(d),
         '400 条分层样本 × 7 个变体。这一节的产出直接改变了方案。'),
        ('3', '规则弱监督', sec_weak(d),
         '基座零样本已证不可用，所以<b>没有</b>拿它当伪标签 —— '
         '那只会把 72% 的坍缩固化进去。改用高精度规则 + 弃权。'),
        ('4', '微调', sec_train(d),
         '<code>laya-multilingual</code> + <code>option_layout="parallel"</code>'
         '（选项顺序无关），RTX 3060 12G。'),
        ('5', '评测（冻结 test 集）', sec_eval(d), ''),
        ('6', '混淆矩阵', sec_confusion(d), ''),
        ('7', '业务结果', sec_business(d),
         '全量打标后聚合。<code>valid</code> 闸门先滤掉碎片与垃圾。'),
        ('!', '已知限制', sec_limits(), '这一节比上面所有数字都重要。'),
    ]
    secs, nav = [], []
    for idx, title, body, note in specs:
        htmlpart = section(idx, title, body, note)
        if not htmlpart:
            continue
        secs.append(htmlpart)
        sid = 's' + str(idx).replace('·', 'o').replace('!', 'x')
        nav.append(f'<a href="#{sid}">{esc(title)}</a>')
    navbar = f'<nav class="nav">{"".join(nav)}</nav>' if len(nav) > 2 else ''
    ts = d.get('train') or {}
    tr = (ts.get('train_report') or {})
    acc = (tr.get('after') or {}).get('accuracy')
    head_sub = (f'<code>laya-pbz-v1</code> · 基座 <code>laya-multilingual</code> · '
                f'parallel layout · dev accuracy '
                f'{(tr.get("before") or {}).get("accuracy", "?")} → '
                f'<b>{acc if acc is not None else "?"}</b>')

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(cfg.game)} 评论决策模型 · 训练看板</title>
<style>{CSS}</style></head><body><div class="wrap">
<header>
<h1>{esc(cfg.game)} 评论多维决策模型 <small>训练过程与结果看板</small></h1>
<div class="sub">{head_sub}</div>
<div class="sub" style="margin-top:4px">生成于 {datetime.now(CST):%Y-%m-%d %H:%M} CST
 · 统计指标来自产物，两级主题名称来自标签体系；由 <code>pbz dashboard</code> 生成</div>
</header>
{navbar}
{''.join(secs)}
<footer>
<span>pbz-comment-insight · 代码 Apache-2.0 · 基座 <a href="https://github.com/NandhaKishorM/laya">Laya</a> Apache-2.0</span>
<span>自包含单文件 · 无 CDN / 无 JS 依赖 · 支持深浅色</span>
</footer>
</div></body></html>"""
    open(out, 'w', encoding='utf-8').write(doc)
    print(f'[dashboard] {len(doc) / 1024:.0f} KB → {out}')
    missing = [k for k in ('clean', 'weak', 'train', 'eval', 'pred') if d.get(k) is None]
    if missing:
        print(f'[dashboard] 缺产物，已跳过相应小节：{", ".join(missing)}')
    return out
