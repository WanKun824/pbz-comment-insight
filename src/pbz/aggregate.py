"""阶段 7：聚合成业务报告。

四张核心表
  1. 主题 × 情感（谁在被夸、谁在被骂）—— 含**赞加权情感**，反映被多少人看到的情绪
  2. 时间线（发售节点前后的主题与负面率迁移）
  3. 视频维度（不同 UP 主评论区的结构差异）
  4. 可行动清单（has_issue=A，按主题分组、按赞排序）
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .config import Config
from .schema import DIMS_BIN, DIMS_UNTRAINED, load_schema
from .topics import localize_predictions, topic_distribution

NEG_SET = {'very_neg', 'neg'}


def run(cfg: Config, *, pred: str | None = None, min_conf: float | None = None) -> pd.DataFrame:
    pred = pred or cfg.path('paths.pred')
    if not os.path.exists(pred):
        raise FileNotFoundError(f'找不到预测结果：{pred}（先跑 `pbz predict`）')
    min_conf = float(cfg.get('aggregate.min_conf', 0.0) if min_conf is None else min_conf)
    drop = set(cfg.get('aggregate.drop_valid', ['fragment', 'spam']))
    weight = cfg.get('aggregate.sentiment_weight', {})
    out_dir = cfg.path('paths.report_dir')
    os.makedirs(out_dir, exist_ok=True)

    sc = load_schema(cfg.path('paths.taxonomy'))
    cn = sc.get('cn_names', {})
    cn_sent = cn.get('sentiment', {})

    df = localize_predictions(pd.read_parquet(pred), sc)
    n0 = len(df)
    df = df[~df['p_valid'].isin(drop)].copy()
    if min_conf > 0:
        df = df[df['c_topic_l1'] >= min_conf]
    print(f'[aggregate] {n0} → {len(df)} 条'
          f'（去 {"/".join(sorted(drop))}{f"、置信≥{min_conf:.2f}" if min_conf else ""}）')

    df['L1'] = df['p_topic_l1']
    df['情感'] = df['p_sentiment'].map(lambda x: cn_sent.get(x, x))
    df['情感分'] = df['p_sentiment'].map(weight).fillna(0).astype(float)
    df['负面'] = df['p_sentiment'].isin(NEG_SET).astype(int)
    df['日期'] = pd.to_datetime(df['时间']).dt.strftime('%m-%d')
    df['加权'] = df['点赞'] + 1

    L = [f'# {cfg.game} B站评论舆情报告\n',
         f'样本 {len(df)} 条（原始 {n0}，已滤除碎片与垃圾）　'
         f'时间 {df["时间"].min()[:10]} ~ {df["时间"].max()[:10]}\n']

    # ---------------- 1. 主题 × 情感
    order = [cn_sent.get(k, k) for k in ['very_neg', 'neg', 'neutral', 'pos', 'very_pos']]
    piv = pd.crosstab(df['L1'], df['情感'])
    piv = piv.reindex(columns=[c for c in order if c in piv.columns], fill_value=0)
    piv['合计'] = piv.sum(1)
    piv['负面率'] = (df.groupby('L1')['负面'].mean() * 100).round(1)
    piv['情感均分'] = df.groupby('L1')['情感分'].mean().round(2)
    piv['赞加权情感'] = df.groupby('L1').apply(
        lambda g: np.average(g['情感分'], weights=g['加权']), include_groups=False).round(2)
    piv = piv.sort_values('合计', ascending=False)
    L += ['## 1. 主题 × 情感\n', piv.to_markdown(), '',
          '> 「赞加权情感」按点赞加权，反映**被多少人看到**的情绪。'
          '与未加权值差距越大，说明该主题的高赞评论倾向越极端。\n']
    piv.to_csv(os.path.join(out_dir, '主题情感矩阵.csv'), encoding='utf-8-sig')

    topics = topic_distribution(df, sc)
    topics.to_csv(os.path.join(out_dir, '二级主题分布.csv'), index=False, encoding='utf-8-sig')
    L += ['### 两级主题分布\n', topics.to_markdown(index=False), '',
          '> 二级主题尚无独立人工评测；这里仅汇总现有预测，不代表已验证的分类准确率。\n']

    # ---------------- 2. 时间线
    tl = df.groupby('日期').agg(条数=('rpid', 'size'), 负面率=('负面', 'mean'),
                              情感均分=('情感分', 'mean'))
    tl['负面率'] = (tl['负面率'] * 100).round(1)
    tl['情感均分'] = tl['情感均分'].round(2)
    cnt = df.groupby(['日期', 'L1']).size().unstack(fill_value=0)
    share = (cnt.div(cnt.sum(1), axis=0) * 100).round(1)
    L += ['## 2. 时间线\n', tl.to_markdown(), '', '### 各天主题占比 (%)\n', share.to_markdown(), '']
    tl.join(share, rsuffix='_占比').to_csv(os.path.join(out_dir, '时间线.csv'), encoding='utf-8-sig')

    # ---------------- 3. 视频维度
    vd = df.groupby('视频标题').agg(
        条数=('rpid', 'size'), 负面率=('负面', 'mean'), 情感均分=('情感分', 'mean'),
        主主题=('L1', lambda s: s.value_counts().index[0]),
        技术问题率=('p_has_issue', lambda s: (s == 'A').mean()),
        购买意向率=('p_purchase_intent', lambda s: (s == 'A').mean()),
        提及他游率=('p_mentions_other_game', lambda s: (s == 'A').mean()))
    for c in ['负面率', '技术问题率', '购买意向率', '提及他游率']:
        vd[c] = (vd[c] * 100).round(1)
    vd['情感均分'] = vd['情感均分'].round(2)
    vd.index = [str(i)[:30] for i in vd.index]
    L += ['## 3. 视频维度\n', vd.sort_values('条数', ascending=False).to_markdown(), '']
    vd.to_csv(os.path.join(out_dir, '视频维度.csv'), encoding='utf-8-sig')

    # ---------------- 4. 可行动清单
    act = df[df['p_has_issue'] == 'A'].sort_values(['c_has_issue', '点赞'], ascending=False)
    L += [f'## 4. 可行动清单（has_issue=A，共 {len(act)} 条）\n']
    for l1, g in act.groupby('L1'):
        L += [f'\n### {l1}（{len(g)} 条，总赞 {g["点赞"].sum():,}）\n']
        for _, r in g.head(8).iterrows():
            L.append(f'- **赞{r["点赞"]}** `{r.get("p_topic_l2", "")}` {str(r["内容"])[:110]}')
    keep = [c for c in ['rpid', 'L1', 'p_topic_l2', 'p_sentiment', 'c_has_issue', '点赞', '内容']
            if c in act.columns]
    act[keep].rename(columns={'L1': '一级主题', 'p_topic_l2': '二级主题'}).to_csv(
        os.path.join(out_dir, '可行动清单.csv'), index=False, encoding='utf-8-sig')

    # ---------------- 5. 二分维度总览
    rows = []
    for b in DIMS_BIN + DIMS_UNTRAINED:
        c = f'p_{b}'
        if c in df:
            rows.append({'维度': b, 'A占比%': round((df[c] == 'A').mean() * 100, 1),
                         'A条数': int((df[c] == 'A').sum()),
                         '平均置信': round(df[f'c_{b}'].mean(), 3),
                         '备注': '未训练，不可用' if b in DIMS_UNTRAINED else ''})
    L += ['\n## 5. 二分维度总览\n', pd.DataFrame(rows).to_markdown(index=False), '']

    p = os.path.join(out_dir, '舆情报告.md')
    open(p, 'w', encoding='utf-8').write('\n'.join(L))
    print(piv.to_string())
    print(f'\n[aggregate] → {p} + 5 个 csv')
    return piv
