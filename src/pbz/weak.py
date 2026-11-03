"""阶段 3：规则弱监督标注（Snorkel 式，高精度优先，允许弃权）。

为什么不用零样本预测当标签
--------------------------
`pbz diag` 已证基座在中文多选上坍缩（topic_l1 72% 压到 creator，
sentiment 把 64–76% 判为负面，而这批数据明显偏正面）。
拿一个 72% 坍缩的输出自训练只会固化错误。

规则冲突或无规则命中时**弃权**（输出 None）。覆盖率低没关系 ——
微调只需要几千条干净样本，2.5 万条脏样本不如 5 千条干净的。
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from .config import Config
from .lexicon import (
    ARGUE,
    BUY,
    CHURN,
    CREATOR_REF,
    DEIXIS,
    DEV_REF,
    FIRST_PERSON,
    GAME_REF,
    ISSUE,
    L1_RE,
    MEME,
    NEG,
    NEG_STRONG,
    OTHER_GAME,
    POS,
    POS_STRONG,
    Q_STRICT,
    SPAM,
    SPOIL,
    SUGGEST,
    TOXIC,
    TURN,
)
from .schema import DIMS_BIN, DIMS_MULTI, DIMS_UNTRAINED


# ---------------------------------------------------------------- 单维度规则
def rule_valid(t: str, L: int, has_topic: bool):
    if SPAM(t):
        return 'spam', 'spam词'
    if L == 0:
        return 'fragment', '纯表情'
    if L <= 4 and not has_topic:
        return 'fragment', '≤4字无主题'
    if L >= 60 and has_topic:
        return 'substantive', '≥60字有主题'
    if L <= 18 and MEME(t) and not has_topic:
        return 'banter', '短+梗词'
    if L >= 30 and has_topic and (POS(t) or NEG(t) or POS_STRONG(t) or NEG_STRONG(t)):
        return 'substantive', '≥30字有主题有评价'
    return None, ''


def rule_topic_l1(t: str, L: int, min_len: int = 15):
    """要求「独占命中」或「压倒性命中」，否则弃权。短句歧义过大，先过长度门。"""
    if L < min_len:
        return None, ''
    hits = {k: len(set(r.findall(t))) for k, r in L1_RE.items()}
    hits = {k: v for k, v in hits.items() if v}
    if not hits:
        return None, ''
    srt = sorted(hits.items(), key=lambda x: -x[1])
    top, n = srt[0]
    if len(srt) == 1:
        return top, f'{top}独占x{n}'
    if n >= 3 and n >= srt[1][1] * 3:
        return top, f'{top}压倒x{n}vs{srt[1][1]}'
    return None, ''


def rule_sentiment(t: str, L: int, min_len: int = 12, meme_cutoff: int = 40):
    """双极 → 弃权；单极 + 转折词 → 降一档强度。

    踩过的坑：最初把「带转折词」直接弃权，覆盖率从 23% 崩到 **2.5%**
    —— 中文长评几乎必有「但/不过」。
    """
    if L < min_len or Q_STRICT(t):
        return None, ''
    if MEME(t) and L < meme_cutoff:                  # 梗句反讽风险
        return None, ''
    ps, p, ns, n = POS_STRONG(t), POS(t), NEG_STRONG(t), NEG(t)
    if (ps or p) and (ns or n):
        return None, ''
    turn = bool(TURN(t))
    if ps:
        return ('pos', f'强正x{ps}+转折') if turn else ('very_pos', f'强正x{ps}')
    if p:
        return 'pos', f'正面x{p}' + ('+转折' if turn else '')
    if ns:
        return ('neg', f'强负x{ns}+转折') if turn else ('very_neg', f'强负x{ns}')
    if n:
        return 'neg', f'负面x{n}' + ('+转折' if turn else '')
    return None, ''


def rule_mixed(t: str, L: int):
    if L < 25:
        return 'B', '过短'
    both = (POS(t) or POS_STRONG(t)) and (NEG(t) or NEG_STRONG(t))
    if both and TURN(t):
        return 'A', '褒贬+转折'
    if not both and L >= 40:
        return 'B', '单极性长评'
    return None, ''


def rule_bin(t: str, fn, name: str, neg_len: int, L: int):
    """二分维度一律两选项 choice（键 A/B）—— 中文 noul 会退化成常数 true。"""
    if fn(t):
        return 'A', name
    if L >= neg_len:
        return 'B', f'{name}未命中'
    return None, ''


def rule_speech_act(t: str, L: int, has_reply: bool):
    if Q_STRICT(t) and L <= 50:                      # 严格疑问最高优先
        return 'question', '疑问句'
    if SUGGEST(t) and not Q_STRICT(t):
        return 'suggest', '建议词'
    if MEME(t) and L <= 20:
        return 'joke', '短+梗'
    if ISSUE(t) and not Q_STRICT(t) and L >= 15 and not (POS_STRONG(t) or NEG_STRONG(t)):
        return 'report', '技术词陈述'
    if has_reply and ARGUE(t) and L >= 10:
        return 'argue', '回复+反驳词'
    if FIRST_PERSON(t) and BUY(t) and not Q_STRICT(t):
        return 'share', '第一人称经历'
    if L >= 60 and (POS_STRONG(t) or NEG_STRONG(t) or POS(t) >= 2 or NEG(t) >= 2) \
            and not Q_STRICT(t) and not ISSUE(t):
        return 'review', '长评+评价词'
    return None, ''


def rule_target(t: str, L: int, has_reply: bool, min_len: int = 8):
    if L == 0:
        return 'none', '纯表情'
    if L < min_len:
        return None, ''
    g, c, d = bool(GAME_REF(t)), bool(CREATOR_REF(t)), bool(DEV_REF(t))
    if d and not c and not g:
        return 'dev', '开发者词独占'
    if c and not g and not d:
        return 'creator', 'UP主词独占'
    if g and not c and not d:
        return 'game', '游戏词独占'
    if has_reply and re.match(r'^(你|您|老哥|兄弟|楼上|这位)', t) and not g and not c:
        return 'other_player', '回复+第二人称开头'
    if OTHER_GAME(t) and not g and not c and not d and not DEIXIS(t):
        return 'other_game', '仅提其他游戏'
    return None, ''


# ---------------------------------------------------------------- 主流程
def label_frame(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    mt = int(cfg.get('weak_label.min_len_topic', 15))
    ms = int(cfg.get('weak_label.min_len_sentiment', 12))
    mg = int(cfg.get('weak_label.min_len_target', 8))
    mc = int(cfg.get('weak_label.meme_sentiment_cutoff', 40))

    dims = DIMS_MULTI + DIMS_BIN
    rec: dict[str, list] = {d: [] for d in dims}
    why: dict[str, list] = {d: [] for d in dims}

    for t, L, hr in zip(df['内容'].astype(str), df['字数'], df['楼层深度'] > 0, strict=True):
        has_topic = any(r.search(t) for r in L1_RE.values())
        got = {
            'valid': rule_valid(t, L, has_topic),
            'topic_l1': rule_topic_l1(t, L, mt),
            'sentiment': rule_sentiment(t, L, ms, mc),
            'target': rule_target(t, L, hr, mg),
            'speech_act': rule_speech_act(t, L, hr),
            'is_mixed': rule_mixed(t, L),
            'has_issue': rule_bin(t, ISSUE, '技术词', 20, L),
            'purchase_intent': rule_bin(t, BUY, '购买词', 25, L),
            'churn_signal': rule_bin(t, CHURN, '劝退词', 25, L),
            'is_spoiler': rule_bin(t, SPOIL, '剧透词', 30, L),
            'is_toxic': rule_bin(t, TOXIC, '辱骂词', 15, L),
            'mentions_other_game': rule_bin(t, OTHER_GAME, '他游词', 10, L),
        }
        for d, (lab, w) in got.items():
            rec[d].append(lab)
            why[d].append(w)

    keep = ['rpid', 'BV号', '视频标题', '层级', '楼层深度', '内容', '父评论', '字数',
            'len_bucket', 'like_bucket', '点赞', '时间', 'state']
    w = df[[c for c in keep if c in df.columns]].copy()
    for d in dims:
        w['w_' + d] = rec[d]
        w['r_' + d] = why[d]
    w['n_labeled'] = w[[f'w_{d}' for d in dims]].notna().sum(axis=1)
    return w


def run(cfg: Config, spotcheck: int = 8) -> pd.DataFrame:
    clean = cfg.path('paths.clean')
    df = pd.read_parquet(clean)
    df = df[~df['is_dup']].reset_index(drop=True)
    print(f'[weak] 输入 {len(df)} 条（已去同视频内重复）')

    w = label_frame(df, cfg)
    out = cfg.path('paths.weak')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    w.to_parquet(out, index=False)

    dims = DIMS_MULTI + DIMS_BIN
    L = ['# 弱监督覆盖率报告 v2\n', f'样本 {len(w)} 条（去重后）\n',
         '## 各维度覆盖率与标签分布\n',
         '| 维度 | 覆盖率 | 标注数 | 分布 |', '|---|---:|---:|---|']
    for d in dims:
        c = w['w_' + d]
        dist = ' · '.join(f'{k} {v}' for k, v in c.value_counts().items())
        L.append(f'| `{d}` | {c.notna().mean() * 100:.1f}% | {c.notna().sum()} | {dist} |')
    L += ['', f'未做规则（留人工）：{", ".join(DIMS_UNTRAINED)}', '',
          '## 每条被标注的维度数\n', '| 维度数 | 条数 |', '|---:|---:|']
    for k, v in w['n_labeled'].value_counts().sort_index().items():
        L.append(f'| {k} | {v} |')
    total = int(w['n_labeled'].sum())
    L += ['', f'**总弱标签决策数：{total:,}**',
          f'（≥6 维度被标注：{(w["n_labeled"] >= 6).sum()} 条）', '', '## 命中规则统计\n']
    for d in DIMS_MULTI:
        L.append(f'\n**{d}**\n')
        L += ['| 规则 | 次数 |', '|---|---:|']
        for k, v in w['r_' + d].replace('', np.nan).value_counts().items():
            L.append(f'| {k} | {v} |')
    open(os.path.join(os.path.dirname(out), 'coverage.md'), 'w', encoding='utf-8').write('\n'.join(L))

    # 抽检文件 —— 规则精度只能靠人工验，这是闭环的必要入口
    sp = []
    for d in dims:
        sub = w[w['w_' + d].notna()]
        for lab, g in sub.groupby('w_' + d):
            for _, r in g.sample(min(spotcheck, len(g)), random_state=1).iterrows():
                sp.append({'维度': d, '弱标签': lab, '命中规则': r['r_' + d], '字数': r['字数'],
                           '内容': str(r['内容'])[:160], '父评论': str(r['父评论'])[:80],
                           '人工判定_对错': '', '正确标签': ''})
    pd.DataFrame(sp).to_csv(os.path.join(os.path.dirname(out), 'spotcheck.csv'),
                            index=False, encoding='utf-8-sig')

    print('\n'.join(L[2:20]))
    print(f'\n[weak] 总决策 {total:,}　抽检 {len(sp)} 条 → {os.path.dirname(out)}')
    return w


def run_probe(cfg: Config) -> None:
    """关键词探针：量化主题分布，为标签体系定档。"""
    from .lexicon import WIDE_PROBES
    df = pd.read_parquet(cfg.path('paths.clean'))
    s = df['内容'].astype(str)
    M = pd.DataFrame({k: s.str.contains(p, regex=True, na=False) for k, p in WIDE_PROBES.items()})
    hits = M.sum(1)

    L = [f'# 关键词探针（{len(s)} 条）\n', '| 探针 | 命中数 | 占比 |', '|---|---:|---:|']
    for k in M.sum().sort_values(ascending=False).index:
        L.append(f'| {k} | {int(M[k].sum())} | {M[k].mean() * 100:.1f}% |')
    long_mask = s.str.len() >= 80
    L += ['', '## 结构性结论\n',
          f'- **{(hits == 0).mean() * 100:.1f}% 命中 0 个探针** → 多为楼中楼碎片，'
          '必须设 valid 闸门，否则会被强分进某个主题',
          f'- **{(hits >= 3).mean() * 100:.1f}% 命中 ≥3 个探针** → 真·多主题，'
          'topic_l1 单选只取主主题，其余靠二分维度补',
          f'- 长评（≥80 字，占 {long_mask.mean() * 100:.1f}%）里只有 '
          f'{(hits[long_mask] == 0).mean() * 100:.1f}% 命中 0 探针 → 高价值信息高度集中在长评']
    out = os.path.join(os.path.dirname(cfg.path('paths.clean')), 'probe.md')
    open(out, 'w', encoding='utf-8').write('\n'.join(L))
    print('\n'.join(L))
    print(f'\n[probe] → {out}')
