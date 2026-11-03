"""阶段 0：从爬虫缓存重建数据集。

修的 bug（阻断级）
------------------
历史 CSV 里 rpid 落成了 float，``321944070160`` → ``3.21944E+11``：
9696 行塌成 1579 个唯一值，**47.7% 的楼中楼无法还原上文**。
``data/raw/cache/*.json`` 里 rpid 是字符串且 root/parent 完好，所以从源头重建。

同时构造送进模型的 state —— 47.7% 是楼中楼、15% 带「回复 @」，
不拼视频标题与父评论，这半数数据语义不完整。
"""
from __future__ import annotations

import glob
import json
import os
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import Config

CST = timezone(timedelta(hours=8))

RE_REPLY = re.compile(r'^回复\s*@([^\s:：]+)\s*[:：]\s*')
RE_EMOJI = re.compile(r'\[[^\[\]]{1,16}\]')
RE_WS = re.compile(r'\s+')
RE_BR = re.compile(r'<br\s*/?>', re.I)

LEN_BINS = [-1, 10, 25, 45, 79, 10 ** 9]
LEN_LABELS = ['L1_≤10', 'L2_11-25', 'L3_26-45', 'L4_46-79', 'L5_≥80']
LIKE_BINS = [-1, 0, 7, 10 ** 9]
LIKE_LABELS = ['K0_0', 'K1_1-7', 'K2_8+']

COLS = ['视频标题', 'BV号', 'UP主', '层级', '楼层深度', 'rpid', 'root_rpid', 'parent_rpid',
        '昵称', 'UID', '等级', '性别', 'IP属地', '时间', '时间戳', '日期', '点赞', '回复数',
        '回复对象', '内容', '父评论', '父评论作者', '主楼内容', '正文_无表情', '字数', '表情数',
        'state', 'state_len', 'len_bucket', 'like_bucket', 'is_dup', '内容_原始']


def flatten_cache(path: str) -> list[dict]:
    """展平一个视频的缓存：主楼 + subs 递归，记录楼层深度。"""
    d = json.load(open(path, encoding='utf-8'))
    info = d['info']
    rows: list[dict] = []

    def walk(node: dict, depth: int, root_rpid: str) -> None:
        parent = str(node.get('parent') or '')
        rows.append({
            '视频标题': info['title'], 'BV号': info['bvid'], 'UP主': info.get('up', ''),
            '视频发布': datetime.fromtimestamp(info['pubdate'], CST).strftime('%Y-%m-%d %H:%M'),
            '楼层深度': depth,
            'rpid': str(node['rpid']),
            'root_rpid': str(root_rpid) if root_rpid else '',
            'parent_rpid': parent if parent != '0' else '',
            '昵称': node.get('uname', ''), 'UID': str(node.get('uid', '')),
            '等级': node.get('level'), '性别': node.get('sex', ''), 'IP属地': node.get('ip', ''),
            '时间戳': node.get('time'),
            '时间': datetime.fromtimestamp(node['time'], CST).strftime('%Y-%m-%d %H:%M:%S'),
            '点赞': node.get('like', 0), '回复数': node.get('rcount', 0),
            '内容_原始': node.get('content', '') or '',
        })
        for sub in node.get('subs') or []:
            walk(sub, depth + 1, node['rpid'] if depth == 0 else root_rpid)

    for top in d['comments']:
        walk(top, 0, '')
    return rows


def build_state(row: pd.Series, game: str) -> str:
    """{game, video, thread?, parent?, comment} —— 顺序即模型读到的顺序。"""
    d: dict[str, str] = {'game': game, 'video': row['视频标题']}
    if row['楼层深度'] > 0:
        if row['主楼内容'] and row['主楼内容'] != row['父评论']:
            d['thread'] = row['主楼内容'][:200]
        if row['父评论']:
            d['parent'] = row['父评论'][:200]
    d['comment'] = row['内容']
    return json.dumps(d, ensure_ascii=False)


def run(cfg: Config, out: str | None = None) -> pd.DataFrame:
    cache = cfg.path('paths.raw_cache')
    files = sorted(glob.glob(os.path.join(cache, '*.json')))
    if not files:
        raise FileNotFoundError(f'缓存目录没有 json：{cache}')

    rows: list[dict] = []
    for f in files:
        rows += flatten_cache(f)
    df = pd.DataFrame(rows)

    # ---------------- 清洗正文
    txt = df['内容_原始'].astype(str).map(lambda s: RE_BR.sub(' ', s))
    df['回复对象'] = txt.str.extract(RE_REPLY, expand=False).fillna('')
    txt = txt.str.replace(RE_REPLY, '', regex=True)
    df['内容'] = txt.map(lambda s: RE_WS.sub(' ', s).strip())
    df['正文_无表情'] = df['内容'].map(lambda s: RE_WS.sub('', RE_EMOJI.sub('', s)))
    df['字数'] = df['正文_无表情'].str.len()
    df['表情数'] = df['内容'].map(lambda s: len(RE_EMOJI.findall(s)))

    # ---------------- 还原父评论（bug-1 的真正修复点）
    book = df.set_index('rpid')[['内容', '昵称']].to_dict('index')

    def lookup(rp: str) -> tuple[str, str]:
        v = book.get(rp)
        return (v['内容'], v['昵称']) if v else ('', '')

    par = df['parent_rpid'].map(lookup)
    df['父评论'] = [p[0] for p in par]
    df['父评论作者'] = [p[1] for p in par]
    df['主楼内容'] = [p[0] for p in df['root_rpid'].map(lookup)]

    miss = (df['父评论'] == '') & (df['楼层深度'] > 0)
    df.loc[miss, '父评论'] = df.loc[miss, '主楼内容']
    still = miss & (df['父评论'] == '') & (df['回复对象'] != '')
    df.loc[still, '父评论'] = '（回复 @' + df.loc[still, '回复对象'] + '，原文未抓取）'

    # ---------------- state 与分层键
    df['state'] = df.apply(lambda r: build_state(r, cfg.game), axis=1)
    df['state_len'] = df['state'].str.len()
    df['len_bucket'] = pd.cut(df['字数'], LEN_BINS, labels=LEN_LABELS)
    df['like_bucket'] = pd.cut(df['点赞'], LIKE_BINS, labels=LIKE_LABELS)
    df['层级'] = np.where(df['楼层深度'] == 0, '主楼', '楼中楼')
    df['日期'] = df['时间'].str[:10]
    df['is_dup'] = df.duplicated(df['BV号'] + '||' + df['正文_无表情'], keep='first') \
        if False else (df['BV号'] + '||' + df['正文_无表情']).duplicated(keep='first') & (df['字数'] > 0)

    df = df[COLS]
    target = out or cfg.path('paths.clean')
    os.makedirs(os.path.dirname(target), exist_ok=True)
    df.to_parquet(target, index=False)
    df.to_csv(os.path.splitext(target)[0] + '.csv', index=False, encoding='utf-8-sig')

    _report(cfg, df, os.path.join(os.path.dirname(target), 'report.md'))
    print(f'[rebuild] {len(df)} 条 → {target}')
    deep = df[df['楼层深度'] > 0]
    print(f'[rebuild] 唯一 rpid {df["rpid"].nunique()}/{len(df)}　'
          f'楼中楼关联父评论 {(deep["父评论"] != "").sum()}/{len(deep)}')
    return df


def _report(cfg: Config, df: pd.DataFrame, path: str) -> None:
    L = ['# 阶段 0 重建报告\n', f'生成：{datetime.now(CST):%Y-%m-%d %H:%M:%S}\n']
    try:
        old = pd.read_csv(cfg.path('paths.raw_csv'), dtype=str)
        L += ['## bug-1 修复验证\n',
              '| 指标 | 旧 CSV | 新 clean |', '|---|---:|---:|',
              f'| 行数 | {len(old)} | {len(df)} |',
              f'| 唯一 rpid | {old["rpid"].nunique()} | {df["rpid"].nunique()} |',
              f'| rpid 完整率 | {old["rpid"].nunique() / len(old) * 100:.1f}% | '
              f'{df["rpid"].nunique() / len(df) * 100:.1f}% |']
        deep = df[df['楼层深度'] > 0]
        L.append(f'| 楼中楼关联父评论 | 0（断链） | {(deep["父评论"] != "").sum()}/{len(deep)} = '
                 f'{(deep["父评论"] != "").mean() * 100:.1f}% |')
        L.append('')
    except Exception:
        pass
    L += ['## 概览\n',
          f'- 视频 {df["BV号"].nunique()}　评论 {len(df)}　去重后 {(~df["is_dup"]).sum()}',
          f'- 主楼 {(df["层级"] == "主楼").sum()}　楼中楼 {(df["层级"] == "楼中楼").sum()}',
          f'- 时间 {df["时间"].min()} ~ {df["时间"].max()}',
          f'- 字数（去表情）中位 {df["字数"].median():.0f}　P90 {df["字数"].quantile(.9):.0f}　最大 {df["字数"].max()}',
          f'- state 长度 中位 {df["state_len"].median():.0f}　P95 {df["state_len"].quantile(.95):.0f}',
          f'- 纯表情/空正文 {(df["字数"] == 0).sum()} 条 = {(df["字数"] == 0).mean() * 100:.2f}%', '',
          '### 长度分桶 × 层级\n', pd.crosstab(df['len_bucket'], df['层级']).to_markdown(), '',
          '### 各视频\n',
          df.groupby('视频标题').agg(条数=('rpid', 'size'), 主楼=('楼层深度', lambda s: (s == 0).sum()),
                                   中位字数=('字数', 'median'), 总赞=('点赞', 'sum')).to_markdown()]
    open(path, 'w', encoding='utf-8').write('\n'.join(L))
