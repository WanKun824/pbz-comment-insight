"""两级主题汉化：模型内部标识保持稳定，预测与业务输出使用中文。"""
from __future__ import annotations

import json
import os
import tempfile
from typing import TYPE_CHECKING

from .config import Config
from .schema import load_schema

if TYPE_CHECKING:
    import pandas as pd


def topic_label(sc: dict, value, parent=None):
    """转换一级或指定分支的二级标签；兼容中文、空值与未知扩展标签。"""
    if not isinstance(value, str):
        return value
    names = sc.get('cn_names', {})
    l1 = names.get('topic_l1', {})
    if parent is None:
        return l1.get(value, value)
    if not isinstance(parent, str):
        return value
    branch = {v: k for k, v in l1.items()}.get(parent, parent)
    return names.get('topic_l2', {}).get(branch, {}).get(value, value)


def localize_predictions(df: pd.DataFrame, sc: dict) -> pd.DataFrame:
    """复制并汉化预测标签及概率字典；重复执行不改变值，不改动概率数字。"""
    out = df.copy()
    if 'p_topic_l1' not in out:
        return out
    parents = out['p_topic_l1'].tolist()
    if 'p_topic_l2' in out:
        out['p_topic_l2'] = [topic_label(sc, value, parent)
                             for value, parent in zip(out['p_topic_l2'], parents, strict=True)]
    if 'probs_json' in out:
        records = []
        for raw, parent in zip(out['probs_json'], parents, strict=True):
            if not isinstance(raw, str):
                records.append(raw)
                continue
            probs = json.loads(raw)
            for dim, branch in [('topic_l1', None), ('topic_l2', parent)]:
                if dim in probs:
                    probs[dim] = {topic_label(sc, key, branch): value
                                  for key, value in probs[dim].items()}
            records.append(json.dumps(probs, ensure_ascii=False))
        out['probs_json'] = records
    out['p_topic_l1'] = [topic_label(sc, value) for value in parents]
    return out


def topic_distribution(df: pd.DataFrame, sc: dict) -> pd.DataFrame:
    """有效评论的两级主题聚合，只含标签和计数，可公开分发。"""
    import pandas as pd

    cols = ['一级主题', '二级主题', '条数']
    if not {'p_topic_l1', 'p_topic_l2'} <= set(df.columns):
        return pd.DataFrame(columns=cols)
    data = df
    if 'p_valid' in data:
        data = data[~data['p_valid'].isin(['fragment', 'spam'])]
    data = localize_predictions(data[['p_topic_l1', 'p_topic_l2']], sc)
    return (data.dropna(subset=['p_topic_l1', 'p_topic_l2'])
            .groupby(['p_topic_l1', 'p_topic_l2']).size().rename('条数').reset_index()
            .rename(columns={'p_topic_l1': '一级主题', 'p_topic_l2': '二级主题'})
            .sort_values(['一级主题', '条数', '二级主题'], ascending=[True, False, True])
            .reset_index(drop=True))


def run(cfg: Config, *, pred: str | None = None, out: str | None = None) -> str:
    """无需模型迁移已有预测文件，临时文件写完后原子替换目标。"""
    import pandas as pd

    source = pred or cfg.path('paths.pred')
    target = os.path.abspath(out or source)
    if not os.path.exists(source):
        raise FileNotFoundError(f'找不到预测结果：{source}（先跑 `pbz predict` 或提供 --pred）')
    sc = load_schema(cfg.path('paths.taxonomy'))
    df = localize_predictions(pd.read_parquet(source), sc)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    # 同目录临时文件保证替换不跨盘；写入失败时原文件仍可使用。
    fd, tmp = tempfile.mkstemp(suffix='.parquet', dir=os.path.dirname(target))
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f'[localize-topics] {len(df):,} 条，两级主题及概率键已汉化 → {target}')
    return target
