"""标签体系的唯一实现来源：加载、键渲染、轮转展开、答案归约。

为什么键要渲染成不透明的 O1..On
--------------------------------
`pbz diag` 在 400 条分层样本上测了 7 个变体。零样本坍缩的主因不是位置偏置，
而是模型按**字面**读英文语义键（README 也警告过 "choice keys are rendered verbatim
and the current checkpoints can follow labels instead of the option descriptions"）：

    topic_l1        collapse  entropy  flip_vs_A  行/条
    语义键·无轮转      0.712    0.508       —        14
    语义键·满轮转      0.730    0.499     17.2%      40     ← 轮转毫无改善，却贵 11.5 倍
    不透明键·无轮转    0.355    0.792     52.5%      14     ← 主因在这里

所以：键用 O1..On，语义名移进 criteria 描述（`中文名：判定要点`），轮转默认关闭。
`labels_v2.json` 的 `render` 段记录了这个决定与实测数字。
"""
from __future__ import annotations

import json
import os
from typing import Any

# ---------------------------------------------------------------- 维度清单
DIMS_MULTI = ['valid', 'topic_l1', 'sentiment', 'target', 'speech_act']
DIMS_BIN = ['is_mixed', 'has_issue', 'purchase_intent', 'churn_signal',
            'is_spoiler', 'is_toxic', 'mentions_other_game']
#  规则不可靠、本版未训练，输出不可用（见 docs/训练实录.md 的已知限制）
DIMS_UNTRAINED = ['is_sarcasm', 'is_actionable']
DIMS_TRAINED = DIMS_MULTI + DIMS_BIN


def load_schema(path: str) -> dict:
    """读 labels_v2.json；`render` / `cn_names` 允许写在 meta 下，这里统一提升到顶层。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f'找不到标签体系文件：{path}')
    sc = json.load(open(path, encoding='utf-8'))
    for k in ('render', 'cn_names'):
        if k not in sc and k in sc.get('meta', {}):
            sc[k] = sc['meta'][k]
    return sc


def pass1_questions(sc: dict) -> dict:
    return sc['pass1']['questions']


def pass2_questions(sc: dict) -> dict:
    """topic_l1 标签 -> 该分支的 topic_l2 问题。"""
    return {k: v['topic_l2'] for k, v in sc['pass2_by_l1'].items()}


def render_opts(sc: dict) -> tuple[dict, str, bool]:
    rd = sc.get('render', {})
    return sc.get('cn_names', {}), rd.get('key_style', 'opaque'), bool(rd.get('prefix_cn', True))


# ---------------------------------------------------------------- 渲染
def clean_q(q: dict) -> dict:
    """剥掉自定义元字段（option_order_rotate / note / gate …），只留 Laya 认的键。"""
    out = {'type': q['type'], 'instructions': q['instructions']}
    if 'criteria' in q:
        out['criteria'] = q['criteria']
    return out


def render_keys(questions: dict, cn_names: dict | None = None,
                key_style: str = 'opaque', prefix_cn: bool = True) -> tuple[dict, dict]:
    """按诊断结论重写选项键。

    返回 ``(questions, keymap)``，``keymap[dim][不透明键] = 语义键``。
    二分维度（≤2 选项）本来就是 A/B，原样保留。
    """
    out: dict[str, dict] = {}
    keymap: dict[str, dict] = {}
    for name, q in questions.items():
        q2 = clean_q(q)
        crit = q.get('criteria') or {}
        if len(crit) <= 2:
            out[name] = q2
            continue
        cn = (cn_names or {}).get(name, {})
        newc, km = {}, {}
        for i, (k, v) in enumerate(crit.items()):
            nk = f'O{i + 1}' if key_style == 'opaque' else (cn.get(k) or k)
            newc[nk] = f'{cn[k]}：{v}' if (prefix_cn and cn.get(k)) else v
            km[nk] = k
        q2['criteria'] = newc
        out[name] = q2
        keymap[name] = km
    return out, keymap


def invert_keymap(keymap: dict) -> dict:
    """语义键 -> 不透明键，构造 gold 时用。"""
    return {d: {v: k for k, v in m.items()} for d, m in keymap.items()}


# ---------------------------------------------------------------- 轮转
def rotation_offsets(k: int, mode: str) -> list[int]:
    if mode in ('1', '0', None) or k <= 2:
        return [0]
    if mode == 'full':
        return list(range(k))
    r = min(int(mode), k)
    return [round(i * k / r) % k for i in range(r)]


def expand(questions: dict, rotate: str = '1') -> tuple[dict, dict]:
    """把需要轮转的问题展开成多个子问题——它们共享同一次前向。

    返回 ``(laya_questions, meta)``，``meta[dim] = (子问题名列表, 选项键顺序)``。
    """
    out: dict[str, dict] = {}
    meta: dict[str, tuple[list[str], list[str]]] = {}
    for name, q in questions.items():
        base = clean_q(q)
        labels = list((q.get('criteria') or {}).keys())
        k = len(labels)
        offs = rotation_offsets(k, rotate) if q.get('option_order_rotate') else [0]
        subs = []
        for o in offs:
            sub = f'{name}@{o}' if len(offs) > 1 else name
            qq = dict(base)
            if o:
                qq['option_order'] = [(i + o) % k for i in range(k)]
            out[sub] = qq
            subs.append(sub)
        meta[name] = (subs, labels)
    return out, meta


def reduce_answers(ans: dict, meta: dict, keymap: dict | None = None) -> dict:
    """轮转平均 + 不透明键映回语义键 → 每个维度一个最终答案。

    每项含 ``label`` / ``p``（top-1 概率）/ ``margin``（与第二名的差）/ ``probs`` / ``n_rot``。
    """
    keymap = keymap or {}
    res: dict[str, dict[str, Any]] = {}
    for name, (subs, labels) in meta.items():
        km = keymap.get(name) or {}
        tot = {l: 0.0 for l in labels}
        n = 0
        for s in subs:
            a = ans.get(s)
            if not a:
                continue
            probs = a.get('probabilities') or {}
            for l in labels:
                tot[l] += float(probs.get(l, 0.0))
            n += 1
        if not n:
            res[name] = {'label': None, 'p': 0.0, 'margin': 0.0, 'probs': {}, 'n_rot': 0}
            continue
        for l in tot:
            tot[l] /= n
        best = max(tot, key=tot.get)
        srt = sorted(tot.values(), reverse=True)
        res[name] = {
            'label': km.get(best, best),
            'p': round(tot[best], 4),
            'margin': round(srt[0] - (srt[1] if len(srt) > 1 else 0.0), 4),
            'probs': {km.get(l, l): round(v, 4) for l, v in tot.items()},
            'n_rot': n,
        }
    return res


# ---------------------------------------------------------------- 模型
def resolve_base_dir(base_id: str, subfolder: str | None = 'multilingual') -> str:
    """HF 模型 id → 本地快照目录（laya.train.finetune 要本地目录）。"""
    from laya.train import resolve_checkpoint_dir
    d = resolve_checkpoint_dir(base_id)
    return os.path.join(d, subfolder) if subfolder else d


def load_agent(path: str, device: str = 'cuda', max_len: int | None = None,
               head_max_len: int | None = None):
    import laya
    agent = laya.load(path, device=device)
    if max_len:
        agent.cfg['max_len'] = max_len
    if head_max_len:
        agent.cfg['head_max_len'] = head_max_len
    return agent


def load_base_agent(base_id: str, subfolder: str | None, device: str = 'cuda',
                    max_len: int | None = None, head_max_len: int | None = None):
    import laya
    agent = laya.load(base_id, subfolder=subfolder, device=device)
    if max_len:
        agent.cfg['max_len'] = max_len
    if head_max_len:
        agent.cfg['head_max_len'] = head_max_len
    return agent
