"""冒烟测试：不加载模型权重，只验证配置、体系、渲染、规则、聚合是否自洽。

    python -m pytest tests/ -q        # 或直接 python tests/test_smoke.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from pbz.config import load_config  # noqa: E402
from pbz.schema import (  # noqa: E402
    DIMS_BIN,
    DIMS_TRAINED,
    expand,
    invert_keymap,
    load_schema,
    pass1_questions,
    pass2_questions,
    reduce_answers,
    render_keys,
    render_opts,
)

CFG = load_config()
SC = load_schema(CFG.path('paths.taxonomy'))


def test_config_paths():
    assert os.path.isdir(CFG.root)
    assert os.path.exists(CFG.path('paths.taxonomy'))
    assert CFG.game


def test_schema_shape():
    p1 = pass1_questions(SC)
    assert set(DIMS_TRAINED) <= set(p1), '体系缺了已训练维度'
    assert len(pass2_questions(SC)) == 8, 'Pass2 应有 8 个 L1 分支'
    for name, q in p1.items():
        assert q['type'] == 'choice', f'{name} 必须是 choice（中文禁用 noul）'
        assert q['instructions'].strip()
        assert len(q['criteria']) >= 2
        for k, v in q['criteria'].items():
            assert str(v).strip(), f'{name}.{k} 的 criteria 描述不能为空'
    # Pass2 每个分支的选项数不得超过 Laya 的可用区间
    for l1, q in pass2_questions(SC).items():
        assert 2 <= len(q['criteria']) <= 12, f'{l1} 的 L2 选项数越界'


def test_binary_dims_are_ab():
    """二分维度必须是两选项 A/B —— 中文 noul 会退化成常数 true。"""
    p1 = pass1_questions(SC)
    for d in DIMS_BIN:
        assert set(p1[d]['criteria']) == {'A', 'B'}, f'{d} 的键必须是 A/B'


def test_render_keys_opaque_and_reversible():
    cn, ks, pc = render_opts(SC)
    assert ks == 'opaque' and pc is True, 'render 方案应为「不透明键 + 中文前缀」'
    q, km = render_keys(pass1_questions(SC), cn, ks, pc)
    # 多选维度被改名成 O1..On，二分维度保持 A/B
    assert set(q['topic_l1']['criteria']) == {f'O{i}' for i in range(1, 9)}
    assert set(q['has_issue']['criteria']) == {'A', 'B'}
    # 语义名进了描述
    assert q['topic_l1']['criteria']['O1'].startswith('战斗：')
    # 映射可逆
    inv = invert_keymap(km)
    for d, m in km.items():
        for opaque, semantic in m.items():
            assert inv[d][semantic] == opaque
    # 只剩 Laya 认的键
    for qq in q.values():
        assert set(qq) <= {'type', 'instructions', 'criteria', 'option_order'}


def test_expand_and_reduce_roundtrip():
    cn, ks, pc = render_opts(SC)
    q, km = render_keys(pass1_questions(SC), cn, ks, pc)
    lq, meta = expand({'topic_l1': q['topic_l1']}, '1')
    assert len(lq) == 1, '轮转关闭时应只有 1 行'
    lq3, meta3 = expand({'topic_l1': dict(q['topic_l1'], option_order_rotate=True)}, '3')
    assert len(lq3) == 3

    fake = {s: {'probabilities': {f'O{i}': (0.9 if i == 5 else 0.0125) for i in range(1, 9)}}
            for s in meta3['topic_l1'][0]}
    out = reduce_answers(fake, meta3, km)
    assert out['topic_l1']['label'] == 'industry', '不透明键应映回语义键'
    assert abs(out['topic_l1']['p'] - 0.9) < 1e-6
    assert out['topic_l1']['n_rot'] == 3


def test_weak_rules_on_known_cases():
    """针对抽检里暴露过的 6 类系统性误判做回归，防止词表改动把老 bug 放回来。

    用例是**合成**的（不引用真实用户评论），但每条都踩中当初出错的那个正则路径。
    对应修正见 docs/训练实录.md 的「v1 → v2 的修正」表。
    """
    from pbz.weak import rule_sentiment, rule_speech_act, rule_target, rule_topic_l1

    # v1 bug：POS_STRONG 的「十分」吃掉了「十分钟」，把时长吐槽判成强正面
    assert rule_sentiment('前面十分钟全是废话，讲机制才用了两分钟', 20)[0] != 'very_pos'
    # v1 bug：梗句里的反讽褒贬被当字面，判成强负面
    assert rule_sentiment('差评，这建模做得也太敬业了吧[doge][doge]', 18)[0] is None
    # v1 bug：疑问句被判情感（「烂了」指任务没做完，不是评价）
    assert rule_sentiment('支线没做完会影响结局吗？我好多支线都烂在半路了', 22)[0] is None
    # v1 bug：「节奏」不分战斗节奏与舆论节奏，把战斗话题归到 community
    assert rule_topic_l1('这游戏的战斗节奏一停就断，必须逼着往前压，格挡和精防都要跟上', 29)[0] == 'combat'
    # v1 bug：「帧」不分帧率与动作帧，把战斗话题归到 tech
    assert rule_topic_l1('红光提示只有几帧的反应时间，太快了根本来不及精防或者闪避', 27)[0] == 'combat'
    # v1 bug：「更新」被当成 UP 主催更词，把他游评价归到 creator
    assert rule_target('那款老游戏发售两年了也不更新补全内容', 18, False)[0] != 'creator'
    # 短句一律弃权 —— 脱离上下文无法判主题
    assert rule_topic_l1('又是魂', 3)[0] is None
    # v1 bug：report 规则抢走了带技术词的提问句（question 一度占 48%）
    assert rule_speech_act('请问这个适配的是索尼还是微软的手柄啊', 18, False)[0] == 'question'


def test_train_jsonl_gold_schema():
    """laya.train 要 {dim: {"probabilities": {key: p}}}，不是裸标签。"""
    p = os.path.join(CFG.path('paths.train_dir'), 'train.jsonl')
    if not os.path.exists(p):
        return
    cn, ks, pc = render_opts(SC)
    q1, km = render_keys(pass1_questions(SC), cn, ks, pc)
    with open(p, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 200:
                break
            row = json.loads(line)
            assert {'state', 'questions', 'gold'} <= set(row)
            json.loads(row['state'])                        # state 必须是合法 JSON
            assert row['gold'] and set(row['gold']) <= set(row['questions'])
            for d, g in row['gold'].items():
                assert 'probabilities' in g, 'gold 必须带 probabilities'
                for key in g['probabilities']:
                    assert key in q1[d]['criteria'], f'{d} 的 gold 键 {key} 不在渲染后的选项里'


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f'  ok   {fn.__name__}')
        except Exception as e:                              # noqa: BLE001
            bad += 1
            print(f'  FAIL {fn.__name__}: {type(e).__name__}: {e}')
    print(f'\n{len(fns) - bad}/{len(fns)} passed')
    raise SystemExit(1 if bad else 0)
