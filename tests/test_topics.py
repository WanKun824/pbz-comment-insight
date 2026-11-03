"""两级汉化回归：覆盖完整词表、分支路由、旧结果迁移与模型兼容。"""
from __future__ import annotations

import copy
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from pbz.config import Config, load_config  # noqa: E402
from pbz.schema import load_schema, pass1_questions, render_keys, render_opts  # noqa: E402
from pbz.topics import localize_predictions, run, topic_distribution, topic_label  # noqa: E402

CFG = load_config()
SC = load_schema(CFG.path('paths.taxonomy'))


def test_all_75_topic_names_are_chinese_and_complete():
    names = SC['cn_names']
    assert set(names['topic_l1']) == set(SC['pass2_by_l1'])
    assert len(set(names['topic_l1'].values())) == 8
    total = 0
    for branch, data in SC['pass2_by_l1'].items():
        mapping = names['topic_l2'][branch]
        assert set(mapping) == set(data['topic_l2']['criteria'])
        assert len(set(mapping.values())) == len(mapping)
        for key, label in mapping.items():
            assert any('\u4e00' <= char <= '\u9fff' for char in label)
            assert topic_label(SC, key, branch) == label
            assert topic_label(SC, key, topic_label(SC, branch)) == label
        total += len(mapping)
    assert total == 67


def test_localization_preserves_probabilities_nulls_and_is_idempotent():
    source = pd.DataFrame({
        'p_topic_l1': ['tech', '技术性能', 'combat', 'future_topic'],
        'p_topic_l2': ['shader', 'fps', None, 'future_subtopic'],
        'c_topic_l1': [0.8, 0.9, 0.7, 0.6],
        'c_topic_l2': [0.6, 0.7, None, 0.5],
        'probs_json': [
            json.dumps({'topic_l1': {'tech': 0.8, 'combat': 0.2},
                        'topic_l2': {'shader': 0.6, 'fps': 0.4}, 'has_issue': {'A': 0.9, 'B': 0.1}}),
            json.dumps({'topic_l1': {'技术性能': 1.0}, 'topic_l2': {'fps': 1.0}}),
            json.dumps({'topic_l1': {'combat': 1.0}}),
            None,
        ],
    }, index=[8, 3, 7, 4])
    result = localize_predictions(source, SC)
    assert source.p_topic_l1.iloc[0] == 'tech', '不应就地修改输入'
    assert result.p_topic_l1.tolist() == ['技术性能', '技术性能', '战斗', 'future_topic']
    assert result.p_topic_l2.dropna().tolist() == ['着色器编译', '帧率与流畅度', 'future_subtopic']
    # pandas 2 的 object 列与 pandas 3 的 str 列使用不同空值标记，缺失位置必须一致。
    pd.testing.assert_series_equal(result.p_topic_l2.isna(), source.p_topic_l2.isna())
    probs = json.loads(result.probs_json.iloc[0])
    assert probs['topic_l1'] == {'技术性能': 0.8, '战斗': 0.2}
    assert probs['topic_l2'] == {'着色器编译': 0.6, '帧率与流畅度': 0.4}
    assert probs['has_issue'] == {'A': 0.9, 'B': 0.1}
    pd.testing.assert_frame_equal(result[['c_topic_l1', 'c_topic_l2']], source[['c_topic_l1', 'c_topic_l2']])
    pd.testing.assert_frame_equal(localize_predictions(result, SC), result)
    assert topic_label(SC, pd.NA) is pd.NA
    assert topic_label(SC, None) is None


def test_l2_names_are_resolved_within_the_parent_branch():
    schema = copy.deepcopy(SC)
    schema['cn_names']['topic_l2']['combat']['shared_key'] = '战斗分支示例'
    schema['cn_names']['topic_l2']['tech']['shared_key'] = '技术分支示例'
    assert topic_label(schema, 'shared_key', 'combat') == '战斗分支示例'
    assert topic_label(schema, 'shared_key', '技术性能') == '技术分支示例'
    assert topic_label(schema, 'shader', 'combat') == 'shader'


def test_training_prompts_still_match_the_existing_checkpoint():
    rendered, _ = render_keys(pass1_questions(SC), *render_opts(SC))
    path = os.path.join(CFG.path('paths.ckpt'), 'questions.json')
    trained = json.load(open(path, encoding='utf-8'))
    for dim, question in trained.items():
        assert rendered[dim] == question, f'{dim} 的模型输入不应因汉化发生变化'


def test_distribution_combines_legacy_and_chinese_labels_and_keeps_gate():
    frame = pd.DataFrame({
        'p_topic_l1': ['tech', '技术性能', 'combat', 'tech'],
        'p_topic_l2': ['shader', '着色器编译', None, 'fps'],
        'p_valid': ['substantive', 'banter', 'fragment', 'spam'],
    })
    assert topic_distribution(frame, SC).to_dict('records') == [
        {'一级主题': '技术性能', '二级主题': '着色器编译', '条数': 2},
    ]
    assert topic_distribution(frame.drop(columns='p_topic_l2'), SC).empty


def test_migrate_file_without_model_and_repeat_safely(tmp_path):
    source, target = tmp_path / 'legacy.parquet', tmp_path / '中文预测.parquet'
    pd.DataFrame({'p_topic_l1': ['tech'], 'p_topic_l2': ['shader']}).to_parquet(source, index=False)
    assert run(CFG, pred=str(source), out=str(target)) == str(target)
    assert pd.read_parquet(source).p_topic_l1.iloc[0] == 'tech'
    first = pd.read_parquet(target)
    run(CFG, pred=str(target))
    pd.testing.assert_frame_equal(first, pd.read_parquet(target))
    assert first.p_topic_l2.iloc[0] == '着色器编译'
    with pytest.raises(FileNotFoundError):
        run(CFG, pred=str(tmp_path / 'missing.parquet'))


def test_predict_routes_with_stable_ids_and_saves_both_levels_in_chinese(tmp_path, monkeypatch):
    from pbz import predict

    calls = []

    class FakeAgent:
        device = 'cpu'
        dtype = 'float32'
        cfg = {'max_len': 768, 'head_max_len': 320}

        def predict_batch(self, states, questions, **kwargs):
            calls.append(questions)
            answers = {}
            for dim, question in questions.items():
                selected = 'O3' if dim == 'topic_l1' else next(iter(question['criteria']))
                answers[dim] = {'probabilities': {key: float(key == selected)
                                                 for key in question['criteria']}}
            return [{'answers': answers} for _ in states]

    source, target = tmp_path / 'input.parquet', tmp_path / 'output.parquet'
    pd.DataFrame({'state': [json.dumps({'comment': '合成的技术性能测试输入'}, ensure_ascii=False)]}).to_parquet(
        source, index=False)
    monkeypatch.setattr(predict, '_agent', lambda *args: FakeAgent())
    result = predict.run(CFG, states=str(source), out=str(target))
    assert len(calls) == 2
    assert set(calls[1]) == {'topic_l2'}
    assert calls[1]['topic_l2']['instructions'] == SC['pass2_by_l1']['tech']['topic_l2']['instructions']
    assert result.p_topic_l1.iloc[0] == '技术性能'
    assert result.p_topic_l2.iloc[0] == '帧率与流畅度'
    probs = json.loads(pd.read_parquet(target).probs_json.iloc[0])
    assert probs['topic_l1']['技术性能'] == 1.0
    assert probs['topic_l2']['帧率与流畅度'] == 1.0


def test_aggregate_accepts_old_and_chinese_predictions(tmp_path):
    from pbz.aggregate import run as aggregate

    frame = pd.DataFrame({
        'rpid': ['synthetic-one', 'synthetic-two'], '视频标题': ['合成视频', '合成视频'],
        '时间': ['2026-10-29 12:00:00', '2026-10-30 12:00:00'], '点赞': [3, 0],
        '内容': ['用于回归的合成内容一', '用于回归的合成内容二'],
        'p_valid': ['substantive', 'substantive'], 'p_topic_l1': ['tech', 'combat'],
        'p_topic_l2': ['shader', 'parry'], 'p_sentiment': ['neg', 'pos'],
        'c_topic_l1': [0.9, 0.8], 'p_has_issue': ['A', 'B'], 'c_has_issue': [0.9, 0.8],
        'p_purchase_intent': ['B', 'B'], 'c_purchase_intent': [0.8, 0.8],
        'p_mentions_other_game': ['B', 'B'], 'c_mentions_other_game': [0.8, 0.8],
    })
    config = Config(root=CFG.root, raw=copy.deepcopy(CFG.raw))
    outputs = []
    for name, data in [('old', frame), ('chinese', localize_predictions(frame, SC))]:
        source, directory = tmp_path / f'{name}.parquet', tmp_path / name
        data.to_parquet(source, index=False)
        config.raw['paths']['report_dir'] = str(directory)
        outputs.append(aggregate(config, pred=str(source)))
        counts = pd.read_csv(directory / '二级主题分布.csv')
        assert set(counts['二级主题']) == {'着色器编译', '拼刀精防'}
        actions = pd.read_csv(directory / '可行动清单.csv')
        assert actions['一级主题'].iloc[0] == '技术性能'
        assert actions['二级主题'].iloc[0] == '着色器编译'
    pd.testing.assert_frame_equal(*outputs)
