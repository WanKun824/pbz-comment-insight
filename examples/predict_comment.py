"""使用发布的微调检查点对单条评论执行两级推理。

    python examples/predict_comment.py --device cuda --comment "手柄适配有问题，希望修复。"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from pbz.config import ensure_utf8_stdout, load_config  # noqa: E402
from pbz.schema import (  # noqa: E402
    DIMS_UNTRAINED,
    expand,
    load_agent,
    load_schema,
    pass1_questions,
    pass2_questions,
    reduce_answers,
    render_keys,
    render_opts,
)
from pbz.topics import topic_label  # noqa: E402


def predict_comment(agent, schema: dict, state: dict) -> dict:
    """复用已加载的 agent；返回中文主题、置信度、间隔及选项概率。"""
    text = json.dumps(state, ensure_ascii=False)

    def ask(questions):
        rendered, keymap = render_keys(questions, *render_opts(schema))
        questions, meta = expand(rendered, '1')
        raw = agent.predict(text, questions)
        return reduce_answers(raw['answers'], meta, keymap)

    answers = ask(pass1_questions(schema))
    branch = answers['topic_l1']['label']
    branches = pass2_questions(schema)
    if answers['valid']['label'] not in {'fragment', 'spam'} and branch in branches:
        answers.update(ask({'topic_l2': branches[branch]}))
    for dim in ['topic_l1', 'topic_l2']:
        if dim in answers:
            parent = branch if dim == 'topic_l2' else None
            answer = answers[dim]
            answer['label'] = topic_label(schema, answer['label'], parent)
            answer['probs'] = {topic_label(schema, key, parent): value
                               for key, value in answer['probs'].items()}
    return {dim: value for dim, value in answers.items() if dim not in DIMS_UNTRAINED}


def main() -> int:
    ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comment', required=True, help='待分类评论正文')
    parser.add_argument('--video', default='影之刃零游戏体验', help='视频标题')
    parser.add_argument('--parent', default='', help='可选的父评论正文')
    parser.add_argument('--thread', default='', help='可选的主楼正文')
    parser.add_argument('--device', default=None, help='cuda / cpu / mps，默认使用项目配置')
    parser.add_argument('--ckpt', default=None, help='可选的本地检查点目录')
    args = parser.parse_args()
    cfg = load_config()
    schema = load_schema(cfg.path('paths.taxonomy'))
    agent = load_agent(args.ckpt or cfg.path('paths.ckpt'), device=args.device or cfg.device,
                       max_len=int(cfg.get('model.max_len', 768)),
                       head_max_len=int(cfg.get('model.head_max_len', 320)))
    state = {'game': cfg.game, 'video': args.video}
    for key in ['thread', 'parent']:
        if getattr(args, key):
            state[key] = getattr(args, key)
    state['comment'] = args.comment
    answers = predict_comment(agent, schema, state)
    summary = {dim: {'label': result['label'], 'confidence': result['p']}
               for dim, result in answers.items()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
