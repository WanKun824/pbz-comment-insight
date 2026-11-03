"""阶段 4：微调 laya-multilingual。

关键配置与理由
--------------
- 基座 ``laya-multilingual``：中文必须。English 检查点在非英语上 macro 仅 0.227，
  且它自己的置信度给不出任何警告（Khmer 0.000 正确率 @ 95.2% 置信）。
- ``option_layout="parallel"``：每个选项从同一个 position id 开始、互不可见，
  决策头无位置编码，所以重排选项只能重排 logits。官方实测换序一致性
  91.7% → **100%**，准确率持平（0.764 vs 0.779）。
  代价：需 transformers>=5，且只能 eager 后端（ONNX / TileLang / compile 均拒绝）。
- 训练后清掉**继承的** ``temperature_by_options``：旧桶值在推理时优先级高于
  新拟合的 per-type 温度，会静默覆盖它。
"""
from __future__ import annotations

import json
import os
import shutil
import time

from .config import Config
from .schema import resolve_base_dir


def run(cfg: Config, *, epochs: int | None = None, out: str | None = None,
        layout: str | None = None, dry: bool = False,
        data: str | None = None, eval_data: str | None = None) -> dict:
    from laya.train import TrainConfig, dry_run, finetune

    tdir = cfg.path('paths.train_dir')
    data = data or os.path.join(tdir, 'train.jsonl')
    eval_data = eval_data or os.path.join(tdir, 'dev.jsonl')
    out = out or cfg.path('paths.ckpt')
    layout = layout or cfg.get('train.option_layout', 'parallel')
    epochs = epochs if epochs is not None else int(cfg.get('train.epochs', 4))

    if not os.path.exists(data):
        raise FileNotFoundError(f'找不到训练集：{data}（先跑 `pbz dataset`）')

    tc = TrainConfig(
        epochs=epochs,
        micro_batch=int(cfg.get('train.micro_batch', 2)),
        grad_accum=int(cfg.get('train.grad_accum', 16)),
        encoder_lr=float(cfg.get('train.encoder_lr', 2.5e-5)),
        head_lr=float(cfg.get('train.head_lr', 1e-4)),
        loss=cfg.get('train.loss', 'rlcd'),
        max_len=int(cfg.get('train.max_len', 640)),
        head_max_len=int(cfg.get('model.head_max_len', 320)),
        option_layout=layout,
        gradient_checkpointing=bool(cfg.get('train.gradient_checkpointing', True)),
        seed=int(cfg.get('train.seed', 0)),
        eval_data=eval_data if os.path.exists(eval_data) else None,
        log_every=50,
    )
    base = resolve_base_dir(cfg.get('model.base_id'), cfg.get('model.base_subfolder'))
    print(f'[train] base   {base}')
    print(f'[train] data   {data}　eval={eval_data}')
    print(f'[train] layout={layout} epochs={epochs} '
          f'micro={tc.micro_batch}x accum={tc.grad_accum} (等效 {tc.micro_batch * tc.grad_accum}) '
          f'max_len={tc.max_len} head={tc.head_max_len} loss={tc.loss}')

    if dry:
        st = dry_run(data, base, tc)
        print('\n' + json.dumps(st, ensure_ascii=False, indent=2, default=str))
        return st

    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        shutil.rmtree(out)
    t0 = time.time()
    summary = finetune(data, base, out, tc, device=cfg.device)
    mins = (time.time() - t0) / 60

    _drop_inherited_temps(out, summary)

    summary['wall_minutes'] = round(mins, 1)
    json.dump(summary, open(os.path.join(out, 'train_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2, default=str)

    rep = (summary.get('train_report') or {})
    before, after = rep.get('before') or {}, rep.get('after') or {}
    print(f'\n[train] 完成 {mins:.1f} min → {out}')
    if before and after:
        print(f'[train] dev accuracy {before.get("accuracy")} → {after.get("accuracy")}　'
              f'ECE {before.get("ece")} → {after.get("ece")}　'
              f'Brier {before.get("brier")} → {after.get("brier")}')
    for qt, c in (summary.get('calibration') or {}).items():
        for issue in (c.get('issues') or []):
            print(f'[train] ⚠ 校准（{qt}）：{issue}')

    # 清理中途快照（和最终权重完全重复，白占 600M+）
    snap = os.path.join(out, 'checkpoint_latest')
    if os.path.isdir(snap):
        shutil.rmtree(snap)
        print('[train] 已删除重复的 checkpoint_latest/')
    return summary


def _drop_inherited_temps(out: str, summary: dict) -> None:
    p = os.path.join(out, 'rl_agent_config.json')
    if not os.path.exists(p):
        return
    c = json.load(open(p, encoding='utf-8'))
    inherited = c.get('temperature_by_options') or {}
    fitted = (summary.get('temperature_by_options') or {})
    if inherited and not fitted:
        c['temperature_by_options'] = {}
        json.dump(c, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        print(f'[train] 清掉继承的 temperature_by_options（{len(inherited)} 项），'
              '否则它会在推理时静默覆盖新拟合的 per-type 温度')
