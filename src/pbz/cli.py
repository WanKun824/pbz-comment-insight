"""统一命令行入口：`pbz <stage>`。

    pbz info                     查看配置与各阶段产物是否就位
    pbz rebuild                  阶段 0：从缓存重建（修 rpid、还原楼中楼、构造 state）
    pbz probe                    关键词探针，量化主题分布
    pbz diag [--n 400]           阶段 2：零样本坍缩诊断
    pbz weak                     阶段 3：规则弱监督标注
    pbz dataset [--cap 450]      阶段 3b：封顶 + 切分 + JSONL + 人工待办
    pbz train [--dry]            阶段 4：微调
    pbz evaluate                 阶段 5：冻结 test 集评测
    pbz predict [--limit N]      阶段 6：全量打标
    pbz localize-topics          将已有预测文件的两级主题及概率键汉化（无需模型）
    pbz aggregate                阶段 7：舆情报告
    pbz dashboard                生成自包含 HTML 看板（过程 + 结果）
    pbz pipeline [--skip-train]  端到端一把跑
"""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .config import load_config


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument('-c', '--config', default=None, help='config.yaml 路径（默认向上自动寻找）')
    p.add_argument('--device', default=None, help='cuda / mps / cpu，覆盖配置')


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog='pbz', description=f'pbz-comment-insight {__version__} —— B站游戏评论多维决策流水线',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('-V', '--version', action='version', version=f'pbz {__version__}')
    sub = ap.add_subparsers(dest='cmd', metavar='<stage>')

    for name, helptext in [('info', '查看配置与产物状态'),
                           ('rebuild', '阶段 0：从缓存重建数据集'),
                           ('probe', '关键词探针，量化主题分布'),
                           ('weak', '阶段 3：规则弱监督标注')]:
        p = sub.add_parser(name, help=helptext)
        _common(p)

    p = sub.add_parser('diag', help='阶段 2：零样本坍缩诊断')
    _common(p)
    p.add_argument('--n', type=int, default=400, help='分层抽样条数')
    p.add_argument('--full-rotation', action='store_true', help='加测满轮转变体（慢 11.5 倍）')
    p.add_argument('--batch-size', type=int, default=None)

    p = sub.add_parser('dataset', help='阶段 3b：构造训练集')
    _common(p)
    p.add_argument('--cap', type=int, default=None, help='每 (维度,标签) 封顶条数')

    p = sub.add_parser('train', help='阶段 4：微调')
    _common(p)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--layout', default=None, choices=['parallel', 'sequential'])
    p.add_argument('--dry', action='store_true', help='只看数据统计，不加载权重')

    p = sub.add_parser('evaluate', help='阶段 5：冻结 test 集评测')
    _common(p)
    p.add_argument('--data', default=None, help='默认 data/train/test.jsonl')
    p.add_argument('--ckpt', default=None)
    p.add_argument('--no-base', action='store_true', help='跳过零样本基线对照')
    p.add_argument('--batch-size', type=int, default=None)

    p = sub.add_parser('predict', help='阶段 6：全量打标')
    _common(p)
    p.add_argument('--out', default=None)
    p.add_argument('--limit', type=int, default=0, help='只跑 N 条，冒烟/测速用')
    p.add_argument('--ckpt', default=None)
    p.add_argument('--base', action='store_true', help='用基座零样本（不加载微调权重）')
    p.add_argument('--rotate', default=None, help='1=关闭 / 3 / full')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--no-pass2', action='store_true')
    p.add_argument('--states', default=None)

    p = sub.add_parser('localize-topics', help='将已有预测的两级主题及概率键汉化，无需模型')
    _common(p)
    p.add_argument('--pred', default=None, help='默认读取配置中的预测 parquet')
    p.add_argument('--out', default=None, help='另存为指定 parquet；默认原子替换输入文件')

    p = sub.add_parser('aggregate', help='阶段 7：舆情报告')
    _common(p)
    p.add_argument('--pred', default=None)
    p.add_argument('--min-conf', type=float, default=None)

    p = sub.add_parser('dashboard', help='生成自包含 HTML 看板')
    _common(p)
    p.add_argument('--out', default=None, help='默认 artifacts/report/dashboard.html')
    p.add_argument('--open', action='store_true', help='生成后用默认浏览器打开')

    p = sub.add_parser('pipeline', help='端到端一把跑')
    _common(p)
    p.add_argument('--skip-train', action='store_true', help='复用已有检查点')
    p.add_argument('--skip-diag', action='store_true')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--cap', type=int, default=None)
    return ap


def cmd_info(cfg) -> None:
    from .schema import DIMS_BIN, DIMS_MULTI, DIMS_UNTRAINED, load_schema
    print(f'pbz-comment-insight {__version__}\n')
    print(cfg.describe())
    try:
        sc = load_schema(cfg.path('paths.taxonomy'))
        p1 = sc['pass1']['questions']
        n2 = sum(len(v['topic_l2']['criteria']) for v in sc['pass2_by_l1'].values())
        print(f'\n体系   : Pass1 {len(p1)} 维度　Pass2 {len(sc["pass2_by_l1"])} 分支 / {n2} 个二级主题')
        print(f'渲染   : 键 {sc.get("render", {}).get("key_style")}　'
              f'中文前缀 {sc.get("render", {}).get("prefix_cn")}')
        print(f'已训练 : {len(DIMS_MULTI + DIMS_BIN)} 个　未训练(不可用)：{", ".join(DIMS_UNTRAINED)}')
    except Exception as e:                            # noqa: BLE001
        print(f'\n[warn] 体系文件读取失败：{e}')
    print('\n产物状态')
    train_jsonl = os.path.join(cfg.path('paths.train_dir'), 'train.jsonl')
    ckpt = cfg.path('paths.ckpt')
    checks = [
        ('clean', cfg.path('paths.clean')),
        ('weak', cfg.path('paths.weak')),
        ('train.jsonl', train_jsonl),
        # 权重单独分发，只有配置没有 safetensors 不算就位
        ('ckpt (weights)', os.path.join(ckpt, 'model.safetensors')),
        ('pred', cfg.path('paths.pred')),
    ]
    for label, p in checks:
        ok = os.path.exists(p)
        size = ''
        if ok:
            size = (f'  {os.path.getsize(p) / 1e6:.1f} MB' if os.path.isfile(p) else
                    f'  {sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) / 1e6:.0f} MB')
        print(f'  [{"x" if ok else " "}] {label:<15}{size}')
    if not os.path.exists(os.path.join(ckpt, 'model.safetensors')):
        print('\n  → 下载权重：https://github.com/WanKun824/pbz-comment-insight/releases/tag/v1.0.0\n'
              f'    将 laya-pbz-v1-weights.zip 解压到 {os.path.relpath(os.path.dirname(ckpt), cfg.root)}/'
              '（包内已包含 laya-pbz-v1/ 目录）。\n'
              '    完整步骤见 docs/模型下载与调用.md。也可用 `pbz train` 自行训练。')


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help()
        return 1

    cfg = load_config(getattr(args, 'config', None))
    if getattr(args, 'device', None):
        cfg.raw.setdefault('model', {})['device'] = args.device

    if args.cmd == 'info':
        cmd_info(cfg)
    elif args.cmd == 'rebuild':
        from . import rebuild
        rebuild.run(cfg)
    elif args.cmd == 'probe':
        from . import weak
        weak.run_probe(cfg)
    elif args.cmd == 'diag':
        from . import diag
        diag.run(cfg, n=args.n, full_rotation=args.full_rotation, batch_size=args.batch_size)
    elif args.cmd == 'weak':
        from . import weak
        weak.run(cfg)
    elif args.cmd == 'dataset':
        from . import dataset
        dataset.run(cfg, cap=args.cap)
    elif args.cmd == 'train':
        from . import trainer
        trainer.run(cfg, epochs=args.epochs, out=args.out, layout=args.layout, dry=args.dry)
    elif args.cmd == 'evaluate':
        from . import evaluate
        evaluate.run(cfg, data=args.data, ckpt=args.ckpt, no_base=args.no_base,
                     batch_size=args.batch_size)
    elif args.cmd == 'predict':
        from . import predict
        predict.run(cfg, out=args.out, limit=args.limit, ckpt=args.ckpt, base=args.base,
                    rotate=args.rotate, batch_size=args.batch_size,
                    no_pass2=args.no_pass2, states=args.states)
    elif args.cmd == 'localize-topics':
        from . import topics
        topics.run(cfg, pred=args.pred, out=args.out)
    elif args.cmd == 'aggregate':
        from . import aggregate
        aggregate.run(cfg, pred=args.pred, min_conf=args.min_conf)
    elif args.cmd == 'dashboard':
        from . import dashboard
        p = dashboard.build(cfg, out=args.out)
        if args.open:
            import webbrowser
            webbrowser.open('file:///' + p.replace(os.sep, '/'))
    elif args.cmd == 'pipeline':
        return _pipeline(cfg, args)
    return 0


def _pipeline(cfg, args) -> int:
    from . import aggregate, dataset, evaluate, predict, rebuild, trainer, weak
    steps: list[tuple[str, callable]] = [
        ('rebuild', lambda: rebuild.run(cfg)),
        ('probe', lambda: weak.run_probe(cfg)),
    ]
    if not args.skip_diag:
        from . import diag
        steps.append(('diag', lambda: diag.run(cfg, n=400)))
    steps += [
        ('weak', lambda: weak.run(cfg)),
        ('dataset', lambda: dataset.run(cfg, cap=args.cap)),
    ]
    if not args.skip_train:
        steps.append(('train', lambda: trainer.run(cfg, epochs=args.epochs)))
    steps += [
        ('evaluate', lambda: evaluate.run(cfg)),
        ('predict', lambda: predict.run(cfg)),
        ('aggregate', lambda: aggregate.run(cfg)),
        ('dashboard', lambda: __import__('pbz.dashboard', fromlist=['build']).build(cfg)),
    ]
    for i, (name, fn) in enumerate(steps, 1):
        print(f'\n{"=" * 72}\n[{i}/{len(steps)}] {name}\n{"=" * 72}')
        try:
            fn()
        except Exception as e:                        # noqa: BLE001
            print(f'\n[pipeline] {name} 失败：{type(e).__name__}: {e}', file=sys.stderr)
            return 1
    print('\n[pipeline] 全部完成')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
