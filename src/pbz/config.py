"""配置加载：config.yaml + 环境变量 + CLI 覆盖。

路径一律相对项目根解析，所以从任何工作目录调用都一致。
"""
from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass, field
from typing import Any

DEFAULT_CONFIG_NAME = 'config.yaml'


def ensure_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，中文标签会直接抛 UnicodeEncodeError。幂等。"""
    enc = (getattr(sys.stdout, 'encoding', '') or '').lower().replace('-', '')
    if enc != 'utf8' and not getattr(sys, '_pbz_utf8', False):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                          errors='replace', line_buffering=True)
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                                          errors='replace', line_buffering=True)
            sys._pbz_utf8 = True            # type: ignore[attr-defined]
        except Exception:
            pass


def find_root(start: str | None = None) -> str:
    """向上找含 config.yaml 的目录；找不到就用包的上两级。"""
    env = os.environ.get('PBZ_ROOT')
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    d = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.exists(os.path.join(d, DEFAULT_CONFIG_NAME)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # src/pbz/config.py -> src/pbz -> src -> <root>
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@dataclass
class Config:
    root: str
    raw: dict[str, Any] = field(default_factory=dict)

    # ---------------- 取值
    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for part in dotted.split('.'):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def path(self, dotted: str, default: str | None = None) -> str:
        """配置里的相对路径 → 绝对路径。"""
        v = self.get(dotted, default)
        if v is None:
            raise KeyError(f'config 缺少路径项：{dotted}')
        return v if os.path.isabs(v) else os.path.normpath(os.path.join(self.root, v))

    def ensure_dir(self, dotted: str, default: str | None = None) -> str:
        p = self.path(dotted, default)
        os.makedirs(p if not os.path.splitext(p)[1] else os.path.dirname(p), exist_ok=True)
        return p

    # ---------------- 便捷属性
    @property
    def game(self) -> str:
        return self.get('project.game', '影之刃零')

    @property
    def device(self) -> str:
        return os.environ.get('PBZ_DEVICE') or self.get('model.device', 'cuda')

    def describe(self) -> str:
        lines = [f'root   : {self.root}', f'game   : {self.game}', f'device : {self.device}']
        for k in ('taxonomy', 'clean', 'weak', 'ckpt', 'pred'):
            try:
                p = self.path(f'paths.{k}')
                lines.append(f'{k:<7}: {p}{"" if os.path.exists(p) else "   (缺)"}')
            except KeyError:
                pass
        return '\n'.join(lines)


_FALLBACK: dict[str, Any] = {
    'project': {'game': '影之刃零'},
    'paths': {
        'taxonomy': 'taxonomy/labels_v2.json',
        'raw_cache': 'data/raw/cache',
        'raw_csv': 'data/raw/comments_all.csv',
        'clean': 'data/clean/comments_clean.parquet',
        'weak': 'data/weak/weak_labels.parquet',
        'train_dir': 'data/train',
        'ckpt': 'artifacts/ckpt/laya-pbz-v1',
        'pred': 'artifacts/pred/pred_full.parquet',
        'eval_dir': 'artifacts/eval',
        'report_dir': 'artifacts/report',
        'log_dir': 'logs',
    },
    'model': {'base_id': 'convaiinnovations/laya', 'base_subfolder': 'multilingual',
              'device': 'cuda', 'max_len': 768, 'head_max_len': 320,
              'batch_size': 8, 'rotate': '1'},
    'weak_label': {'min_len_topic': 15, 'min_len_sentiment': 12, 'min_len_target': 8,
                   'meme_sentiment_cutoff': 40},
    'dataset': {'cap_per_label': 450, 'dev_frac': 0.10, 'test_frac': 0.12,
                'seed': 0, 'golden_n': 600},
    'train': {'epochs': 4, 'micro_batch': 2, 'grad_accum': 16, 'encoder_lr': 2.5e-5,
              'head_lr': 1.0e-4, 'loss': 'rlcd', 'option_layout': 'parallel',
              'gradient_checkpointing': True, 'max_len': 640, 'seed': 0},
    'aggregate': {'min_conf': 0.0, 'drop_valid': ['fragment', 'spam'],
                  'sentiment_weight': {'very_neg': -2, 'neg': -1, 'neutral': 0,
                                       'pos': 1, 'very_pos': 2}},
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | None = None) -> Config:
    ensure_utf8_stdout()
    root = find_root(os.path.dirname(os.path.abspath(path)) if path else None)
    cfg_path = path or os.path.join(root, DEFAULT_CONFIG_NAME)
    data: dict[str, Any] = {}
    if os.path.exists(cfg_path):
        try:
            import yaml
            data = yaml.safe_load(open(cfg_path, encoding='utf-8')) or {}
        except ImportError:
            print('[warn] 未安装 PyYAML，使用内置默认配置', file=sys.stderr)
    return Config(root=root, raw=_deep_merge(_FALLBACK, data))
