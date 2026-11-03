"""pbz-comment-insight —— B站游戏视频评论的多维结构化决策流水线。

六个阶段，每个都是一个可独立运行的子命令（见 `pbz --help`）：

    rebuild   从爬虫缓存重建数据集（修 rpid、还原楼中楼、构造 state）
    probe     关键词探针，量化主题分布
    diag      零样本坍缩诊断（位置偏置 vs 标签文本劫持）
    weak      规则弱监督标注（高精度优先，允许弃权）
    dataset   封顶 + 切分 + 导出 Laya 训练 JSONL + 人工标注待办
    train     微调 laya-multilingual（parallel option layout）
    evaluate  冻结 test 集三方对照 + 覆盖率-准确率曲线
    predict   全量两级推理打标
    aggregate 聚合成舆情报告

设计依据与实测数字见 docs/训练实录.md。
"""

__version__ = '1.0.0'

from .config import Config, load_config  # noqa: F401

__all__ = ['Config', 'load_config', '__version__']
