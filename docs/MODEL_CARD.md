# 模型卡：laya-pbz-v1

## 概要

| 项 | 值 |
|---|---|
| 名称 | `laya-pbz-v1` |
| 基座 | [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) 的 `multilingual` 子目录（mmBERT-base 编码器 + 2 层决策头） |
| 任务 | 中文游戏视频评论的多维 typed decision（`choice` 原语） |
| 训练数据 | 9,689 条规则弱监督决策，来自 5,002 条 B站评论（《影之刃零》7 个视频） |
| 方法 | RLCD（严格恰当评分规则 + GRPO 式策略梯度），4 epochs |
| 选项布局 | **`parallel`** —— 选项顺序无关 |
| 硬件/耗时 | RTX 3060 12G，**60.8 分钟**（1,164 次优化更新） |
| 加载 | `laya.load("artifacts/ckpt/laya-pbz-v1")` |
| 许可 | Apache-2.0（随基座） |

## 训练配置

```python
TrainConfig(
    epochs=4, micro_batch=2, grad_accum=16,        # 等效 batch 32
    encoder_lr=2.5e-5, head_lr=1e-4, loss='rlcd',
    max_len=640, head_max_len=320,
    option_layout='parallel',                       # 需 transformers>=5
    gradient_checkpointing=True, seed=0,
)
```

epoch 平均 loss：1.2307 → 1.0874 → 0.6118 → 0.3114

## 输入格式

state 是一个 JSON 字符串，**字段顺序即模型读到的顺序**：

```json
{"game": "影之刃零",
 "video": "【IGN】10分，《影之刃零》评测：武侠朋克传奇",
 "thread": "<主楼正文，仅楼中楼>",
 "parent": "<父评论正文，仅楼中楼>",
 "comment": "<待判定的评论正文>"}
```

47.7% 的评论是楼中楼、15% 带「回复 @」。**不拼 `video` 与 `parent`，这半数数据语义不完整**，
准确率会明显下降。

问题用 `taxonomy/labels_v2.json` 渲染后的形式提问：
**选项键必须是不透明的 `O1..On`，语义名放在 criteria 描述里**（`中文名：判定要点`）。
训练和推理必须用同一套渲染，否则模型看到的是它没训过的选项文本。
`pbz` 的所有阶段都走 `pbz.schema.render_keys()` 这一个实现。

## 性能（冻结 test 集，771 条 / 1,485 决策）

| 维度 | 类数 | 多数类 | 基座 acc | 本模型 acc | macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|---:|
| `valid` | 4 | 0.382 | 0.253 | 0.971 | 0.731 | 0.026 |
| `topic_l1` | 8 | 0.228 | 0.210 | 0.911 | 0.873 | 0.033 |
| `target` | 6 | 0.296 | 0.214 | 0.949 | 0.952 | 0.048 |
| `speech_act` | 7 | 0.267 | 0.160 | 0.905 | 0.743 | 0.060 |
| `sentiment` | 4 | 0.450 | 0.367 | 0.825 | 0.732 | 0.141 |
| `mentions_other_game` | 2 | 0.559 | 0.516 | 0.946 | 0.945 | 0.051 |
| `has_issue` | 2 | 0.530 | 0.487 | 0.939 | 0.939 | 0.045 |
| `churn_signal` | 2 | 0.897 | 0.621 | 0.931 | 0.838 | 0.044 |
| `purchase_intent` | 2 | 0.667 | 0.678 | 0.908 | 0.895 | 0.068 |
| `is_mixed` | 2 | 0.771 | 0.657 | 0.871 | 0.794 | 0.118 |
| `is_spoiler` | 2 | 0.708 | 0.639 | 0.833 | 0.759 | 0.106 |
| `is_toxic` | 2 | 0.979 | 0.958 | 0.958 | **0.489** | 0.025 |

dev 集（1,188 决策）：accuracy 0.3375 → 0.9251，ECE 0.1553 → 0.0483，
Brier 0.7425 → 0.1301，loss 1.5229 → 0.3020。

推理吞吐：单卡 ~138 决策/秒（14 维度、`batch_size=8`、`sort_by_length=True`），
约 101 ms/条评论。

## 校准

训练后拟合的 `choice` 温度为 **5.0，撞在 `[0.5, 5.0]` 的上限**，
说明模型显著过置信（校准集 mean_confidence 0.954 vs accuracy 0.888）。

按桶拟合的弃答阈值（`target_error=0.1`）：

| 桶 | 阈值 | 样本数 |
|---|---:|---:|
| `choice:2` | 0.6325 | 164 |
| `choice:3-5` | 0.9742 | 70 |
| `choice:6-10` | 0.2996 | 166 |

用法（**一个标量阈值不能跨选项数通用**，这是 Laya #394）：

```python
res = agent.predict(state, questions, min_confidence={
    "choice:2": 0.6325, "choice:3-5": 0.9742, "choice:6-10": 0.2996, "default": 0.8,
})
if res["answers"]["topic_l1"].get("low_confidence"):
    send_to_human(...)
```

## 适用场景

✅ 中文游戏视频评论（B站/YouTube 中文区）的批量结构化打标
✅ 主题分布、情感趋势、技术问题线索的聚合统计
✅ 作为 LLM 的前置分流器：高置信自动处理，低置信才给 LLM

## 不适用 / 不该这么用

❌ **不要用它做自动封禁或内容处罚**：`is_toxic` macro-F1 仅 0.489，实际是全预测「否」。
❌ **不要把概率当作真实后验**：温度撞上限，模型过置信。先在你自己的数据上重拟合。
❌ **不要用 `is_sarcasm` / `is_actionable`**：无训练标签，输出是噪声。
❌ **不要指望 `neutral` 情感**：训练标签里几乎没有这一档。
❌ **不要在非游戏领域直接用**：词表与二级主题是围绕《影之刃零》长出来的。
❌ **不要导出 ONNX / 用 TileLang / `compile=True`**：`parallel` 布局只支持 eager 后端。
❌ **不要当作人工标注的替代**：test 标签本身来自规则，上限受规则精度约束。

## 偏差与风险

- **训练标签的偏差会被完整继承**。规则词表是人写的，它对「正面/负面」「主题归属」的
  划分方式就是模型学到的划分方式。例如规则把「退款」从情感词里删掉了（因为讨论退款政策
  不等于负面），模型也就不会把「退款」当负面信号。
- **类不平衡**：`is_toxic` 1:545、`churn_signal` 1:40。封顶 450 缓解了一部分，
  但极稀有类仍学不动。
- **时间窗狭窄**：数据只覆盖 2026-10-22 ~ 11-02（发售前后 11 天）。
  新梗、新争议点（如 DLC 发布后）会落到训练分布之外。
- **单游戏、单平台、7 个视频**。UP 主构成（IGN / 纯黑 / 狗蛋 / 蕾雅 / 宋少弘 / 官方）
  决定了 `creator` 相关词表，换 UP 主需要重标。

## 复现

```bash
pbz rebuild && pbz weak && pbz dataset --cap 450
pbz train --dry            # 确认 9,689 valid items / 0 skipped / 1,164 updates
pbz train --epochs 4
pbz evaluate
```

全部实测数字与踩过的坑见 `docs/训练实录.md`。
