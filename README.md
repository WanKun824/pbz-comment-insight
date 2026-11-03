# pbz-comment-insight

**B站游戏视频评论的多维结构化决策流水线** —— 用 [Laya](https://github.com/NandhaKishorM/laya)（非自回归 System 1 决策引擎）把海量评论打成带校准概率的结构化字段。

以《影之刃零》（Phantom Blade Zero）7 个视频、9,696 条评论为实例，完整跑通
**重建 → 诊断 → 弱监督 → 微调 → 评测 → 打标 → 报告** 七个阶段。

| 项 | 值 |
|---|---|
| 标签体系 | Pass1 **14 个正交维度** + Pass2 **8 分支 / 67 个二级主题，两级主题均中文输出** |
| 微调 | `laya-multilingual` + `option_layout="parallel"`，RTX 3060 12G，**60.8 分钟** |
| dev 提升 | accuracy **0.3375 → 0.9251**，ECE 0.1553 → 0.0483，Brier 0.743 → 0.130 |
| 推理吞吐 | 单卡 ~138 决策/秒，全量 9,696 条 × 14 维度 = 135,744 决策 **16 分钟** |
| 成本 | $0（全本地，数据不出机器） |

---

## 看板

`artifacts/report/dashboard.html` —— **自包含单文件**训练看板（无 CDN / 无 JS 依赖，
支持深浅色与移动端）。条形图用 CSS 宽度、热力图用 CSS grid、折线图用内联 SVG，
所以离线、内网、GitHub Pages 都能直接看。

十一个小节：概览 · 两级中文主题标签 · 流水线 · 数据重建 · 坍缩诊断 · 规则弱监督 · 微调 · 评测 · 混淆矩阵 · 业务结果 · 已知限制。
统计图表读取本地产物，两级主题名称读取标签体系；历史诊断说明保留既有实验结论。重跑 `pbz dashboard` 即可刷新。

```bash
pbz dashboard --open      # 生成并用默认浏览器打开
```

## 数据与隐私

**本仓库不含任何评论原文、昵称或 UID。** 评论是 B站公开页面的用户内容，
公开再分发会造成隐私问题，所以全部由 `.gitignore` 排除（29 个文件 / 75 MB）。

- `python scripts/privacy_scan.py --tracked-only` 使用本地评论探针复核待发布文件，`python scripts/check_public_files.py` 执行与 CI 相同的规则门禁
- CI 有**隐私门禁**：跟踪文件命中私有产物路径或昵称、UID、评论标识的特征规则时失败
- 本次发布复核：7,195 个昵称、7,774 个 UID、600 条评论片段探针在跟踪文件中零命中
- 入库的只有聚合结果（主题×情感矩阵、两级主题分布、时间线、视频维度）和指标报告

怎么把数据补回来、缓存 json 的字段结构 —— 见 [`data/README.md`](data/README.md)。

## 快速开始

只查看结果：克隆后直接打开 [`artifacts/report/dashboard.html`](artifacts/report/dashboard.html)，无需 Python 或 GPU。
公开 Git 仓库不含评论、训练 JSONL 和预测 parquet；数据准备见 [`data/README.md`](data/README.md)。
**训练好的完整检查点已通过 [GitHub Release v1.0.0](https://github.com/WanKun824/pbz-comment-insight/releases/tag/v1.0.0) 单独分发**，无需重新训练。
权重包约 600 MB，下载、校验、解压、单条调用及批量推理见 [模型下载与调用](docs/模型下载与调用.md)。

```bash
pip install -e .          # 装包与依赖（laya / torch / transformers>=5 …）
pbz info                  # 看配置与各阶段产物是否就位
pbz aggregate             # 需要本地已有预测 parquet，生成舆情报告
```

要从零重跑：

```bash
pbz pipeline              # 端到端（含 ~1 小时微调）
pbz pipeline --skip-train # 复用已有检查点
```

单阶段运行：

```bash
pbz rebuild               # 阶段 0：从爬虫缓存重建（修 rpid、还原楼中楼、构造 state）
pbz probe                 # 关键词探针，量化主题分布
pbz diag --n 400          # 阶段 2：零样本坍缩诊断（位置偏置 vs 标签文本劫持）
pbz weak                  # 阶段 3：规则弱监督标注
pbz dataset --cap 450     # 阶段 3b：封顶 + 切分 + JSONL + 人工标注待办
pbz train --dry           # 阶段 4：先看数据统计
pbz train --epochs 4      #          正式训练
pbz evaluate              # 阶段 5：冻结 test 集三方对照 + 覆盖率曲线
pbz predict               # 阶段 6：全量两级打标，主题输出为中文
pbz localize-topics       # 已有英文预测的两级主题迁移，无需模型，可重复执行
pbz aggregate             # 阶段 7：舆情报告 + 二级主题分布
pbz dashboard             # 刷新自包含中文看板
```

路径、超参、阈值全在 `config.yaml`，命令行参数可覆盖。
从任何工作目录调用都行（自动向上寻找 `config.yaml`，或设 `PBZ_ROOT`）。

---

## 下载训练好的模型

- [完整权重包](https://github.com/WanKun824/pbz-comment-insight/releases/download/v1.0.0/laya-pbz-v1-weights.zip)
- [SHA-256 校验文件](https://github.com/WanKun824/pbz-comment-insight/releases/download/v1.0.0/laya-pbz-v1-weights.zip.sha256)
- [详细操作文档](docs/模型下载与调用.md)

在项目根目录执行（下载命令需要 GitHub CLI，也可以通过上面的链接用浏览器下载两个文件到 `dist/`）：

```bash
gh release download v1.0.0 --repo WanKun824/pbz-comment-insight --pattern "laya-pbz-v1-weights.zip*" --dir dist
python scripts/package.py --verify dist/laya-pbz-v1-weights.zip
python -m zipfile -e dist/laya-pbz-v1-weights.zip artifacts/ckpt
python -m pip install -e . "laya==0.4.21" "transformers==5.3.0"
python examples/predict_comment.py --device cpu --comment "着色器编译等待很久，进游戏后经常掉帧，希望后续能修复这些性能问题。"
```

解压目标是 `artifacts/ckpt`，包里已经包含 `laya-pbz-v1/` 目录。单条示例不依赖本地原始评论；有 NVIDIA GPU 时可使用 `--device cuda`。
完整包已在 Python 3.11、Laya 0.4.21、transformers 5.3.0、PyTorch 2.5.1 环境下通过 CPU 与 GPU 的真实离线推理验证。

## 目录

```
pbz-comment-insight/
├── config.yaml                 所有路径与超参
├── src/pbz/
│   ├── config.py               配置加载 + Windows UTF-8 stdout
│   ├── schema.py               标签体系唯一实现来源：键渲染 / 轮转 / 答案归约
│   ├── topics.py               两级主题中文输出、旧预测迁移、二级聚合
│   ├── lexicon.py              规则词表与关键词探针（weak 与 probe 共用）
│   ├── rebuild.py              阶段 0
│   ├── weak.py                 阶段 3 + probe
│   ├── dataset.py              阶段 3b
│   ├── trainer.py              阶段 4
│   ├── evaluate.py             阶段 5
│   ├── predict.py              阶段 6
│   ├── aggregate.py            阶段 7
│   ├── diag.py                 阶段 2
│   └── cli.py                  `pbz` 命令
├── taxonomy/labels_v2.json     标签体系（含 render 方案与实测数字）
├── docs/
│   ├── 模型下载与调用.md       Release 下载、校验、安装、单条与批量推理
│   ├── 两级中文标签.md         8+67 完整中文对照、输出格式、迁移与模型兼容
│   ├── 标签体系_v2.md          体系设计、旧体系诊断、迁移表
│   ├── 训练实录.md             每个阶段的真实数字与踩过的坑
│   └── MODEL_CARD.md           模型卡：用途、限制、不该怎么用
├── data/
│   ├── raw/cache/              爬虫原始 json（rpid 为字符串，root/parent 完好）
│   ├── raw/comments_all.csv    历史 CSV（rpid 已损坏，仅留作对比）
│   ├── clean/                  重建后的数据 + 重建报告 + 探针报告
│   ├── weak/                   弱标签 + 覆盖率报告 + 344 条抽检
│   └── train/                  train/dev/test.jsonl + stats + golden_todo.xlsx
└── artifacts/
    ├── ckpt/laya-pbz-v1/       微调权重（`laya.load()` 可直接加载）
    ├── pred/pred_full.parquet  全量 9,696 条打标结果
    ├── eval/                   评测报告 + 混淆矩阵 + 覆盖率曲线
    └── report/                 离线看板、舆情报告 + 5 个 csv
```

---

## 标签体系

**核心手术：一列分类拆成 14 个正交问题 + 两级主题。**
原始手动标注有 28 个标签混在一列，其中 8 组同义重复、4 个根本不是主题维度
（`垃圾信息与主题无关` 是有效性、`期待游戏发布` 是购买意向、`游戏热梗` 是言语行为、
`游戏博主互动` 是评价对象）。诊断过程见 `docs/标签体系_v2.md`。

### Pass1（14 问一次前向）

| 维度 | 类型 | 选项 | 作用 |
|---|---|---:|---|
| `valid` | choice | 4 | **前置闸门**：substantive / banter / fragment / spam |
| `topic_l1` | choice | 8 | 战斗 / 内容品质 / 技术性能 / 商业产品 / 行业生态 / 视频与UP主 / 社区舆论 / 玩家自述 |
| `sentiment` | choice | 5 | very_neg → very_pos |
| `is_mixed` | choice | 2 | 是否褒贬并存 |
| `target` | choice | 7 | 评价对象 |
| `speech_act` | choice | 7 | review / report / question / suggest / share / joke / argue |
| `has_issue` `purchase_intent` `churn_signal` `is_spoiler` `is_toxic` `is_sarcasm` `is_actionable` `mentions_other_game` | choice | 2 | 二分标记 |

### Pass2（按 topic_l1 分支，67 个二级主题）

| 一级主题 | 二级数 | 二级主题（完整中文名称） |
|---|---:|---|
| 战斗 | 9 | 战斗手感 / 拼刀精防 / 招式流派 / 战斗难度 / 首领战斗 / 小怪与地形 / 按键操作 / 战斗演出 / 品类归属 |
| 内容品质 | 10 | 主线剧情 / 世界观与武侠味 / 角色塑造 / 台词文案 / 美术画面 / 音乐配音 / 关卡地图 / 序章质量 / 流程与内容量 / 后续更新 |
| 技术性能 | 9 | 帧率与流畅度 / 配置与驱动 / 着色器编译 / 崩溃闪退 / 画质与光追 / 功能故障 / 手柄与键位 / 平台适配 / 存档与成就 |
| 商业产品 | 8 | 定价与性价比 / 销量表现 / 购买决策 / 版本与典藏 / 周边收藏 / 退款售后 / 购买渠道 / 宣发预告 |
| 行业生态 | 7 | 国产大作意义 / 对比黑神话 / 对比日系作品 / 对比国产同行 / 媒体评分 / 审核与政策 / 资本与行业生态 |
| 视频与UP主 | 8 | 解说观点 / 剪辑制作 / 创作者本人 / 催更互动 / 攻略教学 / 商业合作与立场 / 弹幕与评论区 / 选题建议 |
| 社区舆论 | 8 | 舆论节奏 / 黑子水军指控 / 热梗段子 / 谐音审核梗 / 制作人梗 / 玩家圈子 / 玩家争论 / 热度出圈 |
| 玩家自述 | 8 | 购买与开玩 / 游玩进度 / 时间冲突 / 情怀感动 / 个人年度最佳 / 个人偏好 / 玩家身份 / 游玩与健康 |

预测中的 `p_topic_l1`、`p_topic_l2` 及两级概率字典键均为中文。内部标识和模型选项顺序保持兼容，旧权重与训练数据可继续使用，无需为汉化重训。
完整字段说明、67 项内部标识对照及旧结果迁移步骤见 [两级中文标签手册](docs/两级中文标签.md)。

```bash
pbz localize-topics       # 将本地旧英文预测汉化；也支持 --pred / --out 另存
pbz aggregate            # 输出中文报告及 artifacts/report/二级主题分布.csv
pbz dashboard            # 看板显示完整标签表及二级主题分布
```

针对单机买断制 3A 定制：没有抽卡/体力/排位/外挂/服务器经济；
有**类魂品类之争、序章质量、着色器编译、谐音审核梗、制作人女装梗、UP主身体、请假上班、对比黑神话**。

---

## 结果

### 冻结 test 集（771 条 / 1485 决策）

| 维度 | 类数 | 多数类 | 零样本 acc | **微调 acc** | 微调 macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|---:|
| `valid` | 4 | 0.382 | 0.253 | **0.971** | 0.731 | 0.026 |
| `topic_l1` | 8 | 0.228 | 0.210 | **0.911** | 0.873 | 0.033 |
| `target` | 6 | 0.296 | 0.214 | **0.949** | 0.952 | 0.048 |
| `speech_act` | 7 | 0.267 | 0.160 | **0.905** | 0.743 | 0.060 |
| `sentiment` | 4 | 0.450 | 0.367 | **0.825** | 0.732 | 0.141 |
| `mentions_other_game` | 2 | 0.559 | 0.516 | **0.946** | 0.945 | 0.051 |
| `has_issue` | 2 | 0.530 | 0.487 | **0.939** | 0.939 | 0.045 |
| `churn_signal` | 2 | 0.897 | 0.621 | **0.931** | 0.838 | 0.044 |
| `purchase_intent` | 2 | 0.667 | 0.678 | **0.908** | 0.895 | 0.068 |
| `is_mixed` | 2 | 0.771 | 0.657 | **0.871** | 0.794 | 0.118 |
| `is_spoiler` | 2 | 0.708 | 0.639 | **0.833** | 0.759 | 0.106 |
| `is_toxic` | 2 | 0.979 | 0.958 | 0.958 | **0.489** ❌ | 0.025 |

### 业务输出（9,696 条全量 → 6,653 条有效）

| 主题 | 条数 | 负面率 | 情感均分 | 赞加权情感 |
|---|---:|---:|---:|---:|
| 战斗 | 1459 | 35.8% | 0.27 | 0.62 |
| 玩家自述 | 1431 | 20.5% | 0.63 | 0.44 |
| 内容品质 | 1286 | 34.8% | 0.30 | 0.75 |
| 行业生态 | 804 | 23.3% | 0.73 | **1.08** |
| 商业产品 | 540 | 30.2% | 0.38 | 0.45 |
| 技术性能 | 456 | **37.1%** | 0.20 | **0.90** |
| 视频与UP主 | 422 | 21.6% | 0.59 | **−0.02** |
| 社区舆论 | 255 | **38.0%** | 0.18 | 0.99 |

两个可直接汇报的结论：

- **发售日拐点**：10-29 发售当天负面率 26.3% → **33.5%**，`技术性能` 话题占比从 8% 飙到 **18.5%**（着色器编译 / 驱动 / 手柄）；11-02 负面率 **35%**，焦点转向 `内容品质`（31.7%）与 `战斗`（28.4%）。
- **赞加权 vs 未加权的反差**：`技术性能` 未加权 0.20 但加权 0.90 —— 高赞是「优化比黑猴好」，骂的在长尾低赞；`视频与UP主` 反过来（0.59 → **−0.02**），高赞集中在对 UP 主的负面与催更。

---

## 两个改变了方案的发现

### 1. 零样本坍缩的主因是「标签文本劫持」，不是位置偏置

`pbz diag` 在 400 条分层样本上测了 7 个变体：

| topic_l1 变体 | collapse ↓ | entropy ↑ | flip | 行/条 |
|---|---:|---:|---:|---:|
| 语义键·无轮转 | 0.712 | 0.508 | — | 14 |
| 语义键·**满轮转** | 0.730 | 0.499 | 17.2% | **40** |
| 不透明键·无轮转 | **0.355** | **0.792** | 52.5% | 14 |
| 不透明键·中文前缀 | **0.320** | **0.871** | — | 14 |

`option_order` 满轮转 **collapse 毫无改善，成本却是 11.5 倍**（147 → 37 决策/秒）→ 关闭。
换不透明键 `O1..On`（语义名移进 criteria 描述）collapse 0.712 → 0.320 → 采用。

这正是 Laya README 警告过的 *"choice keys are rendered verbatim and the current
checkpoints can follow labels instead of the option descriptions"*。

### 2. 零样本确证不可用 → 没有拿它当伪标签

所有变体都把 64–76% 的评论判为负面，而这批数据明显偏正面（IGN 10 分、年度最佳满天飞）。
拿一个 72% 坍缩的输出自训练只会固化错误，所以改走**规则弱监督 + 弃权**。

规则本身也迭代过一轮：v1 在 344 条人工抽检里暴露 6 类系统性误判
（「十分」吃掉「十分钟」、「节奏」混了战斗节奏与舆论节奏、「帧」吃掉「几帧的反应时间」…），
v2 逐条修完。中间还踩过一个坑：把「带转折词」直接弃权，`sentiment` 覆盖率从 23% 崩到 **2.5%**
—— 中文长评几乎必有「但/不过」。

---

## 已知限制（必须读）

1. **test 标签来自规则弱监督，不是人工标注。** 0.911 衡量的是「模型是否学会了规则体现的
   那套判别」，不等于真实人工一致率。抽检估计规则本身 `topic_l1` 精度约 72–75%，
   模型的人工一致率仍需独立测量，不能将这两个比例相乘推导准确率上限。
   **人工待办（589 条）须先排除与训练数据的重叠，再标注为独立评测集。**
2. **`is_toxic` 完全失败**：macro-F1 0.489 = 全预测 B。训练集正例仅 11 条。
3. **`sentiment` 学不出 `neutral`**：规则从不输出中立，全量预测里 neutral 仅 0.3%。
   这是弱监督的结构性缺陷，必须人工补。
4. **温度拟合到 5.0 撞上限**（校准集置信 0.954 vs accuracy 0.888）→ 模型过置信，
   `is_mixed` / `is_spoiler` 的 ECE 偏高。
5. **`is_sarcasm` / `is_actionable` 未训练**（规则不可靠，无弱标签），输出不可用。
6. **`parallel` 布局的代价**：需 transformers≥5，且只能 eager 后端 ——
   ONNX 导出、TileLang 快路径、`compile=True` 都会拒绝这个检查点。
7. **`IP属地` 全为「未知」**，地域维度做不了。
8. 跨游戏泛化未验证。词表与二级主题是围绕《影之刃零》与这 7 个视频长出来的。
9. **二级主题尚无独立评测**：当前训练 JSONL 未包含 `topic_l2` 监督；两级汉化仅转换名称，不改变或重新验证模型效果。
10. **现有切分并非完全独立**：训练报告已标记 dev 有重叠，本地复核也发现 train/test 存在少量相同输入及标注。正式评测需按内容及上下文去重后重建划分。

---

## 下一步（按收益排序）

1. 排除训练重叠后标注 `golden_todo.xlsx` → 建立独立人工 test 集，测量真实效果
2. 补 `neutral` / `is_toxic` / `is_sarcasm` 的人工正例（各 150+）→ 重训
3. 上线置信度门控，用训练时拟合的按桶阈值：`choice:6-10`=0.300 / `choice:3-5`=0.974 / `choice:2`=0.633
4. 扩到更多视频与其他游戏，验证跨游戏泛化

---

## 环境

| 项 | 版本 |
|---|---|
| Python | ≥3.10（实测 3.13.3） |
| torch | ≥2.4（实测 2.7.1+cu118） |
| transformers | **≥5.0**（`option_layout="parallel"` 的硬要求，实测 5.3.0） |
| laya | ≥0.4.21 |
| GPU | RTX 3060 12G（训练占 11.4G；CPU 可推理，慢） |

```bash
pbz train --device cpu      # 无 GPU 时也能跑，但会很慢
PBZ_DEVICE=mps pbz predict  # Apple silicon
```

## 许可

代码 Apache-2.0。基座权重为 `convaiinnovations/laya`（Apache-2.0）。
评论数据来自 B站公开页面，仅供研究与内部分析，请勿再分发。
