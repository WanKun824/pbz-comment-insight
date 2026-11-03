# 数据目录

**本仓库不包含任何评论原始数据。** 评论里有 B站用户的昵称、UID 与原文，
公开分发会造成隐私问题，所以 `.gitignore` 把它们全部排除了。
用 `python scripts/privacy_scan.py` 可以随时复核哪些文件含个人数据
（当前：26 个文件 / 40 MB，全部在忽略列表里）。

## 仓库里有什么

| 路径 | 内容 | 含个人数据 |
|---|---|---|
| `clean/report.md` | 重建报告（bug-1 修复验证、长度分布） | 否 |
| `clean/probe.md` | 关键词探针的主题分布 | 否 |
| `weak/coverage.md` | 弱监督覆盖率与命中规则统计 | 否 |
| `train/stats.md` | 切分规模与封顶前后分布 | 否 |
| `raw/` `clean/*.parquet` `weak/*.parquet` `train/*.jsonl` | 评论原文 | **是，已忽略** |

## 两级主题输出与旧预测

`p_topic_l1`、`p_topic_l2` 及两级概率字典键统一使用中文；完整映射见
[两级中文标签](../docs/两级中文标签.md)。已有英文预测可运行：

```bash
pbz localize-topics       # 无需模型，转换本地预测文件，可重复执行
pbz aggregate            # 刷新中文报告及二级主题聚合 CSV
pbz dashboard
```

公开可查看 `artifacts/report/二级主题分布.csv`，它只含一级主题、二级主题和条数。
训练 JSONL 与弱标签 parquet 仍使用内部兼容标识，避免改变现有模型的训练输入；
新导出的人工标注待办与主题统计报告使用中文展示。

## 怎么把数据补回来

### 自己抓

需要 `data/raw/cache/<BVID>.json`，结构：

```json
{
  "info": {"bvid": "BV1xxx", "aid": 123, "title": "视频标题",
           "url": "https://...", "up": "UP主名", "up_mid": 123, "pubdate": 1792973100,
           "desc": "简介"},
  "comments": [
    {"rpid": "321944070160",            // 必须是字符串！见下
     "root": "0", "parent": "0",
     "uname": "<昵称>", "uid": "<UID>", "level": 6, "sex": "男", "ip": "未知",
     "time": 1793005982, "like": 8115, "content": "...", "rcount": 2372,
     "subs": [ /* 同结构的楼中楼 */ ]}
  ]
}
```

字段语义见 `src/pbz/rebuild.py::flatten_cache`。换游戏还需要改
`src/pbz/lexicon.py` 的词表和 `taxonomy/labels_v2.json` 的二级主题。

> ⚠️ **`rpid` 一定要存成字符串。** 本项目最初的阻断级 bug 就是它落成了 float：
> `321944070160` → `3.21944E+11`，9696 行塌成 1579 个唯一值，
> 47.7% 的楼中楼无法关联父评论。读 CSV 时同样要 `dtype={'rpid': str}`。

### 然后

```bash
pbz rebuild      # 重建 + 构造 state + 分层键
pbz weak         # 规则弱监督
pbz dataset      # 训练集 + 人工标注待办
pbz train        # 微调（RTX 3060 约 1 小时）
pbz evaluate && pbz predict && pbz aggregate && pbz dashboard
```

或一把跑：`pbz pipeline`

### 只想看结果

`artifacts/report/dashboard.html` 是自包含单文件看板（无 CDN / 无 JS 依赖），
数据已全部聚合，不含任何评论原文 —— 直接用浏览器打开即可。
