# GitHub 发布与维护

代码仓库：<https://github.com/WanKun824/pbz-comment-insight>

仓库发布源码、文档、模型配置、评测指标、中文聚合 CSV 和自包含 HTML 看板。原始评论、用户标识、训练集、预测 parquet 和可行动清单保留在本地。模型权重通过 [Release v1.0.0](https://github.com/WanKun824/pbz-comment-insight/releases/tag/v1.0.0) 的独立附件发布，不进入普通 Git 提交。

## 获取项目

```bash
git clone https://github.com/WanKun824/pbz-comment-insight.git
cd pbz-comment-insight
```

只看结果可以直接打开 `artifacts/report/dashboard.html`。它不依赖 Python、GPU 或外部图表库，包含 8 个一级主题、67 个二级主题的完整中文标签表与现有预测的聚合结果。

需要运行流水线时：

```bash
pip install -e .
pbz info
```

公开仓库不含预测 parquet，因此克隆后不能直接运行 `pbz aggregate`。数据准备见 [`data/README.md`](../data/README.md)，训练和推理约束见 [模型卡](MODEL_CARD.md)。

## 本地修改与推送

先确认目标账号、远程地址和当前分支，不要将访问令牌写进配置文件或远程 URL。

```bash
gh auth status
git remote -v
git branch --show-current
```

两级中文标签修改后，运行：

```bash
python -m pytest tests/ -q
python -m ruff check src tests scripts

# 本地有旧英文预测时，转换名称后刷新报告与看板；无需加载模型
pbz localize-topics
pbz aggregate
pbz dashboard

# 跟踪文件检查不依赖原始数据；内容探针需要本地 clean 数据
python scripts/check_public_files.py
python scripts/privacy_scan.py --tracked-only
```

只暂存需要发布的源码、文档与聚合产物，核对暂存文件后再推送：

```bash
git add src tests taxonomy docs scripts README.md .github
# 按实际变更选择聚合文件，勿添加含评论原文的报告
git add artifacts/report/dashboard.html artifacts/report/二级主题分布.csv
git diff --cached --stat
python scripts/check_public_files.py
git commit -m "feat: 两级主题中文输出与标签文档"
git push origin main
```

私有数据与权重已经列入 `.gitignore`，请保持这些排除规则，不要使用 `git add -f` 强行入库。

## 自动检查

`.github/workflows/ci.yml` 在推送和 PR 到 `main` 时检查：

- Python 3.10 / 3.12 的冒烟测试与两级中文标签回归。
- Ruff 代码检查。
- 8 个一级、67 个二级中文映射的完整性。
- 跟踪文件的私有产物路径和昵称、UID、评论标识特征。

CI 只安装 pandas、numpy、pyarrow 等最小依赖，不加载 Laya 或模型权重。模型路由由合成输入与模拟模型验证，不代表重新测得模型准确率。

工作流状态可查看：

```bash
gh run list --repo WanKun824/pbz-comment-insight --limit 5
```

## 可选：发布 GitHub Pages

Pages 工作流只在仓库已开启 Pages 后部署；未开启时跳过部署，不影响源码推送和 CI。

在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**，
然后在 Actions 页面手动运行 `pages` 工作流，或执行：

```bash
gh workflow run pages.yml --repo WanKun824/pbz-comment-insight
```

配置成功后的站点地址：<https://wankun824.github.io/pbz-comment-insight/>

工作流发布看板为首页，同时复制文档、评测报告和聚合 CSV。它只复制已跟踪的仓库文件，原始评论与可行动清单不会进入 Pages。

## 模型权重单独分发

当前发布：[v1.0.0](https://github.com/WanKun824/pbz-comment-insight/releases/tag/v1.0.0)。用户下载和调用步骤见 [模型下载与调用](模型下载与调用.md)。

维护者重新打包时使用：

```bash
python scripts/package.py --only weights
python scripts/package.py --verify dist/laya-pbz-v1-weights.zip
```

产物为 `dist/laya-pbz-v1-weights.zip` 和同名 `.zip.sha256` 文件。打包器只收集固定检查点的 8 个文件、许可、模型卡和两份操作文档，排除训练数据、快照及其他模型；生成完整 SHA-256 与逐文件 `manifest.json`，打包结束后自动复核。

发布时先将已测试的代码提交推送，再用新版本标签创建草稿 Release：

```bash
gh release create <新版本标签> --repo WanKun824/pbz-comment-insight --target <完整提交SHA> --draft --title "laya-pbz-v1 权重" --notes-file <发布说明文件>
gh release upload <新版本标签> dist/laya-pbz-v1-weights.zip dist/laya-pbz-v1-weights.zip.sha256 --repo WanKun824/pbz-comment-insight
gh release view <新版本标签> --repo WanKun824/pbz-comment-insight
gh release edit <新版本标签> --repo WanKun824/pbz-comment-insight --draft=false
```

每次新发布应同步文档中的固定版本下载链接，核对 Release 附件大小和 SHA-256，并实际解压运行示例。已有发布使用独立标签保留，不覆盖其附件。`full.zip` 含本地评论数据，仅用于内部复现，不作为公开 Release 附件。

## 文档入口

- [README](../README.md)：安装、命令、结果和限制。
- [模型下载与调用](模型下载与调用.md)：Release 附件下载、完整性校验、环境安装、单条与批量推理。
- [两级中文标签](两级中文标签.md)：完整中文标签对照、输出字段、旧预测迁移和模型兼容。
- [模型卡](MODEL_CARD.md)：训练方式、指标、偏差和适用范围。
- [标签体系设计](标签体系_v2.md)：历史设计依据与原始训练计划。
