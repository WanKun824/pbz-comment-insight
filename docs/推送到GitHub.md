# 推送到 GitHub

仓库已在本地初始化并提交好（3 个提交，49 个跟踪文件，零个人数据）。
推送需要你的凭据，下面三条路任选一条。

## 先设你自己的 git 身份

当前提交用的是占位身份（`pbz-comment-insight <noreply@users.noreply.github.com>`）。
要换成你自己的：

```bash
cd D:/pythonProject/pbz-comment-insight
git config user.name  "你的名字"
git config user.email "你的GitHub邮箱"
git commit --amend --reset-author --no-edit      # 只改最后一个提交
# 或者改全部三个：
git rebase --root --exec "git commit --amend --reset-author --no-edit"
```

---

## 方案 A：gh CLI（最省事）

```bash
winget install GitHub.cli        # 或 scoop install gh
gh auth login                    # 浏览器授权

cd D:/pythonProject/pbz-comment-insight
gh repo create pbz-comment-insight --public --source=. --remote=origin --push \
  --description "B站游戏视频评论的多维结构化决策流水线（Laya 微调 / 影之刃零）"
```

**私有仓库**把 `--public` 换成 `--private`。

## 方案 B：网页建仓 + 命令行推送

1. 打开 <https://github.com/new>
2. 仓库名 `pbz-comment-insight`，**不要**勾选 "Add a README / .gitignore / license"（本地已有）
3. 然后：

```bash
cd D:/pythonProject/pbz-comment-insight
git remote add origin https://github.com/<你的用户名>/pbz-comment-insight.git
git push -u origin main
```

会弹出登录框。如果用 HTTPS，密码处要填 **Personal Access Token**（不是账号密码）：
<https://github.com/settings/tokens> → Generate new token (classic) → 勾 `repo` 和 `workflow`。

> `workflow` 这个 scope 是必须的 —— 仓库里有 `.github/workflows/`，
> 没有它推送会被拒。

## 方案 C：SSH

```bash
ssh-keygen -t ed25519 -C "你的邮箱"
cat ~/.ssh/id_ed25519.pub        # 贴到 https://github.com/settings/keys

cd D:/pythonProject/pbz-comment-insight
git remote add origin git@github.com:<你的用户名>/pbz-comment-insight.git
git push -u origin main
```

---

## 推送后：开启 Pages 发布看板

仓库里的 `.github/workflows/pages.yml` 会把
`artifacts/report/dashboard.html` 发成站点首页。只需开一次开关：

**Settings → Pages → Build and deployment → Source** 选 **GitHub Actions**

然后看板地址是：

```
https://<你的用户名>.github.io/pbz-comment-insight/
```

工作流只在 `dashboard.html` / `docs/**` 变化时触发，也可以在
Actions 页手动 `Run workflow`。

---

## 两个工作流在干什么

| 工作流 | 触发 | 内容 |
|---|---|---|
| `ci.yml` | push / PR to main | Python 3.10 + 3.12 跑冒烟测试（7 项，含 6 类 bug 回归）、ruff、标签体系自检、**隐私门禁** |
| `pages.yml` | dashboard 变化 | 把看板 + docs 发到 GitHub Pages |

CI **不需要** torch / laya —— 冒烟测试只验证配置、体系、键渲染、规则回归，
所以几十秒就跑完。

**隐私门禁**会在跟踪文件里出现评论原文 / `uname` / `uid` / 裸 rpid 时让 CI 失败。
本地预演：

```bash
python scripts/privacy_scan.py
```

---

## 权重怎么分发（615 MB，不能进 git）

`artifacts/ckpt/laya-pbz-v1/model.safetensors` 被 `.gitignore` 排除了。三个选择：

**1. GitHub Release**（推荐，单文件上限 2 GB）

```bash
python scripts/package.py --only weights       # 产出 dist/laya-pbz-v1-weights.zip (600 MB)
gh release create v1.0.0 dist/laya-pbz-v1-weights.zip \
  --title "laya-pbz-v1" \
  --notes "微调权重。解压到 artifacts/ckpt/laya-pbz-v1/ 后 \`pbz predict\` 即可用。指标见 docs/MODEL_CARD.md。"
```

**2. Hugging Face Hub**（更适合模型）

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload <你的用户名>/laya-pbz-v1 artifacts/ckpt/laya-pbz-v1 .
```

然后 `config.yaml` 里 `paths.ckpt` 可以直接写 HF id。

**3. Git LFS**（不推荐，免费额度只有 1 GB 存储 / 1 GB 月流量，这个权重一次就吃掉 60%）

```bash
git lfs install
git lfs track "artifacts/ckpt/**/model.safetensors"
# 还要先从 .gitignore 里去掉对应那行
```

---

## 推送前自检

```bash
cd D:/pythonProject/pbz-comment-insight
python scripts/privacy_scan.py          # 应为：含个人数据 29 个文件，全在忽略列表
python tests/test_smoke.py              # 应为 7/7 passed
python -m ruff check src tests scripts  # 应为 All checks passed
git status --short                      # 应为空
git log --oneline                       # 应有 3 个提交
```
