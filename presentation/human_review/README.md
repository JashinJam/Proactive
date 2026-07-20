# U0/U1 人工盲评网页

该工具只读取 U0/U1 冻结盲文件，不加载 key、gold、模型或 GPU。视频通过 HTTP Range 传输；评分按 study 和 reviewer 分开写入 `output/human_reviews/`。

推荐使用后台管理脚本启动。它在独立的 `tmux` 会话中固定端口并记录日志，关闭 SSH 终端不会停止服务：

```bash
cd /home/lanjinxin/workspace/wearable_ai_challenge
bash scripts/human_review_server.sh start 8770
```

检查、查看日志和停止：

```bash
bash scripts/human_review_server.sh status
bash scripts/human_review_server.sh log 100
bash scripts/human_review_server.sh attach
bash scripts/human_review_server.sh stop
```

需要前台调试时才直接运行：

```bash
PYTHONPATH=src /home/lanjinxin/miniconda3/bin/python -m proactive_review.server
```

`attach` 进入服务器控制台，按 `Ctrl+B` 再按 `D` 可退出但不停止服务。直接运行时默认从 `8770` 端口开始并可能自动尝试后续端口；管理脚本使用 `--strict-port`，端口冲突会直接报错，不会静默切换到 `8771`。

如果服务器网络允许入站访问，可以启动浏览器直连模式：

```bash
bash scripts/human_review_server.sh start-public 8770
bash scripts/human_review_server.sh status
```

随后在本地打开 `http://<平时 SSH 使用的服务器地址>:8770`。浏览器首次要求认证时，用户名固定为 `review`，密码由 `status` 命令显示。直连模式监听 `0.0.0.0` 并强制启用认证；是否能从本地连通仍取决于服务器上游防火墙或机房端口策略。

远程服务器使用 SSH 转发：

```bash
ssh -L 8770:127.0.0.1:8770 <用户名>@<服务器地址>
```

每个 session 确认后写入：

```text
output/human_reviews/u0/reviewer_A/ratings.json
output/human_reviews/u0/reviewer_A/ratings.csv
output/human_reviews/u1/reviewer_A/ratings.json
output/human_reviews/u1/reviewer_A/ratings.csv
```

评审 B 使用对应的 `reviewer_B/` 目录。网页中的“导出 CSV”下载当前评审员的同一份 CSV。

完整细则见 `reports/20260717_u0_u1_human_review_protocol.md`。
