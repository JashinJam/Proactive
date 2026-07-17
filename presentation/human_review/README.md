# U0/U1 人工盲评网页

该工具只读取 U0/U1 冻结盲文件，不加载 key、gold、模型或 GPU。视频通过 HTTP Range 传输；评分按 study 和 reviewer 分开写入 `output/human_reviews/`。

从项目根目录启动：

```bash
PYTHONPATH=src /home/lanjinxin/miniconda3/bin/python -m proactive_review.server
```

默认从 `8770` 端口开始；若被占用，会自动尝试后续十个端口。终端会打印实际地址。

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

