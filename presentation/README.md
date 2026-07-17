# EgoProactive PWR-inspired 阶段汇报

本目录包含约 15 分钟的阶段性工作汇报和配套实验结果网页。

## 主要入口

- `PWR_progress_report_15min.ipynb`：展示用 Jupyter Notebook。
- `PWR_progress_report_15min_source.md`：Notebook 的可维护源稿。
- `results_dashboard/`：实验结果、混淆矩阵和视频时间轴网页。
- `build_assets.py`：从冻结实验产物重建网页数据和 Notebook。
- `serve.py`：只读静态网页与视频 Range 服务。
- `human_review/`：U0/U1 双人独立盲评网页；配套写入服务位于 `src/proactive_review/`。

## U0/U1 人工评测网页

从项目根目录启动：

```bash
PYTHONPATH=src /home/lanjinxin/miniconda3/bin/python -m proactive_review.server
```

默认从 `8770` 开始选择空闲端口，终端会打印实际地址。评审结果按 U0/U1 和 A/B 评审员隔离保存到 `output/human_reviews/`；服务不会加载 blind key。详细操作和数据路径见 [human_review/README.md](human_review/README.md)，评分协议见 [U0/U1 双人独立盲评执行细则](../reports/20260717_u0_u1_human_review_protocol.md)。

## 重建

在项目根目录执行：

```bash
/home/lanjinxin/miniconda3/bin/python presentation/build_assets.py
```

脚本会校验数据行数、预测对齐和每个 session 的 chunk 数，随后生成：

```text
presentation/PWR_progress_report_15min.ipynb
presentation/results_dashboard/data.js
presentation/results_dashboard/data_manifest.json
```

Notebook 的默认视频样例使用三个真实关键帧。需要重建该图片时执行：

```bash
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python presentation/build_demo_asset.py
```

## 手动启动结果网页

第一步，进入项目目录：

```bash
cd /home/lanjinxin/workspace/wearable_ai_challenge
```

第二步，正常情况下可直接启动；只有实验产物或 Notebook 源稿更新后才需要重新执行 `build_assets.py`：

```bash
/home/lanjinxin/miniconda3/bin/python presentation/build_assets.py
```

第三步，启动只读网页和视频服务：

```bash
/home/lanjinxin/miniconda3/bin/python presentation/serve.py
```

终端出现以下内容表示启动成功：

```text
Dashboard: http://127.0.0.1:8766
Serving .../presentation/results_dashboard
Video source: .../data/egoproactive/val
```

第四步，在浏览器打开：

```text
http://127.0.0.1:8766
```

样本时间轴可以直接打开：

```text
http://127.0.0.1:8766/#samples
```

服务不加载模型、不使用 GPU，也不会重跑实验。保持启动终端不关闭；演示结束后按 `Ctrl+C` 停止。

### 远程服务器

如果通过 SSH 登录服务器，在本地电脑建立端口转发：

```bash
ssh -L 8766:127.0.0.1:8766 <用户名>@<服务器地址>
```

随后仍在本地浏览器打开 `http://127.0.0.1:8766`。使用 VS Code Remote 时，也可以在 `PORTS` 面板转发服务器端的 `8766` 端口。

### 端口已占用

如果终端提示端口已被占用，可以改用下一个端口：

```bash
/home/lanjinxin/miniconda3/bin/python presentation/serve.py --port 8767
```

此时浏览器地址相应改为 `http://127.0.0.1:8767`。

视频不复制到 `presentation/`。服务只读映射公开验证集 `data/egoproactive/val/*.mp4`，并支持 HTTP Range，因此点击 chunk 后可以跳转到对应时间。

## 证据边界

- 论文信息来自各论文的官方 arXiv 页面和本项目逐篇审计。
- 项目结果来自冻结的本地实验产物；完整公共验证集包含 700 sessions / 9,935 chunks。
- D1、D2 及 R0-F 均使用了公开验证标签进行训练、阈值选择或规则选择，必须标为 `val-supervised`。
- R1 仅有 4 sessions / 50 chunks，是协议试验，不能与完整集结果直接横向比较。
- D1 OOF `0.6341` 是当前科学基线；单一部署阈值的 OOF 模拟是 `0.6330`；全量重拟合 `0.6719` 仅是 train-fit sanity，不用于汇报泛化性能。
