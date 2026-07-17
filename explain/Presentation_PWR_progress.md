# 15 分钟阶段汇报与网页演示说明

## 1. 汇报目标

这次汇报不是逐项罗列代码提交，而是说明一条完整研究决策链：

```text
主动视频文献进阶检索
  -> 为什么选择 PWR-inspired 路线
  -> 如何适配 C1 Small 与公开数据条件
  -> 怎样建立可复现的生成和决策系统
  -> 做了哪些受控实验
  -> 哪些结论通过，哪些尝试被否决
  -> 下一阶段为什么这样安排
```

展示用 Notebook：

[`PWR_progress_report_15min.ipynb`](../presentation/PWR_progress_report_15min.ipynb)

可维护源稿：

[`PWR_progress_report_15min_source.md`](../presentation/PWR_progress_report_15min_source.md)

结果网页：

[`results_dashboard/index.html`](../presentation/results_dashboard/index.html)

## 2. 15 分钟时间安排

| 环节 | 建议时间 | 重点 |
|---|---:|---|
| 标题与结论预告 | 0:40 | 0.6341 OOF、相对 R0 +0.1711、等价加速 9.15% |
| 文献检索地图 | 1:10 | 四篇工作解决的不是同一个子问题，论文分数不能直接横比 |
| MMDuet2 / StreamPro / R3 | 1:15 | 说明可借鉴机制及不能原样部署的原因 |
| C1 任务约束 | 1:05 | interrupt/silent、Macro F1、因果输入、Small 2B |
| PWR 方法 | 1:30 | 显式过程状态、planner/interaction 分工、OOP recovery |
| 选择 PWR 的理由 | 1:00 | 任务匹配、可证伪性和 Small 适配 |
| 当前路线 | 1:00 | R0、R0-F、R1、D1、D2 依次回答什么问题 |
| 工程实现 | 1:25 | InternVL3.5-1B、因果输入、模块边界和标签隔离 |
| D1 特征设计 | 1:00 | 18 scalar + 1 margin + 1,024 hidden |
| 五折 OOF 与消融 | 1:00 | session split、3/1/1 folds、推广门槛 |
| 实验结果 | 1:30 | 校准是最大增量；融合有互补增量；D2 被否决 |
| 视频样例 | 1:15 | 时间轴、utterance、修复与剩余错误 |
| 等价加速与下一步 | 1:05 | shared vision、表示适配、状态复验、RL 触发条件 |

正常语速合计约 15 分钟。展示用 Notebook 只保留台面内容，不包含讲述提示；本文件作为单独的备稿和术语解释材料使用。

## 3. 文献部分应如何讲

### MMDuet2

MMDuet2 将主动交互统一成文本对话：每轮模型生成回答或 `NO REPLY`。它用 SFT 建立格式，再用基于 PAUC 的多轮 GRPO 奖励正确、尽早且不重复的回答。

对本项目的价值是及时性奖励、重复约束和多轮轨迹训练方法。它不是当前主路线，因为基座是 3B，超过 Small 的 2B 限制；训练数据和主动 VideoQA 语义也不同于当前 procedural intervention。

### StreamPro

StreamPro 的关键观察是 streaming trajectory 中 silence 远多于 response，普通交叉熵会把模型推向过度沉默。它先用 CB-Stream Loss 平衡两类控制 token，再用单轮和整条轨迹两级 reward 做 GRPO。

对本项目的价值是类不平衡和轨迹一致性训练。但它使用 3B/4B 模型和很高的训练预算，benchmark 也包含风险预测等不同任务，因此更适合作为后续训练目标参考。

### R3-Streaming

R3 分成 Remember、Respond、Reason：压缩旧记忆、判断证据是否足够、再决定是否调用慢模型。它提醒我们，响应门控、视觉记忆和计算路由是不同问题。

当前不采用 fast/slow 双模型，但保留两点：近期视觉证据应高保真；readiness 或 interrupt head 应与自由回答生成解耦。

### PWR

PWR 与当前赛道最直接对应：第一人称操作任务、步骤指导、何时打断、用户偏离和恢复。核心不是一个 gate，而是显式 procedural state 和 planner/interaction 分工。

当前没有公开获得官方训练代码、模型权重、Pro²Bench 训练标注以及 gold plan/cue targets。因此我们的路线是 PWR-inspired Small reimplementation，不是 official PWR reproduction。

## 4. 当前工作应如何概括

### R0

冻结 InternVL3.5-1B，以当前及过去视频、query 和官方 `dialog[i]` 做因果生成。完整 700 sessions / 9,935 chunks 的 Macro F1 是 0.4630。主要问题是 interrupt recall 低，同时有 633 个非空回答缺少合法 tag，被官方 scorer 按 silent 处理。

### R0-F

只修复冻结 raw response 的意图格式，不重跑模型。Macro F1 提升到 0.5362。这说明自由生成的格式问题与真正的二元决策问题必须分开。

### R1

在 4 sessions / 50 chunks 上测试 null、step、cues、full state。Full state 提高 interrupt recall，但降低 silent recall，没有带来 Macro 增益。它只是未观察到正信号的协议试验，样本量不足以证明程序状态没有价值。

### D1

将回答内容生成和 interrupt/silent gate 分开。每个 chunk 使用：

```text
18 个严格因果标量
+ 1 个 fixed-tag margin
+ 1,024 维最终 causal hidden
= 1,043 维输入
```

线性头包含 1,043 个权重和 1 个 bias，共 1,044 参数。五折按照 session 划分，每轮用三折拟合、一折选择阈值、一折只做测试。

D1 fused 的 OOF Macro F1 是 0.6341。它不是 hidden-only 结果：tag-only 为 0.5313，hidden-only 为 0.6031，scalar+tag 为 0.6172。正结果来自标量、标签 margin 和 hidden 的互补融合。

### D2

width-8 residual MLP 达到 0.6351，只比 D1 高 0.0010；bootstrap 区间跨零，且只有 3/5 folds 改善，因此按预设门槛否决。不能因为单点数字略高就替换 D1。

## 5. 网页演示流程

进入项目目录：

```bash
cd /home/lanjinxin/workspace/wearable_ai_challenge
```

启动服务：

```bash
/home/lanjinxin/miniconda3/bin/python presentation/serve.py
```

打开：

```text
http://127.0.0.1:8766
```

保持该终端运行，演示结束后按 `Ctrl+C` 停止。该服务不会加载模型或占用 GPU。

远程 SSH 使用：

```bash
ssh -L 8766:127.0.0.1:8766 <用户名>@<服务器地址>
```

端口被占用时可以将启动参数改为 `--port 8767`，浏览器地址同步改为 `http://127.0.0.1:8767`。

建议按以下顺序演示：

1. 在“总体进展”展示 R0 → R0-F → D1 scalar → D1 fused。
2. 在“实验诊断”选择 D1 fused，解释 TP/FP/TN/FN 和两类 recall。
3. 查看 D1 消融，说明 hidden 或 tag 单独都不足。
4. 进入“样本时间轴”，点击默认的“修复漏报”样例。
5. 点击时间轴 chunk，视频会跳到对应绝对时间。
6. 切换 R0-F 与 D1 fused，观察 gate 如何改变。
7. 最后打开“剩余漏报”，说明下一阶段仍需步骤完成与偏离证据。

## 6. 默认样例的讲法

默认样例是 session 143：

```text
Task: Replacing batteries in thermostat
Video: 359ed9ce38fdf4dc.mp4
Duration: 140.9 s
Chunks: 22
```

D1 修复了 R0-F 的 10 个 interrupt 漏报。金标中包括：

- 打开恒温器外壳；
- 从不合适的黄油刀切换到平头螺丝刀；
- 取下旧电池；
- 发现电池装反后翻转并重新插入；
- 重新闭合并测试设备。

它与 PWR 的 Out-of-Plan recovery 高度相关。但当前 D1 主要解决打断时机，部分预测话语仍是通用的 `Please continue with the next step.`。汇报时应明确：决策指标已经提升，不代表恢复内容质量已经解决。

## 7. 等价加速的讲法

原始 D1 对 `$silent$` 和 `$interrupt$` 两个固定候选分别重复视觉编码。`shared_vision` 只把这两个候选完全相同的视觉特征计算一次，然后保留两次原始 batch-one 语言前向。

127 chunks 上：

```text
wall time: 500.892 s -> 455.056 s
improvement: 9.15%
peak memory change: 0 B
hidden / margin / decision / answer: 127/127 一致
```

batch-of-two 虽然减少了 API 调用次数，但仍计算两份长序列和视觉输入，最终更慢、更占显存。Prefix cache 改变 BF16/SDPA 的计算形状，tag margin 最大漂移 0.113382，也没有加速，因此两者均被否决。

## 8. 最后的证据边界

- 0.6341 是 public-validation-supervised 的 session-level OOF 结果。
- 统一部署阈值的 OOF 模拟是 0.6330，并通过稳健性门槛。
- 全量重拟合 0.6719 是 train-fit sanity，不能当泛化分数。
- R1 只有 50 chunks，不能与完整集结果直接比较。
- 当前没有 hidden-test 或 leaderboard 提交成绩。
- 下一步不是默认启动 GRPO，而是先完成冻结的表示适配 OOF；若无增益，再做更大规模、预注册的状态价值复验。
