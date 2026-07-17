# Challenge 1 (EgoProactive) — 文献综述 v2

整理时间：2026-07-09（更新自 2026-06-25 v1）
覆盖 C1 方向共 22 篇论文，按技术路线分组。

> **2026-07-13 路线校正**：已完成 PWR v1 正文和 Appendix A--J 的逐节审计，并核查官方训练代码、权重与数据发布状态。详见 [PWR 全文、附录与公开产物正式审计](papers/challenge1_proactive/PWR_audit.md)。PWR 暴露出本综述此前对显式 procedural state、visual cue、OOP recovery 和 plan quality 覆盖不足；在完成新版路线重排前，本文原有的“综合可用性/优先级”判断只作为历史记录，不再视为当前结论。

---

## 目录

- [1. 任务定义](#1-任务定义)
- [2. 概率门控路线 (W2T)](#2-概率门控路线-w2t)
- [3. 独立决策头路线](#3-独立决策头路线)
- [4. RL-based 主动交互路线](#4-rl-based-主动交互路线-2026-新范式)
- [5. 数据驱动 / 反事实增强路线](#5-数据驱动--反事实增强路线)
- [6. 评测基准与工程框架](#6-评测基准与工程框架)
- [7. 特定场景应用](#7-特定场景应用)
- [8. 开源情况汇总](#8-开源情况汇总)
- [9. 核心方法对比与趋势](#9-核心方法对比与趋势)

---

## 1. 任务定义

**C1 EgoProactive**：给定长第一人称视频流 + 用户需求，模型在每个 ~8s chunk 结束时判断主动介入（`$interrupt$<utterance>`）或保持沉默（`$silent$`）。

- 输入：流式第一人称视频（因果，无未来帧）+ 单次用户查询 + 累积对话历史
- 输出：每个 chunk 的二分类决策 + 介入时的指导文本
- 评测：Macro F1（仅评估决策时机正确性，不评估内容质量）
- 数据：验证集 700 sessions / 9935 chunks / 53.9% interrupt ratio

核心挑战：**何时说话（When to Speak）** 比 **说什么（What to Say）** 更难——需要同时具备时序感知、任务步骤理解、沉默克制和介入勇气。

---

## 2. 概率门控路线 (W2T)

核心思路：在每帧（或每 chunk 最后一帧）的 visual token 位置计算 EOS/silence token 的 softmax 概率，超过阈值则沉默，否则生成文本。**不需要额外模型结构**，但依赖阈值标定。

### 2.1 ProAssist (EMNLP 2025) — W2T 路线的代表

- 作者：Yichi Zhang, Seungwhan Moon et al. (Meta, UMich)
- 链接：[arxiv.org/abs/2506.05904](https://arxiv.org/abs/2506.05904)
- 代码/数据：[pro-assist.github.io](https://pro-assist.github.io/) | HF: `594zyc/ProAssist-Dataset`

**架构**：SigLIP-SO400M (frozen, 384×384) → 2-layer MLP Projector → LLaMA-3.1-8B-Instruct (LoRA r=128, α=256)。每帧 5 visual tokens (CLS + 2×2 patches)，2 FPS。

**W2T 机制**：在每段 frames 的最后一帧的最后一个 visual token 位置，计算 EOS token 的 softmax 概率。P(EOS) > θ → 沉默；否则自回归生成文本。中间帧全学 EOS（因为只有最后一帧有 assistant 回复）。

**关键技术**：
- NFS (Negative Frame Sub-sampling, ρ=0.1)：随机采样沉默帧参与训练，缓解 EOS:BOR 极端不平衡（原生 ~97:3）
- IPS (Iterative Progress Summarization)：序列超过 4096 token 时生成进度摘要并重新注入 system prompt

**训练数据**：30K 对话、479h 视频，覆盖 6 个第一人称数据集（ego4d 44%、holoassist 23%、egoexolearn 11%、epicKitchens 11%、assembly101 7%、wtag 3%）。含 `no_talk` user type（~20%），这些用户不追问。

**我们的实验发现（Phase 1 微调报告）**：

| 实验 | 结果 |
|------|------|
| 零样本 (θ=0.9) | Macro F1=0.543, Int F1=0.474, Sil F1=0.611 |
| LoRA 微调 NFS=0.5 | **全静默**（EOS:BOR=13:1 → 模型学会永远预测 EOS） |
| LoRA 微调 NFS=0.05 | Macro F1=0.554, Int P +5.4%, Int R=44.6% |
| NFS=0.05 + LR=2e-4 | **灾难性遗忘**（92% interrupt 输出为重复 header token） |

核心教训：(1) θ 值跨域不可迁移——ProAssist 自身最优 θ=0.3-0.5，C1 最优 θ=0.9，差 3 倍；(2) 单靠 NFS 调参无法根本解决"何时说话"的时序决策问题；(3) 高 LR 下全 LoRA 模块微调会冲刷文本生成能力。

**开发现状**：代码完整开源，权重 HF 可直接下载，数据集已全部下载并解压（143GB）。

---

### 2.2 LiveStarPro (NeurIPS 2025 / arXiv 2026.06) — 零参数困惑度门控

- 作者：Zhenyu Yang et al.
- 发表：NeurIPS 2025 (LiveStar) + arXiv 2026.06 (LiveStarPro 扩展)
- 链接：[arxiv.org/abs/2606.17798](https://arxiv.org/abs/2606.17798)
- 代码/权重：[github.com/sotayang/LiveStarPro](https://github.com/sotayang/LiveStarPro) | HF: `yzy666/LiveStar_8B`
- 数据集：HF: `yzy666/OmniStar-RNG`（10,137 视频, 15 场景）

**SVeD (Streaming Verification Decoding)**：零参数、零额外延迟的激活决策方案。每来一帧，模型计算对预设回应提示的 token 级困惑度。困惑度低（模型确定该说什么）→ 触发；困惑度高（不确定）→ 沉默。完全不需要训练 silence/response 分类器。

```
对比：
  STRIDE:      需要额外的 2B 扩散模块（~113ms 延迟, 额外显存）
  StreamBridge: 需要额外的 0.5B 小模型（独立推理开销）
  SVeD:        零额外参数, 零额外延迟（只用已有 forward pass 算困惑度）
```

**SCAM (Streaming Causal Attention Mask)**：训练时用可变长度流式片段 + 严格因果注意力掩码。第 t 帧只能 attend 第 1~t 帧，片段之间不共享注意力。迫使模型学会从部分观测做增量推理。

**TSHM (Tree-Structured Hierarchical Memory)**：树形层级记忆替代 FIFO 丢弃。新帧→短期缓存（叶子）→缓存满时相邻叶子压缩为事件摘要（中层）→多个事件摘要进一步压缩为任务阶段描述（上层）。把"遗忘"变成"压缩"。

**实验结果**：语义正确性 +28.9%，时机误差 -18.2%（vs 已有在线 Video-LLM）。推理速度 1.58×。LoRA 仅需 2×A800。

**我们的实验发现**：SVeD smoke test (1 sample, 10 chunks) — Naive all-generate → Macro F1=0.33；SVeD gating → Macro F1=0.70。但 LiveStar 输出偏第三人称视频描述而非第二人称程序性指导，需微调纠正。

**与 C1 的关系**：SVeD 是成本最低的激活决策方案。TSHM 树形记忆解决长视频中"10 分钟前的操作步骤如何影响当前介入决策"。SCAM 是严格的因果训练策略。C1 方向中**综合可用性最高**的新工作。

**开发现状**：代码完全开源，权重 HF 可直接下载，数据集已发布。

---

## 3. 独立决策头路线

核心思路：将"是否说话"的决策从主 LLM 中分离出来，通过一个独立的模块（外部小模型 / 内部 MLP 头 / 特殊 token）做二分类。

### 3.1 STRIDE (arXiv 2026.03) — 序列去噪激活检测

- 作者：Junho Kim, Hosu Lee et al. (UIUC, KAIST, Google DeepMind)
- 链接：[arxiv.org/abs/2603.27593](https://arxiv.org/abs/2603.27593)
- 代码/权重：[github.com/interlive-team/STRIDE](https://github.com/interlive-team/STRIDE) | HF: `interlive/STRIDE-2B`

**核心创新：把激活决策建模为序列去噪而非逐点分类。**

正确"介入时机"不是孤立的点，而是连续区间（onset → sustained → offset）。在一个滑动窗口（长度 W）内维护激活序列 `[a_1,...,a_W] ∈ {0,1,[MASK]}`，通过类扩散的迭代过程从全 [MASK] 逐步还原出干净的 0/1 序列。

**训练：边界感知的 Span Masking**。不随机 mask 单点，而是随机掩盖**连续区间**，迫使模型学习从视觉上下文恢复完整激活区间。Loss 仅在被 mask 位置上计算交叉熵。双向 mask predictor（非 causal）。

**推理：K 步迭代去噪**。初始全 [MASK] → 预测所有位置 → 高置信度接受（0 或 1）→ 低置信度重新 mask → 利用已揭开的邻居再预测 → K 步后得到干净序列。检测到连续 1-span 超过阈值 → 触发 → 累积帧缓存传入 Stage 2。

**滑动窗口的状态传递**：上一窗口结尾的确定状态传递到下一窗口，只对新进入帧做 mask。保证跨窗口决策连续，避免边界断裂。

**两阶段架构**：Stage 1: STRIDE-2B 激活检测器（~113ms 延迟）→ Stage 2: 下游 Video-LLM（冻结，即插即用）。

**与 C1 的关系**：STRIDE 的 Stage 1 激活检测器可直接用于 `$interrupt$` vs `$silent$`。Chunk 级决策天然匹配 span-based 激活建模。9935 chunk 标注可用于训练激活预测头。

**开发现状（更新：2026-07-03）**：推理权重已发布，推理代码开源。**训练代码已确认存在**（`stride/train/trainer.py`, `stride/train/run.py`, `scripts/prepare_activation_dataset.py`），非之前以为的纯 TODO。支持 Qwen3-VL/Gemma3/InternVL 三种 backbone。

---

### 3.2 StreamBridge (NeurIPS 2025) — 解耦激活模型

- 作者：Haibo Wang et al. (Apple, Fudan)
- 链接：[arxiv.org/abs/2505.05467](https://arxiv.org/abs/2505.05467)

**核心思路：三个模块，两个不改。** 不改主 Video-LLM，只外加两个组件。

**Memory Buffer（生产者-消费者模式）**：视觉编码器持续编码并推入缓冲区；主 LLM 平时休眠，收到触发信号时才从缓冲区取数据推理；生成的回复也追加回缓冲区作为多轮对话历史。

**Round-Decayed Compression（轮次衰减压缩）**：最大 token 预算下，从最早对话轮次的旧帧开始做相邻帧平均池化，越旧压缩越狠，越新保持越完整。类似人记忆：刚发生的事细节清晰，半小时前只有模糊轮廓。

**Decoupled Activation Model（解耦激活模型）**：轻量 MLLM（如 LLaVA-OV-0.5B）与主 LLM 并行运行。去掉文本生成头换成 **score head（二分类打分头）**。每帧视觉 token 后追加可学习 `[ACT]` token → 隐层向量经 score head → 标量分数 > 阈值 → 触发。与主 LLM 完全解耦，即插即用。

**训练技巧**：只把每段视频的**最后 P% 帧**标为正样本（P ∈ 0%-50%），早期帧标为负样本防止过早触发。Random QA Drop + QA Interval Shift 数据增强。

**实验结果**：OVO-Bench 上 StreamBridge+Qwen2-VL 达 71.30（GPT-4o 为 64.46）。延迟平坦（~3s），离线基准不降反升。

**与 C1 的关系**：激活模型的解耦设计可直接用于 interrupt/silent 决策。Round-Decayed Compression 对 10+ 分钟长视频流直接可用。需额外处理：C1 的介入判断基于"步骤正确性"，比 StreamBridge 的"视频事件发生"更难。

**开发现状**：代码和权重均未公开（Apple 团队）。方法细节在论文中，需自行复现。

---

### 3.3 Proact-VL (ICML 2026) — FLAG token + 内部解耦

- 作者：Weicai Yan et al. (ZJU, MSRA)
- 链接：[arxiv.org/abs/2603.03447](https://arxiv.org/abs/2603.03447)

**核心创新**：不在 LLM 外挂模块，而是在**主模型内部**植入 `<|FLAG|>` token——一个只做决策、不参与文本生成的特殊 token。

流式 chunk 输入 → `<|FLAG|>` token 的隐层向量 h_t → 轻量 MLP 二分类头 → p_t = σ(MLP(h_t)) → p_t > τ 则生成文本，否则 `<silence>`。

**训练目标：多级联合损失**：
- L1（LM loss）：标准交叉熵，管"说得对不对"
- L2（过渡平滑分类损失）：管 silent↔speaking 状态切换的质量，不平滑会导致 flickering
- L3（正则化项）：约束长时间内说话频率，防止话痨或过度沉默

L2+L3 共同保证"在正确时机介入，且不过度介入"。这是 ProAssist 缺乏的——ProAssist 只有 L1，说话频率完全依赖推理时的 θ 调节。

**训练数据**：Live Gaming Benchmark（561h, 12 游戏, 128K 样本）。数据按说话频率分层（0-30%/30-70%/70-100%），覆盖稀疏到密集的回应模式。

**实验结果**：平均响应延迟 0.35s，时机判断 F1=64.87。跨域泛化到 Ego4D 仍保持连贯性。

**与 C1 的关系**：FLAG token + MLP 决策头的端到端内部解耦方案，比 StreamBridge 的外挂方案更简洁（只需训一个模型）。场景差异：Proact-VL 是游戏解说（信息密度高），C1 是任务指导（信息密度低），说话频率分布差异大，需调整 L3 和 τ。

**开发现状**：匿名仓库 `anonymous.4open.science/r/Proact-VL-8699/`（ICML 审稿期临时链接），正式 GitHub 未建。权重未公开。

---

### 3.4 MMDuet2 (ICLR 2026) — RL 训练的"NO REPLY" token ★新增

- 作者：Yueqian Wang et al. (北京大学王选所, 美团)
- 链接：[arxiv.org/abs/2512.06810](https://arxiv.org/abs/2512.06810)
- 代码：[github.com/yellow-binary-tree/mmduet2](https://github.com/yellow-binary-tree/mmduet2)

**核心创新：纯文本 Chat Template 实现"何时说话"——不需要额外架构、不需要手动调阈值、不需要精确时间戳标注。**

**Chat Template 设计**（关键机制）：
```
User:  <image><image>                    ← 每轮 1-2 帧 (2s 间隔)
Assistant: NO REPLY                      ← 沉默
User:  <image><image>
Assistant: People are working at desks  ← 说话
```

整个交互过程（视频输入、回复决策、内容生成）全部格式化为 user/assistant 消息。`"NO REPLY"` token 就是沉默——不需要 EOS-prob、不需要外挂模块、不修改模型架构。这与 C1 的 `$silent$` vs `$interrupt$<text>` 格式完全对应。

**SFT 阶段的问题与 RL 的动机**：
- SFT 后回复频率偏低（"NO REPLY"占绝大多数 turns）
- 回复滞后（往往在关键信息出现数秒后才响应）
- 标注"在哪个精确帧该说话"极其困难——场景级 caption（如 "Tamarind, fish sauce and sugar are added to a heated pan"）无法告诉模型具体哪个帧每种食材出现

**解决方案：RL + PAUC 奖励**。虽然标注精确时间戳难，但判断"两个交互输出哪个更好"容易。PAUC（Proactive Area Under Curve）指标绘制"时间-质量"曲线：
- 同一个时间点，正确性更高的回复→曲线上升→面积更大
- 同样的正确性提升，更早出现的回复→曲线更早上升→面积更大

用 GRPO + 基于 PAUC 的复合奖励训练，模型**自己学会**在最早能正确回答的时刻说话。

**复合奖励函数**：

| 奖励项 | 权重 | 功能 | 对应解决 C1 的哪个问题 |
|--------|:--:|------|------|
| r_PAUC | 3 | 鼓励正确且及时地说话 | Interrupt Recall 不足 |
| r_rep | 2 | 抑制冗余重复回复 | 灾难性遗忘（重复 header token） |
| r_in_span | 0.5 | 确保回复在正确时间片段内 | 误报（不该说时说） |
| r_pfx | 2 | 防止先逐字重复再生成新内容 | 微调后文本生成退化 |

**RL 训练的时序信用分配**：每次只取 20-60s 的短片段训练（而非整个视频），在片段前提供 ground-truth 对话上下文。这避免了稀疏奖励和时间信用分配问题。

**训练数据**：
- 52K 视频（50K YouTube + 2.5K ego-centric from Ego-Exo4D & EgoExoLearn）
- 两种对话类型：1QnA（视频开始一个问题，多时间片段各回答）和 nQnA（多问题多答案，用户可随时提问）
- SFT：16 H800, ~8h；RL (GRPO)：8 H800, ~20h

**关键实验发现**：
- ProactiveVideoQA [WEB] PAUC=53.3 vs MMDuet 38.9 (↑37%)
- 消融：去掉 r_rep → 重复率从 4.2% 升至 17.3%（模型会通过大量重复输出来换取虚高 PAUC）
- 去掉 r_in_span → [EGO] 上直接 FAIL（几乎每 turn 都回应，单样本推理 >20 分钟）
- RL 训练三阶段：过渡期（0-180步，性能下降）→ 成长期（180-450步，快速提升）→ 平台期（450-489步）
- 离线视频理解基准（Video-MME, MVBench, LongVideoBench）性能不退化

**与 C1 的关系（高度契合）**：

| 维度 | MMDuet2 | C1 | 迁移难度 |
|------|---------|-----|:--:|
| 输出格式 | "NO REPLY" vs text | `$silent$` vs `$interrupt$<text>` | **极低**（直接映射） |
| 输入格式 | 每 turn 1-2 frame, `<image>` | 每 chunk 多 frame | 低（适配帧数） |
| 基础模型 | Qwen2.5-VL 3B | 当前 ProAssist (LLaMA-3.1-8B) | 中（可换底座） |
| 训练方式 | SFT + GRPO RL | 当前仅 SFT (LoRA) | 中（需引入 RL 管线） |
| 阈值依赖 | **无**（RL 内化决策） | 当前需人工扫 θ | **重大改进** |
| 代码 | 完整开源 | — | 可直接参考复现 |
| 推理效率 | 3.3 turns, 2m52s（64 样本 on H100） | C1 700 samples × ~14 chunks | 可接受 |

**对我们方案的核心启示**：
1. "NO REPLY" token 路线可以完全替代 ProAssist 的 W2T 机制——不需要手动调 θ，不需要担心跨域迁移
2. PAUC 复合奖励直接针对我们遇到的三个核心问题（Recall 不足、重复输出、文本退化）
3. 其 RL 训练框架可适配 C1 的 chunk 级二元决策——只需要把 "NO REPLY" 替换为 `$silent$`
4. 代码完整开源在 GitHub，意味着我们可以直接参考其 GRPO + PAUC 奖励的实现

---

### 3.5 R3-Streaming (arXiv 2026.05) — 三阶段级联控制框架 ★新增（已读原文）

- 作者：Jinming Liu et al. (SJTU, 东方理工, MSRA)
- 链接：[arxiv.org/abs/2605.17921](https://arxiv.org/abs/2605.17921)
- 原文：`literature/papers/challenge1_proactive/R3-Streaming.md`（560 行完整论文）

**核心创新：将流式视频理解重新定义为级联控制问题——Remember（记忆压缩）→ Respond（就绪判断）→ Reason（计算路由）。每步决策基于前一步的精化信息状态，形成递进依赖链。**

**两个关键实证发现（驱动 R3 设计的动机）**：

1. **Finding 1（历史 token 是注意力噪声）**：通过 JSD 删除分析发现，历史 token 获得了大部分视觉注意力，但删除它们对输出分布的影响远小于删除近期 token。**激进压缩历史帧反而提升性能**——在 StreamingBench 上，Historical=0.01 + Nearby=1.0 比无压缩高 +2.4 分。

2. **Finding 2（模型规模增益非单调）**：小模型在某些流式任务上可以匹敌或超越大模型。Qwen2.5-VL-3B 在部分任务上超过 7B 模型。说明"永远调用大模型"不仅浪费计算，在某些任务上还更差。

**Remember（Active Forgetting，训练无关）**：
将视频历史按时间窗口 W 分为两个区域，分别压缩：
- Nearby zone（`x_{t-W+1:t}`）：保留高保真度，压缩阈值 τ_near >> τ_hist
- Historical zone（`x_{1:t-W}`）：激进压缩为紧凑的 episodic slots
- 默认：Nearby threshold=1.0（不压缩），Historical threshold=0.01（99% 压缩），Nearby window=3 帧
- 压缩算子：使用 TimeChat-Online 的 DTD 算子（消融表明算子选择不重要，age-aware 策略是关键）
- Token 减少 95-96%

**Respond（Proactive Response，轻量就绪头）**：
- 一个轻量 readiness head h 估计当前证据是否足够支撑可靠回答：`p_ready = h(q_t, M_t)`
- p_ready < 0.5 → 输出 `<Routine>`（延迟回答）；否则继续到 Reason 阶段
- **训练方式**：在 Reason SFT + TB-GRPO 完成后，冻结 fast VLM，仅训练 readiness head
- **数据构造（Decision-Boundary-Focused Sampling）**：聚焦在答案可回答性严格取决于特定视觉线索到达的时间段。在 clue timestamp t_c 处：前 3 帧标注为 `<Routine>`（证据不足），t_c 及后 2 帧标注为 Ready。强制模型学会区分线索出现前/后的精确边界
- StreamingBench Proactive split 上从 0.216→0.328（+52%）

**Reason（Adaptive Thinking，TB-GRPO 路由）**：
- Fast model 决定：直接输出 `<Answer>`（生成回答）或 `<Escalate>`（调用 slow/thinking model）
- **SFT Cold-Start**：多响应采样管线。对每个 query，fast model 采样 K=4 个响应 → LLM 打分（开放域）或 exact match（客观题）→ 聚合平均分 s̄ → s̄ ≥ T=2.5 则 `<Answer>`，否则 `<Escalate>`
- **TB-GRPO（Target-Balanced GRPO）**：标准 GRPO 在二值路由上迅速坍塌到"永远 `<Escalate>`"（40 步内 ρ→1.0）。TB-GRPO 引入目标带宽控制 (η=0.3, γ=0.2)：

```
δ_esc = clip(ρ - (η+γ), 0, 1)    # ρ 超上限时激活
δ_ans = clip((η-γ) - ρ, 0, 1)    # ρ 低于下限时激活

正确直接回答: r=2（质量高+延迟低，最高奖励）
正确 escalate:  r=1（质量高但调用慢模型）
错误直接回答:  r=-1
错误 escalate:  r=0

最终: r_i = (1-δ_esc/δ_ans) × r_i^naive - 额外惩罚项
```

- 效果：TB-GRPO escalation ratio 仅 24%（vs Vanilla GRPO 100%，AutoThink 53%），同时 StreamingBench 74.36 最高

**实验结果**：
- OVO-Bench 57.92 (SOTA among streaming MLLMs)；StreamingBench 76.36 (SOTA)
- 95-96% visual token 减少，自适应路由在 StreamingBench 上**超过 slow-only 推理**
- 长视频泛化：MLVU 70.6, Video-MME 65.5
- 记忆-路由协同：Nearby=1.0 时 accuracy ↑ + escalation ratio ↓（更好的近期上下文让 fast model 更有能力直接回答，自然抑制不必要的 escalation）

**与 C1 的关系**：
- Respond 模块的 `<Routine>` vs Ready 决策可映射为 `$silent$` vs `$interrupt$` 的 readiness 判断层——但注意 Respond 是"是否能回答"的就绪判断，C1 还需要额外的"是否应该回答"的介入判断。两者互补
- Finding 1（激进压缩历史提升性能）对 C1 训练数据构造有直接启示——chunk 历史帧的注意力噪声可能干扰模型判断"此时是否该介入"
- Finding 2（非单调模型规模增益）提示我们：C1 的 interrupt/silent 决策不一定要用最大模型，3B 级别的专门训练可能更优
- TB-GRPO 的目标带宽控制机制可借鉴用于防止模型坍塌到"永远沉默"或"永远说话"
- 其系统设计偏效率导向，但两个实证发现对 C1 的模型设计有方法论层面的参考价值

---

## 4. RL-based 主动交互路线 (2026 新范式)

2026 年上半年，RL（特别是 GRPO）在主动视频交互领域集中爆发。以下专列 StreamPro——该路线的最新代表。

### 4.1 StreamPro (arXiv 2026.05) — CB-Stream Loss + 多粒度 RL 奖励 ★新增（已读原文）

- 作者：Ao Li, Zihan Xiao et al. (人大, 小米 MiLM Plus, 北大)
- 链接：[arxiv.org/abs/2605.16381](https://arxiv.org/abs/2605.16381)
- 原文：`literature/papers/challenge1_proactive/StreamPro.md`（983 行完整论文）

**核心创新：从"被动感知"到"主动决策"的完整训练框架。SFT 用 CB-Stream Loss 系统性解决 silence/response 类不平衡，RL 用多粒度奖励（Format + Turn-Level F1 + Trajectory-Level Rubric）联合优化时机和内容。**

**问题定义**：现有流式评测仍遵循"see-then-answer"被动范式——回应仅在显式证据出现后才触发，将主动推理降级为延迟感知。StreamPro 提出 **Proactive Agency** 概念：在部分观测下做出早期可靠决策的能力（如预测未来事件、推断潜在用户需求、在风险完全显现前发出预警）。特别包含**第一人称 Risk Forecasting**（在危险发生 ~3s 前预警，模拟视障用户导航辅助）。

**StreamPro-Bench**：577 视频，1,285 QA pairs，7 任务 × 3 维度：

| 维度 | 任务 | 说明 |
|------|------|------|
| Perception Understanding | Event Understanding, Object Understanding, Anomaly Alert | 持续感知实体状态/动态/变化 |
| Temporal Reasoning | Temporal Perception, Temporal Grounding | 追踪事件发生的精确时间和时序依赖 |
| Proactive Agency | Goal Planning, Risk Forecasting | 基于持续观测主动规划行动；~3s 前预警危险 |

**评价指标 StreamPro-F1**：轨迹级指标，联合评估时序对齐和语义正确性。
- 时间分数：`S_time = max(0, 1 - |t_pred - t_gt|/τ)`，τ 按任务不同（3-6s）
- 语义分数：LLM judge（大多数任务）或 IoU（Temporal Grounding 任务）
- 联合分数：`S = S_time × S_acc`（乘积形式确保一个维度差则整体差）
- 轨迹级 P/R：`P = ΣS_i/N_pred, R = ΣS_i/N_gt`，惩罚过多/过少回应
- 每个任务有独立的 best response timestamp 和 tolerance window（如 Risk Forecasting: optimal = 3s before hazard, tolerance = [-1s, +3s]）

**Stage 1 — SFT with CB-Stream Loss**：

每个 timestep，模型输出 `</Silence>` 或 `</Response>` + 文本。CB-Stream Loss 用**有效样本数**（effective number of samples）对决策 token 类平衡重加权：

```
E_k = (1 - β^n_k) / (1 - β)     # 类别 k 的有效样本量
ŵ_k^CB = (1/E_k) / Σ_j(1/E_j) × |S|   # 类平衡权重
w_i^CB = ŵ_yi^CB (y_i 是决策 token) 或 λ_text (y_i 是语言 token)
L_CB = (1/N) Σ w_i^CB × [-log p_i]
```

- β=0.9999 控制重加权程度
- λ_text=2 平衡决策 token 和语言 token 的优化
- CE baseline on SPB: 6.6；Focal loss (Streamo): 14.2；**CB-Stream: 16.3**

**Stage 2 — GRPO with Multi-Grained Rewards**：

总奖励加权和：`R = 0.1·R_fmt + 0.45·R_turn + 0.45·R_traj`

| 奖励 | 粒度 | 设计 | 关键细节 |
|------|------|------|----------|
| R_fmt | 每步 | 格式合法性：`</Silence>` 或 `</Response>` + text 得 1，其他格式 0 | 全程平均 |
| R_turn | 每个回应 | **加性步分** S' = S_time + S_acc（非乘积！），匹配策略改为每个 gt 取窗口内最优预测 | τ=8（比 benchmark 的 τ 更大），确保 RL 阶段稀疏奖励得到缓解 |
| R_traj | 整个视频 | LLM 为每个样本预生成 N_c 个 binary checklist，评估生成轨迹的 Granularity/Sequencing/Coverage | 使用 Gemini 2.5 Pro 评分 |

**RL 与 Benchmark 的关键区别**：Benchmark 用 S=S_time×S_acc（乘积）保证严格性；RL 训练用 S'=S_time+S_acc（加性）防止整个奖励被单个差组件归零。Benchmark 用 greedy matching；RL 用 optimal matching（每个 gt 取窗口内最高分预测）。这些修改提供了更密集、更有区分度的 RL 优化信号。

**训练配置**：
- Backbone：Qwen2.5-VL-3B / Qwen3-VL-4B
- SFT：64 H100，24h，lr=1e-5，batch=512，1 epoch
- RL (GRPO)：8 H100，24h，lr=1e-6，batch=16，G=8 generations/trajectory，temperature=1.0
- 推理：sliding window of 200 dialogue turns，1 FPS
- SFT 数据混合：StreamPro-SFT-63K + TimeChat-Online-139K + VideoChat-Flash-3K + 287K filtered Streamo-Instruct-465K
- RL 数据：StreamPro-RL-3K（仅 proactive 任务）

**实验结果**：
- StreamPro-Bench：GRPO-4B 达 **41.5**（之前最优 10.4，Streamo-3B 仅 10.4）；Proactive Agency 维度从 2.6→7.6
- OVO-Bench FAR：GRPO-4B 达 **20.6**（Streamo-7B 仅 5.4）
- StreamingBench-RTVU：GRPO-4B 达 **78.9**。RL 后实时性能微降（62.5→57.6），因 RL 阶段仅用 proactive 数据
- 离线基准不退化：VideoMME 60.4, LVBench 52.9
- 消融：τ=8 > τ=3（28.1 vs 25.8）；w_turn:w_traj=0.45:0.45 最优（纯 turn-level 只有 25.5）

**与 C1 的关系**：
- CB-Stream Loss 是替代人工调 NFS 的**系统性方案**——不需要猜测 ρ 值，loss 自适应类频率
- Proactive Agency 中的 Risk Forecasting（egocentric + 3s pre-hazard warning）与 C1 的"操作即将出错时提醒"场景高度一致
- 多粒度 RL 中 Turn-level S_time+S_acc 加性设计解决了 C1 微调时"Interrupt Recall 低但 Precision 高"的 trade-off——加性公式允许模型在内容和时机之间独立改进
- Trajectory-level rubric 可适配 C1：将 C1 的逐 chunk Macro F1 转换为 rubric checklist（如"是否在关键错误出现前预警""是否保持了合理的沉默密度"）
- 代码/权重未开源（小米+人大），但方法描述足够详细可复现

---

## 5. 数据驱动 / 反事实增强路线

### 5.1 Streaming Interventions (arXiv 2026.06) — 反事实合成训练数据

- 作者：Apratim Bhattacharyya et al. (Qualcomm AI Research, York University)
- 链接：[arxiv.org/abs/2606.09547](https://arxiv.org/abs/2606.09547)
- 项目页：[apratimbh.github.io/livecookv2](https://apratimbh.github.io/livecookv2)

**核心发现：现有最强 VLM 在主动纠错任务上几乎全部失败。**

Ego-MC-Bench 结果：
- Gemini-3-Flash: F1=0.18
- Qwen3-VL-32B: F1=0.16
- InternVL3.5-38B: **F1=0.00**（完全失败）
- Qwen2.5-VL-32B: **F1=0.00**

**Ego-CoMist 反事实合成**：把正常烹饪视频反向注入错误——假设用户在某个步骤出错，根据正确步骤反推"如果出错应该是怎样的"，在对应时间点生成纠正提示。微调后 2B 模型 F1=0.20，**超过不微调的 27B 模型**（0.14）。

**与 C1 的关系**：这篇是所有 C1 方向工作中**最接近我们赛道定义的**。两点直接启示：(1) 离线模型在流式错误检测上几乎零能力——必须做针对性的流式微调；(2) 反事实数据合成是解决标注不足的最可行方案。我们已有 Ego-CoMist annotations (12MB)，但缺少对应的视频数据（CaptainCook4D 和 Ego-Exo4D 视频未下载）。

**开发现状**：项目页已上线，代码未公开。

---

### 5.2 VISTA (arXiv 2026.05) — 因果反向推理合成视频

- 作者：Yu-Hsiang Liu et al.
- 链接：[arxiv.org/abs/2605.10579](https://arxiv.org/abs/2605.10579)

**核心思路：从"此刻应该介入"反推事件因果链 → 生成合成视频。** 5 步脚本生成管线：确定干预目标 → 反向推导因果事件序列 → 正向验证因果自洽 → 细化场景参数 → 生成完整脚本 → 视频合成引擎生成第一人称视频。

因果反向推理的要点：标注不是人工打的，而是在脚本生成时就内嵌了。这解决了 C1 数据集手动标注介入时机主观性高、成本大的问题。

三级自主性分类：Reactive（被动响应）→ Proactive Explicit（主动显式）→ Proactive Implicit（主动隐式）。C1 的 `$interrupt$` 可以映射为后两级。

**开发现状**：代码和生成管线未公开。方法描述可用，但无可直接使用的工具。

---

## 6. 评测基准与工程框架

### 6.1 Eyes Wide Open (NeurIPS 2025) — ESTP 任务形式化定义

- 作者：Yulin Zhang et al. (ShanghaiTech, HKU)
- 链接：[arxiv.org/abs/2510.14560](https://arxiv.org/abs/2510.14560)
- 代码：[github.com/SooLab/EyeWO](https://github.com/SooLab/EyeWO)

**核心贡献：首次形式化 ESTP（Ego Streaming Proactive）任务，提出不可能三角**：Proactive Coherence × Just-in-Time Responsiveness × Synchronized Efficiency 三者难以同时满足。

**ask_high 机制**：低分辨率持续监控 → 不确定时发出 `ask_high` → 拿到高分辨率帧 → 做最终决策和精确回答。等价于给模型"凑近看"的能力。

**ESTP-F1 指标**：融合回答正确性 + 时机准确性 + 时序精度。可迁移到 C1 的 Proactive F1 设计。

**三阶段课程训练**：Passive Interval Responsiveness → Proactive JIT Responsiveness + ask_high → Multi-Turn Coherence。渐进式赋予模型被动→主动→多轮能力。

**开发现状**：代码开源，权重未公开，ESTP-Bench 数据未找到独立下载链接。

---

### 6.2 EgoPro-Bench (arXiv 2026.05)

- 作者：Dongchuan Ran et al.
- 链接：[arxiv.org/abs/2605.07299](https://arxiv.org/abs/2605.07299)

首个考虑个性化上下文的主动交互基准。2,400 验证视频 + 12,000+ 训练视频，12 领域。"短思考、好交互"原则——有限 token 预算给意图识别。主要关注交互意图识别而非精确介入时机。数据未公开。

---

### 6.3 OmniPro (arXiv 2026.05)

- 作者：Ruixiang Zhao et al.
- 链接：[arxiv.org/abs/2605.18577](https://arxiv.org/abs/2605.18577)

首个全模态流式主动理解基准。2,700 样本，9 子任务，84% 需音频。双模式评估：Probe Mode（内容理解）和 Online Mode（全主动能力）。关键发现：音频持续带来收益但利用率不均。C1 数据无音频轨，不完全适用。

---

### 6.4 EgoSAT (ECCV 2026) — Past/Present/Future 三方向评测

- 作者：Yijia Lei, Jinzhao Li, Yichi Zhang et al.
- 链接：[arxiv.org/abs/2606.24422](https://arxiv.org/abs/2606.24422)
- 项目页：[leiyj23.github.io/EgoSAT](https://leiyj23.github.io/EgoSAT)

1,997 第一人称视频（165h），~4,800 QA。按时间方向组织评测：Past（回顾）、Present（在线）、Future（前瞻）。推理时约束模型只能看到当前及之前的帧。

关键发现：(1) VLM 在 Past 和 Future 推理上远弱于 Present；(2) 模型置信度与可回答性严重不匹配——"自信地错误"。引入了可回答性诊断。

C1 的介入决策同时需要 Present（当前做什么）、Past（之前做了哪些步骤）、Future（如果继续会出什么错）。EgoSAT 的三方向评测可诊断模型弱点。

**开发现状**：ECCV 2026 已接收，camera-ready 准备中。

---

### 6.5 Harnessing Streaming Video (arXiv 2026.06) — 工程完成度最高的框架

- 作者：Dingyu Yao et al. (中科院信工所, 京东)
- 链接：[arxiv.org/abs/2606.08615](https://arxiv.org/abs/2606.08615)

**核心贡献：Streaming-Train-248K（18+ 数据集，248K 训练样本）+ StreamingHarness 部署框架 + Streaming-Eval（138 视频，15 类场景，6 种能力维度）。**

**三层记忆架构**：短期（200s 原始 visual token）→ 中期（1000s 文本摘要，异步 agent）→ 长期（~12h 高度压缩块）。记忆 agent 在 chunk 边界异步运行，不阻塞推理。

**Prefix-Aware KV Cache**：兼容 vLLM 的 prefix-caching。"记忆文本"每 chunk 预填充一次进 KV cache，后续步骤只处理新帧/新 response token。

**加权 loss 处理类不平衡**：`w_silence_repeated=0.8, w_response=1.5`。两个特殊 token `<response>` 和 `<silence>` 让模型每秒做决策。

**实验结果**：8B 流式 VLM + StreamingHarness 在 Streaming-Eval 上**超越 Claude Opus 4.6, GPT 5.4, Gemini 3.1 Pro 和豆包 Seed 2.0 Pro**。训练成本约 4,096 H200 GPU 小时。

**与 C1 的关系**：三层记忆架构和 StreamingHarness 工程方案直接可用。但其介入决策基于"事件显著性"（发生了一件值得说的事），C1 需要基于"步骤正确性"，需额外步骤知识。代码/数据承诺 CC BY 4.0 开源，截至 2026.06 尚未公开。

---

## 7. 特定场景应用

### 7.1 YETI (arXiv 2025.01)

- 作者：Saptarashmi Bandyopadhyay et al.
- 链接：[arxiv.org/abs/2501.09355](https://arxiv.org/abs/2501.09355)

AR 头显上的程序性任务主动干预。用 SSIM（结构相似性）和任务步骤对齐信号判断用户是否偏离正确步骤。在 HoloAssist 上评估。SSIM + 步骤对齐的双信号机制可用于"纠错"场景，但依赖步骤级标注。

---

### 7.2 EgoSocial (arXiv 2025.10)

- 作者：Xijun Wang et al. (UMD, Google)
- 链接：[arxiv.org/abs/2510.13105](https://arxiv.org/abs/2510.13105)

社交场景中的主动介入时机判断。EgoSocial 数据集：13,500 QA，8 种社交互动线索。Gemini 2.5 Pro 仅 14.4% 准确率。场景与 C1 不同，但 Social-Thinking Graph 的多模态融合思路可参考。

---

### 7.3 Alpha-Service (arXiv 2025.10)

- 作者：Zichen Wen et al. (SJTU)
- 链接：[arxiv.org/abs/2510.14359](https://arxiv.org/abs/2510.14359)

AI 眼镜上的主动服务检测。双模型架构：轻量 Qwen2.5-VL-3B（实时触发）+ Qwen2.5-VL-7B（深度场景分析）。偏向系统设计，双模型（轻量触发+重量分析）架构可借鉴。

---

### 7.4 ROMA (arXiv 2026.01)

- 作者：Xueyun Tian et al.
- 链接：[arxiv.org/abs/2601.10323](https://arxiv.org/abs/2601.10323)

实时全模态主动助手。Speak Head 解耦响应触发和内容生成。两阶段课程训练。依赖音频输入（C1 数据仅为视频），不完全兼容。

---

## 8. 开源情况汇总

仅统计 Challenge 1 相关工作的代码、权重、数据集公开状态。

| 论文 | 代码 | 权重 | 数据集 | 标注 | 备注 |
|------|:--:|:--:|:--:|:--:|------|
| ProAssist (EMNLP'25) | ✅ | ✅ | ✅ | ✅ | 最完整开源，HF: `594zyc/ProAssist-Dataset` |
| LiveStarPro (NeurIPS'25/arXiv'26.06) | ✅ | ✅ | ✅ | ✅ | 两代版本均开源，HF: `yzy666/LiveStar_8B` |
| MMDuet2 (ICLR'26) | ✅ | ✅ | 部分 | — | GitHub: `yellow-binary-tree/mmduet2`，52K 视频数据未确认公开 |
| STRIDE (arXiv'26.03) | ✅ | ✅ | 部分 | — | **训练代码已确认存在**（trainer.py + 数据构造脚本），推理权重 HF: `interlive/STRIDE-2B` |
| Eyes Wide Open (NeurIPS'25) | ✅ | ❌ | ❌ | ❌ | GitHub: `SooLab/EyeWO`，ESTP-Bench 未找到 |
| StreamBridge (NeurIPS'25) | ❌ | ❌ | ❌ | ❌ | Apple 团队，完全未公开 |
| Proact-VL (ICML'26) | 部分 | ❌ | ❌ | ❌ | 匿名审稿仓，正式 GitHub 未建 |
| StreamPro (arXiv'26.05) | ❌ | ❌ | ❌ | ❌ | 小米+人大，未公开 |
| R3-Streaming (arXiv'26.05) | ❌ | ❌ | ❌ | ❌ | SJTU+MSRA，未公开 |
| Streaming Interventions (arXiv'26.06) | ❌ | ❌ | ❌ | ❌ | Qualcomm，项目页已上线 |
| Harnessing Streaming Video (arXiv'26.06) | ❌ | ❌ | ❌ | ❌ | 承诺 CC BY 4.0 开源 |
| EgoSAT (arXiv'26.06, ECCV'26) | ❌ | — | ❌ | ❌ | ECCV 2026 已接收，camera-ready 准备中 |
| EgoPro-Bench (arXiv'26.05) | ❌ | — | ❌ | ❌ | 声称已发布但未找到链接 |
| OmniPro (arXiv'26.05) | ❌ | — | ❌ | ❌ | 84% 需音频 |
| VISTA (arXiv'26.05) | ❌ | — | ❌ | 不适用 | 合成数据框架，未落地 |
| ROMA (arXiv'26.01) | ❌ | ❌ | ❌ | ❌ | 依赖音频 |
| YETI (arXiv'25.01) | ❌ | ❌ | 部分 | 部分 | 复用 HoloAssist |
| EgoSocial (arXiv'25.10) | ❌ | — | 部分 | 部分 | 基于 Ego4D |
| Alpha-Service (arXiv'25.10) | ❌ | ❌ | ❌ | ❌ | 偏系统设计 |

**完整开源可复现**：ProAssist、LiveStarPro、MMDuet2、STRIDE（训练+推理）。
**只开源推理**：Eyes Wide Open。

---

## 9. 核心方法对比与趋势

### 9.1 技术路线的演进

```
2024 H2          2025 H1-H2              2026 H1 (集中爆发)
────────         ────────────            ──────────────────
                  
W2T 路线:
VideoLLM-Online  → ProAssist (EMNLP)     → LiveStarPro (SVeD)
                  (EOS-prob + NFS)         (zero-param PPL gate)

决策头/RL 路线:
                 → STRIDE (03.2026)       → MMDuet2 (ICLR 2026)
                   (mask diffusion)         (RL + "NO REPLY" token)
                 → StreamBridge (NeurIPS)  → StreamPro (05.2026)
                   (decoupled activation)   (CB-Stream + multi-grain RL)
                                           → R3-Streaming (05.2026)
                                             (Remember→Respond→Reason + TB-GRPO)

数据增强路线:
                 → VISTA (05.2026)        → Streaming Interventions (06.2026)
                   (causal reverse synth)   (Ego-CoMist counterfactual)
```

**核心趋势**：
1. **从人工调参到 RL 自适应**：W2T 需要手动标 θ → 2026 年 GRPO-based RL 成为主流范式。MMDuet2 (PAUC)、StreamPro (多粒度)、R3-Streaming (TB-GRPO) 三者各有侧重——分别解决"何时回应""回应什么质量""要不要用大模型"三个递进问题
2. **从外挂模块到零侵入到系统化**：STRIDE/StreamBridge 需额外模块 → MMDuet2 零架构改动 → R3-Streaming 三阶段级联系统
3. **反事实数据合成被实证有效**：Streaming Interventions 证明 2B + 合成数据 > 27B 零样本
4. **RL 训练中 Benchmark 指标 vs 训练奖励的分离设计**：StreamPro 的 Benchmark 用乘积 S_time×S_acc（严格），RL 用加性 S_time+S_acc（密集信号）——这种"评测严、训练宽"的策略是通用经验
5. **流式记忆的新认知**：R3-Streaming 证明激进压缩历史 token **提升**性能（非仅仅节省计算），因为历史 token 是注意力噪声——这挑战了"保留越多上下文越好"的传统假设
6. **评测维度从单一到多元**：F1 only → PAUC (time-quality area) → StreamPro-F1 (task-specific tolerance) → ESTP-F1 (multi-dimensional) → 多粒度轨迹评估

### 9.2 各方法对 C1 的适用性排序（更新，基于原文精读）

| 排名 | 工作 | 核心方法 | 开源 | 对 C1 最直接的价值 |
|:--:|------|----------|:--:|------|
| 1 | **MMDuet2** (ICLR'26) | "NO REPLY" token + PAUC RL | ✅ | 零架构改动的 W2T 替代方案，RL 内化决策无需调 θ |
| 2 | **ProAssist** (EMNLP'25) | W2T + NFS + IPS | ✅ | 最完整基座，实验已跑通零样本+微调管线 |
| 3 | **StreamPro** (arXiv'26.05) | CB-Stream Loss + 多粒度 RL | ❌ | 类不平衡的系统性方案（β=0.9999, λ_text=2）；RL 训练中加性 vs 乘积的分离设计 |
| 4 | **LiveStarPro** (arXiv'26.06) | SVeD + SCAM + TSHM | ✅ | 零成本激活决策 + 流式注意力 + 树形记忆 |
| 5 | **STRIDE** (arXiv'26.03) | 掩码扩散序列去噪 | ✅ | 即插即用激活检测器，训练代码可用 |
| 6 | **Streaming Interventions** (arXiv'26.06) | Ego-CoMist 反事实合成 | ❌ | 解决训练数据瓶颈的思路 |
| 7 | **R3-Streaming** (arXiv'26.05) | Finding 1（历史 token 噪声）+ Finding 2（非单调规模增益）+ TB-GRPO | ❌ | 两个实证发现对 C1 方法论有价值；Respond 的 Decision-Boundary-Focused Sampling 训练策略可复用 |

### 9.3 我们的定位与技术路线建议（更新）

基于文献调研和 Phase 1 实验发现，推荐三条推进路线：

**路线 A（近期，低风险）**：基于 MMDuet2 的 "NO REPLY" → `$silent$` 映射，在 ProAssist 的 LLaMA-3.1-8B 骨架上实现 RL 训练管线。核心改动：(1) 将 Chat Template 中的 "NO REPLY" 替换为 `$silent$`；(2) 引入 PAUC 复合奖励替代 W2T 的 θ 阈值；(3) 冻结 lm_head + 低 LR 保护文本生成能力。**风险最低、工程最简**。

**路线 B（中期，高收益）**：借鉴 StreamPro 的 CB-Stream Loss（β=0.9999, λ_text=2）替代 NFS + 多粒度 RL（Turn-level 用加性 S_time+S_acc 防稀疏，Trajectory-level 用 C1 Macro F1 映射为 binary checklist）。结合 STRIDE 的序列去噪增强时序连贯性。**系统性最强**。

**路线 C（方法论层面，可并行）**：
- 验证 R3-Streaming Finding 1（历史 token 噪声假说）在 C1 数据上的适用性——chunk 历史帧的注意力噪声是否干扰 interrupt/silent 决策
- 验证 Finding 2（非单调规模增益）——3B 模型在 C1 的 binary interrupt/silent 决策上是否可能匹敌或超过 8B 模型
- 采用 Respond 的 Decision-Boundary-Focused Sampling 训练策略——在 interrupt/silent 决策边界附近密集采样，而非均匀采样所有 chunk

---

文献综述 v2.1，最后更新 2026-07-10。基于原文精读修正 StreamPro 和 R3-Streaming 条目（公式、超参数、实验细节）；新增强化学习训练中 Benchmark 指标 vs 训练奖励的分离设计分析；新增 R3-Streaming 两个实证发现（历史 token 噪声假说、非单调规模增益）对 C1 方法论的意义；新增路线 C（方法论层面可并行验证）。
