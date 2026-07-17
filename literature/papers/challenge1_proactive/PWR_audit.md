# PWR 全文、附录与公开产物正式审计

> 论文：**Plan, Watch, Recover: A Benchmark and Architectures for Proactive Procedural Assistance**  
> 版本：[arXiv:2606.04970v1](https://arxiv.org/abs/2606.04970v1)，2026-06-03，53 页  
> 审计日期：2026-07-13  
> 审计范围：正文第 1--8 节、Appendix A--J、arXiv HTML/PDF/source、论文链接的 Hugging Face 数据仓、leaderboard Space、workshop 官方 GitHub、HF/GitHub 公开发布面  
> 本文目的：区分 **PWR 的方法论贡献**、**论文足以支持的实现信息** 和 **当前真正可获得的官方产物**。

---

## 0. 结论先行

### 0.1 是否已经完整阅读

是。本次审计以官方 arXiv v1 为准，逐节核对了正文和 A--J 全部附录，而不是只读摘要、主方法或二手综述。附录源码中的未完成内容也被单独检查：活动附录里仍有 **117 个 `\tbd`**，渲染页约有 **98 个灰色 `--` 占位**。因此，结论不能表述为“论文已经给出全部复现细节”。

### 0.2 PWR 到底是什么

PWR 不是一个单独的 gate，也不是一个可直接下载的小模型。它是一套联合设计：

1. 把主动助手拆成 **Duplex interaction model** 和 **Procedural planner**。
2. 用显式 plan state 跟踪已完成步骤、当前步骤、下一步骤及视觉完成/未完成 cue。
3. interaction model 每个候选时刻决定 `$interrupt$` / `$silent$`，并在需要时生成话语。
4. planner 在开场和 interrupt 后更新 plan；silent 时沿用旧 plan。
5. 用 Pro²Bench 的跨数据集对齐、边界、cue 和 planner target 对两个模型分别做 SFT。
6. 用类平衡采样和更高 interrupt loss 缓解“永远沉默”或决策失衡。

论文的优势来自 **监督数据、显式过程状态、双模型分工、上下文采样和决策校准的组合**，不能归因于某一个 prompt 或门控公式。

### 0.3 公开代码和权重的最终核查结果

截至 2026-07-13，**未找到官方公开的 PWR training code、PWR checkpoints/weights、Pro²Bench train annotations、gold plan/cue targets 或完整 alignment/enrichment 工程**。

论文标题页唯一的 “Data & Code” 链接指向 [facebook/wearable-ai](https://huggingface.co/datasets/facebook/wearable-ai)。该仓当前公开的是竞赛 validation 数据和 starter kit，不是 PWR 训练仓。论文结论写了释放数据、代码和权重，但当前公开状态并不支持这一表述。

这意味着：

- 可以复现 C1 的输入、输出、baseline inference 和官方评分。
- 可以依据论文 **重新实现一个 PWR-inspired 系统**。
- 不能忠实复现论文训练集、训练样本序列化、官方 PWR 训练过程或论文表格结果。
- 在官方产物出现前，我们自己的实现必须称为 **PWR-inspired / paper-spec reimplementation**，不能称为 official PWR reproduction。

### 0.4 对当前 C1 项目的直接含义

PWR 揭示了此前路线中真正缺失的变量：**程序状态、步骤完成 cue、偏离/恢复状态和 plan quality**。但它没有给我们一个可直接拉取训练的现成仓库。主攻 Small 时，合理目标不是照搬两套 27B/大型 MoE 模型，而是把 PWR 的有效信息压缩成小模型可消费的状态，并做可证伪的 oracle/predicted/noisy-plan 对照。

---

## 1. 审计口径与状态标记

本文使用以下状态：

| 标记 | 含义 |
|---|---|
| **已公开** | 官方 URL 上存在，并已检查文件树或内容 |
| **论文给出** | 论文明确描述，但不代表对应文件已经发布 |
| **可重建** | 信息不完整，但可通过自定实现选择构造近似版本 |
| **阻断** | 缺数据、target、schema、代码或权重，无法忠实复现 |
| **冲突** | 论文不同章节、附录或公开竞赛数据之间口径不一致 |

证据优先级为：官方 arXiv/source > 官方 HF/GitHub/API 文件树 > 竞赛公开数据 > 第三方仓库或搜索结果。第三方复现只作为线索，不替代官方发布。

---

## 2. 官方公开产物与穷尽搜索

### 2.1 已核实的官方入口

| 入口 | 2026-07-13 状态 | 审计结论 |
|---|---|---|
| [arXiv abstract](https://arxiv.org/abs/2606.04970v1) | v1，2026-06-03，53 页 | 没有 v2；唯一项目链接是 HF dataset |
| [arXiv HTML](https://arxiv.org/html/2606.04970v1) | 正文和 A--J 可读 | 方法与 prompts 的主要审计来源 |
| [arXiv source](https://arxiv.org/src/2606.04970v1) | TeX、图片、字体 | 含 Appendix I 的 artifact manifest 表，但不含该表列出的 data/config/script 实体；可见大量 `\tbd` |
| [facebook/wearable-ai](https://huggingface.co/datasets/facebook/wearable-ai) | `main` SHA `8b32c442cfd6263fa359a82061fa022be77bc2fa` | validation 数据 + starter kit；不是 PWR train repo |
| [facebook/wearable-ai-leaderboard](https://huggingface.co/spaces/facebook/wearable-ai-leaderboard) | SHA `40c818898cd471d2bd0108924c5d7e0db8606489` | 提交、评分和榜单 Space；无训练代码或模型 |
| [workshop GitHub](https://github.com/wearable-ai-workshop/wearable-ai-workshop.github.io) | HEAD `805e766c59b113fb57d38383e00ca07c655caf90` | 静态站点、规则和 challenge assets；无 PWR 实现 |

### 2.2 HF 数据仓历史审计

`facebook/wearable-ai` 公开 `main` 一共检查到 4 个 commit：

| SHA 前缀 | 日期 | 变化性质 |
|---|---:|---|
| `8b32c442` | 2026-07-09 | README PR |
| `486ad487` | 2026-06-19 | judge 文本修改 |
| `7e032e04` | 2026-06-18 | ConvQA judge 修改 |
| `f6cdc8df` | 2026-05-21 | 初始发布 |

逐个历史树检查后，非视频内容始终只有：

- 根目录 README、LICENSE；
- 三个 challenge validation JSONL；
- `starter_kit` 下的 generation/evaluation/model/tests 等 15 个文件；
- challenge 视频文件。

没有发现：

- `train.jsonl`、Pro²Bench train rows 或 EgoProactive 论文 split；
- plan state、step-complete/incomplete cue、OOP recovery target；
- `stats/stats.json`、`configs/pipeline.yaml`、`splits/*.json`；
- PWR trainer、planner trainer、alignment/enrichment scripts；
- checkpoint、adapter 或 model weights；
- 隐藏的公开 tag 或额外 branch。

本地 `/data1/wearable_ai_challenge_data/starter_kit` 的 15 个非视频文件 blob 与当前远端逐一一致，因此重新下载或完整 clone 不会得到另一套训练管线。

### 2.3 其他公开面搜索

已再次检查：

- HF `facebook` 组织的 model/dataset/space 搜索；
- GitHub exact repository 搜索：论文全名、`Plan Watch Recover`、`EgoProactive`、`Pro2Bench`；
- `facebookresearch` 范围搜索；
- workshop 官方 GitHub 的 branches、tree 和 history；
- 论文作者可确认的公开 GitHub/profile 链接；
- arXiv 页面中的所有 code/GitHub/weights/checkpoint/repository 链接。

结果是：HF `facebook` 下没有 wearable/PWR model，只有该 dataset 和 leaderboard；GitHub 没有官方命中。搜索到的 `Kevincklhhh/ProceduralAgent` 等是第三方实验或复现，不能当作官方代码和权重。

匿名 GitHub 的全站 code search 本身有登录限制，因此这里不宣称数学意义上的“互联网绝对不存在”；准确表述是：**截至审计日期，在所有可验证官方入口、官方组织、公开仓库元数据和精确项目搜索中均未发现发布物。**

### 2.4 当前到底能拿到什么

| 产物 | 状态 |
|---|---|
| PWR 论文正文与 A--J 附录 | **已公开** |
| 竞赛 700 sessions / 9,935 chunks validation 数据 | **已公开** |
| C1 starter-kit 推理、格式与评分代码 | **已公开** |
| leaderboard 提交和评分 Space | **已公开** |
| Pro²Bench 训练语料及 249K/314K row 数据 | **阻断** |
| 论文使用的 EgoProactive 5,065-row eval split | **阻断** |
| gold procedural plans、visual cues、OOP recovery targets | **阻断** |
| step alignment / enrichment 可执行工程 | **阻断** |
| Duplex / planner 训练代码 | **阻断** |
| Llama 4 Maverick PWR 权重 | **阻断** |
| Qwen3.6-VL-27B PWR 权重 | **阻断** |
| 论文逐样本预测与 judge 输出 | **阻断** |

公开 C1 JSONL 的顶层字段只有：

```text
video_path, duration_in_sec, video_intervals, query,
domain, task, answers, dialog
```

其中 `answers` 给出每个 interval 的 `$interrupt$...` / `$silent$`，但没有 `plan`、`current_step`、`step_boundary`、`completion_cue`、`OOP` 或 `recovery_plan` 字段。因而 9,935 个公开决策标签只能直接监督 C1 gate，不能直接还原 PWR planner supervision。

---

## 3. PWR 方法的端到端还原

### 3.1 任务变量

在候选时刻 `t`：

- `q`：用户目标或 query；
- `v_1:t`：到当前为止的因果视频；
- `H_<t`：此前对话；
- `o_t = (v_1:t, H_<t, q)`：当前观测；
- `d_t`：silent / interrupt 决策；
- `u_t`：interrupt 时的自然语言回复；
- `P_t`：结构化 procedural plan state。

传统 monolithic 形式是：

```text
p_Theta(d_t, u_t | o_t)
```

PWR 将其分解为：

```text
p(d_t, u_t, P_t | o_t, P_{t-1})
  = p_theta(d_t, u_t | o_t, P_{t-1})
  * p_phi(P_t | d_t, u_t, o_t, P_{t-1})
```

其中 `theta` 是 interaction model，`phi` 是 planner。

### 3.2 Plan state

论文中的核心状态可概括为：

```text
goal
completed steps
current step
next steps
step-complete visual cues
step-incomplete visual cues
mode                    # recovery 通过重写 plan/steps/cues 表达，不是独立字段
```

Appendix G 的 planner schema 更具体：

```text
mode
goal
steps_completed_so_far
plan                  # 第一项标注 (Current)
step_complete_cues
step_incomplete_cues
```

另一个 JSON DSL 变体使用：

```text
vertical, task, eta, visual_cues,
current, next_few, last_few
```

这两套表示并未统一成一个公开、机器可验证的最终 schema。论文方法上明确的是“显式状态 + cue”，工程上的唯一序列化格式并不明确。

### 3.3 双模型状态机

```text
session opening
    -> planner 初始化 P_0
    -> interaction model 观察当前 clip + history + P_0
         |-- silent:    输出 $silent$，P_t = P_{t-1}
         `-- interrupt: 输出 $interrupt$ + utterance
                          -> planner 结合当前 8s、utterance、旧 plan 更新 P_t
                          -> 后续候选点使用新 plan
```

这一结构有一个重要风险：planner 只在开场和 interrupt 后更新。如果 interaction model 出现 false negative，planner 不会看到本应触发更新的时刻，旧状态会继续传播。论文的 oracle-plan 实验说明 plan quality 很重要，但没有解决这一闭环误差累积。

### 3.4 Interaction context

论文给 interaction model 最多 15 段非连续 8 秒 clip：

- 当前 clip；
- 最多 14 个历史 plan-update anchor clip；
- session opening 作为初始 update anchor；
- 以约 2 FPS 采样。

planner 更新时只看最近 8 秒视觉上下文。未完全明确的细节包括：

- 每个 8 秒 clip 的精确采帧数；
- 边界处 rounding、padding、overlap；
- 不足 15 个 anchor 时的序列化；
- opening clip 的定义；
- Qwen 配置中 `max 30 frames` 与 15 段 8 秒、2 FPS 的映射关系。

### 3.5 两个独立 SFT objective

论文给出的训练形式是：

```text
L = L_theta + L_phi

L_theta = -E_t log p_theta(y_t | o_t, P*_{t-1})
L_phi   = -E_tau log p_phi(P*_tau | u*_tau, o_tau, P*_{tau-1})
```

- interaction model 在所有采样时刻训练；
- silent target 只有 decision token；
- interrupt target 包含 decision token 和 utterance；
- planner 只在 gold interrupt 时刻及 opening 训练；
- 两者独立优化，interaction model 训练时使用 gold previous plan；
- 推理时使用 predicted plan。

因此存在明确的 teacher-forcing gap：训练看到干净 plan，推理看到自身历史预测和 gate 错误造成的状态。

论文文本还说 interrupt loss 加权 `2x`，但公式没有说明这个权重作用于：

- interrupt 样本整体；
- decision token；
- decision + utterance 所有 token；
- 还是 sequence-normalized loss。

这会实质改变决策校准，属于忠实复现阻断项。

### 3.6 推理成本

planner 只在 interrupt 时运行，论文给出期望成本：

```text
C_interaction + p(interrupt) * C_planner
```

这是 PWR 能在 duplex 场景中使用大 planner 的主要理由。不过 Small track 的模型/资源约束下，更现实的实现是把 planner 结果缓存、离线生成，或蒸馏为小状态更新器，而不是在线部署第二个大型模型。

---

## 4. 正文逐节审计

### §1 Introduction

**论文提出的问题**

- 主动助手不仅要识别视觉内容，还要理解当前程序状态；
- 要决定何时打断、说什么，以及用户偏离后如何恢复；
- 现有工作缺少显式 OOP recovery 和统一的可迁移训练 recipe。

**主要贡献声明**

1. EgoProactive：包含 scripted out-of-procedure 错误与恢复的第一视角数据。
2. Pro²Bench：对多个现有程序视频源统一对齐、采样和补充监督。
3. planner--interaction 解耦架构。
4. 在 Llama 4 Maverick 和 Qwen 系列上的 post-training recipe。

**审计意见**

- 引言同时使用 9,935 eval chunks、1,833 OOP 等统计，但后文另有 5,065 和 1,267 等不同口径；不能不加说明地混用。
- “release code/data/weights” 是声明，不是当前可用性事实。
- 对 C1 有价值的是状态建模框架，不是论文同名权重，因为后者目前不存在公开下载入口。

### §2 Plan, Watch, Recover

#### §2.1 Monolithic formulation

将主动决策和生成压在一个模型中。该形式是 PWR 的对照组，不是 PWR 自身。

#### §2.2 Decoupled formulation

把 interaction 和 planner 概率分解，silent 时 plan 不变，interrupt 时 planner 更新。公式说明了模块依赖，但没有给生产级状态机的 retry、parse failure、planner timeout 或 invalid state 处理。

#### §2.3 Plan representation

显式维护 completed/current/next 以及 completion/incompletion cue；OOP 时需重写当前计划和 cue。方法意义明确，最终 token schema 却存在多个版本。

#### §2.4 Temporal context

interaction 使用当前 clip 加历史 plan-update anchors，planner 仅使用最近 clip。这个采样策略比“把整段视频塞进模型”更重要，也更适合 Small 做压缩实验。精确帧数与序列布局缺失。

#### §2.5 Inference

gate 决定是否调用 planner。没有独立 OOP classifier；偏离应通过 interaction decision 和 planner recovery 共同体现。对我们而言，应区分“对外是否说话”和“内部是否更新状态”，否则 false negative 会冻结内部过程状态。

#### §2.6 Training

interaction/planner 分别做单时刻 SFT，均 teacher-force gold plan。训练 loss、mask 和样本格式不充分，且 `2x` interrupt weighting 未形式化。

#### §2.7 Sampling

duplex rows 约 1:1 interrupt/silent；planner rows 来自 opening 和 gold interrupts。C1 公开 JSONL 没有 plan、cue、step boundary 或 recovery target，不能直接执行这一训练。

### §3 EgoProactive and Pro²Bench

#### §3.1 评测对象

论文同时评估 decision timing 和 interrupt content；公开 C1 当前只以决策为核心排名。论文的 G-Mean/PQS/GPT judge 结果不能直接等同于 C1 Macro F1 排名。

#### §3.2 数据

主文核心统计：

| 数据 | 论文主文口径 | 审计备注 |
|---|---:|---|
| EgoProactive | 700 videos/sessions，9,935 eval instances | 与附录的 5,065-row paper eval 不同 |
| Pro²Bench train | 249,584 rows | Appendix F 又报告 duplex 314,290 rows |
| Pro²Bench eval | 42,275 rows | Appendix F 报告 40,008 rows |
| Domains | 14 | 可作为粗粒度 taxonomy；具体映射文件未发布 |

EgoProactive 使用 consumer smart glasses 连续第一视角录制，script 为 5--14 steps，平均 9.3。OOP 采用 correct lead-in -> mistake -> fix -> correct end state 的四阶段设计；边界与文本由 machine proposal 后人工审核/修订。

公开竞赛 700 sessions 的 validation JSONL 不等于论文中用于 Table 结果的 5,065-row eval 表，至少 instance slicing 和标注字段不同。

#### §3.3 Metrics

论文 decision 指标为：

```text
G-Mean = sqrt(Interrupt-F1 * Silent-F1)
```

内容通过 GPT-5.2 对 relevance、specificity、actionability、conciseness 四维评分；PQS 又把 timing/content 组合。OOP 另评 detection recall 和 recovery quality。

而 C1 leaderboard 使用两类 F1 的 arithmetic macro average。二者对极端偏类的惩罚不同：

```text
C1 Macro F1 = (IF1 + SF1) / 2
PWR G-Mean  = sqrt(IF1 * SF1)
```

训练和调阈值时必须同时计算 C1 官方指标，不能用论文 G-Mean 代替。

### §4 Experiments

#### §4.1 训练数据与 held-out 设置

论文报告的 fine-tuning 使用 EgoExo4D、EPIC-KITCHENS、HowTo100M、Ego4D；HoloAssist 和 EgoProactive held out。Appendix F 虽列出 EgoProactive pool rows，但声称报告的 FT 未使用它们。这与附录关于 OOP 模板共享和 recovery supervision 的表述存在张力，见冲突表。

#### §4.2 训练配置

| 配置 | Llama 4 Maverick | Qwen3.6-VL-27B |
|---|---|---|
| Learning rate | `5e-6` | `2e-5` |
| Schedule | cosine | 未充分说明，正文上下文为训练 recipe |
| Weight decay | `0.1` | `0.1` |
| Warmup | 100 | 未明确 |
| Max sequence length | 65,536 | 24K |
| Effective batch | 4 | 128 |
| Micro batch | 2 | 1 |
| Gradient accumulation | 2 | 2，64 ranks |
| Steps | 未给出 | 40,400 |
| Video sampling | anchor clips，约 2 FPS | 2 FPS，max 30 frames |

两者都只在 final assistant response 上算 loss；数据 mixture 采用 square-root proportional sampling；monolithic 与 PWR 对照声称使用相同 recipe。

缺失信息包括 optimizer 的完整参数、precision、gradient clipping、seed、checkpoint cadence、模型 revision、视频 processor 配置、loss normalization、硬件/耗时的完整值。

#### §4.3 主要结果

EgoProactive 上代表性结果：

| Backbone / condition | G-Mean | PQS |
|---|---:|---:|
| Qwen zero-shot | .28 | .24 |
| Qwen monolithic FT | .22 | .46 |
| Qwen PWR | .68 | .44 |
| Llama zero-shot | .49 | .34 |
| Llama monolithic FT | .49 | .40 |
| Llama PWR | .57 | .43 |

论文跨数据集 mean 中，Qwen PWR 约 `.83`，Llama PWR 约 `.76`。关键不是 fine-tuning 必然提升：Qwen monolithic FT 甚至降低决策平衡；显式 plan 和 duplex training 才产生主要增益。

#### §4.4 OOP 与 oracle plan

| 设置 | 论文关键观察 |
|---|---|
| Llama predicted plan | OOP detection recall 78.7，recovery 2.72 |
| Llama oracle plan | detection 99.6，recovery 4.82 |
| Qwen predicted plan | detection 64.9 |
| Qwen oracle plan | detection 54.5，出现反常下降 |

论文把 Qwen 的反常结果归因于视觉上下文限制，但配置细节不足，无法检验。

Oracle-plan 总表中，Llama predicted plan mean 约 `.76/.63`，oracle 约 `.84/.87`；EgoProactive 从 `.57/.43` 到 `.91/.95`。这说明 planner mismatch 是上限瓶颈，也说明先测 oracle plan upper bound 比直接做 RL 更有信息量。

#### §4.5 Monolithic、duplex 与 gate

代表性 mean G-Mean：

| 模型 | Monolithic FT | Duplex/PWR |
|---|---:|---:|
| Qwen | .45 | .83 |
| Llama | .67 | .76 |

简单分离 gate 在 zero-shot 下主要是平衡类别，并不自动得到 PWR 效果。PWR 需要 plan-aware training data 和状态监督。

### §5 Analysis / Discussion

论文强调 plan quality、visual cue 和 recovery。需要额外注意：gold plan 是“记录脚本/数据标注中的计划”，不一定代表现实任务中唯一正确或安全的规范计划。对安全相关步骤，不能只相信自动生成 cue 或脚本顺序。

### §6 Related Work

PWR 与 proactive assistance、procedural video understanding、duplex interaction 和 planning 工作建立联系。对本项目最重要的边界是：STRIDE 等动作区间/边界数据可以帮助学习视觉事件边界，但 **边界标签不等同于 C1 interrupt 标签**；是否应对外介入还取决于用户目标、状态、风险和交流策略。

本节再次出现 5,065 eval、1,267 OOP 等口径，与引言和 Table 1 不一致。

### §7 Conclusion

论文总结称发布 EgoProactive、Pro²Bench annotations、weights 和 code，但没有给具体 model/repo ID，当前官方入口中也不存在这些产物。“PWR matches or exceeds oracle” 的总结也不能概括所有结果，因为 Table 6 多数设置仍显示 oracle plan 显著更强。

### §8 Limitations / Broader Impact

论文承认 scripted OOP、teacher-forcing gap、公平性和安全风险。附录又称 OOP test 与 training 共享 script templates，但主实验称 EgoProactive held out、Table 1 train 为 N/A；这需要原始 split/manifest 才能消歧。

---

## 5. Appendix A--J 逐节审计

### Appendix A: Results: Detailed Analysis

**提供的信息**

- 五个 zero-shot frontier baselines：Llama 4 Maverick、GPT-5.2、Gemini 3.1 Pro、Qwen3-VL-235B、Claude Opus 4.6。
- 条件：No Plan、同家族 ZeroShot Planner、Oracle Planner。
- 每个 `(model, condition)` 运行 3 个 prompt variant，选择最佳 G-Mean；六个数据集复用 prompt。
- frontier median G-Mean `.49`，单数据集最高 `.62`。
- GPT-5.2 在 HowTo No Plan 上 99.7% 预测 interrupt，但 G-Mean 为 0，展示了类别坍缩。
- 加 plan 后常见现象是 SF1 上升、IF1 下降；24/30 model-dataset 对向 silent 偏移。
- Oracle Plan 在 19/30 对反而弱于 ZeroShot Plan，提示 plan utilization/calibration 本身也有问题。

**缺失/问题**

- 5 models x 3 conditions 应为 15 个配置，A.2 写成 18。
- 开头称四种 subjective patterns，实际只列 3 种。
- 承诺给出 3 个 prompt variants 和 per-prompt scores，但 Appendix G 没有完整兑现。
- Qwen3.6-VL-27B FT/FT 的状态在论文内部矛盾：Appendix A 仍写“投稿时训练中”，但正文 Tables 2/3/5 已给出结果，摘要也声称完成跨 backbone 验证。没有 checkpoint 和 predictions，当前无法独立判断最终运行状态。

### Appendix B: Subjective Score Details

**PWR-OP content score `(R, S, A, C, Avg)`**

| Dataset | R | S | A | C | Avg |
|---|---:|---:|---:|---:|---:|
| EE4D | 2.89 | 3.04 | 3.80 | 4.53 | 3.57 |
| EPIC-KITCHENS | 4.61 | 4.59 | 4.78 | 4.93 | 4.73 |
| HowTo | 4.46 | 4.39 | 4.61 | 4.87 | 4.58 |
| EgoProactive | 4.80 | 4.74 | 4.80 | 4.91 | 4.81 |
| HoloAssist | 4.58 | 4.62 | 4.76 | 4.96 | 4.73 |
| Ego4D | 4.71 | 4.67 | 4.79 | 4.93 | 4.78 |

**PWR-OP objective `(IF1, SF1, G-Mean)`**

| Dataset | IF1 | SF1 | G-Mean |
|---|---:|---:|---:|
| EE4D | .66 | .72 | .68 |
| EPIC-KITCHENS | .92 | .90 | .90 |
| HowTo | .86 | .78 | .81 |
| EgoProactive | .92 | .90 | .90 |
| HoloAssist | .91 | .89 | .89 |
| Ego4D | .82 | .75 | .78 |

本附录没有逐样本 judge 输出、置信区间或 judge 重复性原始记录，因此只能核对聚合表，不能重算内容指标。

### Appendix C: More Qualitative Results

Figures 6--13 展示：silent monitoring、step-completion interrupt、外部 distraction、程序性错误及篮球充气的连续错误恢复。

图中可观察的输入/输出结构为：

```text
5 frames
task + source
dialog history
previous/current plan
visual cue
gold response
PWR response
one selected baseline response
```

这些图证明系统能在选定案例上呈现目标行为，不能推出总体 recovery rate。附录没有 case ID、样例选择准则或生成日志。

### Appendix D: Domain Taxonomy

- Table 19：14 个 coarse domains。
- EgoProactive paper eval：5,065 rows。
- EgoExo4D：899 rows。
- Ego4D：19,396 rows。
- HowTo100M：1,321 eval video IDs。
- EPIC-KITCHENS：6,574 rows。
- HoloAssist：2,797 rows。

EgoProactive 的四域分布为 Chef 1,566、Tutorial 1,439、Handyman 1,214、Arts and Crafts 846。Appendix D 说 4 个域，F/H 又说 5 个域并加入 General household，口径冲突。

### Appendix E: Domain-Wise Results

Tables 25--28 报告 domain-wise IF1/SF1/G-Mean，但 PWR-OP 在同一数据集每个域的数值完全相同：

- EgoProactive 每域都是 `.92/.90/.91`；
- EgoExo4D 每域都是 `.66/.72/.69`；
- Ego4D 每域都是 `.82/.75/.79`。

它们看起来是 aggregate 被复制到各域。没有 per-domain confusion counts，因而不能据此得出“跨域稳定”的结论。

### Appendix F: Per-Dataset Statistics

**论文声称统计来源**

```text
stats/stats.json
data/wp/videos.csv
data/wp/instances.jsonl
```

这些文件当前都未公开。

**可见统计**

- 总 eval 40,008：20,804 interrupt / 19,204 silent。
- EgoProactive paper eval 5,065：2,786 interrupt / 2,279 silent，OOP 1,267。
- Duplex train rows 314,290。
- Planner train rows 212,818。
- EgoProactive pool 列出 duplex 4,340、planner 2,502，但已报告 FT 声称不使用它。
- Silent sampling：每视频若有 `K` 个 interrupt，则采约 `K` 个 silent；按时长分层，距 interrupt 至少 5 秒。

**表格质量问题**

- Table 29 的 LaTeX 表头 8 列，数据行字段数不一致；`45/55` 在同一单元格，并出现未定义的 `PS/ST/NR`。
- Table 30 把 EgoProactive 写成 `WearProactive`。
- Table 29 给 EgoProactive 的 `Avg steps` 为 3.5，Table 30 又给出 9.3；可能是列错位、procedure/instance 粒度不同或数据版本不同，论文没有解释。
- Table 30 给出 135 procedures，Table 35 又给 460 unique goals/procedures，定义未解释。
- Table 33 的 `Videos` 列数值逐行等于 eval rows，总计也是 40,008，显然不是视频数。

### Appendix G: Evaluation Protocol

**给出的推理协议**

- greedy decoding，temperature 0；
- 每个 clip 均匀采 8 帧；
- common duplex prompt 要求跟踪当前步骤、识别 completion/OOP，并严格输出 `$interrupt$...` 或 `$silent$`；
- planner 输入包含 attachment、上一 planner 输出、recent visual summary 和最近 user/assistant turn；
- subjective judge 只评 true-positive interrupts；
- self-optimize meta-prompt 让模型自行改写 duplex prompt。

**论文点名但未公开的工程文件**

```text
run_3p_baselines.py
run_condition_b.py
schemas/prompts.py
  MM_PLANNER_SYSTEM_PROMPT
  PROCEDURAL_PLANNER_SYSTEM_PROMPT
subjective_judge.py
  JUDGE_PROMPT
  optimized-prompt loader
```

“点名文件名”不等于已经发布。以上文件不在 arXiv source、HF dataset、leaderboard 或 workshop repo 中。

**缺口/冲突**

- 没有完整 3 prompt variants、rewritten prompts 和 per-prompt scores；
- 没有 parser/fallback、max generation tokens、API model revision、失败重试；
- Table 34 写 Claude judge 其他模型、GPT judge Claude，而 A/B/G 正文称统一用 GPT-5.2 并排除 GPT 自评。

### Appendix H: EgoProactive Collection Details

**采集协议**

- consumer smart glasses，production app `v236+`；
- 视频通常 2--5 分钟，至少 1 分钟，最长 10 分钟；
- script 含 goal、sub-goals、fine-grained action steps；
- 连续第一视角录制，画面中不得出现书面操作说明。

**Table 35 的可见数字**

| 项目 | 数值 |
|---|---:|
| Unique goals | 460 |
| Eval rows | 5,065 |
| OOP rows | 1,267 |
| Interrupt | 2,786 |
| Silent | 2,279 |
| Duplex train rows in pool | 4,340 |
| Planner train rows in pool | 2,502 |

参与者数、采集视频数、接受/拒绝视频数、train/eval 视频数全部仍是 `\tbd`。

**OOP 标注**

- 类型：step omission、incorrect ordering、substitution/technique error；
- onset：错误第一次可从视觉上区分的帧；
- 评价 tolerance：±2 秒；
- 四阶段：correct lead-in -> mistake -> fix -> correct end state；
- naturalness study 和 onset agreement 的样本量/结果均为 `\tbd`；
- 正文声称 supplementary 有 tolerance sensitivity，但 A--J 中没有该分析。

**标注和 cue**

- temporal boundary 由机器提议、人工修正；
- step text 由 VLM 提议、人工编辑；
- OOP onset 人工标注；
- normal completion response 由 VLM template 后人工编辑，OOP response 人写；
- 每步要求至少 3 个 completion 和 3 个 incompletion cues。

实际 cue 数量、human verifiability 和 Cohen's kappa 都是 `\tbd`，cue generation prompt 也没有给出。文中说 safety-critical tag 已发布、demographics 支持 bias analysis，但公开 C1 JSONL 没有这些字段，demographic 数字仍全部缺失。

### Appendix I: Step Alignment Pipeline

**论文承诺的 artifact manifest**

```text
data/*/videos.csv
data/*/instances.jsonl
data/wp/qa_rejections.csv
stats/stats.json
configs/pipeline.yaml
prompts/*.txt
splits/*.json
```

这些路径均未随 arXiv source 或官方仓公开。

**Stage 1: VJEPA2 features**

- ViT-Giant，384 px，64 frames；
- 原视频每 4 帧取 1 帧；
- ImageNet normalization；
- 64-frame window、stride 32、overlap feature averaging；
- feature dimension 1,408；
- 8 x H100、TorchX、24 ffmpeg workers。

batch、吞吐、GPU-hours 仍为 `\tbd`。

**Stage 2: hierarchical clustering**

- Ward linkage；
- tridiagonal adjacency，只允许相邻时间 cluster merge。

pilot 样本量和收益为 `\tbd`。

**Stage 3: dendrogram partition**

- metadata 缺失时 `N = ceil(duration_minutes * 2.5)`；
- 实际目标 `floor(1.3N)`，主动 over-segment；
- 保留完整树供递归细分。

binary-search bounds、停止 tolerance、最短 segment 未给。

**Stage 4: Qwen3-VL-235B captioning**

- 每 segment 最多 16 个均匀帧；
- 多行输出视为多动作，沿 dendrogram 继续 split；
- 16 threads，4--8 videos parallel。

缺 API/model revision、decoding 和 parser。

**Stage 5: Llama 4 Maverick grouping**

- 输入 timestamped captions、activity、reference steps；
- 合并相邻同动作，吸收 `No active task`；
- mistake/interruption/fix 独立成段；
- 输出 `[{caption,start_ts,end_ts}, ...]`。

**Stage 6: Llama 4 alignment**

- text-only；
- 输出与 segment 等长的 1-based step index array；
- `0` 表示无匹配；允许多 segment 对一步，也允许某步无匹配。

**Stage 7: QA**

- 检查全视频 temporal coverage；
- Llama 4 对 coverage/order/relevance 各打 1--10；
- overall `<5.0` 排除。

分数据集 coherence 和 rejection 数字全部为 `\tbd`。附录给了 Stage 4--7 的主要 prompt，但所谓完整 coherence prompt 缺其宣称的 4 个 calibration examples，也没有 cue-generation prompt。

### Appendix J: Enrichment Pipeline

本附录只有四段高层描述：

1. Base conversion：统一 goal、ordered steps、boundaries、dialog；无 JSON schema。
2. Duplex sampling：0.5 秒 stride，即 2 FPS sweep；每点看过去 8 秒；boundary/OOP onset 为 interrupt，其他点采 silent；silent 距 interrupt `delta=3s`；每视频 silent 约等于 interrupt。
3. Cue generation：VLM 生成 complete/incomplete cues 并附在 row 上。
4. Clip/split：URL/ID 去重、video-level split、UUID intersection 检查。

**关键阻断**

- Appendix F 的 silent exclusion 是 5 秒，J 是 3 秒；
- planner sampling 只写“broader temporal context”，没给长度、输入、target、teacher forcing 和 serialization；
- cue 模型、帧数、prompt、schema、人工修订路径缺失；
- 没有 interaction/planner 完整 message 格式和 loss mask；
- 没有可执行脚本、配置或输出 manifest。

---

## 6. 论文内部与竞赛口径冲突表

| 项目 | 口径 A | 口径 B | 复现影响 |
|---|---|---|---|
| EgoProactive eval | §1/Table 1：9,935 | §6/Appendix D/F/H：5,065 | 不知道论文 table 与 C1 public chunks 的映射 |
| Pro²Bench eval | §1/Table 1：42,275 | §4/Appendix F：40,008 | 无 manifest 无法确定过滤版次 |
| Pro²Bench train | Table 1：249,584 | Appendix F：duplex 314,290 | 可能是不同 task rows，但未定义清楚 |
| OOP count | §1：1,833 | 2,786 interrupt + 2,279 silent 中 OOP 1,267；另有总 OOP 3,433 | instance/session/onset 口径混杂 |
| EgoProactive domain | Appendix D：4 | Appendix F/H：5 | domain-wise 结果不可稳健解释 |
| EgoProactive avg steps | Table 29：3.5 | Table 30：9.3 | plan/step 粒度或表格列映射不确定 |
| Procedures/goals | Table 30：135 | Table 35：460 | procedure、script、goal 定义不明 |
| Silent exclusion | Appendix F：5 s | Appendix J：3 s | 直接改变负样本难度和类边界 |
| Judge | A/B/G：GPT-5.2 统一 judge | Table 34：Claude/GPT 交叉 judge | 内容分数不可精确重算 |
| OOP held-out | EgoProactive 未用于 reported FT | 附录称 test 与 train 共享 script templates，且列 pool rows | 数据泄漏/泛化口径需 split 才能判断 |
| Interrupt weight | 文字：`2x` | loss 公式无权重位置 | gate calibration 无法忠实复现 |
| Oracle conclusion | 多表显示 oracle 显著更强 | 结论称 PWR matches/exceeds oracle | 不能只引用结论段 |
| Qwen visual context | 15 x 8s anchors、2 FPS | max 30 frames，且 OOP 分析称 restricted `v_t` | 实际模型视觉输入不明确 |
| Domain-wise PWR-OP | 声称 per-domain table | 同数据集所有域完全同值 | 疑似 aggregate 复制，不能做域结论 |
| Table 33 `Videos` | 列名为视频数 | 数值恰等于 eval rows | 表头或数据错误 |

这些冲突不等于 PWR 的核心思想无效；它们意味着当前 v1 不是一份可执行的 release specification，也不足以验证论文的精确数字。

---

## 7. 忠实复现矩阵

| 模块 | 论文给出的信息 | 公开依赖 | 当前判定 | 忠实复现还缺什么 |
|---|---|---|---|---|
| C1 input/output | `$interrupt$...` / `$silent$`，8s chunks | val JSONL + starter kit | **已公开** | 无 |
| C1 scorer | official Macro F1 pipeline | starter kit | **已公开** | 无 |
| 3P common prompt | 主 prompt | Appendix G | **可重建** | 另外 2 个 variants、optimized prompts |
| 3P decoding | greedy, temp 0, 8 frames/clip | Appendix G | **可重建** | max tokens、parser、API revision、retry |
| Subjective judge | rubric 与主 prompt | Appendix G | **部分可重建** | judge 分工冲突、逐样本结果、calibration |
| EgoProactive C1 val | 700 sessions / 9,935 chunks | HF dataset | **已公开** | 仅能做公开 C1 eval |
| EgoProactive paper eval | 5,065 rows / 1,267 OOP | 无 | **阻断** | split、instances、OOP/cue fields |
| Pro²Bench train | 来源和聚合统计 | 无 | **阻断** | train rows、licenses、manifests |
| Plan schema | 两套示例字段 | Appendix G | **可重建但非唯一** | canonical schema/version |
| Visual cue targets | complete/incomplete 概念 | 无 | **阻断** | cue prompt、targets、人工修订、质量统计 |
| OOP recovery targets | 标注流程概述 | 无 | **阻断** | onset、mistake/fix/recovery plan rows |
| VJEPA2 feature stage | backbone/window/stride | public backbone 可另取 | **可重建** | exact revision、processor、batch、产物 |
| Temporal clustering | Ward + adjacency | 通用库可实现 | **可重建** | thresholds、stopping/min segment |
| Caption splitting | Qwen3-VL-235B + prompt | 模型访问视环境而定 | **部分可重建** | exact revision/API/parser/output |
| Grouping/alignment | Llama 4 + prompts | 模型访问视环境而定 | **部分可重建** | full runnable script、error policy |
| Alignment QA | rubric、threshold 5 | prompt 不完整 | **部分可重建** | 4 calibration examples、stats |
| Duplex sampling | 2 FPS、past 8s、约 1:1 | 自有数据可实现 | **冲突后可重建** | silent exclusion 3s/5s、边界规则 |
| Planner sampling | opening + interrupt 概念 | 无 | **阻断** | broader context、row schema、targets |
| Anchor context | current + up to 14 anchors | 可自行实现 | **可重建但非唯一** | exact frames/layout/padding |
| Duplex SFT messages | high-level condition/target | 无 | **阻断** | chat template、serialization、loss mask |
| Planner SFT messages | high-level condition/target | 无 | **阻断** | exact input/output messages、state parser |
| `2x` interrupt loss | 文字提及 | 无代码 | **阻断** | token/sample weighting 定义 |
| Llama optimizer config | 部分超参 | 论文 | **部分可重建** | steps、seed、precision、clip、revision |
| Qwen optimizer config | 较多超参 | 论文 | **部分可重建** | schedule/warmup/precision、exact model |
| Mixture sampler | sqrt proportional | 论文 | **可重建** | 各源最终 row counts/version |
| Inference state machine | factorization和更新原则 | 论文 | **可重建** | parser failure、invalid plan、timeout |
| Official prediction files | 聚合表 | 无 | **阻断** | logits、decisions、utterances、plans |
| Official PWR checkpoints | 论文称 release | 无 | **阻断** | model IDs / files |
| Official training code | 附录点名若干文件 | 无 | **阻断** | repository / release |

### 7.1 可复现等级

| 等级 | 定义 | 当前状态 |
|---|---|---|
| L0 | 官方 C1 baseline、格式和评分复现 | **可以** |
| L1 | 使用公开 C1 数据做 PWR-inspired prompt/state inference | **可以** |
| L2 | 自建 plan/cue targets，按论文规格重写双模型训练 | **可以研究，但实现选择很多** |
| L3 | 复现论文训练数据、官方模型、表格数字 | **当前不可以** |

---

## 8. 对 Small track 的可执行研究含义

本节只给由审计直接推出的边界，不替代后续完整路线规划。

### 8.1 不应做的事

- 不把当前 HF `facebook/wearable-ai` 当作 PWR training repo。
- 不把第三方 ProceduralAgent 实验当作官方权重或结果。
- 不把 STRIDE 的动作区间/边界 supervision 直接改名为 C1 interrupt label。
- 不在没有 oracle-plan 对照前直接上 GRPO；否则无法判断提升来自状态、gate 还是 reward exploitation。
- 不把论文 5,065-row G-Mean 与公开 C1 9,935-chunk Macro F1 直接横比。

### 8.2 最小但有判别力的 PWR-inspired 实验

同一 Small backbone、同一 C1 split、同一 scorer 下做：

1. **No plan**：当前已有模型或 gate。
2. **Oracle compact plan**：人工/离线生成少量高质量当前步骤与 cue，测上限。
3. **Predicted compact plan**：用可部署的状态预测器替换 oracle。
4. **Noisy plan training**：训练时注入步骤错位、漏更新和旧状态，缩小 teacher-forcing gap。
5. **State update ablation**：只在 interrupt 更新 vs 每个 chunk 内部更新、仅对外 gate 控制发言。

建议的 compact state 不是长篇自然语言 plan，而是：

```text
goal_id / task text
current_step_id
progress: not_started | ongoing | complete | deviated | recovered
completion_evidence
incompletion_or_error_evidence
next_step
confidence
last_update_chunk
```

### 8.3 “粒度”在 PWR 框架中的位置

PWR 的 cue 和 step alignment 恰好说明粒度可能是突破点，但必须把它变成可测变量，而不是再造一个模糊概念：

- 过粗：一个 step 横跨多个动作，模型看见局部进展就误报完成；
- 过细：微动作边界太密，正常连续操作被过度 interrupt；
- 合适粒度：状态变化恰好对应“用户此刻需要知道的决策点”。

最有价值的实验不是先训练一个复杂的 granularity module，而是对相同 session 构造 coarse/medium/fine 三种 plan/cue，测 oracle state 对 C1 IF1、SF1、Macro F1 的敏感性。如果 oracle 粒度显著拉开差距，再把粒度预测专门建模；如果不拉开，重点应回到视觉证据、状态噪声和 gate calibration。

### 8.4 没有官方代码时的定位

后续工程应拆成两条记录：

- **paper-faithful fields**：论文明确给出的状态、采样和 loss 概念；
- **our implementation choices**：schema、边界容差、silent exclusion、parser、噪声策略、Small 蒸馏结构。

这样即使官方仓后来发布，也可以逐项 diff，而不是再次整体推倒。

---

## 9. 后续监测点

由于无法联系主办方，只能对公开状态做轻量监测：

1. arXiv `2606.04970` 是否出现 v2；
2. HF dataset `main` SHA、branches/tags 是否变化；
3. HF `facebook` 组织是否出现 wearable/PWR model；
4. workshop GitHub 是否新增 code/data link；
5. 论文页面是否增加 project/GitHub/model ID。

在这些状态变化前，没有必要反复 clone 现有 700-session 大文件；检查 refs、tree 和非视频文件即可。

---

## 10. 最终判定

PWR 提供了当前 C1 最重要的方法论修正：**主动介入不是孤立二分类，而是有显式程序状态、视觉完成条件和恢复机制约束的时序决策。** 这足以改变我们的研究路线。

但 PWR v1 正文和附录并没有形成可执行的完整 release specification。数据版本冲突、未填统计、artifact manifest 所列实体未公开、训练消息格式和 loss 定义缺失，使得忠实复现被阻断；官方训练代码和权重截至审计日也确实没有可获取入口。

因此，下一阶段正确的表述和目标是：

> 先用公开 C1 数据建立可量化的 PWR-inspired Small 路线，验证 compact procedural state、cue、粒度和 noisy-plan robustness；并保持与未来官方 release 可逐项对齐的实现记录。

而不是等待一个当前不存在的官方仓，也不是把高层论文描述误认为已经拿到完整训练管线。
