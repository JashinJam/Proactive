# D1 Utterance 问题诊断与 Planner / Language 后续工作交接

> 日期：2026-07-16  
> 性质：只读审计结论与初步实验建议  
> 当前权威路线：[`CURRENT_ROUTE.md`](../CURRENT_ROUTE.md)  
> 重要说明：本文**不是新的路线决议**。在用户确认前，本文提出的 `U0/U1/P1/J1` 等阶段均为候选方案，不得替代或静默修改 `CURRENT_ROUTE.md`。

## 0. 交接摘要

本次审计确认了一个需要单独处理的系统缺口：当前 D1 `fused_linear` 的官方 OOF Macro F1 为 `0.6341`，但它只改善了 `$interrupt$ / $silent$` 二元决策，没有同步训练或生成与新决策匹配的具体指导内容。

最关键的事实如下：

1. `Please continue with the next step.` 不是模型依据 scalar、tag margin 或 hidden state 生成的文本，而是代码中写死的 fallback。
2. D1 fused 的 4,613 个预测 interrupt 中，有 2,586 个使用该 fallback，占 `56.06%`。
3. 其中 2,565 个是冻结 R0 原始输出明确为 `$silent$`，随后被 D1 gate 改判为 interrupt；系统没有第二个语言生成步骤，因此只能补固定句。
4. D1 fused 的 3,165 个二元 TP 中，1,647 个使用 fallback，占 `52.04%`。这些样本只能证明“interrupt 标签判对”，不能证明提供了有效指导。
5. 当前路线包含 `oracle -> predicted -> noisy state` 的概念骨架，但尚无完整 planner 训练协议、plan/state 字段级验收指标、utterance 监督损失或语言质量 promotion gate。
6. 尚未运行的 final-language-MLP LoRA 仍是 decision-only 实验，并明确在 utterance generation 时关闭 adapter；它不会修复本报告发现的语言问题。

因此，后续应把以下能力分开定义、监督和评价：

```text
程序理解与状态跟踪：用户现在做到哪一步，是否完成、出错或偏离
Planner：下一步或恢复动作应该是什么
Decision gate：现在是否值得主动说话
Utterance generator：决定说话后，具体、准确、可执行地说什么
```

## 1. 术语与系统边界

- **scalar feature**：一个数值特征，例如“是否首 chunk”“当前 chunk 编号”“已经观察了多少秒”。D1 scalar 分支共使用 18 个严格因果标量。
- **tag margin**：冻结模型对 `$interrupt$` 和 `$silent$` 两个固定候选标签的相对偏好分数。
- **hidden state**：冻结 InternVL 在当前因果 prompt 末端得到的 1,024 维内部表示。
- **decision gate**：只输出 interrupt 或 silent 的开关，不负责写具体指导内容。
- **utterance**：`$interrupt$` 后真正给用户看的自然语言内容。
- **fallback**：上游没有可用 utterance 时，由工程代码填入的固定兜底文本。
- **planner**：根据 query、程序步骤和当前状态确定下一步或恢复动作的模块。
- **state updater**：随每个 chunk 更新 `current_step / progress / evidence / confidence` 的模块；它与“生成一份完整计划”不是同一能力。
- **OOF**：out-of-fold。每个 session 的预测由未使用该 session 训练的折模型产生，但所有折仍来自同一个 public validation 分布。

## 2. 已核实的实际输出链路

当前融合部署路径按以下顺序执行：

```text
当前及过去视频 + query + 官方 prior dialog
                    |
                    v
          R0 先做一次自由生成
          raw_response = 文本 / $silent$
                    |
          +---------+------------------+
          |                            |
          v                            v
  构造 scalar / margin / hidden    保留 raw 文本候选
          |
          v
     D1 gate 判定
       /       \
   silent     interrupt
     |           |
  $silent$   R0 有具体文本 -> 原样复用或修复标签
             R0 是 $silent$ -> 写入固定 fallback
```

证据位置：

- fallback 常量：[`src/proactive_r0/core.py`](../src/proactive_r0/core.py#L14)
- D1 最终答案组装：[`src/proactive_d1/core.py`](../src/proactive_d1/core.py#L631)
- 在线执行顺序：[`src/proactive_d1/deploy.py`](../src/proactive_d1/deploy.py#L192)
- scalar 构造：[`src/proactive_d1/core.py`](../src/proactive_d1/core.py#L192)
- gold utterance 被压缩为 0/1 标签：[`src/proactive_d1/core.py`](../src/proactive_d1/core.py#L318)

因此必须明确：

> scalar、tag margin 和 hidden 只参与“要不要 interrupt”的计算，从未进入最终 utterance 的解码过程。

## 3. 全量 Utterance 统计

统计范围为 700 sessions、9,935 chunks。`模板数`只统计精确文本 `Please continue with the next step.`，Gold 中没有该句。

| 输出 | Macro F1 | Interrupt 数 | 模板数 | 模板占 Interrupt |
|---|---:|---:|---:|---:|
| Gold | - | 5,352 | 0 | 0% |
| R0 | 0.4630 | 2,915 | 39 | 1.34% |
| R0-F | 0.5362 | 3,548 | 39 | 1.10% |
| D1 scalar OOF | 0.6119 | 5,214 | 2,937 | 56.33% |
| D1 fused OOF | **0.6341** | 4,613 | 2,586 | **56.06%** |
| D1 单阈值模拟 | 0.6330 | 4,498 | 2,519 | 56.00% |
| D2 residual MLP | 0.6351 | 4,649 | 2,593 | 55.78% |

D1 fused 的模板来源：

| 来源 | 数量 |
|---|---:|
| R0 raw response 为精确 `$silent$`，D1 改为 interrupt | 2,565 |
| R0 raw response 为无正文 `$interrupt$` | 21 |
| 合计 | 2,586 |

D1 fused interrupt 按二元金标拆分：

| 输出内容 | TP | FP | Precision |
|---|---:|---:|---:|
| 固定 fallback | 1,647 | 939 | 63.69% |
| 非 fallback 文本 | 1,518 | 509 | 74.89% |
| 全部 interrupt | 3,165 | 1,448 | 68.61% |

重要解释：

- `非 fallback` 只表示“存在不同于固定模板的文本”，**不等于文本在语义上正确**。
- R0-F 有 2,144 个非模板 TP；D1 fused 只有 1,518 个。D1 的官方 TP 增长主要由模板型 TP 支撑。
- D1 fused 的非模板 TP 与原始 R0 基本相同：`1,518` 对 `1,514`。
- fallback 出现在 638/700 个 session 中，占 `91.14%` 的 session。
- 对 chunk index 1，即每个 session 的第二个 chunk，D1 fused 共预测 426 个 interrupt，其中 423 个为 fallback，占 `99.30%`。

这说明 D1 学会了在若干位置提高 interrupt 概率，但没有同时获得“此刻具体应当说什么”的内容能力。

## 4. 对“D1 是否在猜 scalar 与答案规律”的诊断

### 4.1 成立的部分

D1 本质上是监督二分类器，目标就是学习 causal feature 与 public gold interrupt/silent 标签之间的统计关系。

消融结果为：

| 特征条件 | OOF Macro F1 |
|---|---:|
| R0-F | 0.5362 |
| 7 个 temporal scalar | **0.6081** |
| temporal + domain + raw-response，共 18 scalar | 0.6119 |
| tag margin only | 0.5313 |
| hidden only | 0.6031 |
| scalar + tag margin + hidden | **0.6341** |

从 `0.5362 -> 0.6081` 的主要增益来自：

- 是否首 chunk；
- chunk 序号；
- 已观察时间；
- interval 时长与间隔；
- 已见 dialog 数量与输入帧数。

这更接近 **annotation-policy calibration**，即学习公开数据的标注节奏，而不是证明当前步骤、完成 cue 或错误恢复已经被理解。

### 4.2 不应过度推断的部分

不能将 D1 描述为“纯 scalar 随机猜测”：

- fused 相对完整 scalar 仍提高 `+0.0222`；
- 五折和四个 domain 的增量具有稳定性；
- tag margin 与 hidden 提供了互补信号。

但这也不能被表述为“已经获得程序理解”：

- hidden-only 仍低于 scalar；
- 当前没有 current-step、cue、plan 或 recovery 监督；
- hidden 的增量也可能包含 prompt、语言形式和时序相关信号，尚未被单独归因。

### 4.3 不是直接 label leakage，但存在 dataset shortcut 风险

当前有效 D1 v2 已删除 answers 后再造 feature，并使用 session-level OOF；同一 session 的 chunk 不跨训练和测试折。因此没有发现逐样本直接读取 gold 的泄漏。

但 OOF 不能消除以下风险：

- 700 sessions 都来自同一个 public validation 分布；
- 699/700 个首 chunk 为 interrupt；
- 第二个 chunk 和 interval 划分具有统一标注规律；
- domain、dialog 长度和 R0 输出形式携带数据集先验；
- R0-F 规则和多轮实验选择都曾查看 public validation 表现。

正确表述应为：

> 当前 D1 是严格因果、session-held-out 的 public-validation-supervised 决策校准结果，但仍可能依赖不能稳定迁移到隐藏测试集的 dataset shortcut。

## 5. 当前各实验实际回答的问题

### R0

- 冻结 InternVL3.5-1B 做自由生成，同时生成决策标签和具体文本。
- 主要问题是漏报：预测 interrupt 率 `29.34%`，Gold 为 `53.87%`。
- 只有 39 个模板输出，说明 R0 在主动选择 interrupt 时通常能生成非模板文本。
- 没有对文本的正确性、具体性和可执行性进行评价。

### R0-F

- 只修复输出格式：将非空但缺标签的文本解释为 interrupt。
- 不重新推理、不训练参数、不增加 planner 或语言能力。
- Macro `0.5362`，主要收益来自首 chunk 格式修复。

### R1 Oracle compact-state pilot

- 4 sessions、50 chunks，手工构造严格因果 Oracle 状态。
- `null / step / cues / full` 均为零样本 prompt 变体，没有训练 planner。
- full 相对 null 增加 interrupt recall，但同时增加 FP，两者 Macro 都是 `0.5169`。
- 只评价二元决策，没有评价计划质量或 utterance 质量。
- 它证明了协议和工程可运行，没有证明 plan state 无效，也没有构成 planner 能力实验。

### D1 scalar / neural fused

- D1 scalar 证明公开标注有很强的时序和响应形式规律。
- D1 fused 证明 margin/hidden 对强 scalar control 有稳定互补增量。
- D1 不训练 utterance；当 gate 推翻 R0 silent 时使用固定 fallback。
- `0.6341` 是当前 active decision baseline，不是完整 assistant-quality baseline。

### D2 residual MLP

- 只测试线性 gate 是否缺少一点非线性容量。
- OOF Macro `0.6351`，仅比 D1 高 `+0.0010`，bootstrap 跨 0，已经否决。
- 模板率仍为 `55.78%`，没有修复语言问题。

### Final-language-MLP LoRA 冻结协议

- 尚未运行完整缓存提取和五折 OOF。
- 训练目标仍是 interrupt/silent tag-margin BCE。
- 输入仍是 scalar、tag margin 和 hidden。
- adapter 在 utterance generation 时明确关闭，只影响 decision feature。
- 该实验只能回答“轻量表示适配能否改善 gate”，不能回答 planner 或 language 问题。

### 阈值与推理加速

- 单阈值审计只验证 fold threshold 向部署 threshold 的运输稳定性。
- shared-vision 只做等价加速，decision、answer 与顺序参考逐元素一致。
- 两者均不改变模型能力或 utterance 内容。

## 6. 当前路线对 Planner 与 Language 的覆盖情况

### 已经存在的概念骨架

`CURRENT_ROUTE.md` 已经提出：

```text
compact state updater
current_step / progress / evidence / next_step / confidence
        -> interrupt-silent decision
        -> interrupt + concise utterance
```

并规划了：

- R1：oracle compact state；
- R2：coarse / medium / fine plan granularity；
- R3：predicted causal state updater；
- R4：noisy-plan robustness；
- R5：再决定监督训练、蒸馏或 GRPO。

### 尚未落实的部分

Planner / state 侧缺少：

- initial procedural plan 的 target 来源；
- static plan 与 dynamic state 的明确分工；
- `current_step / progress / cue / next_step / recovery` 的监督格式；
- field-level loss 和评价指标；
- planner/state updater 的参数、时延和 Small 预算；
- Oracle、predicted、noisy state 的正式 promotion gate。

Language 侧缺少：

- utterance SFT 或其他语言监督；
- 只在 interrupt 样本上计算的语言 loss；
- plan 与 utterance 的一致性约束；
- generic/repetition/groundedness/actionability 指标；
- 人工内容评价协议；
- “官方 Macro 不退化且内容显著提升”的联合 promotion gate。

因此，当前计划的准确状态是：

> 有 plan/state 的研究方向和阶段名称，但没有完整 planner 训练验收方案；utterance 能力尚未形成正式实验路线。

## 7. 建议的模块分解

执行 Agent 不应把 planner、state updater、gate 和 language 混成一个不可归因的模块。

```text
query / task
    |
    v
Initial planner
生成静态步骤、完成/未完成 cues
    |
    v
Causal state updater <---- 当前及过去视频 + prior dialog
current_step / progress / evidence / deviation
    |
    +-------------------+
    |                   |
    v                   v
Decision gate       Recovery / next-step planner
现在是否说话          应该指导哪一步或如何纠错
    |                   |
    +---------+---------+
              v
      Utterance generator
      具体、简洁、可执行的话语
```

建议的监督目标：

```text
L_state：current_step、progress、cue、evidence、deviation、recovery
L_gate：class-balanced interrupt / silent
L_text：仅在 gold interrupt 样本上计算的 utterance token loss
L_consistency：utterance 与 next_step / recovery_action 的一致性
```

联合训练时必须分别归一化各项 loss。不能直接将几十个 utterance token 与一个 decision token 相加，否则语言 loss 会因 token 数量更大而淹没 gate。

## 8. 候选执行阶段

以下为初步方案，执行前需要在 `CURRENT_ROUTE.md` 中得到用户确认并冻结协议。

### U0：全量内容审计

目的：建立语言问题的正式基线，不训练模型。

已完成的自动统计：

- interrupt、fallback、non-fallback 数量；
- fallback 来源；
- TP/FP 拆分；
- 按 chunk 位置的异常；
- R0、R0-F、D1、D2 对照。

待执行：

- 按 domain、task、chunk 位置、TP/FP/FN 分组；
- 统计 session 内重复率、过度泛化句和动作/对象缺失；
- 抽取约 200 个平衡样本，开展双人 blind rubric；
- rubric 至少包含 correctness、specificity、actionability、visual grounding、plan consistency、conciseness、safety。

U0 不需要 GPU。所有结果写入新的实验目录和中文报告，不覆盖现有预测。

### U1：固定 D1 Gate 的 Forced-Interrupt Generation Pilot

目的：回答“R0 只是因为自己的 gate 选择 silent，还是模型确实不知道具体该说什么”。

对象：优先抽取 D1 fused 判 interrupt、R0 raw 为 `$silent$` 的样本；不得根据 gold 文本挑样本。

建议变体：

| 变体 | Decision | 内容生成条件 |
|---|---|---|
| current_fallback | 固定 D1 | 当前固定句 |
| forced_no_state | 固定 D1 | 强制 `$interrupt$` 前缀后继续生成，无 plan |
| forced_oracle_step | 固定 D1 | 加 answer-blind oracle current/next step |
| forced_oracle_full | 固定 D1 | 加 step、progress、evidence、recovery |

严格要求：

- 四个变体的 interrupt/silent 决策必须 100% 相同；
- 只比较 utterance，不允许重新调 threshold；
- Oracle 状态必须由当前视频前缀、query 和 prior dialog 构造，不读当前或未来 answer；
- 先做小规模 GPU smoke，再由用户授权是否扩大；
- 不直接修改当前 `decision_answer()` 部署路径，应建立隔离实验实现。

这个实验是当前最有判别力、成本最低的下一步：

- `forced_no_state` 已明显改善，说明首要问题是 gate/text 接口断裂；
- 只有 oracle state 改善，说明 planner/state 信息是关键；
- 两者都差，说明 1B backbone 的语言/程序能力本身需要 SFT 或 LoRA。

### P1：更大规模 Oracle Plan / State 复验

目的：分别测量状态对 gate 和 utterance 的上限，而不是只看一个综合分数。

要求：

- 旧 4 sessions 仅用于 debug，不再作为正式评价集；
- 样本选择和标注预算在看结果前冻结；
- 明确区分 static plan、dynamic observed state、next step 和 recovery plan；
- 固定现有 D1 split/scorer，并额外报告 state 字段指标；
- Oracle state 输入仍是非部署上限，不能称作提交系统。

建议字段指标：

- current-step exact / ±1 accuracy；
- progress Macro F1；
- completion/error cue precision、recall；
- next-step accuracy；
- state update lag 和 staleness；
- recovery detection / recovery-action accuracy；
- transition validity。

### U2：Utterance Supervision

公开数据中有 5,352 条 gold interrupt utterance，其中 5,332 条文本不同，可作为 fit folds 上的语言 target。

建议比较：

- no-state utterance SFT；
- oracle-state-conditioned utterance SFT；
- predicted-state-conditioned utterance SFT；
- noisy-state-conditioned utterance SFT。

训练边界：

- 只使用 fit folds 的 gold utterance；
- calibration/test fold utterance 只能用于评价；
- silent 样本不计算正文语言 loss；
- public labels 训练出的模型统一标记 `val-supervised`；
- 任何外部 teacher 必须先完成来源、license、可公开性与奖项资格审计。

### P2：Causal Predicted State Updater

当 P1 证明 oracle state 对 gate 或 utterance 至少一项有稳定收益后，再训练 predicted state updater。

必须报告：

- state 本身的字段级准确率；
- oracle-to-predicted 性能差距；
- 错误状态对 gate 和 utterance 的不同影响；
- 每 chunk 内部更新与只在 outward interrupt 更新的对照；
- 参数、时延和峰值显存。

### J1：联合监督消融

只有 gate、state、utterance 三条单独实验都有可解释结果后，才进入联合训练。

最小消融矩阵：

| 变体 | Gate loss | State loss | Text loss |
|---|---:|---:|---:|
| decision-only | ✓ | - | - |
| decision + state | ✓ | ✓ | - |
| decision + text | ✓ | - | ✓ |
| decision + state + text | ✓ | ✓ | ✓ |

必须同时报告：

- 官方 Macro、Interrupt F1、Silent F1、TP/FP/TN/FN；
- state 字段指标；
- fallback/generic/repetition；
- utterance 人评和自动诊断；
- 每折、每 domain、首/中/尾位置结果；
- 参数与部署延迟。

### R4 / GRPO

- 现有 noisy-plan robustness 方向继续保留；
- 一阶 lag、漏更新、错完成、错 step、stale cue、recovery error 应分别测试；
- GRPO 仍不是近期默认步骤；只有监督基线稳定、planner/state 确有价值、语言评价可审计、残差明确后才考虑。

## 9. 语言评价协议建议

官方 C1 只排名 interrupt/silent，因此必须建立独立的研究诊断，且不能把它与官方 Macro 混成一个未经定义的总分。

### 自动指标

- exact fallback rate；
- generic phrase rate；
- session 内重复率；
- utterance 长度与冗余；
- action / object coverage；
- 与 gold utterance 的 BERTScore 或其他语义相似度，只作辅助；
- 与结构化 next-step / recovery-action 的一致率；
- 错误对象、错误动作和不可见事实的 hallucination rate。

### 人工 rubric

建议 1--5 分：

- **Correctness**：指导动作是否正确；
- **Timeliness**：此刻说是否合适；
- **Specificity**：是否指出具体动作、对象或错误；
- **Actionability**：用户能否直接照做；
- **Groundedness**：是否符合当前可见视频与已有状态；
- **Plan consistency**：是否匹配当前步骤、下一步或恢复动作；
- **Conciseness**：是否足够简洁而不丢失必要信息；
- **Safety**：是否可能引导危险或明显错误操作。

评价应分两层：

1. **Gold-interrupt evaluation**：固定“应该说话”，单独测内容能力；
2. **End-to-end evaluation**：只在模型预测 TP 上评价内容，同时保留 FP 的打扰成本。

## 10. Split 与泛化诊断

现有 public set 有 700 sessions、577 个不同 task 名称；112 个 task 名称重复，涉及 235 sessions，最大重复 4 次。

因此建议保留两种不同口径：

- **session-level OOF**：与当前 D1 直接比较，服务排行榜开发；
- **task-held-out diagnostic**：相同或近似 task 不跨训练与测试，用于判断 planner/language 是否只记住步骤模板。

task-held-out 只作为附加诊断，不能替换当前官方 scorer 或静默改变 active baseline。

## 11. 与官方离线 Dialog 的关系

当前每个 chunk 使用数据中官方提供的 `dialog[i]`，并不会把 D1 自己生成的上一轮 utterance 追加回后续输入。

这意味着：

- 当前固定 fallback 不会在离线评测中污染后续 dialog；
- 改善 utterance 也不会通过真实用户反馈自动改善后续 D1 决策；
- 语言监督若改善 gate，只能来自共享表示或联合训练，而不是当前离线闭环；
- 真实交互系统中的“错误指导 -> 用户反应改变 -> 后续状态改变”尚未被 C1 当前离线协议测量。

执行报告必须区分“官方离线 C1 能力”和“真实闭环主动助手能力”。

## 12. 执行 Agent 的强制边界

执行 Agent 开始前必须按项目规范阅读：

1. [`Agent.md`](../Agent.md)
2. [`CURRENT_ROUTE.md`](../CURRENT_ROUTE.md)
3. [`C1_SPEC.md`](../C1_SPEC.md)
4. [`PWR_audit.md`](../literature/papers/challenge1_proactive/PWR_audit.md)
5. D1/R1/D2 对应 README、报告、配置与测试

不得：

- 把本文直接当作已经批准的 active route；
- 修改或覆盖现有 D1/R0/R1/D2 预测和实验目录；
- 复用旧 4-session R1 pilot 作为正式效果评价集；
- 使用未来视频、未来 dialog、当前/未来 gold utterance 构造推理输入；
- 将 public-validation-supervised 结果描述为隐藏测试泛化；
- 将非模板输出直接描述为语义正确；
- 将 final-language-MLP LoRA 描述为语言训练；
- 未经用户授权启动完整多 GPU 提取、训练、外部 API 标注或 leaderboard 提交；
- 在同一批结果上反复搜索 loss 权重、LoRA rank、层数或语言评价阈值。

必须：

- 一个实验只回答一个主要问题；
- 在看结果前冻结样本、split、prompt、loss、评价和 promotion gate；
- 使用官方 scorer 报告决策指标；
- 单独报告 planner/state/language 指标；
- 所有面向汇报的报告使用中文；
- 记录数据、代码、模型、配置、预测和 annotation 的 SHA256；
- 所有部署组件计入 Small 的 2B 总参数预算。

## 13. 推荐执行顺序

若用户批准补充路线，建议顺序为：

```text
U0 全量内容审计
  -> U1 固定 gate 的 forced-generation pilot
  -> P1 更大 Oracle state：分别测 gate 与 utterance 上限
  -> U2 state-conditioned utterance SFT
  -> P2 predicted state updater
  -> J1 decision/state/text 联合消融
  -> noisy-state robustness
  -> 仅在条件满足后考虑 GRPO
```

当前冻结的 D2 final-MLP LoRA OOF 可保留为独立的 gate representation-adaptation 对照，但不能被用来回答本文的 planner/language 问题。是否先运行 D2，还是先执行 U0/U1，应由用户在更新 `CURRENT_ROUTE.md` 时明确决定。

## 14. 交接后的第一项可执行工作

在不启动 GPU、不修改 active route 的前提下，执行 Agent 可以先完成：

1. 将本报告的自动统计固化为可复现只读审计脚本；
2. 输出按 domain、task、chunk position、TP/FP/FN 分组的 utterance 统计；
3. 生成 answer-blind 的人工评价样本清单和 rubric 模板；
4. 起草 U1 forced-generation pilot 的预注册协议；
5. 提交给用户审阅样本选择、GPU 预算和 promotion gate，获得授权后再运行推理。

该工作包不改变模型、预测或排行榜候选，适合作为负责执行 Agent 的起始任务。

## 15. 主要证据索引

- R0 实验：[`reports/20260713_internvl35_1b_no_plan_r0.md`](20260713_internvl35_1b_no_plan_r0.md)
- R0-F 实验：[`reports/20260714_internvl35_1b_r0f_format_ablation.md`](20260714_internvl35_1b_r0f_format_ablation.md)
- R1 pilot：[`reports/20260714_internvl35_1b_oracle_state_r1_pilot_v1.md`](20260714_internvl35_1b_oracle_state_r1_pilot_v1.md)
- D1 scalar：[`reports/20260714_internvl35_1b_causal_scalar_decision_head_d1_oof_v2.md`](20260714_internvl35_1b_causal_scalar_decision_head_d1_oof_v2.md)
- D1 fused：[`reports/20260715_internvl35_1b_neural_decision_head_d1.md`](20260715_internvl35_1b_neural_decision_head_d1.md)
- D1 threshold：[`reports/20260715_internvl35_1b_d1_threshold_robustness.md`](20260715_internvl35_1b_d1_threshold_robustness.md)
- D1 inference：[`reports/20260715_internvl35_1b_d1_inference_optimization.md`](20260715_internvl35_1b_d1_inference_optimization.md)
- D2 residual MLP：[`reports/20260715_internvl35_1b_residual_mlp_d2.md`](20260715_internvl35_1b_residual_mlp_d2.md)
- D2 LoRA feasibility：[`reports/20260715_internvl35_1b_final_mlp_lora_feasibility.md`](20260715_internvl35_1b_final_mlp_lora_feasibility.md)
- D1 fused predictions：[`output/experiments/20260714_internvl35_1b_neural_decision_head_d1_oof_v1/variants/fused_linear/predictions.jsonl`](../output/experiments/20260714_internvl35_1b_neural_decision_head_d1_oof_v1/variants/fused_linear/predictions.jsonl)
- R0 session records：[`output/experiments/20260713_internvl35_1b_no_plan_r0/session_records.jsonl`](../output/experiments/20260713_internvl35_1b_no_plan_r0/session_records.jsonl)
- Gold public data：[`data/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl`](../data/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl)

