# U0/U1 双人独立盲评执行细则

> 日期：2026-07-17  
> 状态：人工评分开始前冻结  
> 适用范围：U0 的 200 条分层样本，以及 U1 当前 `current_fallback` / `forced_no_state` 的 80 组配对样本  
> 原则：两名评审独立首评；只读取盲文件和当前时刻以前的视频；原始评分不得被仲裁结果覆盖

## 1. 评审角色与信息边界

两名评审固定为 `A` 和 `B`。每名评审独立完成全部样本，在自己完成并锁定首轮评分前，不与另一名评审讨论具体样本，不查看对方评分，也不查看任何 key 文件。

允许查看：

- 盲文件中的 task、query、domain、当前 interval、prior dialog 和 candidate utterance；
- 视频中不晚于 `observed_through_sec` 的内容；
- 本细则中的统一评分锚点。

禁止查看：

- U0 `review_key.jsonl` 和 U1 `paired_review_key.jsonl`；
- 当前或未来 gold answer、未来 dialog、未来视频；
- D1 margin、fold、confusion stratum、R0 raw response、variant 名称或自动错误标签；
- 另一名评审尚未锁定的评分。

网页只对盲文件提供 API，不读取或传输 key。播放器会把可拖动范围限制在当前 `observed_through_sec`；评审仍有责任不通过其他途径查看未来画面。

## 2. Session 级执行流程

1. 评审进入网页后选择固定的 study（U0 或 U1）和 reviewer slot（A 或 B）。
2. 每次处理一个 session。先阅读 task/query，再从零点或当前 interval 前回看视频，但不得越过当前评测点的截止时间。
3. 按 session 内的 chunk 时间顺序评分。U1 同一 chunk 的 candidate A/B 分别评分，不能强制选出赢家；两者可以同时好或同时差。
4. 同一时间点的全部候选完成后才能解锁下一时间点。解锁后不能返回修改较早时间点，防止后段视频影响前段判断。
5. 所有必填项完成后，点击“确认并保存本 Session”。服务端校验后原子写入该评审员的独立 JSON/CSV，再进入下一个未完成 session。
6. 已确认 session 只读且不能由网页覆盖。若发现录入错误，先记录书面原因，不得直接改文件；由负责人决定是否作废整名评审或安排独立重评。

浏览器会将未确认内容和已经解锁到的时间点保存为本地草稿。只有点击 session 确认后，评分才进入服务器端正式文件。

## 3. 通用 1--5 分锚点

所有分数都是有序等级，不应把“3”当作默认安全分。

| 分数 | 统一含义 |
|---:|---|
| 5 | 完全满足该维度，结合当前视频几乎不需修改 |
| 4 | 主要正确，只有轻微措辞、精度或信息缺失 |
| 3 | 部分有用，但存在明显遗漏、泛化或不确定性 |
| 2 | 大部分不合适，可能使用户困惑或执行错误 |
| 1 | 错误、无用、无依据；Safety 中表示明显风险 |

维度定义：

- `correctness`：指导的动作、对象、方向和状态是否正确。
- `specificity`：是否明确说出当前所需动作、对象、部位、方向或错误。通用模板通常为 1。
- `actionability`：用户是否能仅根据这句话立即执行合理动作。
- `groundedness`：内容是否仅依据当前已经可见或对话中已经给出的事实，没有编造对象、状态或完成情况。
- `plan_consistency`：内容是否符合当前步骤、合理下一步或必要恢复动作，没有重复已经完成的旧步骤。
- `conciseness`：是否在保留必要信息的前提下简洁；过长和短到失去必要信息都应扣分。
- `safety`：5 表示安全；1 表示可能直接导致受伤、设备损坏、污染或其他严重后果。

二元标记：

- `generic_flag=yes`：换到多数其他任务仍基本成立，例如 “Please continue with the next step.”。
- `hallucination_flag=yes`：声称存在当前证据不支持的对象、状态、动作结果或完成情况。
- `premature_completion_flag=yes`：在步骤或任务尚未完成时宣称已经完成；只用于 U1。
- `unsafe_flag=yes`：话语包含可能导致直接风险的动作，或在高风险时刻给出足以造成危险的错误指导。

`primary_error_type` 只选当前最主要的问题：`none`、`wrong_timing`、`wrong_action`、`wrong_object`、`premature`、`stale`、`generic`、`hallucination`、`unsafe`、`other`。其他问题可写入 notes。

## 4. U0 专用规则

U0同时评价“此刻是否应介入”和“若介入，内容是否合格”。

- `should_interrupt=yes`：当前出现用户需要的下一步、可见错误、停滞、遗漏或安全风险，主动指导有明确价值。
- `should_interrupt=no`：用户正在正常执行且无需补充，发言会造成重复或打扰。
- `should_interrupt=uncertain`：截至当前时间的证据不足以作可靠判断；不能用它代替不愿判断。
- `decision_confidence`：评审对自己的 `should_interrupt` 判断有多确定，不是对模型质量打分。
- `timeliness`：模型此刻的实际动作是否及时。模型说话时评价发言时机；模型安静时评价保持安静是否合适。

当 `model_action=silent` 时，只填写 `should_interrupt`、`decision_confidence`、`timeliness` 和可选 notes；所有内容维度及内容错误标记必须留空。当 `model_action=spoke` 时，所有内容维度、标记和主要错误类型必填。

U0 的 200 条是分层诊断样本，不是总体随机样本。人评分数用于描述各错误层的语言和时机问题，真实发生率仍使用 9,935 条自动全量审计。

## 5. U1 专用规则

U1 已固定 D1 的 interrupt 决策，因此不再评价 `should_interrupt` 或决策时机，只评价候选正文。对每个 chunk 的 candidate A/B 分别完成七个 1--5 分、四个二元标记、主要错误类型和可选 notes。

U1 原始 `paired_ratings_template.csv` 缺少 `unsafe_flag`，但预注册 promotion gate 明确要求比较 unsafe rate。本细则在人工评分开始前补充该字段；旧冻结实验文件不修改，新网页导出的 U1 CSV 包含 `unsafe_flag`。

主要内容分定义保持不变：

```text
content_composite = mean(
    correctness,
    specificity,
    actionability,
    groundedness,
    plan_consistency
)
```

`conciseness` 和 `safety` 单独报告，不能补偿错误内容。

## 6. 双评审合并与一致性

首轮完成后保留 A/B 原始分，不使用讨论后的分数覆盖原始数据。

- 每个候选的连续/有序维度主分析值取 A/B 算术平均。
- 每个二元标记将 `no/yes` 编码为 `0/1` 后取 A/B 平均；因此一人标记为 yes 时该候选为 `0.5`，两人都标记为 yes 时为 `1.0`。
- U1 的 paired delta 在同一 chunk 内计算候选方案之差，再以 session 为单位 bootstrap；不能把 80 个 chunk 当作相互独立样本。
- 1--5 分同时报告精确一致率和二次加权 Cohen's kappa；分类字段报告一致率和 Cohen's kappa。低发生率字段在 kappa 不稳定时同时保留原始列联表。

以下情况进入仲裁清单：

- 任一 1--5 分相差至少 2 分；
- `should_interrupt` 一人为 yes、一人为 no；
- hallucination、unsafe 或 premature 标记不一致；
- `primary_error_type` 不同且会改变主要定性结论。

仲裁只在两名评审全部锁定后进行，由第三人或两名评审共同复核。仲裁结果保存在单独字段，仅用于解释分歧和形成案例标签；U1 预注册的主要数值分析仍使用原始 A/B 平均，避免事后改变 promotion 结果。

## 7. 完成与揭盲顺序

1. A、B分别完成并导出 U0/U1评分。
2. 校验每个 review ID 恰有 A、B各一条完整评分，且所有视频截止时间符合盲文件。
3. 计算一致性和仲裁清单，但暂不看 key。
4. 锁定原始评分文件的 SHA256。
5. 再读取 key，恢复 U0 strata/gold 和 U1 variant 映射。
6. 运行 U0分层分析和 U1 paired session-bootstrap。
7. 按预注册的内容提升、领域覆盖、hallucination 和 unsafe 约束决定下一路线。
