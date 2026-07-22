# U0 双评测员聚合分析协议

> 冻结日期：2026-07-20  
> 范围：U0 200 条困难分层样本的 Reviewer A/B 首轮评分  
> 性质：public-validation 人工诊断，不替代官方 scorer，不估计总体发生率

## 1. 目标

本分析回答三个问题：

1. U0 的时机与内容结论在两名评测员之间是否稳定；
2. 分歧集中在哪些维度、chunk 位置、fallback/非 fallback 和官方 outcome；
3. 哪些 early chunks 最适合进入后续固定 D4 gate 的 utterance/grounding 对照。

本分析不得修改 A/B 原始评分，不得用仲裁值替换主要数值结果，也不得把 200 条分层样本
解释成 9,935 chunks 的总体 prevalence。

## 2. 冻结输入

- Reviewer A：`src/proactive_u0/u0_reviewer_A_ratings.csv`
- Reviewer B：`src/proactive_u0/u0_reviewer_B_ratings.csv`
- 盲评输入：U0 冻结实验的 `review_items_blind.jsonl`
- 独立答案键：U0 冻结实验的 `review_key.jsonl`

程序必须先校验两份评分的 SHA256、每个 `review_id` 恰有 A/B 各一条、无重复和缺失，
再读取答案键做分层分析。

## 3. 条件评分规则

每条样本均要求：

- `should_interrupt`
- `decision_confidence_1_5`
- `timeliness_1_5`

当 `model_action=silent` 时，内容分、内容 flags 和 `primary_error_type` 必须为空。
当 `model_action=spoke` 时，七个内容分、三个 flags 和 `primary_error_type` 必须完整。
空内容分不是 0 分，不能参与均值或一致性计算。

## 4. 主要统计

连续/有序维度：

- 分别报告 A、B 均值及 `B - A`；
- 报告 A/B 算术平均后的主分析值；
- 报告 exact agreement、within-one agreement、mean absolute difference；
- 报告 quadratic-weighted Cohen's kappa、Pearson 和 Spearman correlation；
- 对主均值和 `B - A` 使用 session-level cluster bootstrap 95% CI。

分类/二元维度：

- `should_interrupt`、三个 flags 和 `primary_error_type` 报告原始列联表；
- 报告 exact agreement 和 unweighted Cohen's kappa；
- 低发生率字段不以 kappa 单独决定结论。

内容主汇总：

```text
content_composite = mean(
    correctness,
    specificity,
    actionability,
    groundedness,
    plan_consistency
)
```

`conciseness` 和 `safety` 单独报告，不用于补偿内容错误。

## 5. 分层

至少报告：

- `stratum`：`tp/fp x fallback/nonfallback` 与 `fn_silent`；
- chunk 位置：first、second、2--4、5--9、10+；
- domain；
- fallback、nonfallback、silent；
- official confusion outcome；
- A/B 的 `primary_error_type` 分布。

重点比较 second 与 2--4 early chunks。分层结果只描述冻结困难样本，不外推总体比例。

## 6. 分歧清单

满足任一条件即进入仲裁清单：

- 任一 1--5 分相差至少 2；
- `should_interrupt` 一人为 yes、另一人为 no；
- generic、hallucination 或 unsafe flag 不一致；
- `primary_error_type` 不同。

清单保留 A/B 原始值、盲评上下文标识、官方 stratum 和候选 utterance。主要统计仍使用
原始 A/B 平均；后续人工仲裁只能作为单独解释字段。

## 7. Bootstrap

- 单位：session，即 `input_index`；
- resamples：10,000；
- seed：20260720；
- 每次有放回抽取与原 session 数相同的 session，并保留被抽中 session 的全部样本；
- percentile 95% CI。

## 8. 后续 early-chunk 选择边界

后续 utterance 实验可以使用本分析选择 review-informed development 样本，但必须：

- 明示样本已被人工评分并来自 public validation；
- 固定 D4 gate，不用 U0 分数修改 gate、特征或阈值；
- 不读取未来视频、未来 dialog 或当前/future gold answer 作为生成输入；
- 不把自动文本变化当作 grounding 质量结论；
- 新生成方案需另建盲评包后才能比较 hallucination/groundedness。

## 9. 完成条件

- 输入哈希和 400 条 reviewer rows 完整校验；
- 200 条 item-level A/B 聚合记录可复现；
- 一致性、bootstrap、全部冻结分层和分歧清单落盘；
- 生成中文报告，明确事实、解释边界与 early-chunk 后续选择规则；
- 更新 `CURRENT_ROUTE.md` 中 U0-B 的过时状态。
