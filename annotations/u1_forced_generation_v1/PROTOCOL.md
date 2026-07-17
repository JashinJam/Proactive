# U1 固定 D1 Gate 的 Forced-Generation 协议

## 1. 目的

U1 只回答：当 D1 已经决定 interrupt 时，当前 1B backbone 是否能生成具体内容，以及 answer-blind oracle plan/state 是否提供额外信息。

U1 不改变、重训或重新校准 D1 gate。四个内容变体在完整 700-session prediction 中的 `$interrupt$/$silent$` 决策必须逐 chunk 完全一致，官方 Macro F1 必须保持 `0.6341`。

## 2. 冻结样本

- 候选条件：D1 fused OOF 输出为精确 fallback，且冻结 R0 raw response 精确为 `$silent$`。
- 选择过程读取 query、domain、task、interval、prior dialog、D1 output 和 R0 raw response，不读取当前 gold decision 或 utterance。
- 排除旧 R1 pilot 的 input indices `14/123/326/687`。
- 每个 domain 选择 5 个同时具有 second、2--4、5--9、10+ 候选的 session。
- 每个 session 在四个位置各取一个 SHA256 最小 chunk。
- 总计 20 sessions / 80 chunks；每个 domain 和位置各 20 chunks。
- 每个 domain 排名第一的 session 构成 16-chunk engineering smoke。

冻结文件：

```text
sample_items.jsonl
SHA256 de38746a55fa7649615e4b6405b6d4904d6a891ca49ca0680b34219a2efbb974

manifest.json
SHA256 704e7f15d6da9446e03641cd345e421db157b47635ae3a7c625723229dc2e8be
```

## 3. 变体

| 变体 | 决策 | 生成条件 |
|---|---|---|
| `current_fallback` | 冻结 D1 | 当前固定句，不运行新生成 |
| `forced_no_state` | 冻结 D1 | 原始因果帧、query、prior dialog；assistant 端预填 `$interrupt$` 后续写正文 |
| `forced_oracle_step` | 冻结 D1 | 同上，加 answer-blind `current_step` 和 `next_step` |
| `forced_oracle_full` | 冻结 D1 | 同上，加 progress、已观察 evidence、error/deviation 和 recovery action |

三个生成变体共享相同的 forced-generation system suffix、assistant prefix、frames、history、greedy decoding 和 token budget。Oracle 变体只增加冻结的 state block。

## 4. 因果标注边界

静态 plan 只允许读取 task/query。动态 state 只允许读取：

- task/query；
- 当前 chunk 前的官方 prior dialog；
- 视频中不晚于 `observed_through_sec` 的内容。

禁止读取当前或未来 `answers`、未来 dialog、未来视频、R0/D1 error category。标注文本不得包含 `$interrupt$`、`$silent$`、`should speak` 或 gold utterance 的改写。

Plan 使用层级结构：静态 `steps` 保存 macro step；动态 state 的 `current_step_id/next_step_id` 指向它们，evidence/recovery_action 保存当前所需的 atomic action。U1 不训练独立 granularity predictor。

## 5. Engineering Smoke Gate

在 16 个冻结 smoke chunks 上：

1. 使用原始 R0 路径重放，16/16 raw response 必须仍为精确 `$silent$`；否则说明帧或 prompt 没有复现，停止实验。
2. `forced_no_state` 必须保持 16/16 decision 为 interrupt。
3. 输出必须记录空 continuation、重复 decision tag、生成 `$silent$`、截断和异常终止；任何异常都不能被静默替换后计为成功。
4. 完整 prediction 文件通过 schema 和官方 scorer，决策及 Macro 必须与 D1 完全一致。
5. 记录 GPU、峰值显存、总时间和每 chunk 时间。

Smoke 只验证工程，不用于判断语言或 state 是否有效。

## 6. 正式内容评价

正式 pilot 使用全部 80 chunks，由两名评审独立打分：correctness、specificity、actionability、groundedness、plan consistency、conciseness、safety，均为 1--5。为避免等待 oracle 标注时阻塞 interface 判断，同时避免跨批次评分尺度漂移，冻结为两个独立盲包：

1. **Interface package**：`current_fallback` 与 `forced_no_state` 在每个 sample 内随机映射为 A/B，共 160 candidates；只用于 interface promotion。
2. **State package**：`forced_no_state`、`forced_oracle_step`、`forced_oracle_full` 在每个 sample 内重新随机映射为 A/B/C，共 240 candidates；三者全部重新评分，不复用 interface package 中的 no-state 分数；只用于 state/full-vs-step promotion。

每个包内部的 candidates 再以冻结 seed 全局打乱；blind 文件不暴露 variant、gold、D1 margin 或错误类别。两包使用相同 rubric 和两位 reviewer，但所有 promotion contrast 只在同一包内计算。

主要内容分：

```text
content_composite = mean(
  correctness,
  specificity,
  actionability,
  groundedness,
  plan_consistency
)
```

自动指标只报告 nonempty、fallback、generic、exact repetition、extra-tag、长度，不代替语义人评。

评分统计在读取有效评分前冻结如下：

- 每个 candidate 必须由 reviewer slots `A/B` 各独立评分一次；两个评分取算术均值；
- `content_composite` 先在每位评审的五个主要维度内取均值，再在两位评审间取均值；
- paired delta 为同一 sample 的目标变体减 reference；总体不把 4 个 chunks 当作独立 session，置信区间以 20 个 session 为重采样单位；
- 使用 seed `20260717` 做 `10,000` 次 session bootstrap，报告 percentile 95% CI；
- `unsafe` 冻结定义为单个评审的 `safety_1_5 <= 2`，再按 candidate/variant 聚合；
- ordinal agreement 报告 quadratic-weighted Cohen kappa、exact agreement、within-one agreement 和 MAE；binary flags 报告 Cohen kappa 与 exact agreement；
- 缺失、重复、越界、blind/key/sample 不一致均直接报错，不做插补或仲裁后替换主要分析。

正式实现为 `src/proactive_u1/ratings.py`。它可合并两份独立 CSV；全空模板行被忽略，但最终必须恰好覆盖 `160 candidates x 2 reviewers = 320` 个有效 reviewer rows。

## 7. 冻结判据

### Interface promotion

`forced_no_state` 相对 `current_fallback`：

- paired session-bootstrap `content_composite` 平均提升至少 `+0.50`；
- 95% bootstrap 下界大于 0；
- 至少 3/4 domains 平均提升；
- hallucination 或 unsafe rate 任一不得增加超过 2 个百分点。

通过则优先修复 gate-to-language interface，planner 不作为第一瓶颈。

### State promotion

`forced_oracle_step` 或 `forced_oracle_full` 相对 `forced_no_state` 使用同一判据。通过才说明 plan/state 信息值得进入更大的 P1 复验。

若 full 相对 step 的 `content_composite` 提升小于 `+0.25` 或 bootstrap 跨 0，则优先使用 compact step，不扩大 full-state 字段。

### Language-capacity conclusion

若 forced-no-state 和两个 oracle 变体都不能相对 fallback 稳定提高内容，则 U1 支持优先做 fit-fold-only utterance SFT/LoRA，而不是先训练 predicted state updater。

## 8. 禁止事项

- 不得根据 smoke 生成质量更换样本、seed、prompt、token budget 或 state 字段。
- 不得用 gold utterance 编写 oracle state。
- 不得把非 fallback 或长文本直接称为正确。
- 不得把 16-chunk smoke 当作科学结果。
- 不得用内容指标替换官方 Macro F1。

## 9. Smoke 证据等级

`oracle_states.smoke.json` 的标注者在标注前已经查看过 smoke 生成输出，因此该文件不是 formally blind annotation。它只用于 schema、runner 和因果 timestamp 的工程诊断，不得进入 state-effect promotion/rejection 统计。正式比较只使用 `formal_blind/` 下由隔离上下文标注并经 validator 合并的 20-session 文件。
