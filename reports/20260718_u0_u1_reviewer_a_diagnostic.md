# U0/U1 评测员 A 单人结果诊断与路线调整

## 1. 结论

评测员 A 的结果足以支持一次路线诊断，但不能替代冻结的双评审 promotion。
最明确的结论有四个：

1. 固定句 `Please continue with the next step.` 是严重的语言接口缺陷；强制正文生成明显改善具体性和可操作性。
2. 改善几乎全部发生在中后段。Second chunk 仍大量直接 EOS，且平均不如固定 fallback。
3. 模型很可能把最近 assistant 历史当作程序状态使用；当前视觉是否真正改变输出尚未由因果对照证明。
4. 人最希望早期给出启动指导，而生成器恰好在早期最弱；gate 与 utterance 必须分开设计。

因此，暂停剩余 S1 大规模标注，先做 U1-V 视觉依赖审计和 D3 dialog-only 控制。
已有 S1 资产全部保留，满足恢复条件后再继续。

## 2. 数据边界

- U0：A 完成 200 条，五个困难 strata 各 40 条；没有 TN stratum，不能用于估计全量 prevalence。
- U1：A 完成 160 条，恰好是 80 个 sample 的 `current_fallback/forced_no_state` interface package。
- U1 state package 的 `no-state/oracle-step/oracle-full` 共 240 条不在当前 CSV 中，尚不能判断 state promotion。
- B 评分未读取。机器结果明确记录 `reviewer_b_read=false`。

评分源 SHA256：

```text
U0 A: 4e1476f6576150d1e135b9ad8ef047029c04829d78a234114e93592d6d0c9feb
U1 A: e7e0b1784abccd62e6e4e34190d83e1b2ec78403a8fcd2c4f910b998bdc2ef5d
```

## 3. U1 Interface 结果

| 指标 | 固定 fallback | Forced no-state | 差值 |
|---|---:|---:|---:|
| Content composite | 1.7675 | 2.9400 | +1.1725 |
| Correctness | 1.9500 | 2.9625 | +1.0125 |
| Specificity | 1.0000 | 2.9125 | +1.9125 |
| Actionability | 1.0000 | 2.6625 | +1.6625 |
| Groundedness | 2.9125 | 3.3000 | +0.3875 |
| Plan consistency | 1.9750 | 2.8625 | +0.8875 |
| Generic rate | 98.75% | 28.75% | -70.00pp |
| Hallucination rate | 0% | 2.50% | +2.50pp |
| Unsafe rate（safety <= 2） | 1.25% | 0% | -1.25pp |

按 20 个 session 做 10,000 次 bootstrap，content composite 差值为：

```text
estimate = +1.1725
CI95     = [+0.8875, +1.4500]
```

四个 domain 均为正，但 hallucination 增加 `2.50pp`，超过冻结上限 `2pp`。
因此 A-only 临时 gate 只失败这一项；它是强接口收益和边界风险，而不是正式 promotion。

## 4. 位置与历史效应

| 位置 | Forced composite | Forced fallback rate | 相对固定句 |
|---|---:|---:|---:|
| second | 1.43 | 80% | -0.35 |
| 2--4 | 2.81 | 35% | +1.05 |
| 5--9 | 3.82 | 0% | +2.06 |
| 10+ | 3.70 | 5% | +1.93 |

Second-chunk 差值的 session bootstrap 区间跨零；其他三个位置均明显为正。
实际模型最多读取最近四条 assistant turn：

| 有效 assistant turns | 样本 | fallback rate | composite |
|---:|---:|---:|---:|
| 1 | 27 | 81.48% | 1.3630 |
| 2 | 10 | 0% | 3.8200 |
| 3 | 12 | 8.33% | 3.9500 |
| 4 | 31 | 3.23% | 3.6387 |

这说明 EOS 失败和对话结构高度相关。位置、历史和真实视频进度共同增长，当前结果不能把三者因果分开。

## 5. 视觉利用风险

支持“历史主导”的诊断信号包括：

- 非 fallback 正文中，平均约 39.3% 的内容词出现在最近四条 assistant 历史中；该词面重合只作诊断，不是因果证据。
- 两个 session 在不同 chunk 上生成完全相同正文。
- 典型错误把上一轮的未来指导当成已经完成的事实。例如历史要求继续调整背带，正文却宣布背带已经穿好。
- Forced 生成对 specificity/actionability 的改善很大，对 groundedness 的改善只有 `+0.3875`。

模型确实收到累计真实视频帧，最多 32 帧；必须通过移除当前视觉、移除历史和遮蔽视觉的固定对照，才能判断视觉贡献。

## 6. U0 诊断

U0 是刻意平衡的困难样本，不代表全量验证集分布：

- 官方 FP 80 条中，A 仍认为 22 条应该 interrupt。
- 官方 TP 80 条中，A 认为 18 条不应 interrupt，另有 6 条 uncertain。
- 官方 FN 40 条中，A 认为 17 条应 interrupt、18 条不应 interrupt、5 条 uncertain。
- 模型说话的 160 条中，A 判断 78 条该说、72 条不该说、10 条 uncertain。

这说明官方 interrupt 标签与人的时机判断存在明显语义差异。排行榜优化仍应使用官方 scorer；A 评分用于诊断交互质量，不能替换官方标签。

原始文本质量同样不足：固定 fallback composite `1.30`、generic `98.75%`；非 fallback composite `2.725`，但 hallucination `18.75%`，只有 22/80 被标为无主要错误。

## 7. 路线决策

1. D3 `0.6690` 继续作为排行榜主基线，utterance 结果不改变其官方二元指标地位。
2. S1 在 2/32 sessions、23/444 states 处暂停，不删除协议、标注、contact sheets 或代码。
3. 先做 U1-V：`full/no-assistant-history/no-current-interval-video/masked-video`，直接测视觉与历史贡献。
4. 同时做 D3 dialog-only CPU 控制，量化官方 Macro 增益中可由对话阶段解释的比例。
5. 只有 state package 通过、视觉错误指向 step/progress，或 D3 residual 聚集在步骤转换时，才恢复 S1。

## 8. 复现信息

机器结果：

```text
output/experiments/20260718_u0_u1_reviewer_a_diagnostic_v1/analysis.json
SHA256 2a6445fac5cfe7b14cfebe0fc69d231d78c880daa7963db5c0145b52897c3f92
```

分析代码：

```text
src/proactive_u1/analyze_reviewer_a.py
SHA256 9ac58ca3b93ecfea57f058203e669f9f01c95bbf19e315f554c7d9e0fa953fad
```
