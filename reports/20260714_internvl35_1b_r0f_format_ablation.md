# C1 Small R0-F 格式消融实验报告

> 实验 ID：`20260714_internvl35_1b_response_intent_repair_r0f_valsupervised`  
> 状态：完整公开验证集格式消融已完成  
> 结果分类：**使用验证集监督；不属于留出集泛化证据**  
> 实验产物：[`output/experiments/20260714_internvl35_1b_response_intent_repair_r0f_valsupervised/`](../output/experiments/20260714_internvl35_1b_response_intent_repair_r0f_valsupervised/)  
> 预测文件 SHA256：`cfda7d147ac3203ff5750a5b65fbac54af5f2bcf4aef4d4fa16db700b25c0e37`

## 1. 结论

R0-F 修复了 R0 中一个明确的协议故障：当冻结模型已经输出非空的自然语言指导，但遗漏合法标签时，将该响应视为一次 interrupt 发言，并在前面补上 `$interrupt$`。

在完整的 700 个公开验证 session、9,935 个 chunk 上，官方评分器给出以下结果：

| 指标 | 冻结 R0 | R0-F | 变化 |
|---|---:|---:|---:|
| Macro F1 | 0.4630 | **0.5362** | **+0.0732** |
| G-mean F1 | 0.4541 | **0.5340** | +0.0799 |
| Interrupt precision | 0.5286 | **0.6119** | +0.0833 |
| Interrupt recall | 0.2879 | **0.4056** | +0.1177 |
| Interrupt F1 | 0.3728 | **0.4879** | +0.1151 |
| Silent precision | 0.4571 | **0.5020** | +0.0449 |
| Silent recall | **0.7002** | 0.6995 | -0.0007 |
| Silent F1 | 0.5531 | **0.5845** | +0.0314 |

R0-F 是目前本地完整 700-session 评测中的最高分，但它不是干净的留出集证据。该规则族是在检查公开验证集错误后选定的，并且 633 个被修复的 chunk 中有 630 个恰好是金标 interrupt。因此必须将该结果报告为 `val-supervised`，并通过隐藏测试集确认。

## 2. 精确规则

确定性的修复函数为：

```text
如果 raw 以 $interrupt$ 开头：
    保留 interrupt，并规范化其 utterance
否则，如果 raw 以 $silent$ 开头：
    精确输出 $silent$
否则，如果 raw 为空或只包含空白字符：
    输出 $silent$
否则：
    输出 $interrupt$ + raw
```

该函数不查看标签、阈值、任务、domain、chunk 位置、R0 置信度或未来信息。然而，在分析公开集后才选择这条规则，本身就属于在方法选择层面使用了验证集反馈。

实验没有重新运行模型推理。输入是冻结 R0 的 `session_records.jsonl`，其中包含每条原始响应及其原始 canonical answer。模型权重、prompt、视频帧、对话和 decoding 设置均未改变。

## 3. 为什么放弃 Grammar 方案

在采用 response-intent repair 之前，我们在首个含 14 个 chunk 的 Chef 试验 session 上测试了两种不读取标签的 tag 约束。

### 3.1 V1：贪心 token 前缀有限状态机（FSM）

decoder 只允许能够以 `$silent$` 或 `$interrupt$` 开头的 token 序列。它消除了格式错误，但对每个 context 变体都在 14 个 chunk 中的 13 个选择 interrupt，导致 Silent F1 坍缩为零。

### 3.2 V2：完整标签序列 beam

decoder 使用宽度为 2 的 beam，对两个等长完整标签的累积对数概率进行比较，并且只在选择 interrupt 后继续生成文本。完全相同的 R0 和 null 仍在 14 个 chunk 中的 13 个选择 interrupt；step、cues 和 full 则在 14 个中全部选择 interrupt。Silent F1 再次坍缩为零。

因此，模型对原始标签的 likelihood 不是一个经过校准的 gate。两种 grammar 都没有扩展到完整的 50-chunk 试验或 700-session 集合。对应的 smoke test 产物保留在：

```text
output/experiments/20260714_internvl35_1b_tag_grammar_r1f_pilot_v1_smoke1/
output/experiments/20260714_internvl35_1b_tag_sequence_beam_r1f_pilot_v2_smoke1/
```

这个失败提供了重要信息：可以在保证语法有效的同时彻底破坏决策类别平衡。格式机制必须保留响应意图，而不能用未经校准的 token likelihood 取代原有 gate。

## 4. 被修复样本的总体统计

| 项目 | 数量 |
|---|---:|
| Session 总数 | 700 |
| Chunk 总数 | 9,935 |
| 二元决策发生变化 | 633 |
| 金标为 interrupt 的修复 chunk | 630 |
| 金标为 silent 的修复 chunk | 3 |
| 位于 chunk 位置 0 的修复 | 628 |
| 位于位置 1 / 2 / 3 的修复 | 3 / 1 / 1 |
| 原始响应长度，中位数 / 均值 / 最大值 | 81 / 105.739 / 326 字符 |

混淆矩阵的精确变化如下：

```text
R0:   TP 1541  FP 1374  TN 3209  FN 3811
R0-F: TP 2171  FP 1377  TN 3206  FN 3181
变化:    +630      +3      -3      -630
```

预测 interrupt 比例从 29.34% 上升至 35.71%，仍低于金标 interrupt 的 53.87%。与 grammar smoke test 不同，response-intent repair 没有导致全部预测为 interrupt 的类别坍缩。

## 5. 首 Chunk 效应

公开集中有 699 个金标为 interrupt 的首 chunk，以及 1 个金标为 silent 的首 chunk。

| 首 chunk 指标 | R0 | R0-F |
|---|---:|---:|
| TP / FP / TN / FN | 50 / 0 / 1 / 649 | 677 / 1 / 0 / 22 |
| Interrupt recall | 0.0715 | **0.9685** |
| 预测 interrupt 比例 | 0.0714 | 0.9686 |
| Macro F1 | 0.0683 | 0.4916 |

R0-F 修复了 628 个首 chunk 决策。这与任务公开标注中“助手几乎总会在开头提供指导”的惯例一致，但同时也是主要的隐藏测试集风险：收益强烈依赖该惯例在隐藏集上保持稳定。

首 chunk 的 Macro F1 仍低于 0.5，因为 R0-F 将唯一一个金标为 silent 的首 chunk 预测为 interrupt，使该条件切片没有任何首 chunk true negative，因而 Silent F1 为零。

## 6. 分 Domain 结果

| Domain | R0 Macro | R0-F Macro | 变化 | R0 int. recall | R0-F int. recall |
|---|---:|---:|---:|---:|---:|
| Arts and Crafts | 0.4667 | **0.5389** | +0.0722 | 0.3774 | 0.5057 |
| Chef | 0.4863 | **0.5485** | +0.0622 | 0.3940 | 0.5082 |
| Handyman | 0.4247 | **0.5100** | +0.0853 | 0.1443 | 0.2535 |
| Tutorial | 0.4317 | **0.5144** | +0.0827 | 0.2054 | 0.3241 |

四个 domain 在公开集上均有提升。Handyman 和 Tutorial 的 interrupt recall 绝对值仍然较低，因此格式修复并没有消除 R0 已发现的底层视觉/状态判断弱点。

## 7. Session 层面的稳定性

使用诊断性质的逐 session Macro F1：

| 结果 | Session 数 |
|---|---:|
| 提升 | 628 |
| 不变 | 72 |
| 下降 | 0 |
| Session 平均变化 | +0.0881 |
| Session 变化中位数 | +0.0804 |

广泛的 session 层面一致性是积极信号，但并非独立重复实验：几乎每个得到提升的 session 都包含相同的首 chunk 格式错误模式，而该规则正是在观察到这一模式后才被选中。

## 8. 提交与参赛资格

R0-F 仍然符合 Small 赛道限制：

- 使用相同的 1,060,897,792 参数 Apache-2.0 模型；
- 新增可学习参数为零；
- 未使用外部数据或模型；
- 只包含确定性的源码后处理；
- 预测文件严格包含按源数据顺序排列的 700 行和 9,935 个合法答案。

预测文件已经可以作为公开验证榜候选：

[`predictions.jsonl`](../output/experiments/20260714_internvl35_1b_response_intent_repair_r0f_valsupervised/predictions.jsonl)

尚未向排行榜上传任何内容。外部提交需要用户明确授权，并会消耗每日提交次数。

在测试阶段的 container 中，同一修复必须在线应用到原始生成结果。仅提交这份验证集预测文件本身，并不能提供测试阶段的推理代码。

## 9. 科学解释

已经确立的事实：

1. R0 的相当一部分损失来自响应协议不匹配，而不是模型没有生成自然语言指导；
2. 在 smoke case 上，response-intent repair 的校准明显优于 token-level 或 sequence-level grammar；
3. 该修复在公开集上同时提高两个类别的 F1，并且几乎不改变 silent recall；
4. R1 状态扩展应继续暂停，因为修复后的 R0 在四个试验 session 上仍超过 posthoc 修复后的 full-state（`0.5994` 对 `0.5895`）。

尚未确立的结论：

1. 隐藏测试集也能提升 +0.0732；
2. 所有非空且格式错误的输出都必然代表一次有用干预；
3. 被修复 utterance 的语义质量；
4. 已经解决 session 中段的 false negative、Handyman/Tutorial 状态跟踪或粒度问题；
5. 存在启动 GRPO 或 planner 的依据。

## 10. 复现命令

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_r0f.run \
  --config configs/r0f_internvl35_1b_response_intent_repair.json \
  --output-dir output/experiments/20260714_internvl35_1b_response_intent_repair_r0f_valsupervised
```

重建诊断结果：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_r0f.analyze \
  --experiment-dir output/experiments/20260714_internvl35_1b_response_intent_repair_r0f_valsupervised
```

第二次独立运行官方评分器，得到的 metrics 文件与正式实验产物逐字节完全一致。

## 11. 实验产物指纹

| 对象 | SHA256 |
|---|---|
| 配置 | `61edd2da24c38f9bc749dac4b685cf1c4a7e4f15cae94450864d08f7d143516b` |
| 预测文件 | `cfda7d147ac3203ff5750a5b65fbac54af5f2bcf4aef4d4fa16db700b25c0e37` |
| 官方指标 | `7e169d602e03501eb644d35c90312a4f4f107cab3e1827cf859f87a07f72687e` |
| 对比结果 | `fdb7965d2e9eeeed2cf4a6193944484a808c6813574f0400cb00d0ba7cfa91ee` |
| 分析结果 | `7f1d9b86348dfc1c0bc6f1420100708b373ea86a47069df29f293ce4c9d6699f` |
