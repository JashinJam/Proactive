# C1 Small R1 Oracle 紧凑状态协议试验报告

> 实验 ID：`20260714_internvl35_1b_oracle_state_r1_pilot_v1`  
> 状态：协议试验已完成；**R1 科学门槛未通过**  
> 完成时间：2026-07-14 15:19 CST  
> 实验产物：[`output/experiments/20260714_internvl35_1b_oracle_state_r1_pilot_v1/`](../output/experiments/20260714_internvl35_1b_oracle_state_r1_pilot_v1/)  
> 标注 SHA256：`4e3ea373f0b4e1f4698f254ac0146d3dad32c60e6834f5931566864f373b9435`

## 1. 结论

这项由四个 session 组成的试验证明，R1 的标注、因果校验、多变体推理、断点续跑、产物保存和官方评分管线均可正常工作，但**没有**证明 Oracle 紧凑状态能够提升排行榜指标。

相对于 null 状态包装器对照，full 状态将 interrupt recall 从 `0.3667` 提高到 `0.5000`，同时使 silent recall 从 `0.7500` 降至 `0.5500`。因此两个变体的官方子集 Macro F1 完全相同，均为 `0.5169`。冻结的 R0 预测仍然更高，为 `0.5398`。

观察到的效果是类别间的取舍，而不是状态建模带来的净收益：

- 相对于 null，full 状态改变了 50 个决策中的 14 个；
- 净效应是额外将 8 个 silent 决策改为 interrupt；
- 混淆矩阵变化为 `+4 TP`、`+4 FP`、`-4 TN`、`-4 FN`；
- 预测 interrupt 比例从 `32%` 升至 `48%`，而金标 interrupt 占比为 `60%`；
- 仅使用静态 cues 会造成明显伤害：Macro F1 为 `0.3592`，interrupt recall 为 `0.1667`。

下一项有效实验应是在完全相同的冻结输入上，单独开展受 grammar 约束的格式析因实验。现阶段不要扩大 Oracle 标注规模。

## 2. 研究问题与边界

R1 要回答的问题是：紧凑的程序状态能否找回漏掉的干预，同时不破坏 silent 精度。这项试验被刻意控制在很小的规模，只回答协议是否可运行以及是否存在明显信号。

它不是：

- 排行榜结果或隐藏测试集结果；
- 对 700 个公开 session 的总体估计；
- 可部署的 planner 或状态更新器；
- 官方 PWR 复现；
- 启动训练、GRPO 或专用粒度模块的证据。

Oracle 状态仅用于评测，无法部署。实验没有训练或更新任何参数。

## 3. 冻结的试验样本选择

试验从公开数据的每个 domain 中各选一个 session。选择发生在标注之前，使用 `domain_stratified_sha256_first_k` 规则和种子 `r1-oracle-pilot-v1`。该规则只读取 domain 和视频路径，不查看答案、R0 预测或错误类型。

选中样本保持源 JSONL 中的顺序：

| 输入索引 | Domain | 任务 | Chunk 数 |
|---:|---|---|---:|
| 14 | Chef | Making peanut butter oat balls | 14 |
| 123 | Tutorial | How to loosen a tight screw | 10 |
| 326 | Handyman | Adjusting loose recessed light trim | 12 |
| 687 | Arts and Crafts | Creating a simple finger painting | 14 |
| **合计** | 4 个 domain | 4 个任务 | **50** |

只有在所有生成预测均已持久化后才读取金标。所选集合包含 30 个 interrupt 和 20 个 silent 标签，因此其 interrupt 占比为 60%，与完整公开集的 53.87% 不同。

## 4. 因果标注协议

完整协议冻结在 [`PROTOCOL.md`](../annotations/r1_oracle_pilot_v1/PROTOCOL.md) 中。静态计划只能使用 `task` 和 `query`。在 chunk `i`，动态状态只能使用：

- task 和 query；
- 推理时已经可见的官方 `dialog[i]`；
- 时间不晚于 `video_intervals[i][1]` 的视频帧。

标注时排除 `answers`、未来对话、未来视频、R0 预测和错误类别。50 个 chunk 标注包均按时间顺序检查。每个标注包包含从当前区间内采样的四帧，以及一条不含标签的 `dialog_at_chunk` 记录。

校验器强制检查：

- session 和 chunk 覆盖必须精确完整；
- query 和视频标识必须一致；
- 状态时间戳必须与当前区间结束时间完全一致；
- step ID 和 progress 枚举值必须合法；
- `last_update_chunk <= chunk_index`；
- 不允许未声明的状态突变；
- 状态文本中不能出现目标标签、金标措辞或明确的 speak/silent 决策。

紧凑状态包含以下字段：

```text
goal
current_step
progress: not_started | ongoing | complete | deviated | recovered
completion_cues
incompletion_cues
completion_evidence
incompletion_or_error_evidence
next_step
confidence
last_update_chunk
```

## 5. 受控变体

所有需要生成的变体均使用冻结的 R0 InternVL3.5-1B checkpoint，并保持以下配置不变：BF16/SDPA 加载、每个区间 16 帧、累计上限 32 帧、448 分辨率、最近四轮对话、greedy decoding、64 个新 token、pad ID 151643，以及原有的 response canonicalizer。

| 变体 | 额外输入 |
|---|---|
| `r0_frozen` | 不重新生成；直接从冻结的 R0 预测中提取完全相同的行 |
| `null` | 只包含 `status: unavailable` 的状态包装器 |
| `step` | 当前步骤文本 |
| `cues` | 当前步骤，加上静态的完成/未完成 cues |
| `full` | cues，加上目标、进度、已观察证据、下一步、置信度和更新索引 |

原始官方 system prompt 作为逐字节完全一致的前缀保留，状态块追加到其 system context 中。`null` 用于衡量 prompt 包装器本身的混杂影响；状态内容应主要与 `null` 比较，`r0_frozen` 则保留为真实的 no-plan 参照。

## 6. 模型与运行配置

| 项目 | 冻结值 |
|---|---|
| 模型 | `OpenGVLab/InternVL3_5-1B-HF` |
| Revision | `9191dbccf312b537016f041b25d61c72e7c5c9f3` |
| 参数量 | 总计 1,060,897,792，为 Small 上限的 53.04% |
| 权重 SHA256 | `11effd1da2fc0929957d56de4129d1e7d2aed044ea878d40bf83e95b63d38b39` |
| 设备 | A800 80 GB，GPU 1，启动前无既有进程 |
| 墙钟时间 | 274.838 秒 |
| 峰值已分配显存 | 3,490,579,456 bytes |
| 生成调用数 | 50 个 chunk x 4 个变体 = 200 次 |

## 7. 官方评测结果

下表中的所有实际结果均由 `data/starter_kit/run_evaluation.py` 产生。它只是子集诊断，不能当作可与 700-session 排行榜分数比较的结果。

| 变体 | Macro F1 | Int. P | Int. R | Int. F1 | Silent P | Silent R | Silent F1 | TP/FP/TN/FN | 预测 int. 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `r0_frozen` | **0.5398** | 0.6842 | 0.4333 | 0.5306 | 0.4516 | 0.7000 | 0.5490 | 13/6/14/17 | 38% |
| `null` | 0.5169 | **0.6875** | 0.3667 | 0.4783 | 0.4412 | **0.7500** | **0.5556** | 11/5/15/19 | 32% |
| `step` | 0.4949 | 0.6667 | 0.3333 | 0.4444 | 0.4286 | **0.7500** | 0.5455 | 10/5/15/20 | 30% |
| `cues` | 0.3592 | 0.4545 | 0.1667 | 0.2439 | 0.3590 | 0.7000 | 0.4746 | 5/6/14/25 | 22% |
| `full` | 0.5169 | 0.6250 | **0.5000** | **0.5556** | 0.4231 | 0.5500 | 0.4783 | 15/9/11/15 | 48% |

full 相对 null 的变化：

```text
Macro F1 变化          0.0000
interrupt recall 变化 +0.1333
silent recall 变化    -0.2000
预测 speak 比例变化   +0.1600
混淆矩阵变化          +4 TP, +4 FP, -4 TN, -4 FN
```

## 8. 分 Session 结果

| Domain | R0 | Null | Step | Cues | Full |
|---|---:|---:|---:|---:|---:|
| Chef | **0.4759** | 0.2632 | 0.3778 | 0.2632 | 0.4286 |
| Tutorial | 0.4000 | **0.4949** | 0.3750 | 0.1667 | 0.4000 |
| Handyman | **0.4958** | **0.4958** | **0.4958** | 0.3333 | 0.4375 |
| Arts and Crafts | 0.5238 | 0.5238 | 0.4750 | 0.3778 | **0.5758** |

相对于冻结 R0，full 状态只在 Arts and Crafts session 上有提升。由于每个 domain 只有一个 session，这只是个案观察，不能作为 domain 层面的结论。

## 9. 决策变化分析

相对于 null，full 改变了 14 个决策：

| 变化 | 金标 | Progress | 数量 |
|---|---|---|---:|
| silent -> interrupt | interrupt | ongoing | 5 |
| silent -> interrupt | silent | ongoing | 3 |
| interrupt -> silent | interrupt | not_started | 2 |
| silent -> interrupt | interrupt | deviated | 1 |
| silent -> interrupt | interrupt | complete | 1 |
| silent -> interrupt | silent | complete | 1 |
| interrupt -> silent | interrupt | ongoing | 1 |

因此，full 状态并不是在简单地检测完成或偏离。大多数变化发生在 `ongoing` 阶段，其中增加了 5 个 TP 和 3 个 FP。该表示提高了模型开口干预的倾向，但还不能校准哪些 ongoing 证据真正值得干预。

仅使用静态 cues 会使模型更保守，将预测 interrupt 比例从 32% 降至 22%。一种可能的解释是，1B 零样本模型无法可靠地区分假设性的 cue 描述与已经观察到的事实。这是根据行为作出的推断，不是对模型内部机制的证明。

## 10. 格式混杂

选中的四个 session 的首个 chunk 金标均为 interrupt。首 chunk 结果如下：

| 变体 | TP/FN | Interrupt recall | 格式错误的首 chunk 数 |
|---|---:|---:|---:|
| `r0_frozen` | 1/3 | 0.25 | 3 |
| `null` | 2/2 | 0.50 | 2 |
| `step` | 1/3 | 0.25 | 3 |
| `cues` | 0/4 | 0.00 | 4 |
| `full` | 0/4 | 0.00 | 4 |

四条 full 状态的原始响应均为自然语言指令或错误标签尝试，例如 `$Iinterrupt$ ...`；官方 fallback 会正确地将其按 silent 评分。这仍然是模型的真实失败，不能通过重新标注官方结果来规避。

仅作为 posthoc 诊断，如果把所有格式错误的原始响应重新分类为 interrupt，full 的 Macro F1 将为 `0.5895`，recall 为 `0.6333`。同一 posthoc 规则会使冻结 R0 达到 `0.5994`，因此即使在这个反事实下也不能证明 full 状态占优。这两个数值都不是正式 R1 结果，也不能作为排行榜声明。

## 11. 决策

当前 R1 门槛**未通过**：

1. 没有任何状态变体在 Macro F1 上超过冻结 R0；
2. full 状态在 Macro F1 上没有超过 null；
3. recall 增益被误报代价完全抵消；
4. cues-only 在四个 session 上都造成伤害；
5. full 和 cues 引入了严重的首 chunk 格式混杂；
6. 四个 session 不足以证明结果可重复。

目前不要开展原计划的 16-session 扩展标注。应先在完全相同的 50 个 chunk 上运行预注册、格式受控的析因实验：

- 完全相同的 R0 context + tag grammar；
- null 包装器 + 相同的 tag grammar；
- step/cues/full + 相同的 tag grammar。

grammar 只能约束必需的输出前缀，不能查看标签。如果受约束的 full 仍无法同时超过受约束的 R0 和受约束的 null，则停止扩大这种零样本 Oracle 表示，重新审视状态序列化方式或 state-aware training。只有当它在不发生类别坍缩的前提下产生可重复且非微小的 Macro 增益时，才扩展到预注册的 16-session 集合。

## 12. 复现命令

仅审计：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_r1.run \
  --config configs/r1_internvl35_1b_oracle_state_pilot_v1.json \
  --output-dir output/experiments/20260714_internvl35_1b_oracle_state_r1_pilot_v1_audit \
  --audit-only
```

正式试验：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_r1.run \
  --config configs/r1_internvl35_1b_oracle_state_pilot_v1.json \
  --output-dir output/experiments/20260714_internvl35_1b_oracle_state_r1_pilot_v1 \
  --device cuda:0
```

重建诊断结果：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_r1.analyze \
  --experiment-dir output/experiments/20260714_internvl35_1b_oracle_state_r1_pilot_v1
```

## 13. 实验产物指纹

| 对象 | SHA256 |
|---|---|
| 配置 | `ef4d240938be74a043a680741e6236e5caf98f2b35859b108fccb57fe2879050` |
| 样本选择清单 | `c90fc03cfacc0ba49147eebe892f8e674c3a92f97710324f4046ad325d3c5c1d` |
| Oracle 标注 | `4e3ea373f0b4e1f4698f254ac0146d3dad32c60e6834f5931566864f373b9435` |
| 标注协议 | `e7908e33a1c29d27ab762df8b3f79520eab634f4e4fbefae5ff2dc7a643a085a` |
| 对比 JSON | `0bf1b5b33d050fbc8964edd12f809bd735aef44761df3590d5aadcb26c75931c` |
| 分析 JSON | `9f843365c8a33605e83a5f694fd7bac3bef4e487fd58c0b97d051b240a90f4e7` |

各变体实际预测文件的指纹记录在 [`comparison.json`](../output/experiments/20260714_internvl35_1b_oracle_state_r1_pilot_v1/comparison.json) 中。

