# U2 固定 D4 Gate 的 Early-Chunk 视觉事实先行诊断报告

> 日期：2026-07-20  
> 实验 ID：`20260720_internvl35_1b_d4_early_grounding_u2_v1`  
> 状态：自动机制诊断与盲评包完成；人工质量复核待完成  
> 性质：review-informed public-validation diagnostic，不替代官方 scorer，不改变 D4 决策

## 1. 结论

U2 在固定 D4 interrupt gate 的 21 个 early chunks 上得到三个清晰但边界有限的结论：

1. **assistant history 是当前正文生成的主要脚手架。** 保留完整因果视频但移除 assistant
   history 后，21/21 全部回退到固定句；保留 history 的 `full_history` 只有 8/21 fallback。
2. **当前 interval 视频没有稳定改善 coverage。** 删除当前 interval 后 fallback 反而从
   8/21 降到 7/21，说明当前模型会因画面改变措辞，但尚不能依靠当前视觉稳定启动正文。
3. **predicted-fact 中介只有小幅 cold-start rescue，且未通过冻结门槛。** 在 query/current
   video-only 条件下，事实块把 nonempty 从 3/21 提到 6/21，即救回 3 条、`+14.29pp`，低于
   预注册的 `+20pp`。事实文本还经常写成动作指令而非纯可见观察，正确性尚无人评确认。

因此，U2 自动结果不支持晋级事实先行方案，也不支持修改 D4 gate。当前证据更像是：history
提供语言生成结构，视觉和预测事实只在该结构上调制内容；无 history 的真正 early cold start
仍未解决。

## 2. 样本与因果边界

样本不是人工挑选案例，而是以下冻结条件的完整交集：

1. U0 位置为 second 或 2--4；
2. Reviewer A/B 均判断 `should_interrupt=yes`；
3. D4 五折 OOF 与全公共 refit 均预测 interrupt。

最终为 21 个不同 session：second 9、2--4 12；Arts and Crafts/Chef/Tutorial/Handyman
为 7/5/6/3；原 U0 fallback/nonfallback 为 14/7。人评仅以布尔共识参与取交集，未按分数
排序；gold answer 未参与选择或生成。该集合已使用 public-validation 人评信息，只能用于
development diagnostic，不能称为独立验证集。

生成样本移除了完整 `answers` 字段；各视图不读取未来 dialog、未来帧或未来标签。所有 D4
decision 固定不变，未重新训练 head、调整 threshold 或调用官方 scorer。

## 3. 六视图设计与工程校验

| 视图 | Assistant history | Video | 当前 interval predicted facts |
|---|---|---|---|
| `full_history` | 最近 4 turns | 累计因果帧 | 无 |
| `no_current_video` | 最近 4 turns | 以前 intervals | 无 |
| `query_only_full_video` | 无 | 累计因果帧 | 无 |
| `query_only_current_video` | 无 | 当前 interval | 无 |
| `facts_full_history` | 最近 4 turns | 累计因果帧 | 有 |
| `facts_query_current` | 无 | 当前 interval | 有 |

fact pass 与正文 pass 复用同一 InternVL3.5-1B 模型实例，不增加模型副本或学习参数。每条
事实只接收 query 与当前 interval，greedy 生成最多 32 tokens；同一条事实随后同时用于两个
facts 视图。正文每个视图最多生成 64 tokens。

正式运行完成 21/21 冻结 R0 full-view exact replay、21 次 fact generation 和 126 次正文
generation；D4 decision 改动为 0，official scorer 未调用。物理 GPU 1 在运行前无已有进程，
wall time `208.88s`，峰值显存 `3,480,062,464` bytes。生成模型审计参数量为
`1,060,897,792`，仍在 Small 2B 上限内。

## 4. 自动结果

| View | Fallback | Nonempty | Mean words | Second fallback | 2--4 fallback |
|---|---:|---:|---:|---:|---:|
| `full_history` | 8/21 (38.10%) | 13/21 (61.90%) | 11.48 | 6/9 | 2/12 |
| `no_current_video` | 7/21 (33.33%) | 14/21 (66.67%) | 11.76 | 5/9 | 2/12 |
| `query_only_full_video` | 21/21 (100%) | 0/21 (0%) | 0 | 9/9 | 12/12 |
| `query_only_current_video` | 18/21 (85.71%) | 3/21 (14.29%) | 3.48 | 6/9 | 12/12 |
| `facts_full_history` | 8/21 (38.10%) | 13/21 (61.90%) | 9.81 | 6/9 | 2/12 |
| `facts_query_current` | 15/21 (71.43%) | 6/21 (28.57%) | 8.52 | 4/9 | 11/12 |

关键 paired contrasts：

- `no_current_video` 相对 `full_history` 多 1 条 nonempty，文本 exact 9/21，平均相似度
  `0.7339`。删除当前画面没有降低 coverage，不能把完整视图的输出归因于当前视觉。
- `query_only_full_video` 相对 `full_history` 丢失全部 13 条原 nonfallback，fallback
  `+61.90pp`。这一结果在本样本上直接复现并强化了 U1-V 的 history dependency。
- `facts_query_current` 相对 `query_only_current_video` 救回 3 条，nonempty `+14.29pp`；
  预注册 `+20pp` 条件失败。增益主要出现在 second（5/9 nonempty），2--4 仅 1/12。
- `facts_full_history` 与 `full_history` 的 fallback 完全相同；21 条中有 8 条正文改变，
  平均相似度 `0.8206`。事实块会调制措辞，但没有提升 coverage。

所有显式 completion-claim 计数均为 0，但这只是字符串规则结果，不等于不存在隐式错误、
不当推断或 hallucination。词法 fact overlap 同样只测表面利用，不能证明视觉支持。

## 5. Predicted Facts 的质量边界

fact pass 自动统计为 21/21 非空且非 `unclear`，平均 `10.95` words，通过了冻结的
non-unclear-rate 优先级条件。但人工抽查显示，不少输出是命令式动作描述，而不是协议要求的
“直接可见事实”；部分输出较长、截断或含无法由当前画面自动确认的细节。因此，这个 100%
只说明模型愿意输出文本，不说明事实正确。

3 个 cold-start rescue 也不能直接视为质量提升。例如，模型会从“折纸”“拉线”或“线缆盒”
类预测事实扩写为多步指令，其中可出现画面未支持的后续动作与截断句。该现象正是盲评必须
同时检查 fact correctness、current visual support 和 unsupported claim 的原因。

## 6. 盲评包

已生成随机盲化且不暴露 variant 的两套包：

- 126 个 utterance candidates，A/B 共 252 个评分槽；
- 21 个 predicted facts，A/B 共 42 个评分槽；
- variant key 独立保存，utterance 包不展示 predicted fact block；
- rubric 新增 `current_visual_support_1_5`、`unsupported_visual_claim_flag` 和
  `stale_history_flag`，fact 包独立评 correctness/completeness。

U0 groundedness quadratic kappa 只有 `0.0508`。因此正式双评前必须先用少量独立 calibration
examples 对齐“当前 interval 直接支持”的边界；原始 A/B 分数必须保留，不能用仲裁结果覆盖。
在人评完成前，不比较或晋级任何 U2 内容方案。

## 7. 复现性

独立第二次 GPU 运行完成相同 21 replay、21 facts 和 126 candidates。两次运行的
`analysis.json` 字节完全一致；去除自然变化的 `wall_time_seconds` 后，三份逐条记录语义完全
一致；全部 blind items、keys、templates 和 rubric 字节完全一致。

| Artifact | SHA256 |
|---|---|
| Protocol | `679912125d2c76ec35fc96619dc35ecf96d94458f563410baa12927c707c2d46` |
| Sample items | `4e079328902ebaf1fe44ac32711d9430a69540395c860e1d2bae551cc908fc6f` |
| Sample key | `62c0418de161e46a33a137f2ee96e3a4125d378fbbcd34bf69a7006d69c4e4d4` |
| `analysis.json` | `fa1d7c6be5ca93e025e8a3e4e79fb2a626a7d3745673ec21c1709f9175862bc6` |
| `content_records.jsonl` | `f894a6f33643207fdf1c5ca99ddc8c32647a75c14b150ae5486dd388e17be9f0` |
| `fact_records.jsonl` | `58ccc0555da9f988b51c4c0a8f8ba0a92e50bbd15ac94527b7f123f2d2d9df04` |
| `r0_replay_records.jsonl` | `7f6e42f420b3b7b7f41baef7f49a7c759028a36a5696a27eb4653147a3ace837` |
| Review manifest | `b3ae86995bdadb38e62c07981ab357e45844494f6464092fcd0c2a139dfed8f7` |

代码入口为 `src/proactive_u2/prepare.py` 与 `src/proactive_u2/run.py`；配置和完整实验工件分别
位于 `configs/u2_internvl35_1b_d4_early_grounding_v1.json` 与
`output/experiments/20260720_internvl35_1b_d4_early_grounding_u2_v1/`。

## 8. 路线决定与下一步

1. U2 v1 保留为失败门槛下的机制诊断；不得在同一 21 条上事后修改 fact prompt、帧数、
   history window 或 decoding，再把结果当作确认性证据。
2. 先完成视觉支持 rubric calibration，再由两名独立评测员评价 126 个 utterances 和 21 个
   facts；聚合时同时报告均值、原始列联和一致性，不静默平均 groundedness。
3. 只有当人评显示 fact 质量可靠且正文的 visual support 提升、unsupported/stale 风险不升，
   才能在新的独立冻结样本上设计 v2。当前自动结果本身不满足该条件。
4. D3 `0.6690` 继续作为科学基线，D4 继续作为冻结 leaderboard-engineering candidate；
   U2 不改变任何 official Macro 或提交配置。
5. S1、granularity 和 GRPO 继续冻结。并行等待 U1-B/state-package 评分、teammate input-policy
   工件和官方 Docker 模板；任何外部提交仍需用户明确授权。
