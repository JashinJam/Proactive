# D1 Utterance U0 全量审计报告

> 日期：2026-07-16  
> 实验 ID：`20260716_internvl35_1b_d1_utterance_u0_v1`  
> 状态：自动审计和双人盲评包已完成；人工评分尚未进行  
> 性质：public-validation 上的只读诊断，不是隐藏测试证据，不产生新排行榜候选

## 1. 目的与边界

U0 回答一个独立于官方决策分数的问题：D1 `fused_linear` 在提高 `$interrupt$ / $silent$` Macro F1 后，实际输出了什么内容，以及当前输出链路中固定 fallback 的规模和分布如何。

本实验：

- 读取冻结的 700-session gold、R0 `session_records`、D1 OOF predictions 和 D1 OOF records；
- 对齐并审计全部 9,935 chunks；
- 不加载模型、不使用 GPU、不训练参数、不改变 D1 决策；
- 不把内容质量并入官方 Macro F1；
- 生成 200 条双人盲评清单，但本报告不伪造尚未完成的人工评分。

冻结 D1 官方结果仍为 Macro F1 `0.6341`，TP/FP/TN/FN 为 `3165/1448/3135/2187`。U0 重新推导的 confusion counts 与官方 scorer 逐项一致，但官方 scorer 结果仍是决策指标的唯一来源。

## 2. 对齐与复现检查

执行器逐 chunk 验证：

1. gold、D1 prediction、R0 record 的 session 顺序和 `video_path` 完全一致；
2. `answers`、`video_intervals`、`dialog`、R0 chunks 长度完全一致；
3. D1 OOF `(input_index, chunk_index)` 覆盖 9,935 chunks 且无重复；
4. OOF gold/predicted decision 与 gold/D1 prediction tag 一致；
5. 每个 D1 answer 都可由冻结的 `decision_answer(raw_response, decision)` 精确重建。

单元测试覆盖源对齐、篡改检测、盲文件防泄漏和确定性采样。全量命令连续运行两次，`artifact_manifest.json` SHA256 均为：

```text
92ba38ec6f600086464eb4098d5a9242fcfcf0350fc3ed213aecdb153fd07291
```

盲评输入 SHA256 为：

```text
35fef3e15a4fbe73efdfc61a50e9ca550556434316ac7ff0cc15784d9c298337
```

## 3. 全量结果

### 3.1 Fallback 规模

| 项目 | 数量 | 比例 |
|---|---:|---:|
| D1 predicted interrupt | 4,613 | 46.43% / all chunks |
| 固定 fallback | 2,586 | 56.06% / predicted interrupt |
| 非 fallback 内容 | 2,027 | 43.94% / predicted interrupt |
| fallback TP / FP | 1,647 / 939 | binary precision 63.69% |
| non-fallback TP / FP | 1,518 / 509 | binary precision 74.89% |

这里的 precision 只表示该 chunk 的 interrupt 标签是否正确，不表示 utterance 语义正确。不能据此声称非 fallback 内容有 74.89% 的语义正确率。

2,586 个 fallback 的直接来源为：

- 2,565 个 R0 raw response 是显式 `$silent$`；
- 21 个是空正文 `$interrupt$`；
- 没有其他来源。

因此问题不是生成内容偶尔被后处理丢失，而是 D1 将 R0 的 silent 决策翻转为 interrupt 后，没有重新调用内容生成器。

### 3.2 位置分布

| Chunk 位置 | Pred interrupt | Fallback | Fallback / interrupt |
|---|---:|---:|---:|
| first | 699 | 34 | 4.86% |
| second | 426 | 423 | **99.30%** |
| 2--4 | 1,165 | 886 | **76.05%** |
| 5--9 | 1,254 | 675 | 53.83% |
| 10+ | 1,069 | 568 | 53.13% |

第二 chunk 的异常最强：426 次 D1 interrupt 中只有 3 次携带非 fallback 内容。D1 在该位置学会了提高 interrupt 率，但 R0 的原始生成几乎仍全部选择 silent，导致 gate 与语言输出严重脱节。

### 3.3 Domain 分布

| Domain | Pred interrupt | Fallback | Fallback / interrupt | Non-fallback binary precision |
|---|---:|---:|---:|---:|
| Arts and Crafts | 1,206 | 584 | 48.42% | 72.83% |
| Chef | 1,254 | 532 | 42.42% | 72.02% |
| Handyman | 1,007 | 719 | **71.40%** | 84.38% |
| Tutorial | 1,146 | 751 | **65.53%** | 76.46% |

Handyman 和 Tutorial 的内容缺口明显更严重。Handyman 中已有非 fallback 文本的二元 precision 很高，但数量只有 288；这进一步说明优先解决“已决定说话后如何生成内容”可能比继续扩大通用 decision head 更直接。

### 3.4 重复

- 638/700 sessions 至少出现一次 fallback；
- 554/700 sessions 出现至少一次完全相同 utterance 的后续重复；
- 2,090/4,613 predicted interrupts 是 session 内相同文本第二次或之后出现，占 45.31%；
- 排除 fallback 后，仍有 94 sessions 出现 non-fallback 完全重复。

重复统计不等同于语义冗余判断，但已经证明当前用户体验不是少量孤立模板造成的，而是跨绝大多数 session 的系统性现象。

## 4. 自动语言诊断的解释边界

审计器额外输出了 `generic_only_heuristic`、`action_verb_heuristic` 和 `nonstop_content_token_heuristic`。它们只使用固定英文词表：

- 2,586 个 fallback 全部命中保守 generic-only 规则；
- 4,189/4,613 utterances 命中动作动词词表；
- 2,027/4,613 utterances 含非停用、非动作词内容 token。

这些数字不能判断动作、对象或视觉状态是否正确。例如，错误对象也会通过 content-token heuristic。正式内容结论必须来自视频条件下的盲评。

## 5. 冻结盲评样本

盲评清单共 200 条，五个层各 40 条：

```text
TP + fallback
TP + non-fallback
FP + fallback
FP + non-fallback
FN + silent
```

样本在各层内优先减少 domain、chunk-position 和 task 的重复，SHA256 稳定排序。实现结果为：

- 四个 domain 各 50 条；
- 176 个不同 task；
- first/second/2--4/5--9/10+ 分别为 20/30/49/50/51 条；
- sample seed：`20260716-u0-v1`。

`review_items_blind.jsonl` 不含当前 gold decision/utterance、confusion、fallback 标志、R0 raw response、fold 或 tag margin。`review_key.jsonl` 单独保存，A/B 两名评审提交首轮评分前不得查看。FN 样本只评价是否应该打断和时机，内容维度留空。

## 6. 结论

U0 支持以下结论：

1. D1 `0.6341` 是有效的决策基线，但不是完整 proactive-assistant 质量基线。
2. 当前首要可检验问题是 gate-to-language interface break：D1 改变决策后，没有生成与新决策匹配的内容。
3. 现有证据仍不能判断 1B backbone 在被强制说话时能否生成有效建议，也不能判断 oracle plan/state 是否必要。
4. 下一步应保持 D1 decision 100% 固定，执行 U1 forced-generation pilot；不能用新的 language prompt 顺便重调 gate。

U1 的判据保持不变：forced-no-state 有效则优先修复生成接口；只有 oracle state 有效则扩大 plan/state 路线；两者都无效则优先研究 fit-fold-only utterance supervision 或语言容量。

## 7. 产物

- 完整实验目录：[`output/experiments/20260716_internvl35_1b_d1_utterance_u0_v1/`](../output/experiments/20260716_internvl35_1b_d1_utterance_u0_v1/)
- 全量统计：[`audit.json`](../output/experiments/20260716_internvl35_1b_d1_utterance_u0_v1/audit.json)
- 盲评输入：[`review_items_blind.jsonl`](../output/experiments/20260716_internvl35_1b_d1_utterance_u0_v1/review_items_blind.jsonl)
- 独立答案键：[`review_key.jsonl`](../output/experiments/20260716_internvl35_1b_d1_utterance_u0_v1/review_key.jsonl)
- 评分表：[`ratings_template.csv`](../output/experiments/20260716_internvl35_1b_d1_utterance_u0_v1/ratings_template.csv)
- Rubric：[`review_rubric.md`](../output/experiments/20260716_internvl35_1b_d1_utterance_u0_v1/review_rubric.md)
