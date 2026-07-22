# U2：固定 D4 Gate 的 Early-Chunk 视觉事实先行诊断协议

> 冻结日期：2026-07-20  
> 状态：在任何 U2 模型输出生成前冻结  
> 性质：review-informed public-validation mechanism diagnostic

## 1. 目的

U2 只研究 D4 已决定 interrupt 后如何生成 early-chunk utterance，回答：

1. early fallback 是否主要由 assistant history 冷启动造成；
2. 当前 interval 视频是否实质改变输出；
3. 先提取短视觉事实，再生成 utterance，能否在没有 assistant history 时恢复非空内容；
4. 视觉事实是否能在保留历史时减少陈旧步骤依赖。

U2 不修改 D4 feature、head、threshold 或任何 interrupt/silent 决策，不搜索帧数或历史
窗口，不训练参数，不恢复 state/S1，也不调用 gold answer 生成内容。

## 2. 冻结样本

样本取以下完整交集，不再二次挑选：

1. U0 `position_bin` 为 `1:second` 或 `2-4`；
2. Reviewer A/B 均给出 `should_interrupt=yes`；
3. D4 五折 OOF prediction 为 interrupt；
4. D4 全公共 refit prediction 也为 interrupt。

预期得到 21 chunks：second 9、2--4 12；四域数量为 7/5/6/3；原 U0
fallback/nonfallback 为 14/7。样本已被人工评分并使用公共 validation，因此不是独立
验证集。U0 分数只用于上述布尔交集，不用于选择 prompt、threshold 或排序结果。

生成输入中的公共 JSONL 必须先删除整个 `answers` 字段。U0 key、D4 prediction 和评分
仅用于准备 sample/key，不能进入模型消息。

## 3. 固定 D4 Gate

每个样本的 outward decision 固定为 interrupt，且同时记录 OOF/final 两个冻结 D4 决策。
所有视图都用 `$interrupt$` assistant prefill 生成正文；空 continuation 统一回退到现有固定
句。U2 不重新训练或执行 D4 gate，也不运行官方 scorer，因为所有未入样本决策和样本
decision 都不变。

## 4. 六个正文视图

| 视图 | Dialog | Video | Predicted visual facts |
|---|---|---|---|
| `full_history` | 冻结最近 4 turns | 截至当前的累计因果帧 | 无 |
| `no_current_video` | 同 full | 只保留以前 intervals | 无 |
| `query_only_full_video` | 移除 assistant history | 截至当前的累计因果帧 | 无 |
| `query_only_current_video` | 移除 assistant history | 仅当前 interval | 无 |
| `facts_full_history` | 同 full | 截至当前的累计因果帧 | 注入当前 interval 事实块 |
| `facts_query_current` | 移除 assistant history | 仅当前 interval | 注入同一事实块 |

每个样本只生成一次视觉事实，后两个视图必须复用完全相同的事实文本。

## 5. 视觉事实 Pass

事实 pass 只接收 query 和当前 interval 帧。提示要求：

- 最多三个直接可见的 object/action/state；
- 不给建议，不使用历史步骤，不推断动作已完成；
- 证据不足时输出 `unclear`；
- 最多 32 个新 token，greedy decoding。

事实是模型预测，不是人工标注或 oracle state。事实本身可能 hallucinate，必须原样保存并
进入单独事实核查包。

## 6. 冻结推理设置

- Backbone：`OpenGVLab/InternVL3_5-1B-HF` 固定 revision；
- BF16 + SDPA，greedy decoding；
- 每 interval 16 帧，累计最多 32 帧；
- history 上限 4 turns；
- utterance 最大 64 tokens；事实最大 32 tokens；
- 448 输入尺寸；seed `20260713`；
- 一次模型加载完成全部 pass；运行前检查 GPU，不中断已有进程。

这不是帧数或历史长度搜索；这些值与冻结 D4/U1 一致。

## 7. 自动指标

每个视图报告：

- fallback/nonempty rate、word count、显式 completion claim；
- 与 `full_history` 的 exact match、文本相似度和 fallback 转移；
- utterance 与 predicted fact block 的词法 overlap，仅作利用诊断；
- overall、position 和 domain 分层。

重点 paired contrasts：

- `no_current_video - full_history`：当前视觉敏感性；
- `query_only_full_video - full_history`：assistant history 依赖；
- `facts_query_current - query_only_current_video`：无历史 cold-start rescue；
- `facts_full_history - full_history`：事实块在保留历史时的影响。

在输出前冻结两个 review-priority 条件：

- facts pass 非空/非 `unclear` rate 至少 `0.80`；
- `facts_query_current` 相对 `query_only_current_video` 的 nonempty rate 至少提高 `0.20`。

条件只决定后续优先级，不构成质量晋级。

## 8. 人工质量边界

自动文本变化、fact overlap 和 fallback rescue 均不能证明 grounding。运行完成后必须生成：

- 六视图随机盲化的 utterance review package；
- 独立的 predicted-fact review package；
- variant key 与评分模板。

新 utterance rubric 除原内容维度外，增加 `current_visual_support_1_5`、
`unsupported_visual_claim_flag` 和 `stale_history_flag`。由于 U0 groundedness A/B kappa 仅
`0.0508`，正式评分前必须先用少量 calibration examples 对齐“当前画面支持”的判据。

## 9. 完成条件

- 21 个样本和两个 D4 gate 来源完整校验并哈希冻结；
- 21 次 R0 full-view replay 与冻结记录精确一致；
- 21 个 fact pass、126 个最终正文记录完整；
- 所有生成输入无 `answers`、未来 dialog 和未来帧；
- 自动分析、runtime、环境、代码状态、输入输出哈希和盲评包落盘；
- 中文报告明确区分 mechanism sensitivity、predicted facts 和人工质量证据。
