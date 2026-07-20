# D3-D：官方 dialog policy 信号控制实验

## 1. 问题

D3 `dynamics_fused` 在公共 validation 的 session-level OOF Macro F1 为 `0.6690`，
相对冻结 D1 `0.6341` 提升 `+0.0349`。事后审计发现，对 9,235 个非首 chunk，
`dialog[i]` 是否比 `dialog[i-1]` 新增 assistant turn 与上一 chunk 的 gold interrupt
完全一致。

本实验回答：**D3 的增益有多少可以由这种显式、因果但依赖官方对话构造策略的信号
解释，而不是由视觉/hidden dynamics 解释？**

这是只运行 CPU 线性头的机制控制，不训练 backbone、不重新抽视觉特征、不晋级模型。

## 2. 固定数据与折分

- 使用与 D1/D3 完全相同的 700 sessions / 9,935 chunks；
- 使用冻结的 5-fold domain-stratified session split；
- 每轮仍为 3 个 fit folds、1 个 calibration fold、1 个 test fold；
- L2 网格、LBFGS、class balancing、threshold calibration 和 seed 与 D3 完全相同；
- D1 `fused_linear` 与 D3 `dynamics_fused` 的预测、指标和哈希是固定参照；
- 所有 dialog 特征先从删除 `answers` 后的输入构造；构造时不得读取当前、过去或未来
  gold label，也不得读取 D1/D3 prediction。

公共 validation label 仅按原 D3 规则用于 fit/calibration 和最终 OOF 计分。因此结果仍是
`val-supervised` 开发证据，不是 hidden-test generalization。

## 3. 冻结 dialog-policy 特征

在 chunk `i` 只读取官方 `dialog[i]` 和更早的 dialog prefix。固定 8 个标量：

1. `has_previous_chunk`：是否 `i > 0`；
2. `assistant_added_since_previous`：assistant turn count 是否比 `dialog[i-1]` 增加；
3. `assistant_add_count_since_previous`：本次实际增加的 assistant turn 数；
4. `log1p_visible_assistant_turns`：当前 prefix 可见 assistant turn 总数；
5. `assistant_turns_per_elapsed_chunk`：assistant 总数除以 `max(i, 1)`；
6. `log1p_chunks_since_assistant_addition`：距最近一次新增 assistant turn 经过的 chunk 数；
7. `log1p_last_assistant_text_length`：当前可见最后一条 assistant 文本字符数；
8. `last_assistant_has_interrupt_tag`：最后一条 assistant 文本是否以 `$interrupt$` 开头。

首 chunk 的 `has_previous_chunk=0`、两个新增特征为 0、since-addition 为 0；其他累计
特征只根据首 chunk 当前可见 prefix 计算。固定处理规则为：非 list 的 chunk dialog 或
assistant count 下降直接报错；非 dict/空文本 turn 忽略；一次增加超过 1 时 binary 特征
仍为 1、count 特征保留实际增量，并在审计中计数。不得查看标签后修正规则。

## 4. 固定变体

按以下顺序运行，所有变体仅用于解释，不可晋级：

1. `d1_fused_replay`：原 18 个 D1 scalars + tag margin + 1,024-d current hidden，必须
   逐 chunk 精确复现冻结 D1；
2. `dialog_increment_only`：只用 `has_previous_chunk` 与
   `assistant_added_since_previous`；
3. `dialog_stage_only`：只用全部 8 个 dialog-policy scalars；
4. `d1_fused_plus_dialog_increment`：D1 fused + 前两个 dialog-policy scalars；
5. `d1_fused_plus_dialog_stage`：D1 fused + 全部 8 个 dialog-policy scalars。

不增加文本 embedding、不搜索 history window、不搜索特征子集、不加入 previous gold
label。`assistant_added_since_previous` 与 previous gold 的一致率只在全部 OOF prediction
完成后用标签做只读交叉核对。

## 5. 固定分析

对每个变体报告：

- official Macro F1、interrupt/silent F1、混淆矩阵和 interrupt rate；
- 相对 D1 与 D3 的 Macro F1 差；
- 非首 chunk Macro F1；
- 按 fold、domain、position 的表现；
- 相对 D1 的 5,000 次 session bootstrap；
- 与冻结 D3 的 decision agreement；
- 线性头参数数、每 fold L2/threshold 和标准化系数。

定义 `captured_d3_gain = (variant_macro - 0.6341) / (0.6690 - 0.6341)`。它只是描述
同一公共集上的增益比例，可以小于 0 或大于 1，不是视觉贡献的可加性分解。

## 6. 预注册解释带

以 `d1_fused_plus_dialog_increment` 为最小充分性控制，以
`d1_fused_plus_dialog_stage` 为较丰富控制：

- `captured_d3_gain >= 0.75`：D3 增益“大部分可由 dialog policy 信号重建”；
- `0.25 <= captured_d3_gain < 0.75`：dialog policy “解释部分增益”；
- `captured_d3_gain < 0.25`：该显式 dialog control “只能解释很少增益”。

只有同时满足 session bootstrap 下界大于 0，才把正增益称为稳定。即使控制未重建 D3，
剩余差距也不能自动称为视觉理解，因为 margin、hidden 和 prompt 中仍混合 query、dialog、
历史视觉与当前视觉。

## 7. 路线含义

- 若最小 increment 控制已重建大部分 D3 增益，D3 仍可作为官方协议下的 leaderboard
  模型，但研究表述必须明确它主要利用对话策略/上一动作信号；优先做稳健性和提交打包，
  不应据此扩展 state/granularity。
- 若 dialog controls 只解释很少，同时 U1-V 显示当前视觉有实质影响，再检查 D3 residual
  是否集中在步骤转换，决定是否恢复 S1。
- 无论结果如何，本实验不改 D3 已冻结 head，不用同一 folds 搜索更多 dialog 特征。

## 8. 完成条件

- dialog 特征在删除 answers 后构造且严格因果；
- `d1_fused_replay` 与冻结 D1 predictions/metrics 哈希完全相同；
- 五个变体覆盖全部 9,935 chunks；
- official scorer、bootstrap、分组分析和 previous-gold 事后交叉核对完整；
- 实验工件和中文报告明确区分 leaderboard 可利用信号与视觉/状态理解。
