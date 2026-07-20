# D4：冻结 dialog-stage 排行榜候选协议

## 1. 目标与身份

D4 把 D3-D 中表现最强的唯一固定变体
`D1 fused + 8 answer-stripped dialog-stage scalars` 工程化为可序列化、可在线因果推理的
Small leaderboard candidate。

D3-D 已预先声明所有变体不可晋级，所以 D4 不是对 `0.6846` 的事后“科学晋级”。它是
在排行榜优先目标下，对一个已经冻结候选进行 full-development refit 和部署闭环。
OOF `0.6846` 是公共 validation 开发证据；full-fit 指标只能作训练闭环 sanity check。

## 2. 冻结特征

基础 1,043 维与 D1 `fused_linear` 完全相同：

- 18 个 `response_temporal` 因果标量；
- 1 个固定 tag margin；
- 1,024 维当前 causal hidden state。

按固定顺序追加 D3-D 的 8 个 dialog-policy 标量：

1. `has_previous_chunk`；
2. `assistant_added_since_previous`；
3. `assistant_add_count_since_previous`；
4. `log1p_visible_assistant_turns`；
5. `assistant_turns_per_elapsed_chunk`；
6. `log1p_chunks_since_assistant_addition`；
7. `log1p_last_assistant_text_length`；
8. `last_assistant_has_interrupt_tag`。

总 feature count 为 1,051，线性 head 参数为 1,052。不得增删特征、加入 D3 hidden
delta、改变 history cap、加入对话文本 embedding 或 previous gold label。

## 3. 冻结 full-refit 规则

- 训练范围：全部 700 public-development sessions / 9,935 chunks；
- classifier：class-balanced linear logistic regression；
- optimizer：LBFGS，最多 120 iterations，float64；
- seed：20260714；
- L2 reduction：sum；
- L2：五个冻结 OOF folds 选择值的中位数，预期为 `0.01`；
- threshold：五个冻结 calibration thresholds 的中位数，预期为
  `0.1263874797442615`；
- full-fit prediction 和 full-fit label 不得参与 L2 或 threshold 选择。

## 4. 在线因果实现

每个 session 初始化独立的 dialog state。chunk `i` 只消费官方当前 prefix
`dialog[i]`：

- 非 list dialog 直接报错；
- 非 dict/空文本 turn 忽略；
- assistant count 下降直接报错；
- 一次新增多条时 binary=1、count 保留实际值；
- 首 chunk 的 previous/addition/since-addition 特征按 D3-D 固定为 0。

在线 runner 必须保留 starter 的绝对视频 intervals、累计最多 32 帧、最近 4 turns、
InternVL shared-vision 特征提取、原始响应生成和 submission answer schema。

## 5. 验证顺序

1. 序列化 full-refit head 并重新加载；
2. 在冻结 cache 上逐 session 用在线 dialog state 回放 9,935 chunks；
3. 8 个 dialog features 必须逐值 exact；
4. 9,935 个 decisions 必须 exact，logit 最大差不超过 `1e-6`；
5. 创建部署 config，参数总量必须低于 2B；
6. 在空闲 GPU 上用原始 session 0 的 10 chunks 做 shared-vision smoke；
7. raw response、prompt tokens、tag margin、hidden、8 个 dialog features、decision 和
   answer 必须与冻结 R0/cache/final 工件一致；logit 最大差不超过 `1e-6`。

## 6. 禁止事项

- 不再比较 dialog increment、dialog stage 子集或 D3 dynamics；
- 不搜索 L2、threshold、窗口、feature transform 或 classifier；
- 不把 train-fit score 当 OOF 或 hidden-test 结果；
- 不以 self-fed dialog 稳健性作未经验证的声明；
- 未经用户明确授权，不上传模型、容器或 leaderboard submission。

## 7. 完成条件

- final head、config、records、predictions、metrics 和哈希齐全；
- offline/online 9,935-chunk audit 通过；
- GPU 10-chunk equivalence smoke 通过；
- Small 总参数按 1,060,897,792 backbone + 1,052 head 报告；
- 中文报告明确说明 `0.6846` 的诊断身份、dialog-policy 依赖和部署风险。
