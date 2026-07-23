# D6 主干条件历史注入实验冻结协议

冻结日期：2026-07-22

状态：用户授权、正式 OOF 前预注册、尚无 efficacy 结果。

## 1. 唯一问题与证据边界

D6 只回答一个问题：在 D4.2 `history8` 的严格因果输入上，query-conditioned
当前区间视觉记忆和最后四层 attention LoRA 是否能改善决策表示，并在重新适配
完整 D4 线性头后稳定超过 D4.2 `history8` OOF。

数据仍是 `facebook/wearable-ai` EgoProactive 公开 validation。适配器、线性头、
阈值和候选结构均使用公开标签，因此结果必须称为 post-selection、
`val-supervised` 五折 OOF，不能称为 hidden-test 或独立泛化证据。输出 utterance
始终取 adapter-disabled 的冻结 `history8` generation；D6 只改变二分类表示。

## 2. 冻结模型与输入

- Backbone：`OpenGVLab/InternVL3_5-1B-HF` revision
  `9191dbccf312b537016f041b25d61c72e7c5c9f3`。
- Backbone 参数：1,060,897,792；BF16、SDPA、greedy、official prompt/dialog。
- 输入策略：`max_frames=32`、`frames_per_interval=16`、
  `max_history_turns=8`、`max_new_tokens=64`。
- Vision tower、multi-modal projector、embedding、LM head、language layers 0--23
  及 layers 24--27 的 base weights 全冻结。
- 模型输入在任何 forward 前必须移除 `answers`。Test fold labels 在该 fold 的
  primary 和固定消融预测、logits、标准化器、L2 和 threshold 全部冻结前只能以
  sentinel 形式存在于训练进程之外。
- 每个 chunk 只允许使用当前或更早的绝对 `video_intervals`、当前官方 dialog
  prefix 和 query。不得使用未来视频、未来 dialog、未来标签或全 session 特征。

## 3. 冻结 memory 张量定义

对 D4 uniform cumulative 32-frame 选择结果保留每帧来源 interval。一次冻结
vision forward 得到所有输入帧的 projected patch tokens；memory 只接收其中来源
等于当前 chunk 的 tokens。若当前 interval 在 32-frame 采样中没有 token，立即
报错，不以历史或零 token 替代。

语言层 24 输入处的 `prompt_tokens - 1` 是唯一 assistant-boundary 位置。该位置
hidden 是 query condition。memory 使用 FP32 参数和计算：

1. 一个共享 `LayerNorm(1024, elementwise_affine=True)` 同时归一化 query 和视觉
   patch tokens；
2. 带 bias 的独立 `q_proj/k_proj/v_proj: Linear(1024,128)`；
3. 4 heads、每 head 32 维的 scaled dot-product attention，不使用 attention
   dropout、额外 output projection 或位置编码；
4. 四头 context 按原 head 顺序拼接成 128 维；
5. `GRUCell(128,128)` 以 context 为 input、上一 chunk state 为 hidden；
6. `LayerNorm(128, elementwise_affine=True)` 归一化新 state；
7. `Linear(128,1024,bias=True)` 生成 residual，其 weight 和 bias 均零初始化；
8. residual 只加到 layer-24 前的 assistant-boundary hidden，再转换为 backbone
   dtype；其他 token 不变。

Session state 在每个 session 开始时精确置零。每个 chunk 的 silent/interrupt
batch-1 language forward 共用同一次 vision forward、同一旧 state 和同一新
state；两个 candidate 独立重算出的 query/update 必须数值相同，state 只提交
一次。每个 chunk loss backward 后 state 立即 detach，禁止跨 chunk BPTT。

Memory 参数数为：共享 1024-LN `2,048`，Q/K/V `393,600`，128-LN `256`，
GRUCell `99,072`，injection `132,096`，合计 `627,072`。

## 4. 冻结 LoRA 定义

- 目标：language layers 24、25、26、27 的 self-attention
  `q_proj/k_proj/v_proj/o_proj`，除此之外不得出现 trainable LoRA tensor。
- rank 8、alpha 16、scaling `alpha/rank=2`、dropout 0、无 LoRA bias。
- A 使用 PyTorch `kaiming_uniform_(a=sqrt(5))`；B 精确零初始化。
- A/B 参数与 LoRA residual 计算使用 FP32，residual 在加到 BF16 base output 前
  转回 base dtype。
- Qwen3 projection shape 按 checkpoint 原值审计，四层 LoRA 总参数必须精确为
  `327,680`，共 32 个 trainable A/B tensors。
- Epoch-0 时 memory residual 和所有 LoRA residual 均为零；enabled 路径必须与
  adapter-disabled D4.2 shared-vision 路径等价。

## 5. 冻结训练协议

完全复用 D4.2 `fold_manifest.json`，算法
`domain_stratified_sha256_round_robin`、seed `d1-session-oof-v1`、五折。Test fold
为 `f`，calibration fold 为 `(f+1)%5`，其余三折为 fit。不得重新生成、按 query
分组或替换 manifest。

每折从相同零初始化 adapter 独立训练：

- objective：`interrupt_log_probability - silent_log_probability` margin 的
  `BCEWithLogitsLoss`；fit folds 的 `pos_weight = silent_count/interrupt_count`；
- optimizer：AdamW，memory LR `3e-4`、LoRA LR `1e-4`、weight decay `0.01`；
- gradient global norm clip `1.0`；
- 最多 5 epochs，另记录未训练 epoch-0 control；
- 每 epoch 的 session 顺序由 `seed + 1000*fold + epoch` 的本地 PRNG 确定；
- 梯度按 chunk 累积；累计达到或超过 64 chunks 后，仅在当前 session 完成处
  clip/step/zero。epoch 末在 session 边界 flush 剩余梯度；同一 session 内参数
  绝不变化；
- calibration BCE 使用 fit `pos_weight`，每 epoch 完整按 source order 推理；
  最低 calibration BCE checkpoint 为 best，连续 2 epochs 无严格改善后停止；
- 不使用 scheduler、warmup、mixed-precision scaler、dropout 或 gradient
  checkpointing。

Checkpoint 只能保存 memory/LoRA tensors、这些 tensors 的 optimizer state、
fold/epoch/session-boundary 训练状态、配置与模型 hash、PRNG state 和审计统计；
不得复制 base-model tensor。Resume 只能从 session-boundary checkpoint 开始，
并必须确定性复现未中断训练。

## 6. 冻结决策头与消融

Best adapter 固定后，每折对 700 sessions 进行 label-free primary 特征提取。
特征 schema 精确复用 D4.2 的 1,051 维：18 causal scalar、1 tag margin、1,024
final causal hidden 和 8 dialog-stage scalar。该折三 fit folds 拟合 standardized
float64 class-balanced linear head；calibration fold 只在 L2
`{1e-5,1e-4,1e-3,1e-2}` 中选值并精确选择 Macro-F1 threshold；test fold 仅在
上述对象冻结后解封 labels 和生成最终预测。

`LoRA-disabled` 和 `memory-disabled` 只在该折 test sessions 上运行，沿用 primary
head、primary standardization、primary L2 和 primary threshold，不重新拟合，
不参与候选选择。Adapter-disabled 用于 exact D4.2 reproduction 和 smoke，不是
第三个 promotion candidate。

## 7. 正式运行前硬门

1. 102-chunk、四域、source indices `143,356,472,609` 的 zero-init 审计：离散
   字段精确相同；hidden/tag 差异为零或不超过 D4.3 的数值容差；silent 和
   interrupt memory update 相同；future-frame/query/dialog mutation 不改变已产生
   历史 logits；session reset 有效。
2. Rotation-0 三 fit folds 完整一 epoch并在 calibration fold 完整推理的
   trainability smoke；只报告 loss、梯度、参数变化、时延和资源，不做 efficacy
   结论。
3. Peak allocated GPU memory `<=70 GiB`；从 trainability smoke 线性估计的正式
   单 fold wall time `<=48 hours`；smoke 最长 inference session model time
   `<=240s`。任何失败立即停止，不缩宽度、不减层、不改 rank/LR。

GPU launcher 最多使用 5 张卡，仅选择无外部 compute process 且 free memory
`>=75 GiB` 的 A800；每卡一个 fold。不得终止、迁移或共享他人进程。

## 8. 评价、晋升与停止规则

Primary 与精确复现的 D4.2 `history8=0.6988` 比较。晋升必须同时满足：

- official Macro F1 增量至少 `+0.005`，即至少 `0.7038`；
- 5,000 次 paired-session bootstrap 95% 下界 `>0`；
- 至少 4/5 folds、3/4 domains 严格提升；
- previous-interrupt、previous-silent、non-first-chunk Macro 均不下降；
- interrupt/silent 两类均非退化；
- parameter、causality、memory 和官方 300 秒 session timeout 全部通过。

报告必须包含两类 P/R/F1、Macro、G-mean、TP/FP/TN/FN、预测比例、逐
domain/fold/previous-response/non-first 指标、paired bootstrap、decision changes、
memory residual norm、attention entropy、best epoch、calibration BCE、峰值显存、
单 session/总 wall time、checkpoint/config/code/data/environment hash。

任一门失败即结束本结构族，不在相同 folds 上调整注入层、memory width、head
数、rank、LoRA target、LR、epoch、loss、feature、L2 或 threshold。全部通过时，
才按五折中位 best epoch、L2 和 threshold 做一次全开发集 refit，并执行独立
102-chunk 在线等价/时延审计。外部 leaderboard、registry 或 model upload 仍需
另行授权。

## 9. 参数与许可

- Memory：627,072。
- LoRA：327,680。
- Decision head：1,052。
- Base + D6 components：1,061,853,596，总量低于 Small 2B。
- 训练源：EgoProactive public validation，CC-BY-NC-4.0；无外部训练数据。
- Backbone：Apache-2.0。项目顶层 source license 仍未由 owner 决定，因此本协议
  不构成 prize-source eligibility 声明。
