# C1 Small 最终语言 MLP LoRA / 联合决策损失可行性审计

> 日期：2026-07-15  
> 2026-07-15 原始范围：工程可行性、数值一致性、参数与计算协议；当时不产生 OOF 效果结论  
> 当前有效基线：D1 `fused_linear`，官方 OOF Macro F1 `0.6341`

> **2026-07-16 后续状态：**本报告记录的 feasibility v2 是两 chunk、四状态/候选的历史工程 smoke，不能代表后续全量缓存。正式四状态 cache v1 在 `(input 11, chunk 4)` 暴露 final RMSNorm hidden 差 `0.03125` 后失败；正式六状态 cache v2 随后完成 700 sessions / 9,935 chunks 的全量 exact 审计。冻结五折 OOF 已完成，主候选 Macro F1 `0.6357`，相对 D1 仅 `+0.0016` 且 bootstrap 跨零，未推广。完整结果见 [正式 OOF 报告](20260716_internvl35_1b_final_mlp_lora_oof.md)。以下旧 smoke 的 failed 状态和原始数值保持不变。

## 1. 2026-07-15 当时结论

本轮得到三个需要同时保留的结论：

1. **直接在 9k-token 上反复执行 28 层端到端 LoRA 五折训练不现实。** D1 的 9,935 个 prompt 中位长度为 8,716 tokens，完整 5-fold 多 epoch 重算会远超当前必要成本。
2. **只适配最后一个语言 decoder layer 的 MLP 在结构上可局部训练。** 最后一层 attention 已经完成后，MLP 对每个 token 位置独立；标签打分只需要两个候选各 3 个位置。最终目标严格限定为 layer 27 的 `gate_proj/up_proj/down_proj`，rank 8 共 98,304 个可训练参数。
3. **不能把现有 smoke 事后改写为“全部通过”。** v1 的朴素三位置重放受 BF16 GEMM 批形状影响；v2 虽将实际部署基座路径校正到 hidden/logit/margin 全部零误差，但预注册的“局部适配器与完整序列适配器 margin 差 < 0.01”实测 `0.0105085`，以 `0.0005085` 未通过。

因此本轮正式判定是：

- 对“朴素缓存重放等同于原生完整序列 PEFT LoRA”作 **no-go**；
- 对“缓存校正的、位置局部的联合二分类适配器”作 **有条件 go**，允许进入一个已冻结配置的五折 OOF；
- 这只是允许继续实验，不是模型效果已经优于 `0.6341`，也不是提交候选已经产生。

## 2. 为什么要做这一项

D1 已证明，18 个因果标量、固定标签 margin 和 1,024 维因果 hidden 的线性融合具有互补信号，OOF Macro F1 达到 `0.6341`。随后 width-8 residual MLP 只达到 `0.6351`，增益 `+0.0010` 且 bootstrap 跨零，说明继续搜索普通小 MLP 宽度没有依据。

下一项合理问题不是“再换一个 head”，而是：让视觉语言表示本身接受 interrupt/silent 联合损失，是否能把当前线性头尚未分离的信号推开，同时不改变 utterance generation。

## 3. 模型结构审计

### 3.1 基座

- 模型：`OpenGVLab/InternVL3_5-1B-HF`
- revision：`9191dbccf312b537016f041b25d61c72e7c5c9f3`
- license：Apache-2.0
- 基座总参数：`1,060,897,792`
- 语言模型：28 层 Qwen3 decoder
- hidden size：1,024
- MLP intermediate size：3,072
- dtype：BF16
- attention：SDPA

最后一个语言层的运行时模块为：

```text
model.language_model.layers.27.mlp.gate_proj
model.language_model.layers.27.mlp.up_proj
model.language_model.layers.27.mlp.down_proj
```

权重形状分别为：

```text
gate_proj: [3072, 1024]
up_proj:   [3072, 1024]
down_proj: [1024, 3072]
```

### 3.2 标签与打分位置

当前 tokenizer 下：

```text
$silent$    -> [16337, 68414, 3]
$interrupt$ -> [3, 54805, 3]
```

两个标签均为 3 tokens。完整标签分数是三个自回归位置 log-prob 的和：

```text
margin = log p($interrupt$ | causal context)
       - log p($silent$ | causal context)
```

训练损失为对该 margin 的 class-balanced binary cross entropy。标签只进入 loss，不进入视频、消息、prompt 或缓存构造。

### 3.3 LoRA 参数

rank 8、alpha 16、dropout 0、bias none。每个投影的 LoRA 参数是 `r * (in + out)`，因此：

```text
3 * 8 * (1024 + 3072) = 98,304
```

PEFT 实际审计得到 6 个可训练 tensor，且全部只位于 layer 27 MLP 的 A/B 矩阵；视觉塔、projector、attention、其他 27 个语言层和 LM head 均未匹配。

参数口径：

| 组成 | 参数量 |
|---|---:|
| InternVL 基座 | 1,060,897,792 |
| final-MLP LoRA | 98,304 |
| 基座 + LoRA | 1,060,996,096 |
| 再加 D1 1,044 参数融合头 | 1,060,997,140 |

该系统低于 Small 的 2B 上限。提交表单中的 active params 不能填成 98,304；该值只是 trainable params。由于推理经过整个 dense 基座，total params 和 active params 均约为 `1.06099714` billions。

## 4. 为什么不做普通端到端 LoRA

D1 标签缓存统计为 9,935 chunks：

| prompt tokens | 数值 |
|---|---:|
| min | 440 |
| median | 8,716 |
| mean | 8,404.16 |
| p90 | 8,750 |
| p99 | 8,774 |
| max | 8,835 |

如果每个 fold、每个 epoch 都重新执行视觉塔和 28 个语言层，绝大多数计算只是在重复生成最后三个打分位置之前已经固定的表示。该方式既慢，也会增加共享 GPU 占用，没有必要。

Qwen3 最后一个 decoder layer 的顺序是：

```text
attention residual
  -> post_attention_layernorm
  -> position-local MLP
  -> residual add
  -> final language norm
  -> lm_head
```

适配器只在最后 MLP，因此不会改变该层 attention、KV 或其他 token 的输入。可以一次性执行完整因果前向，只保存两个候选各三个打分位置所需的局部状态，然后从缓存反向训练。

## 5. 缓存设计与 BF16 问题

### 5.1 v1 朴素缓存

最初每个候选保存：

```text
post_attention_residual
post_attention_normalized_mlp_input
```

每 chunk 两候选、各 3 位置、BF16 的理论大小为 24,576 bytes，全量约 233 MiB。

数学上可直接重放：

```text
h = final_norm(residual + adapted_mlp(normalized_input))
```

但 GPU BF16 GEMM 会根据矩阵批形状选择不同内核。完整前向对约 4k--9k tokens 计算 MLP，局部重放只对 3 个位置计算；即使权重与输入逐元素相同，舍入路径也不保证逐位一致。

v1 两个 chunk 上观察到：

| 项目 | 最大绝对差 |
|---|---:|
| base hidden | 0.1875 |
| base logit | 0.125 |
| base margin | 0.0030565262 |

因此 v1 正式 smoke 的精确门失败，不能声称“缓存与完整模型 exact”。

### 5.2 工程 smoke v2：四状态、仅 MLP 校正

v2 每个候选改为保存四个 BF16 tensor：

```text
residual
normalized_input
full_batch_base_mlp_output
local_base_mlp_output
```

部署与训练统一使用：

```text
deployed_mlp_output
  = full_batch_base_mlp_output
  + local_adapted_mlp_output
  - local_base_mlp_output
```

这样完整批形下的基座值原样保留，局部路径只提供 adapter delta。零适配器时，两个 local output 相减为零，严格回到完整基座输出。

缓存大小变为：

```text
49,152 bytes/chunk
488,325,120 bytes / 9,935 chunks
约 465.7 MiB（未压缩）
```

NumPy 不原生支持 BF16，因此正式缓存必须按 `uint16` 保存 BF16 bit pattern，加载后再 bitcast 回 BF16，不能转 FP16 后假装无损。

## 6. GPU smoke 协议

两个固定 chunk 来自 input index 0：chunk 0 为 interrupt，chunk 2 为 silent。选择它们只是为了让 backward smoke 同时覆盖两个类别，不用于估计效果。

严格的数据顺序为：

1. 从 source row 移除 `answers`；
2. 仅用截至当前 interval 的累计帧、query 和可见 dialog 构造 prompt；
3. 构造并冻结两候选缓存；
4. 缓存完成后才读取两个 gold label，送入 BCE loss。

因此 smoke 的特征提取不读标签，但两样本训练本身是 public-validation-supervised。它不产生 Macro F1。

## 7. 实验结果

### 7.1 v1 正式结果

产物：

```text
output/experiments/
  20260715_internvl35_1b_final_mlp_lora_feasibility_v1/
```

v1 通过了：

- 98,304 可训练参数与 6 个 tensor 的目标审计；
- 非零适配器确实改变输出；
- 最后一层 MLP 之前的 residual/normalized state 在启用适配器后零差；
- class-balanced BCE 可反向传播；
- 禁用 adapter 后，R0 raw generation 逐字符串一致；
- 峰值显存约 3.14 GiB。

v1 未通过朴素局部重放的 hidden/logit/margin 精确门。正式状态是 `failed`。

### 7.2 v2 正式结果

产物：

```text
output/experiments/
  20260715_internvl35_1b_final_mlp_lora_feasibility_v2/
```

核心数值：

| 项目 | v2 结果 |
|---|---:|
| 校正后 base hidden 最大差 | 0.0 |
| 校正后 base logit 最大差 | 0.0 |
| 校正后 base margin 最大差 | 0.0 |
| 未校正 base margin 最大差 | 0.0030565262 |
| local vs full adapted margin 最大差 | 0.0105085373 |
| adapter 前 state 最大差 | 0.0 |
| local adapter 最大非零影响 | 3.0 |
| 峰值 GPU 显存 | 3,144,280,064 bytes |
| 完整墙钟 | 14.479 s |
| 两 chunk 缓存构造 | 5.627 s |
| 缓存训练 | 0.197 s |

两样本 backward smoke：

```text
initial loss = 2.4981136
final loss   = 2.6604e-13
optimizer steps = 6
```

loss 中间有震荡，且最终完全过拟合两个样本。这只证明梯度链路有效，不能证明正式学习率或泛化效果。正式 OOF 已另行冻结更保守的 learning rate `3e-4`，不沿用 smoke 的 `5e-3`。

adapter 训练后再次禁用 adapter，chunk 0 的 raw generation 与冻结 R0 完全一致：

```text
Place the stickers on the cover of the notebook.
```

v2 的 11 个门中 10 个通过。唯一失败门是 local/full adapted margin `< 0.01`，实测 `0.0105085373`。正式状态仍为 `failed`，不能事后把阈值放宽成 0.011 并宣称通过。

### 7.3 两个非正式失败尝试

- `...v1_failed_attempt1`：starter 指纹文件路径写错，在模型加载前停止；没有训练。
- `...v1_failed_attempt2`：两样本第 6 步已饱和为零梯度，旧代码错误要求每一步梯度都非零；没有产生效果指标。

这些目录保留用于说明工程修正，不应作为实验结果引用。

## 8. 如何解释 v2 唯一失败项

完整序列 adapter forward 与局部 adapter forward 在实数运算中是同一位置局部函数，但 BF16 下的矩阵形状不同，舍入不同。正式部署已经定义为：

```text
base full-batch candidate forward, adapter disabled
  -> exact cached base output
  -> local cache-corrected adapter delta
  -> adapted tag score / causal hidden
```

旧 smoke 的训练也走同一局部路径。后续全量运行证明，仅校正 MLP 不足以覆盖 final RMSNorm 的 BF16 批形状尾部，因此正式训练改为六状态缓存，并在固定 batch 64 内用 adapter-enabled 与 adapter-disabled 的同批形差值同时校正 MLP、final norm 和 LM-head margin。该修正发生在任何 D2 OOF metric 出现前。

不过，`0.0105085 > 0.01` 是预先写下的门，必须按失败记录。正确做法不是改写 v2，而是在任何 OOF metric 出现前冻结新的正式协议：完整序列 forward 只保留为诊断，不参与候选推广；真正的硬门是零 adapter 对 D1 基座逐元素/逐预测复现，以及局部训练与局部推理一致。

## 9. 冻结的五折 OOF 协议

正式配置：

```text
configs/d2_internvl35_1b_final_mlp_lora_oof.json
```

### 9.1 Split

- 复用 D1 的冻结 5-fold session manifest；
- 每轮 3 folds fit、1 fold calibration、1 fold test；
- 同一 session 不跨折；
- test labels 只在该 fold 的预测冻结后读取。

### 9.2 Adapter 训练

```text
rank=8, alpha=16, dropout=0
AdamW, lr=3e-4, weight_decay=0.01
batch_size=64, max_epochs=20
gradient_clip=1.0
calibration BCE early stopping
patience=3, min_delta=1e-4
```

只允许这一套配置；在 OOF 结果可见后不得搜索 rank、层数、学习率或更多 MLP 目标。

### 9.3 主比较

主 variant 为 `adapted_fused_linear`：

```text
18 个原 D1 response_temporal 标量
+ adapted tag margin
+ adapted 1024 维 causal hidden
```

线性头训练、L2 网格、calibration threshold 和官方 scorer 全部保持 D1 协议。`adapted_tag_only` 与 `adapted_hidden_linear` 只作诊断。

零 adapter 控制必须先逐字节复现 D1 `fused_linear` predictions SHA256：

```text
04183a4083d160662d5f91bff5432a7ca96595dd66b2b0b64f3b430799143ad9
```

### 9.4 推广门槛

相对 D1 `0.6341`，必须同时满足：

- Macro F1 至少 `+0.005`；
- paired-session bootstrap 95% 下界大于 0；
- 至少 4/5 folds 严格提升；
- 非首 chunk 有增益；
- 两类 F1 均不坍塌。

否则停止 LoRA 变体搜索，转向更大规模预注册 oracle-state replication。

## 10. 计算预算

D1 原四分片标签缓存实际消耗：

```text
shard wall time: 8073 / 7614 / 7270 / 7731 s
aggregate GPU time: about 8.52 GPU-hours
four-GPU wall time: about 2.24 h
peak memory: about 3.14 GB/GPU
```

当时针对四状态设计给出的保守预算为：

- 全量缓存：8.0--9.5 GPU-hours；
- 4 张空闲 GPU 并行：约 2.0--2.5 h；
- 峰值显存：预计小于 3.5 GB/GPU；
- 缓存：未压缩约 465.7 MiB；
- 缓存上的五折训练：预计 10--30 min 单卡。

以上是 2026-07-15 feasibility 阶段的估算和当时状态。2026-07-16 经用户授权后，正式六状态缓存实际为 `73,728 bytes/chunk`、`732,487,680 bytes`（`698.554688 MiB`）未压缩；四卡 aggregate `6.8764 GPU-hours`、wall `1.8278 h`、单卡峰值约 `2.906 GiB`。压缩 `features.npz` 为 `604,798,620 bytes`。正式 OOF 单卡耗时 `368.665 s`，峰值 `2.672 GiB`。

## 11. 仍需注意的风险

1. 旧 feasibility v2 只有两个 chunk；它没有覆盖的数值尾部已由四状态 full-cache failure、16-chunk 定向回归和六状态 9,935-chunk 全量审计显式暴露并修正。
2. 公开 validation labels 用于 adapter 与 head 训练，结果只能称为 `val-supervised OOF`，不能当 hidden-test 证据。
3. 98,304 是 trainable 参数，不是 active 参数。
4. utterance generation 必须始终禁用 adapter；否则会同时改变内容生成，破坏单变量归因。
5. 全量缓存已经按 BF16 bit pattern 无损保存，并对 D1 margin/hidden/prompt/key 完成全量零差复现审计。
6. 正式 OOF 未过门；不得再在同一五折上搜索更多 rank、更多层或学习率。

## 12. 产物与复现

核心代码：

```text
src/proactive_d2/final_mlp_lora.py
src/proactive_d2/smoke_final_mlp_lora.py
src/proactive_d2/final_mlp_cache.py
src/proactive_d2/extract_final_mlp_cache.py
src/proactive_d2/merge_final_mlp_cache.py
src/proactive_d2/final_mlp_training.py
src/proactive_d2/run_final_mlp_oof.py
src/proactive_d2/verify_final_mlp_replay.py
src/proactive_d2/tests/test_final_mlp_lora.py
src/proactive_d2/tests/test_final_mlp_training.py
src/proactive_d2/tests/test_merge_final_mlp_cache.py
src/proactive_d2/tests/test_run_final_mlp_oof_protocol.py
```

配置：

```text
configs/d2_internvl35_1b_final_mlp_lora_smoke.json
configs/d2_internvl35_1b_final_mlp_lora_smoke_v2.json
configs/d2_internvl35_1b_final_mlp_lora_oof.json
```

最终完整 D2 单元测试：29/29 通过；`compileall` 通过。旧 v1/v2 smoke 产物保持原样；正式缓存位于 `output/features/20260716_internvl35_1b_final_mlp_cache_d2_v2/`，正式 OOF 位于 `output/experiments/20260715_internvl35_1b_final_mlp_lora_d2_oof_v1/`。完整效果、fold、bootstrap、资源和 SHA256 见 [2026-07-16 正式报告](20260716_internvl35_1b_final_mlp_lora_oof.md)。
