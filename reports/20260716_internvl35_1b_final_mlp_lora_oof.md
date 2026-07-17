# C1 Small 最终语言 MLP LoRA 严格五折 OOF 实验报告

> 日期：2026-07-16  
> 实验：`20260715_internvl35_1b_final_mlp_lora_d2_oof_v1`  
> 主比较：D1 `fused_linear`，官方 OOF Macro F1 `0.6341`  
> 结论：D2 `adapted_fused_linear` 为 `0.6357`，未通过推广门，D1 保持主基线  
> 证据属性：public-validation-supervised session-level OOF，不是 hidden-test 结果

## 1. 结论摘要

本实验回答一个预先限定的问题：只适配 InternVL3.5-1B 最后一个语言层的 MLP，并直接用 `$interrupt$` / `$silent$` 标签 margin 训练，能否在不重算 28 层长序列的前提下，稳定超过冻结 D1 融合头。

最终结果是**有学习信号，但没有足够稳定的排行榜增益**：

- 主候选 `adapted_fused_linear` 官方 Macro F1 为 `0.6357`，相对 D1 `+0.0016`；
- paired-session bootstrap 的增益中位数为 `+0.00155`，95% 区间为 `[-0.00425,+0.00756]`，跨过 0；
- 只有 2/5 test folds 严格提升，未达到 4/5；
- 非首 chunk Macro 从 `0.60454` 提高到 `0.61017`，这一项通过；
- interrupt/silent F1 分别为 `0.6601/0.6114`，没有类别坍塌；
- 五项推广条件中只有“非首 chunk 提升”和“两类非零”通过，最终 promotion 为 `false`。

因此：

1. D1 `fused_linear=0.6341` 继续作为科学基线和当前部署基线；
2. 不在同一五折上继续搜索 rank、层数、学习率、batch size 或更宽的 MLP；
3. 下一研究动作转向更大规模、预注册的 oracle compact-state replication；
4. granularity 专门建模仍需等待 state 信息先证明可重复增益；
5. 本轮没有执行外部 leaderboard 提交。

## 2. 版本术语

本方向存在多次工程迭代。为避免把失败 smoke、失败全量缓存和正式实验混为一谈，统一使用以下名称。

| 名称 | 范围 | 状态 | 核心结论 |
|---|---|---|---|
| feasibility v1 | 2 chunks，2 状态/候选 | failed | 朴素局部 BF16 MLP 重放与完整序列不逐位等价 |
| feasibility v2 | 2 chunks，4 状态/候选 | failed | MLP 校正使零 adapter exact，但预注册 local/full adapted-margin 门差 `0.0005085` |
| formal cache v1 | 4 个全量 shard，4 状态/候选 | failed/stopped | input 11, chunk 4 暴露 final RMSNorm 批形状 hidden 差 `0.03125` |
| formal cache v2 | 4 个全量 shard，6 状态/候选 | complete | MLP 与 final norm 两级校正，700/9,935 全量 exact |
| formal OOF v1 | fixed batch 64，5-fold | complete, rejected | Macro `0.6357`，未过 promotion |

旧 feasibility v1/v2 的 `failed` 状态保持不变。formal cache v2 和 formal OOF 的成功执行不能倒推改写旧 smoke。

## 3. 模型与参数

### 3.1 基座

| 项目 | 值 |
|---|---|
| 模型 | `OpenGVLab/InternVL3_5-1B-HF` |
| revision | `9191dbccf312b537016f041b25d61c72e7c5c9f3` |
| 权重 SHA256 | `11effd1da2fc0929957d56de4129d1e7d2aed044ea878d40bf83e95b63d38b39` |
| license | Apache-2.0 |
| 基座参数 | `1,060,897,792` |
| language layers | 28 |
| hidden / intermediate | `1024 / 3072` |
| dtype / attention | BF16 / SDPA |

### 3.2 LoRA

唯一目标为 language layer 27 MLP：

```text
model.language_model.layers.27.mlp.gate_proj
model.language_model.layers.27.mlp.up_proj
model.language_model.layers.27.mlp.down_proj
```

冻结设置：

```text
rank=8
alpha=16
dropout=0
bias=none
trainable_parameters=98,304
```

参数口径：

| 组成 | 参数量 |
|---|---:|
| InternVL 基座 | 1,060,897,792 |
| final-MLP LoRA | 98,304 |
| 基座 + LoRA | 1,060,996,096 |
| 再加融合线性头 | 1,060,997,140 |

Small 提交口径中的 total params 与 active params 均约为 `1.06099714B`。`98,304` 只是 trainable params，不能填作 active params。

## 4. 整套管线

### 4.1 输入与输出

每个 chunk 的因果输入只包含：

- 当前及历史已见视频帧；
- query；
- 当前 chunk 之前可见的 dialog；
- 最多 4 个历史 turns；
- 不晚于当前 interval end 的信息。

缓存提取前先从 source row 移除 `answers`。提取代码不访问或存储 gold interrupt 标签。

整体流程：

```text
causal frames + query + visible dialog
    -> frozen vision tower/projector
    -> frozen language layers 0--26
    -> frozen layer-27 attention
    -> cache two forced-tag candidates at three scoring positions
    -> fixed-batch local layer-27 MLP LoRA replay
    -> adapted tag margin + adapted causal hidden
    -> 18 scalar features + margin + 1024 hidden
    -> fold-local linear decision head
    -> $silent$ or $interrupt$ + frozen R0 utterance
```

Adapter 只影响二分类表示。utterance generation 始终禁用 adapter，并复用冻结 R0 raw response，从而避免把决策变化和语言内容变化混为一个实验。

### 4.2 标签打分

Tokenizer 下两个标签都是 3 tokens：

```text
$silent$    -> [16337, 68414, 3]
$interrupt$ -> [3, 54805, 3]
```

训练 margin 为完整三 token log-prob 和之差：

```text
margin = log p($interrupt$ | causal context)
       - log p($silent$ | causal context)
```

Adapter loss 是 class-balanced BCE。它不直接优化最终 Macro F1，最终分类仍由 fit/calibration/test 隔离的线性头完成。

## 5. 六状态缓存与 BF16 校正

### 5.1 为什么需要六状态

每个候选、每个 tag position 保存六个 BF16 tensor：

```text
post_attention_residual
post_attention_normalized_mlp_input
full_batch_base_mlp_output
local_base_mlp_output
full_batch_base_final_hidden
local_base_final_hidden
```

四状态 formal cache v1 只处理了 MLP GEMM 的批形状差异，但全量运行到 `(input=11, chunk=4)` 时发现 final RMSNorm 也有批形状舍入差异，最大 hidden 差为 `0.03125`。因此 shard 0 失败，shard 1--3 主动中止；这些目录保留 `FAILURE.md`，不可合并。

六状态 formal cache v2 增加 reference/local final hidden，在提取阶段同时校正 MLP 与 final norm。已知失败 session 的 16 chunks 回归测试中，hidden、logit、D1 hidden、D1 margin 与 prompt 全部精确通过。

### 5.2 正式训练的 same-batch delta

提取发生在 batch 1，而正式训练固定 batch 64。不能在 batch 64 的 adapted MLP 输出中减去 batch 1 缓存的 local base 值。正式重放改为在**同一个 padded batch shape**内计算 enabled/disabled 差值：

```text
delta_mlp_64 = MLP_adapter_enabled(normalized)
             - MLP_adapter_disabled(normalized)

adapted_mlp = full_batch_base_mlp + delta_mlp_64

delta_hidden_64 = final_norm(residual + adapted_mlp)
                - final_norm(residual + full_batch_base_mlp)

adapted_hidden = full_batch_base_final_hidden + delta_hidden_64
```

LM head 同样有 BF16 批形状风险，因此 tag margin 使用：

```text
adapted_margin = cached_full_batch_base_margin
               + replay_adapted_margin_64
               - replay_adapter_disabled_margin_64
```

尾批重复最后一行补到 64；loss 和导出只读取真实行。该路径没有跨 batch-row 运算，padding 不改变真实样本。

### 5.3 存储

NumPy 不原生支持 BF16，所有状态按 `uint16` 保存 BF16 bit pattern，加载时 bitcast 回 BF16。

```text
2 candidates * 3 positions * 6 tensors * 1024 * 2 bytes
= 73,728 bytes/chunk
```

全量状态区：

```text
9,935 * 73,728 = 732,487,680 bytes
= 698.554688 MiB uncompressed
```

最终 `features.npz` 为 `604,798,620` bytes（`576.781 MiB`）。

## 6. 数据与五折协议

### 6.1 数据

| 项目 | 值 |
|---|---|
| sessions | 700 |
| chunks | 9,935 |
| split unit | session |
| public labels | 用于 adapter/head 的 OOF 开发训练 |
| 证据分类 | `val-supervised` |

同一 session 不跨 fold。每轮：

```text
3 folds fit
1 fold calibration
1 fold test
calibration_fold = (test_fold + 1) % 5
```

Runner 为每折复制标签数组，并将 test indices 置为 sentinel `-1`。Adapter、L2 选择和 threshold 选择的 API 只能看到 fit/calibration 二值标签；test predictions 冻结后才使用原始 test labels 评分。

同一 calibration fold 同时承担 adapter epoch early stopping、线性头 L2 选择和 threshold 选择。这是预先声明的联合选择规则，不是 test 泄漏，但会增加 calibration 方差，是结果解释的限制。

### 6.2 冻结训练配置

```text
optimizer=AdamW
learning_rate=3e-4
weight_decay=0.01
batch_size=64
max_epochs=20
include_zero_adapter_epoch=true
early_stopping_metric=class_balanced_calibration_bce
patience=3
min_delta=1e-4
gradient_clip_norm=1.0
```

决策头：

```text
primary=adapted_fused_linear
features=18 scalars + 1 adapted margin + 1024 adapted hidden
classifier=class-balanced linear logistic regression
optimizer=LBFGS
L2 grid=[1e-5, 1e-4, 1e-3, 1e-2]
threshold=exact calibration Macro-F1 selection
```

## 7. 数值与泄漏控制

### 7.1 缓存全量控制

正式合并器重新读取全部 700 个 session NPZ，不信任分片 summary 的自报值，并检查：

- 700 sessions / 9,935 chunks 恰好覆盖一次；
- source order、interval、chunk index、video path；
- session NPZ SHA、键、shape、dtype、字节数；
- 四分片 code state 与 data manifest 一致；
- corrected hidden/logit、candidate hidden、D1 hidden/margin 门；
- prompt token 完全一致；
- cache 不含标签；
- 合并结果直接与固定 D1 cache 比较 hidden、margin、prompt 和 key。

全量结果：所有最大差均为 `0.0`，prompt/key 完全一致。

### 7.2 固定 batch GPU smoke

在 session 11 的 16 个真实 rows 上，以 batch 64 补 48 个 padding：

| 检查 | 结果 |
|---|---:|
| zero-adapter margin 最大差 | 0.0 |
| zero-adapter hidden 最大差 | 0.0 |
| candidate causal hidden 最大差 | 0.0 |
| synthetic gradient loss | 3.7444336 |
| gradient abs max | 0.8083983 |
| 非零梯度元素 | 57,344 |

### 7.3 正式 OOF 前置控制

CPU `audit-only` 重新构造 D1 OOF，predictions SHA256 精确为：

```text
04183a4083d160662d5f91bff5432a7ca96595dd66b2b0b64f3b430799143ad9
```

正式 PEFT 模型加载后，再对全部 9,935 rows 做 fixed-batch 64 zero replay：margin、hidden、candidate hidden 三项最大差均为 `0.0`。

## 8. 正式结果

### 8.1 三个 variant

| Variant | 对应 D1 | D1 Macro | Adapted Macro | 同类增量 | 相对主基线 0.6341 |
|---|---|---:|---:|---:|---:|
| tag only | `tag_only` | 0.5313 | 0.5879 | +0.0566 | -0.0462 |
| hidden linear | `hidden_linear` | 0.6031 | 0.6064 | +0.0033 | -0.0277 |
| fused linear | `fused_linear` | 0.6341 | **0.6357** | **+0.0016** | **+0.0016** |

Tag-only 的显著提升说明 tag-margin adapter 确实学到了监督信号；但 D1 已有标量和 hidden 的互补信息，加入融合头后边际收益只剩 `+0.0016`。

### 8.2 主候选官方指标

| 指标 | D1 | D2 adapted fused | 变化 |
|---|---:|---:|---:|
| Macro F1 | 0.6341 | 0.6357 | +0.0016 |
| G-mean F1 | 0.6341 | 0.6352 | +0.0011 |
| interrupt precision | 0.6861 | 0.6667 | -0.0194 |
| interrupt recall | 0.5914 | 0.6536 | +0.0622 |
| interrupt F1 | 0.6352 | 0.6601 | +0.0249 |
| silent precision | 0.5891 | 0.6045 | +0.0154 |
| silent recall | 0.6840 | 0.6184 | -0.0656 |
| silent F1 | 0.6330 | 0.6114 | -0.0216 |

D2 confusion matrix 为：

```text
TP=3498, FP=1749, TN=2834, FN=1854
predicted interrupt rate=52.8133%
```

相对 D1 共改变 812 个决策：

```text
FN -> TP: 372
TN -> FP: 351
FP -> TN: 50
TP -> FN: 39
```

净结果是 `+333 TP` 与 `+301 FP`。Adapter 主要把系统推向更高 interrupt recall，而不是同时改善两类边界。

### 8.3 每折结果

| Test fold | fit/cal/test chunks | Best epoch / epochs run | Epoch-0 BCE -> best BCE | L2 | Threshold | D2 Macro | D1 Macro | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6003/1922/2010 | 4 / 7 | 3.5407 -> 0.5800 | 0.01 | -0.2621 | 0.6276 | 0.6354 | -0.0078 |
| 1 | 6050/1963/1922 | 7 / 10 | 3.5766 -> 0.5746 | 0.01 | 0.0138 | 0.6427 | 0.6493 | -0.0066 |
| 2 | 6007/1965/1963 | 5 / 8 | 3.7708 -> 0.5997 | 0.01 | -0.0631 | 0.6340 | 0.6169 | +0.0171 |
| 3 | 5895/2075/1965 | 6 / 9 | 3.7761 -> 0.6010 | 0.01 | -0.0044 | 0.6300 | 0.6305 | -0.0005 |
| 4 | 5850/2010/2075 | 5 / 8 | 3.6214 -> 0.5883 | 0.01 | -0.0652 | 0.6397 | 0.6347 | +0.0049 |

五折 calibration BCE 都显著下降，但 test Macro 只有两折提升，说明“拟合 tag margin”不等于“稳定改善最终 session-held-out 决策”。主候选 fused-linear 及诊断 hidden-linear 的五折 head 均选择冻结网格上界 `0.01`；tag-only 五折均选择 `1e-5`。本实验不据此开展后验 L2 扩网格。

### 8.4 Bootstrap 与推广门

```text
session bootstrap repetitions=5000
median delta=+0.001554
P2.5=-0.004250
P97.5=+0.007565
positive fraction=0.7034
```

| Promotion 条件 | 要求 | 结果 | 通过 |
|---|---:|---:|---|
| 最小 Macro 增量 | >= +0.005 | +0.0016 | 否 |
| bootstrap 下界 | > 0 | -0.00425 | 否 |
| 正向 folds | >= 4/5 | 2/5 | 否 |
| 非首 chunk | D2 > D1 | 0.61017 > 0.60454 | 是 |
| 两类非零 | 两类 F1 > 0 | 0.6601 / 0.6114 | 是 |

最终 `promotion_gate_passed=false`。

## 9. 误差与分组分析

### 9.1 位置

| 位置 | chunks | D2 Macro | D1 Macro | Delta |
|---|---:|---:|---:|---:|
| first | 700 | 0.4989 | 0.4993 | -0.0004 |
| second | 700 | 0.5713 | 0.5685 | +0.0027 |
| 2--4 | 2,099 | 0.6049 | 0.6136 | -0.0087 |
| 5--9 | 3,315 | 0.6199 | 0.6128 | +0.0071 |
| 10+ | 3,121 | 0.5918 | 0.5769 | +0.0150 |

812 个改变中只有 1 个发生在 first chunk。增益确实来自较晚的 mid/late-session chunks，但 2--4 区间明显退化，不能称为全时段一致提升。

### 9.2 Domain

| Domain | chunks | D2 Macro | D1 Macro | Delta |
|---|---:|---:|---:|---:|
| Arts and Crafts | 2,391 | 0.6341 | 0.6294 | +0.0046 |
| Chef | 2,952 | 0.6416 | 0.6502 | -0.0086 |
| Handyman | 2,237 | 0.6490 | 0.6352 | +0.0139 |
| Tutorial | 2,355 | 0.6134 | 0.6142 | -0.0008 |

只有 2/4 domains 提升。Handyman 的正增益与 Chef 的负增益都较明显，进一步说明表示适配没有形成稳定的跨领域改进。

## 10. 资源

### 10.1 缓存提取

| Shard | Sessions | Chunks | Wall time (s) | Peak GPU bytes |
|---:|---:|---:|---:|---:|
| 0 | 175 | 2,596 | 6,580.101 | 3,120,379,904 |
| 1 | 175 | 2,481 | 6,060.123 | 3,120,379,904 |
| 2 | 175 | 2,405 | 5,998.382 | 3,120,379,904 |
| 3 | 175 | 2,453 | 6,116.481 | 3,120,381,440 |

```text
aggregate GPU time=6.8764 GPU-hours
four-GPU wall time=1.8278 hours
peak per GPU=2.9061 GiB
merge wall time=92.978 seconds
```

### 10.2 OOF

```text
device=cuda:4
wall time=368.665 seconds (6.144 minutes)
model load + full zero replay=5.398 seconds
peak GPU memory=2,869,462,016 bytes (2.672 GiB)
preexisting GPU processes=[]
```

## 11. 解释与路线决策

本实验不能简单解释为“LoRA 完全无效”：tag-only 从 `0.5313` 提高到 `0.5879`，五折 calibration BCE 也都显著下降，证明缓存训练链路和监督信号有效。更准确的结论是：

1. 最后层 MLP 的 tag-margin 适配主要改变了 interrupt/silent 倾向；
2. D1 的 scalar + fixed margin + hidden 融合已经吸收了大部分可线性利用信号；
3. 剩余增益集中在部分 late chunks、Handyman 和两个 folds，跨 session/domain 不稳定；
4. 在同一 public validation 五折上继续调 LoRA 超参，会把方法搜索变成后验验证集优化。

因此当前停止 final-MLP LoRA 变体搜索。下一步执行更大规模、独立于四-session pilot 的 oracle-state replication，先回答“显式 procedural state 是否提供可重复的决策增益”。只有 state 增益成立后，才进入 coarse/medium/fine granularity sensitivity；GRPO 继续后置。

即使本实验过门，五折 OOF 也对应五套 adapter/head，不能直接提交。本次未过门，因此不进行全量 adapter refit、部署适配或 submission packaging。

## 12. 产物与复现

### 12.1 正式缓存

```text
output/features/20260716_internvl35_1b_final_mlp_cache_d2_v2/
```

| 产物 | SHA256 |
|---|---|
| `features.npz` | `2c4d7d4d69e54e7156404f747a3ff65cd6c6652c4623dd4d50aad9f538dd455e` |
| `records.jsonl` | `1fd3d8062bc553c6b6c944b434d92f4b917f784889c3bcf0ca73d30e31670347` |
| `summary.json` | `dee68387fc90753a99166740d0dd15616bbb09b88888a6d30460ddf47b95d9e4` |
| `data_manifest.json` | `25b211a7aa13f54344bebfad5b0160a3a095d5329abef2d2133a34affdb232a9` |

### 12.2 正式 OOF

```text
output/experiments/20260715_internvl35_1b_final_mlp_lora_d2_oof_v1/
```

| 产物 | SHA256 |
|---|---|
| primary predictions | `bc48f2845fcb53b55ebcb14a1ebe83d5fc5a6a3015b339257e7b7f3f0927c625` |
| primary official metrics | `985b9774957bc13085c988ce2a2e0c11902791071bea617bc61e75b1cabc7374` |
| primary metrics summary | `ad73bf6f63f187f49e3a79e5b521d12423ba88369dfe2923eb06cccb4e56bf3c` |
| `comparison.json` | `2e7a9888af8207549ece312b08ffd06b41fc2f816657a957d90af99b35e0b4f3` |
| effective `config.json` | `8b3d2eb302421421ebfda64f8d2af4205d3889a9d2f4ef34ce8752b06ad8a58d` |

独立第二次 official scorer 输出与正式 primary `metrics.json`、`metrics_summary.json` 逐字节一致。

### 12.3 核心代码

```text
src/proactive_d2/final_mlp_cache.py
src/proactive_d2/extract_final_mlp_cache.py
src/proactive_d2/merge_final_mlp_cache.py
src/proactive_d2/final_mlp_training.py
src/proactive_d2/run_final_mlp_oof.py
src/proactive_d2/verify_final_mlp_replay.py
src/proactive_d2/tests/
```

最终完整 D2 单元测试 `29/29` 通过，`compileall` 通过。正式运行目录包含每折 adapter、adapted features、head 选择、官方 predictions/metrics、环境、代码状态、数据清单和运行日志。
