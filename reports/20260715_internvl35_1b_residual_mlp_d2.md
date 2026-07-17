# InternVL3.5-1B D2 轻量非线性残差决策头实验报告

> 实验 ID：`20260715_internvl35_1b_residual_mlp_d2_oof_v1`  
> 日期：2026-07-15  
> 任务：ECCV 2026 Wearable AI Challenge，EgoProactive Small  
> 结论：**存在微弱正增益，但未通过 promotion gate，不替换 D1**

## 1. 目标与设计理由

D1 已证明 18 个严格因果 scalar 特征、1 个 fixed-tag margin 和 1,024 维冻结 multimodal hidden 的线性融合能达到官方 OOF Macro F1 `0.6341`。D2 要回答的唯一问题是：这些相同特征之间是否还存在一个很小的非线性交互残差，可以在不微调 InternVL 的情况下稳定提升决策。

为了避免一个新 MLP 完全重学并破坏 D1 边界，D2 使用 residual 设计：

```text
1,043 维 frozen fused feature
       |
       +--> D1 standardized linear logit ------------------+
       |                                                   |
       `--> Linear(1043, 8) -> GELU -> Linear(8, 1) -------+
                                                           |
                                               final D2 logit
```

第二个线性层以全零初始化，因此 epoch 0 的 residual 严格为 0。每个 fold 的模型从对应 D1 线性 logit 出发；如果 calibration loss 不支持新增容量，early stopping 可以直接保留 D1。

## 2. 冻结协议

所有结构和训练设置在看到 D2 OOF 指标前写入配置，只运行一个正式配置：

| 项目 | 设置 |
|---|---|
| Backbone | `OpenGVLab/InternVL3_5-1B-HF`，完全冻结 |
| 输入 | 与 D1 相同的 1,043 维 causal fused feature |
| Split | 原 D1 domain-stratified 5-fold session split |
| 每轮 | 3 folds fit / 1 fold calibration / 1 fold test |
| Residual hidden width | 8 |
| Activation | GELU |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `0.01` |
| Batch size | 512 |
| Max epochs | 80 |
| Early stopping | calibration class-balanced BCE，patience 12 |
| Min delta | `1e-4` |
| Gradient clip | `1.0` |
| Dtype / device | float32 / CPU，8 threads |
| Threshold | 训练结束后只在轮转 calibration fold 上选择 |

Test fold 的标签在预测冻结前不参与训练、早停、阈值或架构选择。神经特征缓存本身不读取或存储标签。

## 3. 参数量

Residual MLP 参数为：

```text
1043 * 8 + 8 + 8 + 1 = 8,361
```

加上原 D1 线性权重和 bias：

```text
D2 head = 8,361 + 1,044 = 9,405 parameters
system  = 1,060,897,792 + 9,405
        = 1,060,907,197 parameters
```

它仍属于 Small，且相对 backbone 的参数增量可以忽略。

## 4. D1 复现保护

正式 D2 运行前，每个 fold 都重新执行原 D1 的四个 L2 候选并验证：

1. selected L2 与冻结 diagnostics 相同；
2. calibration threshold 在多线程浮点容差内一致；
3. 每个 test chunk 的 D1 决策完全一致；
4. 700-session D1 predictions 的 SHA256 逐字节一致。

最终 D1 复现哈希仍为：

```text
04183a4083d160662d5f91bff5432a7ca96595dd66b2b0b64f3b430799143ad9
```

前两次尝试分别因 `2.739e-08` 和 `4.881e-06` 的 CPU 线程归约阈值差被过严的数值断言提前终止；均发生在 residual 训练和 D2 指标生成前。失败产物被保留并标记。正式运行把纯数值容差设为 `1e-4`，上述逐 chunk 和整文件行为保护没有放宽。

## 5. Promotion gate

D2 必须同时满足：

| 检查 | 门槛 |
|---|---|
| Macro F1 delta vs D1 | 至少 `+0.005` |
| Paired session bootstrap | 95% 下界大于 0 |
| Non-first chunk | 严格优于 D1 |
| Positive folds | 至少 4/5 |
| 两类 F1 | 均非零 |

## 6. 官方结果

| 模型 | Macro F1 | Interrupt F1 | Silent F1 | TP / FP / TN / FN | Interrupt rate |
|---|---:|---:|---:|---:|---:|
| D1 fused linear | `0.6341` | `0.6352` | `0.6330` | 3165 / 1448 / 3135 / 2187 | `46.43%` |
| D2 residual MLP8 | `0.6351` | `0.6375` | `0.6327` | 3188 / 1461 / 3122 / 2164 | `46.79%` |

全精度 Macro delta：

```text
+0.000993775517393547
```

D2 相对 D1 增加 36 个 interrupt，其中 23 个是新增 TP、13 个是新增 FP；没有相反方向的净 confusion 变化。因此 interrupt F1 小幅上升，而 silent F1 略降。

Non-first chunk Macro 从 `0.60454` 升至 `0.60584`，通过了该单项检查，但增益同样很小。

## 7. Fold 与训练行为

| Test fold | Best epoch | Epochs run | Macro delta vs D1 |
|---:|---:|---:|---:|
| 0 | 0 | 12 | `0.000000` |
| 1 | 5 | 17 | `+0.002593` |
| 2 | 7 | 19 | `+0.001385` |
| 3 | 10 | 22 | `+0.001018` |
| 4 | 0 | 12 | `0.000000` |

fold 0 和 4 的最佳 epoch 为 0，即 calibration BCE 判断任何 residual 更新都不如原 D1；这说明 early stopping 正常工作，并非五折都被迫使用一个过拟合 MLP。其余三折只接受 5--10 个 epoch，增益均低于 `0.003`。

严格正增益 fold 数为 3/5，未达到预设 4/5 门槛。

## 8. Domain 结果

| Domain | D1 Macro | D2 Macro | Delta |
|---|---:|---:|---:|
| Arts and Crafts | `0.62943` | `0.62792` | `-0.00151` |
| Chef | `0.65019` | `0.64938` | `-0.00082` |
| Handyman | `0.63516` | `0.64143` | `+0.00628` |
| Tutorial | `0.61422` | `0.61502` | `+0.00080` |

收益主要来自 Handyman，另外两个 domain 略降，不能解释为跨域稳定的非线性增益。

## 9. Bootstrap 与最终判定

相对 D1 的 5,000 次 paired session bootstrap：

```text
median = +0.000990
95% interval = [-0.001130, +0.003143]
positive fraction = 0.8168
```

区间跨过 0，且上界仍低于预设 `+0.005` 最小有意义增益。D2 只通过“non-first gain”和“两类非坍塌”，未通过最小增益、bootstrap 下界和 positive-fold-count 三项。

因此：

- 不对 D2 做全量 700-session final refit；
- 不生成新的 leaderboard submission；
- 不用后验 width、学习率、patience 或多 seed 搜索追逐这 `+0.001`；
- 当前科学基线和部署候选继续保持 D1 fused linear。

D2 仍相对 scalar OOF `0.6119` 保持明确优势，bootstrap 95% delta 为 `[+0.0131,+0.0334]`；失败的是“是否优于 D1”，不是整个融合路线。

## 10. 路线含义

结果表明，在冻结的最终 prompt hidden 上继续增加一个小型通用非线性 head，边际价值很低。下一步若继续提高决策表示，应该把训练信号送回 InternVL 的最后语言层或 multimodal fusion 表示，而不是继续堆更多离线 MLP。

建议下一实验是受控的 final-layer LoRA / joint binary decision loss 可行性与严格 OOF 设计；它必须保持话语生成与 outward gate 分离，并以 D1 `0.6341` 为基线。更大规模 oracle state replication 仍排在该表示适配实验之后，granularity 和 GRPO 继续不启动。

## 11. 产物与哈希

正式实验目录：

```text
output/experiments/20260715_internvl35_1b_residual_mlp_d2_oof_v1
```

关键哈希：

```text
predictions.jsonl  0cc28e8fdc9764159cea98982c613e8477dde2e3ed1bf3d8b462df2be4e05639
metrics.json        f2741b5c322b32feaf6479b557e01541b172ea22d943ba785baa3ec0dd3b2e2a
diagnostics.json    2986fb8c9e449e26319407f215778f59373378c72083e0a753b92b89ad797c81
comparison.json     76502fe8ba020fcc90ba877637f9eb7bc0bfe3be66c3e2e75cf94cf40b90eb1d
```

正式运行耗时 88.01 秒，仅使用 CPU，没有重新运行视频模型或占用 GPU。

另以 `/tmp/d2_residual_mlp_repro_v1` 独立重跑完整五折流程；`predictions.jsonl`、`metrics.json` 和 `comparison.json` 均与正式产物逐字节一致，以上三个 SHA256 全部复现。
