# InternVL3.5-1B D1 单一部署阈值稳健性审计

> 实验 ID：`20260715_internvl35_1b_d1_threshold_robustness_v1`  
> 日期：2026-07-15  
> 任务：ECCV 2026 Wearable AI Challenge，EgoProactive Small  
> 结论：**通过预设部署阈值稳健性门槛**

## 1. 审计问题

D1 的正式 OOF 结果使用五个轮转模型，每个模型在独立 calibration fold 上选择自己的阈值，官方 Macro F1 为 `0.6341`。最终部署时只有一个在全部 700 个公开开发 session 上拟合的模型，因此只能携带一个阈值。当前 final head 使用五个 calibration 阈值的中位数：

```text
0.1256053793821626
```

本审计回答：把这个单一阈值应用到五个 OOF 模型的未见 test fold logits 时，性能是否发生足以影响路线判断的退化。

该问题只涉及现有决策头的部署稳健性。没有重新抽帧、没有运行 InternVL、没有使用 GPU，也没有改变任何预测话语。

## 2. 证据边界

这是 public-validation-supervised 的部署诊断，不是新的隐藏测试结果。五个原始阈值在 D1 报告中已经可见；本次门槛和统一阈值比较在运行审计前写入冻结配置，但不能把它包装成完全盲的模型选择实验。

统一阈值来自已经序列化的 final head，本审计没有根据 9,935 个 OOF 标签重新挑选一个更好的全局阈值。阈值扫描只用于敏感性解释，不用于回写 final head。

## 3. 复现协议

审计重新构造与 D1 完全相同的输入：

- 700 个完整 session、9,935 个 chunk；
- 原始 domain-stratified 五折 session manifest；
- 18 个严格因果 scalar 特征；
- 1 个 fixed-tag margin；
- 1,024 维冻结 causal hidden state；
- 共 1,043 个输入特征；
- class-balanced linear logistic regression；
- 原始 L2 grid、LBFGS、seed、fit/calibration/test 轮转不变。

每个 test fold 先重新运行全部四个 L2 候选，验证被选 L2 和 calibration 阈值与冻结 diagnostics 一致，再生成 test logits。只有在原始 OOF predictions 的 SHA256 逐字节复现后，才执行统一阈值比较。

原始 OOF 预测哈希成功复现：

```text
04183a4083d160662d5f91bff5432a7ca96595dd66b2b0b64f3b430799143ad9
```

## 4. 预设通过门槛

统一阈值必须同时满足：

| 检查 | 门槛 |
|---|---:|
| 整体 Macro F1 下降 | 不超过 `0.005` |
| 最差 fold Macro F1 下降 | 不超过 `0.020` |
| 最终阈值 `±0.05` 局部扰动的最差下降 | 不超过 `0.005` |
| session bootstrap 的 delta 2.5% 分位 | 不低于 `-0.010` |
| interrupt/silent F1 | 均不低于 `0.600` |

## 5. 主要结果

| 策略 | Macro F1 | Interrupt F1 | Silent F1 | TP / FP / TN / FN | Interrupt rate |
|---|---:|---:|---:|---:|---:|
| 原始 fold-specific 阈值 | `0.6341` | `0.6352` | `0.6330` | 3165 / 1448 / 3135 / 2187 | `46.43%` |
| 单一部署阈值 | `0.6330` | `0.6298` | `0.6361` | 3102 / 1396 / 3187 / 2250 | `45.27%` |

统一阈值的全精度 Macro delta 为：

```text
-0.001130725359119955
```

它让 63 个原本命中的 interrupt 变成 FN，同时减少 52 个 FP，表现为轻微向 silent 偏移。下降远小于预设 `0.005` 容忍线。

官方 scorer 对统一阈值给出 Macro F1 `0.6330`，与内部 confusion count 计算一致。

## 6. Fold 与不确定性

五个 calibration 阈值为：

| Test fold | Calibration 阈值 | 统一阈值相对 Macro delta |
|---:|---:|---:|
| 0 | `-0.111241` | `-0.003232` |
| 1 | `0.095807` | `-0.004242` |
| 2 | `0.368419` | `+0.002630` |
| 3 | `0.125605` | `0.000000` |
| 4 | `0.127800` | `+0.000501` |

阈值原始跨度达到 `0.479660`，但单一中位数在所有 fold 上的实际最差下降只有 `0.004242`。这说明 raw logit calibration 确实存在 fold 波动，但当前决策边界附近并不尖锐。

相对 fold-specific 策略的 5,000 次 paired session bootstrap：

```text
median = -0.001087
95% interval = [-0.004784, +0.002428]
positive fraction = 0.2694
```

区间跨过 0，不能声称统一阈值更优；但其负向尾部仍在预设部署容忍范围内。

## 7. 阈值敏感性

以 final threshold 为中心的 OOF 扫描结果：

| Offset | 阈值 | Macro F1 | 相对 fold-specific delta |
|---:|---:|---:|---:|
| `-0.20` | `-0.0744` | `0.6328` | `-0.0013` |
| `-0.10` | `0.0256` | `0.6344` | `+0.0003` |
| `-0.05` | `0.0756` | `0.6340` | `-0.0001` |
| `0.00` | `0.1256` | `0.6330` | `-0.0011` |
| `+0.05` | `0.1756` | `0.6331` | `-0.0011` |
| `+0.10` | `0.2256` | `0.6302` | `-0.0039` |
| `+0.20` | `0.3256` | `0.6236` | `-0.0105` |

`-0.10` 在这批 OOF logits 上略高于原始结果，但它是审计后可见的全量 OOF 最优方向，不能用于回写模型，否则会引入新的全量标签调参。保留既有中位数阈值是更可辩护的部署选择。

## 8. 结论与下一步

五项预设检查全部通过。D1 final head 的单一阈值不会造成足以推翻 `0.6341` OOF 路线判断的退化；更现实的部署预期应记为约 `0.6330` 的统一阈值 OOF 模拟，而 `0.6719` 仍只是全量公开集 train-fit sanity。

因此，下一步不再继续围绕阈值做后验搜索，转入 D2：在完全相同的 1,043 维冻结特征和 session folds 上，只增加一个预注册、低参数量的非线性 residual MLP，检验线性边界之外是否存在稳定增益。

## 9. 产物

完整实验目录：

```text
output/experiments/20260715_internvl35_1b_d1_threshold_robustness_v1
```

关键文件：

- `audit.json`：全精度指标、fold、bootstrap、阈值扫描与 gate；
- `oof_logits.jsonl`：逐 chunk OOF logit、两种阈值决策和 scalar 对照；
- `predictions.jsonl` / `metrics.json`：统一阈值完整预测与官方评分；
- `fold_calibrated_predictions.jsonl`：逐字节复现的原 D1 OOF 预测；
- `config.json`、`command.sh`、`environment.txt`、`code_state.txt`、`data_manifest.json`：复现信息。

统一阈值预测 SHA256：

```text
8ef2a89eeace2a7056a23bd0cba190f6eacc9fb6b1e7be68774ae323b89ad2cb
```
