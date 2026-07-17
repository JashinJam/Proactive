# C1 Small D1 融合推理等价加速报告

> 主候选：`shared_vision`  
> 状态：**通过等价性、延迟和显存门槛，推广为当前部署路径**  
> 参考路径：`sequential`  
> 否决控制：`batched`、`prefix_cache`  
> 结果分类：推理工程等价性与性能验证，不改变模型、决策头或 OOF 分数

## 1. 结论

D1 融合头原始在线实现对每个 chunk 执行：

1. 一次 R0 自由生成；
2. 一次 `$silent$` 完整多模态候选前向；
3. 一次 `$interrupt$` 完整多模态候选前向。

本轮比较三种优化后，只有 `shared_vision` 同时满足正确性和性能要求。它只计算一次视频编码与多模态投影，然后保留原来的两个 batch=1 语言候选前向，因此不改变 frozen head 所见的 hidden、tag margin 或输出决策。

在四个 domain 各选择一个最短和最长 session 的扩展基准上，共 8 sessions / 127 chunks：

| 项目 | Sequential | Shared vision | 变化 |
|---|---:|---:|---:|
| Session 计算时间总和 | 492.47 s | 446.61 s | **-9.31%** |
| 完整墙钟 | 500.892 s | 455.056 s | **-9.15%** |
| 峰值显存 | 3,466,463,232 B | 3,466,463,232 B | **0%** |
| Hidden 完全一致 | 127/127 | 127/127 | 无变化 |
| Tag margin 完全一致 | 127/127 | 127/127 | 无变化 |
| Decision / answer 一致 | 127/127 | 127/127 | 无变化 |
| Prediction / metrics 文件 | - | 逐字节一致 | 无变化 |

它超过运行前冻结的 5% session 计算时间改善门槛，且没有增加峰值显存，因此正式推广。`batched` 虽然精确等价但更慢、更占显存；`prefix_cache` 同时发生 tag margin 漂移且没有提速，两者均否决。

## 2. 不变项

本轮不改变：

- InternVL3.5-1B-HF 权重与 revision；
- 1,044 参数 `fused_linear` 决策头；
- 18 个严格因果标量、1 个 tag margin、1,024 维 hidden；
- prompt、帧数、历史轮次、decoding 和随机种子；
- 最终 threshold 与响应修复逻辑；
- 官方预测 JSONL schema 和 scorer；
- OOF Macro F1 `0.6341`。

本轮只优化推理中的重复计算。任何候选只要 hidden、margin、logit 或决策超过预注册容差，即使当前小样本最终标签没变，也不得推广。

## 3. 参考路径

`sequential` 是冻结 correctness oracle：

```text
R0 generation:       vision + prompt + autoregressive decode
silent candidate:    vision + prompt + $silent$
interrupt candidate: vision + prompt + $interrupt$
```

正式神经缓存由该路径生成。新的 sequential 8-session 基准再次验证：

- raw response 127/127 完全一致；
- hidden 127/127 与正式缓存逐元素一致；
- tag margin 127/127 完全一致；
- decision 和 answer 127/127 一致；
- 最大 logit 差 `2.3187e-7`，来自 JSON float/线性打分重建，不改变决策。

## 4. 候选一：双候选 batch

### 4.1 设计

把 `$silent$` 和 `$interrupt$` 组成 batch=2，在一次模型调用中评分。直觉上从两次调用降到一次，但视觉塔仍要处理两份重复视频，语言模型也同时处理两个约 4k--8.8k token 的长序列。

### 4.2 10-chunk 结果

| 项目 | Sequential | Batched | 变化 |
|---|---:|---:|---:|
| Session 计算时间 | 35.05 s | 41.40 s | **+18.12%** |
| 完整墙钟 | 43.258 s | 50.772 s | **+17.37%** |
| 峰值显存 | 3.466 GB | 4.145 GB | **+19.59%** |
| Hidden / margin | 完全一致 | 完全一致 | 无漂移 |
| Decision / answer | 10/10 | 10/10 | 无变化 |

结论：数值等价，但 GPU batch=2 的长序列与重复视觉计算增加了时间和显存，否决。

## 5. 候选二：裁剪 prefix cache

### 5.1 设计

先用完整 `$silent$` 候选做一次带 KV cache 的 prefill，取到与参考路径相同的 causal hidden 和 silent score；随后把 cache 裁到标签前，只计算很短的 interrupt 标签后缀。

### 5.2 10-chunk 结果

| 项目 | 结果 |
|---|---:|
| Hidden 最大差 | 0 |
| Tag margin 最大差 | **0.113382** |
| Logit 最大差 | **3.7451e-4** |
| Decision / answer | 10/10 相同 |
| Session 计算时间 | 35.83 s，较参考慢 2.23% |
| 完整墙钟 | 45.864 s，较参考慢 6.02% |
| 峰值显存变化 | +0.01% |

silent score 完全一致，漂移全部来自用 cache 计算的 interrupt 后缀。其原因是 SDPA 在“完整长序列前向”和“长 cache + 短 query”两种张量形状下采用不同数值路径；虽然当前 10 个最终决策未跨 threshold，但该 margin 已不是训练头对应的正式特征。

结论：不满足 margin=`1e-5`、logit=`1e-6` 和至少 10% 提速门槛，否决。不能因为小样本标签恰好不变而把它称为等价优化。

## 6. 候选三：共享视觉特征

### 6.1 设计

`shared_vision` 严格复现 `InternVLModel.forward` 中的内部步骤：

```text
当前视频帧
   |
   +--> vision tower + projector，仅一次
                      |
                      +--> 注入 silent input embeddings
                      |       +--> 原 batch=1 language forward
                      |
                      +--> 注入 interrupt input embeddings
                              +--> 原 batch=1 language forward
```

两个候选的 prompt 完全相同且标签 token 数相同，因此 processor 只需构造一次 silent 输入；interrupt 输入仅替换最后的标签 token。共享的是确定性的 projected image features，两个语言前向的 batch size、序列长度、attention mask 和位置均与 sequential 参考保持一致。

### 6.2 首个 10-chunk smoke

| 项目 | Sequential | Shared vision | 变化 |
|---|---:|---:|---:|
| Session 计算时间 | 35.05 s | 28.15 s | **-19.69%** |
| 完整墙钟 | 43.258 s | 36.377 s | **-15.91%** |
| 峰值显存 | 3,466,037,248 B | 3,466,037,248 B | 0% |

10 个 hidden、margin、raw response、decision 和 answer 全部精确一致。

## 7. 扩展基准设计

为避免只根据一个 session 推广，从每个 domain 中仅按 `video_intervals` 数量选择最短和最长 session。选择不读取答案、R0 错误或模型输出，只用于工程压力测试；session 总长度没有进入任何模型特征。

| 原始索引 | Domain | Chunks | Video |
|---:|---|---:|---|
| 19 | Arts and Crafts | 24 | `0529003148c61a83.mp4` |
| 76 | Arts and Crafts | 4 | `1ad5cc12a5288383.mp4` |
| 314 | Chef | 6 | `7680dce9e6ce8546.mp4` |
| 356 | Tutorial | 26 | `822f4439932ecf6a.mp4` |
| 440 | Handyman | 6 | `9ffaadce7276fe77.mp4` |
| 584 | Handyman | 26 | `d7a644cd2893ebdb.mp4` |
| 609 | Chef | 30 | `e0c237d2d433e4ed.mp4` |
| 650 | Tutorial | 5 | `ed2b9c89c8343f30.mp4` |

两种模式在同一物理 A800 GPU 1 上串行运行；每次启动前该 GPU 只有 4 MiB 驱动占用，未与 GPU 4--7 的既有训练共享。

## 8. 扩展性能结果

| 索引 | Chunks | Sequential | Shared | 改善 |
|---:|---:|---:|---:|---:|
| 19 | 24 | 105.28 s | 94.11 s | +10.61% |
| 76 | 4 | 11.79 s | 12.68 s | -7.55% |
| 314 | 6 | 20.10 s | 21.47 s | -6.82% |
| 356 | 26 | 96.24 s | 81.37 s | +15.45% |
| 440 | 6 | 21.18 s | 19.65 s | +7.22% |
| 584 | 26 | 93.39 s | 88.76 s | +4.96% |
| 609 | 30 | 129.40 s | 115.06 s | +11.08% |
| 650 | 5 | 15.09 s | 13.51 s | +10.47% |

8 个 session 中 6 个更快。两个极短 session 出现 6%--8% 波动，说明 processor、视频解码和调用固定开销会掩盖少量视觉复用收益；24--30 chunk 的长 session 全部更快，且是全量运行的主要成本来源。

聚合结果：

- session 计算时间 `492.47 -> 446.61s`，改善 `9.312%`；
- 完整墙钟 `500.892 -> 455.056s`，改善 `9.151%`；
- 峰值显存完全相同；
- 通过运行前冻结的至少 5% 计算时间改善门槛。

## 9. 扩展等价性结果

在全部 127 chunks 上：

- raw response 127/127 完全一致；
- 1,024 维 hidden 127/127 逐元素完全一致，最大绝对差 `0`；
- tag margin 127/127 完全一致，最大绝对差 `0`；
- decision 127/127 完全一致；
- answer 127/127 完全一致；
- 最大 logit 差 `2.3187e-7 < 1e-6`；
- predictions JSONL 逐字节一致；
- 官方 metrics JSON 逐字节一致。

这意味着 `shared_vision` 不需要重新训练或重新校准头，也不产生新的模型分数；它是冻结 `0.6341` 策略的等价执行实现。

## 10. 推广决定与部署边界

当前部署配置切换为：

```text
configs/d1_internvl35_1b_neural_deploy_shared_vision.json
```

`configs/d1_internvl35_1b_neural_deploy.json` 继续保留为 sequential correctness oracle。`batched` 和 `prefix_cache` 配置及产物保留用于防止以后重复尝试，但不得作为生产路径。

正式推理不要添加 `--record-hidden-state`。该开关只为小规模等价审计保存 1,024 维向量；关闭它不改变计算，只避免完整测试运行产生大量 JSON。

`shared_vision` 直接调用固定 Transformers 版本中的 `InternVLModel.get_image_features`、`get_placeholder_mask` 和 `language_model`。因此 submission/container 必须固定并审计 Transformers 版本；升级库后需重新运行至少本报告的 127-chunk 等价测试。

## 11. 下一步

等价推理优化已完成。下一步进入 submission/container 审计：

1. 核对官方提交物究竟是预测 JSONL、模型目录还是可执行 container；
2. 固定 Python、PyTorch、Transformers、CUDA 和模型文件；
3. 把 shared-vision runner 接入官方测试输入入口，同时保持 scorer 不变；
4. 审计 Small 参数统计、离线权重加载、路径和无网络运行；
5. 在用户明确授权前不执行外部上传或消耗排行榜提交次数。

## 12. 关键产物

| 对象 | SHA256 |
|---|---|
| Shared-vision 配置 | `035cf6f033dbe378d14363c2537d33002c5ed5ec9beaf1e449010a0bdfb69ee8` |
| 127-chunk predictions | `83422f9b2c20a40002168b3960855c1c9b24065618b306676f5034f6dbdbdcd4` |
| 127-chunk metrics | `841f8e146d4e8de1f3e9d35d2f0c9c7158a5352944499c45c523f3b3feb0f57c` |
| Shared consistency audit | `cf6e89ee0e84b907d7ca83d60bea3788202e54544b8575ef1bee29acb27951d3` |
| Promote comparison | `051c77679ede37bf022cca54f9dc32faa3347988764ffc9d2d32708b6fbb0127` |
| Batched reject comparison | `1789c9511a768900822f90563cd4a7f22dba62eb8fe610f381c038c5bc92b704` |
| Prefix-cache reject comparison | `dd9a9ea7b52f513c575e01b31904579d84d75547adaa50b2b415afbc61f22295` |

产物目录：

- Sequential 扩展参考：[`output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_sequential_v2_benchmark8/`](../output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_sequential_v2_benchmark8/)
- Shared-vision 扩展候选：[`output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_shared_vision_v1_benchmark8/`](../output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_shared_vision_v1_benchmark8/)
- Batched 否决控制：[`output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_batched_v1_smoke1/`](../output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_batched_v1_smoke1/)
- Prefix-cache 否决控制：[`output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_prefix_cache_v1_smoke1/`](../output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_prefix_cache_v1_smoke1/)
