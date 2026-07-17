# C1 Small D1 神经融合决策头完整实验报告

> 主实验 ID：`20260714_internvl35_1b_neural_decision_head_d1_oof_v1`  
> 后验复核 ID：`20260715_internvl35_1b_neural_decision_head_d1_oof_l2_followup_v1`  
> 最终头 ID：`20260715_internvl35_1b_neural_decision_head_d1_final_v1`  
> 在线烟测 ID：`20260715_internvl35_1b_neural_decision_head_d1_deploy_v1_smoke1`  
> 状态：神经特征、五折 OOF、正则复核、最终头和在线烟测均已完成  
> 证据分类：**公开验证集监督的 session-level OOF 开发结果，不是隐藏测试集证据**

## 1. 核心结论

D1 神经增量实验得到一个可复现的正结果：在完全相同的 700 个 session、9,935 个 chunk 和五折 session-level OOF 协议下，融合严格因果标量、候选标签分数和冻结多模态隐状态的 `fused_linear` 达到官方 Macro F1 `0.6341`。

它相对已推广的标量头 `0.6119` 提升 `+0.0222`，按 session 配对的 5,000 次 bootstrap 增益中位数为 `+0.0223`，95% 区间为 `[+0.0123, +0.0322]`，增益为正的比例为 `1.0`。五个测试折和四个 domain 全部为正增益，非首 chunk Macro 从 `0.5843` 提升到 `0.6045`，因此这不是首 chunk 规则或单一折造成的偶然结果。

但消融同时表明：

- 标签 margin 单独只有 `0.5313`，原始零阈值会在 `99.15%` 的 chunk 上预测 interrupt，不能直接当 gate；
- 1,024 维 hidden 单独为 `0.6031`，低于标量控制；
- 标量加 margin 为 `0.6172`，有小增益但 bootstrap 下界略低于零；
- 只有 `标量 + margin + hidden` 的融合头稳定超过强标量控制。

因此，当前证据支持的是“冻结隐状态与因果标量具有互补信息”，而不是“hidden 本身已经学会了 C1 决策”。它通过了 D1 神经增量门槛，使 D2 的轻量训练成为可选项，但尚未证明程序状态或粒度建模会进一步有效。

## 2. 为什么做这项实验

此前的有效基线依次为：

| 系统 | 官方 Macro F1 | 主要含义 |
|---|---:|---|
| R0 原始生成 | 0.4630 | 冻结 InternVL3.5-1B 的无计划基线 |
| R0-F 格式修复 | 0.5362 | 修复 malformed 非空响应的决策解释 |
| D1 标量 OOF | 0.6119 | 学习公开集的严格因果时序/响应策略 |

D1 标量头的提升很大，但主要来自时序和标注策略。下一步必须回答：同一冻结 backbone 的当前视觉语言表示，是否还能在强标量控制上提供净增量。若不能，就不应该直接启动 MLP、LoRA、planner 或 GRPO；若能，才有理由把训练资源用于表征与状态接口。

本实验把自由文本生成与二元决策分开：

```text
当前可见视频帧 + query + 已发生 dialog
                  |
                  +--> 冻结 R0 生成 --> 18 个严格因果标量
                  |
                  +--> 固定候选评分 --> tag margin + 1024 维 causal hidden
                                      |
                                      v
                             1044 参数线性 gate
                              /             \
                         $silent$       $interrupt$ + R0 utterance
```

## 3. 数据、模型和 Small 边界

| 项目 | 设置 |
|---|---|
| 数据 | 官方 EgoProactive public validation |
| 规模 | 700 sessions / 9,935 chunks |
| 数据 SHA256 | `feef69ddee605e7070ad0f133636c35739c6964514a46d76da294b6bf1964740` |
| 数据许可 | `CC-BY-NC-4.0` |
| Backbone | `OpenGVLab/InternVL3_5-1B-HF` |
| Revision | `9191dbccf312b537016f041b25d61c72e7c5c9f3` |
| Backbone 参数 | 1,060,897,792 |
| Backbone 权重 SHA256 | `11effd1da2fc0929957d56de4129d1e7d2aed044ea878d40bf83e95b63d38b39` |
| 最终融合头参数 | 1,044 |
| 部署总参数 | 1,060,898,836，低于 Small 的 2B 上限 |
| 外部训练数据 | 无 |
| STRIDE | 未使用 |

模型保持 BF16、SDPA、确定性 decoding。每个 interval 提取 16 帧，历史累计帧按 R0 的均匀策略最多保留 32 帧；最多保留 4 个已经可见的历史对话轮次。所有当前 chunk 的特征都在当前时刻可获得，不使用 session 总长度、最终视频时长、未来 interval、未来 dialog 或未来帧。

## 4. 神经特征设计

### 4.1 固定标签候选分数

对同一个冻结因果 prompt，分别拼接：

```text
$silent$
$interrupt$
```

代码先验证两段文本能被 tokenizer 完整往返，且 token 数相同。随后对每个候选执行 teacher-forced 前向，累计标签各 token 的自回归 log probability：

```text
tag_margin = log P($interrupt$ | causal prompt)
           - log P($silent$ | causal prompt)
```

它不是把模型的自由生成 tag 直接当决策，而是保留连续排序信息，之后由 calibration fold 学习阈值。

### 4.2 最终因果多模态 hidden

从语言模型最后一层取“候选标签开始前的最后一个 prompt token”对应的 1,024 维 hidden。这个位置只依赖候选标签之前的 causal prompt，不应随拼接的 `$silent$` 或 `$interrupt$` 改变。

工程上同时从两个候选前向中取该 hidden，并逐 chunk 校验：

- 全 9,935 chunk 的最大绝对差为 `0.0`；
- 最小 cosine similarity 为 `0.9999997616`。

这证明缓存的 hidden 是候选无关的 prompt-prefix 表示，而不是把待比较标签泄漏进特征。

### 4.3 18 个标量特征

沿用严格因果 `response_temporal`：首 chunk、chunk 编号、已观察结束时间、当前 interval 时长、与上一 interval 的 gap、已见对话比例、模型输入帧比例、四个 domain one-hot，以及 R0/R0-F 决策和当前 raw response 的合法 tag、malformed、空响应与长度属性。

### 4.4 四个消融变体

| 变体 | 输入维度 | 每折头参数 | 用途 |
|---|---:|---:|---|
| `tag_only` | 1 | 2 | 检查候选标签 margin 的独立排序能力 |
| `scalar_tag` | 19 | 20 | 检查 margin 对强标量控制的增量 |
| `hidden_linear` | 1,024 | 1,025 | 检查冻结 hidden 的线性可分性 |
| `fused_linear` | 1,043 | 1,044 | 检查标量、margin 与 hidden 的互补性 |

## 5. 标签隔离和缓存审计

神经特征提取的输入先删除 `answers`。提取器只读取视频、interval、query、domain 和当前已发生 dialog；gold 只在后续 OOF 头训练和评估阶段重新挂接。

合并缓存：

| 项目 | 结果 |
|---|---:|
| Hidden shape | `[9935, 1024]` |
| Hidden dtype | `float32` |
| Session / chunk | 700 / 9,935 |
| Prompt token 范围 | 440--8,835 |
| Tag margin 范围 | -3.5449--15.3449 |
| Tag margin 均值 | 7.4618 |
| 缓存包含标签 | 否 |
| `features.npz` SHA256 | `cc5a5b3c6184987edc5f041eb2cb01a51ccfd88d6328d8308fcce3a4bd9122bf` |
| `records.jsonl` SHA256 | `f25e4a7c64471a044acffed9d1c577131237b3005042879959fa0d42cab23e0c` |

## 6. 四卡特征提取

700 个 session 按连续区间分为四个互不重叠的 175-session shard。每个进程只绑定一张空闲 GPU；启动时均没有既存 GPU 进程。

| Shard | Sessions | Chunks | 用时（秒） | 峰值显存 |
|---:|---:|---:|---:|---:|
| 0 | 175 | 2,596 | 8,073.234 | 3,138,302,976 B |
| 1 | 175 | 2,481 | 7,614.322 | 3,138,247,680 B |
| 2 | 175 | 2,405 | 7,269.900 | 3,138,278,400 B |
| 3 | 175 | 2,453 | 7,730.787 | 3,138,336,768 B |

四个 shard 全部完整，无失败 session。并行墙钟时间由最慢 shard 决定，约 2.24 小时。正式 OOF 头只读取合并缓存，在 CPU 上完成，没有重跑模型推理。

## 7. 五折 OOF 协议

完全复用标量 D1 已冻结的 `domain_stratified_sha256_round_robin` split，manifest SHA256 为 `bd537e9e155586cf3af9f26052fda277fa3e1930e378538346cc197432ff86c0`。同一 session 的所有 chunk 永远在同一折。

测试折为 `f` 时：

```text
test        = f
calibration = (f + 1) mod 5
fit         = 其余三个 fold
```

训练设置：

- class-balanced linear logistic regression；
- PyTorch `float64`；
- LBFGS，最多 120 次迭代；
- 低维头 L2=`0.001`，reduction=`mean`；
- 高维头 L2 网格=`[1e-5, 1e-4, 1e-3, 1e-2]`，reduction=`sum`；
- L2 和 threshold 只由 calibration fold 的 Macro F1 选择；
- test fold 标签在预测冻结后才参与评分；
- bootstrap 以 session 为抽样单位，共 5,000 次。

全部折来自公开验证集，因此 OOF 能减少同 session 泄漏，但不能等价于隐藏测试集。

## 8. 正式 OOF 结果

| 变体 | Macro | Interrupt F1 | Silent F1 | Interrupt 比例 | 相对标量 | 95% bootstrap 区间 |
|---|---:|---:|---:|---:|---:|---:|
| 标量参考 | 0.6119 | 0.6366 | 0.5873 | 0.5248 | - | - |
| `tag_only` | 0.5313 | 0.4983 | 0.5644 | 0.3908 | -0.0806 | [-0.0914, -0.0697] |
| `scalar_tag` | 0.6172 | 0.6341 | 0.6003 | 0.5055 | +0.0053 | [-0.0002, +0.0107] |
| `hidden_linear` | 0.6031 | 0.6280 | 0.5782 | 0.5240 | -0.0088 | [-0.0194, +0.0015] |
| `fused_linear` | **0.6341** | **0.6352** | **0.6330** | **0.4643** | **+0.0222** | **[+0.0123, +0.0322]** |

最佳融合头的完整官方结果：

| 指标 | 数值 |
|---|---:|
| Macro F1 | **0.6341** |
| G-mean F1 | 0.6341 |
| Interrupt precision / recall / F1 | 0.6861 / 0.5914 / 0.6352 |
| Silent precision / recall / F1 | 0.5891 / 0.6840 / 0.6330 |
| TP / FP / TN / FN | 3,165 / 1,448 / 3,135 / 2,187 |

独立再次调用官方 scorer 后，metrics 与正式文件逐字节一致，SHA256 为 `371ef8b347a683d27b45bf0eb90e46138dd22afe35724deabb9418fc69439e01`。

## 9. 稳定性分析

### 9.1 五个测试折

融合头相对标量的 Macro 增益为：

| Fold | 标量 | 融合 | 增益 |
|---:|---:|---:|---:|
| 0 | 0.6046 | 0.6354 | +0.0308 |
| 1 | 0.6143 | 0.6493 | +0.0350 |
| 2 | 0.6052 | 0.6169 | +0.0117 |
| 3 | 0.6185 | 0.6305 | +0.0119 |
| 4 | 0.6139 | 0.6347 | +0.0208 |

没有负增益测试折。

### 9.2 四个 domain

| Domain | 标量 | 融合 | 增益 |
|---|---:|---:|---:|
| Arts and Crafts | 0.6129 | 0.6294 | +0.0165 |
| Chef | 0.6037 | 0.6502 | +0.0464 |
| Handyman | 0.6210 | 0.6352 | +0.0142 |
| Tutorial | 0.6021 | 0.6142 | +0.0121 |

Chef 的增益最大，但其余三域也全部为正。

### 9.3 Chunk 位置

| 位置 | 标量 | 融合 | 增益 |
|---|---:|---:|---:|
| 首 chunk | 0.4996 | 0.4993 | -0.0004 |
| 第 2 个 chunk | 0.4363 | 0.5685 | +0.1323 |
| chunk 2--4 | 0.5701 | 0.6136 | +0.0434 |
| chunk 5--9 | 0.6140 | 0.6128 | -0.0012 |
| chunk 10+ | 0.5494 | 0.5769 | +0.0275 |
| 全部非首 chunk | 0.5843 | **0.6045** | **+0.0203** |

收益集中在第二个、早期和较晚 chunk；首 chunk 没有收益，5--9 区间有极小退化。后续残差建模应优先分析这两个弱区间，而不是继续强化首 chunk 先验。

### 9.4 相对标量头的决策变化

- 修复 707 个标量 FN；
- 修复 969 个标量 FP；
- 新引入 905 个 FN；
- 新引入 566 个 FP；
- 保持正确 4,624 个；
- 保持错误 2,164 个。

融合头净减少 205 个错误。其主要作用是将标量头过多的 interrupt 收回一部分，提高 silent F1，同时保留相近的 interrupt F1。

## 10. Tag margin 的失败含义

原始 margin 以零为阈值时预测 interrupt 比例为 `0.9915`，Macro 仅 `0.3490`，silent F1 约 `0.0039`。其 ROC AUC 为 `0.4451`；gold interrupt 的平均 margin 为 `7.1066`，gold silent 反而为 `7.8766`。

这解释了此前 grammar-constrained smoke 为什么接近全 interrupt：InternVL 对两个候选的绝对语言偏好严重偏向 `$interrupt$`，且方向不等于任务标签方向。margin 仍能在与标量和 hidden 联合标准化后提供一点互补信号，但不能单独作为 C1 gate。

## 11. 后验扩展 L2 复核

首轮融合头五折中有四折选到原网格上界 `0.01`。为排除搜索上界截断，另建配置，把高维 L2 网格扩展为：

```text
[1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
```

这项实验在看到首轮结果后才设计，因此被明确标记为后验二阶段公开集调参，不作为独立确认性证据。

| 变体 | 首轮 | 扩展 L2 | 变化 |
|---|---:|---:|---:|
| `hidden_linear` | 0.6031 | 0.6041 | +0.0010，仍低于标量 |
| `fused_linear` | **0.6341** | 0.6336 | -0.0005 |

扩展网格没有提升最优结果，因此最终头保留干净首轮，而不是选择后验复核。扩展实验的融合 bootstrap 区间仍为正 `[+0.0116, +0.0319]`，说明正信号稳定，但不改变正式最优值。

## 12. 最终单一融合头

OOF 文件由五个不同头组成，不能直接部署。最终头使用首轮五折冻结结果：

- 五折 L2：`[0.01, 0.01, 0.001, 0.01, 0.01]`；
- 最终 L2：中位数 `0.01`；
- 五折 threshold：`[-0.11124065, 0.09580707, 0.36841904, 0.12560538, 0.12780026]`；
- 最终 threshold：中位数 `0.1256053793821626`；
- 不根据全量 train-fit 预测重新选择 L2、特征或 threshold；
- 在全部 700 session / 9,935 chunk 上拟合一个 1,044 参数头。

最终头的 train-fit 官方 Macro 为 `0.6719`，interrupt/silent F1 为 `0.6416/0.7022`，TP/FP/TN/FN 为 `2,893/773/3,810/2,459`，预测 interrupt 比例为 `0.3690`。它只用于验证全量拟合、序列化和重载闭环，不替代 OOF `0.6341`。

独立第二次重拟合得到的头、预测和官方 metrics 均逐字节一致：

| 产物 | SHA256 |
|---|---|
| `decision_head.json` | `b8377d0abf7975f0fcb82dca8f374f34d8cecbb8f45477342f07df609f7312d2` |
| Train-fit predictions | `5ddeb4badc6ace6b65b0a981063d627640d99a71bed3e40062ed6689574ac714` |
| Train-fit metrics | `1a9370a9d36b0aafae0d3020afe6952a59237b4e3985aad91a6d06956c2a1c62` |

## 13. 在线部署验证

现有 `proactive_d1.run_deploy` 已扩展为同时支持：

- `response_temporal` 标量头；
- `fused_linear` 融合头。

融合参考实现每个 chunk 执行三部分计算：

1. 一次确定性 R0 generation，得到 raw response 和 18 个标量；
2. 一次 `$silent$` 固定候选前向；
3. 一次 `$interrupt$` 固定候选前向。

首个 10-chunk session 在空闲物理 GPU 0 上完成：

| 项目 | 结果 |
|---|---:|
| 墙钟时间 | 43.258 秒 |
| 峰值显存 | 3,466,037,248 B |
| 启动前 GPU 进程 | 0 |
| 总参数 | 1,060,898,836 |
| 单 session Macro | 0.7917，仅为烟测，不是模型估计 |

与冻结离线产物逐 chunk 对齐：

| 检查 | 结果 |
|---|---:|
| R0 raw response | 10/10 完全一致 |
| Tag margin | 10/10 完全一致，最大差 0 |
| 最终 decision | 10/10 完全一致 |
| 最终 answer | 10/10 完全一致 |
| Logit 最大绝对差 | `9.03e-8`，低于 `1e-6` 容差 |

机器可读审计位于 `consistency_audit.json`，SHA256 为 `fce1b798dec121b79617a69e285028dbd74cea6bd5179c316c702a8247f12120`。

上述三次前向路径继续作为 correctness oracle。后续等价加速已经完成，见 13.2。

### 13.1 工程测试与官方 starter 状态

- R0 单测 `12/12`；
- R0-F 单测 `3/3`；
- R1 单测 `13/13`；
- D1 单测 `15/15`；
- 官方 proactive scorer 单测 `27/27`；
- 全部主动模块通过 `compileall`。

官方 `model.py`、`run_generate_proactive.py` 和 `run_evaluation.py` 的 SHA256 仍分别为 `3cecb5...`、`7c8adf...`、`072301...`，未被本项目修改。官方 generation 测试自身存在一个版本不一致：测试把 `setup_gpus` mock 成仅接收 keyword 的零位置参数 lambda，而当前官方 runner 调用 `setup_gpus(args.num_gpus, args.model_type)` 两个位置参数，因此 17 项中 5 项通过、12 项在进入被测逻辑前报同一个 `TypeError`。这不是融合 runner 的失败，也不应通过修改官方 starter 来掩盖；项目自己的因果消息、帧累计与在线融合路径由上述本地测试和实际 GPU 烟测覆盖。

### 13.2 Shared-vision 等价加速更新

后续对双候选 batch、裁剪 prefix cache 和共享视觉特征三种方案进行了正式对照。batch 路径虽然精确等价，但 session 计算时间增加 18.12%、显存增加 19.59%；prefix cache 的 tag margin 最大漂移 0.1134 且没有提速，均被否决。

`shared_vision` 只计算一次视频编码与投影，同时保留两个原 batch=1 语言候选前向。在四个 domain 各一短一长的 8 sessions / 127 chunks 上，hidden、margin、decision 和 answer 全部逐元素一致，predictions 与 metrics 逐字节一致；完整墙钟 `500.892 -> 455.056s`，改善 `9.15%`，峰值显存不变。它已推广为当前部署配置：`configs/d1_internvl35_1b_neural_deploy_shared_vision.json`。完整证据见 [融合推理等价加速报告](20260715_internvl35_1b_d1_inference_optimization.md)。

## 14. 证据边界与风险

1. 所有头训练、L2 选择和 threshold 选择都使用了 public validation 标签；`0.6341` 是 session-held-out OOF 开发证据，不是隐藏测试分数。
2. 五折都来自同一公开数据分布，无法验证隐藏集的干预频率、领域比例和标注粒度是否变化。
3. 最终头的 `0.6719` 是 train-fit，不能与 OOF 或排行榜分数横向比较。
4. 单 session 的 `0.7917` 样本极小，只证明部署器能完成官方评分。
5. 线性 hidden-only 未超过标量，说明冻结表示的可用决策信号有限；融合正增益不能被表述为“已解决视觉理解”。
6. 当前没有使用程序状态、动作边界、STRIDE 或粒度标签，因此本实验不直接验证这些方向。
7. Shared-vision 直接使用固定 Transformers 版本的 InternVL 内部接口；升级依赖后必须重跑 127-chunk 等价审计。

## 15. 对后续路线的影响

D1 已完成从“强标量控制”到“稳定神经增量”再到“单一在线部署头”的闭环。推荐优先级：

1. 已完成：融合特征 shared-vision 等价推理加速；
2. 下一步在不消耗外部提交次数的前提下，检查最终 container、模型参数统计和测试输入接口；
3. 以融合 OOF 的残差为对象，定位 5--9 chunk、首 chunk 和各 domain 的 FN/FP；
4. D2 先尝试极小 MLP 或最后语言层 LoRA，并继续使用同一 session OOF；
5. 再把 oracle state 接到已经能学习的 gate 上做更大、预注册的 R1 复验；
6. 只有 state 确有增益后，才启动专门粒度实验；GRPO 继续等待明确残差和稳定 SFT 接口。

排行榜优先并不意味着应立即把所有复杂模块叠加。当前可提交候选已经从标量 `0.6119` 更新为融合 `0.6341`；下一项改动必须与它比较，而不是退回 R0/R0-F。

## 16. 复现命令

神经 OOF：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.run_neural \
  --config configs/d1_internvl35_1b_neural_oof.json
```

后验 L2 复核：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.run_neural \
  --config configs/d1_internvl35_1b_neural_oof_l2_followup.json
```

最终头：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.finalize_neural \
  --config configs/d1_internvl35_1b_neural_final.json
```

单 session 在线烟测：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.run_deploy \
  --config configs/d1_internvl35_1b_neural_deploy.json \
  --output-dir output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_v1_smoke1 \
  --device cuda:0 --max-sessions 1
```

## 17. 主要产物指纹

| 对象 | SHA256 |
|---|---|
| 合并神经特征 | `cc5a5b3c6184987edc5f041eb2cb01a51ccfd88d6328d8308fcce3a4bd9122bf` |
| 正式融合 OOF 预测 | `04183a4083d160662d5f91bff5432a7ca96595dd66b2b0b64f3b430799143ad9` |
| 正式融合 OOF metrics | `371ef8b347a683d27b45bf0eb90e46138dd22afe35724deabb9418fc69439e01` |
| 正式融合 diagnostics | `726071fe259be68e7a66a36b9a759839abe0ffbba1d95a3b262c39bc53d987ee` |
| 正式 comparison | `bff64f4feafe5881c37122873375e54ce7b99d5d424d016395f497d01c8eb4b1` |
| 正式 analysis | `38fdef6fe52e510644b8c9c6e9da94172b46d4833f5f9a379c66ac19107e9b35` |
| 最终决策头 | `b8377d0abf7975f0fcb82dca8f374f34d8cecbb8f45477342f07df609f7312d2` |
| 在线烟测预测 | `4f1429c24b28f2f70bfaf429657a809f85f1622d982a216c0074f945ce7e7031` |
| 在线一致性审计 | `fce1b798dec121b79617a69e285028dbd74cea6bd5179c316c702a8247f12120` |

## 18. 产物位置

- 特征缓存：[`output/features/20260714_internvl35_1b_d1_neural_feature_cache_v1/`](../output/features/20260714_internvl35_1b_d1_neural_feature_cache_v1/)
- 正式 OOF：[`output/experiments/20260714_internvl35_1b_neural_decision_head_d1_oof_v1/`](../output/experiments/20260714_internvl35_1b_neural_decision_head_d1_oof_v1/)
- L2 复核：[`output/experiments/20260715_internvl35_1b_neural_decision_head_d1_oof_l2_followup_v1/`](../output/experiments/20260715_internvl35_1b_neural_decision_head_d1_oof_l2_followup_v1/)
- 最终头：[`output/experiments/20260715_internvl35_1b_neural_decision_head_d1_final_v1/`](../output/experiments/20260715_internvl35_1b_neural_decision_head_d1_final_v1/)
- 在线烟测：[`output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_v1_smoke1/`](../output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_v1_smoke1/)
