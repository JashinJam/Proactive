# C1 Small D4.2 输入策略适配五折 OOF 实验报告

## 结论

D4.2 在全部 700 个 public-validation sessions、9,935 个 chunks 上完成了
policy-matched 五折 OOF。`history8=(32,16,8,64)` 达到 official Macro/G-mean
`0.6988/0.6988`，相对 exact baseline `0.6846` 提升 `+0.0142`；5/5 folds
和 4/4 domains 为正，5,000 次 paired-session bootstrap 的 95% interval 为
`[+0.008166,+0.020363]`。

该结果通过 D4.2 内部选择门槛，但候选来自 D4.1 后的 public-validation
选择，因此属于 `val-supervised` 机制证据，不是 hidden-test 或独立泛化证据。
全量拟合分数 `0.7469` 只表示 train-fit sanity。

## 配置和结果

| 候选 | `(max_frames, frames_per_interval, history, tokens)` | Macro | G-mean | 相对基线 |
|---|---|---:|---:|---:|
| `history8` | `(32,16,8,64)` | `0.6988` | `0.6988` | `+0.0142` |
| `frames16` | `(16,16,4,64)` | `0.6854` | `0.6854` | `+0.0008` |
| baseline | `(32,16,4,64)` | `0.6846` | `0.6846` | `0.0000` |
| `tokens16` | `(32,16,4,16)` | `0.6844` | `0.6843` | `-0.0002` |

`history8` 的 interrupt P/R/F1 为 `0.7468/0.6674/0.7049`，silent
P/R/F1 为 `0.6545/0.7358/0.6928`；TP/FP/TN/FN 为
`3572/1211/3372/1780`，预测 interrupt 比例为 `0.48142929`。

模型保持 InternVL3.5-1B、BF16、greedy、official prompt/dialog 和
`shared_vision` 不变。每个策略重新拟合 1,052 参数的标准化、类别平衡
float64 线性头；三折拟合、一折校准、一折测试，L2 网格为
`{1e-5,1e-4,1e-3,1e-2}`。最终 `history8` head 使用 L2 `0.01`、阈值
`0.12101525136349107`，SHA256 为
`dab9eaf100ea301055ab4d68856d406fb5927864bc96c71f2038688067b904c5`。

## 资源和产物

- 总模型推理时间为 `17363.5358s`，记录的最大 session 模型时间为
  `59.0680s`，峰值分配显存为 `3,120,309,248` bytes。
- 数据为 `facebook/wearable-ai` EgoProactive public val，输入 SHA256 为
  `feef69ddee605e7070ad0f133636c35739c6964514a46d76da294b6bf1964740`，
  许可证为 `CC-BY-NC-4.0`，未使用外部训练数据。
- 实验配置的规范化对象 SHA256 为
  `71b88e99482a9d80bfd401f34604c7df5ab34b0aea723919c33e6fbf8caee453`；
  official scorer SHA256 为
  `072301da6c65b3e30c7581920d178c6d5136305f2db26914df4785f47d809ee1`。
- 顶层 `metrics.json` 和 `predictions.jsonl` 分别是 winner 原始 OOF
  产物的逐字节副本，SHA256 为
  `d3b19c1ffd99ba26418f3081ac1eedae48d18d881411fa3b0ef011782184d3a1`
  和 `d154789b8f41583558878e93b9bb618643a5f64d1ad5b397d84cfd592e31c121`。

## 边界与后续门槛

当前 D4 配置、head 和 `submission/d4_small` 保持不变。`history8` 必须先
通过针对其长历史路径的单卡 GPU 等价 smoke，才能成为新的
leaderboard-engineering 基线；即使晋升，也不能据此声称 hidden-test 改进。
项目顶层源码许可证仍未确定，正式 prize submission 资格尚未闭合。
