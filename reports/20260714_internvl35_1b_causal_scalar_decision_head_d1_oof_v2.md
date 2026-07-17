# C1 Small D1 严格因果标量决策头实验报告

> 实验 ID：`20260714_internvl35_1b_causal_scalar_decision_head_d1_oof_v2`  
> 状态：五折 session-level OOF 标量控制已完成  
> 结果分类：**使用公开验证集监督；属于开发证据，不属于隐藏测试集泛化证据**  
> 实验产物：[`output/experiments/20260714_internvl35_1b_causal_scalar_decision_head_d1_oof_v2/`](../output/experiments/20260714_internvl35_1b_causal_scalar_decision_head_d1_oof_v2/)  
> 最佳 OOF 预测 SHA256：`fc34808d65abf452600bb22b3cd4b3d43ac4e7dceb345542573d10e3248de5ee`

## 1. 结论

D1 首先回答一个比“马上训练 planner 或 LoRA”更基础的问题：冻结的 R0 输出、当前 chunk 的严格因果时序信息和已经可见的对话/帧数量，是否足以支持一个更好的 interrupt/silent 决策边界。

答案是肯定的。在完整 700 个公开验证 session、9,935 个 chunk 上，五折 session-level OOF 的最佳 `response_temporal` 线性头得到官方 Macro F1 `0.6119`，相对 R0-F 的 `0.5362` 提升 `+0.0757`。按 session 配对的 5,000 次 bootstrap，增益的 95% 区间为 `[+0.0660, +0.0853]`，五个测试折和四个领域均为正增益。

但这项结果的正确解释是：**C1 的公开标注中存在很强、可被严格因果标量特征学习的干预时序/标注策略**。它没有证明模型获得了更强的视觉理解或程序状态跟踪能力。仅使用时序特征的 `temporal` 已达到 `0.6081`，加入 domain 和冻结 R0 响应属性后只再增加 `0.0038`，说明主要收益并不来自新的多模态语义表示。

此外，这是一份 OOF 开发评估：每个测试折由不同的拟合头和校准阈值产生预测。它不是一个已经训练完成、可以原样装入官方测试 container 的单一部署头，也不能把该 OOF 预测文件直接当成最终系统提交物。

## 2. 实验动机

R0-F 的官方结果是 Macro F1 `0.5362`，仍有 `3,181` 个 FN 和 `1,377` 个 FP。R1 四-session pilot 没有显示当前零样本文本状态注入的正增益，但样本量不足以否定状态价值，而且实验还受到 tag 格式行为影响。

因此 D1 将问题拆开：

```text
冻结 R0 因果推理结果
        +
当前时刻可获得的标量元数据
        |
        v
19 参数以内的线性 interrupt/silent 头
        |
        +--> silent: $silent$
        |
        +--> interrupt: 复用并修复 R0 utterance
```

若这个低容量控制已经显著提升，就必须先把“标注策略校准”与“视觉/状态理解增益”区分开；否则后续 LoRA、planner、粒度或 GRPO 实验会把简单的决策边界变化误认成表征学习。

## 3. 数据、模型与运行边界

| 项目 | 设置 |
|---|---|
| 数据 | 官方 C1 public validation，700 sessions / 9,935 chunks |
| 数据 SHA256 | `feef69ddee605e7070ad0f133636c35739c6964514a46d76da294b6bf1964740` |
| 冻结 backbone | `OpenGVLab/InternVL3_5-1B-HF` |
| Revision | `9191dbccf312b537016f041b25d61c72e7c5c9f3` |
| Backbone 参数 | 1,060,897,792 |
| 新增头参数 | 每折 8 / 12 / 19，最佳变体为 19 |
| 部署一个头时总参数 | 1,060,897,811，低于 Small 的 2B 限制 |
| 新模型推理 | 无；复用冻结 R0 的 raw response 与因果记录 |
| GPU | 未使用 |
| 训练实现 | PyTorch `float64` 线性 logistic regression |
| 优化器 | LBFGS，最多 120 次迭代，L2=`0.001` |
| 类别处理 | fit fold 上 class-balanced logistic loss |
| 阈值 | calibration fold 上精确最大化 Macro F1 |
| 官方评分器 | `data/starter_kit/run_evaluation.py` |

实验没有使用外部数据、外部模型或 STRIDE 数据。特征构建阶段不读取 gold；gold 只用于 fit fold 训练、calibration fold 阈值选择和冻结预测后的 test fold 评估。

## 4. 五折 OOF 协议

### 4.1 Fold 冻结

使用 `domain_stratified_sha256_round_robin`，seed 为 `d1-session-oof-v1`：

1. 在每个 domain 内仅依据 session 身份的 SHA256 排序；
2. round-robin 分到五折；
3. fold 分配不读取答案；
4. 同一 session 的所有 chunk 永远在同一折。

| Fold | Session 数 | Chunk 数 |
|---:|---:|---:|
| 0 | 141 | 2,010 |
| 1 | 141 | 1,922 |
| 2 | 140 | 1,963 |
| 3 | 139 | 1,965 |
| 4 | 139 | 2,075 |

每个 domain 在每折包含 34--36 个 session。split manifest 的 SHA256 为 `bd537e9e155586cf3af9f26052fda277fa3e1930e378538346cc197432ff86c0`。

### 4.2 每轮训练、校准和测试

当测试折为 `f` 时：

```text
test        = f
calibration = (f + 1) mod 5
fit         = 其余三个 fold
```

fit fold 学习标准化均值、尺度、线性权重和 bias；calibration fold 只选择 logit threshold；test fold 在预测冻结后才计算指标。旋转五轮后，每个 session 恰好获得一次 OOF 预测。

该协议防止同一 session 的相邻 chunk 同时进入训练和测试，也避免用测试折选择阈值。但全部五折仍来自 public validation，因此最终合并分数必须标记 `val-supervised`。

## 5. 严格因果特征

### 5.1 `temporal`：7 个特征

- 当前 chunk 是否为首 chunk；
- `log1p(当前 chunk 编号)`；
- `log1p(当前 interval 的结束时刻)`；
- `log1p(当前 interval 时长)`；
- `log1p(与上一已见 interval 的时间间隔)`；
- 已可见历史对话轮数占配置上限 4 的比例；
- 当前模型实际输入帧数占 R0 上限的比例。

这些值在当前推理时刻都已经确定，不需要知道未来 chunk 或视频最终长度。

### 5.2 `temporal_domain`：11 个特征

在 `temporal` 上加入四个 domain one-hot。该变体用于测量公开领域之间不同的干预频率先验。

### 5.3 `response_temporal`：18 个输入特征、19 个可学习参数

在 `temporal_domain` 上加入冻结 R0 响应属性：

- R0 canonical decision、R0-F decision；
- raw response 是否显式 `$interrupt$`、显式 `$silent$`；
- 是否为非空但 tag 不合法、是否为空；
- `log1p(raw response 字符长度)`。

这里的 raw response 是 R0 在当前因果上下文中已经生成的结果。头只改变二元 gate；若预测 interrupt，则优先复用 R0-F 修复后的 utterance，否则使用固定的空内容 fallback。内容质量没有在 C1 验证指标中单独评分。

### 5.4 明确禁止的特征

有效 v2 不包含：

- session 总 chunk 数；
- 当前 chunk 在完整 session 中的相对位置；
- 视频最终时长；
- 未来 dialog、未来帧或未来 interval；
- 当前样本或测试折的 gold label。

静态审计确认 feature row 不含 gold，fold assignment 不读取 gold，且 R0-F 重建预测 SHA256 与冻结参考完全一致。

## 6. 作废的 v1 与未来长度泄漏

第一次实现曾包含 `relative_position`、`relative_position_squared` 和 `log1p_session_chunks`。它们依赖完整 session 的 interval 总数；在在线处理第 `i` 个 chunk 时，这等价于提前知道未来还会出现多少 chunk，违反因果约束。

该问题在实验后的即时特征/系数审计中发现，发生在报告和任何排行榜提交之前。v1 的指标、预测、bootstrap 和 promotion 判断全部作废，不得作为 C1 结果引用。原始产物已按字节保留到：

```text
/home/lanjinxin/workspace/deprecated/wearable_ai_challenge/
  2026-07-14_d1_future_length_leak/
```

本报告只记录移除全部未来长度特征后重新运行的 v2。保留 v1 是为了审计和防止将来误用，不是保留一个可比较候选。

## 7. 官方 OOF 结果

| 变体 | Macro F1 | Interrupt F1 | Silent F1 | 预测 interrupt 比例 | 相对 R0-F |
|---|---:|---:|---:|---:|---:|
| R0-F | 0.5362 | 0.4879 | 0.5845 | 0.3571 | - |
| `temporal` | 0.6081 | **0.6391** | 0.5771 | 0.5404 | +0.0719 |
| `temporal_domain` | 0.6104 | 0.6284 | **0.5924** | 0.5075 | +0.0742 |
| `response_temporal` | **0.6119** | 0.6366 | 0.5873 | 0.5248 | **+0.0757** |

最佳变体的完整官方指标：

| 指标 | `response_temporal` |
|---|---:|
| Macro F1 | **0.6119** |
| G-mean F1 | 0.6114 |
| Interrupt precision / recall / F1 | 0.6450 / 0.6284 / 0.6366 |
| Silent precision / recall / F1 | 0.5787 / 0.5961 / 0.5873 |
| TP / FP / TN / FN | 3,363 / 1,851 / 2,732 / 1,989 |

第二次用绝对路径独立运行官方评分器，生成的 metrics 与正式文件逐字节一致，SHA256 均为 `e8b392239a418aded4db0209ce45952eb7787a6b18e0f2228acf764713d051bc`。

## 8. 稳定性与切片分析

### 8.1 Session bootstrap

| 变体 | 增益中位数 | 95% 区间 | 增益为正比例 |
|---|---:|---:|---:|
| `temporal` | +0.0718 | [+0.0610, +0.0825] | 1.000 |
| `temporal_domain` | +0.0742 | [+0.0634, +0.0850] | 1.000 |
| `response_temporal` | **+0.0756** | **[+0.0660, +0.0853]** | **1.000** |

bootstrap 的抽样单位是 session，不是把 9,935 个高度相关的 chunk 当作独立样本。

### 8.2 五个测试折

最佳变体相对 R0-F 的各折 Macro 增益依次为 `+0.0587`、`+0.0761`、`+0.0816`、`+0.0748`、`+0.0849`。没有单折负增益，因此总体结果不是由某一个偶然 fold 独自驱动。

### 8.3 四个领域

| Domain | R0-F Macro | D1 Macro | 增益 |
|---|---:|---:|---:|
| Arts and Crafts | 0.5389 | 0.6129 | +0.0740 |
| Chef | 0.5485 | 0.6037 | +0.0552 |
| Handyman | 0.5100 | **0.6210** | **+0.1110** |
| Tutorial | 0.5144 | 0.6021 | +0.0878 |

Handyman 的提升最大，主要因为 R0-F 在该领域过于保守。四领域均提升是积极信号，但 domain one-hot 本身只带来小幅增益，说明简单的领域频率差异不是全部原因。

### 8.4 Chunk 位置

| 位置 | R0-F Macro | D1 Macro | 增益 |
|---|---:|---:|---:|
| 首 chunk | 0.4916 | 0.4996 | +0.0080 |
| 第 2 个 chunk | 0.3140 | 0.4363 | +0.1223 |
| chunk 2--4 | 0.4397 | 0.5701 | +0.1305 |
| chunk 5--9 | 0.5177 | 0.6140 | +0.0964 |
| chunk 10+ | 0.5218 | 0.5494 | +0.0275 |
| 全部非首 chunk | 0.4915 | **0.5843** | **+0.0928** |

这排除了“D1 只是重复 R0-F 首 chunk 规则”的解释。最强增益位于 session 的早中段；首 chunk 反而只贡献很小增量。

## 9. 相对 R0-F 的决策变化

最佳变体：

- 修复 `1,705` 个 R0-F FN；
- 修复 `723` 个 R0-F FP；
- 新引入 `1,197` 个 FP；
- 新引入 `513` 个 FN；
- `3,667` 个原本正确的决策保持正确；
- `2,130` 个原本错误的决策仍未修复。

因此它不是单向提高 interrupt 率的规则。它一边主动回收大量 FN，一边也将一部分 R0-F interrupt 改回 silent。预测 interrupt 比例为 `52.48%`，接近公开金标的 `53.87%`，这恰好也是隐藏测试迁移的风险：若隐藏集干预频率或标注粒度改变，当前阈值可能需要重新校准。

## 10. 系数与科学解释

五折中较稳定的高权重集中在：累计输入帧比例为负、chunk 编号为正、已见历史轮数为负、当前 interval 时长为正、已观察结束时刻为负、首 chunk 为正。由于这些特征相关且每折独立标准化，不能把单个系数解释成因果机制；它们共同形成的是一条公开数据的干预节奏曲线。

关键消融是：

```text
temporal           0.6081
+ domain           0.6104   (+0.0023)
+ R0 response      0.6119   (+0.0015)
```

这说明冻结模型的自然语言/tag 属性只提供了很小的额外信息。D1 已证明一个强标量控制是必要基线，但还没有证明视觉 hidden state 对决策有独立增量。下一项实验必须在完全相同 split 上加入 tag score margin 和最终因果多模态 hidden state，并报告它们相对 `temporal_domain`/`response_temporal` 的 OOF 增量。

## 11. Promotion Gate

| 条件 | 要求 | 结果 |
|---|---|---|
| 相对 R0-F Macro 增益 | 至少 +0.015 | +0.0757，通过 |
| Session bootstrap 下界 | 大于 0 | +0.0660，通过 |
| 两类 F1 | 不坍缩 | 0.6366 / 0.5873，通过 |
| 非首 chunk | 有实际增益 | +0.0928，通过 |
| Domain | 完整报告且无单域退化 | 四域均提升，通过 |

D1 标量控制通过原定 promotion gate。这里“通过”表示值得进入部署固化与神经特征增量实验，不表示已经获得隐藏集可提交分数，也不表示应该立即启动 GRPO。

## 12. 部署与提交边界

当前 `predictions.jsonl` 是五个轮换模型的 OOF 合并结果。它适合科学评估，不是官方测试推理时可复现的单一模型。下一步必须：

1. 冻结最终特征集合和 OOF 选出的训练超参数；
2. 在完整 public validation 上拟合一个最终头；
3. 将标准化统计、权重、bias 和预先确定的阈值序列化；
4. 在 starter kit 的逐 chunk 推理中仅使用当前可见信息复现特征；
5. 明确最终头在完整 public validation 上的重拟合分数是 train-fit sanity check，不再是 OOF 泛化指标。

未执行任何排行榜或外部提交。上传预测、构建测试 container 或消耗提交次数仍需要用户明确授权。

### 12.1 单一部署头已经固化

在 OOF 特征选择和阈值策略冻结后，已生成一个可在测试时在线应用的单一头：

```text
实验 ID：20260714_internvl35_1b_causal_scalar_decision_head_d1_final_v1
产物：output/experiments/
  20260714_internvl35_1b_causal_scalar_decision_head_d1_final_v1/
```

最终头在全部 700 个 public-development session 上重拟合，特征顺序仍是冻结的 `response_temporal`。阈值没有根据这次全量拟合的预测重新优化，而是预先固定为五个 OOF calibration 阈值的中位数 `-0.1295841239905541`。序列化文件包含 18 个标准化均值、尺度和权重，加一个 bias，共 19 个可学习参数。

| 最终头检查项 | 结果 |
|---|---:|
| Train-fit Macro F1 | 0.6136 |
| Train-fit Interrupt F1 | 0.6470 |
| Train-fit Silent F1 | 0.5803 |
| Train-fit TP / FP / TN / FN | 3,491 / 1,949 / 2,634 / 1,861 |
| 预测 interrupt 比例 | 0.5476 |
| 最终总参数 | 1,060,897,811 |
| `decision_head.json` SHA256 | `3db4e2e97b6941cb313ce136ec3f1453bc01f50ab2731a4e2bdfc847a3fac3c3` |

第二次独立重拟合得到的 `decision_head.json`、train-fit predictions 和官方 metrics 均与正式产物逐字节一致。Train-fit 的 `0.6136` 只证明序列化/重载/在线打分没有异常，不是新的泛化分数；科学比较仍使用 session-held-out OOF `0.6119`。

需要注意，官方 starter 的基础接口 `model.generate(frames, messages)` 不传入 domain、当前 interval 或上一 interval 的结束时间，因此不能只替换一个 model class 就完整复现该头。项目已增加 `proactive_d1.run_deploy` 和 `process_session_with_scalar_head`：在逐-session runner 已经持有当前 row/interval 的位置构造与训练共用的 `causal_scalar_values`，随后调用序列化头。也就是说，最终提交需要包含自定义推理 runner；可以修改 starter 推理代码，但官方 scorer、预测 JSONL schema、帧因果策略和标签解析规则不变。

## 13. 下一步

按当前证据，优先顺序调整为：

1. 已完成：实现并验证单一可部署 `response_temporal` 最终头；
2. 已完成：提取完整 tag log-prob margin 与最终因果 hidden，并在相同 split 上完成四项 OOF 消融；
3. `fused_linear` 以 Macro `0.6341` 稳定超过本报告的标量 `0.6119`，已成为新的主比较基线；
4. 已完成：序列化 1,044 参数最终融合头并通过单 session GPU 在线一致性验证；
5. 下一步先做融合路径的等价推理加速与 submission/container 审计，再比较小 MLP、LoRA 或 state-aware supervision；
6. R1 扩样、专门粒度建模和 GRPO 继续等待新的可测残差。完整后续结果见 [D1 神经融合报告](20260715_internvl35_1b_neural_decision_head_d1.md)。

## 14. 复现命令

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.run_scalar \
  --config configs/d1_internvl35_1b_scalar_oof.json \
  --output-dir output/experiments/20260714_internvl35_1b_causal_scalar_decision_head_d1_oof_v2
```

重建冻结预测诊断：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.analyze \
  --experiment-dir output/experiments/20260714_internvl35_1b_causal_scalar_decision_head_d1_oof_v2
```

完整运行耗时 `43.816` 秒，无 GPU、无模型推理重跑。

## 15. 实验产物指纹

| 对象 | SHA256 |
|---|---|
| 配置 | `1dcc2688e453271e544e873b5fbc6d22117837052b6c0f5558972d6e7fe89634` |
| Split manifest | `bd537e9e155586cf3af9f26052fda277fa3e1930e378538346cc197432ff86c0` |
| `temporal` 预测 | `575c2108c9592488955b97861e409a9ea08cee4591e107589173ab47d270dc1b` |
| `temporal_domain` 预测 | `1cb7a3e296ffd940b01055715641451f73ba8a985ffca0fa1b686eecb0214550` |
| `response_temporal` 预测 | `fc34808d65abf452600bb22b3cd4b3d43ac4e7dceb345542573d10e3248de5ee` |
| 最佳官方 metrics | `e8b392239a418aded4db0209ce45952eb7787a6b18e0f2228acf764713d051bc` |
| 对比结果 | `fd852639f0db025f09dddb4a7b562b423b90ee80f80de6e56b0103020b3ca0e8` |
| 分析结果 | `79cec8fadfbb0a656b64c6ba413b65978d08594e17bd465dda0a404c435440a5` |
