# 新实验完整讲解：D2、D3、U0、U1、S0 与 S1 准备

> 更新时间：2026-07-18  
> 范围：2026-07-15 至 2026-07-17 新完成的决策、语言和状态实验  
> 当前决策基线：D3 `dynamics_fused`，session-level OOF Macro F1 `0.6690`  
> 当前语言结论：U1 自动实验完成，人工盲评未完成，不能判断 oracle state 是否改善语言  
> 当前状态结论：S0 zero-shot state decoding 失败，S1 仅完成数据与协议准备，尚未标注和训练  
> 说明：本文用于理解实验动机、输入输出、工程实现和结果边界。正式数字以 `reports/`、冻结产物和官方 scorer 为准。

## 1. 先看全局：这些实验分别在解决什么

这些实验不是沿着一条线简单地“把模型越做越大”，而是在拆分三个原本混在一起的问题：

1. **Decision，决策问题**：当前时刻究竟应该 `$interrupt$` 还是 `$silent$`？
2. **Utterance，语言问题**：已经决定介入后，具体应该说什么？
3. **State，状态理解问题**：用户当前做到哪一步、是否完成、是否出错？

三条实验线的关系如下：

```text
决策线：R0 -> D1 -> D2 residual / D2 LoRA -> D3 dynamics
                         失败              成功但有协议依赖风险

语言线：D1 fallback 问题 -> U0 全量审计 -> U1 强制生成
                                            等待人工盲评

状态线：U1 oracle state -> S0 zero-shot state decoding -> S1 supervised decoder
                              失败                  仅准备完成
```

最重要的总体结论是：

- 单纯把 D1 的决策模块做得更复杂，收益很小且不稳定；
- 加入跨 chunk 的历史变化后，决策分数明显提高；
- 但 D3 的提升并不能全部解释为视觉程序理解，因为官方 dialog 暴露了上一 chunk 的 gold 介入痕迹；
- 当前语言输出中的大量 `Please continue with the next step.` 是硬编码 fallback，不是模型真正生成的具体指导；
- oracle state 会改变模型生成的文本，但尚无人工评分证明改变方向是好的；
- 冻结 1B 模型无法通过 zero-shot 候选打分稳定识别细粒度状态，需要显式监督。

## 2. 阅读结果前必须理解的术语

### 2.1 OOF

OOF 是 `Out-of-Fold`，中文可理解为“折外预测”。

700 个 session 被分成五折。每次轮转：

```text
3 折：fit，训练参数
1 折：calibration，选择 L2、early stopping 或 threshold
1 折：test，只做最终预测
```

轮转五次以后，每个 session 都只在自己没有参加训练和校准的那一轮得到预测。把五轮 test 预测合并，得到完整 700-session OOF 结果。

这比直接在 700 个 session 上训练再在同一批上评分可靠，但仍然属于 `public-validation-supervised`：训练使用了公开验证集的一部分标签，因此它不是隐藏测试集泛化证据。

### 2.2 Session-level split

同一个视频 session 中的所有 chunk 必须进入同一折，不能拆散。

原因是相邻 chunk 的视频、任务、dialog 和状态高度相似。若前半段进入训练、后半段进入测试，模型相当于见过同一个视频，会产生严重的数据泄漏。

### 2.3 Calibration

Calibration fold 是“校准折”。它不负责学习主要模型参数，而是负责决定：

- 哪个 L2 正则强度更合适；
- 训练到第几个 epoch 应停止；
- logit 大于多少时判为 interrupt。

它相当于训练集与测试集之间的开发集。

### 2.4 Logit、threshold 与 margin

`logit` 是分类器输出的连续分数。它越高，模型越倾向 interrupt，但它不一定是已经校准好的概率。

```text
logit >= threshold -> interrupt
logit <  threshold -> silent
```

`tag margin` 是 InternVL 对两个固定标签的相对偏好：

```text
tag_margin
= log p($interrupt$ | 当前因果输入)
- log p($silent$    | 当前因果输入)
```

margin 为正只表示语言模型相对更喜欢 interrupt 标签。D1/D3 仍然要把 margin 和其他特征交给决策头，再用校准阈值分类。

### 2.5 Macro F1

官方 Macro F1 是两类 F1 的平均：

```text
Macro F1 = (Interrupt F1 + Silent F1) / 2
```

它要求 interrupt 和 silent 都做好。单纯把所有 chunk 都判为 interrupt，interrupt recall 可能很高，但 silent F1 会坍塌，Macro F1 不会高。

### 2.6 Paired session bootstrap

Bootstrap 是“有放回重采样”。这里以 session 为单位，随机抽取 700 个 session，可重复抽到同一个 session，然后重新计算新模型减基线的 Macro F1。

重复 5,000 或 10,000 次后得到增益区间。例如 D3 相对 D1：

```text
95% interval = [+0.02654, +0.04372]
```

整个区间大于 0，表示不同 session 构成下，D3 的正增益很稳定。

必须按 session 抽样，不能把 9,935 个强相关 chunk 当作彼此独立样本。

### 2.7 Promotion gate

Promotion gate 是实验前冻结的“推广门槛”。只有同时满足最小增益、bootstrap、五折稳定性、domain 稳定性和类别不坍塌等条件，才允许把新模型升级为主基线。

先定门槛再看结果，是为了防止看到数字后临时修改成功标准。

## 3. D2 第一部分：Residual MLP 决策头

### 3.1 动机

D1 使用的是线性分类器。它的输入是：

```text
18 个 causal scalar
+ 1 个 tag margin
+ 1024 维 hidden
= 1043 维
```

线性头只能学习加权求和。例如：

```text
logit = w1 * 时间 + w2 * margin + w3... * hidden + bias
```

它不擅长直接表达“只有当 A 和 B 同时出现时才介入”这类非线性交互。

D2 residual MLP 要回答：

> D1 已有特征中是否存在少量、稳定的非线性规律，只是线性边界没有表达出来？

如果答案是肯定的，就不必修改 1B backbone，只需增加一个很小的 MLP。

### 3.2 为什么采用 residual，而不是重训一个新 MLP

Residual 是“残差”。结构为：

```text
1043 维标准化输入
    |
    +------> 冻结 D1 linear logit ----------------+
    |                                             |
    +-> Linear(1043, 8) -> GELU -> Linear(8, 1) --+
                                                  |
                                             D2 logit
```

可以写成：

```text
D2_logit = D1_logit + residual(x)
```

第二层在训练开始时全部初始化为 0，因此：

```text
epoch 0: residual(x) = 0
D2_logit = D1_logit
```

这样做的动机是保护 D1：如果校准集不支持非线性修正，early stopping 可以选择 epoch 0，模型就退回原 D1，而不是让一个随机初始化的新网络破坏已经不错的边界。

### 3.3 MLP、GELU 和参数量

`MLP` 是多层感知机，本质上是若干线性层加非线性激活。

`GELU` 是一种平滑激活函数。没有激活函数时，两层线性层仍然等价于一层线性层；加入 GELU 后才有表达非线性交互的能力。

残差分支参数量：

```text
1043 * 8      第一层权重
+ 8           第一层 bias
+ 8           第二层权重
+ 1           第二层 bias
= 8,361
```

加 D1 的 1,044 参数，总 head 为 9,405 参数。

### 3.4 训练设置为什么这样设计

```text
Optimizer        AdamW
Learning rate    1e-3
Batch size       512
Max epochs       80
Early stopping   calibration balanced BCE, patience 12
Device           CPU
```

- `AdamW`：常见的神经网络优化器；
- `BCE`：Binary Cross Entropy，二元交叉熵；
- `class-balanced BCE`：对 interrupt 和 silent 给予平衡权重，防止多数类主导训练；
- `patience 12`：校准损失连续 12 个 epoch 没有有效改善就停止；
- `gradient clip 1.0`：限制梯度过大，避免训练不稳定。

Backbone、1043 维输入、五折划分、D1 基础 logit 都保持冻结。因此实验只检验“非线性残差是否有用”。

### 3.5 结果

| 模型 | Macro F1 | Interrupt F1 | Silent F1 |
|---|---:|---:|---:|
| D1 linear | 0.6341 | 0.6352 | 0.6330 |
| D2 residual MLP | 0.6351 | 0.6375 | 0.6327 |

净增益只有：

```text
+0.000994，约 +0.0010
```

D2 比 D1 多预测了 36 个 interrupt，其中：

```text
23 个是新增 TP
13 个是新增 FP
```

所以 interrupt F1 略升，silent F1 略降。

五折中：

- fold 0 和 fold 4 选择 epoch 0，即不要任何 residual；
- 只有 3/5 folds 有严格正增益；
- bootstrap 95% 区间 `[-0.00113, +0.00314]` 跨过 0；
- Arts 和 Chef 略降，主要收益集中在 Handyman。

### 3.6 正确结论

不能说“MLP 完全没有学习”，因为三折确实接受了 5 至 10 个 epoch 的修正。

更准确的结论是：

> 在 D1 已有的 1043 维表示上，通用小型非线性 head 只能得到约 +0.001 的不稳定边际收益；D1 的主要瓶颈不是决策头太线性或太小。

因此没有继续搜索宽度 16、32、更多层或更多 seed。继续在相同公开五折上搜索，很容易把验证集噪声当成改进。

## 4. D2 第二部分：最后语言层 MLP LoRA

### 4.1 动机：把监督送回特征内部

Residual MLP 只在 backbone 已经输出 1024 hidden 后做修正。若 hidden 本身没有把“是否该介入”的信息组织好，再复杂的外部 head 也难以恢复。

因此第二个 D2 问题是：

> 不完整微调 1B 模型，只适配最后一个语言层的 MLP，能否让 tag margin 和 hidden 更适合 interrupt/silent 分类？

它与 residual MLP 的区别是：

```text
Residual MLP：修改最终分类函数，不修改 InternVL 表示
Final-MLP LoRA：修改 InternVL 最后一层表示，再训练融合分类头
```

### 4.2 LoRA 是什么

LoRA 是 `Low-Rank Adaptation`，低秩适配。

冻结原权重矩阵 W，不直接更新它，而是学习一个低秩增量：

```text
W_adapted = W + scale * B * A
```

若原矩阵很大，A 和 B 的秩 `rank=8` 很小，训练参数会远少于完整微调。

本实验只作用于第 28 个、也就是索引 27 的最后语言层 MLP：

```text
gate_proj
up_proj
down_proj
```

术语可这样理解：

- `up_proj`：把 1024 维 hidden 扩展到 3072 维；
- `gate_proj`：产生门控值，决定中间通道应通过多少；
- `down_proj`：把 3072 维投回 1024 维；
- `rank=8`：LoRA 中间低秩空间只有 8 维；
- `alpha=16`：控制 LoRA 增量缩放；
- `dropout=0`：本实验不随机丢弃 LoRA 通道。

LoRA 新增 98,304 个可训练参数。连同最终融合头，整个系统约 1.060997B 参数，仍低于 Small 的 2B 限制。

### 4.3 LoRA 训练输入与输出

每个 chunk 仍只使用因果输入：

```text
当前及历史可见 frames
+ query
+ 当前可见 prior dialog，最多 4 turns
```

模型分别对两个候选标签评分：

```text
$silent$    -> 3 tokens
$interrupt$ -> 3 tokens
```

每个候选的分数是三个 token 条件 log probability 的和。两者相减得到 adapted tag margin。

LoRA 使用 class-balanced BCE 学习，让 gold interrupt 的 chunk 更偏向 `$interrupt$`，gold silent 的 chunk更偏向 `$silent$`。

训练后得到：

```text
adapted tag margin
+ adapted 1024 hidden
+ 原 18 scalar
-> 新的 fold-local linear head
-> interrupt / silent
```

语言正文没有用 LoRA 生成。生成时关闭 adapter，继续复用冻结 R0 raw response。这样才能只测决策表示适配，而不把语言变化混进指标。

### 4.4 为什么必须做缓存

普通端到端 LoRA 每个 epoch 都要重新运行：

```text
视觉塔 + projector + 28 层语言模型
```

700 sessions、9,935 chunks、每个 chunk 两个标签候选，五折训练会非常昂贵。

本实验冻结第 0 至 26 层和第 27 层 attention，只训练第 27 层 MLP。因此先把最后 MLP 所需的中间状态缓存下来，训练时只重放最后 MLP、final RMSNorm 和 LM head，不再重复完整视频前向。

### 4.5 BF16 batch-shape drift 是什么

BF16 是 16 位脑浮点格式，速度快、显存小，但有效精度低于 float32。

同一条样本分别以 batch 1 和 batch 64 计算矩阵乘法时，GPU 底层可能选择不同 kernel 或归约顺序。数学公式相同，但舍入路径不同，因此结果可能差一点：

```text
batch 1 hidden != batch 64 中同一条样本的 hidden
```

这叫 batch-shape drift，即“批形状导致的数值漂移”。

它不是数据错误，但决策 margin 可能很接近 threshold，小差异也可能改变最终结果，所以不能忽略。

### 4.6 为什么早期缓存失败

工程经历了三次重要失败：

1. feasibility v1 只缓存 residual 和 normalized input，没有消除 MLP 的 BF16 批形状差；
2. feasibility v2 校正 MLP 后，零 adapter 在两个 chunk 上精确，但一个预注册 local/full margin 诊断超过容差；
3. 第一版全量四状态缓存运行到 session 11、chunk 4 时，发现 final RMSNorm 也有最大 `0.03125` 的 hidden 差。

这些失败没有被改写成成功。最终方案增加到每个候选、每个标签位置保存六个 tensor。

### 4.7 六状态缓存怎样校正

保存的六个状态是：

```text
1. post_attention_residual
2. post_attention_normalized_mlp_input
3. full_batch_base_mlp_output
4. local_base_mlp_output
5. full_batch_base_final_hidden
6. local_base_final_hidden
```

训练时在完全相同的 batch 64 形状内分别计算 adapter enabled 和 disabled：

```text
delta_mlp
= MLP_enabled(batch64)
- MLP_disabled(batch64)

adapted_mlp
= trusted_full_batch_base_mlp
+ delta_mlp
```

随后 final norm 和 LM-head margin 也采用同样的 enabled-minus-disabled 差分，再加回可信的完整基线。

直观类比：

> 不直接相信局部重放的绝对值，只相信“在同一种数值环境下，打开 LoRA 相对关闭 LoRA 改变了多少”，再把这个改变量加到正式基线上。

最终 700 sessions / 9,935 chunks 的 zero-adapter hidden、margin、prompt 和 key 与 D1 缓存精确一致。

### 4.8 正式 LoRA 结果

| 变体 | D1 对应结果 | LoRA 后 | 增量 |
|---|---:|---:|---:|
| tag only | 0.5313 | 0.5879 | +0.0566 |
| hidden linear | 0.6031 | 0.6064 | +0.0033 |
| fused linear | 0.6341 | 0.6357 | +0.0016 |

这个表很关键。

Tag-only 提高很多，说明 LoRA 确实学到了 interrupt/silent 监督，不是训练链路坏了。但融合 18 scalar、margin 和 hidden 后，只比强 D1 基线高 `0.0016`。

主结果：

```text
Macro F1          0.6357
Interrupt F1      0.6601
Silent F1         0.6114
Pred interrupt    52.81%
Bootstrap CI      [-0.00425, +0.00756]
Positive folds    2/5
Positive domains  2/4
```

它主要把模型推向“更多 interrupt”：相对 D1 净增加 333 个 TP，但也净增加 301 个 FP。Interrupt recall 上升，silent recall 下降。

### 4.9 正确结论

不能说“LoRA 没学会”。正确说法是：

> 最后一层 MLP LoRA 能明显改变标签偏好，但 D1 的融合头已经吸收了大部分有用信息，剩余决策收益只有 +0.0016，而且跨 fold、domain 不稳定。

因此没有 full refit，也不继续搜索 rank、层、学习率和 batch size。

D2 两部分共同排除了一个重要解释：

> 当前主要问题不是“决策头不够复杂”，也不是“只要轻微适配最后一层就能稳定解决”。

这为 D3 转向“跨时间变化”提供了动机。

## 5. D3：冻结缓存的因果动态决策头

### 5.1 核心动机

D1 只看当前 chunk 的表示：

```text
当前 scalar + 当前 margin + 当前 hidden
```

但程序任务中的关键往往是变化：

- 用户刚刚从准备阶段进入执行阶段；
- 手中物体发生了变化；
- 某一步从未完成变为完成；
- 模型对 interrupt 的偏好突然变化；
- 当前状态与前面长期状态明显不一致。

两个当前画面可能很相似，但“相对上一时刻发生了什么”不同，介入决策也可能不同。

D3 因此检验：

> 不重新训练 backbone，只比较当前 frozen D1 表示与过去表示，能否获得稳定增益？

### 5.2 D3 的完整输入

D3 继承 D1 的 1043 维：

```text
18 scalar
+ 1 current tag margin
+ 1024 current hidden
= 1043
```

再增加：

```text
8 个 dynamics scalar
+ 1024 维 hidden delta
= 1032
```

最终：

```text
1043 + 1032 = 2075 维输入
```

线性头有 2075 个 weight 和 1 个 bias，共 2076 参数。

### 5.3 八个动态标量逐个解释

令：

```text
m_i = 当前 chunk 的 tag margin
h_i = 当前 chunk 的 1024 维 hidden
```

八个标量为：

1. `has_previous_chunk`

   是否存在上一 chunk。首 chunk 为 0，其他为 1。

2. `tag_margin_delta_previous`

   ```text
   m_i - m_(i-1)
   ```

   正值表示模型比上一时刻更倾向 interrupt，负值表示更倾向 silent。

3. `tag_margin_abs_delta_previous`

   ```text
   abs(m_i - m_(i-1))
   ```

   不关心变化方向，只衡量标签偏好变化有多剧烈。

4. `tag_margin_delta_history_mean`

   ```text
   m_i - mean(m_0 ... m_(i-1))
   ```

   衡量当前相对整个历史平均是否异常。

5. `hidden_cosine_previous`

   当前 hidden 与上一 hidden 的余弦相似度。接近 1 表示方向相似，较低表示内部语义表示发生明显变化。

6. `hidden_delta_rms_previous`

   ```text
   RMS(h_i - h_(i-1))
   ```

   RMS 是均方根，衡量 1024 个维度的整体变化幅度。

7. `hidden_cosine_history_mean`

   当前 hidden 与历史 hidden 平均值的余弦相似度。

8. `hidden_delta_rms_history_mean`

   当前 hidden 相对历史平均 hidden 的 RMS 变化。

此外保留完整向量：

```text
hidden_delta = h_i - h_(i-1)
```

八个标量只告诉模型“变化多大”；1024 维 delta 还告诉模型“具体沿哪些内部语义方向变化”。

### 5.4 为什么它是严格因果的

处理 chunk i 时只读取：

- 当前 D1 hidden/margin；
- 上一 hidden/margin；
- 之前 hidden/margin 的累计和与数量。

历史平均不包括当前之后的任何 chunk。每个 session 开始时状态重置，首 chunk 的动态全部为 0。

在线状态只保存：

```text
previous_hidden
previous_margin
hidden_sum
margin_sum
history_count
```

因此不需要未来信息，也不需要保存整个历史序列。

### 5.5 四个变体为什么都要做

| 变体 | 维度 | 目的 |
|---|---:|---|
| d1_fused_replay | 1043 | 确认代码能精确复现 D1 |
| dynamics_scalar | 1051 | 只看八个变化摘要是否有用 |
| dynamics_hidden | 2067 | 只看完整 hidden delta 是否有用 |
| dynamics_fused | 2075 | 同时使用两类动态，是唯一可推广主变体 |

先冻结主变体，避免看到三个结果后挑最高者并假装它是预先选择的。

### 5.6 正式结果

| 变体 | Macro F1 | 相对 D1 |
|---|---:|---:|
| D1 exact replay | 0.6341 | 0 |
| dynamics scalar | 0.6551 | +0.0210 |
| dynamics hidden | 0.6594 | +0.0253 |
| dynamics fused | **0.6690** | **+0.0349** |

D3 主结果：

```text
Interrupt F1      0.6845
Silent F1         0.6535
TP / FP / TN / FN 3560 / 1489 / 3094 / 1792
Pred interrupt    50.82%
Bootstrap CI      [+0.02654, +0.04372]
Positive folds    5/5
Positive domains  4/4
```

它通过全部预注册门槛，成为新的科学 OOF 基线。

### 5.7 D3 到底改对了多少

D3 相对 D1 改变 2,142 个判定：

```text
修复 FN -> TP    755
修复 FP -> TN    493
新引入 TP -> FN  360
新引入 TN -> FP  534
净修复           354
```

也就是说，D3 不是只把系统整体推向 interrupt。它同时纠正了一批漏报和误报，但也产生新的两类错误，最终净收益为正。

首 chunk 只改变 1 个，2,141 个变化发生在非首 chunk。Non-first Macro 从 `0.60454` 提高到 `0.64460`，符合动态特征的设计目标。

### 5.8 最重要的风险：D3 可能在利用官方 dialog policy

这是当前结果中必须讲清楚的部分。

官方数据给每个 chunk 一个 `dialog[i]`，表示当前决策前可见的历史对话。在公开验证集全部 9,235 个非首 chunk 中：

```text
dialog[i] 比 dialog[i-1] 是否新增 assistant turn
恰好等于
上一 chunk 的 gold interrupt/silent
```

一致率是 100%。

原因是：上一 chunk 如果 gold 是 interrupt，下一 chunk 的官方 prior dialog 就加入那条官方 assistant utterance；若上一 chunk gold 是 silent，就不加入。

所以当前输入虽然没有直接给出“上一标签”字段，但对话长度变化等价地暴露了上一 gold action。

D3 的 hidden 是由视频、query 和 dialog 共同产生的。当 dialog 新增一条较长 assistant utterance 时，hidden 和 margin 都会发生很大变化：

```text
没有新增 assistant turn：hidden delta RMS 均值 0.2680
新增 assistant turn：    hidden delta RMS 均值 1.3392
```

D3 很容易识别这两种状态。

### 5.9 这算不算泄漏

需要区分三种说法：

1. **不是当前或未来标签泄漏。** D3 在 chunk i 没有读取 gold answer[i]，也没有读取未来帧或未来 dialog。
2. **在官方 benchmark 协议下是合法因果输入。** prior dialog 本来就在输入中。
3. **它是强协议依赖或 shortcut 风险。** D3 的收益可能主要来自官方如何构造对话历史，而不是识别视频中具体完成了哪一步。

因此可以说：

> D3 是官方协议下真实、稳定的排行榜决策改进。

但不能说：

> D3 已经学会了纯视觉程序进展理解。

### 5.10 为什么分组结果支持这个诊断

| 上一 chunk 后是否新增 assistant turn | 当前 gold interrupt rate | D1 Macro | D3 Macro | 增益 |
|---|---:|---:|---:|---:|
| 否 | 64.01% | 0.5949 | 0.6355 | +0.0406 |
| 是 | 39.42% | 0.5901 | 0.5915 | +0.00145 |

D3 的绝大部分收益来自“上一轮没有新增 assistant turn”的组。这个组当前再次需要 interrupt 的概率更高。

这说明 D3 很可能学到了 benchmark 的介入转移规律，例如：

```text
上一轮没说话 -> 当前更可能该说话
上一轮刚说过 -> 当前更可能保持安静
```

这对官方输入有效，但真实部署若使用模型自己生成的历史，上一轮模型可能判断错、也可能生成不同文本，分布会改变。

### 5.11 Final refit 与部署验证

OOF 的 0.6690 来自五套 fold 模型，不能直接拿一套去提交。工程上又在全部 700 sessions 上拟合了一套最终 head：

```text
L2        0.01，来自五折中位数
threshold 0.14439966662436324，来自五个 calibration threshold 中位数
```

Full-fit Macro `0.7544` 只是训练闭环检查，因为训练和评分是同一批 700 sessions，不能当作泛化性能。

在线状态机重放 9,935 chunks：

- 动态标量与离线缓存最大差 0；
- hidden delta 最大差 0；
- 9,935/9,935 decisions 一致；
- 最大 logit 差 `2.95e-7`。

10-chunk GPU smoke 也精确匹配 raw response、prompt、margin、hidden、dynamics、decision 和 answer。它证明部署代码闭环，不证明 10-chunk Macro 是有效性能估计。

### 5.12 D3 之后真正应该补什么

1. 冻结 D3，不再在同一五折搜索新动态窗口和阈值；
2. 做 `official-dialog / no-assistant / self-fed-dialog` 鲁棒性对比；
3. 单独统计 D3 新增 interrupt 中 fallback utterance 的比例；
4. 继续状态监督路线，确认是否能得到超越 dialog policy 的显式 step/progress 信号；
5. 提交前完成完整 prediction/container packaging，而不是只依赖 10-chunk smoke。

## 6. U0：为什么有大量固定句子

### 6.1 U0 的动机

D1 的 0.6341 只评价 interrupt/silent，不评价说了什么。

D1 的输出组装逻辑是：

```text
D1 决定 silent
-> 输出 $silent$

D1 决定 interrupt 且 R0 原本有正文
-> 输出 $interrupt$ + R0 正文

D1 决定 interrupt 但 R0 原本 silent 或正文为空
-> 输出 $interrupt$Please continue with the next step.
```

所以 `Please continue with the next step.` 是代码中的固定 fallback。它不是模型看完 D1 scalar 后“生成”出来的，也不是 planner 生成的计划。

真正发生的是：

> D1 决策头把 R0 的 silent 翻成了 interrupt，但没有同步让语言模型重新生成正文，只能用固定句子填空。

这就是 `gate-to-language interface break`，即“决策门和语言生成接口脱节”。

### 6.2 全量审计结果

D1 共预测 4,613 个 interrupt：

```text
fallback       2,586，占 56.06%
真实 R0 正文   2,027，占 43.94%
```

fallback 来源：

```text
2,565 个：R0 raw response 明确是 $silent$
21 个：   R0 输出了空正文 $interrupt$
```

因此不是后处理偶尔弄丢了文本，而是超过一半的 D1 介入没有匹配的生成内容。

### 6.3 为什么 second chunk 最严重

| 位置 | D1 interrupt | fallback | fallback 比例 |
|---|---:|---:|---:|
| first | 699 | 34 | 4.86% |
| second | 426 | 423 | **99.30%** |
| 2--4 | 1,165 | 886 | 76.05% |
| 5--9 | 1,254 | 675 | 53.83% |
| 10+ | 1,069 | 568 | 53.13% |

D1 从公开标签规律中学到 second chunk 经常应该介入，但 R0 自由生成几乎总在 second chunk 选择 silent。于是决策被翻转，正文却不存在。

这进一步说明 D1 的高分很大程度上是决策校准成功，不代表完整助手已经能给出高质量指导。

### 6.4 Binary precision 不是语言正确率

Fallback 所在 chunk 的二元 precision 是 63.69%，非 fallback 是 74.89%。这只表示“该 chunk 是否应该 interrupt”，不表示句子内容是否正确。

一句对象完全错误的指导，只要 gold 标签是 interrupt，在官方二元 scorer 中仍然是 TP。

## 7. U1：固定 gate 后强制生成具体正文

### 7.1 动机

U0 发现 fallback 后，需要区分三种可能原因：

1. 只是接口问题：R0 本来选择 silent，但一旦强制它说话，就能说出具体内容；
2. 缺少程序状态：只有告诉模型当前步骤和错误证据后，才能说对；
3. 语言能力或监督不足：即使强制说话、给 oracle state，1B 模型仍然说不好。

U1 因此把 D1 gate 100% 固定，只比较正文生成。

### 7.2 样本怎样选择

选取条件是：

```text
D1 OOF 决定 interrupt
且 R0 raw response 明确选择 silent
```

也就是最典型的 fallback 场景。

正式样本：

- 20 sessions；
- 80 sampled states；
- 四个 domain 各 20；
- second、2--4、5--9、10+ 四个位置段各 20；
- 排除旧 R1 的四个 session。

### 7.3 四种正文条件

```text
current fallback
    固定句 Please continue with the next step.

forced no-state
    不提供状态，只强制模型在 $interrupt$ 后继续生成

forced oracle step
    增加 Current step 和 Next step

forced oracle full
    再增加 progress、完成证据、错误证据和 recovery action
```

所有条件都保持 D1 interrupt/silent 不变。

### 7.4 为什么要真实预填 `$interrupt$`

普通 prompt 只是用文字告诉模型“请输出指导”，模型仍可能先生成 `$silent$`。

U1 在 token 序列的 assistant 位置直接追加 `$interrupt$` 的三个 token，然后让模型从后面 greedy continuation：

```text
完整 prompt + assistant generation marker + $interrupt$
                                               |
                                               v
                                   模型只续写 utterance
```

这叫 `assistant-side prefix prefill`，即助手端前缀预填。它真正固定了已经做出的介入决定。

共同控制 prompt 的核心意思是：

```text
介入决定已经做出；只续写一句简洁、基于当前证据、可执行的英文指导；
不要再输出标签；没有可见证据时不要声称动作已经完成。
```

### 7.5 Oracle state 的输入格式

Step 版本示例：

```text
[Answer-blind oracle procedural state]
Current step: Position and secure the stencil on the paper.
Next step: Apply an even light coat of spray paint through the stencil.
```

Full 版本再加入：

```text
Progress: ongoing
Observed completion evidence: The floral stencil is aligned on the paper.
Observed incomplete/error evidence: Some stencil edges still need to be secured.
Recovery action: Press the stencil flat and secure the lifted edges.
```

`Answer-blind` 表示标注状态时禁止查看当前/future gold answer、模型输出和 decision error，只能看 query、当前 prior dialog 和截至当前 interval 的允许视频。

### 7.6 为什么旧 4-session oracle 被废弃

旧 smoke 的标注者在标注前已经看过对应生成输出。即使没有有意复制，也可能受到输出内容影响，因此不满足 formal blind。

新实验重新标注全部 20 sessions / 80 states，并把静态 plan 与动态状态的可见信息严格隔离。旧文件只保留作 schema 和 runner 工程测试，不能支持 state 效果结论。

### 7.7 视频与对话输入细节

每个 interval 抽 16 帧，累计历史 interval 后最多保留 32 帧。使用最多 4 个 prior dialog turns。

注意：只允许看 `video_intervals_so_far` 明确列出的区间。即使两个 interval 之间的空档时间早于当前时间，也不能查看，因为官方输入没有把空档暴露给模型。

Decoding 固定为：

```text
max_new_tokens = 64
do_sample      = false
seed           = 20260713
```

`do_sample=false` 表示 greedy decoding，每次选择当前最高分 token，避免随机采样带来的比较噪声。

### 7.8 自动结果

| Variant | 非空正文 | 空输出后回退 fallback |
|---|---:|---:|
| no-state | 56/80 | 24 |
| oracle step | 56/80 | 24 |
| oracle full | 57/80 | 23 |

State 确实影响了生成内容：

```text
step vs no-state  43/80 文本不同
full vs no-state  53/80 文本不同
full vs step      45/80 文本不同
```

但“文本不同”不等于“文本更正确”。

Step 没有改变任何样本是否为空；full 只额外恢复一个 second-chunk 输出。自动统计只能证明模型读取了 state block，不能证明它正确使用了 state。

### 7.9 一个具体样本怎样理解

在喷漆模板任务的 second chunk：

```text
fallback:
Please continue with the next step.

no-state:
空输出，仍回退 fallback

oracle step:
空输出，仍回退 fallback

oracle full:
You need to position the stencil on the paper.
```

这个样本说明 full state 可能帮助模型从空输出恢复为具体动作。

但在同一任务另一个 chunk，oracle full 生成了带有可疑对象描述的更复杂句子，而 step 版本更直接。说明字段越多不一定越好：更多状态文字也可能诱导模型复述、混淆或幻觉。

这就是为什么不能凭几个好例子宣布 full state 成功。

### 7.10 为什么官方 Macro 三者完全相同

三种完整 prediction 都保持 9,935 个二元 decision 不变：

```text
Macro F1 = 0.6341
TP/FP/TN/FN = 3165/1448/3135/2187
```

官方 scorer 只看输出是否以 `$interrupt$` 开头，不评价后面的句子。因此正文从 fallback 变成具体指导，官方决策分数仍然相同。

### 7.11 人工盲评为什么是必需的

需要评审结合视频、时间点、query 和 prior dialog，评价：

- correctness：动作和对象是否正确；
- specificity：是否具体，而非空泛；
- actionability：用户能否照着执行；
- groundedness：是否由当前可见证据支持；
- plan consistency：是否符合当前步骤；
- safety：是否安全；
- hallucination：是否捏造看不见的完成、对象或错误。

正式 state package 有 80 样本、三候选，共 240 candidates，两名 reviewer 共 480 行评分。候选身份被打乱，评审看不到 no-state/step/full 标签。

当前评分尚未完成，因此现在不能说：

- oracle state 有效；
- oracle state 无效；
- full state 优于 step；
- no-state 已足够。

## 8. S0：给 oracle plan，模型能否自己读出当前状态

### 8.1 动机

U1 的 oracle state 是人工给的，不可部署。真实系统最终需要模型自己从因果视频和 dialog 预测：

```text
current step
progress
error present
```

S0 先做最低成本可行性测试：不给状态标签训练，只给冻结 InternVL 一份 query-only 四步计划，用固定候选分数做 zero-shot 预测。

S0 不改变 D3 决策，也不读取 U1 人工评分。

### 8.2 Oracle static plan 与 predicted dynamic state

必须区分：

- `oracle static plan`：人工根据 task/query 写的四个宏步骤；
- `predicted dynamic state`：模型根据当前视频和 prior dialog 预测现在处于哪一步、什么进度、是否有错误。

S0 不是完整 deployable planner，因为静态 plan 仍然由 oracle 提供。它只是测试动态状态解码是否有基本信号。

### 8.3 Target 物理隔离

Runner 输入文件只含：

```text
query / task / goal / four-step plan
causal intervals / prior dialog
```

Gold target 单独保存在 targets 文件。Prediction runner 的配置中没有 targets 路径；只有两个 view 的预测完成并冻结哈希后，evaluator 才读取 targets。

这防止推理代码意外读取 gold state。

### 8.4 三个固定分类任务

```text
step:
1=s1, 2=s2, 3=s3, 4=s4

progress:
1=not_started, 2=ongoing, 3=complete,
4=deviated, 5=recovered

error:
1=absent, 2=present
```

总共 4 + 5 + 2 = 11 个候选 language forwards。

使用数字候选是为了：

- 每个候选都是一个 token；
- token 长度相同；
- 避免 `not_started` 和 `recovered` 因 token 数不同而不可比；
- 不让模型自由生成难以解析的 JSON。

每个 state 的视觉塔只运行一次，视觉特征共享给 11 个 language candidate forwards。

### 8.5 模型实际收到的 prompt

以 step 为例，system prompt 追加：

```text
[Structured procedural-state decoding]
The following four-step plan was written from task/query before video inspection.
Task: ...
Goal: ...
Plan:
1. ...
2. ...
3. ...
4. ...
Use only the currently visible video and prior dialog.
Which plan step best describes the work currently being performed,
just completed without moving on, or needing recovery?
Options:
1 = s1
2 = s2
3 = s3
4 = s4
Return only the option digit.
```

模型不是先自由生成再解析，而是分别计算数字 `1/2/3/4` 的 log probability，选择最高者。

### 8.6 V1 为什么是工程失败

V1 smoke 发现，即使候选都是等长单 token，模型也有强烈数字先验：通常天然更喜欢 `1`，然后是 `2`、`3`、`4`。

如果直接选 raw log probability，结果可能只是在测模型喜欢哪个数字，而不是看懂了视频。

### 8.7 V2 content-free calibration

V2 对每个 session、每个 target 先做一次无视频、无 dialog 的基线打分，只保留 query 和 oracle plan：

```text
content_free_score(candidate)
```

正式观察分数为：

```text
observed_score(candidate)
```

校准后：

```text
calibrated_score
= observed_score
- content_free_score
```

直观上是减掉“模型在没看视频时本来就偏爱这个数字多少”，留下视频与 dialog 使候选额外增加的分数。

没有在结果出现后搜索温度、映射、数字排列或阈值。

### 8.8 两个 view 的目的

```text
official_dialog
    使用完整当前可见 prior dialog

no_assistant_history
    删除所有历史 assistant turns，其他输入不变
```

这个对比用于判断：状态预测主要依赖视频，还是主要依赖历史助手说过多少话。

### 8.9 结果怎样看

| View | Step Macro | Progress Macro | Error Macro | Mean task Macro |
|---|---:|---:|---:|---:|
| official dialog | 0.2226 | 0.1348 | 0.5098 | **0.2891** |
| no assistant | 0.2601 | 0.0167 | 0.2024 | **0.1597** |

预先规定：

```text
>= 0.45      强 zero-shot signal
0.35 - 0.45 弱但可用
< 0.35       不足
```

两个 view 都低于 0.35，结论是 `insufficient_zero_shot_signal`。

### 8.10 为什么 no-assistant 的 step accuracy 0.475 仍然很差

No-assistant 预测了 68 次 s1。Gold 中 s1 有 39 次，因此只靠大量猜 s1，就能得到看似不低的 accuracy。

但其他步骤几乎不会预测：

```text
s1=68, s2=2, s3=3, s4=7
```

Macro F1 给每个步骤同等权重，因此只有 0.2601。这个例子说明类别不均衡时不能只看 accuracy。

### 8.11 Official dialog 发生了什么塌缩

Official view 预测分布：

```text
step:     64/80 预测 s4
progress: 70/80 预测 complete
error:    71/80 预测 present
```

No-assistant 则几乎相反：

```text
step:     68/80 预测 s1
progress: 69/80 预测 not_started
error:    74/80 预测 absent
```

这说明模型把“有很多 assistant history”理解成接近任务末端，把“没有 assistant history”理解成任务刚开始。它没有稳定地区分视频中的具体步骤。

### 8.12 Error accuracy 0.75 为什么也不能算成功

80 个 gold 中 error present 有 65 个、absent 只有 15 个。Official view 预测 present 71 次，因此多数类 accuracy 很高。

但 absent 类 F1 只有 0.1667，error Macro F1 只有 0.5098。它主要是在猜多数类。

### 8.13 Progress Macro 的额外注意

正式 80-state 集中没有 `not_started` gold，但协议固定保留五个 progress 类，该类 F1 计为 0。这会使 Macro 更严格。

即使不纠结这个无支持类，模型也只正确识别了 2/47 ongoing，并且 0/10 recovered 被预测为 recovered，所以“progress 解码不足”的结论不会因此改变。

### 8.14 Dialog 有信号，但不是可靠 state

Official 相对 no-assistant 的 composite correctness 提升 `+0.1750`，bootstrap 95% 区间 `[+0.1125,+0.2375]`。

这证明 prior assistant turns 确实包含强阶段信息，但模型利用方式过于粗糙：

```text
有 assistant history -> 末端 / complete / error present
无 assistant history -> 起点 / not_started / error absent
```

它并没有形成可部署的 current-step tracker。

### 8.15 Confidence 为什么不可信

No-assistant mean confidence 高达 0.7876，但没有任何一个 state 的 step、progress、error 三字段全部正确。

因此 softmax confidence 明显过度自信，不能直接当作 planner confidence 或决策门控信号。

## 9. S1：下一步监督状态解码器，目前还没有实验结果

### 9.1 为什么从 S0 转向 S1

S0 失败说明冻结 1B 模型无法仅靠 prompt 和候选概率零样本读出细粒度状态，但不代表 hidden 中完全没有状态信号。

D3 已证明 frozen hidden dynamics 对决策有用。S1 要测试：

> 给少量明确 state supervision，一个很小的线性 decoder 能否从 D1/D3 表示中恢复 step、progress 和 error？

### 9.2 新数据与拆分

S1 选择全新的 32 sessions / 444 contiguous states：

```text
Train     24 sessions / 318 states
Held-out   8 sessions / 126 states
```

四个 domain 均衡，并覆盖 short、middle、long session。排除 U1 formal 20 sessions 和旧 R1 四个 sessions。

选样时不读取 answers、R0/D1/D3 outputs、errors、ratings 或 state labels。

### 9.3 为什么必须标注连续所有 chunk

U1 每个 session 只抽四个位置，适合语言对比，但不适合学习状态转移。

S1 标注连续 chunk，才能观察：

- s1 何时转到 s2；
- ongoing 何时变 complete；
- deviated 何时进入 recovered；
- 模型是否出现一步滞后、跳步或状态抖动。

### 9.4 三个冻结 control

| Variant | 输入 | 要回答的问题 |
|---|---|---|
| temporal_only | 7 个 D1 时间特征 | 只靠位置能猜到多少 state？ |
| current_d1 | 当前 1043 维 D1 表示 | 当前视频/dialog 表示是否超过时间先验？ |
| d3_dynamics | 2075 维 D3 当前+动态表示 | 跨 chunk 变化是否能稳定恢复 state？ |

全部使用标准化线性模型和 class-balanced cross entropy，不增加 MLP、RNN、PCA、平滑或后验特征搜索。

只有 `d3_dynamics` 有资格通过 S1 gate。若它不能稳定超过 temporal-only，就说明所谓 state gain 仍可能只是位置规律。

### 9.5 当前状态

目前已经完成：

- 选出 32 sessions；
- 固定 24/8 train-held-out split；
- 生成净化输入和标注模板；
- 冻结模型变体、L2 网格、指标和 promotion gate。

尚未完成：

- 444 个连续 state 的人工标注；
- state decoder 训练；
- 8-session held-out 评价。

因此不能把 S1 写成“新实验已成功”。它现在只是下一阶段准备。

## 10. 把所有结果连起来分析

### 10.1 决策方面

```text
D1 0.6341
  |
  +-- 更复杂 head：D2 residual 0.6351，失败
  |
  +-- 最后层 LoRA：D2 adapted 0.6357，失败
  |
  +-- 跨 chunk dynamics：D3 0.6690，通过
```

这说明跨时间信息比继续堆 head 容量更有价值。

但 D3 的“时间信息”混合了两部分：

```text
可能的视觉/程序变化
+ 官方 dialog 中上一 gold action 的痕迹
```

目前无法把 0.0349 增益全部归因于第一部分。

### 10.2 语言方面

```text
D1/D3 gate 决定是否说话
        |
        +-- R0 有正文：复用正文
        |
        +-- R0 silent：硬编码 fallback
```

U0 已经证明 D1 超过一半的介入是 fallback。U1 证明强制生成能让 56/80 典型 fallback 样本产生正文，但 24/80 仍为空。

Oracle state 改变了很多文本，却没有自动证据证明更正确。人工盲评是当前真正的语言路线门槛。

D3 仍使用同一 `decision_answer` 输出组装，因此它提高 decision Macro 的同时，并没有自动修复语言内容。D3 新增 interrupt 可能产生更多 fallback，这一数量尚未正式审计。

### 10.3 状态方面

PWR 路线的核心不是在 prompt 中塞一段计划文字，而是持续维护可靠的程序状态。

当前证据是：

- 人工 oracle state 可以作为 U1 的上界输入；
- 冻结模型 zero-shot 不能可靠预测这个 state；
- dialog 中存在阶段信号，但粒度粗且容易塌缩；
- 是否能通过监督从 D3 表示中恢复 state，要等 S1。

所以现在不能直接进入 granularity 或 GRPO。若基础 state 都不可预测，讨论粗、中、细计划粒度或用 RL 优化 planner 都缺少可靠接口。

## 11. 当前最合理的后续顺序

1. 完成 U1 interface package 和 state package 的双人盲评。
2. 对冻结 D3 做 dialog-policy/self-fed robustness 审计，不重新调参。
3. 对 D3 输出做 fallback 与具体 utterance 全量审计。
4. 完成 S1 的 444-state 因果标注和一次冻结 held-out 实验。
5. 只有 U1 证明 state 改善语言、且 S1 证明 state 可预测时，才进入 predicted-state utterance。
6. 只有显式 state 在独立 held-out 上有增益，才研究 granularity。
7. GRPO 最后考虑，用来优化已经明确的残差问题，而不是替代尚未建立的状态监督。

## 12. 最终应当怎样表述当前进展

可以严谨地说：

> 我们先后排除了小型非线性决策头和最后层 LoRA 作为主要增益来源；随后通过预注册五折实验发现，当前与历史多模态表征的严格因果动态可将官方 OOF Macro F1 从 0.6341 提升到 0.6690，且在五折和四个领域均为正。进一步审计表明，该收益包含官方 prior-dialog 对上一 gold intervention 的编码，因此它是有效的官方协议决策提升，但不是纯视觉程序理解已经解决的证据。与此同时，U0/U1 将决策与语言内容拆开，确认 D1 的大量介入使用硬编码 fallback，强制生成和 oracle state 会改变正文，但内容收益仍待双人盲评。S0 则显示冻结 1B 模型的 zero-shot 细粒度状态解码不足，下一步进入独立监督的 S1，而不是直接进入 granularity 或 GRPO。

不能说：

- D3 已经解决 proactive planning；
- 0.6690 是隐藏测试结果；
- `Please continue with the next step.` 是模型规划后的自然语言；
- oracle state 已经被证明有用或无用；
- S1 已经完成；
- D3 full-fit 0.7544 是可报告的泛化分数。

## 13. 对应正式材料

- D2 residual：`reports/20260715_internvl35_1b_residual_mlp_d2.md`
- D2 LoRA：`reports/20260716_internvl35_1b_final_mlp_lora_oof.md`
- D3 dynamics：`reports/20260717_internvl35_1b_causal_dynamics_d3.md`
- U0 utterance audit：`reports/20260716_d1_utterance_u0_audit.md`
- U1 oracle generation：`reports/20260717_u1_formal_blind_oracle_generation.md`
- S0 state decoding：`reports/20260717_internvl35_1b_oracle_plan_state_s0.md`
- 当前路线：`CURRENT_ROUTE.md`
