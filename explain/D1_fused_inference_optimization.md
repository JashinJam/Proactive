# D1 融合推理等价加速讲解

> 更新时间：2026-07-15  
> 正式报告：20260715_internvl35_1b_d1_inference_optimization  
> 当前推广方案：shared_vision  
> 正确性参考：sequential  
> 已否决方案：batched、prefix_cache  
> 状态：等价性、速度和显存门槛均通过，shared_vision 已成为当前部署路径  
> 说明：本阶段只优化执行方式，不改变模型、决策策略或 OOF 0.6341。

## 1. 一句话理解

D1 原来为了得到 tag margin 和 hidden，需要对完全相同的视频分别跑一次 silent 候选和一次 interrupt 候选。

两个候选的图像完全相同，只有最后三个标签 token 不同。shared_vision 因此把视频只编码一次，然后把同一份视觉特征分别交给两次原样的语言模型候选前向。

它删除了一次重复视觉计算，同时故意保留两次语言计算，所以：

- hidden 完全不变；
- tag margin 完全不变；
- 最终 decision 和 answer 完全不变；
- 127-chunk 墙钟时间减少 9.15%；
- 峰值显存不增加。

## 2. 什么叫融合推理

D1 最终 fused head 使用：

~~~text
18 个 causal scalar
+ 1 个 tag margin
+ 1024 维 causal hidden
~~~

在线推理时，每个 chunk 必须现场得到这三类特征：

1. R0 generation 产生 raw response；
2. 根据当前 interval、domain、帧数和 raw response 构造 18 个 scalar；
3. 对 silent 和 interrupt 两个固定标签候选评分；
4. 从候选评分前向中读取 tag margin 和 hidden；
5. 由 1,044 参数 head 计算最终 logit。

这里的 inference 是“使用已经训练好的冻结系统产生预测”，不是训练。

## 3. 什么叫等价加速

等价加速不是：

> 换一种更快的方法，只要当前几个最终答案看起来一样就行。

它要求：

> 删除重复计算，但保持同一模型、同一特征、同一 logit 和同一最终预测。

本项目要求候选满足：

| 检查 | 要求 |
|---|---|
| R0 raw response | 字符串完全一致 |
| 1024 hidden | 最大绝对差不超过 1e-6 |
| tag margin | 最大绝对差不超过 1e-6 |
| fused logit | 最大绝对差不超过 1e-6 |
| interrupt/silent decision | 完全一致 |
| final answer | 完全一致 |
| predictions JSONL | 逐字节一致 |
| official metrics JSON | 逐字节一致 |
| session 计算时间 | 至少改善 5% |
| peak memory | 增幅不超过 10% |

equivalence gate 是等价性门槛。任何一项核心特征超出容差，都不能把方案称为同一冻结策略。

## 4. 为什么中间特征也必须等价

最终决策是：

~~~text
logit = linear_head(
    scalar,
    tag_margin,
    hidden
)

decision = interrupt if logit >= threshold else silent
~~~

假设某个优化让 logit 从：

~~~text
0.1250 -> 0.1254
~~~

而 threshold 是：

~~~text
0.125605...
~~~

当前这个 chunk 可能仍然没有跨过 threshold。但另一个非常接近边界的 chunk 可能会改变类别。

因此仅检查 10 个 decision 相同不够。还必须检查 head 真正使用的 margin、hidden 和 logit。

## 5. 原始 Sequential 路径

sequential 中文是“顺序执行”。它是生成正式神经缓存和训练融合头时使用的 correctness oracle，即正确性参照。

每个 chunk 的高层计算：

~~~text
第一部分：R0 自由生成
  frames
    -> vision tower
    -> projector
    -> language model
    -> autoregressive decode
    -> raw response

第二部分：silent 候选评分
  相同 frames
    -> vision tower
    -> projector
    -> language model(prompt + $silent$)
    -> silent logp + causal hidden

第三部分：interrupt 候选评分
  相同 frames
    -> vision tower
    -> projector
    -> language model(prompt + $interrupt$)
    -> interrupt logp + causal hidden
~~~

所以每个 chunk 至少重复执行三次视觉编码：

~~~text
R0 generation 一次
silent candidate 一次
interrupt candidate 一次
~~~

候选评分部分的视频和 prompt 几乎完全相同，只是末尾三个标签 token 不同。

## 6. 模型内部计算怎样分层

InternVL 可以粗略拆成：

~~~text
pixel frames
    |
    v
Vision tower
提取每帧的视觉 patch 特征
    |
    v
Multimodal projector
把视觉特征转换成语言模型的 1024 维空间
    |
    v
Projected image features
替换 prompt 中的 IMG_CONTEXT 占位位置
    |
    v
Language model
结合视觉、system、query、dialog 和候选标签
    |
    v
Hidden states
    |
    v
LM head
把 hidden 转成词表 logits
~~~

术语解释：

- vision tower：视觉编码器；
- patch：图像被切分的小块；
- projector：把视觉向量转换到语言模型维度的投影层；
- embedding：token 在模型中的连续向量；
- language model：处理整个多模态 token 序列的 28 层 Qwen3；
- LM head：把 1,024 维 hidden 映射成 151,936 个词表分数；
- logits：softmax 之前的词表分数。

## 7. 为什么重复视觉计算值得优化

每个模型输入最多有 32 帧，每帧 448 x 448。

每帧经过视觉塔后形成 256 个投影视觉 token：

~~~text
32 frames x 256 = 8192 visual tokens
~~~

silent 和 interrupt 候选看到的 32 帧完全相同，所以顺序参考路径会对同一批图像重复做两次 vision tower 和 projector。

删除这一份重复计算不会改变候选之间应有的差异，因为候选只在最后三个文字 token 上不同。

## 8. 为什么不能简单复用整个 R0 Generation

R0 generation 也已经编码过同一批视频，但它通过 Transformers 的 generate 流程完成：

- prompt prefill；
- KV cache 建立；
- 逐 token 自回归生成；
- 动态停止；
- raw response 解码。

候选评分则需要：

- 固定的完整 silent 标签分数；
- 固定的完整 interrupt 标签分数；
- 标签前最后一个 prompt hidden；
- 与正式离线缓存完全一致的数值路径。

直接把 R0 generate 内部的视觉或 KV 状态取出来跨接口复用，会进一步侵入 generate 内部实现，正确性风险更高。本轮先只消除两个固定候选之间最明确的重复视觉计算。

因此 shared_vision 后每个 chunk仍有：

~~~text
R0 generation 的一次视觉编码
+ 两个候选共享的一次视觉编码
= 总共两次视觉编码
~~~

而不是降到一次。

## 9. 候选方案一：Batched 双候选

batched 表示把多条样本组成一个 batch 并行送入模型。

方案是：

~~~text
batch row 0 = prompt + $silent$
batch row 1 = prompt + $interrupt$
~~~

然后一次调用 model.model。

表面上：

~~~text
两次 Python model call -> 一次 Python model call
~~~

但实际张量是：

~~~text
videos = [frames, frames]
batch size = 2
~~~

视觉塔仍然处理两份重复视频，语言模型仍然处理两条约 4k 到 8.8k token 的长序列。

## 10. 为什么 Batch 反而更慢

一次 API 调用不等于一份固定计算量。

Sequential：

~~~text
先处理一条长序列
释放或复用中间空间
再处理第二条长序列
~~~

Batched：

~~~text
同时保留两条长序列的激活
同时保留两份视觉输入
同时处理两份 attention
~~~

它减少的主要是 Python 调用和少量调度开销，却没有减少核心 FLOPs，反而提高峰值内存和单次 kernel 压力。

在当前 A800、batch=1 已能有效利用 GPU、序列很长的条件下，batch=2 没有获得并行收益。

## 11. Batched 实测结果

首个 10-chunk session：

| 项目 | Sequential | Batched | 变化 |
|---|---:|---:|---:|
| Session 计算时间 | 35.05 s | 41.40 s | 慢 18.12% |
| 完整墙钟 | 43.258 s | 50.772 s | 慢 17.37% |
| 峰值显存 | 3.466 GB | 4.145 GB | 增加 19.59% |
| Hidden | 完全一致 | 完全一致 | 通过 |
| Tag margin | 完全一致 | 完全一致 | 通过 |
| Decision / answer | 完全一致 | 完全一致 | 通过 |

它是数值等价的，但不是加速，因此否决。

## 12. 候选方案二：Prefix KV Cache

prefix 是两个候选共享的前缀：

~~~text
视频 + system + query + dialog + assistant 起始位置
~~~

两个候选只在末尾不同：

~~~text
共享 prefix + $silent$
共享 prefix + $interrupt$
~~~

Transformer 每层 attention 会为每个历史 token 计算 key 和 value。KV cache 保存这些历史 K/V，使后续 token 不必重新计算整个前缀。

这是大语言模型逐 token 生成加速的常见机制。

## 13. Prefix Cache 方案怎样工作

第一步：

~~~text
完整运行 prompt + $silent$
开启 use_cache
得到：
  causal hidden
  silent logp
  每层 KV cache
~~~

第二步：

~~~text
把 cache 裁剪到标签开始前
只把很短的 $interrupt$ 标签 continuation 送入模型
得到 interrupt logp
~~~

prefill 表示第一次处理完整长前缀并建立 cache。continuation 表示在 cache 后继续处理少量新 token。

从理论计算量看，它应当删除：

- 第二次视觉编码；
- 第二次完整长 prompt 语言前向。

所以它是三种方案中理论加速潜力最大的。

## 14. 为什么数学等价不代表 GPU 数值完全相同

在精确实数数学中：

~~~text
一次处理完整 prompt + interrupt
~~~

应等价于：

~~~text
先处理 prompt 并缓存
再处理 interrupt 后缀
~~~

但 GPU 使用 BF16 和有限精度浮点。浮点加法不满足严格结合律：

~~~text
(a + b) + c
可能不完全等于
a + (b + c)
~~~

SDPA 会根据输入形状选择不同 kernel 或数值路径：

- 完整长序列 forward；
- 长 KV cache 加短 query。

两种路径执行归约、分块和累加的顺序可能不同，因此最终 log probability 会出现漂移。

## 15. Prefix Cache 的漂移在哪里

实测：

- causal hidden 最大差为 0；
- silent score 完全一致；
- interrupt 后缀 score 发生变化；
- tag margin 最大差为 0.113382；
- fused logit 最大差为 3.7451e-4。

hidden 和 silent score 来自原始完整 silent 前向，所以保持一致。漂移全部来自 cache continuation 计算的 interrupt 标签。

10 个最终 decision 恰好都没有跨 threshold，但这只是小样本巧合。

## 16. Prefix Cache 为什么仍被否决

预设门槛要求：

~~~text
tag margin 差 <= 1e-5
logit 差      <= 1e-6
session 时间至少改善 10%
~~~

实际：

~~~text
margin 差 = 0.113382
logit 差  = 0.0003745
session 时间反而慢 2.23%
完整墙钟反而慢 6.02%
~~~

它既不等价，也没有加速。

可能的工程开销包括 cache 建立、保存、裁剪和短 query kernel 调度。但正式结论只依据实测：当前 Transformers、PyTorch、SDPA、BF16 和 A800 组合下，该实现不满足门槛。

## 17. 为什么不能只接受“最终标签一样”

最终 head 是在 sequential 特征上训练和校准的。

如果换成漂移后的 margin：

~~~text
训练特征分布 != 部署特征分布
~~~

这相当于部署时悄悄改变输入特征定义。

即使当前 10 个 decision 相同，也不能保证：

- 其他 9,925 个 public chunks 不变；
- hidden test 中靠近 threshold 的 chunk 不变；
- official Macro F1 不变。

因此 prefix_cache 被保留为 rejected control，而不是部署候选。

## 18. 候选方案三：Shared Vision

shared_vision 只共享视觉塔与 projector 的输出，不共享语言模型前缀。

结构：

~~~text
frames
  |
  v
vision tower + projector
只执行一次
  |
  +----------------------------+
  |                            |
  v                            v
silent input embeddings        interrupt input embeddings
  |                            |
  v                            v
原始 batch=1 language forward  原始 batch=1 language forward
  |                            |
silent logp + hidden            interrupt logp + hidden
~~~

它减少一份重复视觉计算，但保留两个候选原始的语言张量形状和计算顺序。

## 19. Shared Vision 如何构造两个输入

两个候选标签都恰好是三个 token：

~~~text
$silent$    -> 3 tokens
$interrupt$ -> 3 tokens
~~~

代码先让 processor 构造：

~~~text
prompt + $silent$
~~~

得到：

- input_ids；
- attention_mask；
- pixel_values；
- 视觉占位 token。

然后 clone 一份 input_ids，只替换最后三个 token：

~~~text
silent_input_ids
  clone
    -> 把末尾 silent 三 token 换成 interrupt 三 token
    -> interrupt_input_ids
~~~

因为标签等长：

- 总 sequence length 不变；
- prompt length 不变；
- attention mask 不变；
- 视觉 token 位置不变；
- position 编号不变。

## 20. 视觉特征怎样只算一次

代码显式调用：

~~~text
image_features =
    model.get_image_features(pixel_values)
~~~

这一步内部执行：

~~~text
vision tower
+ pixel shuffle
+ multimodal projector
~~~

得到 projected image features。

随后两个候选分别构造普通文字 embedding，再找到 IMG_CONTEXT 占位位置：

~~~text
text_embeddings = token_embedding(input_ids)
image_mask       = input_ids 中视觉占位的位置
inputs_embeds    = 把 image_features 填到 image_mask
~~~

silent 和 interrupt 都填入同一份 image_features。

## 21. 为什么语言模型仍然跑两次

silent 与 interrupt 的最后三个 token 不同。为了严格保持原 sequential 数值路径，代码分别运行：

~~~text
language_model(silent inputs, batch=1)
language_model(interrupt inputs, batch=1)
~~~

每次的：

- batch size；
- sequence length；
- attention mask；
- token positions；
- input dtype；
- language model API；
- use_cache=False；

都与 sequential 候选路径保持一致。

因此不会触发 batch=2 或 cache continuation 的不同数值路径。

## 22. Shared Vision 为什么能够精确等价

视觉部分：

- 两个候选本来输入相同 pixel_values；
- 模型处于 eval 和 inference_mode；
- 视觉计算是确定性的；
- 算一次再复用，与分别算两次得到相同 projected features。

语言部分：

- 仍分别以 batch=1 执行；
- 完整序列仍一次性前向；
- tensor shape 和 attention 路径不变。

所以既删除了真正重复的视觉工作，又没有改变对数概率最敏感的语言计算方式。

## 23. 首个 Smoke 结果

smoke test 是小规模烟测，用于快速确认实现是否能跑、是否有明显错误。

首个 10-chunk session：

| 项目 | Sequential | Shared vision | 变化 |
|---|---:|---:|---:|
| Session 计算时间 | 35.05 s | 28.15 s | 快 19.69% |
| 完整墙钟 | 43.258 s | 36.377 s | 快 15.91% |
| 峰值显存 | 3,466,037,248 B | 3,466,037,248 B | 不变 |
| Hidden / margin | 完全一致 | 完全一致 | 通过 |
| Raw / decision / answer | 完全一致 | 完全一致 | 通过 |

一个 session 仍不足以推广，所以继续做扩展 benchmark。

## 24. Benchmark 怎样选择

benchmark 是性能基准测试。

为了覆盖不同负载，每个 domain 选择：

- chunk 数最少的一个 session；
- chunk 数最多的一个 session。

共：

~~~text
4 domains x 2 sessions = 8 sessions
127 chunks
~~~

选择只使用 domain 和 interval 数量，不看 labels、R0 错误或模型输出。

这里用完整 session 长度选择压力测试样本，不属于因果泄漏，因为长度只用于决定测哪些工程负载，没有作为任何 chunk 的模型特征。

## 25. 为什么需要长短 Session

短 session 中：

- 模型加载；
- processor；
- 视频打开和 seek；
- Python 调用；
- 文件写入；

这些固定开销占比较高，少量 GPU 加速容易被测量波动掩盖。

长 session 中重复候选视觉计算更多，更能显示 shared_vision 的累计收益。

## 26. 扩展 Benchmark 结果

8 sessions 中 6 个更快。两个极短 session 出现 6% 到 8% 的波动。

聚合结果：

| 项目 | Sequential | Shared vision | 改善 |
|---|---:|---:|---:|
| Session 计算时间总和 | 492.47 s | 446.61 s | 9.31% |
| 完整墙钟 | 500.892 s | 455.056 s | 9.15% |
| 峰值显存 | 3,466,463,232 B | 3,466,463,232 B | 0% |

session compute time 是逐 session 核心计算时间之和。

wall time 是程序从启动到结束的完整现实时间，还包含初始化、读写和其他外层开销。

## 27. 为什么只快约 9%，不是快 33%

Sequential 的高层三部分是：

~~~text
R0 generation
silent candidate
interrupt candidate
~~~

shared_vision 只删除：

~~~text
interrupt candidate 中重复的 vision tower + projector
~~~

它没有删除：

- R0 generation；
- R0 自回归 decode；
- silent language forward；
- interrupt language forward；
- 视频解码和抽帧；
- prompt tokenizer；
- scalar 构造；
- head 打分；
- 文件读写。

语言模型处理 4k 到 8.8k token 的两次完整序列仍然很重，所以总体改善约 9% 是合理的。

## 28. 127 Chunk 等价性结果

在全部 127 chunks：

- raw response 127/127 完全一致；
- 1,024 维 hidden 127/127 逐元素完全一致；
- hidden 最大差为 0；
- tag margin 127/127 完全一致；
- margin 最大差为 0；
- decision 127/127 完全一致；
- answer 127/127 完全一致；
- predictions JSONL 逐字节一致；
- official metrics JSON 逐字节一致；
- 最大 logit 审计差为 2.3187e-7，小于 1e-6。

byte-exact 表示文件的每个字节都相同，因此 SHA256 也相同。

## 29. 为什么 OOF 0.6341 不变化

本轮没有：

- 重新训练 head；
- 修改 feature 定义；
- 修改 threshold；
- 修改模型权重；
- 修改 prompt；
- 修改输出规则。

shared_vision 产生与 sequential 相同的 margin 和 hidden，所以对 1,044 参数 head 来说，输入完全相同。

因此：

~~~text
科学策略仍是 fused_linear OOF 0.6341
~~~

没有所谓“加速后的新分数”。加速只改变执行时间。

## 30. 三个候选的最终结论

| 方案 | 数值等价 | 更快 | 显存 | 决定 |
|---|---|---|---|---|
| sequential | 参考 | 参考 | 3.466 GB | correctness oracle |
| batched | 是 | 否，慢 18.12% | 增加 19.59% | 否决 |
| prefix_cache | 否，margin 漂移 | 否，慢 2.23% | 近似不变 | 否决 |
| shared_vision | 是，127/127 | 是，快 9.31% | 不变 | 推广 |

## 31. 当前应该使用哪个配置

部署路径：

~~~text
configs/d1_internvl35_1b_neural_deploy_shared_vision.json
~~~

正确性参照：

~~~text
configs/d1_internvl35_1b_neural_deploy.json
~~~

batched 和 prefix_cache 配置及产物保留用于审计，防止未来重复尝试，但不能作为部署路径。

## 32. 为什么还保留 Sequential

优化代码可能受：

- Transformers 版本；
- PyTorch 版本；
- CUDA；
- SDPA kernel；
- 模型内部 API；

影响。

sequential 是训练缓存使用的原始路径，结构最直接。以后任何库升级或新优化都必须重新与它比较。

因此：

~~~text
shared_vision = 当前部署实现
sequential    = 永久正确性参照
~~~

## 33. Transformers 版本风险

shared_vision 不再只调用顶层 model.model，而是显式调用内部组件：

- get_image_features；
- get_placeholder_mask；
- get_input_embeddings；
- language_model；
- lm_head。

这些内部接口和默认行为可能随 Transformers 版本改变。

submission/container 必须固定：

- Python；
- PyTorch；
- Transformers；
- CUDA；
- 模型 revision；
- processor 配置。

升级后必须重新运行至少 127-chunk 等价 benchmark。

## 34. 为什么正式运行不保存 Hidden 到 JSON

等价审计时使用 record_hidden_state，把每个 chunk 的 1,024 个 hidden 数字写入 session_records，便于逐元素比较。

正式测试不需要把 hidden 交给官方，只需要在内存里立即计算 logit。

关闭记录：

- 不改变 hidden 计算；
- 不改变 decision；
- 减少 JSON 体积；
- 减少序列化和磁盘写入。

所以正式推理不应添加 record-hidden-state。

## 35. 本阶段没有改变因果性

shared_vision 仍然只使用：

- 当前和过去帧；
- 当前 query；
- 官方 dialog[i]；
- 当前 interval 元数据；
- 当前 causal prompt。

它只是让 silent 和 interrupt 候选复用同一份当前视觉特征，没有引入：

- 未来帧；
- 未来 dialog；
- gold label；
- 完整 session 长度特征；
- 跨 chunk 未来 cache。

## 36. 当前还没有优化什么

shared_vision 没有优化：

- R0 generation 与候选评分之间的视觉共享；
- 两个候选的公共语言 prompt 计算；
- 跨 chunk 的历史视觉编码复用；
- 视频解码和 seek；
- 多 session batch；
- 异步 CPU/GPU pipeline；
- container 启动和模型加载。

其中公共语言 prompt 的 prefix cache 已做过一种实现，但因数值漂移和速度失败被否决。未来若有不同后端或精确 kernel，可以重新提出，但必须重新过等价门槛。

## 37. 下一步为什么是 Submission/Container Audit

当前已经具备：

- 最终单一 1,044 参数 head；
- causal 在线 runner；
- shared_vision 等价加速；
- 127-chunk 正确性和性能证据；
- Small 参数合规。

下一阶段重点不再是继续微调速度，而是确认官方测试环境能够完整运行：

1. 官方究竟接收 predictions、模型目录还是 container；
2. 模型和 processor 能否完全离线加载；
3. 依赖版本是否锁定；
4. shared_vision 是否接入官方输入入口；
5. 路径、权限和输出 schema 是否正确；
6. 容器时限和 GPU 环境是否满足；
7. 在用户授权前不上传或消耗提交次数。

## 38. 最准确的结论

错误说法：

> 把两个候选做成 batch，所以 D1 加速了。

正确说法：

> Batch 虽然数值等价，却更慢、更占显存，已被否决。正式加速来自两个候选共享一次 projected vision representation，同时保留两次原 batch=1 语言前向。

错误说法：

> Prefix cache 最终标签一样，所以也是等价的。

正确说法：

> Prefix cache 在 10 chunks 上最终标签碰巧一样，但 tag margin 最大漂移 0.1134，且速度更慢，因此不满足冻结 head 的特征等价定义。

错误说法：

> Shared vision 把 D1 分数提高了。

正确说法：

> Shared vision 不改变 0.6341 策略和任何预测，只把 127-chunk 完整墙钟从 500.892 秒降到 455.056 秒。

## 39. 术语速查

| 英文术语 | 中文解释 |
|---|---|
| inference | 用冻结系统产生预测 |
| optimization | 在目标不变时减少时间或资源消耗 |
| equivalent | 特征、决策和输出保持在冻结容差内一致 |
| sequential | 两个候选依次单独运行 |
| batch | 把多条输入堆在一起并行计算 |
| shared vision | 多个候选复用同一份视觉编码结果 |
| prefix | 多个候选完全相同的前半段输入 |
| prefill | 首次处理完整前缀并建立内部状态 |
| KV cache | 保存 Transformer 历史 key/value 的缓存 |
| continuation | 在已有 cache 后继续处理的新 token |
| vision tower | 把像素转换为视觉特征的编码器 |
| projector | 把视觉特征映射到语言模型维度 |
| projected features | 已转换到语言模型空间的视觉向量 |
| embedding | token 对应的连续向量 |
| attention mask | 标明哪些 token 可参与注意力计算 |
| SDPA | scaled dot-product attention 的 PyTorch 实现 |
| BF16 | 16 位脑浮点数值格式 |
| numerical drift | 不同计算路径引起的小数值变化 |
| tolerance | 允许的最大数值误差 |
| correctness oracle | 新实现必须对齐的正确性参考 |
| smoke test | 小规模快速运行检查 |
| benchmark | 用固定负载比较性能的基准测试 |
| latency | 完成一次任务所需时间 |
| wall time | 从程序启动到结束的现实经过时间 |
| peak memory | 运行期间显存占用的最高值 |
| byte-exact | 两个文件逐字节完全相同 |
| promotion gate | 候选成为正式路径前必须通过的门槛 |
| rejected control | 已测试并明确否决、保留作审计的对照方案 |
| container audit | 检查依赖、入口、环境和离线运行的提交封装审计 |

## 40. 对应材料

- [正式加速报告](../reports/20260715_internvl35_1b_d1_inference_optimization.md)
- [D1 特征提取与优化代码](../src/proactive_d1/internvl_features.py)
- [Shared-vision 部署配置](../configs/d1_internvl35_1b_neural_deploy_shared_vision.json)
- [Sequential 参考配置](../configs/d1_internvl35_1b_neural_deploy.json)
- [Batched 否决配置](../configs/d1_internvl35_1b_neural_deploy_batched.json)
- [Prefix-cache 否决配置](../configs/d1_internvl35_1b_neural_deploy_prefix_cache.json)
- [Sequential 127-chunk 参考产物](../output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_sequential_v2_benchmark8/)
- [Shared-vision 127-chunk 产物](../output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_shared_vision_v1_benchmark8/)
- [Batched 否决产物](../output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_batched_v1_smoke1/)
- [Prefix-cache 否决产物](../output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_prefix_cache_v1_smoke1/)

