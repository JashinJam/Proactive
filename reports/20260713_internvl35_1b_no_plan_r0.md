# C1 Small R0 完整技术与实验报告

> 实验 ID：`20260713_internvl35_1b_no_plan_r0`  
> 状态：**完成并冻结**  
> 正式完成时间：2026-07-14 12:46 CST  
> 目标赛道：ECCV Wearable AI Challenge，Challenge 1 EgoProactive，Small division  
> 正式产物：[`output/experiments/20260713_internvl35_1b_no_plan_r0/`](../output/experiments/20260713_internvl35_1b_no_plan_r0/)  
> 冻结预测 SHA256：`312d0375dd67be2fb244d622a9f302734082f0f251ef0b8dd190b00880879820`

## 阅读导航

- [快速结论](#1-报告结论)
- [任务、数据与标签边界](#3-任务数据与评测边界)
- [模型架构与参数审计](#4-使用模型)
- [冻结配置](#5-冻结配置)
- [完整推理管线](#7-完整推理管线)
- [模块设计](#8-模块设计)
- [失败尝试与正式执行](#10-smoke失败尝试与正式运行决策)
- [正式结果与误差分析](#13-正式结果)
- [R1 交接约束](#16-对-r1-的交接约束)
- [复现命令与产物](#17-可复现命令)

## 1. 报告结论

R0 建立了当前项目第一条满足 Small 参数限制、因果输入约束、完整 700-session 评测、官方 scorer、模型与数据指纹审计、可恢复推理和完整产物记录的 no-plan 基线。

正式结果如下：

| 指标 | 结果 |
|---|---:|
| Macro F1，官方主指标 | **0.4630** |
| G-mean F1 | 0.4541 |
| Interrupt Precision / Recall / F1 | 0.5286 / **0.2879** / 0.3728 |
| Silent Precision / Recall / F1 | 0.4571 / 0.7002 / 0.5531 |
| TP / FP / TN / FN | 1541 / 1374 / 3209 / **3811** |
| 预测 interrupt rate | 29.34% |
| Gold interrupt support | 53.87% |
| Sessions / chunks / skipped chunks | 700 / 9,935 / 0 |

最重要的结论不是单独的 `0.4630`，而是错误结构：

1. R0 明显少说话。预测 interrupt rate 只有 29.34%，而标签中 interrupt 占 53.87%，主要损失来自 3,811 个 FN。
2. 错误高度集中在 session 开始。700 个首 chunk 中 699 个标签是 interrupt，R0 只输出了 50 个合法 interrupt。
3. 633 个 raw response 没有以合法 tag 开头，按官方规则全部作为 silent；其中 628 个发生在首 chunk，630 个对应 gold interrupt。多数 raw response 实际是自然语言指导，只是缺少 `$interrupt$` 前缀。
4. 除去格式不合法样本后，格式合法子集上的条件 Macro F1 为 0.4941，但这是不同样本分布上的诊断，不能替代官方全量分数。
5. 领域差异明显：Chef 为 0.4863、Arts and Crafts 为 0.4667，而 Handyman 仅 0.4247，后者预测 interrupt rate 只有 12.96%。
6. R0 在 session 中段和后段的 Macro F1 约为 0.52，最前 20% 仅为 0.3042。这说明当前失败既有格式问题，也有起始状态和“何时开始指导”的建模问题。

因此，R0 同时给后续工作划出了两个必须隔离的方向：

- R1 科学问题：compact procedural state 是否能在保持 silent precision 的同时提高 interrupt recall。
- 独立工程问题：输出格式和首 chunk policy 如何修复。该问题不能混入 R1 后再把收益归因于 plan-state。

## 2. R0 要回答什么

R0 的假设是：

> 在不加入 plan、状态跟踪器、决策头、微调或 RL 的情况下，一个完全开放、符合 Small 限制的 InternVL3.5-1B 模型，配合官方因果输入和官方 prompt，可以给后续 plan-state 实验提供可复现的零点。

R0 是：

- no-plan、zero-shot 的完整 public-validation 推理；
- 同一模型、同一 prompt、同一帧策略、同一 dialog 策略、同一解码设置下的冻结参照；
- 后续 R1/R2/R3 对照实验必须复用的基线；
- 使用官方 Macro F1 scorer 得到的正式结果。

R0 不是：

- PWR 官方复现；
- 训练或微调实验；
- 使用 public validation 标签调过 prompt、threshold 或 class weight 的结果；
- held-out test 泛化成绩；
- 已提交排行榜的成绩；
- 闭环使用模型自身历史输出的在线 agent。

## 3. 任务、数据与评测边界

### 3.1 输入与输出

每个 session 的输入包含：

| 字段 | 用途 |
|---|---|
| `video_path` | 定位第一视角视频 |
| `video_intervals` | 每次决策对应的绝对 `[start, end]` 时间区间 |
| `query` | 用户的高层任务请求 |
| `dialog[i]` | 第 `i` 个 chunk 时官方提供的可见对话历史 |
| `domain` / `task` | 仅用于结果分组，不进入 prompt |
| `answers` | gold 标签，仅在预测冻结后由官方 scorer 使用 |

输出文件每个 session 一行：

```json
{"video_path": "example.mp4", "answers": ["$silent$", "$interrupt$Do the next step"]}
```

每个 chunk 必须输出：

- `$silent$`；或
- `$interrupt$<utterance>`，且 utterance 非空。

官方 scorer 只判断文本经 `lstrip()` 后是否以 `$interrupt$` 开头；其它任何文本都按 silent 处理。utterance 的内容质量不参与 C1 当前客观指标。

### 3.2 数据快照

| 项目 | 冻结值 |
|---|---|
| 数据仓 | `facebook/wearable-ai` |
| Split | `egoproactive/val` |
| JSONL | `data/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl` |
| JSONL 大小 | 7,817,242 bytes |
| JSONL SHA256 | `feef69ddee605e7070ad0f133636c35739c6964514a46d76da294b6bf1964740` |
| 视频目录 | `data/egoproactive/val` |
| 顶层许可 | CC-BY-NC-4.0 |
| Sessions | 700 |
| Chunks | 9,935 |

数据统计：

| 统计项 | 数值 |
|---|---:|
| 每 session chunks，min / median / mean / max | 4 / 14 / 14.19 / 30 |
| Interval 时长，min / median / mean / max，秒 | 0.1 / 8.0 / 7.28 / 364.1 |
| 相邻 interval 存在正 gap 的数量 | 2,820 |
| 正 gap，median / mean / max，秒 | 8.0 / 14.58 / 432.0 |

这些统计说明不能把数据简单视为连续的固定 8 秒滑窗。R0 始终按每个 interval 的绝对时间戳独立抽帧，正确保留短 interval、长 interval 和时间 gap。

### 3.3 领域构成

| Domain | Sessions | Chunks |
|---|---:|---:|
| Arts and Crafts | 175 | 2,391 |
| Chef | 175 | 2,952 |
| Handyman | 173 | 2,237 |
| Tutorial | 177 | 2,355 |

### 3.4 标签使用声明

R0 没有使用标签训练、微调、选择 prompt、选择阈值或选择 checkpoint。生成逻辑只访问：

```text
video_path, video_intervals, query, dialog
```

需要准确区分两个概念：

- JSONL loader 会解析整行，因此进程内存中的 row 对象物理上仍包含 `answers` 字段；
- `validate_source_rows()`、`process_session()`、消息构造和模型调用从不读取 `row["answers"]`。

所以报告中的“generation does not read gold answers”指生成逻辑没有访问或使用 gold，而不是声称输入文件在字节层面已被剥离标签。最终 scorer 只在 700 行预测全部冻结、合并并完成 SHA256 后读取 gold。

另外，`dialog[i]` 是官方输入的一部分，其中可能包含先前 chunk 的参考 assistant turn。R0 使用当前 chunk 官方提供的 causal dialog，而不是把自身先前预测回填进历史。因此它是官方输入协议下的逐 chunk 条件推理，不是 self-conditioned 闭环 rollout。

## 4. 使用模型

### 4.1 模型身份与 Small 合规性

| 项目 | 冻结值 |
|---|---|
| Hugging Face repo | `OpenGVLab/InternVL3_5-1B-HF` |
| Revision | `9191dbccf312b537016f041b25d61c72e7c5c9f3` |
| 本地路径 | `/home/lanjinxin/model_weights/InternVL3_5-1B-HF` |
| License | Apache-2.0 |
| 权重文件 | `model.safetensors` |
| 权重大小 | 2,121,890,856 bytes，约 1.98 GiB |
| 权重 SHA256 | `11effd1da2fc0929957d56de4129d1e7d2aed044ea878d40bf83e95b63d38b39` |
| 唯一参数总数 | **1,060,897,792** |
| Small 上限 | 2,000,000,000 |
| 上限占比 | 53.04% |
| 推理精度 | BF16 |

参数不是根据模型名估算，而是用 `safetensors.safe_open()` 遍历冻结权重中的每个唯一 tensor，按 shape 乘积求和。加载模型后又用 `sum(parameter.numel())` 复核一次，两者必须等于配置中的 1,060,897,792，否则运行终止。

R0 没有额外 learned gate、planner、adapter、LoRA、classifier 或外部模型，因此全部推理期学习参数就是该 checkpoint 的总参数。

### 4.2 参数构成

| 组件 | 参数量 |
|---|---:|
| Vision tower | 304,012,288 |
| Language model | 751,632,384 |
| Multimodal projector | 5,253,120 |
| 合计 | 1,060,897,792 |

### 4.3 架构细节

视觉编码器（Vision tower）：

| 配置 | 数值 |
|---|---:|
| 输入尺寸 | 448 × 448 |
| Patch size | 14 × 14 |
| Hidden size | 1,024 |
| Layers | 24 |
| Attention heads | 16 |
| MLP intermediate size | 4,096 |
| 激活 | GELU |
| Image sequence length | 256 |
| Downsample ratio | 0.5 |

语言模型（Language model）：

| 配置 | 数值 |
|---|---:|
| 架构 | Qwen3ForCausalLM |
| Hidden size | 1,024 |
| Layers | 28 |
| Attention heads / KV heads | 16 / 8 |
| Head dim | 128 |
| MLP intermediate size | 3,072 |
| Vocabulary size | 151,936 |
| Max position embeddings | 40,960 |
| 激活 | SiLU |
| RoPE theta | 1,000,000 |

Multimodal projector 使用 GELU，参数量为 5.25M。模型配置中的 `image_token_id` 为 151671。

### 4.4 加载方式

模型和 processor 使用 Hugging Face 原生接口加载：

```python
AutoProcessor.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=False,
)

AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    local_files_only=True,
    trust_remote_code=False,
    attn_implementation="sdpa",
).eval()
```

关键点：

- 只读取 pinned 本地 snapshot，不在正式运行中联网；
- `trust_remote_code=False`，不执行仓库自定义 Python；
- attention implementation 固定为 PyTorch SDPA；
- 模型设为 `eval()`，生成在 `torch.inference_mode()` 下执行；
- tokenizer 使用 left padding；
- `pad_token_id=151643`，即 tokenizer 的 `<|endoftext|>`；
- `eos_token_id=151645`；
- `use_cache=True`。

## 5. 冻结配置

正式配置文件：[`configs/r0_internvl35_1b_no_plan.json`](../configs/r0_internvl35_1b_no_plan.json)

配置 SHA256：`0e01be31a7ac3b44d964aff981512b16818a143cd7986c559282d1bd298fb896`

| 类别 | 参数 | 值 | 含义 |
|---|---|---:|---|
| Model | dtype | `bfloat16` | 权重与推理精度 |
| Model | attention implementation | `sdpa` | PyTorch scaled dot-product attention |
| Video | frames per interval | 16 | 每个绝对 interval 均匀采样最多 16 帧 |
| Video | cumulative max frames | 32 | 当前决策可见的累计视频帧上限 |
| Video | frame size | 448 × 448 | 送入 vision tower 的尺寸 |
| Text | max history turns | 4 | query 之后最多保留最近 4 个 dialog turn |
| Generation | max new tokens | 64 | 每个 chunk 最多生成 64 token |
| Generation | do sample | `false` | greedy decoding |
| Generation | use cache | `true` | 启用 KV cache |
| Generation | pad token ID | 151643 | 与 tokenizer pinned 值一致 |
| Reproducibility | seed | 20260713 | PyTorch CPU/CUDA seed |
| State | plan state | `none` | 不加入 plan 或 compact state |
| Evaluation | primary metric | `macro_f1` | 官方主指标 |

R0 没有 temperature、top-p、beam search、threshold、decision head 或 post-hoc probability calibration。`do_sample=false` 时 seed 主要用于控制底层可能的随机路径，决策本身为 greedy。

## 6. 官方 Prompt 与文本上下文

R0 直接从 pinned starter kit 动态加载 `SYSTEM_PROMPT`，没有复制后再私自改写。正式 prompt 为：

```text
You are a proactive AI assistant watching a first-person video of the user performing a procedural task. The user has issued a single high-level query. As the video unfolds you observe a series of short (~8s) chunks; after each chunk you decide whether to speak or stay silent.

Output format (single line, no preamble):
  - If you should speak: start with the literal token `$interrupt$` followed by your suggestion or answer in plain text.
  - If you should stay silent: output the single literal token `$silent$` and nothing else.

Speak when the user asks you something, when an earlier action needs correction, or when you have useful, timely guidance for the next step. Stay silent when nothing useful needs to be said.
```

第 `i` 个 chunk 的文本消息按以下顺序构造：

```text
system: official SYSTEM_PROMPT
user:   session query
history: dialog[i][1:] 的最后 4 个非空 turn
```

`dialog[i][0]` 通常是与 `query` 重复的初始 user turn，因此去掉后再追加 history，避免 query 重复。未知 role 由官方 normalizer 降级为 user，空文本被跳过。

实际 history turn 分布：

| 进入 prompt 的 history turns | Chunk 数 | 占比 |
|---:|---:|---:|
| 0 | 701 | 7.06% |
| 1 | 1,133 | 11.40% |
| 2 | 1,289 | 12.97% |
| 3 | 1,301 | 13.10% |
| 4，达到 cap | 5,511 | 55.47% |

## 7. 完整推理管线

### 7.1 总体数据流

```text
Pinned JSONL row + video
          |
          v
静态审计：文件 SHA / 模型参数 / 视频存在 / interval 与 dialog schema
          |
          v
对 session 的 chunk i 顺序执行
          |
          +--> 只从绝对 interval[i] 解码当前帧
          |
          +--> 追加到 causal cumulative frames
          |
          +--> 对累计帧做确定性均匀下采样，最多 32 帧
          |
          +--> official prompt + query + dialog[i] 最近 4 turns
          |
          +--> InternVL chat template，video 插到第一个 user message
          |
          +--> BF16 greedy generate，最多 64 tokens
          |
          +--> 官方语义一致的 tag canonicalization
          |
          +--> 保存 raw response、规范化原因和最终 answer
          v
每个完整 session：append JSONL + flush + fsync
          |
          v
分片完成：schema 校验 + predictions + diagnostics + runtime
          |
          v
8 分片精确覆盖校验、按 global input_index 合并
          |
          v
冻结 700 行 predictions 和 SHA256
          |
          v
唯一一次官方 full scorer
```

### 7.2 帧提取

R0 直接调用官方 `data/starter_kit/model.py::extract_frames()`。对 interval `[start, end]`：

1. 用 OpenCV 读取原视频 FPS 和总帧数；
2. `start_frame=int(start*fps)`；
3. `end_frame=min(int(end*fps), total_frames-1)`；
4. `n=min(16, end_frame-start_frame+1)`；
5. `step=(end_frame-start_frame)/n`；
6. 选择 `int(start_frame+i*step)`；
7. 去重、排序，逐帧 seek；
8. BGR 转 RGB，再转 PIL Image。

R0 与官方示例脚本有一个更严格的实现差异：官方示例会先预解码全部 interval，再在第 `i` 次决策只传入前缀；R0 在 chunk 到来时才解码当前 interval，从执行层面也不提前读取未来视频帧。

### 7.3 累计视觉记忆

每个 chunk 新抽到的帧追加到 `cumulative_frames`。若累计帧超过 32，使用与官方 starter kit 相同的确定性均匀步长：

```python
stride = len(frames) / 32
selected = [frames[int(i * stride)] for i in range(32)]
```

这不是 recent-window，也不是 age-aware compression。随着 session 变长，32 帧均匀覆盖全部已见 interval；历史与近期没有显式优先级。

实际输入帧统计：

| 模型输入帧数 | Chunk 数 | 解释 |
|---:|---:|---|
| 32 | 9,230 | 92.90%，累计记忆达到 cap |
| 16 | 696 | 7.01%，绝大多数首 chunk |
| 其它 | 9 | 极短 interval 或帧去重造成 |

当前 interval 本身在 9,915/9,935 个 chunk 中抽到了完整 16 帧；其余 20 个是短 interval 或可用帧不足。

### 7.4 448×448 视频处理修正

checkpoint 内存在一处上游配置不一致：

- `video_preprocessor_config.json` 声明 384×384；
- `preprocessor_config.json` 和 vision tower 声明 448×448；
- vision patch size 为 14，pixel shuffle downsample ratio 为 0.5。

384 会形成不适合 0.5 pixel shuffle 的 27×27 patch grid，首次真实 smoke 在模型内部 reshape 时失败。R0 显式把 `processor.video_processor.size` 固定为 448×448，并在加载后检查 vision tower 的 `image_size==(448,448)`。448/14 得到 32×32 grid，可以被 0.5 downsample 正常处理。

这是 checkpoint 配置一致性修复，不是根据标签或分数做的调参。

### 7.5 Multimodal message 与生成

adapter 将 video placeholder 插入第一个 user message：

```text
system message: 纯文本 official prompt
first user:     [video, query text]
history:        后续 user/assistant 文本消息
```

随后：

1. `InternVLProcessor.apply_chat_template(..., add_generation_prompt=True)`；
2. processor 接收 `text=[prompt]`、`videos=[frames]`、`padding=True`；
3. tensor 移到当前逻辑 GPU；
4. 记录输入 `prompt_length`；
5. greedy `model.generate()`；
6. 只切出 prompt 之后的新 token；
7. `skip_special_tokens=True` 解码并 `strip()`。

### 7.6 输出规范化

`canonicalize_response()` 保持官方 scorer 的标签语义，同时保证提交文件结构合法：

| Raw response | 最终输出 | 记录原因 |
|---|---|---|
| 以 `$interrupt$` 开头且有正文 | `$interrupt$<正文>` | 无 |
| 只有 `$interrupt$` | `$interrupt$Please continue with the next step.` | `empty_interrupt_utterance` |
| 以 `$silent$` 开头 | `$silent$` | 有后缀时为 `trimmed_silent_suffix` |
| 其它任何文本 | `$silent$` | `malformed_response_scored_as_silent` |

最后一种行为与官方 `parse_tag()` 一致：非 `$interrupt$` 前缀就是 silent。R0 没有把自然语言擅自包装为 interrupt，因为那会改变决策策略。

### 7.7 可持久化断点续跑

每个 session 完成后写入 `session_records.jsonl`，立即执行：

```text
write -> flush -> os.fsync
```

resume 时先验证：

- 已完成记录不能超过当前分片长度；
- 每条 `input_index` 必须是该分片从起点开始的严格连续前缀；
- prediction 必须存在；
- video_path、session 顺序和 answer 数必须与源数据一致。

因此中断最多损失当前尚未完整落盘的一个 session，不会把半个 session 当作有效记录。

## 8. 模块设计

| 模块 | 责任 | 关键设计 |
|---|---|---|
| `src/proactive_r0/core.py` | 模型无关的 causal 主流程 | official starter 动态加载、消息构造、帧 cap、逐 session 推理、canonicalization、schema 校验 |
| `src/proactive_r0/internvl.py` | InternVL3.5 HF adapter | GPU 预检、processor/model 加载、448 修正、chat template、greedy generation、峰值显存 |
| `src/proactive_r0/run.py` | 单进程/单分片实验运行器 | 静态 fingerprint、参数审计、分片、resume、fsync、artifacts、可选官方评分 |
| `src/proactive_r0/merge.py` | 正式多分片合并 | invariant config 校验、global index exact cover、全量 schema 校验、唯一一次 scorer、最终 manifest |
| `src/proactive_r0/artifacts.py` | 可复现性工具 | SHA256、safetensors 参数统计、Small 限制检查、环境与 nested Git snapshot、原子 JSON 写入 |
| `src/proactive_r0/tests/test_core.py` | 项目回归测试 | 因果顺序、history、frame cap、标签无关验证、GPU 映射、8-way shard bounds |

设计上刻意保持三层边界：

```text
model-agnostic causal policy (core)
              |
model-specific adapter (internvl)
              |
experiment orchestration and audit (run / merge / artifacts)
```

这样 R1 可以保持 `core/run/merge` 的评测与审计契约，只替换或扩展 state 输入，不必重新发明 scorer 或数据顺序逻辑。

## 9. 静态审计与安全检查

每个正式 worker 在加载模型前执行：

1. 数据 JSONL SHA256 校验；
2. starter kit `model.py`、`run_generate_proactive.py`、`run_evaluation.py` SHA256 校验；
3. safetensors 权重 SHA256 和唯一参数数校验；
4. 参数总数不得超过 2B；
5. 700-session 分片范围和全部视频存在性校验；
6. interval 满足 `0 <= start < end` 且按 start 有序；
7. `len(dialog)==len(video_intervals)`；
8. NVML 检查目标物理 GPU 的已有 compute process。

GPU 使用策略：

- 默认检测到已有进程时 warning 并记录 PID/显存，不再一律拒绝；
- `--require-exclusive-gpu` 可切换为严格独占；
- `CUDA_VISIBLE_DEVICES` 的逻辑 ordinal 会被映射回正确物理 GPU，防止检查错卡。

## 10. Smoke、失败尝试与正式运行决策

### 10.1 384×384 smoke 失败

首次真实模型 smoke 在生成任何 prediction 前失败：384 输入形成 27×27 patch grid，与 0.5 pixel shuffle 不兼容。该 run 完成 session 数为 0，保存在：

[`output/experiments/20260713_internvl35_1b_no_plan_smoke_failed_384/`](../output/experiments/20260713_internvl35_1b_no_plan_smoke_failed_384/)

修复后把视频尺寸冻结为 448×448。

### 10.2 单 session smoke

修复后的 smoke：

| 项目 | 数值 |
|---|---:|
| Sessions / chunks | 1 / 10 |
| Wall time | 29.31 s |
| Session generation time | 20.84 s |
| Peak allocated GPU memory | 3,466,036,736 bytes |
| Prediction SHA256 | `3ae429e6287b4df46e3e3d9f297b375c0d6a8800b05dba45cd5d1179ec3bf348` |

该 smoke 只用于运行正确性，不用于 prompt 或参数选择。

### 10.3 pad token 一致性与中止的 4-way run

checkpoint 的 `generation_config.json` 没有声明 pad token，Transformers 会临时把 EOS 151645 当作 pad 并为每个 chunk 打 warning。为消除日志洪泛并冻结行为，adapter 改为显式使用 tokenizer 自带 pad ID 151643。

第一次 4-way run 在中断前后分别经历了隐式 pad fallback 和显式 tokenizer pad。batch size 为 1 且输入没有真实 padding，双 smoke 的预测 SHA 也证明样例输出未变化；但同一个正式结果文件不能混用两套声明。项目因此主动中止该 4-way run，所有目录加上 `ABORTED.md`，禁止 merge 或 score。

### 10.4 双进程同卡预检

两路同时处理相同单 session：

| Worker | Session time | Peak allocated memory | Prediction SHA |
|---|---:|---:|---|
| A | 25.00 s | 3,466,037,248 bytes | `3ae429e6...bf348` |
| B | 24.38 s | 3,466,037,248 bytes | `3ae429e6...bf348` |

两个结果与单进程 smoke 完全一致，证明同卡多进程不会改变该确定性输出，并确认显存足够。

### 10.5 正式 8-way 执行

正式运行把 700 个 session 按原始顺序切成 8 个连续分片：

| Shard | Global indices | Sessions | Chunks | Physical GPU | Wall time，s |
|---:|---|---:|---:|---:|---:|
| 0 | `[0, 88)` | 88 | 1,274 | 5 | 3,632.415 |
| 1 | `[88, 176)` | 88 | 1,344 | 6 | 3,601.190 |
| 2 | `[176, 264)` | 88 | 1,327 | 7 | 3,514.720 |
| 3 | `[264, 352)` | 88 | 1,166 | 5 | 3,011.448 |
| 4 | `[352, 439)` | 87 | 1,283 | 6 | 3,482.845 |
| 5 | `[439, 526)` | 87 | 1,098 | 7 | 2,912.468 |
| 6 | `[526, 613)` | 87 | 1,229 | 5 | 3,284.346 |
| 7 | `[613, 700)` | 87 | 1,214 | 6 | 3,185.147 |

GPU 5/6/7 分别运行 3/3/2 个 worker。GPU 0-3 上已有的重要 STRIDE 进程未被触碰；GPU 4 在正式启动前出现约 28.6 GiB 的外部任务，因此正式 run 完全避开 GPU 4。

每个分片使用相同模型、配置、代码和 seed，只改变 session 范围。分片阶段统一 `--skip-eval`，不各自读取标签评分。

合并器要求：

- shard index 必须恰好为 0..7；
- 每个 shard 的 model/data/starter/inference/evaluation/validation policy 必须与 base config 相同；
- 每个分片的 global index 必须完整且连续；
- 合并后 index 必须严格等于 0..699；
- shape 必须严格等于 700 sessions / 9,935 chunks；
- 每行 video_path、answer 数和格式再次校验。

只有这些检查全部通过，才写最终 predictions 并调用一次官方 scorer。

### 10.6 运行资源

| 项目 | 数值 |
|---|---:|
| 8-way wall time，以最慢 shard 计 | 3,632.415 s，约 60 min 32 s |
| Aggregate GPU process time | 26,624.579 s，约 7.40 GPU-hours |
| Merge and score 记录时间 | 0.471 s |
| 单 worker 最大 peak allocated GPU memory | 3,478,380,544 bytes，约 3.24 GiB |

## 11. 软件环境

| 组件 | 版本 |
|---|---|
| OS | Linux 5.15.0-83-generic，x86_64，glibc 2.35 |
| Python | 3.10.20 |
| Python executable | `/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python` |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.12.1 |
| Safetensors | 0.8.0 |
| OpenCV | 4.13.0 |
| Pillow | 12.2.0 |
| NumPy | 2.2.5 |
| GPU | NVIDIA A800-SXM4-80GB |

checkpoint 自带配置记录的 Transformers 版本为 4.55.0，但正式运行环境为 5.12.1。R0 通过 `trust_remote_code=False`、真实 smoke、输出 SHA 和全量测试固定了当前兼容行为。

## 12. 官方评测定义

官方 scorer 将 interrupt 作为正类：

```text
Interrupt Precision = TP / (TP + FP)
Interrupt Recall    = TP / (TP + FN)
Silent Precision    = TN / (TN + FN)
Silent Recall       = TN / (TN + FP)

Macro F1 = (Interrupt F1 + Silent F1) / 2
G-mean F1 = sqrt(Interrupt F1 * Silent F1)
```

正式 scorer 文件：`data/starter_kit/run_evaluation.py`

SHA256：`072301da6c65b3e30c7581920d178c6d5136305f2db26914df4785f47d809ee1`

完整 700-session scorer 输出在 [`metrics.json`](../output/experiments/20260713_internvl35_1b_no_plan_r0/metrics.json)，其中包含 overall、全部 per-task 指标和每个 session 的逐 chunk gold/pred tag。

## 13. 正式结果

### 13.1 混淆矩阵

| Gold \ Pred | Interrupt | Silent | 合计 |
|---|---:|---:|---:|
| Interrupt | TP = 1,541 | FN = 3,811 | 5,352 |
| Silent | FP = 1,374 | TN = 3,209 | 4,583 |
| 合计 | 2,915 | 7,020 | 9,935 |

从矩阵可以直接看出：R0 并不是简单地“什么都说”，而是明显偏向 silent。Interrupt precision 尚可，为 0.5286；interrupt recall 只有 0.2879。

### 13.2 常数策略参照

以下由相同 gold class counts 和官方公式计算，只用于解释 R0 是否学到了非平凡决策：

| 策略 | Macro F1 | G-mean F1 | Predicted interrupt rate |
|---|---:|---:|---:|
| 永远 silent | 0.3157 | 0.0000 | 0% |
| 永远 interrupt | 0.3501 | 0.0000 | 100% |
| R0 | **0.4630** | **0.4541** | 29.34% |

R0 比两个常数策略都高，且两类 F1 均非零，因此模型确实利用了输入进行区分；但类频率校准严重偏向 silent。

### 13.3 分领域结果

下表是预测冻结后的诊断性聚合，公式与官方 overall 一致，但不是官方排行榜的独立子指标：

| Domain | Chunks | Gold int. | Pred int. | Int. Recall | Silent Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Arts and Crafts | 2,391 | 55.42% | 39.61% | 0.3774 | 0.5807 | 0.4667 |
| Chef | 2,952 | 51.59% | 40.07% | 0.3940 | 0.5920 | **0.4863** |
| Handyman | 2,237 | 53.60% | **12.96%** | **0.1443** | 0.8873 | 0.4247 |
| Tutorial | 2,355 | 55.41% | 21.02% | 0.2054 | 0.7838 | 0.4317 |

Handyman/Tutorial 的 gold interrupt rate 与其它领域相近，但模型说话率显著更低。这不是标签先验差异可以解释的，说明 zero-shot 决策具有明显的 domain-dependent calibration。

### 13.4 按 session 相对进度

把每个 session 的 chunk index 归一化到五等分：

| Session 进度 | Chunks | Gold int. | Pred int. | Int. Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| 0-20% | 2,277 | 68.29% | **10.06%** | **0.0881** | **0.3042** |
| 20-40% | 1,956 | 52.35% | 35.28% | 0.3311 | 0.4621 |
| 40-60% | 2,030 | 48.42% | 42.46% | 0.4466 | 0.5197 |
| 60-80% | 1,956 | 46.52% | 35.94% | 0.3890 | 0.5226 |
| 80-100% | 1,716 | 51.28% | 25.12% | 0.3091 | **0.5267** |

最前 20% 是显著失败区间。中间和尾段 Macro F1 接近，表明模型并非随着累计视频变长而单调退化；真正的问题是初始指导、格式和早期 procedural state。

### 13.5 首 chunk

| 统计 | 数值 |
|---|---:|
| Sessions | 700 |
| Gold interrupt | 699，99.86% |
| Predicted interrupt | 50，7.14% |
| TP / FP / TN / FN | 50 / 0 / 1 / 649 |
| Macro F1 | 0.0683 |

首 chunk 几乎是一个确定性的“开始指导”位置，但模型往往直接生成指导文本而漏掉 `$interrupt$` 前缀，导致官方 scorer 把它判为 silent。

### 13.6 输出格式诊断

| Normalization | 数量 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|
| 无规范化 | 9,261 | 1,514 | 1,362 | 3,204 | 3,181 |
| Malformed -> silent | 633 | 0 | 0 | 3 | **630** |
| Empty interrupt fallback | 39 | 27 | 12 | 0 | 0 |
| Trim silent suffix | 2 | 0 | 0 | 2 | 0 |

Malformed 的位置：

| Chunk index | 数量 |
|---:|---:|
| 0 | 628 |
| 1 | 3 |
| 2 | 1 |
| 3 | 1 |

633 个 malformed 没有空字符串，raw 文本长度 median 为 81 字符、mean 为 105.74，且有 626 种不同文本。典型输出包括：

```text
Place the stickers on the cover of the notebook.
Alright, let's get started.
Place the cable clip on the wall and press it firmly to secure it.
```

这些例子说明模型通常已经决定“给指导”，但没有遵守 tag 协议。R0 按官方 scorer 语义仍必须把它们算作 silent。

### 13.7 后验反事实，不能作为 R0 成绩

为了估计格式问题的上限，预测冻结后做了两个反事实：

| 后验规则 | Macro F1 | Int. Recall | 说明 |
|---|---:|---:|---|
| 把所有 malformed 改判 interrupt | 0.5362 | 0.4056 | 630 TP 同时增加 3 FP |
| 强制所有首 chunk 为 interrupt | 0.5385 | 0.4092 | 利用了 699/700 的 validation 标签结构 |

这两个数字**不是有效 R0、不是未调参结果、不能作为排行榜成绩或泛化结论**。规则是在查看 public validation 标签分布后分析得到，最多说明格式/首 chunk policy 具有很高的潜在价值。

若后续正式测试这些策略，必须：

- 预先冻结规则；
- 与 plan-state 实验分开；
- 在同一 public validation 上选择规则时标为 `val-supervised`；
- 不能把 0.5362/0.5385 当作已完成实验结果。

## 14. 结果解释

### 14.1 R0 学到了什么

- R0 显著超过 always-silent 和 always-interrupt；
- 中后段能同时保持两类非零 F1；
- Chef 和 Arts and Crafts 的决策能力明显高于其它领域；
- 合法 interrupt 的 precision 超过 0.52，说明模型说话时并非随机。

### 14.2 R0 没学好什么

- 没有可靠地输出 tag，尤其是首 chunk；
- 没有把“初次应给出任务起始指导”内化为合法 interrupt；
- 对 Handyman/Tutorial 过度 silent；
- 没有显式 current step、completion/incompletion cue 或 progress state；
- uniform 32-frame memory 没有 recent/history 分区，也不表达动作完成边界。

### 14.3 为什么不能把全部 FN 都归因于缺少 plan

3,811 个 FN 中有 630 个来自已知 malformed fallback。这部分首先是输出协议失败。其余 3,181 个 FN 才更接近“模型输出了合法 silent，但标签需要 interrupt”的决策问题。

因此，后续报告 plan-state 收益时必须同时报告：

```text
official Macro F1
interrupt / silent P-R-F1
predicted interrupt rate
malformed count
first-chunk recall
domain breakdown
```

否则仅看 Macro F1 无法判断增益来自 state、格式遵循还是简单提高说话率。

## 15. 局限性与结论边界

1. **只有 public validation。** 这是零样本 public-validation 结果，不是 hidden test 泛化证据。
2. **输入容器含标签。** 生成代码不访问 `answers`，但没有做到物理 label-stripped 文件隔离。
3. **使用官方提供 dialog。** 不是把模型自己的历史预测回填，因此没有测闭环错误累积。
4. **只有一个模型和一个 prompt。** R0 不能证明 InternVL3.5-1B 是所有 Small backbone 中最优。
5. **格式与决策耦合。** 官方 scorer 把缺 tag 的自然语言指导记为 silent，当前 633 个样本受此影响。
6. **内容不计分。** C1 当前客观 scorer 只看 tag，不能由本结果推出 utterance 内容质量。
7. **均匀帧记忆较朴素。** 32 帧 cap 会逐渐稀释每个 interval，没有 recent-frame 优先或 state compression。
8. **无 plan/state。** R0 不表达 current step、动作完成、错误、恢复或 confidence。
9. **数据许可。** 使用官方 public validation 的 CC-BY-NC-4.0 数据做评测；R0 没有引入 STRIDE 或其它外部训练数据。

## 16. 对 R1 的交接约束

R1 的目标不是“随便改 prompt 看能不能涨分”，而是隔离回答：oracle compact state 是否有价值。

R1 必须冻结以下 R0 条件：

- 同一个 InternVL checkpoint 和 BF16/SDPA 加载方式；
- 16 frames/interval、32 cumulative cap、448×448；
- query、官方 dialog 和最近 4 turns；
- greedy、64 max new tokens、pad ID 151643；
- 相同输出 canonicalization；
- 相同 session 顺序、数据 hash 和官方 scorer；
- 相同标签使用声明。

最小对照：

| Variant | 新增信息 |
|---|---|
| R0 | 无 plan |
| R1-A | current step only |
| R1-B | current step + completion/incompletion cues |
| R1-C | compact full state |

R1 的主要成功判据：

1. interrupt recall 提高；
2. silent precision 不发生不可接受的坍塌；
3. Macro F1 提高且不是只靠提高全局 interrupt rate；
4. malformed count不被悄悄改变，或把格式变化作为独立 ablation；
5. 首 chunk、Handyman、Tutorial 和 session 前 20% 的改善可解释。

建议同时建立独立的 R0-F format ablation，但不与 R1 混合：

- 更强的 tag-constrained decoding；
- grammar/logit constraint；
- 明确的 first-turn policy；
- 非 tag 自然语言的预注册处理规则。

其中任何根据当前 validation 标签选择的规则都必须标记为 `val-supervised`。

## 17. 可复现命令

### 17.1 单分片命令模板

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu> \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONNOUSERSITE=1 \
PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_r0.run \
  --config configs/r0_internvl35_1b_no_plan.json \
  --output-dir output/experiments/20260713_internvl35_1b_no_plan_r0_shard<index>of8 \
  --device cuda:0 \
  --num-shards 8 \
  --shard-index <index> \
  --skip-eval
```

发生安全中断后，在完全相同命令末尾添加：

```text
--resume
```

### 17.2 合并与官方评分

精确命令保存在 [`command.sh`](../output/experiments/20260713_internvl35_1b_no_plan_r0/command.sh)。核心形式为：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONNOUSERSITE=1 \
PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_r0.merge \
  --config configs/r0_internvl35_1b_no_plan.json \
  --output-dir output/experiments/20260713_internvl35_1b_no_plan_r0 \
  --shard-dir output/experiments/20260713_internvl35_1b_no_plan_r0_shard0of8 \
  --shard-dir output/experiments/20260713_internvl35_1b_no_plan_r0_shard1of8 \
  --shard-dir output/experiments/20260713_internvl35_1b_no_plan_r0_shard2of8 \
  --shard-dir output/experiments/20260713_internvl35_1b_no_plan_r0_shard3of8 \
  --shard-dir output/experiments/20260713_internvl35_1b_no_plan_r0_shard4of8 \
  --shard-dir output/experiments/20260713_internvl35_1b_no_plan_r0_shard5of8 \
  --shard-dir output/experiments/20260713_internvl35_1b_no_plan_r0_shard6of8 \
  --shard-dir output/experiments/20260713_internvl35_1b_no_plan_r0_shard7of8
```

## 18. 关键指纹与验证

| 对象 | SHA256 |
|---|---|
| Model weights | `11effd1da2fc0929957d56de4129d1e7d2aed044ea878d40bf83e95b63d38b39` |
| Validation JSONL | `feef69ddee605e7070ad0f133636c35739c6964514a46d76da294b6bf1964740` |
| R0 config | `0e01be31a7ac3b44d964aff981512b16818a143cd7986c559282d1bd298fb896` |
| Starter `model.py` | `3cecb5ab2201b6b4be6aabbdebeeeb5322437e2d8c49d065018230dc9804edbe` |
| Starter proactive runner | `7c8adfe9fa2db5af4aace1cd260ec22132b18d82ef747f8a566eb8095e8fbc34` |
| Official scorer | `072301da6c65b3e30c7581920d178c6d5136305f2db26914df4785f47d809ee1` |
| Final predictions | `312d0375dd67be2fb244d622a9f302734082f0f251ef0b8dd190b00880879820` |
| Full metrics JSON | `409b639a690e6fc6b4a5396d1bef7c88e8222ddbadd45d9cce0cc53dda7f1671` |

官方 scorer 在预测冻结后独立重跑一次，所得完整 metrics 文件与正式 artifact 逐字节相同，`skipped_chunks=0`。

测试：

| Test suite | 结果 |
|---|---:|
| `src/proactive_r0/tests` | 12 / 12 passed |
| 官方 `test_run_evaluation_proactive.py` | 27 / 27 passed |
| `compileall src/proactive_r0` | passed |

项目测试覆盖：

- official stride 风格的 frame cap；
- query/history 消息构造；
- extract/generate 的严格 causal 顺序；
- malformed response fallback；
- 不依赖 gold answer 的 source validation；
- prediction 顺序和 answer 数校验；
- 逻辑 GPU 到物理 GPU 的映射；
- 4-way 和 8-way 连续分片边界。

## 19. 正式产物说明

| 文件 | 内容 |
|---|---|
| [`README.md`](../output/experiments/20260713_internvl35_1b_no_plan_r0/README.md) | 简版实验摘要 |
| [`config.json`](../output/experiments/20260713_internvl35_1b_no_plan_r0/config.json) | 合并后的完整有效配置 |
| [`command.sh`](../output/experiments/20260713_internvl35_1b_no_plan_r0/command.sh) | 精确 merge 命令 |
| [`environment.txt`](../output/experiments/20260713_internvl35_1b_no_plan_r0/environment.txt) | merge 环境和 8 个 inference shard 环境 |
| [`code_state.txt`](../output/experiments/20260713_internvl35_1b_no_plan_r0/code_state.txt) | 关键源码 hash、STRIDE/leaderboard Git 状态 |
| [`data_manifest.json`](../output/experiments/20260713_internvl35_1b_no_plan_r0/data_manifest.json) | 数据、模型、许可、supervision 和 shard provenance |
| [`predictions.jsonl`](../output/experiments/20260713_internvl35_1b_no_plan_r0/predictions.jsonl) | 冻结的 700 行正式预测 |
| [`session_records.jsonl`](../output/experiments/20260713_internvl35_1b_no_plan_r0/session_records.jsonl) | raw response、interval、帧数、normalization 等逐 chunk 审计记录 |
| [`diagnostics.json`](../output/experiments/20260713_internvl35_1b_no_plan_r0/diagnostics.json) | 预测率、格式回退计数、prediction SHA |
| [`metrics.json`](../output/experiments/20260713_internvl35_1b_no_plan_r0/metrics.json) | 官方完整 metrics 和逐 row tags |
| [`metrics_summary.json`](../output/experiments/20260713_internvl35_1b_no_plan_r0/metrics_summary.json) | 不含逐 row 明细的指标摘要 |
| [`scorer.log`](../output/experiments/20260713_internvl35_1b_no_plan_r0/scorer.log) | 官方 scorer 命令和输出 |
| [`runtime.json`](../output/experiments/20260713_internvl35_1b_no_plan_r0/runtime.json) | shard 运行时间、GPU memory、范围和 diagnostics |

每个 `..._shard0of8` 至 `..._shard7of8` 目录还保存各自的 `run.log`、command、environment、runtime、predictions 和 durable session records。

## 20. 汇报用一句话

> 我们完成了首个符合 C1 Small 限制的 1.06B InternVL3.5 全量因果 no-plan 基线，在 700 sessions / 9,935 chunks 上获得官方 Macro F1 0.4630；分析表明主要瓶颈不是过度打断，而是首 chunk tag 遵循失败和整体 interrupt recall 不足，下一步应把格式修复与 compact procedural state 的价值验证拆成两个独立实验。
