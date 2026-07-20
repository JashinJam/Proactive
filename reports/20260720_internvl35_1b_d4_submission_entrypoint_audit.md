# D4 Small 提交入口与容器前置审计报告

> 日期：2026-07-20
> 范围：D4 隐藏测试推理入口、输入输出契约、参数/许可/依赖清单、CPU preflight、GPU smoke
> 结论：模型侧 submission adapter 已完成；官方 Docker 模板适配、项目源码许可证选择和 leaderboard/container 上传仍未执行

## 1. 结论

本轮没有训练模型，也没有改变 D4 的特征、阈值、采样帧数或历史长度。完成的是 D4 从
“项目内部实验 runner”到“可以接在官方 Docker 模板后面的模型入口”的工程转换：

- 新入口接收一个 EgoProactive JSONL、视频目录、模型目录和输出路径；
- 默认要求每个 chunk 都有对齐的官方 `dialog`，并拒绝包含 `answers` 的隐藏测试输入；
- 运行时自动把隐藏输入路径和 SHA256 写入临时配置，不再依赖固定 validation 路径；
- 复用已经验证过的 D4 shared-vision runner，不复制或改写模型逻辑；
- 输出前强制投影为 `video_path` 和 `answers` 两个字段，再检查 session 顺序、chunk 数量和 tag；
- 使用临时文件原子写出最终 `predictions.jsonl`，并生成参数、输入和输出哈希回执。

CPU preflight、48 项 R0/D1/D3/D4 回归测试和一条 10-chunk GPU 闭环均已通过。
新入口生成的预测与 2026-07-19 冻结 D4 smoke **逐字节相同**。

## 2. 为什么需要独立入口

原有 [`proactive_d1.run_deploy`](../src/proactive_d1/run_deploy.py) 是实验复现入口。它会：

- 从固定 config 读取 public validation JSONL 和视频路径；
- 要求输入 SHA256 与冻结 validation 文件完全一致；
- 在实验目录写入 records、环境、评测和复现材料。

这些约束适合保证实验没有悄悄换数据，却不能直接接受组织方挂载的隐藏测试文件。新的
[`proactive_d4.submission`](../src/proactive_d4/submission.py) 只负责边界适配：根据实际
输入生成 runtime config，再调用原 runner。D4 内部模型、head、因果帧处理和 dialog
状态定义保持不变。

官方网站目前明确说明 Test Phase 入选队伍会在 2026-08-08 收到 Docker templates 和
registry credentials；最终容器的基础镜像、挂载目录、CMD、健康检查和资源限制尚未公开。
因此本轮没有自行猜测一套可能冲突的 Dockerfile，而是先稳定其后的模型入口。

## 3. 输入输出契约

隐藏推理输入每行必须包含：

```text
video_path
video_intervals
query
domain
dialog
```

约束如下：

1. `dialog` 必须与 `video_intervals` 等长，`dialog[i]` 表示第 `i` 个候选区间之前的累计历史；
2. 默认不允许出现 `answers`；公开 validation 本地审计只能通过显式
   `--allow-input-answers-for-local-audit` 开关运行；
3. 即使本地审计输入含 `answers`，传给模型前仍由 `strip_answers()` 生成无标签 rows；
4. 视频只按当前和以前的绝对 interval 解码，不读取当前 chunk 结束时间以后的画面；
5. 最终输出只有 `video_path` 和与 intervals 等长的 `answers`；
6. 每个 answer 必须恰好是 `$silent$` 或非空 `$interrupt$<utterance>`。

主调用形式记录在 [`submission/d4_small/README.md`](../submission/d4_small/README.md)。
正式模板发布后只需把组织方路径映射到这些参数。

## 4. 完整推理流程

```text
organizer JSONL + video directory
              |
              v
  hidden-input validator
  - required fields
  - dialog/interval alignment
  - reject answers by default
              |
              v
  generated runtime config
  - actual input SHA256
  - mounted model/head/starter paths
  - frozen D4 inference settings
              |
              v
  existing D4 shared-vision runner
  - cumulative causal frames
  - query + official dialog prefix
  - InternVL raw utterance
  - tag margin + 1024-d hidden
  - 18 scalar + 8 dialog-stage features
  - frozen 1052-parameter linear head
              |
              v
  two-field projection + strict validation
              |
              v
  atomic predictions.jsonl + receipt
```

入口还会在加载 GPU 模型前检查：

- 模型权重 SHA256 和 stored parameter count；
- D4 head SHA256、feature count 和参数数；
- starter `model.py`、`run_generate_proactive.py` 和 scorer 的冻结指纹；
- manifest 与 config 的 `frames_per_interval/max_frames/max_history_turns/max_new_tokens`；
- dtype、attention implementation 和 decision feature mode。

因此同学之后如果改变帧采样或历史长度，必须同步更新正式 config 和 manifest，并重新做
等价 smoke；不能只改容器参数而仍沿用旧登记信息。

## 5. 模型登记与 Small 参数

最终隐藏测试 Docker 使用一个全量 public-development refit 后的单一 D4 head：

| 项目 | 数值 |
|---|---:|
| Backbone | `OpenGVLab/InternVL3_5-1B-HF` |
| Backbone parameters | 1,060,897,792 |
| D4 head parameters | 1,052 |
| Total parameters | 1,060,898,844 |
| Active parameters | 1,060,898,844 |
| Small limit | 2,000,000,000 |
| Backbone/model license | Apache-2.0 |

当前提交表建议填写：

```text
Model name: InternVL3.5-1B-D4-DialogStage
Model license: Apache-2.0
Total params (billions): 1.060898844
Active params (billions): 1.060898844
```

完整机器可读记录见 [`manifest.json`](../submission/d4_small/manifest.json)。InternVL3.5-1B
不是 MoE；vision tower、projector、language model 和 D4 head 都参与推理，所以 total 与
active 相同。

为便于仓库协作，D4 的 1,052 参数线性 head 已逐字节复制到
`submission/d4_small/decision_head.json`。该交付副本与正式 final artifact 的 SHA256
相同；2.12 GB InternVL 基座仍由固定 Hugging Face revision 或本地模型目录提供，不进入
项目 Git 历史。

## 6. 许可证状态

已确认：

- Backbone 权重：Apache-2.0；
- 官方数据：CC-BY-NC-4.0；
- 官方 starter kit：CC-BY-NC-4.0；
- 本项目目前没有顶层 `LICENSE`。

因此当前可以填写 backbone 的 `Model license=Apache-2.0`，但还不能把“获奖所需的源码
开源许可审计”标成完成。给本项目选择 Apache-2.0、MIT 或其他官方认可许可证属于代码
所有者决定，本轮没有越权替用户添加。官方 starter 属于组织方提供的竞赛依赖，正式模板
发布后还要确认它是否由模板自带，以及参赛镜像是否需要重新分发该目录。

## 7. CPU Preflight

实验目录：

```text
output/experiments/20260720_internvl35_1b_d4_submission_preflight_v1/
```

使用 public validation 和显式 local-audit 开关，只选择第一条 session 做路径/视频检查；
模型权重只在 CPU 上读取 metadata 和哈希，没有把模型加载到 GPU。

| 检查 | 结果 |
|---|---:|
| Public input sessions/chunks | 700 / 9,935 |
| 含 `answers` rows | 700，已明确标为 local audit |
| Backbone/head params | 1,060,897,792 / 1,052 |
| Total/active params | 1,060,898,844 / 1,060,898,844 |
| Small 参数检查 | 通过 |
| GPU used | false |
| Wall time | 5.37 s（外层共约 7.45 s） |
| Runtime config SHA256 | `af3a9afe670965a92be8b08a208c298da4aeed6b5d44c3a3060e5e7914acc00b` |

## 8. Submission GPU Smoke

实验目录：

```text
output/experiments/20260720_internvl35_1b_d4_submission_entrypoint_smoke_v1/
```

运行前在沙箱外检查全部 GPU。物理 GPU 0 有约 22.4GB 已有进程，GPU 1--7 空闲；本轮绑定
物理 GPU 1，并启用 `--require-exclusive-gpu`。没有停止、修改或占用其他 GPU 上的进程。

| 项目 | 结果 |
|---|---:|
| Sessions / chunks | 1 / 10 |
| Runner wall | 44.206 s |
| Session compute | 32.23 s |
| Peak GPU memory | 3,466,037,248 bytes |
| Preexisting processes on selected GPU | 0 |
| Predicted interrupts | 6 |
| Official scorer invoked | false |
| Published prediction SHA256 | `cb79b4573dd3551fefafb219b5685be4a0d5d7b1e85bb74d1bf023f67c175de9` |

与 2026-07-19 冻结 D4 smoke 对照：

- 最终 `predictions.jsonl` SHA256 完全相同，文件逐字节相同；
- raw response、R0 answer、prompt tokens、tag margin：10/10 相同；
- 8 个 dialog-stage 特征：10/10 相同；
- decision logit、decision 和 answer：10/10 相同。

这证明新包装层没有改变 D4 推理结果。它不是性能抽样，不能把该 session 的局部分数当作
leaderboard 估计。

## 9. 回归测试

| 测试组 | 通过 |
|---|---:|
| R0 causal core | 12/12 |
| D1 decision/calibration/deploy | 18/18 |
| D3 dynamics/dialog control | 8/8 |
| D4 deploy/submission | 10/10 |
| 合计 | 48/48 |

新增 D4 测试覆盖：拒绝隐藏输入 `answers`、显式 public-audit 例外、dialog 对齐、路径重写、
参数核算、manifest/config 一致性、两字段输出投影和原子发布。`compileall` 与
`git diff --check` 通过。

## 10. OOF 证据与最终 Docker 模型必须分开

D4 的 `0.6846` 来自五折 session-level OOF 合并：每个 session 由没有用该 session 训练的
fold head 预测。这是我们当前最可靠的 public-development 性能估计，但合并文件对应五个
head，不是隐藏测试时使用的单一模型。

最终 Test Docker 使用的是在全部 700 public sessions 上 refit 的单一 head。它在同一训练
集上的 `0.7393` 只是 train-fit sanity，不能当作隐藏测试估计。

机器可读的区别见
[`validation_evidence.json`](../submission/d4_small/validation_evidence.json)。本轮没有偷偷把
OOF predictions 说成单模型产物，也没有上传 OOF 或 train-fit 文件。若 Validation Phase
准备上传 OOF `predictions.jsonl`，提交名称和说明应明确标注它是 five-fold OOF artifact；
是否上传仍需用户授权。

## 11. 当前完成度和下一步

已经完成：

1. 模型侧隐藏输入 adapter；
2. `dialog` 必填与 gold-answer 默认拒绝；
3. 路径解耦、实际输入哈希和无 scorer 推理；
4. 两字段原子输出及回执；
5. 参数、依赖、许可和 OOF/最终模型清单；
6. CPU preflight、48 项回归和真实 GPU smoke。

尚未完成：

1. 选择项目顶层源码许可证；
2. 合入同学最终选定的帧采样/历史长度 config，并重跑 manifest audit 和 smoke；
3. 2026-08-08 后把官方 Docker 模板的固定路径、CMD、健康检查、超时和资源限制接到本入口；
4. 在官方基础镜像中做无网络 build/run 和一条冻结等价测试；
5. 任何 validation prediction、Docker image 或 registry 外部上传。

因此当前主线可以离开“入口是否可行”的问题。等待同学的输入策略结果期间，下一项内部
研究工作回到冻结路线：保留 D3/D4 决策，推进 early-chunk utterance cold-start 和视觉
grounding；若剩余盲评先完成，则优先按既定 gate 合并评分结论。
