# InternVL3.5-1B Oracle-Plan 结构化 State Decoding S0 报告

> 日期：2026-07-17  
> 任务：C1 Small，procedural-state feasibility  
> 结论：冻结 1B 的 zero-shot structured state decoding 不足；官方 dialog 提供强阶段线索，但没有被稳定解码为细粒度 step/progress；停止 S0 prompt 搜索，进入独立监督 S1  
> 性质：oracle static plan + predicted dynamic state；不是 hidden-test、decision OOF 或提交模型证据

## 1. 实验问题

U1 正在通过人工评分回答“oracle state 是否让 utterance 更好”。S0 独立回答：

> 给冻结的 Small backbone 一份只由 query/task 写成的四步 oracle static plan，它能否从当前因果视频和 prior dialog 中识别 current step、progress 与 error state？

S0 不改变 D3 decision，不读取人工评分，不训练模型，也不生成自由格式 JSON。

## 2. Target 隔离

现有 formal oracle 明确是 `evaluation_only_oracle_non_deployable`，因此未用于训练。Preparation 将数据物理拆成：

- `inputs.jsonl`：80 条 query、task、四步 static plan、因果 intervals、prior dialog；
- `targets.jsonl`：step/progress/error 与 annotator confidence；
- prediction runner 配置中不存在 target 路径；
- evaluator 只在两个 view 的 predictions 完整并哈希冻结后读取 target。

| Artifact | SHA256 |
|---|---|
| Runner inputs | `c0d44b933f926460020f6a2d362de3cd2ffed97a66cad5adcc0dddc4ced4a53b` |
| Evaluation targets | `09ae879afc8e3c2eb298ec07244201e2c2b461da55b6b96becbd6f74b06595b8` |
| Preparation manifest | `602a6c785fbb0032d2d50119ab74c5decd3272886a30dcba1c3ac7bca4436b73` |

实验者在模型预测前已经阅读过 state schema、样例和 aggregate distribution。因此本实验只能称为“predictions 出现前冻结、runner target-isolated”，不能称为从未见过 state label 的 benchmark。

## 3. 模型与候选任务

Backbone 仍为 `OpenGVLab/InternVL3_5-1B-HF`，revision `9191dbccf312b537016f041b25d61c72e7c5c9f3`，1,060,897,792 参数、BF16、Apache-2.0。

输入与 R0/D3 对齐：每 interval 16 frames，累计后最多 32 frames，最多 4 个历史 turns。

固定三个 target：

```text
step:     1=s1, 2=s2, 3=s3, 4=s4
progress: 1=not_started, 2=ongoing, 3=complete,
          4=deviated, 5=recovered
error:    1=absent, 2=present
```

所有数字都经过 tokenizer 验证，是唯一、等长、可 round-trip 的单 token。每个 state 只运行一次视觉塔，11 个 language candidates 保持 batch-one。

两个冻结 view：

- `official_dialog`：使用完整 inference-visible prior dialog；
- `no_assistant_history`：只删除 prior assistant turns，其他输入完全相同。

## 4. V1 工程失败与 V2 校准

一个 target-isolated v1 smoke 暴露数字 token 本身有强 `1 > 2 > 3...` 先验。它没有读取 target，也没有进行 formal run，但不能作为 state 能力测试。

V1 被保留为工程失败。V2 在 formal predictions 前冻结 content-free contextual calibration：每个 session/target 先只用 query + oracle plan、无视频、无 dialog 计算 candidate prior，然后：

```text
calibrated_score = observed_logp - content_free_logp
```

不搜索 temperature、mapping、prompt、permutation 或阈值。V2 protocol SHA256 为 `7f47aaf7a0f0be6390a8801eb17f39ab743657527847f0fdb76bbcec46a7db3a`。

## 5. 正式运行

两个 view 均在物理 GPU 7 串行运行，其他 GPU 上已有进程未被访问。

| View | Sessions / states | Wall | Mean state forward | Peak GPU | Prediction SHA256 |
|---|---:|---:|---:|---:|---|
| official dialog | 20 / 80 | 611.892s | 2.640s | 3,120,762,880 B | `a3af9ff8fbbb72e60d3abf9b1e461c4784e1f7dc260406d1a41f2c589ee8db9e` |
| no assistant | 20 / 80 | 639.838s | 2.659s | 3,120,759,808 B | `6661a3917c74a10afa5756eac5a0368ae7ba18e556c4eb41cb61c84fc5389789` |

两份 data manifest 均记录：`target_file_read=false`、`ratings_read=false`、`answers_read=false`。V2 smoke 与 formal 第一个 state 的 prediction、raw/content-free scores 和 token IDs 完全一致。

## 6. 正式结果

### 6.1 主要指标

| View | Step acc / Macro | Progress acc / Macro | Error acc / Macro | Joint step-progress | Mean task Macro | Composite correctness |
|---|---:|---:|---:|---:|---:|---:|
| official dialog | 0.250 / 0.2226 | 0.225 / 0.1348 | 0.750 / 0.5098 | 0.050 | **0.2891** | **0.4083** |
| no assistant | 0.475 / 0.2601 | 0.0125 / 0.0167 | 0.2125 / 0.2024 | 0.000 | **0.1597** | **0.2333** |

两个 view 的 mean task Macro 都低于预注册 weak threshold `0.35`，因此结论均为 `insufficient_zero_shot_signal`。两边都没有任何一个 step/progress/error 三字段全对的 state。

Step ordinal MAE：official `1.7375`，no-assistant `0.9875`。后者更低不是更懂视频，而是大量预测 s1，恰好利用 formal set 中 s1 占比高的分布。

### 6.2 预测塌缩

Official dialog 的预测分布：

```text
step       s4=64, s3=7, s2=7, s1=2
progress   complete=70, deviated=8, ongoing=2
error      present=71, absent=9
```

No-assistant 的预测分布：

```text
step       s1=68, s4=7, s3=3, s2=2
progress   not_started=69, complete=8, deviated=2, ongoing=1
error      absent=74, present=6
```

这说明校准后的候选分数主要把“存在很多 prior assistant turns”和“没有 assistant history”解释成两个极端阶段，而不是细分视频中的真实 current step。

### 6.3 Dialog 影响

Official 相对 no-assistant 的 composite correctness 增益为 `+0.1750`，10,000 次 session bootstrap 95% 区间 `[+0.1125,+0.2375]`，正增益比例 1.0。

按 domain，official composite 为：Arts `0.3667`、Chef `0.4167`、Handyman `0.4000`、Tutorial `0.4500`；四个 domain 都高于对应 no-assistant。

按位置，增益最大的是 `10+`：`0.1000 -> 0.5667`。Second chunk 仅 `0.3333 -> 0.3500`。因此 prior assistant turns 强烈编码“历史上发生过多少指导/动作”的阶段信息，但该信息不能直接等同于视觉 state understanding。

### 6.4 置信度

Official mean confidence `0.6711`，no-assistant `0.7876`，但两边完全正确的 states 都是 0。Frozen candidate softmax 明显过度自信，不能直接作为 deployable state confidence。

## 7. 结论

S0 得到两个同时成立的结论：

1. **状态方向仍有信号。** 删除 assistant history 后 composite 显著下降，证明当前 causal context 中存在可利用的阶段信息。
2. **冻结 1B 的 zero-shot state interface 不够。** Step/progress Macro 很低，预测塌缩到两个端点，不能作为 predicted state updater，更不能直接进入 D3 gate 或 utterance。

因此不再尝试 S0 prompt、候选 token、温度、permutation 或后验校准。正确下一步是 S1：使用新的、独立、连续 state supervision，训练一个很小的 decoder，并用 temporal-only control 判断它是否学到超过位置规律的 state。

## 8. S1 已冻结准备

S1 使用全新的 32 sessions / 444 contiguous states，排除本次 formal 20 和旧 R1 四个 session：

| Split | Sessions | States | Domains | Length bands |
|---|---:|---:|---|---|
| Train | 24 | 318 | 各 6 | short/middle/long = 9/9/6 |
| Held-out | 8 | 126 | 各 2 | short/middle/long = 3/3/2 |

选样 seed `20260717-state-s1-v1`，不读取 answers、模型输出、错误、ratings 或 state labels。正式 sanitized sessions SHA256 为 `cdfe53adf40f0533dc6ac4e8269cadaa810217c4e519a7bd6a2de9a419e02c21`，annotation template SHA256 为 `4e1a42fb21bf48cb82346ed55f1e430336c398d13070f9e59242af2c27fcad5c`。

S1 将比较 `temporal_only / current_d1 / d3_dynamics` 三个线性 state decoder。只有 `d3_dynamics` 在新的 8-session held-out 上通过冻结 gate 后，才允许进入 D3 decision fusion 或 predicted-state utterance。

## 9. 主要产物

- S0 protocol v2：`annotations/state_s0_oracle_plan_v1/PROTOCOL_v2.md`
- S0 runner config：`configs/s0_internvl35_1b_oracle_plan_state_inference_v2.json`
- S0 evaluation：`output/experiments/20260717_internvl35_1b_oracle_plan_state_s0_v2_evaluation/evaluation.json`
- Evaluation SHA256：`b634c4b1d9d360708bd336c2d2c375c132437f19b82afb0bb1691f479a4340d9`
- S1 protocol：`annotations/state_s1_decoder_v1/PROTOCOL.md`
- S1 formal preparation：`annotations/state_s1_decoder_v1/prepared_v2/`

