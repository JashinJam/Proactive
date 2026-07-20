# U1 正式盲态标注与全量 Oracle Generation 报告

> 日期：2026-07-17  
> 状态：20-session / 80-state 正式盲标、oracle-step/full 全量生成、自动诊断与三路盲评包已完成；interface/state 双人评分均未闭环  
> 结论等级：工程与评价协议闭环，不是 state 有效或无效的最终结论

## 1. 核心结论

本轮完成了 U1 中所有不依赖人工评分的步骤：

1. 旧 4-session oracle smoke 被正式降级为 nonblind engineering diagnostic，因为标注者在标注前看过对应生成输出；它不再进入 state-effect 证据。
2. 两个隔离上下文 Agent 重新标注全部 20 sessions / 80 sampled states，只读取净化后的 query/task/prior dialog、明确允许的视频区间和独立协议。
3. 合并 validator 通过 20/80 全覆盖、精确 timestamp、step 引用、confidence、禁用 target marker 和 formal-blind provenance 检查。
4. 正式 `forced_oracle_step/full` 运行 80/80 复现冻结 R0 raw response，并保持全部 9,935 个 D1 interrupt/silent 决策及官方 Macro F1 `0.6341` 完全不变。
5. `step` 产生 56/80 非空正文，与 no-state 相同；`full` 产生 57/80，只多恢复一个 second-chunk 正文。State 会明显改变文本，但自动统计方向混合，不能证明内容更正确。
6. 三路 state 盲评包已经固定为 no-state/step/full 共 240 candidates、480 reviewer rows；在双人评分完成前，不 promotion state、不开始 predicted state、granularity、SFT 或 GRPO。

## 2. 本实验回答什么

U1 固定 D1 的二元 gate，只研究 gate 已经选择 interrupt 后的正文生成：

```text
冻结 D1 决策
    |
    +-- current fallback
    +-- forced no-state
    +-- forced oracle step
    +-- forced oracle full
```

因此本实验不尝试提高官方 Macro F1。它回答的是：

- 1B backbone 在真实 assistant-side `$interrupt$` prefix 后能否生成正文；
- answer-blind procedural state 是否能改善正文；
- compact step 是否足够，还是 full evidence/recovery 字段有额外价值。

官方 decision 指标只作为不变性约束，不能替代内容评价。

## 3. 正式盲态标注

### 3.1 为什么重做全部 20 sessions

旧 `oracle_states.smoke.json` 虽然通过 schema 和 timestamp validator，但标注者此前已查看 smoke generation outputs。即使标注时没有主动复制输出，也无法满足 formally blind 的证据要求。

旧文件现已写明：

```text
annotation_type = engineering_smoke_nonblind_diagnostic_only
scientific_use  = schema and runner engineering only
```

更新后 SHA256：

```text
d666df568c9b1af093770da55d75dc845590964c2a6f858782350d17637b5757
```

正式 oracle 不复用其中任何一个状态，全部 20 sessions / 80 states 重新标注。

### 3.2 净化输入与隔离

从冻结 `sample_items.jsonl` 生成两个净化分片，只保留：

- `input_index/video_path/query/task/domain`；
- 当前 sample 的 chunk、position、interval、`observed_through_sec`；
- `video_intervals_so_far`；
- 当前时刻正式可见的 `prior_dialog`；
- `sample_id`。

显式删除 `frozen_decision`、`current_output`、gold answer、模型输出和 error category。两个 Agent 使用 `fork_turns=none`，不得读取 `CURRENT_ROUTE.md`、reports、U1 outputs、ratings、review key、旧 oracle 或源 gold JSONL。

净化输入 SHA256：

| 分片 | Sessions / States | SHA256 |
|---|---:|---|
| part A | 10 / 40 | `3417b738ec8454cd6444f45850b2c9604bec783387b817f1d44cce7ebfafeeb0` |
| part B | 10 / 40 | `a6814e8af7549c01609ee90b1614ee0ff222ca2a6f7a9fa33fd78f93104f45ab` |

正式隔离协议 SHA256：

```text
06b385faebc961a718b8cb0ba1f4dae61cc625181d92afce8a656733aad1a941
```

### 3.3 因果规则

每个 session 先仅依据 task/query 冻结静态 macro plan，再按 `observed_through_sec` 递增标注动态状态。视频只允许查看 `video_intervals_so_far` 明列区间；即使某段空档的绝对时间早于当前 timestamp，也禁止观看。

动态状态只使用：

- task/query；
- 该 sample 的 prior dialog；
- 当前及历史允许区间中的可见证据。

禁止当前/未来 answer、未来 dialog、未来视频、模型输出、D1/R0 error。标注文本不得包含 decision tag 或 should-speak/interrupt target。

### 3.4 合并与质量统计

| Artifact | SHA256 |
|---|---|
| part A annotations | `68b569bc230daf47a63d8537aaca036fe1a127619671412bce7da16410df68d2` |
| part B annotations | `3d9ce9fbb3396e34d15dd94c3273b4c2407ba79761487e7ab277117c34a949f6` |
| merged formal oracle | `e8f1e0736398d46193009ddb3966599ccc2f8629cfaecdd55f270b5ec6018850` |
| merge manifest | `8cfa5dc349024740659886d43d4ddcbd593f7d1f66ffb8e16dfaa5ba9add82ee` |

根侧 validator 结果：

```text
sessions                         20
sampled_states                   80
causal_timestamps_exact          true
forbidden_target_markers_absent  true
formal_blind_provenance          true
```

描述统计：

| 项目 | 结果 |
|---|---:|
| 每 session macro steps | 4 |
| ongoing / complete | 47 / 16 |
| deviated / recovered | 7 / 10 |
| confidence mean / min / max | 0.924 / 0.67 / 0.99 |
| confidence < 0.8 | 7 |
| 空 `incompletion_or_error_evidence` | 15 |

低置信度保留原值，没有根据生成结果事后改写。

## 4. 模型与完整管线

### 4.1 模型

| 项目 | 配置 |
|---|---|
| Backbone | `OpenGVLab/InternVL3_5-1B-HF` |
| Revision | `9191dbccf312b537016f041b25d61c72e7c5c9f3` |
| License | Apache-2.0 |
| Stored unique params | 1,060,897,792 |
| Precision | BF16 |
| Weight SHA256 | `11effd1da2fc0929957d56de4129d1e7d2aed044ea878d40bf83e95b63d38b39` |

本轮只运行冻结 backbone。D1 gate 的 1,044-parameter head 已提前产生完整 OOF decisions；若按 deployable D1 系统计数，base + head 为 1,060,898,836 params，仍在 Small 2B 限制内。Oracle 标注是 non-deployable upper-bound 输入，不是可提交组件。

### 4.2 因果模型输入

对每个冻结 sample：

1. 按原始 R0 路径累计当前及历史 interval frames；
2. `frames_per_interval=16`，累计后最多保留 `max_frames=32`；
3. 构造 query、最多 4 个历史 turns 和当前可见 frames；
4. 先重放 R0 generation，要求 raw response 精确匹配；
5. 对生成变体在 assistant 端真实预填 `$interrupt$` token，再 greedy continuation；
6. 只替换该 sample 的 utterance 正文，不更改任何 D1 decision；
7. 在完整 700-session prediction 上运行官方 scorer。

共同 decoding：

```text
max_new_tokens   64
do_sample        false
assistant_prefix $interrupt$
seed             20260713
```

### 4.3 State block

`forced_oracle_step` 只加入 `Current step` 与 `Next step`。

`forced_oracle_full` 再加入：

```text
progress
observed completion evidence
observed incomplete/error evidence
recovery action
```

除 state block 外，两变体共享完全相同的 frames、history、prefix、prompt suffix、decoding 和 token budget。

## 5. 执行信息

正式命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_u1.run \
  --config configs/u1_fixed_gate_forced_generation.json \
  --output-dir output/experiments/20260717_internvl35_1b_fixed_gate_forced_generation_u1_v1_oracle_formal_full \
  --device cuda:0 \
  --variants forced_oracle_step forced_oracle_full \
  --oracle-states annotations/u1_forced_generation_v1/oracle_states.formal_blind.json
```

运行前 GPU 0 已有另一个用户的约 23.49 GB 常驻进程，但当时 utilization 为 0；本轮采用用户允许的非独占轻量共享策略。

| 项目 | 结果 |
|---|---:|
| Wall time | 661.63 s |
| Peak GPU memory | 3,501,634,048 bytes，约 3.26 GiB / 3.50 GB |
| Mean step generation | 0.873 s / sample |
| Mean full generation | 0.877 s / sample |
| R0 replay | 80/80 exact |
| 训练/权重修改 | 无 |

## 6. 官方 Decision 结果

三个完整 predictions 的 9,935 个二元 decisions 完全一致：

| 指标 | current fallback | oracle step | oracle full |
|---|---:|---:|---:|
| Macro F1 | 0.6341 | 0.6341 | 0.6341 |
| G-mean F1 | 0.6341 | 0.6341 | 0.6341 |
| Interrupt F1 | 0.6352 | 0.6352 | 0.6352 |
| Silent F1 | 0.6330 | 0.6330 | 0.6330 |
| TP / FP / TN / FN | 3165 / 1448 / 3135 / 2187 | 相同 | 相同 |

Prediction SHA256：

| Variant | SHA256 |
|---|---|
| current fallback | `04183a4083d160662d5f91bff5432a7ca96595dd66b2b0b64f3b430799143ad9` |
| oracle step | `4eb821214f94ad1366f6f8e8dfc5abc25cd9cd60eb1076d35b67080606b8b695` |
| oracle full | `43505925fb83a237b00ada94596ce0d909a8df29567316ed824c0f5b25b6b51a` |

这只证明 U1 没有破坏 D1 gate，不证明 utterance 正确。

## 7. 自动内容诊断

### 7.1 总体

| Variant | 非空 | EOS/fallback | 完成宣告词法命中 | assistant-action 命中 | exact repeats | 非空均值 words |
|---|---:|---:|---:|---:|---:|---:|
| no-state | 56/80 | 24 | 9 | 3 | 2 | 16.38 |
| oracle step | 56/80 | 24 | 5 | 2 | 1 | 16.41 |
| oracle full | 57/80 | 23 | 6 | 1 | 3 | 15.26 |

### 7.2 非空分布

| 分组 | no-state | step | full |
|---|---:|---:|---:|
| Arts and Crafts | 18/20 | 18/20 | 19/20 |
| Chef | 12/20 | 12/20 | 12/20 |
| Handyman | 13/20 | 13/20 | 13/20 |
| Tutorial | 13/20 | 13/20 | 13/20 |
| second chunk | 4/20 | 4/20 | 5/20 |
| position 2--4 | 13/20 | 13/20 | 13/20 |
| position 5--9 | 20/20 | 20/20 | 20/20 |
| position 10+ | 19/20 | 19/20 | 19/20 |

### 7.3 Pairwise 改写

| Contrast | 文本变化 | 两边都空 | reference-only 非空 | target-only 非空 | 两边非空但文本变化 |
|---|---:|---:|---:|---:|---:|
| step vs no-state | 43/80 | 24 | 0 | 0 | 43 |
| full vs no-state | 53/80 | 23 | 0 | 1 | 52 |
| full vs step | 45/80 | 23 | 0 | 1 | 44 |

可以成立的解释：state block 确实被模型利用，因为它改变大量 continuation；但 compact step 完全没有改变 EOS 集合，full 也只多恢复一个输出。

不能成立的解释：完成宣告命中下降不等于 hallucination 下降，文本变化不等于 correctness 提升，full 多一条非空也不等于 full state 更好。非盲抽查显示有改善也有新增错误，因此必须按冻结人评协议判断。

## 8. 三路 State 盲评包

为避免把正在进行的 interface 评分与 state 评分混用，正式评价分两包：

- interface package：fallback vs no-state，160 candidates；当前仍由两名评审填写；
- state package：no-state vs step vs full，240 candidates；三者重新评分，不复用 interface no-state 分数。

State package：

```text
seed             20260717-u1-state-review-v1
samples          80
candidates       240
reviewer rows    480
blind SHA256     f64f7844dc140406dd9b9b7935d87f9b580eb2840f8eee928b0f8c3089dba824
key SHA256       4770d76e29a24f1998430ef1eab02878a71a42cbbe178defcbe3dea8464ed3de
template SHA256  361362f1acafca9c64d74218f065ed73428b64f0ee879bef5720364613ec4d8b
```

盲包已在独立临时目录复现，blind/key/template 哈希相同。评审期间不得读取 `state_review_key.jsonl`。

冻结统计：

- 两名 reviewer 的 1--5 分先在 candidate 内取均值；
- `content_composite = mean(correctness, specificity, actionability, groundedness, plan_consistency)`；
- seed `20260717`，10,000 次 session-level bootstrap；
- state promotion 要求 mean delta `>= +0.50`、95% CI 下界 `> 0`、至少 3/4 domains 为正、hallucination/unsafe rate 增量均 `<= 0.02`；
- `unsafe = safety_1_5 <= 2`；
- full 只有在相对 step 的 mean delta `>= +0.25` 且 bootstrap 下界 `> 0` 时才证明额外字段值得保留；
- 同时报告 quadratic-weighted Cohen kappa、exact/within-one agreement、MAE 和 binary Cohen kappa。

这些规则及实现均在读取任何有效评分前冻结。

## 9. 当前路线判断

本轮尚未通过或失败任何 efficacy gate。当前唯一 promoted baseline 仍是 D1 `0.6341`。

评分完成后的分流：

1. **Interface pass，state 不 pass**：优先解决 gate-to-language 接口和 fit-fold utterance supervision，不训练 planner。
2. **Step/full state pass**：先做更大、独立、预注册的 oracle-state replication；复现后才进入 predicted/noisy state。
3. **No-state 与 oracle 都不 pass**：优先 U2 fit-fold-only utterance SFT/LoRA，说明当前主要限制更可能是 language capacity/supervision。
4. **Full 不显著胜 step**：固定 compact step，不扩大 progress/evidence/recovery 字段。

在上述结论出现前，不进入 granularity predictor、predicted state、GRPO 或额外 prompt/token tuning。

## 10. 主要产物

- [正式盲标协议](../annotations/u1_forced_generation_v1/formal_blind/PROTOCOL.md)
- [合并 oracle states](../annotations/u1_forced_generation_v1/oracle_states.formal_blind.json)
- [Oracle merge manifest](../annotations/u1_forced_generation_v1/formal_blind/manifest.json)
- [完整正式实验](../output/experiments/20260717_internvl35_1b_fixed_gate_forced_generation_u1_v1_oracle_formal_full/)
- [自动 state diagnostics](../output/experiments/20260717_internvl35_1b_fixed_gate_forced_generation_u1_v1_oracle_formal_full/analysis/state_content_diagnostics.json)
- [State blind review](../output/experiments/20260717_internvl35_1b_fixed_gate_forced_generation_u1_v1_oracle_formal_full/state_review/state_review_blind.jsonl)
- [State ratings template](../output/experiments/20260717_internvl35_1b_fixed_gate_forced_generation_u1_v1_oracle_formal_full/state_review/state_ratings_template.csv)
- [U1 frozen protocol](../annotations/u1_forced_generation_v1/PROTOCOL.md)

