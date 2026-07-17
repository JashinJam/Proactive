# U1 固定 D1 Gate Forced-Generation 阶段报告

> 日期：2026-07-16  
> 实验 ID：`20260716_internvl35_1b_fixed_gate_forced_generation_u1_v1`  
> 状态：engineering smoke 与全 80 条 no-state generation 已完成；正式人评和全量 oracle comparison 未完成  
> 结论边界：本报告不 promotion 新系统，也不否决 plan/state

## 1. U1 要回答什么

U0 证明 D1 存在大规模 gate-to-language 断裂，但没有区分：

1. R0 知道说什么，只是它自己的 gate 选择了 silent；
2. 1B backbone 缺少当前 procedural state；
3. backbone 即使获得 state 也不能稳定生成有效建议。

U1 固定 D1 的全部 `$interrupt$/$silent$` 决策，只替换 selected chunks 的 utterance 正文。官方 Macro 必须对所有变体保持 `0.6341`。

## 2. 冻结样本

候选条件为 D1 fused OOF 输出精确 fallback，且冻结 R0 raw response 精确为 `$silent$`。选择代码在读取前删除当前 `answers`；单元测试将全部 gold answers 改写后，样本仍完全一致。

样本结构：

- 排除旧 R1 的 input indices `14/123/326/687`；
- 四个 domain 各 5 sessions；
- 每个 session 取 second、2--4、5--9、10+ 各 1 chunk；
- 共 20 sessions / 80 chunks；
- 每域第一个 session 构成 4 sessions / 16 chunks smoke。

冻结 sample SHA256：

```text
de38746a55fa7649615e4b6405b6d4904d6a891ca49ca0680b34219a2efbb974
```

## 3. 真正的强制生成实现

`forced_no_state` 不是在自然语言 prompt 中要求“必须回答”。实现先构造与 R0 相同的 causal frames、query 和 prior dialog，然后在 chat template 的 assistant generation prompt 后直接追加 `$interrupt$` token IDs，再由模型续写正文。

三个生成变体共享同一个 controlled-generation system suffix、assistant prefix、greedy decoding、`max_new_tokens=64` 和帧策略。Oracle 变体只增加 state block。

输出保留以下失败诊断：

- prefix 后立即 EOS；
- 再次生成 decision tag；
- 生成 `$silent$`；
- 空正文回退到固定句；
- 每条 generated tokens 和耗时。

生成完成后，只在冻结 D1 full prediction 中替换 selected utterance，再运行官方 scorer。runner 对 9,935 chunks 逐项验证决策不变。

## 4. 16-chunk Engineering Smoke

### 4.1 工程校验

| 项目 | 结果 |
|---|---:|
| R0 raw response replay | 16/16 exact `$silent$` |
| Decision invariance | 9,935/9,935 |
| 所有变体 Macro F1 | 0.6341 |
| 峰值显存 | 3.49 GB |
| 全 smoke wall time | 125.85 s |

No-state prediction 在独立 smoke1 和三变体 smoke2 中 SHA256 都是：

```text
89d71fe09d320fd031602eeb2f805b2e8277c07ed7b20d138bde001fea819ec6
```

说明 prefix continuation 可确定性复现。

### 4.2 Oracle smoke 标注及证据限制

四个 session 当时按时间顺序逐段查看，静态 plan 只来自 task/query，动态 state 使用 prior dialog 和当前时间以前视频。validator 检查：

- 4 sessions / 16 sampled states 全覆盖；
- state timestamp 与 sample interval end 精确一致；
- 不含 `$interrupt$`、`$silent$` 或目标决策提示；
- timestamp、schema 和 target-marker 约束通过。

但后续 provenance 审计发现：标注者在写这批状态前已经看过对应 smoke 的生成输出。因此不能把它称为 formally blind annotation，也不能可靠声称模型输出或错误信息完全没有影响判断。该文件现已显式降级为 `engineering_smoke_nonblind_diagnostic_only`，只保留 schema、runner 和 timestamp 工程诊断价值；所有 state-effect 结论必须使用重新隔离标注的 20-session / 80-state 正式文件。

Oracle smoke SHA256：

```text
d666df568c9b1af093770da55d75dc845590964c2a6f858782350d17637b5757
```

### 4.3 自动结果

| 变体 | 非空 | 立即 EOS / fallback |
|---|---:|---:|
| current fallback | 16 | 16 fixed fallback |
| forced no-state | 9/16 | 7/16 |
| forced oracle step | 9/16 | 7/16 |
| forced oracle full | 9/16 | 7/16 |

三个生成变体的空输出发生在同一批样本，state 没有解除 immediate EOS。

非空文本中，full state 对个别样本有作用，例如在线已准备好针线时将错误的“thread the button through the eye”改为放置 button；但也存在：

- grilled-cheese 已组装完成后仍给出不合适的 butter 动作；
- 新电池正在装入时说 remove battery；
- smoke detector 仍在 remount 时宣告完成；
- garment/button 任务被说成 quilt。

这些是工程子集的定性诊断，不是 state promotion/rejection 结论。

## 5. 全 80 条 Forced No-State

### 5.1 工程结果

| 项目 | 结果 |
|---|---:|
| R0 raw replay | 80/80 exact |
| Decision invariance | 9,935/9,935 |
| Official Macro | 0.6341 |
| Non-empty continuation | 56/80（70.0%） |
| Immediate EOS / fallback | 24/80（30.0%） |
| 峰值显存 | 3.48 GB |
| Wall time | 505.25 s |

### 5.2 位置与域

| Position | 非空 / 20 | 非空率 |
|---|---:|---:|
| second | 4/20 | **20%** |
| 2--4 | 13/20 | 65% |
| 5--9 | 20/20 | 100% |
| 10+ | 19/20 | 95% |

| Domain | 非空 / 20 | 非空率 |
|---|---:|---:|
| Arts and Crafts | 18/20 | 90% |
| Chef | 12/20 | 60% |
| Handyman | 13/20 | 65% |
| Tutorial | 13/20 | 65% |

这种强烈的位置依赖说明 immediate EOS 不是随机解码噪声。forced interface 能在中后段得到正文，但不能稳定修复 second-chunk 内容。

### 5.3 自动语言诊断

在 56 条非空输出上：

- 平均 16.38 words，中位数 15.5；
- 9 条命中完成/享用类词法规则；
- 3 条命中 assistant 自己代用户行动的 `let me / I'll / we'll` 规则；
- 2 个 sessions 出现 exact repeated continuation；
- 没有重复 decision tag 或显式生成 `$silent$`。

词法规则不能证明语义错误，但逐项文本检查已出现跨任务对象、错误下一步、过早完成和无依据状态，例如 weather、music、quilt、compartment 等明显异常。因此“56 条非空”不能解读为 70% 有效指导。

## 6. 盲评包

全 80 个 sample 的 current fallback 与 forced no-state 被随机映射为 A/B，共 160 个 candidates。盲文件不暴露 variant、gold、D1 margin 或 error category，两名评审分别填写 correctness、specificity、actionability、groundedness、plan consistency、conciseness、safety。

```text
paired_review_blind.jsonl
SHA256 f9f7160646d1a155e4a4bb5d7c0740d2c0b2245c841ef8db8ef0dd31012b033f

paired_review_key.jsonl
SHA256 015e04a373c75a258d94b76e388f7f514b59ed1ffd314b6fcfb0fa6a2c6ade9d
```

人工评分尚未进行，不能填写预注册的 paired bootstrap 或 promotion gate。

## 7. 当前结论

可以成立：

1. 当前固定 fallback 确实包含接口问题，因为 prefix continuation 可在 56/80 样本产生更具体文本。
2. 单纯接通接口不是完整解决方案：30% 立即 EOS，且非空文本中存在明显时序和 grounding 风险。
3. 非盲 Oracle smoke 以当前 zero-shot serialization 注入后，没有在 16 条工程样本中稳定改善 EOS 或内容利用；这只指导工程排查，不构成 state-effect 证据。

不能成立：

1. 不能根据 16 条 smoke 宣称 state 无效；
2. 不能在无人评时宣称 forced no-state 优于 fallback；
3. 不能把 56/80 非空率当作 correctness；
4. 不能据此直接训练 predicted state updater 或进入 GRPO。

## 8. 未完成工作与下一判据

U1 正式闭环仍需：

1. 两名独立评审完成 160-candidate paired review；
2. 由隔离上下文重新完成全部 20 sessions / 80 sampled chunks 的 formally blind oracle annotation，旧 smoke 不混入；
3. 在完整 80 chunks 运行 oracle-step/full；
4. 按预注册 session-bootstrap 和安全约束决定 interface/state/language 分流。

在此之前，D1 `0.6341` 继续是唯一 promoted baseline。

## 9. 主要产物

- [U1 protocol](../annotations/u1_forced_generation_v1/PROTOCOL.md)
- [冻结 sample](../annotations/u1_forced_generation_v1/sample_items.jsonl)
- [Oracle smoke states](../annotations/u1_forced_generation_v1/oracle_states.smoke.json)
- [三变体 smoke2](../output/experiments/20260716_internvl35_1b_fixed_gate_forced_generation_u1_v1_smoke2/)
- [全 80 条 no-state artifact](../output/experiments/20260716_internvl35_1b_fixed_gate_forced_generation_u1_v1_nostate_full/)
- [Paired blind review](../output/experiments/20260716_internvl35_1b_fixed_gate_forced_generation_u1_v1_nostate_full/analysis/paired_review_blind.jsonl)
- [Paired ratings template](../output/experiments/20260716_internvl35_1b_fixed_gate_forced_generation_u1_v1_nostate_full/analysis/paired_ratings_template.csv)
