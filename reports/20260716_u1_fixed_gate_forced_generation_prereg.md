# U1 固定 Gate Forced-Generation 预注册

> 日期：2026-07-16  
> 实验 ID：`20260716_internvl35_1b_fixed_gate_forced_generation_u1_v1`  
> 状态：样本与判据已冻结，GPU engineering smoke 尚未运行

## 核心问题

U0 已确认 D1 4,613 次 interrupt 中 2,586 次使用固定 fallback。U1 不再讨论“该不该说”，而是在完全固定 D1 决策的条件下区分三种解释：

1. 模型本来能说，只是当前 gate 与生成器接口断裂；
2. 模型缺少当前/下一步骤等 procedural state；
3. 即使给出 oracle state，1B backbone 仍缺少语言或程序能力，需要 utterance supervision。

## 样本冻结

样本选择不读取 gold label/utterance。20 个 session 覆盖四域，每个 session 取 second、2--4、5--9、10+ 各一个 fallback chunk，共 80 chunks。四域和四位置各 20 条。旧 R1 四个 session 已排除。

首个 session/domain 共 4 sessions / 16 chunks 只用于工程 smoke。完整样本 SHA256：

```text
de38746a55fa7649615e4b6405b6d4904d6a891ca49ca0680b34219a2efbb974
```

## 实现要求

`forced_no_state` 必须在 assistant generation prompt 之后真正追加 `$interrupt$` token prefix，再让模型续写；不能只在 prompt 中写“请回答”后仍允许模型选择 silent。三个生成变体共享相同的 forced-generation instruction，oracle 变体只增加 state block。

Smoke 先重放原始 R0 路径，要求 16/16 仍精确输出 `$silent$`，用于证明帧、dialog 和 prompt 装配未漂移。随后才运行 prefix continuation。所有变体会在冻结 D1 全量 predictions 上只替换正文，并用官方 scorer 验证 Macro 和每个 decision 完全不变。

## 评价与路线分流

正式 pilot 的主要评价是双人盲评 `content_composite`，定义为 correctness、specificity、actionability、groundedness、plan consistency 的平均值。

- `forced_no_state` 通过：先实现独立的 decide-then-speak 接口。
- 只有 oracle state 通过：扩大 plan/state 复验，并继续保留层级粒度标注。
- oracle step 与 full 接近：不扩大复杂 full-state schema。
- 所有生成变体都失败：优先做 fit-fold-only utterance SFT/LoRA。

完整 sampling、因果标注边界、smoke gate 和 promotion gate 见 [`PROTOCOL.md`](../annotations/u1_forced_generation_v1/PROTOCOL.md)。
