# 实验报告

本目录用于存放有实验证据支持的当前实验报告和审计报告。每份报告必须明确记录对应的实验产物、配置、代码状态、数据指纹、评分器和局限性。

不要把报告写成相互竞争的路线文档。路线决策统一记录在 [CURRENT_ROUTE.md](../CURRENT_ROUTE.md) 中。

## 当前报告

- [C1 Small R0 完整技术与实验报告](20260713_internvl35_1b_no_plan_r0.md)：冻结的 InternVL3.5-1B no-plan 基线，包含完整管线与配置、官方评测、诊断、局限性和复现命令。
- [C1 Small R1 Oracle 紧凑状态协议试验报告](20260714_internvl35_1b_oracle_state_r1_pilot_v1.md)：四个 session 的因果 Oracle 标注协议、受控状态变体、官方子集指标、格式混杂和 R1 门槛决策。
- [C1 Small R0-F 格式消融实验报告](20260714_internvl35_1b_r0f_format_ablation.md)：完整 700-session response-intent repair、被否决的 tag-grammar smoke test、官方分数、domain/首 chunk 分析和验证集监督边界。
- [C1 Small D1 严格因果标量决策头实验报告](20260714_internvl35_1b_causal_scalar_decision_head_d1_oof_v2.md)：五折 session-level OOF 线性决策校准、严格因果特征审计、作废泄漏版、官方分数、bootstrap/领域/位置分析及部署边界。
- [C1 Small D1 神经融合决策头完整实验报告](20260715_internvl35_1b_neural_decision_head_d1.md)：标签 margin 与 1,024 维因果 hidden 的四卡提取、四项五折 OOF 消融、后验 L2 复核、最终单一融合头、GPU 在线烟测和逐 chunk 一致性审计。
- [C1 Small D1 融合推理等价加速报告](20260715_internvl35_1b_d1_inference_optimization.md)：双候选 batch、prefix cache 与 shared vision 三种实现的正确性/时延/显存对照；shared vision 在 127 chunks 上逐元素等价并将墙钟降低 9.15%，已推广为部署路径。
- [C1 Small D1 单一部署阈值稳健性审计](20260715_internvl35_1b_d1_threshold_robustness.md)：逐字节复现五折 D1 后，将 final head 的单一中位数阈值应用于 OOF logits；官方 Macro F1 `0.6330`，相对 fold-specific `0.6341` 仅下降 `0.00113`，通过全部部署稳健性门槛。
- [C1 Small D2 轻量非线性残差决策头报告](20260715_internvl35_1b_residual_mlp_d2.md)：预注册 width-8 GELU residual MLP 的严格五折 OOF；官方 Macro F1 `0.6351`，相对 D1 仅 `+0.0010` 且 bootstrap 跨 0，未推广。
- [C1 Small 最终语言 MLP LoRA / 联合决策损失可行性审计](20260715_internvl35_1b_final_mlp_lora_feasibility.md)：历史工程审计，记录朴素 BF16 局部重放、四状态 MLP 校正和两次 bounded smoke；旧 failed 状态保留，并附 2026-07-16 后续正式状态说明。
- [C1 Small 最终语言 MLP LoRA 严格五折 OOF 实验报告](20260716_internvl35_1b_final_mlp_lora_oof.md)：四状态 full-cache 尾部失败、六状态 700-session exact cache、fixed-batch same-shape 重放、正式五折指标与 bootstrap；主候选 Macro `0.6357`，仅 `+0.0016` 且 2/5 folds 提升，未推广。
- [D1 Utterance 问题诊断与 Planner / Language 后续工作交接](20260716_d1_utterance_planner_language_handoff.md)：固定 fallback 的来源与全量统计、D1 决策/语言接口缺口、当前 planner/language 覆盖审计，以及供执行 Agent 预注册和实施的候选工作包；本文不是 active route 决议。
- [U0/U1 双人独立盲评执行细则](20260717_u0_u1_human_review_protocol.md)：评分信息边界、逐 session 因果解锁、统一 1--5 分锚点、U0/U1 专用字段、双评审合并、一致性与仲裁规则。人工评分开始前使用此版本。

已完成的 ProAssist Phase 1 报告属于历史材料，已归档至：

```text
/home/lanjinxin/workspace/deprecated/wearable_ai_challenge/
  2026-07-13_pre_pwr_reset/project/reports/phase1_proassist_finetune.md
```

其中的负面实验结论仍有参考价值，但它记录的输出路径、后续步骤以及一项完整预测声明已经不再权威。有关实验产物不一致的问题，请查阅归档清单。
