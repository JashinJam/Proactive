# 实验报告

本目录用于存放有实验证据支持的当前实验报告和审计报告。每份报告必须明确记录对应的实验产物、配置、代码状态、数据指纹、评分器和局限性。

不要把报告写成相互竞争的路线文档。路线决策统一记录在 [CURRENT_ROUTE.md](../CURRENT_ROUTE.md) 中。

## 当前报告

- [7 月 17 日至 7 月 22 日完整工作梳理与汇报报告](20260722_0717_0722_complete_progress_review.md)：合作者分支的阶段性快照，串联 D3、U1、S0、U1-V/D3-D、D4、U0 双评、U2，以及以旧 D4 为基线的 decision-fusion/structured-calibration 实验；当前路线仍以 `CURRENT_ROUTE.md` 为准。
- [历史 D6 低维结构化门槛校准 OOF 报告](20260721_internvl35_1b_structured_calibration_d6.md)：以 D4 `0.6846` 为基线比较 position、previous-action 和 last-two action 分组门槛；主候选为 `0.6747`，明确不晋级。它不是当前运行中的 query-memory LoRA D6。
- [历史 D5 决策融合与 Causal Action-History OOF 报告](20260721_internvl35_1b_decision_fusion_d5.md)：以 D4 `0.6846` 为基线融合 D3 dynamics/action-history；主候选 `0.6912`，但 bootstrap/stability 未通过。它不是当前 D4.2-session-fold D5 基线。
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
- [C1 Small D4 提交入口与容器前置审计](20260720_internvl35_1b_d4_submission_entrypoint_audit.md)：隐藏输入 adapter、`dialog`/gold-answer 契约、参数与许可证清单、CPU preflight、48 项回归和 10-chunk GPU 逐字段等价验证；明确区分五折 OOF `0.6846` 与最终单一 Docker head。
- [C1 Small D4.2 输入策略适配五折 OOF 实验报告](20260721_internvl35_1b_d4_2_adapted_input_policy_oof.md)：四个 policy-matched 线性头的完整五折结果；`history8` 达到 Macro `0.6988`，并明确记录 post-selection、val-supervised 和未完成 GPU 晋升验证的边界。
- [C1 Small D4.3 history8 GPU 等价与候选晋升报告](20260721_internvl35_1b_d4_3_history8_gpu_equivalence.md)：四领域 102-chunk 在线逐字段验证，最大 logit 误差 `1.22e-7`；`history8` 由此成为独立、未上传的活动 leaderboard-engineering 候选。
- [C1 Small D5 D4-session-fold history8 基线报告](20260722_internvl35_1b_d5_session_history8_baseline.md)：直接复用 D4.2 session manifest，predictions 与 metrics 均逐字节复现，完整 OOF Macro 为 `0.6988`。
- [C1 Small D5 因果多尺度采样 D4-session-fold 报告](20260722_internvl35_1b_d5_causal_multiscale_session_oof.md)：冻结的 `16/8/8` 因果采样器取得 Macro `0.6988`、delta `+0.0000`，bootstrap 跨零且仅 2/5 folds 提升，不晋升。
- [C1 Small D5 双视图融合 D4-session-fold 报告](20260722_internvl35_1b_d5_dual_view_session_oof.md)：共享差分与 dialog-gated 差分分别取得 Macro `0.6846/0.6793`，均低于 D4.2 history8 基线，终止该融合族。
- [C1 Small D5 视觉时序残差 D4-session-fold 报告](20260722_internvl35_1b_d5_visual_temporal_session_oof.md)：冻结的 39,073 参数 causal GRU residual 取得 Macro `0.6983`、delta `-0.0005`，未晋升并终止该模型族。
- [C1 Small D5 鲁棒多视图线性头 D4-session-fold 报告](20260722_internvl35_1b_d5_robust_session_oof.md)：四视图等权训练的 clean Macro 为 `0.6918`、delta `-0.0070`，静态门槛失败并在 self-fed 前停止。
- [C1 Small D6 零初始化等价与因果 GPU 审计](20260722_internvl35_1b_d6_zero_init_causality_smoke.md)：冻结四领域 102-chunk 审计对 D4.3 hidden/tag 精确零差异，memory residual 与双候选 update 差异为零，future-only 变异不改变历史输出；仅是正式训练前硬门，不含 efficacy 结论。
- [C1 Small D6 可训练性与资源 GPU 审计](20260722_internvl35_1b_d6_trainability_resource_smoke.md)：rotation-0 完整一轮训练使 48/48 adapter tensors 与 optimizer moments 非零变化；峰值 7.11 GiB、最长 session 26.87 秒、正式单折估算 37.34 小时，全部前置资源门通过，不含 efficacy 结论。
- [D1 Utterance 问题诊断与 Planner / Language 后续工作交接](20260716_d1_utterance_planner_language_handoff.md)：固定 fallback 的来源与全量统计、D1 决策/语言接口缺口、当前 planner/language 覆盖审计，以及供执行 Agent 预注册和实施的候选工作包；本文不是 active route 决议。
- [U0/U1 双人独立盲评执行细则](20260717_u0_u1_human_review_protocol.md)：评分信息边界、逐 session 因果解锁、统一 1--5 分锚点、U0/U1 专用字段、双评审合并、一致性与仲裁规则。人工评分开始前使用此版本。

已完成的 ProAssist Phase 1 报告属于历史材料，已归档至：

```text
/home/lanjinxin/workspace/deprecated/wearable_ai_challenge/
  2026-07-13_pre_pwr_reset/project/reports/phase1_proassist_finetune.md
```

其中的负面实验结论仍有参考价值，但它记录的输出路径、后续步骤以及一项完整预测声明已经不再权威。有关实验产物不一致的问题，请查阅归档清单。
