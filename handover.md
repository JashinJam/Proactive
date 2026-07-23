# Wearable AI Challenge C1 Small 项目交接

> Historical collaborator snapshot imported from `origin/main` commit `c732103`.
> It predates the local D4.2 `history8`, bounded D5, and query-memory LoRA D6
> results. It is retained for provenance and is not an active route document;
> use `CURRENT_ROUTE.md` and `Agent.md` for current decisions.

> 交接快照日期：2026-07-21（Asia/Shanghai）
>
> 项目根目录：`/home/lanjinxin/workspace/wearable_ai_challenge`
>
> 当前分支：`main`
>
> 交接基线提交：`3ed2e0a14149b18a444a05ae3ec9a4fcb775f61d`（`Bundle D4 submission head and entrypoint`）

## 0. 给接手 Agent 的最短说明

我们主攻 ECCV Wearable AI Workshop Challenge 1（EgoProactive）的 **Small 组**，目标是优先提高官方排行榜的 Macro F1，同时保证最终模型能在组织方提供的 Docker 模板中进行因果推理。

目前最成熟的分类路线是 **InternVL3.5-1B + D4 dialog-stage decision head**：

- D4 的 5 折 OOF 公共验证诊断结果为 **Macro F1 0.6846**；
- 已训练一个使用全部 700 个公共 session 重拟合的单一 D4 head，供未来隐藏测试 Docker 使用；
- D4 的模型侧提交入口、Git 跟踪的 head、CPU preflight 和单 session GPU 等价性 smoke 都已完成；
- 还没有向排行榜上传 `predictions.jsonl`，也没有上传 Docker；任何外部提交都必须先得到项目负责人明确授权；
- 当前已重新转为 **decision-first**：公开 schema 和 starter runner 提供 chunk-aligned dialog，因此优先用全自动 OOF/官方 scorer 提升 Macro F1，暂停新增人工内容/state 评测和 utterance 训练；但隐藏测试是否提供相同 dialog、以及 assistant 历史由组织方提供还是模型自回灌，正式规则尚未明确；
- U0 A/B 聚合与 U2 自动诊断已经完成；D5 融合达到 `0.6912` 但稳定性未过门槛，D6 的低维历史门槛主候选又降到 `0.6747`，三组新 split 全部为负；D5/D6 均冻结且未部署，D4 仍是提交候选。

接手后先执行：

1. 完整阅读本文件、`Agent.md`、`CURRENT_ROUTE.md` 和 `C1_SPEC.md`；以 2026-07-21 的 decision-first 路线及 D5/D6 正式报告为当前事实。
2. 运行 `git status --short --branch`，确认没有未知改动。
3. 验证 `submission/d4_small/decision_head.json` 的 SHA256。
4. 不等待 U1-B/state 人评，也不要继续搜索 D5/D6 action-history 特征或门槛；等待并独立验证同学交付的 frame/history 输入配置。
5. 不要重复同学负责的帧采样/对话历史长度搜索，等待其最终配置后做集成和回归验证。

## 1. 任务到底是什么

### 1.1 官方任务

Challenge 1 要求模型针对视频流中的每个候选时间区间，结合：

- 当前 query；
- 截止到该候选区间的因果视频；
- 候选区间之前、按 chunk 对齐的累计 `dialog`；

输出以下两种格式之一：

- `$silent$`
- `$interrupt$<utterance>`

官方分类分数关注 silent/interrupt 的 Macro F1。interrupt 后的语言质量也关系到系统实际效果和人工质量判断，因此本项目将问题拆成两个相互衔接但应独立诊断的模块：

1. **Decision / Gate**：此时是否应该主动打断；
2. **Utterance / Planner**：如果打断，具体应该说什么。

Small 组要求总参数量小于 2B。当前 D4 模型总参数和 active 参数均为 `1.060898844B`，符合 Small 组。

### 1.2 官方验证与测试阶段的输入边界

- 当前是 Validation Phase，本地使用公共的 700-session validation 文件进行实验和生成 `predictions.jsonl`。
- 隐藏 Test Phase 将由组织方运行参赛者提交的 Docker；官方模板预计 2026-08-08 发布，最终 CMD、挂载路径、时间/显存限制仍未知。
- 用户与同学此前确认隐藏测试会提供按候选 chunk 对齐的 `dialog`；2026-07-22 对官方公开规则复核后，只能确认公开集和 starter runner 的该契约，隐藏字段与历史来源仍需组织方或 2026-08-08 Docker 模板确认。因此当前 D4 可部署性是主假设，不是正式规则保证。
- 隐藏测试不会提供 gold `answers`。D4 提交入口默认拒绝任何含 `answers` 的输入，并在推理前再次移除该字段。
- 公共 validation JSONL 本身含 `answers`，仅本地审计时可显式传入 `--allow-input-answers-for-local-audit`；该开关不允许模型看到标签。
- 帧采样数、对话历史长度等推理策略可以调整，但这一方向目前由合作同学负责。我们这边负责 decision、utterance/state、提交契约和最终集成。

## 2. 当前正在做什么

当前工作分成三条线，优先级从高到低如下。

### 2.1 主线 A：把 D4 固化成可交付的 Small 模型

D4 是当前最好的 decision candidate。它在 D1 特征上增加了因果可见的 dialog-stage 标量，用来表达当前对话已经推进到什么阶段，再用一个轻量线性 logistic head 判断是否 interrupt。

当前状态：模型侧提交入口已经完成，关键 learned head 已移到 Git 跟踪的 `submission/d4_small/` 中。等官方 Docker 模板发布后，只需把现有 CLI 映射进官方容器契约，再跑一次等价性 smoke 和完整资源审计。

### 2.2 主线 B：改善 utterance 的早期冷启动和视觉依据

人工审阅显示：

- 较早 chunk 更容易没有自然输出；
- 中后段因为已有更多对话历史，语言更容易生成得像样；
- 模型经常根据此前对话推测下一步，而不是充分利用当前视频；
- 去掉当前区间视频并没有明显恶化 fallback 率，说明视觉确实会影响措辞，但 grounding 不稳定；
- 去掉 assistant history 后 80/80 都退化成 fallback，说明当前 utterance 对语言历史依赖很强。

因此下一步不是立刻重做大规模生成训练，而是在固定 D4 gate 的前提下，专门诊断并改善早期 chunk 的 utterance cold-start 与视觉 grounding。

### 2.3 冻结的内容评测线

U0-A/U0-B 已完成并聚合，U2 自动诊断也已完成；U1-B 和 state package 仍未闭环。由于当前人力有限、官方指标只评决策，且当前主方案假设隐藏推理沿用公开 dialog 契约，新增人工评测已经暂停。这些资产只作为后续内容质量证据保存，不阻塞 decision-first 的自动排行榜实验；隐藏 dialog 契约本身仍是 P0 外部风险。

## 3. 已完成工作与主要进展

### 3.1 官方资料、论文与路线纠偏

- 已审计官方任务、starter kit、规则、提交阶段和 Small 参数约束。
- 已完成 PWR 正文和附录的逐节审计，见 `literature/papers/challenge1_proactive/PWR_audit.md`。
- 已确认目前无法获得 PWR 官方 training code 或 weights，不能把它当作可直接复现的工程依赖。
- 已重新定位 STRIDE：它可作为步骤/动作区间、边界和状态 schema 的参考或可选预训练数据，但 **不能直接当作 C1 interrupt 标签**。
- STRIDE 多源底层视频存在 NC 或许可不清晰问题，外层数据集的 CC-BY-4.0 不能自动覆盖底层视频条款；在不能取得组织方书面确认的情况下，不应让它成为获奖模型的关键依赖。
- 大规模 STRIDE 数据位于 `/data2/download`，约 1.7T，属于外部大数据区。不要为普通代码检查递归扫描、复制或纳入 Git。

### 3.2 R0 到 D4 的实验演进

| 阶段 | 核心变化 | Macro F1 | 结论 |
|---|---|---:|---|
| R0 | InternVL3.5-1B 直接按官方格式生成，不使用 plan | 0.4630 | 正式基础线 |
| R0-F | 只做输出格式修复 | 0.5362 | 证明格式错误损失明显；使用 val 标签监督，不是干净泛化证据 |
| D1 scalar | 从生成过程提取因果标量，训练轻量 gate | 0.6119 | decision head 明显有效 |
| D1 fused | 标量 + tag margin + 1024 维隐藏表征 | 0.6341 | 成为后续主干 |
| D2 residual MLP | 更复杂的残差 MLP head | 0.6351 | 增益小且不稳定，拒绝晋级 |
| D2 final-MLP LoRA | 只调语言模型最后 MLP 的 LoRA 对照 | 0.6357 | 增益小且不稳定，拒绝晋级 |
| D3 dynamics | 加入因果动态特征 | 0.6690 | 按冻结协议晋级 |
| D4 dialog-stage | D1 fused + 8 个对话阶段标量 | 0.6846 | 当前最优 OOF 诊断候选 |
| D4 all-public refit | 全部 700 session 重拟合单一 head | 0.7393 train-fit | 仅训练集拟合 sanity，不是泛化分数 |

关键口径：

- **D4 的 0.6846** 来自 5 折、按 session 划分后合并的 OOF predictions；它对应 5 个 fold head，不是一个可直接用于隐藏测试的单模型。
- **D4 的 0.7393** 是单个 all-public refit head 在自身训练数据上的拟合分数，只用于检查训练是否正常，不能与 OOF 分数等价比较。
- 隐藏测试 Docker 使用的是单一 all-public refit head。
- D4 的原始 D3-D 诊断协议带有 public-validation label supervision，因此 `validation_evidence.json` 明确标记为不可按旧 promotion protocol 自动宣称通过。

对应报告：

- `reports/20260713_internvl35_1b_no_plan_r0.md`
- `reports/20260714_internvl35_1b_r0f_format_ablation.md`
- `reports/20260715_internvl35_1b_neural_decision_head_d1.md`
- `reports/20260715_internvl35_1b_residual_mlp_d2.md`
- `reports/20260716_internvl35_1b_final_mlp_lora_oof.md`
- `reports/20260717_internvl35_1b_causal_dynamics_d3.md`
- `reports/20260719_internvl35_1b_d4_dialog_stage_candidate.md`

### 3.3 D4 模型、管线与提交入口

D4 的单个候选区间推理流程如下：

```text
query + 截止当前区间的视频 + 截止当前区间的 dialog
  -> InternVL3.5-1B shared-vision 前向
  -> D1 fused 特征
     = 18 个因果 scalar + tag margin + 1024 维 hidden
  -> 追加 8 个 dialog-stage scalar
  -> 标准化线性 logistic head（1052 参数）
  -> threshold = 0.1263874797442615
  -> silent 或 interrupt
  -> 若 interrupt，沿用语言生成结果形成 $interrupt$<utterance>
```

D4 final head 配置：

- backbone：`OpenGVLab/InternVL3_5-1B-HF`
- revision：`9191dbccf312b537016f041b25d61c72e7c5c9f3`
- backbone 参数：`1,060,897,792`
- decision head 参数：`1,052`
- 总参数/active 参数：`1,060,898,844`，即 `1.060898844B`
- L2：`0.01`
- threshold：`0.1263874797442615`
- 推理 dtype：`bfloat16`
- attention implementation：`sdpa`
- 当前冻结帧配置：每候选区间 16 帧、最多 32 帧
- 当前冻结历史：最多 4 轮
- 最大新 token：64
- decision feature mode：`shared_vision`

已完成的工程验证：

- D4 online replay：`9,935/9,935` 个 decision 全部匹配，最大 logit 差 `2.55e-7`；
- 旧 deploy 路径的 shared-vision GPU smoke：最大 logit 差 `6.32e-8`；
- submission adapter CPU preflight 已完成；
- submission adapter 单 session、10 chunks GPU smoke 已完成；
- GPU smoke wall time `44.206s`，峰值显存 `3,466,037,248` bytes；
- adapter 输出与冻结 D4 smoke 逐字节一致；
- smoke prediction SHA256：`cb79b4573dd3551fefafb219b5685be4a0d5d7b1e85bb74d1bf023f67c175de9`；
- R0、D1、D3、D4 四组回归测试共 `48/48` 通过。

详细审计见：`reports/20260720_internvl35_1b_d4_submission_entrypoint_audit.md`。

## 4. D4 checkpoint 与路径改动核验

用户为了 Git 管理调整了 D4 checkpoint 和入口路径。核验结果如下。

### 4.1 两个路径分别承担什么职责

实验产物原始路径：

```text
output/experiments/20260719_internvl35_1b_d4_dialog_stage_final_v1/decision_head.json
```

Git 跟踪的提交副本：

```text
submission/d4_small/decision_head.json
```

两份文件已经用字节比较核验为完全一致，SHA256 均为：

```text
531431710a01a71bdd02ffd7a9758428fe282323cc41fae2c1d6859e45408b13
```

`output/` 被 `.gitignore` 忽略，适合保存实验全量产物，但不能保证新 Agent 在 clone 后拥有 learned head。因此将 head 复制到 `submission/d4_small/` 并纳入 Git 是正确的。

### 4.2 为什么冻结 config 仍指向 output

`configs/d4_internvl35_1b_dialog_stage_deploy_shared_vision_v1.json` 中仍然记录：

```text
output/experiments/20260719_internvl35_1b_d4_dialog_stage_final_v1/decision_head.json
```

这是有意保留的 **实验 provenance**，用于说明该冻结实验当时实际读取哪个产物。不要仅因 bundle 路径变化而改写历史 config。

真正的提交入口 `src/proactive_d4/submission.py`：

- 默认 `DEFAULT_HEAD` 指向 `submission/d4_small/decision_head.json`；
- 在 `build_runtime_config()` 中用 CLI/default 的 bundle head 覆盖运行时 `decision_head.path`；
- 测试 `test_bundled_head_matches_manifest` 会检查 bundle 文件存在且哈希与 manifest 一致。

因此“冻结 config 指向 output、提交入口默认指向 submission”不是冲突，而是“实验来源记录”和“可移植运行路径”的职责分离。

### 4.3 相关 Git 状态

- 根仓库现在是正常 Git 仓库，`main` 与 `origin/main` 对齐；
- 根仓库 remote 为 `git@github.com:JashinJam/Proactive.git`；
- 交接基线提交为 `3ed2e0a`；
- 该提交还修复了 `src/proactive_d1/run_deploy.py` 对 `/tmp` 运行时 config 做 `relative_to(PROJECT_ROOT)` 时可能报错的问题；
- 根目录 `README.md` 和 `Agent.md` 的部分文字仍写着“umbrella root 不是 Git repo”，这已经滞后。以实际 Git 状态和本交接为准，后续可单独清理描述；
- `STRIDE/` 与 `wearable-ai-leaderboard/` 是被根仓库忽略的嵌套仓库，分别保持自己的 Git 状态，不应作为根仓库文件直接管理；交接检查时前者在 `ljx` 分支且 clean，后者在 `main` 分支且 clean。

## 5. Utterance、人工评测与 state 路线现状

### 5.1 U0/U1 是什么

- **U0**：审计当前 gate 下原始 utterance 的质量，建立语言质量基准。
- **U1**：固定 gate 后强制尝试生成，比较当前 fallback、forced no-state、oracle-state 等条件，判断问题来自“不会说”、缺状态，还是缺视觉依据。
- 这两类实验主要用于分析 utterance，不应与 decision Macro F1 混成一个指标。

### 5.2 已有 A 评测员结论

报告：`reports/20260718_u0_u1_reviewer_a_diagnostic.md`。

主要发现：

- forced no-state 的 content composite 相对 current fallback 提高 `+1.1725`；
- bootstrap 置信区间为 `[+0.8875, +1.45]`；
- 但 hallucination 增加 `2.5pp`，超过预注册的 `+2pp` 门槛，因此不能直接宣告 U1 晋级；
- second chunks 中 80% 为 fallback，强制输出反而比固定 fallback 更差；
- mid/late chunks 明显更适合生成，说明早期缺上下文是核心困难。

视觉依赖报告：`reports/20260719_u1_visual_reliance_and_d3_dialog_policy_control.md`。

- 移除 assistant history 后，80/80 退化为 fallback；
- 移除当前 interval 视频没有进一步恶化 fallback；
- 视觉会改变措辞，但视觉 grounding 尚不稳定；
- 不能仅凭语言历史的输出流畅度判断模型真正看懂了视频。

### 5.3 人工评分文件的最新实际状态

| 实验 | 评测员 | 状态 | 路径 | SHA256 / 备注 |
|---|---|---|---|---|
| U0 | A | 已完成，200 条 | `output/human_reviews/u0/reviewer_A/ratings.csv` | `4e1476f6576150d1e135b9ad8ef047029c04829d78a234114e93592d6d0c9feb` |
| U0 | B | **已完成，200 条** | `output/human_reviews/u0/reviewer_B/ratings.csv` | `7d9eb9fbea15bdfd5b89d4c4165268b05442f65e42ce8ef8c085ae1a7b93d750` |
| U1 | A | 已完成，160 条 | `output/human_reviews/u1/reviewer_A/ratings.csv` | `e7e0b1784abccd62e6e4e34190d83e1b2ec78403a8fcd2c4f910b998bdc2ef5d` |
| U1 | B | 未完成 | 预期位于 `output/human_reviews/u1/reviewer_B/` | 等待评测员 |
| U1 state 240 candidates | - | 未完成 | 见 state review package | 不阻塞 U0 双人分析 |

补充：

- U0-A 的 Git 跟踪副本：`src/proactive_u0/u0_reviewer_A_ratings.csv`，与 output 文件一致；
- U1-A 的 Git 跟踪副本：`src/proactive_u1/u1_reviewer_A_ratings.csv`，与 output 文件一致；
- U0-B 的项目副本为 `src/proactive_u0/u0_reviewer_B_ratings.csv`，双人聚合报告为 `reports/20260720_u0_dual_reviewer_analysis.md`；
- 准确状态是 U0 A/B 已完成并聚合，U1-B 与 state-package ratings 未完成；当前不再为它们投入新的大规模人工时间；
- 现有 `proactive_u1.ratings` 只适用于 U1 的 `current_fallback/forced_no_state` 对比，不能直接拿来聚合 U0；
- 当前没有专用的 U0 双评测员聚合器。下一步应先实现并测试它，不能强行复用 U1 分析逻辑。

人工评测网页入口：

- 启动脚本：`scripts/human_review_server.sh`
- 前端：`presentation/human_review/`
- 后端：`src/proactive_review/server.py`
- 使用说明：`presentation/human_review/README.md`

### 5.4 State 路线

S0 是零样本 plan/state 解码诊断：

- official-dialog mean task Macro：`0.2891`；
- no-assistant：`0.1597`；
- 低于预设 `0.35` 门槛，说明当前零样本 state 还不足以成为稳定模块。

S1 是监督 state decoder 的准备路线：

- 计划 32 sessions / 444 states；
- 当前仅完成 2/32 sessions、23/444 states；
- 还剩 421 个 state；
- 相关资产在 `annotations/state_s1_decoder_v1/` 和 `annotations/state_s1_decoder_v1/work_v1/`。

S1 目前暂停。只有在人工 state 评分通过冻结门槛，或新的独立 residual audit 明确发现可重复的 step/progress 错误时，才值得继续投入。不要因为“粒度可能重要”就直接启动大规模 state 标注或训练。

## 6. 当前瓶颈和未决问题

### 6.1 官方容器模板尚未发布

当前只有模型侧 adapter，不能提前猜最终 base image、CMD、挂载目录、网络策略、运行时限和健康检查。等 2026-08-08 官方模板发布后再做最后一层封装。

### 6.2 项目源码许可证未决定

- backbone weights：Apache-2.0；
- 官方 starter kit 和官方数据：CC-BY-NC-4.0；
- 当前项目根目录没有 top-level `LICENSE`；
- `submission/d4_small/manifest.json` 因此把 project source license 标记为 unresolved。

这可能影响获奖模型的源码资格与提交材料。选择项目源码许可证是项目负责人的决策，Agent 不应擅自添加许可证。还应在官方模板发布后重新检查 starter kit 是否需要随镜像分发、如何满足其条款。

### 6.3 D4 的公共验证证据有口径限制

- 0.6846 是 OOF 诊断，不代表单一 all-public model 的隐藏测试分数；
- 0.7393 是 train-fit sanity，不是泛化分数；
- 目前拿不到官方 test split 结果；
- Validation Phase 是否上传 5-head OOF predictions，还是某个可明确说明来源的单模型 predictions，仍需负责人决定；
- 不能把 OOF 文件暗示成“单个可部署模型的预测”。

### 6.4 Utterance 的早期冷启动与语言捷径

较早 chunk 的视频证据尚不足或模型不会有效读取，历史对话却能让中后段输出明显变好。模型可能在“根据对话猜流程”，而不是准确识别视频中已发生的动作。下一轮实验必须把“语言变流畅”和“视觉依据更可靠”分开测。

### 6.5 人工评测路线已暂停扩张

U0 A/B 聚合和 U2 自动诊断已经完成；U1-B 和 state 评分仍未闭环，但不再阻塞排行榜决策开发。保留全部冻结资产，不新增大规模人工评分、state 标注或 utterance 训练；只有官方后续增加内容指标或新的自动证据明确要求时才恢复。

### 6.6 同学路线的集成边界

合作同学负责帧采样策略和对话历史长度优化。我们可以验证这些策略在官方 Docker 输入中可用，但不应重复跑相同搜索。等同学给出最终配置、代码或 manifest 后，再合并到 D4 adapter，并重跑 preflight、GPU smoke、哈希和输出契约检查。

## 7. 下一步计划

### P0：D5 自动决策融合（已完成，未晋级）

已严格执行 `annotations/d5_decision_fusion_v1/PROTOCOL.md`：D4 精确复现 `0.6846`；action-history 为 `0.6889`；主并集为 `0.6912`。但 bootstrap 95% 下界为 `-0.00025`，三组稳定性 split 差值为 `+0.0072/-0.0007/+0.0016`，因此未通过门槛。完整结论见 `reports/20260721_internvl35_1b_decision_fusion_d5.md`。

D5 v1 不做 all-public refit、部署集成或窗口/L2/threshold 事后调优。其低维结构化 calibration 后续已由 D6 独立冻结并完成。

### P1：D6 低维结构化门槛（已完成，明确失败）

D6 保持每折 D4 head、L2 和 logit 完全相同，不追加特征列，只按 position、previous action
或最近两次 action 调整 calibration-fold threshold。D4 精确复现 `0.6846`；position 控制为
`0.6855`（`+0.0009`，区间跨 0），主候选 `last2_shrunk` 为 `0.6747`
（`-0.0099`），bootstrap 为 `[-0.01482,-0.00492]`，0/5 folds、0/4 domains 为正，三组
新 split 差值为 `-0.0080/-0.0019/-0.0066`。独立复现的核心工件逐字节一致。完整结论见
`reports/20260721_internvl35_1b_structured_calibration_d6.md`。

D6 不做 threshold transport、部署集成或 GPU smoke；不能把 position 小正控制事后改为主
候选，也不在同一公共数据上改分组、收缩常数或局部门槛目标继续搜索。D5/D6
action-history 分支到此关闭，D4 保持不变。

### P2：接收并集成同学的帧/历史配置

需要对方交付：最终配置、代码差异、所用数据范围、随机种子、指标、运行资源和模型输出样例。合并后：

1. 更新 submission manifest 中冻结推理配置；
2. 确认输入仍然只使用因果视频和 chunk-aligned dialog；
3. 重跑 48 个相关单元测试；
4. 重跑 submission CPU preflight；
5. 在启动前检查 GPU 占用，轻占用时可以运行，但绝不能终止或抢占用户的重要训练进程；
6. 重跑单 session GPU 等价性 smoke，并更新 receipt/hash/report。

### P3：官方 Docker 模板发布后的收口

1. 以官方模板为准创建 Dockerfile/CMD，不猜测契约。
2. 将输入、视频、模型、输出挂载映射到 `python -m proactive_d4.submission`。
3. 确保运行时无网络依赖，模型和 head 都在镜像或官方允许的挂载中。
4. 对隐藏输入模式禁止 `answers`，输出只能含 `video_path` 和 `answers`。
5. 审计显存、时间、磁盘、依赖和许可证。
6. 由负责人授权后再上传 Docker 或 validation predictions。

### P4：有证据再恢复 state / granularity / SFT

- 若 early-utterance residual 明确表现为“步骤位置识别错误”，可恢复小规模 S1；
- 若只是语言表达或视觉 grounding 问题，应优先解决 decoder/视觉使用，不做大规模 state 标签；
- STRIDE 只能做非关键的边界/步骤预训练探索，且要先解决数据来源许可；
- GRPO 不是当前默认下一步，必须在 reward、离线评估和稳定监督基线成熟后再考虑。

## 8. 工程组织结构

```text
wearable_ai_challenge/
├── AGENTS.md                         # 指示 Agent 首先阅读 Agent.md
├── Agent.md                          # 项目规范、事实层级、操作红线（部分状态文字已滞后）
├── CURRENT_ROUTE.md                  # 当前路线摘要（人评状态需更新）
├── C1_SPEC.md                        # 稳定的官方任务事实与输入输出契约
├── README.md                         # 项目导航（“根目录非 Git repo”描述已滞后）
├── handover.md                       # 本交接文件
├── configs/                          # 冻结实验与 deploy 配置
├── models/                           # 模型元数据，不存放 2GB backbone weights
├── src/
│   ├── proactive_r0/                 # 官方因果基线、数据读取、模型运行
│   ├── proactive_r0f/                # 输出格式修复对照
│   ├── proactive_r1/                 # oracle state pilot
│   ├── proactive_d1/                 # scalar/fused gate、shared-vision、通用 deploy runner
│   ├── proactive_d2/                 # residual MLP / final-MLP LoRA 对照，均未晋级
│   ├── proactive_d3/                 # causal dynamics、dialog control、final/deploy
│   ├── proactive_d4/                 # dialog-stage head、final/deploy、submission adapter
│   ├── proactive_d5/                 # D3/D4/action-history 自动 OOF 融合，未晋级
│   ├── proactive_u0/                 # utterance baseline audit
│   ├── proactive_u1/                 # forced utterance、ratings、state ratings、视觉消融
│   ├── proactive_review/             # 双盲人工评测网页后端
│   ├── proactive_state_s0/            # 零样本 state 诊断
│   ├── proactive_state_s1/            # 监督 state decoder 准备/训练，当前暂停
│   ├── data/                          # 项目数据工具
│   ├── evaluation/                    # 项目评估工具
│   └── training/                      # 训练通用组件
├── annotations/                      # 冻结协议、oracle/state 标注与工作包
├── submission/d4_small/              # Git 跟踪的 D4 提交 bundle 与 learned head
├── reports/                          # 正式中文实验报告
├── explain/                          # 面向理解的通俗说明
├── literature/                       # 文献综述、PWR/相关工作审计
├── presentation/                    # dashboard、人工评测页面、汇报材料
├── scripts/                          # 启动和运维脚本
├── data -> /data1/wearable_ai_challenge_data
│                                      # 外部官方数据/starter 的符号链接，根仓库忽略
├── output/                            # 全量实验/评测产物，根仓库忽略
├── STRIDE/                            # 独立嵌套 Git repo，根仓库忽略
└── wearable-ai-leaderboard/           # 独立嵌套 Git repo，根仓库忽略
```

根仓库 `.gitignore` 重点忽略：`/data`、`/output/`、`/STRIDE/`、`/wearable-ai-leaderboard/`。新 clone 只会得到源代码、报告、配置和 `submission/d4_small/decision_head.json`，不会得到公共数据、完整实验输出或 InternVL backbone。

历史归档：

```text
/home/lanjinxin/workspace/deprecated/wearable_ai_challenge/2026-07-13_pre_pwr_reset/
```

日常历史日志：

```text
/home/lanjinxin/workspace/daily_work_log.md
```

日志是 append-only 历史证据，不是当前事实的最高优先级。当前状态优先看本文件、最新报告、冻结 config、manifest 和实际产物。

## 9. 重要文件入口

### 9.1 项目与任务

| 用途 | 文件 |
|---|---|
| Agent 规范 | `Agent.md` |
| 当前路线 | `CURRENT_ROUTE.md` |
| 官方任务稳定事实 | `C1_SPEC.md` |
| 文献总览 | `literature/literature_review.md` |
| PWR 正式审计 | `literature/papers/challenge1_proactive/PWR_audit.md` |
| 原 SFT/GRPO 计划（已归档，仅作历史参考） | `/home/lanjinxin/workspace/deprecated/wearable_ai_challenge/2026-07-13_pre_pwr_reset/project/SFT+GRPO_plan.md` |

### 9.2 D3/D4/D5/D6

| 用途 | 文件/目录 |
|---|---|
| D3 报告 | `reports/20260717_internvl35_1b_causal_dynamics_d3.md` |
| D4 候选报告 | `reports/20260719_internvl35_1b_d4_dialog_stage_candidate.md` |
| D4 提交审计 | `reports/20260720_internvl35_1b_d4_submission_entrypoint_audit.md` |
| D4 deploy 配置 | `configs/d4_internvl35_1b_dialog_stage_deploy_shared_vision_v1.json` |
| 通用 deploy runner | `src/proactive_d1/run_deploy.py` |
| D4 submission CLI | `src/proactive_d4/submission.py` |
| D4 submission tests | `src/proactive_d4/tests/` |
| Git 跟踪的 D4 head | `submission/d4_small/decision_head.json` |
| 提交 manifest | `submission/d4_small/manifest.json` |
| 证据与口径 | `submission/d4_small/validation_evidence.json` |
| 提交使用说明 | `submission/d4_small/README.md` |
| 测试过的依赖 | `submission/d4_small/requirements-tested.txt` |
| D5 冻结协议 | `annotations/d5_decision_fusion_v1/PROTOCOL.md` |
| D5 OOF 配置 | `configs/d5_internvl35_1b_decision_fusion_oof_v1.json` |
| D5 实现与测试 | `src/proactive_d5/` |
| D5 正式报告 | `reports/20260721_internvl35_1b_decision_fusion_d5.md` |
| D6 冻结协议 | `annotations/d6_structured_calibration_v1/PROTOCOL.md` |
| D6 OOF 配置 | `configs/d6_internvl35_1b_structured_calibration_oof_v1.json` |
| D6 实现与测试 | `src/proactive_d6/` |
| D6 正式报告 | `reports/20260721_internvl35_1b_structured_calibration_d6.md` |

### 9.3 Utterance、人评与 state

| 用途 | 文件/目录 |
|---|---|
| utterance/planner 问题交接 | `reports/20260716_d1_utterance_planner_language_handoff.md` |
| U0 审计 | `reports/20260716_d1_utterance_u0_audit.md` |
| U1 预注册 | `reports/20260716_u1_fixed_gate_forced_generation_prereg.md` |
| 双盲人评协议 | `reports/20260717_u0_u1_human_review_protocol.md` |
| A 评测员诊断 | `reports/20260718_u0_u1_reviewer_a_diagnostic.md` |
| 视觉依赖诊断 | `reports/20260719_u1_visual_reliance_and_d3_dialog_policy_control.md` |
| 人评服务 | `scripts/human_review_server.sh` |
| U0 源码/冻结 A 评分 | `src/proactive_u0/` |
| U1 源码/冻结 A 评分 | `src/proactive_u1/` |
| S0 报告 | `reports/20260717_internvl35_1b_oracle_plan_state_s0.md` |
| S1 资产 | `annotations/state_s1_decoder_v1/` |

### 9.4 关键实验产物

| 产物 | 路径 |
|---|---|
| R0 | `output/experiments/20260713_internvl35_1b_no_plan_r0/` |
| D1 OOF | `output/experiments/20260714_internvl35_1b_neural_decision_head_d1_oof_v1/` |
| D3 OOF | `output/experiments/20260717_internvl35_1b_causal_dynamics_d3_oof_v1/` |
| D4 OOF predictions | `output/experiments/20260718_internvl35_1b_d3_dialog_policy_control_v1/variants/d1_fused_plus_dialog_stage/predictions.jsonl` |
| D4 final head 原始产物 | `output/experiments/20260719_internvl35_1b_d4_dialog_stage_final_v1/decision_head.json` |
| D4 CPU preflight | `output/experiments/20260720_internvl35_1b_d4_submission_preflight_v1/` |
| D4 adapter GPU smoke | `output/experiments/20260720_internvl35_1b_d4_submission_entrypoint_smoke_v1/` |
| D5 OOF 与稳定性结果 | `output/experiments/20260721_internvl35_1b_decision_fusion_d5_oof_v1/` |
| D6 OOF 与稳定性结果 | `output/experiments/20260721_internvl35_1b_structured_calibration_d6_oof_v1/` |
| U1 no-state 分析包 | `output/experiments/20260716_internvl35_1b_fixed_gate_forced_generation_u1_v1_nostate_full/analysis/` |
| U1 state review 包 | `output/experiments/20260717_internvl35_1b_fixed_gate_forced_generation_u1_v1_oracle_formal_full/state_review/` |

## 10. 本地环境与依赖

### 10.1 Python 和模型

推荐环境：

```text
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python
```

backbone：

```text
/home/lanjinxin/model_weights/InternVL3_5-1B-HF
```

模型元数据：

```text
models/internvl35_1b_hf.json
```

base `model.safetensors` 预期 SHA256：

```text
11effd1da2fc0929957d56de4129d1e7d2aed044ea878d40bf83e95b63d38b39
```

### 10.2 官方数据与 scorer

公共 validation JSONL：

```text
data/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl
```

官方 scorer：

```text
data/starter_kit/run_evaluation.py
```

`data` 是指向 `/data1/wearable_ai_challenge_data` 的符号链接。新机器或新 clone 必须自行恢复该挂载/链接，不能假设 Git 包含数据。

## 11. 常用验证命令

所有命令均在项目根目录运行。

### 11.1 验证 D4 bundle head

```bash
sha256sum submission/d4_small/decision_head.json
# 预期：531431710a01a71bdd02ffd7a9758428fe282323cc41fae2c1d6859e45408b13
```

### 11.2 D4 submission 单元测试

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m unittest discover -s src/proactive_d4/tests -v
```

### 11.3 四组关键回归测试

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python -m unittest discover -s src/proactive_r0/tests -v
PYTHONNOUSERSITE=1 PYTHONPATH=src /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python -m unittest discover -s src/proactive_d1/tests -v
PYTHONNOUSERSITE=1 PYTHONPATH=src /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python -m unittest discover -s src/proactive_d3/tests -v
PYTHONNOUSERSITE=1 PYTHONPATH=src /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python -m unittest discover -s src/proactive_d4/tests -v
```

当前预期：R0 `12/12`、D1 `18/18`、D3 `8/8`、D4 `10/10`，合计 `48/48`。

测试可能打印 `pynvml` 直接包已废弃的 FutureWarning；容器依赖使用 `nvidia-ml-py`，该 warning 当前不是功能故障。

### 11.4 D4 CPU-only preflight

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d4.submission \
  --input-jsonl data/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl \
  --video-dir data/egoproactive/val \
  --model-dir /home/lanjinxin/model_weights/InternVL3_5-1B-HF \
  --starter-kit-dir data/starter_kit \
  --work-dir /tmp/d4_handoff_preflight \
  --preflight-only \
  --allow-input-answers-for-local-audit \
  --max-sessions 1
```

没有显式传 `--head-path` 时，会使用 Git 跟踪的 bundle head。

### 11.5 未来 Docker 中的模型侧入口

```bash
python -m proactive_d4.submission \
  --input-jsonl /input/egoproactive_test.jsonl \
  --video-dir /input/videos \
  --model-dir /opt/models/InternVL3_5-1B-HF \
  --head-path /opt/wearable_ai/submission/d4_small/decision_head.json \
  --starter-kit-dir /opt/wearable_ai/starter_kit \
  --output-jsonl /output/predictions.jsonl \
  --work-dir /tmp/d4_runtime \
  --device cuda:0
```

上面路径只是 adapter 的预模板示例；最终路径必须服从官方模板。

## 12. 不要踩的坑

1. 不要把 D4 OOF `0.6846` 说成单一部署模型的验证分数，也不要把 train-fit `0.7393` 当成泛化性能。
2. 不要在隐藏测试路径打开 `--allow-input-answers-for-local-audit`；该参数仅用于含 gold labels 的公共文件做契约检查。
3. 不要读取未来视频、未来 dialog 或 `answers`。dialog 可用不等于 gold interrupt 标签可用。
4. 不要把 STRIDE 的步骤边界标签直接改造成 C1 interrupt 标签，也不要在许可未解决时把 STRIDE 变成关键依赖。
5. 不要随意改写冻结 config 中的历史路径；D4 config 原始路径与 submission runtime 路径是有意分工。
6. 不要认为 `output/` 里的关键文件会随 Git 自动交接。新增 learned artifact 若是部署必需，必须进入明确 bundle 并记录哈希。
7. 不要覆盖或删除用户/同学正在进行的实验。启动 GPU 任务前先检查进程和显存；轻占用时可以启动，但不能终止重要进程。
8. 不要重复同学负责的帧采样和历史窗口搜索；以集成验证为主。
9. 不要在没有用户明确授权时上传 predictions、push Docker、发布模型或做其他外部状态变更。
10. 不要擅自选择项目许可证。
11. 不要把尚未完成的 U1-B 或 state 人评写成已完成；U0 A/B 已经完成并聚合，不要恢复旧的“U0-B 未分析”状态。
12. 不要恢复已经归档的 pre-PWR 路线，除非新的证据明确支持。

## 13. 外部提交状态和表单信息

截至本交接：

- validation `predictions.jsonl`：未上传；
- Docker image：未上传；
- leaderboard 官方结果：没有；
- 任何上传：需要用户授权。

当前隐藏测试单模型表单信息：

```text
Model name: InternVL3.5-1B-D4-DialogStage
Model license: Apache-2.0
Total params (billions): 1.060898844
Active params (billions): 1.060898844
```

注意：这里的 `Model license` 指 backbone 模型权重许可证，不等于本项目源码许可证已经解决。

## 14. 交接完成的判定

下一位 Agent 接手后，应能清楚回答：

- 官方每个 chunk 给什么输入、要输出什么；
- D4 为什么优于 D3，以及 0.6846 与 0.7393 的口径差异；
- 为什么有两份 D4 head 路径，哪个被 Git 跟踪、哪个是历史 provenance；
- 当前哪些人评已完成，为什么新的 U1/U2 人评仍处于 decision-first 暂停状态；
- D5 为什么没有晋级、D6 为什么说明分组局部门槛不适合全局 Macro；
- 为什么当前不再继续搜索 action-history 特征/门槛，而是等待同学的输入策略集成；
- 哪些工作由合作同学负责，哪些是本项目当前主线；
- 为什么 STRIDE/state/granularity/GRPO 暂时不是默认主路线；
- 等官方 Docker 模板发布后还需要完成哪些收口工作。

如实际文件、冻结 manifest、最新报告与旧说明发生冲突，先保存证据并核验时间戳、Git 提交和 SHA256，不要凭旧摘要覆盖新状态。
