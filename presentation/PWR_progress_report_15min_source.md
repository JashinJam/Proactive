<style>
.jp-RenderedHTMLCommon, .reveal, .markdown-body, .rendered_html {
  font-family: "Noto Sans SC", "Microsoft YaHei", system-ui, sans-serif;
  color: #17202a;
  line-height: 1.62;
  letter-spacing: 0;
}
.jp-RenderedHTMLCommon h1, .jp-RenderedHTMLCommon h2, .jp-RenderedHTMLCommon h3,
.reveal h1, .reveal h2, .reveal h3,
.markdown-body h1, .markdown-body h2, .markdown-body h3,
.rendered_html h1, .rendered_html h2, .rendered_html h3 { letter-spacing: 0; }
.jp-RenderedHTMLCommon h2, .reveal h2, .markdown-body h2, .rendered_html h2 {
  margin: 0 0 22px;
  padding-bottom: 10px;
  border-bottom: 2px solid #276fbf;
  font-size: 27px;
  line-height: 1.28;
}
.jp-RenderedHTMLCommon h3, .reveal h3, .markdown-body h3, .rendered_html h3 { margin: 0 0 9px; font-size: 17px; }
.jp-RenderedHTMLCommon p, .jp-RenderedHTMLCommon li,
.reveal p, .reveal li, .markdown-body p, .markdown-body li,
.rendered_html p, .rendered_html li { font-size: 14px; line-height: 1.62; }
.jp-RenderedHTMLCommon table, .reveal table, .markdown-body table, .rendered_html table {
  width: 100%;
  margin: 14px 0 18px;
  border-collapse: collapse;
  font-size: 13px;
  line-height: 1.48;
}
.jp-RenderedHTMLCommon th, .jp-RenderedHTMLCommon td,
.reveal th, .reveal td, .markdown-body th, .markdown-body td,
.rendered_html th, .rendered_html td { padding: 9px 10px; border-bottom: 1px solid #d9e0e5; vertical-align: top; }
.jp-RenderedHTMLCommon th, .reveal th, .markdown-body th, .rendered_html th { background: #f1f4f5; color: #46535e; font-weight: 700; }
.report-title {
  margin-top: 6px;
  padding: 40px 8px 28px;
  border-top: 8px solid #276fbf;
  border-bottom: 1px solid #d9e0e5;
}
.report-title h1 { margin: 0 0 16px; font-size: 36px; line-height: 1.25; }
.report-title p { margin: 5px 0; color: #62707c; font-size: 17px; }
.result-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin: 28px 0 20px;
  border-top: 1px solid #d9e0e5;
  border-bottom: 1px solid #d9e0e5;
}
.result-strip > div { padding: 17px 18px; border-right: 1px solid #d9e0e5; }
.result-strip > div:last-child { border-right: 0; }
.result-strip strong { display: block; color: #276fbf; font-size: 27px; line-height: 1.2; }
.result-strip span { display: block; margin-top: 5px; color: #62707c; font-size: 13px; }
.report-grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; margin: 14px 0; }
.method-panel { padding: 18px 20px; border-top: 4px solid #276fbf; background: #f5f7f8; }
.method-panel.red { border-top-color: #bd3d3a; }
.method-panel.green { border-top-color: #23805a; }
.method-panel.amber { border-top-color: #a96513; }
.method-panel p { margin: 7px 0; color: #37434d; font-size: 14px; line-height: 1.62; }
.report-band { margin: 16px 0; padding: 14px 18px; border-left: 4px solid #276fbf; background: #f4f6f7; }
.report-band.red { border-left-color: #bd3d3a; }
.report-band.green { border-left-color: #23805a; }
.report-band.amber { border-left-color: #a96513; }
.report-band h3 { margin: 0 0 6px; }
.report-band p { margin: 0; font-size: 14px; line-height: 1.62; }
.literature-list { display: grid; gap: 0; border-top: 1px solid #d9e0e5; }
.literature-row {
  display: grid;
  grid-template-columns: 135px 175px minmax(0, 1fr);
  gap: 18px;
  padding: 15px 5px;
  border-bottom: 1px solid #d9e0e5;
  align-items: start;
}
.literature-row strong { color: #276fbf; font-size: 15px; }
.literature-row .date { color: #62707c; font-size: 12px; }
.literature-row p { margin: 0; font-size: 13px; line-height: 1.55; }
.pipeline { display: flex; align-items: stretch; gap: 8px; margin: 18px 0; }
.pipeline .node {
  flex: 1;
  min-height: 78px;
  padding: 15px 10px;
  border: 1px solid #d9e0e5;
  background: #fff;
  text-align: center;
  font-size: 13px;
  line-height: 1.48;
}
.pipeline .arrow { display: flex; align-items: center; color: #62707c; font-size: 18px; font-weight: 700; }
.decision-box { display: grid; grid-template-columns: 1fr 86px 1fr; align-items: stretch; margin: 20px 0; }
.decision-box .side { min-height: 112px; padding: 18px; border: 1px solid #d9e0e5; background: #fff; }
.decision-box .mid { display: flex; align-items: center; justify-content: center; color: #276fbf; font-weight: 700; text-align: center; }
.route-list { border-top: 1px solid #d9e0e5; }
.route-step {
  display: grid;
  grid-template-columns: 82px minmax(0, 1fr) 96px;
  gap: 16px;
  align-items: center;
  padding: 12px 4px;
  border-bottom: 1px solid #d9e0e5;
  font-size: 13px;
  line-height: 1.48;
}
.route-step strong { color: #276fbf; font-size: 14px; }
.route-step .status { text-align: right; font-weight: 700; }
.status-done { color: #23805a; }
.status-stop { color: #bd3d3a; }
.status-next { color: #a96513; }
.formula {
  margin: 18px 0;
  padding: 15px 18px;
  background: #17202a;
  color: #fff;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 14px;
  line-height: 1.5;
  text-align: center;
}
.feature-row {
  display: grid;
  grid-template-columns: 170px minmax(0, 1fr) 115px;
  gap: 18px;
  align-items: start;
  padding: 15px 5px;
  border-bottom: 1px solid #d9e0e5;
}
.feature-row strong { color: #276fbf; font-size: 15px; }
.feature-row p { margin: 0; font-size: 13px; }
.feature-row .dimension { color: #62707c; font-size: 13px; text-align: right; }
.sample-layout { display: grid; grid-template-columns: 1.35fr 0.9fr; gap: 24px; align-items: start; }
.sample-image { width: 100%; display: block; border: 1px solid #d9e0e5; }
.sample-facts { border-top: 1px solid #d9e0e5; }
.sample-fact { display: grid; grid-template-columns: 112px 1fr; gap: 12px; padding: 10px 0; border-bottom: 1px solid #d9e0e5; }
.sample-fact span { color: #62707c; font-size: 12px; }
.sample-fact strong { font-size: 13px; }
.web-link {
  display: inline-block;
  margin-top: 14px;
  padding: 9px 14px;
  border: 1px solid #276fbf;
  color: #276fbf;
  font-size: 13px;
  font-weight: 700;
  text-decoration: none;
}
.metric-good { color: #23805a; font-weight: 700; }
.metric-bad { color: #bd3d3a; font-weight: 700; }
.metric-note { color: #a96513; font-weight: 700; }
.source-line { margin-top: 13px; color: #62707c; font-size: 11px !important; }
.small { color: #62707c; font-size: 13px; line-height: 1.58; }
@media (max-width: 850px) {
  .report-grid-2, .result-strip, .sample-layout { grid-template-columns: 1fr; }
  .result-strip > div { border-right: 0; border-bottom: 1px solid #d9e0e5; }
  .result-strip > div:last-child { border-bottom: 0; }
  .literature-row, .feature-row { grid-template-columns: 1fr; gap: 5px; }
  .feature-row .dimension { text-align: left; }
  .pipeline { flex-direction: column; }
  .pipeline .arrow { justify-content: center; }
  .decision-box { grid-template-columns: 1fr; }
  .decision-box .mid { min-height: 48px; }
}
</style>

<div class="report-title">
  <h1>EgoProactive Small：从主动视频文献到 PWR-inspired 决策系统</h1>
  <p>近期文献调研、路线选择、工程实现与阶段实验</p>
  <p>阶段性工作汇报 · 2026-07-15</p>
</div>

<div class="result-strip">
  <div><strong>0.6341</strong><span>D1 五折 OOF Macro F1</span></div>
  <div><strong>+0.1711</strong><span>相对 R0 的 Macro F1 增量</span></div>
  <div><strong>9.15%</strong><span>shared-vision 端到端加速</span></div>
</div>

**汇报主线：** 文献检索明确问题建模，PWR 提供程序状态框架；当前工作先建立可复现的小模型基线和稳定决策接口，再逐步验证状态、表示与训练策略。

<!-- SLIDE -->

## 1. 进阶文献检索：四条互补路线

<div class="literature-list">
  <div class="literature-row">
    <div><strong>MMDuet2</strong><div class="date">2025-12-07 · v1</div></div>
    <p>多轮主动回答时机</p>
    <p>文本式 <code>NO REPLY</code>，SFT 后用 PAUC-style GRPO 优化正确、尽早和少重复的回答。</p>
  </div>
  <div class="literature-row">
    <div><strong>StreamPro</strong><div class="date">2026-05-11 · v1</div></div>
    <p>部分观测下的主动决策</p>
    <p>CB-Stream Loss 处理 silence/response 不平衡，GRPO 同时使用 turn-level 与 trajectory-level reward。</p>
  </div>
  <div class="literature-row">
    <div><strong>R3-Streaming</strong><div class="date">2026-06-01 · v2</div></div>
    <p>记忆、准备度与算力路由</p>
    <p>Remember / Respond / Reason 三级控制；强调近期证据保真和按难度分配计算。</p>
  </div>
  <div class="literature-row">
    <div><strong>Plan, Watch, Recover</strong><div class="date">2026-06-03 · v1</div></div>
    <p>操作步骤、偏离与恢复</p>
    <p>显式 procedural state 与 planner–interaction duplex，与 EgoProactive 的任务语义最直接匹配。</p>
  </div>
</div>

<div class="report-band amber"><h3>比较口径</h3><p>各论文的 benchmark、时序容差、文本评分和模型规模不同，论文分数不能与本项目 Macro F1 直接横向排序。这里比较的是问题建模和可迁移机制。</p></div>

<p class="source-line">官方论文：<a href="https://arxiv.org/abs/2512.06810">MMDuet2</a> · <a href="https://arxiv.org/abs/2605.16381">StreamPro</a> · <a href="https://arxiv.org/abs/2605.17921">R3-Streaming</a> · <a href="https://arxiv.org/abs/2606.04970">PWR</a></p>

<!-- SLIDE -->

## 2. MMDuet2、StreamPro 与 R3：可借鉴机制

<div class="report-grid-2">
  <div class="method-panel red">
    <h3>MMDuet2：优化“何时回答”</h3>
    <p><strong>数据与模型：</strong>52K videos，Qwen2.5-VL-3B。</p>
    <p><strong>训练：</strong>SFT 建立主动对话格式；GRPO 奖励回答正确且尽早，并惩罚重复、越界和冗长前缀。</p>
    <p><strong>本项目借鉴：</strong>及时性奖励和多轮轨迹约束。</p>
    <p><strong>不直接采用：</strong>3B 超过 Small 2B，任务更接近主动 VideoQA，缺少显式操作步骤状态。</p>
  </div>
  <div class="method-panel green">
    <h3>StreamPro：处理类不平衡与轨迹质量</h3>
    <p><strong>模型：</strong>3B / 4B streaming VLM。</p>
    <p><strong>训练：</strong>CB-Stream Loss 提高稀疏 response token 权重；GRPO 同时评价单轮时机和整条响应轨迹。</p>
    <p><strong>本项目借鉴：</strong>类平衡监督和 trajectory-level objective。</p>
    <p><strong>不直接采用：</strong>训练预算高，benchmark 与 C1 的 chunk 二分类口径不同。</p>
  </div>
</div>

<div class="report-band amber"><h3>R3-Streaming 的直接启示</h3><p>旧视觉记忆、是否准备好回答、是否调用慢模型是三个不同问题。当前不采用 fast/slow 双模型，但保留“近期证据高保真”和“readiness head 与自由回答解耦”两项设计。</p></div>

**阶段判断：** 当前首先缺少稳定的 interrupt/silent 决策接口和可检验的过程状态，而不是直接启动大规模 GRPO。

<!-- SLIDE -->

## StreamPro：SFT 与 GRPO 训练流程

<img src="pictures/StreamPro流程.png" alt="StreamPro 的 SFT 与 GRPO 训练流程" style="display:block;width:100%;max-height:620px;object-fit:contain;margin:0 auto;">

<!-- SLIDE -->

## 3. 赛道问题：模型不仅要回答，还要决定何时打断

<div class="decision-box">
  <div class="side"><h3>当前可用输入</h3><span class="small">第一人称视频截至当前 chunk、用户 query、当前时刻之前的官方 dialog、因果内部状态。</span></div>
  <div class="mid">每个<br>chunk</div>
  <div class="side"><h3>唯一决策</h3><code>$silent$</code><br>或<br><code>$interrupt$&lt;utterance&gt;</code></div>
</div>

<div class="report-grid-2">
  <div class="method-panel"><h3>数据与指标</h3><p>700 sessions，9,935 candidate chunks，四个领域。</p><p>Macro F1 同时平均 Interrupt F1 和 Silent F1，因此过度打断与过度沉默都会受罚。</p></div>
  <div class="method-panel amber"><h3>部署边界</h3><p>Small division 总参数不超过 2B。</p><p>推理只能读取当前及过去的信息；不能使用未来帧、未来 dialog、最终 session 长度或由未来信息派生的特征。</p></div>
</div>

`dialog[i]` 是官方提供的、决策 `i` 之前已经发生的对话历史，不是模型自我反馈生成的闭环历史。`video_intervals[i]` 使用绝对时间 `[start, end]`。

<!-- SLIDE -->

## 4. PWR：把主动操作助手建模为“计划—观察—恢复”

<div class="pipeline">
  <div class="node"><strong>Planner</strong><br>初始化或更新 plan state</div><div class="arrow">→</div>
  <div class="node"><strong>Watch</strong><br>当前视频、历史与旧状态</div><div class="arrow">→</div>
  <div class="node"><strong>Interaction</strong><br>silent / interrupt + utterance</div><div class="arrow">→</div>
  <div class="node"><strong>Recover</strong><br>偏离后重写步骤与 cues</div>
</div>

<div class="report-grid-2">
  <div class="method-panel">
    <h3>显式 procedural state</h3>
    <p><code>goal</code> · <code>completed_steps</code> · <code>current_step</code> · <code>next_steps</code></p>
    <p><code>step_complete_cues</code> · <code>step_incomplete_cues</code> · recovery plan</p>
  </div>
  <div class="method-panel green">
    <h3>状态解决的核心歧义</h3>
    <p>同一视觉动作在不同步骤可能需要不同提示；模型还要区分“正在做”“已经完成”“做错并需要恢复”。</p>
    <p>interrupt 因而不是单帧事件检测，而是过程状态变化后的交互动作。</p>
  </div>
</div>

<div class="report-band red"><h3>复现边界</h3><p>截至正式审计，未找到官方 PWR training code、checkpoints、Pro²Bench train annotations、gold plan/cue targets 或完整对齐工程。当前实现是 PWR-inspired Small 路线，不是 official reproduction。</p></div>

<!-- SLIDE -->

## PWR：Planner–Interaction 双模型闭环

<img src="pictures/PWR流程图.png" alt="PWR 的 Planner 与 Duplex Interaction 闭环流程" style="display:block;width:100%;max-height:620px;object-fit:contain;margin:0 auto;">

<!-- SLIDE -->

## PWR：Plan、Watch、Recover 示例

<img src="pictures/PWR样例展示.png" alt="PWR 在制作意式浓缩咖啡任务中的 Plan、Watch 与 Recover 示例" style="display:block;width:100%;max-height:620px;object-fit:contain;margin:0 auto;">

<!-- SLIDE -->

## 5. 为什么选择 PWR-inspired Small 路线

| 选择标准 | PWR 提供的思路 | 本项目的 Small 适配 |
|---|---|---|
| 任务语义 | 第一人称操作指导，决定何时介入、如何恢复 | 与 C1 query、步骤和 interrupt 标签直接对应 |
| 关键变量 | current step、完成/未完成 cue、Out-of-Plan recovery | 构造紧凑 causal state，先测 oracle，再测 predicted/noisy state |
| 系统分工 | planner 与 interaction 解耦 | 状态更新器与二元 decision head 解耦，逐项消融 |
| 可证伪性 | oracle plan 可以测状态上限 | 状态无稳定增量时，暂停 planner 与粒度模型 |
| 参数约束 | 原论文依赖大型模型 | 保留 1.06B backbone，只增加轻量头或小型 adapter |

<div class="report-grid-2">
  <div class="report-band green"><h3>保留</h3><p>过程状态、视觉完成证据、偏离恢复、内部状态与外部发言分离。</p></div>
  <div class="report-band red"><h3>不照搬</h3><p>两套大型在线模型、未公开监督、不可核查训练细节，以及当前尚无必要的 RL。</p></div>
</div>

**实验原则：** 先回答冻结小模型中是否已有可恢复的决策信号，再回答显式状态能否带来可重复增量，最后决定 SFT、LoRA 或 GRPO。

<!-- SLIDE -->

## 6. 当前路线：每个阶段只回答一个问题

<div class="route-list">
  <div class="route-step"><strong>R0</strong><span>冻结无 plan 的因果零样本基线，建立完整复现证据链。</span><span class="status status-done">完成</span></div>
  <div class="route-step"><strong>R0-F</strong><span>隔离自由生成的 tag 格式问题，不重跑模型。</span><span class="status status-done">完成</span></div>
  <div class="route-step"><strong>R1</strong><span>4-session oracle compact-state 协议试验：null / step / cues / full。</span><span class="status status-stop">未过门槛</span></div>
  <div class="route-step"><strong>D1</strong><span>冻结 backbone，训练 session-held-out 标量、tag、hidden 融合决策头。</span><span class="status status-done">已推广</span></div>
  <div class="route-step"><strong>D2</strong><span>测试 width-8 residual MLP，并完成 final-language-MLP LoRA 可行性审计。</span><span class="status status-stop">MLP 否决</span></div>
  <div class="route-step"><strong>复验</strong><span>在稳定 D1 接口上扩大、预注册 oracle state 试验。</span><span class="status status-next">待执行</span></div>
  <div class="route-step"><strong>R2–R4</strong><span>粒度敏感性 → predicted state → noisy-plan robustness。</span><span class="status status-next">条件触发</span></div>
</div>

<div class="report-band amber"><h3>顺序约束</h3><p>状态价值未证实时不建设 planner；粒度没有重复增量时不建设 granularity model；没有稳定监督基线和可测残差时不启动 GRPO。</p></div>

<!-- SLIDE -->

## 7. 工程实现：固定 1.06B backbone，分离生成与决策

<div class="report-grid-2">
  <div class="method-panel">
    <h3>模型与因果输入</h3>
    <p><code>OpenGVLab/InternVL3_5-1B-HF</code>，固定 revision，1,060,897,792 base parameters，Apache-2.0。</p>
    <p>当前 interval 提取 16 帧；累计视觉记忆均匀保留最多 32 帧；最多 4 个当前已可见 dialog turns。</p>
    <p>BF16、SDPA、greedy decoding，输入帧调整为 448×448。</p>
  </div>
  <div class="method-panel green">
    <h3>模块职责</h3>
    <p><code>proactive_r0</code>：帧提取、prompt、自由回答、分片与断点。</p>
    <p><code>proactive_r1</code>：oracle state schema 与受控 prompt 变体。</p>
    <p><code>proactive_d1</code>：特征缓存、五折头、在线 runner 与等价验证。</p>
    <p><code>proactive_d2</code>：非线性 residual 与 LoRA 表示适配审计。</p>
  </div>
</div>

<div class="pipeline">
  <div class="node"><strong>R0 generation</strong><br>生成 raw response / utterance</div><div class="arrow">+</div>
  <div class="node"><strong>Fixed candidates</strong><br><code>$silent$</code> / <code>$interrupt$</code></div><div class="arrow">→</div>
  <div class="node"><strong>D1 features</strong><br>scalar + margin + hidden</div><div class="arrow">→</div>
  <div class="node"><strong>Decision head</strong><br>阈值决定是否发言</div>
</div>

金标 `answers` 在神经特征提取前物理删除；标签只在 OOF 头训练与评估阶段重新挂接。

<!-- SLIDE -->

## 8. D1 输入：三类互补的因果特征

<div class="feature-row">
  <strong>18 causal scalars</strong>
  <p>首 chunk、chunk index、当前结束时间、interval 时长、时间 gap、可见 dialog 比例、输入帧比例、4 个 domain one-hot，以及 R0/R0-F 决策与 raw response 格式属性。</p>
  <div class="dimension">18 维</div>
</div>
<div class="feature-row">
  <strong>Fixed-tag margin</strong>
  <p><code>log P($interrupt$ | prompt) − log P($silent$ | prompt)</code>。它保留连续排序信息，不依赖自由生成是否恰好输出合法 tag。</p>
  <div class="dimension">1 维</div>
</div>
<div class="feature-row">
  <strong>Causal hidden</strong>
  <p>取候选标签开始前最后一个 prompt token 的最终层表示。两个候选路径逐 chunk 验证 prefix hidden 完全相同，避免把待比较标签泄漏进特征。</p>
  <div class="dimension">1,024 维</div>
</div>

<div class="formula">18 + 1 + 1,024 = 1,043 维输入　→　线性权重 1,043 + bias 1 = 1,044 参数</div>

线性头只负责 interrupt/silent gate；预测为 interrupt 时，回答内容优先复用 R0-F utterance，否则使用固定 fallback。

<!-- SLIDE -->

## 9. 实验设计：五折 OOF、消融与推广门槛

<div class="pipeline">
  <div class="node"><strong>3 folds</strong><br>拟合标准化与线性权重</div><div class="arrow">→</div>
  <div class="node"><strong>1 fold</strong><br>只选择 threshold</div><div class="arrow">→</div>
  <div class="node"><strong>1 untouched fold</strong><br>冻结预测后评分</div><div class="arrow">×5</div>
  <div class="node"><strong>Merged OOF</strong><br>每个 session 测试一次</div>
</div>

| 受控变体 | 输入 | 目的 |
|---|---:|---|
| tag only | 1 | 固定标签相对概率能否独立决策 |
| hidden linear | 1,024 | 冻结多模态表示是否线性可分 |
| scalar + tag | 19 | tag margin 对强标量控制的增量 |
| fused linear | 1,043 | scalar、margin 与 hidden 是否互补 |

**推广门槛：** 相对 D1 scalar 至少 `+0.005` Macro F1；paired-session bootstrap 下界为正；两类 F1 不坍缩；中段 chunk、fold 与 domain 的改善具有稳定性。

同一 session 的相邻 chunks 永不跨 train/test。全部五折仍来自公开 validation，因此结果标记为 `val-supervised`。

<!-- SLIDE -->

## 10. 实验结果：校准是主要增量，神经表示提供互补信号

| 实验 | Macro F1 | Interrupt F1 | Silent F1 | 结论 |
|---|---:|---:|---:|---|
| R0 | 0.4630 | 0.3728 | 0.5531 | 明显偏 silent；633 个 malformed |
| R0-F | 0.5362 | 0.4879 | 0.5845 | 格式修复有效，但属 val-supervised 规则 |
| D1 scalar | 0.6119 | 0.6366 | 0.5873 | 决策与标注策略校准是最大增量来源 |
| D1 fused | **0.6341** | 0.6352 | 0.6330 | 两类更平衡；相对 scalar +0.0222 |
| D1 单阈值模拟 | 0.6330 | 0.6298 | 0.6361 | 仅下降 0.00113，部署稳健性通过 |
| D2 width-8 MLP | 0.6351 | 0.6375 | 0.6327 | +0.0010，区间跨零，按门槛否决 |

<div class="report-grid-2">
  <div class="report-band green"><h3>稳定性</h3><p>D1 fused 相对 scalar 的 paired-session bootstrap 95% interval 为 [+0.0123, +0.0322]；5/5 folds、4/4 domains 改善。</p></div>
  <div class="report-band amber"><h3>消融含义</h3><p>tag only 0.5313，hidden only 0.6031，scalar + tag 0.6172。0.6341 不能表述成 hidden 单独带来的结果。</p></div>
</div>

**证据边界：** 0.6341 是 public-validation-supervised 的 OOF 开发估计；全量重拟合 0.6719 只是 train-fit sanity，不用于泛化汇报。

<!-- SLIDE -->

## 11. 具体样本：电池极性错误与恢复过程

<div class="sample-layout">
  <div>
    <img class="sample-image" src="assets/session143_recovery_sequence.jpg" alt="恒温器电池操作在 84、88 和 96 秒的连续画面">
    <p class="source-line">Session 143 · 84 s / 88 s / 96 s：取出设备、检查电池方向、重新装回墙面。</p>
  </div>
  <div>
    <div class="sample-facts">
      <div class="sample-fact"><span>Task</span><strong>Replacing batteries in thermostat</strong></div>
      <div class="sample-fact"><span>时长</span><strong>140.9 s · 22 chunks</strong></div>
      <div class="sample-fact"><span>R0-F → D1</span><strong>修复 10 个 interrupt 漏报</strong></div>
      <div class="sample-fact"><span>金标恢复行为</span><strong>识别电池装反，翻转极性并重新插入</strong></div>
      <div class="sample-fact"><span>当前局限</span><strong>时机改善，但部分 utterance 仍为通用下一步提示</strong></div>
    </div>
    <a class="web-link" href="http://127.0.0.1:8766/#samples" target="_blank">打开交互式视频时间轴</a>
  </div>
</div>

网页同时展示视频、每个 chunk 的绝对时间、gold/prediction、TP/FP/TN/FN，以及 interrupt utterance。点击 chunk 后视频跳转到对应 interval 起点。

<!-- SLIDE -->

## 12. 等价加速、当前系统与下一步

| 推理方案 | 结果 | 等价性 | 决定 |
|---|---|---|---|
| batch-of-two | wall time +17.37%，显存 +19.59% | 特征与决策一致 | 否决：仍计算两份长序列与视觉输入 |
| cropped prefix cache | wall time +6.02%，margin 最大漂移 0.113382 | 不等价 | 否决：BF16/SDPA 计算形状产生漂移 |
| shared vision | 127 chunks：500.892 s → 455.056 s，显存不变 | hidden、margin、decision、answer 127/127 一致 | **推广：端到端加速 9.15%** |

<div class="report-grid-2">
  <div class="method-panel green"><h3>当前系统</h3><p>InternVL3.5-1B + 1,044 参数 D1 fused linear head。</p><p>部署使用 shared vision；sequential 永久保留为正确性参照。</p></div>
  <div class="method-panel amber"><h3>下一阶段</h3><p>先执行唯一冻结的 final-language-MLP LoRA 五折 OOF；若无增益，转向更大、预注册的 oracle-state 复验。</p><p>其后依次验证粒度、predicted state 和 noisy-plan robustness。</p></div>
</div>

**阶段结论：** 已完成可复现 Small 基线、严格因果决策头、完整 OOF 评估和等价部署加速；PWR 的过程状态假设仍需在稳定 D1 接口上做更大规模验证。GRPO 只在状态收益、剩余误差和监督基线都稳定后进入。
