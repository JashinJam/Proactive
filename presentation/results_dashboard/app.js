(function () {
  "use strict";

  const DATA = window.PWR_DASHBOARD_DATA;
  if (!DATA) {
    document.body.innerHTML = "<p>缺少 data.js。请先运行 presentation/build_assets.py。</p>";
    return;
  }

  const experimentsById = Object.fromEntries(DATA.experiments.map((item) => [item.id, item]));
  const progressionIds = ["r0", "r0f", "d1_scalar", "d1_fused", "d1_single_threshold", "d2_mlp"];
  const sampleExperimentIds = ["r0", "r0f", "d1_scalar", "d1_fused", "d1_single_threshold", "d2_mlp"];
  const ablationIds = ["d1_scalar", "d1_tag", "d1_hidden", "d1_scalar_tag", "d1_fused"];

  const state = {
    view: "overview",
    diagnosticExperiment: "d1_fused",
    sampleExperiment: "d1_fused",
    sampleDomain: "all",
    sampleOutcome: "all",
    sampleSearch: "",
    sessionIndex: DATA.featured_sessions[0]?.session_index || 0,
    selectedChunk: 0,
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function pct(value, digits = 1) {
    return `${(Number(value) * 100).toFixed(digits)}%`;
  }

  function f4(value) {
    return Number(value).toFixed(4);
  }

  function signed(value, digits = 4) {
    const number = Number(value);
    return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}`;
  }

  function formatInteger(value) {
    return Number(value).toLocaleString("zh-CN");
  }

  function formatTime(seconds) {
    const safe = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(safe / 60);
    const remainder = Math.floor(safe % 60);
    return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
  }

  function outcomeFor(gold, predicted) {
    if (gold === "interrupt" && predicted === "interrupt") return "tp";
    if (gold === "silent" && predicted === "interrupt") return "fp";
    if (gold === "silent" && predicted === "silent") return "tn";
    return "fn";
  }

  function outcomeLabel(outcome) {
    return {
      tp: "TP：正确打断",
      fp: "FP：多余打断",
      tn: "TN：正确沉默",
      fn: "FN：漏掉打断",
    }[outcome];
  }

  function activateView(view, updateHash = true) {
    const validViews = ["overview", "diagnostics", "r1-efficiency", "samples"];
    state.view = validViews.includes(view) ? view : "overview";
    document.querySelectorAll("[data-view-section]").forEach((section) => {
      const active = section.dataset.viewSection === state.view;
      section.hidden = !active;
      section.classList.toggle("is-active", active);
    });
    document.querySelectorAll("[data-view]").forEach((button) => {
      const active = button.dataset.view === state.view;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", String(active));
    });
    if (updateHash) history.replaceState(null, "", `#${state.view}`);
    if (state.view === "samples") {
      requestAnimationFrame(() => renderSample());
    }
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  function renderMetricChart(container, rows, options = {}) {
    const showClassF1 = options.showClassF1 !== false;
    const baseline = options.baseline || null;
    container.innerHTML = rows
      .map((experiment, index) => {
        const metrics = experiment.metrics;
        const delta = baseline
          ? metrics.macro_f1 - baseline.metrics.macro_f1
          : index > 0
            ? metrics.macro_f1 - rows[index - 1].metrics.macro_f1
            : 0;
        const note = index === 0 && !baseline ? "基线" : `${signed(delta)} Macro`;
        return `
          <div class="metric-row">
            <div class="metric-label">
              <strong>${escapeHtml(experiment.short_label || experiment.label)}</strong>
              <span>${escapeHtml(experiment.status || "")}</span>
            </div>
            <div class="metric-track" aria-label="${escapeHtml(experiment.label)} Macro F1 ${f4(metrics.macro_f1)}">
              <span class="metric-bar macro" style="width:${metrics.macro_f1 * 100}%"></span>
              ${showClassF1 ? `<span class="metric-bar interrupt" style="width:${metrics.interrupt_f1 * 100}%"></span>` : ""}
              ${showClassF1 ? `<span class="metric-bar silent" style="width:${metrics.silent_f1 * 100}%"></span>` : ""}
            </div>
            <div class="metric-value">${f4(metrics.macro_f1)}<small>${escapeHtml(note)}</small></div>
          </div>`;
      })
      .join("");
  }

  function renderOverview() {
    const meta = DATA.metadata;
    const d1 = experimentsById.d1_fused.metrics;
    const r0 = experimentsById.r0.metrics;
    document.getElementById("dataset-stats").innerHTML = `
      <div class="stat-item"><span>Sessions</span><strong>${formatInteger(meta.sessions)}</strong><small>4 个领域</small></div>
      <div class="stat-item"><span>Candidate chunks</span><strong>${formatInteger(meta.chunks)}</strong><small>每个 chunk 二选一</small></div>
      <div class="stat-item"><span>Gold interrupt rate</span><strong>${pct(meta.gold_interrupt_rate, 2)}</strong><small>${formatInteger(meta.gold_interrupts)} interrupt</small></div>
      <div class="stat-item"><span>当前 OOF Macro F1</span><strong>${f4(d1.macro_f1)}</strong><small>较 R0 ${signed(d1.macro_f1 - r0.macro_f1)}</small></div>`;

    const progression = progressionIds.map((id) => experimentsById[id]);
    renderMetricChart(document.getElementById("progress-chart"), progression);
    document.getElementById("key-findings").innerHTML = [
      ["R0 → R0-F", "633 个非空 malformed response 被重新解释；Macro F1 0.4630 → 0.5362，主要暴露输出协议问题。"],
      ["标量校准", "18 个严格因果标量将 Macro F1 提升到 0.6119；主要增益来自决策和标注策略校准。"],
      ["神经融合", "tag margin 与 hidden 单独不足，和标量互补后达到 0.6341；相对标量头 +0.0222。"],
      ["非线性头", "D2 达到 0.6351，但 +0.0010 的区间跨零且仅 3/5 folds 改善，因此不推广。"],
    ]
      .map(([label, text]) => `<div class="finding-item"><strong>${label}</strong><span>${text}</span></div>`)
      .join("");
  }

  function populateExperimentSelect(select, ids, selected) {
    select.innerHTML = ids
      .map((id) => {
        const experiment = experimentsById[id];
        return `<option value="${id}" ${id === selected ? "selected" : ""}>${escapeHtml(experiment.label)}</option>`;
      })
      .join("");
  }

  function renderConfusion() {
    const experiment = experimentsById[state.diagnosticExperiment];
    const metrics = experiment.metrics;
    const actualInterrupt = metrics.tp + metrics.fn;
    const actualSilent = metrics.tn + metrics.fp;
    document.getElementById("confusion-title").textContent = `${experiment.short_label} 混淆矩阵`;
    document.getElementById("confusion-matrix").innerHTML = `
      <div></div><div class="matrix-axis">预测 interrupt</div><div class="matrix-axis">预测 silent</div>
      <div class="matrix-axis">金标<br>interrupt</div>
      <div class="matrix-cell tp"><strong>${formatInteger(metrics.tp)}</strong><span>TP · ${pct(metrics.tp / actualInterrupt)}</span></div>
      <div class="matrix-cell fn"><strong>${formatInteger(metrics.fn)}</strong><span>FN · ${pct(metrics.fn / actualInterrupt)}</span></div>
      <div class="matrix-axis">金标<br>silent</div>
      <div class="matrix-cell fp"><strong>${formatInteger(metrics.fp)}</strong><span>FP · ${pct(metrics.fp / actualSilent)}</span></div>
      <div class="matrix-cell tn"><strong>${formatInteger(metrics.tn)}</strong><span>TN · ${pct(metrics.tn / actualSilent)}</span></div>`;

    const diagnostics = [
      ["Macro F1", metrics.macro_f1],
      ["Interrupt precision", metrics.interrupt_precision],
      ["Interrupt recall", metrics.interrupt_recall],
      ["Silent recall", metrics.silent_recall],
      ["Pred. interrupt rate", metrics.predicted_interrupt_rate],
      ["Support", metrics.support, true],
    ];
    document.getElementById("diagnostic-metrics").innerHTML = diagnostics
      .map(
        ([label, value, integer]) => `
        <div class="diagnostic-card"><span>${label}</span><strong>${integer ? formatInteger(value) : f4(value)}</strong></div>`,
      )
      .join("");

    const domains = DATA.domain_metrics[state.diagnosticExperiment];
    document.getElementById("domain-chart").innerHTML = Object.entries(domains)
      .map(
        ([domain, item]) => `
        <div class="domain-row">
          <span>${escapeHtml(domain)}</span>
          <div class="domain-track"><div class="domain-bar" style="width:${item.macro_f1 * 100}%"></div></div>
          <span class="domain-value">${f4(item.macro_f1)}</span>
          <span class="domain-recall">Int R ${f4(item.interrupt_recall)} · Silent R ${f4(item.silent_recall)}</span>
        </div>`,
      )
      .join("");
  }

  function renderAblationTable() {
    const baseline = experimentsById.d1_scalar.metrics.macro_f1;
    const rows = ablationIds.map((id) => experimentsById[id]);
    document.getElementById("ablation-table").innerHTML = `
      <table>
        <thead><tr><th>变体</th><th>输入</th><th>Macro F1</th><th>Interrupt F1</th><th>Silent F1</th><th>较 scalar</th><th>结论</th></tr></thead>
        <tbody>
          ${rows
            .map((item) => {
              const conclusion = {
                d1_scalar: "强控制",
                d1_tag: "单独不足",
                d1_hidden: "单独不足",
                d1_scalar_tag: "小幅增量",
                d1_fused: "互补增量，推广",
              }[item.id];
              const input = {
                d1_scalar: "18 scalar",
                d1_tag: "1 tag margin",
                d1_hidden: "1,024 hidden",
                d1_scalar_tag: "18 scalar + 1 tag",
                d1_fused: "18 + 1 + 1,024",
              }[item.id];
              return `<tr>
                <td><strong>${escapeHtml(item.short_label)}</strong></td>
                <td>${input}</td>
                <td class="number">${f4(item.metrics.macro_f1)}</td>
                <td class="number">${f4(item.metrics.interrupt_f1)}</td>
                <td class="number">${f4(item.metrics.silent_f1)}</td>
                <td class="number">${signed(item.metrics.macro_f1 - baseline)}</td>
                <td class="status-text ${item.id === "d1_fused" ? "promoted" : ""}">${conclusion}</td>
              </tr>`;
            })
            .join("")}
        </tbody>
      </table>`;
  }

  function renderR1() {
    const variants = DATA.r1_study.variants.map((item) => ({
      ...item,
      short_label: item.label,
      status: item.id === "full" ? "状态最完整" : "",
    }));
    renderMetricChart(document.getElementById("r1-chart"), variants, { showClassF1: true });
  }

  function renderInference() {
    const study = DATA.inference_study;
    const reference = study.smoke.find((item) => item.id === "sequential");
    const table = `
      <div class="table-wrap">
        <table>
          <thead><tr><th>推理模式</th><th>10-chunk wall time</th><th>相对 sequential</th><th>峰值显存</th><th>特征等价</th><th>判定</th></tr></thead>
          <tbody>
            ${study.smoke
              .map((item) => {
                const delta = item.wall_s / reference.wall_s - 1;
                const statusClass = item.id === "shared_vision_smoke" ? "current" : item.id === "sequential" ? "" : "rejected";
                return `<tr>
                  <td><strong>${escapeHtml(item.label)}</strong></td>
                  <td class="number">${item.wall_s.toFixed(3)} s</td>
                  <td class="number">${item.id === "sequential" ? "reference" : pct(delta)}</td>
                  <td class="number">${(item.peak_bytes / 1024 ** 3).toFixed(2)} GiB</td>
                  <td>${item.equivalent ? "是" : `否${item.max_margin_diff ? `；margin diff ${item.max_margin_diff}` : ""}`}</td>
                  <td class="status-text ${statusClass}">${escapeHtml(item.status)}</td>
                </tr>`;
              })
              .join("")}
          </tbody>
        </table>
      </div>`;
    const expanded = study.expanded;
    const cards = `
      <div class="inference-summary">
        <div class="inference-card"><span>127-chunk wall time</span><strong>${expanded.shared_wall_s.toFixed(3)} s</strong></div>
        <div class="inference-card"><span>端到端改善</span><strong>${pct(expanded.wall_improvement)}</strong></div>
        <div class="inference-card"><span>峰值显存变化</span><strong>0 B</strong></div>
        <div class="inference-card"><span>决策 / 答案一致</span><strong>${expanded.decision_equal}</strong></div>
      </div>
      <div class="callout">shared vision 只复用两个固定标签候选之间完全相同的视觉特征；R0 自回归生成和两次 batch-one 语言前向保持不变，因此得到可审计的等价加速。</div>`;
    document.getElementById("inference-comparison").innerHTML = table + cards;
  }

  function sessionHasFilter(session, filter) {
    if (filter === "all") return true;
    if (filter === "recovered_fn" || filter === "recovered_fp") {
      return session.pattern_counts[filter] > 0;
    }
    const predicted = session.predictions[state.sampleExperiment];
    if (!predicted) return false;
    return session.gold.some((gold, index) => {
      const outcome = outcomeFor(gold.decision, predicted[index].decision);
      return filter === "remaining_fn" ? outcome === "fn" : outcome === "fp";
    });
  }

  function filteredSessions() {
    const needle = state.sampleSearch.trim().toLowerCase();
    return DATA.sessions.filter((session) => {
      if (state.sampleDomain !== "all" && session.domain !== state.sampleDomain) return false;
      if (!sessionHasFilter(session, state.sampleOutcome)) return false;
      if (needle && !`${session.task} ${session.query} ${session.video_path}`.toLowerCase().includes(needle)) return false;
      return true;
    });
  }

  function renderSessionSelect() {
    const select = document.getElementById("sample-session");
    const sessions = filteredSessions();
    if (!sessions.some((session) => session.index === state.sessionIndex) && sessions.length) {
      state.sessionIndex = sessions[0].index;
      state.selectedChunk = 0;
    }
    if (!sessions.length) state.sessionIndex = null;
    select.innerHTML = sessions
      .map(
        (session) =>
          `<option value="${session.index}" ${session.index === state.sessionIndex ? "selected" : ""}>${String(session.index).padStart(3, "0")} · ${escapeHtml(session.domain)} · ${escapeHtml(session.task)}</option>`,
      )
      .join("");
    if (!sessions.length) {
      select.innerHTML = '<option value="">没有匹配的 session</option>';
    }
  }

  function timelineBlock(session, item, index, rowType) {
    const interval = session.intervals[index];
    const left = (interval[0] / session.duration) * 100;
    const width = ((interval[1] - interval[0]) / session.duration) * 100;
    const decisionClass = rowType === "outcome" ? `outcome-${item}` : item.decision;
    const label = rowType === "outcome" ? outcomeLabel(item) : item.decision;
    return `<button type="button" class="timeline-block ${decisionClass} ${index === state.selectedChunk ? "is-selected" : ""}" data-chunk-index="${index}" style="left:${left}%;width:${Math.max(width, 0.25)}%" title="Chunk ${index} · ${interval[0].toFixed(1)}–${interval[1].toFixed(1)}s · ${escapeHtml(label)}" aria-label="跳转到 chunk ${index}"></button>`;
  }

  function renderTimeline(session) {
    const prediction = session.predictions[state.sampleExperiment];
    const outcomes = session.gold.map((gold, index) => outcomeFor(gold.decision, prediction[index].decision));
    const rows = [
      ["金标", session.gold, "decision"],
      [experimentsById[state.sampleExperiment].short_label, prediction, "decision"],
      ["对错", outcomes, "outcome"],
    ];
    document.getElementById("timeline").innerHTML = rows
      .map(
        ([label, items, rowType]) => `
        <div class="timeline-row">
          <div class="timeline-row-label" title="${escapeHtml(label)}">${escapeHtml(label)}</div>
          <div class="timeline-track" data-timeline-row="${rowType}">
            ${items.map((item, index) => timelineBlock(session, item, index, rowType)).join("")}
          </div>
        </div>`,
      )
      .join("");
    document.querySelectorAll(".timeline-block").forEach((button) => {
      button.addEventListener("click", () => seekToChunk(Number(button.dataset.chunkIndex), true));
    });
  }

  function decisionMarkup(item) {
    const utterance = item.utterance ? `<p>${escapeHtml(item.utterance)}</p>` : "<p>无 utterance</p>";
    return `<span class="decision-tag ${item.decision}">${item.decision === "interrupt" ? "$interrupt$" : "$silent$"}</span>${utterance}`;
  }

  function renderChunkDetail(session) {
    const index = Math.min(state.selectedChunk, session.intervals.length - 1);
    const interval = session.intervals[index];
    const gold = session.gold[index];
    const prediction = session.predictions[state.sampleExperiment][index];
    const outcome = outcomeFor(gold.decision, prediction.decision);
    document.getElementById("chunk-detail").innerHTML = `
      <div class="detail-index"><h3>当前 chunk</h3><strong>#${index}</strong><span>${interval[0].toFixed(1)}–${interval[1].toFixed(1)} s</span><span class="outcome-badge ${outcome}">${outcomeLabel(outcome)}</span></div>
      <div><h3>金标</h3>${decisionMarkup(gold)}</div>
      <div><h3>${escapeHtml(experimentsById[state.sampleExperiment].label)}</h3>${decisionMarkup(prediction)}</div>`;
  }

  function renderChunkTable(session) {
    const prediction = session.predictions[state.sampleExperiment];
    document.getElementById("chunk-table").innerHTML = `
      <table>
        <thead><tr><th>Chunk</th><th>时间</th><th>金标</th><th>预测</th><th>结果</th></tr></thead>
        <tbody>
          ${session.intervals
            .map((interval, index) => {
              const gold = session.gold[index];
              const pred = prediction[index];
              const outcome = outcomeFor(gold.decision, pred.decision);
              return `<tr class="chunk-row ${index === state.selectedChunk ? "is-selected" : ""}" data-chunk-index="${index}" tabindex="0">
                <td><strong>#${index}</strong></td>
                <td>${interval[0].toFixed(1)}–${interval[1].toFixed(1)} s</td>
                <td class="chunk-cell-answer"><strong>${gold.decision}</strong><span>${escapeHtml(gold.utterance || "")}</span></td>
                <td class="chunk-cell-answer"><strong>${pred.decision}</strong><span>${escapeHtml(pred.utterance || "")}</span></td>
                <td><span class="outcome-badge ${outcome}">${outcome.toUpperCase()}</span></td>
              </tr>`;
            })
            .join("")}
        </tbody>
      </table>`;
    document.querySelectorAll(".chunk-row").forEach((row) => {
      const activate = () => seekToChunk(Number(row.dataset.chunkIndex), true);
      row.addEventListener("click", activate);
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate();
        }
      });
    });
  }

  function seekToChunk(index, autoplay) {
    const session = DATA.sessions[state.sessionIndex];
    if (!session || !session.intervals[index]) return;
    state.selectedChunk = index;
    const video = document.getElementById("sample-video");
    video.currentTime = Math.max(0, session.intervals[index][0] + 0.02);
    if (autoplay) video.play().catch(() => {});
    renderTimeline(session);
    renderChunkDetail(session);
    renderChunkTable(session);
  }

  function updatePlaybackHighlight() {
    const session = DATA.sessions[state.sessionIndex];
    const video = document.getElementById("sample-video");
    if (!session) return;
    const current = video.currentTime || 0;
    const activeIndex = session.intervals.findIndex(([start, end]) => current >= start && current <= end);
    document.querySelectorAll(".timeline-block").forEach((block) => {
      block.classList.toggle("is-playing", Number(block.dataset.chunkIndex) === activeIndex);
    });
    document.getElementById("timeline-position").textContent = `${formatTime(current)} / ${formatTime(session.duration)}`;
  }

  function renderSample() {
    renderSessionSelect();
    const session = DATA.sessions[state.sessionIndex];
    if (!session) {
      document.getElementById("sample-task").textContent = "没有匹配的 session";
      document.getElementById("sample-query").textContent = "调整领域、错误类型或检索条件后重试。";
      document.getElementById("sample-meta").innerHTML = "";
      document.getElementById("timeline").innerHTML = "";
      document.getElementById("chunk-detail").innerHTML = "";
      document.getElementById("chunk-table").innerHTML = "";
      const emptyVideo = document.getElementById("sample-video");
      emptyVideo.removeAttribute("src");
      emptyVideo.load();
      document.getElementById("open-video").href = "#";
      return;
    }
    state.selectedChunk = Math.min(state.selectedChunk, session.intervals.length - 1);
    document.getElementById("sample-task").textContent = session.task;
    document.getElementById("sample-query").textContent = `Query: ${session.query}`;
    document.getElementById("sample-meta").innerHTML = `
      <span class="meta-tag">Session ${session.index}</span>
      <span class="meta-tag">${escapeHtml(session.domain)}</span>
      <span class="meta-tag">${session.intervals.length} chunks</span>
      <span class="meta-tag">${session.duration.toFixed(1)} s</span>`;

    const video = document.getElementById("sample-video");
    const expectedUrl = new URL(session.video_url, window.location.href).href;
    if (video.src !== expectedUrl) {
      document.getElementById("video-status").textContent = "正在加载视频元数据";
      video.src = session.video_url;
      video.load();
    }
    const openVideo = document.getElementById("open-video");
    openVideo.href = session.video_url;
    renderTimeline(session);
    renderChunkDetail(session);
    renderChunkTable(session);
    updatePlaybackHighlight();
  }

  function initializeSamples() {
    populateExperimentSelect(document.getElementById("sample-experiment"), sampleExperimentIds, state.sampleExperiment);
    document.getElementById("sample-domain").innerHTML = ["all", ...DATA.metadata.domains]
      .map((domain) => `<option value="${escapeHtml(domain)}">${domain === "all" ? "全部领域" : escapeHtml(domain)}</option>`)
      .join("");
    document.getElementById("featured-sessions").innerHTML = DATA.featured_sessions
      .map(
        (item) => `
        <button type="button" class="featured-button" data-session-index="${item.session_index}">
          <strong>${escapeHtml(item.label)} · ${item.count} chunks</strong>
          <span>${escapeHtml(item.domain)} · ${escapeHtml(item.task)}</span>
        </button>`,
      )
      .join("");

    document.getElementById("sample-experiment").addEventListener("change", (event) => {
      state.sampleExperiment = event.target.value;
      state.selectedChunk = 0;
      renderSample();
    });
    document.getElementById("sample-domain").addEventListener("change", (event) => {
      state.sampleDomain = event.target.value;
      state.selectedChunk = 0;
      renderSample();
    });
    document.getElementById("sample-outcome").addEventListener("change", (event) => {
      state.sampleOutcome = event.target.value;
      state.selectedChunk = 0;
      renderSample();
    });
    document.getElementById("sample-search").addEventListener("input", (event) => {
      state.sampleSearch = event.target.value;
      state.selectedChunk = 0;
      renderSample();
    });
    document.getElementById("sample-session").addEventListener("change", (event) => {
      if (event.target.value === "") return;
      state.sessionIndex = Number(event.target.value);
      state.selectedChunk = 0;
      renderSample();
    });
    document.querySelectorAll(".featured-button").forEach((button) => {
      button.addEventListener("click", () => {
        state.sampleDomain = "all";
        state.sampleOutcome = "all";
        state.sampleSearch = "";
        state.sessionIndex = Number(button.dataset.sessionIndex);
        state.selectedChunk = 0;
        document.getElementById("sample-domain").value = "all";
        document.getElementById("sample-outcome").value = "all";
        document.getElementById("sample-search").value = "";
        renderSample();
      });
    });

    const video = document.getElementById("sample-video");
    video.addEventListener("timeupdate", updatePlaybackHighlight);
    video.addEventListener("loadedmetadata", () => {
      document.getElementById("video-status").textContent = "";
      updatePlaybackHighlight();
    });
    video.addEventListener("error", () => {
      document.getElementById("video-status").textContent = "视频加载失败；请确认通过 presentation/serve.py 打开网页。";
    });
  }

  function initialize() {
    renderOverview();
    populateExperimentSelect(document.getElementById("diagnostic-experiment"), progressionIds, state.diagnosticExperiment);
    document.getElementById("diagnostic-experiment").addEventListener("change", (event) => {
      state.diagnosticExperiment = event.target.value;
      renderConfusion();
    });
    renderConfusion();
    renderAblationTable();
    renderR1();
    renderInference();
    initializeSamples();
    renderSample();

    document.querySelectorAll(".tab-button").forEach((button) => {
      button.addEventListener("click", () => activateView(button.dataset.view));
    });
    window.addEventListener("hashchange", () => activateView(window.location.hash.slice(1), false));
    activateView(window.location.hash.slice(1) || "overview", false);
  }

  initialize();
})();
