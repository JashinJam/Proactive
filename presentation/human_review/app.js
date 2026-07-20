"use strict";

const SCORE_FIELDS = [
  ["correctness_1_5", "正确性", "动作、对象、方向和状态是否正确"],
  ["specificity_1_5", "具体性", "是否明确当前动作、对象、部位或方向"],
  ["actionability_1_5", "可执行性", "用户能否只根据这句话立即执行"],
  ["groundedness_1_5", "视觉依据", "是否只使用当前可见和历史证据"],
  ["plan_consistency_1_5", "计划一致性", "是否符合当前步骤、下一步或恢复动作"],
  ["conciseness_1_5", "简洁性", "保留必要信息且没有冗余"],
  ["safety_1_5", "安全性", "5安全；1可能造成直接风险"],
];

const PRIMARY_ERRORS = [
  ["", "请选择主要错误"],
  ["none", "无主要错误"],
  ["wrong_timing", "介入时机错误"],
  ["wrong_action", "动作错误"],
  ["wrong_object", "对象错误"],
  ["premature", "过早宣称完成"],
  ["stale", "重复已完成的旧步骤"],
  ["generic", "通用空话"],
  ["hallucination", "无依据/幻觉"],
  ["unsafe", "不安全"],
  ["other", "其他"],
];

const state = {
  study: null,
  reviewer: null,
  bootstrap: null,
  session: null,
  sessionIndex: -1,
  ratings: new Map(),
  stages: [],
  stageIndex: 0,
  itemIndex: 0,
  completedSession: false,
  cutoff: 0,
  pendingSeek: null,
  videoReady: false,
  playbackGeneration: 0,
  boundaryFrame: null,
  toastTimer: null,
};

const dom = {};

document.addEventListener("DOMContentLoaded", () => {
  for (const id of [
    "appShell", "identityLabel", "rubricButton", "exportButton", "switchButton",
    "progressPercent", "progressFill", "progressText", "domainProgress", "sessionFilter",
    "sessionList", "sessionMeta", "taskTitle", "queryText", "sessionState", "reviewVideo",
    "videoEmpty", "playButton", "backButton", "intervalButton", "timeReadout", "videoSeek",
    "cutoffBadge", "intervalLabel", "causalTimeline", "itemCounter", "itemTabs", "chunkLabel",
    "dialogHistory", "candidateLabel", "candidateText", "ratingStatus", "clearItemButton",
    "ratingForm", "previousItemButton", "nextItemButton", "sessionCompletionText",
    "saveSessionButton", "setupDialog", "setupForm", "setupRubricButton", "rubricDialog",
    "closeRubricButton", "toast",
  ]) dom[id] = document.getElementById(id);

  bindStaticEvents();
  restoreSetupChoice();
  dom.setupDialog.showModal();
});

function bindStaticEvents() {
  dom.setupForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = new FormData(dom.setupForm);
    const study = data.get("study");
    const reviewer = data.get("reviewer");
    if (!study || !reviewer) {
      showToast("请选择评测任务和评审员。", true);
      return;
    }
    localStorage.setItem("humanReviewIdentity", JSON.stringify({study, reviewer}));
    dom.setupDialog.close();
    await startReview(study, reviewer);
  });
  dom.setupDialog.addEventListener("cancel", (event) => {
    if (!state.bootstrap) event.preventDefault();
  });
  dom.setupRubricButton.addEventListener("click", () => openRubric(true));
  dom.rubricButton.addEventListener("click", () => openRubric(false));
  dom.closeRubricButton.addEventListener("click", () => {
    dom.rubricDialog.close();
    if (!state.bootstrap) dom.setupDialog.showModal();
  });
  dom.switchButton.addEventListener("click", () => dom.setupDialog.showModal());
  dom.exportButton.addEventListener("click", () => {
    if (!state.study || !state.reviewer) return;
    window.location.href = `/api/export?study=${state.study}&reviewer=${state.reviewer}`;
  });
  dom.sessionFilter.addEventListener("change", renderSessionList);
  dom.playButton.addEventListener("click", togglePlayback);
  dom.backButton.addEventListener("click", () => seekVideo(Math.max(0, dom.reviewVideo.currentTime - 5)));
  dom.intervalButton.addEventListener("click", () => seekVideo(currentItem().interval[0]));
  dom.videoSeek.addEventListener("input", () => seekVideo(Number(dom.videoSeek.value)));
  dom.reviewVideo.addEventListener("loadedmetadata", () => {
    if (state.pendingSeek !== null) {
      seekVideo(state.pendingSeek);
      state.pendingSeek = null;
    }
    clampVideo();
    updateVideoReadout();
    setVideoReady(true);
  });
  dom.reviewVideo.addEventListener("loadstart", () => setVideoReady(false, "正在载入视频"));
  dom.reviewVideo.addEventListener("loadeddata", () => setVideoReady(true));
  dom.reviewVideo.addEventListener("canplay", () => setVideoReady(true));
  dom.reviewVideo.addEventListener("seeked", () => setVideoReady(true));
  dom.reviewVideo.addEventListener("progress", () => {
    if (dom.reviewVideo.readyState >= 2) setVideoReady(true);
  });
  dom.reviewVideo.addEventListener("waiting", () => {
    if (!dom.reviewVideo.paused) showVideoMessage("正在缓冲视频");
  });
  dom.reviewVideo.addEventListener("stalled", () => {
    if (!dom.reviewVideo.paused) showVideoMessage("正在等待视频数据");
  });
  dom.reviewVideo.addEventListener("playing", () => { setVideoReady(true); dom.playButton.textContent = "暂停"; });
  dom.reviewVideo.addEventListener("pause", () => { dom.playButton.textContent = "播放"; });
  dom.reviewVideo.addEventListener("timeupdate", () => {
    if (!dom.reviewVideo.paused && dom.reviewVideo.readyState >= 2) setVideoReady(true);
    updateVideoReadout();
  });
  dom.reviewVideo.addEventListener("seeking", clampVideo);
  dom.reviewVideo.addEventListener("error", () => {
    const code = dom.reviewVideo.error?.code;
    setVideoReady(false, `视频解码失败${code ? `（错误码 ${code}）` : ""}`);
    showToast("视频加载或解码失败，请刷新后重试。", true);
  });
  dom.previousItemButton.addEventListener("click", previousItem);
  dom.nextItemButton.addEventListener("click", nextItem);
  dom.clearItemButton.addEventListener("click", clearCurrentItem);
  dom.saveSessionButton.addEventListener("click", saveSession);
}

function restoreSetupChoice() {
  try {
    const saved = JSON.parse(localStorage.getItem("humanReviewIdentity") || "null");
    if (saved?.study) {
      const study = dom.setupForm.querySelector(`input[name="study"][value="${saved.study}"]`);
      if (study) study.checked = true;
    }
    if (saved?.reviewer) {
      const reviewer = dom.setupForm.querySelector(`input[name="reviewer"][value="${saved.reviewer}"]`);
      if (reviewer) reviewer.checked = true;
    }
  } catch (_) { /* Ignore corrupt local preference. */ }
}

function openRubric(fromSetup) {
  if (fromSetup) dom.setupDialog.close();
  dom.rubricDialog.showModal();
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({error: `HTTP ${response.status}`}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

async function startReview(study, reviewer) {
  try {
    state.study = study;
    state.reviewer = reviewer;
    state.bootstrap = await api(`/api/bootstrap?study=${study}&reviewer=${reviewer}`);
    dom.appShell.hidden = false;
    dom.identityLabel.textContent = `${state.bootstrap.study_label} · 独立评审员 ${reviewer}`;
    renderProgress();
    renderSessionList();
    const remembered = localStorage.getItem(`lastSession:${study}:${reviewer}`);
    let index = state.bootstrap.sessions.findIndex((session) => session.session_id === remembered && !session.completed);
    if (index < 0) index = state.bootstrap.sessions.findIndex((session) => !session.completed);
    if (index < 0) index = 0;
    await loadSession(index);
  } catch (error) {
    showToast(`无法开始评测：${error.message}`, true);
    dom.setupDialog.showModal();
  }
}

function renderProgress() {
  const progress = state.bootstrap.progress;
  const ratio = progress.sessions_total ? progress.sessions_completed / progress.sessions_total : 0;
  dom.progressPercent.textContent = `${Math.round(ratio * 100)}%`;
  dom.progressFill.style.width = `${ratio * 100}%`;
  dom.progressText.textContent = `${progress.sessions_completed} / ${progress.sessions_total} sessions · ${progress.items_completed} / ${progress.items_total} 项`;
  dom.domainProgress.replaceChildren();
  for (const [domain, value] of Object.entries(progress.domains)) {
    const row = element("div", "domain-row");
    row.append(element("span", "", domain), element("span", "", `${value.completed}/${value.total}`));
    dom.domainProgress.append(row);
  }
}

function renderSessionList() {
  if (!state.bootstrap) return;
  const filter = dom.sessionFilter.value;
  dom.sessionList.replaceChildren();
  state.bootstrap.sessions.forEach((session, index) => {
    if (filter === "pending" && session.completed) return;
    if (filter === "completed" && !session.completed) return;
    const button = element("button", `session-entry${index === state.sessionIndex ? " active" : ""}`);
    button.type = "button";
    const number = element("span", "session-number", String(index + 1).padStart(2, "0"));
    const copy = element("span", "session-copy");
    copy.append(element("strong", "", session.task), element("small", "", `${session.domain} · ${session.review_points}个时间点`));
    const dot = element("span", `completion-dot${session.completed ? " done" : ""}`);
    button.append(number, copy, dot);
    button.addEventListener("click", () => loadSession(index));
    dom.sessionList.append(button);
  });
}

async function loadSession(index) {
  const metadata = state.bootstrap.sessions[index];
  if (!metadata) return;
  stopPlayback();
  try {
    const session = await api(`/api/session?study=${state.study}&reviewer=${state.reviewer}&session_id=${encodeURIComponent(metadata.session_id)}`);
    state.session = session;
    state.sessionIndex = index;
    state.completedSession = Boolean(metadata.completed || session.saved);
    state.ratings = new Map(session.items.map((item) => [item.review_id, {review_id: item.review_id}]));
    if (session.saved?.ratings) {
      for (const rating of session.saved.ratings) state.ratings.set(rating.review_id, {...rating});
    }
    state.stages = buildStages(session.items);
    const draft = state.completedSession ? null : loadDraft();
    if (draft) {
      for (const rating of draft.ratings || []) {
        if (state.ratings.has(rating.review_id)) state.ratings.set(rating.review_id, rating);
      }
      state.stageIndex = Math.min(Number(draft.stageIndex) || 0, state.stages.length - 1);
      const allowed = state.stages[state.stageIndex].itemIndices;
      state.itemIndex = allowed.includes(Number(draft.itemIndex)) ? Number(draft.itemIndex) : allowed[0];
    } else if (state.completedSession) {
      state.stageIndex = state.stages.length - 1;
      state.itemIndex = state.stages.at(-1).itemIndices.at(-1);
    } else {
      state.stageIndex = 0;
      state.itemIndex = state.stages[0].itemIndices[0];
    }
    localStorage.setItem(`lastSession:${state.study}:${state.reviewer}`, metadata.session_id);
    dom.reviewVideo.src = `/media/${encodeURIComponent(session.video_path)}`;
    setVideoReady(false, "正在载入视频");
    renderSession();
    setItem(state.itemIndex, true);
    renderSessionList();
  } catch (error) {
    showToast(`读取 Session 失败：${error.message}`, true);
  }
}

function buildStages(items) {
  const stages = [];
  for (let index = 0; index < items.length; index += 1) {
    const item = items[index];
    const key = `${item.chunk_index}:${item.observed_through_sec}`;
    let stage = stages.at(-1);
    if (!stage || stage.key !== key) {
      stage = {key, chunkIndex: item.chunk_index, cutoff: item.observed_through_sec, itemIndices: []};
      stages.push(stage);
    }
    stage.itemIndices.push(index);
  }
  return stages;
}

function renderSession() {
  const session = state.session;
  const metadata = state.bootstrap.sessions[state.sessionIndex];
  dom.sessionMeta.textContent = `${session.domain} · SESSION ${state.sessionIndex + 1}/${state.bootstrap.sessions.length}`;
  dom.taskTitle.textContent = session.task;
  dom.queryText.textContent = session.query;
  dom.sessionState.textContent = state.completedSession ? "已确认锁定" : "评分进行中";
  dom.sessionState.className = `session-state${state.completedSession ? " complete" : ""}`;
  dom.clearItemButton.disabled = state.completedSession;
  dom.saveSessionButton.disabled = state.completedSession;
  dom.saveSessionButton.textContent = state.completedSession ? "本 Session 已保存" : "确认并保存本 Session";
  if (metadata.completed) dom.sessionFilter.value = dom.sessionFilter.value;
}

function setItem(index, seekToInterval = false) {
  if (!state.session || index < 0 || index >= state.session.items.length) return;
  const targetStage = state.stages.findIndex((stage) => stage.itemIndices.includes(index));
  if (!state.completedSession && targetStage !== state.stageIndex) return;
  state.itemIndex = index;
  stopPlayback();
  const item = currentItem();
  state.cutoff = Number(item.observed_through_sec);
  dom.videoSeek.max = String(state.cutoff);
  dom.cutoffBadge.textContent = `截止 ${formatTime(state.cutoff)}`;
  dom.intervalLabel.textContent = `当前片段 ${formatTime(item.interval[0])}–${formatTime(item.interval[1])}`;
  if (seekToInterval && dom.reviewVideo.readyState > 0) seekVideo(Number(item.interval[0]));
  else if (seekToInterval) state.pendingSeek = Number(item.interval[0]);
  else clampVideo();
  renderItemTabs();
  renderTimeline();
  renderContext();
  renderForm();
  updateNavigation();
  persistDraft();
}

function currentItem() { return state.session.items[state.itemIndex]; }
function currentRating() { return state.ratings.get(currentItem().review_id); }

function renderItemTabs() {
  dom.itemTabs.replaceChildren();
  dom.itemCounter.textContent = `时间点 ${state.stageIndex + 1}/${state.stages.length} · 评测项 ${state.itemIndex + 1}/${state.session.items.length}`;
  state.stages.forEach((stage, stageIndex) => {
    stage.itemIndices.forEach((itemIndex, withinStage) => {
      const item = state.session.items[itemIndex];
      const past = stageIndex < state.stageIndex;
      const future = stageIndex > state.stageIndex;
      const accessible = state.completedSession ? itemIndex === state.itemIndex : stageIndex === state.stageIndex;
      const button = element("button", `item-tab${itemIndex === state.itemIndex ? " active" : ""}${isItemComplete(item, state.ratings.get(item.review_id)) ? " filled" : ""}`);
      button.type = "button";
      button.disabled = !accessible;
      let title;
      let detail;
      if (past) {
        title = `时间点 ${stageIndex + 1}`;
        detail = "已锁定";
      } else if (future) {
        title = `时间点 ${stageIndex + 1}`;
        detail = "完成当前点后解锁";
      } else {
        title = state.study === "u1" ? `Candidate ${item.candidate}` : `评测项 ${withinStage + 1}`;
        detail = `Chunk ${item.chunk_index} · ${formatTime(item.observed_through_sec)}`;
      }
      button.append(element("strong", "", title), element("small", "", detail));
      if (accessible) button.addEventListener("click", () => setItem(itemIndex, false));
      dom.itemTabs.append(button);
    });
  });
}

function renderTimeline() {
  const item = currentItem();
  const max = Math.max(state.cutoff, 0.1);
  dom.causalTimeline.replaceChildren();
  for (const interval of item.video_intervals_so_far || []) {
    const segment = element("span", `timeline-segment${sameInterval(interval, item.interval) ? " current" : ""}`);
    segment.style.left = `${Math.max(0, Number(interval[0]) / max * 100)}%`;
    segment.style.width = `${Math.max(0.5, (Number(interval[1]) - Number(interval[0])) / max * 100)}%`;
    dom.causalTimeline.append(segment);
  }
  const stage = state.stages[state.stageIndex];
  const marker = element("button", "timeline-marker active");
  marker.type = "button";
  marker.title = "跳转到当前片段起点";
  marker.style.left = `${Math.min(99.5, Number(item.interval[0]) / max * 100)}%`;
  marker.addEventListener("click", () => seekVideo(Number(item.interval[0])));
  dom.causalTimeline.append(marker);
  if (stage.cutoff !== state.cutoff) throw new Error("Stage cutoff mismatch");
}

function renderContext() {
  const item = currentItem();
  dom.chunkLabel.textContent = `Chunk ${item.chunk_index} · 截止 ${formatTime(item.observed_through_sec)}`;
  dom.dialogHistory.replaceChildren();
  for (const turn of item.prior_dialog || []) {
    const parsed = parseDialogTurn(turn);
    const row = element("div", `dialog-turn ${turn.role === "assistant" ? "assistant" : "user"}`);
    row.append(element("span", "role", parsed.role), element("p", "", parsed.text));
    dom.dialogHistory.append(row);
  }
  if (!(item.prior_dialog || []).length) dom.dialogHistory.append(element("p", "", "当前没有更早的对话。"));
  const candidatePanel = dom.candidateText.closest(".candidate-panel");
  if (state.study === "u0" && item.model_action === "silent") {
    dom.candidateLabel.textContent = "模型动作 · 保持安静";
    dom.candidateText.textContent = "本时刻没有候选话语。只评价是否应介入、判断置信度和时机。";
    candidatePanel.classList.add("silent");
  } else {
    dom.candidateLabel.textContent = state.study === "u1" ? `候选 ${item.candidate}` : "模型动作 · 主动介入";
    dom.candidateText.textContent = item.candidate_utterance || "（空正文）";
    candidatePanel.classList.remove("silent");
  }
}

function parseDialogTurn(turn) {
  let text = String(turn.text || "");
  let suffix = "";
  if (text.startsWith("$interrupt$")) { text = text.slice(11); suffix = " · 主动介入"; }
  else if (text === "$silent$") { text = "保持安静"; suffix = " · 安静"; }
  const role = turn.role === "assistant" ? `Assistant${suffix}` : "User";
  return {role, text};
}

function renderForm() {
  const item = currentItem();
  const rating = currentRating();
  const disabled = state.completedSession;
  dom.ratingForm.replaceChildren();

  if (state.study === "u0") {
    const decisionBand = formBand("介入判断");
    const fields = element("div", "rating-fields");
    fields.append(
      choiceField("should_interrupt", "此刻是否应该主动介入", "依据当前视频判断，而不是猜官方标签", [
        ["yes", "应该介入"], ["no", "应该安静"], ["uncertain", "证据不足"],
      ], rating.should_interrupt, "three", disabled),
      scaleField("decision_confidence_1_5", "判断置信度", "1很不确定；5非常确定", rating.decision_confidence_1_5, disabled),
      scaleField("timeliness_1_5", "实际动作的时机", item.model_action === "spoke" ? "评价此刻发言是否及时" : "评价此刻保持安静是否合适", rating.timeliness_1_5, disabled),
    );
    decisionBand.append(fields);
    dom.ratingForm.append(decisionBand);
  }

  const needsContent = state.study === "u1" || item.model_action === "spoke";
  if (needsContent) {
    const contentBand = formBand("候选内容");
    const fields = element("div", "rating-fields");
    for (const [field, label, description] of SCORE_FIELDS) {
      fields.append(scaleField(field, label, description, rating[field], disabled));
    }
    contentBand.append(fields);
    dom.ratingForm.append(contentBand);

    const flagsBand = formBand("错误标记");
    const flagFields = element("div", "rating-fields");
    flagFields.append(
      choiceField("generic_flag", "通用空话", "换到多数其他任务仍成立", [["no", "否"], ["yes", "是"]], rating.generic_flag, "two", disabled),
      choiceField("hallucination_flag", "幻觉/无依据", "声称当前证据不支持的对象或状态", [["no", "否"], ["yes", "是"]], rating.hallucination_flag, "two", disabled),
    );
    if (state.study === "u1") {
      flagFields.append(choiceField("premature_completion_flag", "过早宣称完成", "尚未结束却声称已经完成", [["no", "否"], ["yes", "是"]], rating.premature_completion_flag, "two", disabled));
    }
    flagFields.append(choiceField("unsafe_flag", "不安全", "可能导致人身、设备或其他直接风险", [["no", "否"], ["yes", "是"]], rating.unsafe_flag, "two", disabled));
    flagsBand.append(flagFields);

    const bottomGrid = element("div", "form-grid-two");
    const selectWrap = element("div", "select-field");
    selectWrap.dataset.field = "primary_error_type";
    const selectLabel = element("label", "", "主要错误类型");
    const select = document.createElement("select");
    select.disabled = disabled;
    for (const [value, label] of PRIMARY_ERRORS) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      option.selected = rating.primary_error_type === value;
      select.append(option);
    }
    select.addEventListener("change", () => setRatingValue("primary_error_type", select.value));
    selectWrap.append(selectLabel, select);
    bottomGrid.append(selectWrap);
    flagsBand.append(bottomGrid);
    dom.ratingForm.append(flagsBand);
  } else {
    const notice = element("div", "silent-notice", "模型保持安静时不填写内容分、内容标记或主要错误类型。是否错过必要介入由上方三项记录。");
    dom.ratingForm.append(notice);
  }

  const notesBand = formBand("评审备注（可选）");
  const notesWrap = element("div", "notes-field");
  const notes = document.createElement("textarea");
  notes.placeholder = "记录关键视觉证据、错误对象或分数理由。不要写入未来信息。";
  notes.maxLength = 2000;
  notes.value = rating.notes || "";
  notes.disabled = disabled;
  notes.addEventListener("input", () => setRatingValue("notes", notes.value, false));
  notesWrap.append(notes);
  notesBand.append(notesWrap);
  dom.ratingForm.append(notesBand);
  updateItemStatus();
}

function formBand(title) {
  const band = element("div", "form-band");
  band.append(element("h3", "", title));
  return band;
}

function scaleField(field, label, description, current, disabled) {
  const wrapper = ratingField(field, label, description);
  const control = element("div", "scale-control");
  for (let value = 1; value <= 5; value += 1) {
    const button = element("button", `${current === value ? "selected " : ""}${value <= 2 ? "low" : value === 3 ? "mid" : "high"}`, String(value));
    button.type = "button";
    button.disabled = disabled;
    button.setAttribute("aria-pressed", String(current === value));
    button.addEventListener("click", () => setRatingValue(field, value));
    control.append(button);
  }
  wrapper.append(control);
  return wrapper;
}

function choiceField(field, label, description, choices, current, columns, disabled) {
  const wrapper = ratingField(field, label, description);
  const control = element("div", `binary-control ${columns}`);
  for (const [value, text] of choices) {
    const button = element("button", current === value ? "selected" : "", text);
    button.type = "button";
    button.disabled = disabled;
    button.setAttribute("aria-pressed", String(current === value));
    button.addEventListener("click", () => setRatingValue(field, value));
    control.append(button);
  }
  wrapper.append(control);
  return wrapper;
}

function ratingField(field, label, description) {
  const wrapper = element("div", "rating-field");
  wrapper.dataset.field = field;
  const copy = element("div", "field-label");
  copy.append(element("strong", "", label), element("small", "", description));
  wrapper.append(copy);
  return wrapper;
}

function setRatingValue(field, value, rerender = true) {
  if (state.completedSession) return;
  const rating = currentRating();
  rating[field] = value;
  state.ratings.set(rating.review_id, rating);
  persistDraft();
  if (rerender) renderForm();
  renderItemTabs();
  updateNavigation();
}

function requiredFields(item) {
  const fields = [];
  if (state.study === "u0") fields.push("should_interrupt", "decision_confidence_1_5", "timeliness_1_5");
  if (state.study === "u1" || item.model_action === "spoke") {
    fields.push(...SCORE_FIELDS.map(([field]) => field), "generic_flag", "hallucination_flag", "unsafe_flag", "primary_error_type");
    if (state.study === "u1") fields.push("premature_completion_flag");
  }
  return fields;
}

function missingFields(item, rating) {
  return requiredFields(item).filter((field) => rating?.[field] === undefined || rating?.[field] === null || rating?.[field] === "");
}

function isItemComplete(item, rating) { return missingFields(item, rating).length === 0; }

function updateItemStatus() {
  const missing = missingFields(currentItem(), currentRating());
  dom.ratingStatus.textContent = state.completedSession ? "已确认，只读" : missing.length ? `还需填写 ${missing.length} 项` : "本项已完整，草稿已保存";
  dom.ratingStatus.style.color = missing.length ? "var(--warn)" : "var(--complete)";
}

function updateNavigation() {
  const stage = state.stages[state.stageIndex];
  const position = stage.itemIndices.indexOf(state.itemIndex);
  dom.previousItemButton.disabled = state.completedSession || position <= 0;
  dom.nextItemButton.disabled = state.completedSession;
  if (position < stage.itemIndices.length - 1) {
    dom.nextItemButton.textContent = "下一评测项";
  } else if (state.stageIndex < state.stages.length - 1) {
    dom.nextItemButton.textContent = "锁定此时间点并继续";
  } else {
    dom.nextItemButton.textContent = "已到最后评测项";
    dom.nextItemButton.disabled = true;
  }
  const allComplete = state.session.items.every((item) => isItemComplete(item, state.ratings.get(item.review_id)));
  const finalStage = state.stageIndex === state.stages.length - 1;
  const completedItems = state.session.items.filter((item) => isItemComplete(item, state.ratings.get(item.review_id))).length;
  dom.sessionCompletionText.textContent = state.completedSession ? "评分已写入服务器" : `${completedItems}/${state.session.items.length}项完整${finalStage ? "" : " · 后续时间点尚未解锁"}`;
  dom.saveSessionButton.disabled = state.completedSession || !allComplete || !finalStage;
}

function previousItem() {
  const indices = state.stages[state.stageIndex].itemIndices;
  const position = indices.indexOf(state.itemIndex);
  if (position > 0) setItem(indices[position - 1], false);
}

function nextItem() {
  const stage = state.stages[state.stageIndex];
  const position = stage.itemIndices.indexOf(state.itemIndex);
  if (position < stage.itemIndices.length - 1) {
    setItem(stage.itemIndices[position + 1], false);
    return;
  }
  const incomplete = stage.itemIndices.filter((index) => {
    const item = state.session.items[index];
    return !isItemComplete(item, state.ratings.get(item.review_id));
  });
  if (incomplete.length) {
    setItem(incomplete[0], false);
    highlightMissing();
    showToast("请先完成当前时间点的全部评分。", true);
    return;
  }
  if (state.stageIndex >= state.stages.length - 1) return;
  const accepted = window.confirm("锁定当前时间点后将显示后续视频，且不能返回修改早期评分。确认继续？");
  if (!accepted) return;
  state.stageIndex += 1;
  setItem(state.stages[state.stageIndex].itemIndices[0], true);
  showToast("当前时间点评分已锁定，已进入下一时间点。", false);
}

function clearCurrentItem() {
  if (state.completedSession) return;
  if (!window.confirm("清空当前评测项的未确认评分？")) return;
  state.ratings.set(currentItem().review_id, {review_id: currentItem().review_id});
  persistDraft();
  renderForm();
  renderItemTabs();
  updateNavigation();
}

async function saveSession() {
  if (state.completedSession) return;
  const incompleteIndex = state.session.items.findIndex((item) => !isItemComplete(item, state.ratings.get(item.review_id)));
  if (incompleteIndex >= 0) {
    const targetStage = state.stages.findIndex((stage) => stage.itemIndices.includes(incompleteIndex));
    if (targetStage === state.stageIndex) setItem(incompleteIndex, false);
    highlightMissing();
    showToast("仍有必填项未完成。", true);
    return;
  }
  if (state.stageIndex !== state.stages.length - 1) {
    showToast("请按时间顺序完成后续评测点。", true);
    return;
  }
  if (!window.confirm("确认提交本 Session？提交后评分会锁定，不能在网页中修改。")) return;
  dom.saveSessionButton.disabled = true;
  dom.saveSessionButton.textContent = "正在保存";
  try {
    const payload = {
      study: state.study,
      reviewer_slot: state.reviewer,
      session_id: state.session.session_id,
      ratings: state.session.items.map((item) => state.ratings.get(item.review_id)),
    };
    const result = await api("/api/save-session", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    state.bootstrap.progress = result.progress;
    const metadata = state.bootstrap.sessions[state.sessionIndex];
    metadata.completed = true;
    clearDraft();
    state.completedSession = true;
    renderProgress();
    renderSessionList();
    showToast("本 Session 已原子保存，正在进入下一项。", false);
    const next = findNextPending(state.sessionIndex);
    if (next >= 0) await loadSession(next);
    else {
      renderSession();
      renderForm();
      updateNavigation();
      showToast("当前评审任务的全部 Session 已完成。", false);
    }
  } catch (error) {
    dom.saveSessionButton.disabled = false;
    dom.saveSessionButton.textContent = "确认并保存本 Session";
    showToast(`保存失败：${error.message}`, true);
  }
}

function findNextPending(afterIndex) {
  const sessions = state.bootstrap.sessions;
  for (let offset = 1; offset <= sessions.length; offset += 1) {
    const index = (afterIndex + offset) % sessions.length;
    if (!sessions[index].completed) return index;
  }
  return -1;
}

function highlightMissing() {
  const missing = new Set(missingFields(currentItem(), currentRating()));
  dom.ratingForm.querySelectorAll("[data-field]").forEach((node) => {
    node.classList.toggle("missing-field", missing.has(node.dataset.field));
  });
}

function draftKey() { return `reviewDraft:${state.study}:${state.reviewer}:${state.session.session_id}`; }

function persistDraft() {
  if (!state.session || state.completedSession) return;
  const payload = {
    source_blind_sha256: state.bootstrap.source_blind_sha256,
    stageIndex: state.stageIndex,
    itemIndex: state.itemIndex,
    ratings: state.session.items.map((item) => state.ratings.get(item.review_id)),
  };
  localStorage.setItem(draftKey(), JSON.stringify(payload));
}

function loadDraft() {
  try {
    const value = JSON.parse(localStorage.getItem(draftKey()) || "null");
    if (!value || value.source_blind_sha256 !== state.bootstrap.source_blind_sha256) return null;
    return value;
  } catch (_) { return null; }
}

function clearDraft() { localStorage.removeItem(draftKey()); }

async function togglePlayback() {
  if (!state.session) return;
  if (!dom.reviewVideo.paused) {
    stopPlayback();
    return;
  }
  const generation = ++state.playbackGeneration;
  dom.playButton.disabled = true;
  try {
    if (dom.reviewVideo.readyState < 1) await waitForVideoEvent("loadedmetadata", 5000);
    if (dom.reviewVideo.currentTime >= state.cutoff - 0.05) {
      seekVideo(currentItem().interval[0]);
    }
    if (generation !== state.playbackGeneration) return;
    await dom.reviewVideo.play();
    if (generation !== state.playbackGeneration) return;
    setVideoReady(true);
    monitorPlaybackBoundary(generation);
  } catch (error) {
    if (generation !== state.playbackGeneration || error?.name === "AbortError") return;
    showToast(`视频无法播放：${error.message}`, true);
  } finally {
    if (generation === state.playbackGeneration) dom.playButton.disabled = false;
  }
}

function stopPlayback() {
  state.playbackGeneration += 1;
  if (state.boundaryFrame !== null) cancelAnimationFrame(state.boundaryFrame);
  state.boundaryFrame = null;
  if (!dom.reviewVideo.paused) dom.reviewVideo.pause();
  dom.playButton.disabled = !state.videoReady;
}

function monitorPlaybackBoundary(generation) {
  if (generation !== state.playbackGeneration || dom.reviewVideo.paused) return;
  if (dom.reviewVideo.currentTime >= state.cutoff - 0.02) {
    dom.reviewVideo.pause();
    if (dom.reviewVideo.currentTime > state.cutoff) dom.reviewVideo.currentTime = state.cutoff;
    updateVideoReadout();
    return;
  }
  state.boundaryFrame = requestAnimationFrame(() => monitorPlaybackBoundary(generation));
}

function setVideoReady(ready, message = "") {
  state.videoReady = ready;
  dom.playButton.disabled = !ready;
  dom.videoEmpty.hidden = ready;
  if (message) dom.videoEmpty.textContent = message;
}

function showVideoMessage(message) {
  dom.videoEmpty.textContent = message;
  dom.videoEmpty.hidden = false;
}

function waitForVideoEvent(name, timeoutMs) {
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("视频加载超时"));
    }, timeoutMs);
    const onEvent = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(new Error("视频加载或解码失败")); };
    const cleanup = () => {
      window.clearTimeout(timeout);
      dom.reviewVideo.removeEventListener(name, onEvent);
      dom.reviewVideo.removeEventListener("error", onError);
    };
    dom.reviewVideo.addEventListener(name, onEvent, {once: true});
    dom.reviewVideo.addEventListener("error", onError, {once: true});
  });
}

function seekVideo(seconds) {
  if (!state.session || !Number.isFinite(seconds)) return;
  dom.reviewVideo.currentTime = Math.max(0, Math.min(state.cutoff, seconds));
  updateVideoReadout();
}

function clampVideo() {
  if (!state.session) return;
  if (dom.reviewVideo.currentTime > state.cutoff) dom.reviewVideo.currentTime = state.cutoff;
}

function updateVideoReadout() {
  const current = Math.min(state.cutoff, Number(dom.reviewVideo.currentTime) || 0);
  dom.timeReadout.textContent = `${formatTime(current)} / ${formatTime(state.cutoff)}`;
  dom.videoSeek.value = String(current);
}

function sameInterval(left, right) {
  return Number(left?.[0]) === Number(right?.[0]) && Number(left?.[1]) === Number(right?.[1]);
}

function formatTime(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  const remainder = value - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`;
}

function element(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null) node.textContent = text;
  return node;
}

function showToast(message, isError) {
  clearTimeout(state.toastTimer);
  dom.toast.textContent = message;
  dom.toast.className = `toast visible${isError ? " error" : ""}`;
  state.toastTimer = setTimeout(() => { dom.toast.className = "toast"; }, 3600);
}
