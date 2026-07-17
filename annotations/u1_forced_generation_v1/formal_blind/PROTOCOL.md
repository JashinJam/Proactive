# U1 正式盲态标注协议

## 目的与隔离

本标注只构造 answer-blind、model-output-blind 的因果 procedural state，用于 U1 oracle upper-bound。标注者不得读取：

- `CURRENT_ROUTE.md`、U1 报告或任何 U1 实验输出；
- `paired_review_key.jsonl`、任何 ratings 文件；
- `oracle_states.smoke.json` 或其他既有 oracle 标注；
- 源数据中的当前/未来 `answers`、R0/D1 输出或错误类别。

允许读取的项目文件仅为分配到的净化 sample JSONL、本协议、只读工具 `src/proactive_u1/contact_sheet.py`，以及 `data/egoproactive/val/<video_path>` 中对应的视频。临时 contact sheet 只能写入分配的 `/tmp` 目录。

## 因果边界

每个 session 必须按 `observed_through_sec` 递增标注。

1. 静态 `goal` 和 `steps` 只能依据 `query` 与 `task` 编写，必须在观看视频前确定。
2. 动态状态可依据 `query`、`task`、该 sample 的 `prior_dialog`，以及 `video_intervals_so_far` 明确列出的历史与当前区间；不得观看这些区间之间的空档视频，即使其绝对时间早于 `observed_through_sec`。
3. 标注较早状态时不得观看其最后一个允许区间之后的内容；完成该状态后才能继续观看下一个 sample 新增的允许区间。
4. 不得推断或改写当前 gold utterance，不得写 `$interrupt$`、`$silent$`、`should speak` 或 `should interrupt`。
5. `completion_evidence`、`incompletion_or_error_evidence` 必须是截至当前时刻可观察的事实；不可见或不确定时保持为空并降低 `confidence`。

## 粒度与字段

`steps` 使用完成任务所需的 macro steps，不把每个手部动作拆成独立 step。当前时刻的 atomic action 写入 evidence 或 `recovery_action`。`current_step_id` 指向当前正在执行、刚完成但尚未进入下一步，或当前需要恢复的 macro step；`next_step_id` 指向后继 macro step，没有后继则为 `null`。

`progress` 只能为：

- `not_started`
- `ongoing`
- `complete`
- `deviated`
- `recovered`

`confidence` 为 `[0, 1]`。没有明确错误时 `recovery_action` 可写下一项最直接动作；任务完成且无需动作时写 `none`。

## 输出格式

输出是 JSON array，每个 session 一个对象，必须覆盖该 session 的 4 个 sample：

```json
{
  "schema_version": 1,
  "status": "complete",
  "input_index": 1,
  "video_path": "example.mp4",
  "query": "...",
  "task": "...",
  "domain": "...",
  "is_smoke_session": false,
  "provenance": {
    "plan_inputs": ["task", "query"],
    "chunk_inputs": ["task", "query", "dialog_at_chunk", "video_through_interval_end"],
    "excluded_inputs": ["answers", "future_dialog", "future_video", "model_outputs", "R0/D1 errors"],
    "annotation_type": "formal_blind_evaluation_only_oracle_non_deployable"
  },
  "goal": "...",
  "steps": [
    {
      "id": "s1",
      "text": "...",
      "completion_cues": ["..."],
      "incompletion_cues": ["..."]
    }
  ],
  "sampled_chunk_states": [
    {
      "sample_id": "...",
      "chunk_index": 1,
      "observed_through_sec": 10.0,
      "current_step_id": "s1",
      "progress": "ongoing",
      "completion_evidence": ["..."],
      "incompletion_or_error_evidence": ["..."],
      "next_step_id": "s2",
      "recovery_action": "...",
      "confidence": 0.8
    }
  ]
}
```

标注文本统一使用简洁英文。不得读取另一分片或修改任何未分配文件。
