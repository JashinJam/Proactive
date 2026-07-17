"""Schema, causal validation, and prompt rendering for R1 oracle state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Literal

from proactive_r0.core import DialogNormalizer, build_messages

StateVariant = Literal["null", "step", "cues", "full"]
STATE_VARIANTS: tuple[StateVariant, ...] = ("null", "step", "cues", "full")
PROGRESS_VALUES = {"not_started", "ongoing", "complete", "deviated", "recovered"}
FORBIDDEN_STATE_MARKERS = (
    "$interrupt$",
    "$silent$",
    "should interrupt",
    "should stay silent",
    "gold decision",
    "gold label",
)


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def _selection_key(seed: str, video_path: str) -> str:
    return text_sha256(f"{seed}\0{video_path}")


def validate_and_select_manifest(
    manifest: dict[str, object], source_rows: list[dict[str, object]]
) -> list[tuple[int, dict[str, object]]]:
    """Verify the label-independent selection algorithm and return its rows."""
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported R1 manifest schema_version")
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Manifest selection must be an object")
    if selection.get("algorithm") != "domain_stratified_sha256_first_k":
        raise ValueError("Unsupported pilot selection algorithm")
    if selection.get("independent_of_gold_labels") is not True:
        raise ValueError("Pilot selection must declare label independence")
    seed = selection.get("seed")
    sessions_per_domain = selection.get("sessions_per_domain")
    listed = selection.get("sessions")
    if not isinstance(seed, str) or not seed:
        raise ValueError("Manifest selection seed must be non-empty")
    if not isinstance(sessions_per_domain, int) or sessions_per_domain <= 0:
        raise ValueError("sessions_per_domain must be positive")
    if not isinstance(listed, list) or not listed:
        raise ValueError("Manifest sessions must be non-empty")

    by_domain: dict[str, list[tuple[str, int, dict[str, object]]]] = {}
    for index, row in enumerate(source_rows):
        domain = row.get("domain")
        video_path = row.get("video_path")
        if not isinstance(domain, str) or not isinstance(video_path, str):
            raise ValueError(f"Source row {index} lacks domain/video_path")
        by_domain.setdefault(domain, []).append(
            (_selection_key(seed, video_path), index, row)
        )

    expected: list[tuple[int, dict[str, object]]] = []
    for domain in sorted(by_domain):
        ranked = sorted(by_domain[domain], key=lambda item: item[0])
        expected.extend((index, row) for _, index, row in ranked[:sessions_per_domain])
    expected.sort(key=lambda item: item[0])

    expected_summary = [
        {
            "input_index": index,
            "video_path": row["video_path"],
            "domain": row["domain"],
            "task": row["task"],
            "query": row["query"],
            "query_sha256": text_sha256(str(row["query"])),
            "chunks": len(row["video_intervals"]),
        }
        for index, row in expected
    ]
    if listed != expected_summary:
        raise ValueError("Manifest sessions do not match the frozen selection algorithm")
    return expected


def _require_text(value: object, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{field} must be non-empty text")
    if len(value) > 500:
        raise ValueError(f"{field} exceeds 500 characters")
    lowered = value.lower()
    for marker in FORBIDDEN_STATE_MARKERS:
        if marker in lowered:
            raise ValueError(f"{field} contains forbidden target marker {marker!r}")
    return value.strip()


def _require_text_list(value: object, field: str, min_items: int = 0) -> list[str]:
    if not isinstance(value, list) or len(value) < min_items:
        raise ValueError(f"{field} must contain at least {min_items} items")
    return [
        _require_text(item, f"{field}[{index}]") for index, item in enumerate(value)
    ]


def validate_annotations(
    annotations: list[dict[str, object]],
    selected: list[tuple[int, dict[str, object]]],
) -> dict[int, dict[str, object]]:
    """Validate exact coverage, source identity, and per-chunk causal timestamps."""
    if len(annotations) != len(selected):
        raise ValueError(
            f"Annotation count {len(annotations)} does not match pilot sessions {len(selected)}"
        )
    by_index: dict[int, dict[str, object]] = {}
    for annotation in annotations:
        input_index = annotation.get("input_index")
        if not isinstance(input_index, int) or input_index in by_index:
            raise ValueError(f"Invalid or duplicate annotation input_index {input_index!r}")
        by_index[input_index] = annotation

    for input_index, row in selected:
        annotation = by_index.get(input_index)
        if annotation is None:
            raise ValueError(f"Missing annotation for source row {input_index}")
        if annotation.get("schema_version") != 1:
            raise ValueError(f"row {input_index}: unsupported annotation schema")
        if annotation.get("status") != "complete":
            raise ValueError(f"row {input_index}: annotation is not complete")
        if annotation.get("video_path") != row.get("video_path"):
            raise ValueError(f"row {input_index}: video_path mismatch")
        if annotation.get("query_sha256") != text_sha256(str(row.get("query", ""))):
            raise ValueError(f"row {input_index}: query fingerprint mismatch")
        provenance = annotation.get("provenance")
        expected_provenance = {
            "plan_inputs": ["task", "query"],
            "chunk_inputs": ["task", "query", "dialog_at_chunk", "video_through_interval_end"],
            "excluded_inputs": ["answers", "future_dialog", "future_video"],
            "annotation_type": "evaluation_only_oracle_non_deployable",
        }
        if provenance != expected_provenance:
            raise ValueError(f"row {input_index}: provenance policy mismatch")

        goal = _require_text(annotation.get("goal"), f"row {input_index}.goal")
        if goal != str(row.get("query", "")).strip():
            raise ValueError(f"row {input_index}: goal must equal the source query")
        steps = annotation.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"row {input_index}: steps must be non-empty")
        step_by_id: dict[str, dict[str, object]] = {}
        ordered_ids: list[str] = []
        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                raise ValueError(f"row {input_index}: step {step_index} must be an object")
            step_id = _require_text(step.get("id"), f"row {input_index}.steps[{step_index}].id")
            if step_id in step_by_id:
                raise ValueError(f"row {input_index}: duplicate step id {step_id}")
            _require_text(step.get("text"), f"row {input_index}.steps[{step_index}].text")
            _require_text_list(
                step.get("completion_cues"),
                f"row {input_index}.steps[{step_index}].completion_cues",
                min_items=1,
            )
            _require_text_list(
                step.get("incompletion_cues"),
                f"row {input_index}.steps[{step_index}].incompletion_cues",
                min_items=1,
            )
            step_by_id[step_id] = step
            ordered_ids.append(step_id)

        states = annotation.get("chunk_states")
        intervals = row.get("video_intervals")
        if not isinstance(states, list) or not isinstance(intervals, list):
            raise ValueError(f"row {input_index}: invalid states or source intervals")
        if len(states) != len(intervals):
            raise ValueError(f"row {input_index}: state count must match intervals")
        previous_last_update = -1
        previous_dynamic: tuple[object, ...] | None = None
        for chunk_index, (state, interval) in enumerate(zip(states, intervals)):
            if not isinstance(state, dict) or not isinstance(interval, list):
                raise ValueError(f"row {input_index} chunk {chunk_index}: invalid state")
            if state.get("chunk_index") != chunk_index:
                raise ValueError(f"row {input_index}: non-contiguous chunk states")
            observed_through = state.get("observed_through_sec")
            interval_end = float(interval[1])
            if not isinstance(observed_through, (int, float)):
                raise ValueError(f"row {input_index} chunk {chunk_index}: bad timestamp")
            if abs(float(observed_through) - interval_end) > 1e-6:
                raise ValueError(
                    f"row {input_index} chunk {chunk_index}: state must be observed exactly through interval end"
                )
            current_step_id = state.get("current_step_id")
            next_step_id = state.get("next_step_id")
            if current_step_id not in step_by_id:
                raise ValueError(f"row {input_index} chunk {chunk_index}: unknown current step")
            if next_step_id is not None and next_step_id not in step_by_id:
                raise ValueError(f"row {input_index} chunk {chunk_index}: unknown next step")
            progress = state.get("progress")
            if progress not in PROGRESS_VALUES:
                raise ValueError(f"row {input_index} chunk {chunk_index}: bad progress")
            completion_evidence = _require_text_list(
                state.get("completion_evidence"),
                f"row {input_index}.chunk_states[{chunk_index}].completion_evidence",
            )
            error_evidence = _require_text_list(
                state.get("incompletion_or_error_evidence"),
                f"row {input_index}.chunk_states[{chunk_index}].incompletion_or_error_evidence",
            )
            confidence = state.get("confidence")
            if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
                raise ValueError(f"row {input_index} chunk {chunk_index}: confidence outside [0,1]")
            last_update = state.get("last_update_chunk")
            if not isinstance(last_update, int) or not 0 <= last_update <= chunk_index:
                raise ValueError(f"row {input_index} chunk {chunk_index}: invalid last_update_chunk")
            if last_update < previous_last_update:
                raise ValueError(f"row {input_index} chunk {chunk_index}: last_update regressed")
            dynamic = (
                current_step_id,
                next_step_id,
                progress,
                tuple(completion_evidence),
                tuple(error_evidence),
                float(confidence),
            )
            if previous_dynamic is not None and last_update == previous_last_update:
                if dynamic != previous_dynamic:
                    raise ValueError(
                        f"row {input_index} chunk {chunk_index}: state changed without an update"
                    )
            previous_last_update = last_update
            previous_dynamic = dynamic
    return by_index


def _step_by_id(annotation: dict[str, object], step_id: str | None) -> dict[str, object] | None:
    if step_id is None:
        return None
    for step in annotation["steps"]:  # type: ignore[index]
        if isinstance(step, dict) and step.get("id") == step_id:
            return step
    raise KeyError(step_id)


def _line(name: str, value: object) -> str:
    if isinstance(value, list):
        rendered = "; ".join(str(item) for item in value) if value else "none observed"
    elif value is None:
        rendered = "none"
    else:
        rendered = str(value)
    return f"{name}: {rendered}"


def render_state(
    annotation: dict[str, object], chunk_index: int, variant: StateVariant
) -> str:
    """Render only the fields authorized for one controlled R1 variant."""
    if variant not in STATE_VARIANTS:
        raise ValueError(f"Unknown state variant: {variant}")
    lines = ["<procedural_state>"]
    if variant == "null":
        lines.append("status: unavailable")
    else:
        states = annotation["chunk_states"]
        if not isinstance(states, list) or not 0 <= chunk_index < len(states):
            raise IndexError(chunk_index)
        state = states[chunk_index]
        if not isinstance(state, dict):
            raise ValueError("chunk state must be an object")
        current = _step_by_id(annotation, str(state["current_step_id"]))
        assert current is not None
        lines.append(_line("current_step", current["text"]))
        if variant in ("cues", "full"):
            lines.append(_line("completion_cues", current["completion_cues"]))
            lines.append(_line("incompletion_cues", current["incompletion_cues"]))
        if variant == "full":
            next_step = _step_by_id(annotation, state.get("next_step_id"))
            lines.extend(
                [
                    _line("goal", annotation["goal"]),
                    _line("progress", state["progress"]),
                    _line("completion_evidence", state["completion_evidence"]),
                    _line(
                        "incompletion_or_error_evidence",
                        state["incompletion_or_error_evidence"],
                    ),
                    _line("next_step", next_step["text"] if next_step else None),
                    _line("confidence", f"{float(state['confidence']):.2f}"),
                    _line("last_update_chunk", state["last_update_chunk"]),
                ]
            )
    lines.append("</procedural_state>")
    return "\n".join(lines)


def build_state_messages(
    row: dict[str, object],
    chunk_index: int,
    system_prompt: str,
    normalize_dialog_turns: DialogNormalizer,
    max_history_turns: int,
    state_block: str,
) -> list[dict[str, str]]:
    """Keep the official message flow and append state to the system context."""
    messages = build_messages(
        row=row,
        chunk_index=chunk_index,
        system_prompt=system_prompt,
        normalize_dialog_turns=normalize_dialog_turns,
        max_history_turns=max_history_turns,
    )
    if not messages or messages[0]["role"] != "system":
        raise ValueError("Official message flow did not start with a system prompt")
    messages[0] = {
        "role": "system",
        "content": f"{messages[0]['content']}\n\n{state_block}",
    }
    return messages


def annotation_paths_from_config(
    project_root: Path, config: dict[str, object]
) -> tuple[Path, Path]:
    oracle = config.get("oracle_state")
    if not isinstance(oracle, dict):
        raise ValueError("Config oracle_state must be an object")

    def resolve(value: object) -> Path:
        path = Path(str(value)).expanduser()
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()

    return resolve(oracle["manifest"]), resolve(oracle["annotations"])


def iter_annotation_text(annotation: dict[str, object]) -> Iterable[str]:
    """Expose annotation strings for audits without serializing target data."""
    yield str(annotation["goal"])
    for step in annotation["steps"]:  # type: ignore[index]
        if isinstance(step, dict):
            yield str(step["text"])
            yield from (str(item) for item in step["completion_cues"])  # type: ignore[index]
            yield from (str(item) for item in step["incompletion_cues"])  # type: ignore[index]
    for state in annotation["chunk_states"]:  # type: ignore[index]
        if isinstance(state, dict):
            yield from (str(item) for item in state["completion_evidence"])  # type: ignore[index]
            yield from (
                str(item) for item in state["incompletion_or_error_evidence"]  # type: ignore[index]
            )
