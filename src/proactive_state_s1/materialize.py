"""Freeze query-only S1 plans and create split-isolated annotation work files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from proactive_r0.artifacts import write_json
from proactive_state_s1.core import load_json, load_jsonl, sha256_file, validate_plan


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def materialize(
    sessions_path: Path,
    template_path: Path,
    plans_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"S1 annotation workspace already exists: {output_dir}")
    sessions = load_jsonl(sessions_path)
    templates = load_json(template_path)
    plans = load_json(plans_path)
    if not isinstance(templates, list) or not isinstance(plans, list):
        raise ValueError("S1 templates and plans must be JSON arrays")
    session_by_index = {int(row["input_index"]): row for row in sessions}
    template_by_index = {
        int(row["input_index"]): row for row in templates if isinstance(row, dict)
    }
    plan_by_index: dict[int, dict[str, object]] = {}
    for plan in plans:
        if not isinstance(plan, dict) or not isinstance(plan.get("input_index"), int):
            raise ValueError("Every S1 plan must contain an integer input_index")
        input_index = int(plan["input_index"])
        if input_index in plan_by_index:
            raise ValueError(f"Duplicate S1 plan input_index {input_index}")
        if set(plan) != {"input_index", "goal", "steps"}:
            raise ValueError(f"S1 plan {input_index} has unexpected identity/dynamic fields")
        plan_by_index[input_index] = plan
    if set(plan_by_index) != set(session_by_index) or set(template_by_index) != set(session_by_index):
        raise ValueError("S1 plan/template coverage differs from selected sessions")

    by_split: dict[str, list[dict[str, object]]] = {"train": [], "heldout": []}
    for session in sessions:
        input_index = int(session["input_index"])
        annotation = dict(template_by_index[input_index])
        annotation["goal"] = plan_by_index[input_index]["goal"]
        annotation["steps"] = plan_by_index[input_index]["steps"]
        validate_plan(annotation, session)
        split = str(session["state_split"])
        if split not in by_split:
            raise ValueError(f"Unknown S1 split {split}")
        by_split[split].append(annotation)

    output_dir.mkdir(parents=True)
    artifacts: dict[str, dict[str, object]] = {}
    for split, rows in by_split.items():
        split_dir = output_dir / split
        split_dir.mkdir()
        path = split_dir / "annotations.json"
        write_json(path, rows)
        artifacts[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "sessions": len(rows),
            "states": sum(len(row["chunk_states"]) for row in rows),
        }
    manifest = {
        "schema_version": 1,
        "status": "query-only plans frozen; dynamic state annotations pending",
        "plan_fields": ["task", "query"],
        "new_s1_video_frames_inspected_before_plan_freeze": False,
        "answers_or_model_outputs_read_for_plans": False,
        "prior_engineering_dialog_exposure_by_plan_author": [20, 25],
        "sources": {
            "sessions": {"path": str(sessions_path), "sha256": sha256_file(sessions_path)},
            "template": {"path": str(template_path), "sha256": sha256_file(template_path)},
            "plans": {"path": str(plans_path), "sha256": sha256_file(plans_path)},
            "protocol": {
                "path": str(PROJECT_ROOT / "annotations/state_s1_decoder_v1/PROTOCOL.md"),
                "sha256": sha256_file(
                    PROJECT_ROOT / "annotations/state_s1_decoder_v1/PROTOCOL.md"
                ),
            },
        },
        "artifacts": artifacts,
        "heldout_annotation_path_is_separate": True,
    }
    write_json(output_dir / "plan_freeze_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = materialize(
        _resolve(args.sessions),
        _resolve(args.template),
        _resolve(args.plans),
        _resolve(args.output_dir),
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
