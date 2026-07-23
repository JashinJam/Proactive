"""Container-facing D4 adapter for hidden EgoProactive inference.

The official test Docker template is not public yet. This adapter keeps the
model-facing contract independent from that future wrapper: it accepts one
EgoProactive JSONL plus a video directory and atomically publishes the standard
``predictions.jsonl``. The frozen experiment runner remains the inference
implementation and receives a generated runtime config with no pinned val path.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Sequence

from proactive_d1.core import strip_answers
from proactive_d1.run_deploy import main as run_deploy_main
from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import (
    load_jsonl,
    validate_prediction_rows,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "d4_internvl35_1b_dialog_stage_deploy_shared_vision_v1.json"
)
DEFAULT_HEAD = (
    PROJECT_ROOT
    / "submission"
    / "d4_small"
    / "decision_head.json"
)
DEFAULT_STARTER_KIT = PROJECT_ROOT / "starter_kit"
DEFAULT_MANIFEST = PROJECT_ROOT / "submission" / "d4_small" / "manifest.json"


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_submission_input_rows(
    rows: Sequence[dict[str, object]],
    *,
    allow_answers_for_local_audit: bool = False,
) -> dict[str, object]:
    """Reject target-bearing or structurally incomplete test inputs early."""
    if not rows:
        raise ValueError("Submission input must contain at least one session")
    rows_with_answers = 0
    total_chunks = 0
    required = ("video_path", "video_intervals", "query", "domain", "dialog")
    for row_index, row in enumerate(rows):
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"row {row_index}: missing required fields {missing}")
        if "answers" in row:
            rows_with_answers += 1
            if not allow_answers_for_local_audit:
                raise ValueError(
                    f"row {row_index}: hidden inference input must not contain answers"
                )
        intervals = row.get("video_intervals")
        dialog = row.get("dialog")
        if not isinstance(intervals, list) or not intervals:
            raise ValueError(f"row {row_index}: video_intervals must be non-empty")
        if not isinstance(dialog, list) or len(dialog) != len(intervals):
            raise ValueError(
                f"row {row_index}: dialog must align with video_intervals"
            )
        total_chunks += len(intervals)
    return {
        "sessions": len(rows),
        "chunks": total_chunks,
        "rows_with_answers": rows_with_answers,
        "hidden_input_contract": rows_with_answers == 0,
    }


def validate_bundle_manifest(
    manifest: dict[str, object],
    config: dict[str, object],
) -> dict[str, object]:
    """Cross-check parameter and frozen-artifact metadata before model loading."""
    model = dict(manifest["model"])  # type: ignore[arg-type]
    head = dict(manifest["decision_head"])  # type: ignore[arg-type]
    submission = dict(manifest["submission_metadata"])  # type: ignore[arg-type]
    config_model = dict(config["model"])  # type: ignore[arg-type]
    config_head = dict(config["decision_head"])  # type: ignore[arg-type]
    config_inference = dict(config["inference"])  # type: ignore[arg-type]
    frozen_inference = dict(manifest["frozen_inference"])  # type: ignore[arg-type]
    backbone_parameters = int(model["parameters"])
    head_parameters = int(head["parameters"])
    system_parameters = backbone_parameters + head_parameters
    if backbone_parameters != int(config_model["total_parameters"]):
        raise ValueError("Bundle manifest backbone parameter count changed")
    if head_parameters != int(config_head["parameters"]):
        raise ValueError("Bundle manifest head parameter count changed")
    if head["sha256"] != config_head["sha256"]:
        raise ValueError("Bundle manifest head fingerprint changed")
    if model["weights_sha256"] != config_model["weights_sha256"]:
        raise ValueError("Bundle manifest model fingerprint changed")
    if system_parameters != int(submission["total_parameters"]):
        raise ValueError("Bundle manifest total parameter accounting changed")
    if system_parameters != int(submission["active_parameters"]):
        raise ValueError("Dense D4 active parameter accounting changed")
    if system_parameters > 2_000_000_000:
        raise ValueError("D4 bundle exceeds the C1 Small parameter limit")
    for name in (
        "frames_per_interval",
        "max_frames",
        "max_history_turns",
        "max_new_tokens",
        "decision_feature_mode",
    ):
        if config_inference.get(name) != frozen_inference.get(name):
            raise ValueError(f"Bundle manifest inference setting changed: {name}")
    if config_model.get("dtype") != frozen_inference.get("dtype"):
        raise ValueError("Bundle manifest inference setting changed: dtype")
    if config_model.get("attention_implementation") != frozen_inference.get(
        "attention_implementation"
    ):
        raise ValueError(
            "Bundle manifest inference setting changed: attention_implementation"
        )
    return {
        "backbone_parameters": backbone_parameters,
        "head_parameters": head_parameters,
        "total_parameters": system_parameters,
        "active_parameters": system_parameters,
        "small_eligible_by_parameter_count": True,
    }


def build_runtime_config(
    frozen_config: dict[str, object],
    *,
    input_path: Path,
    video_dir: Path,
    model_dir: Path,
    head_path: Path,
    starter_kit_dir: Path,
    input_contains_answers: bool,
) -> dict[str, object]:
    """Replace host-specific paths while preserving the frozen D4 policy."""
    config = copy.deepcopy(frozen_config)
    if dict(config["decision_head"])["feature_variant"] != "dialog_stage_fused":  # type: ignore[arg-type]
        raise ValueError("Submission adapter only supports the frozen D4 head")
    data = dict(config["data"])  # type: ignore[arg-type]
    data.update(
        {
            "input": str(input_path),
            "video_folder": str(video_dir),
            "split": "organizer_hidden_test",
            "input_sha256": sha256_file(input_path),
        }
    )
    config["data"] = data
    model = dict(config["model"])  # type: ignore[arg-type]
    model["default_local_path"] = str(model_dir)
    config["model"] = model
    decision_head = dict(config["decision_head"])  # type: ignore[arg-type]
    decision_head["path"] = str(head_path)
    config["decision_head"] = decision_head
    starter = dict(config["starter_kit"])  # type: ignore[arg-type]
    starter["path"] = str(starter_kit_dir)
    config["starter_kit"] = starter
    config["experiment_id"] = "d4_small_hidden_test_runtime"
    config["submission_runtime"] = {
        "adapter": "proactive_d4.submission",
        "input_contains_answers": input_contains_answers,
        "answers_removed_before_inference": True,
        "official_dialog_required": True,
        "future_video_allowed": False,
        "scorer_invoked": False,
    }
    return config


def publish_predictions(
    *,
    source_rows: Sequence[dict[str, object]],
    internal_predictions_path: Path,
    output_path: Path,
    receipt_path: Path,
    manifest: dict[str, object],
    frozen_config: dict[str, object],
    input_path: Path,
) -> dict[str, object]:
    """Validate and atomically publish only the official prediction fields."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {output_path}")
    internal_predictions = load_jsonl(internal_predictions_path)
    predictions = [
        {
            "video_path": prediction.get("video_path"),
            "answers": prediction.get("answers"),
        }
        for prediction in internal_predictions
    ]
    generation_rows = strip_answers(source_rows)
    validation = validate_prediction_rows(generation_rows, predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, predictions)
    parameter_audit = validate_bundle_manifest(manifest, frozen_config)
    receipt = {
        "status": "complete D4 hidden-input inference",
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output_path),
        "predictions_sha256": sha256_file(output_path),
        "validation": validation,
        "parameter_audit": parameter_audit,
        "answers_read_by_inference": False,
        "official_scorer_invoked": False,
    }
    write_json(receipt_path, receipt)
    return receipt


def _require_empty_work_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Submission work directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--work-dir", default="/tmp/wearable_ai_d4")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--head-path", default=str(DEFAULT_HEAD))
    parser.add_argument("--starter-kit-dir", default=str(DEFAULT_STARTER_KIT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    parser.add_argument(
        "--allow-input-answers-for-local-audit",
        action="store_true",
        help="Local public-val audit only; hidden-test invocation must omit this flag.",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.max_sessions is not None and args.max_sessions <= 0:
        parser.error("--max-sessions must be positive")
    if not args.preflight_only and not args.output_jsonl:
        parser.error("--output-jsonl is required unless --preflight-only is set")

    input_path = Path(args.input_jsonl).expanduser().resolve()
    video_dir = Path(args.video_dir).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    head_path = Path(args.head_path).expanduser().resolve()
    starter_kit_dir = Path(args.starter_kit_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    _require_empty_work_dir(work_dir)

    all_rows = load_jsonl(input_path)
    input_audit = validate_submission_input_rows(
        all_rows,
        allow_answers_for_local_audit=args.allow_input_answers_for_local_audit,
    )
    selected_rows = (
        all_rows[: args.max_sessions]
        if args.max_sessions is not None
        else all_rows
    )
    frozen_config = _load_object(config_path)
    manifest = _load_object(manifest_path)
    parameter_audit = validate_bundle_manifest(manifest, frozen_config)
    runtime_config = build_runtime_config(
        frozen_config,
        input_path=input_path,
        video_dir=video_dir,
        model_dir=model_dir,
        head_path=head_path,
        starter_kit_dir=starter_kit_dir,
        input_contains_answers=bool(input_audit["rows_with_answers"]),
    )
    runtime_config_path = work_dir / "runtime_config.json"
    write_json(runtime_config_path, runtime_config)

    internal_dir = work_dir / "run"
    runner_args = [
        "--config",
        str(runtime_config_path),
        "--model-path",
        str(model_dir),
        "--output-dir",
        str(internal_dir),
        "--device",
        args.device,
        "--skip-eval",
    ]
    if args.max_sessions is not None:
        runner_args.extend(["--max-sessions", str(args.max_sessions)])
    if args.require_exclusive_gpu:
        runner_args.append("--require-exclusive-gpu")
    if args.preflight_only:
        runner_args.append("--audit-only")
    run_deploy_main(runner_args)

    if args.preflight_only:
        receipt = {
            "status": "pass",
            "classification": "CPU-only D4 submission preflight",
            "input_audit": input_audit,
            "parameter_audit": parameter_audit,
            "runtime_config_sha256": sha256_file(runtime_config_path),
            "model_loaded": False,
            "gpu_used": False,
        }
        write_json(work_dir / "submission_preflight.json", receipt)
        print(json.dumps(receipt, sort_keys=True))
        return

    receipt = publish_predictions(
        source_rows=selected_rows,
        internal_predictions_path=internal_dir / "predictions.jsonl",
        output_path=Path(args.output_jsonl).expanduser().resolve(),
        receipt_path=work_dir / "submission_receipt.json",
        manifest=manifest,
        frozen_config=frozen_config,
        input_path=input_path,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
