"""Verify D4 online dialog state against the frozen full-fit matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from proactive_d1.core import (
    attach_gold_labels,
    build_label_free_chunks,
    feature_names,
    load_decision_head,
    predict_feature_values,
    strip_answers,
    validate_fold_manifest,
)
from proactive_d1.internvl_features import NeuralDecisionFeatures
from proactive_d1.neural_core import load_aligned_neural_cache
from proactive_d3.dialog_control_core import (
    DIALOG_POLICY_NAMES,
    build_dialog_policy_features,
)
from proactive_d4.deploy import (
    OnlineDialogPolicyState,
    dialog_stage_feature_values,
)
from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGIT_ABS_TOLERANCE = 1e-6


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def verify(final_dir: Path) -> dict[str, object]:
    config = json.loads((final_dir / "config.json").read_text(encoding="utf-8"))
    source_path = _resolve(config["data"]["input"])
    sources = config["sources"]
    source_rows = load_jsonl(source_path)
    label_free_rows = strip_answers(source_rows)
    folds = validate_fold_manifest(
        json.loads(_resolve(sources["split_manifest"]["path"]).read_text(encoding="utf-8")),
        label_free_rows,
    )
    label_free = build_label_free_chunks(
        label_free_rows,
        load_jsonl(_resolve(sources["r0_session_records"]["path"])),
        folds,
        max_history_turns=int(config["features"]["max_history_turns"]),
        max_model_frames=int(config["features"]["max_model_frames"]),
    )
    offline_dialog, _ = build_dialog_policy_features(label_free_rows, label_free)
    examples = attach_gold_labels(label_free, source_rows)
    cache_path = _resolve(sources["neural_features"]["path"])
    cache = load_aligned_neural_cache(
        cache_path,
        examples,
        hidden_size=int(config["features"]["hidden_size"]),
    )
    head_path = final_dir / "decision_head.json"
    head = load_decision_head(json.loads(head_path.read_text(encoding="utf-8")))
    final_records = {
        (int(row["input_index"]), int(row["chunk_index"])): row
        for row in load_jsonl(final_dir / "train_fit_records.jsonl")
    }
    scalar_names = feature_names(
        config["features"]["scalar_variant"],
        sorted({example.feature.domain for example in examples}),
    )
    current_input = -1
    state: OnlineDialogPolicyState | None = None
    max_dialog_difference = 0.0
    max_logit_difference = 0.0
    decision_matches = 0
    for index, example in enumerate(examples):
        if example.feature.input_index != current_input:
            current_input = example.feature.input_index
            state = OnlineDialogPolicyState()
        assert state is not None
        dialog = label_free_rows[example.feature.input_index]["dialog"]
        online_dialog = state.consume(dialog[example.feature.chunk_index], example.feature.chunk_index)
        max_dialog_difference = max(
            max_dialog_difference,
            float(np.max(np.abs(online_dialog - offline_dialog[index]))),
        )
        neural = NeuralDecisionFeatures(
            hidden_state=cache.hidden_state[index],
            silent_log_probability=float(cache.silent_log_probability[index]),
            interrupt_log_probability=float(cache.interrupt_log_probability[index]),
            tag_margin=float(cache.tag_margin[index]),
            prompt_tokens=int(cache.prompt_tokens[index]),
            hidden_max_abs_difference=0.0,
            hidden_cosine_similarity=1.0,
        )
        scalar_values = {
            name: example.feature.values[name] for name in scalar_names
        }
        values = dialog_stage_feature_values(
            scalar_values, neural, online_dialog, head
        )
        decision, logit = predict_feature_values(head, values)
        final = final_records[example.key]
        max_logit_difference = max(
            max_logit_difference, abs(logit - float(final["logit"]))
        )
        decision_matches += int(decision == int(final["predicted_interrupt"]))
    rows = len(examples)
    passed = (
        max_dialog_difference == 0.0
        and max_logit_difference <= LOGIT_ABS_TOLERANCE
        and decision_matches == rows
    )
    return {
        "status": "pass" if passed else "fail",
        "sessions": len(source_rows),
        "chunks": rows,
        "dialog_policy_names": list(DIALOG_POLICY_NAMES),
        "max_dialog_feature_abs_difference": max_dialog_difference,
        "max_logit_abs_difference": max_logit_difference,
        "logit_abs_tolerance": LOGIT_ABS_TOLERANCE,
        "decision_exact_matches": decision_matches,
        "artifacts": {
            "head_sha256": sha256_file(head_path),
            "final_records_sha256": sha256_file(final_dir / "train_fit_records.jsonl"),
            "cache_sha256": sha256_file(cache_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", required=True)
    args = parser.parse_args()
    final_dir = _resolve(args.final_dir)
    result = verify(final_dir)
    write_json(final_dir / "online_state_audit.json", result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit("D4 online-state verification failed")


if __name__ == "__main__":
    main()
