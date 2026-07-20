"""Stream the frozen cache through the online D3 state and final head."""

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
from proactive_d3.core import DYNAMIC_SCALAR_NAMES, build_causal_dynamics
from proactive_d3.deploy import OnlineCausalDynamicsState, dynamics_feature_values
from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGIT_ABS_TOLERANCE = 1e-6


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def verify(final_dir: Path, cache_path: Path) -> dict[str, object]:
    config = json.loads((final_dir / "config.json").read_text(encoding="utf-8"))
    source_path = Path(config["runtime"]["input_path"])
    cache_config = config["neural_cache"]
    features_config = config["features"]
    r0_dir = _resolve(config["r0_reference"]["experiment_dir"])
    split_dir = _resolve(config["split_reference"]["experiment_dir"])
    source = load_jsonl(source_path)
    r0_records = load_jsonl(r0_dir / "session_records.jsonl")
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    folds = validate_fold_manifest(manifest, strip_answers(source))
    label_free = build_label_free_chunks(
        strip_answers(source),
        r0_records,
        folds,
        max_history_turns=int(features_config["max_history_turns"]),
        max_model_frames=int(features_config["max_model_frames"]),
    )
    examples = attach_gold_labels(label_free, source)
    cache = load_aligned_neural_cache(
        cache_path, examples, hidden_size=int(cache_config["hidden_size"])
    )
    offline = build_causal_dynamics(cache)
    head = load_decision_head(
        json.loads((final_dir / "decision_head.json").read_text(encoding="utf-8"))
    )
    final_records = {
        (int(row["input_index"]), int(row["chunk_index"])): row
        for row in load_jsonl(final_dir / "train_fit_records.jsonl")
    }
    domains = sorted({example.feature.domain for example in examples})
    scalar_names = feature_names(features_config["scalar_variant"], domains)
    state: OnlineCausalDynamicsState | None = None
    current_input = -1
    max_scalar_difference = 0.0
    max_delta_difference = 0.0
    max_logit_difference = 0.0
    decision_matches = 0
    for index, example in enumerate(examples):
        if example.feature.input_index != current_input:
            current_input = example.feature.input_index
            state = OnlineCausalDynamicsState(hidden_size=int(cache_config["hidden_size"]))
        assert state is not None
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
        values, dynamics = dynamics_feature_values(
            scalar_values, neural, state, head
        )
        max_scalar_difference = max(
            max_scalar_difference,
            float(np.max(np.abs(dynamics.scalar - offline.scalar[index]))),
        )
        max_delta_difference = max(
            max_delta_difference,
            float(
                np.max(
                    np.abs(dynamics.hidden_delta - offline.hidden_delta[index])
                )
            ),
        )
        decision, logit = predict_feature_values(head, values)
        final = final_records[example.key]
        max_logit_difference = max(
            max_logit_difference, abs(logit - float(final["logit"]))
        )
        decision_matches += int(decision == int(final["predicted_interrupt"]))
    rows = len(examples)
    passed = (
        max_scalar_difference == 0.0
        and max_delta_difference == 0.0
        and max_logit_difference <= LOGIT_ABS_TOLERANCE
        and decision_matches == rows
    )
    return {
        "status": "pass" if passed else "fail",
        "sessions": len(set(cache.input_index.tolist())),
        "chunks": rows,
        "dynamic_scalar_names": list(DYNAMIC_SCALAR_NAMES),
        "max_dynamic_scalar_abs_difference": max_scalar_difference,
        "max_hidden_delta_abs_difference": max_delta_difference,
        "max_logit_abs_difference": max_logit_difference,
        "logit_abs_tolerance": LOGIT_ABS_TOLERANCE,
        "logit_tolerance_reason": (
            "offline training matrices store scalar features as float32, while "
            "online causal scalars retain Python float precision"
        ),
        "decision_exact_matches": decision_matches,
        "artifacts": {
            "head_sha256": sha256_file(final_dir / "decision_head.json"),
            "final_records_sha256": sha256_file(final_dir / "train_fit_records.jsonl"),
            "cache_sha256": sha256_file(cache_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", required=True)
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    final_dir = _resolve(args.final_dir)
    result = verify(final_dir, _resolve(args.cache))
    write_json(final_dir / "online_state_audit.json", result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit("D3 online-state verification failed")


if __name__ == "__main__":
    main()
