"""Run the deterministic, no-GPU U0 audit and prepare blind review files."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import Counter
from pathlib import Path

from proactive_r0.artifacts import code_snapshot, sha256_file, write_json
from proactive_r0.core import load_jsonl, write_jsonl

from .core import (
    FALLBACK_ANSWER,
    build_chunk_records,
    build_review_package,
    finite_json,
    grouped_summary,
    ratings_rows,
    session_summary,
    summarize_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUBRIC = """# U0 双人盲评 Rubric

## 评审输入边界

- 只查看 `review_items_blind.jsonl`、对应视频截至 `observed_through_sec` 的部分，以及其中给出的 prior dialog。
- 不查看 `review_key.jsonl`、gold answer、未来视频、未来 dialog、D1 分数或 R0 raw response。
- A/B 两名评审独立完成，提交前不讨论；分歧在首轮评分完成后再仲裁。
- `model_action=spoke` 时评价决策和内容；`model_action=silent` 时只评价是否应该打断、决策置信度和时机，其余内容项留空。

## 字段

- `should_interrupt`: `yes` / `no` / `uncertain`，表示此刻是否应主动说话。
- `decision_confidence_1_5`: 对上述判断的置信度，1 最低、5 最高。
- `timeliness_1_5`: 此刻说话或保持安静是否及时、不过早也不过晚。
- `correctness_1_5`: 指导动作与对象是否正确。
- `specificity_1_5`: 是否包含足够具体的动作、对象、方向或错误信息。
- `actionability_1_5`: 用户能否直接照做。
- `groundedness_1_5`: 是否只依赖当前可见及历史证据，没有编造不可见事实。
- `plan_consistency_1_5`: 是否符合当前步骤、合理下一步或恢复动作。
- `conciseness_1_5`: 是否简洁且没有丢失必要信息。
- `safety_1_5`: 5 表示安全，1 表示可能直接造成危险或严重错误。
- `generic_flag`: `yes` / `no`，内容是否基本可替换到任意任务而不改变含义。
- `hallucination_flag`: `yes` / `no`，是否声称了当前证据不支持的对象、状态或完成情况。
- `unsafe_flag`: `yes` / `no`。
- `primary_error_type`: `none` / `wrong_timing` / `wrong_action` / `wrong_object` / `premature` / `stale` / `generic` / `hallucination` / `unsafe` / `other`。

## 评分锚点

- 5：完全满足该维度，几乎不需要修改。
- 4：主要正确，只有轻微措辞或信息缺失。
- 3：部分有用，但存在明显遗漏、泛化或不确定性。
- 2：大部分不合适，可能使用户困惑。
- 1：错误、无用、无依据，或在 safety 项中存在明显风险。

文本相似度和是否与 gold 逐字一致不属于评分标准。可接受多种正确表述。
"""

README_TEMPLATE = """# D1 Utterance U0 Audit

本实验是只读、无 GPU 的内容审计。它不训练模型、不改变 D1 决策、不产生排行榜候选。

主要产物：

- `audit.json`: 全量 700 sessions / 9,935 chunks 自动统计；
- `chunk_records.jsonl`: 对齐后的逐 chunk 审计记录；
- `review_items_blind.jsonl`: 200 条盲评输入，不含当前 gold 或内部来源字段；
- `review_key.jsonl`: 与盲评 ID 对齐的答案键，评审提交前不得查看；
- `ratings_template.csv`: A/B 两名独立评审的空白评分表；
- `review_rubric.md`: 中文评分协议。

自动 generic/action/content-token 字段是保守词表诊断，不是语义质量结论。正式内容结论必须来自盲评。
"""


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _check_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty ratings template")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_official_counts(
    audit_overall: dict[str, object], official: dict[str, object]
) -> None:
    mapping = {
        "chunks": "support",
        "tp": "tp",
        "fp": "fp",
        "tn": "tn",
        "fn": "fn",
    }
    for audit_key, official_key in mapping.items():
        if int(audit_overall[audit_key]) != int(official[official_key]):
            raise ValueError(
                f"U0 count differs from official scorer: {audit_key} "
                f"{audit_overall[audit_key]} != {official[official_key]}"
            )


def _artifact_manifest(output_dir: Path, names: list[str]) -> dict[str, object]:
    return {
        "files": {
            name: {
                "bytes": (output_dir / name).stat().st_size,
                "sha256": sha256_file(output_dir / name),
            }
            for name in names
        }
    }


def run(config_path: Path, output_dir: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = config["sources"]
    paths = {name: _resolve(str(value["path"])) for name, value in sources.items()}
    hashes = {
        name: _check_hash(paths[name], str(value["sha256"]))
        for name, value in sources.items()
    }

    source_rows = load_jsonl(paths["gold"])
    prediction_rows = load_jsonl(paths["d1_predictions"])
    r0_rows = load_jsonl(paths["r0_session_records"])
    oof_rows = load_jsonl(paths["d1_oof_records"])
    official_summary = json.loads(
        paths["d1_metrics_summary"].read_text(encoding="utf-8")
    )
    official_overall = official_summary["overall"]

    records = build_chunk_records(
        source_rows, prediction_rows, r0_rows, oof_rows
    )
    strata = {
        str(name): int(count)
        for name, count in config["review_sampling"]["strata"].items()
    }
    blind_rows, key_rows, review_summary = build_review_package(
        records,
        source_rows,
        strata,
        str(config["review_sampling"]["seed"]),
    )

    overall = summarize_records(records)
    _validate_official_counts(overall, official_overall)
    fallback_origin = Counter(
        str(record["raw_response_class"])
        for record in records
        if record["is_fallback"]
    )
    audit: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "automatic audit complete; human blind ratings pending",
        "scope": {
            "sessions": len(source_rows),
            "chunks": len(records),
            "model_inference_run": False,
            "model_training_run": False,
            "d1_decisions_changed": False,
            "official_metric_recomputed": False,
            "content_ranked_by_c1": False,
        },
        "official_d1_metrics_frozen": official_overall,
        "overall": overall,
        "sessions": session_summary(records),
        "fallback_origin_by_raw_response_class": dict(sorted(fallback_origin.items())),
        "by_domain": grouped_summary(records, "domain"),
        "by_position": grouped_summary(records, "position_bin"),
        "by_confusion": grouped_summary(records, "confusion"),
        "by_raw_response_class": grouped_summary(records, "raw_response_class"),
        "by_task": grouped_summary(records, "task"),
        "review_sample": review_summary,
        "heuristic_warning": (
            "generic_only/action_verb/nonstop_content_token are conservative lexical "
            "diagnostics, not semantic correctness or object-grounding metrics."
        ),
    }
    finite_json(audit)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "audit.json", audit)
    write_jsonl(output_dir / "chunk_records.jsonl", records)
    write_jsonl(output_dir / "review_items_blind.jsonl", blind_rows)
    write_jsonl(output_dir / "review_key.jsonl", key_rows)
    _write_csv(output_dir / "ratings_template.csv", ratings_rows(blind_rows))
    _write_text(output_dir / "review_rubric.md", RUBRIC)
    _write_text(output_dir / "README.md", README_TEMPLATE)

    data_manifest = {
        "experiment_id": config["experiment_id"],
        "sources": {
            name: {
                "path": str(paths[name]),
                "sha256": hashes[name],
                "license": value.get("license", "project artifact"),
                "role": value["role"],
            }
            for name, value in sources.items()
        },
        "external_data_used": False,
        "public_validation_labels_read": True,
        "classification": "val-supervised diagnostic audit; not hidden-test evidence",
        "current_or_future_gold_exposed_to_review_input": False,
        "future_video_or_dialog_exposed_to_review_input": False,
    }
    write_json(output_dir / "data_manifest.json", data_manifest)

    tracked = [
        config_path,
        PROJECT_ROOT / "src/proactive_u0/__init__.py",
        PROJECT_ROOT / "src/proactive_u0/core.py",
        PROJECT_ROOT / "src/proactive_u0/run.py",
        PROJECT_ROOT / "src/proactive_u0/tests/test_core.py",
    ]
    write_json(output_dir / "code_state.txt", code_snapshot(PROJECT_ROOT, tracked))
    _write_text(
        output_dir / "environment.txt",
        "\n".join(
            [
                f"python={sys.version.replace(chr(10), ' ')}",
                f"executable={sys.executable}",
                f"platform={platform.platform()}",
                "gpu_used=false",
                "external_service_used=false",
                "",
            ]
        ),
    )
    command = (
        "PYTHONNOUSERSITE=1 PYTHONPATH=src "
        "/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python "
        "-m proactive_u0.run "
        "--config configs/u0_d1_utterance_audit.json "
        f"--output-dir {output_dir.relative_to(PROJECT_ROOT)}\n"
    )
    _write_text(output_dir / "command.sh", command)
    (output_dir / "command.sh").chmod(0o755)
    _write_text(
        output_dir / "run.log",
        "\n".join(
            [
                "U0 automatic audit completed.",
                f"sessions={len(source_rows)}",
                f"chunks={len(records)}",
                f"fallback_answer={FALLBACK_ANSWER}",
                f"review_items={len(blind_rows)}",
                "human_ratings=pending",
                "",
            ]
        ),
    )

    manifest_names = [
        "README.md",
        "audit.json",
        "chunk_records.jsonl",
        "code_state.txt",
        "command.sh",
        "config.json",
        "data_manifest.json",
        "environment.txt",
        "ratings_template.csv",
        "review_items_blind.jsonl",
        "review_key.jsonl",
        "review_rubric.md",
        "run.log",
    ]
    write_json(
        output_dir / "artifact_manifest.json",
        _artifact_manifest(output_dir, manifest_names),
    )
    return {
        "experiment_id": config["experiment_id"],
        "output_dir": str(output_dir),
        "overall": overall,
        "sessions": audit["sessions"],
        "review_sample": review_summary,
        "artifact_manifest_sha256": sha256_file(
            output_dir / "artifact_manifest.json"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(_resolve(args.config), _resolve(args.output_dir))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
