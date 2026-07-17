#!/usr/bin/env python3
"""Build the presentation notebook and dashboard data from frozen artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = PRESENTATION_DIR / "results_dashboard"
DATASET_PATH = PROJECT_ROOT / "data/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl"
NOTEBOOK_SOURCE = PRESENTATION_DIR / "PWR_progress_report_15min_source.md"
NOTEBOOK_OUTPUT = PRESENTATION_DIR / "PWR_progress_report_15min.ipynb"


EXPERIMENTS = [
    {
        "id": "r0",
        "label": "R0 零样本基线",
        "short_label": "R0",
        "stage": "progression",
        "status": "历史基线",
        "supervision": "zero-shot public validation",
        "metrics": "output/experiments/20260713_internvl35_1b_no_plan_r0/metrics_summary.json",
        "predictions": "output/experiments/20260713_internvl35_1b_no_plan_r0/predictions.jsonl",
        "note": "冻结 InternVL3.5-1B，不使用计划状态。",
    },
    {
        "id": "r0f",
        "label": "R0-F 格式修复",
        "short_label": "R0-F",
        "stage": "progression",
        "status": "历史对照",
        "supervision": "val-supervised rule selection",
        "metrics": "output/experiments/20260714_internvl35_1b_response_intent_repair_r0f_valsupervised/metrics_summary.json",
        "predictions": "output/experiments/20260714_internvl35_1b_response_intent_repair_r0f_valsupervised/predictions.jsonl",
        "note": "仅重解释冻结 raw response，不重跑模型。",
    },
    {
        "id": "d1_scalar",
        "label": "D1 严格因果标量头",
        "short_label": "D1 scalar",
        "stage": "progression",
        "status": "已完成",
        "supervision": "five-fold session OOF, val-supervised",
        "metrics": "output/experiments/20260714_internvl35_1b_causal_scalar_decision_head_d1_oof_v2/variants/response_temporal/metrics_summary.json",
        "predictions": "output/experiments/20260714_internvl35_1b_causal_scalar_decision_head_d1_oof_v2/variants/response_temporal/predictions.jsonl",
        "note": "18 个当前时刻可得的时间、领域和 R0 响应属性。",
    },
    {
        "id": "d1_tag",
        "label": "D1 tag margin only",
        "short_label": "tag only",
        "stage": "ablation",
        "status": "消融",
        "supervision": "five-fold session OOF, val-supervised",
        "metrics": "output/experiments/20260714_internvl35_1b_neural_decision_head_d1_oof_v1/variants/tag_only/metrics_summary.json",
        "predictions": "output/experiments/20260714_internvl35_1b_neural_decision_head_d1_oof_v1/variants/tag_only/predictions.jsonl",
        "note": "只使用 log P(interrupt) - log P(silent)。",
    },
    {
        "id": "d1_hidden",
        "label": "D1 hidden only",
        "short_label": "hidden only",
        "stage": "ablation",
        "status": "消融",
        "supervision": "five-fold session OOF, val-supervised",
        "metrics": "output/experiments/20260714_internvl35_1b_neural_decision_head_d1_oof_v1/variants/hidden_linear/metrics_summary.json",
        "predictions": "output/experiments/20260714_internvl35_1b_neural_decision_head_d1_oof_v1/variants/hidden_linear/predictions.jsonl",
        "note": "只使用 1,024 维冻结因果多模态 hidden。",
    },
    {
        "id": "d1_scalar_tag",
        "label": "D1 scalar + tag",
        "short_label": "scalar + tag",
        "stage": "ablation",
        "status": "消融",
        "supervision": "five-fold session OOF, val-supervised",
        "metrics": "output/experiments/20260714_internvl35_1b_neural_decision_head_d1_oof_v1/variants/scalar_tag/metrics_summary.json",
        "predictions": "output/experiments/20260714_internvl35_1b_neural_decision_head_d1_oof_v1/variants/scalar_tag/predictions.jsonl",
        "note": "18 个标量特征加 1 个固定标签 margin。",
    },
    {
        "id": "d1_fused",
        "label": "D1 融合线性头",
        "short_label": "D1 fused",
        "stage": "progression",
        "status": "当前科学基线",
        "supervision": "five-fold session OOF, val-supervised",
        "metrics": "output/experiments/20260714_internvl35_1b_neural_decision_head_d1_oof_v1/variants/fused_linear/metrics_summary.json",
        "predictions": "output/experiments/20260714_internvl35_1b_neural_decision_head_d1_oof_v1/variants/fused_linear/predictions.jsonl",
        "note": "18 scalar + 1 tag margin + 1,024 hidden；每折 1,044 参数。",
    },
    {
        "id": "d1_single_threshold",
        "label": "D1 单一部署阈值模拟",
        "short_label": "D1 deploy threshold",
        "stage": "progression",
        "status": "部署审计通过",
        "supervision": "OOF threshold transport audit, val-supervised",
        "metrics": "output/experiments/20260715_internvl35_1b_d1_threshold_robustness_v1/metrics_summary.json",
        "predictions": "output/experiments/20260715_internvl35_1b_d1_threshold_robustness_v1/predictions.jsonl",
        "note": "五个 OOF 头统一使用已固化的中位阈值 0.125605。",
    },
    {
        "id": "d2_mlp",
        "label": "D2 width-8 残差 MLP",
        "short_label": "D2 MLP",
        "stage": "progression",
        "status": "已否决",
        "supervision": "five-fold session OOF, val-supervised",
        "metrics": "output/experiments/20260715_internvl35_1b_residual_mlp_d2_oof_v1/metrics_summary.json",
        "predictions": "output/experiments/20260715_internvl35_1b_residual_mlp_d2_oof_v1/predictions.jsonl",
        "note": "在 D1 logit 上增加 8,361 参数非线性残差，增益未通过门槛。",
    },
]


SAMPLE_EXPERIMENT_IDS = {
    "r0",
    "r0f",
    "d1_scalar",
    "d1_fused",
    "d1_single_threshold",
    "d2_mlp",
}


R1_EXPERIMENTS = [
    ("r0_frozen", "冻结 R0", "r0_frozen"),
    ("null", "R1 null wrapper", "null"),
    ("step", "current step", "step"),
    ("cues", "step + cues", "cues"),
    ("full", "full compact state", "full"),
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_answer(answer: str) -> dict[str, str]:
    stripped = str(answer).lstrip()
    if stripped.startswith("$interrupt$"):
        return {
            "decision": "interrupt",
            "utterance": stripped[len("$interrupt$") :].strip(),
            "raw": str(answer),
        }
    return {"decision": "silent", "utterance": "", "raw": str(answer)}


def confusion(gold: Iterable[str], predicted: Iterable[str]) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for gold_tag, pred_tag in zip(gold, predicted):
        if gold_tag == "interrupt" and pred_tag == "interrupt":
            counts["tp"] += 1
        elif gold_tag == "silent" and pred_tag == "interrupt":
            counts["fp"] += 1
        elif gold_tag == "silent" and pred_tag == "silent":
            counts["tn"] += 1
        else:
            counts["fn"] += 1
    return counts


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    int_precision = safe_div(tp, tp + fp)
    int_recall = safe_div(tp, tp + fn)
    int_f1 = safe_div(2 * int_precision * int_recall, int_precision + int_recall)
    silent_precision = safe_div(tn, tn + fn)
    silent_recall = safe_div(tn, tn + fp)
    silent_f1 = safe_div(2 * silent_precision * silent_recall, silent_precision + silent_recall)
    support = tp + fp + tn + fn
    return {
        **counts,
        "support": support,
        "macro_f1": (int_f1 + silent_f1) / 2,
        "interrupt_precision": int_precision,
        "interrupt_recall": int_recall,
        "interrupt_f1": int_f1,
        "silent_precision": silent_precision,
        "silent_recall": silent_recall,
        "silent_f1": silent_f1,
        "predicted_interrupt_rate": safe_div(tp + fp, support),
    }


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "macro_f1",
        "gmean_f1",
        "interrupt_precision",
        "interrupt_recall",
        "interrupt_f1",
        "silent_precision",
        "silent_recall",
        "silent_f1",
        "tp",
        "fp",
        "tn",
        "fn",
        "support",
    ]
    result = {key: metrics[key] for key in keys if key in metrics}
    support = result.get("support", 0)
    if support:
        result["predicted_interrupt_rate"] = (result.get("tp", 0) + result.get("fp", 0)) / support
    return result


def load_experiments(dataset_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    expected_paths = [row["video_path"] for row in dataset_rows]
    experiment_data: list[dict[str, Any]] = []
    prediction_rows: dict[str, list[dict[str, Any]]] = {}
    for definition in EXPERIMENTS:
        metrics_path = PROJECT_ROOT / definition["metrics"]
        predictions_path = PROJECT_ROOT / definition["predictions"]
        summary = read_json(metrics_path)
        predictions = read_jsonl(predictions_path)
        if len(predictions) != len(dataset_rows):
            raise ValueError(f"{definition['id']} has {len(predictions)} rows, expected {len(dataset_rows)}")
        predicted_paths = [row["video_path"] for row in predictions]
        if predicted_paths != expected_paths:
            raise ValueError(f"{definition['id']} prediction order does not match the source dataset")
        for source, prediction in zip(dataset_rows, predictions):
            expected_chunks = len(source["video_intervals"])
            if len(prediction["answers"]) != expected_chunks:
                raise ValueError(
                    f"{definition['id']} {source['video_path']} has {len(prediction['answers'])} answers, "
                    f"expected {expected_chunks}"
                )
        gold_tags = [
            parse_answer(answer)["decision"] for source in dataset_rows for answer in source["answers"]
        ]
        predicted_tags = [
            parse_answer(answer)["decision"] for prediction in predictions for answer in prediction["answers"]
        ]
        recomputed = metrics_from_counts(confusion(gold_tags, predicted_tags))
        reported = summary["overall"]
        for key in ("tp", "fp", "tn", "fn", "support"):
            if recomputed[key] != reported[key]:
                raise ValueError(
                    f"{definition['id']} reported {key}={reported[key]}, recomputed {recomputed[key]}"
                )
        if abs(float(recomputed["macro_f1"]) - float(reported["macro_f1"])) > 0.00006:
            raise ValueError(
                f"{definition['id']} reported Macro F1 {reported['macro_f1']}, "
                f"recomputed {recomputed['macro_f1']}"
            )
        prediction_rows[definition["id"]] = predictions
        experiment_data.append(
            {
                **{key: value for key, value in definition.items() if key not in {"metrics", "predictions"}},
                "metrics": compact_metrics(summary["overall"]),
                "source": {
                    "metrics": definition["metrics"],
                    "predictions": definition["predictions"],
                    "metrics_sha256": sha256(metrics_path),
                    "predictions_sha256": sha256(predictions_path),
                },
            }
        )
    return experiment_data, prediction_rows


def build_domain_metrics(
    dataset_rows: list[dict[str, Any]], prediction_rows: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for experiment_id, predictions in prediction_rows.items():
        grouped_gold: dict[str, list[str]] = defaultdict(list)
        grouped_pred: dict[str, list[str]] = defaultdict(list)
        for source, prediction in zip(dataset_rows, predictions):
            domain = source["domain"]
            grouped_gold[domain].extend(parse_answer(answer)["decision"] for answer in source["answers"])
            grouped_pred[domain].extend(parse_answer(answer)["decision"] for answer in prediction["answers"])
        result[experiment_id] = {}
        for domain in sorted(grouped_gold):
            counts = confusion(grouped_gold[domain], grouped_pred[domain])
            result[experiment_id][domain] = metrics_from_counts(counts)
    return result


def build_position_metrics(
    dataset_rows: list[dict[str, Any]], prediction_rows: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, dict[str, Any]]]:
    bins = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for experiment_id in ("r0", "r0f", "d1_scalar", "d1_fused"):
        grouped_gold: dict[str, list[str]] = defaultdict(list)
        grouped_pred: dict[str, list[str]] = defaultdict(list)
        for source, prediction in zip(dataset_rows, prediction_rows[experiment_id]):
            total = len(source["answers"])
            for index, (gold_answer, pred_answer) in enumerate(zip(source["answers"], prediction["answers"])):
                bin_index = min(4, int(index * 5 / total))
                key = bins[bin_index]
                grouped_gold[key].append(parse_answer(gold_answer)["decision"])
                grouped_pred[key].append(parse_answer(pred_answer)["decision"])
        result[experiment_id] = {
            key: metrics_from_counts(confusion(grouped_gold[key], grouped_pred[key])) for key in bins
        }
    return result


def session_pattern_counts(session: dict[str, Any]) -> dict[str, int]:
    gold = [item["decision"] for item in session["gold"]]
    r0f = [item["decision"] for item in session["predictions"]["r0f"]]
    fused = [item["decision"] for item in session["predictions"]["d1_fused"]]
    return {
        "recovered_fn": sum(g == "interrupt" and a == "silent" and b == "interrupt" for g, a, b in zip(gold, r0f, fused)),
        "recovered_fp": sum(g == "silent" and a == "interrupt" and b == "silent" for g, a, b in zip(gold, r0f, fused)),
        "remaining_fn": sum(g == "interrupt" and b == "silent" for g, b in zip(gold, fused)),
        "remaining_fp": sum(g == "silent" and b == "interrupt" for g, b in zip(gold, fused)),
        "correct": sum(g == b for g, b in zip(gold, fused)),
    }


def choose_featured_sessions(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = [
        ("recovered_fn", "修复漏报", "D1 将 R0-F 的 silent 漏报改为正确 interrupt"),
        ("recovered_fp", "修复误报", "D1 将 R0-F 的多余 interrupt 改为正确 silent"),
        ("remaining_fn", "剩余漏报", "D1 仍未触发部分应当 interrupt 的 chunk"),
        ("remaining_fp", "剩余误报", "D1 仍在部分 gold silent chunk 触发 interrupt"),
    ]
    used: set[int] = set()
    featured: list[dict[str, Any]] = []
    for key, label, description in definitions:
        candidates = sorted(
            (
                (session["pattern_counts"][key], session["pattern_counts"]["correct"], index, session)
                for index, session in enumerate(sessions)
                if index not in used
            ),
            key=lambda item: (item[0], item[1], len(item[3]["intervals"])),
            reverse=True,
        )
        score, _, index, session = candidates[0]
        used.add(index)
        featured.append(
            {
                "session_index": index,
                "label": label,
                "description": description,
                "count": score,
                "domain": session["domain"],
                "task": session["task"],
            }
        )
    return featured


def build_sessions(
    dataset_rows: list[dict[str, Any]], prediction_rows: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    for index, source in enumerate(dataset_rows):
        predictions: dict[str, Any] = {}
        for experiment_id in sorted(SAMPLE_EXPERIMENT_IDS):
            predictions[experiment_id] = [
                parse_answer(answer) for answer in prediction_rows[experiment_id][index]["answers"]
            ]
        session = {
            "index": index,
            "video_path": source["video_path"],
            "video_url": f"/media/{source['video_path']}",
            "duration": source["duration_in_sec"],
            "intervals": source["video_intervals"],
            "query": source["query"],
            "domain": source["domain"],
            "task": source["task"],
            "gold": [parse_answer(answer) for answer in source["answers"]],
            "predictions": predictions,
        }
        session["pattern_counts"] = session_pattern_counts(session)
        sessions.append(session)
    return sessions


def load_r1_study() -> dict[str, Any]:
    root = PROJECT_ROOT / "output/experiments/20260714_internvl35_1b_oracle_state_r1_pilot_v1/variants"
    variants = []
    for experiment_id, label, directory in R1_EXPERIMENTS:
        summary = read_json(root / directory / "metrics_summary.json")
        variants.append({"id": experiment_id, "label": label, "metrics": compact_metrics(summary["overall"])})
    return {
        "title": "R1 oracle compact-state protocol pilot",
        "support": 50,
        "sessions": 4,
        "scope": "label-independent domain-stratified subset; protocol pilot only",
        "variants": variants,
        "source": "output/experiments/20260714_internvl35_1b_oracle_state_r1_pilot_v1/",
    }


def inference_study() -> dict[str, Any]:
    comparison_paths = {
        "batched": "output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_batched_v1_smoke1/comparison.json",
        "prefix_cache": "output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_prefix_cache_v1_smoke1/comparison.json",
        "shared_vision_smoke": "output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_shared_vision_v1_smoke1/comparison.json",
        "shared_vision_expanded": "output/experiments/20260715_internvl35_1b_neural_decision_head_d1_deploy_shared_vision_v1_benchmark8/comparison.json",
    }
    comparisons = {key: read_json(PROJECT_ROOT / path) for key, path in comparison_paths.items()}
    batched = comparisons["batched"]
    prefix = comparisons["prefix_cache"]
    shared_smoke = comparisons["shared_vision_smoke"]
    expanded_source = comparisons["shared_vision_expanded"]
    reference = {
        "id": "sequential",
        "label": "sequential",
        "session_compute_s": batched["latency"]["reference_compute_seconds"],
        "wall_s": batched["latency"]["reference_wall_seconds"],
        "peak_bytes": batched["memory"]["reference_peak_bytes"],
        "equivalent": True,
        "decision_equal": True,
        "status": "正确性参照",
    }

    def smoke_entry(
        comparison: dict[str, Any], entry_id: str, label: str, status: str
    ) -> dict[str, Any]:
        audit = comparison["equivalence"]["consistency_audit"]
        entry = {
            "id": entry_id,
            "label": label,
            "session_compute_s": comparison["latency"]["candidate_compute_seconds"],
            "wall_s": comparison["latency"]["candidate_wall_seconds"],
            "peak_bytes": comparison["memory"]["candidate_peak_bytes"],
            "equivalent": comparison["equivalence"]["passed"],
            "decision_equal": audit["decision_exact_matches"] == comparison["chunks"],
            "status": status,
        }
        if audit["max_tag_margin_abs_difference"]:
            entry["max_margin_diff"] = audit["max_tag_margin_abs_difference"]
        return entry

    expanded_audit = expanded_source["equivalence"]["consistency_audit"]
    expanded_latency = expanded_source["latency"]
    expanded_memory = expanded_source["memory"]
    return {
        "title": "D1 等价推理优化",
        "smoke_chunks": 10,
        "expanded_chunks": expanded_source["chunks"],
        "source": "reports/20260715_internvl35_1b_d1_inference_optimization.md",
        "smoke": [
            reference,
            smoke_entry(batched, "batched", "batch-of-two", "否决：更慢且显存更高"),
            smoke_entry(prefix, "prefix_cache", "cropped prefix cache", "否决：特征漂移且无加速"),
            smoke_entry(shared_smoke, "shared_vision_smoke", "shared vision", "进入扩展验证"),
        ],
        "expanded": {
            "reference_wall_s": expanded_latency["reference_wall_seconds"],
            "shared_wall_s": expanded_latency["candidate_wall_seconds"],
            "wall_improvement": expanded_latency["wall_improvement_fraction"],
            "session_compute_improvement": expanded_latency["compute_improvement_fraction"],
            "reference_peak_bytes": expanded_memory["reference_peak_bytes"],
            "shared_peak_bytes": expanded_memory["candidate_peak_bytes"],
            "raw_equal": f"{expanded_audit['raw_response_exact_matches']}/{expanded_source['chunks']}",
            "hidden_equal": f"{expanded_audit['hidden_exact_matches']}/{expanded_source['chunks']}",
            "margin_equal": f"{expanded_audit['tag_margin_exact_matches']}/{expanded_source['chunks']}",
            "decision_equal": f"{expanded_audit['decision_exact_matches']}/{expanded_source['chunks']}",
            "answer_equal": f"{expanded_audit['answer_exact_matches']}/{expanded_source['chunks']}",
            "status": expanded_source["status"],
        },
        "comparison_sources": [
            {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for path in comparison_paths.values()
        ],
    }


def literature_data() -> list[dict[str, Any]]:
    return [
        {
            "id": "mmduet2",
            "title": "MMDuet2",
            "date": "2025-12-07",
            "arxiv": "https://arxiv.org/abs/2512.06810",
            "focus": "多轮主动回答时机与内容",
            "method": "文本式 NO REPLY / response；SFT + PAUC-style GRPO",
            "scale": "Qwen2.5-VL-3B；52K videos",
            "project_use": "及时性奖励、重复回答约束和多轮轨迹设计参考",
        },
        {
            "id": "streampro",
            "title": "StreamPro",
            "date": "2026-05-11",
            "arxiv": "https://arxiv.org/abs/2605.16381",
            "focus": "部分观测下的主动决策与轨迹质量",
            "method": "CB-Stream Loss；turn-level + trajectory-level GRPO",
            "scale": "3B/4B；SFT 64 H100，RL 8 H100",
            "project_use": "类不平衡和轨迹级训练目标参考",
        },
        {
            "id": "r3",
            "title": "R3-Streaming",
            "date": "2026-05-18 / v2 2026-06-01",
            "arxiv": "https://arxiv.org/abs/2605.17921",
            "focus": "记忆、响应准备度与算力路由",
            "method": "Remember / Respond / Reason；TB-GRPO",
            "scale": "3B/7B fast + 4B/8B/32B slow",
            "project_use": "近期信息保真、历史压缩和独立 readiness head 参考",
        },
        {
            "id": "pwr",
            "title": "Plan, Watch, Recover",
            "date": "2026-06-03",
            "arxiv": "https://arxiv.org/abs/2606.04970",
            "focus": "第一人称操作指导、偏离检测与恢复",
            "method": "显式 procedural state；planner + interaction duplex",
            "scale": "论文大型模型；训练代码、权重和 plan/cue targets 未公开",
            "project_use": "当前主路线：压缩为 Small 可部署的状态与决策接口",
        },
    ]


def write_dashboard_data(data: dict[str, Any]) -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    data_path = DASHBOARD_DIR / "data.js"
    data_path.write_text(f"window.PWR_DASHBOARD_DATA={serialized};\n", encoding="utf-8")

    source_files = [DATASET_PATH]
    for definition in EXPERIMENTS:
        source_files.append(PROJECT_ROOT / definition["metrics"])
        source_files.append(PROJECT_ROOT / definition["predictions"])
    r1_root = PROJECT_ROOT / "output/experiments/20260714_internvl35_1b_oracle_state_r1_pilot_v1/variants"
    source_files.extend(r1_root / directory / "metrics_summary.json" for _, _, directory in R1_EXPERIMENTS)
    source_files.extend(PROJECT_ROOT / item["path"] for item in data["inference_study"]["comparison_sources"])
    source_files.extend(
        [
            PROJECT_ROOT / "literature/papers/challenge1_proactive/PWR_audit.md",
            PROJECT_ROOT / "literature/papers/challenge1_proactive/MMDuet2.md",
            PROJECT_ROOT / "literature/papers/challenge1_proactive/StreamPro.md",
            PROJECT_ROOT / "literature/papers/challenge1_proactive/R3-Streaming.md",
        ]
    )
    source_files = list(dict.fromkeys(source_files))
    manifest = {
        "generated_file": str(data_path.relative_to(PROJECT_ROOT)),
        "generated_sha256": sha256(data_path),
        "source_files": [
            {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256(path)} for path in source_files
        ],
        "counts": {
            "sessions": len(data["sessions"]),
            "chunks": data["metadata"]["chunks"],
            "experiments": len(data["experiments"]),
        },
    }
    (DASHBOARD_DIR / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_notebook() -> None:
    source = NOTEBOOK_SOURCE.read_text(encoding="utf-8")
    parts = [part.strip() for part in source.split("\n<!-- SLIDE -->\n") if part.strip()]
    cells = []
    for index, part in enumerate(parts):
        cells.append(
            {
                "cell_type": "markdown",
                "id": f"slide-{index + 1:02d}",
                "metadata": {"slideshow": {"slide_type": "slide"}},
                "source": [line + "\n" for line in part.splitlines()],
            }
        )
    notebook = {
        "cells": cells,
        "metadata": {
            "celltoolbar": "Slideshow",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "rise": {"autolaunch": False, "scroll": True, "transition": "none"},
            "presentation": {
                "title": "EgoProactive Small：PWR-inspired 阶段工作汇报",
                "target_duration_minutes": 15,
                "dashboard_url": "http://127.0.0.1:8766",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def build() -> dict[str, Any]:
    dataset_rows = read_jsonl(DATASET_PATH)
    if len(dataset_rows) != 700:
        raise ValueError(f"Expected 700 sessions, found {len(dataset_rows)}")
    chunk_count = sum(len(row["video_intervals"]) for row in dataset_rows)
    if chunk_count != 9935:
        raise ValueError(f"Expected 9,935 chunks, found {chunk_count}")
    for row in dataset_rows:
        chunk_total = len(row["video_intervals"])
        if len(row["answers"]) != chunk_total or len(row["dialog"]) != chunk_total:
            raise ValueError(f"Source alignment mismatch for {row['video_path']}")
        video_path = PROJECT_ROOT / "data/egoproactive/val" / row["video_path"]
        if not video_path.is_file():
            raise ValueError(f"Missing validation video: {video_path}")

    experiments, prediction_rows = load_experiments(dataset_rows)
    sessions = build_sessions(dataset_rows, prediction_rows)
    domains = sorted({row["domain"] for row in dataset_rows})
    gold_interrupts = sum(
        parse_answer(answer)["decision"] == "interrupt" for row in dataset_rows for answer in row["answers"]
    )
    data = {
        "metadata": {
            "title": "EgoProactive Small / PWR-inspired 实验视图",
            "as_of": "2026-07-15",
            "sessions": len(dataset_rows),
            "chunks": chunk_count,
            "domains": domains,
            "gold_interrupts": gold_interrupts,
            "gold_interrupt_rate": gold_interrupts / chunk_count,
            "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
            "evidence_boundary": "Public-validation-supervised development evidence; not a hidden-test claim.",
        },
        "literature": literature_data(),
        "experiments": experiments,
        "domain_metrics": build_domain_metrics(dataset_rows, prediction_rows),
        "position_metrics": build_position_metrics(dataset_rows, prediction_rows),
        "r1_study": load_r1_study(),
        "inference_study": inference_study(),
        "sessions": sessions,
    }
    data["featured_sessions"] = choose_featured_sessions(sessions)
    write_dashboard_data(data)
    write_notebook()
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true", help="Print the generated asset summary")
    args = parser.parse_args()
    data = build()
    if args.summary:
        print(
            json.dumps(
                {
                    "notebook": str(NOTEBOOK_OUTPUT),
                    "dashboard_data": str(DASHBOARD_DIR / "data.js"),
                    "sessions": len(data["sessions"]),
                    "chunks": data["metadata"]["chunks"],
                    "featured_sessions": data["featured_sessions"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
