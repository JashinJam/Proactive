from __future__ import annotations

import json
import unittest
from pathlib import Path

from proactive_presentation.build import (
    DEFAULT_INPUT,
    INPUT_SHA256,
    PROJECT_ROOT,
    build_dashboard,
    group_feature_names,
    validate_stage_content,
)


PRESENTATION_DIR = PROJECT_ROOT / "presentation" / "2026-07-23"
DATA_DIR = PRESENTATION_DIR / "dashboard" / "data"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class DashboardBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_dashboard(PRESENTATION_DIR, DEFAULT_INPUT)
        cls.catalog = read_json(DATA_DIR / "catalog.json")
        cls.metrics = read_json(DATA_DIR / "metrics.json")
        cls.cases = read_json(DATA_DIR / "cases.json")
        cls.stages = read_json(DATA_DIR / "stages.json")
        cls.session_index = read_json(DATA_DIR / "sessions" / "index.json")["sessions"]

    def test_source_coverage_order_and_hash(self) -> None:
        self.assertEqual(self.manifest["source"]["sha256"], INPUT_SHA256)
        self.assertEqual(self.manifest["source"]["sessions"], 700)
        self.assertEqual(self.manifest["source"]["chunks"], 9935)
        self.assertEqual(self.manifest["session_files"]["count"], 700)
        self.assertEqual([row["input_index"] for row in self.session_index], list(range(700)))

    def test_official_metrics_reproduced(self) -> None:
        comparison = {item["id"]: item for item in self.metrics["d42_comparison"]}
        self.assertEqual(comparison["d42_history8"]["metrics"]["macro_f1"], 0.6988)
        self.assertEqual(comparison["d42_baseline"]["metrics"]["macro_f1"], 0.6846)
        self.assertEqual(
            self.metrics["winner_audit"]["train_fit_sanity"]["macro_f1"], 0.7469
        )
        self.assertEqual(self.manifest["full_refit_official"]["macro_f1"], 0.7469)

    def test_chart_metrics_are_complete(self) -> None:
        chart_ids = {
            "d1_fused",
            "d2_residual",
            "d2_lora",
            "d3_dynamics",
            "d3_dialog_control",
        }
        catalog = {item["id"]: item for item in self.catalog["experiments"]}
        for experiment_id in chart_ids:
            metrics = catalog[experiment_id]["metrics"]
            for key in ("macro_f1", "interrupt_f1", "silent_f1"):
                self.assertIsInstance(metrics[key], (int, float))

        d42 = {item["id"]: item for item in self.metrics["d42_comparison"]}
        baseline = d42["d42_baseline"]
        winner = d42["d42_history8"]
        self.assertEqual(
            [item["fold"] for item in baseline["folds"]],
            [item["fold"] for item in winner["folds"]],
        )
        self.assertEqual(set(baseline["domains"]), set(winner["domains"]))
        self.assertEqual(len(winner["folds"]), 5)
        self.assertEqual(len(winner["domains"]), 4)
        for item in [*d42.values(), *self.metrics["d5_funnel"]]:
            for key in ("macro_f1", "interrupt_f1", "silent_f1"):
                self.assertIsInstance(item["metrics"][key], (int, float))

    def test_evidence_and_comparison_boundaries(self) -> None:
        entries = self.catalog["experiments"]
        ids = {item["id"] for item in entries}
        self.assertEqual(len(ids), len(entries))
        for item in entries:
            self.assertIn("comparison_group", item)
            self.assertIn("evidence_class", item)
            if item["evidence_class"] == "aggregate_only":
                self.assertFalse(item["case_browsable"])
                self.assertNotIn(item["id"], self.catalog["case_config_ids"])
        self.assertFalse(self.manifest["evidence_policy"]["hidden_test_claimed"])
        self.assertFalse(self.manifest["evidence_policy"]["d6_in_rankings"])

    def test_contribution_partition_and_no_hidden_vectors(self) -> None:
        audit = self.manifest["contribution_audit"]
        self.assertEqual(
            audit["feature_groups"],
            {
                "temporal_response_scalar": 18,
                "tag_margin": 1,
                "hidden_block": 1024,
                "dialog_stage": 8,
            },
        )
        self.assertLessEqual(audit["maximum_reconstruction_error"], 1e-9)
        self.assertFalse(audit["hidden_vectors_emitted"])
        sample = (DATA_DIR / "sessions" / "0.json").read_text(encoding="utf-8")
        self.assertNotIn('"hidden_0000"', sample)
        self.assertIn('"hidden_block"', sample)

    def test_unknown_feature_name_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown D4.2 feature name"):
            group_feature_names(["not_a_frozen_feature"])

    def test_featured_cases_match_their_definitions(self) -> None:
        seen_domains = set()
        for featured in self.cases["featured"]:
            session = read_json(DATA_DIR / "sessions" / f"{featured['input_index']}.json")
            chunk = session["chunks"][featured["chunk_index"]]
            gold = chunk["gold"]["decision"]
            baseline = chunk["predictions"]["d42_baseline"]["decision"]
            history8 = chunk["predictions"]["d42_history8"]["decision"]
            if featured["type"] == "repair":
                self.assertNotEqual(baseline, gold)
                self.assertEqual(history8, gold)
            elif featured["type"] == "regression":
                self.assertEqual(baseline, gold)
                self.assertNotEqual(history8, gold)
            else:
                self.assertNotEqual(baseline, gold)
                self.assertNotEqual(history8, gold)
            seen_domains.add(session["domain"])
        self.assertEqual(len(seen_domains), 3)

    def test_d6_is_snapshot_only(self) -> None:
        snapshot = read_json(DATA_DIR / "d6_status.json")
        self.assertFalse(snapshot["efficacy_available"])
        self.assertFalse(snapshot["ranking_eligible"])
        self.assertIn("captured_at", snapshot)

    def test_stage_log_contract_and_d6_boundary(self) -> None:
        stages = self.stages["stages"]
        self.assertEqual(
            [item["id"] for item in stages],
            ["d1-d2", "d3", "d3d-d4", "d42", "d5", "d6"],
        )
        review = stages[0]
        self.assertGreaterEqual(len(review["baseline_facts"]), 4)
        self.assertTrue(review["brief_review"])
        self.assertNotIn("configuration", review)
        self.assertNotIn("analysis", review)
        for stage in stages[1:-1]:
            self.assertTrue(stage["configuration"])
            self.assertTrue(stage["reason"])
            self.assertTrue(stage["result_view"])
            self.assertTrue(stage["analysis"])
        d42 = next(item for item in stages if item["id"] == "d42")
        self.assertGreaterEqual(len(d42["limited_search"]["items"]), 3)
        d6 = stages[-1]
        self.assertNotIn("result_view", d6)
        self.assertNotIn("analysis", d6)
        self.assertGreaterEqual(len(d6["model_flow"]), 6)
        self.assertTrue(d6["why_slow"])

    def test_d6_stage_rejects_premature_results(self) -> None:
        source = read_json(PRESENTATION_DIR / "stage_content.json")
        source["stages"][-1]["analysis"] = ["premature"]
        with self.assertRaisesRegex(ValueError, "must not expose results"):
            validate_stage_content(source)

    def test_frontend_is_single_stage_log_without_runtime_dependencies(self) -> None:
        dashboard = PRESENTATION_DIR / "dashboard"
        html = (dashboard / "index.html").read_text(encoding="utf-8")
        javascript = (dashboard / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="page-select"', html)
        self.assertIn('value="d1-d2"', html)
        self.assertNotIn('value="d1"', html)
        self.assertNotIn('value="d2"', html)
        self.assertIn('value="preview"', html)
        self.assertIn("D4.1 · 输入策略适配", html)
        self.assertNotIn("D4.2", html)
        self.assertNotIn("Presentation", html)
        self.assertNotIn("Explorer", html)
        self.assertIn('fetchJSON("/data/stages.json")', javascript)
        self.assertIn("d42_history8", javascript)
        self.assertIn("D4.1 history8", javascript)
        self.assertNotIn("D4.2", javascript)
        self.assertIn("700 / 9,935", javascript)
        self.assertIn("macroScoreChart", javascript)
        self.assertIn("classF1Chart", javascript)
        self.assertIn("stabilityChart", javascript)
        self.assertIn('section(index++, "模型流程"', javascript)
        self.assertIn('section(index++, "耗时说明"', javascript)
        self.assertNotIn("为什么需要这么长时间", javascript)
        self.assertIn("const lower = 0.5", javascript)
        self.assertNotIn("自适应横轴", javascript)
        self.assertNotIn("横轴从 0 开始", javascript)
        self.assertIn('<progress class="score-track"', javascript)
        self.assertIn('<progress class="class-track', javascript)
        self.assertIn('class="dot-range base"', javascript)
        self.assertNotIn('style="width:', javascript)
        self.assertNotIn('style="left:', javascript)
        self.assertIn("OOF Macro F1", javascript)
        self.assertNotIn("Train-fit Macro F1", javascript)
        self.assertNotIn("OOF held-out development estimate", javascript)
        self.assertNotIn("https://", html)
        self.assertNotIn("cdn", html.lower())


if __name__ == "__main__":
    unittest.main()
