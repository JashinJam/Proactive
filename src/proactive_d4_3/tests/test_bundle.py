from __future__ import annotations

import json
import unittest
from pathlib import Path

from proactive_d4.submission import validate_bundle_manifest
from proactive_r0.artifacts import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class History8BundleTest(unittest.TestCase):
    def test_bundle_matches_deployment_config(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs/d4_2_internvl35_1b_history8_deploy_shared_vision_v1.json").read_text(encoding="utf-8")
        )
        bundle = PROJECT_ROOT / "submission/d4_2_history8_small"
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        audit = validate_bundle_manifest(manifest, config)
        self.assertEqual(audit["total_parameters"], 1_060_898_844)
        self.assertEqual(config["inference"]["max_history_turns"], 8)
        self.assertEqual(
            sha256_file(bundle / "decision_head.json"),
            manifest["decision_head"]["sha256"],
        )


if __name__ == "__main__":
    unittest.main()
