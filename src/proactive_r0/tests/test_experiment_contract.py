from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from proactive_r0.experiment_contract import (
    REQUIRED_ARTIFACTS,
    audit_experiment_contract,
)


class ExperimentContractTest(unittest.TestCase):
    def test_complete_experiment_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in REQUIRED_ARTIFACTS:
                path = root / name
                if name in {"config.json", "data_manifest.json", "metrics.json"}:
                    path.write_text("{}\n", encoding="utf-8")
                elif name == "predictions.jsonl":
                    path.write_text(
                        json.dumps(
                            {"video_path": "a.mp4", "answers": ["$silent$"]}
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                else:
                    path.write_text("complete\n", encoding="utf-8")
            audit = audit_experiment_contract(root)
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["predictions"]["sessions"], 1)  # type: ignore[index]

    def test_missing_and_invalid_artifacts_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("[]\n", encoding="utf-8")
            audit = audit_experiment_contract(root)
        self.assertEqual(audit["status"], "failed")
        self.assertTrue(any("README.md" in error for error in audit["errors"]))
        self.assertTrue(any("invalid config.json" in error for error in audit["errors"]))


if __name__ == "__main__":
    unittest.main()
