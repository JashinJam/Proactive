from __future__ import annotations

import unittest
import inspect

from proactive_d6.contract import labels_for_allowed_sessions
from proactive_d6.run_fold import _train


class ContractTest(unittest.TestCase):
    def test_train_phase_has_no_formal_extraction_dependency(self) -> None:
        source = inspect.getsource(_train)
        self.assertNotIn("primary_records", source)
        self.assertNotIn("representation_audit", source)

    def test_only_allowed_session_labels_are_unsealed(self) -> None:
        rows = [
            {
                "answers": ["$silent$", "$interrupt$now"],
                "video_intervals": [[0, 1], [1, 2]],
            },
            {
                "answers": ["$interrupt$now"],
                "video_intervals": [[0, 1]],
            },
        ]
        labels = labels_for_allowed_sessions(rows, [1])
        self.assertEqual(labels, {(1, 0): 1})
        self.assertFalse(any(key[0] == 0 for key in labels))
