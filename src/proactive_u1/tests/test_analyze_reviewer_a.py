from __future__ import annotations

import unittest

from proactive_u1.analyze_reviewer_a import _require_reviewer_a


class ReviewerAOnlyTest(unittest.TestCase):
    def test_accepts_exact_a_coverage(self) -> None:
        _require_reviewer_a([{"reviewer_slot": "A"}], 1)

    def test_rejects_reviewer_b(self) -> None:
        with self.assertRaisesRegex(ValueError, "refuses reviewer slots"):
            _require_reviewer_a([{"reviewer_slot": "B"}], 1)

    def test_rejects_incomplete_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            _require_reviewer_a([{"reviewer_slot": "A"}], 2)


if __name__ == "__main__":
    unittest.main()
