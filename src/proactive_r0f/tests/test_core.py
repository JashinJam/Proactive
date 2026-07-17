from __future__ import annotations

import unittest

from proactive_r0f.core import repair_response_intent


class ResponseIntentRepairTest(unittest.TestCase):
    def test_explicit_tags_preserve_r0_canonicalization(self) -> None:
        self.assertEqual(repair_response_intent("$silent$"), ("$silent$", None))
        self.assertEqual(
            repair_response_intent(" $silent$ explanation"),
            ("$silent$", "trimmed_silent_suffix"),
        )
        self.assertEqual(
            repair_response_intent("$interrupt$ Do it"),
            ("$interrupt$Do it", None),
        )
        self.assertEqual(
            repair_response_intent("$interrupt$"),
            (
                "$interrupt$Please continue with the next step.",
                "empty_interrupt_utterance",
            ),
        )

    def test_nonempty_malformed_text_becomes_interrupt(self) -> None:
        self.assertEqual(
            repair_response_intent("  Mix the ingredients now."),
            (
                "$interrupt$Mix the ingredients now.",
                "malformed_nonempty_repaired_as_interrupt",
            ),
        )
        self.assertEqual(
            repair_response_intent("$Iinterrupt$ Try again"),
            (
                "$interrupt$$Iinterrupt$ Try again",
                "malformed_nonempty_repaired_as_interrupt",
            ),
        )

    def test_empty_response_stays_silent(self) -> None:
        self.assertEqual(
            repair_response_intent("   "),
            ("$silent$", "empty_raw_response_kept_silent"),
        )


if __name__ == "__main__":
    unittest.main()

