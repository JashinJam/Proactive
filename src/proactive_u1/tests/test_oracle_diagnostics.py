from __future__ import annotations

import unittest

from proactive_u1.oracle_diagnostics import analyze_state_content
from proactive_u1.state_review import STATE_VARIANTS


class OracleDiagnosticsTest(unittest.TestCase):
    def test_counts_pairwise_empty_and_changed_content(self) -> None:
        samples = [
            {"sample_id": "a"},
            {"sample_id": "b"},
        ]
        contents = {
            "forced_no_state": ["", "same"],
            "forced_oracle_step": ["step", "same"],
            "forced_oracle_full": ["", "full"],
        }
        rows = []
        for variant in STATE_VARIANTS:
            for index, sample in enumerate(samples):
                rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "input_index": index,
                        "chunk_index": index,
                        "domain": "Tutorial",
                        "position_bin": "2-4",
                        "variant": variant,
                        "content": contents[variant][index],
                        "used_fallback": not bool(contents[variant][index]),
                    }
                )
        result = analyze_state_content(samples, rows)
        step = result["pairwise"]["forced_oracle_step_vs_forced_no_state"]
        self.assertEqual(step["exact_content_equal"], 1)
        self.assertEqual(step["target_only_nonempty"], 1)
        full = result["pairwise"]["forced_oracle_full_vs_forced_no_state"]
        self.assertEqual(full["both_empty"], 1)
        self.assertEqual(full["both_nonempty_but_changed"], 1)


if __name__ == "__main__":
    unittest.main()
