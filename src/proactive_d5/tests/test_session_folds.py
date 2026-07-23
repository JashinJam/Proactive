from __future__ import annotations

import unittest

from proactive_d1.core import make_fold_manifest
from proactive_d5.session_folds import validate_session_fold_manifest


def rows() -> list[dict[str, object]]:
    return [
        {
            "video_path": f"v{index}.mp4",
            "domain": f"d{index % 4}",
            "query": f"task-{index // 2}",
            "video_intervals": [[0.0, 1.0]] * (index % 7 + 1),
        }
        for index in range(20)
    ]


class SessionFoldTest(unittest.TestCase):
    def test_d4_assignment_is_accepted_and_audited(self) -> None:
        source = rows()
        manifest = make_fold_manifest(source, folds=5, seed="fixed")
        assignments, audit = validate_session_fold_manifest(manifest, source)
        self.assertEqual(set(assignments.values()), set(range(5)))
        self.assertEqual(audit["algorithm"], "domain_stratified_sha256_round_robin")
        self.assertEqual(audit["seed"], "fixed")
        self.assertEqual(
            sum(int(value["sessions"]) for value in audit["folds"].values()),  # type: ignore[union-attr]
            len(source),
        )

    def test_non_d4_algorithm_is_rejected(self) -> None:
        source = rows()
        manifest = make_fold_manifest(source, folds=5, seed="fixed")
        manifest["algorithm"] = "exact_query_grouped_domain_length_greedy_v1"
        with self.assertRaisesRegex(ValueError, "Unsupported D1 split algorithm"):
            validate_session_fold_manifest(manifest, source)


if __name__ == "__main__":
    unittest.main()
