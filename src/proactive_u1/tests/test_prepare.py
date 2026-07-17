from __future__ import annotations

import copy
import unittest

from proactive_u0.core import FALLBACK_ANSWER
from proactive_u1.prepare import prepare_sample, strip_current_answers


def fixtures():
    sources = []
    predictions = []
    r0_rows = []
    domains = ["Arts and Crafts", "Chef", "Handyman", "Tutorial"]
    for input_index in range(700):
        domain = domains[input_index % 4]
        intervals = [[float(i * 8), float((i + 1) * 8)] for i in range(12)]
        answers = ["$silent$"] * 12
        sources.append(
            {
                "video_path": f"video-{input_index}.mp4",
                "video_intervals": intervals,
                "query": f"Query {input_index}",
                "domain": domain,
                "task": f"Task {input_index}",
                "answers": answers,
                "dialog": [[{"role": "user", "text": f"Query {input_index}"}]] * 12,
            }
        )
        predicted = ["$silent$"] * 12
        for chunk_index in (1, 2, 5, 10):
            predicted[chunk_index] = FALLBACK_ANSWER
        predictions.append(
            {"video_path": f"video-{input_index}.mp4", "answers": predicted}
        )
        r0_rows.append(
            {
                "input_index": input_index,
                "video_path": f"video-{input_index}.mp4",
                "chunks": [
                    {
                        "chunk_index": chunk_index,
                        "raw_response": "$silent$",
                    }
                    for chunk_index in range(12)
                ],
            }
        )
    return sources, predictions, r0_rows


class PrepareTest(unittest.TestCase):
    def test_selection_is_label_independent_and_balanced(self) -> None:
        sources, predictions, r0_rows = fixtures()
        changed = copy.deepcopy(sources)
        for row in changed:
            row["answers"] = ["$interrupt$Changed"] * 12
        first = prepare_sample(
            strip_current_answers(sources),
            predictions,
            r0_rows,
            seed="fixed",
            sessions_per_domain=5,
            excluded_input_indices={14, 123, 326, 687},
        )
        second = prepare_sample(
            strip_current_answers(changed),
            predictions,
            r0_rows,
            seed="fixed",
            sessions_per_domain=5,
            excluded_input_indices={14, 123, 326, 687},
        )
        self.assertEqual(first, second)
        rows, manifest, annotations = first
        self.assertEqual(len(rows), 80)
        self.assertEqual(len(annotations), 20)
        self.assertEqual(manifest["coverage"]["by_domain"], {
            "Arts and Crafts": 20,
            "Chef": 20,
            "Handyman": 20,
            "Tutorial": 20,
        })
        self.assertEqual(manifest["selection"]["smoke_chunks"], 16)

    def test_rejects_sources_that_still_contain_answers(self) -> None:
        sources, predictions, r0_rows = fixtures()
        with self.assertRaisesRegex(ValueError, "label-free"):
            prepare_sample(
                sources,
                predictions,
                r0_rows,
                seed="fixed",
                sessions_per_domain=1,
                excluded_input_indices=set(),
            )


if __name__ == "__main__":
    unittest.main()
