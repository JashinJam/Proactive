from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from proactive_d4_1.run_search import discover_idle_gpus


class GpuSelectionTest(unittest.TestCase):
    @mock.patch("proactive_d4_1.run_search.subprocess.run")
    def test_auto_selects_only_idle_gpus(self, run: mock.Mock) -> None:
        run.side_effect = [
            subprocess.CompletedProcess(
                [],
                0,
                stdout="0, GPU-a\n1, GPU-b\n2, GPU-c\n3, GPU-d\n4, GPU-e\n",
                stderr="",
            ),
            subprocess.CompletedProcess([], 0, stdout="GPU-b, 42\n", stderr=""),
        ]
        self.assertEqual(discover_idle_gpus(4), ["0", "2", "3", "4"])

    @mock.patch("proactive_d4_1.run_search.subprocess.run")
    def test_explicit_busy_gpu_fails_safely(self, run: mock.Mock) -> None:
        run.side_effect = [
            subprocess.CompletedProcess(
                [], 0, stdout="0, GPU-a\n1, GPU-b\n", stderr=""
            ),
            subprocess.CompletedProcess([], 0, stdout="GPU-b, 42\n", stderr=""),
        ]
        with self.assertRaisesRegex(RuntimeError, "compute processes"):
            discover_idle_gpus(2, "0,1")

    @mock.patch("proactive_d4_1.run_search.subprocess.run")
    def test_insufficient_idle_gpus_fails(self, run: mock.Mock) -> None:
        run.side_effect = [
            subprocess.CompletedProcess(
                [], 0, stdout="0, GPU-a\n1, GPU-b\n", stderr=""
            ),
            subprocess.CompletedProcess([], 0, stdout="GPU-b, 42\n", stderr=""),
        ]
        with self.assertRaisesRegex(RuntimeError, "requires 2 idle GPUs"):
            discover_idle_gpus(2)


if __name__ == "__main__":
    unittest.main()
