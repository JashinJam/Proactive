from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from proactive_presentation.build import DEFAULT_INPUT, PROJECT_ROOT
from proactive_presentation.server import create_server


PRESENTATION_DIR = PROJECT_ROOT / "presentation" / "2026-07-23"
VIDEO_DIR = Path("/data1/wearable_ai_challenge_data/egoproactive/val")


class PresentationServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = create_server(
            PRESENTATION_DIR, DEFAULT_INPUT, VIDEO_DIR, host="127.0.0.1", port=0
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.video_name = sorted(cls.server.video_allowlist)[0]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def assert_status(self, path: str, expected: int) -> None:
        try:
            with urlopen(self.base + path, timeout=5) as response:
                status = response.status
        except HTTPError as error:
            status = error.code
        self.assertEqual(status, expected)

    def test_health_and_deep_link(self) -> None:
        with urlopen(self.base + "/health", timeout=5) as response:
            payload = json.loads(response.read())
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["sessions"], 700)
        self.assertEqual(payload["chunks"], 9935)
        with urlopen(
            self.base + "/?page=preview&session=0&chunk=1", timeout=5
        ) as response:
            payload = response.read()
            self.assertIn(b"Proactive VLM", payload)
            self.assertIn(b'id="page-select"', payload)

    def test_media_range_and_head(self) -> None:
        request = Request(
            self.base + f"/media/{self.video_name}", headers={"Range": "bytes=0-31"}
        )
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers["Accept-Ranges"], "bytes")
            self.assertTrue(response.headers["Content-Range"].startswith("bytes 0-31/"))
            self.assertEqual(len(response.read()), 32)
        head = Request(self.base + f"/media/{self.video_name}", method="HEAD")
        with urlopen(head, timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"")
            self.assertGreater(int(response.headers["Content-Length"]), 0)

    def test_invalid_media_and_directories_are_rejected(self) -> None:
        self.assert_status("/media/%2e%2e%2fsecret.mp4", 404)
        self.assert_status("/media/not_allowlisted.mp4", 404)
        self.assert_status("/media/not-a-video.txt", 404)
        self.assert_status("/data/sessions/", 404)
        self.assert_status("/unknown", 404)

    def test_unsatisfiable_range(self) -> None:
        request = Request(
            self.base + f"/media/{self.video_name}", headers={"Range": "bytes=999999999999-"}
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 416)
        self.assertTrue(caught.exception.headers["Content-Range"].startswith("bytes */"))


if __name__ == "__main__":
    unittest.main()
