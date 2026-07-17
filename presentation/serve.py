#!/usr/bin/env python3
"""Serve the dashboard and validation videos with byte-range support."""

from __future__ import annotations

import argparse
import errno
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


PRESENTATION_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = PRESENTATION_DIR / "results_dashboard"
VIDEO_DIR = PRESENTATION_DIR.parent / "data/egoproactive/val"
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


class DashboardHandler(SimpleHTTPRequestHandler):
    server_version = "EgoProactivePresentation/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            payload = b'{"status":"ok"}\n'
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path.startswith("/media/"):
            self._serve_video(parsed.path[len("/media/") :], head_only=False)
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/media/"):
            self._serve_video(parsed.path[len("/media/") :], head_only=True)
            return
        super().do_HEAD()

    def _serve_video(self, encoded_name: str, head_only: bool) -> None:
        filename = unquote(encoded_name)
        if not filename or Path(filename).name != filename:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid media path")
            return
        path = VIDEO_DIR / filename
        if not path.is_file() or path.suffix.lower() != ".mp4":
            self.send_error(HTTPStatus.NOT_FOUND, "Video not found")
            return

        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = RANGE_RE.fullmatch(range_header.strip())
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = int(last) if last else size - 1
            elif last:
                suffix_length = int(last)
                start = max(0, size - suffix_length)
                end = size - 1
            if start >= size or end < start:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT

        content_length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return

        with path.open("rb") as handle:
            handle.seek(start)
            remaining = content_length
            while remaining:
                block = handle.read(min(1024 * 1024, remaining))
                if not block:
                    break
                try:
                    self.wfile.write(block)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(block)

    def log_message(self, format: str, *args) -> None:
        if args and str(args[0]).startswith("GET /health"):
            return
        super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    if not (DASHBOARD_DIR / "data.js").is_file():
        raise SystemExit("Missing results_dashboard/data.js; run presentation/build_assets.py first")
    if not VIDEO_DIR.is_dir():
        raise SystemExit(f"Missing video directory: {VIDEO_DIR}")

    try:
        server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise SystemExit(
                f"Port {args.port} is already in use. "
                f"Try: python presentation/serve.py --port {args.port + 1}"
            ) from None
        raise
    print(f"Dashboard: http://{args.host}:{args.port}", flush=True)
    print(f"Serving {DASHBOARD_DIR}", flush=True)
    print(f"Video source: {VIDEO_DIR}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
