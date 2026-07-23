"""Serve the generated experiment log and allowlisted MP4 files over read-only HTTP."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote_to_bytes, urlsplit

from proactive_presentation.build import DEFAULT_INPUT, sha256_file
from proactive_r0.core import load_jsonl


RANGE_PATTERN = re.compile(r"bytes=(\d*)-(\d*)")
STATIC_FILES = {
    "": "index.html",
    "index.html": "index.html",
    "styles.css": "styles.css",
    "app.js": "app.js",
}


def _decode_path(raw_path: str) -> str:
    try:
        return unquote_to_bytes(raw_path).decode("utf-8", errors="strict")
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("URL path is not valid UTF-8") from error


def _is_safe_relative_path(value: str) -> bool:
    if not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in ("", ".", "..") for part in path.parts)


def _parse_range(value: str, size: int) -> tuple[int, int]:
    match = RANGE_PATTERN.fullmatch(value.strip())
    if not match or size <= 0:
        raise ValueError("Unsupported byte range")
    first, last = match.groups()
    if not first and not last:
        raise ValueError("Empty byte range")
    if not first:
        suffix = int(last)
        if suffix <= 0:
            raise ValueError("Invalid suffix range")
        start = max(0, size - suffix)
        end = size - 1
    else:
        start = int(first)
        end = int(last) if last else size - 1
        if start >= size or end < start:
            raise ValueError("Unsatisfiable byte range")
        end = min(end, size - 1)
    return start, end


class PresentationServer(ThreadingHTTPServer):
    """HTTP server carrying immutable runtime paths and source validation state."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        presentation_dir: Path,
        input_jsonl: Path,
        video_dir: Path,
    ) -> None:
        self.presentation_dir = presentation_dir.expanduser().resolve()
        self.dashboard_dir = (self.presentation_dir / "dashboard").resolve()
        self.data_dir = (self.dashboard_dir / "data").resolve()
        self.video_dir = video_dir.expanduser().resolve()
        manifest_path = self.data_dir / "manifest.json"
        if not (self.dashboard_dir / "index.html").is_file():
            raise FileNotFoundError("Dashboard source is missing; expected dashboard/index.html")
        if not manifest_path.is_file():
            raise FileNotFoundError("Built dashboard data is missing; run proactive_presentation.build")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        input_path = input_jsonl.expanduser().resolve()
        actual_sha = sha256_file(input_path)
        built_sha = str(self.manifest.get("source", {}).get("sha256", ""))
        if actual_sha != built_sha:
            raise ValueError(f"Input JSONL SHA differs from built data: {actual_sha} != {built_sha}")
        rows = load_jsonl(input_path)
        if len(rows) != int(self.manifest.get("source", {}).get("sessions", -1)):
            raise ValueError("Input JSONL session count differs from built data")
        names = [str(row.get("video_path", "")) for row in rows]
        if len(set(names)) != len(names) or any(
            not name.endswith(".mp4") or Path(name).name != name for name in names
        ):
            raise ValueError("Input JSONL contains an invalid video allowlist")
        self.video_allowlist = frozenset(names)
        self.media_available = sum((self.video_dir / name).is_file() for name in names)
        super().__init__(address, PresentationRequestHandler)


class PresentationRequestHandler(BaseHTTPRequestHandler):
    """Route only known static assets, generated JSON, health, and allowlisted MP4s."""

    server: PresentationServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        self._route(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._route(send_body=False)

    def do_POST(self) -> None:  # noqa: N802
        self.send_error(HTTPStatus.METHOD_NOT_ALLOWED)

    def _route(self, send_body: bool) -> None:
        raw_path = urlsplit(self.path).path
        try:
            path = _decode_path(raw_path)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        if path == "/health":
            self._serve_health(send_body)
        elif path.startswith("/media/"):
            self._serve_media(path[len("/media/") :], send_body)
        elif path.startswith("/data/"):
            self._serve_data(path[len("/data/") :], send_body)
        else:
            self._serve_static(path.lstrip("/"), send_body)

    def _common_headers(self, content_type: str, content_length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; media-src 'self'; connect-src 'self'",
        )

    def _serve_health(self, send_body: bool) -> None:
        source = self.server.manifest.get("source", {})
        payload = json.dumps(
            {
                "status": "ok",
                "built_at": self.server.manifest.get("built_at"),
                "sessions": source.get("sessions"),
                "chunks": source.get("chunks"),
                "allowlisted_videos": len(self.server.video_allowlist),
                "media_available": self.server.media_available,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._common_headers("application/json; charset=utf-8", len(payload))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if send_body:
            self.wfile.write(payload)

    def _serve_static(self, request_path: str, send_body: bool) -> None:
        filename = STATIC_FILES.get(request_path)
        if filename is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._serve_file(self.server.dashboard_dir / filename, send_body, cache="no-cache")

    def _serve_data(self, request_path: str, send_body: bool) -> None:
        if not _is_safe_relative_path(request_path) or not request_path.endswith(".json"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        candidate = (self.server.data_dir / request_path).resolve()
        try:
            candidate.relative_to(self.server.data_dir)
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._serve_file(candidate, send_body, cache="public, max-age=60")

    def _serve_file(self, path: Path, send_body: bool, cache: str) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type == "application/javascript":
            content_type = "text/javascript"
        self.send_response(HTTPStatus.OK)
        self._common_headers(f"{content_type}; charset=utf-8", size)
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if send_body:
            with path.open("rb") as handle:
                self.wfile.write(handle.read())

    def _serve_media(self, name: str, send_body: bool) -> None:
        if (
            not name
            or "/" in name
            or "\\" in name
            or "\x00" in name
            or Path(name).name != name
            or not name.endswith(".mp4")
            or name not in self.server.video_allowlist
        ):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = self.server.video_dir / name
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        start, end, status = 0, size - 1, HTTPStatus.OK
        if range_header:
            try:
                start, end = _parse_range(range_header, size)
            except (ValueError, OverflowError):
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self._common_headers("video/mp4", length)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "public, max-age=3600")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not send_body:
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                block = handle.read(min(1024 * 1024, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)


def create_server(
    presentation_dir: Path,
    input_jsonl: Path,
    video_dir: Path,
    host: str = "0.0.0.0",
    port: int = 8765,
) -> PresentationServer:
    return PresentationServer((host, port), presentation_dir, input_jsonl, video_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentation-dir", required=True, type=Path)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = create_server(
        args.presentation_dir, args.input_jsonl, args.video_dir, args.host, args.port
    )
    print(
        f"Serving D1-D6 experiment log at http://{args.host}:{server.server_port} "
        f"({server.media_available}/{len(server.video_allowlist)} videos available)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
