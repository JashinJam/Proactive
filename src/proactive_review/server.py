#!/usr/bin/env python3
"""Serve the U0/U1 blind-review UI, videos, and durable rating API."""

from __future__ import annotations

import argparse
import errno
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from proactive_review.core import RatingStore, Study


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "presentation/human_review"
VIDEO_DIR = PROJECT_ROOT / "data/egoproactive/val"
U0_BLIND = (
    PROJECT_ROOT
    / "output/experiments/20260716_internvl35_1b_d1_utterance_u0_v1/review_items_blind.jsonl"
)
U1_BLIND = (
    PROJECT_ROOT
    / "output/experiments/20260716_internvl35_1b_fixed_gate_forced_generation_u1_v1_nostate_full"
    / "analysis/paired_review_blind.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "output/human_reviews"
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")
MAX_REQUEST_BYTES = 2 * 1024 * 1024


class ReviewApplication:
    def __init__(self, output_root: Path) -> None:
        self.studies = {
            "u0": Study("u0", U0_BLIND),
            "u1": Study("u1", U1_BLIND),
        }
        self.store = RatingStore(output_root, self.studies)

    @staticmethod
    def identity(query: dict[str, list[str]]) -> tuple[str, str]:
        study = query.get("study", [""])[0].lower()
        reviewer = query.get("reviewer", [""])[0].upper()
        if study not in {"u0", "u1"}:
            raise ValueError("study must be u0 or u1")
        if reviewer not in {"A", "B"}:
            raise ValueError("reviewer must be A or B")
        return study, reviewer


class ReviewHandler(SimpleHTTPRequestHandler):
    server_version = "EgoProactiveHumanReview/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    @property
    def app(self) -> ReviewApplication:
        return self.server.app  # type: ignore[attr-defined,no-any-return]

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                self._json(HTTPStatus.OK, {"status": "ok"})
                return
            if parsed.path == "/api/bootstrap":
                self._bootstrap(parse_qs(parsed.query))
                return
            if parsed.path == "/api/session":
                self._session(parse_qs(parsed.query))
                return
            if parsed.path == "/api/export":
                self._export(parse_qs(parsed.query))
                return
            if parsed.path.startswith("/media/"):
                self._serve_video(parsed.path[len("/media/") :], head_only=False)
                return
            super().do_GET()
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - last-resort HTTP boundary
            self.log_error("GET failed: %s", exc)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "服务端读取失败"})

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/media/"):
            self._serve_video(parsed.path[len("/media/") :], head_only=True)
            return
        super().do_HEAD()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path != "/api/save-session":
                self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown API endpoint"})
                return
            length_text = self.headers.get("Content-Length", "")
            if not length_text.isdigit():
                raise ValueError("Missing request length")
            length = int(length_text)
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("Request body has an invalid size")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("Request body must be an object")
            study = str(value.get("study", "")).lower()
            reviewer = str(value.get("reviewer_slot", "")).upper()
            session_id = str(value.get("session_id", ""))
            if study not in {"u0", "u1"} or reviewer not in {"A", "B"}:
                raise ValueError("Invalid study or reviewer")
            result = self.app.store.save_session(
                study, reviewer, session_id, value.get("ratings")
            )
            self._json(HTTPStatus.OK, result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - last-resort HTTP boundary
            self.log_error("POST failed: %s", exc)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "评分保存失败"})

    def _bootstrap(self, query: dict[str, list[str]]) -> None:
        study_name, reviewer = self.app.identity(query)
        study = self.app.studies[study_name]
        progress = self.app.store.progress(study_name, reviewer)
        completed = set(progress["completed_session_ids"])
        sessions = []
        for index, session in enumerate(study.sessions):
            row = session.metadata()
            row.update({"index": index, "completed": session.session_id in completed})
            sessions.append(row)
        self._json(
            HTTPStatus.OK,
            {
                "schema_version": 1,
                "study": study_name,
                "study_label": "U0 全量内容审计" if study_name == "u0" else "U1 配对内容评测",
                "reviewer_slot": reviewer,
                "source_blind_sha256": study.source_sha256,
                "sessions": sessions,
                "progress": progress,
            },
        )

    def _session(self, query: dict[str, list[str]]) -> None:
        study_name, reviewer = self.app.identity(query)
        session_id = query.get("session_id", [""])[0]
        session = self.app.studies[study_name].session(session_id)
        saved = self.app.store.session_ratings(study_name, reviewer, session_id)
        self._json(
            HTTPStatus.OK,
            {
                **session.metadata(),
                "study": study_name,
                "reviewer_slot": reviewer,
                "items": list(session.items),
                "saved": saved,
            },
        )

    def _export(self, query: dict[str, list[str]]) -> None:
        study, reviewer = self.app.identity(query)
        path = self.app.store.csv_path(study, reviewer)
        if not path.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"error": "尚无已确认评分可导出"})
            return
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{study}_reviewer_{reviewer}_ratings.csv"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
        payload = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

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

    def log_message(self, format: str, *args: Any) -> None:
        if args and (str(args[0]).startswith("GET /health") or str(args[0]).startswith("GET /media/")):
            return
        super().log_message(format, *args)


class ReviewHTTPServer(ThreadingHTTPServer):
    app: ReviewApplication


def bind_server(host: str, requested_port: int, app: ReviewApplication, strict: bool) -> tuple[ReviewHTTPServer, int]:
    ports = [requested_port] if strict else list(range(requested_port, requested_port + 11))
    for port in ports:
        try:
            server = ReviewHTTPServer((host, port), ReviewHandler)
            server.app = app
            return server, port
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE or port == ports[-1]:
                raise
    raise RuntimeError("No port candidates were attempted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--strict-port", action="store_true", help="Fail instead of trying the next ten ports")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for required in (STATIC_DIR / "index.html", VIDEO_DIR, U0_BLIND, U1_BLIND):
        if not required.exists():
            raise SystemExit(f"Missing required review asset: {required}")

    app = ReviewApplication(args.output_root.resolve())
    try:
        server, port = bind_server(args.host, args.port, app, args.strict_port)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise SystemExit(f"Ports {args.port}-{args.port + 10} are already in use") from None
        raise

    print(f"Human review UI: http://{args.host}:{port}", flush=True)
    print(f"Ratings output: {args.output_root.resolve()}", flush=True)
    print("Blind keys are not loaded by this service.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

