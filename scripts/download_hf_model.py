#!/usr/bin/env python3
"""Download a Hugging Face model repository into models/<organization>/<name>.

The layout matches the project convention in models/readme.md:
    models/公司名/模型名/  ->  same files as the upstream repo
    (config.json, tokenizer files, *.py code, ...)

Weight files (*.safetensors, *.bin, ...) are skipped unless -w is given.

Robustness:
  * metadata and file lists are fetched through the HF-compatible HTTP API
  * every file is streamed in 1 MiB chunks with a per-read socket timeout
  * a failed read reopens the connection with HTTP Range and resumes from the
    last byte already on disk, so a stalled or dropped connection does not
    abort the whole run
  * files are grouped into batches; within a batch up to --workers files are
    downloaded concurrently
  * a live progress bar is drawn on stderr (disabled automatically when
    stderr is not a terminal)

Examples:
  python scripts/download_hf_model.py -m deepseek-ai/DeepSeek-V4.1-Flash
  python scripts/download_hf_model.py -m Qwen/Qwen3-8B -w
  python scripts/download_hf_model.py -m a/b -m c/d -mirror hf-mirror -w
  python scripts/download_hf_model.py -m deepseek-ai/DeepSeek-V4.1-Flash -endpoint https://hf-mirror.com
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
import shutil
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path, PurePosixPath


# Preset mirrors. Key = alias used by -mirror, value = base URL.
MIRRORS = {
    "opencsg":   "https://hub.opencsg.com/hf",
    "hf-mirror": "https://hf-mirror.com",
    "hf":        "https://huggingface.co",
}

# OpenCSG is the default source, matching the project's existing tooling.
DEFAULT_MIRROR = "opencsg"
DEFAULT_ENDPOINT = MIRRORS[DEFAULT_MIRROR]

# models/ lives next to scripts/, i.e. at the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "models"

DEFAULT_WORKERS = 4
DEFAULT_BATCH_SIZE = 8
DEFAULT_TIMEOUT = 60.0        # seconds, per connect and per socket read
DEFAULT_RETRIES = 3           # attempts per file, including the first
BACKOFF_BASE = 2.0            # sleep = BACKOFF_BASE ** attempt
MAX_BACKOFF = 30.0
CHUNK_SIZE = 1024 * 1024      # 1 MiB
RENDER_INTERVAL = 0.2         # seconds between progress redraws
SPEED_WINDOW = 6.0            # seconds used to smooth the speed readout
BAR_WIDTH = 22

WEIGHT_SUFFIXES = {
    ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".msgpack",
    ".onnx", ".gguf", ".tflite", ".ot", ".pb", ".pkl", ".pickle",
    ".npy", ".npz", ".hdf5", ".joblib",
}

MANIFEST_NAME = ".download_manifest.json"

# Metadata queries: the first asks for per-file sizes (blobs=true), the second
# is the plain form used as a fallback for mirrors that reject blobs=true.
_BLOBS_QUERY = "?blobs=true&expand[]=siblings&expand[]=sha"
_PLAIN_QUERY = "?expand[]=siblings&expand[]=sha"

_PRINT_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"endpoint must start with http:// or https://: {endpoint}")
    return endpoint


def resolve_endpoint(args: argparse.Namespace) -> str:
    """Priority: --endpoint > -mirror > HF_ENDPOINT env > DEFAULT_MIRROR."""
    if args.endpoint:
        return normalize_endpoint(args.endpoint)
    if args.mirror:
        return normalize_endpoint(MIRRORS[args.mirror])
    env_endpoint = os.environ.get("HF_ENDPOINT")
    if env_endpoint:
        return normalize_endpoint(env_endpoint)
    return normalize_endpoint(DEFAULT_ENDPOINT)


def auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def is_weight(path: str) -> bool:
    return any(PurePosixPath(path).name.lower().endswith(suffix) for suffix in WEIGHT_SUFFIXES)


def split_repo_id(repo_id: str) -> tuple[str, str]:
    parts = repo_id.split("/")
    if len(parts) != 2 or not all(parts) or any(part in {".", ".."} for part in parts):
        raise ValueError(f"repo id must have the form 'organization/model-name': {repo_id}")
    return parts[0], parts[1]


def quote_repo(repo_id: str) -> str:
    return "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))


def safe_destination(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe repository path: {relative_path}")
    return root.joinpath(*relative.parts)


def build_file_url(endpoint: str, repo_id: str, revision: str, path: str) -> str:
    quoted_path = "/".join(urllib.parse.quote(part, safe="") for part in PurePosixPath(path).parts)
    quoted_revision = urllib.parse.quote(revision, safe="")
    return f"{endpoint}/{quote_repo(repo_id)}/resolve/{quoted_revision}/{quoted_path}?download=true"


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:6.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:6.1f}PB"


def human_speed(bytes_per_second: float) -> str:
    return f"{human_size(bytes_per_second)}/s"


# ---------------------------------------------------------------------------
# Retry-aware HTTP open (used for metadata and size probes, not for streams)
# ---------------------------------------------------------------------------

class RetryExhausted(Exception):
    pass


def _mirror_error_message(error: urllib.error.HTTPError) -> str | None:
    """Some mirrors report logical errors (gated repo, unknown model) as HTTP 500
    with a JSON body like {"code": "AUTH-ERR-0", ...}. Surface those directly
    instead of burning retries on a non-transient failure."""
    try:
        payload = json.loads(error.read().decode("utf-8", "replace"))
    except (ValueError, OSError):
        return None
    if isinstance(payload, dict) and payload.get("code"):
        message = f"{payload['code']}: {payload.get('msg', '')}".strip().rstrip(":")
        if "AUTH-ERR" in message:
            message += " (gated on this mirror? try --token, or -mirror hf)"
        return message
    return None


def open_with_retry(request, *, timeout: float, retries: int, label: str):
    """Open a request, retrying transient failures. Caller must close it."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            message = _mirror_error_message(error)
            if message:
                raise ValueError(message) from error
            # 4xx (except 408/429) are permanent.
            if error.code < 500 and error.code not in (408, 429):
                raise
            last_error = error
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            last_error = error

        if attempt < retries:
            delay = min(BACKOFF_BASE ** attempt, MAX_BACKOFF)
            with _PRINT_LOCK:
                print(f"  ! retry {attempt}/{retries - 1} for {label}: {last_error} "
                      f"(sleep {delay:.1f}s)", file=sys.stderr)
            time.sleep(delay)

    raise RetryExhausted(f"{label}: giving up after {retries} attempts: {last_error}")


def request_json(url: str, token: str | None, timeout: float, retries: int) -> dict:
    request = urllib.request.Request(url, headers=auth_headers(token))
    with open_with_retry(request, timeout=timeout, retries=retries, label=f"GET {url}") as response:
        return json.load(response)


# ---------------------------------------------------------------------------
# Remote size probe
# ---------------------------------------------------------------------------

def probe_remote_size(url: str, token: str | None, timeout: float, retries: int) -> int | None:
    """Best-effort total size: HEAD first, then a ranged GET as a fallback."""
    try:
        request = urllib.request.Request(url, method="HEAD", headers=auth_headers(token))
        with open_with_retry(request, timeout=timeout, retries=retries, label=f"HEAD {url}") as response:
            length = response.headers.get("Content-Length")
        if length is not None and int(length) > 0:
            return int(length)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, RetryExhausted):
        pass

    try:
        headers = auth_headers(token)
        headers["Range"] = "bytes=0-0"
        request = urllib.request.Request(url, headers=headers)
        with open_with_retry(request, timeout=timeout, retries=retries, label=f"GET {url}") as response:
            content_range = response.headers.get("Content-Range")
            response.read(1)
        if content_range and "/" in content_range:
            total = content_range.rsplit("/", 1)[1].strip()
            if total.isdigit():
                return int(total)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, RetryExhausted):
        pass

    return None


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Thread-safe single-line progress bar with a directory of per-file bytes."""

    def __init__(self, total_files: int, enabled: bool) -> None:
        self._lock = threading.Lock()
        self._enabled = enabled
        self._total_files = total_files
        self._sizes: dict[str, int | None] = {}
        self._written: dict[str, int] = {}
        self._done: set[str] = set()
        self._bytes_total = 0
        self._unknown_sizes = 0
        self._samples: deque[tuple[float, int]] = deque(maxlen=1024)
        self._last_render = 0.0
        self._last_len = 0
        self._current = ""

    def register(self, path: str, size: int | None) -> None:
        with self._lock:
            self._sizes[path] = size
            self._written.setdefault(path, 0)
            if size:
                self._bytes_total += size
            else:
                self._unknown_sizes += 1

    def begin(self, path: str, written: int) -> None:
        with self._lock:
            self._written[path] = written
            self._current = path
            self._render(force=True)

    def add(self, path: str, count: int) -> None:
        with self._lock:
            self._written[path] = self._written.get(path, 0) + count
            self._render()

    def complete(self, path: str) -> None:
        with self._lock:
            size = self._sizes.get(path)
            if size:
                self._written[path] = size
            self._done.add(path)
            self._render(force=True)

    def fail(self, path: str) -> None:
        with self._lock:
            self._render(force=True)

    def log(self, message: str) -> None:
        """Print a permanent line without corrupting the progress bar."""
        with self._lock:
            self._clear()
            sys.stderr.write(message + "\n")
            sys.stderr.flush()
            self._last_len = 0
            self._last_render = 0.0
            self._render(force=True)

    def finish(self) -> None:
        with self._lock:
            if not self._enabled:
                return
            self._clear()
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._last_len = 0

    def _clear(self) -> None:
        if not self._enabled or self._last_len == 0:
            return
        sys.stderr.write("\r" + " " * self._last_len + "\r")
        sys.stderr.flush()

    def _render(self, *, force: bool = False) -> None:
        if not self._enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_render < RENDER_INTERVAL:
            return
        self._last_render = now

        done_bytes = sum(self._written.values())
        self._samples.append((now, done_bytes))
        while len(self._samples) > 2 and now - self._samples[0][0] > SPEED_WINDOW:
            self._samples.popleft()
        if len(self._samples) >= 2:
            t0, b0 = self._samples[0]
            elapsed = now - t0
            speed = (done_bytes - b0) / elapsed if elapsed > 0 else 0.0
        else:
            speed = 0.0

        if self._bytes_total > 0:
            ratio = min(done_bytes / self._bytes_total, 1.0)
            filled = int(ratio * BAR_WIDTH)
            bar = "█" * filled + "░" * (BAR_WIDTH - filled)
            percent = f"{ratio * 100:5.1f}%"
            prefix = "≥" if self._unknown_sizes else ""
            amounts = f"{human_size(done_bytes)}/{prefix}{human_size(self._bytes_total)}"
        else:
            bar = "░" * BAR_WIDTH
            percent = "  ?  "
            amounts = f"{human_size(done_bytes)}"

        line = (f"[{len(self._done)}/{self._total_files}] [{bar}] {percent} {amounts} "
                f"@ {human_speed(speed)}")
        if self._current:
            line += f"  {self._current}"

        width = shutil.get_terminal_size((100, 24)).columns
        if len(line) > width - 1:
            line = line[: width - 2] + "…"

        pad = max(self._last_len - len(line), 0)
        sys.stderr.write("\r" + line + " " * pad)
        sys.stderr.flush()
        self._last_len = len(line)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(
    endpoint: str,
    repo_id: str,
    revision: str,
    path: str,
    destination: Path,
    token: str | None,
    *,
    expected_size: int | None,
    timeout: float,
    retries: int,
    tracker: ProgressTracker,
) -> str:
    """Download one file: chunked, resumable, retried. Returns a status string."""
    url = build_file_url(endpoint, repo_id, revision, path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")

    remote_size = expected_size
    if remote_size is None:
        remote_size = probe_remote_size(url, token, timeout, retries)

    if destination.exists():
        local_size = destination.stat().st_size
        if remote_size is not None and local_size == remote_size:
            return "skipped"
        if remote_size is not None:
            destination.unlink(missing_ok=True)

    if temporary.exists() and remote_size is not None and temporary.stat().st_size > remote_size:
        temporary.unlink(missing_ok=True)

    written_start = 0
    success = False
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        resume_from = temporary.stat().st_size if temporary.exists() else 0
        if remote_size is not None and 0 < remote_size <= resume_from:
            written_start = resume_from
            success = True
            break

        headers = auth_headers(token)
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"
        request = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if resume_from > 0 and response.status == 206:
                    mode, written_start = "ab", resume_from
                else:
                    mode, written_start = "wb", 0

                tracker.begin(path, written_start)
                with temporary.open(mode) as output:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                        tracker.add(path, len(chunk))
            success = True
            break
        except (socket.timeout, TimeoutError, http.client.HTTPException,
                urllib.error.URLError, ConnectionError, OSError) as error:
            last_error = error
            if attempt < retries:
                delay = min(BACKOFF_BASE ** attempt, MAX_BACKOFF)
                tracker.log(f"  ! {path}: {type(error).__name__}: {error} "
                            f"— resume at {human_size(written_start)} in {delay:.1f}s "
                            f"(attempt {attempt}/{retries - 1})")
                time.sleep(delay)

    if not success:
        raise RetryExhausted(f"{path}: giving up after {retries} attempts: {last_error}")

    actual_size = temporary.stat().st_size
    if remote_size is not None and actual_size != remote_size:
        raise OSError(f"{path}: incomplete after retries ({actual_size} of {remote_size} bytes)")

    temporary.replace(destination)
    return "resumed" if written_start > 0 else "downloaded"


def _download_one(
    endpoint: str,
    repo_id: str,
    revision: str,
    path: str,
    model_dir: Path,
    token: str | None,
    index: int,
    total: int,
    tracker: ProgressTracker,
    *,
    expected_size: int | None,
    timeout: float,
    retries: int,
) -> tuple[str, str]:
    label = f"[{index}/{total}]"
    try:
        status = download_file(
            endpoint, repo_id, revision, path, safe_destination(model_dir, path), token,
            expected_size=expected_size, timeout=timeout, retries=retries, tracker=tracker,
        )
    except Exception as error:  # noqa: BLE001 - report and keep going
        tracker.fail(path)
        tracker.log(f"{label} FAILED     {path}: {error}")
        return path, f"error: {error}"

    tracker.complete(path)
    size_note = f" {human_size(expected_size)}" if expected_size else ""
    tracker.log(f"{label} {status:<10}{size_note} {path}")
    return path, status


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

def fetch_repo_files(
    endpoint: str,
    repo_id: str,
    revision: str,
    token: str | None,
    timeout: float,
    retries: int,
) -> tuple[str, list[tuple[str, int | None]]]:
    """Return (resolved_revision, [(path, size_or_None), ...])."""
    base = f"{endpoint}/api/models/{quote_repo(repo_id)}/revision/{urllib.parse.quote(revision, safe='')}"
    last_error: Exception | None = None

    for query in (_BLOBS_QUERY, _PLAIN_QUERY):
        try:
            metadata = request_json(base + query, token, timeout, retries)
        except ValueError as error:
            # Mirror-reported error (e.g. AUTH-ERR); the plain query will fail the same way.
            last_error = error
            continue
        except urllib.error.HTTPError as error:
            last_error = error
            if query == _BLOBS_QUERY and error.code in (400, 404, 405, 422, 500):
                continue
            raise

        resolved_revision = metadata.get("sha") or revision
        entries: list[tuple[str, int | None]] = []
        for item in metadata.get("siblings") or []:
            path = item.get("rfilename")
            if not path:
                continue
            size = item.get("size")
            entries.append((path, int(size) if isinstance(size, (int, float)) and size else None))
        return resolved_revision, entries

    raise ValueError(f"cannot list files for {repo_id} at {endpoint}: {last_error}")


def process_repo(
    repo_id: str,
    *,
    endpoint: str,
    revision: str,
    output_root: Path,
    token: str | None,
    include_weights: bool,
    workers: int,
    batch_size: int,
    timeout: float,
    retries: int,
    show_progress: bool,
) -> tuple[int, int, list[str], Path]:
    """Download one repo. Returns (ok, failed, excluded_weights, model_dir)."""
    organization, model_name = split_repo_id(repo_id)
    resolved_revision, entries = fetch_repo_files(endpoint, repo_id, revision, token, timeout, retries)

    if include_weights:
        selected, excluded = list(entries), []
    else:
        selected = [(path, size) for path, size in entries if not is_weight(path)]
        excluded = [path for path, _ in entries if is_weight(path)]

    model_dir = output_root / organization / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    total = len(selected)
    tracker = ProgressTracker(total, enabled=show_progress)
    for path, size in selected:
        tracker.register(path, size)

    results: dict[str, str] = {}
    if total == 0:
        tracker.log(f"[{repo_id}] nothing to download: no matching files in the repo listing.")
    else:
        tracker.log(
            f"[{repo_id}] {total} files @ {resolved_revision[:12]} from {endpoint} "
            f"(weights={'on' if include_weights else 'off'}, "
            f"excluded {len(excluded)} weight files, workers={workers}, batch={batch_size})"
        )
        common = dict(
            endpoint=endpoint, repo_id=repo_id, revision=resolved_revision,
            model_dir=model_dir, token=token, tracker=tracker,
            timeout=timeout, retries=retries,
        )
        batches = [selected[i:i + batch_size] for i in range(0, total, batch_size)]

        for batch_number, batch in enumerate(batches, start=1):
            batch_start = sum(len(b) for b in batches[: batch_number - 1])
            if workers == 1 or len(batch) == 1:
                for offset, (path, size) in enumerate(batch):
                    path_result, status = _download_one(
                        index=batch_start + offset + 1, total=total, path=path,
                        expected_size=size, **common,
                    )
                    results[path_result] = status
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(batch))) as executor:
                    futures = {
                        executor.submit(
                            _download_one, index=batch_start + offset + 1, total=total,
                            path=path, expected_size=size, **common,
                        ): path
                        for offset, (path, size) in enumerate(batch)
                    }
                    for future in concurrent.futures.as_completed(futures):
                        path_result, status = future.result()
                        results[path_result] = status

            batch_failed = sum(1 for status in results.values() if status.startswith("error:"))
            if len(batches) > 1:
                tracker.log(f"-- batch {batch_number}/{len(batches)} done "
                            f"({sum(1 for s in results.values() if not s.startswith('error:'))}/{total} ok, "
                            f"{batch_failed} failed)")

        tracker.finish()

    failed = {path: status for path, status in results.items() if status.startswith("error:")}
    succeeded = {path: status for path, status in results.items() if not status.startswith("error:")}

    manifest = {
        "repo_id": repo_id,
        "organization": organization,
        "model_name": model_name,
        "revision": resolved_revision,
        "endpoint": endpoint,
        "include_weights": include_weights,
        "downloaded_files": sorted(succeeded),
        "download_results": succeeded,
        "failed_files": failed,
        "excluded_weight_files": excluded,
    }
    (model_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if total > 0:
        tracker.log(f"[{repo_id}] {len(succeeded)}/{total} files OK, {len(failed)} failed -> {model_dir}")
    return len(succeeded), len(failed), excluded, model_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

EXAMPLES = """examples:
  %(prog)s -m deepseek-ai/DeepSeek-V4.1-Flash
        metadata + code + tokenizer only (no weights), from OpenCSG
  %(prog)s -m Qwen/Qwen3-8B -w
        include weight files
  %(prog)s -m a/b -m c/d -mirror hf-mirror -w
        two repos, from hf-mirror, with weights
  %(prog)s -m deepseek-ai/DeepSeek-V4.1-Flash --endpoint https://huggingface.co
        explicit endpoint (overrides -mirror)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_hf_model.py",
        description="Download Hugging Face model repos into models/<organization>/<name>/ "
                    "(weights skipped unless -w is given).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    parser.add_argument(
        "-m", "--model",
        dest="models",
        action="append",
        metavar="ORG/NAME",
        help="repository id, e.g. deepseek-ai/DeepSeek-V4.1-Flash; repeat -m to download several",
    )
    parser.add_argument(
        "-w", "--weights", "--include-weights",
        dest="weights",
        action="store_true",
        help="also download weight files (*.safetensors, *.bin, ...); default: skip them",
    )
    parser.add_argument(
        "-mirror", "--mirror",
        choices=sorted(MIRRORS),
        default=None,
        help=f"mirror preset, one of {', '.join(sorted(MIRRORS))} (default: {DEFAULT_MIRROR})",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help=f"explicit HF-compatible endpoint URL; overrides -mirror "
             f"(default: {DEFAULT_MIRROR} -> {DEFAULT_ENDPOINT})",
    )
    parser.add_argument("--revision", default="main", help="branch, tag, or commit (default: main)")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"destination root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="access token; defaults to HF_TOKEN (only needed for gated/private repos)",
    )
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"concurrent downloads within a batch (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"files submitted per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"per-connect and per-read timeout in seconds (default: {DEFAULT_TIMEOUT:g})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"max attempts per file, resuming on failure (default: {DEFAULT_RETRIES})",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=None,
        help="force the progress bar on (default: on when stderr is a terminal)",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="disable the progress bar",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.models:
        print("error: -m/--model is required, e.g. -m deepseek-ai/DeepSeek-V4.1-Flash", file=sys.stderr)
        return 2
    if args.workers < 1 or args.batch_size < 1 or args.retries < 1 or args.timeout <= 0:
        print("error: --workers, --batch-size, --retries must be >= 1 and --timeout must be > 0",
              file=sys.stderr)
        return 2

    try:
        endpoint = resolve_endpoint(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    show_progress = args.progress if args.progress is not None else sys.stderr.isatty()

    print(f"endpoint: {endpoint}")
    print(f"output:   {args.output_root}")
    print(f"models:   {', '.join(args.models)}")
    print(f"weights:  {'on' if args.weights else 'off'}   "
          f"workers: {args.workers}   batch: {args.batch_size}   "
          f"timeout: {args.timeout:g}s   retries: {args.retries}")
    print(flush=True)

    total_ok = 0
    total_fail = 0
    failed_models: list[str] = []

    try:
        for index, repo_id in enumerate(args.models, start=1):
            if len(args.models) > 1:
                print(f"=== [{index}/{len(args.models)}] {repo_id} ===", flush=True)
            try:
                ok, fail, _excluded, _ = process_repo(
                    repo_id,
                    endpoint=endpoint,
                    revision=args.revision,
                    output_root=args.output_root,
                    token=args.token,
                    include_weights=args.weights,
                    workers=args.workers,
                    batch_size=args.batch_size,
                    timeout=args.timeout,
                    retries=args.retries,
                    show_progress=show_progress,
                )
            except (OSError, ValueError, urllib.error.HTTPError, RetryExhausted) as error:
                print(f"[{repo_id}] failed: {error}", file=sys.stderr)
                failed_models.append(repo_id)
                continue
            total_ok += ok
            total_fail += fail
            if fail:
                failed_models.append(repo_id)
    except KeyboardInterrupt:
        print("\ninterrupted: partial .part files are kept and will resume on the next run.",
              file=sys.stderr)
        return 130

    print(f"\nDone. {total_ok} files OK, {total_fail} files failed.", flush=True)
    if failed_models:
        print("models with failures:", file=sys.stderr)
        for repo_id in failed_models:
            print(f"  - {repo_id}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
