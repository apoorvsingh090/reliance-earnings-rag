"""Resilient file downloader.

NSE archives reject bare HEAD requests and non-browser user agents, so we
always use GET with a browser UA, stream to disk, and retry transiently.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {"User-Agent": BROWSER_UA, "Accept": "text/html,application/pdf,*/*"}


class DownloadError(RuntimeError):
    pass


def download_file(
    url: str,
    dest: Path,
    *,
    timeout: int = 60,
    min_bytes: int = 1_000,
    retries: int = 3,
    backoff_s: float = 2.0,
) -> Path:
    """Download URL to dest. Returns dest. Raises DownloadError on failure.

    Idempotency: if dest already exists with >= min_bytes, the download is
    skipped (caller still re-hashes + re-validates the file).
    """
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(
                url, headers=DEFAULT_HEADERS, timeout=timeout, stream=True
            ) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=256 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
            size = dest.stat().st_size
            if size < min_bytes:
                raise DownloadError(f"downloaded only {size} bytes from {url}")
            return dest
        except Exception as exc:  # noqa: BLE001 — retry any transient failure
            last_err = exc
            if attempt < retries:
                time.sleep(backoff_s * attempt)
    raise DownloadError(f"failed to download {url}: {last_err}")
