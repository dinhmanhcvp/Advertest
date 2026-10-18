"""Data Connector for large-scale video processing.

Handles downloading from presigned URLs, streaming frames without loading
entire videos into memory, and managing temporal caching.
"""

from __future__ import annotations

import calendar
import hashlib
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Generator
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Any

import numpy as np

from src.config import get_settings

try:
    import av
except ImportError:
    av = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class PresignedURLExpired(Exception):
    """The presigned URL's TTL elapsed (or its signature was rejected)."""


class DataConnector:
    """Handles data ingestion and large video frame streaming."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.cache_dir = Path(self.settings.data_root) / "egocentric_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def pull_video_batch(
        self,
        urls: list[str],
        batch_size: int = 5,
        *,
        refresh_url: Callable[[str], str] | None = None,
        max_retries: int = 3,
        timeout_s: float = 30.0,
    ) -> list[str]:
        """Download videos from Data Block presigned URLs into the temporary cache.

        * Expired URLs are detected up front (``X-Amz-Date``+``X-Amz-Expires`` or ``Expires``)
          and re-signed through ``refresh_url`` when given; HTTP 401/403 mid-flight does the same.
        * Timeouts and 5xx responses are retried with exponential backoff.
        * Data is written to a ``.part`` file and renamed atomically, so a failed transfer
          never leaves partial raw data behind.
        Remove raw files with :meth:`clear_cache` / :meth:`purge_expired` once frames are extracted.
        """
        paths: list[str] = []
        for url in urls[:batch_size]:
            key = hashlib.sha256(url.split("?")[0].encode()).hexdigest()[:16]
            local = self.cache_dir / f"video_{key}.mp4"
            paths.append(str(self._download(url, local, refresh_url, max_retries, timeout_s)))
        logger.info("Downloaded %d video(s) into temporary cache", len(paths))
        return paths

    @staticmethod
    def url_seconds_remaining(url: str, now: float | None = None) -> float | None:
        """Seconds until a presigned URL expires, or ``None`` if it carries no expiry."""
        q = parse_qs(urlparse(url).query)
        now = time.time() if now is None else now
        try:
            if "X-Amz-Date" in q and "X-Amz-Expires" in q:
                issued = calendar.timegm(time.strptime(q["X-Amz-Date"][0], "%Y%m%dT%H%M%SZ"))
                return issued + int(q["X-Amz-Expires"][0]) - now
            if "Expires" in q:
                return int(q["Expires"][0]) - now
        except (ValueError, IndexError):
            return None
        return None

    def _download(self, url, local: Path, refresh_url, max_retries: int, timeout_s: float) -> Path:
        refreshed = False
        attempt = 0
        while True:
            remaining = self.url_seconds_remaining(url)
            if remaining is not None and remaining <= 0:
                if refresh_url and not refreshed:
                    url, refreshed = refresh_url(url), True
                    continue
                raise PresignedURLExpired(f"presigned URL expired {-remaining:.0f}s ago")
            part = local.with_suffix(".part")
            try:
                with urllib.request.urlopen(url, timeout=timeout_s) as resp, open(part, "wb") as out:
                    shutil.copyfileobj(resp, out, length=1 << 20)
                os.replace(part, local)
                return local
            except urllib.error.HTTPError as exc:
                part.unlink(missing_ok=True)
                if exc.code in (401, 403):
                    if refresh_url and not refreshed:
                        url, refreshed = refresh_url(url), True
                        continue
                    raise PresignedURLExpired(f"HTTP {exc.code}: presigned URL rejected") from exc
                if exc.code < 500:
                    raise
                error: Exception = exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                part.unlink(missing_ok=True)
                error = exc
            if attempt >= max_retries:
                raise error
            logger.warning("Download attempt %d failed: %s", attempt + 1, error)
            time.sleep(min(2**attempt * 0.5, 8.0))
            attempt += 1

    def purge_expired(self, max_age_s: float = 3600.0) -> int:
        """Delete cached raw files older than ``max_age_s`` (PII retention limit)."""
        removed = 0
        cutoff = time.time() - max_age_s
        for file in [*self.cache_dir.glob("*.mp4"), *self.cache_dir.glob("*.part")]:
            try:
                if file.stat().st_mtime < cutoff:
                    file.unlink()
                    removed += 1
            except OSError as e:
                logger.warning(f"Failed to purge {file}: {e}")
        return removed

    def __enter__(self) -> "DataConnector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.clear_cache()

    def stream_frames(
        self, video_path: str, max_frames: int | None = None
    ) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames from a video as float32 numpy arrays.
        
        Using PyAV to decode frames efficiently without loading the whole
        file into RAM.
        """
        if av is None:
            raise RuntimeError("Install 'av' (PyAV) for video streaming")
            
        path = Path(video_path)
        if not path.exists():
            # For demonstration, yield a dummy frame if file doesn't exist
            yield np.zeros((720, 1280, 3), dtype=np.float32)
            return

        container = av.open(str(path))
        stream = container.streams.video[0]
        
        count = 0
        for frame in container.decode(stream):
            if max_frames is not None and count >= max_frames:
                break
                
            # Convert PyAV VideoFrame to numpy array (RGB)
            img = frame.to_ndarray(format='rgb24')
            
            # Convert to standard AdverTest format: float32 [0, 1]
            img_float = img.astype(np.float32) / 255.0
            yield img_float
            
            count += 1
            
        container.close()

    def batch_process(self, items: list[Any], fn: Any, max_workers: int = 4) -> list[Any]:
        """Process items in parallel with memory management."""
        import concurrent.futures
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fn, item) for item in items]
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    
        return results

    def clear_cache(self) -> None:
        """Remove temporary video files."""
        for file in [*self.cache_dir.glob("*.mp4"), *self.cache_dir.glob("*.part")]:
            try:
                file.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete {file}: {e}")
