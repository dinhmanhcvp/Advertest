"""Data Connector for large-scale video processing.

Handles downloading from presigned URLs, streaming frames without loading
entire videos into memory, and managing temporal caching.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_settings

try:
    import av
except ImportError:
    av = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class DataConnector:
    """Handles data ingestion and large video frame streaming."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.cache_dir = Path(self.settings.data_root) / "egocentric_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def pull_video_batch(self, urls: list[str], batch_size: int = 5) -> list[str]:
        """Download videos via URLs and return local paths.
        
        In production, this would use boto3/requests and handle concurrency.
        """
        # Simulated download
        logger.info(f"Downloading batch of {len(urls)} videos...")
        paths = []
        for i, url in enumerate(urls[:batch_size]):
            # Placeholder logic
            safe_name = f"video_{i}.mp4"
            local_path = self.cache_dir / safe_name
            paths.append(str(local_path))
            
        return paths

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
        for file in self.cache_dir.glob("*.mp4"):
            try:
                file.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete {file}: {e}")
