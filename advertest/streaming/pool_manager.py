"""Streaming Attack Pool Manager.

A high-speed state manager that accumulates adversarial frames generated from the 
real-time Kafka stream. It tracks the size of the memory/disk buffer and triggers
callbacks when deployment thresholds are met.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from advertest.core.inference_engine import YOLOBox

logger = logging.getLogger(__name__)


class StreamingPoolManager:
    """Manages the lifecycle of continuous adversarial data streams."""

    def __init__(self, storage_dir: str | Path, trigger_threshold: int = 10000) -> None:
        """
        Parameters
        ----------
        storage_dir : str | Path
            High-speed local SSD storage path for the buffer.
        trigger_threshold : int
            Number of adversarial samples required to trigger a Ray Train job.
        """
        self.storage_dir = Path(storage_dir)
        self.trigger_threshold = trigger_threshold
        
        self.images_dir = self.storage_dir / "images"
        self.labels_dir = self.storage_dir / "labels"
        
        # Ensure clean state
        self.reset_pool()
        
        self.current_count = 0
        self._threshold_callback: Optional[Callable[[Path], None]] = None

    def reset_pool(self) -> None:
        """Clear the storage buffer."""
        if self.storage_dir.exists():
            shutil.rmtree(self.storage_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.current_count = 0
        logger.info("Streaming Pool Manager buffer reset at %s", self.storage_dir)

    def register_threshold_callback(self, callback: Callable[[Path], None]) -> None:
        """Register the function to call when the pool hits the trigger threshold."""
        self._threshold_callback = callback

    def add_sample(self, image: np.ndarray, boxes: List[YOLOBox], metadata: dict) -> None:
        """Add a generated adversarial frame to the high-speed buffer.
        
        If the threshold is reached, this will automatically fire the registered callback.
        """
        # Generate a unique ID based on timestamp or Kafka metadata
        sample_id = metadata.get("frame_id", f"stream_{self.current_count:07d}")
        
        img_path = self.images_dir / f"{sample_id}.jpg"
        lbl_path = self.labels_dir / f"{sample_id}.txt"
        
        # Write to disk buffer
        cv2.imwrite(str(img_path), image)
        with open(lbl_path, "w") as f:
            for box in boxes:
                f.write(box.to_label_str() + "\n")
                
        self.current_count += 1
        
        if self.current_count % 1000 == 0:
            logger.info("Streaming Pool Status: %d / %d samples collected.", 
                        self.current_count, self.trigger_threshold)
            
        # Check threshold trigger
        if self.current_count >= self.trigger_threshold:
            logger.warning("THRESHOLD REACHED (%d samples). Triggering training callback!", 
                           self.trigger_threshold)
            
            if self._threshold_callback:
                # We pass the completed storage directory to the Ray Train trigger
                self._threshold_callback(self.storage_dir)
            else:
                logger.error("Threshold reached, but no callback registered!")
                
            # Note: In a production CT pipeline, we might rotate directories
            # instead of resetting, but for this implementation we reset.
            # We assume the callback is synchronous and handles the payload copying.
            self.reset_pool()
