"""Video Processing Engine.

Integrates DataConnector (PyAV) and the Attack Registry to decode videos,
apply temporal attacks across frames, and reconstruct the output video.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.attacks import get_attack
from src.attacks.base import AttackContext
from src.core.egocentric_types import TemporalBoundary, VideoSample
from src.core.types import Sample
from src.services.data_connector import DataConnector

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Decodes, attacks, and encodes videos for temporal pipelines."""

    def __init__(self, connector: DataConnector | None = None) -> None:
        self.connector = connector or DataConnector()
        # In a real setup, we would use av.open(mode='w') to write video.
        # This module handles the high-level orchestration.

    def process_video_attack(
        self, 
        video_path: str, 
        attack_name: str, 
        severity: int,
        output_path: str,
        fps: float = 30.0,
    ) -> bool:
        """End-to-end processing of a single video through an attack."""
        logger.info(f"Applying {attack_name} (L{severity}) to {video_path}")
        
        # 1. Extract frames to memory (or temp disk if too large)
        frames = list(self.connector.stream_frames(video_path))
        if not frames:
            logger.error("Failed to extract frames.")
            return False
            
        frame_array = np.stack(frames)
        
        # 2. Package into a dummy Sample for the attack interface
        sample = Sample(
            sample_id=Path(video_path).stem,
            image=frame_array[0],
            meta={"video_frames": frame_array}
        )
        
        # 3. Apply the attack
        attack = get_attack(attack_name)
        ctx = AttackContext(rng=np.random.default_rng())
        attacked = attack.apply(sample, severity, ctx)
        
        # 4. Reconstruct video from output frames
        attacked_frames = attacked.meta.get("video_frames")
        if attacked_frames is None or not isinstance(attacked_frames, np.ndarray):
            logger.error("Attack failed to return video frames.")
            return False
            
        self.reconstruct_video(attacked_frames, output_path, fps)
        return True

    def reconstruct_video(self, frames: np.ndarray, output_path: str, fps: float) -> None:
        """Encode float32 frames back to an MP4 video file.
        
        Uses PyAV to write H.264 video.
        """
        import av
        
        container = av.open(output_path, mode='w')
        stream = container.add_stream('libx264', rate=int(fps))
        
        height, width = frames.shape[1:3]
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        
        # Convert float32 [0, 1] back to uint8 [0, 255]
        frames_uint8 = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
        
        for i in range(frames_uint8.shape[0]):
            frame = av.VideoFrame.from_ndarray(frames_uint8[i], format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
                
        # Flush stream
        for packet in stream.encode():
            container.mux(packet)
            
        container.close()
        logger.info(f"Reconstructed video saved to {output_path}")

    def extract_action_segments(
        self, video_path: str, boundaries: list[TemporalBoundary]
    ) -> list[VideoSample]:
        """Chop a long video into shorter VideoSamples based on action boundaries."""
        frames = list(self.connector.stream_frames(video_path))
        frame_array = np.stack(frames)
        
        samples = []
        for i, b in enumerate(boundaries):
            if b.start_frame < 0 or b.end_frame > frame_array.shape[0]:
                continue
                
            segment = frame_array[b.start_frame:b.end_frame]
            sample = VideoSample(
                sample_id=f"{Path(video_path).stem}-seg{i}",
                frames=segment,
                temporal_boundaries=(b,),
            )
            samples.append(sample)
            
        return samples
