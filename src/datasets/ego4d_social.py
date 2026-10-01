"""Ego4D Social task dataset adapter."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

from src.core.egocentric_types import TemporalAction, TemporalBoundary, VideoSample
from src.core.types import Task
from src.datasets import DATASETS
from src.datasets.base import BaseDataset
from src.config import get_settings


@DATASETS.register
class Ego4DSocialDataset(BaseDataset):
    """Ego4D Social interactions dataset."""

    name: ClassVar[str] = "ego4d_social"
    task: ClassVar[Task] = "video_semantic"
    version: ClassVar[str] = "v1"
    owner: ClassVar[str] = "egocentric"

    def __init__(self, data_root: str | None = None, split: str = "train") -> None:
        super().__init__()
        settings = get_settings()
        self.data_root = Path(data_root or getattr(settings, "ego4d_data_root", "data/ego4d"))
        self.split = split

    def __iter__(self) -> Generator[VideoSample, None, None]:
        # Ego4D Social has a primary JSON file with annotations
        annot_file = self.data_root / "annotations" / f"social_{self.split}.json"
        
        if not annot_file.exists():
            print(f"Warning: Ego4D annotations not found at {annot_file}")
            return
            
        with open(annot_file, "r") as f:
            data = json.load(f)
            
        video_dir = self.data_root / "v1" / "clips"
        
        # Structure of Ego4D social json (simplified for the adapter)
        for video_id, video_data in data.get("videos", {}).items():
            video_path = video_dir / f"{video_id}.mp4"
            
            if not video_path.exists():
                continue
                
            actions = []
            boundaries = []
            
            # Example parsing of temporal labels (conversations, looking-at)
            for clip in video_data.get("clips", []):
                for segment in clip.get("social_segments", []):
                    start_sec = segment.get("start_time", 0.0)
                    end_sec = segment.get("end_time", 0.0)
                    label = segment.get("action_type", "social_interaction")
                    
                    # Assume 30fps for default conversion
                    start_frame = int(start_sec * 30)
                    end_frame = int(end_sec * 30)
                    
                    boundary = TemporalBoundary(
                        start_frame=start_frame,
                        end_frame=end_frame,
                        action_label=label
                    )
                    boundaries.append(boundary)
                    
                    actions.append(
                        TemporalAction(
                            boundary=boundary,
                            verb=label,
                            narration=segment.get("description", "")
                        )
                    )
                    
            # For a proper implementation, we would extract frames using DataConnector
            # For the dataset iterator, we will yield a dummy VideoSample with the correct metadata
            # and let the VideoProcessor handle frame extraction when needed to save memory.
            import numpy as np
            yield VideoSample(
                sample_id=video_id,
                frames=np.zeros((0, 720, 1280, 3), dtype=np.float32), # Deferred loading
                fps=30.0,
                temporal_boundaries=tuple(boundaries),
                actions=tuple(actions),
                meta={"video_path": str(video_path)}
            )

    def __len__(self) -> int:
        annot_file = self.data_root / "annotations" / f"social_{self.split}.json"
        if not annot_file.exists():
            return 0
        with open(annot_file, "r") as f:
            data = json.load(f)
            return len(data.get("videos", {}))
