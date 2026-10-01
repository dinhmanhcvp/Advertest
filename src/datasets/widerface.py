"""WIDERFACE Dataset adapter."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

from src.core.types import Box, Sample, Task
from src.datasets import DATASETS
from src.datasets.base import BaseDataset
from src.config import get_settings


@DATASETS.register
class WIDERFaceDataset(BaseDataset):
    """WIDERFACE benchmark dataset for face detection."""

    name: ClassVar[str] = "widerface"
    task: ClassVar[Task] = "pii_detection"
    version: ClassVar[str] = "1.0.0"
    owner: ClassVar[str] = "egocentric"

    def __init__(self, data_root: str | None = None, split: str = "train") -> None:
        super().__init__()
        settings = get_settings()
        self.data_root = Path(data_root or getattr(settings, "widerface_data_root", "data/widerface"))
        self.split = split
        self._samples_cache: list[tuple[str, list[Box]]] | None = None

    def _load_annotations(self) -> list[tuple[str, list[Box]]]:
        if self._samples_cache is not None:
            return self._samples_cache

        annot_file = self.data_root / "wider_face_split" / f"wider_face_{self.split}_bbx_gt.txt"
        if not annot_file.exists():
            print(f"Warning: WIDERFACE annotations not found at {annot_file}")
            return []

        samples = []
        with open(annot_file, "r") as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            img_path = lines[i].strip()
            i += 1
            
            if i >= len(lines):
                break
                
            num_boxes_str = lines[i].strip()
            try:
                num_boxes = int(num_boxes_str)
            except ValueError:
                num_boxes = 0
                
            i += 1
            
            boxes = []
            if num_boxes == 0:
                # WIDERFACE convention: 0 boxes is sometimes followed by a single line of zeros
                i += 1
            else:
                for _ in range(num_boxes):
                    parts = list(map(int, lines[i].strip().split()[:4]))
                    x1, y1, w, h = parts
                    boxes.append(
                        Box(
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x1 + w),
                            y2=float(y1 + h),
                            label="Face"
                        )
                    )
                    i += 1
            
            samples.append((img_path, boxes))
            
        self._samples_cache = samples
        return samples

    def __iter__(self) -> Generator[Sample, None, None]:
        samples = self._load_annotations()
        
        img_dir = self.data_root / f"WIDER_{self.split}" / "images"
        
        for img_rel_path, boxes in samples:
            img_path = img_dir / img_rel_path
            if not img_path.exists():
                continue
                
            import numpy as np
            from PIL import Image
            
            try:
                img = Image.open(img_path).convert("RGB")
                img_array = np.array(img, dtype=np.float32) / 255.0
            except Exception:
                continue
                
            yield Sample(
                sample_id=str(Path(img_rel_path).with_suffix('')),
                image=img_array,
                boxes=tuple(boxes),
            )

    def __len__(self) -> int:
        return len(self._load_annotations())
