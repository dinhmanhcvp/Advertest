"""YOLOv7-face adapter for face detection."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

import numpy as np

from src.adapters import MODELS
from src.adapters.base import ModelAdapter
from src.core.types import Box, DetectionPrediction, ModelInfo, Sample, Task
from src.config import get_settings


@MODELS.register
class YOLOv7FaceAdapter(ModelAdapter):
    """YOLOv7-face detection adapter (derronqi)."""

    name: ClassVar[str] = "yolov7_face"
    task: ClassVar[Task] = "pii_detection"
    version: ClassVar[str] = "v0.1"
    supports_gradients: ClassVar[bool] = False
    owner: ClassVar[str] = "egocentric"

    def __init__(
        self,
        *,
        weights: str | None = None,
        score_threshold: float = 0.25,
        max_detections: int = 100,
    ) -> None:
        super().__init__(score_threshold=score_threshold, max_detections=max_detections)
        settings = get_settings()
        self.weights = weights or getattr(settings, "yolov7_face_weights_path", "data/weights/yolov7-lite-t.pt")
        self._model: Any | None = None
        self._device: Any | None = None
        
        # Add third_party to sys.path to allow importing from the cloned repo
        repo_dir = Path(__file__).parent.parent / "third_party" / "yolov7_face"
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            task=self.task,
            version=f"{self.version}:{self.weights}",
            supports_gradients=self.supports_gradients,
            classes=("Face",),
        )

    def predict(self, samples: Sequence[Sample]) -> list[DetectionPrediction]:
        model = self._load()
        import torch
        from utils.general import non_max_suppression_face, scale_coords
        from utils.datasets import letterbox
        
        predictions: list[DetectionPrediction] = []
        for sample in samples:
            started = perf_counter()
            
            # Preprocess image
            img0 = (sample.image * 255.0).round().astype(np.uint8)
            img = letterbox(img0, 640, stride=32, auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            
            tensor_img = torch.from_numpy(img).to(self._device)
            tensor_img = tensor_img.float() / 255.0
            if len(tensor_img.shape) == 3:
                tensor_img = tensor_img[None]
                
            # Inference
            with torch.no_grad():
                pred = model(tensor_img)[0]
                
            # Postprocess
            pred = non_max_suppression_face(pred, self.score_threshold, 0.45)
            
            boxes: list[Box] = []
            if len(pred) > 0 and pred[0] is not None:
                det = pred[0].cpu().numpy()
                # det format: [x1, y1, x2, y2, conf, cls, landmarks...]
                
                # Scale boxes back to original image size
                det[:, :4] = scale_coords(tensor_img.shape[2:], det[:, :4], img0.shape).round()
                
                for *xyxy, conf, cls, *landmarks in det:
                    if conf > self.score_threshold:
                        boxes.append(
                            Box(
                                x1=float(xyxy[0]), 
                                y1=float(xyxy[1]), 
                                x2=float(xyxy[2]), 
                                y2=float(xyxy[3]), 
                                label="Face", 
                                score=float(conf)
                            )
                        )
                        
            # Sort by score descending and truncate
            boxes.sort(key=lambda b: b.score, reverse=True)
            boxes = boxes[: self.max_detections]
            
            predictions.append(
                DetectionPrediction(
                    sample_id=sample.sample_id,
                    boxes=tuple(boxes),
                    latency_ms=(perf_counter() - started) * 1000.0,
                )
            )
            
        return predictions

    def _load(self) -> Any:
        if self._model is None:
            import torch
            from models.experimental import attempt_load
            
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weights_path = Path(self.weights)
            
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights not found at {weights_path}. Please run download_face_data.py")
                
            self._model = attempt_load(str(weights_path), map_location=self._device)
            self._model.eval()
            
        return self._model
