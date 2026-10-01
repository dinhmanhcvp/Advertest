"""Stage 1: Error Extraction Pipeline.

Compares model predictions against ground truth to identify "Hard Negatives".
These errors are then categorized (or pushed to Label Studio for human
classification) to feed into the Attack Pool Factory.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import get_settings
from src.core.egocentric_types import ErrorCase, ErrorCategory, TemporalBoundary
from src.core.types import DetectionPrediction, Sample

logger = logging.getLogger(__name__)


class ErrorExtractor:
    """Extracts and classifies prediction errors."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.iou_threshold = self.settings.egocentric_error_iou_threshold

    def extract_pii_errors(
        self, predictions: list[DetectionPrediction], samples: list[Sample]
    ) -> list[ErrorCase]:
        """Extract spatial errors for PII detection."""
        errors: list[ErrorCase] = []
        
        sample_map = {s.sample_id: s for s in samples}
        
        for pred in predictions:
            sample = sample_map.get(pred.sample_id)
            if not sample:
                continue
                
            # Naive comparison: if a prediction has low IoU with any GT,
            # or if a GT is completely missed, it's a hard negative.
            # For brevity, this is a simplified simulation of the logic.
            
            for box in pred.boxes:
                # Find best matching GT
                best_iou = 0.0
                best_gt_label = "none"
                for gt in sample.boxes:
                    iou = self._box_iou(box, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_label = gt.label
                        
                if best_iou < self.iou_threshold:
                    error = ErrorCase(
                        case_id=f"err-pii-{pred.sample_id}-{len(errors)}",
                        sample_id=pred.sample_id,
                        error_category=self.classify_pii_error(sample),
                        iou=best_iou,
                        confidence=box.score,
                        predicted_label=box.label,
                        ground_truth_label=best_gt_label,
                        severity_estimate=3,  # placeholder
                    )
                    errors.append(error)
                    
                # NEW METRIC: High Confidence False Positive (Background Clutter)
                # Confidence > 0.8 but IoU is exactly 0.0
                elif box.score > 0.8 and best_iou == 0.0:
                    error = ErrorCase(
                        case_id=f"err-fp-{pred.sample_id}-{len(errors)}",
                        sample_id=pred.sample_id,
                        error_category="semantic_confusion",
                        iou=0.0,
                        confidence=box.score,
                        predicted_label=box.label,
                        ground_truth_label="none",
                        severity_estimate=5,  # Needs high severity retraining
                    )
                    errors.append(error)

        return errors

    def extract_temporal_errors(
        self, predictions: list[Any], gt_boundaries: list[TemporalBoundary]
    ) -> list[ErrorCase]:
        """Extract temporal errors for Video Semantic Labeling.
        
        In a real implementation, predictions would be Action segments.
        """
        errors: list[ErrorCase] = []
        
        # NEW METRIC: Temporal Consistency Error (Flickering)
        # If the model predicts an action, then misses it, then predicts it again.
        # This simulates comparing prediction boundary continuity against ground truth.
        for pred in predictions:
            # Assuming pred is a TemporalBoundary for simulation
            for gt in gt_boundaries:
                # If prediction is fragmented but gt is continuous
                if gt.duration_frames > 30 and hasattr(pred, "duration_frames") and pred.duration_frames < 10:
                    # Simulated flickering condition
                    errors.append(
                        ErrorCase(
                            case_id=f"err-temporal-{gt.action_label}-{len(errors)}",
                            sample_id="video_stream",
                            error_category="semantic_temporal_break",
                            iou=pred.iou(gt) if hasattr(pred, "iou") else 0.0,
                            confidence=getattr(pred, "confidence", 0.0),
                            predicted_label=getattr(pred, "action_label", "none"),
                            ground_truth_label=gt.action_label,
                            severity_estimate=4,
                        )
                    )
        
        return errors

    def classify_pii_error(self, sample: Sample) -> ErrorCategory:
        """Auto-classify the error cause based on image heuristics.
        
        Falls back to 'pii_blur' if undetermined (to be corrected in Label Studio).
        """
        # A real implementation would run Laplacian variance for blur,
        # brightness histograms for exposure, or metadata analysis for fisheye.
        
        if sample.meta.get("lens_type") == "fisheye":
            return "pii_fisheye"
            
        # Default fallback
        return "pii_blur"

    @staticmethod
    def _box_iou(box1: Any, box2: Any) -> float:
        """Calculate intersection over union for two boxes."""
        x_left = max(box1.x1, box2.x1)
        y_top = max(box1.y1, box2.y1)
        x_right = min(box1.x2, box2.x2)
        y_bottom = min(box1.y2, box2.y2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        iou = intersection_area / float(box1_area + box2_area - intersection_area + 1e-6)
        return iou
