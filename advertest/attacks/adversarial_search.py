import logging
import numpy as np
from typing import Any, Dict

logger = logging.getLogger(__name__)

class AdversarialSearchController:
    """
    Curriculum Learning & Dynamic Severity Search.
    Instead of randomly assigning severity 1-5 or relying on VLM text,
    this uses Binary Search to find the *minimum* severity required to 
    drop the YOLO IoU below 0.5. 
    Prevents gradient vanishing from generating overly destroyed images.
    """

    def __init__(self, attack_engine: Any, evaluator: Any):
        self.attack_engine = attack_engine
        self.evaluator = evaluator # E.g., Yolov7Evaluator instance

    def find_minimum_breaking_severity(self, image: np.ndarray, boxes: list, tag: str) -> int:
        """
        Binary search for severity [1, 2, 3, 4, 5].
        Returns the optimal severity, or 5 if it never breaks.
        """
        low = 1
        high = 5
        optimal_sev = 5
        
        # Get baseline prediction on the clean image (Assuming clean has high IoU)
        # clean_preds, clean_scores = self.evaluator.infer_single(image)
        
        while low <= high:
            mid = (low + high) // 2
            
            # Apply attack at 'mid' severity
            result = self.attack_engine.apply(image, boxes, tag=tag, severity=mid)
            
            if result.discarded:
                # BBox constraint failed at this severity, meaning it's too severe physically
                # Search lower
                high = mid - 1
                continue
                
            # Run inference on the attacked image
            pred_boxes, pred_scores = self.evaluator.infer_single(result.image)
            
            # In real implementation: Compare pred_boxes with result.boxes using compute_iou
            # We mock the IOU drop here: if severity >= breaking point, model fails.
            iou = self._mock_compute_iou(pred_boxes, result.boxes, severity=mid)
            
            if iou < 0.5:
                # Model broke! This severity is sufficient. Can we go lower?
                optimal_sev = mid
                high = mid - 1
            else:
                # Model survived. We need higher severity.
                low = mid + 1
                
        logger.info(f"Binary Search found optimal Severity {optimal_sev}/5 for tag {tag}")
        return optimal_sev

    def _mock_compute_iou(self, pred, gt, severity) -> float:
        """Mock function. High severity -> Low IoU."""
        # E.g., if severity is 3, iou is 0.4. If severity is 1, iou is 0.8
        return max(0.0, 1.0 - (severity * 0.25))
