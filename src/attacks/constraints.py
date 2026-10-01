"""Constraints and Data Integrity Engine.

Provides the ExclusionMatrix and BBoxIntegrityChecker to ensure that synthesized
adversarial data is physically plausible and does not introduce label noise.
"""

from typing import List
from src.core.types import Sample

class ExclusionMatrix:
    """Prevents physically impossible combinations of corruptions."""
    
    # Define conflicting tags (e.g., cannot be both pitch black and glaring sun)
    CONFLICTS = {
        frozenset({"error_lowlight", "error_flare"}),
        frozenset({"error_rain", "error_flare"}),
    }
    
    @classmethod
    def check_conflicts(cls, active_tags: List[str]) -> bool:
        """Return True if the active tags contain a conflict, False otherwise."""
        tags_set = set(active_tags)
        for conflict_pair in cls.CONFLICTS:
            if conflict_pair.issubset(tags_set):
                return True
        return False

class BBoxIntegrityChecker:
    """Checks if bounding boxes remain valid after geometric distortion."""
    
    @classmethod
    def check(cls, original: Sample, attacked: Sample, max_drift: float = 0.5) -> bool:
        """
        Verify that bounding boxes haven't drifted out of frame significantly.
        Returns False if the sample should be DISCARDED.
        
        Note: In a true geometric transform (like GridDistortion), we should warp the 
        bboxes directly. For this prototype, we ensure the boxes are strictly 
        within image bounds, or we simulate a check.
        """
        if not original.boxes or not attacked.boxes:
            return True
            
        # For prototype simplicity: if bounding box coordinates go completely
        # out of bounds or shrink to zero area, we discard.
        h, w = attacked.image.shape[:2]
        
        valid_boxes = 0
        for box in attacked.boxes:
            # Clip box to image dimensions
            x1 = max(0, min(w, box.x1))
            y1 = max(0, min(h, box.y1))
            x2 = max(0, min(w, box.x2))
            y2 = max(0, min(h, box.y2))
            
            area = (x2 - x1) * (y2 - y1)
            original_area = (box.x2 - box.x1) * (box.y2 - box.y1)
            
            if original_area > 0 and (area / original_area) > (1.0 - max_drift):
                valid_boxes += 1
                
        # If no valid boxes remain but there were boxes originally, discard.
        if valid_boxes == 0 and len(original.boxes) > 0:
            return False
            
        return True
