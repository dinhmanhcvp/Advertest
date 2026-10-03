"""Integrity module to ensure annotations remain valid after adversarial attacks."""

from __future__ import annotations

from src.core.egocentric_types import TemporalBoundary
from src.core.types import Box, Sample


class BBoxIntegrityChecker:
    """Validates and corrects bounding boxes after spatial/geometric attacks."""

    @staticmethod
    def check_and_pad(box: Box, image_width: int, image_height: int) -> Box | None:
        """
        Check if a bounding box has been pushed out of bounds.
        If it's 100% out of bounds, returns None (discard).
        If partially out, clamp to the edges of the image.
        """
        # Original area
        orig_area = box.width * box.height
        if orig_area <= 0:
            return None

        # Clamp to image boundaries
        x1 = max(0.0, min(box.x1, float(image_width)))
        y1 = max(0.0, min(box.y1, float(image_height)))
        x2 = max(0.0, min(box.x2, float(image_width)))
        y2 = max(0.0, min(box.y2, float(image_height)))

        new_width = x2 - x1
        new_height = y2 - y1

        if new_width <= 0 or new_height <= 0:
            return None

        new_area = new_width * new_height
        
        # If box lost more than 50% of its area, discard
        if new_area / orig_area < 0.5:
            return None

        return Box(
            x1=x1, y1=y1, x2=x2, y2=y2,
            label=box.label,
            score=box.score
        )

    @classmethod
    def apply_to_sample(cls, sample: Sample) -> Sample | None:
        """Apply checker to all boxes in a sample. Returns None if ALL boxes are invalid."""
        if not sample.boxes:
            return sample

        H, W = sample.shape[:2]
        new_boxes = []
        for box in sample.boxes:
            checked_box = cls.check_and_pad(box, W, H)
            if checked_box is not None:
                new_boxes.append(checked_box)

        # Discard the whole sample if it lost all its GT faces due to extreme warping
        if not new_boxes:
            return None

        from dataclasses import replace
        return replace(sample, boxes=tuple(new_boxes))


class TemporalBoundaryWarper:
    """Warp temporal boundaries (action segments) when frames are dropped/added."""

    @staticmethod
    def warp_boundaries(
        boundaries: tuple[TemporalBoundary, ...],
        dropped_indices: list[int]
    ) -> tuple[TemporalBoundary, ...]:
        """
        Given a list of original frame indices that were dropped,
        recalculate the new start and end frames for action boundaries.
        """
        if not dropped_indices:
            return boundaries

        # Sort for fast processing
        dropped_sorted = sorted(dropped_indices)

        new_boundaries = []
        for b in boundaries:
            start = b.start_frame
            end = b.end_frame

            # Count how many drops occurred BEFORE start
            drops_before_start = sum(1 for d in dropped_sorted if d < start)
            new_start = max(0, start - drops_before_start)

            # Count drops BEFORE or AT end
            drops_before_end = sum(1 for d in dropped_sorted if d < end)
            new_end = max(0, end - drops_before_end)

            # If the segment was entirely dropped (start == end), discard
            if new_end > new_start:
                new_boundaries.append(
                    TemporalBoundary(
                        start_frame=new_start,
                        end_frame=new_end,
                        action_label=b.action_label,
                        confidence=b.confidence,
                        instance_id=b.instance_id
                    )
                )

        return tuple(new_boundaries)
