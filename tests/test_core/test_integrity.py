"""Unit Tests for Integrity Module (BBox and Temporal Warping)."""

import pytest

from src.core.egocentric_types import TemporalBoundary
from src.core.integrity import BBoxIntegrityChecker, TemporalBoundaryWarper
from src.core.types import Box, Sample


def test_bbox_integrity_clamp_and_discard():
    """Test that out-of-bounds boxes are clamped, and 100% OOB are discarded."""
    W, H = 640, 480
    
    # 1. Valid Box
    box_valid = Box(x1=100.0, y1=100.0, x2=200.0, y2=200.0, label="Face")
    res_valid = BBoxIntegrityChecker.check_and_pad(box_valid, W, H)
    assert res_valid is not None
    assert res_valid.x1 == 100.0
    assert res_valid.x2 == 200.0
    
    # 2. Partially OOB (Clamping expected, area > 50% remains)
    # Area = 200 * 200 = 40000. 
    # Clamped Area = 100 * 200 = 20000. Ratio = 0.5. Since it's exactly 0.5, our logic returns it.
    box_partial = Box(x1=-100.0, y1=100.0, x2=100.0, y2=300.0, label="Face")
    res_partial = BBoxIntegrityChecker.check_and_pad(box_partial, W, H)
    assert res_partial is not None
    assert res_partial.x1 == 0.0
    assert res_partial.x2 == 100.0
    
    # 3. Partially OOB (Discarded because Area < 50%)
    # Area = 200 * 200 = 40000. Clamped Area = 50 * 200 = 10000. Ratio = 0.25 -> Discard
    box_discard = Box(x1=-150.0, y1=100.0, x2=50.0, y2=300.0, label="Face")
    assert BBoxIntegrityChecker.check_and_pad(box_discard, W, H) is None
    
    # 4. 100% OOB
    box_oob = Box(x1=700.0, y1=100.0, x2=800.0, y2=200.0, label="Face")
    assert BBoxIntegrityChecker.check_and_pad(box_oob, W, H) is None


def test_bbox_integrity_sample_apply():
    """Test applying integrity checker to an entire sample."""
    sample = Sample(
        sample_id="test",
        image=None,  # Dummy
        boxes=(
            Box(x1=100, y1=100, x2=200, y2=200, label="F1"),
            Box(x1=1000, y1=1000, x2=1100, y2=1100, label="F2"), # OOB
        )
    )
    # Assume 640x480 resolution through a mock shape
    sample = type("MockSample", (), {"shape": (480, 640, 3), "boxes": sample.boxes, "sample_id": "test", "with_image": sample.with_image})()
    
    # Mocking since Sample is a dataclass and we bypassed image shape. 
    # Let's just create a real numpy array.
    import numpy as np
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    real_sample = Sample(sample_id="t", image=img, boxes=(
        Box(x1=10, y1=10, x2=50, y2=50, label="F1"),
        Box(x1=700, y1=700, x2=800, y2=800, label="F2"), # OOB
    ))
    
    res = BBoxIntegrityChecker.apply_to_sample(real_sample)
    assert res is not None
    assert len(res.boxes) == 1
    assert res.boxes[0].label == "F1"
    
    # All OOB -> None
    all_oob = Sample(sample_id="t2", image=img, boxes=(
        Box(x1=700, y1=700, x2=800, y2=800, label="F2"),
    ))
    assert BBoxIntegrityChecker.apply_to_sample(all_oob) is None


def test_temporal_boundary_warping():
    """Test that dropped frames correctly recalculate temporal boundaries."""
    boundaries = (
        TemporalBoundary(start_frame=10, end_frame=20, action_label="Run"),
        TemporalBoundary(start_frame=30, end_frame=40, action_label="Jump")
    )
    
    # Drop frames: 5 (before both), 15 (inside Run), 25 (between), 35 (inside Jump)
    dropped_indices = [5, 15, 25, 35]
    
    res = TemporalBoundaryWarper.warp_boundaries(boundaries, dropped_indices)
    
    # 1. Run (10 to 20):
    # drops before start (5) -> start shifts by 1 -> 9
    # drops before end (5, 15) -> end shifts by 2 -> 18
    assert res[0].start_frame == 9
    assert res[0].end_frame == 18
    
    # 2. Jump (30 to 40):
    # drops before start (5, 15, 25) -> start shifts by 3 -> 27
    # drops before end (5, 15, 25, 35) -> end shifts by 4 -> 36
    assert res[1].start_frame == 27
    assert res[1].end_frame == 36
    
    # 3. Test complete erasure
    short = (TemporalBoundary(start_frame=10, end_frame=12, action_label="Tap"),)
    res_short = TemporalBoundaryWarper.warp_boundaries(short, [10, 11])
    assert len(res_short) == 0  # Erased
