"""Tests for the Egocentric Engine domain types."""

import numpy as np
from src.core.egocentric_types import (
    ErrorCase,
    PIIAnnotation,
    TemporalBoundary,
    VideoSample,
)

def test_temporal_boundary_overlap():
    b1 = TemporalBoundary(start_frame=10, end_frame=20, action_label="run")
    b2 = TemporalBoundary(start_frame=15, end_frame=25, action_label="jump")
    b3 = TemporalBoundary(start_frame=25, end_frame=30, action_label="stop")
    
    assert b1.overlaps(b2)
    assert not b1.overlaps(b3)
    assert b1.duration_frames == 10
    
def test_temporal_iou():
    b1 = TemporalBoundary(start_frame=10, end_frame=20, action_label="run")
    b2 = TemporalBoundary(start_frame=15, end_frame=25, action_label="jump")
    
    # intersection: 15-20 (5 frames)
    # union: 10-25 (15 frames)
    # IoU = 5 / 15 = 0.333...
    assert abs(b1.iou(b2) - (5.0 / 15.0)) < 1e-6

def test_pii_annotation_area():
    ann = PIIAnnotation(
        x1=10, y1=10, x2=20, y2=30, pii_class="face"
    )
    assert ann.area == 200.0
    assert ann.as_box_tuple() == (10, 10, 20, 30)

def test_video_sample_properties():
    frames = np.zeros((10, 720, 1280, 3), dtype=np.float32)
    sample = VideoSample(
        sample_id="test_vid",
        frames=frames,
        fps=30.0
    )
    
    assert sample.num_frames == 10
    assert sample.height == 720
    assert sample.width == 1280
    assert abs(sample.duration_seconds - (10.0 / 30.0)) < 1e-6

def test_error_case_properties():
    err = ErrorCase(
        case_id="e1",
        sample_id="s1",
        error_category="pii_fisheye"
    )
    assert err.is_pii_error
    assert not err.is_semantic_error
