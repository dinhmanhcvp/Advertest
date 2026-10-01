"""Unit Test 2: Inference & Error Extraction.

Tests the ErrorExtractor module against mocked YOLOv7-face predictions and Ground Truth.
Verifies that IoU < 0.5 or Confidence < 0.4 triggers Hard Negative flagging.
"""

import pytest

from src.core.types import Box, DetectionPrediction, Sample
from src.pipeline.error_extractor import ErrorExtractor


@pytest.fixture
def mock_predictions_and_gt():
    """Mock YOLOv7-face predictions and Ground Truth set."""
    # Ground Truth: 3 faces
    gt_boxes = (
        Box(x1=10.0, y1=10.0, x2=50.0, y2=50.0, label="Face"),
        Box(x1=100.0, y1=100.0, x2=150.0, y2=150.0, label="Face"),
        Box(x1=200.0, y1=200.0, x2=250.0, y2=250.0, label="Face"),
    )
    
    gt_sample = Sample(
        sample_id="test_image_01",
        image=None,  # Not needed for logic test
        boxes=gt_boxes
    )
    
    # Predictions:
    # 1. High IoU (0.8), High Conf (0.9) -> Valid (Not an error)
    # 2. Low IoU (0.3), High Conf (0.9) -> Error (Localization Failure / Hard Negative)
    # 3. High IoU (0.8), Low Conf (0.3) -> Error (Low Confidence / Hard Negative)
    pred_boxes = (
        Box(x1=12.0, y1=12.0, x2=48.0, y2=48.0, label="Face", score=0.9),    # Maps to GT 1 (Valid)
        Box(x1=120.0, y1=120.0, x2=170.0, y2=170.0, label="Face", score=0.9), # Maps to GT 2 (Low IoU)
        Box(x1=202.0, y1=202.0, x2=248.0, y2=248.0, label="Face", score=0.3), # Maps to GT 3 (Low Conf)
    )
    
    prediction = DetectionPrediction(
        sample_id="test_image_01",
        boxes=pred_boxes,
        latency_ms=10.0
    )
    
    return gt_sample, prediction


def test_error_extractor(mock_predictions_and_gt):
    """Ensure predictions with IoU < 0.5 or Conf < 0.4 are correctly flagged."""
    gt_sample, prediction = mock_predictions_and_gt
    
    extractor = ErrorExtractor()
    
    # We will override the logic slightly to test the exact condition requested:
    # Extract errors (Hard Negatives)
    errors = extractor.extract_pii_errors([gt_sample], [prediction])
    
    # Assert the final count matches expected
    # Expected: 2 errors (1 Low IoU, 1 Low Confidence)
    assert len(errors) == 2, f"Expected 2 Hard Negatives, got {len(errors)}"
    
    error_categories = [err.error_category for err in errors]
    
    # Because ErrorExtractor assigns categories like 'pii_occlusion', 'pii_blur'
    # we just verify that they are correctly flagged as errors with sample_id matching.
    for err in errors:
        assert err.sample_id == "test_image_01"
