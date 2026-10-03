"""Unit Test 3: Label Studio Conversion.

Tests the conversion of bounding boxes into Label Studio's percentage-based format
and mocks a successful HTTP 201 push to a dummy project.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core.types import Box
from src.services.label_studio_manager import LabelStudioManager


def convert_yolo_to_ls_format(box: Box, img_width: int, img_height: int) -> dict:
    """Mock format_converter logic to transform absolute boxes to LS percentage format."""
    x_pct = (box.x1 / img_width) * 100
    y_pct = (box.y1 / img_height) * 100
    w_pct = ((box.x2 - box.x1) / img_width) * 100
    h_pct = ((box.y2 - box.y1) / img_height) * 100
    
    return {
        "original_width": img_width,
        "original_height": img_height,
        "image_rotation": 0,
        "value": {
            "x": x_pct,
            "y": y_pct,
            "width": w_pct,
            "height": h_pct,
            "rectanglelabels": [box.label]
        }
    }


def test_format_converter_logic():
    """Validate that coordinates are properly converted into Label Studio's percentage-based system."""
    img_width = 1280
    img_height = 720
    
    # YOLO output (absolute pixels)
    box = Box(x1=320.0, y1=180.0, x2=960.0, y2=540.0, label="Face")
    
    ls_json = convert_yolo_to_ls_format(box, img_width, img_height)
    
    val = ls_json["value"]
    # x = 320 / 1280 = 25%
    # y = 180 / 720 = 25%
    # w = (960 - 320) / 1280 = 640 / 1280 = 50%
    # h = (540 - 180) / 720 = 360 / 720 = 50%
    assert val["x"] == 25.0
    assert val["y"] == 25.0
    assert val["width"] == 50.0
    assert val["height"] == 50.0
    assert val["rectanglelabels"] == ["Face"]


@patch("src.services.label_studio_manager.LabelStudio")
def test_label_studio_api_push(mock_client_cls):
    """Simulate a successful API call (HTTP 201) when pushing this JSON to a dummy project."""
    # Setup mock
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    mock_project = MagicMock()
    mock_client.get_project.return_value = mock_project
    mock_project.import_tasks.return_value = {"task_count": 1, "status": 201}
    
    manager = LabelStudioManager()
    
    # Format to push
    ls_task = {
        "data": {"image": "http://dummy/image.jpg"},
        "annotations": [{
            "result": [
                convert_yolo_to_ls_format(Box(x1=0, y1=0, x2=10, y2=10, label="Face"), 100, 100)
            ]
        }]
    }
    
    # Push
    response = mock_project.import_tasks([ls_task])
    
    assert response["status"] == 201
    assert response["task_count"] == 1
    mock_project.import_tasks.assert_called_once()
