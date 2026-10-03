"""Integration Test: End-to-End Sanity Check.

Executes the pipeline sequentially: Inference -> Error Extractor -> Attack Factory -> Format Converter.
Mocks external endpoints (Label Studio).
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.core.egocentric_types import ErrorCase
from src.core.types import Box, DetectionPrediction, Sample
from src.pipeline.orchestrator import Orchestrator


@pytest.fixture
def mock_micro_batch():
    """Generate 2 dummy images mimicking Ego4D constraints."""
    img1 = np.ones((720, 1280, 3), dtype=np.float32)
    img2 = np.zeros((720, 1280, 3), dtype=np.float32)
    
    samples = [
        Sample(sample_id="ego4d_001", image=img1, boxes=(Box(10, 10, 50, 50, "Face"),)),
        Sample(sample_id="ego4d_002", image=img2, boxes=(Box(100, 100, 150, 150, "Face"),))
    ]
    
    # Mock inference output with low confidence/IoU to trigger Error Extractor
    preds = [
        DetectionPrediction(sample_id="ego4d_001", boxes=(Box(10, 10, 50, 50, "Face", score=0.2),), latency_ms=10.0),
        DetectionPrediction(sample_id="ego4d_002", boxes=(Box(200, 200, 250, 250, "Face", score=0.9),), latency_ms=10.0)
    ]
    
    return samples, preds


@patch("src.agents.auto_triage.AutoTriageAgent._simulate_vlm_confidence", return_value=1.0)
@patch("src.services.label_studio_manager.LabelStudio")
@patch("src.pipeline.data_packager.DataPackager.package_approved_pool")
def test_e2e_pipeline_sanity(mock_package, mock_ls_client, mock_vlm_conf, mock_micro_batch):
    """Run E2E pipeline without throwing any exceptions."""
    samples, preds = mock_micro_batch
    
    # Mock Label Studio client to avoid HTTP errors
    mock_project = MagicMock()
    mock_project.id = 99
    mock_ls_client().get_project.return_value = mock_project
    mock_ls_client().create_project.return_value = mock_project
    
    # Mock packager to return a dummy path
    mock_package.return_value = "/tmp/mock_dataset_v1.0"
    
    orchestrator = Orchestrator()
    
    # Override orchestrator's _run_extraction to use our micro batch
    def _mock_run_extraction(config):
        errors = orchestrator.error_extractor.extract_pii_errors(samples, preds)
        # Cast to ErrorCase
        return [
            ErrorCase(case_id=f"err-{i}", sample_id=e.sample_id, error_category=e.error_category)
            for i, e in enumerate(errors)
        ]
        
    orchestrator._run_extraction = _mock_run_extraction
    
    # Run the pipeline
    run_config = {"batch_size": 2}
    try:
        results = orchestrator.run_full_pipeline(run_config)
    except Exception as e:
        pytest.fail(f"Pipeline threw an unexpected exception: {e}")
        
    # Assertions
    assert results is not None
    assert "extracted_errors" in results
    assert results["extracted_errors"] == 2  # 1 low conf, 1 low IoU
    assert "generated_pools" in results
    assert "packaged_datasets" in results
    assert len(results["packaged_datasets"]) > 0
    
    # Verify Label Studio interactions
    assert mock_ls_client().create_project.called
    assert mock_project.import_tasks.called
