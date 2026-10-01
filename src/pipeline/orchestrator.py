"""Pipeline Orchestrator.

High-level coordinator that runs the entire Egocentric Engine workflow:
Error Extraction -> Label Studio Sync -> Attack Pool Generation -> HITL -> Packaging.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import get_settings
from src.core.egocentric_types import ErrorCase
from src.agents.auto_triage import AutoTriageAgent
from src.pipeline.attack_pool_factory import AttackPoolFactory
from src.pipeline.data_packager import DataPackager
from src.pipeline.error_extractor import ErrorExtractor
from src.services.data_connector import DataConnector
from src.services.label_studio_manager import LabelStudioManager

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the End-to-End Egocentric Data Engine pipeline."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.error_extractor = ErrorExtractor()
        self.ls_manager = LabelStudioManager()
        self.factory = AttackPoolFactory()
        self.triage_agent = AutoTriageAgent()
        self.packager = DataPackager()
        self.data_connector = DataConnector()

    def run_full_pipeline(self, run_config: dict[str, Any]) -> dict[str, Any]:
        """Execute the complete E2E pipeline (simulation)."""
        logger.info("Starting Full Pipeline Run")
        results = {}
        
        # Stage 1: Error Extraction
        logger.info("--- Stage 1: Error Extraction ---")
        errors = self._run_extraction(run_config)
        results["extracted_errors"] = len(errors)
        
        if not errors:
            logger.warning("No errors extracted. Pipeline stopping.")
            return results
            
        # Stage 1.5: Push to Label Studio for classification
        logger.info("--- Label Studio: Pushing Error Tasks ---")
        project_id = self.ls_manager.create_project("Error Analysis", "pii")
        self.ls_manager.push_error_tasks(project_id, errors)
        
        # (In reality, we would pause here and wait for human labeling.
        # For demonstration, we proceed assuming classification is done).
        
        # Stage 2: Attack Pool Generation
        logger.info("--- Stage 2: Attack Pool Generation ---")
        # We group by the automatically classified error_category
        categories = {e.error_category for e in errors}
        pools = []
        for cat in categories:
            # Generate spatial pool for PII
            pool = self.factory.generate_spatial_pool(cat, source_samples=[]) # Dummy empty samples
            
            # Auto-Triage: VLM Agent filters out corrupted samples
            approved_pool, rejected_entries = self.triage_agent.triage_pool(pool)
            
            # Auto-approve the remaining ones for pipeline automation testing
            from dataclasses import replace
            if len(approved_pool.entries) > 0:
                approved_pool = replace(approved_pool, status="approved")
                pools.append(approved_pool)
            
        results["generated_pools"] = len(pools)
        
        # Stage 3: Label Studio Review Push
        logger.info("--- Stage 3: HITL Review Push ---")
        review_project_id = self.ls_manager.create_project("Pool Review", "pii")
        for pool in pools:
            self.ls_manager.push_review_tasks(review_project_id, pool)
            
        # Stage 4: Package and Export
        logger.info("--- Stage 4: Data Packaging ---")
        packaged_paths = []
        for pool in pools:
            path = self.packager.package_approved_pool(pool, version="v1.0")
            if path:
                self.packager.generate_data_card(pool, path)
                self.packager.export_retrain_ready(path, format="coco")
                packaged_paths.append(path)
                
        results["packaged_datasets"] = packaged_paths
        
        logger.info("Full Pipeline Run Complete")
        return results

    def _run_extraction(self, config: dict[str, Any]) -> list[ErrorCase]:
        """Simulate extraction step."""
        # This would call error_extractor.extract_pii_errors() with real data.
        return [
            ErrorCase(
                case_id="err-1",
                sample_id="img_001",
                error_category="pii_fisheye",
            ),
            ErrorCase(
                case_id="err-2",
                sample_id="vid_002",
                error_category="semantic_temporal_break",
            )
        ]
