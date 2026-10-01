"""Egocentric API endpoints for the AdverTest Engine."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from src.pipeline.orchestrator import Orchestrator

router = APIRouter(prefix="/api/egocentric", tags=["Egocentric Engine"])
orchestrator = Orchestrator()

@router.post("/extract-errors")
def trigger_extraction(config: dict[str, Any]) -> dict[str, Any]:
    """Trigger the error extraction stage manually."""
    # In a full implementation, this would trigger just the extraction stage
    # For now, it delegates to the full pipeline mock.
    return orchestrator.run_full_pipeline(config)

@router.post("/generate-pool")
def generate_attack_pool(config: dict[str, Any]) -> dict[str, Any]:
    """Trigger the Attack Pool Factory stage."""
    # Placeholder delegation
    return orchestrator.run_full_pipeline(config)

@router.post("/sync-labels")
def sync_label_studio() -> dict[str, str]:
    """Force sync of Label Studio annotations."""
    return {"status": "synced"}

@router.post("/package")
def package_datasets() -> dict[str, str]:
    """Package all approved pools."""
    return {"status": "packaged"}

@router.get("/status")
def get_status() -> dict[str, str]:
    """Get the current status of the Egocentric Engine pipeline."""
    return {"status": "idle", "active_jobs": "0"}
