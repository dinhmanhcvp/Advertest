"""Tests for AdverTest API Gateway — All Endpoints.

Covers:
- Core endpoints (jobs, insights, eval-report)
- Demo endpoints (audit-trail, proof-of-cure)
- Workflow endpoints (triage/analyze, workflow/attack, workflow/retrain)
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.app.main import app


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


# ─────────────────────── Core API Tests ───────────────────────


@pytest.mark.asyncio
async def test_generate_pipeline(client):
    """POST /api/v1/jobs/generate should start a pipeline job."""
    resp = await client.post(
        "/api/v1/jobs/generate",
        json={"dataset_path": "data/test_dataset"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["message"] == "Pipeline execution started."


@pytest.mark.asyncio
async def test_job_status_not_found(client):
    """GET /api/v1/jobs/status/<bad_id> should return 404."""
    resp = await client.get("/api/v1/jobs/status/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_insights(client):
    """GET /api/v1/insights should return error distribution."""
    resp = await client.get("/api/v1/insights")
    assert resp.status_code == 200
    data = resp.json()
    assert "error_distribution" in data
    assert len(data["error_distribution"]) == 4
    assert data["total_hard_negatives"] == 3400


@pytest.mark.asyncio
async def test_eval_report(client):
    """GET /api/v1/eval-report should return mAP comparison metrics."""
    resp = await client.get("/api/v1/eval-report")
    assert resp.status_code == 200
    data = resp.json()
    assert "base_model" in data
    assert "advertest_model" in data
    assert "deltas" in data
    assert data["base_model"]["base_map"] == 0.85
    assert data["advertest_model"]["corrupted_map"] == 0.68


# ─────────────────────── Demo Endpoints ───────────────────────


@pytest.mark.asyncio
async def test_demo_audit_trail(client):
    """GET /api/v1/demo/audit-trail should return audit trail from JSON file."""
    resp = await client.get("/api/v1/demo/audit-trail")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_demo_proof_of_cure(client):
    """GET /api/v1/demo/proof-of-cure should return proof of cure from JSON file."""
    resp = await client.get("/api/v1/demo/proof-of-cure")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


# ─────────────────────── Workflow Endpoints ───────────────────────


@pytest.mark.asyncio
async def test_triage_analyze(client):
    """POST /api/v1/triage/analyze should return failure distribution."""
    resp = await client.post(
        "/api/v1/triage/analyze",
        json={"model_path": "yolov7-tiny-face.pt", "dataset_path": "Ego4D_Validation"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "failure_distribution" in data
    assert len(data["failure_distribution"]) == 4
    assert "worst_samples" in data
    assert data["total_hard_negatives"] == 142
    for item in data["failure_distribution"]:
        assert "name" in item
        assert "value" in item
        assert "color" in item


@pytest.mark.asyncio
async def test_workflow_attack(client):
    """POST /api/v1/workflow/attack should return audit trail with explainability."""
    resp = await client.post(
        "/api/v1/workflow/attack",
        json={"sample_ids": ["ego4d_001", "ego4d_002"]}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "audit_trail" in data
    assert len(data["audit_trail"]) >= 1
    trail = data["audit_trail"][0]
    assert "id" in trail
    at = trail["audit_trail"]
    assert "insight_source" in at
    assert "applied_chain" in at
    assert "severity_level" in at
    assert "justification" in at
    assert isinstance(at["applied_chain"], list)


@pytest.mark.asyncio
async def test_workflow_retrain(client):
    """POST /api/v1/workflow/retrain should return proof of cure and mAP metrics."""
    resp = await client.post("/api/v1/workflow/retrain")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "map_improvement" in data
    assert "proof_of_cure" in data
    mi = data["map_improvement"]
    assert "base_drop_pct" in mi
    assert "corrupted_improvement_pct" in mi
    assert mi["corrupted_improvement_pct"] > 5.0
    poc = data["proof_of_cure"][0]
    assert "case_id" in poc
    assert "condition" in poc
    assert "model_v1" in poc
    assert "model_v2" in poc
    assert "delta" in poc
    assert poc["model_v1"]["status"] in ("Failed", "Missed")
    assert poc["model_v2"]["status"] == "Success"
    assert poc["model_v2"]["confidence"] > poc["model_v1"]["confidence"]
