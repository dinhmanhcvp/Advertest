from fastapi import APIRouter
import json
import os

router = APIRouter()

DEMO_ARTIFACTS_DIR = "data/demo_artifacts"

@router.get("/demo/audit-trail")
async def get_demo_audit_trail():
    """Reads and returns the contents of the local audit_trail.json file."""
    file_path = os.path.join(DEMO_ARTIFACTS_DIR, "audit_trail.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return []

@router.get("/demo/proof-of-cure")
async def get_demo_proof_of_cure():
    """Reads and returns the contents of the local proof_of_cure.json file."""
    file_path = os.path.join(DEMO_ARTIFACTS_DIR, "proof_of_cure.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return []
