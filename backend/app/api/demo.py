import json
import os
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()

DEMO_ARTIFACTS_DIR = Path("data/demo_artifacts")
LIVE_DIR = Path("data/demo_output")


def _demo_mode() -> bool:
    return os.environ.get("ENABLE_DEMO_MODE", os.environ.get("DEMO_MODE", "")).strip().lower() in ("1", "true", "yes")


def _serve(name: str):
    """Live artifact when present and DEMO_MODE is off; otherwise the pre-computed one."""
    stem, ext = name.rsplit(".", 1)
    candidates = [DEMO_ARTIFACTS_DIR / name]
    if not _demo_mode():
        candidates.insert(0, LIVE_DIR / f"{stem}.live.{ext}")
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return []


@router.get("/demo/audit-trail")
async def get_demo_audit_trail():
    """Explainability log: insight_source, applied_chain, severity_level, justification."""
    return _serve("audit_trail.json")


@router.get("/demo/proof-of-cure")
async def get_demo_proof_of_cure():
    """A/B evaluation: model_v1 (failed) vs model_v2 (cured) with bbox + 5-point landmarks."""
    return _serve("proof_of_cure.json")
