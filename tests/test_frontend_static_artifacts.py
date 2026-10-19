"""The Vercel deploy serves frontend/public/demo/*.json when no API is reachable: keep them identical to the source of truth."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_static_demo_artifacts_match_data_demo_artifacts():
    for name in ("audit_trail.json", "proof_of_cure.json", "eval_report.json"):
        src = (ROOT / "data" / "demo_artifacts" / name).read_bytes().replace(b"\r\n", b"\n")
        dst = (ROOT / "frontend" / "public" / "demo" / name).read_bytes().replace(b"\r\n", b"\n")
        assert src == dst, f"frontend/public/demo/{name} is stale: copy it from data/demo_artifacts/"
