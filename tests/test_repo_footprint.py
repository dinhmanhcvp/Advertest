"""Guard: the repository must stay lightweight (all heavy work runs on the Colab worker)."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAVY = {".pt", ".pth", ".onnx", ".ckpt", ".safetensors", ".mp4", ".zip", ".tar", ".bin", ".npy", ".npz", ".jpg", ".jpeg", ".png"}


def _tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [ROOT / p for p in out.splitlines() if (ROOT / p).exists()]


def test_no_heavy_binaries_tracked():
    bad = [p.name for p in _tracked() if p.suffix.lower() in HEAVY]
    assert not bad, f"heavy artifacts must live on the Colab worker / Core Data Lake, not in git: {bad}"


def test_tracked_footprint_under_5mb():
    total = sum(p.stat().st_size for p in _tracked())
    assert total < 5_000_000, f"tracked files total {total / 1e6:.2f} MB (limit 5 MB)"
