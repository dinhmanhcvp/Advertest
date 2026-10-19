"""Latency of the attack engine on a 1080p frame (CPU) + landmark-geometry evidence.

Usage: python scripts/benchmark_attacks.py [--frames N]
Prints milliseconds per frame for each failure tag and a linear extrapolation to 1M frames on one core.
"""

import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("colab_pipeline", ROOT / "colab" / "colab_zero_disk_pipeline.py")
cp = importlib.util.module_from_spec(spec)
sys.modules["colab_pipeline"] = cp
spec.loader.exec_module(cp)


def bench(fn, n):
    fn()  # warm-up
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), min(ts), max(ts)


def main(frames: int) -> None:
    w, h = 1920, 1080
    img = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    box = (900.0, 400.0, 1020.0, 540.0)
    lm = np.array([[930, 440], [990, 440], [960, 470], [935, 505], [985, 505]], float)
    print(f"1080p frame, CPU, median of {frames} runs (ms)\n")
    print(f"{'step':34s} {'median':>9s} {'min':>8s} {'max':>8s}")
    rows = {}
    for sev in (1, 3, 5):
        ker = cp.imu_kernel(sev, 3.0, 10.0)
        rows[f"imu filter2D (severity {sev}, {ker.shape[0]}px kernel)"] = bench(lambda ker=ker: cp.cv2.filter2D(img, -1, ker), frames)
    rows["radial remap of the image (k1,k2)"] = bench(lambda: cp.distort_image(img, 0.5, 0.2), frames)
    rows["radial map of 5 landmarks + box"] = bench(
        lambda: (cp.distort_points(lm, w, h, 0.5, 0.2), cp.box_after_distortion(box, w, h, 0.5, 0.2)), max(frames, 50)
    )
    for name, (med, lo, hi) in rows.items():
        print(f"{name:34s} {med:9.2f} {lo:8.2f} {hi:8.2f}")
    print("\nfull synthesize() per failure tag (severity 5)")
    total = []
    for tag in cp.CANONICAL_TAGS:
        rng = np.random.default_rng(1)
        med, lo, hi = bench(lambda tag=tag, rng=rng: cp.synthesize(img, [box], [lm], tag, 5, rng), frames)
        total.append(med)
        print(f"{tag:34s} {med:9.2f} {lo:8.2f} {hi:8.2f}")
    avg = sum(total) / len(total)
    print(f"\nmean over tags: {avg:.1f} ms/frame -> 1M frames on ONE core ~ {avg * 1e6 / 3.6e6:.1f} core-hours")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=5)
    main(ap.parse_args().frames)
