"""SYNTHETIC placeholder artifacts for DEMO_MODE (NOT measured results).

The committed files in data/demo_artifacts/ are real outputs of colab/colab_zero_disk_pipeline.py; this script only
exists to restore a plausible demo if those files are ever deleted.

Regenerate placeholder artifacts in data/demo_artifacts/.

Deterministic (seeded). Produces 200 hard-negative audit records over the four
canonical failure tags (42.5 / 31.0 / 18.5 / 8.0 %), three proof-of-cure cases
and the blind-evaluation-gate report, so the UI never depends on GPU or live data.

All box / landmark coordinates live in a 160x120 reference frame.
"""
import json
import random
from pathlib import Path

OUT = Path("data/demo_artifacts")
random.seed(7)

tags = (["error_motion_blur"] * 85 + ["error_fisheye"] * 62
        + ["error_sensor_noise"] * 37 + ["error_occlusion"] * 16)
random.shuffle(tags)
just = {
    "error_fisheye": "Baseline failed on peripheral edges; GoPro Hero10 radial intrinsics (k1,k2) selected at severity {s}.",
    "error_motion_blur": "Baseline lost the face during rapid locomotion; {hz:.1f} Hz biomechanical pitch trajectory selected at severity {s}.",
    "error_sensor_noise": "Exposure/ISO failure cluster; tone curve + ISO noise (never combined with flare/low-light) selected at severity {s}.",
    "error_occlusion": "Partial-face misses from hands/straps; occluder patch selected at severity {s}.",
}
chain = {"error_fisheye": ["OpticalDistortion"], "error_motion_blur": ["OpticalDistortion", "BiomechanicalBlur"],
         "error_sensor_noise": ["ToneCurve", "ISONoise"], "error_occlusion": ["HandOcclusion"]}

audit = []
for i, t in enumerate(tags):
    s = random.randint(2, 5)
    hz = random.uniform(2, 5)
    audit.append({
        "id": f"ego4d_{i + 1:03d}",
        "image_url": f"/assets/attacked_ego4d_{i + 1:03d}_{t}_s{s}.jpg",
        "iou": round(random.uniform(0.02, 0.48), 2),
        "confidence": round(random.uniform(0.05, 0.39), 2),
        "audit_trail": {"insight_source": t, "applied_chain": chain[t], "severity_level": s,
                        "justification": just[t].format(s=s, hz=hz)},
    })


def landmarks(b):
    x, y, w, h = b
    return [[round(x + w * fx, 1), round(y + h * fy, 1)]
            for fx, fy in [(0.3, 0.36), (0.7, 0.36), (0.5, 0.56), (0.33, 0.78), (0.67, 0.78)]]


def case(cid, cond, tag, c1, c2, b1, b2):
    return {"case_id": cid, "condition": cond, "failure_tag": tag,
            "delta": f"+{c2 - c1:.2f} Confidence Gain",
            "model_v1": {"status": "Failed", "confidence": c1, "bbox": b1, "landmarks": None},
            "model_v2": {"status": "Success", "confidence": c2, "bbox": b2, "landmarks": landmarks(b2)}}


poc = [
    case("eval_ego4d_112", "Chest-Mount Lens Distortion", "error_fisheye", 0.25, 0.88, [10, 20, 50, 60], [12, 22, 48, 58]),
    case("eval_ego4d_127", "2Hz Walking Blur", "error_motion_blur", 0.18, 0.81, [84, 30, 44, 56], [82, 28, 46, 58]),
    case("eval_ego4d_141", "Backlit Glare", "error_sensor_noise", 0.31, 0.84, [58, 26, 46, 58], [56, 24, 48, 60]),
]
report = {"base_model": {"base_map": 0.914, "corrupted_map": 0.482},
          "advertest_model": {"base_map": 0.911, "corrupted_map": 0.768},
          "deltas": {"base_drop": 0.003, "corrupted_improvement": 0.286},
          "hard_negative_recovery": 0.894,
          "gate": {"max_base_drop": 0.01, "min_corrupted_gain": 0.05, "accepted": True}}

OUT.mkdir(parents=True, exist_ok=True)
for name, obj in [("audit_trail.json", audit), ("proof_of_cure.json", poc), ("eval_report.json", report)]:
    (OUT / name).write_text(json.dumps(obj, indent=2), encoding="utf8")
print("wrote", len(audit), "audit records")
