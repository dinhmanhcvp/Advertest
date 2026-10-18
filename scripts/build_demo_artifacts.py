"""Regenerate the pre-computed DEMO_MODE artifacts in data/demo_artifacts/.

Deterministic (seeded). Produces 142 hard-negative audit records over the four
canonical failure tags, so the demo UI never depends on GPU or live data.
"""
import json
import random
from pathlib import Path

OUT = Path("data/demo_artifacts")
random.seed(7)

tags = (["error_fisheye"] * 50 + ["error_motion_blur"] * 43
        + ["error_sensor_noise"] * 30 + ["error_occlusion"] * 19)
random.shuffle(tags)
# Keep the three Proof-of-Cure cases consistent with their audit records.
tags[0], tags[6], tags[11] = "error_sensor_noise", "error_fisheye", "error_motion_blur"
just = {
    "error_fisheye": "Baseline IoU collapsed on peripheral faces; radial distortion calibrated to GoPro intrinsics (k1,k2) selected at severity {s}.",
    "error_motion_blur": "Baseline confidence < 0.4 under head-bob; biomechanical IMU blur ({hz:.1f} Hz sine + polynomial trajectory) selected at severity {s}.",
    "error_sensor_noise": "Exposure/ISO failure cluster dominates hard negatives; tone-curve + ISO noise (never combined with flare/low-light) selected at severity {s}.",
    "error_occlusion": "Partial-face misses from hands/straps; occluder patch selected at severity {s}.",
}
chain = {"error_fisheye": ["OpticalDistortion"], "error_motion_blur": ["BiomechanicalBlur"],
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

def landmarks(x, y):
    return [[x + dx, y + dy] for dx, dy in [(28, 28), (76, 26), (52, 56), (32, 86), (72, 85)]]

def case(cid, cond, tag, c1, c2, box):
    return {"case_id": cid, "condition": cond, "failure_tag": tag,
            "delta": f"+{c2 - c1:.2f} Confidence Gain",
            "model_v1": {"status": "Failed" if c1 < 0.4 else "Missed", "confidence": c1, "bbox": box, "landmarks": None},
            "model_v2": {"status": "Success", "confidence": c2, "bbox": box, "landmarks": landmarks(box[0], box[1])}}

poc = [
    case("ego4d_001", "Extreme Over-exposure (Chest Mount)", "error_sensor_noise", 0.25, 0.88, [120, 90, 110, 120]),
    case("ego4d_007", "Fisheye Edge Distortion (GoPro)", "error_fisheye", 0.31, 0.84, [60, 70, 100, 115]),
    case("ego4d_012", "Walking Head-bob Motion Blur (3 Hz)", "error_motion_blur", 0.18, 0.81, [140, 100, 95, 110]),
]
report = {"base_model": {"base_map": 0.85, "corrupted_map": 0.41},
          "advertest_model": {"base_map": 0.84, "corrupted_map": 0.68},
          "deltas": {"base_drop": 0.01, "corrupted_improvement": 0.27},
          "gate": {"max_base_drop": 0.01, "min_corrupted_gain": 0.05, "accepted": True}}

OUT.mkdir(parents=True, exist_ok=True)
for name, obj in [("audit_trail.json", audit), ("proof_of_cure.json", poc), ("eval_report.json", report)]:
    (OUT / name).write_text(json.dumps(obj, indent=2), encoding="utf8")
print("wrote", len(audit), "audit records")
