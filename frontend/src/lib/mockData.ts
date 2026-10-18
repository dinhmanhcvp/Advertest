import { landmarksFor, TAGS, TAG_ORDER } from "./taxonomy";
import type {
  AuditTrail,
  Box,
  CureCase,
  EvalReport,
  FailureSlice,
  FailureTag,
  HardNegative,
  StudioSample,
  TrackFrame,
  WorkspaceData,
} from "./types";

/**
 * Local fallback store. Shapes and values mirror data/demo_artifacts/*.json so the
 * workspace is fully populated even when the FastAPI backend is not running.
 */

export const TOTAL_HARD_NEGATIVES = 200;
const COUNTS: Record<FailureTag, number> = {
  error_motion_blur: 85,
  error_fisheye: 62,
  error_sensor_noise: 37,
  error_occlusion: 16,
};

export const MOCK_SLICES: FailureSlice[] = TAG_ORDER.map((tag) => ({
  tag,
  label: TAGS[tag].label,
  count: COUNTS[tag],
  pct: COUNTS[tag] / TOTAL_HARD_NEGATIVES,
}));

/** Small deterministic PRNG so server and client render identical rows (no hydration drift). */
function lcg(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

function buildHardNegatives(): HardNegative[] {
  const rnd = lcg(7);
  const pattern: FailureTag[] = [
    "error_motion_blur",
    "error_fisheye",
    "error_motion_blur",
    "error_sensor_noise",
    "error_fisheye",
    "error_occlusion",
    "error_motion_blur",
    "error_sensor_noise",
  ];
  const rows = Array.from({ length: 24 }, (_, i): HardNegative => {
    const tag = pattern[i % pattern.length];
    return {
      id: `ego4d_${String(i * 7 + 8).padStart(3, "0")}`,
      mount: TAGS[tag].mount,
      tag,
      conf: Math.round((0.05 + rnd() * 0.34) * 100) / 100,
      iou: Math.round((0.02 + rnd() * 0.3) * 100) / 100,
    };
  });
  return rows.sort((a, b) => a.iou - b.iou);
}

export const MOCK_HARD_NEGATIVES = buildHardNegatives();

const JUSTIFICATION: Record<FailureTag, (s: number, hz: number) => string> = {
  error_motion_blur: (s, hz) =>
    `Baseline failed heavily on peripheral edges during rapid locomotion. Applied GoPro Hero10 radial intrinsics coupled with a ${hz.toFixed(1)}Hz biomechanical pitch trajectory at severity ${s}.`,
  error_fisheye: (s) =>
    `Baseline IoU collapsed on faces near the frame edge. Applied GoPro Hero10 radial intrinsics (k1, k2) at severity ${s}.`,
  error_sensor_noise: (s) =>
    `Failure cluster concentrated in over/under-exposed frames. Applied tone-curve + ISO noise at severity ${s}; flare and low-light are mutually excluded.`,
  error_occlusion: (s) => `Partial-face misses caused by hands and straps. Applied occluder patch at severity ${s}.`,
};

const CHAIN: Record<FailureTag, string[]> = {
  error_motion_blur: ["OpticalDistortion", "BiomechanicalBlur"],
  error_fisheye: ["OpticalDistortion"],
  error_sensor_noise: ["ToneCurve", "ISONoise"],
  error_occlusion: ["HandOcclusion"],
};

export function makeAudit(tag: FailureTag, severity: number, hz: number, source = "Ego4D_Analysis"): AuditTrail {
  return {
    insight_source: source,
    applied_chain: CHAIN[tag],
    severity_level: severity,
    justification: JUSTIFICATION[tag](severity, hz),
  };
}

export function makeSample(
  id: string,
  tag: FailureTag,
  opts: { source?: StudioSample["source"]; box?: Box; hz?: number; severity?: number; pan?: number } = {},
): StudioSample {
  const box = opts.box ?? [58, 24, 44, 58];
  const hz = opts.hz ?? 3.2;
  const severity = opts.severity ?? 4;
  return {
    id,
    source: opts.source ?? "Ego4D",
    tag,
    mount: TAGS[tag].mount,
    box,
    landmarks: landmarksFor(box),
    hz,
    severity,
    pan: opts.pan ?? 12,
    audit: makeAudit(tag, severity, hz),
  };
}

export const MOCK_SAMPLES: StudioSample[] = [
  makeSample("atk_wider_001", "error_motion_blur", { source: "WIDER FACE", severity: 5, hz: 3.2, pan: 18 }),
  makeSample("atk_ego4d_008", "error_motion_blur", { box: [46, 30, 42, 54], hz: 2.4, severity: 3, pan: 8 }),
  makeSample("atk_ego4d_015", "error_fisheye", { box: [18, 28, 46, 58], severity: 4, pan: 0 }),
  makeSample("atk_ego4d_032", "error_sensor_noise", { box: [74, 22, 42, 56], severity: 4, pan: 0 }),
  makeSample("atk_ego4d_045", "error_occlusion", { box: [62, 34, 40, 52], severity: 3, pan: 0 }),
  makeSample("atk_ego4d_059", "error_fisheye", { box: [100, 26, 40, 54], severity: 5, pan: 0 }),
];

function cure(
  id: string,
  condition: string,
  tag: FailureTag,
  c1: number,
  c2: number,
  b1: Box,
  b2: Box,
): CureCase {
  return {
    case_id: id,
    condition,
    failure_tag: tag,
    delta: `+${(c2 - c1).toFixed(2)} Confidence Gain`,
    model_v1: { status: "Failed", confidence: c1, bbox: b1, landmarks: null },
    model_v2: { status: "Success", confidence: c2, bbox: b2, landmarks: landmarksFor(b2) },
  };
}

export const MOCK_CASES: CureCase[] = [
  cure("eval_ego4d_112", "Chest-Mount Lens Distortion", "error_fisheye", 0.25, 0.88, [10, 20, 50, 60], [12, 22, 48, 58]),
  cure("eval_ego4d_127", "2Hz Walking Blur", "error_motion_blur", 0.18, 0.81, [84, 30, 44, 56], [82, 28, 46, 58]),
  cure("eval_ego4d_141", "Backlit Glare", "error_sensor_noise", 0.31, 0.84, [58, 26, 46, 58], [56, 24, 48, 60]),
];

export const MOCK_REPORT: EvalReport = {
  base_model: { base_map: 0.914, corrupted_map: 0.482 },
  advertest_model: { base_map: 0.911, corrupted_map: 0.768 },
  hard_negative_recovery: 0.894,
  gate: { max_base_drop: 0.01, min_corrupted_gain: 0.05, accepted: true },
};

/** Subject turns rapidly: frame t is obliterated by motion blur (conf 0.18 < 0.40). */
export const MOCK_TRACK: TrackFrame[] = [
  { label: "Frame t-2", offset: -2, box: [48, 34, 40, 52], blur: 0, conf: 0.91, detected: true },
  { label: "Frame t-1", offset: -1, box: [55, 33, 40, 52], blur: 0.9, conf: 0.74, detected: true },
  { label: "Frame t", offset: 0, box: [62, 32, 40, 52], blur: 4.2, conf: 0.18, detected: false },
  { label: "Frame t+1", offset: 1, box: [69, 31, 40, 52], blur: 1.4, conf: 0.57, detected: true },
  { label: "Frame t+2", offset: 2, box: [76, 30, 40, 52], blur: 0, conf: 0.89, detected: true },
];

export const MOCK_WORKSPACE: WorkspaceData = {
  source: "local",
  slices: MOCK_SLICES,
  hardNegatives: MOCK_HARD_NEGATIVES,
  samples: MOCK_SAMPLES,
  cases: MOCK_CASES,
  report: MOCK_REPORT,
  track: MOCK_TRACK,
};
