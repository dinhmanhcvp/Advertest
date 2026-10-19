/** Reference frame for every coordinate in the product: 160 x 120 units. */
export const FRAME_W = 160;
export const FRAME_H = 120;

export type Pt = [number, number];
/** [x, y, w, h] in reference-frame units. */
export type Box = [number, number, number, number];

export type FailureTag = "error_motion_blur" | "error_fisheye" | "error_sensor_noise" | "error_occlusion";

export interface FailureSlice {
  tag: FailureTag;
  label: string;
  pct: number;
  count: number;
}

export interface HardNegative {
  id: string;
  mount: string;
  tag: FailureTag;
  conf: number;
  iou: number;
}

export interface AuditTrail {
  insight_source: string;
  applied_chain: string[];
  severity_level: number;
  justification: string;
}

export interface StudioSample {
  id: string;
  source: "WIDER FACE" | "Ego4D";
  tag: FailureTag;
  mount: string;
  box: Box;
  landmarks: Pt[];
  hz: number;
  severity: number;
  pan: number;
  audit: AuditTrail;
}

export interface ModelResult {
  status: string;
  confidence: number;
  bbox: Box;
  landmarks: Pt[] | null;
}

export interface CureCase {
  case_id: string;
  condition: string;
  failure_tag: FailureTag;
  delta: string;
  /** Optional real frame (data URI, letter-boxed to the 160x120 reference frame) from the GPU worker. */
  image?: string;
  model_v1: ModelResult;
  model_v2: ModelResult;
}

export interface EvalReport {
  base_model: { base_map: number; corrupted_map: number };
  advertest_model: { base_map: number; corrupted_map: number };
  hard_negative_recovery: number;
  gate: {
    max_base_drop: number;
    min_corrupted_gain: number;
    accepted: boolean;
    corrupted_gain_ci95?: [number, number];
    gain_statistically_supported?: boolean;
    beats_clean_only_control?: boolean;
  };
  /** Same init / epochs / clean images as V2 but without adversarial samples: rules out "just more WIDER data". */
  control_model?: { base_map: number; corrupted_map: number; note?: string } | null;
  stats?: Record<"clean" | "corrupted", Record<string, { mean: number; lo: number; hi: number; n_images: number }>>;
  corruption_repeats?: { v2_minus_v1_mean: number; v2_minus_v1_std: number; draws: unknown[] };
  recall_at_fppi?: Record<"clean" | "corrupted", Record<"v1" | "v2", Record<string, number>>>;
  provenance?: { target_domain?: string; n_blind?: number; n_train?: number; epochs?: number; ego4d_real?: unknown };
}

export interface TrackFrame {
  label: string;
  offset: number;
  box: Box;
  blur: number;
  conf: number;
  detected: boolean;
}

/** api = FastAPI, artifact = measured Colab run served as static files, local = built-in placeholder store. */
export type DataSource = "api" | "artifact" | "local";

export interface WorkspaceData {
  source: DataSource;
  slices: FailureSlice[];
  hardNegatives: HardNegative[];
  samples: StudioSample[];
  cases: CureCase[];
  report: EvalReport;
  track: TrackFrame[];
}
