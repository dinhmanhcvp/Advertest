// Static registry for the Closed-Loop Studio: models, datasets, failure
// taxonomy ("insights") and the adversarial attack library.

// ───────────────────────────── Failure taxonomy (insights) ─────────────────

export const INSIGHT_TAGS = [
  "motion_blur",
  "low_light",
  "glare",
  "lens_distortion",
  "occlusion",
  "weather",
  "compression",
];

export const INSIGHTS = {
  motion_blur: {
    id: "motion_blur",
    label: "Nhòe chuyển động",
    short: "Motion blur",
    color: "#a78bfa",
    hypothesis: "Đối tượng di chuyển nhanh trong thời gian phơi sáng dài làm mất biên cạnh — backbone suy giảm đặc trưng tần số cao.",
    remedy: "Tăng cường mẫu blur theo hướng chuyển động, thêm temporal context.",
  },
  low_light: {
    id: "low_light",
    label: "Thiếu sáng · nhiễu cảm biến",
    short: "Low light",
    color: "#60a5fa",
    hypothesis: "SNR thấp ở vùng tối khiến contrast của đối tượng gần nền; mô hình phụ thuộc texture chi tiết.",
    remedy: "Bổ sung mẫu low-light + ISO noise, gamma augmentation.",
  },
  glare: {
    id: "glare",
    label: "Chói sáng · lóa nắng",
    short: "Glare",
    color: "#fbbf24",
    hypothesis: "Vùng cháy sáng (HDR clipping) xóa thông tin màu/biên của đối tượng, đặc biệt khi ngược nắng.",
    remedy: "Thêm sun-flare, overexposure, tone-mapping augmentation.",
  },
  lens_distortion: {
    id: "lens_distortion",
    label: "Méo ống kính",
    short: "Lens distortion",
    color: "#34d399",
    hypothesis: "Camera góc rộng làm biến dạng hình học ở rìa khung hình, lệch so với phân phối huấn luyện.",
    remedy: "Augment bằng fisheye / grid distortion với tâm ngẫu nhiên.",
  },
  occlusion: {
    id: "occlusion",
    label: "Che khuất một phần",
    short: "Occlusion",
    color: "#fb7185",
    hypothesis: "Đối tượng bị vật khác che > 30% — mô hình thiếu khả năng suy luận từ phần nhìn thấy.",
    remedy: "Cutout / random erasing có kiểm soát vị trí quanh đối tượng.",
  },
  weather: {
    id: "weather",
    label: "Thời tiết xấu · mưa sương",
    short: "Weather",
    color: "#22d3ee",
    hypothesis: "Hạt mưa, sương mù làm giảm contrast toàn cục và tạo nhiễu cấu trúc trên biên đối tượng.",
    remedy: "Tổng hợp rain + fog + giảm contrast theo lớp.",
  },
  compression: {
    id: "compression",
    label: "Nén ảnh · mất chi tiết",
    short: "Compression",
    color: "#f472b6",
    hypothesis: "Artifact nén JPEG/H.264 và độ phân giải thấp làm vỡ khối, mất texture nhỏ.",
    remedy: "Augment JPEG quality thấp, downscale-upscale, noise.",
  },
};

// ───────────────────────────── Attack library ──────────────────────────────

/**
 * `aff` = how strongly the attack pushes each failure condition (0‒1).  The
 * engine adds `aff × severity/5 × K` to the sample's condition intensities and
 * lets the model's own sensitivity profile turn that into a score drop.
 */
export const ATTACKS = [
  { id: "motion_blur", name: "Motion Blur", family: "Optical", desc: "Nhòe theo hướng chuyển động", aff: { motion_blur: 1, weather: 0.1 } },
  { id: "defocus_blur", name: "Defocus Blur", family: "Optical", desc: "Mất nét do lệch tiêu cự", aff: { motion_blur: 0.55, compression: 0.3 } },
  { id: "gaussian_noise", name: "Sensor Noise", family: "Sensor", desc: "Nhiễu ISO / Gaussian", aff: { low_light: 0.75, compression: 0.3 } },
  { id: "low_light", name: "Low Light", family: "Sensor", desc: "Giảm phơi sáng, lệch tông xanh", aff: { low_light: 1 } },
  { id: "sun_flare", name: "Sun Flare", family: "Environment", desc: "Lóa nắng, tia sáng ngược", aff: { glare: 1, weather: 0.1 } },
  { id: "overexposure", name: "Overexposure", family: "Sensor", desc: "Cháy sáng, clipping HDR", aff: { glare: 0.85 } },
  { id: "fisheye", name: "Fisheye", family: "Geometric", desc: "Méo thùng góc rộng", aff: { lens_distortion: 1, occlusion: 0.1 } },
  { id: "grid_distortion", name: "Grid Distortion", family: "Geometric", desc: "Biến dạng lưới phi tuyến", aff: { lens_distortion: 0.6, compression: 0.1 } },
  { id: "rain", name: "Rain Streaks", family: "Environment", desc: "Vệt mưa & giảm contrast", aff: { weather: 1, motion_blur: 0.2 } },
  { id: "fog", name: "Fog / Haze", family: "Environment", desc: "Sương mù, mờ khí quyển", aff: { weather: 0.8, glare: 0.3, low_light: 0.2 } },
  { id: "cutout", name: "Cutout", family: "Occlusion", desc: "Che khuất cục bộ", aff: { occlusion: 1 } },
  { id: "jpeg", name: "JPEG Artifacts", family: "Digital", desc: "Nén mạnh, vỡ khối", aff: { compression: 1, low_light: 0.1 } },
  { id: "color_shift", name: "Color Shift", family: "Digital", desc: "Lệch cân bằng trắng", aff: { low_light: 0.3, weather: 0.2, compression: 0.1 } },
];

export const ATTACK_BY_ID = Object.fromEntries(ATTACKS.map((a) => [a.id, a]));

export const FAMILY_COLOR = {
  Optical: "#a78bfa",
  Sensor: "#60a5fa",
  Environment: "#22d3ee",
  Geometric: "#34d399",
  Occlusion: "#fb7185",
  Digital: "#f472b6",
};

// Pairs that compound each other beyond the sum of their parts (physically
// co-occurring in real captures).  Key is the sorted pair joined by "|".
export const SYNERGY = {
  "low_light|motion_blur": 0.28,
  "fog|rain": 0.3,
  "low_light|rain": 0.22,
  "gaussian_noise|jpeg": 0.2,
  "motion_blur|rain": 0.15,
  "fog|low_light": 0.15,
  "defocus_blur|overexposure": 0.15,
  "fisheye|sun_flare": 0.15,
  "cutout|motion_blur": 0.12,
  "fisheye|grid_distortion": -0.1, // redundant geometry — diminishing
  "gaussian_noise|low_light": 0.18,
  "color_shift|fog": 0.1,
};

export const synergyOf = (a, b) => SYNERGY[[a, b].sort().join("|")] || 0;

export const SEVERITY_LABELS = ["", "Nhẹ", "Vừa", "Rõ", "Mạnh", "Cực đoan"];

// ───────────────────────────── Models ──────────────────────────────────────

/** `sens` = how much each failure condition hurts this model (0‒1). */
export const MODELS = [
  {
    id: "yolov7-face-lite",
    name: "YOLOv7-Face Lite",
    arch: "YOLOv7-tiny",
    params: "6.2M",
    task: "Face detection",
    base: 0.93,
    sens: { motion_blur: 0.92, low_light: 0.72, glare: 0.6, lens_distortion: 0.78, occlusion: 0.7, weather: 0.42, compression: 0.5 },
  },
  {
    id: "yolov8n-det",
    name: "YOLOv8n Detector",
    arch: "YOLOv8-nano",
    params: "3.2M",
    task: "Object detection",
    base: 0.92,
    sens: { motion_blur: 0.7, low_light: 0.88, glare: 0.66, lens_distortion: 0.55, occlusion: 0.8, weather: 0.74, compression: 0.46 },
  },
  {
    id: "rtdetr-r18",
    name: "RT-DETR R18",
    arch: "Transformer · ResNet-18",
    params: "20M",
    task: "Object detection",
    base: 0.95,
    sens: { motion_blur: 0.5, low_light: 0.62, glare: 0.82, lens_distortion: 0.64, occlusion: 0.42, weather: 0.58, compression: 0.7 },
  },
  {
    id: "retinaface-r50",
    name: "RetinaFace R50",
    arch: "ResNet-50 · FPN",
    params: "27M",
    task: "Face detection",
    base: 0.96,
    sens: { motion_blur: 0.58, low_light: 0.5, glare: 0.45, lens_distortion: 0.86, occlusion: 0.52, weather: 0.34, compression: 0.62 },
  },
];

export const MODEL_BY_ID = Object.fromEntries(MODELS.map((m) => [m.id, m]));

// ───────────────────────────── Datasets ────────────────────────────────────

/** `mix` = relative frequency of each failure condition being dominant. */
export const DATASETS = [
  {
    id: "wider-face-hard",
    name: "WIDER FACE · Hard",
    scene: "portrait",
    size: 800,
    classes: ["frontal", "profile", "small-face", "masked"],
    note: "Mặt người trong điều kiện khó: nhỏ, che, nhòe",
    pressure: 1.0,
    mix: { motion_blur: 1.2, low_light: 0.9, glare: 0.5, lens_distortion: 0.4, occlusion: 1.3, weather: 0.3, compression: 0.9 },
  },
  {
    id: "vin-urban-night",
    name: "VinDrive · Urban Night",
    scene: "street",
    size: 960,
    classes: ["car", "pedestrian", "cyclist", "truck"],
    note: "Đô thị ban đêm, nhiều ngược sáng & mưa",
    pressure: 1.12,
    mix: { motion_blur: 0.9, low_light: 1.6, glare: 1.2, lens_distortion: 0.5, occlusion: 0.8, weather: 1.1, compression: 0.5 },
  },
  {
    id: "bdd-lite-fisheye",
    name: "BDD100K · Surround Lite",
    scene: "street",
    size: 720,
    classes: ["car", "pedestrian", "cyclist", "truck"],
    note: "Camera surround góc rộng, méo rìa khung hình",
    pressure: 0.95,
    mix: { motion_blur: 0.7, low_light: 0.6, glare: 0.7, lens_distortion: 1.7, occlusion: 1.0, weather: 0.8, compression: 0.6 },
  },
];

export const DATASET_BY_ID = Object.fromEntries(DATASETS.map((d) => [d.id, d]));

export const RETRAIN_BATCH_TARGET = 24;
