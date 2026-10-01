/**
 * Standard system configuration schemas, architecture definitions, and attack presets
 * Tailored for PII Protection and Face Detection.
 */

export const PROBLEM_TYPES = [
  { id: "detection2d", name: "Face Detection", desc: "Phát hiện & định vị khuôn mặt (PII)", icon: "Target" },
  { id: "segmentation", name: "Face Parsing", desc: "Phân đoạn chi tiết khuôn mặt", icon: "Layers" },
  { id: "detection3d", name: "3D Head Pose", desc: "Ước lượng góc nghiêng đầu 3D", icon: "Box", comingSoon: true },
  { id: "classification", name: "Attribute Classification", desc: "Phân loại thuộc tính", icon: "Grid", comingSoon: true },
  { id: "tracking", name: "Face Tracking", desc: "Theo dõi khuôn mặt trong video", icon: "Activity", comingSoon: true },
];

export const MODEL_ARCHITECTURES = [
  {
    id: "yolov7_face",
    name: "YOLOv7-Face",
    family: "YOLO",
    defaultModel: "yolov7-face.pt",
    variants: ["yolov7-face.pt", "yolov7-face-lite.pt"],
  },
  {
    id: "retinaface",
    name: "RetinaFace",
    family: "ResNet",
    defaultModel: "retinaface_resnet50.pth",
    variants: ["retinaface_resnet50.pth", "retinaface_mobilenet0.25.pth"],
  },
  {
    id: "mtcnn",
    name: "MTCNN",
    family: "Cascade",
    defaultModel: "mtcnn_default",
    variants: ["mtcnn_default"],
  },
  {
    id: "anonymization",
    name: "Anonymization ONNX",
    family: "Privacy ONNX",
    defaultModel: "yolov7-face-blur.onnx",
    variants: ["yolov7-face-blur.onnx (Local)"],
  }
];

export const CLASS_LABELS_DATA = [
  { id: 0, name: "face", count: 18453 },
  { id: 1, name: "masked_face", count: 4232 },
  { id: 2, name: "blurred_face", count: 1652 },
];

export const AVAILABLE_DATASETS = [
  {
    id: "widerface_val",
    name: "WIDER FACE Validation Set",
    task: "detection2d",
    format: "JPEG + TXT",
    samples: 3226,
    classes: 1,
    size: "1.8 GB",
    resolution: "Various",
    isLocal: true,
    localPath: "data/widerface/val/",
    tags: ["local", "widerface", "pii", "ready"],
    description: "Bộ dữ liệu chuẩn WIDER FACE dùng cho đánh giá phát hiện khuôn mặt với nhiều góc độ, độ sáng và tỷ lệ.",
    classLabels: [
      { id: 0, name: "face", count: 39708 }
    ],
  },
  {
    id: "ego4d_social",
    name: "Ego4D Social Benchmark",
    task: "detection2d",
    format: "MP4 / Frames + JSON",
    samples: 5400,
    classes: 1,
    size: "8.4 GB",
    resolution: "1920 × 1080",
    isLocal: true,
    localPath: "data/ego4d/social/",
    tags: ["local", "ego4d", "egocentric", "pipeline"],
    description: "Dữ liệu góc nhìn thứ nhất (Egocentric) từ Ego4D tập trung vào tương tác xã hội và theo dõi khuôn mặt.",
    classLabels: [
      { id: 0, name: "face", count: 12450 }
    ],
  }
];

export const AVAILABLE_MODELS = [
  {
    id: "local_yolov7_face",
    name: "YOLOv7-Face (Face Detection & Landmarks)",
    family: "YOLO",
    task: "detection2d",
    params: "36.5M",
    fileSize: "75.1 MB",
    inputSize: "640 × 640",
    format: ".pt (PyTorch)",
    isLocal: true,
    localPath: "checkpoints/yolov7-face.pt",
    mapBaseline: "92.8% mAP",
    fpsRtx4090: "145 FPS",
    description: "YOLOv7 tinh chỉnh cho bài toán phát hiện khuôn mặt và 5 điểm mốc (landmarks) chuẩn xác.",
    architecture: "yolov7_face",
  },
  {
    id: "local_retinaface",
    name: "RetinaFace ResNet50",
    family: "ResNet",
    task: "detection2d",
    params: "29.5M",
    fileSize: "112.5 MB",
    inputSize: "Dynamic",
    format: ".pth (PyTorch)",
    isLocal: true,
    localPath: "checkpoints/retinaface_resnet50.pth",
    mapBaseline: "95.6%",
    fpsRtx4090: "80 FPS",
    description: "Mô hình RetinaFace độ chính xác cao cho nhận diện PII trong đám đông.",
    architecture: "retinaface",
  },
  {
    id: "local_anonymization",
    name: "YOLOv7-Face Blur ONNX",
    family: "Privacy ONNX",
    task: "detection2d",
    params: "36.5M",
    fileSize: "140 MB",
    inputSize: "640 × 640",
    format: ".onnx (ONNX Runtime)",
    isLocal: true,
    localPath: "checkpoints/anonymization/yolov7-face-blur.onnx",
    mapBaseline: "91.4%",
    fpsRtx4090: "210 FPS",
    description:
      "Mô hình ONNX nhận diện và làm mờ khuôn mặt tự động để bảo vệ quyền riêng tư PII trước khi đưa vào hệ thống nội bộ.",
    architecture: "anonymization",
  },
];

export const ATTACK_PRESETS = [
  {
    id: "preset_egocentric_motion",
    name: "🏃‍♂️ Chuyển động góc nhìn thứ nhất (Egocentric Motion)",
    desc: "Mô phỏng camera rung lắc mạnh và mờ chuyển động phi tuyến tính từ ngực người đeo.",
    task: "detection2d",
    attacks: [
      { id: "biomechanical_blur", name: "Biomechanical Blur", severity: 4, eps: "Severity 4", alpha: "Motion", iters: 1 },
      { id: "motion_blur", name: "Motion Blur", severity: 3, eps: "Severity 3", alpha: "Linear", iters: 1 },
    ],
  },
  {
    id: "preset_lighting_stress",
    name: "💡 Ánh sáng cực đoan (Lighting & Shadows)",
    desc: "Kiểm thử ngược sáng và bóng đổ lên khuôn mặt.",
    task: "detection2d",
    attacks: [
      { id: "error_overexposure", name: "Overexposure", severity: 4, eps: "Severity 4", alpha: "Brightness", iters: 1 },
      { id: "gaussian_noise", name: "Gaussian Noise", severity: 2, eps: "Severity 2", alpha: "L2", iters: 1 },
    ],
  },
  {
    id: "preset_gradient_pii",
    name: "⚡ Tấn công PII Gradient (White-box)",
    desc: "Tổ hợp tấn công PGD nhằm qua mặt hệ thống che mờ khuôn mặt tự động.",
    task: "detection2d",
    attacks: [
      { id: "pgd", name: "PGD (Projected Gradient)", severity: 3, eps: "8/255", alpha: "2/255", iters: 20 },
    ],
  },
];

export const ATTACK_CATEGORIES = [
  {
    category: "Chuyển động & Góc nhìn (Egocentric)",
    task_compatibility: ["detection2d", "segmentation", "classification"],
    attacks: [
      {
        id: "biomechanical_blur",
        name: "Biomechanical Blur",
        fullName: "Nhòe chuyển động sinh học 2Hz",
        norm: "Spatial",
        defaultEps: "Severity 3",
        isWeather: false,
        task_compatibility: ["detection2d", "segmentation"],
        desc: "Mô phỏng quỹ đạo rung lắc nhịp độ 2Hz khi người đeo camera (Ego4D) đi bộ.",
        previewClean: "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=600&auto=format&fit=crop&q=80",
        previewAttacked: "https://images.unsplash.com/photo-1517411032315-54ef2cb783bb?w=600&auto=format&fit=crop&q=80",
      },
      {
        id: "motion_blur",
        name: "Motion Blur",
        fullName: "Mờ do xoay camera đột ngột",
        norm: "Spatial",
        defaultEps: "Severity 3",
        isWeather: false,
        task_compatibility: ["detection2d", "segmentation"],
        desc: "Làm nhòe tuyến tính khi camera xoay ngang tốc độ cao.",
        previewClean: "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=600&auto=format&fit=crop&q=80",
        previewAttacked: "https://images.unsplash.com/photo-1509114397022-ed747cca3f65?w=600&auto=format&fit=crop&q=80",
      },
    ],
  },
  {
    category: "Môi trường & Chiếu sáng",
    task_compatibility: ["detection2d", "segmentation"],
    attacks: [
      {
        id: "error_overexposure",
        name: "Overexposure",
        fullName: "Ngược sáng & Dư sáng",
        norm: "Brightness",
        defaultEps: "Severity 3",
        isWeather: true,
        task_compatibility: ["detection2d", "segmentation"],
        desc: "Mô phỏng nguồn sáng chói từ phía sau làm tối khuôn mặt (chống ngược sáng) hoặc cháy sáng.",
        previewClean: "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=600&auto=format&fit=crop&q=80",
        previewAttacked: "https://images.unsplash.com/photo-1515694346937-94d85e41e6f0?w=600&auto=format&fit=crop&q=80",
      },
      {
        id: "gaussian_noise",
        name: "Gaussian Noise",
        fullName: "Nhiễu hạt cảm biến ISO cao",
        norm: "L2",
        defaultEps: "Severity 2",
        isWeather: true,
        task_compatibility: ["detection2d", "segmentation", "classification"],
        desc: "Thêm nhiễu phân phối chuẩn Gaussian mô phỏng nhiễu cảm biến camera an ninh thiếu sáng.",
        previewClean: "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=600&auto=format&fit=crop&q=80",
        previewAttacked: "https://images.unsplash.com/photo-1550684848-fac1c5b4e853?w=600&auto=format&fit=crop&q=80",
      },
    ]
  },
  {
    category: "Gradient-based (White-box)",
    task_compatibility: ["detection2d", "segmentation", "classification"],
    attacks: [
      {
        id: "pgd",
        name: "PGD",
        fullName: "Projected Gradient Descent",
        norm: "Linf",
        defaultEps: "8/255",
        task_compatibility: ["detection2d", "segmentation", "classification"],
        desc: "Tấn công gradient lặp từng bước có phép chiếu, chuẩn vàng để qua mặt hệ thống nhận diện khuôn mặt.",
        previewClean: "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=600&auto=format&fit=crop&q=80",
        previewAttacked: "https://images.unsplash.com/photo-1550684848-fac1c5b4e853?w=600&auto=format&fit=crop&q=80",
      },
      {
        id: "fgsm",
        name: "FGSM",
        fullName: "Fast Gradient Sign Method",
        norm: "Linf",
        defaultEps: "8/255",
        task_compatibility: ["detection2d", "segmentation", "classification"],
        desc: "Tạo nhiễu đối kháng nhanh một bước (single-step).",
        previewClean: "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=600&auto=format&fit=crop&q=80",
        previewAttacked: "https://images.unsplash.com/photo-1550684848-fac1c5b4e853?w=600&auto=format&fit=crop&q=80",
      },
    ],
  }
];
