"""Real YOLOv7-Face Inference & Hard Negative Extraction.

NO MOCKS.  This module loads a real YOLOv7-face PyTorch checkpoint, runs
inference on a folder of images, compares predictions against YOLO-format
ground truth labels, computes IoU, and filters Hard Negatives into a
dedicated output directory.

Requirements:
    pip install torch torchvision opencv-python numpy

Weight path resolution order:
    1. Constructor argument ``weights``
    2. Environment variable ``YOLOV7_FACE_WEIGHTS``
    3. Fallback ``data/weights/yolov7-lite-t.pt``

YOLOv7-face repo must be cloned into ``third_party/yolov7_face/``::

    git clone https://github.com/derronqi/yolov7-face.git third_party/yolov7_face
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────── data types ──────

@dataclass
class YOLOBox:
    """YOLO-format bounding box  [class_id  x_center  y_center  w  h]  (normalised 0-1)."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        x1 = int((self.x_center - self.width / 2) * img_w)
        y1 = int((self.y_center - self.height / 2) * img_h)
        x2 = int((self.x_center + self.width / 2) * img_w)
        y2 = int((self.y_center + self.height / 2) * img_h)
        return x1, y1, x2, y2

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float,
                  img_w: int, img_h: int, class_id: int = 0) -> "YOLOBox":
        return cls(
            class_id=class_id,
            x_center=((x1 + x2) / 2) / img_w,
            y_center=((y1 + y2) / 2) / img_h,
            width=(x2 - x1) / img_w,
            height=(y2 - y1) / img_h,
        )

    def to_label_str(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


@dataclass
class DetectionResult:
    """Result for one image after inference + GT comparison."""
    image_path: Path
    gt_boxes: list[YOLOBox]
    pred_boxes: list[YOLOBox]
    pred_scores: list[float]
    ious: list[float]              # best IoU for each GT box
    is_hard_negative: bool = False  # True if any GT box is a hard negative


@dataclass
class HardNegative:
    """A single hard-negative sample ready for Label Studio upload."""
    image_path: Path
    gt_boxes: list[YOLOBox]
    pred_boxes: list[YOLOBox]
    pred_scores: list[float]
    max_iou: float
    min_conf: float
    reason: str                    # "low_iou" | "low_conf" | "missed"


# ──────────────────────────────────── IoU computation ──────

def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# ──────────────────────────────────── I/O helpers ──────

def load_yolo_labels(path: Path) -> list[YOLOBox]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        boxes.append(YOLOBox(
            class_id=int(parts[0]),
            x_center=float(parts[1]),
            y_center=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4]),
        ))
    return boxes


def save_yolo_labels(boxes: list[YOLOBox], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for box in boxes:
            f.write(box.to_label_str() + "\n")


# ──────────────────────────────────── Evaluator ──────

class Yolov7Evaluator:
    """Real YOLOv7-face evaluator — loads a PyTorch checkpoint and runs
    production inference.  Zero mocks.

    Parameters
    ----------
    weights : str or Path
        Path to the ``.pt`` checkpoint file.  Resolved in order:
        1. This argument.
        2. ``YOLOV7_FACE_WEIGHTS`` environment variable.
        3. ``data/weights/yolov7-lite-t.pt`` fallback.
    repo_dir : str or Path
        Path to the cloned ``yolov7-face`` repository (contains ``models/``,
        ``utils/``).  Default: ``third_party/yolov7_face``.
    score_threshold : float
        Minimum confidence to keep a detection.
    iou_threshold : float
        NMS IoU threshold.
    img_size : int
        Inference input size (pixels).
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        repo_dir: str | Path = "third_party/yolov7_face",
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
    ) -> None:
        # Resolve weights path
        if weights is not None:
            self.weights = Path(weights)
        elif os.environ.get("YOLOV7_FACE_WEIGHTS"):
            self.weights = Path(os.environ["YOLOV7_FACE_WEIGHTS"])
        else:
            self.weights = Path("data/weights/yolov7-lite-t.pt")

        self.repo_dir = Path(repo_dir)
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        self._model = None
        self._device = None

        # Inject the yolov7-face repo into sys.path for its internal imports
        repo_str = str(self.repo_dir.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    # ── Lazy model loading ──

    def _load_model(self):
        """Load the YOLOv7-face PyTorch model.  Called once on first inference."""
        if self._model is not None:
            return

        import torch
        from models.experimental import attempt_load

        if not self.weights.exists():
            raise FileNotFoundError(
                f"YOLOv7-face weights not found at: {self.weights.resolve()}\n"
                f"Download them or set YOLOV7_FACE_WEIGHTS env var."
            )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading YOLOv7-face weights from %s onto %s", self.weights, self._device)
        self._model = attempt_load(str(self.weights), map_location=self._device)
        self._model.eval()
        logger.info("YOLOv7-face model loaded successfully.")

    # ── Single-image inference ──

    def infer_single(self, image_bgr: np.ndarray) -> tuple[list[YOLOBox], list[float]]:
        """Run inference on one BGR uint8 image.

        Returns
        -------
        boxes : list[YOLOBox]
            Predicted boxes in normalised YOLO format.
        scores : list[float]
            Confidence score for each box.
        """
        self._load_model()
        import torch
        from utils.general import non_max_suppression_face, scale_coords
        from utils.datasets import letterbox

        h0, w0 = image_bgr.shape[:2]

        # Preprocess
        img = letterbox(image_bgr, self.img_size, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC → CHW, BGR → RGB
        img = np.ascontiguousarray(img)

        tensor = torch.from_numpy(img).to(self._device).float() / 255.0
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            raw_pred = self._model(tensor)[0]

        # NMS
        nms_pred = non_max_suppression_face(raw_pred, self.score_threshold, self.iou_threshold)

        boxes: list[YOLOBox] = []
        scores: list[float] = []

        if len(nms_pred) > 0 and nms_pred[0] is not None:
            det = nms_pred[0].cpu().numpy()
            # Scale coords back to original image
            det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], image_bgr.shape).round()

            for *xyxy, conf, cls_id, *_ in det:
                if conf >= self.score_threshold:
                    box = YOLOBox.from_xyxy(
                        float(xyxy[0]), float(xyxy[1]),
                        float(xyxy[2]), float(xyxy[3]),
                        w0, h0, class_id=int(cls_id),
                    )
                    boxes.append(box)
                    scores.append(float(conf))

        return boxes, scores

    # ── Folder-level evaluation ──

    def evaluate_folder(
        self,
        image_dir: str | Path,
        iou_hard_neg_threshold: float = 0.5,
        conf_hard_neg_threshold: float = 0.4,
    ) -> list[DetectionResult]:
        """Run inference on every image in ``image_dir``, compare with GT
        labels (YOLO ``.txt`` files), and tag hard negatives.

        A sample is a hard negative if:
        - Any GT box has best-matching prediction IoU < ``iou_hard_neg_threshold``, OR
        - Any prediction's confidence < ``conf_hard_neg_threshold``, OR
        - A GT box has no matching prediction at all (missed detection).
        """
        image_dir = Path(image_dir)
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        results: list[DetectionResult] = []

        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in image_exts
        )

        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        logger.info("Evaluating %d images in %s", len(image_paths), image_dir)

        for img_path in image_paths:
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                logger.warning("Cannot read %s — skipping", img_path)
                continue

            h, w = image_bgr.shape[:2]
            gt_boxes = load_yolo_labels(img_path.with_suffix(".txt"))
            pred_boxes, pred_scores = self.infer_single(image_bgr)

            # Compute IoU: for each GT box, find best matching prediction
            ious: list[float] = []
            for gt in gt_boxes:
                gt_xyxy = gt.to_xyxy(w, h)
                best_iou = 0.0
                for pred in pred_boxes:
                    pred_xyxy = pred.to_xyxy(w, h)
                    iou = compute_iou(gt_xyxy, pred_xyxy)
                    best_iou = max(best_iou, iou)
                ious.append(best_iou)

            # Determine if this is a hard negative
            is_hard = False
            if gt_boxes:
                # Case 1: Any GT box with low IoU or missed entirely
                if any(iou < iou_hard_neg_threshold for iou in ious):
                    is_hard = True
                # Case 2: Any prediction with low confidence
                if any(score < conf_hard_neg_threshold for score in pred_scores):
                    is_hard = True
                # Case 3: GT boxes exist but no predictions at all
                if not pred_boxes:
                    is_hard = True

            results.append(DetectionResult(
                image_path=img_path,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                ious=ious,
                is_hard_negative=is_hard,
            ))

        total_hard = sum(1 for r in results if r.is_hard_negative)
        logger.info(
            "Evaluation complete: %d/%d images are hard negatives (IoU<%.2f or Conf<%.2f)",
            total_hard, len(results), iou_hard_neg_threshold, conf_hard_neg_threshold,
        )
        return results

    # ── Hard negative extraction ──

    def extract_hard_negatives(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.4,
    ) -> list[HardNegative]:
        """Run full evaluation and copy hard negatives (images + labels +
        prediction labels) into ``output_dir``.

        Returns the list of HardNegative objects for downstream use
        (e.g. Label Studio upload).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.evaluate_folder(
            image_dir,
            iou_hard_neg_threshold=iou_threshold,
            conf_hard_neg_threshold=conf_threshold,
        )

        hard_negatives: list[HardNegative] = []

        for result in results:
            if not result.is_hard_negative:
                continue

            # Determine reason
            max_iou = min(result.ious) if result.ious else 0.0
            min_conf = min(result.pred_scores) if result.pred_scores else 0.0
            if not result.pred_boxes and result.gt_boxes:
                reason = "missed"
            elif max_iou < iou_threshold:
                reason = "low_iou"
            else:
                reason = "low_conf"

            hn = HardNegative(
                image_path=result.image_path,
                gt_boxes=result.gt_boxes,
                pred_boxes=result.pred_boxes,
                pred_scores=result.pred_scores,
                max_iou=max_iou,
                min_conf=min_conf,
                reason=reason,
            )
            hard_negatives.append(hn)

            # Copy image
            dst_img = output_dir / result.image_path.name
            shutil.copy2(result.image_path, dst_img)

            # Save GT labels
            gt_label_path = output_dir / f"{result.image_path.stem}_gt.txt"
            save_yolo_labels(result.gt_boxes, gt_label_path)

            # Save prediction labels
            pred_label_path = output_dir / f"{result.image_path.stem}_pred.txt"
            save_yolo_labels(result.pred_boxes, pred_label_path)

            # Save metadata
            meta_path = output_dir / f"{result.image_path.stem}_meta.txt"
            with open(meta_path, "w") as f:
                f.write(f"reason={reason}\n")
                f.write(f"max_iou={max_iou:.4f}\n")
                f.write(f"min_conf={min_conf:.4f}\n")
                f.write(f"n_gt={len(result.gt_boxes)}\n")
                f.write(f"n_pred={len(result.pred_boxes)}\n")

        logger.info(
            "Extracted %d hard negatives to %s",
            len(hard_negatives), output_dir.resolve(),
        )
        return hard_negatives


# ──────────────────────────── CLI entry point ──────

def main() -> None:
    """Standalone CLI for hard-negative extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="YOLOv7-face Evaluator — Extract hard negatives from a folder of images."
    )
    parser.add_argument("--images", type=str, required=True,
                        help="Directory containing images and YOLO .txt labels")
    parser.add_argument("--output", type=str, default="data/hard_negatives",
                        help="Directory to save hard negatives")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to yolov7-face .pt checkpoint")
    parser.add_argument("--repo", type=str, default="third_party/yolov7_face",
                        help="Path to cloned yolov7-face repository")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold below which a sample is a hard negative")
    parser.add_argument("--conf-threshold", type=float, default=0.4,
                        help="Confidence threshold below which a sample is a hard negative")
    parser.add_argument("--img-size", type=int, default=640)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    evaluator = Yolov7Evaluator(
        weights=args.weights,
        repo_dir=args.repo,
        img_size=args.img_size,
    )

    hard_negs = evaluator.extract_hard_negatives(
        image_dir=args.images,
        output_dir=args.output,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
    )

    print(f"\n{'='*60}")
    print(f"  Hard Negatives Extracted: {len(hard_negs)}")
    print(f"  Output: {Path(args.output).resolve()}")
    reasons = {}
    for hn in hard_negs:
        reasons[hn.reason] = reasons.get(hn.reason, 0) + 1
    for reason, count in sorted(reasons.items()):
        print(f"    {reason}: {count}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
