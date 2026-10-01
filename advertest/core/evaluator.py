"""Blind Evaluation Gate for AdverTest Pipeline.

Evaluates the retrained model against a held-out Gold Standard Ego4D
validation set to ensure robustness improvements without catastrophic forgetting.

Requirements:
    pip install pycocotools
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    raise ImportError("pycocotools is required: pip install pycocotools")

from advertest.core.inference_engine import Yolov7Evaluator
from advertest.eval.temporal_tracker import TemporalTracker

logger = logging.getLogger(__name__)


class RobustnessEvaluator:
    """The Blind Evaluation Gate for Model Deployment."""

    def __init__(self, old_weights_path: str, new_weights_path: str, repo_dir: str = "third_party/yolov7_face"):
        self.old_evaluator = Yolov7Evaluator(weights=old_weights_path, repo_dir=repo_dir)
        self.new_evaluator = Yolov7Evaluator(weights=new_weights_path, repo_dir=repo_dir)

    def _convert_yolo_dir_to_coco_dict(self, image_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Convert a directory of images and YOLO labels into COCO dictionary format."""
        import cv2
        
        coco_gt = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": "Face"}]
        }
        
        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in image_exts])
        
        anno_id = 1
        image_id_map = {}
        
        for img_id, img_path in enumerate(image_paths, 1):
            image_id_map[img_path.name] = img_id
            
            # We need width/height for COCO format
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            coco_gt["images"].append({
                "id": img_id,
                "file_name": img_path.name,
                "width": w,
                "height": h
            })
            
            label_path = img_path.with_suffix(".txt")
            if label_path.exists():
                for line in label_path.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        xc, yc, nw, nh = map(float, parts[1:5])
                        
                        # YOLO to COCO [x_min, y_min, width, height] in pixels
                        bw = nw * w
                        bh = nh * h
                        bx = (xc * w) - (bw / 2)
                        by = (yc * h) - (bh / 2)
                        
                        coco_gt["annotations"].append({
                            "id": anno_id,
                            "image_id": img_id,
                            "category_id": class_id,
                            "bbox": [bx, by, bw, bh],
                            "area": bw * bh,
                            "iscrowd": 0
                        })
                        anno_id += 1
                        
        return coco_gt, image_id_map

    def _run_coco_eval(self, evaluator: Yolov7Evaluator, image_dir: Path, coco_gt_dict: Dict[str, Any], image_id_map: Dict[str, int]) -> float:
        """Run inference on the directory and compute mAP@0.5:0.95 using pycocotools."""
        import cv2
        
        # Write ground truth to a temporary file for pycocotools
        gt_path = image_dir / "temp_coco_gt.json"
        with open(gt_path, "w") as f:
            json.dump(coco_gt_dict, f)
            
        coco_gt = COCO(str(gt_path))
        gt_path.unlink() # Cleanup
        
        coco_dt_list = []
        tracker = TemporalTracker(max_age=3, min_iou=0.3)
        
        # Run inference
        for img_info in coco_gt_dict["images"]:
            img_path = image_dir / img_info["file_name"]
            img_id = img_info["id"]
            
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                continue
                
            pred_boxes, pred_scores = evaluator.infer_single(image_bgr)
            h, w = image_bgr.shape[:2]
            
            # Format predictions for TemporalTracker: [x1, y1, x2, y2, conf]
            raw_detections = []
            for box, score in zip(pred_boxes, pred_scores):
                bw = box.width * w
                bh = box.height * h
                bx = (box.x_center * w) - (bw / 2)
                by = (box.y_center * h) - (bh / 2)
                raw_detections.append([bx, by, bx + bw, by + bh, score])
                
            # Temporal Persistence step
            smoothed_detections = tracker.update(raw_detections)
            
            for det in smoothed_detections:
                x1, y1, x2, y2, conf = det
                bw = x2 - x1
                bh = y2 - y1
                
                coco_dt_list.append({
                    "image_id": img_id,
                    "category_id": 0,  # Assuming class 0 (Face) as defined in _convert_yolo_dir_to_coco_dict
                    "bbox": [x1, y1, bw, bh],
                    "score": conf
                })
                
        if not coco_dt_list:
            return 0.0
            
        coco_dt = coco_gt.loadRes(coco_dt_list)
        
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # return mAP @ IoU=0.50:0.95 (which is the first index in stats)
        return float(coco_eval.stats[0])

    def calculate_map_drop(
        self,
        clean_val_dir: str | Path,
        corrupted_val_dir: str | Path
    ) -> Dict[str, Any]:
        """Evaluate both models on Clean and Corrupted validation sets.
        
        Applies the deployment gate logic:
        - Base_mAP cannot drop by more than 1%.
        - Corrupted_mAP (mPC) must improve by at least 5%.
        """
        clean_val_dir = Path(clean_val_dir)
        corrupted_val_dir = Path(corrupted_val_dir)
        
        logger.info("Preparing COCO annotations for evaluation...")
        clean_gt, clean_map = self._convert_yolo_dir_to_coco_dict(clean_val_dir)
        corrupted_gt, corrupted_map = self._convert_yolo_dir_to_coco_dict(corrupted_val_dir)
        
        logger.info("Evaluating Original Model (Baseline)...")
        old_base_map = self._run_coco_eval(self.old_evaluator, clean_val_dir, clean_gt, clean_map)
        old_corr_map = self._run_coco_eval(self.old_evaluator, corrupted_val_dir, corrupted_gt, corrupted_map)
        
        logger.info("Evaluating New Retrained Model...")
        new_base_map = self._run_coco_eval(self.new_evaluator, clean_val_dir, clean_gt, clean_map)
        new_corr_map = self._run_coco_eval(self.new_evaluator, corrupted_val_dir, corrupted_gt, corrupted_map)
        
        # Logic Gate
        base_drop = old_base_map - new_base_map
        corr_improvement = new_corr_map - old_corr_map
        
        # Convert metrics to percentages
        base_drop_pct = base_drop * 100
        corr_improvement_pct = corr_improvement * 100
        
        status = "PASS"
        reasons = []
        
        if base_drop_pct > 1.0:
            status = "FAIL"
            reasons.append(f"Catastrophic Forgetting: Base mAP dropped by {base_drop_pct:.2f}% (Limit: 1.0%).")
            
        if corr_improvement_pct < 5.0:
            status = "FAIL"
            reasons.append(f"Insufficient Robustness: Corrupted mAP improved by {corr_improvement_pct:.2f}% (Required: 5.0%).")
            
        if status == "PASS":
            reasons.append("Model met all robustness improvement and retention criteria.")

        results = {
            "status": status,
            "metrics": {
                "original_model": {
                    "base_map": old_base_map,
                    "corrupted_map": old_corr_map
                },
                "new_model": {
                    "base_map": new_base_map,
                    "corrupted_map": new_corr_map
                }
            },
            "deltas": {
                "base_drop_pct": base_drop_pct,
                "corrupted_improvement_pct": corr_improvement_pct
            },
            "reasons": reasons
        }
        
        logger.info("=========================================")
        logger.info("BLIND EVALUATION GATE: %s", status)
        for r in reasons:
            logger.info(" - %s", r)
        logger.info("=========================================")
        
        return results

# ──────────────────────────── CLI entry point ──────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="AdverTest Blind Evaluation Gate")
    parser.add_argument("--old-weights", type=str, required=True, help="Path to original model weights")
    parser.add_argument("--new-weights", type=str, required=True, help="Path to retrained model weights")
    parser.add_argument("--clean-val", type=str, required=True, help="Directory of clean Ego4D validation images")
    parser.add_argument("--corr-val", type=str, required=True, help="Directory of corrupted Ego4D validation images")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    
    evaluator = RobustnessEvaluator(args.old_weights, args.new_weights)
    evaluator.calculate_map_drop(args.clean_val, args.corr_val)

if __name__ == "__main__":
    main()
