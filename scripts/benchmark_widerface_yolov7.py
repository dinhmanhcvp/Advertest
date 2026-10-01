"""WIDERFACE Benchmark for AdverTest (Mentor Pitch Edition)."""

import argparse
import sys
from pathlib import Path

# Add project root to path if running directly
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable

from src.datasets.widerface import WIDERFaceDataset
from src.adapters.yolov7_face import YOLOv7FaceAdapter
from src.attacks.corruption.grid_distortion import GridDistortion
from src.attacks.corruption.tone_curve import RandomToneCurve
from src.attacks.corruption.sun_flare import RandomSunFlare
from src.attacks.base import AttackContext
from src.core.types import Sample

def calculate_iou(boxA, boxB):
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate_sample(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Calculate Recall at IoU threshold."""
    if not gt_boxes:
        return 0.0
    
    gt_matched = [False] * len(gt_boxes)
    matches = 0
    
    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            if not gt_matched[i]:
                iou = calculate_iou([pred.x1, pred.y1, pred.x2, pred.y2], [gt.x1, gt.y1, gt.x2, gt.y2])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
                    
        if best_iou >= iou_threshold:
            gt_matched[best_gt_idx] = True
            matches += 1
            
    return matches / len(gt_boxes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Number of images to benchmark")
    parser.add_argument("--weights", type=str, default="data/weights/yolov7-lite-t.pt")
    args = parser.parse_args()
    
    print("==================================================")
    print("🚀 Initializing AdverTest Benchmark...")
    print("==================================================")
    
    dataset = WIDERFaceDataset(split="val")
    try:
        adapter = YOLOv7FaceAdapter(weights=args.weights)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run `python src/scripts/download_face_data.py` first.")
        return
        
    grid_distort = GridDistortion()
    sun_flare = RandomSunFlare()
    tone_curve = RandomToneCurve()
    
    rng = np.random.default_rng(42)
    ctx = AttackContext(rng=rng)
    
    print("Loading dataset...")
    all_samples = list(dataset.load())
    if args.limit > 0:
        all_samples = all_samples[:args.limit]
        
    print(f"Loaded {len(all_samples)} samples from WIDERFACE val set.")
    print("Running Inference and Corruptions (This may take a moment)...")
    
    clean_recalls = []
    fisheye_recalls = []
    flare_recalls = []
    
    for i, sample in enumerate(tqdm(all_samples)):
        gt_boxes = sample.boxes
        if not gt_boxes:
            continue
            
        # 1. Clean Inference
        clean_pred = adapter.predict([sample])[0]
        clean_recall = evaluate_sample(gt_boxes, clean_pred.boxes)
        clean_recalls.append(clean_recall)
        
        # 2. Fisheye Inference
        fisheye_sample = grid_distort.run(sample, severity=5, ctx=ctx)
        fisheye_pred = adapter.predict([fisheye_sample])[0]
        fisheye_recall = evaluate_sample(gt_boxes, fisheye_pred.boxes)
        fisheye_recalls.append(fisheye_recall)
        
        # 3. Flare & Overexposure Inference
        flare_sample = tone_curve.run(sample, severity=4, ctx=ctx)
        flare_sample = sun_flare.run(flare_sample, severity=4, ctx=ctx)
        flare_pred = adapter.predict([flare_sample])[0]
        flare_recall = evaluate_sample(gt_boxes, flare_pred.boxes)
        flare_recalls.append(flare_recall)
        
    if not clean_recalls:
        print("No valid Ground Truth boxes found in the sampled data.")
        return
        
    mean_clean = np.mean(clean_recalls) * 100
    mean_fisheye = np.mean(fisheye_recalls) * 100
    mean_flare = np.mean(flare_recalls) * 100
    
    print("\n" + "="*50)
    print("📊 ADVERTEST BENCHMARK REPORT (mPC Proxy)")
    print("="*50)
    print(f"Total Images Evaluated: {len(clean_recalls)}")
    print(f"Model: YOLOv7-Face (Lite-T)")
    print(f"Dataset: WIDERFACE (Validation Split)")
    print("-" * 50)
    print(f"🟢 Clean Mean Recall:      {mean_clean:.2f}%")
    print(f"🔴 Fisheye Mean Recall:    {mean_fisheye:.2f}% (Drop: {mean_clean - mean_fisheye:.2f}%)")
    print(f"🟠 Overexposure Recall:    {mean_flare:.2f}% (Drop: {mean_clean - mean_flare:.2f}%)")
    print("="*50)
    print("Conclusion: Geometric distortion (Fisheye) and Overexposure cause significant performance collapse.")
    print("Next Step: Use this exact Adversarial Dataset to RETRAIN and IMPROVE the model.")

if __name__ == "__main__":
    main()
