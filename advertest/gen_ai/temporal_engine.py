"""Temporal Sequence Processor for Video-to-Video GenAI.

Upgrades Phase 2 from Image-to-Image translation to Video-to-Video translation
(e.g., using Stable Video Diffusion or AnimateDiff). Extracts Optical Flow to
enforce temporal consistency and applies Exponential Moving Average (EMA)
filtering to prevent bounding box jitter in generated frames.

Requirements:
    pip install opencv-python numpy
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np

# Use the core YOLOBox data structure
from advertest.core.inference_engine import YOLOBox

logger = logging.getLogger(__name__)


class TemporalSequenceProcessor:
    """Processes video frame sequences for temporally consistent GenAI."""

    def __init__(self, ema_alpha: float = 0.4) -> None:
        """
        Parameters
        ----------
        ema_alpha : float
            Smoothing factor for the Exponential Moving Average filter.
            Lower = more smoothing (less responsive). Higher = less smoothing.
        """
        self.ema_alpha = ema_alpha

    def export_to_svd_format(self, input_dir: str | Path, output_dir: str | Path, fps: int = 30) -> None:
        """Prepare frame sequences and optical flow data for Stable Video Diffusion.

        Calculates dense optical flow using Farneback's algorithm to provide
        motion conditioning for video generation pipelines.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        frames_out = output_dir / "frames"
        flow_out = output_dir / "flow"
        frames_out.mkdir(parents=True, exist_ok=True)
        flow_out.mkdir(parents=True, exist_ok=True)

        image_exts = {".jpg", ".jpeg", ".png"}
        frame_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in image_exts])
        
        if not frame_paths:
            logger.error("No frames found in %s", input_dir)
            return

        logger.info("Exporting %d frames for SVD/AnimateDiff to %s", len(frame_paths), output_dir)
        
        manifest: List[Dict[str, Any]] = []
        
        # Read the first frame
        prev_img = cv2.imread(str(frame_paths[0]))
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        
        # Copy first frame
        cv2.imwrite(str(frames_out / frame_paths[0].name), prev_img)
        manifest.append({
            "frame": frame_paths[0].name,
            "flow_file": None,  # First frame has no previous optical flow
            "fps": fps
        })

        for i in range(1, len(frame_paths)):
            curr_path = frame_paths[i]
            curr_img = cv2.imread(str(curr_path))
            curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
            
            # Calculate dense optical flow (Farneback)
            # flow shape will be (H, W, 2)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Save flow as uncompressed NumPy array for lossless consumption by GenAI
            flow_filename = f"flow_{i:04d}.npy"
            np.save(str(flow_out / flow_filename), flow)
            
            # Save current frame
            cv2.imwrite(str(frames_out / curr_path.name), curr_img)
            
            manifest.append({
                "frame": curr_path.name,
                "flow_file": flow_filename,
                "fps": fps
            })
            
            # Step forward
            prev_gray = curr_gray

        # Save SVD manifest
        with open(output_dir / "svd_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
            
        logger.info("Extraction complete. Optical flow maps saved as NumPy arrays.")

    def temporal_smoothing(self, labels_dir: str | Path, output_dir: str | Path) -> None:
        """Apply EMA to eliminate inter-frame jitter in bounding boxes.
        
        Reads a sequence of YOLO labels (assumed to be ordered chronologically),
        smooths the coordinates across time, and saves the stabilized labels.
        """
        labels_dir = Path(labels_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        label_paths = sorted(labels_dir.glob("*.txt"))
        
        if not label_paths:
            logger.error("No YOLO label files found in %s", labels_dir)
            return
            
        logger.info("Applying Temporal Smoothing (EMA, alpha=%.2f) to %d frames...", 
                    self.ema_alpha, len(label_paths))

        # We track smoothed boxes for each class ID across frames.
        # Dictionary format: {class_id: last_smoothed_box}
        # Note: This is a simplified tracker assuming max 1 instance per class per frame.
        # For multiple instances of the same class, a robust DeepSORT tracker would be needed.
        trackers: Dict[int, YOLOBox] = {}
        
        for path in label_paths:
            smoothed_boxes_for_frame = []
            
            for line in path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    current_box = YOLOBox(
                        class_id=class_id,
                        x_center=float(parts[1]),
                        y_center=float(parts[2]),
                        width=float(parts[3]),
                        height=float(parts[4])
                    )
                    
                    if class_id in trackers:
                        # Apply Exponential Moving Average
                        prev_box = trackers[class_id]
                        smoothed_box = YOLOBox(
                            class_id=class_id,
                            x_center=(self.ema_alpha * current_box.x_center) + ((1 - self.ema_alpha) * prev_box.x_center),
                            y_center=(self.ema_alpha * current_box.y_center) + ((1 - self.ema_alpha) * prev_box.y_center),
                            width=(self.ema_alpha * current_box.width) + ((1 - self.ema_alpha) * prev_box.width),
                            height=(self.ema_alpha * current_box.height) + ((1 - self.ema_alpha) * prev_box.height)
                        )
                    else:
                        # First appearance
                        smoothed_box = current_box
                        
                    # Update tracker memory
                    trackers[class_id] = smoothed_box
                    smoothed_boxes_for_frame.append(smoothed_box)
            
            # Save smoothed boxes
            out_path = output_dir / path.name
            with open(out_path, "w") as f:
                for b in smoothed_boxes_for_frame:
                    f.write(b.to_label_str() + "\n")
                    
        logger.info("Temporal smoothing complete. Stabilized labels saved to %s", output_dir)


# ──────────────────────────── CLI entry point ──────

def main() -> None:
    """Standalone CLI for Temporal Sequence Processing."""
    import argparse

    parser = argparse.ArgumentParser(description="Temporal Sequence Processor for V2V GenAI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Export to SVD
    export_parser = subparsers.add_parser("export-flow", help="Extract optical flow for SVD/AnimateDiff")
    export_parser.add_argument("--input", type=str, required=True, help="Directory of input video frames")
    export_parser.add_argument("--output", type=str, required=True, help="Output directory for frames + flow")
    
    # Smooth labels
    smooth_parser = subparsers.add_parser("smooth-boxes", help="Apply EMA to bounding boxes to prevent jitter")
    smooth_parser.add_argument("--labels", type=str, required=True, help="Directory of raw generated YOLO labels")
    smooth_parser.add_argument("--output", type=str, required=True, help="Output directory for smoothed labels")
    smooth_parser.add_argument("--alpha", type=float, default=0.4, help="EMA smoothing factor (0.0 to 1.0)")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    
    processor = TemporalSequenceProcessor(ema_alpha=getattr(args, "alpha", 0.4))
    
    if args.command == "export-flow":
        processor.export_to_svd_format(args.input, args.output)
    elif args.command == "smooth-boxes":
        processor.temporal_smoothing(args.labels, args.output)


if __name__ == "__main__":
    main()
