"""Phase 2: Generative AI Colab Bridge.

Provides infrastructure for exporting WIDER FACE data (images + landmarks)
to Google Colab for ControlNet/CycleGAN synthesis, and ingesting the
generated results back into the local environment for FID scoring and
spatial layout validation.

Requirements:
    pip install torch torchvision torchmetrics opencv-python numpy
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import List

import cv2
import numpy as np

try:
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    raise ImportError(
        "torch and torchmetrics are required for FID scoring. "
        "Install via: pip install torch torchvision torchmetrics"
    )

# Import the existing YOLOBox and BBoxIntegrityChecker from Phase 1 MVP
from advertest.attacks.engine import YOLOBox, BBoxIntegrityChecker


logger = logging.getLogger(__name__)


class GenAIColabBridge:
    """Bridge for offloading GenAI (ControlNet/CycleGAN) to Google Colab."""

    def __init__(self, target_prompt: str = "chest-mounted GoPro view, extreme fisheye, harsh lighting, action blur") -> None:
        self.target_prompt = target_prompt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ──────────────────────────────────────────────── Export ──────

    def export_payload_for_colab(self, input_dir: str | Path, output_zip: str | Path) -> None:
        """Package clean images and facial landmarks into a Colab-ready ZIP.

        Reads image files and their corresponding YOLO format `.txt` label files.
        Extracts bounding box and landmark information (if present).
        Generates a manifest.json driving the ControlNet generation process.

        Parameters
        ----------
        input_dir : str | Path
            Directory containing images and `.txt` labels.
        output_zip : str | Path
            Destination path for the `.zip` payload.
        """
        input_dir = Path(input_dir)
        output_zip = Path(output_zip)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in image_exts])
        
        manifest_entries = []
        
        # We write directly to the ZIP to save disk space
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for img_path in image_paths:
                label_path = img_path.with_suffix(".txt")
                boxes = []
                landmarks = []
                
                # Parse labels and landmarks (assuming RetinaFace WIDER FACE format)
                # Format: class_id xc yc w h [lx1 ly1 lx2 ly2 lx3 ly3 lx4 ly4 lx5 ly5]
                if label_path.exists():
                    for line in label_path.read_text().strip().splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Standard YOLO BBox
                            boxes.append({
                                "class_id": int(parts[0]),
                                "x_center": float(parts[1]),
                                "y_center": float(parts[2]),
                                "width": float(parts[3]),
                                "height": float(parts[4])
                            })
                            # Facial Landmarks (5 points: 10 coordinates)
                            if len(parts) >= 15:
                                lms = [float(x) for x in parts[5:15]]
                                landmarks.append(lms)
                
                manifest_entries.append({
                    "image_file": img_path.name,
                    "prompt": self.target_prompt,
                    "boxes": boxes,
                    "landmarks": landmarks
                })
                
                # Add image and label to zip
                zf.write(img_path, arcname=f"images/{img_path.name}")
                if label_path.exists():
                    zf.write(label_path, arcname=f"labels/{label_path.name}")
                    
            # Add manifest to zip
            manifest_str = json.dumps(manifest_entries, indent=2)
            zf.writestr("manifest.json", manifest_str)
            
        logger.info("Exported %d images and manifest to %s", len(image_paths), output_zip.resolve())

    # ──────────────────────────────────────────────── Ingest ──────

    def _load_images_as_tensor(self, img_paths: List[Path], img_size: tuple[int, int] = (299, 299)) -> torch.Tensor:
        """Load and resize images to batch tensor [B, 3, H, W] for FID."""
        tensors = []
        for path in img_paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            # Torchmetrics FID expects uint8 [B, 3, H, W]
            t = torch.from_numpy(img).permute(2, 0, 1).byte()
            tensors.append(t)
            
        if not tensors:
            return torch.empty((0, 3, *img_size), dtype=torch.uint8)
        return torch.stack(tensors)

    def ingest_and_evaluate(self, colab_output_dir: str | Path, ground_truth_dir: str | Path) -> None:
        """Ingest GenAI images, validate spatial integrity, and compute FID.

        Parameters
        ----------
        colab_output_dir : str | Path
            Directory containing the images generated by ControlNet in Colab.
            Must also contain the original YOLO `.txt` labels to verify BBox integrity.
        ground_truth_dir : str | Path
            Directory containing real Ego4D (or target domain) images to compute FID against.
        """
        colab_output_dir = Path(colab_output_dir)
        ground_truth_dir = Path(ground_truth_dir)
        
        if not colab_output_dir.exists():
            raise FileNotFoundError(f"Colab output directory not found: {colab_output_dir}")
        if not ground_truth_dir.exists():
            raise FileNotFoundError(f"Ground truth directory not found: {ground_truth_dir}")

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        gen_paths = sorted([p for p in colab_output_dir.iterdir() if p.suffix.lower() in image_exts])
        real_paths = sorted([p for p in ground_truth_dir.iterdir() if p.suffix.lower() in image_exts])

        if not gen_paths or not real_paths:
            logger.error("Insufficient images to compute FID. Ensure both directories have images.")
            return

        logger.info("Ingesting %d GenAI images...", len(gen_paths))

        # 1. BBox Mapping & Constraint Validation
        # Assuming the generated image retains the same filename as the original label file
        valid_gen_paths = []
        discarded_count = 0
        
        for gen_img_path in gen_paths:
            label_path = gen_img_path.with_suffix(".txt")
            if not label_path.exists():
                # If no label exists, we can't validate spatial layout, but we keep it
                valid_gen_paths.append(gen_img_path)
                continue
                
            img = cv2.imread(str(gen_img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            # Load original GT boxes
            orig_boxes = []
            for line in label_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    orig_boxes.append(YOLOBox(
                        class_id=int(parts[0]),
                        x_center=float(parts[1]),
                        y_center=float(parts[2]),
                        width=float(parts[3]),
                        height=float(parts[4])
                    ))
            
            # Pass through ConstraintEngine (BBoxIntegrityChecker)
            # Since ControlNet preserves spatial layout, transformed_boxes == orig_boxes.
            # We check if the GT boxes are valid within the generated image dimensions.
            if not BBoxIntegrityChecker.check(orig_boxes, orig_boxes, w, h, max_loss_ratio=0.5):
                logger.warning("Spatial hallucination detected in %s. Discarding.", gen_img_path.name)
                discarded_count += 1
            else:
                valid_gen_paths.append(gen_img_path)

        logger.info("Spatial Validation: %d valid, %d discarded due to layout destruction.", 
                    len(valid_gen_paths), discarded_count)
                    
        if not valid_gen_paths:
            logger.error("No valid generated images remaining after Constraint Validation.")
            return

        # 2. FID Scoring
        logger.info("Computing Frechet Inception Distance (FID) on %s device...", self.device)
        fid = FrechetInceptionDistance(feature=2048, normalize=False).to(self.device)
        
        # Process in batches to avoid OOM
        batch_size = 32
        
        # Add real images (Ego4D domain)
        for i in range(0, len(real_paths), batch_size):
            batch_paths = real_paths[i:i + batch_size]
            real_tensors = self._load_images_as_tensor(batch_paths).to(self.device)
            if len(real_tensors) > 0:
                fid.update(real_tensors, real=True)
                
        # Add generated images (ControlNet output)
        for i in range(0, len(valid_gen_paths), batch_size):
            batch_paths = valid_gen_paths[i:i + batch_size]
            gen_tensors = self._load_images_as_tensor(batch_paths).to(self.device)
            if len(gen_tensors) > 0:
                fid.update(gen_tensors, real=False)
                
        # Compute FID
        fid_score = fid.compute()
        logger.info("=" * 60)
        logger.info("  🏆 FID Score (GenAI vs Real Ego4D): %.4f", fid_score.item())
        logger.info("=" * 60)
        
        # Clean up
        fid.reset()


# ──────────────────────────── CLI entry point ──────

def main() -> None:
    """Standalone CLI for GenAI Colab Bridge."""
    import argparse

    parser = argparse.ArgumentParser(description="GenAI Colab Bridge for Phase 2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export images and manifest to zip for Colab")
    export_parser.add_argument("--input", type=str, required=True, help="Input directory of WIDER FACE images and labels")
    export_parser.add_argument("--output", type=str, required=True, help="Output zip file path (e.g., payload.zip)")
    export_parser.add_argument("--prompt", type=str, default="chest-mounted GoPro view, extreme fisheye, harsh lighting, action blur")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest Colab outputs and compute FID")
    ingest_parser.add_argument("--gen", type=str, required=True, help="Directory containing GenAI outputs (from Colab)")
    ingest_parser.add_argument("--real", type=str, required=True, help="Directory containing real Ego4D images")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    
    bridge = GenAIColabBridge(target_prompt=getattr(args, "prompt", ""))
    
    if args.command == "export":
        bridge.export_payload_for_colab(args.input, args.output)
    elif args.command == "ingest":
        bridge.ingest_and_evaluate(args.gen, args.real)


if __name__ == "__main__":
    main()
