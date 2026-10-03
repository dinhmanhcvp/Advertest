#!/usr/bin/env python3
"""AdverTest MVP Demo — End-to-End Pipeline.

Reads clean images + YOLO labels from  data/raw/,
routes them through the InsightRouter → AttackEngine,
and saves side-by-side comparison panels to  data/demo_output/.

Usage::

    python run_demo.py                       # uses data/raw/ by default
    python run_demo.py --input path/to/imgs  # custom input
    python run_demo.py --severity 5          # force max severity
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Resolve project root so imports work regardless of cwd ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from advertest.attacks.engine import YOLOBox, AttackResult
from advertest.core.insight_router import InsightRouter


# ──────────────────────────────────────────── I/O helpers ──────

def load_yolo_labels(label_path: Path) -> list[YOLOBox]:
    """Parse a YOLO label txt file into YOLOBox objects."""
    boxes: list[YOLOBox] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append(YOLOBox(class_id=cls_id, x_center=xc, y_center=yc, width=w, height=h))
    return boxes


def save_yolo_labels(boxes: list[YOLOBox], label_path: Path) -> None:
    """Write YOLO boxes back to a txt file."""
    with open(label_path, "w") as f:
        for box in boxes:
            f.write(box.to_label_str() + "\n")


def find_image_label_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """Discover (image, label) pairs in a flat directory."""
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    pairs = []
    for img_path in sorted(input_dir.iterdir()):
        if img_path.suffix.lower() not in image_exts:
            continue
        label_path = img_path.with_suffix(".txt")
        pairs.append((img_path, label_path))
    return pairs


# ──────────────────────────────────── Visualisation ──────

TAG_COLORS = {
    "error_fisheye":      (255, 100, 50),   # orange
    "error_blur":         (100, 50, 255),    # purple
    "error_overexposure": (50, 200, 255),    # yellow
    "clean":              (100, 255, 100),   # green
}


def draw_boxes(img: np.ndarray, boxes: list[YOLOBox],
               color: tuple[int, int, int] = (0, 255, 0),
               thickness: int = 2) -> np.ndarray:
    """Draw YOLO boxes on an image (BGR uint8)."""
    canvas = img.copy()
    h, w = canvas.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box.to_xyxy(w, h)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f"cls={box.class_id}"
        cv2.putText(canvas, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def create_comparison_panel(
    original: np.ndarray,
    attacked: np.ndarray,
    orig_boxes: list[YOLOBox],
    atk_boxes: list[YOLOBox],
    tag: str,
    severity: int,
) -> np.ndarray:
    """Build a side-by-side panel with text overlay."""
    h, w = original.shape[:2]

    # Resize to uniform height if aspect ratios differ
    target_h = 480
    scale = target_h / h
    new_w = int(w * scale)
    orig_resized = cv2.resize(original, (new_w, target_h))
    atk_resized = cv2.resize(attacked, (new_w, target_h))

    # Draw boxes
    color = TAG_COLORS.get(tag, (0, 255, 0))

    # Scale boxes manually for resized images
    def scale_boxes(boxes: list[YOLOBox]) -> list[YOLOBox]:
        return boxes  # YOLO coords are normalised, so they scale automatically

    left = draw_boxes(orig_resized, scale_boxes(orig_boxes), color=(0, 255, 0))
    right = draw_boxes(atk_resized, scale_boxes(atk_boxes), color=color)

    # Add text headers
    header_h = 40
    panel_w = new_w * 2 + 20  # 20px gap
    panel = np.zeros((target_h + header_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # dark background

    # Left header
    cv2.putText(panel, "ORIGINAL", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
    # Right header
    tag_display = tag.replace("error_", "").upper()
    header_text = f"ADVERTEST [{tag_display}] sev={severity}"
    cv2.putText(panel, header_text, (new_w + 30, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Paste images
    panel[header_h:header_h + target_h, 0:new_w] = left
    panel[header_h:header_h + target_h, new_w + 20:new_w * 2 + 20] = right

    return panel


# ────────────────────────────────────────────────── Demo with synthetic data ──────

def generate_synthetic_demo_data(output_dir: Path, n_images: int = 5) -> None:
    """Create synthetic test images + YOLO labels for the demo if data/raw/ is empty."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)

    for i in range(n_images):
        w, h = 640, 480
        # Generate a colourful test image with some structure
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Gradient background
        for y in range(h):
            ratio = y / h
            img[y, :, 0] = int(40 + 80 * ratio)   # B
            img[y, :, 1] = int(60 + 100 * ratio)  # G
            img[y, :, 2] = int(120 + 100 * (1 - ratio))  # R

        # Draw some "face-like" shapes
        n_faces = int(rng.integers(1, 4))
        boxes = []
        for j in range(n_faces):
            cx = int(rng.uniform(100, w - 100))
            cy = int(rng.uniform(80, h - 80))
            fw = int(rng.uniform(40, 100))
            fh = int(rng.uniform(50, 120))

            # Draw ellipse (face) + circle (head contour)
            face_color = (
                int(rng.uniform(180, 240)),
                int(rng.uniform(150, 210)),
                int(rng.uniform(130, 190)),
            )
            cv2.ellipse(img, (cx, cy), (fw // 2, fh // 2), 0, 0, 360, face_color, -1)
            cv2.circle(img, (cx, cy - fh // 4), fw // 5, (40, 40, 40), -1)   # "eyes"
            cv2.circle(img, (cx - fw // 6, cy - fh // 4), fw // 10, (255, 255, 255), -1)
            cv2.circle(img, (cx + fw // 6, cy - fh // 4), fw // 10, (255, 255, 255), -1)

            x1, y1 = max(0, cx - fw // 2 - 10), max(0, cy - fh // 2 - 10)
            x2, y2 = min(w, cx + fw // 2 + 10), min(h, cy + fh // 2 + 10)
            boxes.append(YOLOBox.from_xyxy(x1, y1, x2, y2, w, h, class_id=0))

        img_path = output_dir / f"sample_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)

        label_path = output_dir / f"sample_{i:03d}.txt"
        save_yolo_labels(boxes, label_path)

    print(f"  ✓ Generated {n_images} synthetic demo images in {output_dir}")


# ──────────────────────────────────────────────────── Main ──────

def main() -> None:
    parser = argparse.ArgumentParser(description="AdverTest MVP Demo Pipeline")
    parser.add_argument("--input", type=str, default="data/raw",
                        help="Directory containing images and YOLO label .txt files")
    parser.add_argument("--output", type=str, default="data/demo_output",
                        help="Directory to save generated adversarial data")
    parser.add_argument("--severity", type=int, default=None,
                        help="Force a specific severity (1-5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("  🚀  AdverTest MVP Demo — Phase 1 Local Pipeline")
    print("=" * 60)

    # If input dir is empty, generate synthetic data
    pairs = find_image_label_pairs(input_dir)
    if not pairs:
        print(f"\n  ⚠ No images found in {input_dir}. Generating synthetic demo data...")
        generate_synthetic_demo_data(input_dir)
        pairs = find_image_label_pairs(input_dir)

    print(f"\n  📂 Input:  {input_dir}  ({len(pairs)} image(s))")
    print(f"  📂 Output: {output_dir}")

    # Initialise router
    router = InsightRouter(seed=args.seed)
    print(f"\n{router.summary()}\n")

    stats = {"total": 0, "generated": 0, "discarded": 0, "clean": 0}

    for img_path, label_path in pairs:
        stats["total"] += 1
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ✗ Failed to read {img_path.name}")
            continue

        boxes = load_yolo_labels(label_path)

        # Route through the insight-driven engine
        routed = router.route(image, boxes, force_severity=args.severity)
        result = routed.attack_result

        tag = routed.selected_tag
        sev = routed.severity

        if result.discarded:
            stats["discarded"] += 1
            print(f"  ✗ {img_path.name:<20s}  [{tag}] sev={sev}  → DISCARDED (BBox integrity)")
            continue

        if not routed.was_routed:
            stats["clean"] += 1
            print(f"  ○ {img_path.name:<20s}  [clean]  → passthrough (control group)")
            # Still save it
            out_img_path = output_dir / f"{img_path.stem}_clean{img_path.suffix}"
            cv2.imwrite(str(out_img_path), image)
            continue

        stats["generated"] += 1

        # Save attacked image
        out_img_path = output_dir / f"{img_path.stem}_{tag}_s{sev}{img_path.suffix}"
        cv2.imwrite(str(out_img_path), result.image)

        # Save transformed labels
        out_label_path = output_dir / f"{img_path.stem}_{tag}_s{sev}.txt"
        save_yolo_labels(result.boxes, out_label_path)

        # Save side-by-side comparison
        panel = create_comparison_panel(image, result.image, boxes, result.boxes, tag, sev)
        panel_path = output_dir / f"{img_path.stem}_{tag}_s{sev}_compare.jpg"
        cv2.imwrite(str(panel_path), panel)

        tag_short = tag.replace("error_", "").upper()
        print(f"  ✓ {img_path.name:<20s}  [{tag_short:>13s}] sev={sev}  → saved")

    print()
    print("─" * 60)
    print(f"  📊 Results:  {stats['generated']} generated  |  "
          f"{stats['discarded']} discarded  |  "
          f"{stats['clean']} clean  |  "
          f"{stats['total']} total")
    print(f"  📁 Output saved to: {output_dir.resolve()}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
