"""AdverTest Attack Engine — Constraint-Driven Physics Simulation.

This is the production-ready MVP engine that implements three core egocentric
corruptions (FisheyeDistortion, KineticBlur, SensorDegradation), gated by
an ExclusionMatrix (conflict validator) and a BBox Integrity Checker.

All corruptions operate on uint8 BGR images (OpenCV convention) and accept
YOLO-format labels  [class_id, x_center, y_center, width, height]  normalised
to [0, 1].  The engine converts internally as needed and returns the attacked
image + transformed YOLO labels (or None if the bbox integrity gate rejects
the result).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError:
    raise ImportError("albumentations is required: pip install albumentations")


# ─────────────────────────────────────────────────── data containers ──────

@dataclass
class YOLOBox:
    """One YOLO-format bounding box (normalised 0-1)."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """Convert to pixel (x1, y1, x2, y2)."""
        x1 = int((self.x_center - self.width / 2) * img_w)
        y1 = int((self.y_center - self.height / 2) * img_h)
        x2 = int((self.x_center + self.width / 2) * img_w)
        y2 = int((self.y_center + self.height / 2) * img_h)
        return x1, y1, x2, y2

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int,
                  img_w: int, img_h: int, class_id: int = 0) -> "YOLOBox":
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return cls(class_id=class_id, x_center=xc, y_center=yc, width=w, height=h)

    def to_label_str(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


@dataclass
class AttackResult:
    """Container returned by the engine for one image."""
    image: np.ndarray                        # uint8 BGR
    boxes: list[YOLOBox]                     # transformed YOLO labels
    tag: str                                 # e.g. "error_fisheye"
    severity: int
    discarded: bool = False                  # True if bbox gate rejected it


# ──────────────────────────────────────────── Exclusion Matrix (Safety) ──────

class ExclusionMatrix:
    """Physics conflict gate — prevents impossible augmentation combos.

    If `SunFlare` is active, `LowLight` or `HeavyRain` cannot co-occur.
    """

    CONFLICTS: set[frozenset[str]] = {
        frozenset({"error_overexposure", "error_lowlight"}),
        frozenset({"error_overexposure", "error_heavy_rain"}),
        frozenset({"error_flare",        "error_lowlight"}),
        frozenset({"error_flare",        "error_heavy_rain"}),
    }

    @classmethod
    def validate(cls, tags: Sequence[str]) -> bool:
        """Return True if the combination is physically valid."""
        tag_set = set(tags)
        for pair in cls.CONFLICTS:
            if pair.issubset(tag_set):
                return False
        return True


# ──────────────────────────────────────────── BBox Integrity Checker ──────

class BBoxIntegrityChecker:
    """Discard a transformation if any bbox loses > threshold of its area."""

    @staticmethod
    def check(
        original_boxes: list[YOLOBox],
        transformed_boxes: list[YOLOBox],
        img_w: int,
        img_h: int,
        max_loss_ratio: float = 0.50,
    ) -> bool:
        """Return True if transformation is acceptable, False to DISCARD."""
        if not original_boxes:
            return True

        for orig, trans in zip(original_boxes, transformed_boxes):
            ox1, oy1, ox2, oy2 = orig.to_xyxy(img_w, img_h)
            orig_area = max(1, (ox2 - ox1) * (oy2 - oy1))

            tx1, ty1, tx2, ty2 = trans.to_xyxy(img_w, img_h)
            # Clip to image bounds
            cx1 = max(0, min(img_w, tx1))
            cy1 = max(0, min(img_h, ty1))
            cx2 = max(0, min(img_w, tx2))
            cy2 = max(0, min(img_h, ty2))
            clipped_area = max(0, (cx2 - cx1) * (cy2 - cy1))

            loss = 1.0 - (clipped_area / orig_area)
            if loss > max_loss_ratio:
                return False
        return True


# ──────────────────────────────── Individual Attack Implementations ──────

class FisheyeDistortion:
    """Barrel / pincushion distortion simulating wide-angle egocentric lens.

    Uses a radial remapping (not albumentations GridDistortion) for true
    barrel-distortion physics:  r' = r * (1 + k1*r² + k2*r⁴).
    Edges are distorted far more than the center — matching real GoPro lenses.
    """

    SEVERITY_K1 = (0.15, 0.30, 0.50, 0.75, 1.00)
    SEVERITY_K2 = (0.05, 0.10, 0.15, 0.25, 0.35)

    def __call__(
        self,
        image: np.ndarray,
        boxes: list[YOLOBox],
        severity: int = 3,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, list[YOLOBox]]:
        h, w = image.shape[:2]
        k1 = self.SEVERITY_K1[min(severity, 5) - 1]
        k2 = self.SEVERITY_K2[min(severity, 5) - 1]

        # Build radial remap
        cx, cy = w / 2, h / 2
        max_r = math.sqrt(cx**2 + cy**2)

        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = xs - cx
        dy = ys - cy
        r = np.sqrt(dx**2 + dy**2) / max_r      # normalised radius [0, 1]
        r_distorted = r * (1 + k1 * r**2 + k2 * r**4)

        # Scale factor per pixel
        scale = np.where(r > 1e-6, r_distorted / r, 1.0)
        map_x = (dx * scale + cx).astype(np.float32)
        map_y = (dy * scale + cy).astype(np.float32)

        distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

        # Transform boxes through the same radial warp
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.to_xyxy(w, h)
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            warped_corners = self._warp_points(corners, cx, cy, max_r, k1, k2)
            nx1 = int(warped_corners[:, 0].min())
            ny1 = int(warped_corners[:, 1].min())
            nx2 = int(warped_corners[:, 0].max())
            ny2 = int(warped_corners[:, 1].max())
            new_boxes.append(YOLOBox.from_xyxy(nx1, ny1, nx2, ny2, w, h, box.class_id))

        return distorted, new_boxes

    @staticmethod
    def _warp_points(pts: np.ndarray, cx: float, cy: float,
                     max_r: float, k1: float, k2: float) -> np.ndarray:
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        r = np.sqrt(dx**2 + dy**2) / max_r
        r_d = r * (1 + k1 * r**2 + k2 * r**4)
        scale = np.where(r > 1e-6, r_d / r, 1.0)
        out = np.column_stack([dx * scale + cx, dy * scale + cy])
        return out


class KineticBlur:
    """Chain of linear MotionBlur + custom radial (rotational) blur.

    Simulates the compound motion artefact from walking gait (linear sway)
    and torso rotation (angular twist) on a chest-mounted camera.
    """

    SEVERITY_KERNEL = (5, 9, 15, 21, 31)
    SEVERITY_ANGLE  = (1.5, 3.0, 5.0, 7.0, 10.0)

    def __call__(
        self,
        image: np.ndarray,
        boxes: list[YOLOBox],
        severity: int = 3,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, list[YOLOBox]]:
        k = self.SEVERITY_KERNEL[min(severity, 5) - 1]
        angle = self.SEVERITY_ANGLE[min(severity, 5) - 1]

        # Step 1 — linear motion blur via albumentations
        aug = A.MotionBlur(blur_limit=(k, k), p=1.0)
        blurred = aug(image=image)["image"]

        # Step 2 — rotational (radial) blur
        blurred = self._rotational_blur(blurred, angle, steps=max(3, severity + 2))

        # Blur doesn't shift geometry → boxes unchanged
        return blurred, list(boxes)

    @staticmethod
    def _rotational_blur(img: np.ndarray, max_angle: float, steps: int = 5) -> np.ndarray:
        if max_angle < 0.5:
            return img
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        acc = np.zeros_like(img, dtype=np.float32)
        angles = np.linspace(-max_angle, max_angle, steps)
        for a in angles:
            M = cv2.getRotationMatrix2D(center, a, 1.0)
            warped = cv2.warpAffine(img, M, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            acc += warped.astype(np.float32)
        return (acc / steps).astype(np.uint8)


class SensorDegradation:
    """Overexposure (tone-curve clipping) + ISO sensor noise + sun flare.

    Models the egocentric camera auto-exposure failure when transitioning
    from indoor to outdoor and the resulting HDR clipping artefact.
    """

    SEVERITY_GAMMA = (1.3, 1.6, 2.0, 2.5, 3.0)
    SEVERITY_NOISE = (5, 10, 20, 35, 50)

    def __call__(
        self,
        image: np.ndarray,
        boxes: list[YOLOBox],
        severity: int = 3,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, list[YOLOBox]]:
        if rng is None:
            rng = np.random.default_rng()

        img = image.copy()

        # 1. Tone-curve overexposure (gamma push + clip)
        gamma = self.SEVERITY_GAMMA[min(severity, 5) - 1]
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in range(256)]).astype(np.uint8)
        img = cv2.LUT(img, table)

        # 2. ISO noise (Gaussian approximation)
        sigma = self.SEVERITY_NOISE[min(severity, 5) - 1]
        noise = rng.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 3. Sun flare — soft radial gradient in upper quadrant
        h, w = img.shape[:2]
        fx = int(rng.uniform(w * 0.2, w * 0.8))
        fy = int(rng.uniform(0, h * 0.35))
        radius = int(min(w, h) * rng.uniform(0.15, 0.3) * (1 + severity * 0.15))

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (fx, fy), radius, 1.0, -1)
        ksize = (radius // 2) * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        # Add rays
        for _ in range(int(rng.uniform(3, 6))):
            angle = float(rng.uniform(0, 2 * math.pi))
            length = int(radius * rng.uniform(1.5, 3.0))
            thickness = int(rng.uniform(2, 8))
            x2 = int(fx + math.cos(angle) * length)
            y2 = int(fy + math.sin(angle) * length)
            cv2.line(mask, (fx, fy), (x2, y2), 0.4, thickness)

        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        mask = np.clip(mask, 0, 1)

        intensity = 0.3 + severity * 0.12
        flare = (mask[:, :, None] * np.array([220, 230, 255], dtype=np.float32) * intensity)
        img = np.clip(img.astype(np.float32) + flare, 0, 255).astype(np.uint8)

        # Sensor degradation doesn't move geometry → boxes unchanged
        return img, list(boxes)


class BiomechanicalBlur:
    """
    Simulates non-linear egocentric camera motion (e.g., walking head-bob or quick panning)
    by generating a mathematically modeled 2D blur kernel and convolving it with the image.
    """
    def __call__(
        self,
        image: np.ndarray,
        boxes: list[YOLOBox],
        severity: int = 3,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, list[YOLOBox]]:
        if rng is None:
            rng = np.random.default_rng()
            
        severity = max(1, min(5, severity))
        kernel_size = 5 + (severity * 10)
        k = kernel_size
        kernel = np.zeros((k, k), dtype=np.float32)
        center = k // 2

        amplitude = k * 0.4
        frequency = 2.0
        num_points = 200

        for i in range(num_points):
            t = i / float(num_points)
            x = center + (t - 0.5) * amplitude
            y = center + amplitude * 0.3 * math.sin(2 * math.pi * frequency * t)
            x, y = int(round(x)), int(round(y))
            if 0 <= x < k and 0 <= y < k:
                kernel[y, x] += 1.0

        kernel = cv2.GaussianBlur(kernel, (3, 3), 0)
        if np.sum(kernel) > 0:
            kernel /= np.sum(kernel)
        else:
            kernel[center, center] = 1.0

        blurred_image = cv2.filter2D(image, -1, kernel)
        
        return blurred_image, list(boxes)


# ─────────────────────────────────────────────────── Attack Engine ──────

class AttackEngine:
    """Constraint-driven, physics-based attack engine.

    Usage::

        engine = AttackEngine()
        result = engine.apply(image, boxes, tag="error_fisheye", severity=4)
        if result.discarded:
            print("BBox integrity violation — frame discarded.")
        else:
            cv2.imwrite("out.jpg", result.image)
    """

    TAG_MAP: dict[str, type] = {
        "error_fisheye":      FisheyeDistortion,
        "error_blur":         KineticBlur,
        "error_overexposure": SensorDegradation,
        "error_biomechanical": BiomechanicalBlur,
    }

    def __init__(self, bbox_loss_threshold: float = 0.50, seed: int = 42):
        self.bbox_loss_threshold = bbox_loss_threshold
        self.rng = np.random.default_rng(seed)
        # Eagerly instantiate the attacks
        self._attacks: dict[str, object] = {tag: cls() for tag, cls in self.TAG_MAP.items()}

    def apply(
        self,
        image: np.ndarray,
        boxes: list[YOLOBox],
        tag: str,
        severity: int = 3,
        extra_tags: Sequence[str] = (),
    ) -> AttackResult:
        """Run a single attack, gated by exclusion matrix + bbox integrity."""

        # ── 1. Exclusion Matrix gate ──
        all_tags = [tag] + list(extra_tags)
        if not ExclusionMatrix.validate(all_tags):
            return AttackResult(
                image=image, boxes=boxes, tag=tag,
                severity=severity, discarded=True,
            )

        # ── 2. Dispatch to the correct attack ──
        attack_fn = self._attacks.get(tag)
        if attack_fn is None:
            # "clean" or unknown tag → return original
            return AttackResult(image=image, boxes=boxes, tag=tag,
                                severity=severity, discarded=False)

        attacked_image, new_boxes = attack_fn(image, boxes, severity=severity, rng=self.rng)  # type: ignore[operator]

        # ── 3. BBox Integrity gate ──
        h, w = image.shape[:2]
        if not BBoxIntegrityChecker.check(boxes, new_boxes, w, h,
                                          max_loss_ratio=self.bbox_loss_threshold):
            return AttackResult(
                image=image, boxes=boxes, tag=tag,
                severity=severity, discarded=True,
            )

        return AttackResult(
            image=attacked_image, boxes=new_boxes, tag=tag,
            severity=severity, discarded=False,
        )
