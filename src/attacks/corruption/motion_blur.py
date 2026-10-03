"""Group A: Motion blur (Egocentric Camera Shake)."""

from __future__ import annotations

import cv2
import numpy as np
import albumentations as A
from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, BaseAttack, AttackParams
from src.core.types import Sample, AttackGroup
from pydantic import Field

class MotionBlurParams(AttackParams):
    """Parameters for MultiAxisMotionBlur."""
    blur_limit_per_severity: tuple[int, ...] = Field(
        default=(3, 7, 11, 15, 21),
        description="Kernel size for motion blur. Must be odd.",
    )
    rotation_angle_per_severity: tuple[float, ...] = Field(
        default=(2.0, 4.0, 6.0, 8.0, 10.0),
        description="Max angle in degrees for rotational blur.",
    )

def apply_rotational_blur(image: np.ndarray, angle: float, steps: int = 5) -> np.ndarray:
    """Applies a rotational (radial) blur to simulate camera twist."""
    if angle <= 0.5:
        return image
        
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    blended = np.zeros_like(image, dtype=np.float32)
    angles = np.linspace(-angle, angle, steps)
    
    for a in angles:
        M = cv2.getRotationMatrix2D(center, a, 1.0)
        warped = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        blended += warped.astype(np.float32)
        
    blended /= steps
    return blended.astype(np.uint8)


@ATTACKS.register
class MultiAxisMotionBlur(BaseAttack):
    """Simulates dynamic motion blur common in egocentric walking/running.
    
    Combines linear motion blur (direction shake) and rotational blur (body twist)
    to accurately reproduce body-worn camera artifacts.
    """

    name: ClassVar[str] = "multi_axis_motion_blur"
    group: ClassVar[AttackGroup] = "A"
    category = "corruption"
    params_model = MotionBlurParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        params = self.resolve_parameters(severity)
        blur_limit = int(params["blur_limit"])
        angle = float(params["rotation_angle"])
        
        # Ensure blur_limit is odd and >= 3
        if blur_limit < 3:
            blur_limit = 3
        if blur_limit % 2 == 0:
            blur_limit += 1
            
        # Ensure image is in [0, 255] uint8 format for OpenCV/Albumentations
        img_uint8 = np.clip(sample.image * 255.0, 0, 255).astype(np.uint8)
        
        # Step 1: Linear Motion Blur
        transform = A.Compose([
            A.MotionBlur(blur_limit=(blur_limit, blur_limit), p=1.0)
        ])
        
        # Albumentations expects (H, W, C)
        linear_blurred = transform(image=img_uint8)["image"]
        
        # Step 2: Rotational Blur (Body Twist)
        final_blurred = apply_rotational_blur(linear_blurred, angle=angle, steps=severity + 2)
        
        # Convert back to [0, 1] float32
        img_float = final_blurred.astype(np.float32) / 255.0
        
        return sample.with_image(img_float)
