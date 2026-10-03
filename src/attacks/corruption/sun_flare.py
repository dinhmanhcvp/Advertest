"""Group B: Sun Flare."""

from __future__ import annotations

import math
from typing import ClassVar, cast

import cv2
import numpy as np

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, Sample


class SunFlareParams(AttackParams):
    flare_roi: tuple[float, float, float, float] = (0, 0, 1, 0.5)
    flare_radius: int = 400


@ATTACKS.register
class RandomSunFlare(BaseAttack):
    """Simulate Egocentric Sun Flare (Auto-Exposure Lag)."""

    name: ClassVar[str] = "random_sun_flare"
    group: ClassVar[AttackGroup] = "B"
    category: ClassVar[str | None] = "weather"
    params_model: ClassVar[type[AttackParams]] = SunFlareParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        img = sample.image.copy()
        h, w = img.shape[:2]
        
        # Determine flare center (upper half usually)
        x = int(ctx.rng.uniform(0, w))
        y = int(ctx.rng.uniform(0, int(h * 0.5)))
        
        params = cast(SunFlareParams, self.params)
        radius = int((params.flare_radius * w / 1920) * (1 + severity * 0.2))
        
        flare_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(flare_mask, (x, y), radius, 1.0, -1)
        
        # Apply intense gaussian blur to the mask to create a soft glow
        blur_ksize = (radius // 2) * 2 + 1
        flare_mask = cv2.GaussianBlur(flare_mask, (blur_ksize, blur_ksize), 0)
        
        # Generate some rays
        num_rays = int(ctx.rng.uniform(3, 7))
        for _ in range(num_rays):
            angle = ctx.rng.uniform(0, 2 * math.pi)
            ray_length = int(radius * ctx.rng.uniform(1.5, 3.0))
            ray_thickness = int(ctx.rng.uniform(2, 10))
            x2 = int(x + math.cos(angle) * ray_length)
            y2 = int(y + math.sin(angle) * ray_length)
            cv2.line(flare_mask, (x, y), (x2, y2), 0.5, ray_thickness)
            
        flare_mask = cv2.GaussianBlur(flare_mask, (blur_ksize // 2 * 2 + 1, blur_ksize // 2 * 2 + 1), 0)
        flare_mask = np.clip(flare_mask, 0, 1)
        flare_mask = np.expand_dims(flare_mask, axis=-1)
        
        # Flare color (whitish yellow)
        flare_color = np.array([0.9, 0.95, 1.0], dtype=np.float32)
        
        # Blend
        intensity = 0.3 + (severity * 0.1)
        blended = img + (flare_mask * flare_color * intensity)
        
        # Add global wash out (overexposure)
        wash_out = (severity * 0.05)
        blended = blended + wash_out
        
        img_out = np.clip(blended, 0.0, 1.0).astype(np.float32)
        return sample.with_image(img_out)
