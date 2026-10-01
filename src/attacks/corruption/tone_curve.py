"""Group B: Tone Curve (HDR Loss / Exposure)."""

from __future__ import annotations

from typing import ClassVar

import cv2
import numpy as np

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, Sample


class ToneCurveParams(AttackParams):
    """Parameters for Tone Curve corruption."""
    gamma_limit: tuple[float, float] = (0.5, 2.0)
    contrast_limit: tuple[float, float] = (0.5, 1.5)


@ATTACKS.register
class RandomToneCurve(BaseAttack):
    """Simulate Egocentric HDR Loss via S-Curve/Gamma distortion."""

    name: ClassVar[str] = "random_tone_curve"
    group: ClassVar[AttackGroup] = "B"
    category: ClassVar[str | None] = "weather"
    params_model: ClassVar[type[AttackParams]] = ToneCurveParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        img = sample.image
        # severity: 1 to 5
        
        # Scale severity to standard deviation of gamma/contrast changes
        gamma_dev = severity * 0.15
        contrast_dev = severity * 0.1
        
        gamma = ctx.rng.uniform(1.0 - gamma_dev, 1.0 + gamma_dev)
        contrast = ctx.rng.uniform(1.0 - contrast_dev, 1.0 + contrast_dev)
        
        # Convert float32 [0, 1] to uint8 [0, 255] for cv2 LUT
        img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        
        img_gamma = cv2.LUT(img_uint8, table)
        
        # Apply contrast
        img_float = img_gamma.astype(np.float32)
        mean = np.mean(img_float, axis=(0, 1), keepdims=True)
        img_contrast = (img_float - mean) * contrast + mean
        
        img_out = np.clip(img_contrast / 255.0, 0.0, 1.0).astype(np.float32)
        return sample.with_image(img_out)
