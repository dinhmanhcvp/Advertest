"""Group A: Barrel / pincushion (fish-eye) lens distortion using Kornia (GPU)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import Field

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class FisheyeParams(AttackParams):
    """Radial distortion coefficients per severity level."""

    k1_per_severity: tuple[float, ...] = (0.10, 0.25, 0.45, 0.65, 0.85)
    p1: float = Field(default=0.0, ge=-0.1, le=0.1)
    p2: float = Field(default=0.0, ge=-0.1, le=0.1)


@ATTACKS.register
class Fisheye(BaseAttack):
    """Barrel / pincushion lens distortion for egocentric robustness testing."""

    name: ClassVar[str] = "fisheye"
    group: ClassVar[AttackGroup] = "A"
    cost_class: ClassVar[CostClass] = "cheap"
    owner: ClassVar[str] = "egocentric"
    reference: ClassVar[str] = "GPU-Accelerated Fisheye via Kornia"
    params_model: ClassVar[type[AttackParams]] = FisheyeParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        try:
            import torch
            import kornia
        except ImportError:
            raise RuntimeError("Install 'torch' and 'kornia' for GPU-accelerated fisheye")

        params: FisheyeParams = self.params  # type: ignore[assignment]
        k1 = self.level(severity, params.k1_per_severity)

        # Convert image to tensor on device
        if isinstance(sample.image, np.ndarray):
            # Numpy shape is (H, W, C), Kornia expects (B, C, H, W)
            tensor = torch.from_numpy(sample.image).to(ctx.device)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # Add batch dim
            was_numpy = True
        else:
            tensor = sample.image.to(ctx.device)
            was_numpy = False
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)

        B, C, H, W = tensor.shape
        focal = max(H, W)
        cx, cy = W / 2.0, H / 2.0

        # Create camera matrix
        K = torch.tensor([
            [[focal, 0, cx],
             [0, focal, cy],
             [0, 0, 1]]
        ], dtype=tensor.dtype, device=tensor.device).repeat(B, 1, 1)

        # Distortion coeffs: [k1, k2, p1, p2] - Brown Conrady
        dist_coeff = torch.tensor([
            [k1, 0.0, params.p1, params.p2]
        ], dtype=tensor.dtype, device=tensor.device).repeat(B, 1)

        # Kornia undistort
        distorted = kornia.geometry.calibration.undistort_image(tensor, K, dist_coeff)

        # Back to original format if it was numpy
        if was_numpy:
            out_img = distorted.squeeze(0).permute(1, 2, 0).cpu().numpy()
            return sample.with_image(out_img)
        else:
            if sample.image.ndim == 3:
                distorted = distorted.squeeze(0)
            return sample.with_image(distorted)
