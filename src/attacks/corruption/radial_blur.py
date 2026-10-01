"""Group A: Radial motion blur from rapid head rotation using Kornia (GPU)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class RadialBlurParams(AttackParams):
    """Maximum blur radius (in pixels at image edge) per severity."""

    max_radius_per_severity: tuple[float, ...] = (4.0, 8.0, 14.0, 22.0, 32.0)
    num_samples: int = 12


@ATTACKS.register
class RadialBlur(BaseAttack):
    """Radial motion blur from centre outward, simulating fast head turns."""

    name: ClassVar[str] = "radial_blur"
    group: ClassVar[AttackGroup] = "A"
    cost_class: ClassVar[CostClass] = "cheap"
    owner: ClassVar[str] = "egocentric"
    reference: ClassVar[str] = "GPU-Accelerated Radial Blur via Kornia"
    params_model: ClassVar[type[AttackParams]] = RadialBlurParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        try:
            import torch
            import kornia
        except ImportError:
            raise RuntimeError("Install 'torch' and 'kornia' for GPU-accelerated radial blur")

        params: RadialBlurParams = self.params  # type: ignore[assignment]
        max_radius = self.level(severity, params.max_radius_per_severity)
        num_samples = max(2, params.num_samples)

        if isinstance(sample.image, np.ndarray):
            tensor = torch.from_numpy(sample.image).to(ctx.device)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            was_numpy = True
        else:
            tensor = sample.image.to(ctx.device)
            was_numpy = False
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)

        B, C, H, W = tensor.shape
        cx, cy = W / 2.0, H / 2.0
        max_dist = np.sqrt(cx * cx + cy * cy)

        if max_dist < 1e-6:
            return sample

        result = torch.zeros_like(tensor)

        # Apply multi-scale affine warps (zoom) via Kornia
        for i in range(num_samples):
            t = (i - num_samples / 2.0) / num_samples
            scale = 1.0 + t * (max_radius / max_dist)
            
            # Affine matrix
            m = torch.tensor([[
                [scale, 0, cx * (1 - scale)],
                [0, scale, cy * (1 - scale)]
            ]], dtype=tensor.dtype, device=tensor.device).repeat(B, 1, 1)

            warped = kornia.geometry.transform.warp_affine(
                tensor, m, dsize=(H, W), mode='bilinear', padding_mode='reflection', align_corners=True
            )
            result += warped

        result /= num_samples

        if was_numpy:
            out_img = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
            return sample.with_image(out_img)
        else:
            if sample.image.ndim == 3:
                result = result.squeeze(0)
            return sample.with_image(result)
