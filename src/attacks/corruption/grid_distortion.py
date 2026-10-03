"""Group A: Geometric Grid Distortion."""

from __future__ import annotations

from typing import ClassVar, cast

import cv2
import numpy as np

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, Sample


class GridDistortionParams(AttackParams):
    num_steps: int = 5


@ATTACKS.register
class GridDistortion(BaseAttack):
    """Simulate non-linear 3D projection distortion (edge-weighted)."""

    name: ClassVar[str] = "grid_distortion"
    group: ClassVar[AttackGroup] = "A"
    category: ClassVar[str | None] = "geometric"
    params_model: ClassVar[type[AttackParams]] = GridDistortionParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        img = sample.image
        h, w = img.shape[:2]
        
        distort_limit = severity * 0.03
        
        # Generate meshgrid
        params = cast(GridDistortionParams, self.params)
        steps = params.num_steps
        x_steps = np.linspace(0, w, steps + 1)
        y_steps = np.linspace(0, h, steps + 1)
        
        # Distort the grid points
        x_dist = ctx.rng.uniform(-distort_limit, distort_limit, size=(steps + 1, steps + 1)) * w
        y_dist = ctx.rng.uniform(-distort_limit, distort_limit, size=(steps + 1, steps + 1)) * h
        
        # Edge-weighting: more distortion on the edges, less in center
        x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, steps + 1), np.linspace(-1, 1, steps + 1))
        radial_weight = np.sqrt(x_grid**2 + y_grid**2)
        
        x_dist *= radial_weight
        y_dist *= radial_weight
        
        # Full map arrays
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)
        
        for i in range(steps):
            for j in range(steps):
                y0, y1 = int(y_steps[i]), int(y_steps[i + 1])
                x0, x1 = int(x_steps[j]), int(x_steps[j + 1])
                
                # Bilinear interpolation for the block
                y_coords, x_coords = np.mgrid[y0:y1, x0:x1]
                
                x_alpha = (x_coords - x0) / (x1 - x0)
                y_alpha = (y_coords - y0) / (y1 - y0)
                
                dx = (x_dist[i, j] * (1 - x_alpha) * (1 - y_alpha) +
                      x_dist[i, j + 1] * x_alpha * (1 - y_alpha) +
                      x_dist[i + 1, j] * (1 - x_alpha) * y_alpha +
                      x_dist[i + 1, j + 1] * x_alpha * y_alpha)
                      
                dy = (y_dist[i, j] * (1 - x_alpha) * (1 - y_alpha) +
                      y_dist[i, j + 1] * x_alpha * (1 - y_alpha) +
                      y_dist[i + 1, j] * (1 - x_alpha) * y_alpha +
                      y_dist[i + 1, j + 1] * x_alpha * y_alpha)
                      
                map_x[y0:y1, x0:x1] = x_coords + dx
                map_y[y0:y1, x0:x1] = y_coords + dy
                
        img_out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        img_out = np.clip(img_out, 0.0, 1.0).astype(np.float32)
        
        return sample.with_image(img_out)
