"""Group C: Variable Frame Rate (VFR) temporal distortion.

Simulates inconsistent frame rates typical of constrained egocentric
cameras (e.g. thermal throttling, varying processing load). It stretches
and compresses time by selectively duplicating or skipping frames, while
maintaining the overall sequence length.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class VariableFramerateParams(AttackParams):
    """Configuration for VFR attack.
    
    ``variance_per_severity`` controls the amplitude of the frame sampling distortion.
    """

    variance_per_severity: tuple[float, ...] = (0.5, 1.0, 2.0, 3.5, 5.0)


@ATTACKS.register
class VariableFramerate(BaseAttack):
    """Simulates inconsistent frame rates (VFR) in video sequences."""

    name: ClassVar[str] = "variable_framerate"
    group: ClassVar[AttackGroup] = "C"
    cost_class: ClassVar[CostClass] = "cheap"
    owner: ClassVar[str] = "egocentric"
    reference: ClassVar[str] = "Egocentric Vision: Variable Frame Rate (VFR) effects"
    params_model: ClassVar[type[AttackParams]] = VariableFramerateParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        params: VariableFramerateParams = self.params  # type: ignore[assignment]
        variance = self.level(severity, params.variance_per_severity)

        video_frames = sample.meta.get("video_frames")
        if video_frames is None or not isinstance(video_frames, np.ndarray):
            return sample

        num_frames = video_frames.shape[0]
        if num_frames <= 1:
            return sample

        # Generate a non-linear time mapping
        # We use a sine wave + random noise to create time stretching/compressing
        t = np.linspace(0, 1, num_frames)
        # Random low-frequency phase and amplitude
        phase = ctx.rng.random() * 2 * np.pi
        freq = 1.0 + ctx.rng.random() * 2.0
        
        # Calculate ideal sampling points
        distortion = np.sin(t * freq * 2 * np.pi + phase) * (variance / num_frames)
        mapped_t = t + distortion
        
        # Ensure monotonicity to prevent going backwards in time
        mapped_t = np.maximum.accumulate(mapped_t)
        
        # Normalize back to [0, 1] and map to frame indices
        mapped_t = (mapped_t - mapped_t.min()) / (mapped_t.max() - mapped_t.min() + 1e-6)
        indices = np.clip(np.round(mapped_t * (num_frames - 1)), 0, num_frames - 1).astype(int)

        vfr_frames = video_frames[indices]

        new_meta = {**sample.meta, "video_frames": vfr_frames}
        return Sample(
            sample_id=sample.sample_id,
            image=vfr_frames[0].astype(np.float32),
            boxes=sample.boxes,
            mask=sample.mask,
            depth=sample.depth,
            lidar=sample.lidar,
            camera_views=sample.camera_views,
            lidar_frame=sample.lidar_frame,
            boxes3d=sample.boxes3d,
            anonymized=sample.anonymized,
            meta=new_meta,
        )
