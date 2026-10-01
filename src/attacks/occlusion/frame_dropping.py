"""Group C: Frame dropping temporal occlusion attack.

Simulates missing frames due to sensor dropout, buffering issues, or
network lag. Replaces dropped frames with the previous valid frame (freeze)
or zero-fills them, depending on the fill mode.

Reference:
    - Egocentric Vision: robustness to sensor dropout and missing frames.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class FrameDroppingParams(AttackParams):
    """Configuration for frame dropping.
    
    ``drop_rate_per_severity`` controls the fraction of frames dropped.
    """

    drop_rate_per_severity: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    fill_mode: Literal["freeze", "zero"] = "freeze"


@ATTACKS.register
class FrameDropping(BaseAttack):
    """Simulates frame dropping or sensor dropout in video sequences."""

    name: ClassVar[str] = "frame_dropping"
    group: ClassVar[AttackGroup] = "C"
    cost_class: ClassVar[CostClass] = "cheap"
    owner: ClassVar[str] = "egocentric"
    reference: ClassVar[str] = "Egocentric Vision: robustness to sensor dropout"
    params_model: ClassVar[type[AttackParams]] = FrameDroppingParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        params: FrameDroppingParams = self.params  # type: ignore[assignment]
        drop_rate = self.level(severity, params.drop_rate_per_severity)

        video_frames = sample.meta.get("video_frames")
        if video_frames is None or not isinstance(video_frames, np.ndarray):
            return sample

        num_frames = video_frames.shape[0]
        if num_frames <= 1:
            return sample

        dropped = video_frames.copy()
        
        # Decide which frames to drop (excluding the first frame to ensure we have a fallback for freeze)
        drop_mask = ctx.rng.random(num_frames) < drop_rate
        drop_mask[0] = False  # Never drop the first frame

        dropped_indices = []
        for i in range(1, num_frames):
            if drop_mask[i]:
                dropped_indices.append(i)
                if params.fill_mode == "freeze":
                    dropped[i] = dropped[i - 1]
                else:
                    dropped[i] = np.zeros_like(dropped[i])

        new_meta = {**sample.meta, "video_frames": dropped, "dropped_indices": dropped_indices}
        return Sample(
            sample_id=sample.sample_id,
            image=dropped[0].astype(np.float32),
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
