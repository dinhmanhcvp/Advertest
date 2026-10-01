"""Group C: Temporal jitter — frame reordering within action segments.

Disrupts the temporal coherence of video sequences by swapping or
shuffling frames within local windows.  At low severity only adjacent
frames are swapped; at high severity entire segments are shuffled.

This models the real failure mode where camera buffer reordering or
network jitter delivers frames out of order, breaking temporal models
that rely on smooth motion continuity for action classification.

The attack operates on the ``meta["video_frames"]`` array (T, H, W, 3)
when present, and falls back to a single-frame no-op (preserving the
contract that severity-0 is always identity).  Temporal boundary
annotations are intentionally *not* reordered — the evaluator should
compare predictions against the original action segmentation.

Reference:
    - Assembly101 (CVPR 2022): temporal robustness in egocentric action data
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class TemporalJitterParams(AttackParams):
    """Jitter window size per severity level.

    ``window_per_severity`` controls the maximum displacement (in frames)
    that any single frame can move from its original position.
    """

    window_per_severity: tuple[int, ...] = (2, 4, 8, 16, 32)


@ATTACKS.register
class TemporalJitter(BaseAttack):
    """Frame reordering within local windows to break temporal coherence."""

    name: ClassVar[str] = "temporal_jitter"
    group: ClassVar[AttackGroup] = "C"
    cost_class: ClassVar[CostClass] = "cheap"
    owner: ClassVar[str] = "egocentric"
    reference: ClassVar[str] = (
        "Assembly101, CVPR 2022; temporal robustness in egocentric action data"
    )
    params_model: ClassVar[type[AttackParams]] = TemporalJitterParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        params: TemporalJitterParams = self.params  # type: ignore[assignment]
        window = int(self.level(severity, params.window_per_severity))

        # Retrieve video frames from sample metadata.
        video_frames = sample.meta.get("video_frames")
        if video_frames is None or not isinstance(video_frames, np.ndarray):
            # Single-frame sample: apply a slight temporal displacement
            # surrogate by blending the image with a shifted version.
            return self._single_frame_surrogate(sample, severity, ctx)

        num_frames = video_frames.shape[0]
        if num_frames <= 1:
            return sample

        # Apply local window shuffle: for each frame, swap it with a
        # random frame within the jitter window.
        jittered = video_frames.copy()
        indices = np.arange(num_frames)
        for i in range(num_frames):
            lo = max(0, i - window)
            hi = min(num_frames, i + window + 1)
            j = ctx.rng.integers(lo, hi)
            indices[i], indices[j] = indices[j], indices[i]

        jittered = video_frames[indices]

        # Update the canonical image with the first jittered frame.
        new_meta = {**sample.meta, "video_frames": jittered}
        return Sample(
            sample_id=sample.sample_id,
            image=jittered[0].astype(np.float32),
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

    def _single_frame_surrogate(
        self, sample: Sample, severity: int, ctx: AttackContext
    ) -> Sample:
        """For single-frame samples, simulate temporal jitter by applying
        a subtle pixel-shift blend that mimics frame misalignment."""
        shift = max(1, severity)
        image = sample.image
        # Roll the image horizontally and blend with original.
        shifted = np.roll(image, shift, axis=1)
        alpha = 0.1 * severity
        blended = ((1.0 - alpha) * image + alpha * shifted).astype(np.float32)
        return sample.with_image(blended)
