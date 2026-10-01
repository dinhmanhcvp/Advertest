"""Egocentric vision domain types for PII Detection & Video Semantic Labeling.

Extends the core AdverTest type system with dedicated types for the
Egocentric Data Engine pipeline:

* **PII Detection** (spatial): annotated bounding boxes for sensitive data
  in first-person camera frames distorted by fisheye, motion blur, etc.
* **Video Semantic Labeling** (temporal): action segments with temporal
  boundaries for classifying sequences of human actions in egocentric video.
* **Error Analysis**: structured representation of hard negatives extracted
  by comparing model predictions against ground truth.
* **Attack Pools**: collections of adversarial samples ready for HITL review.

References:
    - Assembly101 (CVPR 2022): zero-discard policy for noisy egocentric data
    - Expectation Over Transformation (EoT): targeted adversarial robustness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

# ---------------------------------------------------------------------------
# PII Detection types
# ---------------------------------------------------------------------------

PIIClass = Literal[
    "face",
    "credit_card",
    "screen",
    "license_plate",
    "document",
    "handwriting",
    "qr_code",
]

ErrorCategory = Literal[
    "pii_blur",
    "pii_fisheye",
    "pii_occlusion",
    "pii_overexposure",
    "pii_low_light",
    "semantic_occlusion",
    "semantic_temporal_break",
    "semantic_motion_blur",
    "semantic_label_confusion",
    "semantic_confusion",
]


@dataclass(frozen=True, slots=True)
class PIIAnnotation:
    """Bounding box annotation for a PII region in a single frame.

    Coordinates follow the AdverTest convention: pixel space, ``x1 < x2``
    and ``y1 < y2``.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    pii_class: PIIClass
    confidence: float = 1.0
    occlusion_level: float = 0.0  # 0.0 = fully visible, 1.0 = fully occluded
    is_truncated: bool = False
    instance_id: str = ""

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def as_box_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


# ---------------------------------------------------------------------------
# Video Semantic Labeling types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemporalBoundary:
    """One action segment within a video clip."""

    start_frame: int
    end_frame: int
    action_label: str
    confidence: float = 1.0
    instance_id: str = ""

    @property
    def duration_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    def overlaps(self, other: TemporalBoundary) -> bool:
        return self.start_frame < other.end_frame and other.start_frame < self.end_frame

    def iou(self, other: TemporalBoundary) -> float:
        """Temporal IoU between two segments."""
        intersection_start = max(self.start_frame, other.start_frame)
        intersection_end = min(self.end_frame, other.end_frame)
        intersection = max(0, intersection_end - intersection_start)
        union = (
            self.duration_frames + other.duration_frames - intersection
        )
        if union <= 0:
            return 0.0
        return intersection / union


@dataclass(frozen=True, slots=True)
class TemporalAction:
    """Complete action annotation with metadata for semantic labeling."""

    boundary: TemporalBoundary
    verb: str = ""
    noun: str = ""
    narration: str = ""
    is_fine_grained: bool = False


@dataclass(frozen=True, slots=True, eq=False)
class VideoSample:
    """A video clip for temporal analysis.

    Unlike :class:`~src.core.types.Sample` (one image), this holds a sequence
    of frames with temporal boundary annotations. ``frames`` is a 4-D array
    of shape ``(T, H, W, 3)`` in ``float32 [0, 1]``.
    """

    sample_id: str
    frames: np.ndarray  # (T, H, W, 3) float32 [0, 1]
    fps: float = 30.0
    temporal_boundaries: tuple[TemporalBoundary, ...] = ()
    actions: tuple[TemporalAction, ...] = ()
    pii_annotations_per_frame: tuple[tuple[PIIAnnotation, ...], ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return self.frames.shape[0]

    @property
    def height(self) -> int:
        return self.frames.shape[1]

    @property
    def width(self) -> int:
        return self.frames.shape[2]

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.num_frames / self.fps

    def get_frame(self, index: int) -> np.ndarray:
        """Single frame as (H, W, 3) float32."""
        return self.frames[index]

    def with_frames(self, frames: np.ndarray) -> VideoSample:
        """Return a copy with replaced frames, preserving annotations."""
        from dataclasses import replace
        return replace(self, frames=frames)


# ---------------------------------------------------------------------------
# Error Analysis types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ErrorCase:
    """A hard negative extracted by comparing prediction vs ground truth.

    Used as input to Label Studio for classification and to the
    AttackPoolFactory for targeted augmentation.
    """

    case_id: str
    sample_id: str
    error_category: ErrorCategory
    iou: float = 0.0
    confidence: float = 0.0
    predicted_label: str = ""
    ground_truth_label: str = ""
    severity_estimate: int = 0  # auto-estimated difficulty 1-5
    frame_index: int | None = None  # for video errors
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_pii_error(self) -> bool:
        return self.error_category.startswith("pii_")

    @property
    def is_semantic_error(self) -> bool:
        return self.error_category.startswith("semantic_")


# ---------------------------------------------------------------------------
# Attack Pool types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class AttackPoolEntry:
    """One attacked sample in a pool, with provenance metadata."""

    entry_id: str
    source_sample_id: str
    attack_name: str
    severity: int
    image: np.ndarray | None = None  # (H, W, 3) for spatial
    frames: np.ndarray | None = None  # (T, H, W, 3) for temporal
    error_category: ErrorCategory | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AttackPool:
    """Collection of adversarial samples generated from a single error category.

    Ready for HITL review via Label Studio.
    """

    pool_id: str
    error_category: ErrorCategory
    attack_name: str
    severity_range: tuple[int, int]  # (min_severity, max_severity)
    entries: tuple[AttackPoolEntry, ...] = ()
    source_error_ids: tuple[str, ...] = ()
    status: Literal["pending", "in_review", "approved", "rejected"] = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.entries)

    def approved_entries(self) -> tuple[AttackPoolEntry, ...]:
        """Entries from an approved pool (all or none based on pool status)."""
        if self.status != "approved":
            return ()
        return self.entries
