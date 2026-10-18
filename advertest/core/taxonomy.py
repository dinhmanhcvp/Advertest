"""Canonical failure-tag taxonomy for the central metadata registry.

Every component (triage, attack router, audit trail, API) emits tags from
``CANONICAL_TAGS`` only. Legacy engine-internal tags are mapped through
``normalize_tag`` so the Core Data Lake never sees fragmented labels.
"""

from __future__ import annotations

CANONICAL_TAGS: tuple[str, ...] = (
    "error_fisheye",
    "error_motion_blur",
    "error_sensor_noise",
    "error_occlusion",
)

TAG_COLORS: dict[str, str] = {
    "error_fisheye": "#8b5cf6",
    "error_motion_blur": "#f59e0b",
    "error_sensor_noise": "#22d3ee",
    "error_occlusion": "#ef4444",
}

LEGACY_ALIASES: dict[str, str] = {
    "error_blur": "error_motion_blur",
    "error_biomechanical": "error_motion_blur",
    "error_overexposure": "error_sensor_noise",
    "error_lowlight": "error_sensor_noise",
    "error_flare": "error_sensor_noise",
    "error_noise": "error_sensor_noise",
}

CLEAN_TAG = "clean"


def normalize_tag(tag: str) -> str:
    """Map any legacy tag to its canonical registry tag (``clean`` is preserved)."""
    if tag in CANONICAL_TAGS or tag == CLEAN_TAG:
        return tag
    return LEGACY_ALIASES.get(tag, tag)


def normalize_distribution(dist: dict[str, float]) -> dict[str, float]:
    """Collapse a legacy distribution onto canonical tags, dropping ``clean``."""
    out: dict[str, float] = {}
    for tag, value in dist.items():
        canon = normalize_tag(tag)
        if canon == CLEAN_TAG:
            continue
        out[canon] = out.get(canon, 0.0) + value
    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in out.items()}
