"""Stage 2: Attack Pool Generation.

Factory for generating targeted adversarial data pools based on the 
error categories identified in Stage 1 and classified via Label Studio.
"""

from __future__ import annotations

import logging
from typing import Any

from src.attacks import get_attack
from src.attacks.base import AttackContext
from src.core.egocentric_types import AttackPool, AttackPoolEntry, ErrorCategory, ErrorCase
from src.core.integrity import BBoxIntegrityChecker, TemporalBoundaryWarper
from src.core.types import Sample
from src.core.egocentric_types import VideoSample

logger = logging.getLogger(__name__)


class AttackPoolFactory:
    """Factory for routing error categories to targeted adversarial attacks."""

    def __init__(self, rng_seed: int = 42) -> None:
        self.rng = __import__("numpy").random.default_rng(rng_seed)

    def generate_spatial_pool(
        self, error_category: ErrorCategory, source_samples: list[Sample]
    ) -> AttackPool:
        """Generate targeted spatial attacks for PII detection errors."""
        pool_id = f"pool-{error_category}-{self.rng.integers(1000, 9999)}"
        entries: list[AttackPoolEntry] = []
        
        # Map error category to relevant attack plugins
        attack_names = self._route_spatial_attacks(error_category)
        
        for sample in source_samples:
            for attack_name in attack_names:
                attack = get_attack(attack_name)
                
                # Apply across severity levels 1 to 5
                for severity in range(1, 6):
                    ctx = AttackContext(rng=self.rng)
                    attacked_sample = attack.apply(sample, severity, ctx)
                    
                    # Ensure BBox Integrity
                    checked_sample = BBoxIntegrityChecker.apply_to_sample(attacked_sample)
                    if checked_sample is None:
                        logger.warning(f"Discarding {sample.sample_id} - {attack_name} - completely out of bounds.")
                        continue
                    
                    entry = AttackPoolEntry(
                        entry_id=f"{pool_id}-{sample.sample_id}-{attack_name}-L{severity}",
                        source_sample_id=sample.sample_id,
                        attack_name=attack_name,
                        severity=severity,
                        image=checked_sample.image,
                        error_category=error_category,
                    )
                    entries.append(entry)
                    
        return AttackPool(
            pool_id=pool_id,
            error_category=error_category,
            attack_name="mixed_spatial",
            severity_range=(1, 5),
            entries=tuple(entries),
        )

    def generate_temporal_pool(
        self, error_category: ErrorCategory, source_videos: list[VideoSample]
    ) -> AttackPool:
        """Generate targeted temporal attacks for video semantic errors."""
        pool_id = f"pool-{error_category}-{self.rng.integers(1000, 9999)}"
        entries: list[AttackPoolEntry] = []
        
        attack_names = self._route_temporal_attacks(error_category)
        
        for video in source_videos:
            for attack_name in attack_names:
                attack = get_attack(attack_name)
                
                # For video samples, we convert them temporarily to Sample
                # objects so we can reuse the standard BaseAttack contract.
                base_sample = Sample(
                    sample_id=video.sample_id,
                    image=video.get_frame(0),
                    meta={"video_frames": video.frames}
                )
                
                for severity in range(1, 6):
                    ctx = AttackContext(rng=self.rng)
                    attacked_sample = attack.apply(base_sample, severity, ctx)
                    
                    # Handle Temporal Boundaries if frames were dropped
                    dropped_indices = attacked_sample.meta.get("dropped_indices")
                    new_boundaries = video.temporal_boundaries
                    if dropped_indices:
                        new_boundaries = TemporalBoundaryWarper.warp_boundaries(
                            video.temporal_boundaries, dropped_indices
                        )
                    
                    entry = AttackPoolEntry(
                        entry_id=f"{pool_id}-{video.sample_id}-{attack_name}-L{severity}",
                        source_sample_id=video.sample_id,
                        attack_name=attack_name,
                        severity=severity,
                        frames=attacked_sample.meta.get("video_frames"),
                        error_category=error_category,
                        metadata={"temporal_boundaries": new_boundaries},
                    )
                    
                    entries.append(entry)

        return AttackPool(
            pool_id=pool_id,
            error_category=error_category,
            attack_name="mixed_temporal",
            severity_range=(1, 5),
            entries=tuple(entries),
        )

    def generate_eot_pool(
        self, sample: Sample, attack_names: list[str], n_transforms: int = 50
    ) -> AttackPool:
        """Expectation Over Transformation (EoT).
        
        Generates N random variants of the same sample using a combination
        of attacks for robustness training.
        """
        pool_id = f"pool-eot-{sample.sample_id}"
        entries: list[AttackPoolEntry] = []
        
        for i in range(n_transforms):
            attack_name = self.rng.choice(attack_names)
            severity = int(self.rng.integers(1, 6))
            attack = get_attack(str(attack_name))
            
            ctx = AttackContext(rng=self.rng)
            attacked = attack.apply(sample, severity, ctx)
            
            entry = AttackPoolEntry(
                entry_id=f"{pool_id}-T{i}",
                source_sample_id=sample.sample_id,
                attack_name=str(attack_name),
                severity=severity,
                image=attacked.image,
            )
            entries.append(entry)
            
        return AttackPool(
            pool_id=pool_id,
            error_category="pii_blur",  # fallback
            attack_name="eot_mix",
            severity_range=(1, 5),
            entries=tuple(entries),
        )

    def _route_spatial_attacks(self, error_category: ErrorCategory) -> list[str]:
        """Map error categories to relevant spatial attack plugins."""
        routes = {
            "pii_fisheye": ["fisheye"],
            "pii_blur": ["motion_blur", "radial_blur"],
            "pii_occlusion": ["object_occlusion", "random_erasing"],
            "pii_overexposure": ["brightness", "contrast"],
            "pii_low_light": ["iso_noise", "shot_noise"],
        }
        return routes.get(error_category, ["gaussian_noise"])

    def _route_temporal_attacks(self, error_category: ErrorCategory) -> list[str]:
        """Map error categories to relevant temporal attack plugins."""
        routes = {
            "semantic_temporal_break": ["frame_dropping", "temporal_jitter"],
            "semantic_motion_blur": ["variable_framerate", "radial_blur"],
            "semantic_occlusion": ["camera_dropout", "frame_freeze"],
            "semantic_label_confusion": ["temporal_jitter"],
        }
        return routes.get(error_category, ["frame_dropping"])
