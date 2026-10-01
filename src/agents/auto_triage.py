"""Auto-Triage Agent.

Uses a simulated VLM (Vision-Language Model) to pre-review adversarial samples
before they are sent to human annotators on Label Studio.
"""

import logging
from typing import ClassVar

import numpy as np

from src.core.egocentric_types import AttackPool, AttackPoolEntry

logger = logging.getLogger(__name__)


class AutoTriageAgent:
    """Pre-reviews Attack Pools to filter out completely corrupted samples."""

    # Threshold for borderline confidence. 
    # If the simulated VLM's confidence in recognizing the image is <= this,
    # it is considered completely corrupted and rejected.
    BORDERLINE_THRESH: ClassVar[float] = 0.4

    def __init__(self, rng_seed: int = 42):
        self.rng = np.random.default_rng(rng_seed)

    def _simulate_vlm_confidence(self, severity: int) -> float:
        """
        Simulate a VLM scoring the image's legibility.
        Severity 1: high confidence (~0.9)
        Severity 5: low confidence (~0.2)
        """
        # Base confidence drops as severity increases.
        # Severity 1 -> 0.9, Severity 2 -> 0.7, ..., Severity 5 -> 0.1
        base_confidence = max(0.1, 1.0 - (severity * 0.18))
        
        # Add some noise
        noise = self.rng.uniform(-0.15, 0.15)
        
        return float(np.clip(base_confidence + noise, 0.0, 1.0))

    def triage_pool(self, pool: AttackPool) -> tuple[AttackPool, list[AttackPoolEntry]]:
        """
        Filters an AttackPool.
        
        Returns:
            (approved_pool, rejected_entries)
        """
        approved_entries = []
        rejected_entries = []

        for entry in pool.entries:
            confidence = self._simulate_vlm_confidence(entry.severity)
            
            if confidence > self.BORDERLINE_THRESH:
                approved_entries.append(entry)
            else:
                rejected_entries.append(entry)

        logger.info(
            f"[AutoTriage] Pool {pool.pool_id} - Approved: {len(approved_entries)}, "
            f"Rejected: {len(rejected_entries)}"
        )

        approved_pool = AttackPool(
            pool_id=pool.pool_id,
            error_category=pool.error_category,
            attack_name=pool.attack_name,
            severity_range=pool.severity_range,
            entries=tuple(approved_entries),
        )

        return approved_pool, rejected_entries
