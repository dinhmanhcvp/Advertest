"""Label Studio Insight Router & Probabilistic Attack Dispatcher.

In production, Label Studio sends back an error-distribution JSON via its
REST API after human annotators classify each failure case.  When Label
Studio is not connected, a built-in default distribution is used as a
fallback to steer each incoming image to the correct attack pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from advertest.attacks.engine import AttackEngine, AttackResult, YOLOBox


# ──────────────────────────────── Default Error Distribution ──────

def get_default_insight_distribution() -> dict[str, float]:
    """Default error-distribution when Label Studio is not connected.

    In a real deployment this is fetched from the Label Studio REST API::

        GET /api/projects/{id}/exports/{export_id}

    The returned dict maps each error-tag to its prevalence in the validation
    failures. Values must sum to 1.0.
    """
    return {
        "error_fisheye":       0.35,  # 35 % lens distortion
        "error_blur":          0.20,  # 20 % kinetic motion blur
        "error_biomechanical": 0.15,  # 15 % IMU/biomechanical blur
        "error_overexposure":  0.15,  # 15 % sensor / HDR clipping
        "clean":               0.15,  # 15 % control group
    }


# ──────────────────────────────────────────────── Insight Router ──────

@dataclass
class RoutedResult:
    """Wraps an AttackResult with routing metadata."""
    attack_result: AttackResult
    selected_tag: str
    severity: int
    was_routed: bool     # False for "clean" passthrough


class InsightRouter:
    """Probabilistic router: steers images to attack pipelines based on
    the error distribution from Label Studio insights.

    Usage::

        router = InsightRouter()
        result = router.route(image, boxes)
        if not result.attack_result.discarded:
            cv2.imwrite("out.jpg", result.attack_result.image)
    """

    def __init__(
        self,
        insights: dict[str, float] | None = None,
        severity_range: tuple[int, int] = (2, 5),
        bbox_loss_threshold: float = 0.50,
        seed: int = 42,
    ):
        self.insights = insights or get_default_insight_distribution()
        self.severity_range = severity_range
        self.rng = np.random.default_rng(seed)
        self.engine = AttackEngine(bbox_loss_threshold=bbox_loss_threshold, seed=seed)

        # Pre-compute cumulative probabilities for fast sampling
        self._tags = list(self.insights.keys())
        self._probs = np.array([self.insights[t] for t in self._tags])
        # Normalise just in case
        self._probs = self._probs / self._probs.sum()

    def route(
        self,
        image: np.ndarray,
        boxes: list[YOLOBox],
        force_tag: str | None = None,
        force_severity: int | None = None,
    ) -> RoutedResult:
        """Select an attack tag via weighted random sampling and apply it.

        Parameters
        ----------
        image : ndarray
            Input BGR uint8 image.
        boxes : list[YOLOBox]
            YOLO-format bounding boxes for this image.
        force_tag : str, optional
            Override the probabilistic selection (useful for debugging).
        force_severity : int, optional
            Override the random severity.

        Returns
        -------
        RoutedResult
            Contains the attacked image, transformed boxes, and metadata.
        """
        # 1. Select tag
        if force_tag:
            tag = force_tag
        else:
            tag = self.rng.choice(self._tags, p=self._probs)

        # 2. Select severity
        if force_severity:
            severity = force_severity
        else:
            severity = int(self.rng.integers(self.severity_range[0],
                                              self.severity_range[1] + 1))

        # 3. "clean" passthrough
        if tag == "clean":
            from advertest.attacks.engine import AttackResult
            return RoutedResult(
                attack_result=AttackResult(
                    image=image, boxes=boxes, tag="clean",
                    severity=0, discarded=False,
                ),
                selected_tag="clean",
                severity=0,
                was_routed=False,
            )

        # 4. Route to engine
        result = self.engine.apply(image, boxes, tag=tag, severity=severity)
        return RoutedResult(
            attack_result=result,
            selected_tag=tag,
            severity=severity,
            was_routed=True,
        )

    def summary(self) -> str:
        """Human-readable summary of the insight distribution."""
        lines = ["┌─── Label Studio Insight Distribution ───┐"]
        for tag, prob in self.insights.items():
            bar = "█" * int(prob * 40)
            lines.append(f"│ {tag:<25s} {prob:5.1%}  {bar}")
        lines.append("└─────────────────────────────────────────┘")
        return "\n".join(lines)
