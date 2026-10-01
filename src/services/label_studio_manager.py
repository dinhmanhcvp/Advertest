"""Label Studio integration for HITL workflows.

Manages project creation, task import (for Error Analysis and Review),
and annotation synchronization using the Label Studio Python SDK.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

try:
    from label_studio_sdk.client import LabelStudio
except ImportError:
    LabelStudio = None  # type: ignore[assignment,misc]

from src.config import get_settings
from src.core.egocentric_types import AttackPool, ErrorCase

logger = logging.getLogger(__name__)

# Basic labeling configs for our egocentric tasks
PII_LABEL_CONFIG = """
<View>
  <Image name="image" value="$image" />
  <RectangleLabels name="pii" toName="image">
    <Label value="Face" background="#FFA39E" />
    <Label value="Credit Card" background="#D4380D" />
    <Label value="Screen" background="#FFC069" />
    <Label value="License Plate" background="#AD8B00" />
    <Label value="Document" background="#87E8DE" />
  </RectangleLabels>
  <Choices name="error_category" toName="image" choice="single-radio">
    <Choice value="pii_blur" />
    <Choice value="pii_fisheye" />
    <Choice value="pii_occlusion" />
    <Choice value="pii_overexposure" />
    <Choice value="pii_low_light" />
  </Choices>
</View>
"""

VIDEO_LABEL_CONFIG = """
<View>
  <Video name="video" value="$video" />
  <VideoRectangle name="box" toName="video" />
  <TimelineLabels name="action" toName="video">
    <Label value="Semantic Action" background="#2F54EB" />
  </TimelineLabels>
  <Choices name="error_category" toName="video" choice="single-radio">
    <Choice value="semantic_occlusion" />
    <Choice value="semantic_temporal_break" />
    <Choice value="semantic_motion_blur" />
    <Choice value="semantic_label_confusion" />
  </Choices>
</View>
"""

class LabelStudioManager:
    """Manages communication with Label Studio."""

    def __init__(self) -> None:
        self.settings = get_settings()
        if LabelStudio is None:
            self.client = None
            logger.warning("label-studio-sdk not installed. LabelStudioManager is in dummy mode.")
        elif not self.settings.label_studio_api_token:
            self.client = None
            logger.warning("LABEL_STUDIO_API_TOKEN not set. LabelStudioManager is in dummy mode.")
        else:
            self.client = LabelStudio(
                base_url=self.settings.label_studio_url,
                api_key=self.settings.label_studio_api_token,
            )

    def create_project(self, name: str, project_type: Literal["pii", "video"]) -> str:
        """Create a new project and return its ID."""
        if self.client is None:
            logger.info(f"[Dummy] Created Label Studio project '{name}' of type {project_type}")
            return "dummy-project-id"

        label_config = PII_LABEL_CONFIG if project_type == "pii" else VIDEO_LABEL_CONFIG
        
        project = self.client.projects.create(
            title=name,
            label_config=label_config,
            description=f"Auto-generated {project_type} project from AdverTest.",
        )
        return str(project.id)

    def push_error_tasks(self, project_id: str, errors: list[ErrorCase]) -> None:
        """Push a batch of ErrorCases to Label Studio for classification."""
        if self.client is None:
            logger.info(f"[Dummy] Pushed {len(errors)} error tasks to project {project_id}")
            return

        tasks = []
        for error in errors:
            # We assume the images/videos are served via a URL or local file path
            # known to Label Studio (e.g. through local storage sync).
            # We use placeholder URLs in this snippet.
            data = {}
            if error.is_pii_error:
                data["image"] = f"/data/local-files/?d={error.sample_id}.jpg"
            else:
                data["video"] = f"/data/local-files/?d={error.sample_id}.mp4"
            
            data["case_id"] = error.case_id
            data["predicted_label"] = error.predicted_label
            data["ground_truth"] = error.ground_truth_label
            
            tasks.append(data)
            
        self.client.tasks.create_many(project_id=int(project_id), request=tasks)

    def push_review_tasks(self, project_id: str, pool: AttackPool) -> None:
        """Push an Attack Pool to Label Studio for HITL review."""
        if self.client is None:
            logger.info(f"[Dummy] Pushed Attack Pool '{pool.pool_id}' ({pool.size} items) to project {project_id}")
            return

        tasks = []
        for entry in pool.entries:
            data = {
                "entry_id": entry.entry_id,
                "attack": entry.attack_name,
                "severity": entry.severity,
            }
            if pool.error_category.startswith("pii_"):
                data["image"] = f"/data/local-files/?d={entry.entry_id}.jpg"
            else:
                data["video"] = f"/data/local-files/?d={entry.entry_id}.mp4"
            tasks.append(data)
            
        self.client.tasks.create_many(project_id=int(project_id), request=tasks)

    def get_classified_errors(self, project_id: str) -> list[dict[str, Any]]:
        """Fetch completed annotations from the project."""
        if self.client is None:
            logger.info(f"[Dummy] Fetched classifications from project {project_id}")
            return []
            
        # Simplified: fetch tasks with annotations.
        # In a real setup, we'd parse the task JSON and extract the chosen error_category.
        tasks = self.client.tasks.list(project_id=int(project_id))
        results = []
        for task in tasks:
            if hasattr(task, 'annotations') and task.annotations:
                results.append(task.model_dump())
        return results

    def sync_annotations(self, project_id: str) -> None:
        """Force a sync of target storage if used."""
        if self.client is None:
            return
        # Implement storage sync logic here via SDK if needed
        pass
