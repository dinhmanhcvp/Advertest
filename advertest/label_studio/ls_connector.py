"""Real Label Studio API Connector.

NO MOCKS. This module uses the official label-studio-sdk (v1.0.0+) to
upload Hard Negatives to a real Label Studio project, along with their
YOLO predicted bounding boxes as pre-annotations. It also fetches
completed human annotations to compute the true error distribution.

Requirements:
    pip install label-studio-sdk python-dotenv
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

try:
    from label_studio_sdk.client import LabelStudio
except ImportError:
    raise ImportError("label-studio-sdk is required: pip install label-studio-sdk")

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("python-dotenv is required: pip install python-dotenv")

from advertest.core.inference_engine import HardNegative

logger = logging.getLogger(__name__)


class LSConnector:
    """Production connector for Label Studio.

    Uses ``LABEL_STUDIO_URL`` and ``LABEL_STUDIO_API_KEY`` from the environment
    or a ``.env`` file.
    """

    def __init__(self, env_file: str | Path = ".env") -> None:
        if Path(env_file).exists():
            load_dotenv(dotenv_path=env_file)

        url = os.environ.get("LABEL_STUDIO_URL")
        api_key = os.environ.get("LABEL_STUDIO_API_KEY")

        if not url or not api_key:
            raise ValueError(
                "Missing Label Studio credentials. Set LABEL_STUDIO_URL and "
                "LABEL_STUDIO_API_KEY in the environment or a .env file."
            )

        self.client = LabelStudio(
            base_url=url,
            api_key=api_key,
        )
        logger.info("Connected to Label Studio at %s", url)

    # ──────────────────────────────────── Upload Hard Negatives ──────

    def upload_hard_negatives(
        self,
        project_id: int,
        hard_negatives: List[HardNegative],
        image_base_url: str = "/data/local-files/?d=",
    ) -> None:
        """Upload Hard Negatives as new tasks with pre-annotations.

        Parameters
        ----------
        project_id : int
            The Label Studio project ID.
        hard_negatives : List[HardNegative]
            The list of hard negatives extracted by Yolov7Evaluator.
        image_base_url : str
            The base URL where Label Studio can fetch the images.
            If using Local Storage, it usually looks like `/data/local-files/?d=`.
        """
        if not hard_negatives:
            logger.info("No hard negatives to upload.")
            return

        tasks = []
        for hn in hard_negatives:
            # We assume Label Studio has Local Storage configured pointing to
            # the folder containing these images.
            image_url = f"{image_base_url}{hn.image_path.name}"
            
            # Format predictions as Label Studio pre-annotations
            # https://labelstud.io/guide/predictions.html
            results = []
            for box, score in zip(hn.pred_boxes, hn.pred_scores):
                # YOLO format is normalized (0-1). Label Studio expects percentages (0-100)
                x = (box.x_center - box.width / 2) * 100
                y = (box.y_center - box.height / 2) * 100
                w = box.width * 100
                h = box.height * 100

                result = {
                    "from_name": "pii",      # Name of the RectangleLabels tag in config
                    "to_name": "image",      # Name of the Image tag in config
                    "type": "rectanglelabels",
                    "value": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "rectanglelabels": ["Face"] # Map class_id to label string if needed
                    },
                    "score": score
                }
                results.append(result)
            
            task_data = {
                "data": {
                    "image": image_url,
                    "reason": hn.reason,
                    "max_iou": hn.max_iou,
                    "min_conf": hn.min_conf,
                },
                "predictions": [{
                    "model_version": "yolov7-face",
                    "result": results
                }]
            }
            tasks.append(task_data)

        # Upload in chunks (or all at once if supported by SDK version)
        try:
             # The new SDK might use create_many on tasks
             self.client.tasks.create_many(project_id=project_id, request=tasks) # type: ignore
        except AttributeError:
             # Fallback to importing tasks directly via API if SDK doesn't expose it
             import requests
             url = f"{self.client._base_url}/api/projects/{project_id}/import"
             headers = {"Authorization": f"Token {self.client._api_key}", "Content-Type": "application/json"}
             resp = requests.post(url, headers=headers, json=tasks)
             resp.raise_for_status()

        logger.info("Uploaded %d tasks with pre-annotations to project %d", len(tasks), project_id)

    # ──────────────────────────────────── Fetch Insights ──────

    def fetch_insights(self, project_id: int, choices_tag_name: str = "error_category") -> Dict[str, float]:
        """Fetch completed annotations and compute the error distribution.

        Queries the Label Studio API for completed tasks, parses the specific
        Choices tag where human reviewers selected the root cause (e.g. error_fisheye),
        and returns a normalized frequency distribution.

        Parameters
        ----------
        project_id : int
            The Label Studio project ID.
        choices_tag_name : str
            The 'name' attribute of the <Choices> tag in your Label Studio config.

        Returns
        -------
        Dict[str, float]
            A dictionary mapping error tags to their frequency (summing to 1.0).
            Example: {"error_fisheye": 0.45, "clean": 0.15}
        """
        logger.info("Fetching insights from project %d...", project_id)
        
        # Use the SDK to get all tasks (with annotations)
        tasks = []
        try:
            tasks_iter = self.client.tasks.list(project_id=project_id)
            for t in tasks_iter:
                # Assuming t is a pydantic model or similar in the new SDK
                if hasattr(t, 'model_dump'):
                    tasks.append(t.model_dump())
                elif hasattr(t, 'dict'):
                    tasks.append(t.dict())
                else:
                    tasks.append(t)
        except Exception as e:
            logger.error("Failed to fetch tasks via SDK. Attempting raw API call. Error: %s", e)
            import requests
            url = f"{self.client._base_url}/api/tasks?project={project_id}&with_annotations=true"
            headers = {"Authorization": f"Token {self.client._api_key}"}
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            tasks = resp.json()

        counts = defaultdict(int)
        total_annotated = 0

        for task in tasks:
            # Handle dictionary vs object access
            annotations = task.get("annotations", []) if isinstance(task, dict) else getattr(task, "annotations", [])
            
            if not annotations:
                continue

            # Take the most recent annotation (or could average across annotators)
            latest_anno = annotations[-1]
            anno_result = latest_anno.get("result", []) if isinstance(latest_anno, dict) else getattr(latest_anno, "result", [])

            for result in anno_result:
                # We are looking for the choices classification
                res_dict = result if isinstance(result, dict) else (result.model_dump() if hasattr(result, 'model_dump') else result.dict())
                if res_dict.get("from_name") == choices_tag_name and res_dict.get("type") == "choices":
                    # 'value' contains 'choices' which is a list of selected strings
                    selected_choices = res_dict.get("value", {}).get("choices", [])
                    if selected_choices:
                        # Assuming single choice
                        choice = selected_choices[0]
                        counts[choice] += 1
                        total_annotated += 1
        
        if total_annotated == 0:
            logger.warning("No completed annotations found in project %d. Returning uniform fallback.", project_id)
            # Return a default fallback if no data exists yet
            fallback = {"error_fisheye": 0.25, "error_blur": 0.25, "error_overexposure": 0.25, "clean": 0.25}
            return fallback

        # Normalize to probabilities
        distribution = {k: v / total_annotated for k, v in counts.items()}
        
        # Ensure 'clean' is always present for control group passthrough
        if "clean" not in distribution:
            distribution["clean"] = 0.0

        logger.info("Fetched distribution from %d annotations: %s", total_annotated, distribution)
        return distribution


# ──────────────────────────── CLI entry point ──────

def main() -> None:
    """Standalone CLI for testing Label Studio connection."""
    import argparse

    parser = argparse.ArgumentParser(description="Label Studio Connector Test")
    parser.add_argument("--project", type=int, required=True, help="Label Studio Project ID")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        connector = LSConnector()
        insights = connector.fetch_insights(project_id=args.project)
        print("\n=== Live Insights ===")
        for k, v in insights.items():
            print(f"{k}: {v:.2%}")
    except Exception as e:
        logger.error("Failed to connect or fetch insights: %s", e)


if __name__ == "__main__":
    main()
