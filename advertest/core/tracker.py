"""MLOps Experiment Tracking Integration.

Logs the robustness metrics (Base mAP vs Corrupted mAP) to Weights & Biases (wandb).
This ensures that every deployment gate decision is audited and versioned.

Requirements:
    pip install wandb
"""

from __future__ import annotations

import logging
from typing import Any, Dict

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore

logger = logging.getLogger(__name__)


def log_eval_metrics(run_name: str, metrics_dict: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Log the Blind Evaluation Gate metrics to Weights & Biases.

    Parameters
    ----------
    run_name : str
        The name of the tracking run (e.g., "yolov7-advertest-v2").
    metrics_dict : Dict[str, Any]
        The output dictionary from `RobustnessEvaluator.calculate_map_drop()`.
        Must contain "metrics" and "deltas".
    config : Dict[str, Any]
        Configuration parameters (e.g., dataset version, attack engine severity).
    """
    if wandb is None:
        logger.warning("wandb is not installed. Experiment tracking is disabled.")
        return

    try:
        # Initialize W&B run
        run = wandb.init(
            project="advertest-robustness",
            name=run_name,
            config=config,
            reinit=True
        )

        if run is None:
            logger.warning("wandb.init() failed. Ensure WANDB_API_KEY is set.")
            return

        status = metrics_dict.get("status", "UNKNOWN")
        deltas = metrics_dict.get("deltas", {})
        metrics = metrics_dict.get("metrics", {})

        # Prepare payload
        payload = {
            "deployment_status": 1 if status == "PASS" else 0,
            "base_drop_pct": deltas.get("base_drop_pct", 0.0),
            "corrupted_improvement_pct": deltas.get("corrupted_improvement_pct", 0.0),
            
            "original_base_map": metrics.get("original_model", {}).get("base_map", 0.0),
            "original_corrupted_map": metrics.get("original_model", {}).get("corrupted_map", 0.0),
            
            "new_base_map": metrics.get("new_model", {}).get("base_map", 0.0),
            "new_corrupted_map": metrics.get("new_model", {}).get("corrupted_map", 0.0),
        }

        # Log metrics to W&B
        wandb.log(payload)
        logger.info("Successfully logged metrics to Weights & Biases (Run: %s).", run.id)
        
        # Finish run
        run.finish()

    except Exception as e:
        logger.error("Failed to log to wandb: %s", e)
