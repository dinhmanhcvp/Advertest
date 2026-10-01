"""Data Drift Detection & Auto-Remediation Trigger.

Monitors the incoming production data streams for Data Drift (Distribution Shift).
If the real-world data starts drifting away from the original training baseline
(e.g., users mount cameras differently, lighting conditions change), this script
fires an alert and can autonomously trigger the AdverTest pipeline to synthesize
new adversarial data to patch the blind spot.

Requirements:
    pip install scipy requests numpy
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
try:
    from scipy.stats import ks_2samp
except ImportError:
    raise ImportError("scipy is required for Kolmogorov-Smirnov tests: pip install scipy")
import requests

logger = logging.getLogger(__name__)


class DriftMonitor:
    """Detects statistical shifts in data distributions."""

    def __init__(self, slack_webhook_url: str = None, api_gateway_url: str = "http://localhost:8000"):
        self.slack_webhook_url = slack_webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        self.api_gateway_url = api_gateway_url
        
    def _extract_features(self, dataset_path: Path) -> np.ndarray:
        """Mock feature extraction.
        
        In production, this would extract latent embeddings from the YOLO backbone
        or calculate image-level heuristics (brightness, contrast, edge density).
        For this demonstration, we return a mock distribution array.
        """
        # Simulating extracting 100 features representing the "brightness/blur" distribution
        return np.random.normal(loc=0.5, scale=0.1, size=100)

    def detect_drift(self, baseline_path: str, production_path: str, p_value_threshold: float = 0.05) -> bool:
        """Perform Kolmogorov-Smirnov test to detect data drift.
        
        Parameters
        ----------
        baseline_path : str
            Path to the original training dataset.
        production_path : str
            Path to the newly ingested production data (e.g., last 24h).
        p_value_threshold : float
            Threshold for statistical significance. Default 0.05.
            
        Returns
        -------
        bool
            True if drift is detected, False otherwise.
        """
        logger.info("Extracting features from Baseline: %s", baseline_path)
        baseline_features = self._extract_features(Path(baseline_path))
        
        logger.info("Extracting features from Production Data: %s", production_path)
        # We artificially shift the mean to simulate drift for the mock if we want
        prod_features = self._extract_features(Path(production_path))
        
        # In this mock, we force a slight shift to ensure the test works occasionally
        prod_features += np.random.normal(loc=0.03, scale=0.01, size=len(prod_features))
        
        # Perform 2-sample Kolmogorov-Smirnov test
        statistic, p_value = ks_2samp(baseline_features, prod_features)
        
        logger.info("KS Test Statistic: %.4f | P-Value: %.4f", statistic, p_value)
        
        if p_value < p_value_threshold:
            logger.warning("🚨 DATA DRIFT DETECTED! Distributions are significantly different (p < %s)", p_value_threshold)
            return True
            
        logger.info("✅ Data distribution is stable.")
        return False

    def send_alert(self, message: str) -> None:
        """Send an alert to a Slack/Discord webhook."""
        if not self.slack_webhook_url:
            logger.warning("No webhook URL configured. Skipping alert: %s", message)
            return
            
        try:
            payload = {"text": message}
            response = requests.post(self.slack_webhook_url, json=payload)
            if response.status_code == 200:
                logger.info("Alert successfully sent to webhook.")
            else:
                logger.error("Failed to send alert: HTTP %d", response.status_code)
        except Exception as e:
            logger.error("Webhook exception: %s", e)

    def auto_remediate(self, production_path: str) -> None:
        """Trigger the AdverTest API to synthesize new data addressing the drift."""
        logger.info("Initiating Auto-Remediation Workflow...")
        endpoint = f"{self.api_gateway_url}/api/v1/jobs/generate"
        
        try:
            payload = {"dataset_path": production_path}
            response = requests.post(endpoint, json=payload)
            
            if response.status_code == 200:
                job_data = response.json()
                logger.info("Successfully triggered AdverTest CT Pipeline! Job ID: %s", job_data.get("job_id"))
            else:
                logger.error("Failed to trigger API: HTTP %d - %s", response.status_code, response.text)
        except Exception as e:
            logger.error("API call exception: %s", e)


def run_cron_job(baseline_dir: str, production_dir: str):
    """The main execution function designed to be run via daily cron."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger.info("Starting Daily Data Drift Check...")
    
    monitor = DriftMonitor()
    
    is_drifting = monitor.detect_drift(baseline_dir, production_dir)
    
    if is_drifting:
        msg = "⚠️ *Data Drift Detected in Production.* Real-world egocentric data has shifted away from the training baseline. Initiating AdverTest Auto-Remediation..."
        monitor.send_alert(msg)
        
        # Autonomously call the API to generate new adversarial data
        monitor.auto_remediate(production_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AdverTest Drift Monitor")
    parser.add_argument("--baseline", type=str, default="data/clean_training_set", help="Path to baseline dataset")
    parser.add_argument("--production", type=str, default="data/raw_ego4d_streams", help="Path to recent production data")
    
    args = parser.parse_args()
    run_cron_job(args.baseline, args.production)
