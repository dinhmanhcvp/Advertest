"""Dataset Compiler & Versioning.

Compiles the final hybrid adversarial dataset by merging the clean WIDER FACE
baseline with human-approved Attack Pools from Label Studio. Generates a
cryptographic Data Card (JSON) for data provenance.

Requirements:
    pip install label-studio-sdk
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

from advertest.label_studio.ls_connector import LSConnector

logger = logging.getLogger(__name__)


class DatasetCompiler:
    """Merges clean data with approved adversarial data into a unified dataset."""

    def __init__(self, ls_connector: LSConnector | None = None) -> None:
        self.ls = ls_connector or LSConnector()

    def merge_approved_pools(
        self,
        project_id: int,
        clean_dataset_dir: str | Path,
        adversarial_source_dir: str | Path,
        output_dir: str | Path,
        approval_tag: str = "approved"
    ) -> Dict[str, Any]:
        """Query Label Studio for HITL-approved samples and merge them with clean data.

        Parameters
        ----------
        project_id : int
            Label Studio project ID containing the reviewed adversarial tasks.
        clean_dataset_dir : str | Path
            Path to the original WIDER FACE dataset (contains images/ and labels/ subdirectories).
        adversarial_source_dir : str | Path
            Path where the locally generated adversarial images reside before upload to LS.
        output_dir : str | Path
            Destination for the new hybrid YOLO dataset.
        approval_tag : str
            The specific tag value in the Label Studio choices indicating HITL approval.

        Returns
        -------
        Dict[str, Any]
            Metadata about the compilation process.
        """
        clean_dataset_dir = Path(clean_dataset_dir)
        adversarial_source_dir = Path(adversarial_source_dir)
        output_dir = Path(output_dir)

        out_images = output_dir / "images"
        out_labels = output_dir / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        logger.info("Starting Dataset Compilation...")
        
        # 1. Copy Clean Baseline
        clean_count = 0
        clean_imgs_dir = clean_dataset_dir / "images"
        clean_lbls_dir = clean_dataset_dir / "labels"
        
        if clean_imgs_dir.exists():
            for img_path in clean_imgs_dir.glob("*.*"):
                if img_path.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
                    continue
                shutil.copy2(img_path, out_images / img_path.name)
                lbl_path = clean_lbls_dir / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    shutil.copy2(lbl_path, out_labels / lbl_path.name)
                clean_count += 1
        
        logger.info("Copied %d clean baseline samples.", clean_count)

        # 2. Fetch Approved Adversarial Samples from Label Studio
        logger.info("Querying Label Studio Project %d for '%s' tasks...", project_id, approval_tag)
        
        try:
            tasks_iter = self.ls.client.tasks.list(project_id=project_id)
            tasks = [t.model_dump() if hasattr(t, "model_dump") else t for t in tasks_iter]
        except Exception as e:
            logger.error("Failed to fetch tasks via SDK: %s. Attempting raw API.", e)
            import requests
            url = f"{self.ls.client._base_url}/api/tasks?project={project_id}&with_annotations=true"
            headers = {"Authorization": f"Token {self.ls.client._api_key}"}
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            tasks = resp.json()

        adv_count = 0
        severity_dist = defaultdict(int)
        
        for task in tasks:
            annotations = task.get("annotations", [])
            if not annotations:
                continue
                
            latest_anno = annotations[-1]
            anno_result = latest_anno.get("result", [])
            
            # Check if this task was marked as 'approved' by HITL
            is_approved = False
            for res in anno_result:
                if res.get("type") == "choices":
                    choices = res.get("value", {}).get("choices", [])
                    if approval_tag in choices:
                        is_approved = True
                        break
            
            if is_approved:
                # The image URL typically looks like /data/local-files/?d=filename.jpg
                image_url = task.get("data", {}).get("image", "")
                filename = image_url.split("=")[-1] if "=" in image_url else Path(image_url).name
                
                # Copy from local adversarial source to output
                src_img = adversarial_source_dir / filename
                if src_img.exists():
                    shutil.copy2(src_img, out_images / filename)
                    # We also copy the transformed label
                    src_lbl = adversarial_source_dir / f"{src_img.stem}.txt"
                    if src_lbl.exists():
                        shutil.copy2(src_lbl, out_labels / src_lbl.name)
                    
                    adv_count += 1
                    
                    # Try to extract severity from filename (e.g. image_error_fisheye_s4.jpg)
                    if "_s" in filename:
                        sev_part = filename.split("_s")[-1].split(".")[0]
                        if sev_part.isdigit():
                            severity_dist[int(sev_part)] += 1
                else:
                    logger.warning("Approved file %s not found in local source %s", filename, adversarial_source_dir)

        logger.info("Merged %d approved adversarial samples.", adv_count)
        
        return {
            "clean_count": clean_count,
            "adversarial_count": adv_count,
            "total_count": clean_count + adv_count,
            "severity_distribution": dict(severity_dist),
            "output_dir": str(output_dir.resolve())
        }

    def generate_data_card(self, output_dir: str | Path, compilation_stats: Dict[str, Any]) -> None:
        """Generate a cryptographic Data Card (JSON) for provenance and reproducibility.

        Computes a SHA-256 hash of the final dataset directory contents and saves
        the distribution metrics.
        """
        output_dir = Path(output_dir)
        
        # 1. Compute Dataset Checksum
        logger.info("Computing SHA-256 checksum for dataset...")
        hasher = hashlib.sha256()
        
        # Hash all labels and images deterministically
        all_files = sorted(output_dir.rglob("*.*"))
        for filepath in all_files:
            if filepath.name == "data_card.json":
                continue
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
                    
        dataset_hash = hasher.hexdigest()
        
        # 2. Calculate Ratios
        total = compilation_stats["total_count"]
        clean_ratio = compilation_stats["clean_count"] / total if total > 0 else 0
        adv_ratio = compilation_stats["adversarial_count"] / total if total > 0 else 0
        
        data_card = {
            "version": "1.0",
            "dataset_checksum_sha256": dataset_hash,
            "total_samples": total,
            "ratios": {
                "clean": float(f"{clean_ratio:.4f}"),
                "adversarial": float(f"{adv_ratio:.4f}")
            },
            "severity_distribution": compilation_stats["severity_distribution"],
            "description": "Hybrid Ego-centric Adversarial Dataset generated by AdverTest."
        }
        
        card_path = output_dir / "data_card.json"
        with open(card_path, "w") as f:
            json.dump(data_card, f, indent=4)
            
        logger.info("Data Card generated: %s", card_path.resolve())
        logger.info("Dataset SHA-256: %s", dataset_hash)
