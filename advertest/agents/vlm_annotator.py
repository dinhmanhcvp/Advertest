"""Zero-HITL VLM Annotator for AdverTest.

Replaces the manual Label Studio tagging process by using a Vision-Language Model
(e.g., GPT-4o, Gemini 1.5 Pro) to autonomously analyze "Hard Negatives" and
route them directly back into the Physics Attack Engine.

Requirements:
    pip install openai opencv-python pydantic
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# Use the core YOLOBox data structure and HardNegative from Phase 1 MVP
from advertest.core.inference_engine import YOLOBox, HardNegative
from advertest.attacks.engine import AttackEngine

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai package is required: pip install openai")

logger = logging.getLogger(__name__)


class VLMErrorAnalyzer:
    """Autonomously analyzes detection failures using a Vision-Language Model."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o") -> None:
        """
        Parameters
        ----------
        api_key : str | None
            OpenAI (or equivalent VLM provider) API key. 
            Defaults to OPENAI_API_KEY environment variable.
        model : str
            The specific Vision-Language Model to use.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("VLM API key is missing. Set OPENAI_API_KEY.")
            
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        
        # We reuse the existing Attack Engine to complete the Zero-HITL loop
        self.attack_engine = AttackEngine()

        self.system_prompt = """
You are an Expert Computer Vision QA Engineer.
Your task is to visually analyze why an object detection model failed on a specific image.

The image you receive has two types of bounding boxes drawn on it:
- GREEN boxes: The Ground Truth (where the object actually is).
- RED boxes: The Model's Prediction (where the model thought the object was).

Visually analyze why the red box missed the green box, or why there is no red box at all.
You must categorize the primary cause into EXACTLY ONE of the following tags from our taxonomy:
1. "error_blur" (Motion blur or out of focus)
2. "error_fisheye" (Severe lens distortion / GoPro effect)
3. "error_overexposure" (Harsh lighting, sun flare, clipping)
4. "occlusion" (Object is hidden behind something)

Output ONLY valid JSON matching this exact schema:
{
  "primary_cause": "one of the 4 tags above",
  "confidence_score": float between 0.0 and 1.0,
  "reasoning": "A concise 1-sentence explanation of your visual findings"
}
"""

    def _draw_boxes_and_encode(self, hn: HardNegative) -> str:
        """Draw GT (Green) and Pred (Red) boxes on the image and encode to Base64."""
        img_path = str(hn.image_path)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
            
        h, w = img.shape[:2]
        line_thickness = max(2, int(w * 0.005))
        
        # Draw Ground Truth boxes in GREEN (BGR: 0, 255, 0)
        for box in hn.gt_boxes:
            x1, y1, x2, y2 = box.to_xyxy(w, h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)
            
        # Draw Predicted boxes in RED (BGR: 0, 0, 255)
        for box in hn.pred_boxes:
            x1, y1, x2, y2 = box.to_xyxy(w, h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), line_thickness)
            
        # Encode as JPEG
        success, buffer = cv2.imencode('.jpg', img)
        if not success:
            raise ValueError("Failed to encode image to JPEG buffer.")
            
        # Convert to Base64
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_str}"

    def analyze_failure(self, hn: HardNegative) -> Dict[str, Any]:
        """Send the annotated image to the VLM and return the parsed JSON analysis."""
        logger.info("Encoding image %s for VLM analysis...", hn.image_path.name)
        base64_image = self._draw_boxes_and_encode(hn)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={ "type": "json_object" },
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this failure case."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image,
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.0,
            )
            
            result_text = response.choices[0].message.content
            analysis = json.loads(result_text)
            
            logger.info("VLM Analysis complete: %s (Confidence: %.2f)", 
                        analysis.get("primary_cause"), 
                        analysis.get("confidence_score", 0.0))
            return analysis
            
        except Exception as e:
            logger.error("VLM API call failed: %s", e)
            return {
                "primary_cause": "error_blur", # Fallback default
                "confidence_score": 0.0,
                "reasoning": f"API Error: {e}"
            }

    def zero_hitl_loop(self, hard_negatives: List[HardNegative], output_dir: str | Path) -> None:
        """The fully autonomous Zero-HITL pipeline.
        
        Takes a batch of Hard Negatives, asks the VLM for the root cause, and 
        immediately feeds that insight into the Physics Attack Engine to generate 
        new adversarial examples targeting that specific weakness.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting Zero-HITL Loop on %d Hard Negatives...", len(hard_negatives))
        
        attack_map = {
            "error_fisheye": "fisheye",
            "error_blur": "blur",
            "error_overexposure": "exposure",
            "occlusion": "fisheye" # Defaulting occlusion to fisheye warp for this demo
        }
        
        success_count = 0
        
        for idx, hn in enumerate(hard_negatives):
            logger.info("[%d/%d] Processing %s", idx + 1, len(hard_negatives), hn.image_path.name)
            
            # 1. VLM Inference
            analysis = self.analyze_failure(hn)
            primary_cause = analysis.get("primary_cause", "error_blur")
            
            # 2. Route to Attack Engine
            img_bgr = cv2.imread(str(hn.image_path))
            if img_bgr is None:
                continue
                
            strategy = attack_map.get(primary_cause, "blur")
            severity = 5 # Force max severity for adversarial generation
            
            logger.info("  -> Routing to AttackEngine: Strategy=%s, Severity=%d", strategy, severity)
            
            res_img, res_boxes, discarded = self.attack_engine.apply(
                img_bgr, hn.gt_boxes, strategy, severity
            )
            
            if discarded:
                logger.warning("  -> Discarded by ConstraintEngine (BBox Integrity violation).")
                continue
                
            # 3. Save resulting adversarial image and labels
            out_img_path = output_dir / f"{hn.image_path.stem}_vlm_{strategy}_s{severity}.jpg"
            out_lbl_path = output_dir / f"{hn.image_path.stem}_vlm_{strategy}_s{severity}.txt"
            
            cv2.imwrite(str(out_img_path), res_img)
            with open(out_lbl_path, "w") as f:
                for b in res_boxes:
                    f.write(b.to_label_str() + "\n")
                    
            success_count += 1
            
        logger.info("Zero-HITL Loop Complete. Successfully generated %d adversarial samples at %s", 
                    success_count, output_dir.resolve())


# ──────────────────────────── CLI entry point ──────

def main() -> None:
    """Standalone CLI for Zero-HITL Agent testing."""
    import argparse
    from advertest.core.inference_engine import Yolov7Evaluator

    parser = argparse.ArgumentParser(description="Zero-HITL VLM Annotator")
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLO weights")
    parser.add_argument("--images", type=str, required=True, help="Directory containing clean validation images")
    parser.add_argument("--output", type=str, default="data/pools/vlm_generated", help="Output directory")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    
    # 1. Extract a few hard negatives to test
    logger.info("Step 1: Extracting Hard Negatives...")
    evaluator = Yolov7Evaluator(weights=args.weights)
    hard_negs = evaluator.extract_hard_negatives(args.images, "data/temp/hn_vlm_test")
    
    if not hard_negs:
        logger.info("No hard negatives found. VLM Agent has nothing to analyze.")
        return
        
    # 2. Run VLM Zero-HITL Loop
    logger.info("Step 2: Activating VLM Agent...")
    try:
        agent = VLMErrorAnalyzer() # Relies on OPENAI_API_KEY env var
        agent.zero_hitl_loop(hard_negs, args.output)
    except ValueError as e:
        logger.error(e)


if __name__ == "__main__":
    main()
