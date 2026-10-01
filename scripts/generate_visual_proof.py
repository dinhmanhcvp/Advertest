"""Generate Visual Proofs for AdverTest Pitch Deck."""

import os
import urllib.request

import cv2
import numpy as np

# Adjust imports to local repository
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.types import Sample
from src.attacks.base import AttackContext

# Import our custom attacks
from src.attacks.corruption.tone_curve import RandomToneCurve
from src.attacks.corruption.sun_flare import RandomSunFlare
from src.attacks.corruption.grid_distortion import GridDistortion

def download_sample_image(url: str, save_path: str) -> np.ndarray:
    if not os.path.exists(save_path):
        req = urllib.request.urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        cv2.imwrite(save_path, img)
    else:
        img = cv2.imread(save_path)
        
    # Return as RGB float32 [0, 1] for our pipeline
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def save_image(img_float: np.ndarray, save_path: str):
    img_uint8 = np.clip(img_float * 255, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)
    print(f"Saved {save_path}")

def main():
    out_dir = Path("outputs/visual_proof")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Get a sample face image from Unsplash (simulate WIDER FACE)
    sample_url = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?q=80&w=800&auto=format&fit=crop"
    base_img_path = str(out_dir / "original_sample.jpg")
    img_float = download_sample_image(sample_url, base_img_path)
    
    # Wrap in Sample
    sample = Sample(sample_id="test_001", image=img_float)
    
    # Prepare Context
    rng = np.random.default_rng(42)
    ctx = AttackContext(rng=rng)
    
    # 2. Instantiate Attacks
    tone_curve = RandomToneCurve()
    sun_flare = RandomSunFlare()
    grid_distort = GridDistortion()
    
    # 3. Apply Attacks (Chaining)
    # Target: error_fisheye (GridDistortion) + error_overexposure (ToneCurve + SunFlare)
    severity = 4
    
    print("Applying Grid Distortion...")
    attacked = grid_distort.run(sample, severity, ctx)
    
    print("Applying Tone Curve...")
    attacked = tone_curve.run(attacked, severity, ctx)
    
    print("Applying Sun Flare...")
    attacked = sun_flare.run(attacked, severity, ctx)
    
    # 4. Save results
    save_image(attacked.image, str(out_dir / "simulated_egocentric.jpg"))
    
    print("Visual Proof generation complete! Check outputs/visual_proof/ directory.")

if __name__ == "__main__":
    main()
