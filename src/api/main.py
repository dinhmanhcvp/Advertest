from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.types import Sample
from src.attacks.base import AttackContext
from src.attacks.corruption.tone_curve import RandomToneCurve
from src.attacks.corruption.sun_flare import RandomSunFlare
from src.attacks.corruption.grid_distortion import GridDistortion

app = FastAPI(title="AdverTest API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate attacks
tone_curve = RandomToneCurve()
sun_flare = RandomSunFlare()
grid_distort = GridDistortion()

@app.post("/api/simulate")
async def simulate(file: UploadFile = File(...), tags: str = Form(...), severity: int = Form(4)):
    # 1. Read image from request
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return Response(status_code=400, content="Invalid image")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img.astype(np.float32) / 255.0
    
    # 2. Setup context
    sample = Sample(sample_id="api_test", image=img_float)
    rng = np.random.default_rng()
    ctx = AttackContext(rng=rng)
    
    # 3. Parse requested errors
    try:
        attack_list = json.loads(tags)
    except json.JSONDecodeError:
        attack_list = []
        
    attacked = sample
    
    # 4. Route attacks (Dynamic Chaining)
    if "error_fisheye" in attack_list:
        attacked = grid_distort.run(attacked, severity, ctx)
        
    if "error_overexposure" in attack_list:
        attacked = tone_curve.run(attacked, severity, ctx)
        attacked = sun_flare.run(attacked, severity, ctx)
        
    # Note: If error_blur was fully implemented we'd add it here.
    
    # 5. Encode output image
    res_img = np.clip(attacked.image * 255, 0, 255).astype(np.uint8)
    res_img_bgr = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
    _, encoded_img = cv2.imencode('.jpg', res_img_bgr)
    
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

# Serve Frontend static files if web directory exists
web_dir = Path(__file__).parent.parent.parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
