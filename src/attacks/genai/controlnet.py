"""Phase 2: Generative AI Adapter for ControlNet Egocentric Synthesis."""

import cv2
import numpy as np
from typing import ClassVar, Any
from pydantic import Field

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, BaseAttack, AttackParams
from src.core.types import Sample, AttackGroup

try:
    import torch
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    _HAS_DIFFUSERS = True
except ImportError:
    _HAS_DIFFUSERS = False


class ControlNetParams(AttackParams):
    """Parameters for ControlNet Egocentric Generation."""
    prompt: str = Field(
        default="first-person view, body-worn camera perspective, fisheye lens distortion, action camera, realistic lighting, outdoor street",
        description="Text prompt guiding the SD generation."
    )
    negative_prompt: str = Field(
        default="monochrome, lowres, bad anatomy, worst quality, low quality",
        description="Negative prompt to avoid artifacts."
    )
    guidance_scale: float = Field(default=7.5)
    num_inference_steps: int = Field(default=20)


@ATTACKS.register
class ControlNetEgocentric(BaseAttack):
    """
    Uses ControlNet (Canny Edge) to synthesize true 3D camera angles
    while preserving the identity/structure of the original image.
    """
    
    name: ClassVar[str] = "controlnet_egocentric"
    group: ClassVar[AttackGroup] = "D"  # GenAI / Advanced
    category = "synthesis"
    params_model = ControlNetParams
    
    def __init__(self, **params: Any):
        super().__init__(**params)
        self.pipe = None
        
    def _init_pipeline(self, device="cuda"):
        """Lazy load the pipeline only when executed on GPU."""
        if self.pipe is not None:
            return
            
        print("🚀 Initializing ControlNet SD1.5 Pipeline...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        # Ensure it moves to correct device if offload is not sufficient
        if device == "cuda" and torch.cuda.is_available():
            self.pipe.to("cuda")

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        params = self.resolve_parameters(severity)
        
        # 1. Extract Condition Image (Canny Edge)
        img_uint8 = np.clip(sample.image * 255.0, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        # We use a slight blur to reduce edge noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)
        edges_3c = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # 2. Mock Mode (For Local/Low-end machines during Pitch)
        if not _HAS_DIFFUSERS or not ctx.use_gpu:
            # We return the Canny Edge visualization overlaid with a Mock Text
            # to prove the condition extraction works.
            mock_img = cv2.addWeighted(img_uint8, 0.3, edges_3c, 0.7, 0)
            cv2.putText(mock_img, "GEN-AI PHASE 2 (MOCK MODE)", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(mock_img, "Run on Colab (GPU) for Full Render", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return sample.with_image(mock_img.astype(np.float32) / 255.0)
            
        # 3. Full Render (On GPU)
        self._init_pipeline(ctx.device)
        
        # Convert Canny array to PIL Image for diffusers
        from PIL import Image
        condition_image = Image.fromarray(edges_3c)
        
        # Generate new image
        import torch
        generator = torch.manual_seed(int(ctx.rng.integers(0, 2**31 - 1)))
        
        output = self.pipe(
            params["prompt"],
            image=condition_image,
            negative_prompt=params["negative_prompt"],
            num_inference_steps=int(params["num_inference_steps"]),
            guidance_scale=float(params["guidance_scale"]),
            generator=generator
        ).images[0]
        
        # Convert back to float32 [0, 1] numpy array
        result_array = np.array(output).astype(np.float32) / 255.0
        
        return sample.with_image(result_array)
