"""
Model registry and data schemas for AI Image Studio.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# Default models registry
DEFAULT_MODELS = {
    "sdxl-base-1.0": {
        "id": "sdxl-base-1.0",
        "name": "Stable Diffusion XL Base 1.0",
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline": "StableDiffusionXLPipeline",
        "default_resolution": [768, 1024],
        "min_vram_gb": 8,
        "recommended_steps": 30,
        "default_guidance_scale": 7.0,
        "fp16": True,
        "license": "CreativeML OpenRAIL++-M",
        "public_auth_required": False
    },
    "sd-1.5": {
        "id": "sd-1.5",
        "name": "Stable Diffusion 1.5",
        "repo_id": "runwayml/stable-diffusion-v1-5",
        "pipeline": "StableDiffusionPipeline",
        "default_resolution": [512, 512],
        "min_vram_gb": 4,
        "recommended_steps": 25,
        "default_guidance_scale": 7.5,
        "fp16": True,
        "license": "CreativeML OpenRAIL-M",
        "public_auth_required": False
    },
    "ssd-1b": {
        "id": "ssd-1b",
        "name": "Segmind SSD-1B",
        "repo_id": "segmind/SSD-1B",
        "pipeline": "StableDiffusionXLPipeline",
        "default_resolution": [1024, 1024],
        "min_vram_gb": 6,
        "recommended_steps": 25,
        "default_guidance_scale": 7.0,
        "fp16": True,
        "license": "Apache 2.0 compatible",
        "public_auth_required": False
    },
    "sdxl-turbo": {
        "id": "sdxl-turbo",
        "name": "Stable Diffusion XL Turbo",
        "repo_id": "stabilityai/sdxl-turbo",
        "pipeline": "AutoPipelineForText2Image",
        "default_resolution": [512, 512],
        "min_vram_gb": 6,
        "recommended_steps": 4,
        "default_guidance_scale": 0.0,
        "fp16": True,
        "license": "CreativeML OpenRAIL++-M Non-Commercial",
        "public_auth_required": False
    },
    "realistic-vision-v5.1": {
        "id": "realistic-vision-v5.1",
        "name": "Realistic Vision v5.1",
        "repo_id": "SG161222/Realistic_Vision_V5.1_noVAE",
        "pipeline": "StableDiffusionPipeline",
        "default_resolution": [512, 768],
        "min_vram_gb": 4,
        "recommended_steps": 25,
        "default_guidance_scale": 6.0,
        "fp16": True,
        "license": "CreativeML OpenRAIL-M",
        "public_auth_required": False
    },
    "flux-schnell": {
        "id": "flux-schnell",
        "name": "FLUX.1-schnell (4-bit NF4 Quantized)",
        "repo_id": "black-forest-labs/FLUX.1-schnell",
        "pipeline": "FluxPipeline",
        "default_resolution": [768, 1024],
        "min_vram_gb": 8,
        "recommended_steps": 4,
        "default_guidance_scale": 0.0,
        "max_sequence_length": 256,
        "fp16": False,
        "license": "Apache 2.0",
        "public_auth_required": False
    }
}

def get_model_specs(model_id: str) -> Dict[str, Any]:
    """
    Returns architecture specs and default generation parameters for a given model_id or repo_id.
    Dynamically identifies FLUX variants (schnell, dev), SDXL variants (base, turbo, ssd), and SD 1.5.
    """
    mid = (model_id or "").lower()

    if "flux" in mid:
        is_dev = "dev" in mid
        if is_dev:
            return {
                "family": "flux",
                "variant": "flux-dev",
                "default_width": 768,
                "default_height": 1024,
                "default_steps": 28,
                "max_recommended_steps": 50,
                "default_guidance_scale": 3.5,
                "max_sequence_length": 512,
                "supports_negative_prompt": False,
                "step_multiple": 16
            }
        else:
            return {
                "family": "flux",
                "variant": "flux-schnell",
                "default_width": 768,
                "default_height": 1024,
                "default_steps": 4,
                "max_recommended_steps": 8,
                "default_guidance_scale": 0.0,
                "max_sequence_length": 256,
                "supports_negative_prompt": False,
                "step_multiple": 16
            }
    elif "turbo" in mid:
        return {
            "family": "sdxl",
            "variant": "sdxl-turbo",
            "default_width": 512,
            "default_height": 512,
            "default_steps": 4,
            "max_recommended_steps": 10,
            "default_guidance_scale": 0.0,
            "max_sequence_length": 77,
            "supports_negative_prompt": False,
            "step_multiple": 8
        }
    elif "xl" in mid or "ssd" in mid:
        return {
            "family": "sdxl",
            "variant": "sdxl",
            "default_width": 768,
            "default_height": 1024,
            "default_steps": 30,
            "max_recommended_steps": 60,
            "default_guidance_scale": 7.0,
            "max_sequence_length": 77,
            "supports_negative_prompt": True,
            "step_multiple": 8
        }
    elif "1-5" in mid or "1.5" in mid or "v5.1" in mid:
        return {
            "family": "sd15",
            "variant": "sd-1.5",
            "default_width": 512,
            "default_height": 768,
            "default_steps": 25,
            "max_recommended_steps": 60,
            "default_guidance_scale": 7.0,
            "max_sequence_length": 77,
            "supports_negative_prompt": True,
            "step_multiple": 8
        }
    else:
        return {
            "family": "generic",
            "variant": "generic",
            "default_width": 768,
            "default_height": 1024,
            "default_steps": 30,
            "max_recommended_steps": 60,
            "default_guidance_scale": 7.0,
            "max_sequence_length": 77,
            "supports_negative_prompt": True,
            "step_multiple": 8
        }

DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, out of focus, pixelated, low resolution, "
    "cartoon, anime, illustration, CGI, 3D render, plastic skin, waxy skin, overly smooth skin, "
    "artificial appearance, unrealistic anatomy, distorted body proportions, malformed body, "
    "deformed face, asymmetrical eyes, crossed eyes, poorly drawn hands, extra fingers, "
    "missing fingers, fused fingers, extra arms, extra legs, duplicate body parts, "
    "unnatural pose, twisted limbs, broken anatomy, floating limbs, cropped head, "
    "cropped feet, incomplete body, cut-off legs, unnatural lighting, harsh shadows, "
    "overexposed, underexposed, oversaturated, excessive skin smoothing, distorted clothing, "
    "unnatural fabric, text, watermark, logo, signature"
)

class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt describing the desired image")
    negative_prompt: Optional[str] = Field(default=DEFAULT_NEGATIVE_PROMPT, max_length=2000, description="Text elements to exclude")
    width: Optional[int] = Field(default=768, ge=256, le=1536, description="Image width (must be multiple of 8 or 16)")
    height: Optional[int] = Field(default=1024, ge=256, le=1536, description="Image height (must be multiple of 8 or 16)")
    steps: Optional[int] = Field(default=4, ge=1, le=60, description="Denoising diffusion steps (4 for FLUX.1-schnell)")
    guidance_scale: Optional[float] = Field(default=0.0, ge=0.0, le=15.0, description="Guidance scale (0.0 for FLUX.1-schnell)")
    seed: Optional[int] = Field(default=None, ge=0, le=4294967295, description="Deterministic seed")
    num_images: Optional[int] = Field(default=1, ge=1, le=4, description="Number of images to generate (batch size)")
    format: Optional[str] = Field(default="png", description="Output format (png)")
    model_id: Optional[str] = Field(default="flux-schnell", description="Model to use (defaults to flux-schnell)")
    character_id: Optional[str] = Field(default=None, description="Optional character profile identifier")
    preset: Optional[str] = Field(default=None, description="Optional photography or aspect ratio preset name")
    idempotency_key: Optional[str] = Field(default=None, description="Client idempotency key to prevent duplicate jobs")

class ModelLoadRequest(BaseModel):
    model_id: str = Field(..., description="ID or Hugging Face repo ID to load")
    hf_token: Optional[str] = Field(default=None, description="Optional Hugging Face access token for gated models")
    idempotency_key: Optional[str] = Field(default=None, description="Client idempotency key")

class VramMetrics(BaseModel):
    total_gb: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float

class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool
    device_name: Optional[str] = None
    cuda_version: Optional[str] = None
    cuda_device_count: int = 0
    cuda_visible_devices: Optional[str] = None
    vram: Optional[VramMetrics] = None
    current_model: Optional[str] = None
    model_loaded: bool = False
    generation_ready: bool = False
    active_jobs: int = 0
    queue_length: int = 0
    running_job_id: Optional[str] = None

class JobResultItem(BaseModel):
    index: int
    seed: int
    url: str
    width: int
    height: int

class JobResponse(BaseModel):
    job_id: str
    idempotency_key: Optional[str] = None
    type: str
    status: str  # queued | downloading | loading | running | completed | failed | cancelled
    stage: Optional[str] = None  # waiting | downloading | loading | diffusion | encoding | completed
    progress: Optional[int] = None  # 0-100 percentage or None
    message: Optional[str] = None
    model_id: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    queue_wait_seconds: Optional[float] = None
    execution_duration_seconds: Optional[float] = None
    duration_seconds: Optional[float] = None  # Backwards compatibility
    vram: Optional[VramMetrics] = None
    error: Optional[str] = None
    results: Optional[List[JobResultItem]] = None
    image_url: Optional[str] = None  # Backwards compatibility
    effective_parameters: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None

class CancelResponse(BaseModel):
    job_id: str
    status: str
    message: str
